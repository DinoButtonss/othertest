"""
tetris_vision.py  --  perception for the SNES Tetris & Dr. Mario bot

What changed vs. the old detector (all found by replaying your screen recording):

 1. GEOMETRY.  The old code sampled a 10x20 lattice laid over a hand-drawn rectangle that
    was then trimmed by 3.5 % on each side.  The real cells were ~7 % bigger than the
    lattice, so columns 0 and 9 were sampled half-way into the neighbouring cell and the
    error grew toward the edges.  Here the board is mapped with a 4-corner perspective
    transform (cv2.getPerspectiveTransform) straight onto an exact 10x20 grid.  No trim.

 2. OCCUPANCY.  A cell is "filled" when most of its inner area is bright.  Background
    texture, dots and the maze pattern are dark (V ~ 30-60); blocks are V > 180.  No
    per-colour hacks, no specular heuristics.

 3. SPAWN.  Instead of fuzzy template scoring, a spawn is accepted only when EXACTLY four
    new cells appear in the top rows, the pattern matches one of the 7 SNES spawn shapes
    exactly, and it sits in the known spawn columns.  The stack snapshot is the last
    grid captured while the spawn area was empty - no 35-frame timers, and real holes /
    overhangs are never "corrected" away.

 4. NEXT BOX.  Read from a calibrated quad (the dark piece area only, label excluded),
    classified from a 2x3 / 1x4 / 2x2 occupancy pattern, voted over several frames, and
    only trusted when it keeps agreeing with the pieces that actually spawn.  If it is not
    trusted the AI searches one piece deep.  It never guesses 'T' any more.
"""
import json
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:                      # running on a laptop / in replay mode
    HAS_PICAMERA2 = False

BOARD_COLS = 10
BOARD_ROWS = 20

# SNES spawn shapes (matrix, left column of bounding box at spawn).  Spawn row is always 0.
SNES_SPAWN = {
    'I': ([[1, 1, 1, 1]], 3),
    'O': ([[1, 1], [1, 1]], 4),
    'T': ([[1, 1, 1], [0, 1, 0]], 4),
    'S': ([[0, 1, 1], [1, 1, 0]], 4),
    'Z': ([[1, 1, 0], [0, 1, 1]], 4),
    'J': ([[1, 1, 1], [0, 0, 1]], 4),
    'L': ([[1, 1, 1], [1, 0, 0]], 4),
}

CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")


# ----------------------------------------------------------------------------- geometry
def order_corners(pts):
    """Return corners as TL, TR, BR, BL regardless of click order."""
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()          # y - x
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


class Warper:
    def __init__(self, corners, out_w, out_h):
        src = order_corners(corners)
        dst = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = np.linalg.inv(self.M)
        self.size = (int(out_w), int(out_h))

    def __call__(self, frame):
        return cv2.warpPerspective(frame, self.M, self.size, flags=cv2.INTER_LINEAR)

    def to_frame(self, pts):
        """Map points from the warped image back into camera-frame pixels (for drawing overlays)."""
        p = np.float32(pts).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.Minv).reshape(-1, 2)


def load_calibration(path=CALIBRATION_FILE, frame_size=None):
    """Loads calibration.json.  Supports the new 4-corner format and (with a warning) the
    old playfield/next_box rectangles.  If frame_size is given and differs from the size
    the file was made at, the corners are scaled automatically."""
    if not os.path.exists(path):
        raise RuntimeError("No calibration.json found. Run:  python3 calibrate.py")
    with open(path) as f:
        cal = json.load(f)

    if "board_corners" not in cal:           # legacy rectangles
        pf, nb = cal["playfield"], cal["next_box"]

        def rect_to_corners(r):
            x, y, w, h = r["left"], r["top"], r["width"], r["height"]
            return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

        cal["board_corners"] = rect_to_corners(pf)
        cal["next_corners"] = rect_to_corners(nb)
        cal["legacy"] = True
        cal.setdefault("frame_size", [640, 360])      # the old calibrate.py captured at 640x360
        print("WARNING: old rectangle calibration loaded. Re-run calibrate.py for the 4-corner "
              "version - it is noticeably more accurate.")

    src_size = cal.get("frame_size")
    if frame_size and src_size and tuple(src_size) != tuple(frame_size):
        sx, sy = frame_size[0] / float(src_size[0]), frame_size[1] / float(src_size[1])
        for key in ("board_corners", "next_corners"):
            cal[key] = [[p[0] * sx, p[1] * sy] for p in cal[key]]
    return cal


def save_calibration(data, path=CALIBRATION_FILE):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# ----------------------------------------------------------------------------- board
class BoardObservation:
    __slots__ = ("grid", "cover", "sat", "sat_min", "banner", "warp")

    def __init__(self, grid, cover, sat, sat_min, banner, warp):
        self.grid = grid          # bool (20,10)  strict occupancy
        self.cover = cover        # float (20,10) fraction of the cell that is lit
        self.sat = sat            # float (20,10) mean saturation of the lit pixels
        self.sat_min = sat_min
        self.banner = banner      # True when WAIT / LOSE / PAUSE style text is on the board
        self.warp = warp          # warped BGR board image (for debugging)

    def cells(self, cover_thr):
        """Occupancy at a different strictness (used for dim cells under the top bezel)."""
        return (self.cover >= cover_thr) & (self.sat >= self.sat_min)


class BoardReader:
    """Warps the playfield to an exact grid and decides which cells hold a block.

    ADAPTIVE (adaptive=True, default): nothing is tied to one colour or one brightness.
      bg_v   per-cell brightness of the EMPTY background, learned continuously from cells that are
             quiet (no block, no neighbour lit) - follows room light, glare haze, exposure drift.
      ref_v  brightness of the blocks currently on the board (dim / dark-palette levels read lower).
      lit    = brighter than the midpoint between background and blocks
      colour = the lit pixels must be colourful (saturation >= sat_min, itself scaled to the blocks'
               own saturation), which rejects white text and white glare.
    No hue is ever used, so purple / orange / any palette works as long as blocks are clearly
    brighter than the background.  With default lighting the midpoint is ~140, same as before.
    """

    def __init__(self, corners, cell=32, v_thr=140, cover_thr=0.85, margin=0.2, sat_min=90, adaptive=True):
        self.cell = cell
        self.warper = Warper(corners, BOARD_COLS * cell, BOARD_ROWS * cell)
        self.v_thr = float(v_thr)
        self.cover_thr = cover_thr
        self.sat_min_default = float(sat_min)
        self.sat_min = float(sat_min)
        self.m = max(1, int(round(cell * margin)))
        self.adaptive = adaptive
        self.bg_v = np.full((BOARD_ROWS, BOARD_COLS), 60.0, dtype=np.float32)
        self.bg_init = False
        self.ref_v = 220.0
        self.sat_ref = 200.0
        self.thr_map = np.full((BOARD_ROWS, BOARD_COLS), self.v_thr, dtype=np.float32)
        self.health = {"bg": 60.0, "ref": 220.0, "contrast": 160.0, "sat_min": self.sat_min}

    # --- optional manual override (key 'b'): learn the background of an EMPTY board right now.
    def learn_empty(self, frames):
        p95 = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.float32)
        for fr in frames:
            v4 = self._v4(self.warper(fr))
            cur = np.percentile(v4.transpose(0, 2, 1, 3).reshape(BOARD_ROWS, BOARD_COLS, -1), 95, axis=2)
            p95 = np.maximum(p95, cur)
        self.bg_v = p95.astype(np.float32)
        self.bg_init = True
        self._update_thr()

    def _v4(self, warp):
        v = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)[:, :, 2]
        c, m = self.cell, self.m
        return v.reshape(BOARD_ROWS, c, BOARD_COLS, c)[:, m:c - m, :, m:c - m]

    def _bg_smooth(self):
        """Glare and haze are SMOOTH over many cells; a bezel edge or a reflection on one cell is not.
        So a cell's background may not be much brighter than the median of its 5x5 neighbourhood."""
        nb = cv2.medianBlur(self.bg_v.astype(np.float32), 5)
        return np.minimum(self.bg_v, nb + 25.0)

    def _update_thr(self):
        bg = self._bg_smooth()
        thr = bg + 0.55 * (self.ref_v - bg)
        thr = np.clip(thr, np.maximum(bg + 35.0, 55.0), 200.0)
        # Near normal lighting use exactly the threshold that was tuned on real footage; the adaptive
        # value only takes over when conditions genuinely differ (dim, hazy, glare, other palette).
        thr = np.where(np.abs(thr - self.v_thr) <= 12.0, self.v_thr, thr)
        self.thr_map = thr.astype(np.float32)

    def _adapt(self, v4, s4, cover, sat, grid, banner, std_v):
        if banner:
            return
        p95 = np.percentile(v4.transpose(0, 2, 1, 3).reshape(BOARD_ROWS, BOARD_COLS, -1), 95, axis=2)

        # Glare / haze: bright but colourless and smooth.  It is background, so absorb it quickly
        # (a block is never colourless).  This is what keeps a hot spot on the glass from being read
        # as blocks or as a WAIT/LOSE banner.
        glare = (sat < 70) & (p95 > 100) & (std_v < 45)
        # ... and only where it is a smooth AREA (>= 4 of the 3x3 neighbourhood agree), not a lone bright cell
        crowd = cv2.boxFilter(glare.astype(np.float32), -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
        glare &= crowd >= 4
        if glare.any():
            self.bg_v[glare] = 0.6 * self.bg_v[glare] + 0.4 * p95[glare]

        near = cv2.dilate((cover >= 0.3).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        quiet = (cover < 0.03) & ~near & ~glare & (p95 < 0.55 * self.ref_v + 20.0)
        if quiet.any():
            if not self.bg_init:
                self.bg_v[:] = float(np.median(p95[quiet]))
                self.bg_v[quiet] = p95[quiet]
                self.bg_init = True
            else:
                self.bg_v[quiet] = 0.97 * self.bg_v[quiet] + 0.03 * p95[quiet]

        bg = float(np.median(self._bg_smooth()))
        if int(grid.sum()) >= 4:
            vmean = v4.mean(axis=(1, 3))
            ref_now = float(np.percentile(vmean[grid], 25))
            self.ref_v = float(np.clip(0.9 * self.ref_v + 0.1 * ref_now, 100.0, 255.0))
            self.sat_ref = 0.9 * self.sat_ref + 0.1 * float(np.median(sat[grid]))
            self.sat_min = float(np.clip(0.45 * self.sat_ref, 45.0, self.sat_min_default))
        else:
            # Nothing is detected as a block yet (very dim / dark palette?).  If something clearly
            # brighter than the background and colourful exists, let the block level come DOWN to it.
            # saturation measured at each cell's brightest pixels (not just at 'lit' pixels, of which
            # there may be none yet when the picture is very dim)
            peak_mask = v4 >= (p95[:, None, :, None] - 5.0)
            sat_peak = (s4 * peak_mask).sum(axis=(1, 3)) / np.maximum(peak_mask.sum(axis=(1, 3)), 1)
            colourful = (sat_peak >= 0.45 * self.sat_ref) & ~glare
            if colourful.any():
                peak = float(np.percentile(p95[colourful], 99))
                if peak - bg > 40.0 and peak < self.ref_v - 25.0:
                    self.ref_v = float(np.clip(0.9 * self.ref_v + 0.1 * peak, 60.0, 255.0))
        self._update_thr()
        self.health = {"bg": bg, "ref": self.ref_v, "contrast": self.ref_v - bg, "sat_min": self.sat_min}

    def read(self, frame):
        warp = self.warper(frame)
        hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
        c, m = self.cell, self.m
        v4 = hsv[:, :, 2].reshape(BOARD_ROWS, c, BOARD_COLS, c)[:, m:c - m, :, m:c - m]
        s4 = hsv[:, :, 1].reshape(BOARD_ROWS, c, BOARD_COLS, c)[:, m:c - m, :, m:c - m]

        thr = self.thr_map[:, None, :, None]
        lit = v4 > thr
        cover = lit.mean(axis=(1, 3))
        # mean saturation of the LIT pixels: real blocks are strongly coloured,
        # white text such as WAIT / LOSE / PAUSE and white glare are not.
        n_lit = np.maximum(lit.sum(axis=(1, 3)), 1)
        sat = (s4 * lit).sum(axis=(1, 3)) / n_lit
        bright = cover >= self.cover_thr
        grid = bright & (sat >= self.sat_min)

        # Banner text (WAIT / LOSE / PAUSE) = several colourless cells with PARTIAL coverage and sharp
        # strokes (high brightness variance inside the cell).  A glare spot is smooth, so it is not text.
        std_v = v4.std(axis=(1, 3))
        text_like = (cover >= 0.2) & (cover <= 0.92) & (sat < 60) & (std_v >= 40)
        banner = int(text_like[3:17, 1:9].sum()) >= 4
        if self.adaptive:
            self._adapt(v4, s4, cover, sat, grid, banner, std_v)
        return BoardObservation(grid, cover.astype(np.float32), sat.astype(np.float32), self.sat_min, banner, warp)


def _decisiveness_cost(frames, corners, cell=24, v_thr=140, margin=0.12, reward=0.6):
    """Lower = every cell is either clearly lit or clearly dark = the grid is aligned with the blocks.
    The 'reward' term (fully lit cells) stops the search from sliding off a few blocks when the
    frames contain little stack - tested on early/mid/late frames of the recording."""
    wp = Warper(corners, BOARD_COLS * cell, BOARD_ROWS * cell)
    m = max(1, int(round(cell * margin)))
    total = 0.0
    for fr in frames:
        v = cv2.cvtColor(wp(fr), cv2.COLOR_BGR2HSV)[:, :, 2]
        lit = (v > v_thr).astype(np.float32)
        lit4 = lit.reshape(BOARD_ROWS, cell, BOARD_COLS, cell)[:, m:cell - m, :, m:cell - m]
        cov = lit4.mean(axis=(1, 3))
        total += float((cov * (1.0 - cov)).sum()) - reward * float((cov > 0.9).sum())
    return total


def refine_corners(corners, frames, shift_px=7.0, rounds=3):
    """Fine-tunes hand-clicked corners.  Searches translation and x/y scale (coordinate descent)
    for the grid placement that makes every cell clearly lit or clearly dark.  Pass a few frames
    that contain settled blocks (e.g. the camera frames at the last few spawns).  Returns
    (new_corners, cost_before, cost_after).  Use it only when the frames contain blocks."""
    base = order_corners(corners).astype(np.float64)
    ctr = base.mean(axis=0)
    p = {"dx": 0.0, "dy": 0.0, "sx": 1.0, "sy": 1.0}

    def build(q):
        c = base.copy()
        c[:, 0] = (c[:, 0] - ctr[0]) * q["sx"] + ctr[0] + q["dx"]
        c[:, 1] = (c[:, 1] - ctr[1]) * q["sy"] + ctr[1] + q["dy"]
        return c

    cost0 = _decisiveness_cost(frames, build(p))
    best = cost0
    for r in range(rounds):
        shrink = 0.5 ** r
        for key, span, steps in (("dx", shift_px * shrink, 9), ("dy", shift_px * shrink, 9),
                                 ("sx", 0.04 * shrink, 9), ("sy", 0.04 * shrink, 9)):
            centre = p[key]
            for v in np.linspace(centre - span, centre + span, steps):
                q = dict(p)
                q[key] = float(v)
                c = _decisiveness_cost(frames, build(q))
                if c < best - 1e-6:
                    best = c
                    p = q
    return build(p).astype(np.float32), cost0, best


# ----------------------------------------------------------------------------- spawn tracker
def _match_spawn(cells, max_r0=3):
    """cells: iterable of (r, c).  Returns (piece, left_col) if they are exactly a spawn shape."""
    cells = set(cells)
    if len(cells) not in (4,):
        return None
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    rel = {(r - r0, c - c0) for r, c in cells}
    if r0 > max_r0:                 # spawns on row 0; at fast levels it can already be a few rows down when first seen
        return None
    for name, (mat, spawn_col) in SNES_SPAWN.items():
        tmpl = {(r, c) for r, row in enumerate(mat) for c, v in enumerate(row) if v}
        if rel == tmpl and abs(c0 - spawn_col) <= 1:       # allow 1 col slack: piece may already be moving
            return name, c0
    return None


class SpawnEvent:
    def __init__(self, t, piece, left_col, stack, cells):
        self.t = t
        self.t_first = t
        self.piece = piece
        self.left_col = left_col
        self.stack = stack            # bool (20,10): board WITHOUT the new piece
        self.cells = cells


class SpawnTracker:
    """Turns a stream of occupancy grids into 'a new piece appeared' events."""

    def __init__(self, confirm_s=0.10, min_confirm_frames=3, rearm_empty_s=0.15):
        self.confirm_s = confirm_s
        self.min_confirm_frames = min_confirm_frames
        self.rearm_empty_s = rearm_empty_s
        self.recent = deque(maxlen=3)         # last grids seen with an empty spawn area
        self.armed = False
        self.empty_since = None
        self.cand = None
        self.cand_first_t = None
        self.cand_frames = 0
        self.last_stack = None
        self.rejects = 0
        self.was_banner = False
        self.cand_stack = None
        self.cand_stacks = []

    def reset(self):
        self.recent.clear()
        self.armed = False
        self.empty_since = None
        self.cand = None
        self.cand_frames = 0
        self.cand_stack = None

    def _snapshot(self):
        if not self.recent:
            return None
        if len(self.recent) < 3:
            return self.recent[-1].copy()
        a, b, c = list(self.recent)
        return ((a.astype(np.uint8) + b + c) >= 2)      # 2-of-3 vote kills single-frame flicker

    # cover thresholds tried, strictest first, when looking for a freshly spawned piece
    SPAWN_LEVELS = (0.85, 0.65, 0.50, 0.30)
    ZONE_ROWS = 4      # spawn area that must be empty before a spawn can be detected (rows 0..ZONE_ROWS-1)
    NEW_ROWS = 6       # rows searched for the 4 new cells
    MAX_R0 = 3         # highest row index the piece's top may be on when first seen

    def update(self, obs, t):
        """obs: BoardObservation.  Returns a SpawnEvent or None."""
        grid, banner = obs.grid, obs.banner
        if banner:
            self.reset()
            self.was_banner = True
            return None
        if self.was_banner:
            # WAIT / LOSE / PAUSE just went away.  At game start the board is empty by
            # definition, so seed an empty snapshot (the first piece can spawn immediately).
            # Mid-game this is harmless: a non-empty board simply fails the 4-new-cells test
            # until the spawn area has been seen empty again.
            self.was_banner = False
            for _ in range(3):
                self.recent.append(np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool))
            self.armed = True

        weakest = obs.cells(self.SPAWN_LEVELS[-1])
        if not weakest[0:self.ZONE_ROWS, 2:8].any():                      # spawn area (top 4 rows) empty even at the loosest level
            self.recent.append(grid.copy())
            if self.empty_since is None:
                self.empty_since = t
            if not self.armed and (t - self.empty_since) >= self.rearm_empty_s - 1e-6:
                self.armed = True
            self.cand = None
            self.cand_frames = 0
            self.cand_stack = None
            return None

        self.empty_since = None
        if not self.armed:
            return None
        snap = self._snapshot()
        if snap is None:
            return None

        # Look for exactly one valid 4-cell spawn shape in rows 0-3, strict threshold first.
        # (A piece's top row can sit under the bezel and read dim, so we relax step by step.
        #  Requiring EXACTLY a tetromino at the spawn columns keeps glow from sneaking in.)
        match = None
        new_cells = None
        n_new = 0
        for lvl in self.SPAWN_LEVELS:
            g = obs.cells(lvl)
            nc = np.zeros_like(grid)
            nc[0:self.NEW_ROWS, 2:8] = g[0:self.NEW_ROWS, 2:8] & ~snap[0:self.NEW_ROWS, 2:8]
            # Only the spawn columns (2..7) are examined, and the tetromino is found as a connected
            # component: stray lit cells elsewhere (bezel glare, reflections, text) cannot veto it.
            n_lab, lab = cv2.connectedComponents(nc.astype(np.uint8), connectivity=4)
            found = []
            for j in range(1, n_lab):
                comp = lab == j
                if int(comp.sum()) != 4:
                    continue
                rs, cs = np.nonzero(comp)
                m = _match_spawn(zip(rs.tolist(), cs.tolist()), self.MAX_R0)
                if m is not None:
                    found.append((m, comp))
            n_new = max(n_new, int(nc.sum()))
            if len(found) == 1:
                match, new_cells = found[0][0], found[0][1]
                break
        if match is None:
            self.cand = None
            self.cand_frames = 0
            if n_new > 0:
                self.rejects += 1
            return None

        # The game only spawns AFTER the previous piece has locked (and any line-clear
        # animation has finished), so the stack is "what is lit now, minus the new piece".
        # (An older snapshot is wrong: the last piece can still be in the air in it when it
        # hard-drops and the next piece spawns right after.)
        stack_now = grid & ~new_cells
        same_stack = (self.cand_stack is not None and
                      int((stack_now ^ self.cand_stack).sum()) <= 1)
        if self.cand is not None and self.cand[0] == match[0] and same_stack:
            self.cand_frames += 1
            self.cand_stacks.append(stack_now)
        else:
            self.cand = match
            self.cand_frames = 1
            self.cand_first_t = t
            self.cand_stacks = [stack_now]
        self.cand_stack = stack_now
        if (self.cand_frames >= self.min_confirm_frames
                and (t - self.cand_first_t) >= self.confirm_s - 1e-6):
            # 2-of-3 vote over the last confirming frames removes single-frame flicker
            last = self.cand_stacks[-3:]
            voted = (sum(x.astype(np.uint8) for x in last) * 2) > len(last)
            voted = voted | (sum(x.astype(np.uint8) for x in last) == len(last))
            self.armed = False
            self.last_stack = voted
            rs, cs = np.nonzero(new_cells)
            ev = SpawnEvent(t, match[0], match[1], voted, list(zip(rs.tolist(), cs.tolist())))
            ev.t_first = self.cand_first_t           # when the piece was FIRST seen (before the confirm frames)
            return ev
        return None


# ----------------------------------------------------------------------------- next box
_NEXT_PATTERNS = {}
for _name, (_mat, _col) in SNES_SPAWN.items():
    if _name in ('I', 'O'):
        continue
    _a = np.array(_mat, dtype=np.uint8)
    _NEXT_PATTERNS[_name] = [_a, _a[::-1, ::-1]]        # spawn orientation + 180deg (same piece type)


class NextReader:
    """Reads the piece in the NEXT box from a calibrated quad around the dark piece area."""

    def __init__(self, corners, out_w=160, out_h=80):
        self.warper = Warper(corners, out_w, out_h)
        self.w, self.h = out_w, out_h
        self.last_mask = None

    def read(self, frame):
        """Returns (piece_or_None, debug_mask)."""
        img = self.warper(frame)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        bg = float(np.percentile(v, 25))
        top = float(np.percentile(v, 98))
        if top - bg < 40.0:                                  # nothing brighter than the background
            self.last_mask = np.zeros_like(v)
            return None, self.last_mask
        thr = bg + 0.5 * (top - bg)                          # relative: works for dim / dark palettes
        mask = (v > thr).astype(np.uint8)
        mask[:2, :] = 0
        mask[-2:, :] = 0
        mask[:, :2] = 0
        mask[:, -2:] = 0
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        self.last_mask = mask * 255

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return None, self.last_mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        area = int(stats[best, cv2.CC_STAT_AREA])
        if area < 0.012 * self.w * self.h:                       # nothing piece-sized
            return None, self.last_mask
        x, y, bw, bh = (int(stats[best, k_]) for k_ in
                        (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        if bw < 4 or bh < 4:
            return None, self.last_mask
        comp = (labels[y:y + bh, x:x + bw] == best).astype(np.float32)
        aspect = bw / float(bh)
        fill = area / float(bw * bh)

        if aspect >= 2.8 and fill > 0.75:
            return 'I', self.last_mask
        if 0.8 <= aspect <= 1.25 and fill > 0.75:
            return 'O', self.last_mask
        if not (1.2 <= aspect <= 2.1):
            return None, self.last_mask

        sub = np.zeros((2, 3), dtype=np.uint8)
        for r in range(2):
            for c in range(3):
                y0, y1 = int(round(r * bh / 2.0)), int(round((r + 1) * bh / 2.0))
                x0, x1 = int(round(c * bw / 3.0)), int(round((c + 1) * bw / 3.0))
                patch = comp[y0:y1, x0:x1]
                sub[r, c] = 1 if patch.size and patch.mean() > 0.5 else 0
        if int(sub.sum()) != 4:
            return None, self.last_mask
        for name, variants in _NEXT_PATTERNS.items():
            for pat in variants:
                if np.array_equal(sub, pat):
                    return name, self.last_mask
        return None, self.last_mask


class NextTracker:
    """Votes NEXT readings over time and checks them against the pieces that really spawn.

    Timing matters: the NEXT box changes in the SAME frame the new piece first appears.  So
      * "what NEXT promised" is looked up from just BEFORE the piece was first seen (t_first - 0.12 s);
      * votes taken after t_first already describe the NEW next piece and are kept, so the lookahead
        is available almost immediately instead of after a fresh round of votes."""

    def __init__(self, vote_len=6, need=4, post_need=3, trust_window=12, trust_min=0.75):
        self.votes = deque(maxlen=24)            # (t, piece or None)
        self.hist = deque(maxlen=300)            # (t, stable value at that time)
        self.vote_len = vote_len
        self.need = need
        self.post_need = post_need
        self.since = None                        # t_first of the latest spawn
        self.stable = None
        self.history = deque(maxlen=trust_window)   # did "what NEXT promised" match the real spawn?
        self.trust_min = trust_min

    def _stable(self):
        if self.since is None:
            recent = [p for _, p in list(self.votes)[-self.vote_len:] if p]
            need = self.need
        else:
            recent = [p for t, p in self.votes if t >= self.since and p]
            need = self.post_need
        if recent:
            top = max(set(recent), key=recent.count)
            if recent.count(top) >= need:
                return top
        return None

    def push(self, piece, t):
        self.votes.append((t, piece))
        self.stable = self._stable()
        self.hist.append((t, self.stable))

    def on_spawn(self, piece, t, t_first=None):
        t_first = t if t_first is None else t_first
        ref = t_first - 0.12
        expected = None
        for ts, st in reversed(self.hist):
            if ts <= ref:
                expected = st
                break
        if expected is not None:
            self.history.append(expected == piece)
        self.since = t_first
        self.stable = self._stable()
        return expected

    @property
    def trusted(self):
        """Lookahead is only used once the NEXT reader has PROVEN itself: at least 4 spawns where
        the piece it promised was the piece that really spawned, and >= 75 % over the recent
        window.  Until then (or if it never produces a stable reading) the AI searches one piece
        deep, which is safe."""
        if len(self.history) < 4:
            return False
        return (sum(self.history) / float(len(self.history))) >= self.trust_min

    def accuracy(self):
        if not self.history:
            return None
        return sum(self.history) / float(len(self.history))


# ----------------------------------------------------------------------------- frame sources
class CameraSource:
    """Picamera2 (IMX708) with a reader thread so read() always returns the newest frame."""

    def __init__(self, index=0, size=(1280, 720), lens_pos=None, fps=30):
        self.size = size
        self.picam2 = None
        self.cap = None
        self.lock = threading.Lock()
        self.latest = None
        self.stamp = 0.0
        self.running = True
        self.lens_pos = float(lens_pos) if lens_pos is not None else 2.2
        self.use_picam = HAS_PICAMERA2

        if self.use_picam:
            try:
                self.picam2 = Picamera2(camera_num=index)
                cfg = self.picam2.create_video_configuration(
                    sensor={"output_size": (2304, 1296)},          # full 16:9 field of view, no crop
                    main={"size": tuple(size), "format": "RGB888"},  # 'RGB888' is BGR order = OpenCV order
                    buffer_count=3,
                    controls={"FrameDurationLimits": (int(1e6 / fps), int(1e6 / fps))},
                )
                self.picam2.configure(cfg)
                self.picam2.start()
                if lens_pos is not None:
                    self.picam2.set_controls({"AfMode": 0, "LensPosition": self.lens_pos})
                else:
                    self.picam2.set_controls({"AfMode": 2, "AfSpeed": 1})
                time.sleep(1.5)                                   # let exposure / white balance settle ...
                self.picam2.set_controls({"AeEnable": False, "AwbEnable": False})   # ... then lock them
            except Exception as e:
                print("Picamera2 init failed (%s) - falling back to cv2.VideoCapture" % e)
                self.use_picam = False
        if not self.use_picam:
            self.cap = cv2.VideoCapture(index)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            frame = None
            if self.use_picam:
                frame = self.picam2.capture_array()
            else:
                ok, fr = self.cap.read()
                frame = fr if ok else None
            if frame is not None:
                with self.lock:
                    self.latest = frame
                    self.stamp = time.time()
            else:
                time.sleep(0.002)

    def read(self):
        with self.lock:
            if self.latest is None:
                return False, None, 0.0
            return True, self.latest.copy(), self.stamp

    def set_focus(self, lens_pos):
        self.lens_pos = max(0.0, min(10.0, float(lens_pos)))
        if self.use_picam:
            self.picam2.set_controls({"AfMode": 0, "LensPosition": self.lens_pos})

    def autofocus(self):
        if self.use_picam:
            self.picam2.set_controls({"AfMode": 2, "AfSpeed": 1})

    def frame_size(self):
        ok, f, _ = self.read()
        return (f.shape[1], f.shape[0]) if ok else tuple(self.size)

    def release(self):
        self.running = False
        self.thread.join(timeout=0.5)
        if self.use_picam and self.picam2 is not None:
            self.picam2.stop()
            self.picam2.close()
        elif self.cap is not None:
            self.cap.release()


class VideoSource:
    """Replay a recorded video (or the raw capture saved with --record)."""

    def __init__(self, path, step=1, start=0.0, realtime=False):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video: %s" % path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.step = max(1, int(step))
        self.idx = 0
        self.realtime = realtime
        if start > 0:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
            self.idx = int(start * self.fps)
        self._t0 = time.time()
        self._v0 = self.idx / self.fps

    def read(self):
        for _ in range(self.step - 1):
            self.cap.grab()
            self.idx += 1
        ok, frame = self.cap.read()
        if not ok:
            return False, None, 0.0
        t = self.idx / self.fps
        self.idx += 1
        if self.realtime:
            wait = (t - self._v0) - (time.time() - self._t0)
            if wait > 0:
                time.sleep(wait)
        return True, frame, t

    def frame_size(self):
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def release(self):
        self.cap.release()
