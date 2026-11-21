"""
tetris_test.py  --  main loop (camera -> vision -> AI -> solenoids)
4-solenoid mode (LEFT, RIGHT, ROTATE_CW, DROP) with 3x-CW rotation, stale-press cancel,
fixed NEXT lookahead timing, full-width 2-ply search, live fall-speed time budgeting,
and deeper level-scaled DROP (Levels 0-7 only, hard OFF at Level 8+).
"""
import argparse
import json
import statistics
import threading
import time
from collections import deque

import cv2
import numpy as np

from tetris_shapes import SHAPES, rotate_piece
from tetris_vision import (BOARD_COLS, BOARD_ROWS, SNES_SPAWN, BoardReader, CameraSource, NextReader,
                           NextTracker, SpawnTracker, VideoSource, load_calibration, refine_corners)


def format_action(item):
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return "%s(%dms)" % (item[0], int(round(float(item[1]) * 1000.0)))
    return str(item)


class PrintOnlyController:
    """Never touches hardware - just prints the button presses (replay / --no-hw)."""

    def __init__(self, *args, **kwargs):
        self.timings = {
            "dpad_press_s": 0.080,
            "rot_press_s": 0.075,
            "release_s": 0.120,
            "consecutive_same_extra_s": 0.040,
            "drop_hold_s": 0.180,
        }

    def cancel(self):
        pass

    def dispatch_sequence(self, seq):
        print("   [no-hw] ", " -> ".join(format_action(a) for a in seq))

    def is_busy(self):
        return False

    def strike_s(self):
        return 0.080

    def press_period_s(self):
        return 0.240

    def estimate_sequence_s(self, seq):
        if not seq:
            return 0.0
        total = 0.0
        for idx, item in enumerate(seq):
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                name, hold = item[0], float(item[1])
            else:
                name = item
                hold = self.timings["dpad_press_s"] if name in ("LEFT", "RIGHT") else self.timings["rot_press_s"]
            total += hold
            if idx + 1 < len(seq):
                nxt = seq[idx + 1][0] if isinstance(seq[idx + 1], (tuple, list)) else seq[idx + 1]
                wait = self.timings["release_s"]
                if nxt == name:
                    wait += self.timings["consecutive_same_extra_s"]
                total += wait
        return total

    def cleanup(self):
        pass


try:
    from hardware_controller import HardwareController, DEFAULT_BOOST_DUTY_PCT
except ImportError:
    HardwareController = None
    DEFAULT_BOOST_DUTY_PCT = 45.0


# --------------------------------------------------------------------------- AI selection
def make_ai(name):
    import tetris_ai_v2

    def ai(board, piece, nxt, press_fn=None, budget=None):
        mv, sc, _ = tetris_ai_v2.find_best_move(
            board,
            piece,
            next_piece_name=nxt,
            beam_width=999,
            press_fn=press_fn,
            budget=budget,
        )
        return mv, sc

    return ai


# --------------------------------------------------------------------------- button sequencing
def trim_matrix(m):
    rows = [r for r in m if any(r)]
    if not rows:
        return []
    cols = [c for c in range(len(rows[0])) if any(r[c] for r in rows)]
    return [[r[c] for c in cols] for r in rows]


SNES_SPAWN_LEFT_COL_BY_CW_CLICKS = {
    'I': {0: 3, 1: 5, 2: 3, 3: 5},
    'O': {0: 4, 1: 4, 2: 4, 3: 4},
    'T': {0: 4, 1: 4, 2: 4, 3: 5},
    'S': {0: 4, 1: 5, 2: 4, 3: 5},
    'Z': {0: 4, 1: 5, 2: 4, 3: 5},
    'J': {0: 4, 1: 4, 2: 4, 3: 5},
    'L': {0: 4, 1: 4, 2: 4, 3: 5},
}


class ActionSequencer:
    """Turns (piece, AI rotation index, target left column) into solenoid button presses.
    Defaults to force_3cw=True for 4-solenoid hardware (LEFT, RIGHT, ROTATE_CW, DROP)."""

    def __init__(self, force_3cw=True):
        self.spawn_mats = {name: mat for name, (mat, _) in SNES_SPAWN.items()}
        self.force_3cw = force_3cw

    def real_cw_clicks(self, piece, ai_rot):
        target = SHAPES[piece]
        for _ in range(ai_rot):
            target = rotate_piece(target)
        target = trim_matrix(target)
        cur = self.spawn_mats[piece]
        for clicks in range(4):
            if trim_matrix(cur) == target:
                return clicks
            cur = rotate_piece(cur)
        return ai_rot % 4

    def get_split_sequence(self, piece, ai_rot, target_col, observed_spawn_col=None):
        clicks = self.real_cw_clicks(piece, ai_rot)
        if piece in ('I', 'S', 'Z'):
            clicks %= 2
        elif piece == 'O':
            clicks = 0
        table = SNES_SPAWN_LEFT_COL_BY_CW_CLICKS.get(piece, {})
        spawn_col = table.get(clicks, 4)
        if observed_spawn_col is not None:
            spawn_col += observed_spawn_col - table.get(0, 4)

        rot_seq = []
        if clicks == 1:
            rot_seq.append('ROTATE_CW')
        elif clicks == 2:
            rot_seq += ['ROTATE_CW', 'ROTATE_CW']
        elif clicks == 3:
            if self.force_3cw:
                rot_seq += ['ROTATE_CW', 'ROTATE_CW', 'ROTATE_CW']
            else:
                rot_seq.append('ROTATE_CCW')

        dx = target_col - spawn_col
        shift_seq = (['RIGHT'] * dx) if dx > 0 else (['LEFT'] * (-dx))
        return rot_seq, shift_seq, clicks

    def get_full_sequence(self, piece, ai_rot, target_col, observed_spawn_col=None):
        rot_seq, shift_seq, _ = self.get_split_sequence(piece, ai_rot, target_col, observed_spawn_col)
        return rot_seq + shift_seq


# --------------------------------------------------------------------------- fall-speed estimator
class FallSpeedTracker:
    """Measures live natural gravity (sec_per_row). Resets cleanly on new games/banners
    and pauses once DROP engages so soft-drop motion never corrupts the level estimate."""

    MIN_SAMPLES = 4

    def __init__(self, window=8):
        self.samples = deque(maxlen=window)
        self._stack = None
        self._last_row = None
        self._last_t = None
        self._ignore_after_t = None
        self._prev_stack_cells = 0
        self.pieces_since_sample = 0

    def reset(self):
        self.samples.clear()
        self._stack = None
        self._last_row = None
        self._last_t = None
        self._ignore_after_t = None
        self.pieces_since_sample = 0

    def ignore_after(self, t_cutoff):
        """Stop recording gravity samples for the current piece once DROP starts firing."""
        self._ignore_after_t = t_cutoff

    def on_spawn(self, ev):
        cells = int(ev.stack.sum())
        if cells == 0 and self._prev_stack_cells >= 8:
            self.reset()
        self._prev_stack_cells = cells
        self._stack = ev.stack
        self._last_row = min(r for r, _ in ev.cells) if ev.cells else 0
        self._last_t = ev.t
        self._ignore_after_t = None
        self.pieces_since_sample += 1

    def update(self, obs, t):
        if obs.banner:
            if self.samples or self._stack is not None:
                self.reset()
            return
        if self._stack is None:
            return
        if self._ignore_after_t is not None and t >= self._ignore_after_t:
            return
        active = obs.grid & ~self._stack
        if int(active.sum()) != 4:
            return
        rs, _ = np.nonzero(active)
        top_r = int(rs.min())
        if self._last_row is None:
            self._last_row = top_r
            self._last_t = t
            return
        if top_r == self._last_row + 1:
            dt = t - self._last_t
            if 0.045 <= dt <= 1.50:
                self.samples.append(dt)
                self.pieces_since_sample = 0
            self._last_row = top_r
            self._last_t = t
        elif top_r > self._last_row + 1:
            self._last_row = top_r
            self._last_t = t

    @property
    def sec_per_row(self):
        if len(self.samples) < self.MIN_SAMPLES:
            return None
        return float(statistics.median(self.samples))


# --------------------------------------------------------------------------- level-scaled DROP planner
class DropPlanner:
    """Deterministic, safe soft-drop calculator for SNES Tetris & Dr. Mario (Levels 0-7 only).
    - Hard OFF above Level 7 (spr < 0.185 s/row -> never fires).
    - Progressively requires taller open falls + rarer deterministic cooldown as level increases.
    - Stops 1.5 to 2.2 rows BEFORE lock so DROP never bleeds into the next piece.
    - Enforces a hard 0.45 s solenoid hold ceiling.
    """

    SOFT_DROP_SPR = 2.0 / 60.0988   # SNES soft drop is 1/2G = 2 frames/row (0.03328 s/row)
    RELEASE_LAG_S = 0.025           # mechanical plunger spring-retraction cushion
    MIN_HOLD_S = 0.080              # skip trivial micro-drops
    MIN_STACK_TOP_ROW = 6           # never drop if any block is in the top 6 rows (0..5)
    GRAVITY_PROBE_GAP = 6           # skip one drop after 6 pieces with no fresh gravity sample

    # (level, min_spr, every_n_eligible, min_rows_remaining, safety_rows_early)
    LEVEL_TABLE = (
        (0, 0.757, 1,  5.0, 1.5),
        (1, 0.674, 1,  6.0, 1.5),
        (2, 0.591, 2,  7.0, 1.8),
        (3, 0.507, 2,  8.0, 1.8),
        (4, 0.424, 3,  9.0, 2.0),
        (5, 0.341, 4, 10.0, 2.0),
        (6, 0.258, 5, 11.0, 2.2),
        (7, 0.185, 6, 12.0, 2.2),
    )

    def __init__(self, enabled=True, max_hold_s=0.450):
        self.enabled = enabled
        self.max_hold_s = max(self.MIN_HOLD_S, min(0.450, float(max_hold_s)))
        self._eligible_count = 0
        self._last_level = None

    def reset(self):
        self._eligible_count = 0
        self._last_level = None

    def estimate_level(self, spr):
        """Returns estimated level 0..7, or 8 for Level 8+, or None if speed not yet confirmed."""
        if spr is None:
            return None
        for lvl, min_spr, _, _, _ in self.LEVEL_TABLE:
            if spr >= min_spr:
                return lvl
        return 8

    def plan_hold_s(self, board, spr, start_row, land_row, decision_elapsed_s, pre_drop_s, pieces_since_sample):
        """Returns (hold_s_or_None, est_level, reason_str)."""
        est_level = self.estimate_level(spr)
        if not self.enabled:
            return None, est_level, "disabled"
        if spr is None or est_level is None:
            return None, None, "warming_up_spr"
        if est_level > 7:
            return None, est_level, "level_8+_hard_off"
        if land_row is None or land_row < 0:
            return None, est_level, "no_land_row"

        if any(board[r][c] for r in range(self.MIN_STACK_TOP_ROW) for c in range(BOARD_COLS)):
            return None, est_level, "high_stack"

        if pieces_since_sample >= self.GRAVITY_PROBE_GAP:
            return None, est_level, "gravity_probe"

        _, _, every_n, min_rows_left, safety_rows = self.LEVEL_TABLE[est_level]

        t_grav = max(0.0, decision_elapsed_s + pre_drop_s)
        cur_row = float(start_row) + (t_grav / spr) + 0.5
        rows_remaining = float(land_row) - cur_row

        if rows_remaining < min_rows_left:
            return None, est_level, "fall_too_short"

        safe_drop_rows = rows_remaining - safety_rows
        raw_hold = (safe_drop_rows * self.SOFT_DROP_SPR) - self.RELEASE_LAG_S
        if raw_hold < self.MIN_HOLD_S:
            return None, est_level, "hold_too_short"

        if self._last_level != est_level:
            self._eligible_count = 0
            self._last_level = est_level
        self._eligible_count += 1
        if ((self._eligible_count - 1) % every_n) != 0:
            return None, est_level, "cooldown"

        hold_s = round(min(self.max_hold_s, raw_hold), 3)
        return hold_s, est_level, "ok"


# --------------------------------------------------------------------------- the bot
class Bot:
    ALIGN_AFTER = 5
    ALIGN_MIN_STACK = 12
    ALIGN_FRAMES = 5

    def __init__(self, cal, hw, ai_name='v2', use_next='auto', next_wait=0.22, log_path=None, verbose=True,
                 autoalign=True, sync_align=False, force_3cw=True, time_budget='auto',
                 enable_drop=True, max_drop_s=0.450):
        self.board_corners = np.array(cal["board_corners"], dtype=np.float32)
        self.reader = BoardReader(self.board_corners)
        self.autoalign = autoalign
        self.sync_align = sync_align
        self.align_frames = []
        self.align_stack_cells = 0
        self.aligned = False
        self._align_result = None
        self.nreader = NextReader(cal["next_corners"])
        self.tracker = SpawnTracker()
        self.ntrk = NextTracker()
        self.fall = FallSpeedTracker()
        self.drop_planner = DropPlanner(enabled=enable_drop, max_hold_s=max_drop_s)
        self.ai = make_ai(ai_name)
        self.seq = ActionSequencer(force_3cw=force_3cw)
        self.hw = hw
        self.use_next = use_next
        self.next_wait = next_wait
        self.time_budget = time_budget
        self.pending = None
        self.verbose = verbose
        self.log = open(log_path, "a") if log_path else None
        self.last_event = None
        self.last_decision = None
        self.n_pieces = 0
        self.mismatches = 0

    def _next_for_ai(self):
        if self.use_next == 'off':
            return None
        if self.use_next == 'on':
            return self.ntrk.stable
        return self.ntrk.stable if self.ntrk.trusted else None

    def _align_worker(self, frames):
        new, c0, c1 = refine_corners(self.board_corners, frames, rounds=3)
        shift = float(np.abs(new - self.board_corners).max())
        self._align_result = (new, c0, c1, shift)

    def start_align(self, frames):
        if self.sync_align:
            self._align_worker(frames)
        else:
            threading.Thread(target=self._align_worker, args=(list(frames),), daemon=True).start()

    def _apply_align(self):
        new, c0, c1, shift = self._align_result
        self._align_result = None
        ok = c1 < c0 - 0.5 and shift < 10.0
        if ok:
            old = self.reader
            self.reader = BoardReader(new, cell=old.cell)
            self.reader.bg_v = old.bg_v
            self.reader.bg_init = old.bg_init
            self.reader.ref_v = old.ref_v
            self.reader.sat_ref = old.sat_ref
            self.reader.sat_min = old.sat_min
            self.reader.thr_map = old.thr_map
            self.board_corners = new
        if self.verbose:
            print("auto-align: cost %.1f -> %.1f, max corner move %.1f px -> %s" % (
                c0, c1, shift, "applied" if ok else "kept original grid"))

    def process(self, frame, t):
        if self._align_result is not None:
            self._apply_align()
        obs = self.reader.read(frame)
        nxt, nmask = self.nreader.read(frame)
        self.ntrk.push(nxt, t)
        if obs.banner:
            self.drop_planner.reset()
        self.fall.update(obs, t)

        ev = self.tracker.update(obs, t)
        if ev is not None:
            if hasattr(self.hw, "cancel"):
                self.hw.cancel()
            self.fall.on_spawn(ev)
            expected = self.ntrk.on_spawn(ev.piece, t, t_first=ev.t_first)
            if expected is not None and expected != ev.piece:
                self.mismatches += 1
            self.pending = (ev, expected)
            self.last_event = ev
            self.align_frames = (self.align_frames + [frame.copy()])[-self.ALIGN_FRAMES:]
            self.align_stack_cells = int(ev.stack.sum())
            if (self.autoalign and not self.aligned and len(self.align_frames) >= self.ALIGN_AFTER
                    and self.align_stack_cells >= self.ALIGN_MIN_STACK):
                self.aligned = True
                self.start_align(self.align_frames)

        decision = None
        if self.pending is not None:
            ev, expected = self.pending
            spr = self.fall.sec_per_row
            fast_mode = (spr is not None and spr <= 0.25)
            wait_for_next = (not fast_mode) and self.use_next != 'off' and (
                self.use_next == 'on' or self.ntrk.trusted
            )
            if (not wait_for_next) or self.ntrk.stable is not None or (t - ev.t_first) >= self.next_wait:
                decision = self._decide(ev, expected, t)
                self.pending = None
        return obs, nmask, decision

    def _decide(self, ev, expected, t):
        import tetris_ai_v2

        nxt = self._next_for_ai()
        board = ev.stack.astype(int).tolist()
        spr = self.fall.sec_per_row
        stack_high = any(board[r][c] for r in range(11) for c in range(BOARD_COLS))
        elapsed = max(0.0, t - ev.t_first)

        use_budget = False
        budget = None
        press_fn = None
        if self.time_budget == 'on' or (
            self.time_budget == 'auto' and spr is not None and (spr <= 0.30 or stack_high)
        ):
            use_budget = spr is not None
            if use_budget:
                press_period = self.hw.press_period_s() if hasattr(self.hw, "press_period_s") else 0.240
                strike = self.hw.strike_s() if hasattr(self.hw, "strike_s") else 0.080
                budget = {
                    'spr': spr,
                    'elapsed': elapsed,
                    'press_period': press_period,
                    'strike': strike,
                    'margin': 0.08,
                    'spawn_col': ev.left_col,
                }
                press_fn = lambda name, rot, col: len(
                    self.seq.get_full_sequence(name, rot, col, observed_spawn_col=ev.left_col)
                )

        t0 = time.perf_counter()
        move, score = self.ai(board, ev.piece, nxt, press_fn=press_fn, budget=budget)
        ms = (time.perf_counter() - t0) * 1000.0
        self.n_pieces += 1

        drop_hold_s = None
        est_level = self.drop_planner.estimate_level(spr)
        drop_reason = "no_move"

        if move is None:
            seq = []
            if self.verbose:
                print("t=%.1f  %s: NO LEGAL MOVE (board full?)" % (t, ev.piece))
        else:
            rot, col = move
            move_seq = self.seq.get_full_sequence(ev.piece, rot, col, observed_spawn_col=ev.left_col)
            seq = list(move_seq)

            land_row = tetris_ai_v2.get_landing_row(board, ev.piece, rot, col)
            start_row = min(r for r, _ in ev.cells) if ev.cells else 0
            now = time.time()
            decision_elapsed_s = max(0.0, (now - ev.t_first) if ev.t_first > 1e9 else (t - ev.t_first + ms / 1000.0))

            if move_seq:
                move_s = self.hw.estimate_sequence_s(move_seq)
                rel_s = self.hw.timings.get("release_s", 0.120) if hasattr(self.hw, "timings") else 0.120
                pre_drop_s = move_s + rel_s
            else:
                pre_drop_s = 0.0

            drop_hold_s, est_level, drop_reason = self.drop_planner.plan_hold_s(
                board=board,
                spr=spr,
                start_row=start_row,
                land_row=land_row,
                decision_elapsed_s=decision_elapsed_s,
                pre_drop_s=pre_drop_s,
                pieces_since_sample=self.fall.pieces_since_sample,
            )

            if drop_hold_s is not None:
                seq.append(("DROP", drop_hold_s))
                self.fall.ignore_after(t + pre_drop_s - 0.05)

            seq_str = " -> ".join(format_action(a) for a in seq) if seq else "(HOLD SPAWN COLUMN - NO BUTTONS NEEDED)"
            if self.verbose:
                lvl_str = ("L%d" % est_level) if est_level is not None else "L?"
                print("t=%.1f  piece=%s next=%s (%s, NEXT trusted=%s)  move=%s  AI %.1f ms%s\n   %s" % (
                    t, ev.piece, nxt, lvl_str, self.ntrk.trusted, move, ms,
                    ("  [budget spr=%.2fs]" % spr) if use_budget else "",
                    seq_str))
            if seq:
                self.hw.dispatch_sequence(seq)

        planned_s = (
            self.hw.estimate_sequence_s(seq)
            if hasattr(self.hw, "estimate_sequence_s")
            else (max(0.0, (len(seq) - 1) * 0.240 + 0.080) if seq else 0.0)
        )
        now = time.time()
        frame_age_ms = round((now - t) * 1000.0, 1) if t > 1e9 else None

        rec = {
            "t": round(t, 2),
            "piece": ev.piece,
            "spawn_col": ev.left_col,
            "next_used": nxt,
            "next_expected_before_spawn": expected,
            "next_trusted": self.ntrk.trusted,
            "move": list(move) if move else None,
            "seq": [format_action(a) for a in seq],
            "ai_ms": round(ms, 1),
            "first_seen_to_decision_s": round(elapsed + ms / 1000.0, 3),
            "frame_age_ms": frame_age_ms,
            "planned_press_s": round(planned_s, 3),
            "sec_per_row": round(spr, 3) if spr is not None else None,
            "est_level": est_level,
            "drop_hold_s": drop_hold_s,
            "drop_reason": drop_reason,
            "time_budgeted": use_budget,
            "vision_health": dict(self.reader.health),
            "stack": ["".join("#" if v else "." for v in row) for row in ev.stack],
        }
        if self.log:
            self.log.write(json.dumps(rec) + "\n")
            self.log.flush()
        self.last_decision = rec
        return rec


# --------------------------------------------------------------------------- debug drawing
def draw_debug(frame, bot, obs, nmask, fps, show_grid, recording):
    vis = frame.copy()
    bw = bot.reader.warper
    cell = bot.reader.cell
    corners = bw.to_frame([[0, 0], [BOARD_COLS * cell, 0], [BOARD_COLS * cell, BOARD_ROWS * cell], [0, BOARD_ROWS * cell]])
    cv2.polylines(vis, [np.int32(corners)], True, (0, 255, 0), 1)
    if show_grid:
        for c in range(1, BOARD_COLS):
            p = bw.to_frame([[c * cell, 0], [c * cell, BOARD_ROWS * cell]])
            cv2.line(vis, tuple(int(v) for v in p[0]), tuple(int(v) for v in p[1]), (0, 160, 0), 1)
        for r in range(1, BOARD_ROWS):
            p = bw.to_frame([[0, r * cell], [BOARD_COLS * cell, r * cell]])
            cv2.line(vis, tuple(int(v) for v in p[0]), tuple(int(v) for v in p[1]), (0, 160, 0), 1)
    stack = bot.tracker.last_stack
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if obs.grid[r, c]:
                ctr = bw.to_frame([[(c + 0.5) * cell, (r + 0.5) * cell]])[0]
                in_stack = stack is not None and stack[r, c]
                cv2.circle(vis, (int(ctr[0]), int(ctr[1])), 3, (0, 255, 0) if in_stack else (0, 0, 255), -1)
    nw = bot.nreader.warper
    nc = nw.to_frame([[0, 0], [nw.size[0], 0], [nw.size[0], nw.size[1]], [0, nw.size[1]]])
    cv2.polylines(vis, [np.int32(nc)], True, (0, 255, 255), 1)

    ev = bot.last_event
    h = bot.reader.health
    spr = bot.fall.sec_per_row
    est_lvl = bot.drop_planner.estimate_level(spr)
    lvl_txt = ("L%d+" % est_lvl) if (est_lvl is not None and est_lvl >= 8) else (("L%d" % est_lvl) if est_lvl is not None else "L?")
    lines = [
        "FPS %.0f %s | spr %s (%s)" % (fps, "REC" if recording else "", ("%.2fs" % spr) if spr else "-", lvl_txt),
        "banner" if obs.banner else "cells %d | light bg %.0f blk %.0f ctr %.0f%s" % (
            int(obs.grid.sum()), h["bg"], h["ref"], h["contrast"],
            " LOW CONTRAST" if h["contrast"] < 70 else ""),
        "last piece %s | NEXT stable=%s trusted=%s acc=%s" % (
            ev.piece if ev else "-", bot.ntrk.stable, bot.ntrk.trusted,
            ("%.0f%%" % (100 * bot.ntrk.accuracy())) if bot.ntrk.accuracy() is not None else "-"),
    ]
    if bot.last_decision:
        seq_txt = " ".join(bot.last_decision["seq"]) if bot.last_decision["seq"] else "(STAY)"
        lines.append("do: " + seq_txt)
    for i, l in enumerate(lines):
        cv2.putText(vis, l, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return vis


# --------------------------------------------------------------------------- main
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["camera", "video"], default="camera")
    ap.add_argument("--video")
    ap.add_argument("--step", type=int, default=1, help="process every Nth frame of --video")
    ap.add_argument("--realtime", action="store_true", help="play --video at real speed")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--size", default="1280x720", help="camera capture size; must match calibrate.py")
    ap.add_argument("--fps", type=int, default=30, help="camera capture fps")
    ap.add_argument("--solenoids", action="store_true", help="use real solenoid timings (default: LED timings)")
    ap.add_argument("--boost-duty", type=float, default=DEFAULT_BOOST_DUTY_PCT)
    ap.add_argument("--strike-ms", type=float, default=None, help="override solenoid strike duration in ms")
    ap.add_argument("--recharge-ms", type=float, default=None, help="override solenoid release delay in ms")
    ap.add_argument("--no-drop", action="store_true", help="disable the DROP solenoid completely")
    ap.add_argument("--max-drop-ms", type=float, default=450.0, help="hard max DROP hold duration in ms (max 450)")
    ap.add_argument("--no-hw", action="store_true", help="never touch hardware (print button presses)")
    ap.add_argument("--ai", choices=["v2", "legacy"], default="v2")
    ap.add_argument("--use-next", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--time-budget", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--force-3cw", action="store_true", default=True,
                    help="always use 3x ROTATE_CW instead of ROTATE_CCW (default: enabled)")
    ap.add_argument("--record", nargs="?", const="auto", default=None, help="save raw frames (no overlay)")
    ap.add_argument("--log", default="decisions.jsonl")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--no-autoalign", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--grid", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.source == "video":
        if not args.video:
            raise SystemExit("--video is required with --source video")
        src = VideoSource(args.video, step=args.step, realtime=args.realtime)
    else:
        size = tuple(int(v) for v in args.size.lower().split("x"))
        src = CameraSource(index=args.camera_index, size=size, fps=args.fps)
        time.sleep(0.5)

    frame_size = src.frame_size()
    cal = load_calibration(args.calibration, frame_size=frame_size) if args.calibration else \
        load_calibration(frame_size=frame_size)
    print("capture size", frame_size, "| AI:", args.ai, "| use-next:", args.use_next,
          "| budget:", args.time_budget, "| drop:", "OFF" if args.no_drop else ("<=%dms" % int(args.max_drop_ms)))

    if args.no_hw or args.source == "video":
        hw = PrintOnlyController()
    elif HardwareController is None:
        print("\n*** WARNING: hardware_controller.py not found - button presses will only be PRINTED. ***\n")
        hw = PrintOnlyController()
    else:
        hw = HardwareController(use_solenoid_timings=args.solenoids, boost_duty_pct=args.boost_duty)
        if args.strike_ms is not None:
            hw.timings["dpad_press_s"] = args.strike_ms / 1000.0
            hw.timings["rot_press_s"] = args.strike_ms / 1000.0
        if args.recharge_ms is not None:
            hw.timings["release_s"] = args.recharge_ms / 1000.0

    bot = Bot(cal, hw, ai_name=args.ai, use_next=args.use_next, log_path=args.log,
              autoalign=not args.no_autoalign, sync_align=(args.source == "video"),
              force_3cw=args.force_3cw, time_budget=args.time_budget,
              enable_drop=not args.no_drop, max_drop_s=args.max_drop_ms / 1000.0)

    show_grid = args.grid
    writer = None
    rec_on = args.record is not None
    last_stamp = None
    fps_hist = deque(maxlen=30)
    learn_frames = None
    try:
        while True:
            ok, frame, t = src.read()
            if not ok:
                if args.source == "video":
                    break
                time.sleep(0.005)
                continue
            if args.source == "camera":
                if t == last_stamp:
                    time.sleep(0.002)
                    continue
                last_stamp = t

            if rec_on:
                if writer is None:
                    path = args.record if args.record not in (None, "auto") else time.strftime("raw_%Y%m%d_%H%M%S.avi")
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
                    print("recording raw frames to", path)
                writer.write(frame)

            if learn_frames is not None:
                learn_frames.append(frame.copy())
                if len(learn_frames) >= 10:
                    bot.reader.learn_empty(learn_frames)
                    learn_frames = None
                    print("Empty-board background learned.")

            t_loop = time.perf_counter()
            obs, nmask, decision = bot.process(frame, t)
            fps_hist.append(time.perf_counter() - t_loop)

            if not args.headless:
                fps = 1.0 / max(1e-6, sum(fps_hist) / len(fps_hist))
                vis = draw_debug(frame, bot, obs, nmask, fps, show_grid, writer is not None)
                cv2.imshow("Tetris bot", vis)
                cv2.imshow("board", cv2.resize(obs.warp, None, fx=0.6, fy=0.6))
                if nmask is not None:
                    cv2.imshow("next mask", nmask)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                elif k == ord('g'):
                    show_grid = not show_grid
                elif k == ord('k') and len(bot.align_frames) >= 2:
                    bot.start_align(bot.align_frames)
                    print("Re-aligning the grid to the blocks on screen...")
                elif k == ord('b'):
                    learn_frames = []
                    print("Learning empty-board background... keep the board EMPTY.")
                elif k == ord('r'):
                    rec_on = not rec_on
                    if not rec_on and writer is not None:
                        writer.release()
                        writer = None
                elif k in (ord('+'), ord('=')) and args.source == "camera":
                    src.set_focus(src.lens_pos + 0.25)
                elif k == ord('-') and args.source == "camera":
                    src.set_focus(src.lens_pos - 0.25)
                elif k == ord('a') and args.source == "camera":
                    src.autofocus()
                elif k == ord('t'):
                    hw.dispatch_sequence(['LEFT', 'RIGHT', 'ROTATE_CW', ('DROP', 0.15)])
    finally:
        if writer is not None:
            writer.release()
        src.release()
        try:
            hw.cleanup()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print("pieces handled: %d | NEXT-box mismatches: %d" % (bot.n_pieces, bot.mismatches))


if __name__ == "__main__":
    main()
