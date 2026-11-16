"""
tetris_test.py  --  main loop (camera -> vision -> AI -> solenoids)

    python3 tetris_test.py                          # live camera, solenoids OFF (LED timings) by default
    python3 tetris_test.py --solenoids              # real solenoid timings
    python3 tetris_test.py --record                 # also save RAW camera frames (no overlay) to raw_*.avi
    python3 tetris_test.py --source video --video raw_20261115_101500.avi --no-hw --headless
                                                    # replay a recording through the exact same pipeline

Keys (live window):  q quit | k re-run grid auto-align | b learn EMPTY-board background | g toggle grid
                     r toggle raw recording | + / - / a focus | t test the hardware sequence

Pipeline per frame
    BoardReader (4-corner warp, exact 10x20 grid)  ->  SpawnTracker (exact tetromino match)
    NextReader  ->  NextTracker (votes, and only trusted after it matches real spawns)
    spawn event -> AI (tetris_ai_v2, 1-piece search unless NEXT is trusted) -> ActionSequencer -> hardware
"""
import argparse
import json
import threading
import time
from collections import deque

import cv2
import numpy as np

from tetris_shapes import SHAPES, rotate_piece
from tetris_vision import (BOARD_COLS, BOARD_ROWS, SNES_SPAWN, BoardReader, CameraSource, NextReader,
                           NextTracker, SpawnTracker, VideoSource, load_calibration, refine_corners)

class PrintOnlyController:
    """Never touches hardware - just prints the button presses (replay / --no-hw)."""

    def __init__(self, *args, **kwargs):
        pass

    def dispatch_sequence(self, seq):
        print("   [no-hw] ", " -> ".join(seq))

    def cleanup(self):
        pass


try:
    from hardware_controller import HardwareController
except ImportError:
    HardwareController = None


# --------------------------------------------------------------------------- AI selection
CHAMPION_DNA = {'lines_1': -250.0, 'lines_2': -80.0, 'lines_3': 100.0, 'lines_4': 12000.0,
                'height': 75.0, 'holes': 25000.0, 'bumpiness': 320.0, 'wells': 300.0}
CLEANUP_DNA = {'lines_1': 900.0, 'lines_2': 2000.0, 'lines_3': 3500.0, 'lines_4': 12000.0,
               'height': 140.0, 'holes': 20000.0, 'bumpiness': 240.0, 'wells': 60.0}


def make_ai(name):
    """Returns f(board_rows_of_ints, piece, next_piece_or_None) -> ((rot, col), score) or (None, score)"""
    if name == 'legacy':
        from tetris_ai_search import find_best_move as legacy_find

        def ai(board, piece, nxt):
            top = any(1 in r for r in board[:8])
            dna = CLEANUP_DNA if top else CHAMPION_DNA
            beam = 4 if any(1 in r for r in board[:10]) else 12
            mv, sc, _ = legacy_find(board, piece, nxt, dna, beam_width=beam)
            return mv, sc
        return ai

    import tetris_ai_v2

    def ai(board, piece, nxt):
        mv, sc, _ = tetris_ai_v2.find_best_move(board, piece, nxt)
        return mv, sc
    return ai


# --------------------------------------------------------------------------- button sequencing
def trim_matrix(m):
    rows = [r for r in m if any(r)]
    if not rows:
        return []
    cols = [c for c in range(len(rows[0])) if any(r[c] for r in rows)]
    return [[r[c] for c in cols] for r in rows]


# Leftmost column of the piece on the SNES 10-wide board AFTER rotating `r` times CW from SNES spawn
# (copied unchanged from your original tetris_test.py):
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
    """Turns (piece, AI rotation index, target left column) into button presses.
    Same tables as before, plus: if vision SAW the piece spawn one column away from the table
    value (e.g. it had already started moving), all columns are shifted by that difference."""

    def __init__(self):
        self.spawn_mats = {name: mat for name, (mat, _) in SNES_SPAWN.items()}

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

    def get_full_sequence(self, piece, ai_rot, target_col, observed_spawn_col=None):
        clicks = self.real_cw_clicks(piece, ai_rot)
        if piece in ('I', 'S', 'Z'):
            clicks %= 2
        elif piece == 'O':
            clicks = 0
        table = SNES_SPAWN_LEFT_COL_BY_CW_CLICKS.get(piece, {})
        spawn_col = table.get(clicks, 4)
        if observed_spawn_col is not None:
            spawn_col += observed_spawn_col - table.get(0, 4)

        seq = []
        if clicks == 1:
            seq.append('ROTATE_CW')
        elif clicks == 2:
            seq += ['ROTATE_CW', 'ROTATE_CW']
        elif clicks == 3:
            seq.append('ROTATE_CCW')
        dx = target_col - spawn_col
        seq += (['RIGHT'] * dx) if dx > 0 else (['LEFT'] * (-dx))
        seq.append('DROP')
        return seq


# --------------------------------------------------------------------------- the bot
class Bot:
    ALIGN_AFTER = 5          # run the one-time grid fine-alignment after this many spawns ...
    ALIGN_MIN_STACK = 12     # ... and only once the stack has at least this many cells (enough blocks to align on)
    ALIGN_FRAMES = 5         # number of newest spawn frames used (5/5 tested best; 2-4 can mis-align)

    def __init__(self, cal, hw, ai_name='v2', use_next='auto', next_wait=0.45, log_path=None, verbose=True,
                 autoalign=True, sync_align=False):
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
        self.ai = make_ai(ai_name)
        self.seq = ActionSequencer()
        self.hw = hw
        self.use_next = use_next
        self.next_wait = next_wait
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
        return self.ntrk.stable if self.ntrk.trusted else None      # auto

    # ---- one-time fine alignment of the grid (see refine_corners in tetris_vision.py) -------------
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

        ev = self.tracker.update(obs, t)
        if ev is not None:
            expected = self.ntrk.on_spawn(ev.piece, t)
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
            wait_for_next = self.use_next != 'off' and (self.use_next == 'on' or self.ntrk.trusted)
            ev, expected = self.pending
            if (not wait_for_next) or self.ntrk.stable is not None or (t - ev.t) >= self.next_wait:
                decision = self._decide(ev, expected, t)
                self.pending = None
        return obs, nmask, decision

    def _decide(self, ev, expected, t):
        nxt = self._next_for_ai()
        board = ev.stack.astype(int).tolist()
        t0 = time.perf_counter()
        move, score = self.ai(board, ev.piece, nxt)
        ms = (time.perf_counter() - t0) * 1000.0
        self.n_pieces += 1
        if move is None:
            seq = []
            if self.verbose:
                print("t=%.1f  %s: NO LEGAL MOVE (board full?)" % (t, ev.piece))
        else:
            rot, col = move
            seq = self.seq.get_full_sequence(ev.piece, rot, col, observed_spawn_col=ev.left_col)
            if self.verbose:
                print("t=%.1f  piece=%s next=%s (NEXT trusted=%s acc=%s)  move=%s  AI %.1f ms\n   %s" % (
                    t, ev.piece, nxt, self.ntrk.trusted, self.ntrk.accuracy(), move, ms, " -> ".join(seq)))
            self.hw.dispatch_sequence(seq)
        rec = {"t": round(t, 2), "piece": ev.piece, "spawn_col": ev.left_col, "next_used": nxt,
               "next_expected_before_spawn": expected, "next_trusted": self.ntrk.trusted,
               "move": list(move) if move else None, "seq": seq, "ai_ms": round(ms, 1),
               "stack": ["".join("#" if v else "." for v in row) for row in ev.stack]}
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
    lines = ["FPS %.0f %s" % (fps, "REC" if recording else ""),
             "banner" if obs.banner else "cells %d" % int(obs.grid.sum()),
             "last piece %s | NEXT stable=%s trusted=%s acc=%s" % (
                 ev.piece if ev else "-", bot.ntrk.stable, bot.ntrk.trusted, bot.ntrk.accuracy())]
    if bot.last_decision:
        lines.append("do: " + " ".join(bot.last_decision["seq"]))
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
    ap.add_argument("--solenoids", action="store_true", help="use real solenoid timings (default: LED timings)")
    ap.add_argument("--no-hw", action="store_true", help="never touch hardware (print button presses)")
    ap.add_argument("--ai", choices=["v2", "legacy"], default="v2")
    ap.add_argument("--use-next", choices=["auto", "on", "off"], default="auto",
                    help="auto = use the NEXT box only after it has proven itself against real spawns")
    ap.add_argument("--record", nargs="?", const="auto", default=None, help="save raw frames (no overlay)")
    ap.add_argument("--log", default="decisions.jsonl")
    ap.add_argument("--calibration", default=None, help="calibration file (default: calibration.json next to the scripts)")
    ap.add_argument("--no-autoalign", action="store_true", help="skip the one-time automatic grid fine-alignment")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--grid", action="store_true", help="draw the 10x20 grid on the live view")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.source == "video":
        if not args.video:
            raise SystemExit("--video is required with --source video")
        src = VideoSource(args.video, step=args.step, realtime=args.realtime)
    else:
        size = tuple(int(v) for v in args.size.lower().split("x"))
        src = CameraSource(index=args.camera_index, size=size)
        time.sleep(0.5)

    frame_size = src.frame_size()
    cal = load_calibration(args.calibration, frame_size=frame_size) if args.calibration else \
        load_calibration(frame_size=frame_size)
    print("capture size", frame_size, "| AI:", args.ai, "| use-next:", args.use_next)

    if args.no_hw or args.source == "video":
        hw = PrintOnlyController()
    elif HardwareController is None:
        print("\n*** WARNING: hardware_controller.py not found - button presses will only be PRINTED. ***\n")
        hw = PrintOnlyController()
    else:
        hw = HardwareController(use_solenoid_timings=args.solenoids)
    bot = Bot(cal, hw, ai_name=args.ai, use_next=args.use_next, log_path=args.log,
              autoalign=not args.no_autoalign, sync_align=(args.source == "video"))

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
                writer.write(frame)                       # RAW frame, before any drawing

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
                    hw.dispatch_sequence(['LEFT', 'RIGHT', 'ROTATE_CW', 'DROP'])
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
            pass                                     # headless OpenCV build
        print("pieces handled: %d | NEXT-box mismatches: %d" % (bot.n_pieces, bot.mismatches))


if __name__ == "__main__":
    main()
