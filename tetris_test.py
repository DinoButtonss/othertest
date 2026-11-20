"""
tetris_test.py  --  main loop (camera -> vision -> AI -> solenoids)
Includes zero-false-positive vision verification for ALL buttons (ROTATE_CW, ROTATE_CCW, LEFT, RIGHT).
"""
import argparse
import json
import threading
import time
from collections import deque

import cv2
import numpy as np

from tetris_shapes import SHAPES, rotate_piece"""
tetris_test.py  --  main loop (camera -> vision -> AI -> solenoids)
Fast single-burst execution with a dedicated CCW-only camera failsafe (3x ROTATE_CW backup).
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

    def is_busy(self):
        return False

    def cleanup(self):
        pass


try:
    from hardware_controller import HardwareController, DEFAULT_BOOST_DUTY_PCT
except ImportError:
    HardwareController = None
    DEFAULT_BOOST_DUTY_PCT = 45.0


# --------------------------------------------------------------------------- AI selection
CHAMPION_DNA = {
    'lines_1': -4500.0,
    'lines_2': -1500.0,
    'lines_3': 2000.0,
    'lines_4': 45000.0,
    'height': 85.0,
    'holes': 25000.0,
    'bumpiness': 320.0,
    'wells': 950.0,
}
CLEANUP_DNA = {
    'lines_1': 600.0,
    'lines_2': 1800.0,
    'lines_3': 4500.0,
    'lines_4': 30000.0,
    'height': 140.0,
    'holes': 20000.0,
    'bumpiness': 240.0,
    'wells': 300.0,
}


def make_ai(name):
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


def _mat_to_rel(m):
    t = trim_matrix(m)
    return frozenset((r, c) for r, row in enumerate(t) for c, v in enumerate(row) if v)


SNES_ROT_SHAPES = {}
for _name, (_mat, _) in SNES_SPAWN.items():
    _cur = _mat
    _states = {}
    for _clicks in range(4):
        _rel = _mat_to_rel(_cur)
        if _rel not in _states:
            _states[_rel] = _clicks
        _cur = rotate_piece(_cur)
    SNES_ROT_SHAPES[_name] = _states


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
    """Turns (piece, AI rotation index, target left column) into solenoid button presses."""

    def __init__(self, force_3cw=False):
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


# --------------------------------------------------------------------------- the bot
class Bot:
    ALIGN_AFTER = 5
    ALIGN_MIN_STACK = 12
    ALIGN_FRAMES = 5

    SETTLE_S = 0.08
    VERIFY_TIMEOUT_S = 0.18
    CONFIRM_FRAMES = 2

    def __init__(self, cal, hw, ai_name='v2', use_next='auto', next_wait=0.45, log_path=None, verbose=True,
                 autoalign=True, sync_align=False, force_3cw=False, ccw_failsafe=True):
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
        self.seq = ActionSequencer(force_3cw=force_3cw)
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
        self.ccw_failsafe = ccw_failsafe
        self._ccw_state = None

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
            self.reader.thr_map = old.thr_map
            self.board_corners = new
        if self.verbose:
            print("auto-align: cost %.1f -> %.1f, max corner move %.1f px -> %s" % (
                c0, c1, shift, "applied" if ok else "kept original grid"))

    def _observe_active_rotation(self, obs, stack, piece):
        active = obs.grid & ~stack
        sub = active[:15, :]
        if int(sub.sum()) != 4:
            return None
        n_lab, _ = cv2.connectedComponents(sub.astype(np.uint8), connectivity=4)
        if n_lab != 2:
            return None
        rs, cs = np.nonzero(sub)
        r0, c0 = int(rs.min()), int(cs.min())
        rel = frozenset((int(r - r0), int(c - c0)) for r, c in zip(rs, cs))
        return SNES_ROT_SHAPES.get(piece, {}).get(rel, None)

    def _update_ccw_failsafe(self, obs, t):
        """Dedicated CCW-only failsafe: if ROTATE_CCW didn't register (piece is still at rot=0
        for 2 consecutive frames), fire 3x ROTATE_CW + shifts."""
        st = self._ccw_state
        if st is None:
            return
        hw_busy = self.hw.is_busy() if hasattr(self.hw, "is_busy") else False

        if st["phase"] == "WAIT_CCW":
            if hw_busy:
                st["done_t"] = None
                return
            if st["done_t"] is None:
                st["done_t"] = t
            if (t - st["done_t"]) >= self.SETTLE_S:
                st["phase"] = "VERIFY_CCW"
                st["verify_start_t"] = t
            return

        obs_clicks = self._observe_active_rotation(obs, st["stack"], st["piece"])

        # CCW succeeded (piece is now at rot=3) -> send shifts right away!
        if obs_clicks == 3:
            if st["shift_seq"]:
                self.hw.dispatch_sequence(st["shift_seq"])
            self._ccw_state = None
            return

        # CCW missed (piece is STILL at spawn rot=0) -> confirm 2 frames, then fire 3x ROTATE_CW
        if obs_clicks == 0:
            st["zero_count"] += 1
            if st["zero_count"] >= self.CONFIRM_FRAMES:
                recovery = ['ROTATE_CW', 'ROTATE_CW', 'ROTATE_CW'] + st["shift_seq"]
                if self.verbose:
                    print("   [CCW failsafe] %s stayed at rot=0 -> firing 3x ROTATE_CW + shifts" % st["piece"])
                self.hw.dispatch_sequence(recovery)
                self._ccw_state = None
                return
        else:
            st["zero_count"] = 0

        # Unclear frame or timeout -> never risk a false retry, just send shifts
        if (t - st["verify_start_t"]) >= self.VERIFY_TIMEOUT_S:
            if st["shift_seq"]:
                self.hw.dispatch_sequence(st["shift_seq"])
            self._ccw_state = None

    def process(self, frame, t):
        if self._align_result is not None:
            self._apply_align()
        obs = self.reader.read(frame)
        nxt, nmask = self.nreader.read(frame)
        self.ntrk.push(nxt, t)
        if self.ccw_failsafe:
            self._update_ccw_failsafe(obs, t)

        ev = self.tracker.update(obs, t)
        if ev is not None:
            self._ccw_state = None
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
            rot_seq, shift_seq, target_clicks = self.seq.get_split_sequence(
                ev.piece, rot, col, observed_spawn_col=ev.left_col
            )
            seq = rot_seq + shift_seq
            seq_str = " -> ".join(seq) if seq else "(HOLD SPAWN COLUMN - NO BUTTONS NEEDED)"
            if self.verbose:
                print("t=%.1f  piece=%s next=%s (NEXT trusted=%s acc=%s)  move=%s  AI %.1f ms\n   %s" % (
                    t, ev.piece, nxt, self.ntrk.trusted, self.ntrk.accuracy(), move, ms, seq_str))

            # ONLY split & verify when ROTATE_CCW is used; all other moves fire in one fast burst!
            if self.ccw_failsafe and rot_seq == ['ROTATE_CCW']:
                self.hw.dispatch_sequence(['ROTATE_CCW'])
                self._ccw_state = {
                    "phase": "WAIT_CCW",
                    "piece": ev.piece,
                    "stack": ev.stack,
                    "shift_seq": shift_seq,
                    "done_t": None,
                    "verify_start_t": None,
                    "zero_count": 0,
                }
            elif seq:
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
    ap.add_argument("--solenoids", action="store_true", help="use real solenoid timings (default: LED timings)")
    ap.add_argument("--boost-duty", type=float, default=DEFAULT_BOOST_DUTY_PCT,
                    help="boost converter PWM duty cycle %%")
    ap.add_argument("--no-hw", action="store_true", help="never touch hardware (print button presses)")
    ap.add_argument("--ai", choices=["v2", "legacy"], default="v2")
    ap.add_argument("--use-next", choices=["auto", "on", "off"], default="auto",
                    help="auto = use the NEXT box only after it has proven itself against real spawns")
    ap.add_argument("--force-3cw", action="store_true",
                    help="always use 3x ROTATE_CW instead of ROTATE_CCW (skips CCW completely)")
    ap.add_argument("--no-ccw-failsafe", action="store_true",
                    help="disable the CCW camera failsafe")
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
        hw = HardwareController(use_solenoid_timings=args.solenoids, boost_duty_pct=args.boost_duty)
    bot = Bot(cal, hw, ai_name=args.ai, use_next=args.use_next, log_path=args.log,
              autoalign=not args.no_autoalign, sync_align=(args.source == "video"),
              force_3cw=args.force_3cw, ccw_failsafe=not args.no_ccw_failsafe)

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
                    hw.dispatch_sequence(['LEFT', 'RIGHT', 'ROTATE_CW', 'ROTATE_CCW'])
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
from tetris_vision import (BOARD_COLS, BOARD_ROWS, SNES_SPAWN, BoardReader, CameraSource, NextReader,
                           NextTracker, SpawnTracker, VideoSource, load_calibration, refine_corners)


class PrintOnlyController:
    """Never touches hardware - just prints the button presses (replay / --no-hw)."""

    def __init__(self, *args, **kwargs):
        pass

    def dispatch_sequence(self, seq):
        print("   [no-hw] ", " -> ".join(seq))

    def is_busy(self):
        return False

    def cleanup(self):
        pass


try:
    from hardware_controller import HardwareController, DEFAULT_BOOST_DUTY_PCT
except ImportError:
    HardwareController = None
    DEFAULT_BOOST_DUTY_PCT = 45.0


# --------------------------------------------------------------------------- AI selection
CHAMPION_DNA = {
    'lines_1': -4500.0,
    'lines_2': -1500.0,
    'lines_3': 2000.0,
    'lines_4': 45000.0,
    'height': 85.0,
    'holes': 25000.0,
    'bumpiness': 320.0,
    'wells': 950.0,
}
CLEANUP_DNA = {
    'lines_1': 600.0,
    'lines_2': 1800.0,
    'lines_3': 4500.0,
    'lines_4': 30000.0,
    'height': 140.0,
    'holes': 20000.0,
    'bumpiness': 240.0,
    'wells': 300.0,
}


def make_ai(name):
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


def _mat_to_rel(m):
    t = trim_matrix(m)
    return frozenset((r, c) for r, row in enumerate(t) for c, v in enumerate(row) if v)


# Precompute normalized 4-cell shapes for every CW click state (0..3) of every piece
SNES_ROT_SHAPES = {}
for _name, (_mat, _) in SNES_SPAWN.items():
    _cur = _mat
    _states = {}
    for _clicks in range(4):
        _rel = _mat_to_rel(_cur)
        if _rel not in _states:
            _states[_rel] = _clicks
        _cur = rotate_piece(_cur)
    SNES_ROT_SHAPES[_name] = _states


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
    """Turns (piece, AI rotation index, target left column) into solenoid button presses."""

    def __init__(self, ccw_failsafe=False):
        self.spawn_mats = {name: mat for name, (mat, _) in SNES_SPAWN.items()}
        self.ccw_failsafe = ccw_failsafe

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
        """Returns (rot_seq, shift_seq, target_clicks) so rotations and shifts can both be verified."""
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
            if self.ccw_failsafe:
                rot_seq += ['ROTATE_CW', 'ROTATE_CW', 'ROTATE_CW']
            else:
                rot_seq.append('ROTATE_CCW')

        dx = target_col - spawn_col
        shift_seq = (['RIGHT'] * dx) if dx > 0 else (['LEFT'] * (-dx))
        return rot_seq, shift_seq, clicks

    def get_full_sequence(self, piece, ai_rot, target_col, observed_spawn_col=None):
        rot_seq, shift_seq, _ = self.get_split_sequence(piece, ai_rot, target_col, observed_spawn_col)
        return rot_seq + shift_seq


# --------------------------------------------------------------------------- the bot
class Bot:
    ALIGN_AFTER = 5
    ALIGN_MIN_STACK = 12
    ALIGN_FRAMES = 5

    SETTLE_S = 0.10          # wait after solenoids finish before reading camera
    VERIFY_TIMEOUT_S = 0.22  # max time to wait for a clean 4-cell read before skipping retry
    CONFIRM_FRAMES = 2       # require 2 consecutive frames of the SAME mismatch before retrying

    def __init__(self, cal, hw, ai_name='v2', use_next='auto', next_wait=0.45, log_path=None, verbose=True,
                 autoalign=True, sync_align=False, ccw_failsafe=False):
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
        self.seq = ActionSequencer(ccw_failsafe=ccw_failsafe)
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
        self._move_state = None

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
            self.reader.thr_map = old.thr_map
            self.board_corners = new
        if self.verbose:
            print("auto-align: cost %.1f -> %.1f, max corner move %.1f px -> %s" % (
                c0, c1, shift, "applied" if ok else "kept original grid"))

    def _observe_active_piece(self, obs, stack, piece):
        """Returns (obs_clicks, obs_left_col) ONLY if there is an exact, unambiguous
        4-cell connected falling piece matching a known orientation of `piece` in rows 0..14.
        Otherwise returns (None, None)."""
        active = obs.grid & ~stack
        sub = active[:15, :]
        if int(sub.sum()) != 4:
            return None, None
        n_lab, _ = cv2.connectedComponents(sub.astype(np.uint8), connectivity=4)
        if n_lab != 2:  # must be a single 4-connected piece
            return None, None
        rs, cs = np.nonzero(sub)
        r0, c0 = int(rs.min()), int(cs.min())
        rel = frozenset((int(r - r0), int(c - c0)) for r, c in zip(rs, cs))
        clicks = SNES_ROT_SHAPES.get(piece, {}).get(rel, None)
        if clicks is None:
            return None, None
        if piece in ('I', 'S', 'Z'):
            clicks %= 2
        return clicks, c0

    def _start_shift_phase(self, st, shift_seq):
        if not shift_seq:
            self._move_state = None
            return
        self.hw.dispatch_sequence(shift_seq)
        st["phase"] = "WAIT_SHIFT"
        st["done_t"] = None
        st["verify_start_t"] = None
        st["wrong_obs"] = None
        st["wrong_count"] = 0

    def _update_failsafe(self, obs, t):
        """Zero-false-positive verification state machine for ROTATE + LEFT/RIGHT."""
        st = self._move_state
        if st is None:
            return

        hw_busy = self.hw.is_busy() if hasattr(self.hw, "is_busy") else False

        # --- Wait for current solenoid burst to finish + screen settle ---
        if st["phase"] in ("WAIT_ROT", "WAIT_SHIFT"):
            if hw_busy:
                st["done_t"] = None
                return
            if st["done_t"] is None:
                st["done_t"] = t
            if (t - st["done_t"]) >= self.SETTLE_S:
                st["phase"] = "VERIFY_ROT" if st["phase"] == "WAIT_ROT" else "VERIFY_SHIFT"
                st["verify_start_t"] = t
                st["wrong_obs"] = None
                st["wrong_count"] = 0
            return

        obs_clicks, obs_col = self._observe_active_piece(obs, st["stack"], st["piece"])

        # --- Stage 1: Verify Rotation ---
        if st["phase"] == "VERIFY_ROT":
            target_rot = st["target_clicks"]

            if obs_clicks == target_rot:
                # Rotation succeeded! Compute exact shifts from the ACTUAL observed column
                dx = st["target_col"] - obs_col
                exact_shifts = (['RIGHT'] * dx) if dx > 0 else (['LEFT'] * (-dx))
                self._start_shift_phase(st, exact_shifts)
                return

            if obs_clicks is not None and obs_clicks != target_rot:
                if st["wrong_obs"] == obs_clicks:
                    st["wrong_count"] += 1
                else:
                    st["wrong_obs"] = obs_clicks
                    st["wrong_count"] = 1

                if st["wrong_count"] >= self.CONFIRM_FRAMES:
                    mod = 2 if st["piece"] in ('I', 'S', 'Z') else 4
                    delta = (target_rot - obs_clicks) % mod
                    retry_rot = ['ROTATE_CW'] * delta
                    if self.verbose:
                        print("   [ROT failsafe] %s at rot=%d (want %d) -> firing %s + shifts" % (
                            st["piece"], obs_clicks, target_rot, " -> ".join(retry_rot)))
                    # Fire recovery rotations + planned shifts, then verify final column!
                    self._start_shift_phase(st, retry_rot + st["shift_seq"])
                    return
            else:
                st["wrong_obs"] = None
                st["wrong_count"] = 0

            if (t - st["verify_start_t"]) >= self.VERIFY_TIMEOUT_S:
                # Unclear camera frame -> DO NOT retry rotation, just proceed to shifts
                self._start_shift_phase(st, st["shift_seq"])
            return

        # --- Stage 2: Verify Horizontal Column (LEFT / RIGHT) ---
        if st["phase"] == "VERIFY_SHIFT":
            target_col = st["target_col"]

            if obs_col == target_col:
                # Landed on the exact target column!
                self._move_state = None
                return

            if obs_col is not None and obs_col != target_col:
                if st["wrong_obs"] == obs_col:
                    st["wrong_count"] += 1
                else:
                    st["wrong_obs"] = obs_col
                    st["wrong_count"] = 1

                if st["wrong_count"] >= self.CONFIRM_FRAMES:
                    dx = target_col - obs_col
                    retry_shift = (['RIGHT'] * dx) if dx > 0 else (['LEFT'] * (-dx))
                    if self.verbose:
                        print("   [SHIFT failsafe] %s at col=%d (want %d) -> firing %s" % (
                            st["piece"], obs_col, target_col, " -> ".join(retry_shift)))
                    self.hw.dispatch_sequence(retry_shift)
                    # Max 1 shift retry per piece so it can never loop
                    self._move_state = None
                    return
            else:
                st["wrong_obs"] = None
                st["wrong_count"] = 0

            if (t - st["verify_start_t"]) >= self.VERIFY_TIMEOUT_S:
                # Unclear camera frame -> DO NOT retry
                self._move_state = None

    def process(self, frame, t):
        if self._align_result is not None:
            self._apply_align()
        obs = self.reader.read(frame)
        nxt, nmask = self.nreader.read(frame)
        self.ntrk.push(nxt, t)
        self._update_failsafe(obs, t)

        ev = self.tracker.update(obs, t)
        if ev is not None:
            self._move_state = None
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
            rot_seq, shift_seq, target_clicks = self.seq.get_split_sequence(
                ev.piece, rot, col, observed_spawn_col=ev.left_col
            )
            seq = rot_seq + shift_seq
            seq_str = " -> ".join(seq) if seq else "(HOLD SPAWN COLUMN - NO BUTTONS NEEDED)"
            if self.verbose:
                print("t=%.1f  piece=%s next=%s (NEXT trusted=%s acc=%s)  move=%s  AI %.1f ms\n   %s" % (
                    t, ev.piece, nxt, self.ntrk.trusted, self.ntrk.accuracy(), move, ms, seq_str))

            base_state = {
                "piece": ev.piece,
                "stack": ev.stack,
                "target_clicks": target_clicks,
                "target_col": col,
                "shift_seq": shift_seq,
                "done_t": None,
                "verify_start_t": None,
                "wrong_obs": None,
                "wrong_count": 0,
            }
            if rot_seq:
                self.hw.dispatch_sequence(rot_seq)
                base_state["phase"] = "WAIT_ROT"
                self._move_state = base_state
            elif shift_seq:
                self.hw.dispatch_sequence(shift_seq)
                base_state["phase"] = "WAIT_SHIFT"
                self._move_state = base_state

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
    ap.add_argument("--solenoids", action="store_true", help="use real solenoid timings (default: LED timings)")
    ap.add_argument("--boost-duty", type=float, default=DEFAULT_BOOST_DUTY_PCT,
                    help="boost converter PWM duty cycle %%")
    ap.add_argument("--no-hw", action="store_true", help="never touch hardware (print button presses)")
    ap.add_argument("--ai", choices=["v2", "legacy"], default="v2")
    ap.add_argument("--use-next", choices=["auto", "on", "off"], default="auto",
                    help="auto = use the NEXT box only after it has proven itself against real spawns")
    ap.add_argument("--force-3cw", action="store_true",
                    help="always use 3x ROTATE_CW instead of trying ROTATE_CCW first")
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
        hw = HardwareController(use_solenoid_timings=args.solenoids, boost_duty_pct=args.boost_duty)
    bot = Bot(cal, hw, ai_name=args.ai, use_next=args.use_next, log_path=args.log,
              autoalign=not args.no_autoalign, sync_align=(args.source == "video"),
              ccw_failsafe=args.force_3cw)

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
                    hw.dispatch_sequence(['LEFT', 'RIGHT', 'ROTATE_CW', 'ROTATE_CCW'])
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
