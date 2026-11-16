"""
replay_check.py  --  vision self-check on a recorded clip (no hardware, no AI needed)

    python3 replay_check.py clip.avi --calibration calibration.json [--step 1] [--annotate out.mp4]

Use it on the raw clip saved by `tetris_test.py --record`.  It prints every detected spawn with the
number of cells in the stack at that moment.  What to look for:

  * one SPAWN line per real piece (if you count more pieces in the clip than lines, vision missed some)
  * stack cell counts that grow by exactly 4 per piece (minus 10 per cleared line).  An ODD count
    means a phantom or a missed cell (unless 2-player garbage rows arrived, which add 9 cells).
  * the NEXT-box accuracy line: how often the piece shown in NEXT really spawned next.
"""
import argparse

import cv2

from tetris_test import Bot, draw_debug
from tetris_vision import VideoSource, load_calibration


class _NoHW:
    def dispatch_sequence(self, seq):
        pass

    def cleanup(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--no-autoalign", action="store_true")
    ap.add_argument("--annotate", default=None, help="write an annotated video here")
    args = ap.parse_args()

    src = VideoSource(args.video, step=args.step)
    cal = load_calibration(args.calibration, frame_size=src.frame_size()) if args.calibration \
        else load_calibration(frame_size=src.frame_size())
    bot = Bot(cal, _NoHW(), use_next="auto", verbose=True, autoalign=not args.no_autoalign, sync_align=True)
    writer = None
    events = []
    n = 0
    while True:
        ok, frame, t = src.read()
        if not ok:
            break
        n += 1
        obs, nmask, dec = bot.process(frame, t)
        if dec:
            events.append(dec)
            cells = sum(row.count("#") for row in dec["stack"])
            flag = "  <-- ODD cell count (phantom/missed cell?)" if cells % 2 else ""
            print("SPAWN t=%6.1fs  piece=%s  col=%d  stack_cells=%d%s" % (t, dec["piece"], dec["spawn_col"], cells, flag))
        if args.annotate:
            vis = draw_debug(frame, bot, obs, nmask, 0.0, True, False)
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(args.annotate, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
            writer.write(vis)
    if writer:
        writer.release()
    print("\nframes processed: %d   spawns found: %d" % (n, len(events)))
    acc = bot.ntrk.accuracy()
    print("NEXT-box accuracy vs real spawns: %s  (trusted for lookahead: %s)" % (
        "n/a (never produced a stable reading)" if acc is None else "%.0f%%" % (acc * 100), bot.ntrk.trusted))


if __name__ == "__main__":
    main()
