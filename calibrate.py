"""
calibrate.py  --  4-corner calibration for the Tetris bot

    python3 calibrate.py                         # live Pi camera (1280x720)
    python3 calibrate.py --video clip.mp4        # calibrate on a recorded clip
    python3 calibrate.py --image frame.png       # calibrate on a still image

What you do
  1. (camera only) aim + focus with the live preview:  + / -  focus,  a  autofocus,  SPACE  freeze.
     TIP: freeze while a piece is sitting in the TOP row of the board - then you can see exactly
     where row 0 is.
  2. BOARD:  drag a loose box around the playfield, press ENTER (zooms in), then click the four
     corners of the 10x20 cell area in this order:  top-left, top-right, bottom-right, bottom-left.
     Click on the edges of the DARK play area where blocks are drawn - not on the bezel.
     Right-click or 'u' undoes the last point.
  3. A warped preview with a 10x20 grid appears.  Blocks should sit exactly inside grid cells.
     Press y to accept, r to redo.
  4. NEXT BOX:  same thing for the dark area where the next piece is drawn.  Leave the yellow
     "NEXT:" label OUT of the quad if you can (click just below it).
  5. calibration.json is written.  Re-run this any time the camera or monitor moves.
"""
import argparse
import sys

import cv2
import numpy as np

from tetris_vision import (BOARD_COLS, BOARD_ROWS, CameraSource, VideoSource, Warper,
                           order_corners, save_calibration)


class ClickCollector:
    """Collects up to n clicks.  Left click adds, right click removes the last one."""

    def __init__(self, n=4):
        self.n = n
        self.pts = []

    def on_mouse(self, event, x, y, flags=0, param=None):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.pts) < self.n:
            self.pts.append((float(x), float(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and self.pts:
            self.pts.pop()

    def undo(self):
        if self.pts:
            self.pts.pop()


def click_corners(frame, title, prompt, max_view=(1500, 900)):
    """Zoomed 4-click picker.  Returns 4 points in ORIGINAL frame coordinates, or None (aborted)."""
    h, w = frame.shape[:2]
    print("\n" + prompt)
    print("  Drag a loose box around the area, press ENTER.  (ESC = skip zoom and use the whole frame)")
    roi = cv2.selectROI(title, frame, fromCenter=False, showCrosshair=False)
    x0, y0, rw, rh = [int(v) for v in roi]
    if rw < 20 or rh < 20:
        x0, y0, rw, rh = 0, 0, w, h
    crop = frame[y0:y0 + rh, x0:x0 + rw]
    scale = min(max_view[0] / float(rw), max_view[1] / float(rh))
    scale = max(1.0, min(scale, 6.0))
    view = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    col = ClickCollector(4)
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, col.on_mouse)
    labels = ["TL", "TR", "BR", "BL"]
    while True:
        vis = view.copy()
        for i, (px, py) in enumerate(col.pts):
            cv2.circle(vis, (int(px), int(py)), 5, (0, 0, 255), -1)
            cv2.putText(vis, labels[i], (int(px) + 8, int(py) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if len(col.pts) >= 2:
            cv2.polylines(vis, [np.int32(col.pts)], len(col.pts) == 4, (0, 255, 0), 1)
        nxt = labels[len(col.pts)] if len(col.pts) < 4 else "ENTER = accept"
        cv2.putText(vis, "next: " + nxt + "   (right-click / u = undo, ESC = abort)", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(title, vis)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            cv2.destroyWindow(title)
            return None
        if k == ord('u'):
            col.undo()
        if k in (13, 10, 32) and len(col.pts) == 4:
            break
    cv2.destroyWindow(title)
    return [(x0 + px / scale, y0 + py / scale) for px, py in col.pts]


def preview_board(frame, corners):
    cell = 36
    wp = Warper(corners, BOARD_COLS * cell, BOARD_ROWS * cell)
    img = wp(frame)
    for c in range(BOARD_COLS + 1):
        cv2.line(img, (c * cell, 0), (c * cell, BOARD_ROWS * cell), (0, 255, 255), 1)
    for r in range(BOARD_ROWS + 1):
        cv2.line(img, (0, r * cell), (BOARD_COLS * cell, r * cell), (0, 255, 255), 1)
    return img


def preview_next(frame, corners):
    wp = Warper(corners, 320, 160)
    img = wp(frame)
    for c in range(1, 4):
        cv2.line(img, (c * 80, 0), (c * 80, 160), (0, 255, 255), 1)
    cv2.line(img, (0, 80), (320, 80), (0, 255, 255), 1)
    return img


def pick(frame, name, prompt, previewer):
    while True:
        pts = click_corners(frame, "Calibrate - " + name, prompt)
        if pts is None:
            print("Aborted.")
            sys.exit(1)
        corners = order_corners(pts)
        win = "Preview - " + name + "  (y = accept, r = redo)"
        cv2.imshow(win, previewer(frame, corners))
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k in (ord('y'), 13):
                cv2.destroyWindow(win)
                return [[float(x), float(y)] for x, y in corners]
            if k in (ord('r'), 27):
                cv2.destroyWindow(win)
                break


def grab_frame(args):
    """Returns (frame, source_info_dict, release_fn)."""
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit("Cannot read image " + args.image)
        return img, {"source": "image", "path": args.image}, (lambda: None)
    if args.video:
        src = VideoSource(args.video, start=args.start)
        ok, frame, _ = src.read()
        if not ok:
            sys.exit("Cannot read video " + args.video)
        return frame, {"source": "video", "path": args.video}, src.release

    size = tuple(int(v) for v in args.size.lower().split("x"))
    cam = CameraSource(index=args.camera_index, size=size)
    print("\nLIVE PREVIEW:  + / - focus   a autofocus   SPACE/ENTER freeze and calibrate")
    manual = False
    frame = None
    while True:
        ok, fr, _ = cam.read()
        if ok:
            frame = fr
            vis = fr.copy()
            txt = "Manual focus %.2f" % cam.lens_pos if manual else "Autofocus"
            cv2.putText(vis, txt + "  |  SPACE = freeze", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Aim & focus", vis)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 32) and frame is not None:
            break
        if k in (ord('+'), ord('=')):
            manual = True
            cam.set_focus(cam.lens_pos + 0.25)
        elif k == ord('-'):
            manual = True
            cam.set_focus(cam.lens_pos - 0.25)
        elif k == ord('a'):
            manual = False
            cam.autofocus()
        elif k == ord('q'):
            cam.release()
            sys.exit(0)
    cv2.destroyWindow("Aim & focus")
    info = {"source": "camera", "camera_index": args.camera_index,
            "lens_position": cam.lens_pos if manual else None}
    return frame, info, cam.release


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--image")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into --video to grab the frame")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--size", default="1280x720", help="camera capture size (default 1280x720)")
    ap.add_argument("--out", default=None, help="calibration file to write (default: calibration.json)")
    args = ap.parse_args()

    frame, info, release = grab_frame(args)
    release()
    h, w = frame.shape[:2]

    board = pick(frame, "BOARD",
                 "BOARD: click the 4 corners of the 10x20 cell area (TL, TR, BR, BL).",
                 preview_board)
    nxt = pick(frame, "NEXT BOX",
               "NEXT BOX: click the 4 corners of the dark area where the NEXT piece is drawn "
               "(below the yellow 'NEXT:' label).", preview_next)

    data = {"version": 2, "frame_size": [w, h], "board_corners": board, "next_corners": nxt}
    data.update(info)
    if args.out:
        save_calibration(data, args.out)
        print("\nSaved", args.out)
    else:
        save_calibration(data)
        print("\nSaved calibration.json")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
