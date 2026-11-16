# Tetris bot upgrade - what to copy and how to run it

## Files (put them next to your existing ones)

| file | what it is |
|---|---|
| `tetris_test.py` | **replaces** your main loop |
| `tetris_vision.py` | new: board reader, spawn tracker, NEXT reader, camera/video sources |
| `tetris_ai_v2.py` | new AI (replaces the brain + search at runtime) |
| `calibrate.py` | **replaces** your calibrator (4 corners instead of 2 rectangles) |
| `replay_check.py` | new: checks the vision on a recorded clip, no hardware |
| `ai_selftest.py` | new: checks the AI on your Pi, no camera |

Keep everything else. `tetris_brain.py`, `tetris_ai_search.py`, the trainer and the genetics
script are untouched (they are only used with `--ai legacy`).
`tetris_test.py` still imports your `hardware_controller.py` (`HardwareController`,
`dispatch_sequence`, `cleanup`) exactly as before.
The rotation / column tables in the sequencer are copied unchanged from your file.

## 1. Calibrate (do this first, and again whenever the camera or monitor moves)

    python3 calibrate.py

1. Aim + focus (`+` / `-` / `a`), then SPACE. Tip: freeze while a piece sits in the TOP row.
2. BOARD: drag a loose box around the playfield, ENTER, then click the 4 corners of the 10x20
   cell area: top-left, top-right, bottom-right, bottom-left (right-click / `u` = undo).
   Click on the dark play area where blocks are drawn, not on the bezel.
3. In the preview, blocks must sit inside grid cells. `y` accepts, `r` redoes.
4. NEXT BOX: the 4 corners of the dark area where the next piece is drawn, BELOW the yellow
   "NEXT:" label.

Capture size is 1280x720 now (it was 640x360). Your old `calibration.json` still loads and is
scaled, but re-calibrate for the better version.

## 2. Check it without playing

    python3 tetris_test.py --record          # play a bit; saves raw_*.avi (RAW frames, no overlay)
    python3 replay_check.py raw_XXXX.avi --calibration calibration.json

You should see one `SPAWN` line per real piece and a stack-cell count that grows by 4 per
piece (minus 10 per cleared line). An ODD count means a phantom or a missed cell (2-player
garbage rows add 9 cells, so ignore that case). Send me that clip if anything looks off.

## 3. Run

    python3 tetris_test.py                   # LED timings
    python3 tetris_test.py --solenoids       # real solenoid timings
    python3 tetris_test.py --no-hw           # never touches hardware, prints button presses

Keys: `q` quit | `g` grid overlay | `k` re-run grid auto-align | `b` learn empty-board background
| `r` toggle raw recording | `+ - a` focus | `t` hardware test.
Every decision is appended to `decisions.jsonl` (piece, board, move, buttons, AI ms).

## What changed and why

* **AI.** The old DNA topped out after ~60-240 pieces in your own simulator even with perfect
  input. The new evaluator (Dellacherie features, bitboards) survived 20,000 pieces in the same
  simulator without topping out. It plays for survival and mostly clears 1-2 lines at a time.
* **No more fake NEXT = 'T'.** The NEXT reader is only used after it has matched at least four
  real spawns (>= 75 %); until then the AI searches one piece deep, which is safe.
  `--use-next off` disables it, `--use-next on` forces it.
* **Spawn detection** = exactly four new cells forming one of the 7 SNES spawn shapes in the top
  rows. The stack is "what is lit, minus those four cells", so real holes and overhangs are never
  "corrected" away. If vision sees a piece spawn a column away from your table, the button
  sequence is shifted to match.
* **Cells** need to be bright AND strongly coloured, and mostly filled. That removes the WAIT/LOSE
  text and the glow around bright blocks that created phantom cells.
* **Grid auto-align.** Hand-clicked corners are a few pixels off, and the stack is sensitive to
  that. After 5 spawns (with >= 12 stack cells) the grid is fine-tuned once in the background.
  `--no-autoalign` turns it off, `k` runs it again.

## Known limits - please read

* Tested on ONE recorded clip (8 spawns, no line clears, no garbage, level 0) and in simulation.
  Nothing was run on your Pi, camera, or solenoids.
* On that clip: 8 of 8 spawns found with the right piece and column, stack counts exact. With
  deliberately mis-clicked corners the first ~5 pieces can read a cell or two wrong before
  auto-align runs; after it the counts were exact (8 of 9 bad starts).
* The NEXT-box reader could NOT be validated (that clip's NEXT area was too damaged). Expect to
  tune it from a raw clip. The AI does not need it: without lookahead it still survived 20,000
  simulated pieces.
* The game speeds up with level. At high levels the button presses may not finish before the piece
  falls. I did not model the solenoid timing.
* The sequence is open loop (it does not re-check where the piece ended up before DROP).
