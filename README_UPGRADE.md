# Tetris bot - update 2 (speed, palettes, lighting, hardware)

Copy ALL of these over your project (same folder as your other files):

| file | change |
|---|---|
| `tetris_test.py` | main loop - **your edits kept** (no DROP, `--boost-duty`, `t` test key) + new features below |
| `tetris_vision.py` | adaptive lighting/palette reading, faster-level spawn tracking, fixed NEXT timing |
| `tetris_ai_v2.py` | AI now understands "how many presses can I afford before the piece lands" |
| `hardware_controller.py` | **small patch to your file**: `cancel()`, stale-press abort, timing helpers (nothing else changed) |
| `calibrate.py`, `replay_check.py`, `ai_selftest.py` | unchanged |
| `timing_report.py` | new: latency / speed summary of a play session (`python3 timing_report.py decisions.jsonl`) |
| `speed_sim.py` | new: model of fall speed vs button-press time (`python3 speed_sim.py --press-ms 285 150`) |

## What the hardware numbers mean for faster levels

Your controller needs **285 ms per press** (65 ms strike + 220 ms capacitor recharge) and `DROP` is off, so
gravity decides how much time a piece has. A typical placement is 3-6 presses = 0.9-1.7 s.
Your level 0 measured about 0.9 s/row (recording). If later levels follow an NES-like curve (I could not
verify that for this game) they fall roughly 0.2 s/row around level 7, 0.1 s/row around level 9-10 and
faster after that. `speed_sim.py` is a MODEL (not a measurement):

| fall speed | 285 ms/press, cancel stale presses | + time-budgeted AI | 150 ms/press hardware |
|---|---|---|---|
| 0.20 s/row | perfect | perfect | perfect |
| 0.10 s/row | ~30 of 200 pieces survive, 73-78 % land as planned | ~42, 89-91 % | all 100 in a short run, ~100 % |
| 0.08 s/row | ~19 of 200, 53 % | ~31, 89 % | 117 of 300 in an earlier run (vs 21-27 at 285 ms) |

So: **vision latency is not the bottleneck - button speed is.** Measured on your Pi camera loop the vision
adds ~0.10 s from first sight to dispatch (the confirm frames); the AI takes 1-5 ms.
What this update does about it:
* **Stale-press cancel.** When a new piece appears the bot calls `hw.cancel()`. Before, presses left over
  from a piece that had already landed kept firing and hit the NEXT piece.
* **Time budget.** The bot measures the fall speed from the camera (`sec_per_row`). When pieces fall
  <= 0.25 s/row it only picks placements whose presses can finish before the piece lands, and it stops waiting
  for the NEXT box (saves 0.3-0.45 s). `--time-budget off|auto|on`.
* **Hardware ideas that matter most (not done - need your decision, they change the solenoid wiring/duty):**
  shorter recharge (`--recharge-ms`; 150 ms/press raised survival at 0.08 s/row from ~21-27 to 117 pieces (of 300) in the model),
  enabling DROP, and holding LEFT/RIGHT to use the game's auto-repeat for long moves. Check coil heating.
  Try values with `python3 tetris_test.py --solenoids --strike-ms 65 --recharge-ms 150`.

Measure your own numbers: play a session, then `python3 timing_report.py decisions.jsonl`.

## Other palettes (purple, burnt orange, ...) and lighting

Nothing in the vision uses hue: spawn shapes are matched on shape only, and cells are "colourful and clearly
brighter than the background". The thresholds now ADAPT to the live background brightness, block
brightness and block saturation, and a hot spot / haze on the glass is absorbed as background. I tested by
altering your recording (same frames, changed light) - pieces found correctly out of 14:

| condition | old fixed thresholds | now |
|---|---|---|
| normal | 13 | 14 |
| dim x0.6 / x0.35 | 11 / 0 | 13 / 14 |
| veiling haze +55 / +90 | 10 / 4 | 11 / 11 |
| white glare spot | 0 | 13 |
| drifting + pulsing glare spot | 6 | 14 |
| brightness ramp 1.0 -> 0.45 | 11 | 14 |
| dark purple palette (hue shift + dim) | 0 | 13 |

Limits: these are synthetic changes of one recording, not real purple/orange levels. Heavy haze still costs a
few pieces. Near normal lighting the code deliberately reproduces the threshold tuned on your footage.
The live window shows `light: bg .. block .. contrast ..` and warns `LOW CONTRAST` below 70.
Send me a raw clip (`--record`) of a later level and I will check it for real.

## Other changes

* **NEXT box:** it was actually accurate on your clean footage (10 of 10); the "trust" score was wrong because it
  compared the box AFTER it had already updated. Fixed. Lookahead now starts after 4 proven pieces and is
  available within ~0.1 s of the spawn.
* **Faster spawns:** the tracker now also catches a piece that has already fallen 1-3 rows when first seen (found
  a piece the old version missed in your clip) and ignores stray lit cells at the board edges.
* **Auto-align** keeps the learned lighting when it re-aligns the grid.
* New options: `--fps`, `--time-budget`, `--strike-ms`, `--recharge-ms`.

## Honest limits

* No real-hardware run of the new `cancel()` path: tested with the controller in mock-GPIO mode (strikes 285 ms
  apart, cancel stops the remaining presses, a newer sequence supersedes a running one).
* Fall-speed estimate from your screen recording is noisy (the remote-desktop stream duplicates frames);
  the live camera loop will be cleaner. The estimate only changes behaviour when it says <= 0.25 s/row.
* The time-budget helps at <= 0.10 s/row in the model but is slightly worse than plain cancel around 0.12 s/row,
  so it only switches on when fall speed is measured fast.
* One stray "piece" appears at the very start of your new recording (window still being moved); the grid
  must sit exactly on the board.
