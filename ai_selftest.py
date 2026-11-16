"""
ai_selftest.py  --  checks the new AI on your machine (no camera, no console needed)

    python3 ai_selftest.py                 # 3 seeds x 1000 pieces, with and without NEXT lookahead
    python3 ai_selftest.py --pieces 5000 --noise 0.005

Prints how many pieces were placed before topping out (== pieces means it never died) and the
average / worst time per decision - run it on the Raspberry Pi to see real decision latency.
--noise p  randomly misreads each board cell with probability p before every decision, to show
how much vision errors cost.  (For reference, the old DNA topped out after ~60-240 pieces.)
"""
import argparse
import random
import time

import tetris_ai_v2 as ai
from tetris_shapes import SHAPES


def play(seed, pieces, use_next, noise):
    rng, nrng = random.Random(seed), random.Random(seed + 999)
    names = list(SHAPES.keys())
    seq = [rng.choice(names) for _ in range(pieces + 2)]
    rows = [0] * ai.ROWS
    placed, total, worst = 0, 0.0, 0.0
    for i in range(pieces):
        seen = ai.rows_to_board(rows)
        if noise > 0:
            for r in range(ai.ROWS):
                for c in range(ai.COLS):
                    if nrng.random() < noise:
                        seen[r][c] ^= 1
        t0 = time.perf_counter()
        mv, _, _ = ai.find_best_move(seen, seq[i], seq[i + 1] if use_next else None)
        dt = time.perf_counter() - t0
        total += dt
        worst = max(worst, dt)
        if mv is None:
            break
        rot, col = mv
        _, h, _, masks = [p for p in ai.PIECES[seq[i]] if p[0] == rot][0]
        r, sh = ai._drop(rows, masks, h, col)
        if r < 0:
            break
        rows, _ = ai._clear(ai._place(rows, r, sh))
        placed += 1
    return placed, total / max(1, placed), worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.0)
    a = ap.parse_args()
    for use_next in (True, False):
        res = [play(s, a.pieces, use_next, a.noise) for s in range(1, a.seeds + 1)]
        print("%-22s survived %s / %d   avg %.1f ms   worst %.1f ms" % (
            "with NEXT lookahead" if use_next else "no lookahead (1-piece)",
            [r[0] for r in res], a.pieces,
            1000 * sum(r[1] for r in res) / len(res), 1000 * max(r[2] for r in res)))


if __name__ == "__main__":
    main()
