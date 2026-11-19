"""
tetris_ai_v2.py  --  drop-in replacement for tetris_ai_search.find_best_move

Why a new evaluator?
    In simulation the original DNA/heuristic topped out after ~60-240 pieces even
    with PERFECT inputs. This module uses the classic Dellacherie feature set
    (landing height, eroded cells, row/column transitions, holes, cumulative wells),
    which is a well-studied, hand-tuned heuristic that survives for very long
    games with 1-piece lookahead. It runs on integer bitboards so it is fast
    enough for a Raspberry Pi.

API (same shape as the old module so tetris_test.py barely changes):
    find_best_move(board, current_piece, next_piece, weights=None, beam_width=None)
        -> ((rot, col), score, placed_board_before_line_clear)

    * board        : list of 20 lists of 0/1, row 0 = TOP (same as everywhere else)
    * next_piece   : a piece name, or None when the NEXT box could not be read
                     confidently.  None => safe 1-piece search (never a fake guess).
    * rot          : index into rotate_piece() applied to tetris_shapes.SHAPES[...]
                     (same convention as before, so ActionSequencer still works)
    * col          : left-most column of the piece's bounding box
"""
from tetris_shapes import SHAPES, rotate_piece

ROWS, COLS = 20, 10
FULL = (1 << COLS) - 1

# ---------------------------------------------------------------- tuning
# Classic Dellacherie weights (Thiery & Scherrer formulation).
W_DEFAULT = {
    'landing': -1.0,
    'eroded': 1.0,
    'row_trans': -1.0,
    'col_trans': -1.0,
    'holes': -4.0,
    'wells': -1.0,
    # tiny tie-break so equal boards prefer fewer button presses
    'move_cost': -0.002,
}

# Blend used for 2-piece search: how much of the first board's quality is kept.
LOOKAHEAD_BLEND = 0.25
DEFAULT_BEAM = 8          # finalists from stage 1 that get a full stage-2 search

# SNES spawn column of the piece's bounding box (used only for the move-cost tie-break)
SPAWN_COL = {'I': 3, 'O': 4, 'T': 4, 'S': 4, 'Z': 4, 'J': 4, 'L': 4}


def _unique_rots(name):
    if name == 'O':
        return (0,)
    if name in ('I', 'S', 'Z'):
        return (0, 1)
    return (0, 1, 2, 3)


def _build_pieces():
    out = {}
    for name, shape in SHAPES.items():
        m = shape
        lst = []
        for rot in range(4):
            if rot in _unique_rots(name):
                h, w = len(m), len(m[0])
                masks = tuple(sum(1 << c for c in range(w) if m[r][c]) for r in range(h))
                lst.append((rot, h, w, masks))
            m = rotate_piece(m)
        out[name] = lst
    return out


PIECES = _build_pieces()

# row transition lookup (walls count as filled)
_ROW_TRANS = []
for _v in range(1 << COLS):
    x = (_v << 1) | 1 | (1 << (COLS + 1))
    _ROW_TRANS.append(bin((x ^ (x >> 1)) & ((1 << (COLS + 1)) - 1)).count('1'))
_POP = [bin(i).count('1') for i in range(1 << COLS)]


def board_to_rows(board):
    rows = []
    for r in board:
        v = 0
        for c in range(COLS):
            if r[c]:
                v |= 1 << c
        rows.append(v)
    return rows


def rows_to_board(rows):
    return [[(v >> c) & 1 for c in range(COLS)] for v in rows]


def _drop(rows, masks, h, x):
    """Hard-drop a piece placed at column x. Returns (top_row, shifted_masks) or (-1, None)."""
    sh = [m << x for m in masks]
    for i in range(h):
        if rows[i] & sh[i]:
            return -1, None
    r = 0
    while r + h < ROWS:
        blocked = False
        for i in range(h):
            if rows[r + 1 + i] & sh[i]:
                blocked = True
                break
        if blocked:
            break
        r += 1
    return r, sh


def _place(rows, r, sh):
    new = rows[:]
    for i, m in enumerate(sh):
        new[r + i] |= m
    return new


def _clear(rows):
    kept = [v for v in rows if v != FULL]
    n = ROWS - len(kept)
    if n:
        kept = [0] * n + kept
    return kept, n


def _features(rows_after, landing_height, eroded):
    """Dellacherie features on the board AFTER line clears."""
    # first non-empty row from the top
    top = ROWS
    for i in range(ROWS):
        if rows_after[i]:
            top = i
            break

    row_t = 0
    for i in range(top, ROWS):
        row_t += _ROW_TRANS[rows_after[i]]

    col_t = 0
    holes = 0
    wells = 0
    for c in range(COLS):
        bit = 1 << c
        lb = 1 << (c - 1) if c > 0 else 0
        rb = 1 << (c + 1) if c < COLS - 1 else 0
        prev_filled = False          # above the stack is empty
        seen_block = False
        depth = 0
        for i in range(top, ROWS):
            v = rows_after[i]
            filled = bool(v & bit)
            if filled != prev_filled:
                col_t += 1
            prev_filled = filled
            if filled:
                seen_block = True
                depth = 0
            else:
                if seen_block:
                    holes += 1
                left_f = True if c == 0 else bool(v & lb)
                right_f = True if c == COLS - 1 else bool(v & rb)
                if left_f and right_f:
                    depth += 1
                    wells += depth
                else:
                    depth = 0
        if not prev_filled:          # the floor counts as filled
            col_t += 1
    return row_t, col_t, holes, wells


def _score_state(rows_after, landing_height, eroded, weights):
    row_t, col_t, holes, wells = _features(rows_after, landing_height, eroded)
    return (weights['landing'] * landing_height
            + weights['eroded'] * eroded
            + weights['row_trans'] * row_t
            + weights['col_trans'] * col_t
            + weights['holes'] * holes
            + weights['wells'] * wells)


def _enumerate(rows, name, weights, with_cost=False, press_fn=None, budget=None):
    """All reachable (rot, col) placements for a piece with their evaluated results.

    press_fn(name, rot, col) -> number of button presses needed.
    budget = {"spr": seconds per row, "elapsed": seconds since the piece appeared,
              "press_period": s per press (strike + recharge), "strike": s, "lead": s, "margin": s}
    With a budget, a placement is only allowed if its presses can FINISH before the piece lands
    there:  (presses-1)*press_period + strike + lead  <=  landing_row*spr - elapsed - margin.
    If nothing fits, the placements that need the fewest presses are kept (best effort)."""
    out = []
    infeasible = []
    for rot, h, w, masks in PIECES[name]:
        for x in range(COLS - w + 1):
            r, sh = _drop(rows, masks, h, x)
            if r < 0:
                continue
            presses = None
            if press_fn is not None:
                presses = press_fn(name, rot, x)
            placed = _place(rows, r, sh)
            after, n = _clear(placed)
            eroded = 0
            if n:
                for i in range(h):
                    if placed[r + i] == FULL:
                        eroded += _POP[masks[i]]
                eroded *= n
            landing = (ROWS - (r + h - 1)) + (h - 1) / 2.0
            sc = _score_state(after, landing, eroded, weights)
            if with_cost:
                if presses is not None:
                    sc += weights['move_cost'] * presses
                else:
                    sc += weights['move_cost'] * (abs(x - SPAWN_COL.get(name, 4)) + (rot if rot < 3 else 1))
            item = (sc, (rot, x), placed, after)
            if budget is not None and presses is not None:
                need = max(0, presses - 1) * budget["press_period"] + (budget["strike"] if presses else 0.0) + budget["lead"]
                avail = r * budget["spr"] - budget["elapsed"] - budget["margin"]
                if need > avail:
                    infeasible.append((presses, need - avail, item))
                    continue
            out.append(item)
    if not out and infeasible:
        infeasible.sort(key=lambda t: (t[1], t[0]))          # least overshoot, then fewest presses
        best = infeasible[0][1]
        out = [t[2] for t in infeasible if t[1] <= best + 0.15]
    return out


def find_best_move(board, current_piece_name, next_piece_name=None, weights=None, beam_width=None,
                   press_fn=None, budget=None):
    """See module docstring. `weights` is accepted for API compatibility; pass a dict
    with the keys of W_DEFAULT to override, otherwise W_DEFAULT is used (the old
    GA 'DNA' dicts are ignored on purpose)."""
    if current_piece_name not in PIECES:
        return None, -999999, None
    w = W_DEFAULT
    if weights and 'landing' in weights:
        w = weights

    rows = board_to_rows(board)
    stage1 = _enumerate(rows, current_piece_name, w, with_cost=True, press_fn=press_fn, budget=budget)
    if not stage1:
        return None, -999999, None
    stage1.sort(key=lambda t: t[0], reverse=True)

    if next_piece_name not in PIECES:
        best = stage1[0]
        return best[1], best[0], rows_to_board(best[2])

    bw = DEFAULT_BEAM if beam_width is None else beam_width
    best_total = None
    best_item = stage1[0]
    for s1, mv1, placed1, after1 in stage1[:max(1, bw)]:
        s2_list = _enumerate(after1, next_piece_name, w)
        if s2_list:
            best2 = max(t[0] for t in s2_list)
            total = LOOKAHEAD_BLEND * s1 + (1.0 - LOOKAHEAD_BLEND) * best2
        else:
            total = -1e6            # next piece cannot be placed => this line loses
        if best_total is None or total > best_total:
            best_total = total
            best_item = (s1, mv1, placed1, after1)
    return best_item[1], best_total, rows_to_board(best_item[2])
