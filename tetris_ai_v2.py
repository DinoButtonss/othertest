"""
tetris_ai_v2.py  --  drop-in replacement for tetris_ai_search.find_best_move

Updated to strongly prioritize 4-line clears (Tetris) over single-line clears
while keeping fast bitboard evaluation for Raspberry Pi and safe high-stack survival.
"""
from tetris_shapes import SHAPES, rotate_piece

ROWS, COLS = 20, 10
FULL = (1 << COLS) - 1

# ---------------------------------------------------------------- tuning
# Prioritizes 4-line Tetris clears and penalizes 1-line clears when safe,
# while switching smoothly to survival mode if the stack gets high.
W_DEFAULT = {
    'landing': -1.0,
    'eroded': 1.0,
    'row_trans': -1.2,
    'col_trans': -1.2,
    'holes': -5.0,
    'wells': 1.2,
    'lines_1': -35.0,
    'lines_2': -10.0,
    'lines_3': 15.0,
    'lines_4': 250.0,
    'tetris_well_bonus': 18.0,
    # tiny tie-break so equal boards prefer fewer button presses
    'move_cost': -0.002,
}

# Survival weights when the stack reaches the upper half of the board
W_HIGH_STACK = {
    'landing': -1.5,
    'eroded': 1.5,
    'row_trans': -1.5,
    'col_trans': -1.5,
    'holes': -6.0,
    'wells': 0.4,
    'lines_1': 12.0,
    'lines_2': 28.0,
    'lines_3': 50.0,
    'lines_4': 180.0,
    'tetris_well_bonus': 4.0,
    'move_cost': -0.002,
}

# Blend used for 2-piece search: how much of the first board's quality is kept.
LOOKAHEAD_BLEND = 0.25
DEFAULT_BEAM = 10          # finalists from stage 1 that get a full stage-2 search

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


def _col_height(rows, c, top):
    bit = 1 << c
    for i in range(top, ROWS):
        if rows[i] & bit:
            return ROWS - i
    return 0


def _features(rows_after, landing_height, eroded, lines_cleared, weights):
    """Dellacherie + Tetris well & multi-line clear features on the board AFTER line clears."""
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
    col_heights = [0] * COLS

    for c in range(COLS):
        bit = 1 << c
        lb = 1 << (c - 1) if c > 0 else 0
        rb = 1 << (c + 1) if c < COLS - 1 else 0
        prev_filled = False
        seen_block = False
        depth = 0
        ch = 0
        for i in range(top, ROWS):
            v = rows_after[i]
            filled = bool(v & bit)
            if filled and ch == 0:
                ch = ROWS - i
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
        if not prev_filled:
            col_t += 1
        col_heights[c] = ch

    # Encourage keeping column 9 (rightmost) open as a clean 4-line Tetris well
    # when columns 0..8 are fairly even and not dangerously high.
    well_bonus = 0.0
    if top >= 6:
        right_h = col_heights[COLS - 1]
        left_min = min(col_heights[:COLS - 1])
        left_max = max(col_heights[:COLS - 1])
        well_depth = max(0, left_min - right_h)
        if right_h == 0 and (left_max - left_min) <= 3:
            well_bonus = weights.get('tetris_well_bonus', 18.0) * min(well_depth, 4) * 0.25

    line_bonus = 0.0
    if lines_cleared == 1:
        line_bonus = weights.get('lines_1', -35.0)
    elif lines_cleared == 2:
        line_bonus = weights.get('lines_2', -10.0)
    elif lines_cleared == 3:
        line_bonus = weights.get('lines_3', 15.0)
    elif lines_cleared >= 4:
        line_bonus = weights.get('lines_4', 250.0)

    return row_t, col_t, holes, wells, well_bonus + line_bonus


def _score_state(rows_after, landing_height, eroded, lines_cleared, weights):
    row_t, col_t, holes, wells, extra = _features(
        rows_after, landing_height, eroded, lines_cleared, weights
    )
    return (weights['landing'] * landing_height
            + weights['eroded'] * eroded
            + weights['row_trans'] * row_t
            + weights['col_trans'] * col_t
            + weights['holes'] * holes
            + weights['wells'] * wells
            + extra)


def _enumerate(rows, name, weights, with_cost=False):
    """All reachable (rot, col) placements for a piece with their evaluated results."""
    out = []
    for rot, h, w, masks in PIECES[name]:
        for x in range(COLS - w + 1):
            r, sh = _drop(rows, masks, h, x)
            if r < 0:
                continue
            placed = _place(rows, r, sh)
            after, n = _clear(placed)
            eroded = 0
            if n:
                for i in range(h):
                    if placed[r + i] == FULL:
                        eroded += _POP[masks[i]]
                eroded *= n
            landing = (ROWS - (r + h - 1)) + (h - 1) / 2.0
            sc = _score_state(after, landing, eroded, n, weights)
            if with_cost:
                sc += weights['move_cost'] * (abs(x - SPAWN_COL.get(name, 4)) + (rot if rot < 3 else 1))
            out.append((sc, (rot, x), placed, after))
    return out


def find_best_move(board, current_piece_name, next_piece_name=None, weights=None, beam_width=None):
    if current_piece_name not in PIECES:
        return None, -999999, None

    if weights and 'landing' in weights:
        w = weights
    else:
        # Use survival weights if any block is in the top 7 rows; otherwise prioritize 4-line clears
        w = W_HIGH_STACK if any(board[r][c] for r in range(7) for c in range(COLS)) else W_DEFAULT

    rows = board_to_rows(board)
    stage1 = _enumerate(rows, current_piece_name, w, with_cost=True)
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
            total = -1e6
        if best_total is None or total > best_total:
            best_total = total
            best_item = (s1, mv1, placed1, after1)
    return best_item[1], best_total, rows_to_board(best_item[2])
