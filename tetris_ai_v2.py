"""
tetris_ai_v2.py  --  Fast bitboard AI locked to a Column 9 Tetris well with strict hole avoidance,
plus get_landing_row() and live fall-speed time budgeting for tetris_test.py.
"""
from tetris_shapes import SHAPES, rotate_piece

ROWS, COLS = 20, 10
FULL = (1 << COLS) - 1

# Safe mode: flat 9-col stack (Cols 0..8) + strictly open Col 9 for 4-line Tetrises
W_DEFAULT = {
    'height': 0.35,
    'holes': 60.0,          # Heavy penalty so it avoids creating/burying holes
    'bumpiness': 2.0,       # Keeps Cols 0..8 flat with no internal canyons
    'well_scale': 3.5,      # Strong commitment to keeping Col 9 empty for I-bars
    'lines_1': -30.0,       # Discourage single-line clears when stack is low
    'lines_2': -8.0,
    'lines_3': 20.0,
    'lines_4': 600.0,       # Massive reward for 4-line Tetris!
    'move_cost': -0.002,
}

# Survival mode when any block reaches the top 8 rows
W_HIGH_STACK = {
    'height': 0.90,
    'holes': 45.0,
    'bumpiness': 1.5,
    'well_scale': 0.0,      # Stop saving a well and clear lines to survive
    'lines_1': 35.0,
    'lines_2': 80.0,
    'lines_3': 160.0,
    'lines_4': 600.0,
    'move_cost': -0.002,
}

LOOKAHEAD_BLEND = 0.20
DEFAULT_BEAM = 10
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


def get_landing_row(board, piece_name, rot, col):
    """Returns the top landing row for (piece_name, rot, col), or -1 if invalid."""
    if piece_name not in PIECES:
        return -1
    rows = board_to_rows(board)
    for r_idx, h, w, masks in PIECES[piece_name]:
        if r_idx == rot:
            if col < 0 or col + w > COLS:
                return -1
            land_r, _ = _drop(rows, masks, h, col)
            return land_r
    return -1


def _score_state(rows_after, lines_cleared, weights):
    heights = [0] * COLS
    total_holes = 0.0

    for c in range(COLS):
        bit = 1 << c
        block_found = False
        blocks_above = 0
        for r in range(ROWS):
            if rows_after[r] & bit:
                if not block_found:
                    heights[c] = ROWS - r
                    block_found = True
                blocks_above += 1
            elif block_found:
                total_holes += 1.0 + (min(blocks_above, 5) * 0.25)

    # Strictly use Column 9 (rightmost) as the Tetris well
    main = heights[:-1]
    well_h = heights[9]

    tower_span = max(main) - min(main)
    height_cost = sum(heights) + (max(heights) * 2.5) + (tower_span * 1.8)

    bump = 0.0
    for i in range(8):
        diff = abs(main[i] - main[i + 1])
        bump += diff * 1.75 if diff >= 2 else diff

    for i in range(1, 8):
        wall_min = min(main[i - 1], main[i + 1])
        if wall_min - main[i] >= 2:
            bump += (wall_min - main[i]) * 1.5

    if well_h > heights[8]:
        bump += (well_h - heights[8]) * 4.0

    well_score = 0.0
    if weights['well_scale'] > 0.0:
        avg_main = sum(main) / 9.0
        min_main = min(main)
        well_penalty = (-7.0 * well_h) if well_h > 0 else 4.0
        true_depth = max(0.0, float(min_main - well_h))
        dev_depth = max(0.0, avg_main - well_h) * 0.6
        well_score = min(true_depth, 4.0) + min(dev_depth, 2.5) + well_penalty

    line_reward = 0.0
    if lines_cleared == 1:
        line_reward = weights['lines_1']
    elif lines_cleared == 2:
        line_reward = weights['lines_2']
    elif lines_cleared == 3:
        line_reward = weights['lines_3']
    elif lines_cleared >= 4:
        line_reward = weights['lines_4']

    return (
        line_reward
        - (weights['height'] * height_cost)
        - (weights['holes'] * total_holes)
        - (weights['bumpiness'] * bump)
        + (weights['well_scale'] * well_score)
    )


def _enumerate(rows, name, weights, with_cost=False):
    out = []
    for rot, h, w, masks in PIECES[name]:
        for x in range(COLS - w + 1):
            r, sh = _drop(rows, masks, h, x)
            if r < 0:
                continue
            placed = _place(rows, r, sh)
            after, n = _clear(placed)
            sc = _score_state(after, n, weights)
            if with_cost:
                sc += weights['move_cost'] * (abs(x - SPAWN_COL.get(name, 4)) + (rot if rot < 3 else 1))
            out.append((sc, (rot, x), placed, after))
    return out


def find_best_move(board, current_piece_name, next_piece_name=None, weights=None,
                   beam_width=None, press_fn=None, budget=None):
    if current_piece_name not in PIECES:
        return None, -999999, None

    if weights and 'holes' in weights and 'well_scale' in weights:
        w = weights
    else:
        w = W_HIGH_STACK if any(board[r][c] for r in range(8) for c in range(COLS)) else W_DEFAULT

    rows = board_to_rows(board)
    stage1 = _enumerate(rows, current_piece_name, w, with_cost=True)
    if not stage1:
        return None, -999999, None

    # Live fall-speed time budget filter (from tetris_test.py)
    if press_fn is not None and budget is not None:
        spr = budget.get('spr')
        if spr:
            elapsed = budget.get('elapsed', 0.0)
            press_period = budget.get('press_period', 0.240)
            strike = budget.get('strike', 0.080)
            margin = budget.get('margin', 0.08)
            feasible = []
            for item in stage1:
                rot, col = item[1]
                n_press = press_fn(current_piece_name, rot, col)
                need_s = (max(0, n_press - 1) * press_period + strike) if n_press > 0 else 0.0
                land_r = get_landing_row(board, current_piece_name, rot, col)
                avail_s = max(0.0, land_r * spr - elapsed - margin)
                if need_s <= avail_s:
                    feasible.append(item)
            if feasible:
                stage1 = feasible

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
