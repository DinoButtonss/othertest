SHAPES = {
    'I': [
        [1, 1, 1, 1]
    ],
    'O': [
        [1, 1],
        [1, 1]
    ],
    'T': [
        [1, 1, 1],
        [0, 1, 0]
    ],
    'S': [
        [0, 1, 1],
        [1, 1, 0]
    ],
    'Z': [
        [1, 1, 0],
        [0, 1, 1]
    ],
    'J': [
        [1, 1, 1],
        [0, 0, 1]
    ],
    'L': [
        [1, 1, 1],
        [1, 0, 0]
    ]
}

def rotate_piece(piece_matrix):
    rotated = [list(row) for row in zip(*piece_matrix[::-1])]
    return rotated