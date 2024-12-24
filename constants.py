"""
Define constants to be imported by modules.

NUM_PLIES_FOR_DRAW: the number of plies (moves by either player) after which the game is taken to be a draw.
"""

# # 3x3 Quoridor:
# BOARD_SIZE = 3
# NUM_WALLS = 1
# NUM_PLIES_FOR_DRAW = 14  # (1 wall placement + max 6 moves from goal)*2

# 5x5 Quoridor:
# BOARD_SIZE = 5
# NUM_WALLS = 2
# NUM_PLIES_FOR_DRAW = 28  # (2 wall placements + max 12 moves from goal)*2

# # 9x9 Quoridor:
BOARD_SIZE = 9
NUM_WALLS = 10
NUM_PLIES_FOR_DRAW = 116  # (10 wall placements + max 48 moves from goal)*2