# TODO: Performance increases.
# TODO: Make sure all the modules work with a general board size and number of walls.
#    Currently only tested size 3 with 1 wall.
from tensorflow.python.ops.logging_ops import Print

# ====================
# Execution of Learning Cycle
# ====================

# Importing packages
from dual_network import dual_network
from self_play import self_play
from train_network import train_network
from evaluate_network import evaluate_network
from evaluate_best_player import evaluate_best_player
from constants import BOARD_SIZE

# Number of NUM_EPOCH
NUM_TRAIN_CYCLE = 2

# Main function
if __name__ == '__main__':
    print(f'Board size {BOARD_SIZE}')

    # Creating the dual network
    dual_network()

    for i in range(NUM_TRAIN_CYCLE):
        print(f'\nBegin training cycle {i+1}/{NUM_TRAIN_CYCLE} ====================')
        # self-play part
        print('\nBegin self-play ====================')
        self_play()

        # parameter update part
        print('\nUpdate network parameters ====================')
        train_network()

        # Evaluating new parameters
        print('\nEvaluate new parameters ====================')
        update_best_player = evaluate_network()

        # Evaluating the best player
        print('\nEvaluate best model against baseline algorithms ====================')
        if update_best_player:
            evaluate_best_player()
