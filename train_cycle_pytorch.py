# TODO: Performance increases.
# TODO: Make sure all the modules work with a general board size and number of walls.
#    Tested size 3 with 1 wall and size 5 with 2 walls.
#    Create a system which saves the data and models separately for each board size.
from tensorflow.python.ops.logging_ops import Print

# ====================
# Execution of Learning Cycle
# ====================

# Importing packages
from dual_network_pytorch import create_dual_network
from self_play_pytorch import self_play
from train_network_pytorch import train_network
from evaluate_network_pytorch import evaluate_network
from evaluate_best_player_pytorch import evaluate_best_player
from constants import BOARD_SIZE

# Number of NUM_EPOCH
NUM_TRAIN_CYCLE = 1

# Main function
if __name__ == '__main__':
    print(f'Board size {BOARD_SIZE}')

    # Creating the dual network
    create_dual_network()

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
        if update_best_player:
            print('\nEvaluate best model against baseline algorithms ====================')
            evaluate_best_player()
