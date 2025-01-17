# TODO: Performance increases: https://pytorch.org/serve/performance_checklist.html
# TODO: Create a system which saves the data and models separately for each board size.

#  TODO: track model performance over training cycles and the number of self-play games performed.
# ====================
# Execution of Learning Cycle
# ====================

# Importing packages
from pv_network_cnn import create_network
from self_play import self_play
from train_network import train_network
from evaluate_network import evaluate_network
from evaluate_agents import evaluate_best_player
from constants import BOARD_SIZE, PV_NETWORK_NAME
import torch

NUM_TRAIN_CYCLE = 1000 # Number of training cycles

# Main function
if __name__ == '__main__':
    print(f'Model {PV_NETWORK_NAME} on board size {BOARD_SIZE}')

    # Creating the PV network
    create_network()

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

        # # Evaluating the best player
        # if update_best_player:
        #     print('\nEvaluate best model against baseline algorithms ====================')
        #     evaluate_best_player()
