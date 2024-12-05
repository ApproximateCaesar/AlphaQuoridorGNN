# TODO: Performance increases.

# ====================
# Execution of Learning Cycle
# ====================

# Importing packages
from dual_network import dual_network
from self_play import self_play
from train_network import train_network
from evaluate_network import evaluate_network
from evaluate_best_player import evaluate_best_player

# Number of NUM_EPOCH
NUM_TRAIN_CYCLE = 2

# Main function
if __name__ == '__main__':
    # Creating the dual network
    dual_network()

    for i in range(NUM_TRAIN_CYCLE):
        print(f'\nBEGIN TRAINING CYCLE {i+1}/{NUM_TRAIN_CYCLE} ====================')
        # self-play part
        print('\nBegin self-play ====================')
        self_play()
        print('Self play', i, 'complete')

        # parameter update part
        print('\nUpdate network parameters ====================')
        train_network()
        print('\nParameter update', i, 'complete\n')

        # Evaluating new parameters
        print('\nEvaluate new parameters ====================')
        update_best_player = evaluate_network()
        print('Evaluating new parameters', i, 'complete\n')

        # Evaluating the best player
        print('\nEvaluate best model against baseline algorithms ====================')
        # if update_best_player:
        #     evaluate_best_player()