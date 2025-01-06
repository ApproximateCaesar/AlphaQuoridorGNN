# ====================
# New Parameter Evaluation Section
# ====================

# Import packages
from game_logic import State
from pv_mcts import pv_mcts_action
import torch
from pv_network_cnn import CNNNetwork
from shutil import copy
from constants import PV_NETWORK_PATH

# Prepare parameters
EN_GAME_COUNT = 15 # Number of games per evaluation (originally 400)
EN_TEMPERATURE = 1.0 # Temperature of the Boltzmann distribution

# Points for the first player
def first_player_point(ended_state):
    # 1: first player wins, 0: first player loses, 0.5: draw
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    return 0.5

# Execute one game
def play(next_actions):
    # Generate state
    state = State()

    # Loop until the game ends
    while True:
        # When the game ends
        if state.is_done():
            break

        # Get action
        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = next_action(state)

        # Get the next state
        state = state.next(action)

    # Return points for the first player
    return first_player_point(state)

# Replace the best player
def update_best_player():
    copy(PV_NETWORK_PATH + 'latest.pth', PV_NETWORK_PATH + 'best.pth')
    print('Latest model is better than current best. Replacing best model with latest.')

# Network evaluation
def evaluate_network():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load latest model
    model0 = CNNNetwork()
    model0.prep_for_inference(PV_NETWORK_PATH + 'latest.pth')

    # load best model
    model1 = CNNNetwork()
    model1.prep_for_inference(PV_NETWORK_PATH + 'best.pth')

    # Generate a function to select actions using PV MCTS
    next_action0 = pv_mcts_action(model0, EN_TEMPERATURE, device)
    next_action1 = pv_mcts_action(model1, EN_TEMPERATURE, device)
    next_actions = (next_action0, next_action1)

    # Repeat multiple matches
    total_point = 0
    for i in range(EN_GAME_COUNT):
        # Execute one game
        if i % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(list(reversed(next_actions)))

        # Output
        print('\rEvaluating latest model against current best (game {}/{})'.format(i + 1, EN_GAME_COUNT), end='')
    print('')

    # Calculate average points
    average_point = total_point / EN_GAME_COUNT
    print('Average points of latest model against current best:', average_point)

    # Clean up
    del model0
    del model1
    torch.cuda.empty_cache()

    # Replace the best player
    if average_point > 0.5:
        update_best_player()
        return True
    else:
        return False

# Operation check
if __name__ == '__main__':
    evaluate_network()
