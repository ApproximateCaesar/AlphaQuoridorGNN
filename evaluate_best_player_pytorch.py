# ====================
# Evaluation of Best Player
# ====================

# Import packages
from game import State, random_action, alpha_beta_action, mcts_action
from pv_mcts_pytorch import pv_mcts_action
import torch
from dual_network_pytorch import DualNetwork, DN_INPUT_SHAPE, DN_POLICY_OUTPUT_SIZE, DN_FILTERS, DN_RESIDUAL_NUM
from pathlib import Path
import numpy as np

# Prepare parameters
EP_GAME_COUNT = 10  # Number of games per evaluation

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
    while not state.is_done():
        print(state)
        # Get action
        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = next_action(state)

        # Get the next state
        state = state.next(action)

    # Return points for the first player
    return first_player_point(state)

# Evaluation of any algorithm
def evaluate_algorithm_of(label, next_actions):
    # Repeat multiple matches
    total_point = 0
    for i in range(EP_GAME_COUNT):
        # Execute one game
        if i % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(list(reversed(next_actions)))

        # Output
        print('\rEvaluate {}/{}'.format(i + 1, EP_GAME_COUNT), end='')
    print('')

    # Calculate average points
    average_point = total_point / EP_GAME_COUNT
    print(label, average_point)

# Evaluation of the best player
def evaluate_best_player():
    # Load best model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = './model_pytorch/best.pth'
    model = DualNetwork(DN_INPUT_SHAPE[0], DN_FILTERS, DN_RESIDUAL_NUM, DN_POLICY_OUTPUT_SIZE)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = torch.jit.script(model)  # converting model to torchscript increases performance
    model.to(device)
    model.eval()

    # Generate a function to select actions using PV MCTS
    next_pv_mcts_action = pv_mcts_action(model, 0.0, device)

    # VS Random
    next_actions = (next_pv_mcts_action, random_action)
    evaluate_algorithm_of('VS_Random', next_actions)

    # TODO: limit search depth to use with board size > 3
    # # VS Alpha-Beta
    next_actions = (next_pv_mcts_action, alpha_beta_action)
    evaluate_algorithm_of('VS_AlphaBeta', next_actions)

    # VS Monte Carlo Tree Search
    next_actions = (next_pv_mcts_action, mcts_action)
    evaluate_algorithm_of('VS_MCTS', next_actions)

    # Clean up
    del model
    torch.cuda.empty_cache()

# Operation check
if __name__ == '__main__':
    evaluate_best_player()
