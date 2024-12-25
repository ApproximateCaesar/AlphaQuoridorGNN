# ====================
# Self-Play
# ====================

import os
import pickle
from datetime import datetime
from copy import deepcopy
import torch
import numpy as np

from code_profiling_util import time_this_function, speedtest
from game_logic import State
from pv_mcts import pv_mcts_policy
from pv_network_cnn import Network, INPUT_SHAPE, NUM_FILTERS, POLICY_OUTPUT_SIZE, NUM_RESIDUAL_BLOCKS
# TODO: hardcode model constants into DualNetwork so I don't always have to import them

# Parameters
SP_GAME_COUNT = 50  # Number of games for self-play (25000 in the original version)
SP_TEMPERATURE = 1.0 # Temperature parameter for Boltzmann distribution

def first_player_value(ended_state):
    """Calculate the value of the first player based on the game result.
    1: First player wins, -1: First player loses, 0: Draw"""
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0

def write_data(history):
    """Save training data to a file."""
    now = datetime.now()
    os.makedirs('./data/', exist_ok=True)  # Create folder if it does not exist
    path = './data/{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    with open(path, mode='wb') as f:
        pickle.dump(history, f)


def play(model, device):
    """Execute one self-play game."""
    history = []

    state = State()
    while not state.is_done():  # while the game hasn't ended

        # Get the probability distribution of legal moves
        scores = pv_mcts_policy(model, deepcopy(state), SP_TEMPERATURE, device)

        # Add state and policy to training data
        policy = [0] * POLICY_OUTPUT_SIZE
        for action, action_prob in zip(state.legal_actions(), scores):
            policy[action] = action_prob
        history.append([state.to_array(), policy, None])

        # Choose an action based on the scores
        action = np.random.choice(state.legal_actions(), p=scores)

        # Transition to the next state
        state = state.next(action)

    # Add value to training data
    value = first_player_value(state)
    for i in range(len(history)):
        history[i][2] = value
        value = -value

    return history


def self_play():
    """Perform self-play games and save the training data."""
    # Training data
    history = []
    # TODO: create a load model function
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'model/best.pth'
    model = Network(INPUT_SHAPE[0], NUM_FILTERS, NUM_RESIDUAL_BLOCKS, POLICY_OUTPUT_SIZE)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = torch.jit.script(model)  # converting model to torchscript increases performance
    model.to(device)
    model.eval()

    for i in range(SP_GAME_COUNT):
        # Execute one game
        h = play(model, device)
        history.extend(h)

        # Output progress
        print(f'\rSelf-play (game {i + 1}/{SP_GAME_COUNT})', end='')
    print('')

    # Save training data
    write_data(history)

    # Clean up
    del model
    torch.cuda.empty_cache()

if __name__ == '__main__':
    self_play()
