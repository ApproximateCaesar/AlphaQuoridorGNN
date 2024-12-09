# ====================
# Self-Play Part
# ====================
from diagnostics import time_this_function, profile_this_function, speedtest
# Importing packages
from game import State
from pv_mcts import pv_mcts_scores
from dual_network import DN_POLICY_OUTPUT_SIZE
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np
import pickle
import os
from copy import deepcopy

import diagnostics

from cProfile import Profile
from pstats import Stats

# Preparing parameters
SP_GAME_COUNT = 10  # Number of games for self-play (25000 in the original version)
SP_TEMPERATURE = 1.0  # Temperature parameter for Boltzmann distribution

# Value of the first player
def first_player_value(ended_state):
    # 1: First player wins, -1: First player loses, 0: Draw
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0

# Saving training data
def write_data(history):
    now = datetime.now()
    os.makedirs('./data/', exist_ok=True)  # Create folder if it does not exist
    path = './data/{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    with open(path, mode='wb') as f:
        pickle.dump(history, f)


# TODO: Increase performance of play(). Currently takes 35.5s (10-call average) for 5x5 board.
# Executing one game
@time_this_function
def play(model):
    # Training data
    history = []

    # Generating the state
    state = State()

    while not state.is_done(): # while the game hasn't ended

        # Getting the probability distribution of legal moves
        scores = pv_mcts_scores(model, deepcopy(state), SP_TEMPERATURE)

        # Adding the state and policy to the training data
        policies = [0] * DN_POLICY_OUTPUT_SIZE
        for action, policy in zip(state.legal_actions(), scores):
            policies[action] = policy
        history.append([state.pieces_array(), policies, None])

        # Getting the action
        action = np.random.choice(state.legal_actions(), p=scores)

        # Getting the next state
        state = state.next(action)

    # Adding the value to the training data
    value = first_player_value(state)
    for i in range(len(history)):
        history[i][2] = value
        value = -value
    return history

# Self-Play
def self_play():
    # Training data
    history = []

    # Loading the best player's model
    model = load_model('./model/best.keras')

    # Executing multiple games
    for i in range(SP_GAME_COUNT):
        # Executing one game
        h = play(model)
        history.extend(h)

        # Output
        print('\rSelf-play (game {}/{})'.format(i+1, SP_GAME_COUNT), end='')
    print('')

    # Saving the training data
    write_data(history)

    # Clearing the model
    K.clear_session()
    del model

# Running the function
if __name__ == '__main__':
    self_play()