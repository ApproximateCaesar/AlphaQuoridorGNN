# ====================
# Policy-Value Monte Carlo Tree Search Implementation
# ====================

import torch
import numpy as np
from game_logic import State
from pv_network_cnn import CNNNetwork
from constants import PV_NETWORK_PATH
from math import sqrt
from copy import deepcopy
import random
from pathlib import Path
from train_network import preprocess_input
from code_profiling_util import profile_this_function

# Prepare parameters
PV_EVALUATE_COUNT = 50  # Number of simulations per inference (original is 1600)

def pv_mcts_policy(model, state, temperature, device):
    """Use PUCT-based Monte Carlo Tree Search to return an improved policy (distribution over legal actions),
    compared to the prior policy provided by the neural network."""
    # Define Monte Carlo Tree Search node
    class Node:
        def __init__(self, state, p):
            self.state = state  # State
            self.p = p  # prior probability of action that led to this state
            self.w = 0  # Cumulative value
            self.n = 0  # Number of simulations
            self.child_nodes = None  # Child nodes

        # Calculate value of the state
        def evaluate(self):
            # If the game is over
            if self.state.is_done():
                # Get value from the game result.
                # Note: Don't need to account for win since a non-drawn game always ends on the loser's turn.
                value = -1 if self.state.is_lose() else 0
                # Update cumulative value and number of simulations
                self.w += value
                self.n += 1
                return value

            # If there are no child nodes
            elif not self.child_nodes:
                # Get policy and value from neural network inference
                prior_policy, value = model.predict(self.state, device)
                # Update cumulative value and number of simulations
                self.w += value
                self.n += 1

                # Expand child nodes
                self.child_nodes = [
                    Node(self.state.next(action), action_prob)
                    for action, action_prob in zip(self.state.legal_actions(), prior_policy)
                ]
                return value

            # If there are child nodes
            else:
                # Get value from the evaluation of the child node with the maximum PUCT score
                value = -self.next_child_node().evaluate()
                # Update cumulative value and number of simulations
                self.w += value
                self.n += 1
                return value

        # Get child node with the maximum PUCT score
        def next_child_node(self):
            # Calculate PUCT scores
            C_PUCT = 1.25  # TODO: explore this parameter. Originally 1.0.
            t = sum([child.n for child in self.child_nodes])
            pucb_values = [
                (-child.w / child.n if child.n else 0.0) + C_PUCT * child.p * sqrt(t) / (1 + child.n)
                for child in self.child_nodes
            ]
            # Return child node with the maximum PUCT score
            return self.child_nodes[np.argmax(pucb_values)]

    # Create a node for the current state
    root_node = Node(state, 0)

    # Perform multiple evaluations
    for _ in range(PV_EVALUATE_COUNT):
        root_node.evaluate()

    # Probability distribution of legal moves
    child_visit_counts = [child.n for child in root_node.child_nodes]  # number of MCTS visits to each child node
    if temperature == 0:  # choose most-visited child
        action = np.argmax(child_visit_counts)
        mcts_policy = np.zeros(len(child_visit_counts))
        mcts_policy[action] = 1 # Give most visited child probability 1
    else:  # Add variation with Boltzmann distribution
        mcts_policy = boltzman(child_visit_counts, temperature)
    return mcts_policy

# Action selection with Monte Carlo Tree Search
def pv_mcts_action(model, temperature=0, device='cpu'):
    """Returns a function of the game state that selects an action based on PV-MCTS."""
    def pv_mcts_action(state):
        policy = pv_mcts_policy(model, deepcopy(state), temperature, device)
        return np.random.choice(state.legal_actions(), p=policy)
    return pv_mcts_action

# Boltzmann distribution
def boltzman(xs, temperature):
    """Boltzmann distribution """
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]


# Confirm operation
if __name__ == '__main__':
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = PV_NETWORK_PATH + 'best.pth'
    model = CNNNetwork()
    model.prep_for_inference(model_path)

    # Play game using PV-MCTS
    state = State()
    next_action = pv_mcts_action(model, 1.0, device)
    while not state.is_done():
        action = next_action(state)
        state = state.next(action)
        print(state)
