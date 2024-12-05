# ====================
# Monte Carlo Tree Search Implementation
# ====================

# Import packages
from game import State
from dual_network import DN_INPUT_SHAPE
from math import sqrt
from tensorflow.keras.models import load_model
from pathlib import Path
import numpy as np
from copy import deepcopy
import random

import diagnostics

# Prepare parameters
PV_EVALUATE_COUNT = 50 # Number of simulations per inference (original is 1600)

# Inference
def predict(model, state):
    # Reshape input data for inference
    a, b, c = DN_INPUT_SHAPE
    x = np.array(state.pieces_array())
    x = x.reshape(c, a, b).transpose(1, 2, 0).reshape(1, a, b, c)

    # Inference
    y = model.predict(x, batch_size=1, verbose=False)

    # Get policy
    policies = y[0][0][list(state.legal_actions())] # Only legal moves
    policies /= np.sum(policies) if np.sum(policies) else 1 # Convert to a probability distribution summing to 1

    # Get value
    value = y[1][0][0]
    return policies, value

# Convert list of nodes to list of scores
def nodes_to_scores(nodes):
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores

# Get Monte Carlo Tree Search scores
def pv_mcts_scores(model, state, temperature):
    # Define Monte Carlo Tree Search node
    class Node:
        # Initialize node
        def __init__(self, state, p):
            self.state = state # State
            self.p = p # Policy
            self.w = 0 # Cumulative value
            self.n = 0 # Number of simulations
            self.child_nodes = None  # Child nodes

        # Calculate value of the state
        def evaluate(self):
            # If the game is over
            if self.state.is_done():
                # Get value from the game result
                value = -1 if self.state.is_lose() else 0

                # Update cumulative value and number of simulations
                self.w += value
                self.n += 1
                return value

            # If there are no child nodes
            if not self.child_nodes:
                # Get policy and value from neural network inference
                policies, value = predict(model, self.state)

                # Update cumulative value and number of simulations
                self.w += value
                self.n += 1

                # Expand child nodes
                self.child_nodes = []
                for action, policy in zip(self.state.legal_actions(), policies):
                    self.child_nodes.append(Node(self.state.next(action), policy))
                return value

            # If there are child nodes
            else:
                # Get value from the evaluation of the child node with the maximum arc evaluation value
                value = -self.next_child_node().evaluate()

                # Update cumulative value and number of simulations
                self.w += value
                self.n += 1
                return value

        # Get child node with the maximum arc evaluation value
        def next_child_node(self):
            # Calculate arc evaluation value
            C_PUCT = 1.0
            t = sum(nodes_to_scores(self.child_nodes))
            pucb_values = []
            for child_node in self.child_nodes:
                pucb_values.append((-child_node.w / child_node.n if child_node.n else 0.0) +
                    C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n))

            # Return child node with the maximum arc evaluation value
            return self.child_nodes[np.argmax(pucb_values)]

    # Create a node for the current state
    root_node = Node(state, 0)

    # Perform multiple evaluations
    for _ in range(PV_EVALUATE_COUNT):
        root_node.evaluate()

    # Probability distribution of legal moves
    scores = nodes_to_scores(root_node.child_nodes)
    if temperature == 0: # Only the maximum value is 1
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else: # Add variation with Boltzmann distribution
        scores = boltzman(scores, temperature)
    return scores

# Action selection with Monte Carlo Tree Search
def pv_mcts_action(model, temperature=0):
    def pv_mcts_action(state):
        scores = pv_mcts_scores(model, deepcopy(state), temperature)

        return np.random.choice(state.legal_actions(), p=scores)
    return pv_mcts_action

# Boltzmann distribution
def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]

def random_action():
    def random_action(state):
        legal_actions = state.legal_actions()
        action = random.randint(0, len(legal_actions) - 1)

        return legal_actions[action]
    return random_action

# Confirm operation
if __name__ == '__main__':
    # Load model
    path = sorted(Path('./model').glob('*.keras'))[-1]
    model = load_model(str(path))

    # Generate state
    state = State()

    # Create function to get actions with Monte Carlo Tree Search
    next_action = pv_mcts_action(model, 1.0)

    # Loop until the game is over
    while True:
        # If the game is over
        if state.is_done():
            break

        # Get action
        action = next_action(state)

        # Get next state
        state = state.next(action)

        # Print state
        print(state)
