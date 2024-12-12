import torch
import numpy as np
from game import State
from dual_network_pytorch import DualNetwork, DN_INPUT_SHAPE, DN_FILTERS, DN_POLICY_OUTPUT_SIZE, DN_RESIDUAL_NUM
from math import sqrt
from copy import deepcopy
import random
from pathlib import Path

# Prepare parameters
PV_EVALUATE_COUNT = 50  # Number of simulations per inference (original is 1600)

# Inference
def predict(model, state, device):
    # Reshape input data for inference
    C, H, W = DN_INPUT_SHAPE
    x = np.array(state.pieces_array())
    x = x.reshape(C, H, W)  # Shape: (C, H, W)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():  # disable gradient calculation to save memory during inference
        policy, value = model(x)

    policy = policy[0][list(state.legal_actions())]  # Remove batch dimension and restrict to legal actions

    # Normalize policy distribution over the subset of legal actions
    policy /= torch.sum(policy) if torch.sum(policy) else 1

    policy = policy.cpu().numpy()  # tensor on GPU to numpy array on CPU
    value = value.item()  # GPU tensor to float

    return policy, value

# Convert list of nodes to list of scores
def nodes_to_scores(nodes):
    scores = [child.n for child in nodes]
    return scores

# Get Monte Carlo Tree Search scores
def pv_mcts_scores(model, state, temperature, device):
    # Define Monte Carlo Tree Search node
    class Node:
        def __init__(self, state, p):
            self.state = state  # State
            self.p = p  # probability of action that led to this state
            self.w = 0  # Cumulative value
            self.n = 0  # Number of simulations
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
            elif not self.child_nodes:
                # Get policy and value from neural network inference
                policy, value = predict(model, self.state, device)
                # Update cumulative value and number of simulations
                self.w += value
                self.n += 1

                # Expand child nodes
                self.child_nodes = [
                    Node(self.state.next(action), action_prob)
                    for action, action_prob in zip(self.state.legal_actions(), policy)
                ]
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
            pucb_values = [
                (-child.w / child.n if child.n else 0.0) + C_PUCT * child.p * sqrt(t) / (1 + child.n)
                for child in self.child_nodes
            ]
            # Return child node with the maximum arc evaluation value
            return self.child_nodes[np.argmax(pucb_values)]

    # Create a node for the current state
    root_node = Node(state, 0)

    # Perform multiple evaluations
    for _ in range(PV_EVALUATE_COUNT):
        root_node.evaluate()

    # Probability distribution of legal moves
    scores = nodes_to_scores(root_node.child_nodes)
    if temperature == 0:  # Only the maximum value is 1
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:  # Add variation with Boltzmann distribution
        scores = boltzman(scores, temperature)
    return scores

# Action selection with Monte Carlo Tree Search
def pv_mcts_action(model, temperature=0, device='cpu'):
    """Returns a function of the game state that selects an action based on PV-MCTS."""
    def pv_mcts_action(state):
        scores = pv_mcts_scores(model, deepcopy(state), temperature, device)
        return np.random.choice(state.legal_actions(), p=scores)
    return pv_mcts_action

# Boltzmann distribution
def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]

def random_action():
    """Returns a function of the game state that selects a uniformly random action."""
    def random_action(state):
        legal_actions = state.legal_actions()
        action = random.randint(0, len(legal_actions) - 1)
        return legal_actions[action]
    return random_action

# Confirm operation
if __name__ == '__main__':
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = sorted(Path('./model_pytorch').glob('*.pth'))[-1]
    model = DualNetwork(DN_INPUT_SHAPE[0], DN_FILTERS, DN_RESIDUAL_NUM, DN_POLICY_OUTPUT_SIZE)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Play game using PV-MCTS
    state = State()
    next_action = pv_mcts_action(model, 1.0, device)
    while not state.is_done():
        action = next_action(state)
        state = state.next(action)
        print(state)
