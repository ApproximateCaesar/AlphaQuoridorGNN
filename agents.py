"""
Here we define a selection of Quoridor-playing agents that employ different algorithms.
Each agent is represented by a function that takes in the game state and returns an action.
"""

import random
import math
from game_logic import State

# Agent that randomly selects an action
def random_action(state):
    legal_actions = state.legal_actions()
    action = random.randint(0, len(legal_actions) - 1)
    return legal_actions[action]

# Alpha-beta pruning:

def heuristic_eval(state):
    """
    Heuristic evaluation function to estimate the value of a game state in minimax search.
    """

    return 0


def alpha_beta(state, alpha, beta, depth):
    """
    Alpha-beta pruning using depth-limited search and heuristic evaluation function.

    :param state: Current game state
    :param alpha: Alpha value for pruning
    :param beta: Beta value for pruning
    :param depth: Remaining depth to search
    :return: Estimated value of the state
    """
    # Terminal conditions
    if depth == 0 or state.is_lose() or state.is_draw():
        if state.is_lose():
            return -1  # Loss is -1
        if state.is_draw():
            return 0  # Draw is 0
        return heuristic_eval(state)  # Evaluate the state at depth 0

    # Search over legal actions
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -beta, -alpha, depth - 1)
        if score > alpha:
            alpha = score

        # Beta cutoff
        if alpha >= beta:
            return alpha

    return alpha


# Select the best action using alpha-beta pruning
def alpha_beta_action(state, max_depth=2):
    """
    Select the best action using alpha-beta pruning.

    :param state: Current game state
    :param max_depth: Maximum depth to search
    :return: The action with the maximum state value
    """
    best_action = None
    alpha = -float('inf')

    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -float('inf'), -alpha, max_depth)
        if score > alpha:
            best_action = action
            alpha = score

    return best_action

# Basic MCTS:

def playout(state):
    # Loss is -1
    if state.is_lose():
        return -1

    # Draw is 0
    if state.is_draw():
        return 0

    # Next state value
    return -playout(state.next(random_action(state)))


# Return the index of the maximum value
def argmax(collection):
    return collection.index(max(collection))


# Select an action using Monte Carlo Tree Search
def mcts_action(state):
    # Node for Monte Carlo Tree Search
    class Node:
        # Initialization
        def __init__(self, state):
            self.state = state  # State
            self.w = 0  # Cumulative value
            self.n = 0  # Number of trials
            self.child_nodes = None  # Child nodes

        # Evaluation
        def evaluate(self):
            # When the game ends
            if self.state.is_done():
                # Get value from the game result
                value = -1 if self.state.is_lose() else 0  # Loss is -1, draw is 0

                # Update cumulative value and number of trials
                self.w += value
                self.n += 1
                return value

            # When there are no child nodes
            if not self.child_nodes:
                # Get value from playout
                value = playout(self.state)

                # Update cumulative value and number of trials
                self.w += value
                self.n += 1

                # Expand child nodes
                if self.n == 10:
                    self.expand()
                return value

            # When there are child nodes
            else:
                # Get value from evaluating the child node with the maximum UCB1
                value = -self.next_child_node().evaluate()

                # Update cumulative value and number of trials
                self.w += value
                self.n += 1
                return value

        # Expand child nodes
        def expand(self):
            legal_actions = self.state.legal_actions()
            self.child_nodes = []
            for action in legal_actions:
                self.child_nodes.append(Node(self.state.next(action)))

        # Get the child node with the maximum UCB1
        def next_child_node(self):
            # Return the child node with n=0
            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node

            # Calculate UCB1
            t = 0
            for c in self.child_nodes:
                t += c.n
            ucb1_values = []
            for child_node in self.child_nodes:
                ucb1_values.append(-child_node.w / child_node.n + 2 * (2 * math.log(t) / child_node.n) ** 0.5)

            # Return the child node with the maximum UCB1
            return self.child_nodes[argmax(ucb1_values)]

    # Generate the root node
    root_node = Node(state)
    root_node.expand()

    # Evaluate the root node 100 times
    for _ in range(100):
        root_node.evaluate()

    # Return the action with the maximum number of trials
    legal_actions = state.legal_actions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)
    return legal_actions[argmax(n_list)]

# Test an agent
if __name__ == '__main__':
    # Generate the state
    state = State()
    # Display as a string
    print(state)
    print()

    # Loop until the game ends
    while not state.is_done():

        # Get the next state
        state = state.next(mcts_action(state))
        # Display as a string
        print(state)
        print()