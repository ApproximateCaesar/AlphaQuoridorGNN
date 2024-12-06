# TODO: figure out who goes first. The AI should go first 50 percent of the time.
#   Pretty sure it doesn't matter because the AI plays itself - it always goes both first and second.
# TODO: Make board_size a constant defined outside of classes/functions so we can easily change it

# ====================
# Quoridor (3 x 3), wall = 1
# ====================

# Importing packages
import random
import math
from collections import deque
import copy
from copy import deepcopy

import diagnostics

# Game state
class State:
    def __init__(self, board_size=3, num_walls=1, player=None, enemy=None, walls=None, depth=0):
        self.N = board_size
        N = self.N
        if N % 2 == 0:
            raise ValueError('The board size must be an odd number.')
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.player = player if player is not None else [0] * 2 # Position, number of walls
        self.enemy = enemy if enemy is not None else [0] * 2
        self.walls = walls if walls is not None else [0] * ((N - 1) ** 2)
        self.depth = depth # number of plies (moves by either player) played
        self.draw_depth = 30 # number of moves after which the game is assumed to be a draw

        if player is None or enemy is None:
            init_pos = N * (N - 1) + N // 2
            self.player[0] = init_pos
            self.player[1] = num_walls
            self.enemy[0] = init_pos 
            self.enemy[1] = num_walls

     # Check if it's a loss
    def is_lose(self):
        if self.enemy[0] // self.N == 0:
            return True
        return False

    # Check if it's a draw
    def is_draw(self):
        return self.depth >= self.draw_depth
    
    # Check if the game is over
    def is_done(self):
        return self.is_lose() or self.is_draw()
    
    def pieces_array(self):
        N = self.N
        def pieces_of(pieces):
            tables = []

            table = [0] * (N ** 2)
            table[pieces[0]] = 1
            tables.append(table)
                
            table = [pieces[1]] * (N ** 2)
            tables.append(table)

            return tables
        
        def walls_of(walls):
            tables = []

            table_h = [0] * (N ** 2)
            table_v = [0] * (N ** 2)

            for wp in range((N - 1) ** 2):
                x, y = wp // (N - 1), wp % (N - 1)

                if x < (N - 1) // 2 and y < (N - 1) // 2:
                    pos = N * x + y
                elif x > (N - 1) // 2 and y < (N - 1) // 2:
                    pos = N * x + (y + 1)
                elif x < (N - 1) // 2 and y > (N - 1) // 2:
                    pos = N * (x + 1) + y
                else:
                    pos = N * (x + 1) + (y + 1)

                if walls[wp] == 1:
                    table_h[pos] = 1
                elif walls[wp] == 2:
                    table_v[pos] = 1
                
            tables.append(table_h)
            tables.append(table_v)

            return tables
        
        return [pieces_of(self.player), pieces_of(self.enemy), walls_of(self.walls)]

    def legal_actions(self):
        """
        Each possible action is represented by an integer in the range 0 to (N ** 2 + 2 * (N - 1) ** 2 - 1).
        Actions 0 to (N ** 2 - 1): Move to a position
        Actions (N ** 2) to (N ** 2 + (N - 1) ** 2 - 1): Place a horizontal wall
        Actions (N ** 2 + (N - 1) ** 2) to (N ** 2 + 2 * (N - 1) ** 2 - 1): Place a vertical wall
        """
        actions = []
        actions.extend(self.legal_actions_pos(self.player[0]))

        if self.player[1] > 0:
            for pos in range((self.N - 1) ** 2):
                actions.extend(self.legal_actions_wall(pos))
                
        return actions

    def legal_actions_pos(self, pos):
        actions = []

        N = self.N
        walls = self.walls
        ep = self.enemy[0]

        x, y = pos // N, pos % N
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N:
                np = N * nx + ny
                wp = (N - 1) * nx + ny

                if nx < x:
                    if y == 0:
                        if walls[wp] != 1:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if nx > 0 and walls[wp - (N - 1)] != 1:
                                    nnp = np - N
                                    actions.append(nnp)
                                elif (nx == 0 and walls[wp] != 2) or (nx > 0 and walls[wp - (N - 1)] != 2 and walls[wp] != 2):
                                    nnp = np + 1
                                    actions.append(nnp)
                    elif y == (N - 1):
                        if walls[wp - 1] != 1:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if nx > 0 and walls[wp - (N - 1) - 1] != 1:
                                    nnp = np -  N
                                    actions.append(nnp)
                                elif (nx == 0 and walls[wp - 1] != 2) or (nx > 0 and walls[wp - (N - 1) - 1] != 2 and walls[wp - 1] != 2):
                                    nnp = np - 1
                                    actions.append(nnp)
                    else:
                        if walls[wp - 1] != 1 and walls[wp] != 1:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if nx > 0 and walls[wp - (N - 1)] != 1 and walls[wp - (N - 1) - 1] != 1:
                                    nnp = np - N
                                    actions.append(nnp)
                                else:
                                    if (nx == 0 and walls[wp - 1] != 2) or (nx > 0 and walls[wp - (N - 1) - 1] != 2 and walls[wp - 1] != 2):
                                        nnp = np - 1
                                        actions.append(nnp)
                                    if (nx == 0 and walls[wp] != 2) or (nx > 0 and walls[wp - (N - 1)] != 2 and walls[wp] != 2):
                                        nnp = np + 1
                                        actions.append(nnp)
                if nx > x:
                    if y == 0:
                        if walls[wp - (N - 1)] != 1:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if nx < (N - 1) and walls[wp] != 1:
                                    nnp = np + N
                                    actions.append(nnp)
                                elif (nx == (N - 1) and walls[wp - (N - 1)] != 2) or (nx < (N - 1) and walls[wp - (N - 1)] != 2 and walls[wp] != 2):
                                    nnp = np + 1
                                    actions.append(nnp)
                    elif y == (N - 1):
                        if walls[wp - (N - 1) - 1] != 1:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if nx < (N - 1) and walls[wp - 1] != 1:
                                    nnp = np + N
                                    actions.append(nnp)
                                elif (nx == (N - 1) and walls[wp - (N - 1) - 1] != 2) or (nx < (N - 1) and walls[wp - (N - 1) - 1] != 2 and walls[wp - 1] != 2):
                                    nnp = np - 1
                                    actions.append(nnp)
                    else:
                        if walls[wp - (N - 1) - 1] != 1 and walls[wp - (N - 1)] != 1:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if nx < (N - 1) and walls[wp - 1] != 1 and walls[wp] != 1:
                                    nnp = np + N
                                    actions.append(nnp)
                                else:
                                    if (nx == (N - 1) and walls[wp - (N - 1) - 1] != 2) or (nx < (N - 1) and walls[wp - (N - 1) - 1] != 2 and walls[wp - 1] != 2):
                                        nnp = np - 1
                                        actions.append(nnp)
                                    if (nx == (N - 1) and walls[wp - (N - 1)] != 2) or (nx < (N - 1) and walls[wp - (N - 1)] != 2 and walls[wp] != 2):
                                        nnp = np + 1
                                        actions.append(nnp)
                if ny < y:
                    if x == 0:
                        if walls[wp] != 2:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if ny > 0 and walls[wp - 1] != 2:
                                    nnp = np - 1
                                    actions.append(nnp)
                                elif (ny == 0 and walls[wp] != 1) or (ny > 0 and walls[wp - 1] != 1 and walls[wp] != 1):
                                    nnp = np + N
                                    actions.append(nnp)
                    elif x == (N - 1):
                        if walls[wp - (N - 1)] != 2:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if ny > 0 and walls[wp - (N - 1) - 1] != 2:
                                    nnp = np - 1
                                    actions.append(nnp)
                                elif (ny == 0 and walls[wp - (N - 1)] != 1) or (ny > 0 and walls[wp - (N - 1) - 1] != 2 and walls[wp - (N - 1)] != 1):
                                    nnp = np - N
                                    actions.append(nnp)
                    else:
                        if walls[wp - (N - 1)] != 2 and walls[wp] != 2:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if ny > 0 and walls[wp - (N - 1) - 1] != 2 and walls[wp - 1] != 2:
                                    nnp = np - 1
                                    actions.append(nnp)
                                else:
                                    if (ny == 0 and walls[wp - (N - 1)] != 1) or (ny > 0 and walls[wp - (N - 1) - 1] != 2 and walls[wp - (N - 1)] != 1):
                                        nnp = np - N
                                        actions.append(nnp)
                                    if (ny == 0 and walls[wp] != 1) or (ny > 0 and (walls[wp - 1] != 1 or walls[wp] != 1)):
                                        nnp = np + N
                                        actions.append(nnp)
                if ny > y:
                    if x == 0:
                        if walls[wp - 1] != 2:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if ny < (N - 1) and walls[wp] != 2:
                                    nnp = np + 1
                                    actions.append(nnp)
                                elif (ny == (N - 1) and walls[wp - 1] != 1) or (ny < (N - 1) and walls[wp - 1] != 1 and walls[wp] != 1):
                                    nnp = np + N
                                    actions.append(nnp)
                    elif x == (N - 1):
                        if walls[wp - (N - 1) - 1] != 2:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if ny < (N - 1) and walls[wp - (N - 1)] != 2:
                                    nnp = np + 1
                                    actions.append(nnp)
                                elif (ny == (N - 1) and walls[wp - (N - 1) - 1] != 1) or (ny < (N - 1) and walls[wp - (N - 1) - 1] != 1 and walls[wp - (N - 1)] != 1):
                                    nnp = np - N
                                    actions.append(nnp)
                    else:
                        if walls[wp - (N - 1) - 1] != 2 and walls[wp - 1] != 2:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if ny < (N - 1) and walls[wp - (N - 1)] != 2 and walls[wp] != 2:
                                    nnp = np + 1
                                    actions.append(nnp)
                                else:
                                    if (ny == (N - 1) and walls[wp - (N - 1) - 1] != 1) or (ny < (N - 1) and walls[wp - (N - 1) - 1] != 1 and walls[wp - (N - 1)] != 1):
                                        nnp = np - N
                                        actions.append(nnp)
                                    if (ny == (N - 1) and walls[wp - 1] != 1) or (ny < (N - 1) and (walls[wp - 1] != 1 or walls[wp] != 1)):
                                        nnp = np + N
                                        actions.append(nnp)

        return actions

    def legal_actions_wall(self, pos):
        N = self.N
        walls = self.walls
        def can_place_wall(orientation, pos):
            if walls[pos] != 0:
                return False
            x, y = pos // (N - 1), pos % (N - 1)
            if orientation == 1:
                if y == 0:
                    if walls[pos + 1] == 1:
                        return False
                elif y == (N - 2):
                    if walls[pos - 1] == 1:
                        return False
                else:
                    if walls[pos - 1] == 1 or walls[pos + 1] == 1:
                        return False
            else:
                if x == 0:
                    if walls[pos + (N - 1)] == 2:
                        return False
                elif x == (N - 2):
                    if walls[pos - (N - 1)] == 2:
                        return False
                else:
                    if walls[pos - (N - 1)] == 2 or walls[pos + (N - 1)] == 2:
                        return False
            return True

        def can_reach_goal(orientation, pos):
            def bfs(state):
                queue = deque([state.player[0]])
                visited = set()
                while queue:
                    pos = queue.popleft()
                    nps = state.legal_actions_pos(pos)
                    for np in nps:
                        x, y = np // N, np % N
                        if y == 0:
                            return True
                        if np not in visited:
                            visited.add(np)
                            queue.append(np)
                return False

            self.walls[pos] = orientation

            player_state = State(board_size=N, player=self.player.copy(), enemy=self.enemy.copy(), walls=deepcopy(self.walls), depth=self.depth)

            can_reach_player = bfs(player_state)

            action = pos
            if orientation == 1:
                action += N ** 2
            else:
                action += N ** 2 + (N - 1) ** 2

            enemy_state = player_state.next(action)

            can_reach_enemy = bfs(enemy_state)

            self.walls[pos] = 0

            return can_reach_player and can_reach_enemy
    
        actions = []

        if can_place_wall(1, pos) and can_reach_goal(1, pos):
            actions.append(N ** 2 + pos)
        if can_place_wall(2, pos) and can_reach_goal(2, pos):
            actions.append(N ** 2 + (N - 1) ** 2 + pos)

        return actions
    
    def rotate_walls(self):
        N = self.N
        rotated_walls = [0] * len(self.walls)
        for i in range((N - 1) ** 2):
            rotated_walls[i] = self.walls[(N - 1) ** 2 - 1 - i]
        self.walls = rotated_walls
    
    def next(self, action):
        N = self.N
        # Create the next state
        state = State(board_size=N, player=self.player.copy(), enemy=self.enemy.copy(), walls=deepcopy(self.walls), depth=self.depth + 1)

        if action < N ** 2:
            # Move piece
            state.player[0] = action
        elif action < N ** 2 + (N - 1) ** 2:
            # Place horizontal wall
            pos = action - N ** 2
            state.walls[pos] = 1
            state.player[1] -= 1
        else:
            # Place vertical wall
            pos = action - N ** 2 - (N - 1) ** 2
            state.walls[pos] = 2
            state.player[1] -= 1

        state.rotate_walls()

        # Swap players
        state.player, state.enemy = state.enemy, state.player

        return state
    
    # Check if it's the first player's turn
    def is_first_player(self):
        return self.depth % 2 == 0

    def __str__(self):
        """Display the game state as a string."""
        N = self.N
        is_first_player = self.is_first_player()

        board = [['o'] * (2 * N - 1) for _ in range(2 * N - 1)]
        for i in range(2 * N - 1):
            for j in range(2 * N - 1):
                if i % 2 == 1 and j % 2 == 1:
                    board[i][j] = 'x'

        p_pos = self.player[0] if is_first_player else self.enemy[0]
        e_pos = self.enemy[0] if is_first_player else self.player[0]

        e_pos = N ** 2 - 1 - e_pos

        p_x, p_y = p_pos // N, p_pos % N
        e_x, e_y = e_pos // N, e_pos % N

        board[2 * p_x][2 * p_y] = 'P'
        board[2 * e_x][2 * e_y] = 'E'
        
        turn_info = "<Enemy's Turn>" if is_first_player else "<Player's Turn>"

        if not is_first_player:
            self.rotate_walls()

        # Set walls
        for i in range(N - 1):
            for j in range(N - 1):
                pos = i * (N - 1) + j
                if self.walls[pos] == 1:
                    board[2 * i + 1][2 * j] = '-'
                    board[2 * i + 1][2 * (j + 1)] = '-'
                if self.walls[pos] == 2:
                    board[2 * i][2 * j + 1] = '|'
                    board[2 * (i + 1)][2 * j + 1] = '|'

        if not is_first_player:
            self.rotate_walls()

        board_str = '\n'.join([''.join(row) for row in board])
        return turn_info + '\n' + board_str


# Randomly select an action
def random_action(state):
    legal_actions = state.legal_actions()
    action = random.randint(0, len(legal_actions) - 1)
    return legal_actions[action]

# Calculate state value using alpha-beta pruning
def alpha_beta(state, alpha, beta):
    # Loss is -1
    if state.is_lose():
        return -1

    # Draw is 0
    if state.is_draw():
        return 0

    # Calculate state values for legal actions
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -beta, -alpha)
        if score > alpha:
            alpha = score

        # If the best score for the current node exceeds the parent node, stop the search
        if alpha >= beta:
            return alpha

    # Return the maximum value of the state values for legal actions
    return alpha

# TODO: Looks like alpha-beta algorithm doesn't limit its search depth. If not implement this with eval function.
# Select an action using alpha-beta pruning
def alpha_beta_action(state):
    # Calculate state values for legal actions
    best_action = 0
    alpha = -float('inf')
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -float('inf'), -alpha)
        if score > alpha:
            best_action = action
            alpha = score

    # Return the action with the maximum state value
    return best_action

# Playout
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
    class node:
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
                self.child_nodes.append(node(self.state.next(action)))

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
                ucb1_values.append(-child_node.w/child_node.n + 2*(2*math.log(t)/child_node.n)**0.5)

            # Return the child node with the maximum UCB1
            return self.child_nodes[argmax(ucb1_values)]

    # Generate the root node
    root_node = node(state)
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

# Running the function
if __name__ == '__main__':
    # Generate the state
    state = State()

    # Loop until the game ends
    while True:
        # When the game ends
        if state.is_done():
            break

        # Get the next state
        state = state.next(random_action(state))

        # Display as a string
        print(state)
        print()
