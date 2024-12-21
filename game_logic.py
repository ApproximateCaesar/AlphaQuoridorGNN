# TODO: figure out how the game state and actions are represented.

# ====================
# Quoridor game logic
# ====================

# Importing packages
from collections import deque
import copy
from copy import deepcopy
from constants import BOARD_SIZE, NUM_WALLS, NUM_PLIES_FOR_DRAW
import diagnostics

MOVEMENT_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # directions in which pawns can move

# TODO: see if using __slots__ (https://wiki.python.org/moin/UsingSlots) for the State class provides a MCTS speedup
# Game state
class State:
    """
    :param walls: 1 by (N-1)^2 int array giving wall positions from the current player's perspective.
        walls[i] > 0 implies there is a wall placed in the 2x2 square with top-left tile at position (i//N, i%N).
        walls[i] = 0 if no wall, 1 if horizontal wall, and 2 if vertical wall.
    """

    def __init__(self, board_size=BOARD_SIZE, num_walls=NUM_WALLS, player=None, enemy=None, walls=None, depth=0):
        self.N = board_size
        N = self.N
        if N % 2 == 0:
            raise ValueError('The board size must be an odd number.')
        self.player = player if player is not None else [0] * 2  # Position, number of walls
        self.enemy = enemy if enemy is not None else [0] * 2
        self.walls = walls if walls is not None else [0] * ((N - 1) ** 2)
        self.depth = depth  # number of plies (moves by either player) played

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
        return self.depth >= NUM_PLIES_FOR_DRAW

    # Check if the game is over
    def is_done(self):
        return self.is_lose() or self.is_draw()

    def pieces_array(self):
        """
        :returns: Game state represented as a
        """
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
        for dx, dy in MOVEMENT_DIRECTIONS:
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
                                elif (nx == 0 and walls[wp] != 2) or (
                                        nx > 0 and walls[wp - (N - 1)] != 2 and walls[wp] != 2):
                                    nnp = np + 1
                                    actions.append(nnp)
                    elif y == (N - 1):
                        if walls[wp - 1] != 1:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if nx > 0 and walls[wp - (N - 1) - 1] != 1:
                                    nnp = np - N
                                    actions.append(nnp)
                                elif (nx == 0 and walls[wp - 1] != 2) or (
                                        nx > 0 and walls[wp - (N - 1) - 1] != 2 and walls[wp - 1] != 2):
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
                                    if (nx == 0 and walls[wp - 1] != 2) or (
                                            nx > 0 and walls[wp - (N - 1) - 1] != 2 and walls[wp - 1] != 2):
                                        nnp = np - 1
                                        actions.append(nnp)
                                    if (nx == 0 and walls[wp] != 2) or (
                                            nx > 0 and walls[wp - (N - 1)] != 2 and walls[wp] != 2):
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
                                elif (nx == (N - 1) and walls[wp - (N - 1)] != 2) or (
                                        nx < (N - 1) and walls[wp - (N - 1)] != 2 and walls[wp] != 2):
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
                                elif (nx == (N - 1) and walls[wp - (N - 1) - 1] != 2) or (
                                        nx < (N - 1) and walls[wp - (N - 1) - 1] != 2 and walls[wp - 1] != 2):
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
                                    if (nx == (N - 1) and walls[wp - (N - 1) - 1] != 2) or (
                                            nx < (N - 1) and walls[wp - (N - 1) - 1] != 2 and walls[wp - 1] != 2):
                                        nnp = np - 1
                                        actions.append(nnp)
                                    if (nx == (N - 1) and walls[wp - (N - 1)] != 2) or (
                                            nx < (N - 1) and walls[wp - (N - 1)] != 2 and walls[wp] != 2):
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
                                elif (ny == 0 and walls[wp - (N - 1)] != 1) or (
                                        ny > 0 and walls[wp - (N - 1) - 1] != 2 and walls[wp - (N - 1)] != 1):
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
                                    if (ny == 0 and walls[wp - (N - 1)] != 1) or (
                                            ny > 0 and walls[wp - (N - 1) - 1] != 2 and walls[wp - (N - 1)] != 1):
                                        nnp = np - N
                                        actions.append(nnp)
                                    if (ny == 0 and walls[wp] != 1) or (
                                            ny > 0 and (walls[wp - 1] != 1 or walls[wp] != 1)):
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
                                elif (ny == (N - 1) and walls[wp - 1] != 1) or (
                                        ny < (N - 1) and walls[wp - 1] != 1 and walls[wp] != 1):
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
                                elif (ny == (N - 1) and walls[wp - (N - 1) - 1] != 1) or (
                                        ny < (N - 1) and walls[wp - (N - 1) - 1] != 1 and walls[wp - (N - 1)] != 1):
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
                                    if (ny == (N - 1) and walls[wp - (N - 1) - 1] != 1) or (
                                            ny < (N - 1) and walls[wp - (N - 1) - 1] != 1 and walls[wp - (N - 1)] != 1):
                                        nnp = np - N
                                        actions.append(nnp)
                                    if (ny == (N - 1) and walls[wp - 1] != 1) or (
                                            ny < (N - 1) and (walls[wp - 1] != 1 or walls[wp] != 1)):
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
                visited = set()
                visited.add(state.player[0])  # mark root node as visited
                queue = deque([state.player[0]])
                while queue:
                    position = queue.popleft()
                    if position // N == 0:  # reached goal (farthest row)
                        return True
                    else:  # search child nodes (adjacent positions)
                        new_positions = state.legal_actions_pos(position)
                        for new_position in new_positions:
                            if new_position not in visited:
                                visited.add(new_position)
                                queue.append(new_position)
                return False
            # TODO: potential speedup by modifying then searching self (and reverting afterwards) instead of cloning.
            # Check if player can still reach goal
            player_state = State(board_size=N, player=self.player.copy(), enemy=self.enemy.copy(),
                                 walls=deepcopy(self.walls), depth=self.depth)
            player_state.walls[pos] = orientation

            can_reach_player = bfs(player_state)

            # Check if enemy can still reach goal
            action = pos
            if orientation == 1:
                action += N ** 2
            else:
                action += N ** 2 + (N - 1) ** 2

            enemy_state = player_state.next(action)
            can_reach_enemy = bfs(enemy_state)

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
        state = State(board_size=N, player=self.player.copy(), enemy=self.enemy.copy(), walls=deepcopy(self.walls),
                      depth=self.depth + 1)

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

    # TODO: maybe rewrite this in terms of first and second player to be less confusing
    def __str__(self):
        """
        Create a string representation of the game state.
        The board is shown from the perspective of the first player (the second player is considered the enemy).
        """
        N = self.N
        is_first_player = self.is_first_player()

        # Create empty board
        board = [['□'] * (2 * N - 1) for _ in range(2 * N - 1)]  # '□' denotes a square on which a pawn can be placed
        for i in range(2 * N - 1):
            for j in range(2 * N - 1):
                if i % 2 == 1 or j % 2 == 1:
                    board[i][j] = ' '  # ' ' denotes grooves inbetween squares in which walls can be placed


        # mark player and enemy positions
        p_pos = self.player[0] if is_first_player else self.enemy[0]
        e_pos = self.enemy[0] if is_first_player else self.player[0]

        e_pos = N ** 2 - 1 - e_pos

        p_x, p_y = p_pos // N, p_pos % N
        e_x, e_y = e_pos // N, e_pos % N

        board[2 * p_x][2 * p_y] = 'P'  # player position
        board[2 * e_x][2 * e_y] = 'E'  # enemy position

        turn_info = "<Player's Turn>" if is_first_player else "<Enemy's Turn>"

        # mark wall positions
        if not is_first_player:  # rotate to correct player's perspective
            self.rotate_walls()

        for i in range(N - 1):
            for j in range(N - 1):
                pos = i * (N - 1) + j
                if self.walls[pos] == 1:  # horizontal wall
                    board[2 * i + 1][2 * j] = '-'
                    board[2 * i + 1][2 * (j + 1)] = '-'
                if self.walls[pos] == 2:  # vertical wall
                    board[2 * i][2 * j + 1] = '|'
                    board[2 * (i + 1)][2 * j + 1] = '|'

        if not is_first_player:  # rotate back so as not to affect state
            self.rotate_walls()

        # display who won
        win_info = ''
        if self.is_done():
            if self.is_draw():
                win_info = '\n\nDraw'
            elif is_first_player:
                win_info = '\n\nPlayer has lost'
            else:
                win_info = '\n\nPlayer has won'

        board_str = '\n'.join([''.join(row) for row in board])
        return turn_info + '\n' + board_str + win_info

