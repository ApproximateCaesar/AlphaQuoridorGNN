# TODO: figure out how the game state and actions are represented.

# ====================
# Quoridor game logic
# ====================

# Importing packages
from collections import deque
from constants import BOARD_SIZE, NUM_WALLS, NUM_PLIES_FOR_DRAW
import code_profiling_util

MOVEMENT_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # directions in which pawns can move (U,D,L,R)

# TODO: see if using __slots__ (https://wiki.python.org/moin/UsingSlots) for the State class provides a MCTS speedup
# Game state
class State:
    """
    :param walls: 1 by (N-1)^2 int array giving wall positions from the current player's perspective.
        walls[i] > 0 implies there is a wall placed in the 2x2 square with top-left tile at linear index i, that is, at (row, col) = (i//N, i%N).
        walls[i] = 0 if no wall, 1 if horizontal wall, and 2 if vertical wall.
    :param player: 1 by 2 int array [position, number of walls]. Position is given as a linear index from the player's own perspective.
    :param enemy: 1 by 2 int array [position, number of walls]. Position is given as a linear index from the enemy's own perspective.
    :param plies_played: number of plies (1 ply = 1 turn taken by 1 player) played so far.
    """

    def __init__(self, board_size=BOARD_SIZE, num_walls=NUM_WALLS, player=None, enemy=None, walls=None, plies_played=0):
        self.N = board_size
        N = self.N
        if N % 2 == 0:
            raise ValueError('The board size must be an odd number.')
        self.player = player if player is not None else [0] * 2
        self.enemy = enemy if enemy is not None else [0] * 2
        self.walls = walls if walls is not None else [0] * ((N - 1) ** 2)
        self.plies_played = plies_played

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
        return self.plies_played >= NUM_PLIES_FOR_DRAW

    # Check if the game is over
    def is_done(self):
        return self.is_lose() or self.is_draw()

    def pieces_array(self):
        """
        :returns: Game state represented as a
        """
        N = self.N

        def pawn_table(player):
            tables = []

            table = [0] * (N ** 2)
            table[player[0]] = 1
            tables.append(table)

            table = [player[1]] * (N ** 2)
            tables.append(table)

            return tables

        def wall_table(walls):
            tables = []

            table_h = [0] * (N ** 2)
            table_v = [0] * (N ** 2)

            for wall_index in range((N - 1) ** 2):
                # get linear index of top-left tile within the 2x2 square in which the wall is placed
                top_left_tile_index = N * (wall_index // (N - 1)) + (wall_index % (N - 1))

                if walls[wall_index] == 1:
                    table_h[top_left_tile_index] = 1
                elif walls[wall_index] == 2:
                    table_v[top_left_tile_index] = 1

            tables.append(table_h)
            tables.append(table_v)

            return tables
        return [pawn_table(self.player), pawn_table(self.enemy), wall_table(self.walls)]


    def to_array(self):
        """
        :returns: Array containing the game state variables [player, enemy, walls].
        """
        return [self.player.copy(), self.enemy.copy(), self.walls.copy()]


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

    # TODO: test this function extensively and compare speed to old implementation (legal_actions_pos).
    def legal_actions_pos(self, pos):
        """
        Get all legal pawn moves.

        Parameters:
        - pos: int, current pawn position (linear index).

        Returns:
        - list of int: Linear indices of legal moves.
        """

        actions = []
        N = self.N
        walls = self.walls

        x, y = pos // N, pos % N
        enemy_pos = (N**2 - 1) - self.enemy[0]  # convert to player's perspective
        enemy_x, enemy_y = enemy_pos // N, enemy_pos % N

        def get_linear_index(nx, ny):
            return nx * N + ny

        def is_within_board(nx, ny):
            return 0 <= nx < N and 0 <= ny < N

        def is_wall_blocking(x, y, nx, ny):
            """Check if a wall is blocking movement between (x, y) and (nx, ny)."""
            if nx > x:  # Moving down
                blocking_bottom_right = (walls[x * (N - 1) + y] == 1 if y < N - 1 else False)
                blocking_bottom_left = (walls[x * (N - 1) + y - 1] == 1 if y > 0 else False)
                return blocking_bottom_right or blocking_bottom_left

            if nx < x:  # Moving up
                blocking_top_right = (walls[(x - 1) * (N - 1) + y] == 1 if y < N - 1 else False)
                blocking_top_left = (walls[(x - 1) * (N - 1) + y - 1] == 1 if y > 0 else False)
                return blocking_top_right or blocking_top_left

            if ny > y:  # Moving right
                blocking_bottom_right = (walls[x * (N - 1) + y] == 2 if x < N - 1 else False)
                blocking_top_right = (walls[(x - 1) * (N - 1) + y] == 2 if x > 0 else False)
                return blocking_bottom_right or blocking_top_right

            if ny < y:  # Moving left
                blocking_bottom_left = (walls[x * (N - 1) + (y - 1)] == 2 if x < N - 1 else False)
                blocking_top_left = (walls[(x - 1) * (N - 1) + (y - 1)] == 2 if x > 0 else False)
                return blocking_bottom_left or blocking_top_left

            return False  # No movement

        for dx, dy in MOVEMENT_DIRECTIONS:
            nx, ny = x + dx, y + dy
            if is_within_board(nx, ny):
                if not is_wall_blocking(x, y, nx, ny):  # Check if direct move is valid
                    np = get_linear_index(nx, ny)
                    if (nx, ny) == (enemy_x, enemy_y):  # If enemy is blocking, consider jumping
                        nnx, nny = nx + dx, ny + dy
                        if is_within_board(nnx, nny) and not is_wall_blocking(nx, ny, nnx, nny):
                            actions.append(get_linear_index(nnx, nny))  # straight jump
                        else:  # Consider diagonal jumps
                            if dx != 0:  # Moving vertically, check left and right
                                if is_within_board(nx, ny - 1) and not is_wall_blocking(nx, ny, nx, ny - 1):
                                    actions.append(get_linear_index(nx, ny - 1))
                                if is_within_board(nx, ny + 1) and not is_wall_blocking(nx, ny, nx, ny + 1):
                                    actions.append(get_linear_index(nx, ny + 1))
                            elif dy != 0:  # Moving horizontally, check up and down
                                if is_within_board(nx - 1, ny) and not is_wall_blocking(nx, ny, nx - 1, ny):
                                    actions.append(get_linear_index(nx - 1, ny))
                                if is_within_board(nx + 1, ny) and not is_wall_blocking(nx, ny, nx + 1, ny):
                                    actions.append(get_linear_index(nx + 1, ny))
                    else:  # Direct move
                        actions.append(np)

        return actions

    # TODO: REPLACE THIS BUGGY MESS
    def legal_actions_pos_old(self, pos):
        actions = []

        N = self.N
        walls = self.walls
        ep = self.enemy[0]

        x, y = pos // N, pos % N
        for dx, dy in MOVEMENT_DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N:  # within the game board
                np = N * nx + ny  # linear index of the proposed new pawn position
                wp = (N - 1) * nx + ny

                # move up
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
                # move down
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
                # move left
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
                        if walls[wp - (N - 1)] != 2 and walls[wp] != 2:  # not directly blocked by wall
                            if np + ep != N ** 2 - 1:  # not blocked by enemy
                                actions.append(np)
                            else:  # blocked by enemy - test for possible jumps
                                if ny > 0 and walls[wp - (N - 1) - 1] != 2 and walls[wp - 1] != 2:  # straight jump
                                    nnp = np - 1
                                    actions.append(nnp)
                                else:  # diagonal jumps
                                    if (ny == 0 and walls[wp - (N - 1)] != 1) or (
                                            ny > 0 and walls[wp - (N - 1) - 1] != 1 and walls[wp - (N - 1)] != 1):
                                        nnp = np - N
                                        actions.append(nnp)

                                    if (ny == 0 and walls[wp] != 1) or (
                                            ny > 0 and walls[wp - 1] != 1 and walls[wp] != 1):
                                        nnp = np + N
                                        actions.append(nnp)
                # move right
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
                                            ny < (N - 1) and walls[wp - 1] != 1 and walls[wp] != 1):
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
        # TODO: potential speedup by not calling can_reach_goal if wall isn't touching any other wall.
        def can_reach_goal(orientation, pos):
            def bfs(state):
                visited = set()

                visited.add(state.player[0])  # mark root node as visited
                queue = deque([state.player[0]])
                while queue:
                    # print(queue)
                    # print(visited)
                    position = queue.popleft()

                    # print(f'pop {position} == {(position//N, position%N)}')
                    if position // N == 0:  # reached goal (farthest row)
                        return True
                    else:  # search child nodes (adjacent positions)
                        new_positions = state.legal_actions_pos(position)
                        # print([(position//N, position%N) for position in new_positions])
                        for new_position in new_positions:
                            if new_position not in visited:
                                visited.add(new_position)
                                queue.append(new_position)
                return False
            # TODO: potential speedup by modifying then searching self (and reverting afterwards) instead of cloning.
            # Check if player can still reach goal
            player_state = State(board_size=N, player=self.player.copy(), enemy=self.enemy.copy(),
                                 walls=self.walls.copy(), plies_played=self.plies_played)
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
        state = State(board_size=N, player=self.player.copy(), enemy=self.enemy.copy(), walls=self.walls.copy(),
                      plies_played=self.plies_played + 1)

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
        return self.plies_played % 2 == 0

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

