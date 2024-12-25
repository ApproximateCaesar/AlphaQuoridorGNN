from game_logic import State

state = State()

state.walls[24] = 1
state.walls[27] = 1
state.walls[32] = 2
state.walls[36] = 2
state.walls[37] = 1
state.walls[41] = 1
state.walls[42] = 2
state.walls[43] = 1
# state.walls[26] = 2 # illegal wall placed by enemy

state.player[0] = 40
state.enemy[0] = 32

print(state)

N = 9
print(state.legal_actions_wall(pos=26))  # shows as legal




