# TODO: fix bug where game freezes when invalid wall placement is attempted.
#       Could also be just attempting to place a wall when you have none.
# TODO: be able to choose which colour you start as (black or white).


# Importing necessary packages and modules
import torch
from game import State
from pv_mcts_pytorch import pv_mcts_action, random_action
from dual_network_pytorch import DualNetwork, DN_INPUT_SHAPE, DN_POLICY_OUTPUT_SIZE, DN_FILTERS, DN_RESIDUAL_NUM
import tkinter as tk

# Loading the best player's model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = './model_pytorch/best.pth'
model = DualNetwork(DN_INPUT_SHAPE[0], DN_FILTERS, DN_RESIDUAL_NUM, DN_POLICY_OUTPUT_SIZE)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Defining the Game UI
class GameUI(tk.Frame):
    # Initialization
    def __init__(self, master=None, model=None):
        tk.Frame.__init__(self, master)
        self.master.title('Quoridor')

        # Generating the game state
        self.state = State()
        self.N = self.state.N
        self.D = 200  # Cell size (pixels)
        self.L = self.N * self.D  # Canvas size

        self.select = -1  # Selection (-1: none, 0~(N*N-1): square)
        self.placing_wall = False  # Flag to indicate if we are placing a wall

        # Creating the function for action selection using PV MCTS
        self.next_action = pv_mcts_action(model, device=device) if model else random_action()

        # Main frame layout
        self.grid()

        # Creating the canvas for the game board
        self.c = tk.Canvas(self, width=self.L, height=self.L, highlightthickness=0)
        self.c.bind('<Button-1>', self.turn_of_human)
        self.c.grid(row=1, column=1, padx=10, pady=10)

        # Displaying the player's walls on the left
        self.player_walls_frame = tk.Frame(self)
        self.player_walls_frame.grid(row=1, column=2, padx=10, pady=10)
        self.player_walls = tk.Label(self.player_walls_frame, text="Player Walls", anchor="center", justify=tk.CENTER, font=('Helvetica', 24))
        self.player_walls.pack()

        # Displaying the enemy's walls on the right
        self.enemy_walls_frame = tk.Frame(self)
        self.enemy_walls_frame.grid(row=1, column=0, padx=10, pady=10)
        self.enemy_walls = tk.Label(self.enemy_walls_frame, text="Enemy Walls", anchor="center", justify=tk.CENTER, font=('Helvetica', 24))
        self.enemy_walls.pack()

        # Displaying the action buttons below the game board
        self.controls_frame = tk.Frame(self)
        self.controls_frame.grid(row=2, column=1, padx=10, pady=10)
        self.wall_button = tk.Button(self.controls_frame, text="Place Wall", command=self.place_wall_mode)
        self.wall_button.pack()

        self.wall_direction = tk.StringVar(value="horizontal")
        self.wall_horizontal_button = tk.Radiobutton(self.controls_frame, text="Horizontal", variable=self.wall_direction, value="horizontal")
        self.wall_vertical_button = tk.Radiobutton(self.controls_frame, text="Vertical", variable=self.wall_direction, value="vertical")
        self.wall_horizontal_button.pack()
        self.wall_vertical_button.pack()

        # Result message
        self.result_message = tk.Label(self, text="", font=('Helvetica', 60))
        self.result_message.grid(row=0, column=1, pady=10)

        # Updating the drawing
        self.on_draw()

    def place_wall_mode(self):
        self.placing_wall = not self.placing_wall
        self.wall_button.config(text="Move Piece" if self.placing_wall else "Place Wall")

    # Human's turn
    def turn_of_human(self, event):
        N = self.N
        D = self.D
        # If the game is over
        if self.state.is_done():
            return

        # If it is not the first player's turn
        if not self.state.is_first_player():
            return

        # Calculate the selection and move position
        if self.placing_wall:
            x, y = (event.x - D // 2) // D, (event.y - D // 2) // D
            print(x, y)
            if 0 <= x < N - 1 and 0 <= y < N - 1:
                self.place_wall(x, y)
        else:
            x, y = event.x // D, event.y // D
            self.select = N * y + x
            action = self.select

            # Convert selection and move to action

            # If the action is not legal
            if not (action in self.state.legal_actions()):
                self.select = -1
                self.on_draw()
                return

            # Get the next state
            self.state = self.state.next(action)
            self.select = -1
            self.on_draw()

        # AI's turn
        self.master.after(500, self.turn_of_ai)

    def place_wall(self, x, y):
        N = self.N
        # Adjusted logic for placing walls at grid points
        if self.wall_direction.get() == "horizontal":
            action = N ** 2 + (N - 1) * y + x
        else:
            action = N ** 2 + (N - 1) ** 2 + (N - 1) * y + x

        # Check if the action is legal
        if action in self.state.legal_actions():
            # Get the next state
            self.state = self.state.next(action)
            self.placing_wall = False
            self.wall_button.config(text="Place Wall")
            self.on_draw()
        else:
            self.placing_wall = False
            self.wall_button.config(text="Place Wall")
            self.on_draw()

    # AI's turn
    def turn_of_ai(self):
        # If the game is over
        if self.state.is_done():
            self.display_result()
            self.master.after(1000, self.reset_game)
            return

        # Get the action
        action = self.next_action(self.state)

        # Get the next state
        self.state = self.state.next(action)
        self.on_draw()

        if self.state.is_done():
            self.display_result()
            self.master.after(1000, self.reset_game)
            return

    def display_result(self):
        is_lose = self.state.is_lose() if self.state.is_first_player() else not self.state.is_lose()
        if is_lose:
            self.result_message.config(text="You Lose", fg="blue")
        else:
            self.result_message.config(text="You Win", fg="red")
    
    def reset_game(self):
        self.state = State()
        self.on_draw()
        self.result_message.config(text="")

    # Draw the piece
    def draw_piece(self, index, color):
        N = self.N
        D = self.D
        x = (index % N) * D
        y = (index // N) * D
        margin = D // 10
        self.c.create_oval(x + margin, y + margin, x + D - margin, y + D - margin, fill=color, outline='black')

    # Draw the walls
    def draw_walls(self):
        N = self.N
        D = self.D
        for i in range(len(self.state.walls)):
            x, y = i % (N - 1), i // (N - 1)
            if self.state.walls[i] == 1:
                x1, y1 = x * D, (y + 1) * D
                x2, y2 = (x + 2) * D, (y + 1) * D
                self.c.create_line(x1, y1, x2, y2, width=16.0, fill='#D1B575')
            elif self.state.walls[i] == 2:
                x1, y1 = (x + 1) * D, y * D
                x2, y2 = (x + 1) * D, (y + 2) * D
                self.c.create_line(x1, y1, x2, y2, width=16.0, fill='#D1B575')

    # Update the drawing
    def on_draw(self):
        N = self.N
        D = self.D
        L = self.L
        is_first_player = self.state.is_first_player()

        # Grid
        self.c.delete('all')
        self.c.create_rectangle(0, 0, L, L, width=0.0, fill='#4B4B4B')
        for i in range(1, N):
            self.c.create_line(i * D, 0, i * D, L, width=16.0, fill='#8B0000')
            self.c.create_line(0, i * D, L, i * D, width=16.0, fill='#8B0000')

        # Pieces
        p_pos = self.state.player[0] if is_first_player else self.state.enemy[0]
        e_pos = self.state.enemy[0] if is_first_player else self.state.player[0]
        e_pos = N ** 2 - 1 - e_pos

        self.draw_piece(p_pos, '#D2B48C')
        self.draw_piece(e_pos, '#5D3A3A')

        p_walls = self.state.player[1] if is_first_player else self.state.enemy[1]
        e_walls = self.state.enemy[1] if is_first_player else self.state.player[1]

        # Update the wall count
        self.player_walls.config(text=f"Player Walls\n{p_walls}")
        self.enemy_walls.config(text=f"Enemy Walls\n{e_walls}")

        if not is_first_player:
            self.state.rotate_walls()

        # Walls
        self.draw_walls()

        if not is_first_player:
            self.state.rotate_walls()

# Run the game UI
if __name__ == '__main__':
    f = GameUI(model=model)
    f.pack()
    f.mainloop()
