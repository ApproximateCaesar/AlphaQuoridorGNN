"""
Define the Policy-Value (PV) Network using a CNN architecture.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from BaseNetwork import BaseNetwork
import numpy as np

# Parameters
from constants import BOARD_SIZE, PV_NETWORK_PATH
NUM_FILTERS = 128  # Number of kernels/filters in the convolutional layer (256 in AlphaZero)
NUM_RESIDUAL_BLOCKS = 16  # Number of residual blocks (19 in AlphaZero)
INPUT_SHAPE = (6, BOARD_SIZE, BOARD_SIZE)  # Input shape: (Channels, Height, Width) for PyTorch Conv2d
POLICY_OUTPUT_SIZE = BOARD_SIZE ** 2 + 2 * (BOARD_SIZE - 1) ** 2  # Number of possible actions


# Convolutional layer with batch normalization and ReLU
class ConvBN(nn.Module):
    def __init__(self, num_channels, num_filters):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(num_channels, num_filters, kernel_size=3, padding='same', bias=False)
        self.bn = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv_bn1 = ConvBN(num_filters,num_filters)
        self.conv_bn2 = ConvBN(num_filters,num_filters)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv_bn1(x))
        x = self.conv_bn2(x)
        x += residual  # this is called a "skip connection"
        return F.relu(x)


# Network model
class CNNNetwork(BaseNetwork):

    @property
    def name(self):
        return self._name

    def __init__(self):
        super(CNNNetwork, self).__init__()
        self._name = 'CNN'

        # NN layers
        self.conv = ConvBN(INPUT_SHAPE[0], NUM_FILTERS)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(NUM_FILTERS) for _ in range(NUM_RESIDUAL_BLOCKS)]
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(NUM_FILTERS, POLICY_OUTPUT_SIZE),
            nn.Softmax(dim=1)
        )

        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(NUM_FILTERS, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.residual_blocks(x)
        x = self.global_avg_pool(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

    def preprocess_input(self, game_state_arrays):
        """Processes raw input given as a list of game state arrays, where each game state is of the form
        [player, enemy, walls], i.e. the output of State.to_array(). Input is converted to the form accepted
        by the neural network.
        :returns processed_input: transformed input to be accepted by the neural network."""
        N = BOARD_SIZE
        num_states = len(game_state_arrays)
        processed_input = np.zeros((num_states, *INPUT_SHAPE), dtype=np.float32)

        for i, (player, enemy, walls) in enumerate(game_state_arrays):
            processed_input[i, 0, player[0] // N, player[0] % N] = 1  # Player pawn table
            processed_input[i, 1, :, :] = player[1]  # Player remaining walls table

            processed_input[i, 2, enemy[0] // N, enemy[0] % N] = 1  # Enemy pawn table
            processed_input[i, 3, :, :] = enemy[1]  # Enemy remaining walls table

            # Wall tables
            for wall_index, wall in enumerate(walls):
                if wall != 0:
                    top_left_tile_index = N * (wall_index // (N - 1)) + (wall_index % (N - 1))
                    row, col = divmod(top_left_tile_index, N)
                    if wall == 1:  # Horizontal wall
                        processed_input[i, 4, row, col] = 1
                    elif wall == 2:  # Vertical wall
                        processed_input[i, 5, row, col] = 1

        return processed_input


    def predict(self, state, device):
        """Predict the policy and value for a game state given a State object.
                :returns: policy, value, where policy is a normalised PMF over all legal actions (as a 1D numpy array)
                and value is a float between -1 and 1."""
        # Reshape input data for inference
        x = self.preprocess_input([state.to_array()])
        x = torch.tensor(x, dtype=torch.float32).to(device)

        with torch.inference_mode():  # disable gradient calculation to provide inference speedup
            # with torch.autocast(device_type=device):  # using mixed precision showed a slight slowdown
            policy, value = self.optimised_model(x)

        policy = policy[0][list(state.legal_actions())]  # Remove batch dimension and restrict to legal actions

        # Normalize policy distribution over the subset of legal actions
        policy /= torch.sum(policy) if torch.sum(policy) else 1

        policy = policy.cpu().numpy()  # tensor on GPU to numpy array on CPU
        value = value.item()  # GPU tensor to float

        return policy, value

    def train_model(self, data_loader, optimizer, loss_fn, device='cpu', num_epochs=10):
        pass


# Function to create the policy-value network
def create_network():

    # Do nothing if the model is already created
    if os.path.exists(PV_NETWORK_PATH + 'best.pth'):
        return

    # Initialize the model
    model = CNNNetwork()

    # Save the model
    os.makedirs(PV_NETWORK_PATH, exist_ok=True)
    torch.save(model.state_dict(), PV_NETWORK_PATH + 'best.pth')


# Running the function
if __name__ == '__main__':
    create_network()


