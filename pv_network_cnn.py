"""
Define the Policy-Value (PV) Network using a CNN architecture.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from BaseNetwork import BaseNetwork

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
        pass

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


