# ====================
# Creating the Dual Network using Pytorch in order to use GPU computation.
# ====================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Parameters
from constants import BOARD_SIZE
DN_FILTERS = 128  # Number of kernels/filters in the convolutional layer (256 in the original version)
DN_RESIDUAL_NUM = 16  # Number of residual blocks (19 in the original version)
DN_INPUT_SHAPE = (6, BOARD_SIZE, BOARD_SIZE)  # Input shape (Channels, Height, Width) for PyTorch Conv2d
DN_POLICY_OUTPUT_SIZE = BOARD_SIZE ** 2 + 2 * (BOARD_SIZE - 1) ** 2  # Number of possible actions


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
    def __init__(self, num_channels, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv_bn1 = ConvBN(num_channels,num_filters)
        self.conv_bn2 = ConvBN(num_channels,num_filters)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv_bn1(x))
        x = self.conv_bn2(x)
        x += residual  # this is called a "skip connection"
        return F.relu(x)


# Dual network model
class DualNetwork(nn.Module):
    def __init__(self, num_channels, num_filters, num_residual_blocks, policy_output_size):
        super(DualNetwork, self).__init__()
        self.conv = ConvBN(num_channels,num_filters)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_channels,num_filters) for _ in range(num_residual_blocks)]
        )
        # TODO: verify the NN architecture is correct from hereon
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters, policy_output_size),
            nn.Softmax(dim=1)
        )

        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.residual_blocks(x)
        x = self.global_avg_pool(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


# Function to create the dual network
def create_dual_network():
    model_path = './model_pytorch/best.pth'

    # Do nothing if the model is already created
    if os.path.exists(model_path):
        return

    # Initialize the model
    model = DualNetwork(DN_INPUT_SHAPE[0], DN_FILTERS, DN_RESIDUAL_NUM, DN_POLICY_OUTPUT_SIZE)
    print(model)
    # Save the model
    os.makedirs('./model_pytorch/', exist_ok=True)
    torch.save(model.state_dict(), model_path)


# Running the function
if __name__ == '__main__':
    create_dual_network()


