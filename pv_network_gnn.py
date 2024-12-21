"""
Define the Policy-Value (PV) Network using a GNN architecture.
"""
# TODO: try using DGL to speedup training/inference.
# TODO: implement abstract neural network class to subclass for CNN and GNN.
#  This class should have an abstract method for training the model (prefilled with code) and an abstract method
#   for data preprocessing which is called by the training method. Change history file to only store a minimal game rep as (player,enemy,walls).

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# Parameters
from constants import BOARD_SIZE
NUM_FEATURES = 6  # Node feature size (similar to input channels in CNN)
HIDDEN_DIM = 128  # Hidden dimension for GCN layers
NUM_GCN_LAYERS = 3  # Number of GCN layers
POLICY_OUTPUT_SIZE = BOARD_SIZE ** 2 + 2 * (BOARD_SIZE - 1) ** 2  # Number of possible actions

# Graph-based Policy-Value Network
class GraphPolicyValueNetwork(nn.Module):
    def __init__(self, num_features, hidden_dim, num_gcn_layers, policy_output_size):
        super(GraphPolicyValueNetwork, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.policy_output_size = policy_output_size

        # GCN layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(num_features, hidden_dim))
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, policy_output_size),
            nn.Softmax(dim=1)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

    def forward(self, x, edge_index, batch):
        # Pass through GCN layers with ReLU activations
        for layer in self.gcn_layers:
            x = F.relu(layer(x, edge_index))

        # Pooling to aggregate graph-level representation
        x = global_mean_pool(x, batch)

        # Policy and value outputs
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


# Function to create the dual network
def create_network():
    model_path = 'model/best.pth'

    # Do nothing if the model is already created
    if os.path.exists(model_path):
        return

    # Initialize the model
    model = GraphPolicyValueNetwork(NUM_FEATURES, HIDDEN_DIM, NUM_GCN_LAYERS, POLICY_OUTPUT_SIZE)

    # Save the model
    os.makedirs('model/', exist_ok=True)
    torch.save(model.state_dict(), model_path)


# Running the function
if __name__ == '__main__':
    create_network()
