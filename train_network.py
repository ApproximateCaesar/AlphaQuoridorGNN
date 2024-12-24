import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import numpy as np
import pickle
from pv_network_cnn import Network, INPUT_SHAPE, POLICY_OUTPUT_SIZE, NUM_FILTERS, NUM_RESIDUAL_BLOCKS


NUM_EPOCH = 100
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data():
    # Load the latest training data
    history_path = sorted(Path('./data').glob('*.history'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)

# def preprocess_input(game_state_arrays):
#     """Processes raw input given as a list of game state arrays, where each game state is of the form
#      [player, enemy, walls], i.e. the output of State.to_array(). Input is converted to the form accepted
#      by the neural network.
#      :returns input: transformed input to be accepted by the neural network."""
#
#     N = self.N
#     player, enemy, walls = game_state_arrays
#
#     def pieces_of(pieces):
#         tables = []
#
#         table = [0] * (N ** 2)
#         table[pieces[0]] = 1
#         tables.append(table)
#
#         table = [pieces[1]] * (N ** 2)
#         tables.append(table)
#
#         return tables
#
#     def walls_of(walls):
#         tables = []
#
#         table_h = [0] * (N ** 2)
#         table_v = [0] * (N ** 2)
#
#         for wp in range((N - 1) ** 2):
#             x, y = wp // (N - 1), wp % (N - 1)
#
#             if x < (N - 1) // 2 and y < (N - 1) // 2:
#                 pos = N * x + y
#             elif x > (N - 1) // 2 and y < (N - 1) // 2:
#                 pos = N * x + (y + 1)
#             elif x < (N - 1) // 2 and y > (N - 1) // 2:
#                 pos = N * (x + 1) + y
#             else:
#                 pos = N * (x + 1) + (y + 1)
#
#             if walls[wp] == 1:
#                 table_h[pos] = 1
#             elif walls[wp] == 2:
#                 table_v[pos] = 1
#
#         tables.append(table_h)
#         tables.append(table_v)
#
#         return tables
#
#     return [pieces_of(self.player), pieces_of(self.enemy), walls_of(self.walls)]


def train_network():
    # Load the training data
    history = load_data()
    s, p, v = zip(*history)

    # Reshape the input data for training
    C, H, W = INPUT_SHAPE
    s = np.array(s).reshape(len(s), C, H, W)  # Shape: (N, C, H, W)
    p = np.array(p)  # Policy targets
    v = np.array(v)  # Value targets
    print(s)

    # Convert data to PyTorch tensors
    s = torch.tensor(s, dtype=torch.float32)
    p = torch.tensor(p, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)

    # Create a DataLoader for the dataset
    dataset = TensorDataset(s, p, v)

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load the model
    model = Network(INPUT_SHAPE[0], NUM_FILTERS, NUM_RESIDUAL_BLOCKS, POLICY_OUTPUT_SIZE)

    model_path = 'model/best.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    model = model.to(DEVICE)

    # Define the loss functions and optimizer
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # TODO: seems like lr=0.0001 is needed?

    # Learning rate scheduler
    def lr_lambda(epoch):
        if epoch >= 80:
            return 0.25  # Scale the LR to 25% of the initial value
        elif epoch >= 50:
            return 0.5   # Scale the LR to 50% of the initial value
        else:
            return 1.0   # Keep the LR at its initial value
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    for epoch in range(NUM_EPOCH):
        model.train()  # set model to training mode (might be unnecessary)

        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0

        for batch_idx, (state, policy_target, value_target) in enumerate(data_loader):
            # Move data to the device
            state = state.to(DEVICE)
            policy_target = policy_target.to(DEVICE)
            value_target = value_target.to(DEVICE)

            # Forward pass
            policy_pred, value_pred = model(state)

            # Compute losses
            policy_loss = policy_loss_fn(policy_pred, policy_target)
            value_loss = value_loss_fn(value_pred.squeeze(), value_target)

            # Total loss
            loss = policy_loss + value_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()

        # Update the learning rate
        lr_scheduler.step()

        # Print progress
        print(f"\rEpoch {epoch + 1}/{NUM_EPOCH} | Policy Loss: {epoch_policy_loss:.4f} | Value Loss: {epoch_value_loss:.4f}", end='')
    print('')

    # Save the latest model
    latest_model_path = 'model/latest.pth'
    torch.save(model.state_dict(), latest_model_path)


if __name__ == '__main__':
    train_network()
