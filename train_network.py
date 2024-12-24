import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import numpy as np
import pickle
from pv_network_cnn import Network, INPUT_SHAPE, POLICY_OUTPUT_SIZE, NUM_FILTERS, NUM_RESIDUAL_BLOCKS, BOARD_SIZE


NUM_EPOCH = 100
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data():
    # Load the latest training data
    history_path = sorted(Path('./data').glob('*.history'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)


# TODO: simplify this function. maybe directly produce 2d arrays from raw input. Replace all references of pieces_array. Finally make this method a cnn class method.
def preprocess_input(game_state_arrays):
    """Processes raw input given as a list of game state arrays, where each game state is of the form
     [player, enemy, walls], i.e. the output of State.to_array(). Input is converted to the form accepted
     by the neural network.
     :returns input: transformed input to be accepted by the neural network."""
    N = BOARD_SIZE

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

    processed_input = [[pawn_table(state[0]), pawn_table(state[1]), wall_table(state[2])] for state in game_state_arrays]
    C, H, W = INPUT_SHAPE
    processed_input = np.array(processed_input).reshape(len(processed_input), C, H, W)  # Shape: (N, C, H, W)
    print(processed_input)

    return processed_input




def train_network():
    # Load the training data
    history = load_data()
    s, p, v = zip(*history)

    # Reshape the input data for training
    # C, H, W = INPUT_SHAPE
    # s = np.array(s).reshape(len(s), C, H, W)  # Shape: (N, C, H, W)
    s = preprocess_input(s)
    p = np.array(p)  # Policy targets
    v = np.array(v)  # Value targets

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
