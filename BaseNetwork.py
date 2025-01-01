

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseNetwork(nn.Module, ABC):
    """Abstract class representing a policy-value neural network"""
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @property
    @abstractmethod
    def name(self):
        """Name of the network architecture to be used in file extensions."""
        pass

    @abstractmethod
    def train_model(self, data_loader, optimizer, loss_fn, device='cpu', num_epochs=10):
        """Performs model training."""
        pass


    @abstractmethod
    def preprocess_input(self, game_state_arrays):
        """Processes raw input given as a list of game state arrays, where each game state is of the form
            [player, enemy, walls], i.e. the output of State.to_array(). Input is converted to the form accepted
            by the neural network.
            :returns processed_input: transformed input to be accepted by the neural network."""
        pass

