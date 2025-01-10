from zmq.backend import first

from constants import BOARD_SIZE
import torch
import torch_tensorrt
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseNetwork(nn.Module, ABC):
    """Abstract class representing a policy-value neural network"""
    def __init__(self):
        super(BaseNetwork, self).__init__()
        self.optimised_model = None  # compiled copy of the model for faster inference

    @property
    @abstractmethod
    def name(self):
        """Name of the network architecture to be used in file extensions."""
        pass

    def prep_for_inference(self, model_path):
        """Loads state_dict of parameters located at model_path and prepares the model for inference."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_state_dict(torch.load(model_path, map_location=device))
        self.eval()  # set to evaluation mode
        # self.optimised_model = torch.jit.script(self)  # slower optimisation but less dependencies
        self.to(device)
        self.optimised_model = torch_tensorrt.compile(
            self,
            inputs=[torch_tensorrt.Input((1, *(6, BOARD_SIZE, BOARD_SIZE)))],  # Specify the input shape
            enabled_precisions={torch.float32},  # Use FP32 precision
        )


    @abstractmethod
    def predict(self, state, device):
        """Predict the policy and value for a game state given a State object.
        :returns: policy, value, where policy is a normalised PMF over all legal actions (as a 1D numpy array)
        and value is a float between -1 and 1."""
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

