

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseNetwork(nn.Module, ABC):
    """Abstract class representing a policy-value neural network"""
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def train_model(self, data_loader, optimizer, loss_fn, device='cpu', num_epochs=10):
        """Common training logic."""
        self.to(device)
        self.train()

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in data_loader:
                inputs, labels = self.preprocess_data(batch)  # Architecture-specific preprocessing
                inputs, labels = [x.to(device) for x in inputs], labels.to(device)

                optimizer.zero_grad()
                outputs = self.forward(*inputs)  # Call the forward method with preprocessed inputs
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

    @abstractmethod
    def preprocess_data(self, batch):
        """Preprocess the data before training (to be implemented by subclasses)."""
        pass