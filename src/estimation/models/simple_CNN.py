import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.feedforward import FeedForward

class SimpleCNN(nn.Module):
    def __init__(
            self
    ):
        super(SimpleCNN, self).__init__()

        self.in_ref = nn.Conv2d(3, 16, kernel_size=3, padding='same')

        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding='same')
        self.pool1 = nn.AvgPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.pool2 = nn.AvgPool2d(2,2)

        self.ff = FeedForward(64*50*50, 100, 4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.in_ref(x))
        h = F.relu(self.conv1(h))
        h = self.pool1(h)
        h = F.relu(self.conv2(h))
        h = self.pool2(h)
        h = h.view(-1, 64*50*50)
        h = self.ff(h)

        return h
