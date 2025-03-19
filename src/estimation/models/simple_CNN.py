import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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
        self.pool2 = nn.AvgPool2d(4,4)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.pool3 = nn.AvgPool2d(5,5)

        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(800, 400)
        self.fc3 = nn.Linear(400, 83)

        self.sm = nn.Softmax(dim=1)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.in_ref(x))
        h = F.relu(self.conv1(h))
        h = self.pool1(h)
        h = F.relu(self.conv2(h))
        h = self.pool2(h)
        h = F.relu(self.conv3(h))
        h = self.pool3(h)

        h = rearrange(h, 'b c h w -> b h w c')
        h = F.silu(self.fc1(h))
        h = h.view(-1, 5*5*32)
        h = F.silu(self.fc2(h))
        h = F.silu(self.fc3(h))

        return self.sm(h)
