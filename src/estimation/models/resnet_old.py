import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..layers.resnet import ResLayer

class ResNet(nn.Module):
    def __init__(
            self
    ):
        super(ResNet, self).__init__()

        self.in_ref = nn.Conv2d(3, 8, kernel_size=7, padding='same')

        self.conv1 = ResLayer(8, 32, kernel_size=5, dropout=0.1)
        self.pool1 = nn.AvgPool2d(2,2)
        self.conv2 = ResLayer(32, 128, kernel_size=3, dropout=0.1)
        self.pool2 = nn.AvgPool2d(4,4)
        self.conv3 = ResLayer(128, 512, kernel_size=3, dropout=0.1)
        self.pool3 = nn.AvgPool2d(5,5)

        self.fc1 = nn.Linear(512,64)
        self.fc2 = nn.Linear(1600,500)
        self.fc3 = nn.Linear(500,83)

        self.sm = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_ref(x)
        h = self.pool1(F.relu(self.conv1(h)))
        h = self.pool2(F.relu(self.conv2(h)))
        h = self.pool3(F.relu(self.conv3(h)))

        h = rearrange(h, 'b c h w -> b h w c')
        h = F.silu(self.fc1(h))

        h = rearrange(h, 'b h w c -> b (h w c)')
        h = F.silu(self.fc2(h))
        h = F.silu(self.fc3(h))

        return self.sm(h)

