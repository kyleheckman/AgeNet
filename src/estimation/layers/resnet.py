import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ResLayer(nn.Module):
    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            kernel_size: int = 3,
            dropout: float = 0,
            norm: bool = True
    ):
        super(ResLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding='same')

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

        self.dropout = nn.Dropout(p=dropout)

        if norm:
            self.norm = nn.BatchNorm2d(in_ch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.conv1(x))
        if hasattr(self, 'norm'):
            h1 = self.norm(h1)
        h1 = self.dropout(h1)
        h2 = F.relu(self.conv2(h1))
        return h2 + self.skip(x)