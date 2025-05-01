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
            dropout: float = 0, #No dropout
            norm: bool = True
    ):
        super(ResLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding='same') #First convolutional layer
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding='same') #Second convolutional layer. Changes the number of channels

        #Define the skip connection path
        if in_ch != out_ch:
            # If input and output channels differ, use a 1x1 convolution
            # to project the input 'x' to the same number of channels as the output 'h2'.
            # This allows the addition operation in the forward pass.
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) #Why 1?
        else:
            # If input and output channels are the same, the skip connection
            # is just an identity mapping (it does nothing, passes 'x' through unchanged).
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