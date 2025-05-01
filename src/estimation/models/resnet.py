import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os # Added os for path joining below
import sys # Added sys for path manipulation below


current_model_dir = os.path.dirname(os.path.abspath(__file__))
estimation_dir = os.path.dirname(current_model_dir)
src_dir = os.path.dirname(estimation_dir)
root_dir_from_model = os.path.dirname(src_dir)
if root_dir_from_model not in sys.path:
    sys.path.insert(0, root_dir_from_model)


try:
    from ..layers.resnet import ResLayer
except ImportError:
    print("Relative import of ResLayer failed in models/resnet.py. Check structure/path.")
    exit(1)


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

        # Fully Connected Layers
        # Calculate input size to fc2 dynamically based on pooling
        # Input 200x200 -> pool1(2x2) -> 100x100 -> pool2(4x4) -> 25x25 -> pool3(5x5) -> 5x5
        # After conv3, feature map size is 5x5 with 512 channels
        # After fc1 (applied per location), size is 5x5 with 64 channels
        fc1_output_channels = 64
        fc2_input_size = 5 * 5 * fc1_output_channels # 5x5 spatial, 64 channels = 1600

        self.fc1 = nn.Linear(512, fc1_output_channels) 
        self.fc2 = nn.Linear(fc2_input_size, 500)
        self.fc3 = nn.Linear(500, 83) # Output raw logits for 83 classes (ages 8-90)

        # Removed Softmax layer: self.sm = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, 3, 200, 200)
        h = self.in_ref(x)
        h = self.pool1(F.relu(self.conv1(h))) # Shape: (batch, 32, 100, 100)
        h = self.pool2(F.relu(self.conv2(h))) # Shape: (batch, 128, 25, 25)
        h = self.pool3(F.relu(self.conv3(h))) # Shape: (batch, 512, 5, 5)

        # Apply fc1 channel-wise using rearrange
        h = rearrange(h, 'b c h w -> b h w c') # Shape: (batch, 5, 5, 512)
        h = F.silu(self.fc1(h)) # Shape: (batch, 5, 5, 64)

        # Flatten for fc2
        h = rearrange(h, 'b h w c -> b (h w c)') # Shape: (batch, 1600)
        h = F.silu(self.fc2(h)) # Shape: (batch, 500)

        # Output logits from fc3
        logits = self.fc3(h) # Shape: (batch, 83)

        # Removed Softmax application: return self.sm(h)
        return logits