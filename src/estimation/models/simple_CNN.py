import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os 
import sys 



current_model_dir = os.path.dirname(os.path.abspath(__file__))
estimation_dir = os.path.dirname(current_model_dir)
src_dir = os.path.dirname(estimation_dir)
root_dir_from_model = os.path.dirname(src_dir)
if root_dir_from_model not in sys.path:
    sys.path.insert(0, root_dir_from_model)

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

        #Fully Connected Layers
        # Calculate input sizes dynamically
        # Input 200x200 -> pool1(2x2) -> 100x100 -> pool2(4x4) -> 25x25 -> pool3(5x5) -> 5x5
        # After conv3, feature map size is 5x5 with 128 channels
        # After fc1 (applied per location), size is 5x5 with 32 channels
        fc1_output_channels = 32
        fc2_input_size = 5 * 5 * fc1_output_channels # 5x5 spatial, 32 channels = 800

        self.fc1 = nn.Linear(128, fc1_output_channels) # Applied channel-wise effectively by rearrange later
        self.fc2 = nn.Linear(fc2_input_size, 400)
        self.fc3 = nn.Linear(400, 83) # Output raw logits for 83 classes

        # Removed Softmax layer: self.sm = nn.Softmax(dim=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, 3, 200, 200)
        h = F.relu(self.in_ref(x))          # Shape: (batch, 16, 200, 200)
        h = F.relu(self.conv1(h))          # Shape: (batch, 32, 200, 200)
        h = self.pool1(h)                   # Shape: (batch, 32, 100, 100)
        h = F.relu(self.conv2(h))          # Shape: (batch, 64, 100, 100)
        h = self.pool2(h)                   # Shape: (batch, 64, 25, 25)
        h = F.relu(self.conv3(h))          # Shape: (batch, 128, 25, 25)
        h = self.pool3(h)                   # Shape: (batch, 128, 5, 5)

        # Apply fc1 channel-wise using rearrange
        h = rearrange(h, 'b c h w -> b h w c') # Shape: (batch, 5, 5, 128)
        h = F.silu(self.fc1(h))             # Shape: (batch, 5, 5, 32)

        # Flatten for fc2
        h = h.view(-1, 5*5*32)              # Shape: (batch, 800) Using original view

        h = F.silu(self.fc2(h))             # Shape: (batch, 400)

        # Output logits from fc3
        logits = self.fc3(h)                # Shape: (batch, 83)

        # Removed Softmax application: return self.sm(h)
        return logits