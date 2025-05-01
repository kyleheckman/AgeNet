import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(
            self,
            in_size: int,
            out_size: int,
            layers: int
    ):
        super(FeedForward, self).__init__()

        ff_layers = []
        fc_in = in_size
        for _ in range(layers-1):
            fc_out = fc_in // 4
            ff_layers.append(nn.Linear(fc_in, fc_out))
            ff_layers.append(nn.SiLU())
            fc_in = fc_out
        
        ff_layers.append(nn.Linear(fc_in, out_size))
        ff_layers.append(nn.SiLU())

        self.layers = nn.ModuleList(ff_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.layers:
            x = module(x)
        return x