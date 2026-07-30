import torch
from torch import nn
from torch import Tensor

class MLPRegressor(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, d_out)
        )
    
    def forward(self, x:Tensor) -> Tensor:
        return self.model(x)

