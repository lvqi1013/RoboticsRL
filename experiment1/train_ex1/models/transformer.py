import torch
from torch import nn
from torch import Tensor

class TransformerRegressor(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        d_model = 128
        self.value_embed = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.zeros(1, d_in, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )        
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.value_embed(x.unsqueeze(-1)) + self.pos
        encoded = self.encoder(tokens)
        return self.head(encoded.mean(dim=1))        