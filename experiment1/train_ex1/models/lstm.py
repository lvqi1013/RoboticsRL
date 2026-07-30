import torch
from torch import nn
from torch import Tensor

class LSTMRegressor(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()

        self.lstm_model = nn.LSTM(
            input_size=1,
            hidden_size=192,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        self.head = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, d_out))
    
    def forward(self,  x: Tensor) -> Tensor:
        seq = x.unsqueeze(-1)
        out, _ = self.lstm_model(seq)
        return self.head(out[:, -1, :])
    
"""
输入 x: [Batch, Seq_Len] 
       ↓ unsqueeze(-1)
序列数据: [Batch, Seq_Len, 1]
       ↓ LSTM (2层, hidden=192)
LSTM输出: [Batch, Seq_Len, 192]
       ↓ 取最后一个时间步 [:, -1, :]
特征向量: [Batch, 192]
       ↓ LayerNorm + Linear
输出 y: [Batch, d_out]
"""