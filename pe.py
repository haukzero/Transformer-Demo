import math
import torch
from torch import nn


class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(max_len).unsqueeze(1)
        # 先用 exp 计算可以加速计算过程
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(1e4) / d_model))
        # sin: 2i  cos: 2i + 1
        pe[ :, 0::2 ] = torch.sin(position * div_term)
        pe[ :, 1::2 ] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        # 设置为缓冲区, 以便在不通设备间传输模型时保持其状态
        self.register_buffer('pe', pe)

    def forward(self, x):
        x += self.pe[ :, :x.shape[ 1 ], :x.shape[ 2 ] ]
        return self.dropout(x)


if __name__ == '__main__':
    d_model = 4
    max_len = 9
    x = torch.randn(1, max_len, d_model)
    pe = PositionEncoding(d_model, max_len=max_len)
    print(f"x = {x}")
    print(f"pe(x) = {pe(x)}")
