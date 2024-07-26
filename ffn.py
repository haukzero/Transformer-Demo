from torch import nn


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.5, device='cpu'):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff).to(device)
        self.fc2 = nn.Linear(d_ff, d_model).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return self.dropout(x)
