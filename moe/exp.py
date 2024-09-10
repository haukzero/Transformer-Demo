import torch
from torch import nn


class Expert(nn.Module):
    def __init__(self, d_model, d_exp, dropout=0.1):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(d_model, d_exp)
        self.fc2 = nn.Linear(d_exp, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.dropout(self.fc2(x))
