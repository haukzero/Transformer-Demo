import utils
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head, device='cpu'):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        self.W_q = nn.Linear(d_model, n_head * d_k).to(device)
        self.W_k = nn.Linear(d_model, n_head * d_k).to(device)
        self.W_v = nn.Linear(d_model, n_head * d_v).to(device)
        self.fc = nn.Linear(n_head * d_v, d_model).to(device)
        self.ln = nn.LayerNorm(d_model).to(device)

    def forward(self, x_q, x_k, x_v, mask):
        x, batch_size = x_q, x_q.shape[0]

        # q: (batch_size, n_head, len_q, d_k)
        q = self.W_q(x_q).reshape(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        # k: (batch_size, n_head, len_k, d_k)
        k = self.W_k(x_k).reshape(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        # v: (batch_size, n_head, len_v, d_v)
        v = self.W_v(x_v).reshape(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)

        # mask: (batch_size, len_q, len_k) -> (batch_size, n_head, len_q, len_k)
        mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        # attn: (batch_size, n_head, len_q, d_v)
        attn = utils.attention(q, k, v, mask)
        # attn: (batch_size, len_q, n_head * d_v)
        attn = attn.transpose(1, 2).reshape(batch_size, -1, self.n_head * self.d_v)

        return self.ln(self.fc(attn) + x)

