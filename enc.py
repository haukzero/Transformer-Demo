import utils
from torch import nn
from pe import PositionEncoding
from mha import MultiHeadAttention
from ffn import FFN


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, n_head, d_ff, dropout=0.5, device='cpu'):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, d_k, d_k, n_head, device)
        self.ffn = FFN(d_model, d_ff, dropout, device)

    def forward(self, x, mask):
        # x: (batch_size, n_seq, d_model)
        x = self.mha(x, x, x, mask)
        return self.ffn(x)


class Encoder(nn.Module):
    def __init__(self, n, n_vocab, d_model,
                 d_k, n_head, d_ff, pad_token=0,
                 max_len=5000, dropout=0.5, device='cpu'):
        super().__init__()
        self.pad_token = pad_token
        self.device = device
        self.embd = nn.Embedding(n_vocab, d_model).to(device)
        self.pe = PositionEncoding(d_model, max_len, dropout).to(device)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_k, n_head, d_ff, dropout, device)
            for _ in range(n)
        ])

    def forward(self, x):
        # x: (batch_size, n_seq)

        pad_mask = utils.get_pad_mask(x, x, self.pad_token).to(self.device)
        mask = utils.bool_mask(pad_mask)

        # x: (batch_size, n_seq, d_model)
        x = self.embd(x)
        x = self.pe(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x
