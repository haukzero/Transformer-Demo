import torch
from torch import nn
from enc import Encoder
from dec import Decoder


class Transformer(nn.Module):
    def __init__(self, n, n_vocab, d_model,
                 d_k, d_v, n_head, d_ff,
                 pad_token=0, max_len=5000,
                 dropout=0.5, device='cpu'):
        super().__init__()
        self.encoder = Encoder(n, n_vocab, d_model,
                               d_k, n_head, d_ff, pad_token,
                               max_len, dropout, device)
        self.decoder = Decoder(n, n_vocab, d_model, d_k,
                               d_v, n_head, d_ff, pad_token,
                               max_len, dropout, device)
        self.proj = nn.Linear(d_model, n_vocab).to(device)

    def forward(self, x_enc, x_dec):
        enc_output = self.encoder(x_enc)
        dec_output = self.decoder(x_dec, x_enc, enc_output)
        # logits: (batch_size, n_vocab, n_vocab)
        logits = self.proj(dec_output)
        return logits.view(-1, logits.size(-1))