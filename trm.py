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
        self.device = device
        self.encoder = Encoder(n, n_vocab, d_model,
                               d_k, n_head, d_ff, pad_token,
                               max_len, dropout, device)
        self.decoder = Decoder(n, n_vocab, d_model, d_k,
                               d_v, n_head, d_ff, pad_token,
                               max_len, dropout, device)
        self.decoder.embd.weight = self.encoder.embd.weight
        self.proj = nn.Linear(d_model, n_vocab).to(device)

    def forward(self, x_enc, x_dec):
        enc_output = self.encoder(x_enc)
        dec_output = self.decoder(x_dec, x_enc, enc_output)
        # logits: (batch_size, n_vocab, n_vocab)
        logits = self.proj(dec_output)
        return logits.view(-1, logits.size(-1))

    def greedy_decoder(self, x_enc, start_token, end_token, pad_token=0):
        self.decoder.use_kv_cache()
        enc_output = self.encoder(x_enc)
        dec_input = torch.ones_like(x_enc).type_as(x_enc) * pad_token
        next_token = torch.ones(x_enc.shape[ 0 ]).to(self.device) * start_token
        end_token = torch.ones(x_enc.shape[ 0 ]).to(self.device) * end_token
        stop_flag = torch.ones(x_enc.shape[ 0 ]).to(self.device) * pad_token
        i = 0
        pred = torch.ones_like(x_enc).type_as(x_enc) * pad_token
        while (not (torch.all(next_token == stop_flag))
               and (not torch.all(next_token == end_token))
               and i < x_enc.shape[ 1 ]):
            dec_input[ :, i ] = next_token
            dec_output = self.decoder(dec_input, x_enc, enc_output, i)
            logits = self.proj(dec_output)
            pred = logits.argmax(dim=-1, keepdim=False)
            next_token = pred[ :, i ]
            i += 1
        self.decoder.clear_cache()
        return pred
