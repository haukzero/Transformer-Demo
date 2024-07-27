import utils
from torch import nn
from pe import PositionEncoding
from mha import MultiHeadAttention
from ffn import FFN


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head, d_ff, dropout=0.5, device='cpu', use_cache=False):
        super().__init__()
        self.self_mha = MultiHeadAttention(d_model,
                                           d_k,
                                           d_k,
                                           n_head,
                                           device,
                                           1 if use_cache else 0)
        self.mha = MultiHeadAttention(d_model,
                                      d_k,
                                      d_v,
                                      n_head,
                                      device,
                                      2 if use_cache else 0)
        self.ffn = FFN(d_model, d_ff, dropout, device)

    def forward(self, x, x_enc, dec_self_mask, dec_enc_mask, i=None):
        # x: (batch_size, n_seq, d_model)
        # x_enc: (batch_size, n_seq, d_model)
        x = self.self_mha(x, x, x, dec_self_mask, i)
        x = self.mha(x, x_enc, x_enc, dec_enc_mask)
        return self.ffn(x)


class Decoder(nn.Module):
    def __init__(self, n, n_vocab, d_model,
                 d_k, d_v, n_head, d_ff,
                 pad_token=0, max_len=5000,
                 dropout=0.5, device='cpu',
                 use_cache=False):
        super().__init__()
        self.pad_token = pad_token
        self.device = device
        self.embd = nn.Embedding(n_vocab, d_model).to(device)
        self.pe = PositionEncoding(d_model, max_len, dropout).to(device)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_k, d_v, n_head, d_ff, dropout, device, use_cache)
            for _ in range(n)
        ])

    def forward(self, x, enc_input, enc_output, i=None):
        # x: (batch_size, n_seq)
        # enc_input: (batch_size, n_seq)
        # enc_output: (batch_size, n_seq, d_model)

        dec_self_pad_mask = utils.get_pad_mask(x, x, self.pad_token).to(self.device)
        dec_self_attn_mask = utils.get_attn_mask(x).to(self.device)
        dec_self_mask = utils.bool_mask(dec_self_pad_mask, dec_self_attn_mask)

        dec_enc_pad_mask = utils.get_pad_mask(x, enc_input, self.pad_token).to(self.device)
        dec_enc_mask = utils.bool_mask(dec_enc_pad_mask)

        # x: (batch_size, n_seq, d_model)
        x = self.embd(x)
        x = self.pe(x)

        for layer in self.layers:
            x = layer(x, enc_output, dec_self_mask, dec_enc_mask, i)

        return x

    def use_kv_cache(self):
        for layer in self.layers:
            layer.self_mha.cache_flag = 1
            layer.mha.cache_flag = 2

    def clear_cache(self):
        for layer in self.layers:
            layer.self_mha.cache_flag = 0
            layer.self_mha.past_q = None
            layer.self_mha.past_k = None
            layer.self_mha.past_v = None

            layer.mha.cache_flag = 0
            layer.mha.past_k = None
            layer.mha.past_v = None
