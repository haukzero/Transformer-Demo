import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn


def load_config(filename='model/config.json'):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def load_vocab(filename='model/vocab.json'):
    with open(filename, 'r') as f:
        vocab = json.load(f)
    return vocab


def sen2vec(sen, vocab='model/vocab.json'):
    with open(vocab, 'r') as f:
        vocab = json.load(f)
    vec = [ ]
    for sentence in sen:
        sen_list = sentence.split(' ')
        v = [ vocab[ i ] for i in sen_list ]
        vec.append(v)
    return torch.LongTensor(vec)


def get_pad_mask(seq_q, seq_k, pad_token=0):
    batch_size, len_q = seq_q.shape
    len_k = seq_k.shape[ 1 ]
    # 先放成 (batch_size, 1, len_k) 的形状
    mask = (seq_q == pad_token).unsqueeze(1)
    # 再填充成 (batch_size, len_q, len_k) 的形状
    return mask.expand(batch_size, len_q, len_k).byte()


def get_attn_mask(seq):
    mask_shape = [ seq.shape[ 0 ], seq.shape[ 1 ], seq.shape[ 1 ] ]
    # 上三角为 1
    mask = np.triu(np.ones(mask_shape), k=1)
    # (batch_size, n_seq, n_seq) == (batch_size, len_q, len_k)
    return torch.from_numpy(mask).byte()


def bool_mask(pad_mask, attn_mask=None):
    attn_mask = torch.zeros_like(pad_mask) if attn_mask is None else attn_mask
    mask = (pad_mask + attn_mask) > 0
    return mask


def attention(Q, K, V, mask):
    d_k = K.shape[ -1 ]
    scores = (Q @ K.transpose(-1, -2)) / np.sqrt(d_k)
    # 用 -1e10 代替 -inf 防止出现 nan
    scores.masked_fill_(mask, -1e10)
    return nn.Softmax(dim=-1)(scores) @ V


def draw_loss(losses):
    x = np.arange(len(losses))
    plt.plot(x, losses)
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.show()


if __name__ == '__main__':
    sentences = [
        '<sta> I very very very love you <pad> <end>',
        'I love you very much <pad> <pad> <pad> <pad>'
    ]
    print(sen2vec(sentences))
