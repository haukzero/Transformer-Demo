import utils
import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head, device='cpu', cache_flag=0):
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

        # 两种 kv-cache
        # cache_flag = 1 : 在 decoder self-attention 处, 把前面的 token 对应的 q, k, v 存起来,
        # 只用当前的 token 去做线性映射得到对应 q, k, v 并将其与之前的连起来去做 attention
        # cache_flag = 2： 在与 encoder 做交互的 attention 时, k、v 都不变可以直接存来用
        self.cache_flag = cache_flag
        self.past_q = None
        self.past_k = None
        self.past_v = None

    def _cal_q(self, x_q):
        batch_size = x_q.shape[ 0 ]
        # q: (batch_size, n_head, len_q, d_k)
        q = self.W_q(x_q).reshape(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        return q

    def _cal_kv(self, x_k, x_v):
        batch_size = x_k.shape[ 0 ]
        # k: (batch_size, n_head, len_k, d_k)
        k = self.W_k(x_k).reshape(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        # v: (batch_size, n_head, len_v, d_v)
        v = self.W_v(x_v).reshape(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)
        return k, v

    def forward(self, x_q, x_k, x_v, mask, i=None):
        x, batch_size = x_q.clone(), x_q.shape[ 0 ]
        q, k, v = None, None, None

        # 不使用 kv-cache
        if self.cache_flag == 0:
            q = self._cal_q(x_q)
            k, v = self._cal_kv(x_k, x_v)

        # cache 的第一种情况
        elif self.cache_flag == 1:
            x_q, x_k, x_v = x_q[ :, i, : ], x_k[ :, i, : ], x_v[ :, i, : ]
            q = self._cal_q(x_q)
            k, v = self._cal_kv(x_k, x_v)

            # 第一次初始化 cache
            if self.past_q is None:
                self.past_q, self.past_k, self.past_v = q, k, v
            # 否则把之前的和现在的连起来
            else:
                q = torch.cat((self.past_q, q), dim=-2)
                k = torch.cat((self.past_k, k), dim=-2)
                v = torch.cat((self.past_v, v), dim=-2)

        # cache 的第二种情况
        elif self.cache_flag == 2:
            x_q = x_q[ :, i, : ]
            q = self._cal_q(x_q)
            # 第一次初始化 cache
            if self.past_k is None:
                k, v = self._cal_kv(x_k, x_v)
                self.past_k, self.past_v = k, v
            # 否则直接用之前存起来的
            else:
                k, v = self.past_k, self.past_v

        # mask: (batch_size, len_q, len_k) -> (batch_size, n_head, len_q, len_k)
        mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        # attn: (batch_size, n_head, len_q, d_v)
        attn = utils.attention(q, k, v, mask)
        # attn: (batch_size, len_q, n_head * d_v)
        attn = attn.transpose(1, 2).reshape(batch_size, -1, self.n_head * self.d_v)
        # 当使用第一种 cache 的时候, attn 与 x 的 n_seq 可能不一致, 用零矩阵补齐
        if self.cache_flag == 1:
            zero_shape = x.shape[ 1 ] - attn.shape[ 1 ]
            zero_attn = torch.zeros((attn.shape[ 0 ], zero_shape, attn.shape[ -1 ])).to(x.device)
            attn = torch.cat((attn, zero_attn), dim=-2)

        return self.ln(self.fc(attn) + x)
