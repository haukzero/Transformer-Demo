import torch
from torch import nn
from torch.nn import functional as F


# simple router
class Router(nn.Module):
    def __init__(self, d_model, n_expert, top_k):
        super(Router, self).__init__()
        self.fc = nn.Linear(d_model, n_expert)
        self.top_k = top_k

    def forward(self, x):
        """
        :param x: (batch_size, seq_len, d_model)
        :return: router_output: (batch_size, seq_len, n_expert),
                ids: (batch_size, seq_len, top_k)
        """
        # logits: (batch_size, seq_len, n_expert)
        logits = self.fc(x)
        top_k_logits, ids = logits.topk(self.top_k, dim=-1)
        # 创建一个全是负无穷的张量, 然后将 top_k 按位置填充
        mask = torch.full_like(logits, -torch.inf)
        sparse_logits = mask.scatter(-1, ids, top_k_logits)
        # 经过 softmax 后, -inf 全部变成 0, 表示不把这份工作交给对应下标的 expert 做
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, ids


"""
本质上, 我们不希望所有 token 都发送给同一组 "受青睐" 的 expert, 
需要一个良好平衡, 因此，将标准正态噪声添加到来自门控线性层的 logits
"""


# noisy router
class NoisyRouter(nn.Module):
    def __init__(self, d_model, n_expert, top_k):
        super().__init__()
        self.fc = nn.Linear(d_model, n_expert)
        self.noisy_fc = nn.Linear(d_model, n_expert)
        self.top_k = top_k

    def forward(self, x):
        logits = self.fc(x)

        noise_logits = self.noisy_fc(x)
        noise = torch.randn_like(noise_logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, ids = noisy_logits.topk(self.top_k, dim=-1)
        mask = torch.full_like(logits, -torch.inf)
        sparse_logits = mask.scatter(-1, ids, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, ids


if __name__ == '__main__':
    x = torch.randn(2, 8, 32)
    nr = NoisyRouter(32, 8, 4)
    router_output, ids = nr(x)
    print(router_output.shape)
    print(ids.shape)
