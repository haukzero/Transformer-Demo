import torch
from torch import nn
from .exp import Expert
from .router import NoisyRouter


class MoE(nn.Module):
    def __init__(self, d_model, n_expert, d_exp, top_k, dropout=0.1):
        super(MoE, self).__init__()
        self.top_k = top_k
        self.router = NoisyRouter(d_model, n_expert, top_k)
        self.experts = nn.ModuleList([
            Expert(d_model, d_exp, dropout)
            for _ in range(n_expert)
        ])

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)

        # router_output: (batch_size, seq_len, n_expert)
        # ids: (batch_size, seq_len, top_k)
        router_output, ids = self.router(x)

        output = torch.zeros_like(x)

        # flatten, 即把每个 batch 拼在一起
        # flat_x: (batch_size * seq_len, d_model)
        # flat_router: (batch_size * seq_len, n_expert)
        flat_x = x.view(-1, x.shape[ -1 ])
        flat_router = router_output.view(-1, router_output.shape[ -1 ])

        # 以每个 expert 为单位进行操作, 即把当前 expert 处理的 token 都进行加权
        for i, expert in enumerate(self.experts):
            expert_mask = (ids == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            # 若当前 expert 是任意一个 token 的 top-k
            if flat_mask.any():
                # 选出由当前 expert 参与处理的 token
                # expert_x: (n_token_expert, d_model)
                expert_x = flat_x[ flat_mask ]
                expert_output = expert(expert_x)

                # 计算当前 expert 对于有作用的 token 的权重分数
                # scores: (n_token_expert, 1)
                scores = flat_router[ flat_mask, i ].unsqueeze(-1)
                # weight_output: (n_token_expert, d_model)
                weight_output = expert_output * scores

                # 结果叠加
                output[ expert_mask ] += weight_output

        return output


if __name__ == '__main__':
    batch_size = 4
    seq_len = 16
    d_model = 64
    d_exp = 128
    n_expert = 8
    top_k = 2

    x = torch.randn(batch_size, seq_len, d_model)
    moe = MoE(d_model, n_expert, d_exp, top_k)
    print(moe(x).shape)
