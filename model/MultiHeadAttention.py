import torch
import torch.nn as nn

from model.Attention import scaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, batchSize, context, embedding, heads, groups, mask=False):
        super(MultiHeadAttention, self).__init__()

        self.batchSize = batchSize
        self.context = context
        self.embedding = embedding
        self.heads = heads
        self.groups = groups
        self.mask = mask
        self.headDim = embedding // heads

        assert self.headDim == float(embedding / heads)

        # number of heads should be divisible by groups
        assert self.heads % self.groups == 0

        if mask:
            self.maskVector = torch.ones(self.context, self.context)
            self.maskVector[self.maskVector.tril(diagonal=0).bool()] = 0
            self.maskVector = self.maskVector.masked_fill(
                self.maskVector == 1, float("-inf")
            )

        self.Q = nn.ModuleList(
            [nn.Linear(self.embedding, self.headDim) for _ in range(self.heads)]
        )
        self.K = nn.ModuleList(
            [nn.Linear(self.embedding, self.headDim) for _ in range(self.groups)]
        )
        self.V = nn.ModuleList(
            [nn.Linear(self.embedding, self.headDim) for _ in range(self.groups)]
        )

    def forward(self, x):
        out = []
        per_group = self.heads // self.groups

        for i in range(self.groups):
            k = self.K[i](x)
            v = self.V[i](x)
            for j in range(per_group):
                q = self.Q[i * per_group + j](x)
                if self.mask:
                    out.append(scaledDotProductAttention(q, k, v, self.maskVector))
                else:
                    out.append(scaledDotProductAttention(q, k, v))
        out = torch.cat(out, dim=2)

        return out
