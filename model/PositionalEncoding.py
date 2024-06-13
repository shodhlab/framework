import math
import torch
import torch.nn as nn


class Learned(nn.Module):
    def __init__(self, contextLength, embeddingDim):
        super(Learned, self).__init__()

        self.contextLength = contextLength
        self.embeddingDim = embeddingDim

        self.pos_embedding_layer = nn.Embedding(self.contextLength, self.embeddingDim)

    def forward(self, x):
        pos_indices = torch.arange(self.contextLength).to(x.device)
        pos_embedding = self.pos_embedding_layer(pos_indices)
        x = x + pos_embedding
        return x


class Cosine(nn.Module):
    def __init__(self, contextLength, embeddingDim):
        super(Cosine, self).__init__()

        self.contextLength = contextLength
        self.embeddingDim = embeddingDim

        self.posVector = torch.zeros(
            self.contextLength, self.embeddingDim, dtype=torch.float32
        )
        for pos in range(self.contextLength):
            for i in range(0, self.embeddingDim, 2):
                self.posVector[pos, i] = math.sin(
                    pos / (10000 ** ((2 * i) / self.embeddingDim))
                )
                self.posVector[pos, i + 1] = math.cos(
                    pos / (10000 ** ((2 * i) / self.embeddingDim))
                )

    def forward(self, x):
        x = x + self.posVector.to(x.device).bfloat16()
        return x


class RoPE(nn.Module):
    def __init__(self, context_length, embedding_dim):
        super(RoPE, self).__init__()
        self.dim = embedding_dim
        self.inv_freq = 1.0 / (
            10000 ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim)
        )
        self.context_length = context_length

    def forward(self, x):
        seq_len = self.context_length
        t = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.device))
        pos_enc = (
            torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
            .to(x.device)
            .bfloat16()
        )
        return x * pos_enc