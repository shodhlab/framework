import math
import torch
import torch.nn as nn

class scaledDotProductAttention(nn.Module):
    def __init__(self, contextLength, embeddingDim, dropout):
        super(scaledDotProductAttention, self).__init__()
        self.contextLength = contextLength
        self.embeddingDim = embeddingDim
        self.dropout = dropout
        self.dropoutLayer = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        k = k.transpose(-2, -1)
        scores = torch.matmul(q, k)
        scores = scores / math.sqrt(self.embeddingDim)
        scores = scores + mask.to(scores.dtype).to(scores.device)
        attention = nn.Softmax(dim=-1)(scores)
        out = torch.matmul(attention, v)
        out = self.dropoutLayer(out)
        return out


class additiveAttention(nn.Module):
    def __init__(self, contextLength, embeddingDim, dropout):
        super(additiveAttention, self).__init__()
        self.contextLength = contextLength
        self.embeddingDim = embeddingDim
        self.dropout = dropout

        self.queryExpand = nn.Linear(embeddingDim, contextLength)
        self.keyExpand = nn.Linear(embeddingDim, contextLength)
        self.dropoutLayer = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        q = self.queryExpand(q)
        k = self.keyExpand(k)
        energy = torch.tanh(q + k)
        energy = energy + mask.to(energy.dtype).to(energy.device)
        attention = nn.Softmax(dim=-1)(energy)
        out = torch.matmul(attention, v)
        out = self.dropoutLayer(out)
        return out
