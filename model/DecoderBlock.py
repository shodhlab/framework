import torch.nn as nn

from model.FeedForward import FeedForward
from model.MultiHeadAttention import MultiHeadAttention
from model.Normalizations import cRMSNorm


class DecoderBlock(nn.Module):
    def __init__(self, batchSize, contextLength, embeddingDim, numHeads, dropout):
        super(DecoderBlock, self).__init__()

        self.batchSize = batchSize
        self.contextLength = contextLength
        self.embeddingDim = embeddingDim
        self.numHeads = numHeads
        self.dropout = dropout
        # self.precision = precision

        self.MHA = MultiHeadAttention(
            self.batchSize,
            self.contextLength,
            self.embeddingDim,
            self.numHeads,
            self.dropout,
            # self.precision
        )
        self.FF = FeedForward(
            self.batchSize,
            self.contextLength,
            self.embeddingDim,
            self.dropout,
            # self.precision
        )
        self.normalisation = cRMSNorm(self.embeddingDim)

    def forward(self, x):
        h = self.normalisation(x)
        h = self.MHA(h)
        x = x + h
        h = self.normalisation(x)
        h = self.FF(h)
        x = x + h
        return x