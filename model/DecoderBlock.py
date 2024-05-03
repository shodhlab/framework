import torch.nn as nn
from model.FeedForward import FeedForward
from model.MultiHeadAttention import MultiHeadAttention


class DecoderBlock(nn.Module):
    def __init__(self, batchSize, context, embedding, heads, groups):
        super(DecoderBlock, self).__init__()

        self.batchSize = batchSize
        self.context = context
        self.embedding = embedding
        self.heads = heads
        self.groups = groups

        self.maskedMHA = MultiHeadAttention(
            self.batchSize, self.context, self.embedding, self.heads, self.groups, True
        )
        self.feedForward = FeedForward(self.batchSize, self.context, self.embedding)
        self.normalisation = nn.LayerNorm(self.embedding)

    def forward(self, x):
        y = self.maskedMHA(x)
        x = self.normalisation(x)
        x = x + y
        y = self.feedForward(x)
        x = self.normalisation(x)
        x = x + y
        return x
