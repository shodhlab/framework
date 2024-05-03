import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, batchSize, context, embedding):
        super(FeedForward, self).__init__()

        self.batchSize = batchSize
        self.context = context
        self.embedding = embedding

        self.inTranform = nn.Linear(embedding, embedding)
        self.activation = nn.GELU()
        self.outTransform = nn.Linear(embedding, embedding)

    def forward(self, x):
        x = self.inTranform(x)
        x = self.activation(x)
        x = self.outTransform(x)
        return x
