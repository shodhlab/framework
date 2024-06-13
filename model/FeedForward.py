import torch.nn as nn
from model.Activations import SwiGLU


class FeedForward(nn.Module):

    def __init__(self, batchSize, contextLength, embeddingDim, dropout):
        super(FeedForward, self).__init__()

        self.batchSize = batchSize
        self.contextLength = contextLength
        self.embeddingDim = embeddingDim
        self.dropout = dropout

        self.dropoutLayer = nn.Dropout(self.dropout)
        self.inTranform = nn.Linear(embeddingDim, embeddingDim)
        self.activation = SwiGLU()
        self.outTransform = nn.Linear(embeddingDim, embeddingDim)

    def forward(self, x):
        x = self.inTranform(x)
        x = self.dropoutLayer(x)
        x = self.activation(x)
        x = self.dropoutLayer(x)
        x = self.outTransform(x)
        return x
