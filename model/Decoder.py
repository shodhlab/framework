import torch.nn as nn
from model.DecoderBlock import DecoderBlock


class Decoder(nn.Module):
    def __init__(self, batchSize, context, embedding, heads, layers, groups):
        super(Decoder, self).__init__()

        self.batchSize = batchSize
        self.context = context
        self.embedding = embedding
        self.heads = heads
        self.layers = layers
        self.groups = groups

        self.decoderBlocks = nn.ModuleList(
            [
                DecoderBlock(
                    self.batchSize,
                    self.context,
                    self.embedding,
                    self.heads,
                    self.groups,
                )
                for _ in range(self.layers)
            ]
        )

    def forward(self, x):
        for i in range(self.layers):
            x = self.decoderBlocks[i](x)
        return x
