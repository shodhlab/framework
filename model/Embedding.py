import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, context_length, vocab_size, embedding_len):
        super(InputEmbedding, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_len)
        self.pos_embedding_layer = nn.Embedding(context_length, embedding_len)
        self.context_length = context_length

    def forward(self, token_ids):
        token_embedding = self.embedding_layer(token_ids)
        pos_indices = torch.arange(self.context_length).to(token_ids.device)
        pos_embedding = self.pos_embedding_layer(pos_indices)
        final_embedding = token_embedding + pos_embedding
        return final_embedding
