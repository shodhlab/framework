import math
import torch
import torch.nn as nn


def scaledDotProductAttention(Q, K, V, mask=None):
    K = K.transpose(-2, -1)
    scores = torch.matmul(Q, K) / math.sqrt(Q.size(-1))
    if mask is not None:
        scores = scores + mask.to(K.device)
    attention = nn.Softmax(dim=1)(scores).bfloat16()
    out = torch.matmul(attention, V)
    return out
