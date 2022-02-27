import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from collections import defaultdict
from torch.autograd import Variable


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        # q:[b_size, len_q, d_model] -> [b_size, len_q, head x d_k] -> [b_size, len_q, head, d_k] -> [b_size, head, len_q, d_k]

        bs = q.size(0)
        residual = q

        # q:[bs x head x len_q x d_k]
        # k:[bs x head x len_k x d_k]
        # v:[bs x head x len_k x d_k]
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)

        # 计算attention
        # scores:[bs x head x len_q x len_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # mask:[bs x len_q x len_k]
        if mask is not None:
            # mask:[bs x 1 x len_q x len_k]
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        # attn:[bs x head x len_q x len_k]
        attn = self.dropout1(F.softmax(scores, dim=-1))

        # out:[bs x len_q x d_model]
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        out = self.dropout2(self.proj(out))
        out = self.layer_norm(residual + out)
        return out, attn


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        out = self.dropout1(F.relu(self.linear_1(x)))
        out = self.dropout2(F.relu(self.linear_2(out)))
        out = self.layer_norm(residual + out)
        return out
