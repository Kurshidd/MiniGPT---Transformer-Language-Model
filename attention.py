import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attention = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        attention = attention.masked_fill(
            self.mask[:T, :T] == 0,
            float('-inf')
        )

        attention = F.softmax(attention, dim=-1)

        attention = self.dropout(attention)

        out = attention @ v

        out = out.transpose(1, 2).contiguous()

        out = out.view(B, T, C)

        out = self.proj(out)

        return out