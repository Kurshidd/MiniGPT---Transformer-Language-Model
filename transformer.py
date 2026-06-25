import torch.nn as nn

from attention import MultiHeadSelfAttention

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        block_size,
        dropout=0.1
    ):
        super().__init__()

        self.attention = MultiHeadSelfAttention(
            embed_dim,
            num_heads,
            block_size,
            dropout
        )

        self.ln1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):

        x = x + self.attention(self.ln1(x))

        x = x + self.ffn(self.ln2(x))

        return x