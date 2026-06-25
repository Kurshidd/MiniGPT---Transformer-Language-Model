import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import TransformerBlock


class GPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        block_size,
        num_heads,
        num_layers,
        dropout=0.1
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        self.position_embedding = nn.Embedding(
            block_size,
            embed_dim
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    block_size,
                    dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(embed_dim)

        self.lm_head = nn.Linear(
            embed_dim,
            vocab_size
        )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        token_emb = self.token_embedding(idx)

        pos = torch.arange(
            T,
            device=idx.device
        )

        pos_emb = self.position_embedding(pos)

        x = token_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(B * T, C)

            targets = targets.view(B * T)

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=300,
        temperature=0.8,
        top_k=20
    ):

        self.eval()

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]

            logits = logits / temperature

            probs = torch.softmax(
                logits,
                dim=-1
            )

            values, indices = torch.topk(
                probs,
                top_k
            )

            sampled_index = torch.multinomial(
                values,
                num_samples=1
            )

            next_token = torch.gather(
                indices,
                -1,
                sampled_index
            )

            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

        return idx