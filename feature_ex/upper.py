import torch
import torch.nn as nn


class AttentionFeedForwardBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8):
        super().__init__()

        self.norm_attn = nn.LayerNorm(dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.ln = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Linear(512, dim)
        )

    def forward(self, x):

        x_norm = self.norm_attn(x)

        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)

        x = x + attn_out

        x_ln = self.ln(x)

        mlp_out = self.mlp(x_ln)

        x = x + mlp_out

        return x


class LinearTokenization(nn.Module):
    def __init__(self, image_size=256, token_dim=256):
        super().__init__()

        self.proj = nn.Linear(image_size, token_dim)

    def forward(self, x):

        x = x.squeeze(1)
        x = self.proj(x)          

        return x


class UpperFeatureExtractor(nn.Module):
    def __init__(self, dim=256, num_heads=8):
        super().__init__()

        self.tokenization = LinearTokenization(
            image_size=256,
            token_dim=dim
        )

        self.block1 = AttentionFeedForwardBlock(
            dim=dim,
            num_heads=num_heads
        )

        self.block2 = AttentionFeedForwardBlock(
            dim=dim,
            num_heads=num_heads
        )

    def forward(self, x, return_projection=False):
        # Only raw images need tokenization. Later FEBs receive fused tokens.
        if x.ndim == 4:
            x = self.tokenization(x)
        elif x.ndim != 3 or x.shape[-2:] != (256, 256):
            raise ValueError("Expected a raw slice batch or [B,256,256] tokens")
        projection_tokens = self.block1(x)
        output = self.block2(projection_tokens)
        return (output, projection_tokens) if return_projection else output


