import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class ZAxisSelfAttention(nn.Module):
    """
    Z轴向 Self-Attention

    输入:
        [B, Z, H, W]

    输出:
        [B, Z, H, W]
    """

    def __init__(self, attn_dim=64, num_heads=8, chunk_size=256):
        super().__init__()

        self.attn_dim = attn_dim
        self.chunk_size = chunk_size

        # 把每个空间位置上的标量特征映射到高维特征
        self.input_proj = nn.Linear(1, attn_dim)

        # Z轴向多头自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # LayerNorm
        self.norm = nn.LayerNorm(attn_dim)

        # 前馈网络
        self.mlp = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 2),
            nn.GELU(),
            nn.Linear(attn_dim * 2, attn_dim)
        )

        # 投影回单通道
        self.output_proj = nn.Linear(attn_dim, 1)

    def forward(self, x):
        """
        x: [B, Z, H, W]
        """

        B, Z, H, W = x.shape
        x_reshape = x.permute(0, 2, 3, 1)
        x_reshape = x_reshape.reshape(B * H * W, Z, 1)

        chunks = []
        for seq in x_reshape.split(self.chunk_size, dim=0):
            if self.training and torch.is_grad_enabled():
                chunks.append(checkpoint(self._attend, seq, use_reentrant=False))
            else:
                chunks.append(self._attend(seq))
        out = torch.cat(chunks, dim=0).reshape(B, H, W, Z)
        return out.permute(0, 3, 1, 2)

    def _attend(self, sequence):
        feat = self.input_proj(sequence)

        # ------------------------------------------------
        # Z轴向 Self-Attention
        # Query = Key = Value = feat
        # ------------------------------------------------

        attn_out, _ = self.self_attn(
            query=feat,
            key=feat,
            value=feat,
            need_weights=False
        )

        # 残差连接
        feat = feat + attn_out

        # 前馈网络 + 残差连接
        feat = feat + self.mlp(self.norm(feat))

        out = self.output_proj(feat)

        # A chunk of independent feature positions; forward restores the grid.
        return out

