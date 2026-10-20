"""A single shared age FiLM module per model, used either pre- or post-FE."""
import torch
from torch import nn


class AgeFiLM(nn.Module):
    def __init__(self, channels, embedding_dim=256):
        super().__init__()
        if embedding_dim % 2:
            raise ValueError("embedding_dim must be even")
        self.channels = channels
        self.register_buffer("frequencies", torch.exp(-torch.arange(embedding_dim // 2).float() * (torch.log(torch.tensor(10000.0)) / (embedding_dim // 2))))
        self.affine = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.GELU(), nn.Linear(embedding_dim, 2 * channels))
        nn.init.zeros_(self.affine[-1].weight)
        nn.init.zeros_(self.affine[-1].bias)

    def forward(self, features, age):
        phase = age.reshape(-1, 1).float() / 100 * self.frequencies
        encoded = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(1)
        gamma, beta = self.affine(encoded).chunk(2, dim=-1)
        shape = (len(age), self.channels) + (1,) * (features.ndim - 2)
        return features * (1 + gamma.reshape(shape)) + beta.reshape(shape)
