"""
Generator G: MLP maps noise z to pool weights w (Softmax), S_probe = w @ P_pool.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """Selector: z -> w (sum to 1), S_probe = sum_i w_i * P_i."""

    def __init__(self, latent_dim: int, pool_size: int, hidden: int = 256, num_layers: int = 3):
        super().__init__()
        self.pool_size = pool_size
        layers = []
        in_d = latent_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_d, hidden), nn.ReLU(inplace=True), nn.LayerNorm(hidden)]
            in_d = hidden
        layers += [nn.Linear(in_d, pool_size)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) -> w: (B, pool_size), sum to 1."""
        logits = self.mlp(z)
        return F.softmax(logits, dim=-1)

    def probe(self, w: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
        """w: (B, pool_size), pool: (pool_size, T) -> (B, T)."""
        return torch.einsum("bp,pt->bt", w, pool)
