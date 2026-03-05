"""
Temporal Attention — Multi-Head Self-Attention for Temporal Fusion
====================================================================
Fuses T encoded feature maps into a single feature map using
multi-head self-attention across the temporal dimension.

Used by all domain models to learn temporal risk progression
(e.g., fire spread over days, seasonal deforestation patterns,
wet/dry season hydrology, cumulative soil degradation).

Components:
    TemporalAttention  — Full multi-head self-attention for bottleneck
    TemporalSkipFusion — Lightweight learned temporal weighting for skips
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TemporalAttention(nn.Module):
    """Multi-head self-attention for fusing T encoded feature maps.

    Processes T timesteps with multi-head attention, computes
    attention weights per pixel across time, and produces a weighted
    sum that emphasises the most informative frames.

    Parameters
    ----------
    channels : int
        Number of channels in the feature maps.
    num_heads : int
        Number of attention heads (default 4).
    max_timesteps : int
        Maximum number of timesteps for positional encoding (default 16).
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        max_timesteps: int = 16,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, \
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.scale = self.head_dim ** -0.5

        # Multi-head projections
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # Layer normalisation for stable attention
        self.norm = nn.GroupNorm(1, channels)

        # Learnable temporal positional encoding
        self.pos_enc = nn.Parameter(
            torch.randn(1, max_timesteps, channels, 1, 1) * 0.02
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, T, C, H, W]
            T encoded feature maps.

        Returns
        -------
        out : Tensor [B, C, H, W]
            Single fused feature map.
        """
        B, T, C, H, W = x.shape

        # Add positional encoding
        x = x + self.pos_enc[:, :T, :, :, :]

        # Normalise
        x_flat = x.reshape(B * T, C, H, W)
        x_flat = self.norm(x_flat)

        # Compute Q, K, V
        qkv = self.qkv(x_flat)  # [B*T, 3*C, H, W]
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim, H, W)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # q, k, v: [B, T, num_heads, head_dim, H, W]

        # Per-pixel attention across time (per head)
        # Reshape for dot product: [B, num_heads, H, W, T, head_dim]
        q = q.permute(0, 2, 4, 5, 1, 3)
        k = k.permute(0, 2, 4, 5, 1, 3)
        v = v.permute(0, 2, 4, 5, 1, 3)

        # Attention scores: [B, num_heads, H, W, T, T]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Weighted values: [B, num_heads, H, W, T, head_dim]
        out = torch.matmul(attn, v)

        # Extract last timestep's attended representation
        # The last query has attended to all prior frames via self-attention,
        # producing a richer summary than a flat mean which dilutes weights
        out = out[..., -1, :]  # [B, num_heads, H, W, head_dim]

        # Reshape: [B, C, H, W]
        out = out.permute(0, 1, 4, 2, 3)  # [B, num_heads, head_dim, H, W]
        out = out.reshape(B, C, H, W)

        # Output projection
        out = self.proj_out(out)

        return out


class TemporalSkipFusion(nn.Module):
    """Lightweight temporal fusion for skip connections.

    Instead of discarding all but the last timestep, this module learns
    per-channel temporal weights via a small MLP, then computes a
    weighted sum across timesteps.  Much cheaper than full attention
    but preserves temporal information in skip connections.

    Parameters
    ----------
    channels : int
        Number of channels in the skip features.
    max_timesteps : int
        Maximum number of timesteps supported.
    """

    def __init__(self, channels: int, max_timesteps: int = 16) -> None:
        super().__init__()
        self.channels = channels
        self.max_timesteps = max_timesteps

        # Small MLP: pool spatial → [B, T, C] → temporal weights [B, T, 1]
        self.temporal_mlp = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, 1),
        )

        # Layer norm applied per-timestep
        self.norm = nn.GroupNorm(1, channels)

        # Learnable temporal position bias
        self.pos_bias = nn.Parameter(
            torch.zeros(1, max_timesteps, 1, 1, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, T, C, H, W]
            T skip-connection feature maps.

        Returns
        -------
        out : Tensor [B, C, H, W]
            Temporally fused skip feature map.
        """
        B, T, C, H, W = x.shape

        # Add temporal position bias (recency signal)
        x = x + self.pos_bias[:, :T]

        # Global average pool per timestep: [B, T, C]
        pooled = x.mean(dim=(-2, -1))  # [B, T, C]

        # Compute temporal attention weights: [B, T, 1]
        weights = self.temporal_mlp(pooled)  # [B, T, 1]
        weights = F.softmax(weights, dim=1)  # normalize across T

        # Expand weights for spatial broadcast: [B, T, 1, 1, 1]
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        # Weighted sum across timesteps: [B, C, H, W]
        out = (x * weights).sum(dim=1)

        return out


if __name__ == "__main__":
    attn = TemporalAttention(channels=768, num_heads=4)
    params = sum(p.numel() for p in attn.parameters())
    print(f"TemporalAttention params: {params:,}")

    x = torch.randn(2, 5, 768, 8, 8)  # B=2, T=5
    out = attn(x)
    print(f"Input: {x.shape}  Output: {out.shape}")

    # Single timestep
    x1 = torch.randn(2, 1, 768, 8, 8)
    out1 = attn(x1)
    print(f"Single timestep: {x1.shape} → {out1.shape}")

    # ── TemporalSkipFusion test ──
    print("\n--- TemporalSkipFusion ---")
    for ch, name in [(96, "s1"), (192, "s2"), (384, "s3")]:
        skip_fuser = TemporalSkipFusion(channels=ch)
        sp = sum(p.numel() for p in skip_fuser.parameters())
        x_skip = torch.randn(2, 5, ch, 64, 64)
        out_skip = skip_fuser(x_skip)
        print(f"  {name} (ch={ch}): {sp:,} params  "
              f"{x_skip.shape} → {out_skip.shape}")
