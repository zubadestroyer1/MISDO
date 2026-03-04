"""
Shared Temporal Attention module for MISDO temporal models.
Used by FireRiskNet, ForestLossNet, and SoilRiskNet to fuse
T encoded feature maps into a single feature map via learned attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class TemporalAttention(nn.Module):
    """Fuses T encoded feature maps into one via learned temporal attention.

    Processes T timesteps, computes attention weights per pixel across time,
    and produces a weighted sum that emphasises the most informative frames.

    Parameters
    ----------
    channels : int
        Number of channels in the feature maps.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # Project to lower-dim for attention computation
        mid = max(channels // 4, 8)
        self.query = nn.Conv2d(channels, mid, 1)
        self.key = nn.Conv2d(channels, mid, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.scale = mid ** -0.5

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
        x_flat = x.view(B * T, C, H, W)

        q = self.query(x_flat).view(B, T, -1, H, W)  # [B, T, mid, H, W]
        k = self.key(x_flat).view(B, T, -1, H, W)
        v = self.value(x_flat).view(B, T, C, H, W)

        # Per-pixel attention across time
        attn = (q * k).sum(dim=2, keepdim=True) * self.scale  # [B, T, 1, H, W]
        attn = torch.softmax(attn, dim=1)

        out = (attn * v).sum(dim=1)  # [B, C, H, W]
        return out
