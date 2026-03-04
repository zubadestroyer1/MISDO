"""
FireRiskNet — Temporal Fire Risk Model
========================================
ResNet-18-style encoder with U-Net decoder and temporal attention
for fire detection from multi-temporal deforestation-edge data.

Input  : [B, T, 6, 256, 256]  or  [B, 6, 256, 256]  (backward compat)
Output : [B, 1, 256, 256]  (fire risk mask, sigmoid, [0, 1])

Temporal attention fuses T encoded feature maps before the decoder,
allowing the model to learn fire risk progression over time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.temporal import TemporalAttention


class ResBlock(nn.Module):
    """Residual block with two 3×3 convolutions and GroupNorm."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.gelu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return F.gelu(x + residual)


class FireRiskNet(nn.Module):
    """ResNet-18-style encoder + temporal attention + U-Net decoder.

    Supports both temporal [B, T, 6, H, W] and single-frame [B, 6, H, W].
    """

    IN_CHANNELS: int = 6

    def __init__(self) -> None:
        super().__init__()

        # ── Encoder (weight-shared across T) ──
        self.enc1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )

        self.enc2 = nn.Sequential(
            ResBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )

        self.enc3 = nn.Sequential(
            ResBlock(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.GELU(),
        )

        self.bottleneck = nn.Sequential(
            ResBlock(256),
            ResBlock(256),
        )

        # ── Temporal Attention ──
        self.temporal_attn = TemporalAttention(256)

        # ── Decoder ──
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            ResBlock(128),
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ResBlock(64),
        )

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )

    def _encode(self, x: Tensor):
        """Run encoder, returns (bottleneck, skip2, skip1)."""
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        return b, e2, e1

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 5:
            # Temporal input: [B, T, C, H, W]
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)

            # Encode all timesteps (weight-shared)
            b_all, e2_all, e1_all = self._encode(x_flat)

            # Reshape for temporal attention
            _, Cb, Hb, Wb = b_all.shape
            b_temporal = b_all.view(B, T, Cb, Hb, Wb)
            b_fused = self.temporal_attn(b_temporal)  # [B, 256, 32, 32]

            # Use last timestep's skip connections for decoder
            e2 = e2_all.view(B, T, *e2_all.shape[1:])[:, -1]
            e1 = e1_all.view(B, T, *e1_all.shape[1:])[:, -1]
        else:
            # Single frame: [B, C, H, W]
            b_fused, e2, e1 = self._encode(x)

        # Decode
        d3 = self.up3(b_fused)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        d1 = self.up1(d2)
        out = self.head(d1)

        return torch.sigmoid(out)


if __name__ == "__main__":
    model = FireRiskNet()
    # Test single frame
    x = torch.randn(1, 6, 256, 256)
    y = model(x)
    print(f"Single frame: {y.shape}  [{y.min():.3f}, {y.max():.3f}]")
    # Test temporal
    x_t = torch.randn(1, 5, 6, 256, 256)
    y_t = model(x_t)
    print(f"Temporal (T=5): {y_t.shape}  [{y_t.min():.3f}, {y_t.max():.3f}]")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
