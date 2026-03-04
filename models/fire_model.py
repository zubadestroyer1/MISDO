"""
FireRiskNet — VIIRS Active Fire Risk Model
============================================
ResNet-18-style encoder with U-Net decoder for fire detection from
VIIRS I-band imagery.

Input  : [B, 6, 256, 256]  (VIIRS I1–I5 + FRP)
Output : [B, 1, 256, 256]  (fire risk mask, sigmoid, [0, 1])

~2.5M parameters — runnable on Mac CPU/MPS.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
    """ResNet-18-style encoder + U-Net decoder for fire risk.

    Architecture:
        Encoder: 6→64 (stride 2) → ResBlock → 128 (stride 2) → ResBlock → 256 (stride 2) → ResBlock
        Decoder: 256→128 (up) + skip → 64 (up) + skip → 1 (up)
    """

    IN_CHANNELS: int = 6

    def __init__(self) -> None:
        super().__init__()

        # ── Encoder ──
        self.enc1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )  # → [B, 64, 128, 128]

        self.enc2 = nn.Sequential(
            ResBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )  # → [B, 128, 64, 64]

        self.enc3 = nn.Sequential(
            ResBlock(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.GELU(),
        )  # → [B, 256, 32, 32]

        self.bottleneck = nn.Sequential(
            ResBlock(256),
            ResBlock(256),
        )

        # ── Decoder ──
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),  # 128+128 skip
            nn.GroupNorm(8, 128),
            nn.GELU(),
            ResBlock(128),
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),  # 64+64 skip
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

    def forward(self, x: Tensor) -> Tensor:
        # Encode
        e1 = self.enc1(x)        # [B, 64, 128, 128]
        e2 = self.enc2(e1)       # [B, 128, 64, 64]
        e3 = self.enc3(e2)       # [B, 256, 32, 32]

        b = self.bottleneck(e3)  # [B, 256, 32, 32]

        # Decode with skip connections
        d3 = self.up3(b)                           # [B, 128, 64, 64]
        d3 = self.dec3(torch.cat([d3, e2], dim=1)) # [B, 128, 64, 64]

        d2 = self.up2(d3)                           # [B, 64, 128, 128]
        d2 = self.dec2(torch.cat([d2, e1], dim=1))  # [B, 64, 128, 128]

        d1 = self.up1(d2)         # [B, 32, 256, 256]
        out = self.head(d1)       # [B, 1, 256, 256]

        return torch.sigmoid(out)


if __name__ == "__main__":
    model = FireRiskNet()
    x = torch.randn(1, 6, 256, 256)
    y = model(x)
    print(f"FireRiskNet: {y.shape}  range [{y.min():.3f}, {y.max():.3f}]")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
