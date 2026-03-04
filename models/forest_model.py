"""
ForestLossNet — Hansen Global Forest Change Risk Model
========================================================
EfficientNet-style encoder with depthwise-separable convolutions and a
pixel-shuffle decoder for forest-loss detection.

Input  : [B, 5, 256, 256]  (treecover, lossyear, gain, red, NIR)
Output : [B, 1, 256, 256]  (deforestation risk mask, sigmoid, [0, 1])

~3.2M parameters — runnable on Mac CPU/MPS.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv: depthwise 3×3 → pointwise 1×1."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                            groups=in_ch, bias=False)
        self.gn1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.gn2 = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(self.gn1(self.dw(x)))
        x = F.gelu(self.gn2(self.pw(x)))
        return x


class MBConvBlock(nn.Module):
    """Mobile-inverted bottleneck block (simplified MBConv).

    expand → depthwise 3×3 → squeeze → residual.
    """

    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        mid = channels * expansion
        self.expand = nn.Conv2d(channels, mid, 1, bias=False)
        self.gn1 = nn.GroupNorm(min(8, mid), mid)
        self.dw = nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False)
        self.gn2 = nn.GroupNorm(min(8, mid), mid)
        self.squeeze = nn.Conv2d(mid, channels, 1, bias=False)
        self.gn3 = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.gelu(self.gn1(self.expand(x)))
        x = F.gelu(self.gn2(self.dw(x)))
        x = self.gn3(self.squeeze(x))
        return F.gelu(x + residual)


class PixelShuffleUp(nn.Module):
    """Upsample 2× via pixel shuffle (sub-pixel convolution)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, 1)
        self.shuffle = nn.PixelShuffle(2)
        self.gn = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.shuffle(x)
        return F.gelu(self.gn(x))


class ForestLossNet(nn.Module):
    """EfficientNet-style encoder + pixel-shuffle decoder.

    Architecture:
        Stem: 5→48 (stride 2) → 128×128
        Stage 1: 48→96 (stride 2, MBConv×2) → 64×64
        Stage 2: 96→192 (stride 2, MBConv×2) → 32×32
        Decoder: PixelShuffle up ×3 with skip connections → 1
    """

    IN_CHANNELS: int = 5

    def __init__(self) -> None:
        super().__init__()

        # ── Encoder ──
        self.stem = nn.Sequential(
            nn.Conv2d(5, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU(),
        )  # → [B, 48, 128, 128]

        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(48, 96, stride=2),
            MBConvBlock(96),
            MBConvBlock(96),
        )  # → [B, 96, 64, 64]

        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(96, 192, stride=2),
            MBConvBlock(192),
            MBConvBlock(192),
        )  # → [B, 192, 32, 32]

        self.bottleneck = nn.Sequential(
            MBConvBlock(192),
        )

        # ── Decoder ──
        self.up3 = PixelShuffleUp(192, 96)
        self.dec3 = nn.Sequential(
            nn.Conv2d(192, 96, 1, bias=False),  # 96 + 96 skip
            nn.GroupNorm(8, 96),
            nn.GELU(),
            MBConvBlock(96),
        )

        self.up2 = PixelShuffleUp(96, 48)
        self.dec2 = nn.Sequential(
            nn.Conv2d(96, 48, 1, bias=False),  # 48 + 48 skip
            nn.GroupNorm(8, 48),
            nn.GELU(),
            MBConvBlock(48),
        )

        self.up1 = PixelShuffleUp(48, 24)
        self.head = nn.Sequential(
            nn.Conv2d(24, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Encode
        e1 = self.stem(x)        # [B, 48, 128, 128]
        e2 = self.stage1(e1)     # [B, 96, 64, 64]
        e3 = self.stage2(e2)     # [B, 192, 32, 32]

        b = self.bottleneck(e3)  # [B, 192, 32, 32]

        # Decode with skips
        d3 = self.up3(b)                            # [B, 96, 64, 64]
        d3 = self.dec3(torch.cat([d3, e2], dim=1))  # [B, 96, 64, 64]

        d2 = self.up2(d3)                            # [B, 48, 128, 128]
        d2 = self.dec2(torch.cat([d2, e1], dim=1))   # [B, 48, 128, 128]

        d1 = self.up1(d2)        # [B, 24, 256, 256]
        out = self.head(d1)      # [B, 1, 256, 256]

        return torch.sigmoid(out)


if __name__ == "__main__":
    model = ForestLossNet()
    x = torch.randn(1, 5, 256, 256)
    y = model(x)
    print(f"ForestLossNet: {y.shape}  range [{y.min():.3f}, {y.max():.3f}]")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
