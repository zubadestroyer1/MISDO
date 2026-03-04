"""
ForestLossNet — Temporal Forest Loss Model
=============================================
EfficientNet-style encoder with MBConv blocks, pixel-shuffle decoder,
and temporal attention for deforestation prediction.

Input  : [B, T, 5, 256, 256]  or  [B, 5, 256, 256]  (backward compat)
Output : [B, 1, 256, 256]  (deforestation risk mask, sigmoid, [0, 1])
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.temporal import TemporalAttention


class DepthwiseSeparableConv(nn.Module):
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
    """EfficientNet encoder + temporal attention + pixel-shuffle decoder."""

    IN_CHANNELS: int = 5

    def __init__(self) -> None:
        super().__init__()

        # ── Encoder ──
        self.stem = nn.Sequential(
            nn.Conv2d(5, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU(),
        )

        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(48, 96, stride=2),
            MBConvBlock(96),
            MBConvBlock(96),
        )

        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(96, 192, stride=2),
            MBConvBlock(192),
            MBConvBlock(192),
        )

        self.bottleneck = nn.Sequential(MBConvBlock(192))

        # ── Temporal Attention ──
        self.temporal_attn = TemporalAttention(192)

        # ── Decoder ──
        self.up3 = PixelShuffleUp(192, 96)
        self.dec3 = nn.Sequential(
            nn.Conv2d(192, 96, 1, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            MBConvBlock(96),
        )

        self.up2 = PixelShuffleUp(96, 48)
        self.dec2 = nn.Sequential(
            nn.Conv2d(96, 48, 1, bias=False),
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

    def _encode(self, x: Tensor):
        e1 = self.stem(x)
        e2 = self.stage1(e1)
        e3 = self.stage2(e2)
        b = self.bottleneck(e3)
        return b, e2, e1

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
            b_all, e2_all, e1_all = self._encode(x_flat)

            _, Cb, Hb, Wb = b_all.shape
            b_temporal = b_all.view(B, T, Cb, Hb, Wb)
            b_fused = self.temporal_attn(b_temporal)

            e2 = e2_all.view(B, T, *e2_all.shape[1:])[:, -1]
            e1 = e1_all.view(B, T, *e1_all.shape[1:])[:, -1]
        else:
            b_fused, e2, e1 = self._encode(x)

        d3 = self.up3(b_fused)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        d1 = self.up1(d2)
        out = self.head(d1)

        return torch.sigmoid(out)


if __name__ == "__main__":
    model = ForestLossNet()
    x = torch.randn(1, 5, 256, 256)
    y = model(x)
    print(f"Single frame: {y.shape}  [{y.min():.3f}, {y.max():.3f}]")
    x_t = torch.randn(1, 5, 5, 256, 256)
    y_t = model(x_t)
    print(f"Temporal (T=5): {y_t.shape}  [{y_t.min():.3f}, {y_t.max():.3f}]")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
