"""
SoilRiskNet — SMAP Soil Degradation Risk Model
=================================================
Dilated-convolution encoder capturing multi-scale moisture patterns,
with a compact decoder for drought/degradation risk estimation.

Input  : [B, 4, 256, 256]  (moisture, veg_water, temperature, freeze_thaw)
Output : [B, 1, 256, 256]  (soil degradation risk, sigmoid, [0, 1])

~2.1M parameters — runnable on Mac CPU/MPS.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DilatedBlock(nn.Module):
    """Dilated convolution block with residual connection.

    Uses dilation to capture long-range spatial dependencies without
    increasing parameter count (SMAP data has ~9 km resolution with
    broad moisture gradients).
    """

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3,
                               padding=dilation, dilation=dilation, bias=False)
        self.gn1 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3,
                               padding=1, dilation=1, bias=False)
        self.gn2 = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.gelu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return F.gelu(x + residual)


class MultiScaleDilatedEncoder(nn.Module):
    """Encoder with parallel dilated convolutions at rates 1, 2, 4.

    Captures moisture patterns at multiple spatial scales simultaneously,
    inspired by ASPP (Atrous Spatial Pyramid Pooling).
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # Use 24 per branch (divisible by 8 for GroupNorm)
        branch_ch = 24

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(min(8, branch_ch), branch_ch),
            nn.GELU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(min(8, branch_ch), branch_ch),
            nn.GELU(),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(min(8, branch_ch), branch_ch),
            nn.GELU(),
        )

        # Merge (branch_ch * 3 may differ from out_ch due to integer division)
        self.merge = nn.Sequential(
            nn.Conv2d(branch_ch * 3, out_ch, 1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b4 = self.branch4(x)
        return self.merge(torch.cat([b1, b2, b4], dim=1))


class SoilRiskNet(nn.Module):
    """Dilated-conv encoder + compact decoder for soil risk.

    Architecture:
        ASPP stem: 4→64 (multi-scale dilated)
        Enc 1: DilatedBlock×2 → stride 2 → 128×128
        Enc 2: DilatedBlock×2 → stride 2 → 64×64
        Enc 3: DilatedBlock×2 → stride 2 → 32×32
        Decoder: upsample × 3 with skip connections → 1
    """

    IN_CHANNELS: int = 4

    def __init__(self) -> None:
        super().__init__()

        # ── ASPP Stem ──
        self.aspp = MultiScaleDilatedEncoder(4, 64)  # [B, 64, 256, 256]

        # ── Encoder ──
        self.enc1 = nn.Sequential(
            DilatedBlock(64, dilation=1),
            DilatedBlock(64, dilation=2),
        )
        self.down1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # → 128

        self.proj2 = nn.Conv2d(64, 128, 1, bias=False)
        self.enc2 = nn.Sequential(
            DilatedBlock(128, dilation=1),
            DilatedBlock(128, dilation=2),
        )
        self.down2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)  # → 64

        self.proj3 = nn.Conv2d(128, 256, 1, bias=False)
        self.enc3 = nn.Sequential(
            DilatedBlock(256, dilation=1),
            DilatedBlock(256, dilation=4),
        )

        # ── Decoder ──
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),  # 128 + 128 skip
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),  # 64 + 64 skip
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # ASPP stem
        s = self.aspp(x)              # [B, 64, 256, 256]

        # Encode
        e1 = self.enc1(s)             # [B, 64, 256, 256]
        e1_down = self.down1(e1)      # [B, 64, 128, 128]

        e2 = self.enc2(self.proj2(e1_down))   # [B, 128, 128, 128]
        e2_down = self.down2(e2)              # [B, 128, 64, 64]

        e3 = self.enc3(self.proj3(e2_down))   # [B, 256, 64, 64]

        # Decode with skips
        d3 = self.up3(e3)                             # [B, 128, 128, 128]
        d3 = self.dec3(torch.cat([d3, e2], dim=1))    # [B, 128, 128, 128]

        d2 = self.up2(d3)                              # [B, 64, 256, 256]
        d2 = self.dec2(torch.cat([d2, e1], dim=1))     # [B, 64, 256, 256]

        out = self.head(d2)  # [B, 1, 256, 256]
        return torch.sigmoid(out)


if __name__ == "__main__":
    model = SoilRiskNet()
    x = torch.randn(1, 4, 256, 256)
    y = model(x)
    print(f"SoilRiskNet: {y.shape}  range [{y.min():.3f}, {y.max():.3f}]")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
