"""
SoilRiskNet — Temporal Soil Degradation Model
================================================
Dilated-convolution encoder with ASPP stem and temporal max-pooling
for soil degradation risk from deforestation history.

Input  : [B, T, 4, 256, 256]  or  [B, 4, 256, 256]  (backward compat)
Output : [B, 1, 256, 256]  (soil degradation risk, sigmoid, [0, 1])

Uses temporal max-pool instead of attention — soil degradation is
cumulative, so the worst moisture state across time dominates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DilatedBlock(nn.Module):
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
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        branch_ch = 24
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(min(8, branch_ch), branch_ch), nn.GELU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(min(8, branch_ch), branch_ch), nn.GELU(),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(min(8, branch_ch), branch_ch), nn.GELU(),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(branch_ch * 3, out_ch, 1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch), nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.merge(torch.cat([
            self.branch1(x), self.branch2(x), self.branch4(x),
        ], dim=1))


class SoilRiskNet(nn.Module):
    """Dilated-conv encoder + temporal max-pool + compact decoder.

    Supports both temporal [B, T, 4, H, W] and single-frame [B, 4, H, W].
    """

    IN_CHANNELS: int = 4

    def __init__(self) -> None:
        super().__init__()

        self.aspp = MultiScaleDilatedEncoder(4, 64)

        self.enc1 = nn.Sequential(DilatedBlock(64, 1), DilatedBlock(64, 2))
        self.down1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)

        self.proj2 = nn.Conv2d(64, 128, 1, bias=False)
        self.enc2 = nn.Sequential(DilatedBlock(128, 1), DilatedBlock(128, 2))
        self.down2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)

        self.proj3 = nn.Conv2d(128, 256, 1, bias=False)
        self.enc3 = nn.Sequential(DilatedBlock(256, 1), DilatedBlock(256, 4))

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128), nn.GELU(),
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )

    def _encode(self, x: Tensor):
        s = self.aspp(x)
        e1 = self.enc1(s)
        e1_down = self.down1(e1)
        e2 = self.enc2(self.proj2(e1_down))
        e2_down = self.down2(e2)
        e3 = self.enc3(self.proj3(e2_down))
        return e3, e2, e1

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 5:
            # Temporal: [B, T, C, H, W]
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
            e3_all, e2_all, e1_all = self._encode(x_flat)

            # Temporal max-pool (worst-case moisture state dominates)
            e3_t = e3_all.view(B, T, *e3_all.shape[1:])
            e3_fused = e3_t.max(dim=1).values  # [B, C, H, W]

            e2 = e2_all.view(B, T, *e2_all.shape[1:])[:, -1]
            e1 = e1_all.view(B, T, *e1_all.shape[1:])[:, -1]
        else:
            e3_fused, e2, e1 = self._encode(x)

        d3 = self.up3(e3_fused)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        out = self.head(d2)
        return torch.sigmoid(out)


if __name__ == "__main__":
    model = SoilRiskNet()
    x = torch.randn(1, 4, 256, 256)
    y = model(x)
    print(f"Single frame: {y.shape}  [{y.min():.3f}, {y.max():.3f}]")
    x_t = torch.randn(1, 5, 4, 256, 256)
    y_t = model(x_t)
    print(f"Temporal (T=5): {y_t.shape}  [{y_t.min():.3f}, {y_t.max():.3f}]")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
