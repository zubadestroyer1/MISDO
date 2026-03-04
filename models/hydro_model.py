"""
HydroRiskNet — SRTM/HydroSHEDS Water-Pollution Risk Model
============================================================
Multi-scale Feature Pyramid Network with attention-gated skip
connections for erosion/runoff risk estimation.

Input  : [B, 5, 256, 256]  (elevation, slope, aspect, flow_acc, flow_dir)
Output : [B, 1, 256, 256]  (water-pollution risk, sigmoid, [0, 1])

~3.8M parameters — runnable on Mac CPU/MPS.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AttentionGate(nn.Module):
    """Additive attention gate for skip connections.

    Learns to suppress irrelevant spatial regions in encoder features
    using context from the decoder path.
    """

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int) -> None:
        super().__init__()
        self.w_gate = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.w_skip = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, gate: Tensor, skip: Tensor) -> Tensor:
        g = self.w_gate(gate)
        s = self.w_skip(skip)
        # Ensure spatial dims match (gate may be smaller)
        if g.shape[2:] != s.shape[2:]:
            g = F.interpolate(g, size=s.shape[2:], mode="bilinear", align_corners=False)
        attn = self.psi(F.relu(g + s))
        return skip * attn


class FPNBlock(nn.Module):
    """Feature Pyramid convolution block: conv → norm → GELU × 2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class HydroRiskNet(nn.Module):
    """Multi-scale FPN with attention-gated skip connections.

    Architecture:
        Encoder: 5→64 → 128 → 256 (3 levels, stride 2 each)
        FPN lateral connections at each level
        Decoder: attention-gated skip + upsample × 3 → 1
    """

    IN_CHANNELS: int = 5

    def __init__(self) -> None:
        super().__init__()

        # ── Multi-scale Encoder ──
        self.enc1 = FPNBlock(5, 64)
        self.pool1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)   # 256→128

        self.enc2 = FPNBlock(64, 128)
        self.pool2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)  # 128→64

        self.enc3 = FPNBlock(128, 256)
        self.pool3 = nn.Conv2d(256, 256, 3, stride=2, padding=1)  # 64→32

        self.bottleneck = FPNBlock(256, 256)

        # ── FPN lateral connections ──
        self.lat3 = nn.Conv2d(256, 256, 1)
        self.lat2 = nn.Conv2d(128, 128, 1)
        self.lat1 = nn.Conv2d(64, 64, 1)

        # ── Attention gates ──
        self.attn3 = AttentionGate(gate_ch=256, skip_ch=256, inter_ch=128)
        self.attn2 = AttentionGate(gate_ch=128, skip_ch=128, inter_ch=64)
        self.attn1 = AttentionGate(gate_ch=64, skip_ch=64, inter_ch=32)

        # ── Decoder ──
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = FPNBlock(512, 128)   # 256 + 256 skip

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = FPNBlock(256, 64)    # 128 + 128 skip

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = FPNBlock(128, 32)    # 64 + 64 skip

        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Encode
        e1 = self.enc1(x)            # [B, 64, 256, 256]
        e2 = self.enc2(self.pool1(e1))  # [B, 128, 128, 128]
        e3 = self.enc3(self.pool2(e2))  # [B, 256, 64, 64]

        b = self.bottleneck(self.pool3(e3))  # [B, 256, 32, 32]

        # FPN lateral features
        l3 = self.lat3(e3)  # [B, 256, 64, 64]
        l2 = self.lat2(e2)  # [B, 128, 128, 128]
        l1 = self.lat1(e1)  # [B, 64, 256, 256]

        # Decode with attention-gated skips
        d3 = self.up3(b)                                    # [B, 256, 64, 64]
        a3 = self.attn3(d3, l3)                             # [B, 256, 64, 64]
        d3 = self.dec3(torch.cat([d3, a3], dim=1))          # [B, 128, 64, 64]

        d2 = self.up2(d3)                                    # [B, 128, 128, 128]
        a2 = self.attn2(d2, l2)                              # [B, 128, 128, 128]
        d2 = self.dec2(torch.cat([d2, a2], dim=1))           # [B, 64, 128, 128]

        d1 = self.up1(d2)                                    # [B, 64, 256, 256]
        a1 = self.attn1(d1, l1)                              # [B, 64, 256, 256]
        d1 = self.dec1(torch.cat([d1, a1], dim=1))           # [B, 32, 256, 256]

        out = self.head(d1)  # [B, 1, 256, 256]
        return torch.sigmoid(out)


if __name__ == "__main__":
    model = HydroRiskNet()
    x = torch.randn(1, 5, 256, 256)
    y = model(x)
    print(f"HydroRiskNet: {y.shape}  range [{y.min():.3f}, {y.max():.3f}]")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
