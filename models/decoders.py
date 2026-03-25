"""
UNet++ Decoder — Nested Dense Skip Connections
=================================================
Dense decoder with nested skip pathways for high-accuracy dense
prediction.  Compatible with the ConvNeXt V2 backbone's multi-scale
feature maps.

Includes a DilatedContextModule (ASPP-style) at the bottleneck to
capture long-range spatial impact propagation (1–5 km at 30 m res).

Output uses sigmoid activation for smooth gradients on impact deltas.

Reference:
    Zhou et al., "UNet++: A Nested U-Net Architecture for Medical
    Image Segmentation", DLMIA 2018.

The decoder receives 4 multi-scale encoder features:
    s1: [B, 96,  H/4,  W/4]
    s2: [B, 192, H/8,  W/8]
    s3: [B, 384, H/16, W/16]
    s4: [B, 768, H/32, W/32]   ← bottleneck

And produces a full-resolution output [B, 1, H, W] via nested
dense skip connections with deep supervision.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

class _BilinearUp2x(nn.Module):
    """Bilinear 2× upsampling — drop-in replacement for ConvTranspose2d stride 2.

    Avoids the checkerboard artifacts that ConvTranspose2d can produce.
    """
    def forward(self, x: Tensor) -> Tensor:
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


# ═══════════════════════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════════════════════

class ConvBnGelu(nn.Module):
    """Conv2d → GroupNorm → GELU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.norm = nn.GroupNorm(
            num_groups=min(32, out_ch), num_channels=out_ch
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class DoubleConv(nn.Module):
    """Two consecutive ConvBnGelu blocks — the standard U-Net conv unit."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBnGelu(in_ch, out_ch),
            ConvBnGelu(out_ch, out_ch),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample (×2) via bilinear interpolation + 1×1 channel projection."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: Tensor, target_size: Tuple[int, int]) -> Tensor:
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.proj(x)


# ═══════════════════════════════════════════════════════════════════════════
# Dilated Context Module (ASPP-style)
# ═══════════════════════════════════════════════════════════════════════════

class DilatedContextModule(nn.Module):
    """Multi-rate dilated convolutions for long-range spatial context.

    Captures impact propagation at multiple spatial scales (1–5 km at
    30 m resolution).  Applied at the bottleneck (H/32) where each
    pixel represents ~1 km.

    Inspired by ASPP (Atrous Spatial Pyramid Pooling) from DeepLab.

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    rates : list of int
        Dilation rates for each parallel branch (default [1, 3, 6, 12]).
    """

    def __init__(
        self,
        channels: int,
        rates: list[int] | None = None,
    ) -> None:
        super().__init__()
        if rates is None:
            rates = [1, 3, 6, 12]

        mid = channels // 4

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    channels, mid, kernel_size=3,
                    padding=r, dilation=r, bias=False,
                ),
                nn.GroupNorm(min(16, mid), mid),
                nn.GELU(),
            )
            for r in rates
        ])

        # Global average pooling branch for image-level context
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.GELU(),
        )

        # Fuse all branches back to original channel count
        self.fuse = nn.Sequential(
            nn.Conv2d(mid * (len(rates) + 1), channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, channels), channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        branch_outs = [branch(x) for branch in self.branches]

        # Global context — broadcast to spatial dims
        global_ctx = self.global_branch(x)
        global_ctx = F.interpolate(
            global_ctx, size=x.shape[2:], mode="bilinear", align_corners=False,
        )
        branch_outs.append(global_ctx)

        fused = self.fuse(torch.cat(branch_outs, dim=1))
        return x + fused  # Residual connection


# ═══════════════════════════════════════════════════════════════════════════
# UNet++ Decoder
# ═══════════════════════════════════════════════════════════════════════════

class UNetPPDecoder(nn.Module):
    """UNet++ (nested U-Net) decoder with dense skip connections.

    Notation: X_{i,j} where i = depth (0=shallowest) and j = dense block index.

    The encoder provides X_{0,0}, X_{1,0}, X_{2,0}, X_{3,0} (= s1, s2, s3, s4).

    The nested paths compute:
        X_{2,1} from X_{3,0}↑ + X_{2,0}
        X_{1,1} from X_{2,0}↑ + X_{1,0}
        X_{1,2} from X_{2,1}↑ + X_{1,0} + X_{1,1}
        X_{0,1} from X_{1,0}↑ + X_{0,0}
        X_{0,2} from X_{1,1}↑ + X_{0,0} + X_{0,1}
        X_{0,3} from X_{1,2}↑ + X_{0,0} + X_{0,1} + X_{0,2}

    Parameters
    ----------
    encoder_dims : tuple
        Channel counts from the encoder stages (default ConvNeXt-V2 Base).
    decoder_dim : int
        Uniform channel count for all decoder nodes.
    deep_supervision : bool
        If True, output intermediate predictions for auxiliary loss.
    """

    def __init__(
        self,
        encoder_dims: Tuple[int, ...] = (96, 192, 384, 768),
        decoder_dim: int = 128,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision
        d = decoder_dim
        e = encoder_dims  # (96, 192, 384, 768)

        # ── Dilated context at bottleneck (long-range impact) ──
        self.context = DilatedContextModule(e[3])

        # ── Row 2 (depth=2): one node ──
        # X_{2,1}: upsample X_{3,0} + skip X_{2,0}
        self.up_30 = UpBlock(e[3], d)
        self.conv_21 = DoubleConv(d + e[2], d)   # concat(up(X30), X20)

        # ── Row 1 (depth=1): two nodes ──
        # X_{1,1}: upsample X_{2,0} + skip X_{1,0}
        self.up_20 = UpBlock(e[2], d)
        self.conv_11 = DoubleConv(d + e[1], d)   # concat(up(X20), X10)

        # X_{1,2}: upsample X_{2,1} + skip X_{1,0} + X_{1,1}
        self.up_21 = UpBlock(d, d)
        self.conv_12 = DoubleConv(d + e[1] + d, d)   # concat(up(X21), X10, X11)

        # ── Row 0 (depth=0, shallowest): three nodes ──
        # X_{0,1}: upsample X_{1,0} + skip X_{0,0}
        self.up_10 = UpBlock(e[1], d)
        self.conv_01 = DoubleConv(d + e[0], d)   # concat(up(X10), X00)

        # X_{0,2}: upsample X_{1,1} + skip X_{0,0} + X_{0,1}
        self.up_11 = UpBlock(d, d)
        self.conv_02 = DoubleConv(d + e[0] + d, d)   # concat(up(X11), X00, X01)

        # X_{0,3}: upsample X_{1,2} + skip X_{0,0} + X_{0,1} + X_{0,2}
        self.up_12 = UpBlock(d, d)
        self.conv_03 = DoubleConv(d + e[0] + d + d, d)   # concat(up(X12), X00, X01, X02)

        # ── Final upsample to full resolution (H/4 → H) ──
        # Uses bilinear interpolation + Conv2d instead of ConvTranspose2d
        # to prevent checkerboard artifacts at higher resolutions.
        self.final_up = nn.Sequential(
            _BilinearUp2x(),
            nn.Conv2d(d, d // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(16, d // 2), d // 2),
            nn.GELU(),
            _BilinearUp2x(),
            nn.Conv2d(d // 2, d // 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, d // 4), d // 4),
            nn.GELU(),
        )

        # ── Output heads ──
        self.head = nn.Conv2d(d // 4, 1, kernel_size=1)

        # Deep supervision heads (predict at X_{0,1}, X_{0,2}, X_{1,2})
        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Sequential(
                    _BilinearUp2x(),
                    nn.Conv2d(d, d // 2, 3, padding=1, bias=False),
                    nn.GroupNorm(min(16, d // 2), d // 2),
                    nn.GELU(),
                    _BilinearUp2x(),
                    nn.Conv2d(d // 2, 1, 1),
                )
                for _ in range(3)
            ])

    def forward(
        self,
        features: Dict[str, Tensor],
        return_deep: bool = False,
    ) -> Tensor | Tuple[Tensor, List[Tensor]]:
        """Decode multi-scale features into a full-resolution risk mask.

        Parameters
        ----------
        features : dict
            {"s1": [B,96,H/4,W/4], "s2": ..., "s3": ..., "s4": ...}
        return_deep : bool
            If True AND deep_supervision enabled, return auxiliary outputs.

        Returns
        -------
        output : Tensor [B, 1, H, W]
            Risk mask with sigmoid activation.
        deep_outputs : list of Tensor (only if return_deep=True)
            Auxiliary predictions from intermediate nodes.
        """
        x00 = features["s1"]  # [B, 96,  H/4,  W/4]
        x10 = features["s2"]  # [B, 192, H/8,  W/8]
        x20 = features["s3"]  # [B, 384, H/16, W/16]
        x30 = features["s4"]  # [B, 768, H/32, W/32]

        # Apply dilated context for long-range impact propagation
        x30 = self.context(x30)

        # ── Row 2 ──
        x21 = self.conv_21(torch.cat([
            self.up_30(x30, x20.shape[2:]),
            x20,
        ], dim=1))

        # ── Row 1 ──
        x11 = self.conv_11(torch.cat([
            self.up_20(x20, x10.shape[2:]),
            x10,
        ], dim=1))

        x12 = self.conv_12(torch.cat([
            self.up_21(x21, x10.shape[2:]),
            x10, x11,
        ], dim=1))

        # ── Row 0 ──
        x01 = self.conv_01(torch.cat([
            self.up_10(x10, x00.shape[2:]),
            x00,
        ], dim=1))

        x02 = self.conv_02(torch.cat([
            self.up_11(x11, x00.shape[2:]),
            x00, x01,
        ], dim=1))

        x03 = self.conv_03(torch.cat([
            self.up_12(x12, x00.shape[2:]),
            x00, x01, x02,
        ], dim=1))

        # ── Final upsample (H/4 → H) and prediction ──
        out = self.final_up(x03)
        # Sigmoid activation for output — provides smooth gradients
        # everywhere, unlike ReLU+clamp which kills gradients for ~50%
        # of outputs when inputs are near-zero (common in Siamese delta).
        out = torch.sigmoid(self.head(out))

        if return_deep and self.deep_supervision:
            deep = [
                torch.sigmoid(self.ds_heads[0](x01)),
                torch.sigmoid(self.ds_heads[1](x02)),
                torch.sigmoid(self.ds_heads[2](x12)),
            ]
            return out, deep

        return out


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    decoder = UNetPPDecoder(
        encoder_dims=(96, 192, 384, 768),
        decoder_dim=128,
        deep_supervision=True,
    )
    params = sum(p.numel() for p in decoder.parameters())
    print(f"UNet++ decoder params: {params:,}")

    # Simulate encoder features for 256×256 input
    features = {
        "s1": torch.randn(1, 96, 64, 64),     # H/4
        "s2": torch.randn(1, 192, 32, 32),     # H/8
        "s3": torch.randn(1, 384, 16, 16),     # H/16
        "s4": torch.randn(1, 768, 8, 8),       # H/32
    }

    out = decoder(features)
    print(f"Output: {out.shape}  range [{out.min():.3f}, {out.max():.3f}]")

    out, deep = decoder(features, return_deep=True)
    print(f"Output: {out.shape}")
    for i, d in enumerate(deep):
        print(f"  Deep {i}: {d.shape}")
