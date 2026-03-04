"""
MISDO Module 2 — Shared Backbone & Agent Decoders
===================================================
ConvNeXt-Tiny encoder (in_channels=20) producing a dense latent feature map,
plus four lightweight U-Net-style decoder heads that output per-pixel risk
masks at full spatial resolution.

Backbone output : [B, 256, 64, 64]
Each head output : [B, 1, 256, 256]
Stacked output  : [B, 4, 256, 256]
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════
# ConvNeXt building blocks
# ═══════════════════════════════════════════════════════════════════════════

class ConvNeXtBlock(nn.Module):
    """Simplified ConvNeXt block: depth-wise conv → LayerNorm → 1×1 → GELU → 1×1."""

    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)  # equivalent to LayerNorm per channel
        self.pw1 = nn.Conv2d(dim, dim * expansion, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * expansion, dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x                         # Shape: [B, C, H, W]
        x = self.dw_conv(x)                  # Shape: [B, C, H, W]
        x = self.norm(x)
        x = self.pw1(x)                      # Shape: [B, C*4, H, W]
        x = self.act(x)
        x = self.pw2(x)                      # Shape: [B, C, H, W]
        return x + residual


class ConvNeXtStage(nn.Module):
    """One downsampling stage: optional strided conv → N ConvNeXt blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 2,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if downsample:
            layers.append(
                nn.Sequential(
                    nn.GroupNorm(1, in_channels),
                    nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
                )
            )
        elif in_channels != out_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        for _ in range(depth):
            layers.append(ConvNeXtBlock(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


# ═══════════════════════════════════════════════════════════════════════════
# Shared ConvNeXt-Tiny Backbone
# ═══════════════════════════════════════════════════════════════════════════

class ConvNeXtBackbone(nn.Module):
    """Lightweight ConvNeXt encoder.

    Input  : [B, 20, 256, 256]
    Output : [B, 256, 64, 64]

    Architecture (4× spatial downsample total):
        Stem  : 20 → 64,  stride 2   → 128×128
        Stage1: 64 → 128, stride 2   → 64×64
        Stage2: 128 → 256, stride 1  → 64×64  (no further downsampling)
    """

    def __init__(self, in_channels: int = 20, base_dim: int = 64) -> None:
        super().__init__()
        # Stem — patchify with stride-2 conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, base_dim),
        )  # [B, 64, 128, 128]

        self.stage1 = ConvNeXtStage(base_dim, base_dim * 2, depth=2, downsample=True)
        # [B, 128, 64, 64]

        self.stage2 = ConvNeXtStage(base_dim * 2, base_dim * 4, depth=2, downsample=False)
        # [B, 256, 64, 64]

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)       # Shape: [B, 64, 128, 128]
        x = self.stage1(x)     # Shape: [B, 128, 64, 64]
        x = self.stage2(x)     # Shape: [B, 256, 64, 64]
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Lightweight U-Net–style Decoder Head
# ═══════════════════════════════════════════════════════════════════════════

class DecoderHead(nn.Module):
    """Maps latent [B, 256, 64, 64] → [B, 1, 256, 256] with Sigmoid.

    Two transposed-conv upsample stages (×2 each = ×4 total).
    """

    def __init__(self, in_channels: int = 256) -> None:
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=2, stride=2),
            nn.GroupNorm(1, 128),
            nn.GELU(),
        )  # [B, 128, 128, 128]

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.GroupNorm(1, 64),
            nn.GELU(),
        )  # [B, 64, 256, 256]

        self.head = nn.Conv2d(64, 1, kernel_size=1)  # [B, 1, 256, 256]

    def forward(self, x: Tensor) -> Tensor:
        x = self.up1(x)            # Shape: [B, 128, 128, 128]
        x = self.up2(x)            # Shape: [B, 64, 256, 256]
        x = self.head(x)           # Shape: [B, 1, 256, 256]
        return torch.sigmoid(x)    # Bounded [0, 1]


# Named aliases for clarity (identical architecture, independent weights)
class HydrologyHead(DecoderHead):
    """Soil erosion / runoff risk decoder."""
    pass


class BiodiversityHead(DecoderHead):
    """Habitat fragmentation risk decoder."""
    pass


class ClimateHead(DecoderHead):
    """Carbon / biomass loss risk decoder."""
    pass


class FireHead(DecoderHead):
    """Wildfire wind-tunnel risk decoder."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Full Perception Module
# ═══════════════════════════════════════════════════════════════════════════

class MISDOPerception(nn.Module):
    """Backbone + 4 decoder heads.

    Input  : [B, 20, 256, 256]
    Output : Agent_Masks [B, 4, 256, 256]
    """

    HEAD_NAMES: List[str] = ["hydrology", "biodiversity", "climate", "fire"]

    def __init__(self, in_channels: int = 20) -> None:
        super().__init__()
        self.backbone = ConvNeXtBackbone(in_channels=in_channels)
        self.heads = nn.ModuleDict(
            {
                "hydrology": HydrologyHead(),
                "biodiversity": BiodiversityHead(),
                "climate": ClimateHead(),
                "fire": FireHead(),
            }
        )

    def forward(self, x: Tensor) -> Tensor:
        latent: Tensor = self.backbone(x)  # Shape: [B, 256, 64, 64]

        masks: list[Tensor] = []
        for name in self.HEAD_NAMES:
            mask = self.heads[name](latent)  # Shape: [B, 1, 256, 256]
            masks.append(mask)

        agent_masks: Tensor = torch.cat(masks, dim=1)  # Shape: [B, 4, 256, 256]
        return agent_masks


# ═══════════════════════════════════════════════════════════════════════════
# Real Perception Module (trained domain-specific sub-models)
# ═══════════════════════════════════════════════════════════════════════════

class RealMISDOPerception(nn.Module):
    """Loads 4 trained domain-specific models and stacks their outputs.

    Each sub-model takes its own domain-specific input and produces
    a [B, 1, 256, 256] risk mask. Stacked output: [B, 4, 256, 256].

    Input  : Dict of domain tensors (fire, forest, hydro, soil)
    Output : Agent_Masks [B, 4, 256, 256]
    """

    HEAD_NAMES: List[str] = ["fire", "forest", "hydro", "soil"]

    def __init__(self, weights_dir: str = "weights") -> None:
        super().__init__()
        import os
        from models.fire_model import FireRiskNet
        from models.forest_model import ForestLossNet
        from models.hydro_model import HydroRiskNet
        from models.soil_model import SoilRiskNet

        self.sub_models = nn.ModuleDict({
            "fire": FireRiskNet(),
            "forest": ForestLossNet(),
            "hydro": HydroRiskNet(),
            "soil": SoilRiskNet(),
        })

        # Load trained weights if available
        for name in self.HEAD_NAMES:
            path = os.path.join(weights_dir, f"{name}_model.pt")
            if os.path.exists(path):
                state = torch.load(path, map_location="cpu", weights_only=True)
                self.sub_models[name].load_state_dict(state)
                print(f"  [RealPerception] Loaded weights: {path}")
            else:
                print(f"  [RealPerception] No weights found at {path}, using random init")

    def forward(self, domain_inputs: dict) -> Tensor:
        """Run each sub-model on its domain input and stack results.

        Parameters
        ----------
        domain_inputs : dict
            Keys: "fire", "forest", "hydro", "soil"
            Values: Tensor [B, C_i, 256, 256] where C_i varies per domain.
        """
        masks: list[Tensor] = []
        for name in self.HEAD_NAMES:
            x = domain_inputs[name]
            mask = self.sub_models[name](x)  # [B, 1, 256, 256]
            masks.append(mask)

        return torch.cat(masks, dim=1)  # [B, 4, 256, 256]


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Legacy test
    model = MISDOPerception(in_channels=20)
    dummy = torch.randn(1, 20, 256, 256)
    out = model(dummy)
    print(f"Agent_Masks shape: {out.shape}")   # Expected: [1, 4, 256, 256]
    print(f"  min={out.min():.4f}  max={out.max():.4f}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {total_params:,}")

