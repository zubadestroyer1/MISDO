"""
MISDO Module 2 — Shared Backbone & Agent Decoders
===================================================
Orchestrates the 4 domain-specific models, optionally fusing their
bottleneck features via CrossDomainFusion before decoding.

Legacy (MISDOPerception):
    ConvNeXt-Tiny encoder (in_channels=20) + 4 decoder heads.
    Input  : [B, 20, 256, 256]
    Output : [B, 4, 256, 256]

Real (RealMISDOPerception):
    4 trained ConvNeXt-V2 + UNet++ domain models with cross-domain
    feature fusion at the bottleneck level.
    Input  : Dict of domain tensors (fire, forest, hydro, soil)
    Output : [B, 4, 256, 256]
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════
# ConvNeXt building blocks (legacy backbone — kept for backward compat)
# ═══════════════════════════════════════════════════════════════════════════

class ConvNeXtBlock(nn.Module):
    """Simplified ConvNeXt block: depth-wise conv → LayerNorm → 1×1 → GELU → 1×1."""

    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pw1 = nn.Conv2d(dim, dim * expansion, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * expansion, dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
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
# Legacy Shared ConvNeXt-Tiny Backbone
# ═══════════════════════════════════════════════════════════════════════════

class ConvNeXtBackbone(nn.Module):
    """Lightweight ConvNeXt encoder (legacy).

    Input  : [B, 20, 256, 256]
    Output : [B, 256, 64, 64]
    """

    def __init__(self, in_channels: int = 20, base_dim: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, base_dim),
        )
        self.stage1 = ConvNeXtStage(base_dim, base_dim * 2, depth=2, downsample=True)
        self.stage2 = ConvNeXtStage(base_dim * 2, base_dim * 4, depth=2, downsample=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Legacy Decoder Head
# ═══════════════════════════════════════════════════════════════════════════

class DecoderHead(nn.Module):
    """Maps latent [B, 256, 64, 64] → [B, 1, 256, 256] with Sigmoid."""

    def __init__(self, in_channels: int = 256) -> None:
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=2, stride=2),
            nn.GroupNorm(1, 128),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.GroupNorm(1, 64),
            nn.GELU(),
        )
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.up1(x)
        x = self.up2(x)
        x = self.head(x)
        return torch.sigmoid(x)


class HydrologyHead(DecoderHead):
    pass

class BiodiversityHead(DecoderHead):
    pass

class ClimateHead(DecoderHead):
    pass

class FireHead(DecoderHead):
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Legacy Full Perception Module
# ═══════════════════════════════════════════════════════════════════════════

class MISDOPerception(nn.Module):
    """Legacy backbone + 4 decoder heads.

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
        latent: Tensor = self.backbone(x)
        masks: list[Tensor] = []
        for name in self.HEAD_NAMES:
            mask = self.heads[name](latent)
            masks.append(mask)
        agent_masks: Tensor = torch.cat(masks, dim=1)
        return agent_masks


# ═══════════════════════════════════════════════════════════════════════════
# Real Perception Module (trained domain-specific sub-models)
# ═══════════════════════════════════════════════════════════════════════════

class RealMISDOPerception(nn.Module):
    """Loads 4 trained ConvNeXt-V2 + UNet++ domain models, fuses their
    bottleneck features via CrossDomainFusion, and stacks decoded outputs.

    Each sub-model's encoder produces a bottleneck feature map.
    CrossDomainFusion exchanges information between all 4 domain
    bottlenecks before each decoder runs, enabling early cross-domain
    interactions (e.g., slope × fire → amplified erosion risk).

    Input  : Dict of domain tensors (fire, forest, hydro, soil)
    Output : Agent_Masks [B, 4, 256, 256]
    """

    HEAD_NAMES: List[str] = ["fire", "forest", "hydro", "soil"]

    def __init__(self, weights_dir: str = "weights", use_fusion: bool = True) -> None:
        super().__init__()
        import os
        from models.fire_model import FireRiskNet
        from models.forest_model import ForestLossNet
        from models.hydro_model import HydroRiskNet
        from models.soil_model import SoilRiskNet
        from models.fusion import CrossDomainFusion

        self.sub_models = nn.ModuleDict({
            "fire": FireRiskNet(),
            "forest": ForestLossNet(),
            "hydro": HydroRiskNet(),
            "soil": SoilRiskNet(),
        })

        # Load trained weights if available (tolerant of shape mismatches
        # from architecture changes, e.g., ConvTranspose2d → Conv2d)
        for name in self.HEAD_NAMES:
            path = os.path.join(weights_dir, f"{name}_model.pt")
            if os.path.exists(path):
                saved_state = torch.load(path, map_location="cpu", weights_only=True)
                model_state = self.sub_models[name].state_dict()
                # Filter out keys with shape mismatches
                compatible = {}
                skipped = []
                for k, v in saved_state.items():
                    if k in model_state and v.shape == model_state[k].shape:
                        compatible[k] = v
                    else:
                        skipped.append(k)
                self.sub_models[name].load_state_dict(compatible, strict=False)
                n_loaded = len(compatible)
                n_total = len(model_state)
                print(f"  [RealPerception] Loaded {n_loaded}/{n_total} params from {path}")
                if skipped:
                    print(f"    (skipped {len(skipped)} incompatible keys, will retrain)")
            else:
                print(f"  [RealPerception] No weights found at {path}, using random init")

        # Cross-domain feature fusion
        self.use_fusion = use_fusion
        if use_fusion:
            domain_channels = {
                name: self.sub_models[name].BOTTLENECK_CHANNELS
                for name in self.HEAD_NAMES
            }
            self.fusion = CrossDomainFusion(domain_channels)
            print(f"  [RealPerception] CrossDomainFusion enabled "
                  f"({sum(p.numel() for p in self.fusion.parameters()):,} params)")

    def forward(self, domain_inputs: dict) -> Tensor:
        """Run each sub-model's encoder, fuse bottlenecks, then decode.

        Parameters
        ----------
        domain_inputs : dict
            Keys: "fire", "forest", "hydro", "soil"
            Values: Tensor [B, C_i, 256, 256] where C_i varies per domain.
        """
        # Phase 1: Encode all domains
        bottlenecks: dict = {}
        all_skips: dict = {}
        for name in self.HEAD_NAMES:
            x = domain_inputs[name]
            model = self.sub_models[name]
            b, skips = model.encode(x)
            bottlenecks[name] = b
            all_skips[name] = skips

        # Phase 2: Cross-domain fusion (enriches bottlenecks)
        if self.use_fusion:
            bottlenecks = self.fusion(bottlenecks)

        # Phase 3: Decode each domain from enriched bottleneck
        masks: list[Tensor] = []
        for name in self.HEAD_NAMES:
            model = self.sub_models[name]
            mask = model.decode(bottlenecks[name], all_skips[name])
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
    print(f"Agent_Masks shape: {out.shape}")
    print(f"  min={out.min():.4f}  max={out.max():.4f}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {total_params:,}")
