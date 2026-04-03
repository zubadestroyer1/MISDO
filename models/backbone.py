"""
ConvNeXt-V2 Backbone — Shared Encoder for All Domain Models
==============================================================
Hierarchical feature extractor based on ConvNeXt V2 with Global
Response Normalization (GRN).  Produces multi-scale feature maps
for use with UNet++ or other dense-prediction decoders.

Reference:
    Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets
    with Masked Autoencoders", CVPR 2023.

Architecture (default "Base" config):
    Stem   : in_channels → 96,  stride 4          → H/4
    Stage 1: 96   → 96,   depth 3                 → H/4
    Stage 2: 96   → 192,  stride 2, depth 3       → H/8
    Stage 3: 192  → 384,  stride 2, depth 9       → H/16
    Stage 4: 384  → 768,  stride 2, depth 3       → H/32

Output: dict of multi-scale features {"s1": ..., "s2": ..., "s3": ..., "s4": ...}
        plus the final bottleneck at stage 4.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════
# Global Response Normalization (ConvNeXt V2 key innovation)
# ═══════════════════════════════════════════════════════════════════════════

class GRN(nn.Module):
    """Global Response Normalization layer.

    Adaptively recalibrates feature responses based on their global
    spatial aggregation, replacing Layer Scale from ConvNeXt V1.

    For each channel c:
        gx = ||X_c||_2   (L2 norm across spatial dims)
        nx = gx / mean(gx across channels)
        output = X * nx + bias
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        # Global feature norm per channel: [B, C, 1, 1]
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        # Normalize across channels
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


# ═══════════════════════════════════════════════════════════════════════════
# ConvNeXt V2 Block
# ═══════════════════════════════════════════════════════════════════════════

class ConvNeXtV2Block(nn.Module):
    """ConvNeXt V2 block: DWConv7×7 → LayerNorm → PW1 → GELU → GRN → PW2.

    Parameters
    ----------
    dim : int
        Number of input/output channels.
    expansion : int
        Channel expansion ratio for the inverted bottleneck (default 4).
    drop_path : float
        Stochastic depth drop probability (default 0.0).
    """

    def __init__(self, dim: int, expansion: int = 4, drop_path: float = 0.0) -> None:
        super().__init__()
        mid = dim * expansion

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, mid)
        self.act = nn.GELU()
        self.grn = GRN(mid)
        self.pwconv2 = nn.Linear(mid, dim)

        # Stochastic depth
        self.drop_path_rate = drop_path
        if drop_path > 0.0:
            self.drop_path_mask = True
        else:
            self.drop_path_mask = False

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        # Depthwise conv (spatial mixing)
        x = self.dwconv(x)

        # Channel-last for LayerNorm and linear layers
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)

        # GRN operates on channel-first
        x = x.permute(0, 3, 1, 2)  # [B, mid, H, W]
        x = self.grn(x)

        # Back to channel-last for final projection
        x = x.permute(0, 2, 3, 1)  # [B, H, W, mid]
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Stochastic depth
        if self.drop_path_mask and self.training:
            keep_prob = 1.0 - self.drop_path_rate
            shape = (x.shape[0], 1, 1, 1)
            mask = torch.bernoulli(
                torch.full(shape, keep_prob, device=x.device, dtype=x.dtype)
            )
            x = x / keep_prob * mask

        return residual + x


# ═══════════════════════════════════════════════════════════════════════════
# ConvNeXt V2 Stage (downsample + N blocks)
# ═══════════════════════════════════════════════════════════════════════════

class ConvNeXtV2Stage(nn.Module):
    """One stage: optional downsample → N ConvNeXt V2 blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        downsample: bool = True,
        drop_path_rates: List[float] | None = None,
    ) -> None:
        super().__init__()
        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth

        layers: list[nn.Module] = []

        if downsample:
            layers.append(nn.Sequential(
                _ChannelFirstToLast(),      # [B,C,H,W] → [B,H,W,C]
                nn.LayerNorm(in_channels, eps=1e-6),
                _ChannelLastToFirst(),      # [B,H,W,C] → [B,C,H,W]
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            ))
        elif in_channels != out_channels:
            layers.append(nn.Sequential(
                _ChannelFirstToLast(),
                nn.LayerNorm(in_channels, eps=1e-6),
                _ChannelLastToFirst(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            ))

        for i in range(depth):
            layers.append(ConvNeXtV2Block(out_channels, drop_path=drop_path_rates[i]))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class _ChannelLastToFirst(nn.Module):
    """Permute [B, H, W, C] → [B, C, H, W]."""
    def forward(self, x: Tensor) -> Tensor:
        return x.permute(0, 3, 1, 2) if x.dim() == 4 else x


class _ChannelFirstToLast(nn.Module):
    """Permute [B, C, H, W] → [B, H, W, C]."""
    def forward(self, x: Tensor) -> Tensor:
        return x.permute(0, 2, 3, 1) if x.dim() == 4 else x


# ═══════════════════════════════════════════════════════════════════════════
# Full ConvNeXt V2 Backbone
# ═══════════════════════════════════════════════════════════════════════════

class ConvNeXtV2Backbone(nn.Module):
    """ConvNeXt V2 hierarchical encoder with multi-scale feature output.

    Parameters
    ----------
    in_channels : int
        Number of input channels (varies per domain: 4–6).
    dims : tuple
        Channel dimensions per stage (default: Base config).
    depths : tuple
        Number of blocks per stage.
    drop_path_rate : float
        Maximum stochastic depth drop rate (linearly increases).

    Returns
    -------
    features : dict
        {"s1": [B, 96, H/4, W/4], "s2": [B, 192, H/8, W/8],
         "s3": [B, 384, H/16, W/16], "s4": [B, 768, H/32, W/32]}
    """

    def __init__(
        self,
        in_channels: int = 6,
        dims: Tuple[int, ...] = (96, 192, 384, 768),
        depths: Tuple[int, ...] = (3, 3, 9, 3),
        drop_path_rate: float = 0.1,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.depths = depths

        # Stochastic depth schedule (linearly increasing)
        total_depth = sum(depths)
        dp_rates = [
            drop_path_rate * i / (total_depth - 1)
            for i in range(total_depth)
        ]

        # Stem: patchify with stride-4 conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            _ChannelFirstToLast(),
            nn.LayerNorm(dims[0], eps=1e-6),
            _ChannelLastToFirst(),
        )

        # Build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage_dp = dp_rates[cur:cur + depths[i]]
            stage = ConvNeXtV2Stage(
                in_channels=dims[i - 1] if i > 0 else dims[0],
                out_channels=dims[i],
                depth=depths[i],
                downsample=(i > 0),  # stages 1–3 downsample
                drop_path_rates=stage_dp,
            )
            self.stages.append(stage)
            cur += depths[i]

        # Final layer norm on bottleneck
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

        if pretrained:
            self._load_pretrained(in_channels)
        else:
            self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _load_pretrained(self, in_channels: int) -> None:
        """Load ImageNet-22k pretrained ConvNeXt-V2 Base weights via timm.

        timm handles in_chans adaptation automatically by repeating
        the 3-channel stem weights cyclically for in_chans > 3.
        We match parameters by shape + name suffix to handle the
        different module nesting between timm and our architecture.
        """
        try:
            import timm
        except ImportError:
            print("    \u26a0 timm not installed \u2014 using random initialisation.")
            print("    Install with: pip install timm")
            self._init_weights()
            return

        print(f"    Loading pretrained ConvNeXt-V2 Base weights "
              f"(in_chans={in_channels})...")
        ref = timm.create_model(
            'convnextv2_base.fcmae_ft_in22k_in1k_384',
            pretrained=True,
            in_chans=in_channels,
        )

        # Collect pretrained parameters
        ref_params = {}
        for name, param in ref.named_parameters():
            ref_params[name] = param.data.clone()

        # Match by shape + name suffix to handle module nesting differences
        own_state = self.state_dict()
        loaded = 0
        used_ref_keys: set = set()
        for own_key in sorted(own_state.keys()):
            own_val = own_state[own_key]
            # Try direct key match first
            if own_key in ref_params and ref_params[own_key].shape == own_val.shape:
                own_state[own_key] = ref_params[own_key]
                used_ref_keys.add(own_key)
                loaded += 1
                continue
            # Try suffix match (handles different module nesting)
            parts = own_key.split('.')
            suffix_str = '.'.join(parts[-2:]) if len(parts) >= 2 else own_key
            for ref_key, ref_val in ref_params.items():
                if ref_key in used_ref_keys:
                    continue
                if ref_key.endswith(suffix_str) and ref_val.shape == own_val.shape:
                    own_state[own_key] = ref_val
                    used_ref_keys.add(ref_key)
                    loaded += 1
                    break

        self.load_state_dict(own_state)
        total = len(own_state)
        print(f"    \u2713 Loaded {loaded}/{total} pretrained parameters")
        del ref  # free memory

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.stem(x)  # [B, dims[0], H/4, W/4]

        features: Dict[str, Tensor] = {}
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features[f"s{i + 1}"] = x
            # s1: [B, 96,  H/4,  W/4]
            # s2: [B, 192, H/8,  W/8]
            # s3: [B, 384, H/16, W/16]
            # s4: [B, 768, H/32, W/32]

        # Apply final norm to bottleneck
        b = features["s4"]
        b = b.permute(0, 2, 3, 1)
        b = self.norm(b)
        b = b.permute(0, 3, 1, 2)
        features["s4"] = b

        return features

    @property
    def bottleneck_channels(self) -> int:
        return self.dims[-1]


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    for in_ch, name in [(7, "fire"), (6, "forest"), (6, "hydro"), (6, "soil")]:
        backbone = ConvNeXtV2Backbone(in_channels=in_ch)
        x = torch.randn(1, in_ch, 256, 256)
        features = backbone(x)
        params = sum(p.numel() for p in backbone.parameters())
        print(f"\n{name} (in={in_ch})  params={params:,}")
        for k, v in features.items():
            print(f"  {k}: {v.shape}")
