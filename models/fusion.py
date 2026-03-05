"""
CrossDomainFusion — Early Feature Fusion Module
==================================================
Lightweight cross-domain attention that exchanges information between
the bottleneck features of all 4 domain encoders before decoding.

This allows the model to learn interactions like:
    - high slope + recent clearing → amplified hydro risk
    - low soil moisture + high fire risk → compounding drought/fire risk
    - dense forest canopy + high flow accumulation → erosion sensitivity

Architecture:
    1. Project each domain's bottleneck to a shared dimension (1×1 conv)
    2. Concatenate all projected features
    3. Apply multi-head 1×1 cross-attention
    4. Produce per-domain residuals added back to each encoder's bottleneck

~40k learnable parameters. Near-zero initialized so initial outputs
match the unfused baseline (existing weights load cleanly).
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossDomainFusion(nn.Module):
    """Cross-domain feature fusion via shared-space attention.

    Parameters
    ----------
    domain_channels : dict
        Mapping from domain name to its bottleneck channel count.
        e.g. {"fire": 768, "forest": 768, "hydro": 768, "soil": 768}
    shared_dim : int
        Shared projection dimension for fusion (default 96).
    """

    def __init__(
        self,
        domain_channels: Dict[str, int],
        shared_dim: int = 96,
    ) -> None:
        super().__init__()
        self.domain_names = list(domain_channels.keys())
        self.shared_dim = shared_dim
        n_domains = len(self.domain_names)

        # ── Project each domain → shared space ──
        self.proj_in = nn.ModuleDict({
            name: nn.Conv2d(ch, shared_dim, 1, bias=False)
            for name, ch in domain_channels.items()
        })

        # ── Cross-domain attention ──
        total_in = shared_dim * n_domains
        self.cross_attn = nn.Sequential(
            nn.Conv2d(total_in, total_in, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(total_in, total_in, 1, bias=False),
            nn.GELU(),
        )

        # ── Attention gate per domain (softmax across domains) ──
        self.gate = nn.Conv2d(total_in, n_domains, 1, bias=True)

        # ── Project back to each domain's channel count ──
        self.proj_out = nn.ModuleDict({
            name: nn.Conv2d(shared_dim, ch, 1, bias=False)
            for name, ch in domain_channels.items()
        })

        # ── Near-zero initialization for residual path ──
        for name in self.domain_names:
            nn.init.zeros_(self.proj_out[name].weight)

    def forward(
        self, bottlenecks: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """Fuse bottleneck features across domains.

        Parameters
        ----------
        bottlenecks : dict
            {domain_name: Tensor [B, C_i, H_i, W_i]}
            Spatial dims may differ across domains; we interpolate to
            the smallest common size.

        Returns
        -------
        enriched : dict
            {domain_name: Tensor [B, C_i, H_i, W_i]}
            Original bottleneck + fusion residual.
        """
        # Determine common spatial size (smallest across domains)
        sizes = {name: (t.shape[2], t.shape[3]) for name, t in bottlenecks.items()}
        min_h = min(s[0] for s in sizes.values())
        min_w = min(s[1] for s in sizes.values())

        # Project each domain to shared space and resize
        projected: List[Tensor] = []
        for name in self.domain_names:
            feat = bottlenecks[name]
            p = self.proj_in[name](feat)
            if p.shape[2] != min_h or p.shape[3] != min_w:
                p = F.interpolate(p, size=(min_h, min_w), mode="bilinear", align_corners=False)
            projected.append(p)

        # Concatenate all domains: [B, shared_dim * N, H, W]
        concat = torch.cat(projected, dim=1)

        # Cross-domain attention
        attended = self.cross_attn(concat)

        # Per-domain gating
        gates = torch.softmax(self.gate(attended), dim=1)

        # Split attended back into per-domain chunks
        chunks = torch.chunk(attended, len(self.domain_names), dim=1)

        # Produce enriched bottlenecks
        enriched: Dict[str, Tensor] = {}
        for i, name in enumerate(self.domain_names):
            gated = chunks[i] * gates[:, i:i+1, :, :]
            residual = self.proj_out[name](gated)

            orig_h, orig_w = sizes[name]
            if residual.shape[2] != orig_h or residual.shape[3] != orig_w:
                residual = F.interpolate(
                    residual, size=(orig_h, orig_w),
                    mode="bilinear", align_corners=False,
                )

            enriched[name] = bottlenecks[name] + residual

        return enriched


if __name__ == "__main__":
    domain_channels = {
        "fire": 768,
        "forest": 768,
        "hydro": 768,
        "soil": 768,
    }
    fusion = CrossDomainFusion(domain_channels)
    params = sum(p.numel() for p in fusion.parameters())
    print(f"CrossDomainFusion parameters: {params:,}")

    bottlenecks = {
        "fire": torch.randn(1, 768, 8, 8),
        "forest": torch.randn(1, 768, 8, 8),
        "hydro": torch.randn(1, 768, 8, 8),
        "soil": torch.randn(1, 768, 8, 8),
    }

    enriched = fusion(bottlenecks)
    for name, t in enriched.items():
        orig = bottlenecks[name]
        diff = (t - orig).abs().mean().item()
        print(f"  {name}: {t.shape}  residual_mean={diff:.6f}")
