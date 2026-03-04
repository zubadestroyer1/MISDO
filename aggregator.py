"""
MISDO Module 3 — Learnable Parameter-Conditioned Spatial Aggregator
====================================================================
Learned fusion of the 4 agent risk masks, conditioned on user-provided
priority weights.  Uses a small 1×1 conv network to capture non-linear
cross-domain interactions, followed by Gaussian smoothing for contiguous
risk regions and hard boolean constraints for Slope and Distance-to-River.

Pipeline:
    Agent_Masks  [B, 4, 256, 256]  ─┐
    User_Weights [B, 4]             ─┤→ Learned Fusion + Gaussian Blur → Final_Harm_Mask [B, 1, 256, 256]
    slope        [B, 1, H, W]      ─┤  (hard constraints applied post-blur)
    river_prox   [B, 1, H, W]      ─┘
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Hard-constraint thresholds
SLOPE_THRESHOLD: float = 0.8
RIVER_THRESHOLD: float = 0.05

# Gaussian smoothing parameters
GAUSSIAN_SIGMA: float = 4.0
GAUSSIAN_KERNEL_SIZE: int = 21  # must be odd


def _make_gaussian_kernel(kernel_size: int, sigma: float) -> Tensor:
    """Create a 2D Gaussian kernel for spatial smoothing."""
    ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]


class ConditionedAggregator(nn.Module):
    """Learned fusion of 4 agent risk masks with user-priority weights.

    Instead of a simple weighted sum, this module uses a small 1×1
    convolutional network that takes the 4 risk masks and 4 user-weight
    channels (broadcast to spatial dimensions) as an 8-channel input.
    The network learns non-linear cross-domain interactions:
        - fire × low moisture → exponentially worse
        - high slope + deforestation → landslide risk (multiplicative)
        - correlated risks in adjacent domains

    Architecture (learnable fusion):
        Conv1×1(8→32)  + GELU
        Conv1×1(32→16) + GELU
        Conv1×1(16→1)  + Sigmoid

    ~700 learnable parameters.

    Inputs
    ------
    agent_masks : Tensor [B, 4, 256, 256]
        Per-pixel risk outputs from the 4 decoder heads.
    user_weights : Tensor [B, 4]
        User-defined priority vector (e.g., [0.9, 0.1, 0.5, 0.2]).
    slope : Tensor [B, 1, H, W] or None
        Slope data for hard constraints.
    river_proximity : Tensor [B, 1, H, W] or None
        River/flow proximity for hard constraints.

    Returns
    -------
    final_harm_mask : Tensor [B, 1, 256, 256]
        Combined risk surface in [0, 1], with hard no-go zones forced to 1.0.
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Learned Fusion Network ──
        # 8 input channels: 4 agent masks + 4 user weight channels
        self.fusion = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1, bias=True),
        )

        # Initialise the final layer's bias to 0 and weights small
        # so initial outputs are near sigmoid(0) = 0.5
        nn.init.xavier_uniform_(self.fusion[0].weight, gain=0.5)
        nn.init.xavier_uniform_(self.fusion[2].weight, gain=0.5)
        nn.init.xavier_uniform_(self.fusion[4].weight, gain=0.5)
        nn.init.zeros_(self.fusion[0].bias)
        nn.init.zeros_(self.fusion[2].bias)
        nn.init.zeros_(self.fusion[4].bias)

        # Pre-compute Gaussian kernel (registered as buffer so it moves with .to())
        kernel = _make_gaussian_kernel(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
        self.register_buffer("_gaussian_kernel", kernel)

    def forward(
        self,
        agent_masks: Tensor,
        user_weights: Tensor,
        slope: Tensor | None = None,
        river_proximity: Tensor | None = None,
    ) -> Tensor:
        B, C, H, W = agent_masks.shape
        # agent_masks  Shape: [B, 4, 256, 256]
        # user_weights Shape: [B, 4]

        # --- Broadcast user weights to spatial dimensions ---
        # [B, 4] → [B, 4, H, W]
        w_spatial = user_weights[:, :, None, None].expand(B, C, H, W)

        # --- Learned fusion ---
        # Concatenate: [B, 8, H, W]  (4 masks + 4 weight channels)
        fusion_input = torch.cat([agent_masks, w_spatial], dim=1)

        # Non-linear learned fusion → [B, 1, H, W]
        harm_mask = torch.sigmoid(self.fusion(fusion_input))

        # --- Gaussian smoothing for spatial coherence ---
        pad = GAUSSIAN_KERNEL_SIZE // 2
        harm_mask = F.pad(harm_mask, (pad, pad, pad, pad), mode="reflect")
        harm_mask = F.conv2d(harm_mask, self._gaussian_kernel)  # [B, 1, H, W]
        harm_mask = harm_mask.clamp(0.0, 1.0)

        # --- Boolean hard constraints (deterministic, no grad) ---
        if slope is not None:
            # Ensure slope is [B, 1, H, W]
            if slope.dim() == 3:
                slope = slope.unsqueeze(1)
            no_go_slope = slope > SLOPE_THRESHOLD
            harm_mask = torch.where(no_go_slope, torch.ones_like(harm_mask), harm_mask)

        if river_proximity is not None:
            # Ensure river_proximity is [B, 1, H, W]
            if river_proximity.dim() == 3:
                river_proximity = river_proximity.unsqueeze(1)
            no_go_river = river_proximity < RIVER_THRESHOLD
            harm_mask = torch.where(no_go_river, torch.ones_like(harm_mask), harm_mask)

        return harm_mask


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agg = ConditionedAggregator()
    total_params = sum(p.numel() for p in agg.parameters())
    print(f"ConditionedAggregator parameters: {total_params:,}")

    masks = torch.rand(2, 4, 256, 256)
    weights = torch.tensor([[0.9, 0.1, 0.5, 0.2], [0.3, 0.8, 0.4, 0.6]])

    result = agg(masks, weights)
    print(f"Final_Harm_Mask shape: {result.shape}")  # [2, 1, 256, 256]
    print(f"  min={result.min():.4f}  max={result.max():.4f}")
    print(f"  mean={result.mean():.4f}  std={result.std():.4f}")

    # Test with hard constraints
    slope = torch.rand(2, 1, 256, 256)
    river = torch.rand(2, 1, 256, 256)
    result2 = agg(masks, weights, slope=slope, river_proximity=river)
    print(f"\nWith hard constraints:")
    print(f"  min={result2.min():.4f}  max={result2.max():.4f}")
    no_go_pct = (result2 >= 0.999).float().mean() * 100
    print(f"  no_go_pct={no_go_pct:.1f}%")

    # Test that different weights produce different outputs
    w1 = torch.tensor([[0.9, 0.1, 0.1, 0.1]])
    w2 = torch.tensor([[0.1, 0.1, 0.1, 0.9]])
    single_mask = torch.rand(1, 4, 256, 256)
    r1 = agg(single_mask, w1)
    r2 = agg(single_mask, w2)
    diff = (r1 - r2).abs().mean().item()
    print(f"\n  Weight sensitivity test:")
    print(f"    fire-heavy vs soil-heavy mean diff: {diff:.4f}")
