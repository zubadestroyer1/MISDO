"""
MISDO Module 3 — Hybrid Deterministic + Learnable Aggregator
===============================================================
Fuses 4 agent risk masks conditioned on user-provided priority weights.

Pipeline:
    1. Deterministic weighted sum — ALWAYS respects user weights
    2. Optional learned cross-domain correction (near-zero init)
    3. Forest masking — non-forest pixels excluded from harm
    4. Gaussian spatial smoothing — contiguous risk regions
    5. Hard boolean constraints — slope and distance-to-river no-go zones
    6. Safety mask — inverted harm mask showing recommended harvest zones

Outputs:
    harm_mask  : [B, 1, H, W]  — high = dangerous to deforest
    safety_mask: [B, 1, H, W]  — high = safe to harvest (1 - harm, forest only)
    recommended: [B, 1, H, W]  — binary mask of top-percentile safest zones
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
GAUSSIAN_SIGMA: float = 3.0
GAUSSIAN_KERNEL_SIZE: int = 17  # must be odd


def _make_gaussian_kernel(kernel_size: int, sigma: float) -> Tensor:
    """Create a 2D Gaussian kernel for spatial smoothing."""
    ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]


class LearnedCorrection(nn.Module):
    """Optional learned cross-domain interaction network.

    Captures non-linear relationships between risk domains that a
    simple weighted sum cannot express, e.g.:
        - fire × low moisture → exponentially worse
        - high slope + deforestation → landslide risk
        - correlated risks in adjacent domains

    Near-zero initialized so outputs are negligible without training.

    Architecture:
        Conv1×1(4→32) + GELU + Conv1×1(32→16) + GELU + Conv1×1(16→1)
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1, bias=True),
        )

        # Scale factor for the correction (learnable, starts at 0)
        self.scale = nn.Parameter(torch.zeros(1))

        # Initialize weights very small
        for m in self.net:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, agent_masks: Tensor) -> Tensor:
        """Produce a correction term from raw agent masks.

        Parameters
        ----------
        agent_masks : Tensor [B, 4, H, W]

        Returns
        -------
        correction : Tensor [B, 1, H, W]
            Near-zero without training, learned interactions with training.
        """
        raw = self.net(agent_masks)  # [B, 1, H, W]
        return torch.tanh(raw) * torch.sigmoid(self.scale)


class ConditionedAggregator(nn.Module):
    """Hybrid deterministic + learnable fusion of 4 agent risk masks.

    The deterministic weighted sum guarantees that changing user weights
    ALWAYS changes the output. The optional learned correction adds
    non-linear cross-domain interactions when trained.

    Parameters
    ----------
    use_learned_correction : bool
        If True, include the learned cross-domain correction pathway.

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

    def __init__(self, use_learned_correction: bool = True) -> None:
        super().__init__()
        self.use_learned_correction = use_learned_correction

        if use_learned_correction:
            self.correction = LearnedCorrection()

        # Pre-compute Gaussian kernel
        kernel = _make_gaussian_kernel(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
        self.register_buffer("_gaussian_kernel", kernel)

    def forward(
        self,
        agent_masks: Tensor,
        user_weights: Tensor,
        slope: Tensor | None = None,
        river_proximity: Tensor | None = None,
        forest_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute the harm mask.

        Parameters
        ----------
        agent_masks : Tensor [B, 4, H, W]
        user_weights : Tensor [B, 4]
        slope : Tensor [B, 1, H, W] or None
        river_proximity : Tensor [B, 1, H, W] or None
        forest_mask : Tensor [B, 1, H, W] or None
            1 = forested, 0 = non-forest.  Non-forest pixels get harm = 0
            (deforesting them is meaningless).

        Returns
        -------
        harm_mask : Tensor [B, 1, H, W]
        """
        B, C, H, W = agent_masks.shape

        # ── Step 1: Deterministic weighted sum ──
        w = user_weights.clamp(min=1e-8)
        w_norm = w / w.sum(dim=1, keepdim=True)  # [B, 4]
        w_spatial = w_norm[:, :, None, None]  # [B, 4, 1, 1]
        harm_mask = (agent_masks * w_spatial).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        # ── Step 2: Optional learned correction ──
        if self.use_learned_correction:
            correction = self.correction(agent_masks)
            harm_mask = (harm_mask + correction).clamp(0.0, 1.0)

        # ── Step 3: Mask out non-forest pixels BEFORE smoothing ──
        # This prevents high-risk non-forest areas from bleeding into
        # adjacent forest zones via the Gaussian kernel.
        if forest_mask is not None:
            if forest_mask.dim() == 3:
                forest_mask = forest_mask.unsqueeze(1)
            harm_mask = harm_mask * forest_mask

        # ── Step 4: Gaussian smoothing for spatial coherence ──
        pad = GAUSSIAN_KERNEL_SIZE // 2
        harm_mask = F.pad(harm_mask, (pad, pad, pad, pad), mode="reflect")
        harm_mask = F.conv2d(harm_mask, self._gaussian_kernel)
        harm_mask = harm_mask.clamp(0.0, 1.0)

        # Re-apply forest mask after smoothing to ensure non-forest stays 0
        if forest_mask is not None:
            harm_mask = harm_mask * forest_mask

        # ── Step 5: Hard constraints (slope + river) ──
        if slope is not None:
            if slope.dim() == 3:
                slope = slope.unsqueeze(1)
            harm_mask = torch.where(slope > SLOPE_THRESHOLD, torch.ones_like(harm_mask), harm_mask)

        if river_proximity is not None:
            if river_proximity.dim() == 3:
                river_proximity = river_proximity.unsqueeze(1)
            harm_mask = torch.where(river_proximity < RIVER_THRESHOLD, torch.ones_like(harm_mask), harm_mask)

        return harm_mask

    def compute_safety_mask(
        self,
        harm_mask: Tensor,
        forest_mask: Tensor | None = None,
        safety_threshold: float = 0.3,
    ) -> dict:
        """Derive safety mask and recommended harvest zones from harm mask.

        Parameters
        ----------
        harm_mask : Tensor [B, 1, H, W]
            Output of forward().
        forest_mask : Tensor [B, 1, H, W] or None
            1 = forested.  Recommendations only apply to forested pixels.
        safety_threshold : float
            Harm values below this are considered safe to harvest.

        Returns
        -------
        dict with:
            'safety_mask': Tensor [B, 1, H, W] — 1.0 = safest, 0.0 = most harmful
            'recommended': Tensor [B, 1, H, W] — binary mask of safe harvest zones
        """
        safety_mask = 1.0 - harm_mask

        # Recommended zones: low harm AND forested
        recommended = (harm_mask < safety_threshold).float()
        if forest_mask is not None:
            if forest_mask.dim() == 3:
                forest_mask = forest_mask.unsqueeze(1)
            recommended = recommended * forest_mask
            safety_mask = safety_mask * forest_mask

        return {
            'safety_mask': safety_mask,
            'recommended': recommended,
        }


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
    print(f"Final_Harm_Mask shape: {result.shape}")
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

    # ── CRITICAL: Weight sensitivity test ──
    w1 = torch.tensor([[0.9, 0.1, 0.1, 0.1]])
    w2 = torch.tensor([[0.1, 0.1, 0.1, 0.9]])
    single_mask = torch.rand(1, 4, 256, 256)
    r1 = agg(single_mask, w1)
    r2 = agg(single_mask, w2)
    diff = (r1 - r2).abs().mean().item()
    print(f"\n  Weight sensitivity test:")
    print(f"    fire-heavy vs soil-heavy mean diff: {diff:.4f}")
    assert diff > 0.01, f"FAIL: weights should produce different outputs (diff={diff})"
    print(f"    ✓ PASSED — weights produce visibly different outputs")
