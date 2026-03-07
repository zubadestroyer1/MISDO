"""
VIIRS Fire Dataset — Counterfactual Fire Impact (Synthetic)
=============================================================
Generates synthetic fire data where the target is the INCREASE in
fire risk in surrounding forest caused by deforestation, not just
existing fire locations.

Input:  [B, 7, H, W]  — 6 fire features + deforestation mask
Target: [B, 1, H, W]  — fire impact delta [0, 1]
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import binary_dilation, gaussian_filter
from torch import Tensor
from torch.utils.data import Dataset


class VIIRSFireDataset(Dataset):
    """Synthetic VIIRS fire dataset for counterfactual impact training.

    Generates landscapes with deforestation events and computes
    how fire risk INCREASES near clearings (dry edges, debris fuel,
    wind exposure).
    """

    def __init__(
        self,
        num_samples: int = 256,
        image_size: int = 256,
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        rng = np.random.RandomState(self.seed + idx)
        H = W = self.image_size

        # Generate forest cover with natural variation
        base_cover = rng.uniform(0.4, 0.9)
        forest = np.clip(
            gaussian_filter(rng.randn(H, W), sigma=20) * 0.3 + base_cover,
            0, 1,
        ).astype(np.float32)

        # Generate deforestation patches
        deforestation_mask = np.zeros((H, W), dtype=np.float32)
        n_patches = rng.randint(1, 5)
        for _ in range(n_patches):
            cx, cy = rng.randint(30, H - 30), rng.randint(30, W - 30)
            rx, ry = rng.randint(5, 25), rng.randint(5, 25)
            yy, xx = np.ogrid[-cx:H - cx, -cy:W - cy]
            mask = (xx ** 2 / (ry ** 2 + 1e-8) + yy ** 2 / (rx ** 2 + 1e-8)) < 1
            deforestation_mask[mask] = 1.0

        # Create cleared landscape
        cleared_forest = forest.copy()
        cleared_forest[deforestation_mask > 0.5] = 0.0

        # Fire-related input channels
        non_forest = cleared_forest < 0.3
        edge_zone = binary_dilation(non_forest, iterations=3).astype(np.float32)
        exposure = edge_zone * (cleared_forest > 0.3).astype(np.float32)
        dryness = 1.0 - cleared_forest
        brightness = rng.uniform(0.2, 0.8) * np.ones((H, W), dtype=np.float32)
        frp_proxy = dryness * rng.uniform(0.1, 0.5)

        # Stack: [7, H, W]
        obs = np.stack([
            cleared_forest, np.zeros_like(forest),  # forest, recent_loss
            exposure, dryness,  # exposure, dryness
            brightness, frp_proxy,  # brightness proxy, FRP proxy
            deforestation_mask,  # deforestation mask
        ], axis=0)

        # ── Counterfactual fire target ──
        # Fire risk increases near clearing edges (dry, wind-exposed forest)
        near_clearing = binary_dilation(
            deforestation_mask > 0.5, iterations=10
        ).astype(np.float32)

        # Impact = proximity to clearing × dryness × slope-like factor
        forest_mask = (cleared_forest > 0.3).astype(np.float32)
        fire_impact = near_clearing * forest_mask * dryness

        # Add stochastic variation
        noise = gaussian_filter(rng.randn(H, W) * 0.1, sigma=5)
        fire_impact = np.clip(fire_impact + noise, 0, None)

        fi_max = fire_impact.max()
        if fi_max > 1e-8:
            fire_impact = fire_impact / fi_max

        fire_impact = gaussian_filter(fire_impact, sigma=2.0)
        target = np.clip(fire_impact, 0, 1)[np.newaxis, :, :]

        return (
            torch.from_numpy(obs.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )
