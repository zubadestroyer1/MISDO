"""
SMAP Soil Dataset — Counterfactual Soil Impact (Synthetic)
============================================================
Generates synthetic soil data where the target is the INCREASE
in soil degradation caused by deforestation of nearby areas.

Input:  [B, 5, H, W]  — 4 soil features + deforestation mask
Target: [B, 1, H, W]  — soil degradation impact delta [0, 1]
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import Tensor
from torch.utils.data import Dataset


class SMAPSoilDataset(Dataset):
    """Synthetic SMAP soil dataset for counterfactual degradation training.

    Generates landscapes with deforestation patches and computes
    the INCREASE in soil degradation (moisture loss, topsoil erosion)
    in neighbouring areas caused by the clearing.
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

        # Forest cover
        forest = np.clip(
            gaussian_filter(rng.randn(H, W), sigma=20) * 0.3 + 0.65,
            0, 1,
        ).astype(np.float32)

        # Slope (terrain proxy)
        slope = np.clip(
            gaussian_filter(rng.randn(H, W), sigma=15) * 0.3 + 0.2,
            0, 1,
        ).astype(np.float32)

        # Deforestation patches
        deforestation_mask = np.zeros((H, W), dtype=np.float32)
        n_patches = rng.randint(1, 5)
        for _ in range(n_patches):
            cx, cy = rng.randint(25, H - 25), rng.randint(25, W - 25)
            rx, ry = rng.randint(5, 20), rng.randint(5, 20)
            yy, xx = np.ogrid[-cx:H - cx, -cy:W - cy]
            mask = (xx ** 2 / (ry ** 2 + 1e-8) + yy ** 2 / (rx ** 2 + 1e-8)) < 1
            deforestation_mask[mask] = 1.0

        cleared_forest = forest.copy()
        cleared_forest[deforestation_mask > 0.5] = 0.0

        # Soil proxies derived from forest cover
        moisture = cleared_forest * 0.8 + 0.2
        veg_water = cleared_forest * 0.9
        temp = 1.0 - cleared_forest * 0.6

        # Stack: [5, H, W]
        obs = np.stack([
            moisture, veg_water, temp, slope, deforestation_mask,
        ], axis=0)

        # ── Counterfactual soil degradation target ──
        # Soil degradation is highest on steep slopes near clearings
        import scipy.ndimage as ndi
        distance = ndi.distance_transform_edt(1 - deforestation_mask)

        # Impact decays with distance (localised effect ~10px = ~300m)
        proximity = np.exp(-distance / 10.0)

        # Soil impact = proximity × slope × (1 - remaining forest cover)
        forest_mask = (cleared_forest > 0.1).astype(np.float32)
        soil_impact = proximity * slope * (1.0 - cleared_forest) * forest_mask

        # Cleared areas themselves have maximum degradation
        soil_impact[deforestation_mask > 0.5] = slope[deforestation_mask > 0.5]

        # Add stochastic noise
        noise = gaussian_filter(rng.randn(H, W) * 0.05, sigma=3)
        soil_impact = np.clip(soil_impact + noise, 0, None)

        si_max = soil_impact.max()
        if si_max > 1e-8:
            soil_impact = soil_impact / si_max

        soil_impact = gaussian_filter(soil_impact, sigma=2.0)
        target = np.clip(soil_impact, 0, 1)[np.newaxis, :, :]

        return (
            torch.from_numpy(obs.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )
