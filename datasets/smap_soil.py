"""
SMAP Soil Dataset — Counterfactual Soil Impact (Synthetic)
============================================================
Generates synthetic soil data where the target is the INCREASE
in soil degradation caused by deforestation of nearby areas.

Input:  [B, 7, H, W]  — forest + smap + terrain + deforestation mask
Target: [B, 1, H, W]  — soil degradation impact delta [0, 1]

Channels (7):
    0: forest_cover (canopy %, deforestation-aware)
    1: smap_soil_moisture (baseline moisture, normalised)
    2: slope (SRTM, normalised degrees)
    3: elevation (SRTM DEM, normalised)
    4: aspect (sin-encoded, [0, 1))
    5: flow_accumulation (log-normalised)
    6: deforestation_mask (binary: 1=cleared)
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

        # Terrain — generate elevation first, derive slope/aspect/flow_acc
        elevation_raw = gaussian_filter(rng.randn(H, W), sigma=25) * 500 + 1000
        elevation_raw = elevation_raw.astype(np.float32)
        e_min, e_max = elevation_raw.min(), elevation_raw.max()
        elevation = ((elevation_raw - e_min) / (e_max - e_min + 1e-8)).astype(np.float32)

        dy, dx = np.gradient(elevation_raw)
        slope_raw = np.sqrt(dx ** 2 + dy ** 2)
        slope = np.clip(slope_raw / (slope_raw.max() + 1e-8), 0, 1).astype(np.float32)
        aspect = ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi)).astype(np.float32)

        flow_acc_raw = gaussian_filter(np.maximum(0, -dy), sigma=10)
        fa_max = flow_acc_raw.max()
        flow_acc = (flow_acc_raw / (fa_max + 1e-8)).astype(np.float32)

        # Synthetic SMAP soil moisture — mimics baseline moisture conditions
        smap_norm = np.clip(
            gaussian_filter(rng.randn(H, W), sigma=15) * 0.2 + 0.5,
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

        # Stack: [7, H, W] — matches RealSoilDataset channel layout
        obs = np.stack([
            cleared_forest, smap_norm, slope, elevation,
            aspect, flow_acc, deforestation_mask,
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
