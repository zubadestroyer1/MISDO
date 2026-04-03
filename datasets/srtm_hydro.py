"""
SRTM Hydro Dataset — Counterfactual Erosion Impact (Synthetic)
================================================================
Generates synthetic terrain data where the target is the INCREASE
in downstream erosion/pollution caused by upstream deforestation.

Input:  [B, 7, H, W]  — 5 terrain features + ndssi_baseline + deforestation mask
Target: [B, 1, H, W]  — erosion impact delta [0, 1]

Channels (7):
    0: elevation (DEM, normalised)
    1: slope (degrees, normalised)
    2: aspect (sin-encoded, [0, 1))
    3: flow_accumulation (log-normalised)
    4: forest_cover (canopy %, deforestation-aware)
    5: ndssi_baseline (baseline water quality, normalised)
    6: deforestation_mask (binary: 1=cleared)
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import Tensor
from torch.utils.data import Dataset


class SRTMHydroDataset(Dataset):
    """Synthetic SRTM/hydro dataset for counterfactual erosion training.

    Generates terrain with deforestation patches and computes
    the INCREASE in downstream erosion caused by the clearing.
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

        # Generate realistic terrain
        elevation = gaussian_filter(rng.randn(H, W), sigma=30) * 500 + 1000
        elevation = elevation.astype(np.float32)
        e_min, e_max = elevation.min(), elevation.max()
        elev_norm = (elevation - e_min) / (e_max - e_min + 1e-8)

        # Terrain derivatives
        dy, dx = np.gradient(elevation)
        slope = np.sqrt(dx ** 2 + dy ** 2)
        slope_norm = np.clip(slope / (slope.max() + 1e-8), 0, 1).astype(np.float32)
        aspect = ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi)).astype(np.float32)

        # Flow accumulation proxy
        flow_acc = gaussian_filter(np.maximum(0, -dy), sigma=10)
        fa_max = flow_acc.max()
        flow_acc = (flow_acc / (fa_max + 1e-8)).astype(np.float32)

        # Forest cover
        forest = np.clip(
            gaussian_filter(rng.randn(H, W), sigma=20) * 0.25 + 0.65,
            0, 1,
        ).astype(np.float32)

        # Deforestation patches
        deforestation_mask = np.zeros((H, W), dtype=np.float32)
        n_patches = rng.randint(1, 4)
        for _ in range(n_patches):
            cx, cy = rng.randint(30, H - 30), rng.randint(30, W - 30)
            rx, ry = rng.randint(5, 20), rng.randint(5, 20)
            yy, xx = np.ogrid[-cx:H - cx, -cy:W - cy]
            mask = (xx ** 2 / (ry ** 2 + 1e-8) + yy ** 2 / (rx ** 2 + 1e-8)) < 1
            deforestation_mask[mask] = 1.0

        cleared_forest = forest.copy()
        cleared_forest[deforestation_mask > 0.5] = 0.0

        # Synthetic NDSSI baseline — mimics pre-clearing water quality.
        # Higher values = more suspended sediment in baseline conditions.
        ndssi_baseline = np.clip(
            gaussian_filter(rng.randn(H, W), sigma=15) * 0.2 + 0.3,
            0, 1,
        ).astype(np.float32)

        # Stack: [7, H, W] — matches RealHydroDataset channel layout
        obs = np.stack([
            elev_norm, slope_norm, aspect, flow_acc,
            cleared_forest, ndssi_baseline, deforestation_mask,
        ], axis=0)

        # ── Counterfactual erosion target ──
        # Curvature
        d2y = np.gradient(dy, axis=0)
        d2x = np.gradient(dx, axis=1)
        curvature = np.clip(-(d2y + d2x), 0, None)
        curv_max = curvature.max()
        curvature_norm = curvature / (curv_max + 1e-8)

        # Erosion BEFORE clearing (land already exposed)
        exposed_before = (forest < 0.3).astype(np.float32)
        erosion_before = exposed_before * slope_norm * (1.0 + curvature_norm)

        # Erosion AFTER clearing
        exposed_after = ((cleared_forest < 0.3) | (deforestation_mask > 0.5)).astype(np.float32)
        erosion_after = exposed_after * slope_norm * (1.0 + curvature_norm)

        # Delta
        erosion_delta = np.clip(erosion_after - erosion_before, 0, 1)

        # Propagate downstream
        pollution = gaussian_filter(
            erosion_delta * (1.0 + flow_acc * 3.0), sigma=5.0,
        )
        p_max = pollution.max()
        if p_max > 1e-8:
            pollution = pollution / p_max

        target = np.clip(pollution, 0, 1)[np.newaxis, :, :]

        return (
            torch.from_numpy(obs.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )
