"""
Hansen GFC Dataset — Counterfactual Cascade Deforestation (Synthetic)
=======================================================================
Generates synthetic forest data where the target is the CASCADE
deforestation effect — how clearing one area causes additional
forest loss in surrounding areas.

Input:  [B, 6, H, W]  — 5 forest features + deforestation mask
Target: [B, 1, H, W]  — cascade deforestation impact delta [0, 1]
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import binary_dilation, gaussian_filter
from torch import Tensor
from torch.utils.data import Dataset


class HansenGFCDataset(Dataset):
    """Synthetic Hansen GFC dataset for counterfactual cascade training.

    Generates landscapes with deforestation patches and computes
    how much additional forest loss occurs near clearings due to
    edge effects, fragmentation, and access-road pressure.
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

        # Treecover baseline
        treecover = np.clip(
            gaussian_filter(rng.randn(H, W), sigma=25) * 0.2 + 0.7,
            0, 1,
        ).astype(np.float32)

        # Generate deforestation patches
        deforestation_mask = np.zeros((H, W), dtype=np.float32)
        n_patches = rng.randint(1, 6)
        for _ in range(n_patches):
            cx, cy = rng.randint(20, H - 20), rng.randint(20, W - 20)
            rx, ry = rng.randint(8, 30), rng.randint(8, 30)
            yy, xx = np.ogrid[-cx:H - cx, -cy:W - cy]
            mask = (xx ** 2 / (ry ** 2 + 1e-8) + yy ** 2 / (rx ** 2 + 1e-8)) < 1
            deforestation_mask[mask] = 1.0

        # Post-clearing forest
        cleared = treecover.copy()
        cleared[deforestation_mask > 0.5] = 0.0

        gain = (rng.rand(H, W) < 0.05).astype(np.float32)
        ndvi_proxy = cleared * 0.8 + 0.2
        canopy_change = np.zeros_like(treecover)

        # Stack: [6, H, W]
        obs = np.stack([
            cleared, np.zeros_like(treecover), gain,
            ndvi_proxy, canopy_change, deforestation_mask,
        ], axis=0)

        # ── Counterfactual cascade target ──
        # Cascade deforestation: forest near clearings is more likely to be lost
        # due to edge drying, fragmentation, access roads
        forest_mask = (cleared > 0.3).astype(np.float32)

        # Impact decays with distance from clearing edge
        import scipy.ndimage as ndi
        distance = ndi.distance_transform_edt(1 - deforestation_mask)
        cascade_prob = np.exp(-distance / 15.0)  # ~15px = ~450m decay

        # Higher cascade probability in thinner forest
        vulnerability = (1.0 - cleared) * forest_mask
        cascade_impact = cascade_prob * forest_mask * (0.3 + 0.7 * vulnerability)

        # Add stochastic variation
        noise = gaussian_filter(rng.randn(H, W) * 0.05, sigma=3)
        cascade_impact = np.clip(cascade_impact + noise, 0, None)

        ci_max = cascade_impact.max()
        if ci_max > 1e-8:
            cascade_impact = cascade_impact / ci_max

        cascade_impact = gaussian_filter(cascade_impact, sigma=2.0)
        target = np.clip(cascade_impact, 0, 1)[np.newaxis, :, :]

        return (
            torch.from_numpy(obs.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )
