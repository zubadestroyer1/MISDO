"""
Hansen Global Forest Change Dataset
=====================================
Generates synthetic data matching Hansen GFC structure (Landsat-derived).

Channels (5):
    0: treecover2000 (percent canopy cover)     — [0, 1] (maps 0–100%)
    1: lossyear (year of loss, 0=no loss, norm) — [0, 1] (maps 0–23)
    2: gain (binary forest gain 2000–2012)      — {0, 1}
    3: red band composite (Landsat B3/B4 norm)  — [0, 1]
    4: NIR band composite (Landsat B4/B5 norm)  — [0, 1]

Target: Binary forest-loss mask [1, 256, 256] where lossyear > 0 AND
        treecover2000 > threshold.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


def _fractal_forest(S: int, g: torch.Generator, octaves: int = 5) -> Tensor:
    """Generate fractal-like forest cover using multi-octave noise.

    Produces spatially-correlated patterns mimicking real forest/non-forest
    boundaries at ~30 m Landsat resolution.
    """
    result = torch.zeros(S, S)
    for i in range(octaves):
        freq = 2 ** i
        weight = 1.0 / (freq + 1)
        small = torch.rand(
            max(S // (2 ** (octaves - i - 1)), 4),
            max(S // (2 ** (octaves - i - 1)), 4),
            generator=g,
        )
        upsampled = F.interpolate(
            small.unsqueeze(0).unsqueeze(0),
            size=(S, S),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        result += upsampled * weight
    # Normalise to [0, 1]
    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    return result


class HansenGFCDataset(Dataset):
    """Synthetic Hansen Global Forest Change dataset.

    Uses fractal noise to generate realistic forest-cover patterns, then
    simulates deforestation patches as rectangles and irregular clearings.
    """

    NUM_CHANNELS: int = 5

    def __init__(
        self,
        num_samples: int = 64,
        spatial_size: int = 256,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.spatial_size = spatial_size
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        S = self.spatial_size
        g = torch.Generator().manual_seed(self.seed + idx)

        # --- Treecover 2000: fractal forest ---
        treecover = _fractal_forest(S, g)  # [S, S] in [0, 1]

        # --- Loss patches: random rectangular clearings ---
        loss_mask = torch.zeros(S, S)
        lossyear_map = torch.zeros(S, S)
        n_clearings = torch.randint(3, 12, (1,), generator=g).item()

        for _ in range(n_clearings):
            # Random position and size
            rh = torch.randint(10, 40, (1,), generator=g).item()
            rw = torch.randint(10, 40, (1,), generator=g).item()
            ry = torch.randint(0, S - rh, (1,), generator=g).item()
            rx = torch.randint(0, S - rw, (1,), generator=g).item()
            year = torch.randint(1, 24, (1,), generator=g).item()  # 2001–2023

            # Only clear where there IS forest
            patch = treecover[ry:ry+rh, rx:rx+rw]
            forest_here = (patch > 0.3).float()
            loss_mask[ry:ry+rh, rx:rx+rw] = torch.max(
                loss_mask[ry:ry+rh, rx:rx+rw], forest_here
            )
            # Assign year only where forested
            mask_update = (forest_here > 0) & (lossyear_map[ry:ry+rh, rx:rx+rw] == 0)
            lossyear_map[ry:ry+rh, rx:rx+rw][mask_update] = year / 23.0

        # --- Gain: some regrowth in older clearings ---
        gain = torch.zeros(S, S)
        early_loss = (lossyear_map > 0) & (lossyear_map < 0.5)
        regrowth_prob = torch.rand(S, S, generator=g)
        gain[early_loss & (regrowth_prob > 0.7)] = 1.0

        # --- Landsat composites ---
        # Red: higher in cleared areas, lower in forest
        red = (1 - treecover * 0.7) * 0.6 + torch.randn(S, S, generator=g) * 0.05
        red = red.clamp(0, 1)
        # NIR: higher in forest (vegetation), lower in cleared
        nir = treecover * 0.8 + torch.randn(S, S, generator=g) * 0.05
        nir = nir.clamp(0, 1)

        # Apply deforestation to composites
        red = red + loss_mask * 0.2
        nir = nir * (1 - loss_mask * 0.4)

        observation = torch.stack([
            treecover,
            lossyear_map,
            gain,
            red.clamp(0, 1),
            nir.clamp(0, 1),
        ], dim=0)  # [5, S, S]

        # Target: binary loss mask where treecover was > 30%
        target = (loss_mask * (treecover > 0.3).float()).unsqueeze(0)  # [1, S, S]

        return observation, target


if __name__ == "__main__":
    ds = HansenGFCDataset(num_samples=4, seed=0)
    obs, tgt = ds[0]
    print(f"Hansen obs: {obs.shape}  min={obs.min():.3f}  max={obs.max():.3f}")
    print(f"Hansen tgt: {tgt.shape}  loss_pixels={tgt.sum().item():.0f}")
