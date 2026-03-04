"""
SMAP L3 Soil Moisture Dataset
===============================
Generates synthetic data matching SMAP radiometer-derived products.

Channels (4):
    0: surface_soil_moisture (m³/m³ normalised) — [0, 1]  (maps 0–0.5 m³/m³)
    1: vegetation_water_content (kg/m² norm)    — [0, 1]  (maps 0–10 kg/m²)
    2: soil_temperature (K normalised)          — [0, 1]  (maps 240–330 K)
    3: freeze_thaw (binary flag)                — {0, 1}

Target: Soil degradation risk [1, 256, 256] — continuous [0, 1], derived from
        low moisture + high temperature = drought/degradation risk.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


def _smooth_field(S: int, g: torch.Generator, base_scale: int = 32) -> Tensor:
    """Generate spatially-correlated field via upsampled low-res noise."""
    grid = max(S // base_scale, 2)
    low = torch.rand(1, 1, grid, grid, generator=g)
    high = F.interpolate(low, size=(S, S), mode="bilinear", align_corners=False)
    return high.squeeze()  # [S, S]


class SMAPSoilDataset(Dataset):
    """Synthetic SMAP soil moisture dataset.

    Generates spatially-correlated moisture fields with physically-coupled
    vegetation and temperature layers. Captures the inverse relationship
    between soil moisture and surface temperature.
    """

    NUM_CHANNELS: int = 4

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

        # --- Base moisture field (spatially correlated) ---
        moisture_base = _smooth_field(S, g, base_scale=32)
        # Add fine-scale variability
        fine_noise = torch.rand(S, S, generator=g) * 0.1
        moisture = (moisture_base * 0.8 + fine_noise).clamp(0, 1)

        # --- Dry patches (desert / drought areas) ---
        n_dry = torch.randint(1, 5, (1,), generator=g).item()
        for _ in range(n_dry):
            cx = torch.randint(30, S - 30, (1,), generator=g).item()
            cy = torch.randint(30, S - 30, (1,), generator=g).item()
            sigma = torch.rand(1, generator=g).item() * 30 + 15
            yy, xx = torch.meshgrid(
                torch.arange(S, dtype=torch.float32),
                torch.arange(S, dtype=torch.float32),
                indexing="ij",
            )
            dry_zone = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            moisture = moisture * (1 - dry_zone * 0.7)

        moisture = moisture.clamp(0, 1)

        # --- Vegetation water content: correlated with moisture ---
        veg_water = moisture * 0.7 + _smooth_field(S, g, base_scale=16) * 0.3
        veg_water = veg_water.clamp(0, 1)

        # --- Soil temperature: inversely correlated with moisture ---
        # Wet areas are cooler, dry areas are warmer
        temp_base = 1 - moisture * 0.5 + _smooth_field(S, g, base_scale=48) * 0.2
        temp_noise = torch.randn(S, S, generator=g) * 0.03
        soil_temp = (temp_base + temp_noise).clamp(0, 1)

        # --- Freeze/thaw: cold regions ---
        freeze_thaw = (soil_temp < 0.2).float()

        observation = torch.stack([
            moisture,
            veg_water,
            soil_temp,
            freeze_thaw,
        ], dim=0)  # [4, S, S]

        # --- Target: drought / degradation risk ---
        # High risk = low moisture + high temperature + low vegetation
        drought_risk = (1 - moisture) * 0.4 + soil_temp * 0.35 + (1 - veg_water) * 0.25
        # Normalise
        drought_risk = (drought_risk - drought_risk.min()) / (drought_risk.max() - drought_risk.min() + 1e-8)
        target = drought_risk.unsqueeze(0)  # [1, S, S]

        return observation, target


if __name__ == "__main__":
    ds = SMAPSoilDataset(num_samples=4, seed=0)
    obs, tgt = ds[0]
    print(f"SMAP obs: {obs.shape}  min={obs.min():.3f}  max={obs.max():.3f}")
    print(f"SMAP tgt: {tgt.shape}  mean_risk={tgt.mean():.3f}")
