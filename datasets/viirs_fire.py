"""
VIIRS VNP14IMG Active Fire Dataset
===================================
Generates synthetic data matching VIIRS I-band structure.

Channels (6):
    0: I1 (0.64 µm visible reflectance)         — [0, 1]
    1: I2 (0.86 µm NIR reflectance)              — [0, 1]
    2: I3 (1.61 µm SWIR reflectance)             — [0, 1]
    3: I4 (3.74 µm MIR brightness temp, norm)    — [0, 1] (maps ~250–500 K)
    4: I5 (11.45 µm TIR brightness temp, norm)   — [0, 1] (maps ~200–350 K)
    5: FRP (fire radiative power, norm)           — [0, 1]

Target: Binary fire mask [1, 256, 256] derived from I4 brightness temperature
        exceedance over contextual background.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

# Physical constants (normalized to [0, 1])
_BG_I4: float = 0.3    # background MIR BT (~310 K normalised)
_BG_I5: float = 0.45   # background TIR BT (~280 K normalised)
_FIRE_I4: float = 0.85  # fire MIR BT (~450 K normalised)
_FIRE_I5: float = 0.65  # fire TIR BT (~320 K normalised)


class VIIRSFireDataset(Dataset):
    """Synthetic VIIRS active-fire dataset with realistic spatial hotspots.

    Fire locations are generated as Gaussian clusters to mimic real wildfire
    point-spread patterns at 375 m resolution.
    """

    NUM_CHANNELS: int = 6

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

        # --- Background layers ---
        # Visible / NIR / SWIR — spatially correlated vegetation reflectance
        base_veg = torch.rand(1, S, S, generator=g) * 0.4 + 0.2  # [0.2, 0.6]
        noise = torch.randn(3, S, S, generator=g) * 0.05
        i1 = (base_veg * 0.5 + noise[0:1]).clamp(0, 1)   # visible — lower
        i2 = (base_veg * 0.9 + noise[1:2]).clamp(0, 1)    # NIR — higher for veg
        i3 = (base_veg * 0.3 + noise[2:3]).clamp(0, 1)    # SWIR — moderate

        # Brightness temperatures — background
        i4 = torch.full((1, S, S), _BG_I4) + torch.randn(1, S, S, generator=g) * 0.03
        i5 = torch.full((1, S, S), _BG_I5) + torch.randn(1, S, S, generator=g) * 0.03

        # FRP — background near zero
        frp = torch.zeros(1, S, S)

        # --- Generate fire clusters ---
        fire_mask = torch.zeros(1, S, S)
        n_clusters = torch.randint(2, 8, (1,), generator=g).item()

        for _ in range(n_clusters):
            cx = torch.randint(20, S - 20, (1,), generator=g).item()
            cy = torch.randint(20, S - 20, (1,), generator=g).item()
            sigma = torch.rand(1, generator=g).item() * 8 + 3  # 3–11 px spread

            yy, xx = torch.meshgrid(
                torch.arange(S, dtype=torch.float32),
                torch.arange(S, dtype=torch.float32),
                indexing="ij",
            )
            gaussian = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            fire_mask[0] = torch.max(fire_mask[0], gaussian)

        # Threshold to get binary fire pixels
        fire_binary = (fire_mask > 0.3).float()
        fire_soft = fire_mask.clamp(0, 1)

        # --- Apply fire signatures to bands ---
        i4 = i4 + fire_soft * (_FIRE_I4 - _BG_I4)
        i5 = i5 + fire_soft * (_FIRE_I5 - _BG_I5)
        frp = fire_soft * (torch.rand(1, S, S, generator=g) * 0.5 + 0.5)

        # Burnt scar: reduce visible reflectance near fires
        i1 = (i1 * (1 - fire_soft * 0.5)).clamp(0, 1)
        i2 = (i2 * (1 - fire_soft * 0.6)).clamp(0, 1)

        # Clamp all
        observation = torch.cat([
            i1.clamp(0, 1),
            i2.clamp(0, 1),
            i3.clamp(0, 1),
            i4.clamp(0, 1),
            i5.clamp(0, 1),
            frp.clamp(0, 1),
        ], dim=0)  # [6, S, S]

        return observation, fire_binary  # [6, 256, 256], [1, 256, 256]


if __name__ == "__main__":
    ds = VIIRSFireDataset(num_samples=4, seed=0)
    obs, tgt = ds[0]
    print(f"VIIRS obs: {obs.shape}  min={obs.min():.3f}  max={obs.max():.3f}")
    print(f"VIIRS tgt: {tgt.shape}  fire_pixels={tgt.sum().item():.0f}")
