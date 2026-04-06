"""
⚠️  DEPRECATED — Early Prototype Module
========================================
This module is from the early MISDO prototype and is NOT used in
production training.  Production training uses:
  - ``train_real_models.py`` + ``datasets/real_datasets.py``

Kept for backward compatibility with older test scripts only.
Do NOT add new functionality here.

Original description:
    Synthetic Earth Observation dataset generator for local development.

Channels (20 total):
    0-9   : Sentinel-2 Optical / NIR / SWIR
    10-11 : Sentinel-1 SAR  (VV / VH)
    12    : GEDI LiDAR      (Canopy Height Model)
    13-15 : SRTM Topography (Elevation, Slope, Aspect)
    16-17 : ERA5 Climate    (Precipitation, Wind Speed)
    18-19 : Proximity Grids (Distance to River, Distance to Road)
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Channel constants
# ---------------------------------------------------------------------------
NUM_CHANNELS: int = 20
SPATIAL_SIZE: int = 256

CHANNEL_NAMES: list[str] = [
    # Sentinel-2
    "S2_B02", "S2_B03", "S2_B04", "S2_B05", "S2_B06",
    "S2_B07", "S2_B08", "S2_B8A", "S2_B11", "S2_B12",
    # Sentinel-1
    "SAR_VV", "SAR_VH",
    # GEDI
    "GEDI_CHM",
    # SRTM
    "SRTM_Elev", "SRTM_Slope", "SRTM_Aspect",
    # ERA5
    "ERA5_Precip", "ERA5_Wind",
    # Proximity
    "Prox_River", "Prox_Road",
]

assert len(CHANNEL_NAMES) == NUM_CHANNELS


# ---------------------------------------------------------------------------
# Mock Earth-Observation Dataset
# ---------------------------------------------------------------------------
class MockEODataset(Dataset):
    """Generates synthetic multi-modal EO tensors via ``torch.randn``.

    Each sample returns:
        observation_tensor : Tensor[20, 256, 256]  — normalised to [0, 1]
        initial_forest_mask: Tensor[1, 256, 256]   — all 1.0
    """

    def __init__(
        self,
        num_samples: int = 64,
        num_channels: int = NUM_CHANNELS,
        spatial_size: int = SPATIAL_SIZE,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.spatial_size = spatial_size
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # Deterministic per-sample seed for reproducibility
        generator = torch.Generator().manual_seed(self.seed + idx)

        # Raw synthetic observation  — Shape: [C, H, W]
        raw: Tensor = torch.randn(
            self.num_channels,
            self.spatial_size,
            self.spatial_size,
            generator=generator,
        )

        # Normalise to [0, 1] via sigmoid
        observation: Tensor = torch.sigmoid(raw)  # Shape: [20, 256, 256]

        # Forest mask — starts fully forested
        forest_mask: Tensor = torch.ones(
            1, self.spatial_size, self.spatial_size
        )  # Shape: [1, 256, 256]

        return observation, forest_mask


# ---------------------------------------------------------------------------
# Real Earth-Observation Dataset (composing domain-specific loaders)
# ---------------------------------------------------------------------------
class RealEODataset(Dataset):
    """Composes all 4 domain-specific datasets into a unified interface.

    Each sample returns data from all 4 sources (VIIRS, Hansen, SRTM, SMAP)
    as a dictionary of (observation, target) tuples.
    """

    def __init__(
        self,
        num_samples: int = 64,
        spatial_size: int = SPATIAL_SIZE,
        seed: int = 42,
    ) -> None:
        super().__init__()
        from datasets.viirs_fire import VIIRSFireDataset
        from datasets.hansen_gfc import HansenGFCDataset
        from datasets.srtm_hydro import SRTMHydroDataset
        from datasets.smap_soil import SMAPSoilDataset

        self.datasets = {
            "fire": VIIRSFireDataset(num_samples, spatial_size, seed),
            "forest": HansenGFCDataset(num_samples, spatial_size, seed),
            "hydro": SRTMHydroDataset(num_samples, spatial_size, seed),
            "soil": SMAPSoilDataset(num_samples, spatial_size, seed),
        }
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return {
            name: ds[idx] for name, ds in self.datasets.items()
        }


# ---------------------------------------------------------------------------
# DataLoader helper
# ---------------------------------------------------------------------------
def get_dataloader(
    num_samples: int = 64,
    batch_size: int = 4,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Convenience wrapper that returns a ready-to-iterate DataLoader."""
    dataset = MockEODataset(num_samples=num_samples, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    loader = get_dataloader(num_samples=8, batch_size=2)
    for obs, mask in loader:
        print(f"observation : {obs.shape}  min={obs.min():.3f}  max={obs.max():.3f}")
        print(f"forest_mask : {mask.shape} min={mask.min():.3f}  max={mask.max():.3f}")
        break
