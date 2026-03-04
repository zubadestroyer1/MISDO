"""
MISDO Dataset Package — Domain-Specific Data Loaders
=====================================================
Each sub-module provides a PyTorch Dataset that generates physically-realistic
synthetic data matching its real-world satellite source.

Modules:
    viirs_fire   — VIIRS VNP14IMG active fire (6 channels)
    hansen_gfc   — Hansen Global Forest Change (5 channels)
    srtm_hydro   — SRTM DEM + HydroSHEDS (5 channels)
    smap_soil    — SMAP L3 Soil Moisture (4 channels)
"""

from __future__ import annotations

from typing import Dict

from torch.utils.data import Dataset

from .viirs_fire import VIIRSFireDataset
from .hansen_gfc import HansenGFCDataset
from .srtm_hydro import SRTMHydroDataset
from .smap_soil import SMAPSoilDataset


def load_all_datasets(
    num_samples: int = 64,
    spatial_size: int = 256,
    seed: int = 42,
) -> Dict[str, Dataset]:
    """Return a dictionary of all 4 domain-specific datasets."""
    return {
        "fire": VIIRSFireDataset(num_samples=num_samples, spatial_size=spatial_size, seed=seed),
        "forest": HansenGFCDataset(num_samples=num_samples, spatial_size=spatial_size, seed=seed),
        "hydro": SRTMHydroDataset(num_samples=num_samples, spatial_size=spatial_size, seed=seed),
        "soil": SMAPSoilDataset(num_samples=num_samples, spatial_size=spatial_size, seed=seed),
    }
