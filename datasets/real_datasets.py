"""
MISDO — Real Hansen GFC Dataset (Multi-Temporal)
==================================================
Loads real Hansen Global Forest Change chips and constructs
temporal sequences showing deforestation progression.

Each sample provides T=5 yearly snapshots of forest state plus
a target mask of the next year's deforestation.
"""

from __future__ import annotations

import json
import os
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class RealHansenDataset(Dataset):
    """Multi-temporal Hansen GFC dataset using real satellite data.

    For each chip, constructs T temporal frames by progressively
    masking out pixels according to their lossyear value.

    Parameters
    ----------
    tiles_dir : str
        Path to datasets/real_tiles/
    split : str
        'train' or 'test'
    T : int
        Number of temporal frames (default 5)
    train_end_year : int
        Last year for training labels (inclusive, 1-indexed where 1=2001)
    """

    def __init__(
        self,
        tiles_dir: str = "datasets/real_tiles",
        split: str = "train",
        T: int = 5,
        train_end_year: int = 20,  # 2020
    ) -> None:
        self.tiles_dir = tiles_dir
        self.split = split
        self.T = T
        self.train_end_year = train_end_year

        manifest_file = os.path.join(tiles_dir, "manifest.json")
        with open(manifest_file) as f:
            manifest = json.load(f)

        self.entries = manifest[split]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        entry = self.entries[idx]
        data = np.load(entry["file"])

        treecover = data["treecover2000"].astype(np.float32)  # [256, 256] 0-100
        lossyear = data["lossyear"].astype(np.float32)  # [256, 256] 0-23
        gain = data["gain"].astype(np.float32)  # [256, 256] 0-1

        # Normalise treecover to [0, 1]
        tc_norm = treecover / 100.0

        # Build temporal frames
        # For train: years 1-20 (2001-2020)
        # For test: years 16-23 (2016-2023) to predict 2021-2023
        if self.split == "train":
            # Spread T frames across training years
            year_steps = np.linspace(1, self.train_end_year, self.T + 1).astype(int)
        else:
            # Test: use recent years
            year_steps = np.linspace(
                self.train_end_year - self.T,
                23,  # latest year
                self.T + 1,
            ).astype(int)

        frames = []  # T frames, each [C, H, W]
        for t in range(self.T):
            year = year_steps[t]

            # Forest state at this year: pixels lost before this year are gone
            forest_at_year = tc_norm.copy()
            lost_mask = (lossyear > 0) & (lossyear <= year)
            forest_at_year[lost_mask] = 0.0

            # Binary loss since last frame
            prev_year = year_steps[t - 1] if t > 0 else 0
            recent_loss = ((lossyear > prev_year) & (lossyear <= year)).astype(
                np.float32
            )

            # NDVI proxy: forest cover × spatial smoothness
            ndvi_proxy = forest_at_year * 0.8 + 0.2 * (1.0 - recent_loss)

            # Canopy change rate
            if t > 0:
                prev_forest = frames[-1][0]  # previous forest state
                canopy_change = forest_at_year - prev_forest
            else:
                canopy_change = np.zeros_like(forest_at_year)

            # Stack channels: [5, H, W]
            frame = np.stack([
                forest_at_year,  # current forest cover
                recent_loss,  # binary loss this period
                gain,  # forest gain (same across time)
                ndvi_proxy,  # vegetation index proxy
                canopy_change,  # change from previous frame
            ], axis=0)
            frames.append(frame)

        # Target: loss in the NEXT period after the last frame
        target_year_start = year_steps[self.T - 1]
        target_year_end = year_steps[self.T]
        target_loss = (
            (lossyear > target_year_start) & (lossyear <= target_year_end)
        ).astype(np.float32)
        target = target_loss[np.newaxis, :, :]  # [1, H, W]

        # Stack temporal frames: [T, C, H, W]
        obs = np.stack(frames, axis=0)

        return torch.from_numpy(obs), torch.from_numpy(target)


class RealFireDataset(Dataset):
    """Fire risk derived from deforestation patterns.

    Fire risk = recent deforestation edges × terrain dryness × exposure.
    Uses Hansen loss patterns as fire proxy since areas with recent clearing
    near forest edges are the primary fire risk zones in the Amazon.
    """

    def __init__(
        self,
        tiles_dir: str = "datasets/real_tiles",
        split: str = "train",
        T: int = 5,
        train_end_year: int = 20,
    ) -> None:
        self.tiles_dir = tiles_dir
        self.split = split
        self.T = T
        self.train_end_year = train_end_year

        manifest_file = os.path.join(tiles_dir, "manifest.json")
        with open(manifest_file) as f:
            manifest = json.load(f)
        self.entries = manifest[split]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        entry = self.entries[idx]
        data = np.load(entry["file"])

        tc = data["treecover2000"].astype(np.float32) / 100.0
        lossyear = data["lossyear"].astype(np.float32)
        slope = data.get("slope", np.zeros_like(tc))
        elevation = data.get("elevation", np.zeros_like(tc))

        if self.split == "train":
            year_steps = np.linspace(1, self.train_end_year, self.T + 1).astype(int)
        else:
            year_steps = np.linspace(
                self.train_end_year - self.T, 23, self.T + 1
            ).astype(int)

        frames = []
        for t in range(self.T):
            year = year_steps[t]
            prev_year = year_steps[t - 1] if t > 0 else 0

            # Forest at this year
            forest = tc.copy()
            forest[(lossyear > 0) & (lossyear <= year)] = 0.0

            # Recent loss (fire trigger)
            recent = ((lossyear > prev_year) & (lossyear <= year)).astype(np.float32)

            # Edge exposure: distance from forest boundary
            from scipy.ndimage import binary_dilation
            non_forest = forest < 0.3
            edge_zone = binary_dilation(non_forest, iterations=3).astype(np.float32)
            exposure = edge_zone * (forest > 0.3).astype(np.float32)

            # Dryness proxy: low canopy = drier
            dryness = 1.0 - forest

            # Stack [6, H, W]
            frame = np.stack([
                forest, recent, exposure, dryness, slope, elevation,
            ], axis=0)
            frames.append(frame)

        # Fire risk target: areas that burned in next period
        # (approximated as newly deforested areas near edges)
        t_start = year_steps[self.T - 1]
        t_end = year_steps[self.T]
        next_loss = ((lossyear > t_start) & (lossyear <= t_end)).astype(np.float32)
        # Fire is more likely at edges of existing deforestation
        from scipy.ndimage import binary_dilation
        existing_clear = (lossyear > 0) & (lossyear <= t_start)
        fire_zone = binary_dilation(existing_clear, iterations=5).astype(np.float32)
        fire_target = (next_loss * fire_zone)[np.newaxis, :, :]

        obs = np.stack(frames, axis=0)
        return torch.from_numpy(obs.astype(np.float32)), torch.from_numpy(fire_target)


class RealHydroDataset(Dataset):
    """Hydrological risk from terrain + deforestation.

    Models how upstream deforestation increases downstream water pollution
    through the flow accumulation network.
    """

    def __init__(
        self,
        tiles_dir: str = "datasets/real_tiles",
        split: str = "train",
        train_end_year: int = 20,
    ) -> None:
        self.tiles_dir = tiles_dir
        self.split = split
        self.train_end_year = train_end_year

        manifest_file = os.path.join(tiles_dir, "manifest.json")
        with open(manifest_file) as f:
            manifest = json.load(f)
        self.entries = manifest[split]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        entry = self.entries[idx]
        data = np.load(entry["file"])

        tc = data["treecover2000"].astype(np.float32) / 100.0
        lossyear = data["lossyear"].astype(np.float32)
        slope = data.get("slope", np.zeros_like(tc))
        aspect = data.get("aspect", np.zeros_like(tc))
        flow_acc = data.get("flow_acc", np.zeros_like(tc))
        elevation = data.get("elevation", np.zeros_like(tc))

        # Forest state at train_end_year
        year = self.train_end_year if self.split == "train" else 23
        forest = tc.copy()
        forest[(lossyear > 0) & (lossyear <= year)] = 0.0

        # Input: [5, H, W]
        obs = np.stack([elevation, slope, aspect, flow_acc, forest], axis=0)

        # Target: water pollution risk
        # Deforested areas on slopes with high flow accumulation
        # cause downstream sediment loading
        deforested = (lossyear > 0) & (lossyear <= year)
        erosion = deforested.astype(np.float32) * slope * 2.0
        erosion = np.clip(erosion, 0, 1)

        # Propagate downstream via flow accumulation
        from scipy.ndimage import gaussian_filter
        pollution = gaussian_filter(erosion * flow_acc, sigma=5.0)
        pollution = pollution / (pollution.max() + 1e-8)
        target = pollution[np.newaxis, :, :]

        return (
            torch.from_numpy(obs.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )


class RealSoilDataset(Dataset):
    """Soil degradation risk from terrain + deforestation history.

    Models how exposed soil on steep deforested slopes leads to
    erosion and degradation.
    """

    def __init__(
        self,
        tiles_dir: str = "datasets/real_tiles",
        split: str = "train",
        T: int = 5,
        train_end_year: int = 20,
    ) -> None:
        self.tiles_dir = tiles_dir
        self.split = split
        self.T = T
        self.train_end_year = train_end_year

        manifest_file = os.path.join(tiles_dir, "manifest.json")
        with open(manifest_file) as f:
            manifest = json.load(f)
        self.entries = manifest[split]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        entry = self.entries[idx]
        data = np.load(entry["file"])

        tc = data["treecover2000"].astype(np.float32) / 100.0
        lossyear = data["lossyear"].astype(np.float32)
        slope = data.get("slope", np.zeros_like(tc))
        elevation = data.get("elevation", np.zeros_like(tc))

        if self.split == "train":
            year_steps = np.linspace(1, self.train_end_year, self.T + 1).astype(int)
        else:
            year_steps = np.linspace(
                self.train_end_year - self.T, 23, self.T + 1
            ).astype(int)

        frames = []
        for t in range(self.T):
            year = year_steps[t]
            forest = tc.copy()
            forest[(lossyear > 0) & (lossyear <= year)] = 0.0

            # Soil moisture proxy: forest = moist, no forest = dry
            moisture = forest * 0.8 + 0.2
            # Vegetation water content
            veg_water = forest * 0.9
            # Temperature proxy: no canopy = hotter
            temp = 1.0 - forest * 0.6

            frame = np.stack([moisture, veg_water, temp, slope], axis=0)
            frames.append(frame)

        # Target: soil degradation
        # Steep deforested areas suffer most
        final_year = year_steps[self.T]
        final_forest = tc.copy()
        final_forest[(lossyear > 0) & (lossyear <= final_year)] = 0.0

        degradation = (1.0 - final_forest) * slope * 2.0
        from scipy.ndimage import gaussian_filter
        degradation = gaussian_filter(degradation, sigma=3.0)
        degradation = np.clip(degradation, 0, 1)
        target = degradation[np.newaxis, :, :]

        obs = np.stack(frames, axis=0)
        return (
            torch.from_numpy(obs.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )
