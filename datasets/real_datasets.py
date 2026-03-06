"""
MISDO — Real Satellite Dataset Classes (Multi-Temporal)
=========================================================
Loads real Hansen GFC + SRTM + VIIRS chips for all 4 domain models.

All 4 dataset classes load from the SAME .npz chip files — each class
constructs domain-specific input channels from the shared underlying data.
When real SRTM or VIIRS data is present in the chip, it is used
automatically; otherwise, proxy features are constructed as fallback.

Chip data keys (from download_real_data.py):
    Hansen:  treecover2000, lossyear, gain
    SRTM:    srtm_elevation, srtm_slope, srtm_aspect, srtm_flow_acc,
             srtm_flow_dir, has_real_srtm
    VIIRS:   viirs_fire_count, viirs_mean_frp, viirs_max_bright_ti4,
             viirs_max_bright_ti5, viirs_confidence, viirs_persistence,
             viirs_fire_year_XX (per-year fire counts, XX=12..23),
             has_real_viirs
    Geo:     bounds [west, south, east, north]

Label provenance:
    Forest: Ground-truth from Hansen GFC lossyear (✅ real observed).
    Fire:   When VIIRS available, uses real fire observations per year.
            When VIIRS unavailable, proxy from deforestation adjacency.
    Hydro:  Physics-informed proxy: erosion(slope, deforestation) propagated
            downstream via flow accumulation. Emphasises water pollution.
    Soil:   Physics-informed proxy: cumulative exposure(deforestation, slope)
            with temporal compounding. Emphasises topsoil degradation.
"""

from __future__ import annotations

import json
import os
from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import binary_dilation, gaussian_filter
from torch import Tensor
from torch.utils.data import Dataset


def _resolve_chip_path(tiles_dir: str, raw_path: str) -> str:
    """Resolve chip file path, handling both absolute and relative paths.

    Manifest files may store paths relative to the project root or
    relative to the tiles directory.  This function tries multiple
    strategies so training works regardless of the working directory.
    """
    # 1. Absolute path that exists
    if os.path.isabs(raw_path) and os.path.exists(raw_path):
        return raw_path
    # 2. Relative path that resolves from cwd
    if os.path.exists(raw_path):
        return os.path.abspath(raw_path)
    # 3. Resolve relative to tiles_dir using last 2 path components (split/chip.npz)
    parts = raw_path.replace("\\", "/").split("/")
    if len(parts) >= 2:
        candidate = os.path.join(tiles_dir, parts[-2], parts[-1])
        if os.path.exists(candidate):
            return candidate
    # 4. Just the basename inside each split dir
    basename = os.path.basename(raw_path)
    for split_name in ("train", "test"):
        candidate = os.path.join(tiles_dir, split_name, basename)
        if os.path.exists(candidate):
            return candidate
    # 5. Return as-is — will fail with a clear FileNotFoundError
    return raw_path


class RealHansenDataset(Dataset):
    """Multi-temporal Hansen GFC dataset for ForestLossNet.

    Constructs T temporal frames showing deforestation progression.
    Uses real Hansen lossyear as ground-truth labels.

    Input:  [T, 5, 256, 256]  — temporal forest state
    Target: [1, 256, 256]     — binary deforestation in next period
    """

    def __init__(
        self,
        tiles_dir: str = "datasets/real_tiles",
        split: str = "train",
        T: int = 5,
        train_end_year: int = 18,  # 2018
        year_start: int | None = None,
    ) -> None:
        self.tiles_dir = os.path.abspath(tiles_dir)
        self.split = split
        self.T = T
        self.train_end_year = train_end_year
        self.year_start = year_start

        manifest_file = os.path.join(self.tiles_dir, "manifest.json")
        with open(manifest_file) as f:
            manifest = json.load(f)
        self.entries = manifest[split]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        entry = self.entries[idx]
        file_path = _resolve_chip_path(self.tiles_dir, entry["file"])
        data = np.load(file_path)

        treecover = data["treecover2000"].astype(np.float32)
        lossyear = data["lossyear"].astype(np.float32)
        gain = data["gain"].astype(np.float32)

        tc_norm = treecover / 100.0

        # Determine temporal window
        start = self.year_start if self.year_start is not None else (
            1 if self.split == "train" else self.train_end_year - self.T
        )
        end = self.train_end_year if self.split == "train" else 23
        year_steps = np.round(np.linspace(start, end, self.T + 1)).astype(int)

        frames = []
        for t in range(self.T):
            year = year_steps[t]
            forest_at_year = tc_norm.copy()
            lost_mask = (lossyear > 0) & (lossyear <= year)
            forest_at_year[lost_mask] = 0.0

            prev_year = year_steps[t - 1] if t > 0 else 0
            recent_loss = (
                (lossyear > prev_year) & (lossyear <= year)
            ).astype(np.float32)

            ndvi_proxy = forest_at_year * 0.8 + 0.2 * (1.0 - recent_loss)

            if t > 0:
                prev_forest = frames[-1][0]
                canopy_change = forest_at_year - prev_forest
            else:
                canopy_change = np.zeros_like(forest_at_year)

            frame = np.stack([
                forest_at_year, recent_loss, gain, ndvi_proxy, canopy_change,
            ], axis=0)
            frames.append(frame)

        # Target: loss in the next period
        target_year_start = year_steps[self.T - 1]
        target_year_end = year_steps[self.T]
        target_loss = (
            (lossyear > target_year_start) & (lossyear <= target_year_end)
        ).astype(np.float32)
        target = target_loss[np.newaxis, :, :]

        obs = np.stack(frames, axis=0)
        return torch.from_numpy(obs), torch.from_numpy(target)


class RealFireDataset(Dataset):
    """Fire risk dataset using real VIIRS fire detections when available.

    When real VIIRS data exists, uses per-year fire rasters aligned to
    the temporal frame years — each timestep gets the fire activity for
    that specific year.
    Falls back to proxy fire features derived from Hansen lossyear.

    Input:  [T, 6, 256, 256]  — fire-related features per timestep
    Target: [1, 256, 256]     — fire risk in next period
    """

    def __init__(
        self,
        tiles_dir: str = "datasets/real_tiles",
        split: str = "train",
        T: int = 5,
        train_end_year: int = 18,  # 2018
        year_start: int | None = None,
    ) -> None:
        self.tiles_dir = os.path.abspath(tiles_dir)
        self.split = split
        self.T = T
        self.train_end_year = train_end_year
        self.year_start = year_start

        manifest_file = os.path.join(self.tiles_dir, "manifest.json")
        with open(manifest_file) as f:
            manifest = json.load(f)
        self.entries = manifest[split]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        entry = self.entries[idx]
        file_path = _resolve_chip_path(self.tiles_dir, entry["file"])
        data = np.load(file_path)

        tc = data["treecover2000"].astype(np.float32) / 100.0
        lossyear = data["lossyear"].astype(np.float32)

        # Check for real VIIRS data
        has_viirs = (
            data.get("has_real_viirs", np.array([0]))[0] > 0
            if "has_real_viirs" in data
            else False
        )

        # Use real SRTM terrain if available
        slope = np.nan_to_num(data.get("srtm_slope", np.zeros_like(tc))).astype(np.float32)
        elevation = np.nan_to_num(data.get("srtm_elevation", np.zeros_like(tc))).astype(np.float32)

        # Determine temporal window
        start = self.year_start if self.year_start is not None else (
            1 if self.split == "train" else self.train_end_year - self.T
        )
        end = self.train_end_year if self.split == "train" else 23
        year_steps = np.round(np.linspace(start, end, self.T + 1)).astype(int)

        frames = []
        for t in range(self.T):
            year = year_steps[t]
            prev_year = year_steps[t - 1] if t > 0 else 0

            forest = tc.copy()
            forest[(lossyear > 0) & (lossyear <= year)] = 0.0

            recent = (
                (lossyear > prev_year) & (lossyear <= year)
            ).astype(np.float32)

            if has_viirs:
                # Use per-year VIIRS fire raster if available for this timestep
                year_key = f"viirs_fire_year_{year:02d}"
                if year_key in data:
                    fire_at_year = data[year_key].astype(np.float32)
                else:
                    # Fall back to aggregate fire count
                    fire_at_year = data["viirs_fire_count"].astype(np.float32)

                frp = data["viirs_mean_frp"].astype(np.float32)
                bright_ti4 = data["viirs_max_bright_ti4"].astype(np.float32)
                bright_ti5 = data["viirs_max_bright_ti5"].astype(np.float32)

                # Stack [6, H, W]: forest, recent_loss, fire_at_year, FRP, ti4, ti5
                frame = np.stack([
                    forest, recent, fire_at_year, frp, bright_ti4, bright_ti5,
                ], axis=0)
            else:
                # Proxy fire channels
                non_forest = forest < 0.3
                edge_zone = binary_dilation(
                    non_forest, iterations=3
                ).astype(np.float32)
                exposure = edge_zone * (forest > 0.3).astype(np.float32)
                dryness = 1.0 - forest

                frame = np.stack([
                    forest, recent, exposure, dryness, slope, elevation,
                ], axis=0)

            frames.append(frame)

        # Fire target: areas that burned in next period
        t_start = year_steps[self.T - 1]
        t_end = year_steps[self.T]
        next_loss = (
            (lossyear > t_start) & (lossyear <= t_end)
        ).astype(np.float32)

        if has_viirs:
            # Use per-year VIIRS data for the target period if available
            target_year_key = f"viirs_fire_year_{t_end:02d}"
            if target_year_key in data:
                fire_target = data[target_year_key].astype(np.float32)
            else:
                fire_target = data["viirs_persistence"].astype(np.float32)
            # Combine with deforestation signal (fire often precedes clearing)
            combined = np.clip(fire_target + next_loss * 0.5, 0, 1)
            target = combined[np.newaxis, :, :]
        else:
            existing_clear = (lossyear > 0) & (lossyear <= t_start)
            fire_zone = binary_dilation(
                existing_clear, iterations=5
            ).astype(np.float32)
            fire_target = (next_loss * fire_zone)[np.newaxis, :, :]
            target = fire_target

        obs = np.stack(frames, axis=0)
        return (
            torch.from_numpy(obs.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )


class RealHydroDataset(Dataset):
    """Hydrological risk from real SRTM terrain + deforestation.

    Physics-informed proxy target: Erosion on deforested slopes is
    propagated downstream via flow accumulation to model water
    pollution risk. Uses curvature and flow direction for more
    physically realistic routing than Gaussian blurring alone.

    Input:  [5, 256, 256]  — elevation, slope, aspect, flow_acc, forest_cover
    Target: [1, 256, 256]  — water pollution risk (proxy)

    Note: Channel 5 is forest cover (not flow_direction) because the
    model needs to know WHERE trees were removed to predict erosion.
    """

    def __init__(
        self,
        tiles_dir: str = "datasets/real_tiles",
        split: str = "train",
        train_end_year: int = 18,  # 2018
    ) -> None:
        self.tiles_dir = os.path.abspath(tiles_dir)
        self.split = split
        self.train_end_year = train_end_year

        manifest_file = os.path.join(self.tiles_dir, "manifest.json")
        with open(manifest_file) as f:
            manifest = json.load(f)
        self.entries = manifest[split]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        entry = self.entries[idx]
        file_path = _resolve_chip_path(self.tiles_dir, entry["file"])
        data = np.load(file_path)

        tc = data["treecover2000"].astype(np.float32) / 100.0
        lossyear = data["lossyear"].astype(np.float32)

        # Use real SRTM terrain if available, otherwise fallback
        elevation = np.nan_to_num(data.get("srtm_elevation", np.zeros_like(tc))).astype(np.float32)
        slope = np.nan_to_num(data.get("srtm_slope", np.zeros_like(tc))).astype(np.float32)
        aspect = np.nan_to_num(data.get("srtm_aspect", np.zeros_like(tc))).astype(np.float32)
        flow_acc = np.nan_to_num(data.get("srtm_flow_acc", np.zeros_like(tc))).astype(np.float32)

        # Forest state at analysis year
        year = self.train_end_year if self.split == "train" else 23
        forest = tc.copy()
        forest[(lossyear > 0) & (lossyear <= year)] = 0.0

        # Input: [5, H, W]
        obs = np.stack([elevation, slope, aspect, flow_acc, forest], axis=0)

        # Target: water pollution risk
        # Erosion potential = deforested area × slope × curvature factor
        deforested = ((lossyear > 0) & (lossyear <= year)).astype(np.float32)

        # Curvature: second derivative of elevation (concave = sediment trap)
        dy, dx = np.gradient(elevation)
        d2y = np.gradient(dy, axis=0)
        d2x = np.gradient(dx, axis=1)
        curvature = np.clip(-(d2y + d2x), 0, None)  # positive = convex (high erosion)
        curv_max = curvature.max()
        curvature_norm = curvature / (curv_max + 1e-8)

        # Erosion = deforested × slope × (1 + curvature boost)
        erosion = deforested * slope * (1.0 + curvature_norm)
        erosion = np.clip(erosion, 0, 1)

        # Propagate downstream: weight by flow accumulation and directional blur
        # Larger flow_acc = more upstream contributing area = more pollution
        pollution = gaussian_filter(erosion * (1.0 + flow_acc * 3.0), sigma=5.0)
        p_max = pollution.max()
        if p_max > 1e-8:
            pollution = pollution / p_max
        target = pollution[np.newaxis, :, :]

        return (
            torch.from_numpy(obs.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )


class RealSoilDataset(Dataset):
    """Soil degradation risk from real SRTM terrain + deforestation.

    Physics-informed proxy target: Cumulative soil exposure on steep
    deforested slopes leads to topsoil loss. Unlike the hydro model
    which focuses on downstream water pollution, this model predicts
    in-situ soil degradation severity with temporal compounding.

    Input:  [T, 4, 256, 256]  — soil/vegetation state per timestep
    Target: [1, 256, 256]     — soil degradation risk (proxy)
    """

    def __init__(
        self,
        tiles_dir: str = "datasets/real_tiles",
        split: str = "train",
        T: int = 5,
        train_end_year: int = 18,  # 2018
        year_start: int | None = None,
    ) -> None:
        self.tiles_dir = os.path.abspath(tiles_dir)
        self.split = split
        self.T = T
        self.train_end_year = train_end_year
        self.year_start = year_start

        manifest_file = os.path.join(self.tiles_dir, "manifest.json")
        with open(manifest_file) as f:
            manifest = json.load(f)
        self.entries = manifest[split]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        entry = self.entries[idx]
        file_path = _resolve_chip_path(self.tiles_dir, entry["file"])
        data = np.load(file_path)

        tc = data["treecover2000"].astype(np.float32) / 100.0
        lossyear = data["lossyear"].astype(np.float32)

        # Use real SRTM terrain if available
        slope = np.nan_to_num(data.get("srtm_slope", np.zeros_like(tc))).astype(np.float32)
        elevation = np.nan_to_num(data.get("srtm_elevation", np.zeros_like(tc))).astype(np.float32)

        # Determine temporal window
        start = self.year_start if self.year_start is not None else (
            1 if self.split == "train" else self.train_end_year - self.T
        )
        end = self.train_end_year if self.split == "train" else 23
        year_steps = np.round(np.linspace(start, end, self.T + 1)).astype(int)

        frames = []
        for t in range(self.T):
            year = year_steps[t]
            forest = tc.copy()
            forest[(lossyear > 0) & (lossyear <= year)] = 0.0

            # Soil moisture proxy: forest cover correlates with soil moisture
            moisture = forest * 0.8 + 0.2
            # Vegetation water content proxy
            veg_water = forest * 0.9
            # Temperature proxy: canopy reduces surface temp
            temp = 1.0 - forest * 0.6

            frame = np.stack([moisture, veg_water, temp, slope], axis=0)
            frames.append(frame)

        # Target: cumulative soil degradation with temporal compounding
        # Unlike hydro (which propagates downstream), soil degradation
        # is in-situ and compounds over time
        cumulative_exposure = np.zeros_like(tc)
        for t in range(len(year_steps) - 1):
            y_start = year_steps[t]
            y_end = year_steps[t + 1]
            deforested_at_t = (
                (lossyear > 0) & (lossyear <= y_end)
            ).astype(np.float32)
            # Duration of exposure in years (earlier clearing = more damage)
            years_exposed = float(year_steps[-1] - y_start)
            # Degradation compounds: longer exposure × steeper slope = worse
            cumulative_exposure += deforested_at_t * slope * years_exposed

        # Normalise
        ce_max = cumulative_exposure.max()
        if ce_max > 1e-8:
            cumulative_exposure = cumulative_exposure / ce_max

        # Local smoothing (soil degradation is localised, not propagated downstream)
        degradation = gaussian_filter(cumulative_exposure, sigma=2.0)
        degradation = np.clip(degradation, 0, 1)
        target = degradation[np.newaxis, :, :]

        obs = np.stack(frames, axis=0)
        return (
            torch.from_numpy(obs.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )
