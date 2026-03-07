"""
MISDO — Real Satellite Dataset Classes (Counterfactual Impact)
================================================================
Loads real Hansen GFC + SRTM + VIIRS chips for all 4 domain models.
Each dataset uses a TEMPORAL COUNTERFACTUAL approach:

  Input:  landscape at T₁ + deforestation mask (T₁→T₂)
  Target: impact delta — how the metric changed in SURROUNDING
          still-forested areas BECAUSE of the clearing

Key design features:
  • Decoupled spatial/temporal split — ``split`` controls which
    chip files are loaded (train vs test spatial tiles).
    ``temporal_split`` controls which year range is sampled:
        train:    events years  1–16,  impact observed by year 18
        test:     events years 17–18,  impact observed by year 20
        validate: events years 19–21,  impact observed by year 23
  • Sliding temporal windows — each __getitem__ randomly samples
    (T₁, T₂) within the temporal_split's valid range, multiplying
    effective dataset size ~100× compared to fixed windows.
  • Control-pixel baseline subtraction — isolates causal signal
    by subtracting background metric change in pixels far from
    clearings.
  • Single-patch augmentation — 50% chance to show only one
    deforestation patch, bridging train/inference gap.
  • Deforestation mask as extra input channel — model knows
    WHERE clearing happened to predict impact on surroundings.

Chip data keys (from download_real_data.py):
    Hansen:  treecover2000, lossyear, gain
    SRTM:    srtm_elevation, srtm_slope, srtm_aspect, srtm_flow_acc,
             srtm_flow_dir, has_real_srtm
    VIIRS:   viirs_fire_count, viirs_mean_frp, viirs_max_bright_ti4,
             viirs_max_bright_ti5, viirs_confidence, viirs_persistence,
             viirs_fire_year_XX (per-year fire counts, XX=12..23),
             has_real_viirs
    Geo:     bounds [west, south, east, north]

Label provenance (* = proxy, clients should be informed):
    Forest: Hansen GFC lossyear — cascade deforestation (✅ real observed).
    Fire:   VIIRS per-year rasters when available (✅ real observed),
            forest-loss proxy when VIIRS key unavailable (⚠ proxy *).
    Hydro:  Physics-informed proxy: erosion delta downstream (⚠ proxy *).
            No ground-truth erosion measurements are used.
    Soil:   Physics-informed proxy: cumulative soil exposure delta (⚠ proxy *).
            No ground-truth soil degradation measurements are used.
"""

from __future__ import annotations

import json
import os
from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import binary_dilation, gaussian_filter, label
from torch import Tensor
from torch.utils.data import Dataset


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════


def compute_global_target_scale(
    model_name: str,
    tiles_dir: str,
    split: str = "train",
    n_samples: int = 200,
    percentile: float = 95.0,
) -> float:
    """Pre-scan chips to compute a global target normalisation scale.

    Instead of normalising each chip's target to [0, 1] independently
    (which destroys magnitude information), this computes the Pth
    percentile of per-chip target maxima across the dataset.  Using
    this as a fixed divisor preserves relative magnitudes across chips.

    Parameters
    ----------
    model_name : str
        One of 'forest', 'fire', 'hydro', 'soil'.
    tiles_dir : str
        Path to the real_tiles directory.
    split : str
        Which spatial split to scan (default 'train').
    n_samples : int
        Max chips to scan (default 200 — fast, ~30s).
    percentile : float
        Percentile of per-chip maxima to use as scale (default 95th).

    Returns
    -------
    scale : float
        The computed global target scale.  Pass this as ``target_scale``
        to the dataset constructors for all splits (train, test, validate).
    """
    manifest_file = os.path.join(os.path.abspath(tiles_dir), "manifest.json")
    with open(manifest_file) as f:
        manifest = json.load(f)
    entries = manifest.get(split, [])

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(entries))[:n_samples]
    maxima = []

    abs_tiles = os.path.abspath(tiles_dir)
    for idx in indices:
        entry = entries[idx]
        file_path = _resolve_chip_path(abs_tiles, entry["file"])
        if not os.path.exists(file_path):
            continue
        try:
            data = np.load(file_path)
        except Exception:
            continue

        tc = data["treecover2000"].astype(np.float32) / 100.0
        lossyear = data["lossyear"].astype(np.float32)

        # Use a representative temporal window (mid-range)
        t1, t2, t_impact = 5, 10, 14

        deforestation_mask = (
            (lossyear > t1) & (lossyear <= t2)
        ).astype(np.float32)
        if deforestation_mask.sum() < 5:
            continue

        forest_at_t2 = tc.copy()
        forest_at_t2[(lossyear > 0) & (lossyear <= t2)] = 0.0
        forest_mask = (forest_at_t2 > 0.3).astype(np.float32)

        if model_name == "forest":
            additional_loss = (
                (lossyear > t2) & (lossyear <= t_impact)
            ).astype(np.float32)
            raw_target = _control_baseline(
                additional_loss, deforestation_mask, forest_mask, near_radius=15,
            )

        elif model_name == "fire":
            # Use proxy (worst-case — no VIIRS)
            future_loss = (
                (lossyear > t2) & (lossyear <= t_impact)
            ).astype(np.float32)
            near_clearing = binary_dilation(
                deforestation_mask > 0.5, iterations=5
            ).astype(np.float32)
            fire_change = future_loss * near_clearing
            raw_target = _control_baseline(
                fire_change, deforestation_mask, forest_mask, near_radius=15,
            )

        elif model_name == "hydro":
            slope = np.nan_to_num(
                data.get("srtm_slope", np.zeros_like(tc))
            ).astype(np.float32)
            elevation = np.nan_to_num(
                data.get("srtm_elevation", np.zeros_like(tc))
            ).astype(np.float32)
            flow_acc = np.nan_to_num(
                data.get("srtm_flow_acc", np.zeros_like(tc))
            ).astype(np.float32)

            dy, dx = np.gradient(elevation)
            d2y = np.gradient(dy, axis=0)
            d2x = np.gradient(dx, axis=1)
            curvature = np.clip(-(d2y + d2x), 0, None)
            curv_max = curvature.max()
            curvature_norm = curvature / (curv_max + 1e-8)

            forest_at_t1 = tc.copy()
            forest_at_t1[(lossyear > 0) & (lossyear <= t1)] = 0.0
            exposed_before = (forest_at_t1 < 0.3).astype(np.float32)
            erosion_before = exposed_before * slope * (1.0 + curvature_norm)
            exposed_after = (
                (forest_at_t2 < 0.3) | (deforestation_mask > 0.5)
            ).astype(np.float32)
            erosion_after = exposed_after * slope * (1.0 + curvature_norm)
            erosion_delta = np.clip(erosion_after - erosion_before, 0, 1)
            raw_target = gaussian_filter(
                erosion_delta * (1.0 + flow_acc * 3.0), sigma=5.0
            )

        elif model_name == "soil":
            slope = np.nan_to_num(
                data.get("srtm_slope", np.zeros_like(tc))
            ).astype(np.float32)
            cumulative_with = np.zeros_like(tc)
            cumulative_without = np.zeros_like(tc)
            for y in range(t1 + 1, t_impact + 1):
                years_since = float(t_impact - y + 1)
                deforested_with = (
                    (lossyear > 0) & (lossyear <= y)
                ).astype(np.float32)
                cumulative_with += deforested_with * slope * years_since
                deforested_without = (
                    (lossyear > 0) & (lossyear <= y)
                    & ~((lossyear > t1) & (lossyear <= t2))
                ).astype(np.float32)
                cumulative_without += deforested_without * slope * years_since
            raw_target = np.clip(cumulative_with - cumulative_without, 0, None)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        chip_max = float(raw_target.max())
        if chip_max > 1e-8:
            maxima.append(chip_max)

    if not maxima:
        print(f"  ⚠ Could not compute target scale for {model_name} "
              f"(no valid chips). Using 1.0")
        return 1.0

    scale = float(np.percentile(maxima, percentile))
    if scale < 1e-8:
        scale = float(np.max(maxima))
    print(f"  Target scale for {model_name}: {scale:.6f} "
          f"(p{percentile:.0f} of {len(maxima)} chips)")
    return scale

def _resolve_chip_path(tiles_dir: str, raw_path: str) -> str:
    """Resolve chip file path, handling both absolute and relative paths."""
    if os.path.isabs(raw_path) and os.path.exists(raw_path):
        return raw_path
    if os.path.exists(raw_path):
        return os.path.abspath(raw_path)
    parts = raw_path.replace("\\", "/").split("/")
    if len(parts) >= 2:
        candidate = os.path.join(tiles_dir, parts[-2], parts[-1])
        if os.path.exists(candidate):
            return candidate
    basename = os.path.basename(raw_path)
    for split_name in ("train", "test"):
        candidate = os.path.join(tiles_dir, split_name, basename)
        if os.path.exists(candidate):
            return candidate
    return raw_path


def _select_single_patch(
    deforestation_mask: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Select ONE connected deforestation patch from the full mask.

    Used for single-patch augmentation — bridges the gap between
    multi-patch training data and single-tile inference queries.
    """
    labeled, num_features = label(deforestation_mask > 0.5)
    if num_features <= 1:
        return deforestation_mask
    # Pick a random connected component
    patch_id = rng.randint(1, num_features + 1)
    return (labeled == patch_id).astype(np.float32)


def _control_baseline(
    metric_change: np.ndarray,
    deforestation_mask: np.ndarray,
    forest_mask: np.ndarray,
    near_radius: int = 15,
) -> np.ndarray:
    """Subtract background change to isolate causal impact signal.

    Compares metric change near clearings vs. far from clearings
    within the same chip.  Returns the EXCESS change attributable
    to the clearing.

    Parameters
    ----------
    metric_change : ndarray [H, W]
        Raw change in the metric (e.g., forest loss, fire count).
    deforestation_mask : ndarray [H, W]
        Binary mask of cleared areas.
    forest_mask : ndarray [H, W]
        Binary mask of remaining forest.
    near_radius : int
        Dilation radius defining "near clearing" zone.
    """
    # Define "near clearing" zone (impacted area)
    near_zone = binary_dilation(
        deforestation_mask > 0.5, iterations=near_radius
    ).astype(np.float32)
    near_forest = near_zone * forest_mask

    # "Far from clearing" = forested but outside the near zone
    far_forest = (1.0 - near_zone) * forest_mask

    # Compute background rate (change in far-from-clearing forest)
    far_sum = (metric_change * far_forest).sum()
    far_count = far_forest.sum()
    background_rate = far_sum / (far_count + 1e-8) if far_count > 100 else 0.0

    # Excess = observed change - background rate
    excess = metric_change - background_rate
    # Only keep positive excess (clearing caused MORE of the effect)
    excess = np.clip(excess, 0, None)
    # Only measure in forested areas (not in the cleared zone itself)
    excess = excess * forest_mask

    return excess


def _sample_window(
    split: str,
    rng: np.random.RandomState,
    train_end: int = 18,
    min_window: int = 2,
    max_window: int = 5,
) -> Tuple[int, int, int]:
    """Sample random (T₁, T₂, T_impact) window within split bounds.

    Strict temporal separation ensures NO overlap between splits:
        Train:    events in years  1–16,  impact observed by year 18
        Test:     events in years 17–18,  impact observed by year 20
        Validate: events in years 19–21,  impact observed by year 23

    Returns
    -------
    t1 : int
        Start year (event window start).
    t2 : int
        End year (event window end; deforestation happens in [t1, t2]).
    t_impact : int
        Year by which to measure impact (t2 + delta).
    """
    if split == "train":
        # Events must end by year 16 so that with a 2-year impact window,
        # impact is fully observed by year 18.  This leaves years 17+
        # completely untouched for test/validate.
        delta = rng.randint(min_window, max_window + 1)
        max_t2 = min(16, train_end - delta)  # cap at year 16
        min_t1 = 1
        if max_t2 <= min_t1 + 1:
            t1, t2, t_impact = 1, 8, min(8 + delta, train_end)
        else:
            t2 = rng.randint(min_t1 + 2, max_t2 + 1)  # ensure t2 >= 3
            max_win = min(max_window, t2 - min_t1)
            window = max(1, rng.randint(1, max_win + 1)) if max_win >= 1 else 1
            t1 = max(min_t1, t2 - window)
            t_impact = t2 + delta
    elif split == "test":
        # Test: events in years 17-18, impact observed by 20
        # Strictly non-overlapping with train (≤16) and validate (≥19)
        # numpy randint upper is exclusive, so randint(17, 19) → {17, 18}
        t1 = rng.randint(17, 19)
        t2 = min(t1 + rng.randint(0, 2), 18)
        t_impact = min(t2 + 2, 20)
    else:
        # Validate: events in years 19-21, impact observed by 23
        # Strictly non-overlapping with train (≤16) and test (17-18)
        t1 = rng.randint(19, 22)  # 19, 20, or 21
        t2 = min(t1 + rng.randint(0, 3), 21)
        t_impact = min(t2 + 2, 23)

    return int(t1), int(t2), int(t_impact)


# ═══════════════════════════════════════════════════════════════════════════
# Forest Impact Dataset
# ═══════════════════════════════════════════════════════════════════════════

class RealHansenDataset(Dataset):
    """Multi-temporal Hansen GFC dataset — CASCADE DEFORESTATION IMPACT.

    Target: how much additional forest loss occurs in surrounding
    pixels AFTER the clearing event (cascade / contagion effect).

    Input:  [T, 6, 256, 256]  — temporal forest state + deforestation mask
    Target: [1, 256, 256]     — cascade deforestation impact delta [0, 1]
    """

    def __init__(
        self,
        tiles_dir: str = "datasets/real_tiles",
        split: str = "train",
        T: int = 5,
        train_end_year: int = 18,
        year_start: int | None = None,
        target_scale: float | None = None,
        temporal_split: str | None = None,
    ) -> None:
        self.tiles_dir = os.path.abspath(tiles_dir)
        self.split = split
        self.temporal_split = temporal_split if temporal_split is not None else split
        self.T = T
        self.train_end_year = train_end_year
        self.year_start = year_start
        self.target_scale = target_scale
        self._epoch_seed: int = 0

        manifest_file = os.path.join(self.tiles_dir, "manifest.json")
        with open(manifest_file) as f:
            manifest = json.load(f)
        self.entries = manifest.get(split, [])

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for reproducible per-epoch randomisation."""
        self._epoch_seed = epoch * 100003

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        entry = self.entries[idx]
        file_path = _resolve_chip_path(self.tiles_dir, entry["file"])
        data = np.load(file_path)
        rng = np.random.RandomState(idx + self._epoch_seed)

        treecover = data["treecover2000"].astype(np.float32)
        lossyear = data["lossyear"].astype(np.float32)
        gain = data["gain"].astype(np.float32)

        tc_norm = treecover / 100.0

        # ── Sample temporal window ──
        t1, t2, t_impact = _sample_window(
            self.temporal_split, rng, self.train_end_year,
        )

        # ── Deforestation mask: cleared between t1 and t2 ──
        deforestation_mask = (
            (lossyear > t1) & (lossyear <= t2)
        ).astype(np.float32)

        # Single-patch augmentation: 50% chance during training
        if self.temporal_split == "train" and rng.random() < 0.5 and deforestation_mask.sum() > 10:
            deforestation_mask = _select_single_patch(deforestation_mask, rng)

        no_deforestation = np.zeros_like(deforestation_mask)

        # ── Build temporal frames ──
        start = self.year_start if self.year_start is not None else (
            max(1, t1 - self.T + 1)
        )
        year_steps = np.round(np.linspace(start, t2, self.T)).astype(int)

        frames_factual = []      # landscape WITHOUT the clearing query
        frames_counterfactual = []  # landscape WITH the clearing query
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
                prev_forest = frames_counterfactual[-1][0]
                canopy_change = forest_at_year - prev_forest
            else:
                canopy_change = np.zeros_like(forest_at_year)

            # Factual: deforestation_mask = zeros (no clearing proposed)
            frame_f = np.stack([
                forest_at_year, recent_loss, gain, ndvi_proxy,
                canopy_change, no_deforestation,
            ], axis=0)
            frames_factual.append(frame_f)

            # Counterfactual: deforestation_mask = real clearing
            frame_cf = np.stack([
                forest_at_year, recent_loss, gain, ndvi_proxy,
                canopy_change, deforestation_mask,
            ], axis=0)
            frames_counterfactual.append(frame_cf)

        # ── Counterfactual target: cascade forest loss ──
        forest_at_t2 = tc_norm.copy()
        forest_at_t2[(lossyear > 0) & (lossyear <= t2)] = 0.0

        additional_loss = (
            (lossyear > t2) & (lossyear <= t_impact)
        ).astype(np.float32)

        forest_mask = (forest_at_t2 > 0.3).astype(np.float32)

        cascade_impact = _control_baseline(
            additional_loss, deforestation_mask, forest_mask, near_radius=15,
        )

        if self.target_scale is not None:
            cascade_impact = cascade_impact / (self.target_scale + 1e-8)
        else:
            ci_max = cascade_impact.max()
            if ci_max > 1e-8:
                cascade_impact = cascade_impact / ci_max

        cascade_impact = gaussian_filter(cascade_impact, sigma=2.0)
        cascade_impact = np.clip(cascade_impact, 0, 1)
        target = cascade_impact[np.newaxis, :, :]

        obs_f = np.stack(frames_factual, axis=0)
        obs_cf = np.stack(frames_counterfactual, axis=0)
        return (
            torch.from_numpy(obs_f),
            torch.from_numpy(obs_cf),
            torch.from_numpy(target),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Fire Impact Dataset
# ═══════════════════════════════════════════════════════════════════════════

class RealFireDataset(Dataset):
    """Fire risk impact dataset — FIRE INCREASE NEAR CLEARINGS.

    Target: increase in fire activity in surrounding forest after
    nearby deforestation (uses real VIIRS per-year data when available).

    Input:  [T, 7, 256, 256]  — fire features + deforestation mask
    Target: [1, 256, 256]     — fire impact delta [0, 1]
    """

    def __init__(
        self,
        tiles_dir: str = "datasets/real_tiles",
        split: str = "train",
        T: int = 5,
        train_end_year: int = 18,
        year_start: int | None = None,
        target_scale: float | None = None,
        temporal_split: str | None = None,
    ) -> None:
        self.tiles_dir = os.path.abspath(tiles_dir)
        self.split = split
        self.temporal_split = temporal_split if temporal_split is not None else split
        self.T = T
        self.train_end_year = train_end_year
        self.year_start = year_start
        self.target_scale = target_scale
        self._epoch_seed: int = 0

        manifest_file = os.path.join(self.tiles_dir, "manifest.json")
        with open(manifest_file) as f:
            manifest = json.load(f)
        self.entries = manifest.get(split, [])

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for reproducible per-epoch randomisation."""
        self._epoch_seed = epoch * 100003

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        entry = self.entries[idx]
        file_path = _resolve_chip_path(self.tiles_dir, entry["file"])
        data = np.load(file_path)
        rng = np.random.RandomState(idx + self._epoch_seed)

        tc = data["treecover2000"].astype(np.float32) / 100.0
        lossyear = data["lossyear"].astype(np.float32)

        has_viirs = (
            data.get("has_real_viirs", np.array([0]))[0] > 0
            if "has_real_viirs" in data
            else False
        )

        slope = np.nan_to_num(data.get("srtm_slope", np.zeros_like(tc))).astype(np.float32)
        elevation = np.nan_to_num(data.get("srtm_elevation", np.zeros_like(tc))).astype(np.float32)

        # ── Sample temporal window ──
        t1, t2, t_impact = _sample_window(
            self.temporal_split, rng, self.train_end_year,
        )

        # ── Deforestation mask ──
        deforestation_mask = (
            (lossyear > t1) & (lossyear <= t2)
        ).astype(np.float32)

        if self.temporal_split == "train" and rng.random() < 0.5 and deforestation_mask.sum() > 10:
            deforestation_mask = _select_single_patch(deforestation_mask, rng)

        no_deforestation = np.zeros_like(deforestation_mask)

        # ── Build temporal frames ──
        start = self.year_start if self.year_start is not None else (
            max(1, t1 - self.T + 1)
        )
        year_steps = np.round(np.linspace(start, t2, self.T)).astype(int)

        frames_factual = []
        frames_counterfactual = []
        for t in range(self.T):
            year = year_steps[t]
            prev_year = year_steps[t - 1] if t > 0 else 0

            forest = tc.copy()
            forest[(lossyear > 0) & (lossyear <= year)] = 0.0

            recent = (
                (lossyear > prev_year) & (lossyear <= year)
            ).astype(np.float32)

            if has_viirs:
                year_key = f"viirs_fire_year_{year:02d}"
                if year_key in data:
                    fire_at_year = data[year_key].astype(np.float32)
                else:
                    fire_at_year = data["viirs_fire_count"].astype(np.float32)

                frp = data["viirs_mean_frp"].astype(np.float32)
                bright_ti4 = data["viirs_max_bright_ti4"].astype(np.float32)
                bright_ti5 = data["viirs_max_bright_ti5"].astype(np.float32)

                # Factual: no clearing mask
                frame_f = np.stack([
                    forest, recent, fire_at_year, frp,
                    bright_ti4, bright_ti5, no_deforestation,
                ], axis=0)
                # Counterfactual: real clearing mask
                frame_cf = np.stack([
                    forest, recent, fire_at_year, frp,
                    bright_ti4, bright_ti5, deforestation_mask,
                ], axis=0)
            else:
                # Proxy fire channels
                non_forest = forest < 0.3
                edge_zone = binary_dilation(
                    non_forest, iterations=3
                ).astype(np.float32)
                exposure = edge_zone * (forest > 0.3).astype(np.float32)
                dryness = 1.0 - forest

                # Factual: no clearing mask
                frame_f = np.stack([
                    forest, recent, exposure, dryness,
                    slope, elevation, no_deforestation,
                ], axis=0)
                # Counterfactual: real clearing mask
                frame_cf = np.stack([
                    forest, recent, exposure, dryness,
                    slope, elevation, deforestation_mask,
                ], axis=0)

            frames_factual.append(frame_f)
            frames_counterfactual.append(frame_cf)

        # ── Counterfactual fire target ──
        forest_at_t2 = tc.copy()
        forest_at_t2[(lossyear > 0) & (lossyear <= t2)] = 0.0
        forest_mask = (forest_at_t2 > 0.3).astype(np.float32)

        if has_viirs:
            fire_before = np.zeros_like(tc)
            fire_after = np.zeros_like(tc)
            count_before = 0
            count_after = 0

            for y in range(max(12, t1 - 2), t1 + 1):
                key = f"viirs_fire_year_{y:02d}"
                if key in data:
                    fire_before += data[key].astype(np.float32)
                    count_before += 1

            for y in range(t2, min(24, t_impact + 1)):
                key = f"viirs_fire_year_{y:02d}"
                if key in data:
                    fire_after += data[key].astype(np.float32)
                    count_after += 1

            if count_before > 0:
                fire_before /= count_before
            if count_after > 0:
                fire_after /= count_after

            fire_change = fire_after - fire_before
            fire_change = np.clip(fire_change, 0, None)
        else:
            future_loss = (
                (lossyear > t2) & (lossyear <= t_impact)
            ).astype(np.float32)
            near_clearing = binary_dilation(
                deforestation_mask > 0.5, iterations=5
            ).astype(np.float32)
            fire_change = future_loss * near_clearing

        fire_impact = _control_baseline(
            fire_change, deforestation_mask, forest_mask, near_radius=15,
        )

        if self.target_scale is not None:
            fire_impact = fire_impact / (self.target_scale + 1e-8)
        else:
            fi_max = fire_impact.max()
            if fi_max > 1e-8:
                fire_impact = fire_impact / fi_max

        fire_impact = gaussian_filter(fire_impact, sigma=2.0)
        fire_impact = np.clip(fire_impact, 0, 1)
        target = fire_impact[np.newaxis, :, :]

        obs_f = np.stack(frames_factual, axis=0)
        obs_cf = np.stack(frames_counterfactual, axis=0)
        return (
            torch.from_numpy(obs_f.astype(np.float32)),
            torch.from_numpy(obs_cf.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Hydro Impact Dataset
# ═══════════════════════════════════════════════════════════════════════════

class RealHydroDataset(Dataset):
    """Hydrological impact dataset — EROSION DELTA DOWNSTREAM.

    Physics-informed proxy target: erosion risk INCREASE on downstream
    cells after upstream deforestation, compared to the baseline
    (pre-clearing) condition.

    Input:  [6, 256, 256]  — terrain + forest + deforestation mask
    Target: [1, 256, 256]  — water pollution impact delta [0, 1]
    """

    def __init__(
        self,
        tiles_dir: str = "datasets/real_tiles",
        split: str = "train",
        train_end_year: int = 18,
        target_scale: float | None = None,
        temporal_split: str | None = None,
    ) -> None:
        self.tiles_dir = os.path.abspath(tiles_dir)
        self.split = split
        self.temporal_split = temporal_split if temporal_split is not None else split
        self.train_end_year = train_end_year
        self.target_scale = target_scale
        self._epoch_seed: int = 0

        manifest_file = os.path.join(self.tiles_dir, "manifest.json")
        with open(manifest_file) as f:
            manifest = json.load(f)
        self.entries = manifest.get(split, [])

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for reproducible per-epoch randomisation."""
        self._epoch_seed = epoch * 100003

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        entry = self.entries[idx]
        file_path = _resolve_chip_path(self.tiles_dir, entry["file"])
        data = np.load(file_path)
        rng = np.random.RandomState(idx + self._epoch_seed)

        tc = data["treecover2000"].astype(np.float32) / 100.0
        lossyear = data["lossyear"].astype(np.float32)

        elevation = np.nan_to_num(data.get("srtm_elevation", np.zeros_like(tc))).astype(np.float32)
        slope = np.nan_to_num(data.get("srtm_slope", np.zeros_like(tc))).astype(np.float32)
        aspect = np.nan_to_num(data.get("srtm_aspect", np.zeros_like(tc))).astype(np.float32)
        flow_acc = np.nan_to_num(data.get("srtm_flow_acc", np.zeros_like(tc))).astype(np.float32)

        # ── Sample temporal window ──
        t1, t2, t_impact = _sample_window(
            self.temporal_split, rng, self.train_end_year,
        )

        # ── Deforestation mask ──
        deforestation_mask = (
            (lossyear > t1) & (lossyear <= t2)
        ).astype(np.float32)

        if self.temporal_split == "train" and rng.random() < 0.5 and deforestation_mask.sum() > 10:
            deforestation_mask = _select_single_patch(deforestation_mask, rng)

        no_deforestation = np.zeros_like(deforestation_mask)

        # Forest state at t2
        forest = tc.copy()
        forest[(lossyear > 0) & (lossyear <= t2)] = 0.0

        # Factual input: [6, H, W] — terrain + forest + NO deforestation mask
        obs_f = np.stack([
            elevation, slope, aspect, flow_acc, forest, no_deforestation,
        ], axis=0)

        # Counterfactual input: [6, H, W] — terrain + forest + deforestation mask
        obs_cf = np.stack([
            elevation, slope, aspect, flow_acc, forest, deforestation_mask,
        ], axis=0)

        # ── Counterfactual erosion target ──
        dy, dx = np.gradient(elevation)
        d2y = np.gradient(dy, axis=0)
        d2x = np.gradient(dx, axis=1)
        curvature = np.clip(-(d2y + d2x), 0, None)
        curv_max = curvature.max()
        curvature_norm = curvature / (curv_max + 1e-8)

        forest_at_t1 = tc.copy()
        forest_at_t1[(lossyear > 0) & (lossyear <= t1)] = 0.0

        exposed_before = (forest_at_t1 < 0.3).astype(np.float32)
        erosion_before = exposed_before * slope * (1.0 + curvature_norm)

        exposed_after = ((forest < 0.3) | (deforestation_mask > 0.5)).astype(np.float32)
        erosion_after = exposed_after * slope * (1.0 + curvature_norm)

        erosion_delta = np.clip(erosion_after - erosion_before, 0, 1)

        pollution_delta = gaussian_filter(
            erosion_delta * (1.0 + flow_acc * 3.0), sigma=5.0
        )

        if self.target_scale is not None:
            pollution_delta = pollution_delta / (self.target_scale + 1e-8)
        else:
            p_max = pollution_delta.max()
            if p_max > 1e-8:
                pollution_delta = pollution_delta / p_max

        pollution_delta = np.clip(pollution_delta, 0, 1)
        target = pollution_delta[np.newaxis, :, :]

        return (
            torch.from_numpy(obs_f.astype(np.float32)),
            torch.from_numpy(obs_cf.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Soil Impact Dataset
# ═══════════════════════════════════════════════════════════════════════════

class RealSoilDataset(Dataset):
    """Soil degradation impact dataset — EXPOSURE DELTA.

    Physics-informed proxy target: soil exposure INCREASE in
    neighbouring cells after clearing, with temporal compounding
    (longer exposure = worse degradation).  Compared to pre-clearing
    baseline to isolate the clearing's contribution.

    Input:  [T, 5, 256, 256]  — soil state + deforestation mask
    Target: [1, 256, 256]     — soil degradation impact delta [0, 1]
    """

    def __init__(
        self,
        tiles_dir: str = "datasets/real_tiles",
        split: str = "train",
        T: int = 5,
        train_end_year: int = 18,
        year_start: int | None = None,
        target_scale: float | None = None,
        temporal_split: str | None = None,
    ) -> None:
        self.tiles_dir = os.path.abspath(tiles_dir)
        self.split = split
        self.temporal_split = temporal_split if temporal_split is not None else split
        self.T = T
        self.train_end_year = train_end_year
        self.year_start = year_start
        self.target_scale = target_scale
        self._epoch_seed: int = 0

        manifest_file = os.path.join(self.tiles_dir, "manifest.json")
        with open(manifest_file) as f:
            manifest = json.load(f)
        self.entries = manifest.get(split, [])

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for reproducible per-epoch randomisation."""
        self._epoch_seed = epoch * 100003

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        entry = self.entries[idx]
        file_path = _resolve_chip_path(self.tiles_dir, entry["file"])
        data = np.load(file_path)
        rng = np.random.RandomState(idx + self._epoch_seed)

        tc = data["treecover2000"].astype(np.float32) / 100.0
        lossyear = data["lossyear"].astype(np.float32)

        slope = np.nan_to_num(data.get("srtm_slope", np.zeros_like(tc))).astype(np.float32)
        elevation = np.nan_to_num(data.get("srtm_elevation", np.zeros_like(tc))).astype(np.float32)

        # ── Sample temporal window ──
        t1, t2, t_impact = _sample_window(
            self.temporal_split, rng, self.train_end_year,
        )

        # ── Deforestation mask ──
        deforestation_mask = (
            (lossyear > t1) & (lossyear <= t2)
        ).astype(np.float32)

        if self.temporal_split == "train" and rng.random() < 0.5 and deforestation_mask.sum() > 10:
            deforestation_mask = _select_single_patch(deforestation_mask, rng)

        no_deforestation = np.zeros_like(deforestation_mask)

        # ── Build temporal frames ──
        start = self.year_start if self.year_start is not None else (
            max(1, t1 - self.T + 1)
        )
        year_steps = np.round(np.linspace(start, t2, self.T)).astype(int)

        frames_factual = []
        frames_counterfactual = []
        for t in range(self.T):
            year = year_steps[t]
            forest = tc.copy()
            forest[(lossyear > 0) & (lossyear <= year)] = 0.0

            moisture = forest * 0.8 + 0.2
            veg_water = forest * 0.9
            temp = 1.0 - forest * 0.6

            # Factual: no clearing mask
            frame_f = np.stack([
                moisture, veg_water, temp, slope, no_deforestation,
            ], axis=0)
            frames_factual.append(frame_f)

            # Counterfactual: real clearing mask
            frame_cf = np.stack([
                moisture, veg_water, temp, slope, deforestation_mask,
            ], axis=0)
            frames_counterfactual.append(frame_cf)

        # ── Counterfactual soil degradation target ──
        forest_at_t1 = tc.copy()
        forest_at_t1[(lossyear > 0) & (lossyear <= t1)] = 0.0

        cumulative_with = np.zeros_like(tc)
        cumulative_without = np.zeros_like(tc)

        for y in range(t1 + 1, t_impact + 1):
            years_since = float(t_impact - y + 1)

            deforested_with = (
                (lossyear > 0) & (lossyear <= y)
            ).astype(np.float32)
            cumulative_with += deforested_with * slope * years_since

            deforested_without = (
                (lossyear > 0) & (lossyear <= y)
                & ~((lossyear > t1) & (lossyear <= t2))
            ).astype(np.float32)
            cumulative_without += deforested_without * slope * years_since

        soil_delta = np.clip(cumulative_with - cumulative_without, 0, None)

        if self.target_scale is not None:
            soil_delta = soil_delta / (self.target_scale + 1e-8)
        else:
            sd_max = soil_delta.max()
            if sd_max > 1e-8:
                soil_delta = soil_delta / sd_max

        degradation = gaussian_filter(soil_delta, sigma=2.0)
        degradation = np.clip(degradation, 0, 1)
        target = degradation[np.newaxis, :, :]

        obs_f = np.stack(frames_factual, axis=0)
        obs_cf = np.stack(frames_counterfactual, axis=0)
        return (
            torch.from_numpy(obs_f.astype(np.float32)),
            torch.from_numpy(obs_cf.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )
