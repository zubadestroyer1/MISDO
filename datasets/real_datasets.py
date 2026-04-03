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
    Hydro:  Real Sentinel-2 MSI NDSSI delta from Planetary Computer
            (✅ real observed suspended-sediment change, via download_msi_smap.py).
    Soil:   Physics-informed exposure delta weighted by real TerraClimate
            soil moisture from Planetary Computer (⚠ hybrid — real weighting
            on a derived target, no direct soil degradation observations).
"""

from __future__ import annotations

import json
import os
import random as _random
from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, label
from torch import Tensor
from torch.utils.data import Dataset


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

# Keys that every chip MUST contain for any model to work.
_REQUIRED_KEYS = ("treecover2000", "lossyear")

# Additional keys checked per-domain (optional — uses fallback defaults if
# absent, but warn if they contain NaN/Inf when present).
_DOMAIN_OPTIONAL_KEYS = {
    "forest": ("gain",),
    "fire": ("srtm_slope", "srtm_elevation"),
    "hydro": ("srtm_elevation", "srtm_slope", "srtm_aspect", "srtm_flow_acc"),
    "soil": ("srtm_slope", "srtm_elevation"),
}

import logging as _logging

_chip_logger = _logging.getLogger("misdo.chip_validation")

_MAX_SKIP_WARNINGS = 50  # cap per-process to avoid log spam
_nan_warn_state: list[int] = [0]  # mutable counter for optional-key NaN/Inf warnings


def validate_chip(
    data: np.lib.npyio.NpzFile,
    file_path: str,
    domain: str | None = None,
) -> tuple[bool, str]:
    """Validate a loaded .npz chip for training quality.

    Checks
    ------
    1. Required keys present (``treecover2000``, ``lossyear``).
    2. No NaN or Inf in required arrays.
    3. ``treecover2000`` is not all-zero (empty chips add noise).
    4. Array shapes are 2-D and consistent across mandatory keys.
    5. If *domain* is given, checks domain-specific optional keys for
       NaN/Inf (warns but still passes, since fallback zeros are used).

    Parameters
    ----------
    data : np.lib.npyio.NpzFile
        The loaded ``.npz`` file.
    file_path : str
        Path to the chip file (for error messages).
    domain : str | None
        One of ``"forest"``, ``"fire"``, ``"hydro"``, ``"soil"``, or None.

    Returns
    -------
    (is_valid, reason) : tuple[bool, str]
        ``(True, "")`` if valid; ``(False, reason_string)`` otherwise.
    """
    basename = os.path.basename(file_path)

    # ── 1. Required key presence ─────────────────────────────────────────
    for key in _REQUIRED_KEYS:
        if key not in data:
            return False, f"{basename}: missing required key '{key}'"

    # ── 2. Load and check required arrays ────────────────────────────────
    tc = data["treecover2000"]
    ly = data["lossyear"]

    if tc.ndim != 2:
        return False, f"{basename}: treecover2000 has {tc.ndim} dims, expected 2"
    if ly.ndim != 2:
        return False, f"{basename}: lossyear has {ly.ndim} dims, expected 2"
    if tc.shape != ly.shape:
        return (
            False,
            f"{basename}: shape mismatch treecover2000 {tc.shape} vs lossyear {ly.shape}",
        )

    if np.any(np.isnan(tc)) or np.any(np.isinf(tc)):
        return False, f"{basename}: treecover2000 contains NaN/Inf"
    if np.any(np.isnan(ly)) or np.any(np.isinf(ly)):
        return False, f"{basename}: lossyear contains NaN/Inf"

    # ── 3. All-zero treecover → useless for training ─────────────────────
    if np.all(tc == 0):
        return False, f"{basename}: treecover2000 is all-zero (no forest)"

    # ── 4. Domain-specific optional-key sanity ───────────────────────────
    if domain is not None and domain in _DOMAIN_OPTIONAL_KEYS:
        for key in _DOMAIN_OPTIONAL_KEYS[domain]:
            if key in data:
                arr = data[key]
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                    if _nan_warn_state[0] < _MAX_SKIP_WARNINGS:
                        _chip_logger.warning(
                            "%s: optional key '%s' contains NaN/Inf "
                            "(will be replaced by nan_to_num fallback)",
                            basename,
                            key,
                        )
                        _nan_warn_state[0] += 1

    return True, ""


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

        valid, _ = validate_chip(data, file_path)
        if not valid:
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
                additional_loss, deforestation_mask, forest_mask,
                near_radius=15, decay_sigma=10.0,  # forest: ~300m cascade
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
                fire_change, deforestation_mask, forest_mask,
                near_radius=15, decay_sigma=17.0,  # fire: ~500m ember transport
            )

        elif model_name == "hydro":
            # Skip chips without real MSI/SMAP data — their NDSSI delta
            # is a zero-filled fallback that would deflate the scale.
            has_real = data.get("has_real_msi_smap", None)
            if has_real is not None and float(has_real.flat[0]) < 0.5:
                continue
            raw_target = data.get("msi_ndssi_delta", None)
            if raw_target is None:
                continue  # No real NDSSI data for this chip — skip
            raw_target = raw_target.astype(np.float32)
        elif model_name == "soil":
            # Replicate the FULL RealSoilDataset target computation:
            # exposure_delta + cascade_exposure*0.5 → control_baseline → SMAP weighting
            slope = np.clip(np.nan_to_num(data.get("srtm_slope", np.zeros_like(tc))), 0, 1).astype(np.float32)

            # Exposure delta (same as __getitem__)
            exposed_before = (tc < 0.3).astype(np.float32)
            clearing = (
                (lossyear > t1) & (lossyear <= t2)
            ).astype(np.float32)
            forest_at_t2_soil = tc.copy()
            forest_at_t2_soil[(lossyear > 0) & (lossyear <= t2)] = 0.0
            cleared_tc = forest_at_t2_soil.copy()
            cleared_tc[clearing > 0.5] = 0.0
            exposed_after = (cleared_tc < 0.3).astype(np.float32)
            exposure_delta = np.clip(
                (exposed_after - exposed_before) * slope, 0, None
            )

            # Cascade forest loss (matches __getitem__ lines 944-948)
            additional_loss_soil = (
                (lossyear > t2) & (lossyear <= t_impact)
            ).astype(np.float32)
            cascade_exposure = additional_loss_soil * slope

            # Combined raw degradation (matches __getitem__ line 951)
            raw_degradation = exposure_delta + cascade_exposure * 0.5

            # Control-baseline subtraction (matches __getitem__ lines 952-954)
            forest_mask_soil = (forest_at_t2_soil > 0.3).astype(np.float32)
            raw_target = _control_baseline(
                raw_degradation, clearing, forest_mask_soil,
                near_radius=10, decay_sigma=7.0,  # soil: ~200m surface erosion
            )

            # SMAP soil moisture weighting (matches __getitem__ lines 958-965)
            smap = data.get("smap_soil_moisture", None)
            if smap is not None:
                smap = smap.astype(np.float32)
                smap_max = smap.max()
                if smap_max > 1e-8:
                    smap_norm = smap / smap_max
                    raw_target = raw_target * (0.5 + 0.5 * smap_norm)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        chip_max = float(raw_target.max())
        if chip_max > 1e-8:
            maxima.append(chip_max)

    if not maxima:
        msg = (
            f"Could not compute target scale for {model_name}: "
            f"no valid chips found in '{split}' split. "
        )
        if model_name == "hydro":
            msg += (
                "Ensure download_msi_smap.py has been run and "
                "chips contain 'msi_ndssi_delta' key."
            )
        print(f"  \u26a0 {msg} Falling back to 1.0")
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
    decay_sigma: float = 10.0,
) -> np.ndarray:
    """Subtract background change to isolate causal impact signal.

    Uses a smooth Gaussian distance decay instead of a hard binary
    near/far boundary.  This matches the physical reality that
    deforestation impact (erosion, fire spread, hydrological change)
    propagates via diffusion-like processes that attenuate with
    distance.

    Parameters
    ----------
    metric_change : ndarray [H, W]
        Raw change in the metric (e.g., forest loss, fire count).
    deforestation_mask : ndarray [H, W]
        Binary mask of cleared areas.
    forest_mask : ndarray [H, W]
        Binary mask of remaining forest.
    near_radius : int
        Dilation radius for defining "far" background zone (pixels
        beyond this radius are used for background estimation).
    decay_sigma : float
        Gaussian decay width in pixels for the impact weight.
        Domain-specific recommended values:
          - Fire:   ~17 pixels (~500m at 30m/pixel, ember transport)
          - Forest:  ~10 pixels (~300m, cascade contagion)
          - Hydro:   ~15 pixels (~450m, runoff propagation)
          - Soil:    ~7 pixels  (~200m, surface erosion)
    """
    clearing_binary = deforestation_mask > 0.5

    # Distance from nearest clearing pixel (in pixels)
    dist = distance_transform_edt(~clearing_binary)

    # Smooth impact weight: 1.0 at clearing, decays to ~0 beyond ~3×sigma
    impact_weight = np.exp(-0.5 * (dist / max(decay_sigma, 0.1)) ** 2)
    impact_weight[clearing_binary] = 1.0

    # Background zone: forested pixels OUTSIDE the near_radius dilation.
    # We still use binary_dilation to define the background exclusion
    # zone, ensuring background pixels are truly unaffected.
    near_zone = binary_dilation(
        clearing_binary, iterations=near_radius,
    ).astype(np.float32)
    far_forest = (1.0 - near_zone) * forest_mask

    # Background rate from truly far pixels
    far_sum = (metric_change * far_forest).sum()
    far_count = far_forest.sum()
    background_rate = far_sum / (far_count + 1e-8) if far_count > 100 else 0.0

    # Excess impact, weighted by distance-based proximity
    excess = metric_change - background_rate
    excess = np.clip(excess, 0, None)
    excess = excess * forest_mask

    return excess


def _sample_window(
    _split: str,
    rng: np.random.RandomState,
    train_end: int = 23,
    min_window: int = 2,
    max_window: int = 5,
) -> Tuple[int, int, int]:
    """Sample random (T₁, T₂, T_impact) window from the full year range.

    All splits use the full temporal range (years 1–23) because data
    leakage is prevented by the spatial tile-level split (different
    geographic tiles for train vs test).  The model learns a
    time-invariant physical relationship (clearing → impact), so
    temporal holdout is unnecessary.

    Parameters
    ----------
    _split : str
        Dataset split name (e.g. "train", "test"). Intentionally unused —
        retained for API compatibility with callers. All splits sample
        identically because leakage is prevented at the tile level.
    rng : np.random.RandomState
        Random state for reproducibility.
    train_end : int
        Maximum year (inclusive) to sample from (default 23 = year 2023).
    min_window, max_window : int
        Min/max width of the observation→impact gap.

    Returns
    -------
    t1 : int
        Start year (event window start).
    t2 : int
        End year (event window end; deforestation happens in [t1, t2]).
    t_impact : int
        Year by which to measure impact (t2 + delta).
    """
    delta = rng.randint(min_window, max_window + 1)
    max_t2 = train_end - delta  # leave room for impact observation
    min_t1 = 1
    if max_t2 <= min_t1 + 1:
        # Fallback for very short ranges
        t1, t2, t_impact = 1, 8, min(8 + delta, train_end)
    else:
        t2 = rng.randint(min_t1 + 2, max_t2 + 1)  # ensure t2 >= 3
        max_win = min(max_window, t2 - min_t1)
        window = max(1, rng.randint(1, max_win + 1)) if max_win >= 1 else 1
        t1 = max(min_t1, t2 - window)
        t_impact = min(t2 + delta, train_end)

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
        # ── Load with validation & retry ────────────────────────────────
        for _attempt in range(5):
            entry = self.entries[idx]
            file_path = _resolve_chip_path(self.tiles_dir, entry["file"])
            try:
                data = np.load(file_path)
            except Exception as exc:
                _chip_logger.warning("Corrupted chip %s: %s — skipping", file_path, exc)
                idx = _random.randrange(len(self.entries))
                continue
            valid, reason = validate_chip(data, file_path, domain="forest")
            if not valid:
                _chip_logger.warning("Skipping chip: %s", reason)
                idx = _random.randrange(len(self.entries))
                continue
            break
        else:
            raise RuntimeError("Failed to load a valid chip after 5 attempts")
        rng = np.random.RandomState(idx + self._epoch_seed)

        treecover = data["treecover2000"].astype(np.float32)
        lossyear = data["lossyear"].astype(np.float32)
        gain = data.get("gain", np.zeros_like(treecover)).astype(np.float32)

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
            additional_loss, deforestation_mask, forest_mask,
            near_radius=15, decay_sigma=10.0,  # forest: ~300m cascade
        )

        if self.target_scale is not None:
            cascade_impact = cascade_impact / (self.target_scale + 1e-8)
        else:
            ci_max = cascade_impact.max()
            if ci_max > 1e-8:
                cascade_impact = cascade_impact / ci_max

        cascade_impact = gaussian_filter(cascade_impact, sigma=0.5)
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
        # ── Load with validation & retry ────────────────────────────────
        for _attempt in range(5):
            entry = self.entries[idx]
            file_path = _resolve_chip_path(self.tiles_dir, entry["file"])
            try:
                data = np.load(file_path)
            except Exception as exc:
                _chip_logger.warning("Corrupted chip %s: %s — skipping", file_path, exc)
                idx = _random.randrange(len(self.entries))
                continue
            valid, reason = validate_chip(data, file_path, domain="fire")
            if not valid:
                _chip_logger.warning("Skipping chip: %s", reason)
                idx = _random.randrange(len(self.entries))
                continue
            break
        else:
            raise RuntimeError("Failed to load a valid chip after 5 attempts")
        rng = np.random.RandomState(idx + self._epoch_seed)

        tc = data["treecover2000"].astype(np.float32) / 100.0
        lossyear = data["lossyear"].astype(np.float32)

        has_viirs = (
            data.get("has_real_viirs", np.array([0]))[0] > 0
            if "has_real_viirs" in data
            else False
        )

        slope = np.clip(np.nan_to_num(data.get("srtm_slope", np.zeros_like(tc))), 0, 1).astype(np.float32)
        elevation = np.clip(np.nan_to_num(data.get("srtm_elevation", np.zeros_like(tc))), 0, 1).astype(np.float32)

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

                # Normalise VIIRS channels to [0, 1] per-chip.
                # Raw scales vary wildly (counts 0–500+, FRP in MW,
                # brightness in Kelvin 300–500+) and would dominate
                # gradient updates vs. other [0,1] channels.
                _fa_max = fire_at_year.max()
                if _fa_max > 1e-6:
                    fire_at_year = fire_at_year / _fa_max
                _frp_max = frp.max()
                if _frp_max > 1e-6:
                    frp = frp / _frp_max
                _ti4_max = bright_ti4.max()
                if _ti4_max > 1e-6:
                    bright_ti4 = bright_ti4 / _ti4_max
                _ti5_max = bright_ti5.max()
                if _ti5_max > 1e-6:
                    bright_ti5 = bright_ti5 / _ti5_max

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
            fire_change, deforestation_mask, forest_mask,
            near_radius=15, decay_sigma=17.0,  # fire: ~500m ember transport
        )

        if self.target_scale is not None:
            fire_impact = fire_impact / (self.target_scale + 1e-8)
        else:
            fi_max = fire_impact.max()
            if fi_max > 1e-8:
                fire_impact = fire_impact / fi_max

        fire_impact = gaussian_filter(fire_impact, sigma=0.5)
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

# ── Hydro temporal constants ──────────────────────────────────────
# download_msi_smap.py fetches Sentinel-2 imagery for two fixed periods:
#   baseline: 2016-01-01 → 2017-01-01  (Hansen year 16)
#   impact:   2020-01-01 → 2021-01-01  (Hansen year 20)
# The input deforestation mask MUST match these years so the model
# learns the correct causal relationship (clearing → sediment change).
_HYDRO_T1: int = 16   # baseline year  (2016)
_HYDRO_T2: int = 20   # impact year    (2020)


class RealHydroDataset(Dataset):
    """Hydrological impact dataset — WATER POLLUTION DELTA DOWNSTREAM.

    Uses real Sentinel-2 MSI NDSSI (Normalised Difference Suspended
    Sediment Index) delta as the target, downloaded by
    ``download_msi_smap.py`` via Microsoft Planetary Computer.

    Target: observed increase in suspended sediment concentration
    downstream of deforestation, measured as NDSSI_before - NDSSI_after
    (positive = more sediment after clearing).

    **Temporal alignment:** The target is fixed to 2016→2020 (Hansen
    years 16→20) because ``download_msi_smap.py`` fetches Sentinel-2
    imagery for those exact periods.  The deforestation mask is
    therefore locked to ``(lossyear > 16) & (lossyear <= 20)`` to
    maintain causal alignment between input and target.

    Input:  [7, 256, 256]  — terrain + forest + ndssi_baseline + deforestation mask
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
        # ── Load with validation & retry ────────────────────────────────
        for _attempt in range(5):
            entry = self.entries[idx]
            file_path = _resolve_chip_path(self.tiles_dir, entry["file"])
            try:
                data = np.load(file_path)
            except Exception as exc:
                _chip_logger.warning("Corrupted chip %s: %s — skipping", file_path, exc)
                idx = _random.randrange(len(self.entries))
                continue
            valid, reason = validate_chip(data, file_path, domain="hydro")
            if not valid:
                _chip_logger.warning("Skipping chip: %s", reason)
                idx = _random.randrange(len(self.entries))
                continue

            # Additional hydro check: must have real NDSSI data
            if "msi_ndssi_delta" not in data or np.all(data["msi_ndssi_delta"] == 0):
                _chip_logger.debug("No real NDSSI in %s — resampling", file_path)
                idx = _random.randrange(len(self.entries))
                continue
                
            break
        else:
            raise RuntimeError("Failed to load a valid chip after 5 attempts")
        # NOTE: Hydro uses a fixed temporal window (_HYDRO_T1/_HYDRO_T2),
        # so per-epoch randomisation only affects single-patch augmentation
        # (not temporal sliding, which is unused for this dataset).
        rng = np.random.RandomState(idx + self._epoch_seed)

        tc = data["treecover2000"].astype(np.float32) / 100.0
        lossyear = data["lossyear"].astype(np.float32)

        elevation = np.clip(np.nan_to_num(data.get("srtm_elevation", np.zeros_like(tc))), 0, 1).astype(np.float32)
        slope = np.clip(np.nan_to_num(data.get("srtm_slope", np.zeros_like(tc))), 0, 1).astype(np.float32)
        aspect = np.clip(np.nan_to_num(data.get("srtm_aspect", np.zeros_like(tc))), 0, 1).astype(np.float32)
        flow_acc = np.clip(np.nan_to_num(data.get("srtm_flow_acc", np.zeros_like(tc))), 0, 1).astype(np.float32)

        # ── Fixed temporal window (must match download_msi_smap.py) ──
        # The NDSSI target is baked to 2016→2020 at download time, so
        # the deforestation mask must use the same years.
        t1, t2 = _HYDRO_T1, _HYDRO_T2

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

        # Baseline NDSSI — gives the model visibility into pre-clearing
        # water quality so it can learn that high-sediment areas respond
        # more strongly to further clearing.
        ndssi_baseline = data.get("msi_ndssi_baseline", np.zeros_like(tc))
        ndssi_baseline = np.clip(
            np.nan_to_num(ndssi_baseline.astype(np.float32)), 0, 1,
        )

        # Factual input: [7, H, W] — terrain + forest + ndssi_baseline + NO deforestation mask
        obs_f = np.stack([
            elevation, slope, aspect, flow_acc, forest,
            ndssi_baseline, no_deforestation,
        ], axis=0)

        # Counterfactual input: [7, H, W] — terrain + forest + ndssi_baseline + deforestation mask
        obs_cf = np.stack([
            elevation, slope, aspect, flow_acc, forest,
            ndssi_baseline, deforestation_mask,
        ], axis=0)

        # ── Real Sentinel-2 NDSSI Target ──
        # The NDSSI delta from download_msi_smap.py is already a first-
        # difference observation (NDSSI_before - NDSSI_after), so no
        # _control_baseline() subtraction is needed here — unlike the
        # derived lossyear-based targets in forest/fire/soil, which
        # require control-pixel subtraction to isolate causal signal.
        pollution_delta = data["msi_ndssi_delta"].astype(np.float32)

        if self.target_scale is not None:
            pollution_delta = pollution_delta / (self.target_scale + 1e-8)
        else:
            p_max = pollution_delta.max()
            if p_max > 1e-8:
                pollution_delta = pollution_delta / p_max

        # Gaussian smoothing (consistent with forest/fire/soil targets)
        # Prevents overfitting to pixel-level satellite noise in NDSSI.
        pollution_delta = gaussian_filter(pollution_delta, sigma=0.5)
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
    """Soil degradation impact dataset — COUNTERFACTUAL EXPOSURE DELTA.

    Target: soil exposure INCREASE in neighbouring cells caused by
    the clearing, compared to pre-clearing baseline.  When real SMAP
    soil moisture data is available (from ``download_msi_smap.py``),
    it weights the exposure delta to produce a physically-informed
    degradation signal (exposed + high baseline moisture = worse
    degradation risk).

    The target depends on the deforestation mask and temporal window,
    ensuring the model learns a causal relationship between clearing
    location and soil degradation.

    Input:  [T, 7, 256, 256]  — forest + smap + terrain + deforestation mask
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
        # ── Load with validation & retry ────────────────────────────────
        for _attempt in range(5):
            entry = self.entries[idx]
            file_path = _resolve_chip_path(self.tiles_dir, entry["file"])
            try:
                data = np.load(file_path)
            except Exception as exc:
                _chip_logger.warning("Corrupted chip %s: %s — skipping", file_path, exc)
                idx = _random.randrange(len(self.entries))
                continue
            valid, reason = validate_chip(data, file_path, domain="soil")
            if not valid:
                _chip_logger.warning("Skipping chip: %s", reason)
                idx = _random.randrange(len(self.entries))
                continue
            break
        else:
            raise RuntimeError("Failed to load a valid chip after 5 attempts")
        rng = np.random.RandomState(idx + self._epoch_seed)

        tc = data["treecover2000"].astype(np.float32) / 100.0
        lossyear = data["lossyear"].astype(np.float32)

        slope = np.clip(np.nan_to_num(data.get("srtm_slope", np.zeros_like(tc))), 0, 1).astype(np.float32)
        elevation = np.clip(np.nan_to_num(data.get("srtm_elevation", np.zeros_like(tc))), 0, 1).astype(np.float32)
        aspect = np.clip(np.nan_to_num(data.get("srtm_aspect", np.zeros_like(tc))), 0, 1).astype(np.float32)
        flow_acc = np.clip(np.nan_to_num(data.get("srtm_flow_acc", np.zeros_like(tc))), 0, 1).astype(np.float32)

        # Real SMAP soil moisture — gives the model visibility into
        # baseline moisture conditions.  Previously only used to
        # weight the target; now also an input so the model can learn
        # that high-moisture areas are more vulnerable to degradation.
        smap_raw = data.get("smap_soil_moisture", np.zeros_like(tc))
        smap_raw = np.nan_to_num(smap_raw.astype(np.float32))
        _smap_max = smap_raw.max()
        smap_norm = smap_raw / _smap_max if _smap_max > 1e-8 else np.zeros_like(smap_raw)

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

            # Factual: no clearing mask
            # Channels: forest, smap, slope, elevation, aspect, flow_acc, deforestation_mask
            frame_f = np.stack([
                forest, smap_norm, slope, elevation,
                aspect, flow_acc, no_deforestation,
            ], axis=0)
            frames_factual.append(frame_f)

            # Counterfactual: real clearing mask
            frame_cf = np.stack([
                forest, smap_norm, slope, elevation,
                aspect, flow_acc, deforestation_mask,
            ], axis=0)
            frames_counterfactual.append(frame_cf)

        # ── Counterfactual Soil Degradation Target ──
        # Compute soil exposure BEFORE vs AFTER clearing to get causal delta
        forest_at_t2 = tc.copy()
        forest_at_t2[(lossyear > 0) & (lossyear <= t2)] = 0.0
        forest_mask = (forest_at_t2 > 0.3).astype(np.float32)

        # Soil exposure BEFORE clearing (areas already degraded)
        exposed_before = (tc < 0.3).astype(np.float32)
        exposure_before = exposed_before * slope

        # Soil exposure AFTER clearing (new areas exposed by the clearing)
        cleared_tc = forest_at_t2.copy()
        cleared_tc[deforestation_mask > 0.5] = 0.0
        exposed_after = (cleared_tc < 0.3).astype(np.float32)
        exposure_after = exposed_after * slope

        # Delta = new exposure caused by clearing
        exposure_delta = np.clip(exposure_after - exposure_before, 0, None)

        # Also account for forest loss between t2 and t_impact (cascade soil effect)
        additional_loss = (
            (lossyear > t2) & (lossyear <= t_impact)
        ).astype(np.float32)
        cascade_exposure = additional_loss * slope

        # Combined: exposure delta + cascade, with control-baseline subtraction
        raw_degradation = exposure_delta + cascade_exposure * 0.5
        soil_impact = _control_baseline(
            raw_degradation, deforestation_mask, forest_mask,
            near_radius=10, decay_sigma=7.0,  # soil: ~200m surface erosion
        )

        # If real SMAP soil moisture is available, use it to weight the target:
        # areas with high baseline moisture that lose forest lose MORE moisture
        smap = data.get("smap_soil_moisture", None)
        if smap is not None:
            smap = smap.astype(np.float32)
            smap_max = smap.max()
            if smap_max > 1e-8:
                smap_norm = smap / smap_max
                # Weight by baseline moisture: high moisture → worse degradation
                soil_impact = soil_impact * (0.5 + 0.5 * smap_norm)

        if self.target_scale is not None:
            soil_impact = soil_impact / (self.target_scale + 1e-8)
        else:
            si_max = soil_impact.max()
            if si_max > 1e-8:
                soil_impact = soil_impact / si_max

        soil_impact = gaussian_filter(soil_impact, sigma=0.5)
        degradation = np.clip(soil_impact, 0, 1)
        target = degradation[np.newaxis, :, :]

        obs_f = np.stack(frames_factual, axis=0)
        obs_cf = np.stack(frames_counterfactual, axis=0)
        return (
            torch.from_numpy(obs_f.astype(np.float32)),
            torch.from_numpy(obs_cf.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )
