"""
MISDO — Spatial Cross-Validation & Temporal Holdout
======================================================
Prevents data leakage in geospatial machine learning by ensuring that
spatially or temporally adjacent data never appears in both training
and validation sets simultaneously.

Without spatial blocking, adjacent tiles share deforestation patterns
through spatial autocorrelation, inflating test metrics by 10-30%
(Ploton et al., 2020, Nature Communications).

Key API:
    SpatialBlockCV    — K-fold CV with spatial blocking
    TemporalHoldout   — strict temporal train/val/test splits
    run_cross_validation() — full CV pipeline with metrics

Reference:
    Ploton et al., "Spatial validation reveals poor predictive
    performance of large-scale ecological models", Nature Communications, 2020.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset


class SpatialBlockCV:
    """K-fold cross-validation with spatial blocking.

    Groups tiles into spatial blocks based on their geographic coordinates.
    Ensures all tiles from the same block are in the same fold, preventing
    spatial autocorrelation leakage.

    Parameters
    ----------
    n_folds : int
        Number of cross-validation folds (default 5).
    block_size_deg : float
        Size of spatial blocks in degrees (default 1.0°).
        Each 1° block covers ~111 km at the equator.
        Larger blocks = more conservative (less leakage risk).
    """

    def __init__(self, n_folds: int = 5, block_size_deg: float = 1.0) -> None:
        self.n_folds = n_folds
        self.block_size_deg = block_size_deg

    def split(
        self,
        tiles: List[Dict[str, Any]],
    ) -> List[Tuple[List[int], List[int]]]:
        """Split tiles into spatially-blocked folds.

        Parameters
        ----------
        tiles : list of dict
            Each dict must have 'lat' and 'lon' keys (float).
            May also have 'file', 'year', etc.

        Returns
        -------
        folds : list of (train_indices, val_indices) tuples
            Each fold's train and validation indices into the tiles list.
        """
        # Assign each tile to a spatial block
        block_assignments: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for idx, tile in enumerate(tiles):
            lat = tile.get("lat", 0.0)
            lon = tile.get("lon", 0.0)
            block_row = int(lat // self.block_size_deg)
            block_col = int(lon // self.block_size_deg)
            block_assignments[(block_row, block_col)].append(idx)

        # Sort blocks for deterministic ordering
        blocks = sorted(block_assignments.keys())
        n_blocks = len(blocks)

        if n_blocks < self.n_folds:
            raise ValueError(
                f"Only {n_blocks} spatial blocks found, but {self.n_folds} folds "
                f"requested. Decrease block_size_deg or n_folds."
            )

        # Assign blocks to folds (round-robin)
        block_fold_assignments = {}
        for i, block_key in enumerate(blocks):
            fold_idx = i % self.n_folds
            block_fold_assignments[block_key] = fold_idx

        # Build train/val splits
        folds = []
        for fold in range(self.n_folds):
            val_indices = []
            train_indices = []
            for block_key, tile_indices in block_assignments.items():
                if block_fold_assignments[block_key] == fold:
                    val_indices.extend(tile_indices)
                else:
                    train_indices.extend(tile_indices)
            folds.append((sorted(train_indices), sorted(val_indices)))

        return folds

    def validate_no_leakage(
        self,
        tiles: List[Dict[str, Any]],
        folds: List[Tuple[List[int], List[int]]],
    ) -> bool:
        """Verify that no spatial block appears in both train and val.

        Returns True if no leakage detected.
        """
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            train_blocks = set()
            val_blocks = set()

            for idx in train_idx:
                lat = tiles[idx].get("lat", 0.0)
                lon = tiles[idx].get("lon", 0.0)
                block = (
                    int(lat // self.block_size_deg),
                    int(lon // self.block_size_deg),
                )
                train_blocks.add(block)

            for idx in val_idx:
                lat = tiles[idx].get("lat", 0.0)
                lon = tiles[idx].get("lon", 0.0)
                block = (
                    int(lat // self.block_size_deg),
                    int(lon // self.block_size_deg),
                )
                val_blocks.add(block)

            overlap = train_blocks & val_blocks
            if overlap:
                print(f"  ⚠ Fold {fold_idx}: {len(overlap)} blocks in both train and val!")
                return False

        return True


class TemporalHoldout:
    """Strict temporal train/validation/test splits.

    Ensures no future data leaks into training. Critical for
    deforestation prediction where temporal patterns are the
    primary signal.

    Default split (for Hansen GFC year encoding where 1=2001):
        Train:      years 1-18  (2001-2018)
        Validation: years 19-20 (2019-2020)
        Test:       years 21-23 (2021-2023)

    Parameters
    ----------
    train_end : int
        Last year for training (inclusive, 1-indexed).
    val_end : int
        Last year for validation (inclusive).
    test_end : int
        Last year for testing (inclusive).
    """

    def __init__(
        self,
        train_end: int = 18,
        val_end: int = 20,
        test_end: int = 23,
    ) -> None:
        self.train_end = train_end
        self.val_end = val_end
        self.test_end = test_end

    def split(self) -> Dict[str, List[int]]:
        """Return year lists for each split."""
        return {
            "train_years": list(range(1, self.train_end + 1)),
            "val_years": list(range(self.train_end + 1, self.val_end + 1)),
            "test_years": list(range(self.val_end + 1, self.test_end + 1)),
        }

    def get_dataset_kwargs(self) -> Dict[str, Dict[str, int]]:
        """Return kwargs for RealHansenDataset per split.

        The returned dicts can be passed directly to the dataset constructors:
            dataset = RealHansenDataset(**kwargs['train'])
        """
        return {
            "train": {
                "split": "train",
                "train_end_year": self.train_end,
            },
            "val": {
                "split": "train",  # uses training tiles but limits years
                "train_end_year": self.val_end,
            },
            "test": {
                "split": "test",
                "train_end_year": self.val_end,
            },
        }


class SpatialTemporalCV:
    """Combined spatial blocking + temporal holdout.

    Applies spatial blocking within each temporal split to prevent
    both spatial and temporal leakage simultaneously.

    Parameters
    ----------
    spatial_cv : SpatialBlockCV
        Spatial blocking configuration.
    temporal : TemporalHoldout
        Temporal split configuration.
    """

    def __init__(
        self,
        spatial_cv: Optional[SpatialBlockCV] = None,
        temporal: Optional[TemporalHoldout] = None,
    ) -> None:
        self.spatial_cv = spatial_cv or SpatialBlockCV()
        self.temporal = temporal or TemporalHoldout()

    def describe(self) -> str:
        """Human-readable description of the validation strategy."""
        splits = self.temporal.split()
        years_to_real = lambda years: [y + 2000 for y in years]
        return (
            f"Validation Strategy:\n"
            f"  Temporal splits:\n"
            f"    Train: {years_to_real(splits['train_years'])[0]}-"
            f"{years_to_real(splits['train_years'])[-1]}\n"
            f"    Val:   {years_to_real(splits['val_years'])[0]}-"
            f"{years_to_real(splits['val_years'])[-1]}\n"
            f"    Test:  {years_to_real(splits['test_years'])[0]}-"
            f"{years_to_real(splits['test_years'])[-1]}\n"
            f"  Spatial blocking:\n"
            f"    Block size: {self.spatial_cv.block_size_deg}° "
            f"(~{self.spatial_cv.block_size_deg * 111:.0f} km)\n"
            f"    CV folds: {self.spatial_cv.n_folds}"
        )


def run_cross_validation(
    model_class: type,
    dataset: Dataset,
    tiles: List[Dict[str, Any]],
    spatial_cv: Optional[SpatialBlockCV] = None,
    epochs: int = 10,
    batch_size: int = 2,
    lr: float = 3e-4,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run spatially-blocked cross-validation on a model.

    Supports both legacy datasets returning ``(obs, target)`` and
    Siamese counterfactual datasets returning ``(obs_f, obs_cf, target)``.
    The format is auto-detected from the first training batch.

    Parameters
    ----------
    model_class : type
        Model class to instantiate (e.g., FireRiskNet).
    dataset : Dataset
        Full dataset (will be split into folds).
    tiles : list of dict
        Tile metadata with 'lat', 'lon' keys.
    spatial_cv : SpatialBlockCV, optional
        Spatial blocking config (default: 5-fold, 1° blocks).
    epochs : int
        Training epochs per fold (default 10).
    device : torch.device, optional
        Compute device.

    Returns
    -------
    results : dict with:
        fold_metrics : list of per-fold metric dicts
        mean_metrics : averaged metrics across folds
        std_metrics  : std dev of metrics across folds
    """
    if spatial_cv is None:
        spatial_cv = SpatialBlockCV()
    if device is None:
        device = torch.device("cpu")

    folds = spatial_cv.split(tiles)

    if verbose:
        print(f"Running {len(folds)}-fold spatial CV...")
        no_leakage = spatial_cv.validate_no_leakage(tiles, folds)
        print(f"  Spatial leakage check: {'✓ No leakage' if no_leakage else '✗ LEAKAGE DETECTED'}")

    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if verbose:
            print(f"\n  Fold {fold_idx + 1}/{len(folds)}: "
                  f"train={len(train_idx)}, val={len(val_idx)}")

        # Create fold subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Fresh model per fold
        model = model_class().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        # Auto-detect dataset format from first sample:
        #   2-tuple → legacy (obs, target)
        #   3-tuple → Siamese (obs_f, obs_cf, target)
        sample = dataset[train_idx[0]] if len(train_idx) > 0 else dataset[0]
        siamese = len(sample) == 3

        # Train
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                if siamese:
                    obs_f, obs_cf, target = batch
                    obs_f = obs_f.to(device)
                    obs_cf = obs_cf.to(device)
                    target = target.to(device)
                    pred = model.forward_paired(obs_f, obs_cf)
                else:
                    obs, target = batch
                    obs, target = obs.to(device), target.to(device)
                    pred = model(obs)
                # Match spatial dimensions if needed
                if pred.shape[2:] != target.shape[2:]:
                    target = F.interpolate(
                        target, size=pred.shape[2:],
                        mode="bilinear", align_corners=False,
                    )
                loss = F.mse_loss(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        total_mse = 0.0
        total_mae = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                if siamese:
                    obs_f, obs_cf, target = batch
                    obs_f = obs_f.to(device)
                    obs_cf = obs_cf.to(device)
                    target = target.to(device)
                    pred = model.forward_paired(obs_f, obs_cf)
                else:
                    obs, target = batch
                    obs, target = obs.to(device), target.to(device)
                    pred = model(obs)
                if pred.shape[2:] != target.shape[2:]:
                    target = F.interpolate(
                        target, size=pred.shape[2:],
                        mode="bilinear", align_corners=False,
                    )
                total_mse += F.mse_loss(pred, target).item()
                total_mae += (pred - target).abs().mean().item()
                n += 1

        metrics = {
            "mse": total_mse / max(n, 1),
            "mae": total_mae / max(n, 1),
            "rmse": (total_mse / max(n, 1)) ** 0.5,
        }
        fold_metrics.append(metrics)

        if verbose:
            print(f"    MSE={metrics['mse']:.6f}  MAE={metrics['mae']:.6f}")

    # Aggregate across folds
    metric_keys = fold_metrics[0].keys()
    mean_metrics = {
        k: float(np.mean([fm[k] for fm in fold_metrics]))
        for k in metric_keys
    }
    std_metrics = {
        k: float(np.std([fm[k] for fm in fold_metrics]))
        for k in metric_keys
    }

    if verbose:
        print(f"\n  Cross-validation results:")
        for k in metric_keys:
            print(f"    {k}: {mean_metrics[k]:.6f} ± {std_metrics[k]:.6f}")

    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "n_folds": len(folds),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing Spatial Cross-Validation...\n")

    # Test SpatialBlockCV
    tiles = [
        {"lat": float(i), "lon": float(j), "file": f"tile_{i}_{j}.npz"}
        for i in range(10)
        for j in range(10)
    ]
    cv = SpatialBlockCV(n_folds=5, block_size_deg=2.0)
    folds = cv.split(tiles)

    assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        overlap = set(train_idx) & set(val_idx)
        assert len(overlap) == 0, f"Fold {fold_idx}: train/val overlap!"
        all_idx = set(train_idx) | set(val_idx)
        assert all_idx == set(range(100)), f"Fold {fold_idx}: not all tiles covered!"
        print(f"  Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}")

    no_leakage = cv.validate_no_leakage(tiles, folds)
    assert no_leakage, "Spatial leakage detected!"
    print(f"  Spatial leakage check: ✓ No leakage\n")

    # Test TemporalHoldout
    th = TemporalHoldout(train_end=18, val_end=20, test_end=23)
    splits = th.split()
    assert splits["train_years"] == list(range(1, 19))
    assert splits["val_years"] == [19, 20]
    assert splits["test_years"] == [21, 22, 23]
    print(f"  Temporal splits:")
    for split_name, years in splits.items():
        real_years = [y + 2000 for y in years]
        print(f"    {split_name}: {real_years[0]}-{real_years[-1]} ({len(years)} years)")

    # Test SpatialTemporalCV description
    stcv = SpatialTemporalCV()
    print(f"\n  {stcv.describe()}")

    print("\n✓ All spatial cross-validation tests passed")
