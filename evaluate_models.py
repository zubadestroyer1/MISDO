"""
MISDO — Post-Training Model Evaluation
==========================================
Comprehensive statistical analysis of trained models on the
held-out validation set (2021–2023, unseen spatial tiles).

Metrics computed per model:
    ── Pixel-level ──
    • MSE, MAE, RMSE
    • Soft Dice coefficient, Hard Dice, IoU (Jaccard)
    • Precision, Recall, F1 (at threshold 0.5)
    • AUROC (area under ROC curve)
    • Pearson correlation (spatial pattern fidelity)
    • Macro-averaged Dice (per-sample, then mean)

    ── Signal-region ──
    • MSE, MAE, Pearson restricted to impact zone (target > 0.01)

    ── Spatial ──
    • SSIM (structural similarity index, Wang et al. 2004)
    • Gradient-matching MSE (captures edge quality)

    ── Calibration ──
    • Expected calibration error (ECE)
    • Reliability curve data points

    ── Distribution ──
    • Prediction range, mean, std
    • Target range, mean, std
    • KS-test statistic (distribution similarity)

Reports saved to weights/evaluation_report.json

Usage:
    python evaluate_models.py
    python evaluate_models.py --model fire --tiles-dir datasets/real_tiles
    python evaluate_models.py --threshold 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from datasets.real_datasets import (
    RealFireDataset, RealHansenDataset, RealHydroDataset, RealSoilDataset,
    compute_global_target_scale,
)
from models.fire_model import FireRiskNet
from models.forest_model import ForestLossNet
from models.hydro_model import HydroRiskNet
from models.soil_model import SoilRiskNet


# ═══════════════════════════════════════════════════════════════════════════
# Test-Time Augmentation (copied from train_real_models.py for independence)
# ═══════════════════════════════════════════════════════════════════════════

def _tta_predict(
    model: nn.Module,
    obs_f: torch.Tensor,
    obs_cf: torch.Tensor,
    aspect_channel_idx: int | None = None,
) -> torch.Tensor:
    """Test-time augmentation: average predictions over 8 geometric transforms.

    For models with aspect channels (hydro), applies the same directional
    corrections as RandomFlipRotate to keep inputs physically consistent.

    Parameters
    ----------
    model : nn.Module
        Unwrapped model (not DP/DDP wrapped).
    obs_f, obs_cf : Tensor
        Factual and counterfactual observations.
    aspect_channel_idx : int | None
        Channel index of aspect for directional correction.

    Returns
    -------
    Tensor : averaged prediction over all 8 augmentations.
    """
    preds = []
    for k in range(4):  # 0°, 90°, 180°, 270° CCW rotations
        for do_hflip in (False, True):
            f = obs_f.clone()
            cf = obs_cf.clone()

            # Apply horizontal flip
            if do_hflip:
                f = torch.flip(f, [-1])
                cf = torch.flip(cf, [-1])
                if aspect_channel_idx is not None:
                    if f.ndim == 5:
                        f[:, :, aspect_channel_idx] = (
                            1.0 - f[:, :, aspect_channel_idx]) % 1.0
                        cf[:, :, aspect_channel_idx] = (
                            1.0 - cf[:, :, aspect_channel_idx]) % 1.0
                    else:
                        f[:, aspect_channel_idx] = (
                            1.0 - f[:, aspect_channel_idx]) % 1.0
                        cf[:, aspect_channel_idx] = (
                            1.0 - cf[:, aspect_channel_idx]) % 1.0

            # Apply rotation
            if k > 0:
                f = torch.rot90(f, k, [-2, -1])
                cf = torch.rot90(cf, k, [-2, -1])
                if aspect_channel_idx is not None:
                    shift = k * 0.25
                    if f.ndim == 5:
                        f[:, :, aspect_channel_idx] = (
                            f[:, :, aspect_channel_idx] + shift) % 1.0
                        cf[:, :, aspect_channel_idx] = (
                            cf[:, :, aspect_channel_idx] + shift) % 1.0
                    else:
                        f[:, aspect_channel_idx] = (
                            f[:, aspect_channel_idx] + shift) % 1.0
                        cf[:, aspect_channel_idx] = (
                            cf[:, aspect_channel_idx] + shift) % 1.0

            # Forward (no deep supervision needed for inference)
            p = model.forward_paired(f, cf)

            # Undo rotation on output
            if k > 0:
                p = torch.rot90(p, -k, [-2, -1])
            # Undo horizontal flip on output
            if do_hflip:
                p = torch.flip(p, [-1])

            preds.append(p)

    return torch.stack(preds).mean(0)


# ═══════════════════════════════════════════════════════════════════════════
# Metric Functions
# ═══════════════════════════════════════════════════════════════════════════

def _safe_div(a: float, b: float) -> float:
    return a / b if b > 1e-12 else 0.0


def compute_pixel_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all pixel-level metrics.

    Binary classification metrics (precision, recall, F1) and AUROC
    use a **fixed target threshold of 0.5** to define positives,
    regardless of the prediction threshold.  This ensures AUROC values
    are comparable across different ``--threshold`` settings and that
    the "positive class" has a stable, semantically meaningful definition.

    Dice and IoU are reported as both:
      - ``soft_dice`` / ``soft_iou``: continuous (no binarisation)
      - ``hard_dice``: binarised at the specified threshold
    """
    pred_flat = pred.ravel()
    tgt_flat = target.ravel()

    # Regression metrics
    mse = float(np.mean((pred_flat - tgt_flat) ** 2))
    mae = float(np.mean(np.abs(pred_flat - tgt_flat)))
    rmse = float(np.sqrt(mse))

    # Pearson correlation
    if pred_flat.std() > 1e-8 and tgt_flat.std() > 1e-8:
        pearson = float(np.corrcoef(pred_flat, tgt_flat)[0, 1])
    else:
        pearson = 0.0

    # [S-4 fix] AUROC always uses 0.5-binarised targets so it is
    # comparable across different --threshold settings.
    auroc_labels = (tgt_flat > 0.5).astype(np.float32)

    # Binary classification metrics at the user-specified threshold
    pred_bin = (pred_flat > threshold).astype(np.float32)
    tgt_bin = (tgt_flat > 0.5).astype(np.float32)  # fixed reference

    tp = float((pred_bin * tgt_bin).sum())
    fp = float((pred_bin * (1 - tgt_bin)).sum())
    fn = float(((1 - pred_bin) * tgt_bin).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    # [S-2 fix] Soft Dice and IoU (continuous predictions, no binarisation)
    smooth = 1.0
    intersection = float((pred_flat * tgt_flat).sum())
    soft_dice = (2.0 * intersection + smooth) / (
        pred_flat.sum() + tgt_flat.sum() + smooth
    )
    soft_iou = (intersection + smooth) / (
        pred_flat.sum() + tgt_flat.sum() - intersection + smooth
    )

    # Hard Dice at the specified threshold (matches training's Dice sweep)
    hard_inter = float((pred_bin * tgt_bin).sum())
    hard_dice = (2.0 * hard_inter + smooth) / (
        pred_bin.sum() + tgt_bin.sum() + smooth
    )

    # [C-1 fix] AUROC with proper trapezoidal integration
    auroc = _compute_auroc(pred_flat, auroc_labels)

    return {
        "mse": round(mse, 6),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "pearson": round(pearson, 4),
        "soft_dice": round(soft_dice, 4),
        "hard_dice": round(hard_dice, 4),
        "soft_iou": round(soft_iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4),
    }


def _find_optimal_threshold(
    pred: np.ndarray,
    target: np.ndarray,
    thresholds: List[float] | None = None,
    target_threshold: float = 0.5,
) -> Tuple[float, float]:
    """Find the prediction threshold that maximises F1 score.

    [C-2 fix] The target is binarised at a FIXED reference threshold
    (default 0.5), and only the prediction threshold is swept.  This
    produces a true "operating point" — the prediction threshold that
    best separates model outputs into the fixed positive/negative classes.

    Parameters
    ----------
    target_threshold : float
        Fixed threshold for binarising targets (default 0.5).

    Returns (best_threshold, best_f1).
    """
    if thresholds is None:
        thresholds = [i * 0.05 for i in range(1, 20)]  # 0.05 to 0.95

    pred_flat = pred.ravel()
    tgt_flat = target.ravel()
    # Fixed target binarisation — defines what "positive" means
    tgt_bin = (tgt_flat > target_threshold).astype(np.float32)
    best_f1 = 0.0
    best_th = 0.5

    for th in thresholds:
        pred_bin = (pred_flat > th).astype(np.float32)
        tp = float((pred_bin * tgt_bin).sum())
        fp = float((pred_bin * (1 - tgt_bin)).sum())
        fn = float(((1 - pred_bin) * tgt_bin).sum())
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    return best_th, best_f1


def _compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC without sklearn using the trapezoidal rule.

    [C-1 fix] Uses proper trapezoidal integration:
      area += (fpr - prev_fpr) * (prev_tpr + tpr) / 2
    instead of the right-rectangle approximation that underestimates AUC.

    Handles edge cases:
      - All-positive or all-negative → returns 0.5 (undefined)
      - Single sample → returns 0.5
    """
    n_pos = float(labels.sum())
    n_neg = float(len(labels) - n_pos)

    if n_pos == 0 or n_neg == 0:
        return 0.5  # undefined — all one class

    # Sort by score descending
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    # Walk through sorted samples, accumulating TPR/FPR
    tp = 0.0
    fp = 0.0
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0

    for label in sorted_labels:
        if label > 0.5:
            tp += 1
        else:
            fp += 1
        fpr = fp / n_neg
        tpr = tp / n_pos
        # Trapezoidal rule: area of trapezoid between prev and current point
        auc += (fpr - prev_fpr) * (prev_tpr + tpr) / 2.0
        prev_fpr = fpr
        prev_tpr = tpr

    return float(auc)


def compute_signal_region_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    signal_threshold: float = 0.01,
) -> Dict[str, float]:
    """[R-2] Metrics restricted to the impact zone (target > signal_threshold).

    For sparse impact prediction where 95%+ of pixels are near-zero,
    global metrics are dominated by the trivial background.  This function
    isolates model quality on the pixels that actually matter — near-clearing
    forested areas with measurable change signal.

    Standard practice in medical imaging ("lesion-level Dice", BraTS / Menze
    et al. 2015) and remote sensing change detection (LEVIR-CD, Chen & Shi 2020).
    """
    tgt_flat = target.ravel()
    pred_flat = pred.ravel()

    # Signal region: pixels where the target has any measurable impact
    signal_mask = tgt_flat > signal_threshold
    n_signal = int(signal_mask.sum())

    if n_signal < 10:
        return {
            "signal_n_pixels": n_signal,
            "signal_fraction": round(float(n_signal / max(len(tgt_flat), 1)), 6),
            "signal_mse": None,
            "signal_mae": None,
            "signal_pearson": None,
        }

    pred_sig = pred_flat[signal_mask]
    tgt_sig = tgt_flat[signal_mask]

    mse = float(np.mean((pred_sig - tgt_sig) ** 2))
    mae = float(np.mean(np.abs(pred_sig - tgt_sig)))

    if pred_sig.std() > 1e-8 and tgt_sig.std() > 1e-8:
        pearson = float(np.corrcoef(pred_sig, tgt_sig)[0, 1])
    else:
        pearson = 0.0

    return {
        "signal_n_pixels": n_signal,
        "signal_fraction": round(float(n_signal / len(tgt_flat)), 6),
        "signal_mse": round(mse, 6),
        "signal_mae": round(mae, 6),
        "signal_pearson": round(pearson, 4),
    }


def compute_spatial_metrics(
    pred: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:
    """Compute spatial structure metrics (SSIM + gradient matching).

    [S-1 fix] SSIM constants follow Wang et al. (2004):
      c1 = (k1 * L)^2,  c2 = (k2 * L)^2
    where k1=0.01, k2=0.03, L=data_range.

    NOTE: This is a simplified *global* SSIM (whole-image statistics),
    not the sliding-window local SSIM used in losses.py's SSIMLoss.
    Both are valid — global SSIM is standard for small patches (~256px).
    """
    # SSIM constants (Wang et al. 2004) — data_range = 1.0 for [0,1] predictions
    data_range = 1.0
    c1 = (0.01 * data_range) ** 2  # = 1e-4
    c2 = (0.03 * data_range) ** 2  # = 9e-4

    # Structural Similarity Index (simplified, per-image)
    mu_p = pred.mean()
    mu_t = target.mean()
    sigma_p = pred.std()
    sigma_t = target.std()
    sigma_pt = float(((pred - mu_p) * (target - mu_t)).mean())

    ssim = float(
        (2 * mu_p * mu_t + c1) * (2 * sigma_pt + c2)
        / ((mu_p**2 + mu_t**2 + c1) * (sigma_p**2 + sigma_t**2 + c2))
    )

    # Gradient-matching MSE
    pred_dx = pred[:, 1:] - pred[:, :-1]
    pred_dy = pred[1:, :] - pred[:-1, :]
    tgt_dx = target[:, 1:] - target[:, :-1]
    tgt_dy = target[1:, :] - target[:-1, :]
    grad_mse = float(
        np.mean((pred_dx - tgt_dx) ** 2) + np.mean((pred_dy - tgt_dy) ** 2)
    )

    return {
        "ssim": round(ssim, 4),
        "gradient_mse": round(grad_mse, 6),
    }


def compute_calibration(
    pred: np.ndarray,
    target: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Expected calibration error and reliability curve."""
    pred_flat = pred.ravel()
    tgt_flat = target.ravel()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    reliability = []

    for i in range(n_bins):
        mask = (pred_flat >= bin_edges[i]) & (pred_flat < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_pred = pred_flat[mask].mean()
        bin_true = tgt_flat[mask].mean()
        bin_weight = mask.sum() / len(pred_flat)
        ece += abs(bin_pred - bin_true) * bin_weight
        reliability.append({
            "bin_center": round((bin_edges[i] + bin_edges[i + 1]) / 2, 2),
            "predicted": round(float(bin_pred), 4),
            "observed": round(float(bin_true), 4),
            "count": int(mask.sum()),
        })

    return {
        "ece": round(float(ece), 4),
        "reliability_curve": reliability,
    }


def compute_distribution_stats(
    pred: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:
    """Distribution statistics and KS test.

    [S-3 fix] When there are >1000 unique values, we subsample the actual
    data points (preserving the true CDF shape) instead of replacing them
    with np.linspace(0, 1, 1000) which assumed data range [0, 1] and
    missed sharp CDF jumps in sparse regions.
    """
    pred_flat = pred.ravel()
    tgt_flat = target.ravel()

    # Kolmogorov-Smirnov statistic (manual, no scipy needed)
    all_vals = np.sort(np.unique(np.concatenate([pred_flat, tgt_flat])))
    if len(all_vals) > 1000:
        # Subsample actual data points to preserve CDF shape
        subsample_idx = np.linspace(0, len(all_vals) - 1, 1000, dtype=int)
        all_vals = all_vals[subsample_idx]

    max_diff = 0.0
    for v in all_vals:
        cdf_p = (pred_flat <= v).mean()
        cdf_t = (tgt_flat <= v).mean()
        max_diff = max(max_diff, abs(cdf_p - cdf_t))

    return {
        "pred_mean": round(float(pred_flat.mean()), 4),
        "pred_std": round(float(pred_flat.std()), 4),
        "pred_min": round(float(pred_flat.min()), 4),
        "pred_max": round(float(pred_flat.max()), 4),
        "target_mean": round(float(tgt_flat.mean()), 4),
        "target_std": round(float(tgt_flat.std()), 4),
        "target_min": round(float(tgt_flat.min()), 4),
        "target_max": round(float(tgt_flat.max()), 4),
        "ks_statistic": round(float(max_diff), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Model Evaluation
# ═══════════════════════════════════════════════════════════════════════════

# [R-1] Added aspect_channel_idx for TTA parity with training.
MODEL_CONFIGS = {
    "fire": {
        "model_cls": FireRiskNet,
        "dataset_cls": RealFireDataset,
        "temporal": True,
        "aspect_channel_idx": None,
    },
    "forest": {
        "model_cls": ForestLossNet,
        "dataset_cls": RealHansenDataset,
        "temporal": True,
        "aspect_channel_idx": None,
    },
    "hydro": {
        "model_cls": HydroRiskNet,
        "dataset_cls": RealHydroDataset,
        "temporal": False,
        "aspect_channel_idx": 2,   # channel 2 = SRTM aspect
    },
    "soil": {
        "model_cls": SoilRiskNet,
        "dataset_cls": RealSoilDataset,
        "temporal": True,
        "aspect_channel_idx": 4,   # channel 4 = SRTM aspect (matches training)
    },
}


def evaluate_model(
    name: str,
    config: dict,
    tiles_dir: str,
    weights_dir: str,
    device: torch.device,
    threshold: float = 0.5,
    max_samples: int = 500,
    use_tta: bool = True,
) -> Dict:
    """Evaluate a single trained model on the validation set.

    Parameters
    ----------
    use_tta : bool
        If True (default), use 8× test-time augmentation to match
        training's final validation (train_real_models.py:1176).
        This produces metrics comparable to training's reported values.
    """
    print(f"\n{'─' * 50}")
    print(f"  Evaluating: {name}")
    print(f"{'─' * 50}")

    # Load model
    model = config["model_cls"]().to(device)
    weight_path = os.path.join(weights_dir, f"{name}_model.pt")
    if not os.path.exists(weight_path):
        print(f"  ✗ No weights found at {weight_path}")
        return {"error": "no weights"}

    state = torch.load(weight_path, map_location=device, weights_only=True)
    # Defence-in-depth: strip "module." prefix from state dict keys if
    # the checkpoint was saved from a DataParallel / DDP-wrapped model.
    # New training code (post-audit) saves base_model.state_dict() directly,
    # but this handles older checkpoints gracefully.
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    # Load validation dataset (held-out spatial tiles, full year range)
    # MUST replicate train_real_models.py's 50/50 test-tile split:
    #   First half  = checkpoint selection (seen during training as test set)
    #   Second half = held-out validation (never used during training)
    # We evaluate ONLY on the held-out second half.
    print(f"  Computing global target scale for {name}...")
    target_scale = compute_global_target_scale(name, tiles_dir, split="train")

    DatasetCls = config["dataset_cls"]
    if config["temporal"]:
        full_test_ds = DatasetCls(tiles_dir=tiles_dir, split="test", T=5, train_end_year=23,
                            target_scale=target_scale)
    else:
        full_test_ds = DatasetCls(tiles_dir=tiles_dir, split="test", train_end_year=23,
                            target_scale=target_scale)

    # Replicate training's 50/50 split: evaluate only the held-out second half
    n_test_total = len(full_test_ds)
    n_test_half = n_test_total // 2
    val_indices = list(range(n_test_half, n_test_total))
    val_ds = torch.utils.data.Subset(full_test_ds, val_indices)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    tta_label = "8× TTA" if use_tta else "single-pass"
    print(f"  Validation samples: {len(val_ds)} (held-out second half of {n_test_total} test tiles)")
    print(f"  Threshold: {threshold}  Inference: {tta_label}")

    # [M-3 fix] Auto-detect dataset format (2-tuple vs 3-tuple)
    sample = full_test_ds[0]
    siamese = len(sample) == 3

    aspect_ch = config.get("aspect_channel_idx")

    # Collect predictions and targets
    all_preds = []
    all_targets = []
    n_samples = min(max_samples, len(val_ds))

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_samples:
                break

            if siamese:
                obs_f, obs_cf, target = batch
                obs_f = obs_f.to(device)
                obs_cf = obs_cf.to(device)
                # [R-1] Use TTA to match training's final validation
                if use_tta:
                    pred = _tta_predict(model, obs_f, obs_cf,
                                        aspect_channel_idx=aspect_ch)
                else:
                    pred = model.forward_paired(obs_f, obs_cf)
            else:
                obs, target = batch
                obs = obs.to(device)
                target = target
                pred = model(obs)

            # Match spatial dimensions
            if pred.shape[2:] != target.shape[2:]:
                target = F.interpolate(
                    target, size=pred.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            if pred.shape[1] != target.shape[1]:
                target = target[:, :pred.shape[1]]

            all_preds.append(pred.cpu().numpy().squeeze())
            all_targets.append(target.cpu().numpy().squeeze())

    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)

    print(f"  Evaluated {len(all_preds)} samples")
    print(f"  Pred shape: {all_preds_arr.shape}, "
          f"range [{all_preds_arr.min():.4f}, {all_preds_arr.max():.4f}]")
    print(f"  Target shape: {all_targets_arr.shape}, "
          f"range [{all_targets_arr.min():.4f}, {all_targets_arr.max():.4f}]")

    # ── Aggregate pixel metrics across all samples (micro-averaged) ──
    pixel_metrics = compute_pixel_metrics(all_preds_arr, all_targets_arr, threshold)

    # [R-3] Macro-averaged Dice (per-sample, then mean)
    per_sample_dice = []
    for p, t in zip(all_preds, all_targets):
        p_flat, t_flat = p.ravel(), t.ravel()
        inter = (p_flat * t_flat).sum()
        sample_dice = (2.0 * inter + 1.0) / (p_flat.sum() + t_flat.sum() + 1.0)
        per_sample_dice.append(float(sample_dice))
    pixel_metrics["soft_dice_macro"] = round(float(np.mean(per_sample_dice)), 4)
    pixel_metrics["soft_dice_macro_std"] = round(float(np.std(per_sample_dice)), 4)

    # ── Optimal threshold search ──
    opt_th, opt_f1 = _find_optimal_threshold(all_preds_arr, all_targets_arr)
    opt_metrics = compute_pixel_metrics(all_preds_arr, all_targets_arr, opt_th)
    pixel_metrics["optimal_threshold"] = round(opt_th, 3)
    pixel_metrics["f1_at_optimal"] = round(opt_f1, 4)
    pixel_metrics["precision_at_optimal"] = opt_metrics["precision"]
    pixel_metrics["recall_at_optimal"] = opt_metrics["recall"]

    print(f"\n  Pixel Metrics (threshold={threshold}):")
    print(f"    MSE={pixel_metrics['mse']:.6f}  MAE={pixel_metrics['mae']:.6f}  "
          f"RMSE={pixel_metrics['rmse']:.6f}")
    print(f"    Pearson={pixel_metrics['pearson']:.4f}  "
          f"SoftDice={pixel_metrics['soft_dice']:.4f}  "
          f"HardDice={pixel_metrics['hard_dice']:.4f}  "
          f"IoU={pixel_metrics['soft_iou']:.4f}")
    print(f"    SoftDice(macro)={pixel_metrics['soft_dice_macro']:.4f} "
          f"± {pixel_metrics['soft_dice_macro_std']:.4f}")
    print(f"    P={pixel_metrics['precision']:.4f}  R={pixel_metrics['recall']:.4f}  "
          f"F1={pixel_metrics['f1']:.4f}  AUROC={pixel_metrics['auroc']:.4f}")
    print(f"  Optimal Threshold: {opt_th:.3f}")
    print(f"    P={opt_metrics['precision']:.4f}  R={opt_metrics['recall']:.4f}  "
          f"F1={opt_f1:.4f}")

    # ── Signal-region metrics (R-2) ──
    signal_metrics = compute_signal_region_metrics(all_preds_arr, all_targets_arr)
    print(f"\n  Signal-Region Metrics (target > 0.01):")
    print(f"    Signal pixels: {signal_metrics['signal_n_pixels']:,} "
          f"({signal_metrics['signal_fraction'] * 100:.2f}% of total)")
    if signal_metrics["signal_mse"] is not None:
        print(f"    MSE={signal_metrics['signal_mse']:.6f}  "
              f"MAE={signal_metrics['signal_mae']:.6f}  "
              f"Pearson={signal_metrics['signal_pearson']:.4f}")
    else:
        print(f"    (too few signal pixels for reliable metrics)")

    # ── Per-sample spatial metrics (average across samples) ──
    ssim_scores = []
    grad_mse_scores = []
    for p, t in zip(all_preds, all_targets):
        spatial = compute_spatial_metrics(p, t)
        ssim_scores.append(spatial["ssim"])
        grad_mse_scores.append(spatial["gradient_mse"])

    spatial_metrics = {
        "ssim_mean": round(float(np.mean(ssim_scores)), 4),
        "ssim_std": round(float(np.std(ssim_scores)), 4),
        "gradient_mse_mean": round(float(np.mean(grad_mse_scores)), 6),
    }
    print(f"\n  Spatial Metrics:")
    print(f"    SSIM={spatial_metrics['ssim_mean']:.4f} ± {spatial_metrics['ssim_std']:.4f}")
    print(f"    Gradient MSE={spatial_metrics['gradient_mse_mean']:.6f}")

    # ── Calibration ──
    calibration = compute_calibration(all_preds_arr, all_targets_arr)
    print(f"\n  Calibration:")
    print(f"    ECE={calibration['ece']:.4f}")

    # ── Distribution ──
    distribution = compute_distribution_stats(all_preds_arr, all_targets_arr)
    print(f"\n  Distribution:")
    print(f"    Pred:   mean={distribution['pred_mean']:.4f}  "
          f"std={distribution['pred_std']:.4f}  "
          f"range=[{distribution['pred_min']:.4f}, {distribution['pred_max']:.4f}]")
    print(f"    Target: mean={distribution['target_mean']:.4f}  "
          f"std={distribution['target_std']:.4f}  "
          f"range=[{distribution['target_min']:.4f}, {distribution['target_max']:.4f}]")
    print(f"    KS statistic={distribution['ks_statistic']:.4f}")

    # ── Quality Assessment ──
    # [R-5] Compute activation fractions for quality grading
    pred_active = float((all_preds_arr.ravel() > 0.05).mean())
    tgt_active = float((all_targets_arr.ravel() > 0.05).mean())
    quality = _assess_quality(
        pixel_metrics, spatial_metrics, distribution, name,
        pred_active_frac=pred_active, tgt_active_frac=tgt_active,
        signal_metrics=signal_metrics,
    )
    print(f"\n  Quality Assessment: {quality['grade']}")
    for note in quality["notes"]:
        print(f"    • {note}")

    return {
        "model": name,
        "params": params,
        "n_samples": len(all_preds),
        "threshold": threshold,
        "use_tta": use_tta,
        "pixel_metrics": pixel_metrics,
        "signal_metrics": signal_metrics,
        "spatial_metrics": spatial_metrics,
        "calibration": calibration,
        "distribution": distribution,
        "quality": quality,
    }


def _assess_quality(
    pixel: Dict, spatial: Dict, dist: Dict, model_name: str,
    pred_active_frac: float = 0.0,
    tgt_active_frac: float = 0.0,
    signal_metrics: Dict | None = None,
) -> Dict:
    """Auto-assess model quality based on metric thresholds.

    [R-5] Enhanced with activation-fraction comparison and signal-region
    Pearson to catch failure modes the original grading missed:
      - "diffuse haze" (model predicts low values everywhere)
      - "dead neurons" (model activates on far fewer pixels than target)
      - Weak causal signal (good global MSE but poor signal-region Pearson)
    """
    notes = []
    score = 0  # 0-12 (expanded from 0-10)

    # Check for mode collapse
    if dist["pred_std"] < 0.01:
        notes.append("⚠ Mode collapse — predictions near-constant")
    elif dist["pred_std"] < 0.05:
        notes.append("⚠ Low variance — predictions lack spatial diversity")
        score += 1
    else:
        score += 2

    # Pearson correlation
    if pixel["pearson"] > 0.5:
        notes.append("✓ Strong spatial correlation with target")
        score += 3
    elif pixel["pearson"] > 0.2:
        notes.append("~ Moderate spatial correlation")
        score += 2
    else:
        notes.append("✗ Weak spatial correlation")

    # Dice / IoU (uses soft_dice for consistency with previous grading)
    if pixel["soft_dice"] > 0.3:
        notes.append("✓ Reasonable segmentation overlap")
        score += 2
    elif pixel["soft_dice"] > 0.1:
        notes.append("~ Low segmentation overlap")
        score += 1
    else:
        notes.append("✗ Very low segmentation overlap")

    # SSIM
    if spatial["ssim_mean"] > 0.5:
        notes.append("✓ Good structural similarity")
        score += 2
    elif spatial["ssim_mean"] > 0.2:
        notes.append("~ Moderate structural similarity")
        score += 1

    # KS test
    if dist["ks_statistic"] < 0.1:
        notes.append("✓ Prediction distribution matches target well")
        score += 1
    elif dist["ks_statistic"] > 0.3:
        notes.append("⚠ Prediction distribution diverges from target")

    # [R-5] Activation fraction comparison
    if tgt_active_frac > 1e-6:
        activation_ratio = pred_active_frac / max(tgt_active_frac, 1e-8)
        if activation_ratio < 0.1:
            notes.append("⚠ Under-activation — model predicts far fewer signal pixels than target")
        elif activation_ratio > 10.0:
            notes.append("⚠ Over-activation — model predicts signal over too much of the image")
        else:
            notes.append("✓ Activation fraction within 10× of target")
            score += 1

    # [R-5] Signal-region Pearson (is the model learning the causal signal?)
    if signal_metrics is not None and signal_metrics.get("signal_pearson") is not None:
        sp = signal_metrics["signal_pearson"]
        if sp > 0.3:
            notes.append("✓ Signal-region correlation indicates causal learning")
            score += 1
        elif sp > 0.1:
            notes.append("~ Weak signal-region correlation")
        else:
            notes.append("✗ No signal-region correlation — model may not capture impact")

    # Grade (expanded scale: max score now 12)
    if score >= 10:
        grade = "A — Excellent"
    elif score >= 7:
        grade = "B — Good"
    elif score >= 5:
        grade = "C — Acceptable"
    elif score >= 3:
        grade = "D — Needs improvement"
    else:
        grade = "F — Model is not working"

    return {"grade": grade, "score": score, "notes": notes}


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MISDO — Post-training model evaluation"
    )
    parser.add_argument(
        "--model", default="all",
        choices=["all", "fire", "forest", "hydro", "soil"],
    )
    parser.add_argument("--tiles-dir", default="datasets/real_tiles")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Binary classification threshold (default: 0.5)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=500,
        help="Max validation samples to evaluate (default: 500)",
    )
    parser.add_argument(
        "--no-tta", action="store_true",
        help="Disable test-time augmentation (faster but less accurate)",
    )
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print("  MISDO — Post-Training Model Evaluation")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  TTA: {'disabled' if args.no_tta else 'enabled (8× geometric)'}")

    models_to_eval = (
        list(MODEL_CONFIGS.keys()) if args.model == "all"
        else [args.model]
    )

    all_results = {}
    t0 = time.time()

    for name in models_to_eval:
        config = MODEL_CONFIGS[name]
        results = evaluate_model(
            name, config, args.tiles_dir, args.weights_dir,
            device, args.threshold, args.max_samples,
            use_tta=not args.no_tta,
        )
        all_results[name] = results

    elapsed = time.time() - t0

    # ── Summary Table ──
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n  {'Model':<10} {'Grade':<16} {'MSE':<10} {'SoftDice':<10} "
          f"{'Pearson':<10} {'AUROC':<8} {'SSIM':<8}")
    print(f"  {'─' * 72}")

    for name, r in all_results.items():
        if "error" in r:
            print(f"  {name:<10} {'ERROR':<16}")
            continue
        p = r["pixel_metrics"]
        s = r["spatial_metrics"]
        q = r["quality"]
        print(f"  {name:<10} {q['grade']:<16} {p['mse']:<10.6f} "
              f"{p['soft_dice']:<10.4f} {p['pearson']:<10.4f} "
              f"{p['auroc']:<8.4f} {s['ssim_mean']:<8.4f}")

    print(f"\n  Total evaluation time: {elapsed:.1f}s")

    # Save report
    report_path = os.path.join(args.weights_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Report saved: {report_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
