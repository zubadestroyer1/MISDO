"""
MISDO — Post-Training Model Evaluation
==========================================
Comprehensive statistical analysis of trained models on the
held-out validation set (2021–2023, unseen spatial tiles).

Metrics computed per model:
    ── Pixel-level ──
    • MSE, MAE, RMSE
    • Dice coefficient, IoU (Jaccard)
    • Precision, Recall, F1 (at threshold 0.5)
    • AUROC (area under ROC curve)
    • Pearson correlation (spatial pattern fidelity)

    ── Spatial ──
    • SSIM (structural similarity index)
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
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from datasets.real_datasets import (
    RealFireDataset, RealHansenDataset, RealHydroDataset, RealSoilDataset,
)
from models.fire_model import FireRiskNet
from models.forest_model import ForestLossNet
from models.hydro_model import HydroRiskNet
from models.soil_model import SoilRiskNet


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
    """Compute all pixel-level metrics."""
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

    # Binary classification metrics at threshold
    pred_bin = (pred_flat > threshold).astype(np.float32)
    tgt_bin = (tgt_flat > threshold).astype(np.float32)

    tp = float((pred_bin * tgt_bin).sum())
    fp = float((pred_bin * (1 - tgt_bin)).sum())
    fn = float(((1 - pred_bin) * tgt_bin).sum())
    tn = float(((1 - pred_bin) * (1 - tgt_bin)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    # Dice and IoU (soft, using continuous predictions)
    smooth = 1.0
    intersection = float((pred_flat * tgt_flat).sum())
    dice = (2.0 * intersection + smooth) / (
        pred_flat.sum() + tgt_flat.sum() + smooth
    )
    iou = (intersection + smooth) / (
        pred_flat.sum() + tgt_flat.sum() - intersection + smooth
    )

    # AUROC (manual computation, no sklearn dependency)
    auroc = _compute_auroc(pred_flat, tgt_bin)

    return {
        "mse": round(mse, 6),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "pearson": round(pearson, 4),
        "dice": round(dice, 4),
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4),
    }


def _compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC without sklearn using the trapezoidal rule."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5  # undefined — all one class

    # Sort by score descending
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    # Accumulate TPR and FPR
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    tp = 0.0
    fp = 0.0
    auc = 0.0
    prev_fpr = 0.0

    for label in sorted_labels:
        if label > 0.5:
            tp += 1
        else:
            fp += 1
            fpr = fp / n_neg
            tpr = tp / n_pos
            auc += (fpr - prev_fpr) * tpr
            prev_fpr = fpr

    return float(auc)


def compute_spatial_metrics(
    pred: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:
    """Compute spatial structure metrics (SSIM + gradient matching)."""
    # Structural Similarity Index (simplified, per-image)
    mu_p = pred.mean()
    mu_t = target.mean()
    sigma_p = pred.std()
    sigma_t = target.std()
    sigma_pt = float(((pred - mu_p) * (target - mu_t)).mean())

    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
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
    """Distribution statistics and KS test."""
    pred_flat = pred.ravel()
    tgt_flat = target.ravel()

    # Kolmogorov-Smirnov statistic (manual, no scipy needed)
    all_vals = np.sort(np.unique(np.concatenate([pred_flat, tgt_flat])))
    if len(all_vals) > 1000:
        all_vals = np.linspace(0, 1, 1000)
    
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

MODEL_CONFIGS = {
    "fire": {
        "model_cls": FireRiskNet,
        "dataset_cls": RealFireDataset,
        "temporal": True,
    },
    "forest": {
        "model_cls": ForestLossNet,
        "dataset_cls": RealHansenDataset,
        "temporal": True,
    },
    "hydro": {
        "model_cls": HydroRiskNet,
        "dataset_cls": RealHydroDataset,
        "temporal": False,
    },
    "soil": {
        "model_cls": SoilRiskNet,
        "dataset_cls": RealSoilDataset,
        "temporal": True,
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
) -> Dict:
    """Evaluate a single trained model on the validation set."""
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
    model.load_state_dict(state, strict=False)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    # Load validation dataset (2021-2023, held-out spatial tiles)
    DatasetCls = config["dataset_cls"]
    if config["temporal"]:
        val_ds = DatasetCls(tiles_dir=tiles_dir, split="test", T=5, train_end_year=20)
    else:
        val_ds = DatasetCls(tiles_dir=tiles_dir, split="test", train_end_year=20)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    print(f"  Validation samples: {len(val_ds)}")
    print(f"  Threshold: {threshold}")

    # Collect predictions and targets
    all_preds = []
    all_targets = []
    n_samples = min(max_samples, len(val_ds))

    with torch.no_grad():
        for i, (obs, target) in enumerate(val_loader):
            if i >= n_samples:
                break
            obs = obs.to(device)
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

    # ── Aggregate pixel metrics across all samples ──
    pixel_metrics = compute_pixel_metrics(all_preds_arr, all_targets_arr, threshold)
    print(f"\n  Pixel Metrics:")
    print(f"    MSE={pixel_metrics['mse']:.6f}  MAE={pixel_metrics['mae']:.6f}  "
          f"RMSE={pixel_metrics['rmse']:.6f}")
    print(f"    Pearson={pixel_metrics['pearson']:.4f}  "
          f"Dice={pixel_metrics['dice']:.4f}  IoU={pixel_metrics['iou']:.4f}")
    print(f"    P={pixel_metrics['precision']:.4f}  R={pixel_metrics['recall']:.4f}  "
          f"F1={pixel_metrics['f1']:.4f}  AUROC={pixel_metrics['auroc']:.4f}")

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
    quality = _assess_quality(pixel_metrics, spatial_metrics, distribution, name)
    print(f"\n  Quality Assessment: {quality['grade']}")
    for note in quality["notes"]:
        print(f"    • {note}")

    return {
        "model": name,
        "params": params,
        "n_samples": len(all_preds),
        "threshold": threshold,
        "pixel_metrics": pixel_metrics,
        "spatial_metrics": spatial_metrics,
        "calibration": calibration,
        "distribution": distribution,
        "quality": quality,
    }


def _assess_quality(
    pixel: Dict, spatial: Dict, dist: Dict, model_name: str,
) -> Dict:
    """Auto-assess model quality based on metric thresholds."""
    notes = []
    score = 0  # 0-10

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

    # Dice / IoU
    if pixel["dice"] > 0.3:
        notes.append("✓ Reasonable segmentation overlap")
        score += 2
    elif pixel["dice"] > 0.1:
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

    # Grade
    if score >= 8:
        grade = "A — Excellent"
    elif score >= 6:
        grade = "B — Good"
    elif score >= 4:
        grade = "C — Acceptable"
    elif score >= 2:
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
        )
        all_results[name] = results

    elapsed = time.time() - t0

    # ── Summary Table ──
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n  {'Model':<10} {'Grade':<16} {'MSE':<10} {'Dice':<8} "
          f"{'Pearson':<10} {'AUROC':<8} {'SSIM':<8}")
    print(f"  {'─' * 70}")

    for name, r in all_results.items():
        if "error" in r:
            print(f"  {name:<10} {'ERROR':<16}")
            continue
        p = r["pixel_metrics"]
        s = r["spatial_metrics"]
        q = r["quality"]
        print(f"  {name:<10} {q['grade']:<16} {p['mse']:<10.6f} "
              f"{p['dice']:<8.4f} {p['pearson']:<10.4f} "
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
