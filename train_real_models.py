"""
MISDO — Real Data Training Pipeline
======================================
Trains all 4 domain-specific models on real Hansen GFC satellite data.

Train split: years 2001–2020
Test split:  years 2021–2023

Usage:
    python train_real_models.py --model all --epochs 30
    python train_real_models.py --model fire --epochs 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from datasets.real_datasets import (
    RealFireDataset,
    RealHansenDataset,
    RealHydroDataset,
    RealSoilDataset,
)
from models.fire_model import FireRiskNet
from models.forest_model import ForestLossNet
from models.hydro_model import HydroRiskNet
from models.soil_model import SoilRiskNet


# ═══════════════════════════════════════════════════════════════════
# Loss Functions
# ═══════════════════════════════════════════════════════════════════

class FocalBCE(nn.Module):
    """Focal Binary Cross-Entropy for sparse fire/loss events."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = target * pred + (1 - target) * (1 - pred)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class DiceBCE(nn.Module):
    """Dice + BCE combined loss for segmentation."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target)
        intersection = (pred * target).sum()
        dice = 1.0 - (2 * intersection + 1) / (pred.sum() + target.sum() + 1)
        return bce + dice


class GradientMSE(nn.Module):
    """MSE + gradient loss for spatially smooth predictions."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        grad = F.mse_loss(pred_dx, target_dx) + F.mse_loss(pred_dy, target_dy)
        return mse + 0.5 * grad


class SmoothMSE(nn.Module):
    """MSE with spatial smoothness penalty."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        smooth = F.mse_loss(pred[:, :, 1:, :], pred[:, :, :-1, :]) + \
                 F.mse_loss(pred[:, :, :, 1:], pred[:, :, :, :-1])
        return mse + 0.1 * smooth


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "fire": {
        "model_cls": FireRiskNet,
        "dataset_cls": RealFireDataset,
        "loss_cls": FocalBCE,
        "temporal": True,
        "lr": 3e-4,
    },
    "forest": {
        "model_cls": ForestLossNet,
        "dataset_cls": RealHansenDataset,
        "loss_cls": DiceBCE,
        "temporal": True,
        "lr": 3e-4,
    },
    "hydro": {
        "model_cls": HydroRiskNet,
        "dataset_cls": RealHydroDataset,
        "loss_cls": GradientMSE,
        "temporal": False,
        "lr": 1e-3,
    },
    "soil": {
        "model_cls": SoilRiskNet,
        "dataset_cls": RealSoilDataset,
        "loss_cls": SmoothMSE,
        "temporal": True,
        "lr": 1e-3,
    },
}


def train_single_model(
    name: str,
    config: dict,
    epochs: int,
    tiles_dir: str,
    weights_dir: str,
    device: torch.device,
) -> dict:
    """Train a single model on real data and return metrics."""
    print(f"\n{'=' * 50}")
    print(f"  Training {name}")
    print(f"{'=' * 50}")

    # Dataset
    DatasetCls = config["dataset_cls"]
    if config["temporal"]:
        train_ds = DatasetCls(tiles_dir=tiles_dir, split="train", T=5)
        test_ds = DatasetCls(tiles_dir=tiles_dir, split="test", T=5)
    else:
        train_ds = DatasetCls(tiles_dir=tiles_dir, split="train")
        test_ds = DatasetCls(tiles_dir=tiles_dir, split="test")

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

    # Model
    model = config["model_cls"]().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    print(f"  Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = config["loss_cls"]()

    # Training loop
    train_losses = []
    test_losses = []
    best_test_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for obs, target in train_loader:
            obs = obs.to(device)
            target = target.to(device)

            pred = model(obs)
            # Ensure shapes match
            if pred.shape != target.shape:
                target = target[:, :1]

            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # Test
        model.eval()
        test_loss = 0.0
        n_test = 0
        with torch.no_grad():
            for obs, target in test_loader:
                obs = obs.to(device)
                target = target.to(device)
                pred = model(obs)
                if pred.shape != target.shape:
                    target = target[:, :1]
                test_loss += criterion(pred, target).item()
                n_test += 1

        avg_test = test_loss / max(n_test, 1)
        test_losses.append(avg_test)

        if avg_test < best_test_loss:
            best_test_loss = avg_test
            # Save best model
            weight_path = os.path.join(weights_dir, f"{name}_model.pt")
            torch.save(model.state_dict(), weight_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: train_loss={avg_train:.6f}  "
                  f"test_loss={avg_test:.6f}  best={best_test_loss:.6f}")

    # Final test metrics
    model.eval()
    total_mse = 0.0
    total_dice = 0.0
    n_samples = 0
    with torch.no_grad():
        for obs, target in test_loader:
            obs = obs.to(device)
            target = target.to(device)
            pred = model(obs)
            if pred.shape != target.shape:
                target = target[:, :1]

            total_mse += F.mse_loss(pred, target).item()
            p, t = pred.flatten(), target.flatten()
            intersection = (p * t).sum()
            dice = (2 * intersection + 1) / (p.sum() + t.sum() + 1)
            total_dice += dice.item()
            n_samples += 1

    metrics = {
        "model": name,
        "params": params,
        "epochs": epochs,
        "final_train_loss": round(train_losses[-1], 6),
        "best_test_loss": round(best_test_loss, 6),
        "test_mse": round(total_mse / max(n_samples, 1), 6),
        "test_dice": round(total_dice / max(n_samples, 1), 4),
        "train_losses": [round(l, 6) for l in train_losses],
        "test_losses": [round(l, 6) for l in test_losses],
    }

    print(f"\n  Final: MSE={metrics['test_mse']:.6f}  Dice={metrics['test_dice']:.4f}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MISDO models on real data")
    parser.add_argument("--model", default="all",
                        choices=["all", "fire", "forest", "hydro", "soil"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--tiles-dir", default="datasets/real_tiles")
    parser.add_argument("--weights-dir", default="weights")
    args = parser.parse_args()

    os.makedirs(args.weights_dir, exist_ok=True)

    # Check data exists
    manifest_path = os.path.join(args.tiles_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print("ERROR: No real data found. Run: python datasets/download_real_data.py")
        sys.exit(1)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Select models
    models_to_train = (
        list(MODEL_CONFIGS.keys()) if args.model == "all"
        else [args.model]
    )

    all_metrics = {}
    t0 = time.time()

    for name in models_to_train:
        config = MODEL_CONFIGS[name]
        metrics = train_single_model(
            name, config, args.epochs, args.tiles_dir, args.weights_dir, device
        )
        all_metrics[name] = metrics

    # Save metrics
    metrics_path = os.path.join(args.weights_dir, "real_training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'=' * 50}")
    print(f"  All training complete in {elapsed:.1f}s")
    print(f"  Metrics saved to {metrics_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
