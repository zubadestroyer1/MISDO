"""
MISDO — Train All Domain-Specific Models
==========================================
Self-supervised training for 4 dataset-specific risk models using
physically-realistic synthetic data.

Usage:
    python train_models.py --model all --epochs 30
    python train_models.py --model fire --epochs 50

Models are saved to weights/ directory.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from datasets.viirs_fire import VIIRSFireDataset
from datasets.hansen_gfc import HansenGFCDataset
from datasets.srtm_hydro import SRTMHydroDataset
from datasets.smap_soil import SMAPSoilDataset
from models.fire_model import FireRiskNet
from models.forest_model import ForestLossNet
from models.hydro_model import HydroRiskNet
from models.soil_model import SoilRiskNet


# ═══════════════════════════════════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════════════════════════════════

class FocalBCELoss(nn.Module):
    """Binary cross-entropy with focal weighting for class imbalance.

    Used for fire detection where positive pixels are sparse.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for segmentation with fragmented patches."""

    def __init__(self, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.dice_weight = dice_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # BCE component
        bce = F.binary_cross_entropy(pred, target)

        # Dice component
        smooth = 1.0
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        intersection = (pred_flat * target_flat).sum()
        dice = 1 - (2 * intersection + smooth) / (
            pred_flat.sum() + target_flat.sum() + smooth
        )

        return bce * (1 - self.dice_weight) + dice * self.dice_weight


class GradientMSELoss(nn.Module):
    """MSE + gradient-matching regulariser for smooth risk surfaces.

    Preserves spatial gradients in the prediction, important for
    continuous risk fields like water pollution and soil degradation.
    """

    def __init__(self, grad_weight: float = 0.3) -> None:
        super().__init__()
        self.grad_weight = grad_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        mse = F.mse_loss(pred, target)

        # Spatial gradients (Sobel-like)
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        grad_loss = F.mse_loss(pred_dx, tgt_dx) + F.mse_loss(pred_dy, tgt_dy)

        return mse + self.grad_weight * grad_loss


class SmoothMSELoss(nn.Module):
    """MSE + spatial smoothness regulariser for continuous risk fields."""

    def __init__(self, smooth_weight: float = 0.2) -> None:
        super().__init__()
        self.smooth_weight = smooth_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        mse = F.mse_loss(pred, target)

        # Total variation smoothness
        tv_h = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs().mean()
        tv_w = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean()
        smoothness = tv_h + tv_w

        return mse + self.smooth_weight * smoothness


# ═══════════════════════════════════════════════════════════════════════════
# Model configurations
# ═══════════════════════════════════════════════════════════════════════════

MODEL_CONFIGS: Dict[str, Dict] = {
    "fire": {
        "model_class": FireRiskNet,
        "dataset_class": VIIRSFireDataset,
        "loss_class": FocalBCELoss,
        "lr": 1e-3,
        "description": "VIIRS Active Fire Detection",
    },
    "forest": {
        "model_class": ForestLossNet,
        "dataset_class": HansenGFCDataset,
        "loss_class": DiceBCELoss,
        "lr": 1e-3,
        "description": "Hansen Forest Loss Detection",
    },
    "hydro": {
        "model_class": HydroRiskNet,
        "dataset_class": SRTMHydroDataset,
        "loss_class": GradientMSELoss,
        "lr": 8e-4,
        "description": "SRTM/HydroSHEDS Water-Pollution Risk",
    },
    "soil": {
        "model_class": SoilRiskNet,
        "dataset_class": SMAPSoilDataset,
        "loss_class": SmoothMSELoss,
        "lr": 1e-3,
        "description": "SMAP Soil Degradation Risk",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_single_model(
    name: str,
    epochs: int = 30,
    num_samples: int = 64,
    batch_size: int = 4,
    device: torch.device = torch.device("cpu"),
    save_dir: str = "weights",
) -> Dict[str, float]:
    """Train one domain-specific model and save weights."""

    cfg = MODEL_CONFIGS[name]
    print(f"\n{'='*70}")
    print(f"  Training: {cfg['description']} ({name})")
    print(f"{'='*70}")

    # Dataset and loader
    dataset = cfg["dataset_class"](
        num_samples=num_samples, spatial_size=256, seed=42
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Model
    model = cfg["model_class"]().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Loss, optimizer, scheduler
    criterion = cfg["loss_class"]()
    optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Training
    best_loss = float("inf")
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for obs, target in loader:
            obs = obs.to(device)
            target = target.to(device)

            pred = model(obs)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.6f}  "
                  f"best={best_loss:.6f}  lr={lr:.2e}")

    # Save weights
    os.makedirs(save_dir, exist_ok=True)
    weight_path = os.path.join(save_dir, f"{name}_model.pt")
    torch.save(model.state_dict(), weight_path)
    print(f"  ✓ Saved weights to {weight_path}")

    # Validation pass
    model.eval()
    val_dataset = cfg["dataset_class"](
        num_samples=8, spatial_size=256, seed=999
    )
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    val_loss = 0.0
    pred_stats = {"min": 1.0, "max": 0.0, "mean": 0.0}
    n_val = 0

    with torch.no_grad():
        for obs, target in val_loader:
            obs = obs.to(device)
            target = target.to(device)
            pred = model(obs)
            val_loss += criterion(pred, target).item()
            pred_stats["min"] = min(pred_stats["min"], pred.min().item())
            pred_stats["max"] = max(pred_stats["max"], pred.max().item())
            pred_stats["mean"] += pred.mean().item()
            n_val += 1

    pred_stats["mean"] /= max(n_val, 1)
    val_loss /= max(n_val, 1)

    print(f"  Validation loss: {val_loss:.6f}")
    print(f"  Pred range: [{pred_stats['min']:.4f}, {pred_stats['max']:.4f}]  "
          f"mean={pred_stats['mean']:.4f}")

    return {
        "train_loss_final": history[-1],
        "val_loss": val_loss,
        "best_train_loss": best_loss,
        "params": total_params,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Train MISDO domain models")
    parser.add_argument("--model", type=str, default="all",
                        choices=["fire", "forest", "hydro", "soil", "all"],
                        help="Which model to train (default: all)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--samples", type=int, default=64,
                        help="Number of training samples (default: 64)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default: 4)")
    args = parser.parse_args()

    # Device selection: MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    models_to_train = (
        list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    )

    results: Dict[str, Dict] = {}
    t0 = time.time()

    for name in models_to_train:
        results[name] = train_single_model(
            name=name,
            epochs=args.epochs,
            num_samples=args.samples,
            batch_size=args.batch_size,
            device=device,
        )

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*70}")
    print(f"  TRAINING SUMMARY  (total time: {elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"{'Model':<12} {'Params':>10} {'Train Loss':>12} {'Val Loss':>12}")
    print(f"{'-'*46}")
    for name, r in results.items():
        print(f"{name:<12} {r['params']:>10,} {r['train_loss_final']:>12.6f} "
              f"{r['val_loss']:>12.6f}")
    print()


if __name__ == "__main__":
    main()
