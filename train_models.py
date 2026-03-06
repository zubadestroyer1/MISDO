"""
MISDO — Train All Domain-Specific Models
==========================================
Training pipeline for 4 dataset-specific risk models using
physically-realistic synthetic data.

Features:
    ✓ Consolidated production-grade loss functions (NaN-safe)
    ✓ UNet++ deep supervision auxiliary losses
    ✓ Data augmentation (random flips, rotations, brightness)
    ✓ Proper train/validation split (80/20 by seed)
    ✓ Best checkpoint saving (val loss, not last epoch)
    ✓ Early stopping (patience=10 epochs)
    ✓ Learning rate warmup + cosine annealing
    ✓ MPS-safe DataLoader and device detection

Usage:
    python train_models.py --model all --epochs 30
    python train_models.py --model fire --epochs 50

Models are saved to weights/ directory.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from datasets.viirs_fire import VIIRSFireDataset
from datasets.hansen_gfc import HansenGFCDataset
from datasets.srtm_hydro import SRTMHydroDataset
from datasets.smap_soil import SMAPSoilDataset
from losses import (
    FocalBCELoss,
    DiceBCELoss,
    GradientMSELoss,
    SmoothMSELoss,
    DeepSupervisionWrapper,
)
from models.fire_model import FireRiskNet
from models.forest_model import ForestLossNet
from models.hydro_model import HydroRiskNet
from models.soil_model import SoilRiskNet


# ═══════════════════════════════════════════════════════════════════════════
# Data augmentation
# ═══════════════════════════════════════════════════════════════════════════

class RandomAugmentation:
    """On-the-fly spatial augmentation for observation-target pairs.

    Applies random horizontal/vertical flips and 90° rotations
    consistently to both input and target.
    """

    def __call__(
        self, obs: Tensor, target: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            obs = obs.flip(-1)
            target = target.flip(-1)

        # Random vertical flip
        if torch.rand(1).item() > 0.5:
            obs = obs.flip(-2)
            target = target.flip(-2)

        # Random 90° rotation (0, 90, 180, or 270 degrees)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            obs = torch.rot90(obs, k, dims=(-2, -1))
            target = torch.rot90(target, k, dims=(-2, -1))

        # Random brightness/contrast perturbation (input only)
        if torch.rand(1).item() > 0.5:
            brightness = torch.rand(1).item() * 0.1 - 0.05  # [-0.05, 0.05]
            obs = (obs + brightness).clamp(0, 1)

        return obs, target


class AugmentedDataset(Dataset):
    """Wraps an existing dataset with on-the-fly augmentation."""

    def __init__(self, base_dataset: Dataset, augment: bool = True) -> None:
        self.base = base_dataset
        self.augment = augment
        self.aug = RandomAugmentation()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        obs, target = self.base[idx]
        if self.augment:
            obs, target = self.aug(obs, target)
        return obs, target


# ═══════════════════════════════════════════════════════════════════════════
# Device utilities
# ═══════════════════════════════════════════════════════════════════════════

def _get_device() -> torch.device:
    """Select the best available device: MPS > CUDA > CPU."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_dataloader_kwargs(device: torch.device) -> dict:
    """MPS/CUDA-safe DataLoader settings."""
    if device.type == "cuda":
        return {"num_workers": min(4, os.cpu_count() or 1), "pin_memory": True}
    return {"num_workers": 0, "pin_memory": False}


# ═══════════════════════════════════════════════════════════════════════════
# LR Scheduler with Warmup
# ═══════════════════════════════════════════════════════════════════════════

class WarmupCosineScheduler:
    """Linear warmup for first N epochs, then cosine annealing to zero."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int) -> None:
        """Update learning rate (1-indexed epoch)."""
        if epoch <= self.warmup_epochs:
            scale = epoch / max(self.warmup_epochs, 1)
        else:
            progress = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale

    def get_last_lr(self) -> list:
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ═══════════════════════════════════════════════════════════════════════════
# Model configurations
# ═══════════════════════════════════════════════════════════════════════════

MODEL_CONFIGS: Dict[str, Dict] = {
    "fire": {
        "model_class": FireRiskNet,
        "dataset_class": VIIRSFireDataset,
        "loss_class": FocalBCELoss,
        "lr": 5e-4,
        "description": "VIIRS Active Fire Detection",
    },
    "forest": {
        "model_class": ForestLossNet,
        "dataset_class": HansenGFCDataset,
        "loss_class": DiceBCELoss,
        "lr": 5e-4,
        "description": "Hansen Forest Loss Detection",
    },
    "hydro": {
        "model_class": HydroRiskNet,
        "dataset_class": SRTMHydroDataset,
        "loss_class": GradientMSELoss,
        "lr": 3e-4,
        "description": "SRTM/HydroSHEDS Water-Pollution Risk",
    },
    "soil": {
        "model_class": SoilRiskNet,
        "dataset_class": SMAPSoilDataset,
        "loss_class": SmoothMSELoss,
        "lr": 5e-4,
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
    patience: int = 10,
) -> Dict[str, float]:
    """Train one domain-specific model and save best weights."""

    cfg = MODEL_CONFIGS[name]
    print(f"\n{'='*70}")
    print(f"  Training: {cfg['description']} ({name})")
    print(f"{'='*70}")

    # ── Train / Validation split (80/20 by seed offset) ──
    train_samples = int(num_samples * 0.8)
    val_samples = num_samples - train_samples

    train_dataset = AugmentedDataset(
        cfg["dataset_class"](
            num_samples=train_samples, spatial_size=256, seed=42
        ),
        augment=True,
    )
    val_dataset = AugmentedDataset(
        cfg["dataset_class"](
            num_samples=val_samples, spatial_size=256, seed=10000
        ),
        augment=False,
    )

    dl_kwargs = _get_dataloader_kwargs(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **dl_kwargs)

    print(f"  Train samples: {train_samples}  Val samples: {val_samples}")

    # Model
    model = cfg["model_class"]().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Loss with deep supervision wrapper, optimizer, scheduler
    base_criterion = cfg["loss_class"]()
    criterion = DeepSupervisionWrapper(base_criterion, aux_weight=0.3)
    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    warmup_epochs = max(1, epochs // 10)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs)
    print(f"  LR warmup: {warmup_epochs} epochs")

    # Training state
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = {"train": [], "val": []}

    os.makedirs(save_dir, exist_ok=True)
    weight_path = os.path.join(save_dir, f"{name}_model.pt")

    for epoch in range(epochs):
        scheduler.step(epoch + 1)

        # ── Train ──
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for obs, target in train_loader:
            obs = obs.to(device)
            target = target.to(device)

            # Forward with deep supervision
            bottleneck, skips = model.encode(obs)
            features = {"s1": skips["s1"], "s2": skips["s2"], "s3": skips["s3"], "s4": bottleneck}
            pred_result = model.decoder(features, return_deep=True)

            if isinstance(pred_result, tuple):
                pred, deep_outputs = pred_result
            else:
                pred, deep_outputs = pred_result, None

            loss = criterion(pred, target, deep_outputs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        history["train"].append(avg_train_loss)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        n_val = 0
        pred_stats = {"min": 1.0, "max": 0.0, "mean": 0.0}

        with torch.no_grad():
            for obs, target in val_loader:
                obs = obs.to(device)
                target = target.to(device)
                pred = model(obs)
                val_loss += base_criterion(pred, target).item()
                pred_stats["min"] = min(pred_stats["min"], pred.min().item())
                pred_stats["max"] = max(pred_stats["max"], pred.max().item())
                pred_stats["mean"] += pred.mean().item()
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)
        pred_stats["mean"] /= max(n_val, 1)
        history["val"].append(avg_val_loss)

        # ── Save best checkpoint ──
        improved = avg_val_loss < best_val_loss
        if improved:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), weight_path)
        else:
            epochs_without_improvement += 1

        # ── Logging ──
        if (epoch + 1) % 5 == 0 or epoch == 0 or improved:
            lr = scheduler.get_last_lr()[0]
            marker = " ★" if improved else ""
            print(f"  Epoch {epoch+1:3d}/{epochs}  "
                  f"train={avg_train_loss:.6f}  val={avg_val_loss:.6f}  "
                  f"best={best_val_loss:.6f} (ep{best_epoch})  "
                  f"lr={lr:.2e}{marker}")

        # ── Early stopping ──
        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch+1} "
                  f"(no improvement for {patience} epochs)")
            break

    print(f"  ✓ Best weights saved to {weight_path} (epoch {best_epoch})")
    print(f"  Pred range: [{pred_stats['min']:.4f}, {pred_stats['max']:.4f}]  "
          f"mean={pred_stats['mean']:.4f}")

    return {
        "train_loss_final": history["train"][-1],
        "val_loss": best_val_loss,
        "best_epoch": best_epoch,
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
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (default: 10)")
    args = parser.parse_args()

    device = _get_device()
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
            patience=args.patience,
        )

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*70}")
    print(f"  TRAINING SUMMARY  (total time: {elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"{'Model':<12} {'Params':>10} {'Train Loss':>12} {'Val Loss':>12} {'Best Ep':>8}")
    print(f"{'-'*54}")
    for name, r in results.items():
        print(f"{name:<12} {r['params']:>10,} {r['train_loss_final']:>12.6f} "
              f"{r['val_loss']:>12.6f} {r['best_epoch']:>8}")
    print()


if __name__ == "__main__":
    main()
