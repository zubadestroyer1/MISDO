"""
MISDO — Real Data Training Pipeline (Production-Ready)
=========================================================
Trains all 4 domain-specific models on real Hansen GFC satellite data
with production-grade training features.

Features:
    - Linear LR warmup (first 10% of epochs)
    - Gradient accumulation (effective batch size 16-64 on A100)
    - Data augmentation (random flips + 90° rotations)
    - Automatic Mixed Precision (AMP) for CUDA/MPS
    - UNet++ deep supervision auxiliary losses
    - Gradient clipping with max norm
    - Proper epoch timing and logging
    - Checkpoint saving (best model by test loss)
    - MPS-safe DataLoader settings

Temporal Split Strategy:
    Train:    spatial "train" tiles, years 2001–2018
    Test:     spatial "train" tiles, years 2019–2020 (monitoring)
    Validate: spatial "test" tiles, years 2021–2023 (true hold-out)

Usage:
    python train_real_models.py --model all --epochs 60
    python train_real_models.py --model fire --epochs 80 --amp
    python train_real_models.py --model all --epochs 60 --accumulation-steps 4 --amp
"""

from __future__ import annotations

import argparse
import json
import math
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


# ═══════════════════════════════════════════════════════════════════
# Data Augmentation
# ═══════════════════════════════════════════════════════════════════

class RandomFlipRotate:
    """Random augmentation: horizontal/vertical flips + 90° rotations.

    Applied identically to both input and target tensors.
    Supports both [B, C, H, W] and [B, T, C, H, W] temporal inputs.
    """

    def __call__(
        self, obs: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            obs = torch.flip(obs, [-1])
            target = torch.flip(target, [-1])

        # Random vertical flip
        if torch.rand(1).item() > 0.5:
            obs = torch.flip(obs, [-2])
            target = torch.flip(target, [-2])

        # Random 90° rotation (0, 90, 180, or 270°)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            obs = torch.rot90(obs, k, [-2, -1])
            target = torch.rot90(target, k, [-2, -1])

        return obs, target


# ═══════════════════════════════════════════════════════════════════
# Device Utilities
# ═══════════════════════════════════════════════════════════════════

def _get_amp_device_type(device: torch.device) -> str:
    """Return the correct device type string for torch.amp.autocast."""
    if device.type == "cuda":
        return "cuda"
    elif device.type == "mps":
        return "mps"
    return "cpu"


def _get_dataloader_kwargs(device: torch.device) -> dict:
    """Return MPS/CUDA-safe DataLoader keyword arguments.

    MPS does not support CUDA-style pinned memory, so pin_memory
    must be False when using Apple Silicon GPUs.
    """
    if device.type == "cuda":
        return {"num_workers": min(8, os.cpu_count() or 1), "pin_memory": True}
    # MPS and CPU: no pinned memory, no multiprocessing workers
    # (MPS + multiprocess workers can cause silent hangs)
    return {"num_workers": 0, "pin_memory": False}


# ═══════════════════════════════════════════════════════════════════
# Learning Rate Scheduler with Warmup
# ═══════════════════════════════════════════════════════════════════

class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    warmup_epochs : int
        Number of warmup epochs (linear ramp from 0 to base_lr).
    total_epochs : int
        Total number of training epochs.
    """

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
        """Update learning rate at the start of each epoch (1-indexed)."""
        if epoch <= self.warmup_epochs:
            # Linear warmup: scale from 0 → base_lr
            scale = epoch / max(self.warmup_epochs, 1)
        else:
            # Cosine annealing from base_lr → 0
            progress = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "fire": {
        "model_cls": FireRiskNet,
        "dataset_cls": RealFireDataset,
        "loss_cls": FocalBCELoss,
        "temporal": True,
        "lr": 3e-4,
    },
    "forest": {
        "model_cls": ForestLossNet,
        "dataset_cls": RealHansenDataset,
        "loss_cls": DiceBCELoss,
        "temporal": True,
        "lr": 3e-4,
    },
    "hydro": {
        "model_cls": HydroRiskNet,
        "dataset_cls": RealHydroDataset,
        "loss_cls": GradientMSELoss,
        "temporal": False,
        "lr": 1e-3,
    },
    "soil": {
        "model_cls": SoilRiskNet,
        "dataset_cls": RealSoilDataset,
        "loss_cls": SmoothMSELoss,
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
    accumulation_steps: int = 4,
    use_amp: bool = False,
    patience: int = 10,
) -> dict:
    """Train a single model on real data with production-grade features.

    Parameters
    ----------
    accumulation_steps : int
        Number of gradient accumulation steps (effective batch = batch_size × steps).
    use_amp : bool
        Enable automatic mixed precision (requires CUDA/MPS).
    patience : int
        Early stopping patience — stop training if test loss hasn't
        improved for this many consecutive epochs (default 10).
    """
    # ── Temporal Split Strategy ──────────────────────────────────────
    # Train:    spatial "train" tiles, years 2001-2018  (year_start=1, end=18)
    # Test:     spatial "train" tiles, years 2014-2020  (year_start=14, end=20)
    #           → context uses recent history, TARGET (predicted period)
    #             is years 2019-2020 which is non-overlapping with train
    # Validate: spatial "test" tiles,  years 2018-2023  (year_start=18, end=23)
    #           → true hold-out: unseen locations AND unseen time period
    # ──────────────────────────────────────────────────────────────────

    # Dataset
    DatasetCls = config["dataset_cls"]
    if config["temporal"]:
        train_ds = DatasetCls(tiles_dir=tiles_dir, split="train", T=5, train_end_year=18)
        test_ds = DatasetCls(tiles_dir=tiles_dir, split="train", T=5, train_end_year=20, year_start=14)
        val_ds = DatasetCls(tiles_dir=tiles_dir, split="test", T=5, train_end_year=23, year_start=18)
    else:
        train_ds = DatasetCls(tiles_dir=tiles_dir, split="train", train_end_year=18)
        test_ds = DatasetCls(tiles_dir=tiles_dir, split="train", train_end_year=20)
        val_ds = DatasetCls(tiles_dir=tiles_dir, split="test", train_end_year=23)

    batch_size = 16 if device.type == "cuda" else 2
    dl_kwargs = _get_dataloader_kwargs(device)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, **dl_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **dl_kwargs,
    )

    effective_batch = batch_size * accumulation_steps

    # Model
    model = config["model_cls"]().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    print(f"  Temporal split: train ≤2018 | test 2014–2020 | validate 2018–2023")
    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}, Validate: {len(val_ds)}")
    print(f"  Effective batch size: {effective_batch}")

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=0.01
    )
    warmup_epochs = max(1, epochs // 10)  # 10% warmup
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs)
    print(f"  Warmup epochs: {warmup_epochs}")

    # Loss + Deep Supervision + Augmentation
    base_criterion = config["loss_cls"]()
    criterion = DeepSupervisionWrapper(base_criterion, aux_weight=0.3)
    augment = RandomFlipRotate()

    # AMP setup — device-aware autocast type
    amp_device_type = _get_amp_device_type(device)
    amp_enabled = use_amp and device.type != "cpu"
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    # Training loop
    train_losses: list[float] = []
    test_losses: list[float] = []
    best_test_loss = float("inf")
    epochs_without_improvement = 0
    stopped_epoch = epochs  # will be overwritten if early-stopped

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Train ──
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, (obs, target) in enumerate(train_loader):
            obs = obs.to(device, non_blocking=dl_kwargs.get("pin_memory", False))
            target = target.to(device, non_blocking=dl_kwargs.get("pin_memory", False))

            # Data augmentation
            obs, target = augment(obs, target)

            # Forward pass with optional AMP
            with torch.amp.autocast(device_type=amp_device_type, enabled=amp_enabled):
                # Forward with deep supervision for auxiliary losses
                bottleneck, skips = model.encode(obs)
                features = {"s1": skips["s1"], "s2": skips["s2"], "s3": skips["s3"], "s4": bottleneck}
                pred_result = model.decoder(features, return_deep=True)

                if isinstance(pred_result, tuple):
                    pred, deep_outputs = pred_result
                else:
                    pred, deep_outputs = pred_result, None

                # Shape assertion
                assert pred.shape[0] == target.shape[0], (
                    f"Batch size mismatch: pred {pred.shape} vs target {target.shape}"
                )
                # Ensure spatial dims match (resize target if needed)
                if pred.shape[2:] != target.shape[2:]:
                    target = F.interpolate(
                        target, size=pred.shape[2:],
                        mode="bilinear", align_corners=False,
                    )
                # Ensure channel dim matches
                if pred.shape[1] != target.shape[1]:
                    target = target[:, :pred.shape[1]]

                loss = criterion(pred, target, deep_outputs)

                # Scale loss by accumulation steps for correct gradient magnitude
                loss = loss / accumulation_steps

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Step optimizer every accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accumulation_steps  # undo scaling for logging
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # ── Test (2019-2020 temporal window, same spatial tiles) ──
        model.eval()
        test_loss = 0.0
        n_test = 0
        with torch.no_grad():
            for obs, target in test_loader:
                obs = obs.to(device)
                target = target.to(device)
                pred = model(obs)
                if pred.shape[2:] != target.shape[2:]:
                    target = F.interpolate(
                        target, size=pred.shape[2:],
                        mode="bilinear", align_corners=False,
                    )
                if pred.shape[1] != target.shape[1]:
                    target = target[:, :pred.shape[1]]
                # Test loss without deep supervision (main output only)
                test_loss += base_criterion(pred, target).item()
                n_test += 1

        avg_test = test_loss / max(n_test, 1)
        test_losses.append(avg_test)

        if avg_test < best_test_loss:
            best_test_loss = avg_test
            epochs_without_improvement = 0
            weight_path = os.path.join(weights_dir, f"{name}_model.pt")
            torch.save(model.state_dict(), weight_path)
        else:
            epochs_without_improvement += 1

        epoch_time = time.time() - epoch_start
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}: train={avg_train:.6f}  "
                  f"test={avg_test:.6f}  best={best_test_loss:.6f}  "
                  f"lr={current_lr:.2e}  time={epoch_time:.1f}s"
                  f"  patience={epochs_without_improvement}/{patience}")

        # Early stopping
        if epochs_without_improvement >= patience:
            stopped_epoch = epoch
            print(f"  ⚡ Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    # ── Final Validation (2021+, held-out spatial tiles) ──
    print(f"\n  ── Final Validation (2021+, held-out tiles) ──")
    model.load_state_dict(torch.load(
        os.path.join(weights_dir, f"{name}_model.pt"),
        map_location=device, weights_only=True,
    ))
    model.eval()
    total_mse = 0.0
    total_dice = 0.0
    n_samples = 0
    with torch.no_grad():
        for obs, target in val_loader:
            obs = obs.to(device)
            target = target.to(device)
            pred = model(obs)
            if pred.shape[2:] != target.shape[2:]:
                target = F.interpolate(
                    target, size=pred.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            if pred.shape[1] != target.shape[1]:
                target = target[:, :pred.shape[1]]

            total_mse += F.mse_loss(pred, target).item()
            p, t = pred.flatten(), target.flatten()
            intersection = (p * t).sum()
            dice = (2 * intersection + 1) / (p.sum() + t.sum() + 1)
            total_dice += dice.item()
            n_samples += 1

    val_mse = round(total_mse / max(n_samples, 1), 6)
    val_dice = round(total_dice / max(n_samples, 1), 4)

    metrics = {
        "model": name,
        "params": params,
        "epochs": stopped_epoch,
        "max_epochs": epochs,
        "early_stopped": stopped_epoch < epochs,
        "effective_batch_size": effective_batch,
        "warmup_epochs": warmup_epochs,
        "amp_enabled": use_amp,
        "temporal_split": {"train": "2001-2018", "test": "2019-2020", "validate": "2021-2023"},
        "final_train_loss": round(train_losses[-1], 6),
        "best_test_loss": round(best_test_loss, 6),
        "val_mse": val_mse,
        "val_dice": val_dice,
        "train_losses": [round(l, 6) for l in train_losses],
        "test_losses": [round(l, 6) for l in test_losses],
    }

    print(f"  Validation: MSE={val_mse:.6f}  Dice={val_dice:.4f}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MISDO models on real data")
    parser.add_argument("--model", default="all",
                        choices=["all", "fire", "forest", "hydro", "soil"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--tiles-dir", default="datasets/real_tiles")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size × this)")
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed precision")
    parser.add_argument("--early-stop-patience", type=int, default=10,
                        help="Stop training if test loss hasn't improved for N epochs (default: 10)")
    args = parser.parse_args()

    os.makedirs(args.weights_dir, exist_ok=True)

    # Check data exists
    manifest_path = os.path.join(args.tiles_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print("ERROR: No real data found. Run: python datasets/download_real_data.py")
        sys.exit(1)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
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
            name, config, args.epochs, args.tiles_dir, args.weights_dir,
            device, args.accumulation_steps, args.amp, args.early_stop_patience,
        )
        all_metrics[name] = metrics

    # Save metrics
    metrics_path = os.path.join(args.weights_dir, "real_training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  All training complete in {elapsed:.1f}s")
    print(f"  Metrics saved to {metrics_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
