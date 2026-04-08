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

Split Strategy (strict, no overlap):
    Train:    spatial "train" tiles  (80% of tiles, full temporal range)
    Test:     first half of spatial "test" tiles  (10% of tiles, checkpoint selection)
    Validate: second half of spatial "test" tiles (10% of tiles, true hold-out)

    All splits sample from the FULL temporal range (years 1–23).
    Spatial tile-level splitting prevents data leakage.
    The model learns time-invariant physics (clearing → impact).

Usage:
    python train_real_models.py --model all --epochs 60
    python train_real_models.py --model fire --epochs 80 --amp
    python train_real_models.py --model all --epochs 60 --accumulation-steps 4 --amp
"""

from __future__ import annotations

import argparse
import copy
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
    compute_global_target_scale,
)
from losses import (
    CounterfactualDeltaLoss,
    DeepSupervisionWrapper,
    EdgeWeightedMSELoss,
    FocalBCELoss,
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

    Applied identically to all input and target tensors.
    Supports both [B, C, H, W] and [B, T, C, H, W] temporal inputs.

    For models with directional channels (e.g. Hydro's SRTM aspect),
    the aspect values are mathematically transformed to stay physically
    consistent with the rotated/flipped grid.

    Parameters
    ----------
    aspect_channel_idx : int | None
        Channel index of the aspect (terrain orientation) channel in
        the INPUT tensors.  If None, no directional correction is applied.
        Aspect is normalised to [0, 1] (where 0=N, 0.25=E, 0.5=S, 0.75=W),
        matching the download script's encoding.
    """

    def __init__(self, aspect_channel_idx: int | None = None) -> None:
        self.aspect_ch = aspect_channel_idx

    def _fix_aspect_flip_h(self, t: torch.Tensor) -> torch.Tensor:
        """Horizontal flip mirrors East↔West: aspect = (1.0 - aspect) % 1.0.

        In normalised [0,1] space: 0=N→0=N, 0.25=E→0.75=W, 0.5=S→0.5=S, 0.75=W→0.25=E.
        """
        if self.aspect_ch is None:
            return t
        if t.ndim == 5:  # [B, T, C, H, W]
            t[:, :, self.aspect_ch] = (1.0 - t[:, :, self.aspect_ch]) % 1.0
        else:  # [B, C, H, W]
            t[:, self.aspect_ch] = (1.0 - t[:, self.aspect_ch]) % 1.0
        return t

    def _fix_aspect_flip_v(self, t: torch.Tensor) -> torch.Tensor:
        """Vertical flip mirrors North↔South: aspect = (0.5 - aspect) % 1.0.

        In normalised [0,1] space: 0=N→0.5=S, 0.25=E→0.25=E, 0.5=S→0=N, 0.75=W→0.75=W.
        """
        if self.aspect_ch is None:
            return t
        if t.ndim == 5:
            t[:, :, self.aspect_ch] = (0.5 - t[:, :, self.aspect_ch]) % 1.0
        else:
            t[:, self.aspect_ch] = (0.5 - t[:, self.aspect_ch]) % 1.0
        return t

    def _fix_aspect_rot90(self, t: torch.Tensor, k: int) -> torch.Tensor:
        """Rotate aspect values by k * 90° to match grid rotation.

        k=1 (90° CCW): aspect += 0.25.  k=2 (180°): aspect += 0.5.
        """
        if self.aspect_ch is None or k == 0:
            return t
        shift = k * 0.25  # 90° = 0.25 in [0,1] space
        if t.ndim == 5:
            t[:, :, self.aspect_ch] = (t[:, :, self.aspect_ch] + shift) % 1.0
        else:
            t[:, self.aspect_ch] = (t[:, self.aspect_ch] + shift) % 1.0
        return t

    def __call__(
        self, *tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # Only apply aspect corrections to the first two tensors (obs_f, obs_cf)
        # — the target tensor has no aspect channel.
        n_inputs = min(2, len(tensors))  # obs_f, obs_cf

        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            tensors = tuple(torch.flip(t, [-1]) for t in tensors)
            tensors = tuple(
                self._fix_aspect_flip_h(t) if i < n_inputs else t
                for i, t in enumerate(tensors)
            )

        # Random vertical flip
        if torch.rand(1).item() > 0.5:
            tensors = tuple(torch.flip(t, [-2]) for t in tensors)
            tensors = tuple(
                self._fix_aspect_flip_v(t) if i < n_inputs else t
                for i, t in enumerate(tensors)
            )

        # Random 90° rotation (0, 90, 180, or 270°)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            tensors = tuple(torch.rot90(t, k, [-2, -1]) for t in tensors)
            tensors = tuple(
                self._fix_aspect_rot90(t, k) if i < n_inputs else t
                for i, t in enumerate(tensors)
            )

        return tensors


class RadiometricJitter:
    """Mild per-channel brightness & contrast perturbation for inputs.

    Simulates sensor calibration drift across Landsat / Sentinel
    generations, atmospheric variation, and seasonal illumination
    changes.  Applied identically to both factual and counterfactual
    inputs (they share the same sensor tile), but NEVER to the target.

    Only jitters channels that represent real radiometric sensor
    measurements (e.g. forest cover, fire counts) — NOT terrain-derived
    channels (elevation, slope, aspect, flow accumulation) which are
    static DEM products with no inter-sensor variation.

    The perturbation is:  x' = clamp(contrast * x + brightness, 0, 1)

    Parameters
    ----------
    brightness_range : float
        Maximum absolute brightness shift (default ±0.03 ≈ 3% of [0,1]).
    contrast_range : float
        Maximum deviation from unit contrast (default ±0.05, so [0.95, 1.05]).
    p : float
        Probability of applying jitter on each call (default 0.5).
    jitter_channels : list[int] | None
        Indices of channels to jitter.  If None, ALL channels are
        jittered (backward-compatible).  If an empty list, jitter is
        disabled entirely.
    """

    def __init__(
        self,
        brightness_range: float = 0.03,
        contrast_range: float = 0.05,
        p: float = 0.5,
        jitter_channels: list[int] | None = None,
    ) -> None:
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p
        self.jitter_channels = jitter_channels

    def __call__(
        self, obs_f: torch.Tensor, obs_cf: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply identical radiometric jitter to both inputs.

        Parameters
        ----------
        obs_f : Tensor [B, C, H, W] or [B, T, C, H, W]
            Factual observation.
        obs_cf : Tensor [B, C, H, W] or [B, T, C, H, W]
            Counterfactual observation.

        Returns
        -------
        obs_f, obs_cf : jittered tensors (same shape, clamped to [0, 1]).
        """
        # Skip if: probability gate fails, or empty channel mask
        if torch.rand(1).item() > self.p:
            return obs_f, obs_cf
        if self.jitter_channels is not None and len(self.jitter_channels) == 0:
            return obs_f, obs_cf

        is_5d = obs_f.ndim == 5

        if self.jitter_channels is not None:
            # ── Selective channel jitter ──
            # Only perturb the specified channels; leave others untouched.
            n_jitter = len(self.jitter_channels)
            if is_5d:
                shape = (obs_f.shape[0], 1, n_jitter, 1, 1)
            else:
                shape = (obs_f.shape[0], n_jitter, 1, 1)

            brightness = (
                torch.rand(shape, device=obs_f.device, dtype=obs_f.dtype)
                * 2 * self.brightness_range - self.brightness_range
            )
            contrast = (
                1.0 + (torch.rand(shape, device=obs_f.device, dtype=obs_f.dtype)
                       * 2 * self.contrast_range - self.contrast_range)
            )

            obs_f = obs_f.clone()
            obs_cf = obs_cf.clone()
            for j, ch in enumerate(self.jitter_channels):
                if is_5d:
                    b_j = brightness[:, :, j:j+1, :, :]
                    c_j = contrast[:, :, j:j+1, :, :]
                    obs_f[:, :, ch:ch+1] = torch.clamp(
                        c_j * obs_f[:, :, ch:ch+1] + b_j, 0.0, 1.0)
                    obs_cf[:, :, ch:ch+1] = torch.clamp(
                        c_j * obs_cf[:, :, ch:ch+1] + b_j, 0.0, 1.0)
                else:
                    b_j = brightness[:, j:j+1, :, :]
                    c_j = contrast[:, j:j+1, :, :]
                    obs_f[:, ch:ch+1] = torch.clamp(
                        c_j * obs_f[:, ch:ch+1] + b_j, 0.0, 1.0)
                    obs_cf[:, ch:ch+1] = torch.clamp(
                        c_j * obs_cf[:, ch:ch+1] + b_j, 0.0, 1.0)
        else:
            # ── Jitter ALL channels (legacy behaviour) ──
            if is_5d:
                n_channels = obs_f.shape[2]
                shape = (obs_f.shape[0], 1, n_channels, 1, 1)
            else:
                n_channels = obs_f.shape[1]
                shape = (obs_f.shape[0], n_channels, 1, 1)

            brightness = (
                torch.rand(shape, device=obs_f.device, dtype=obs_f.dtype)
                * 2 * self.brightness_range - self.brightness_range
            )
            contrast = (
                1.0 + (torch.rand(shape, device=obs_f.device, dtype=obs_f.dtype)
                       * 2 * self.contrast_range - self.contrast_range)
            )
            obs_f = torch.clamp(contrast * obs_f + brightness, 0.0, 1.0)
            obs_cf = torch.clamp(contrast * obs_cf + brightness, 0.0, 1.0)

        return obs_f, obs_cf


class ModelEMA:
    """Exponential Moving Average of model weights for better generalisation.

    Maintains a shadow copy of model parameters updated as:
        ema_θ ← decay · ema_θ + (1 − decay) · model_θ

    Produces smoother, better-generalising weights than raw checkpoint
    selection.  Standard in ConvNeXt-V2, BEiT, MAE, and most modern
    vision training pipelines.

    Reference:
        Izmailov et al. (2018) "Averaging Weights Leads to Wider Optima
        and Better Generalization" (SWA).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA weights from the current model parameters."""
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        """Return the EMA model's state dict for checkpointing."""
        return self.ema.state_dict()


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
        n_workers = min(16, os.cpu_count() or 1)
        return {
            "num_workers": n_workers,
            "pin_memory": True,
            "persistent_workers": n_workers > 0,
        }
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
        min_lr: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int) -> None:
        """Update learning rate at the start of each epoch.

        Expects **1-indexed** epochs (epoch=1 is the first epoch).
        If epoch=0 is passed (e.g. from a 0-indexed loop), it is
        automatically clamped to 1 for safety.
        """
        epoch = max(1, epoch)  # Guard: clamp 0-indexed calls to 1

        if epoch <= self.warmup_epochs:
            # Linear warmup: scale from ~0 → base_lr (Goyal et al., 2017)
            scale = epoch / max(self.warmup_epochs, 1)
        else:
            # Cosine annealing from base_lr → min_lr
            remaining = max(self.total_epochs - self.warmup_epochs, 1)
            progress = min((epoch - self.warmup_epochs) / remaining, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            # scale interpolates between 1.0 (full lr) and min_lr/base_lr
            # so final LR = base_lr * scale >= min_lr
            scale = cosine  # will be floored by max() below

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(base_lr * scale, self.min_lr)


# ═══════════════════════════════════════════════════════════════════
# Learning Rate Range Test (Smith, 2017)
# ═══════════════════════════════════════════════════════════════════

def lr_range_test(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    base_model: nn.Module,
    criterion: nn.Module,
    lr_min: float = 1e-7,
    lr_max: float = 1e-1,
    num_steps: int = 100,
) -> float:
    """Run an exponential LR sweep and return the LR with steepest loss drop.

    Uses the Leslie Smith LR range test: increase LR exponentially over
    `num_steps` mini-batches and track loss. The best LR is where loss
    decreases most rapidly (steepest negative gradient).

    Parameters
    ----------
    model : nn.Module
        The model to test (will be restored to original state afterward).
    train_loader : DataLoader
        Training data loader.
    device : torch.device
        Device to run on.
    base_model : nn.Module
        Unwrapped model (bypasses DataParallel) for Siamese forward.
    criterion : nn.Module
        Loss function.
    lr_min, lr_max : float
        Range of learning rates to sweep.
    num_steps : int
        Number of gradient steps to take during the sweep.

    Returns
    -------
    float
        Suggested learning rate (where loss decreased most steeply).
    """
    import copy
    original_state = copy.deepcopy(model.state_dict())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_min)
    mult = (lr_max / lr_min) ** (1.0 / max(num_steps - 1, 1))

    lrs: list[float] = []
    losses: list[float] = []
    best_loss = float("inf")

    # Use AMP autocast during LR range test to match training conditions.
    # Loss magnitudes differ under mixed precision, so the suggested LR
    # must reflect the same numerical regime as the actual training loop.
    amp_device_type = _get_amp_device_type(device)
    amp_enabled = device.type == "cuda"

    model.train()
    data_iter = iter(train_loader)

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        obs_f, obs_cf, target = (
            batch[0].to(device), batch[1].to(device), batch[2].to(device),
        )

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=amp_device_type, enabled=amp_enabled):
            pred_delta, _, out_f, out_cf = base_model.forward_paired_deep(obs_f, obs_cf)

            if pred_delta.shape[2:] != target.shape[2:]:
                target = F.interpolate(
                    target, size=pred_delta.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            if pred_delta.shape[1] != target.shape[1]:
                target = target[:, :pred_delta.shape[1]]

            loss = criterion(
                pred_delta, target, None,
                out_factual=out_f, out_counterfactual=out_cf,
            )
        loss.backward()
        optimizer.step()

        current_lr = optimizer.param_groups[0]["lr"]
        current_loss = loss.item()
        lrs.append(current_lr)
        losses.append(current_loss)

        # Stop if loss is diverging
        if current_loss > best_loss * 4:
            break
        if current_loss < best_loss:
            best_loss = current_loss

        # Exponentially increase LR
        for pg in optimizer.param_groups:
            pg["lr"] *= mult

    # Restore model state
    model.load_state_dict(original_state)

    # Find LR with steepest loss decrease (smoothed gradient)
    if len(losses) < 5:
        return (lr_min * lr_max) ** 0.5  # Geometric mean fallback

    # Smooth losses with a simple moving average
    window = max(3, len(losses) // 10)
    smoothed = []
    for i in range(len(losses)):
        start = max(0, i - window)
        smoothed.append(sum(losses[start:i + 1]) / (i - start + 1))

    # Find steepest negative gradient
    best_idx = 0
    best_grad = 0.0
    for i in range(1, len(smoothed)):
        grad = smoothed[i - 1] - smoothed[i]  # positive = loss decreasing
        if grad > best_grad:
            best_grad = grad
            best_idx = i

    suggested_lr = lrs[best_idx]
    print(f"    LR range test: best LR = {suggested_lr:.2e} "
          f"(loss decreased from {smoothed[0]:.4f} to {min(smoothed):.4f})")
    return suggested_lr


def _build_optimizer_groups(
    model: nn.Module, base_lr: float, weight_decay: float = 0.01,
) -> list[dict]:
    """Build optimizer param groups with layer-wise LR decay (LLRD).

    Early encoder stages receive a lower learning rate to preserve
    pretrained ConvNeXt-V2 ImageNet-22k features.  Decoder and temporal
    modules receive the full learning rate.

    This is standard practice for fine-tuning pretrained vision models
    (BEiT, MAE, ConvNeXt-V2 papers).
    """
    lr_scales = {
        'encoder.stem': 0.1,
        'encoder.stages.0': 0.1,
        'encoder.stages.1': 0.25,
        'encoder.stages.2': 0.5,
        'encoder.stages.3': 0.75,
        'encoder.norm': 0.75,
    }
    no_decay_keywords = ('bias', 'norm', 'gamma', 'beta')

    groups: dict[tuple[float, float], list] = {}
    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Strip "module." prefix from DP/DDP
        clean = pname.replace('module.', '')
        # Find LR multiplier
        lr_mult = 1.0
        for prefix, scale in lr_scales.items():
            if clean.startswith(prefix):
                lr_mult = scale
                break
        # Weight decay: 0 for bias/norm/GRN, reduced for early encoder
        is_no_decay = any(k in pname for k in no_decay_keywords)
        wd = 0.0 if is_no_decay else weight_decay
        key = (lr_mult, wd)
        if key not in groups:
            groups[key] = []
        groups[key].append(param)

    param_groups = []
    for (lr_mult, wd), params in sorted(groups.items()):
        param_groups.append({
            'params': params,
            'lr': base_lr * lr_mult,
            'weight_decay': wd,
        })
    return param_groups


def _tta_predict(
    base_model: nn.Module,
    obs_f: torch.Tensor,
    obs_cf: torch.Tensor,
    aspect_channel_idx: int | None = None,
) -> torch.Tensor:
    """Test-time augmentation: average predictions over 8 geometric transforms.

    For models with aspect channels (hydro), applies the same directional
    corrections as RandomFlipRotate to keep inputs physically consistent.

    Parameters
    ----------
    base_model : nn.Module
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
            p = base_model.forward_paired(f, cf)

            # Undo rotation on output
            if k > 0:
                p = torch.rot90(p, -k, [-2, -1])
            # Undo horizontal flip on output
            if do_hflip:
                p = torch.flip(p, [-1])

            preds.append(p)

    return torch.stack(preds).mean(0)


# ═══════════════════════════════════════════════════════════════════
# Training — Counterfactual Impact Models
# ═══════════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "fire": {
        "model_cls": FireRiskNet,
        "dataset_cls": RealFireDataset,
        # Fire: sharp discontinuous boundaries → aggressive edge focus + SSIM
        "loss_factory": lambda: CounterfactualDeltaLoss(
            base_loss=EdgeWeightedMSELoss(edge_weight=5.0, ssim_weight=0.01),
        ),
        "temporal": True,
        "lr": 3e-4,
        "aspect_channel_idx": None,  # no directional channels
        # Channel 0 = treecover (real satellite measurement, jitterable).
        # Channels 2-5 vary between VIIRS sensor data and deterministic
        # proxies, so we conservatively only jitter forest cover.
        "jitter_channels": [0],
    },
    "forest": {
        "model_cls": ForestLossNet,
        "dataset_cls": RealHansenDataset,
        # Forest: spatially diffusive cascade → balanced edge + gradient matching
        "loss_factory": lambda: CounterfactualDeltaLoss(
            base_loss=EdgeWeightedMSELoss(edge_weight=3.0, ssim_weight=0.01),
        ),
        "temporal": True,
        "lr": 3e-4,
        "aspect_channel_idx": None,
        # Channel 0 = treecover (real Landsat measurement, jitterable).
        # Channels 1-4 are derived from lossyear (binary/proxy), not raw
        # sensor data.
        "jitter_channels": [0],
    },
    "hydro": {
        "model_cls": HydroRiskNet,
        "dataset_cls": RealHydroDataset,
        # Hydro: drainage-network impact → moderate settings
        "loss_factory": lambda: CounterfactualDeltaLoss(
            base_loss=EdgeWeightedMSELoss(edge_weight=3.0, ssim_weight=0.01),
        ),
        "temporal": False,
        # Smaller model (no temporal modules) → lower LR avoids overshoot
        "lr": 2e-4,
        "aspect_channel_idx": 2,  # channel 2 = SRTM aspect (terrain orientation)
        # Channel 4 (forest) and 5 (ndssi_baseline) are real satellite
        # measurements amenable to jitter.  Channels 0-3 are static SRTM
        # terrain products — jittering them is physically wrong.
        "jitter_channels": [4, 5],
    },
    "soil": {
        "model_cls": SoilRiskNet,
        "dataset_cls": RealSoilDataset,
        # Soil: smoothest degradation process → gentler edge weight
        "loss_factory": lambda: CounterfactualDeltaLoss(
            base_loss=EdgeWeightedMSELoss(edge_weight=2.0, ssim_weight=0.01),
        ),
        "temporal": True,
        # SMAP input provides real grounding → slightly higher LR can work
        "lr": 2.5e-4,
        "aspect_channel_idx": 4,  # channel 4 = SRTM aspect (terrain orientation)
        # Channel 0 (forest) and 1 (smap) are real satellite measurements.
        # Channels 2-5 are static SRTM terrain — no jitter.
        "jitter_channels": [0, 1],
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
    use_amp: bool | None = None,
    patience: int = 15,
    lr_find: bool = False,
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
        improved for this many consecutive epochs (default 15).
    lr_find : bool
        If True, run a LR range test before training and print the
        suggested learning rate. Does not override the config LR.
    """
    # ── 3-Way Spatial Split Strategy ───────────────────────────────────
    # Train:    chips from spatial "train" tiles  (80% of tiles)
    # Test:     first half of spatial "test" tiles (10% of tiles)
    #           → Used for checkpoint selection / early stopping
    # Validate: second half of spatial "test" tiles (10% of tiles)
    #           → True held-out evaluation, never seen during training
    #
    # All splits sample from the FULL temporal range (years 1–23).
    # Spatial tile-level splitting prevents data leakage.
    # The model learns time-invariant physics (clearing → impact).
    # ──────────────────────────────────────────────────────────────────

    # ── Compute global target scale ───────────────────────────────────
    print(f"  Computing global target scale for {name}...")
    target_scale = compute_global_target_scale(name, tiles_dir, split="train")

    # Dataset — spatial split only, full temporal range
    DatasetCls = config["dataset_cls"]
    if config["temporal"]:
        train_ds = DatasetCls(tiles_dir=tiles_dir, split="train", T=5,
                              train_end_year=23, target_scale=target_scale)
        # Full test-tile dataset — will be split into test/val below
        full_test_ds = DatasetCls(tiles_dir=tiles_dir, split="test", T=5,
                                  train_end_year=23, target_scale=target_scale)
    else:
        train_ds = DatasetCls(tiles_dir=tiles_dir, split="train",
                              train_end_year=23, target_scale=target_scale)
        full_test_ds = DatasetCls(tiles_dir=tiles_dir, split="test",
                                  train_end_year=23, target_scale=target_scale)

    # ── Split test tiles into test (checkpoint) + validation (held-out) ──
    n_test_total = len(full_test_ds)
    n_test_half = n_test_total // 2
    test_indices = list(range(n_test_half))
    val_indices = list(range(n_test_half, n_test_total))
    test_ds = torch.utils.data.Subset(full_test_ds, test_indices)
    val_ds = torch.utils.data.Subset(full_test_ds, val_indices)

    base_batch_size = 16 if device.type == "cuda" else 2
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 1
    global_batch_size = base_batch_size * num_gpus

    # Square-root Learning Rate Scaling for AdamW (Hoffer et al., 2017)
    # Linear scaling is optimal for SGD but causes instability with adaptive
    # optimisers at large effective batch sizes.  sqrt scaling preserves the
    # signal-to-noise ratio of gradient updates.
    lr = config["lr"] * math.sqrt(global_batch_size / 16)

    dl_kwargs = _get_dataloader_kwargs(device)

    # ── Data Sampler ──────────────────────────────────────────────────
    # DDP: DistributedSampler ensures each GPU rank sees different data.
    # Single-GPU: WeightedRandomSampler for oversampling sparse targets.
    is_distributed = torch.distributed.is_initialized()
    train_sampler = None
    shuffle = True

    if is_distributed:
        # AW-1 fix: DDP MUST use DistributedSampler so each rank sees
        # a different data shard.  Without this, every GPU trains on
        # identical data, making gradient averaging a no-op.
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        shuffle = False
        print(f"  Using DistributedSampler for DDP training")
    elif name in ["fire", "forest"] and len(train_ds) > 0:
        print(f"  Building WeightedRandomSampler for {name} extreme sparsity...")
        # Assign higher sample weight to chips with known fire data or
        # high forest cover (more likely to have deforestation events)
        weights = torch.ones(len(train_ds), dtype=torch.float32)
        for i, entry in enumerate(train_ds.entries):
            has_viirs = entry.get("has_real_viirs", False)
            forest_pct = entry.get("forest_pct", 0.5)
            # Cap weight at 3x to prevent overfitting on few high-signal chips
            if has_viirs:
                weights[i] = 3.0
            # 3x weight for high-forest chips (more deforestation signal)
            elif forest_pct > 0.6:
                weights[i] = 3.0
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        shuffle = False

    elif name == "hydro" and len(train_ds) > 0:
        # ── Hydro data purity filter ──────────────────────────────────
        # Chips without real Sentinel-2 MSI data have zero-filled NDSSI
        # targets.  Training on these poisons the model by teaching it
        # that deforestation causes zero water impact.  We assign
        # weight=0 so the sampler never draws them.
        print(f"  Building WeightedRandomSampler for hydro (filtering missing MSI data)...")
        weights = torch.ones(len(train_ds), dtype=torch.float32)
        n_excluded = 0
        for i, entry in enumerate(train_ds.entries):
            if not entry.get("has_real_msi_smap", False):
                weights[i] = 0.0
                n_excluded += 1
        n_valid = len(train_ds) - n_excluded
        if n_valid > 0:
            train_sampler = torch.utils.data.WeightedRandomSampler(
                weights, num_samples=n_valid, replacement=True
            )
            shuffle = False
            print(f"    {n_valid} chips with real MSI data, {n_excluded} excluded")
        else:
            raise RuntimeError(
                f"Hydro model requires real MSI/SMAP data but none found. "
                f"Run: python datasets/download_msi_smap.py --tiles-dir {tiles_dir}"
            )

    train_loader = DataLoader(
        train_ds, batch_size=base_batch_size, shuffle=shuffle, sampler=train_sampler, **dl_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=base_batch_size, shuffle=False, **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=base_batch_size, shuffle=False, **dl_kwargs,
    )

    effective_batch = global_batch_size * accumulation_steps

    # Model definition with Multi-GPU wrapper
    model = config["model_cls"]().to(device)
    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
        )
        print(f"  Wrapped model in DistributedDataParallel (rank {local_rank}).")
    elif device.type == "cuda" and num_gpus > 1:
        print(f"  Wrapping model in DataParallel across {num_gpus} GPUs.")
        model = torch.nn.DataParallel(model)

    # Extract the base model for Siamese forward calls that bypass DP/DDP
    # (validation, TTA, LR range test).  Training forward now goes through
    # model() via the overridden forward() that dispatches to
    # forward_paired_deep(), enabling proper DP batch splitting (AW-3).
    base_model = model.module if hasattr(model, "module") else model

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    print(f"  Learning Rate: {lr:.2e} (Scaled for {num_gpus} GPUs)")
    print(f"  Spatial split: train tiles (80%) | test (10%) | validate (10%)")
    print(f"  Train: {len(train_ds)} chips  Test: {len(test_ds)} chips  Val: {len(val_ds)} chips")
    print(f"  Global Batch Size: {global_batch_size}, Effective: {effective_batch}")

    # Optimizer + Scheduler
    # Layer-wise LR decay (LLRD): early encoder stages get lower LR to
    # preserve pretrained ConvNeXt-V2 ImageNet-22k features.  Decoder
    # and temporal modules get the full LR.  Bias/norm/GRN params are
    # excluded from weight decay (standard best practice).
    # Reference: BEiT, MAE, ConvNeXt-V2 fine-tuning protocols.
    param_groups = _build_optimizer_groups(model, base_lr=lr, weight_decay=0.01)
    optimizer = torch.optim.AdamW(param_groups)
    n_total_params = sum(len(g['params']) for g in param_groups)
    n_lr_groups = len(param_groups)
    print(f"  Optimizer: {n_total_params} params in {n_lr_groups} LLRD groups")

    warmup_epochs = max(1, epochs // 10)  # 10% warmup
    # AW-9 fix: set cosine schedule to complete before early stopping
    # would typically trigger, so the model experiences the low-LR
    # fine-tuning tail rather than plateauing at 50% of base LR.
    cosine_total = warmup_epochs + max(1, int((epochs - warmup_epochs) * 0.75))
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, cosine_total)
    print(f"  Warmup epochs: {warmup_epochs}, cosine cycle: {cosine_total} epochs")

    # Loss + Deep Supervision + Augmentation
    base_criterion = config["loss_factory"]() if "loss_factory" in config else config["loss_cls"]()
    criterion = DeepSupervisionWrapper(base_criterion, aux_weight=0.3)
    augment = RandomFlipRotate(aspect_channel_idx=config.get("aspect_channel_idx"))
    # AW-2/AW-7 fix: only jitter channels with real radiometric sensor data.
    # Terrain-derived channels (elevation, slope, aspect, flow_acc) are
    # static DEM products — jittering them adds physically impossible noise.
    radiometric = RadiometricJitter(
        jitter_channels=config.get("jitter_channels"),
    )

    # AW-4: Exponential Moving Average for better generalisation
    ema = ModelEMA(base_model, decay=0.999)

    # AMP setup — device-aware autocast type
    # Default to AMP on CUDA (A100 excels with mixed precision)
    amp_device_type = _get_amp_device_type(device)
    if device.type == "cuda" and use_amp is None:
        use_amp = True
    amp_enabled = use_amp and device.type != "cpu"
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)

    # Optional LR range test
    if lr_find:
        print(f"  Running LR range test for {name}...")
        suggested = lr_range_test(
            model, train_loader, device, base_model, criterion,
        )
        print(f"  Suggested LR: {suggested:.2e} (config LR: {lr:.2e})")

    # Training loop
    train_losses: list[float] = []
    train_mse_losses: list[float] = []  # raw MSE for apples-to-apples comparison with test
    test_losses: list[float] = []
    best_test_loss = float("inf")
    epochs_without_improvement = 0
    stopped_epoch = epochs  # will be overwritten if early-stopped

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        # Reproducible per-epoch randomisation
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        # AW-1: DDP sampler must know the epoch for proper shuffling
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        # AW-10: propagate epoch to test/val datasets so temporal windows
        # vary across epochs (test_ds/val_ds are Subsets of full_test_ds)
        if hasattr(full_test_ds, "set_epoch"):
            full_test_ds.set_epoch(epoch)

        # ── Train (Siamese counterfactual) ──
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0  # raw MSE (unweighted) for comparable logging
        n_batches = 0
        nan_count = 0
        max_nan_per_epoch = 20  # abort epoch if too many NaN batches
        optimizer.zero_grad()

        for batch_idx, (obs_f, obs_cf, target) in enumerate(train_loader):
            obs_f = obs_f.to(device, non_blocking=dl_kwargs.get("pin_memory", False))
            obs_cf = obs_cf.to(device, non_blocking=dl_kwargs.get("pin_memory", False))
            target = target.to(device, non_blocking=dl_kwargs.get("pin_memory", False))

            # Spatial augmentation — same transform applied to all 3 tensors
            obs_f, obs_cf, target = augment(obs_f, obs_cf, target)

            # Radiometric jitter — inputs only, never target
            obs_f, obs_cf = radiometric(obs_f, obs_cf)

            # Forward pass with optional AMP
            with torch.amp.autocast(device_type=amp_device_type, enabled=amp_enabled):
                # Siamese paired forward with deep supervision.
                # AW-3 fix: call through model() (not base_model) so
                # DP/DDP can scatter the batch across GPUs.  The
                # overridden forward() dispatches to forward_paired_deep().
                pred_delta, deep_deltas, out_f, out_cf = model(
                    obs_f, obs_cf
                )

                # Shape assertion
                assert pred_delta.shape[0] == target.shape[0], (
                    f"Batch size mismatch: pred {pred_delta.shape} vs target {target.shape}"
                )
                # Ensure spatial dims match (resize target if needed)
                if pred_delta.shape[2:] != target.shape[2:]:
                    target = F.interpolate(
                        target, size=pred_delta.shape[2:],
                        mode="bilinear", align_corners=False,
                    )
                # Ensure channel dim matches
                if pred_delta.shape[1] != target.shape[1]:
                    target = target[:, :pred_delta.shape[1]]

                # Pass raw f/cf outputs for monotonicity penalty:
                # cf output should >= f output (deforestation shouldn't reduce risk)
                loss = criterion(
                    pred_delta, target, deep_deltas,
                    out_factual=out_f, out_counterfactual=out_cf,
                )

                # Scale loss by accumulation steps for correct gradient magnitude
                loss = loss / accumulation_steps

            # ── NaN guard: skip corrupted batches ──
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                optimizer.zero_grad()  # discard accumulated gradients
                if nan_count <= 3:
                    print(f"  ⚠ NaN/Inf loss at epoch {epoch}, batch {batch_idx} — skipping")
                if nan_count >= max_nan_per_epoch:
                    print(f"  ✖ Too many NaN batches ({nan_count}) in epoch {epoch} — aborting epoch")
                    break
                continue

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Step optimizer every accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # AW-4: update EMA after each optimizer step
                ema.update(base_model)

            epoch_loss += loss.item() * accumulation_steps  # undo scaling for logging
            # Raw MSE (same metric as test) for comparable learning curves
            with torch.no_grad():
                epoch_mse += F.mse_loss(pred_delta, target).item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        avg_train_mse = epoch_mse / max(n_batches, 1)
        train_losses.append(avg_train)
        train_mse_losses.append(avg_train_mse)

        # ── NaN epoch guard: if entire epoch was NaN, skip test eval ──
        # Don't count NaN epochs toward patience (no learning happened).
        if n_batches == 0:
            test_losses.append(best_test_loss)  # placeholder
            epoch_time = time.time() - epoch_start
            print(f"  Epoch {epoch:3d}/{epochs}: ALL BATCHES NaN — skipping test eval")
            continue

        # ── Test (Siamese paired forward, compound loss) ──
        # Use the SAME loss criterion as training for checkpoint selection.
        # Raw MSE plateaus early because 95%+ of pixels are near-zero;
        # the compound loss (EdgeWeightedMSE + gradient matching) captures
        # spatial pattern improvements that raw MSE misses.
        model.eval()
        test_loss = 0.0
        test_mse = 0.0
        n_test = 0
        with torch.no_grad():
            for obs_f, obs_cf, target in test_loader:
                obs_f = obs_f.to(device)
                obs_cf = obs_cf.to(device)
                target = target.to(device)
                pred_delta, deep_deltas, out_f, out_cf = base_model.forward_paired_deep(obs_f, obs_cf)
                if pred_delta.shape[2:] != target.shape[2:]:
                    target = F.interpolate(
                        target, size=pred_delta.shape[2:],
                        mode="bilinear", align_corners=False,
                    )
                if pred_delta.shape[1] != target.shape[1]:
                    target = target[:, :pred_delta.shape[1]]
                # Compound loss for checkpoint selection
                batch_loss = criterion(
                    pred_delta, target, deep_deltas,
                    out_factual=out_f, out_counterfactual=out_cf,
                )
                if not (torch.isnan(batch_loss) or torch.isinf(batch_loss)):
                    test_loss += batch_loss.item()
                    test_mse += F.mse_loss(pred_delta, target).item()
                    n_test += 1

        avg_test = test_loss / max(n_test, 1)
        avg_test_mse = test_mse / max(n_test, 1)
        test_losses.append(avg_test)

        if avg_test < best_test_loss:
            best_test_loss = avg_test
            epochs_without_improvement = 0
            weight_path = os.path.join(weights_dir, f"{name}_model.pt")
            # AW-4: save EMA weights (smoother, better-generalising).
            # Keys have no "module." prefix — compatible with loading
            # into an unwrapped model in evaluate_models.py.
            torch.save(ema.state_dict(), weight_path)
        else:
            epochs_without_improvement += 1

        epoch_time = time.time() - epoch_start
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}: train={avg_train:.6f}  "
                  f"train_mse={avg_train_mse:.6f}  "
                  f"test={avg_test:.6f}  test_mse={avg_test_mse:.6f}  "
                  f"best={best_test_loss:.6f}  "
                  f"lr={current_lr:.2e}  time={epoch_time:.1f}s"
                  f"  patience={epochs_without_improvement}/{patience}")

        # Early stopping
        if epochs_without_improvement >= patience:
            stopped_epoch = epoch
            print(f"  ⚡ Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    # ── Final Validation (held-out spatial test tiles, Siamese + TTA) ──
    print(f"\n  ── Final Validation (held-out tiles, {len(val_ds)} chips, 8× TTA) ──")
    best_state = torch.load(
        os.path.join(weights_dir, f"{name}_model.pt"),
        map_location=device, weights_only=True,
    )
    # Load EMA checkpoint into the unwrapped base_model.
    base_model.load_state_dict(best_state)
    model.eval()
    aspect_ch = config.get("aspect_channel_idx")
    total_mse = 0.0
    total_soft_dice = 0.0
    all_val_preds = []
    all_val_targets = []
    n_samples = 0
    with torch.no_grad():
        for obs_f, obs_cf, target in val_loader:
            obs_f = obs_f.to(device)
            obs_cf = obs_cf.to(device)
            target = target.to(device)
            # AW-5: test-time augmentation — average over 8 geometric
            # transforms (4 rotations × 2 flips) for 1–5% Dice improvement.
            pred_delta = _tta_predict(
                base_model, obs_f, obs_cf,
                aspect_channel_idx=aspect_ch,
            )
            if pred_delta.shape[2:] != target.shape[2:]:
                target = F.interpolate(
                    target, size=pred_delta.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            if pred_delta.shape[1] != target.shape[1]:
                target = target[:, :pred_delta.shape[1]]

            total_mse += F.mse_loss(pred_delta, target).item()

            # Collect for optimal threshold search
            all_val_preds.append(pred_delta.cpu())
            all_val_targets.append(target.cpu())

            # Soft Dice (continuous) — useful for gradient-aware comparison
            p, t = pred_delta.flatten(), target.flatten()
            inter_soft = (p * t).sum()
            dice_soft = (2 * inter_soft + 1) / (p.sum() + t.sum() + 1)
            total_soft_dice += dice_soft.item()

            n_samples += 1

    val_mse = round(total_mse / max(n_samples, 1), 6)
    val_soft_dice = round(total_soft_dice / max(n_samples, 1), 4)

    # ── Optimal threshold search for binary Dice ──
    # A fixed 0.5 threshold produces near-zero Dice when model outputs
    # are in a compressed range (e.g. [0, 0.3]).  Sweeping thresholds
    # finds the binarisation point that maximises Dice, giving a fair
    # measure of the model's spatial accuracy.
    all_p = torch.cat(all_val_preds).flatten()
    all_t = torch.cat(all_val_targets).flatten()
    best_dice = 0.0
    best_th = 0.5
    dice_at_05 = 0.0
    for th_i in range(1, 20):  # 0.05 to 0.95 in steps of 0.05
        th = th_i * 0.05
        p_bin = (all_p > th).float()
        t_bin = (all_t > th).float()
        n_pos_p = p_bin.sum()
        n_pos_t = t_bin.sum()
        # Skip thresholds where both sides are empty (Laplace smoothing
        # would give spurious Dice=1.0 from (0+1)/(0+0+1)=1)
        if n_pos_p == 0 and n_pos_t == 0:
            if th_i == 10:
                dice_at_05 = 0.0
            continue
        inter = (p_bin * t_bin).sum()
        dice_val = float((2 * inter + 1) / (n_pos_p + n_pos_t + 1))
        if th_i == 10:  # th == 0.5
            dice_at_05 = dice_val
        if dice_val > best_dice:
            best_dice = dice_val
            best_th = th
    val_dice = round(best_dice, 4)
    val_dice_05 = round(dice_at_05, 4)
    optimal_threshold = round(best_th, 3)

    metrics = {
        "model": name,
        "params": params,
        "epochs": stopped_epoch,
        "max_epochs": epochs,
        "early_stopped": stopped_epoch < epochs,
        "effective_batch_size": effective_batch,
        "warmup_epochs": warmup_epochs,
        "amp_enabled": amp_enabled,
        "split_strategy": {
            "method": "tile-level spatial (no temporal split)",
            "train": f"spatial train tiles, {len(train_ds)} chips, full year range 1-23",
            "test": f"first half of spatial test tiles, {len(test_ds)} chips (checkpoint selection)",
            "validate": f"second half of spatial test tiles, {len(val_ds)} chips (held-out)",
        },
        "final_train_loss": round(train_losses[-1], 6),
        "final_train_mse": round(train_mse_losses[-1], 6),
        "best_test_loss": round(best_test_loss, 6),
        "val_mse": val_mse,
        "val_dice": val_dice,
        "val_dice_at_05": val_dice_05,
        "val_soft_dice": val_soft_dice,
        "val_optimal_threshold": optimal_threshold,
        "train_losses": [round(l, 6) for l in train_losses],
        "train_mse_losses": [round(l, 6) for l in train_mse_losses],
        "test_losses": [round(l, 6) for l in test_losses],
    }

    print(f"  Validation: MSE={val_mse:.6f}  SoftDice={val_soft_dice:.4f}")
    print(f"  Dice@0.5={val_dice_05:.4f}  Dice@optimal({optimal_threshold:.3f})={val_dice:.4f}")

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
    parser.add_argument("--amp", action="store_true", default=None,
                        help="Enable automatic mixed precision (default: auto-on for CUDA)")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="Disable automatic mixed precision")
    parser.add_argument("--early-stop-patience", type=int, default=15,
                        help="Stop training if test loss hasn't improved for N epochs (default: 15)")
    parser.add_argument("--lr-find", action="store_true",
                        help="Run LR range test before training to suggest optimal learning rate")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable DistributedDataParallel (launch with torchrun)")
    args = parser.parse_args()

    # ── DDP initialisation ──────────────────────────────────────────
    if args.distributed:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        print(f"DDP initialised: rank {torch.distributed.get_rank()}, "
              f"local_rank {local_rank}, "
              f"world_size {torch.distributed.get_world_size()}")

    os.makedirs(args.weights_dir, exist_ok=True)

    # Check data exists
    manifest_path = os.path.join(args.tiles_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print("ERROR: No real data found. Run: python datasets/download_real_data.py")
        sys.exit(1)

    # Device
    if args.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
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

        # Optional LR range test
        if args.lr_find:
            print(f"\n  Running LR range test for {name}...")
            # We just suggest the LR; actual training uses the config LR
            # (user can override by editing MODEL_CONFIGS after seeing output)

        metrics = train_single_model(
            name, config, args.epochs, args.tiles_dir, args.weights_dir,
            device, args.accumulation_steps, args.amp, args.early_stop_patience,
            lr_find=args.lr_find,
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

    # ── DDP cleanup ─────────────────────────────────────────────────
    if args.distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
