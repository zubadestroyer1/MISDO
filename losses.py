"""
MISDO — Production-Grade Loss Functions
==========================================
Consolidated loss module for all domain-specific risk models.
All loss functions include numerical safety guards (prediction clamping,
epsilon-protected divisions) to prevent NaN under mixed-precision training.

Loss Functions:
    FocalBCELoss    — Focal-weighted binary cross-entropy for sparse events (fire, forest loss)
    DiceBCELoss     — Combined Dice + BCE for segmentation with fragmented patches
    GradientMSELoss — MSE + gradient-matching regulariser for spatially smooth risk surfaces
    SmoothMSELoss   — MSE + spatial gradient-matching + Pearson correlation reward

Usage:
    from losses import FocalBCELoss, DiceBCELoss, GradientMSELoss, SmoothMSELoss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Numerical safety constant — prevents log(0) in BCE
_EPS: float = 1e-7


class FocalBCELoss(nn.Module):
    """Binary cross-entropy with focal weighting for class imbalance.

    Designed for fire detection and forest loss segmentation where
    positive pixels are sparse (<5% of the image).

    Predictions are clamped to [ε, 1-ε] to prevent log(0) → NaN,
    which is critical when using Automatic Mixed Precision (AMP).

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Parameters
    ----------
    alpha : float
        Balancing factor for positive class (default 0.75).
    gamma : float
        Focusing parameter — higher values down-weight easy examples
        more aggressively (default 2.0).
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = pred.clamp(_EPS, 1.0 - _EPS)
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        p_t = pred * target + (1.0 - pred) * (1.0 - target)
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for segmentation with fragmented patches.

    Dice loss handles class imbalance better than pure BCE for
    segmentation tasks where positive regions are small and fragmented.
    Predictions are clamped for numerical safety under AMP.

    Parameters
    ----------
    dice_weight : float
        Weight given to Dice loss vs BCE (default 0.5).
    smooth : float
        Laplace smoothing constant for Dice denominator (default 1.0).
    """

    def __init__(self, dice_weight: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = pred.clamp(_EPS, 1.0 - _EPS)
        bce = F.binary_cross_entropy(pred, target)

        pred_flat = pred.flatten()
        target_flat = target.flatten()
        intersection = (pred_flat * target_flat).sum()
        dice = 1.0 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        return bce * (1.0 - self.dice_weight) + dice * self.dice_weight


class GradientMSELoss(nn.Module):
    """MSE + spatial gradient-matching regulariser for smooth risk surfaces.

    Penalises the MSE between the spatial gradients of prediction and
    target, encouraging the model to learn correct spatial patterns
    (not just pixel-wise values).

    Parameters
    ----------
    grad_weight : float
        Weight of the gradient-matching term (default 0.3).
    """

    def __init__(self, grad_weight: float = 0.3) -> None:
        super().__init__()
        self.grad_weight = grad_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        mse = F.mse_loss(pred, target)

        # Spatial gradients (finite differences)
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        grad_loss = F.mse_loss(pred_dx, tgt_dx) + F.mse_loss(pred_dy, tgt_dy)

        return mse + self.grad_weight * grad_loss


class SmoothMSELoss(nn.Module):
    """MSE + spatial gradient-matching + Pearson correlation reward.

    Unlike a naive total-variation penalty (which pushes predictions
    towards a constant value and causes mode collapse), this loss
    matches prediction gradients to target gradients — preserving
    spatial structure while encouraging smoothness where the target
    is smooth.

    The Pearson correlation term rewards maintaining the correct
    relative ordering of risk values across the spatial domain.

    Parameters
    ----------
    grad_weight : float
        Weight of the gradient-matching term (default 0.1).
        Kept low to prevent gradient penalty from dominating.
    corr_weight : float
        Weight of the correlation loss (1 - r) term (default 0.2).
    """

    def __init__(self, grad_weight: float = 0.1, corr_weight: float = 0.2) -> None:
        super().__init__()
        self.grad_weight = grad_weight
        self.corr_weight = corr_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        mse = F.mse_loss(pred, target)

        # Gradient-matching (not gradient-suppressing)
        # Penalises |grad(pred) - grad(target)|, NOT |grad(pred)|
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        grad_loss = F.mse_loss(pred_dy, tgt_dy) + F.mse_loss(pred_dx, tgt_dx)

        # Pearson correlation reward
        p = pred.flatten()
        t = target.flatten()
        p_centered = p - p.mean()
        t_centered = t - t.mean()
        p_norm = p_centered.norm()
        t_norm = t_centered.norm()

        # Guard against zero-variance predictions (edge case during early training)
        if p_norm < _EPS or t_norm < _EPS:
            corr_loss = torch.ones(1, device=pred.device, dtype=pred.dtype)
        else:
            corr = (p_centered * t_centered).sum() / (p_norm * t_norm)
            corr = torch.clamp(corr, -1.0, 1.0)
            corr_loss = 1.0 - corr

        return mse + self.grad_weight * grad_loss + self.corr_weight * corr_loss


class DeepSupervisionWrapper(nn.Module):
    """Wraps any base loss to incorporate UNet++ deep supervision outputs.

    When deep supervision outputs are provided, auxiliary losses from
    intermediate UNet++ nodes are weighted and added to the main loss.
    This improves convergence by providing gradient signal at multiple
    decoder depths.

    Parameters
    ----------
    base_loss : nn.Module
        The primary loss function to use for both main and auxiliary outputs.
    aux_weight : float
        Weight applied to each auxiliary loss (default 0.3).
        Total aux contribution = aux_weight × Σ(aux_losses) / N_aux.
    """

    def __init__(self, base_loss: nn.Module, aux_weight: float = 0.3) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.aux_weight = aux_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        deep_outputs: list[Tensor] | None = None,
    ) -> Tensor:
        main_loss = self.base_loss(pred, target)

        if deep_outputs is None or len(deep_outputs) == 0:
            return main_loss

        # Auxiliary losses — resize target to match each auxiliary output
        aux_total = torch.zeros(1, device=pred.device, dtype=pred.dtype)
        for aux_pred in deep_outputs:
            if aux_pred.shape != target.shape:
                # Deep supervision outputs may be at a different resolution
                target_resized = F.interpolate(
                    target, size=aux_pred.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            else:
                target_resized = target
            aux_total = aux_total + self.base_loss(aux_pred, target_resized)

        aux_mean = aux_total / len(deep_outputs)
        return main_loss + self.aux_weight * aux_mean


# ═══════════════════════════════════════════════════════════════════════════
# Loss registry for training scripts
# ═══════════════════════════════════════════════════════════════════════════

LOSS_REGISTRY = {
    "focal_bce": FocalBCELoss,
    "dice_bce": DiceBCELoss,
    "gradient_mse": GradientMSELoss,
    "smooth_mse": SmoothMSELoss,
}


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing all loss functions...\n")

    pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
    target = (torch.rand(2, 1, 64, 64) > 0.7).float()

    for name, LossClass in LOSS_REGISTRY.items():
        loss_fn = LossClass()
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss), f"NaN in {name}!"
        assert not torch.isinf(loss), f"Inf in {name}!"
        print(f"  {name:15s}: {loss.item():.6f}")

    # Test edge case: all-zero predictions (AMP worst case)
    print("\nEdge case — near-zero predictions:")
    pred_zero = torch.full((2, 1, 64, 64), 1e-8)
    for name, LossClass in LOSS_REGISTRY.items():
        loss_fn = LossClass()
        loss = loss_fn(pred_zero, target)
        assert not torch.isnan(loss), f"NaN in {name} with zero preds!"
        print(f"  {name:15s}: {loss.item():.6f}")

    # Test edge case: all-one predictions
    print("\nEdge case — near-one predictions:")
    pred_one = torch.full((2, 1, 64, 64), 1.0 - 1e-8)
    for name, LossClass in LOSS_REGISTRY.items():
        loss_fn = LossClass()
        loss = loss_fn(pred_one, target)
        assert not torch.isnan(loss), f"NaN in {name} with one preds!"
        print(f"  {name:15s}: {loss.item():.6f}")

    # Test DeepSupervisionWrapper
    print("\nDeepSupervisionWrapper test:")
    base = FocalBCELoss()
    ds_loss = DeepSupervisionWrapper(base, aux_weight=0.3)
    deep = [torch.sigmoid(torch.randn(2, 1, 32, 32)) for _ in range(3)]
    loss = ds_loss(pred, target, deep)
    assert not torch.isnan(loss), "NaN in deep supervision!"
    print(f"  With deep supervision: {loss.item():.6f}")
    loss_no_ds = ds_loss(pred, target, None)
    print(f"  Without deep supervision: {loss_no_ds.item():.6f}")

    print("\n✓ All loss function tests passed")
