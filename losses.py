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


class EdgeWeightedMSELoss(nn.Module):
    """MSE + gradient-matching with upweighting near deforestation edges.

    For counterfactual impact prediction, the strongest signal is in
    pixels immediately surrounding cleared areas.  This loss upweights
    those pixels to focus learning on the high-impact zone.

    Parameters
    ----------
    edge_weight : float
        Multiplier for pixels within ``edge_radius`` of the deforestation
        boundary (default 3.0 — 3× weight for edge pixels).
    edge_radius : int
        Number of dilation iterations to define the edge zone (default 5,
        ~150 m at 30 m resolution).
    grad_weight : float
        Weight of the gradient-matching term (default 0.2).
    base_weight : float
        Weight for non-edge pixels (default 1.0).
    """

    def __init__(
        self,
        edge_weight: float = 3.0,
        edge_radius: int = 5,
        grad_weight: float = 0.2,
        base_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.edge_weight = edge_weight
        self.edge_radius = edge_radius
        self.grad_weight = grad_weight
        self.base_weight = base_weight

        # Dilation kernel for edge detection (3×3 max pool approximates dilation)
        self._kernel_size = 2 * edge_radius + 1

    def _compute_edge_mask(self, target: Tensor) -> Tensor:
        """Compute a weight mask that upweights pixels near non-zero target regions.

        Uses max-pooling as a GPU-friendly approximation of binary dilation.
        """
        # Detect regions with significant impact signal
        # Threshold lowered to 0.01 to capture weak targets after
        # global target_scale normalisation (many targets < 0.05).
        has_signal = (target > 0.01).float()

        # Dilate to find edge zone
        dilated = F.max_pool2d(
            has_signal,
            kernel_size=self._kernel_size,
            stride=1,
            padding=self.edge_radius,
        )

        # Edge zone = dilated minus original signal area
        edge_zone = (dilated - has_signal).clamp(0, 1)

        # Weight map: base_weight everywhere + extra weight on edges + signal areas
        weight_map = (
            torch.ones_like(target) * self.base_weight
            + edge_zone * (self.edge_weight - self.base_weight)
            + has_signal * self.edge_weight
        )

        return weight_map

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # Weighted MSE
        weight_map = self._compute_edge_mask(target)
        mse = (weight_map * (pred - target) ** 2).mean()

        # Gradient matching
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        grad_loss = F.mse_loss(pred_dy, tgt_dy) + F.mse_loss(pred_dx, tgt_dx)

        return mse + self.grad_weight * grad_loss


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
        **kwargs,
    ) -> Tensor:
        main_loss = self.base_loss(pred, target, **kwargs)

        if deep_outputs is None or len(deep_outputs) == 0:
            return main_loss

        # Auxiliary losses — resize target to match each auxiliary output
        # Note: kwargs (out_factual etc.) are NOT passed to auxiliaries —
        # monotonicity is only enforced on the main output.
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

class CounterfactualDeltaLoss(nn.Module):
    """Loss for Siamese counterfactual training.

    Combines EdgeWeightedMSELoss (or a custom base_loss) on the predicted delta with:
    1. A monotonicity penalty — deforestation should NOT decrease risk,
       so counterfactual output should be >= factual output.
    2. Gradient matching on the delta itself.

    Parameters
    ----------
    base_loss : nn.Module | None
        Custom base loss. If None, defaults to EdgeWeightedMSELoss.
    edge_weight : float
        Upweight pixels near deforestation edges if using default MSE (default 3.0).
    mono_weight : float
        Weight of the monotonicity penalty (default 0.1).
    """

    def __init__(
        self,
        base_loss: nn.Module | None = None,
        edge_weight: float = 3.0,
        mono_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss if base_loss is not None else EdgeWeightedMSELoss(edge_weight=edge_weight)
        self.mono_weight = mono_weight

    def forward(
        self,
        pred_delta: Tensor,
        target_delta: Tensor,
        out_factual: Tensor | None = None,
        out_counterfactual: Tensor | None = None,
    ) -> Tensor:
        """Compute counterfactual loss.

        Parameters
        ----------
        pred_delta : Tensor [B, 1, H, W]
            Predicted impact delta (clamped cf - f).
        target_delta : Tensor [B, 1, H, W]
            Ground-truth impact delta.
        out_factual : Tensor, optional
            Raw factual output (for monotonicity penalty).
        out_counterfactual : Tensor, optional
            Raw counterfactual output (for monotonicity penalty).
        """
        loss = self.base_loss(pred_delta, target_delta)

        # Monotonicity penalty: cf should >= f
        if out_factual is not None and out_counterfactual is not None:
            violation = F.relu(out_factual - out_counterfactual)
            mono_penalty = violation.mean()
            loss = loss + self.mono_weight * mono_penalty

        return loss


# ═══════════════════════════════════════════════════════════════════════════
# Loss registry for training scripts
# ═══════════════════════════════════════════════════════════════════════════

LOSS_REGISTRY = {
    "focal_bce": FocalBCELoss,
    "dice_bce": DiceBCELoss,
    "gradient_mse": GradientMSELoss,
    "smooth_mse": SmoothMSELoss,
    "edge_weighted_mse": EdgeWeightedMSELoss,
    "counterfactual_delta": CounterfactualDeltaLoss,
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
