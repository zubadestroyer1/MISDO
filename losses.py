"""
MISDO — Production-Grade Loss Functions
==========================================
Consolidated loss module for all domain-specific risk models.
All loss functions include numerical safety guards (prediction clamping,
epsilon-protected divisions) to prevent NaN under mixed-precision training.

Loss Functions:
    FocalBCELoss            — Focal-weighted binary cross-entropy for sparse events
    DiceBCELoss             — Combined Dice + BCE for fragmented segmentation
    GradientMSELoss         — MSE + gradient-matching regulariser for smooth risk surfaces
    SmoothMSELoss           — MSE + gradient-matching + Pearson correlation reward
    SSIMLoss                — Differentiable Structural Similarity Index loss
    EdgeWeightedMSELoss     — Charbonnier + SSIM + edge upweighting + gradient matching
    CounterfactualDeltaLoss — Main training loss: EdgeWeightedMSE on delta + monotonicity penalty
    DeepSupervisionWrapper  — Wraps any loss to add auxiliary losses from UNet++ nodes

Usage:
    from losses import (
        CounterfactualDeltaLoss, DeepSupervisionWrapper, EdgeWeightedMSELoss,
        FocalBCELoss, DiceBCELoss, GradientMSELoss, SmoothMSELoss, SSIMLoss,
    )
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Numerical safety constant — prevents log(0) in BCE
_EPS: float = 1e-7

# Charbonnier epsilon — controls the L1/L2 transition point.
# At ε = 1e-3, errors > ~0.001 get near-L1 gradient (outlier-robust),
# while errors < ~0.001 get smooth L2 gradient (stable near zero).
# This is standard in image super-resolution (EDSR, SRResNet).
_CHARB_EPS: float = 1e-3


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
        # Per-class alpha weighting (Lin et al. 2017): alpha for positives,
        # (1 - alpha) for negatives.  With alpha=0.75 this gives a 3:1
        # positive-to-negative weight ratio to counteract class imbalance.
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
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


# ═══════════════════════════════════════════════════════════════════════════
# SSIM — Structural Similarity Index
# ═══════════════════════════════════════════════════════════════════════════

def _gaussian_kernel_2d(kernel_size: int = 11, sigma: float = 1.5) -> Tensor:
    """Create a 2D Gaussian kernel for SSIM computation.

    Returns a normalised [1, 1, K, K] kernel suitable for F.conv2d.
    The separable construction (outer product of 1-D Gaussians) is
    mathematically identical to a direct 2-D Gaussian but numerically
    cleaner for small kernels.
    """
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g1d = torch.exp(-coords ** 2 / (2.0 * sigma ** 2))
    g2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)  # outer product
    g2d = g2d / g2d.sum()  # normalise to sum=1
    return g2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]


class SSIMLoss(nn.Module):
    """Differentiable Structural Similarity Index (SSIM) loss.

    Measures local luminance, contrast, and structural correlation
    between prediction and target using Gaussian-weighted sliding
    windows.  Returns ``1 - mean(SSIM_map)`` so that lower = better.

    Implemented from scratch with pure PyTorch — no external
    dependencies.  Numerically safe under AMP (float16) via epsilon
    guards in the denominator.

    Reference:
        Wang et al., "Image Quality Assessment: From Error Visibility
        to Structural Similarity", IEEE TIP 2004.

    Parameters
    ----------
    kernel_size : int
        Size of the Gaussian sliding window (default 11).
    sigma : float
        Standard deviation of the Gaussian kernel (default 1.5).
    data_range : float
        Dynamic range of the input (default 1.0 for [0, 1] data).
    """

    def __init__(
        self,
        kernel_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.data_range = data_range

        # Stability constants (Wang et al. 2004)
        self.C1 = (0.01 * data_range) ** 2  # 1e-4 for range=1.0
        self.C2 = (0.03 * data_range) ** 2  # 9e-4 for range=1.0

        # Register the Gaussian kernel as a persistent buffer so it
        # moves to the correct device/dtype with the module.
        kernel = _gaussian_kernel_2d(kernel_size, sigma)
        self.register_buffer("_kernel", kernel)

    def _ssim_map(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute per-pixel SSIM map (values in [-1, 1])."""
        pad = self.kernel_size // 2
        C = pred.shape[1]  # number of channels (typically 1)

        # Expand kernel to match input channels (groups=C for depthwise conv)
        # Cast to input dtype for AMP float16 compatibility — the registered
        # buffer is float32 but inputs may arrive as float16 under autocast.
        kernel = self._kernel.to(dtype=pred.dtype).expand(C, 1, -1, -1)

        # Local means via Gaussian-weighted convolution
        mu_p = F.conv2d(pred, kernel, padding=pad, groups=C)
        mu_t = F.conv2d(target, kernel, padding=pad, groups=C)

        mu_p_sq = mu_p * mu_p
        mu_t_sq = mu_t * mu_t
        mu_pt = mu_p * mu_t

        # Local variances and covariance
        # Var(X) = E[X²] - E[X]²  via the Gaussian-weighted window
        sigma_p_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu_p_sq
        sigma_t_sq = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu_t_sq
        sigma_pt = F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu_pt

        # Clamp variances to zero (numerical noise can make them slightly negative)
        sigma_p_sq = sigma_p_sq.clamp(min=0.0)
        sigma_t_sq = sigma_t_sq.clamp(min=0.0)

        # SSIM formula (Wang et al. 2004, Eq. 13)
        numerator = (2.0 * mu_pt + self.C1) * (2.0 * sigma_pt + self.C2)
        denominator = (mu_p_sq + mu_t_sq + self.C1) * (sigma_p_sq + sigma_t_sq + self.C2)

        return numerator / denominator

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute SSIM loss = 1 - mean(SSIM_map)."""
        ssim_map = self._ssim_map(pred, target)
        return 1.0 - ssim_map.mean()


class EdgeWeightedMSELoss(nn.Module):
    """Charbonnier + SSIM + gradient-matching with edge upweighting.

    The primary loss for counterfactual impact prediction.  Combines:

    1. **Charbonnier pixel-wise loss** (√(error² + ε²)) — more robust
       to outliers than MSE, produces sharper edges on sparse targets.
       Standard in image super-resolution (EDSR, SRResNet, SwinIR).
    2. **SSIM structural similarity** — captures local luminance,
       contrast, and structure correlation in sliding windows.
    3. **Edge upweighting** — 3–5× weight on pixels near deforestation
       boundaries where the impact signal is strongest.
    4. **Gradient matching** — penalises differences in spatial gradients
       to encourage correct spatial patterns.

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
    ssim_weight : float
        Weight of the SSIM structural similarity term (default 0.1).
    base_weight : float
        Weight for non-edge pixels (default 1.0).
    charb_eps : float
        Charbonnier epsilon controlling L1/L2 transition (default 1e-3).
    focal_gamma : float
        Focal exponent for down-weighting easy background pixels.
        Higher values push more gradient budget toward hard impact-zone
        pixels where the model most needs to learn (default 2.0).
    """

    def __init__(
        self,
        edge_weight: float = 3.0,
        edge_radius: int = 5,
        grad_weight: float = 0.2,
        ssim_weight: float = 0.1,
        base_weight: float = 1.0,
        charb_eps: float = _CHARB_EPS,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.edge_weight = edge_weight
        self.edge_radius = edge_radius
        self.grad_weight = grad_weight
        self.ssim_weight = ssim_weight
        self.base_weight = base_weight
        self.charb_eps = charb_eps
        self.focal_gamma = focal_gamma

        # Dilation kernel for edge detection (max pool approximates dilation)
        self._kernel_size = 2 * edge_radius + 1

        # SSIM module for structural similarity (only created if weight > 0)
        if ssim_weight > 0:
            self.ssim_loss = SSIMLoss(kernel_size=11, sigma=1.5, data_range=1.0)
        else:
            self.ssim_loss = None

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
        # ── Weighted Charbonnier loss ──
        # Charbonnier: sqrt((pred - target)² + ε²)
        # Behaves like L2 for small errors (smooth gradient at zero),
        # and like L1 for large errors (robust to outliers, sharper edges).
        weight_map = self._compute_edge_mask(target)
        error_sq = (pred - target) ** 2
        charbonnier = torch.sqrt(error_sq + self.charb_eps ** 2)

        # Focal modulation: down-weight easy background pixels.
        # For target≈0, pred≈0 pixels: focal_mod → 1^γ → 1 (no change)
        # would naively be strongest. Instead, we use target magnitude
        # as the "easy" indicator: pixels with higher target signal
        # get full weight, while background pixels (target≈0) that the
        # model already predicts correctly get reduced weight.
        # focal_mod = (target + ε)^γ ensures impact-zone pixels get
        # full gradient while zero-background pixels get scaled down.
        if self.focal_gamma > 0:
            # Scale between [ε^γ, 1.0] — impact pixels get ~1.0
            focal_mod = (target.clamp(0, 1) + 1e-6) ** self.focal_gamma
            # Normalise so the max modulator is 1.0
            focal_mod = focal_mod / (focal_mod.max() + 1e-8)
            # Ensure minimum weight so background still contributes
            focal_mod = focal_mod.clamp(min=0.05)
            pixel_loss = (weight_map * focal_mod * charbonnier).mean()
        else:
            pixel_loss = (weight_map * charbonnier).mean()

        # ── SSIM structural similarity ──
        if self.ssim_loss is not None:
            ssim_term = self.ssim_loss(pred, target)
        else:
            ssim_term = 0.0

        # ── Gradient matching ──
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        grad_loss = F.mse_loss(pred_dy, tgt_dy) + F.mse_loss(pred_dx, tgt_dx)

        return pixel_loss + self.ssim_weight * ssim_term + self.grad_weight * grad_loss


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

        # Auxiliary losses — resize target to match each auxiliary output.
        # Note: kwargs (out_factual etc.) are NOT passed to auxiliaries —
        # monotonicity is only enforced on the main output.
        #
        # We use nearest-neighbor interpolation for downscaling so that
        # sparse signal pixels (target > 0.01) retain their values exactly
        # rather than being blurred below the EdgeWeightedMSELoss threshold
        # by bilinear averaging.  This preserves edge-mask quality at lower
        # auxiliary resolutions.
        aux_total = torch.zeros(1, device=pred.device, dtype=pred.dtype)
        for aux_pred in deep_outputs:
            if aux_pred.shape != target.shape:
                # Deep supervision outputs may be at a different resolution
                target_resized = F.interpolate(
                    target, size=aux_pred.shape[2:],
                    mode="nearest",
                )
            else:
                target_resized = target
            aux_total = aux_total + self.base_loss(aux_pred, target_resized)

        aux_mean = aux_total / len(deep_outputs)
        return main_loss + self.aux_weight * aux_mean

class CounterfactualDeltaLoss(nn.Module):
    """Loss for Siamese counterfactual training.

    Combines EdgeWeightedMSELoss (or a custom base_loss) on the predicted
    delta with three regularisation terms:

    1. **Monotonicity penalty** — deforestation should NOT decrease risk,
       so counterfactual output should be >= factual output.
    2. **Baseline grounding** — the factual (no-deforestation) branch
       should predict near-zero impact, anchoring the network to a
       physically meaningful reference and preserving sigmoid dynamic
       range for the counterfactual branch.  Grounded in TARNet
       (Shalit et al., ICML 2017) treatment-effect regularisation.

    Without grounding, out_f can drift to any value in (0, 1) because
    only the delta (out_cf − out_f) is supervised.  If out_f ≈ 0.6,
    then out_cf is capped near 1.0 by sigmoid saturation, limiting
    the maximum expressible delta to ~0.4 and reducing gradient flow
    by up to 81% in the counterfactual branch.

    Parameters
    ----------
    base_loss : nn.Module | None
        Custom base loss. If None, defaults to EdgeWeightedMSELoss.
    edge_weight : float
        Upweight pixels near deforestation edges if using default MSE (default 3.0).
    mono_weight : float
        Weight of the monotonicity penalty (default 0.1).
    grounding_weight : float
        Weight of the L1 grounding loss on the factual branch (default 0.05).
        Anchors out_f toward zero to preserve sigmoid dynamic range for
        out_cf and enforce counterfactual consistency (zero treatment →
        zero effect).  Uses L1 (not L2) for constant gradient magnitude
        regardless of current out_f value.
    """

    def __init__(
        self,
        base_loss: nn.Module | None = None,
        edge_weight: float = 3.0,
        mono_weight: float = 0.1,
        grounding_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss if base_loss is not None else EdgeWeightedMSELoss(edge_weight=edge_weight)
        self.mono_weight = mono_weight
        self.grounding_weight = grounding_weight

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
            Raw factual output (for monotonicity and grounding penalties).
        out_counterfactual : Tensor, optional
            Raw counterfactual output (for monotonicity penalty).
        """
        loss = self.base_loss(pred_delta, target_delta)

        if out_factual is not None and out_counterfactual is not None:
            # Monotonicity penalty: cf should >= f
            violation = F.relu(out_factual - out_counterfactual)
            mono_penalty = violation.mean()
            loss = loss + self.mono_weight * mono_penalty

        # Baseline grounding: the factual (no-deforestation) branch
        # output should be near zero.  Physically, "no deforestation"
        # should produce "zero impact delta baseline."
        #
        # This removes the translational degree of freedom that allows
        # out_f to drift away from zero (wasting sigmoid dynamic range
        # that out_cf needs) and enforces counterfactual consistency.
        #
        # Uses L1 (abs().mean()) for constant gradient magnitude
        # regardless of current out_f value — L2 would barely act
        # when out_f is small and act aggressively when large,
        # creating training instability during early epochs.
        if out_factual is not None and self.grounding_weight > 0:
            grounding = out_factual.abs().mean()
            loss = loss + self.grounding_weight * grounding

        return loss


# ═══════════════════════════════════════════════════════════════════════════
# Loss registry for training scripts
# ═══════════════════════════════════════════════════════════════════════════

LOSS_REGISTRY = {
    "focal_bce": FocalBCELoss,
    "dice_bce": DiceBCELoss,
    "gradient_mse": GradientMSELoss,
    "smooth_mse": SmoothMSELoss,
    "ssim": SSIMLoss,
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
        print(f"  {name:20s}: {loss.item():.6f}")

    # Test edge case: all-zero predictions (AMP worst case)
    print("\nEdge case — near-zero predictions:")
    pred_zero = torch.full((2, 1, 64, 64), 1e-8)
    for name, LossClass in LOSS_REGISTRY.items():
        loss_fn = LossClass()
        loss = loss_fn(pred_zero, target)
        assert not torch.isnan(loss), f"NaN in {name} with zero preds!"
        print(f"  {name:20s}: {loss.item():.6f}")

    # Test edge case: all-one predictions
    print("\nEdge case — near-one predictions:")
    pred_one = torch.full((2, 1, 64, 64), 1.0 - 1e-8)
    for name, LossClass in LOSS_REGISTRY.items():
        loss_fn = LossClass()
        loss = loss_fn(pred_one, target)
        assert not torch.isnan(loss), f"NaN in {name} with one preds!"
        print(f"  {name:20s}: {loss.item():.6f}")

    # ── SSIM correctness tests ──
    print("\n── SSIM correctness tests ──")
    ssim = SSIMLoss()
    # Perfect match → SSIM = 1.0, loss = 0.0
    ssim_perfect = ssim(pred, pred)
    assert abs(ssim_perfect.item()) < 1e-5, (
        f"SSIM(pred, pred) should be ~0, got {ssim_perfect.item():.6f}"
    )
    print(f"  SSIM(pred, pred):   {ssim_perfect.item():.6f}  (expected ~0.0)")
    # Random pair → SSIM ≈ 0, loss ≈ 1.0
    pred2 = torch.sigmoid(torch.randn(2, 1, 64, 64))
    ssim_random = ssim(pred, pred2)
    assert ssim_random.item() > 0.3, (
        f"SSIM(pred, random) should be large, got {ssim_random.item():.6f}"
    )
    print(f"  SSIM(pred, random): {ssim_random.item():.6f}  (expected ~0.5–1.0)")
    # All zeros → should not NaN
    ssim_zero = ssim(pred_zero, pred_zero)
    assert not torch.isnan(ssim_zero), "SSIM NaN on zero inputs!"
    print(f"  SSIM(zero, zero):   {ssim_zero.item():.6f}  (expected ~0.0)")

    # ── Charbonnier vs MSE comparison ──
    print("\n── Charbonnier behaviour test ──")
    ew = EdgeWeightedMSELoss(ssim_weight=0.0)  # Charbonnier only, no SSIM
    loss_charb = ew(pred, target)
    assert not torch.isnan(loss_charb), "NaN in Charbonnier!"
    print(f"  EdgeWeightedMSE (Charbonnier, no SSIM): {loss_charb.item():.6f}")
    ew_ssim = EdgeWeightedMSELoss(ssim_weight=0.1)
    loss_charb_ssim = ew_ssim(pred, target)
    assert not torch.isnan(loss_charb_ssim), "NaN in Charbonnier+SSIM!"
    print(f"  EdgeWeightedMSE (Charbonnier + SSIM):   {loss_charb_ssim.item():.6f}")
    # SSIM component should add to the loss
    assert loss_charb_ssim > loss_charb, (
        "Adding SSIM should increase loss (structural imperfection adds penalty)"
    )

    # ── DeepSupervisionWrapper test ──
    print("\nDeepSupervisionWrapper test (FocalBCE):")
    base = FocalBCELoss()
    ds_loss = DeepSupervisionWrapper(base, aux_weight=0.3)
    deep = [torch.sigmoid(torch.randn(2, 1, 32, 32)) for _ in range(3)]
    loss = ds_loss(pred, target, deep)
    assert not torch.isnan(loss), "NaN in deep supervision!"
    print(f"  With deep supervision: {loss.item():.6f}")
    loss_no_ds = ds_loss(pred, target, None)
    print(f"  Without deep supervision: {loss_no_ds.item():.6f}")

    # ── CounterfactualDeltaLoss monotonicity test ──
    print("\nCounterfactualDeltaLoss monotonicity test:")
    cf_loss = CounterfactualDeltaLoss()
    out_f = torch.sigmoid(torch.randn(2, 1, 64, 64))
    out_cf = torch.sigmoid(torch.randn(2, 1, 64, 64))
    loss_mono = cf_loss(pred, target, out_factual=out_f, out_counterfactual=out_cf)
    assert not torch.isnan(loss_mono), "NaN in CounterfactualDeltaLoss with monotonicity!"
    loss_no_mono = cf_loss(pred, target)
    assert not torch.isnan(loss_no_mono), "NaN in CounterfactualDeltaLoss without monotonicity!"
    print(f"  With monotonicity penalty:    {loss_mono.item():.6f}")
    print(f"  Without monotonicity penalty: {loss_no_mono.item():.6f}")
    assert loss_mono >= loss_no_mono, "Monotonicity penalty should increase loss!"

    # ── Production combination: DS + CounterfactualDelta + EdgeWeightedMSE ──
    print("\nProduction loss combination test (DS + CounterfactualDelta):")
    cf_base = CounterfactualDeltaLoss(
        base_loss=EdgeWeightedMSELoss(edge_weight=5.0, ssim_weight=0.15),
    )
    ds_cf = DeepSupervisionWrapper(cf_base, aux_weight=0.3)
    deep_deltas = [torch.sigmoid(torch.randn(2, 1, 32, 32)) for _ in range(3)]
    loss_prod = ds_cf(
        pred, target, deep_deltas,
        out_factual=out_f, out_counterfactual=out_cf,
    )
    assert not torch.isnan(loss_prod), "NaN in production loss combination!"
    print(f"  Full production loss (fire-like config): {loss_prod.item():.6f}")
    loss_prod_no_aux = ds_cf(
        pred, target, None,
        out_factual=out_f, out_counterfactual=out_cf,
    )
    print(f"  Without deep supervision: {loss_prod_no_aux.item():.6f}")

    # ── AMP float16 stability test ──
    print("\n── AMP float16 stability test ──")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    pred_f16 = pred.to(device).half()
    target_f16 = target.to(device).half()
    for name, LossClass in LOSS_REGISTRY.items():
        loss_fn = LossClass().to(device)
        loss = loss_fn(pred_f16, target_f16)
        assert not torch.isnan(loss), f"NaN in {name} under float16!"
        assert not torch.isinf(loss), f"Inf in {name} under float16!"
        print(f"  {name:20s}: {loss.item():.6f} (float16)")

    print("\n✓ All loss function tests passed")
