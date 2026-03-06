"""
MISDO — GradCAM Explainability for Risk Predictions
=======================================================
Provides visual explanations of *why* each pixel is predicted as
high or low risk. Essential for government procurement where
decisions must be auditable and interpretable.

Key API:
    GradCAM(model, target_layer)           — compute attribution maps
    generate_attribution_report(model, x)  — full explainability report

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.

Example output interpretation:
    "This area is flagged as high fire risk because of:
     - 73% attribution to deforestation edge exposure (channel 2)
     - 18% attribution to recent clearing activity (channel 1)
     - 9% attribution to terrain dryness (channel 3)"
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GradCAM:
    """Gradient-weighted Class Activation Mapping for dense prediction.

    Hooks into a target layer of the model to compute gradient-weighted
    activation maps showing which spatial regions contributed most to
    the model's predictions.

    For dense prediction (segmentation), the "target" is the average
    prediction value, or a specific target mask region.

    Parameters
    ----------
    model : nn.Module
        The risk prediction model.
    target_layer : nn.Module
        The layer to hook into (typically the bottleneck or last encoder stage).
        Default uses the last stage of the encoder.

    Usage
    -----
    >>> cam = GradCAM(model, model.encoder.stages[-1])
    >>> attribution = cam.generate(input_tensor)
    >>> cam.cleanup()  # remove hooks when done
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

        # Storage for hook outputs
        self._activations: Optional[Tensor] = None
        self._gradients: Optional[Tensor] = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, input: Tensor, output: Tensor) -> None:
        """Forward hook: save activations."""
        if isinstance(output, dict):
            # ConvNeXtV2Backbone returns a dict — get last stage
            output = list(output.values())[-1]
        self._activations = output.detach()

    def _save_gradient(self, module: nn.Module, grad_input: Tuple, grad_output: Tuple) -> None:
        """Backward hook: save gradients."""
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: Tensor,
        target_mask: Optional[Tensor] = None,
        upsample: bool = True,
    ) -> Tensor:
        """Generate GradCAM attribution map.

        Parameters
        ----------
        input_tensor : Tensor [B, C, H, W] or [B, T, C, H, W]
            Model input.
        target_mask : Tensor [B, 1, H, W], optional
            If provided, compute attributions for predictions within
            this mask only. If None, attributes to mean prediction.
        upsample : bool
            If True, upsample the attribution map to input spatial
            resolution via bilinear interpolation.

        Returns
        -------
        attribution : Tensor [B, H, W]
            Per-pixel attribution map normalised to [0, 1].
            High values = regions the model relied on most.
        """
        self.model.eval()

        # Enable gradients for this inference pass
        input_tensor = input_tensor.detach().requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]

        # Define target for backprop
        if target_mask is not None:
            # Attribution for specific region
            target = (output * target_mask).sum()
        else:
            # Attribution for mean predicted risk
            target = output.mean()

        # Backward pass to get gradients at target layer
        self.model.zero_grad()
        target.backward(retain_graph=False)

        if self._gradients is None or self._activations is None:
            raise RuntimeError(
                "Hooks did not fire. Ensure target_layer is in the model's forward path."
            )

        # GradCAM: weight activations by global-average-pooled gradients
        # gradients: [B, C, h, w], activations: [B, C, h, w]
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # [B, 1, h, w]

        # ReLU — only keep positive attributions
        cam = F.relu(cam)

        # Normalise per-sample to [0, 1]
        B = cam.shape[0]
        for b in range(B):
            cam_max = cam[b].max()
            if cam_max > 0:
                cam[b] = cam[b] / cam_max

        # Upsample to input resolution
        if upsample:
            if input_tensor.dim() == 5:
                H, W = input_tensor.shape[3], input_tensor.shape[4]
            else:
                H, W = input_tensor.shape[2], input_tensor.shape[3]
            cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)

        return cam.squeeze(1).detach()  # [B, H, W]

    def cleanup(self) -> None:
        """Remove hooks to allow normal model use."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()
        self._activations = None
        self._gradients = None


def _get_target_layer(model: nn.Module) -> nn.Module:
    """Automatically find the best layer for GradCAM in a DomainRiskNet.

    Prefers the last encoder stage (highest-level spatial features).
    """
    if hasattr(model, "encoder") and hasattr(model.encoder, "stages"):
        return model.encoder.stages[-1]
    elif hasattr(model, "encoder"):
        # Fallback: use the encoder itself
        return model.encoder
    else:
        raise ValueError(
            "Cannot auto-detect target layer. "
            "Pass target_layer explicitly to GradCAM()."
        )


def generate_attribution_report(
    model: nn.Module,
    obs: Tensor,
    target_mask: Optional[Tensor] = None,
    channel_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Generate a complete explainability report for a prediction.

    Parameters
    ----------
    model : nn.Module
        Trained domain risk model.
    obs : Tensor [B, C, H, W] or [B, T, C, H, W]
        Input observation tensor.
    target_mask : Tensor [B, 1, H, W], optional
        Focus attribution on specific regions.
    channel_names : list of str, optional
        Names for input channels (for reporting).

    Returns
    -------
    dict with:
        attribution_map : np.ndarray [H, W]
            Per-pixel attribution map (0 = not important, 1 = most important).
        top_regions : list of (row, col, score) tuples
            Top 10 highest-attribution pixel locations.
        mean_attribution : float
            Average attribution intensity.
        high_attribution_fraction : float
            Fraction of pixels with attribution > 0.5.
        channel_importance : dict (if channel_names provided)
            Per-channel importance scores computed via input gradient analysis.
    """
    target_layer = _get_target_layer(model)
    cam = GradCAM(model, target_layer)

    try:
        # Generate attribution map
        attr_map = cam.generate(obs, target_mask=target_mask, upsample=True)
        attr_np = attr_map[0].cpu().numpy()  # First sample [H, W]

        # Find top attribution regions
        flat_indices = np.argsort(attr_np.ravel())[::-1][:10]
        H, W = attr_np.shape
        top_regions = [
            (int(idx // W), int(idx % W), float(attr_np.ravel()[idx]))
            for idx in flat_indices
        ]

        report = {
            "attribution_map": attr_np,
            "top_regions": top_regions,
            "mean_attribution": float(attr_np.mean()),
            "high_attribution_fraction": float((attr_np > 0.5).mean()),
        }

        # Channel importance via input gradient analysis
        if channel_names is not None:
            channel_importance = _compute_channel_importance(model, obs, channel_names)
            report["channel_importance"] = channel_importance

    finally:
        cam.cleanup()

    return report


def _compute_channel_importance(
    model: nn.Module,
    obs: Tensor,
    channel_names: List[str],
) -> Dict[str, float]:
    """Compute per-channel importance via input gradient magnitude.

    For each input channel, measures how much the output changes
    when that channel's values change (gradient magnitude).
    """
    model.eval()
    obs_grad = obs.detach().clone().requires_grad_(True)

    output = model(obs_grad)
    if isinstance(output, tuple):
        output = output[0]
    output.mean().backward()

    if obs_grad.grad is None:
        return {name: 0.0 for name in channel_names}

    # Handle temporal [B, T, C, H, W] and single-frame [B, C, H, W]
    grad = obs_grad.grad.detach()
    if grad.dim() == 5:
        # Temporal: average over time and spatial dims
        importance = grad.abs().mean(dim=(0, 1, 3, 4))  # [C]
    else:
        importance = grad.abs().mean(dim=(0, 2, 3))  # [C]

    # Normalise to sum to 1
    total = importance.sum()
    if total > 0:
        importance = importance / total

    n_channels = min(len(channel_names), importance.shape[0])
    return {
        channel_names[i]: float(importance[i])
        for i in range(n_channels)
    }


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from models.fire_model import FireRiskNet

    print("Testing GradCAM Explainability...\n")

    model = FireRiskNet()
    model.eval()

    x = torch.randn(1, 6, 256, 256)

    # Test with channel names
    channel_names = ["I1_vis", "I2_NIR", "I3_SWIR", "I4_MIR", "I5_TIR", "FRP"]
    report = generate_attribution_report(model, x, channel_names=channel_names)

    print(f"  Attribution map shape: {report['attribution_map'].shape}")
    print(f"  Mean attribution: {report['mean_attribution']:.4f}")
    print(f"  High attribution fraction: {report['high_attribution_fraction']:.4f}")
    print(f"  Top 3 regions: {report['top_regions'][:3]}")

    if "channel_importance" in report:
        print(f"\n  Channel importance:")
        for ch, imp in sorted(
            report["channel_importance"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"    {ch:12s}: {imp:.4f} ({imp*100:.1f}%)")

    # Verify
    assert report["attribution_map"].shape == (256, 256)
    assert 0.0 <= report["attribution_map"].min()
    assert report["attribution_map"].max() <= 1.0 + 1e-6
    assert len(report["top_regions"]) == 10

    print("\n✓ All GradCAM explainability tests passed")
