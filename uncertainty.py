"""
MISDO — Uncertainty Quantification via MC Dropout
====================================================
Provides per-pixel confidence intervals on risk predictions by running
multiple stochastic forward passes with dropout active at inference time.

This is essential for government-grade decisions: a point prediction of
0.7 risk means nothing without knowing if the confidence interval is
[0.65, 0.75] (act on it) vs [0.3, 0.95] (need more data).

Key API:
    enable_mc_dropout(model, p=0.1)  — inject dropout into decoder
    predict_with_uncertainty(model, x, n_samples=20) — returns mean, std, CI

Reference:
    Gal & Ghahramani, "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning", ICML 2016.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor


def enable_mc_dropout(
    model: nn.Module,
    p: float = 0.1,
    target_modules: tuple = (nn.Conv2d,),
) -> int:
    """Inject Dropout2d layers after target modules in the decoder.

    Only modifies the decoder sub-network — the encoder features remain
    deterministic and are computed once, then the decoder is sampled
    multiple times for efficiency.

    Parameters
    ----------
    model : nn.Module
        A DomainRiskNet or similar model with a `decoder` attribute.
    p : float
        Dropout probability (default 0.1). Keep low to avoid
        destroying spatial structure; 0.05–0.15 is typical for
        dense prediction.
    target_modules : tuple of nn.Module types
        Insert dropout after these layer types.

    Returns
    -------
    n_inserted : int
        Number of dropout layers inserted.
    """
    if not hasattr(model, "decoder"):
        raise ValueError(
            "Model must have a 'decoder' attribute. "
            "Expected a DomainRiskNet or similar architecture."
        )

    n_inserted = 0
    decoder = model.decoder

    # Walk all sequential containers in the decoder and inject dropout
    for name, module in decoder.named_modules():
        if isinstance(module, nn.Sequential):
            new_layers = []
            for layer in module:
                new_layers.append(layer)
                if isinstance(layer, target_modules):
                    new_layers.append(_MCDropout2d(p))
                    n_inserted += 1
            # Replace the sequential with the augmented version
            if n_inserted > 0:
                # Rebuild the sequential in-place
                module.__init__(*new_layers)

    return n_inserted


class _MCDropout2d(nn.Module):
    """Dropout2d that stays active during both training AND inference.

    Standard nn.Dropout2d is disabled when model.eval() is called.
    This version always drops, which is required for Monte Carlo sampling.
    """

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        # Always apply dropout regardless of self.training
        if self.p <= 0.0:
            return x
        return nn.functional.dropout2d(x, p=self.p, training=True)

    def __repr__(self) -> str:
        return f"_MCDropout2d(p={self.p}, always_on=True)"


@torch.no_grad()
def predict_with_uncertainty(
    model: nn.Module,
    x: Tensor,
    n_samples: int = 20,
    confidence_level: float = 0.90,
) -> Dict[str, Tensor]:
    """Run MC Dropout inference and return uncertainty estimates.

    Performs multiple stochastic forward passes through the decoder
    while keeping the encoder deterministic. Returns per-pixel
    mean prediction, standard deviation, and credible intervals.

    Parameters
    ----------
    model : nn.Module
        Model with MC Dropout enabled via `enable_mc_dropout()`.
    x : Tensor
        Input tensor [B, C, H, W] or [B, T, C, H, W].
    n_samples : int
        Number of stochastic forward passes (default 20).
        More samples = tighter uncertainty estimates but slower.
        Recommended: 10 for quick checks, 30+ for production.
    confidence_level : float
        Confidence level for credible intervals (default 0.90).

    Returns
    -------
    dict with keys:
        mean : Tensor [B, 1, H, W]
            Average prediction across MC samples (more robust than single pass).
        std : Tensor [B, 1, H, W]
            Per-pixel standard deviation (epistemic uncertainty).
        confidence_lower : Tensor [B, 1, H, W]
            Lower bound of credible interval.
        confidence_upper : Tensor [B, 1, H, W]
            Upper bound of credible interval.
        entropy : Tensor [B, 1, H, W]
            Predictive entropy: -p*log(p) - (1-p)*log(1-p).
            High entropy = model is uncertain about the prediction.
        samples : Tensor [n_samples, B, 1, H, W]
            Raw MC samples (for advanced analysis).
    """
    was_training = model.training

    # Encode once (deterministic) — reuse for all MC samples
    # This is ~20× faster than encoding each time
    bottleneck, skips = model.encode(x)

    # Build features dict for decoder
    features = {
        "s1": skips["s1"],
        "s2": skips["s2"],
        "s3": skips["s3"],
        "s4": bottleneck,
    }

    # Collect MC samples by running the decoder multiple times
    # MC Dropout is always active due to _MCDropout2d
    samples = []
    for _ in range(n_samples):
        pred = model.decoder(features)
        if isinstance(pred, tuple):
            pred = pred[0]  # discard deep supervision outputs
        samples.append(pred)

    # Restore original mode
    model.train(was_training)

    # Stack and compute statistics
    stacked = torch.stack(samples, dim=0)  # [N, B, 1, H, W]
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)

    # Credible interval via percentiles
    alpha = (1.0 - confidence_level) / 2.0
    lower_pct = alpha * 100
    upper_pct = (1.0 - alpha) * 100

    # torch.quantile needs float
    stacked_float = stacked.float()
    confidence_lower = torch.quantile(stacked_float, alpha, dim=0).to(mean.dtype)
    confidence_upper = torch.quantile(stacked_float, 1.0 - alpha, dim=0).to(mean.dtype)

    # Predictive entropy: -p*log(p) - (1-p)*log(1-p)
    eps = 1e-7
    p_clamped = mean.clamp(eps, 1.0 - eps)
    entropy = -(p_clamped * p_clamped.log() + (1 - p_clamped) * (1 - p_clamped).log())

    return {
        "mean": mean,
        "std": std,
        "confidence_lower": confidence_lower,
        "confidence_upper": confidence_upper,
        "entropy": entropy,
        "samples": stacked,
    }


def uncertainty_summary(result: Dict[str, Tensor]) -> Dict[str, float]:
    """Compute scalar summary statistics from uncertainty output.

    Useful for logging and monitoring uncertainty calibration.
    """
    return {
        "mean_prediction": result["mean"].mean().item(),
        "mean_uncertainty": result["std"].mean().item(),
        "max_uncertainty": result["std"].max().item(),
        "mean_entropy": result["entropy"].mean().item(),
        "ci_width_mean": (
            result["confidence_upper"] - result["confidence_lower"]
        ).mean().item(),
        "ci_width_max": (
            result["confidence_upper"] - result["confidence_lower"]
        ).max().item(),
        "high_uncertainty_frac": (
            (result["std"] > 0.1).float().mean().item()
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from models.fire_model import FireRiskNet

    print("Testing MC Dropout Uncertainty Quantification...\n")

    model = FireRiskNet()
    n = enable_mc_dropout(model, p=0.1)
    print(f"  Injected {n} MC Dropout layers into decoder")

    x = torch.randn(1, 6, 256, 256)
    result = predict_with_uncertainty(model, x, n_samples=10)

    print(f"  Mean shape:     {result['mean'].shape}")
    print(f"  Std shape:      {result['std'].shape}")
    print(f"  CI lower shape: {result['confidence_lower'].shape}")
    print(f"  CI upper shape: {result['confidence_upper'].shape}")
    print(f"  Entropy shape:  {result['entropy'].shape}")
    print(f"  Samples shape:  {result['samples'].shape}")

    stats = uncertainty_summary(result)
    print(f"\n  Summary statistics:")
    for k, v in stats.items():
        print(f"    {k:25s}: {v:.6f}")

    # Verify correctness
    assert result["mean"].shape == (1, 1, 256, 256)
    assert result["std"].shape == (1, 1, 256, 256)
    assert (result["std"] >= 0).all(), "Std should be non-negative"
    assert (result["confidence_lower"] <= result["mean"]).all()
    assert (result["confidence_upper"] >= result["mean"]).all()
    assert (result["entropy"] >= 0).all(), "Entropy should be non-negative"

    print("\n✓ All MC Dropout uncertainty tests passed")
