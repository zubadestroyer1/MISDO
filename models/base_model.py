"""
DomainRiskNet — Base Class for All Domain-Specific Risk Models
================================================================
Shared encoder-decoder architecture that eliminates code duplication
across Fire, Forest, Hydro, and Soil models.

Architecture:
    Encoder  : ConvNeXt-V2 Base (96→192→384→768), ~23M params
    Temporal : Multi-head self-attention on bottleneck + learned
               skip-level temporal fusion
    Decoder  : UNet++ with nested dense skip connections, ~5M params

Subclasses only need to set IN_CHANNELS.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.backbone import ConvNeXtV2Backbone
from models.decoders import UNetPPDecoder
from models.temporal import TemporalAttention, TemporalSkipFusion


# Encoder dimensions (ConvNeXt-V2 Base config)
_DIMS: Tuple[int, ...] = (96, 192, 384, 768)
_DEPTHS: Tuple[int, ...] = (3, 3, 9, 3)


class DomainRiskNet(nn.Module):
    """Base class for ConvNeXt-V2 + UNet++ domain risk models.

    Supports both temporal [B, T, C, H, W] and single-frame [B, C, H, W].

    Subclasses only need to set the class attribute ``IN_CHANNELS``.

    Parameters
    ----------
    in_channels : int
        Number of input channels for this domain's satellite data.
    """

    IN_CHANNELS: int  # must be set by subclass
    TEMPORAL: bool = True  # subclasses can set False to skip temporal modules
    BOTTLENECK_CHANNELS: int = _DIMS[-1]  # 768

    def __init__(self, in_channels: int | None = None) -> None:
        super().__init__()
        ch = in_channels if in_channels is not None else self.IN_CHANNELS

        # ── Encoder ──
        self.encoder = ConvNeXtV2Backbone(
            in_channels=ch,
            dims=_DIMS,
            depths=_DEPTHS,
            # CRITICAL: In a Siamese counterfactual setup, we compute delta = cf - f.
            # If drop_path rate > 0, timm applies different random dropout masks to
            # the factual and counterfactual branches. The resulting 'delta' is dominated
            # by dropout noise. Because targets are 99.9% zeros, the network learns to
            # squash all outputs to zero to silence this noise, freezing the model.
            drop_path_rate=0.0,
            pretrained=True,
        )

        # ── Temporal modules (only when needed) ──
        if self.TEMPORAL:
            self.temporal_attn = TemporalAttention(_DIMS[-1])
            self.skip_temporal = nn.ModuleDict({
                f"s{i+1}": TemporalSkipFusion(channels=_DIMS[i])
                for i in range(3)  # s1, s2, s3 (s4 is handled by temporal_attn)
            })
        else:
            self.temporal_attn = None
            self.skip_temporal = None

        # ── Decoder ──
        self.decoder = UNetPPDecoder(
            encoder_dims=_DIMS,
            decoder_dim=128,
            deep_supervision=True,
        )

    def _run_encoder(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Run encoder, return (bottleneck, skip_dict)."""
        features = self.encoder(x)
        bottleneck = features["s4"]
        skips = {
            "s1": features["s1"],
            "s2": features["s2"],
            "s3": features["s3"],
        }
        return bottleneck, skips

    def encode(self, x: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        """Public encode: handles temporal dim, returns (bottleneck, skips).

        For temporal inputs [B, T, C, H, W]:
            - Bottleneck is fused via multi-head self-attention
            - Skip connections are fused via learned temporal weighting
              (instead of discarding all but the last frame)

        Returns
        -------
        bottleneck : Tensor [B, 768, H/32, W/32]
        skips : dict with keys 's1', 's2', 's3'
        """
        if x.dim() == 5:
            if self.temporal_attn is None:
                raise ValueError(
                    f"{self.__class__.__name__} received 5D temporal input "
                    f"but TEMPORAL=False. Use [B, C, H, W] input instead."
                )
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
            b_all, skips_all = self._run_encoder(x_flat)

            # Temporal attention on bottleneck
            _, Cb, Hb, Wb = b_all.shape
            b_temporal = b_all.view(B, T, Cb, Hb, Wb)
            b_fused = self.temporal_attn(b_temporal)

            # Temporal fusion on skip connections (learned weighting, not just last frame)
            skips = {}
            for key, val in skips_all.items():
                val_temporal = val.view(B, T, *val.shape[1:])  # [B, T, C, H, W]
                skips[key] = self.skip_temporal[key](val_temporal)

            return b_fused, skips
        else:
            return self._run_encoder(x)

    def decode(self, bottleneck: Tensor, skips: Dict[str, Tensor]) -> Tensor:
        """Decode from bottleneck + skips → [B, 1, H, W] risk mask."""
        features = {
            "s1": skips["s1"],
            "s2": skips["s2"],
            "s3": skips["s3"],
            "s4": bottleneck,
        }
        return self.decoder(features)

    def forward(
        self, x: Tensor, x_counterfactual: Tensor | None = None,
    ) -> Tensor | Tuple[Tensor, list, Tensor, Tensor]:
        """Forward pass — supports both single-input and Siamese paired modes.

        When ``x_counterfactual`` is provided, dispatches to
        ``forward_paired_deep`` which returns the full 4-tuple needed
        for training (delta, deep_deltas, out_f, out_cf).  This allows
        ``DataParallel``/``DistributedDataParallel`` wrappers (which only
        parallelise ``.forward()``) to properly split both inputs across
        GPUs.

        Parameters
        ----------
        x : Tensor
            Factual observation (or single input for non-Siamese mode).
        x_counterfactual : Tensor | None
            If provided, counterfactual observation for Siamese paired forward.
        """
        if x_counterfactual is not None:
            return self.forward_paired_deep(x, x_counterfactual)
        bottleneck, skips = self.encode(x)
        return self.decode(bottleneck, skips)

    # ── Siamese Counterfactual Interface ──────────────────────────────

    def forward_paired(
        self, x_factual: Tensor, x_counterfactual: Tensor
    ) -> Tensor:
        """Paired forward pass — independent branch decoding.

        Runs the encoder+decoder on both inputs independently, then
        computes the impact delta as (cf - f) in decoded output space.
        This ensures the decoder always processes normal feature
        statistics (all-positive from GELU), not zero-centered
        differences that would be killed by ReLU.

        Returns
        -------
        impact_delta : Tensor [B, 1, H, W]
            Clamped impact delta in [0, 1].
        """
        b_f, s_f = self.encode(x_factual)
        b_cf, s_cf = self.encode(x_counterfactual)

        # Decode each branch with normal feature statistics
        out_f = self.decode(b_f, s_f)
        out_cf = self.decode(b_cf, s_cf)

        # Delta = counterfactual - factual. 
        # We do not clamp here to ensure gradients flow freely if cf < f.
        delta = out_cf - out_f
        return delta

    def forward_paired_deep(
        self, x_factual: Tensor, x_counterfactual: Tensor
    ) -> Tuple[Tensor, list, Tensor, Tensor]:
        """Paired forward with deep supervision outputs for training.

        Both branches are decoded with deep supervision so that
        auxiliary deltas are computed at matched abstraction levels
        (aux_cf[i] - aux_f[i]), not from mismatched representations.

        Returns
        -------
        impact_delta : Tensor [B, 1, H, W]
            Clamped delta prediction.
        deep_deltas : list[Tensor]
            Auxiliary deltas from matched intermediate decoder nodes.
        out_factual : Tensor [B, 1, H, W]
            Raw decoded factual output (for monotonicity penalty).
        out_counterfactual : Tensor [B, 1, H, W]
            Raw decoded counterfactual output (for monotonicity penalty).
        """
        b_f, s_f = self.encode(x_factual)
        b_cf, s_cf = self.encode(x_counterfactual)

        # Decode BOTH branches with deep supervision so auxiliary
        # deltas are computed at matched abstraction levels.
        feat_f = {
            "s1": s_f["s1"], "s2": s_f["s2"],
            "s3": s_f["s3"], "s4": b_f,
        }
        feat_cf = {
            "s1": s_cf["s1"], "s2": s_cf["s2"],
            "s3": s_cf["s3"], "s4": b_cf,
        }

        result_f = self.decoder(feat_f, return_deep=True)
        result_cf = self.decoder(feat_cf, return_deep=True)

        if isinstance(result_f, tuple):
            out_f, deep_f = result_f
        else:
            out_f, deep_f = result_f, []

        if isinstance(result_cf, tuple):
            out_cf, deep_cf = result_cf
        else:
            out_cf, deep_cf = result_cf, []

        # Main delta = cf - f in output space
        delta = out_cf - out_f

        # Deep supervision deltas: matched intermediate outputs.
        # aux_cf[i] and aux_f[i] come from the same decoder node
        # (x01, x02, x12), so they encode features at the same
        # level of abstraction — gradient signal is clean.
        deep_deltas = []
        for aux_cf, aux_f in zip(deep_cf, deep_f):
            deep_deltas.append(aux_cf - aux_f)

        return delta, deep_deltas, out_f, out_cf


if __name__ == "__main__":
    # Quick test with a concrete subclass
    class TestNet(DomainRiskNet):
        IN_CHANNELS = 6

    model = TestNet()
    params = sum(p.numel() for p in model.parameters())
    print(f"DomainRiskNet(in=6) parameters: {params:,}")

    # Single frame
    x = torch.randn(1, 6, 256, 256)
    y = model(x)
    print(f"Single frame: {y.shape}  [{y.min():.3f}, {y.max():.3f}]")

    # Temporal
    x_t = torch.randn(1, 3, 6, 256, 256)
    y_t = model(x_t)
    print(f"Temporal (T=3): {y_t.shape}  [{y_t.min():.3f}, {y_t.max():.3f}]")

    # Encode/decode round-trip
    b, s = model.encode(x)
    print(f"Bottleneck: {b.shape}")
    y2 = model.decode(b, s)
    print(f"Decode: {y2.shape}  matches forward: {torch.allclose(y, y2)}")
