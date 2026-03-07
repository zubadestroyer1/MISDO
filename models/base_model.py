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
    BOTTLENECK_CHANNELS: int = _DIMS[-1]  # 768

    def __init__(self, in_channels: int | None = None) -> None:
        super().__init__()
        ch = in_channels if in_channels is not None else self.IN_CHANNELS

        # ── Encoder ──
        self.encoder = ConvNeXtV2Backbone(
            in_channels=ch,
            dims=_DIMS,
            depths=_DEPTHS,
            drop_path_rate=0.1,
        )

        # ── Temporal Attention (bottleneck) ──
        self.temporal_attn = TemporalAttention(_DIMS[-1])

        # ── Temporal Skip Fusion (skip connections) ──
        # Learns temporal weights for each skip scale instead of using
        # only the last timestep
        self.skip_temporal = nn.ModuleDict({
            f"s{i+1}": TemporalSkipFusion(channels=_DIMS[i])
            for i in range(3)  # s1, s2, s3 (s4 is handled by temporal_attn)
        })

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

    def forward(self, x: Tensor) -> Tensor:
        bottleneck, skips = self.encode(x)
        return self.decode(bottleneck, skips)

    # ── Siamese Counterfactual Interface ──────────────────────────────

    def forward_paired(
        self, x_factual: Tensor, x_counterfactual: Tensor
    ) -> Tensor:
        """Paired forward pass for counterfactual impact prediction.

        Runs the SAME encoder-decoder (shared weights) on two inputs:
          • x_factual:        landscape WITHOUT the proposed clearing
          • x_counterfactual: landscape WITH    the proposed clearing

        Returns
        -------
        impact_delta : Tensor [B, 1, H, W]
            Clamped difference (counterfactual - factual) in [0, 1].
            Positive values indicate INCREASED risk due to clearing.
        """
        out_f = self.forward(x_factual)
        out_cf = self.forward(x_counterfactual)
        return torch.clamp(out_cf - out_f, 0.0, 1.0)

    def forward_paired_deep(
        self, x_factual: Tensor, x_counterfactual: Tensor
    ) -> Tuple[Tensor, list]:
        """Paired forward with deep supervision outputs for training.

        Returns
        -------
        impact_delta : Tensor [B, 1, H, W]
        deep_deltas  : list[Tensor]  — auxiliary impact deltas from
                        intermediate UNet++ nodes (for deep supervision loss).
        """
        # Factual pass
        b_f, s_f = self.encode(x_factual)
        feat_f = {"s1": s_f["s1"], "s2": s_f["s2"], "s3": s_f["s3"], "s4": b_f}
        result_f = self.decoder(feat_f, return_deep=True)
        if isinstance(result_f, tuple):
            out_f, deep_f = result_f
        else:
            out_f, deep_f = result_f, []

        # Counterfactual pass (same weights)
        b_cf, s_cf = self.encode(x_counterfactual)
        feat_cf = {"s1": s_cf["s1"], "s2": s_cf["s2"], "s3": s_cf["s3"], "s4": b_cf}
        result_cf = self.decoder(feat_cf, return_deep=True)
        if isinstance(result_cf, tuple):
            out_cf, deep_cf = result_cf
        else:
            out_cf, deep_cf = result_cf, []

        # Main delta
        delta = torch.clamp(out_cf - out_f, 0.0, 1.0)

        # Deep supervision deltas
        deep_deltas = []
        for d_f, d_cf in zip(deep_f, deep_cf):
            deep_deltas.append(torch.clamp(d_cf - d_f, 0.0, 1.0))

        return delta, deep_deltas


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
