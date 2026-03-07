"""
FireImpactNet — Fire Risk Impact Model
==========================================
ConvNeXt-V2 + UNet++ for predicting how deforestation at a given
location increases fire risk in surrounding forest.

Input  : [B, T, 7, 256, 256]  or  [B, 7, 256, 256]
Output : [B, 1, 256, 256]  (fire impact delta, ReLU+clamp, [0, 1])

Channels (7) — with real VIIRS data:
    0: forest_cover (canopy %, deforestation-aware)
    1: recent_loss (binary: clearing in this timestep)
    2: viirs_fire_count (per-year fire detections, normalised)
    3: viirs_mean_frp (mean fire radiative power, normalised)
    4: viirs_max_bright_ti4 (max MIR brightness temperature)
    5: viirs_max_bright_ti5 (max TIR brightness temperature)
    6: deforestation_mask (binary: 1=cleared between T₁ and T₂)

Channels (7) — proxy fallback (no VIIRS):
    0: forest_cover
    1: recent_loss
    2: exposure (edge-zone fire exposure proxy)
    3: dryness (1 - forest_cover)
    4: slope (SRTM, normalised)
    5: elevation (SRTM, normalised)
    6: deforestation_mask
"""

from __future__ import annotations

import torch
from models.base_model import DomainRiskNet


class FireRiskNet(DomainRiskNet):
    """ConvNeXt-V2 + UNet++ for fire impact prediction.

    Predicts how much fire risk increases in surrounding forest
    when the areas indicated by the deforestation mask are cleared.
    """
    IN_CHANNELS: int = 7


if __name__ == "__main__":
    model = FireRiskNet()
    params = sum(p.numel() for p in model.parameters())
    print(f"FireRiskNet parameters: {params:,}")

    # Test single frame
    x = torch.randn(1, 7, 256, 256)
    y = model(x)
    print(f"Single frame: {y.shape}  [{y.min():.3f}, {y.max():.3f}]")

    # Test encode/decode
    b, s = model.encode(x)
    print(f"Bottleneck: {b.shape}")
    y2 = model.decode(b, s)
    print(f"Decode: {y2.shape}  matches forward: {torch.allclose(y, y2)}")
