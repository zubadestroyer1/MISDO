"""
ForestImpactNet — Deforestation Cascade Impact Model
======================================================
ConvNeXt-V2 + UNet++ for predicting how deforestation at a given
location triggers cascade forest loss in surrounding areas.

Input  : [B, T, 6, 256, 256]  or  [B, 6, 256, 256]
Output : [B, 1, 256, 256]  (cascade deforestation risk delta, [0, 1])

Channels (6):
    0: forest_cover (canopy %, deforestation-aware)
    1: recent_loss (binary: clearing in this timestep)
    2: gain (binary forest gain 2000–2012)
    3: ndvi_proxy (forest_cover * 0.8 + 0.2 * (1 - recent_loss))
    4: canopy_change (forest delta from previous timestep)
    5: deforestation_mask (binary: 1=cleared between T₁ and T₂)
"""

from __future__ import annotations

import torch
from models.base_model import DomainRiskNet


class ForestLossNet(DomainRiskNet):
    """ConvNeXt-V2 + UNet++ for cascade deforestation impact.

    Predicts how much additional forest loss occurs in surrounding
    areas when the indicated tiles are cleared (edge effects,
    fragmentation pressure, road access).
    """
    IN_CHANNELS: int = 6


if __name__ == "__main__":
    model = ForestLossNet()
    params = sum(p.numel() for p in model.parameters())
    print(f"ForestLossNet parameters: {params:,}")

    x = torch.randn(1, 6, 256, 256)
    y = model(x)
    print(f"Single frame: {y.shape}  [{y.min():.3f}, {y.max():.3f}]")

    b, s = model.encode(x)
    print(f"Bottleneck: {b.shape}")
    y2 = model.decode(b, s)
    print(f"Decode: {y2.shape}  matches forward: {torch.allclose(y, y2)}")
