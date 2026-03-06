"""
HydroRiskNet — Water-Pollution Risk Model
============================================
ConvNeXt-V2 + UNet++ for continuous water-pollution / erosion risk
estimation from terrain and hydrological data.

Input  : [B, T, 5, 256, 256]  or  [B, 5, 256, 256]
Output : [B, 1, 256, 256]  (water-pollution risk, sigmoid, [0, 1])

Channels (5):
    0: elevation (DEM, normalised)
    1: slope (degrees, normalised)
    2: aspect (sin-encoded)
    3: flow_accumulation (log-normalised)
    4: forest_cover (canopy %, deforestation-aware)
"""

from __future__ import annotations

import torch
from models.base_model import DomainRiskNet


class HydroRiskNet(DomainRiskNet):
    """ConvNeXt-V2 + UNet++ for continuous water-pollution risk."""
    IN_CHANNELS: int = 5


if __name__ == "__main__":
    model = HydroRiskNet()
    params = sum(p.numel() for p in model.parameters())
    print(f"HydroRiskNet parameters: {params:,}")

    x = torch.randn(1, 5, 256, 256)
    y = model(x)
    print(f"Single frame: {y.shape}  [{y.min():.3f}, {y.max():.3f}]")

    b, s = model.encode(x)
    print(f"Bottleneck: {b.shape}")
    y2 = model.decode(b, s)
    print(f"Decode: {y2.shape}  matches forward: {torch.allclose(y, y2)}")
