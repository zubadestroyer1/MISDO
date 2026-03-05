"""
SoilRiskNet — Soil Degradation Risk Model
============================================
ConvNeXt-V2 + UNet++ for continuous soil degradation risk estimation
from satellite-derived soil moisture data.

Input  : [B, T, 4, 256, 256]  or  [B, 4, 256, 256]
Output : [B, 1, 256, 256]  (soil degradation risk, sigmoid, [0, 1])

Channels (4):
    0: surface_soil_moisture (m³/m³ normalised)
    1: vegetation_water_content (kg/m² normalised)
    2: soil_temperature (K normalised)
    3: freeze_thaw (binary flag)
"""

from __future__ import annotations

import torch
from models.base_model import DomainRiskNet


class SoilRiskNet(DomainRiskNet):
    """ConvNeXt-V2 + UNet++ for continuous soil degradation risk."""
    IN_CHANNELS: int = 4


if __name__ == "__main__":
    model = SoilRiskNet()
    params = sum(p.numel() for p in model.parameters())
    print(f"SoilRiskNet parameters: {params:,}")

    x = torch.randn(1, 4, 256, 256)
    y = model(x)
    print(f"Single frame: {y.shape}  [{y.min():.3f}, {y.max():.3f}]")

    b, s = model.encode(x)
    print(f"Bottleneck: {b.shape}")
    y2 = model.decode(b, s)
    print(f"Decode: {y2.shape}  matches forward: {torch.allclose(y, y2)}")
