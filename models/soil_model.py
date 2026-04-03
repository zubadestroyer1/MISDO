"""
SoilImpactNet — Soil Degradation Impact Model
================================================
ConvNeXt-V2 + UNet++ for predicting how deforestation increases
soil degradation in surrounding areas.

Input  : [B, T, 7, 256, 256]  or  [B, 7, 256, 256]
Output : [B, 1, 256, 256]  (soil degradation impact delta, [0, 1])

Channels (7):
    0: forest_cover (canopy %, deforestation-aware)
    1: smap_soil_moisture (baseline moisture, normalised)
    2: slope (SRTM, normalised degrees)
    3: elevation (SRTM DEM, normalised)
    4: aspect (SRTM, sin-encoded, [0, 1))
    5: flow_accumulation (log-normalised)
    6: deforestation_mask (binary: 1=cleared between T₁ and T₂)
"""

from __future__ import annotations

import torch
from models.base_model import DomainRiskNet


class SoilRiskNet(DomainRiskNet):
    """ConvNeXt-V2 + UNet++ for soil degradation impact prediction.

    Predicts how much soil degradation (moisture loss, temperature
    increase, topsoil erosion) occurs in surrounding areas when the
    indicated tiles are cleared.
    """
    IN_CHANNELS: int = 7


if __name__ == "__main__":
    model = SoilRiskNet()
    params = sum(p.numel() for p in model.parameters())
    print(f"SoilRiskNet parameters: {params:,}")

    x = torch.randn(1, 7, 256, 256)
    y = model(x)
    print(f"Single frame: {y.shape}  [{y.min():.3f}, {y.max():.3f}]")

    b, s = model.encode(x)
    print(f"Bottleneck: {b.shape}")
    y2 = model.decode(b, s)
    print(f"Decode: {y2.shape}  matches forward: {torch.allclose(y, y2)}")
