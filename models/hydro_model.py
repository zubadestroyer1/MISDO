"""
HydroImpactNet — Water-Pollution Impact Model
================================================
ConvNeXt-V2 + UNet++ for predicting how deforestation increases
downstream water pollution / erosion risk.

Input  : [B, 7, 256, 256]  (non-temporal, single-frame)
Output : [B, 1, 256, 256]  (water-pollution impact delta, [0, 1])

Temporal mode is DISABLED for hydro because the Sentinel-2 MSI
NDSSI target is baked to a fixed 2016→2020 window at download
time (download_msi_smap.py), making temporal sliding unnecessary.
The encoder still supports temporal inputs architecturally, but
the training config sets ``temporal=False``.

Channels (7):
    0: elevation (DEM, normalised)
    1: slope (degrees, normalised)
    2: aspect (sin-encoded)
    3: flow_accumulation (log-normalised)
    4: forest_cover (canopy %, deforestation-aware)
    5: ndssi_baseline (Sentinel-2 baseline water quality, normalised)
    6: deforestation_mask (binary: 1=cleared between T₁ and T₂)
"""

from __future__ import annotations

import torch
from models.base_model import DomainRiskNet


class HydroRiskNet(DomainRiskNet):
    """ConvNeXt-V2 + UNet++ for water-pollution impact prediction.

    Predicts how much downstream erosion/runoff risk increases
    when the indicated upstream tiles are cleared.
    """
    IN_CHANNELS: int = 7
    TEMPORAL: bool = False


if __name__ == "__main__":
    model = HydroRiskNet()
    params = sum(p.numel() for p in model.parameters())
    print(f"HydroRiskNet parameters: {params:,}")

    x = torch.randn(1, 7, 256, 256)
    y = model(x)
    print(f"Single frame: {y.shape}  [{y.min():.3f}, {y.max():.3f}]")

    b, s = model.encode(x)
    print(f"Bottleneck: {b.shape}")
    y2 = model.decode(b, s)
    print(f"Decode: {y2.shape}  matches forward: {torch.allclose(y, y2)}")
