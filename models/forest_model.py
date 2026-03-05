"""
ForestLossNet — Deforestation Detection Model
================================================
ConvNeXt-V2 + UNet++ for deforestation detection from multi-band
Landsat-derived imagery.

Input  : [B, T, 5, 256, 256]  or  [B, 5, 256, 256]
Output : [B, 1, 256, 256]  (deforestation risk mask, sigmoid, [0, 1])

Channels (5):
    0: treecover2000 (percent canopy cover)
    1: lossyear (year of loss, normalised)
    2: gain (binary forest gain 2000–2012)
    3: red band composite (Landsat)
    4: NIR band composite (Landsat)
"""

from __future__ import annotations

import torch
from models.base_model import DomainRiskNet


class ForestLossNet(DomainRiskNet):
    """ConvNeXt-V2 + UNet++ for binary deforestation segmentation."""
    IN_CHANNELS: int = 5


if __name__ == "__main__":
    model = ForestLossNet()
    params = sum(p.numel() for p in model.parameters())
    print(f"ForestLossNet parameters: {params:,}")

    x = torch.randn(1, 5, 256, 256)
    y = model(x)
    print(f"Single frame: {y.shape}  [{y.min():.3f}, {y.max():.3f}]")

    b, s = model.encode(x)
    print(f"Bottleneck: {b.shape}")
    y2 = model.decode(b, s)
    print(f"Decode: {y2.shape}  matches forward: {torch.allclose(y, y2)}")
