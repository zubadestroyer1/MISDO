"""
FireRiskNet — Fire Risk Detection Model
==========================================
ConvNeXt-V2 + UNet++ for active fire detection from multi-band
satellite imagery.

Input  : [B, T, 6, 256, 256]  or  [B, 6, 256, 256]
Output : [B, 1, 256, 256]  (fire risk mask, sigmoid, [0, 1])

Channels (6):
    0: I1 (0.64 µm visible reflectance)
    1: I2 (0.86 µm NIR reflectance)
    2: I3 (1.61 µm SWIR reflectance)
    3: I4 (3.74 µm MIR brightness temp)
    4: I5 (11.45 µm TIR brightness temp)
    5: FRP (fire radiative power)
"""

from __future__ import annotations

import torch
from models.base_model import DomainRiskNet


class FireRiskNet(DomainRiskNet):
    """ConvNeXt-V2 + UNet++ for binary fire segmentation."""
    IN_CHANNELS: int = 6


if __name__ == "__main__":
    model = FireRiskNet()
    params = sum(p.numel() for p in model.parameters())
    print(f"FireRiskNet parameters: {params:,}")

    # Test single frame
    x = torch.randn(1, 6, 256, 256)
    y = model(x)
    print(f"Single frame: {y.shape}  [{y.min():.3f}, {y.max():.3f}]")

    # Test encode/decode
    b, s = model.encode(x)
    print(f"Bottleneck: {b.shape}")
    y2 = model.decode(b, s)
    print(f"Decode: {y2.shape}  matches forward: {torch.allclose(y, y2)}")
