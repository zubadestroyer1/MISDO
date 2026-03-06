# FireRiskNet — ConvNeXt-V2 + UNet++ Fire Risk Model

## Overview

FireRiskNet detects active wildfires and fire risk from VIIRS I-band satellite imagery. It uses a **ConvNeXt-V2 Base** encoder with **UNet++** decoder for state-of-the-art binary segmentation at ~375 m resolution.

## Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | ConvNeXt-V2 Base (96→192→384→768), 4 stages with GRN |
| **Decoder** | UNet++ with nested dense skip connections |
| **Temporal** | Multi-head self-attention (4 heads) with positional encoding |
| **Parameters** | ~34M |
| **Input** | `[B, T, 6, 256, 256]` or `[B, 6, 256, 256]` |
| **Output** | `[B, 1, 256, 256]` — sigmoid fire risk mask [0, 1] |

### Encoder Stages

```
Stem    : 6 → 96,  stride 4, patchify        → H/4   (64×64)
Stage 1 : 96 → 96,  depth 3, ConvNeXt-V2     → H/4   (64×64)
Stage 2 : 96 → 192, stride 2, depth 3        → H/8   (32×32)
Stage 3 : 192 → 384, stride 2, depth 9       → H/16  (16×16)
Stage 4 : 384 → 768, stride 2, depth 3       → H/32  (8×8)
```

### UNet++ Decoder

Dense nested skip connections aggregate features across all encoder scales:
- X_{2,1}, X_{1,1}, X_{1,2}, X_{0,1}, X_{0,2}, X_{0,3}
- Deep supervision available for auxiliary training loss
- Final upsample: H/4 → H via two transposed convolutions

## Input Channels

| Channel | Band | Description | Range |
|---------|------|-------------|-------|
| 0 | I1 | Visible reflectance (0.64 µm) | [0, 1] |
| 1 | I2 | NIR reflectance (0.86 µm) | [0, 1] |
| 2 | I3 | SWIR reflectance (1.61 µm) | [0, 1] |
| 3 | I4 | MIR brightness temp (3.74 µm) | [0, 1] |
| 4 | I5 | TIR brightness temp (11.45 µm) | [0, 1] |
| 5 | FRP | Fire radiative power | [0, 1] |

## Loss Function

**Focal BCE Loss** with α=0.75, γ=2.0 and prediction clamping to prevent NaN.

## Training

- **Data augmentation**: random flips, rotations, brightness jitter
- **Optimizer**: AdamW (lr=5e-4, weight_decay=1e-4)
- **Scheduler**: warmup cosine annealing
- **UNet++ deep supervision** auxiliary losses (aux_weight=0.3)
- **Best checkpoint saving** on validation loss
- **Early stopping** (patience=10)
- **NaN-safe losses** with prediction clamping for AMP
