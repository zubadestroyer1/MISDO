# HydroRiskNet — ConvNeXt-V2 + UNet++ Water-Pollution Risk Model

## Overview

HydroRiskNet estimates water-pollution and erosion risk from SRTM elevation data and HydroSHEDS flow accumulation products. It uses a **ConvNeXt-V2 Base** encoder with **UNet++** decoder for continuous risk regression.

## Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | ConvNeXt-V2 Base (96→192→384→768), 4 stages with GRN |
| **Decoder** | UNet++ with nested dense skip connections |
| **Temporal** | Multi-head self-attention (4 heads) with positional encoding |
| **Parameters** | ~34M |
| **Input** | `[B, T, 5, 256, 256]` or `[B, 5, 256, 256]` |
| **Output** | `[B, 1, 256, 256]` — sigmoid water-pollution risk [0, 1] |

### Encoder Stages

```
Stem    : 5 → 96,  stride 4, patchify        → H/4   (64×64)
Stage 1 : 96 → 96,  depth 3, ConvNeXt-V2     → H/4   (64×64)
Stage 2 : 96 → 192, stride 2, depth 3        → H/8   (32×32)
Stage 3 : 192 → 384, stride 2, depth 9       → H/16  (16×16)
Stage 4 : 384 → 768, stride 2, depth 3       → H/32  (8×8)
```

## Input Channels

| Channel | Band | Description | Range |
|---------|------|-------------|-------|
| 0 | elevation | SRTM DEM (normalised) | [0, 1] |
| 1 | slope | Terrain slope (degrees, normalised) | [0, 1] |
| 2 | aspect | Aspect (sin-encoded) | [0, 1] |
| 3 | flow_accumulation | HydroSHEDS flow (log-normalised) | [0, 1] |
| 4 | flow_direction | D8 flow direction (normalised) | [0, 1] |

## Loss Function

**Gradient MSE Loss** (grad_weight=0.3) — MSE plus spatial gradient matching to preserve risk surface topology.

## Training

- **Data augmentation**: random flips, rotations, brightness jitter
- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4)
- **Scheduler**: cosine annealing
- **Best checkpoint saving** on validation loss
- **Early stopping** (patience=10)
