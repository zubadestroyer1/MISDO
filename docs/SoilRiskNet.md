# SoilRiskNet — ConvNeXt-V2 + UNet++ Soil Degradation Model

## Overview

SoilRiskNet estimates soil degradation and drought risk from SMAP L3 soil moisture data. It uses a **ConvNeXt-V2 Base** encoder with **UNet++** decoder for continuous risk regression.

## Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | ConvNeXt-V2 Base (96→192→384→768), 4 stages with GRN |
| **Decoder** | UNet++ with nested dense skip connections |
| **Temporal** | Multi-head self-attention (4 heads) with positional encoding |
| **Parameters** | ~34M |
| **Input** | `[B, T, 4, 256, 256]` or `[B, 4, 256, 256]` |
| **Output** | `[B, 1, 256, 256]` — sigmoid soil degradation risk [0, 1] |

### Encoder Stages

```
Stem    : 4 → 96,  stride 4, patchify        → H/4   (64×64)
Stage 1 : 96 → 96,  depth 3, ConvNeXt-V2     → H/4   (64×64)
Stage 2 : 96 → 192, stride 2, depth 3        → H/8   (32×32)
Stage 3 : 192 → 384, stride 2, depth 9       → H/16  (16×16)
Stage 4 : 384 → 768, stride 2, depth 3       → H/32  (8×8)
```

## Input Channels

| Channel | Band | Description | Range |
|---------|------|-------------|-------|
| 0 | surface_soil_moisture | Soil moisture (m³/m³ normalised) | [0, 1] |
| 1 | vegetation_water_content | Vegetation water (kg/m² normalised) | [0, 1] |
| 2 | soil_temperature | Soil temperature (K normalised) | [0, 1] |
| 3 | freeze_thaw | Binary freeze/thaw flag | {0, 1} |

## Loss Function

**Smooth MSE Loss** (smooth_weight=0.05, corr_weight=0.2) — MSE plus light total-variation smoothness and Pearson correlation reward for spatial pattern matching.

## Training

- **Data augmentation**: random flips, rotations, brightness jitter
- **Optimizer**: AdamW (lr=5e-4, weight_decay=1e-4)
- **Scheduler**: cosine annealing
- **Best checkpoint saving** on validation loss
- **Early stopping** (patience=10)
