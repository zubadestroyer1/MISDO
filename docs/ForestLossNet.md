# ForestLossNet — ConvNeXt-V2 + UNet++ Forest Loss Model

## Overview

ForestLossNet detects deforestation and forest loss from Hansen Global Forest Change data (Landsat-derived). It uses a **ConvNeXt-V2 Base** encoder with **UNet++** decoder for state-of-the-art binary segmentation at ~30 m resolution.

## Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | ConvNeXt-V2 Base (96→192→384→768), 4 stages with GRN |
| **Decoder** | UNet++ with nested dense skip connections |
| **Temporal** | Multi-head self-attention (4 heads) with positional encoding |
| **Parameters** | ~34M |
| **Input** | `[B, T, 5, 256, 256]` or `[B, 5, 256, 256]` |
| **Output** | `[B, 1, 256, 256]` — sigmoid deforestation risk mask [0, 1] |

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
| 0 | treecover2000 | Percent canopy cover (year 2000) | [0, 1] |
| 1 | lossyear | Year of forest loss (normalised) | [0, 1] |
| 2 | gain | Binary forest gain (2000–2012) | {0, 1} |
| 3 | red | Landsat red band composite | [0, 1] |
| 4 | NIR | Landsat NIR band composite | [0, 1] |

## Loss Function

**Dice + BCE Loss** (dice_weight=0.5) — balances pixel-level accuracy with region-level overlap.

## Training

- **Data augmentation**: random flips, rotations, brightness jitter
- **Optimizer**: AdamW (lr=5e-4, weight_decay=1e-4)
- **Scheduler**: cosine annealing
- **Best checkpoint saving** on validation loss
- **Early stopping** (patience=10)
