# ForestLossNet — Cascade Deforestation Impact Model

## Overview

ForestLossNet predicts how deforestation at one location **triggers cascade forest loss in surrounding areas** (edge effects, fragmentation pressure, road access). It learns from real Hansen GFC temporal data showing which forest pixels were lost after nearby clearings.

## Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | ConvNeXt-V2 Base (96→192→384→768), 4 stages with GRN |
| **Decoder** | UNet++ with nested dense skip connections + DilatedContextModule (ASPP) |
| **Temporal** | Multi-head self-attention (4 heads) with positional encoding |
| **Parameters** | ~40M |
| **Input** | `[B, T, 6, 256, 256]` or `[B, 6, 256, 256]` |
| **Output** | `[B, 1, 256, 256]` — cascade deforestation impact delta [0, 1] |

## Input Channels (6)

| Channel | Description | Range |
|---------|-------------|-------|
| 0 | treecover2000 — Percent canopy cover (year 2000) | [0, 1] |
| 1 | lossyear — Year of forest loss (normalised) | [0, 1] |
| 2 | gain — Binary forest gain (2000–2012) | {0, 1} |
| 3 | ndvi_proxy — NDVI vegetation health estimate | [0, 1] |
| 4 | canopy_change — Change from previous timestep | [-1, 1] |
| 5 | **deforestation_mask** — binary: 1=cleared between T₁ and T₂ | {0, 1} |

## Target

Cascade deforestation delta: additional forest loss in surrounding still-forested pixels **caused by** the clearing event (from Hansen `lossyear`), with control-pixel baseline subtraction.

## Loss Function

**Edge-Weighted MSE** — 3× upweight near deforestation boundaries + gradient matching.

## Training

- **Data source**: Real Hansen GFC `lossyear` — fully real observed data (no physics proxy needed)
- **Sliding temporal windows**: random (T₁, T₂) per chip
- **Single-patch augmentation**: 50% chance to show one clearing
- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.01)
- **Deep supervision** + early stopping
