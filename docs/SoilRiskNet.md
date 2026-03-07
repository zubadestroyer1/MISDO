# SoilRiskNet — Soil Degradation Impact Model

## Overview

SoilRiskNet predicts how deforestation **increases soil degradation** (moisture loss, topsoil erosion, temperature increase) in surrounding and cleared areas. It uses a physics-informed proxy target with temporal compounding, anchored to real deforestation events.

## Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | ConvNeXt-V2 Base (96→192→384→768), 4 stages with GRN |
| **Decoder** | UNet++ with nested dense skip connections + DilatedContextModule (ASPP) |
| **Temporal** | Multi-head self-attention (4 heads) with positional encoding |
| **Parameters** | ~40M |
| **Input** | `[B, T, 5, 256, 256]` or `[B, 5, 256, 256]` |
| **Output** | `[B, 1, 256, 256]` — soil degradation impact delta [0, 1] |

## Input Channels (5)

| Channel | Description | Range |
|---------|-------------|-------|
| 0 | surface_soil_moisture — Forest-derived moisture proxy | [0, 1] |
| 1 | vegetation_water_content — Forest-derived water proxy | [0, 1] |
| 2 | soil_temperature — Forest-derived temperature proxy | [0, 1] |
| 3 | slope — Terrain slope (normalised) | [0, 1] |
| 4 | **deforestation_mask** — binary: 1=cleared between T₁ and T₂ | {0, 1} |

## Target

Soil degradation impact delta: cumulative soil exposure WITH the clearing event minus cumulative exposure WITHOUT it. Longer exposure = worse degradation (temporal compounding).

## Loss Function

**Edge-Weighted MSE** — upweight near deforestation edges + gradient matching.

## Training

- **Data source**: Physics proxy anchored to real Hansen deforestation events
- **Sliding temporal windows**: random (T₁, T₂) per chip with temporal compounding
- **Single-patch augmentation**: 50% chance
- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.01)
- **Deep supervision** + early stopping
