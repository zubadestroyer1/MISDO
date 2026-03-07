# HydroRiskNet — Erosion Impact Model

## Overview

HydroRiskNet predicts how upstream deforestation **increases downstream erosion and water pollution**. It uses a physics-informed proxy target (slope × exposure × flow accumulation) anchored to real deforestation events from Hansen GFC.

## Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | ConvNeXt-V2 Base (96→192→384→768), 4 stages with GRN |
| **Decoder** | UNet++ with nested dense skip connections + DilatedContextModule (ASPP) |
| **Parameters** | ~40M |
| **Input** | `[B, 6, 256, 256]` (static — no temporal dimension) |
| **Output** | `[B, 1, 256, 256]` — erosion impact delta [0, 1] |

## Input Channels (6)

| Channel | Description | Range |
|---------|-------------|-------|
| 0 | elevation — SRTM DEM (normalised) | [0, 1] |
| 1 | slope — Terrain slope (degrees, normalised) | [0, 1] |
| 2 | aspect — Aspect (sin-encoded) | [0, 1] |
| 3 | flow_accumulation — HydroSHEDS flow (log-normalised) | [0, 1] |
| 4 | forest_cover — Canopy cover (deforestation-aware) | [0, 1] |
| 5 | **deforestation_mask** — binary: 1=cleared between T₁ and T₂ | {0, 1} |

## Target

Erosion impact delta: the INCREASE in downstream erosion compared to pre-clearing baseline. Computed as:
```
erosion_after(clearing + existing exposure) − erosion_before(existing exposure only)
```
Propagated downstream via flow accumulation with Gaussian diffusion.

## Loss Function

**Edge-Weighted MSE** — upweight near deforestation edges + gradient matching.

## Training

- **Data source**: Physics proxy anchored to real Hansen deforestation events + real SRTM terrain
- **Sliding temporal windows**: random (T₁, T₂) per chip
- **Single-patch augmentation**: 50% chance
- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.01)
- **Deep supervision** + early stopping
