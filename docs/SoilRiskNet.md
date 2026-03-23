# SoilRiskNet — Soil Degradation Impact Model

## Overview

SoilRiskNet predicts how deforestation **increases soil degradation** (moisture loss, topsoil erosion, temperature increase) in surrounding and cleared areas. It uses a Siamese counterfactual approach with physics-informed proxy targets featuring temporal compounding, anchored to real deforestation events.

> **⚠️ Known Issue:** The current trained weights suffer from mode collapse (Grade D). The model outputs near-zero predictions due to aggressive target normalisation (target_scale=45.0). Retraining with a capped target scale is recommended.

## Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | ConvNeXt-V2 Base (96→192→384→768), 4 stages with GRN |
| **Decoder** | UNet++ with nested dense skip connections |
| **Temporal** | Multi-head self-attention on bottleneck + learned skip-level temporal fusion |
| **Inference** | Siamese paired forward: `impact_delta = clamp(output_cf - output_f, 0, 1)` |
| **Parameters** | ~40.6M |
| **Input** | `[B, T, 5, 256, 256]` or `[B, 5, 256, 256]` |
| **Output** | `[B, 1, 256, 256]` — soil degradation impact delta [0, 1] |

## Input Channels (5)

| Channel | Description | Range |
|---------|-------------|-------|
| 0 | moisture — Forest-derived soil moisture proxy (`forest * 0.8 + 0.2`) | [0, 1] |
| 1 | veg_water — Forest-derived vegetation water content (`forest * 0.9`) | [0, 1] |
| 2 | temp — Forest-derived soil temperature proxy (`1 - forest * 0.6`) | [0, 1] |
| 3 | slope — Terrain slope (SRTM, normalised degrees) | [0, 1] |
| 4 | **deforestation_mask** — binary: 1=cleared between T₁ and T₂ | {0, 1} |

## Target

Soil degradation impact delta: cumulative soil exposure WITH the clearing event minus cumulative exposure WITHOUT it. Longer exposure = worse degradation (temporal compounding). Normalised by global target scale (p95 = 45.0 — this is the cause of mode collapse).

## Loss Function

**CounterfactualDeltaLoss** — wraps Edge-Weighted MSE (3× upweight near deforestation edges + gradient matching) with a monotonicity penalty ensuring counterfactual output ≥ factual output. Combined with UNet++ deep supervision (aux_weight=0.3).

## Training

- **Siamese counterfactual**: paired forward pass (factual vs counterfactual) with shared weights
- **Data source**: Physics proxy anchored to real Hansen deforestation events
- **Spatial-only split**: 80% train tiles / 20% test+validate tiles, all years 1–23
- **Data augmentation**: random flips + 90° rotations + RadiometricJitter (brightness/contrast p=0.5)
- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.01)
- **Scheduler**: 10% linear warmup → cosine annealing
- **Gradient accumulation**: 4× steps (effective batch = 64 on A100)
- **Deep supervision**: UNet++ auxiliary losses (weight=0.3)
- **Early stopping**: patience=10 (triggered at epoch 11 due to mode collapse)
- **AMP**: enabled on CUDA
