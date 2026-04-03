# SoilRiskNet — Soil Degradation Impact Model

## Overview

SoilRiskNet predicts how deforestation **increases soil degradation** (moisture loss, topsoil erosion, temperature increase) in surrounding and cleared areas. It uses a Siamese counterfactual approach with real TerraClimate SMAP soil moisture data and terrain-derived features, anchored to real deforestation events from Hansen GFC.

## Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | ConvNeXt-V2 Base (96→192→384→768), 4 stages with GRN |
| **Decoder** | UNet++ with nested dense skip connections |
| **Temporal** | Multi-head self-attention on bottleneck + learned skip-level temporal fusion |
| **Inference** | Siamese paired forward: `impact_delta = clamp(output_cf - output_f, 0, 1)` |
| **Parameters** | 40,623,431 (~40.6M) |
| **Input** | `[B, T, 7, 256, 256]` or `[B, 7, 256, 256]` |
| **Output** | `[B, 1, 256, 256]` — soil degradation impact delta [0, 1] |

## Input Channels (7)

| Channel | Description | Range |
|---------|-------------|-------|
| 0 | forest_cover — Canopy cover %, deforestation-aware | [0, 1] |
| 1 | smap_soil_moisture — TerraClimate soil moisture (per-chip normalised) | [0, 1] |
| 2 | slope — SRTM terrain slope (normalised) | [0, 1] |
| 3 | elevation — SRTM DEM (normalised) | [0, 1] |
| 4 | aspect — SRTM terrain aspect (normalised to unit range) | [0, 1] |
| 5 | flow_accumulation — HydroSHEDS flow (log-normalised) | [0, 1] |
| 6 | **deforestation_mask** — binary: 1=cleared between T₁ and T₂ | {0, 1} |

> **Note:** Channel 1 (`smap_soil_moisture`) uses real TerraClimate monthly soil moisture data from Planetary Computer when available (`has_real_msi_smap=True`). Falls back to a forest-derived proxy (`forest * 0.8 + 0.2`) for chips without SMAP data. Terrain channels (slope, elevation, aspect, flow_acc) are derived from real SRTM DEM via `derive_terrain()`.

## Target

Soil degradation impact delta: cumulative soil exposure WITH the clearing event minus cumulative exposure WITHOUT it. Longer exposure = worse degradation (temporal compounding). Normalised by global target scale (p95).

## Loss Function

**CounterfactualDeltaLoss** — wraps a composite loss (Focal Charbonnier ε=1e-6, γ=2.0 + SSIM + edge-weighted MSE with 3× deforestation edge upweight) with a monotonicity penalty ensuring counterfactual output ≥ factual output. Combined with UNet++ deep supervision (aux_weight=0.3).

## Training

- **Siamese counterfactual**: paired forward pass (factual vs counterfactual) with shared weights
- **Data source**: Real TerraClimate SMAP soil moisture + real SRTM terrain + Hansen deforestation events
- **Spatial-only split**: 80% train tiles / 20% test+validate tiles, all years 1–23
- **Data augmentation**: random flips + 90° rotations (aspect-aware) + RadiometricJitter on channels [0, 1] (p=0.5)
- **Optimizer**: AdamW (lr=2.5e-4, weight_decay=0.01) with decay-group exclusions for bias/norm layers
- **Scheduler**: 10% linear warmup → cosine annealing (min_lr floor)
- **Gradient accumulation**: 4× steps (effective batch = 64 on A100)
- **Deep supervision**: UNet++ auxiliary losses (weight=0.3), full-resolution re-computation
- **Aspect-aware augmentation**: horizontal/vertical flips correctly mirror aspect channel (idx=4)
- **Early stopping**: patience=15
- **AMP**: enabled on CUDA
