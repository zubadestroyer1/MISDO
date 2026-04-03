# HydroRiskNet — Erosion Impact Model

## Overview

HydroRiskNet predicts how upstream deforestation **increases downstream erosion and water pollution**. It uses a Siamese counterfactual approach with real Sentinel-2 NDSSI sediment observations (when available) and physics-informed proxy targets (slope × exposure × flow accumulation) anchored to real deforestation events from Hansen GFC.

This is the only **non-temporal** model — it processes static terrain features (elevation, slope, flow, NDSSI) without temporal attention, making it the simplest and fastest of the four models.

## Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | ConvNeXt-V2 Base (96→192→384→768), 4 stages with GRN |
| **Decoder** | UNet++ with nested dense skip connections |
| **Temporal** | Not used in training (static inputs) — architecture supports it but bypassed |
| **Inference** | Siamese paired forward: `impact_delta = clamp(output_cf - output_f, 0, 1)` |
| **Parameters** | 38,201,540 (~38.2M) |
| **Input** | `[B, 7, 256, 256]` (static — no temporal dimension) |
| **Output** | `[B, 1, 256, 256]` — erosion impact delta [0, 1] |

## Input Channels (7)

| Channel | Description | Range |
|---------|-------------|-------|
| 0 | elevation — SRTM DEM (normalised) | [0, 1] |
| 1 | slope — Terrain slope (degrees, normalised) | [0, 1] |
| 2 | aspect — Terrain aspect (normalised to unit range) | [0, 1] |
| 3 | flow_accumulation — HydroSHEDS flow (log-normalised) | [0, 1] |
| 4 | forest_cover — Canopy cover (deforestation-aware) | [0, 1] |
| 5 | ndssi_baseline — Sentinel-2 NDSSI (pre-clearing sediment baseline) | [-1, 1] |
| 6 | **deforestation_mask** — binary: 1=cleared between T₁ and T₂ | {0, 1} |

> **Note:** Channel 5 (`ndssi_baseline`) uses real Sentinel-2 L2A Blue/NIR bands when available (`has_real_msi_smap=True`), with NDSSI = (Blue−NIR)/(Blue+NIR) per Hossain et al. 2010. Falls back to zeros for chips without MSI data.

## Target

Erosion impact delta: the INCREASE in downstream erosion compared to pre-clearing baseline. Computed as:
```
erosion_after(clearing + existing exposure) − erosion_before(existing exposure only)
```
Propagated downstream via flow accumulation with Gaussian diffusion. Normalised by global target scale (p95 ≈ 1.47).

## Loss Function

**CounterfactualDeltaLoss** — wraps a composite loss (Focal Charbonnier ε=1e-6, γ=2.0 + SSIM + edge-weighted MSE with 3× deforestation edge upweight) with a monotonicity penalty ensuring counterfactual output ≥ factual output. Combined with UNet++ deep supervision (aux_weight=0.3).

## Training

- **Siamese counterfactual**: paired forward pass (factual vs counterfactual) with shared weights
- **Data source**: Real Sentinel-2 NDSSI + physics proxy anchored to real Hansen deforestation events + real SRTM terrain
- **Spatial-only split**: 80% train tiles / 20% test+validate tiles, all years 1–23
- **Data augmentation**: random flips + 90° rotations (aspect-aware) + RadiometricJitter on channels [4, 5] (p=0.5)
- **Optimizer**: AdamW (lr=2e-4, weight_decay=0.01) with decay-group exclusions for bias/norm layers
- **Scheduler**: 10% linear warmup → cosine annealing (min_lr floor)
- **Gradient accumulation**: 4× steps (effective batch = 64 on A100)
- **Deep supervision**: UNet++ auxiliary losses (weight=0.3), full-resolution re-computation
- **Aspect-aware augmentation**: horizontal/vertical flips correctly mirror aspect channel (idx=2)
- **Early stopping**: patience=15
- **AMP**: enabled on CUDA
