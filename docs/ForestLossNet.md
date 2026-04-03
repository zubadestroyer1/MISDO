# ForestLossNet — Cascade Deforestation Impact Model

## Overview

ForestLossNet predicts how deforestation at one location **triggers cascade forest loss in surrounding areas** (edge effects, fragmentation pressure, road access). It uses a Siamese counterfactual approach: the same encoder-decoder processes two versions of the landscape (with and without a clearing event), and the clamped difference reveals the causal cascade impact delta.

Trained on real Hansen GFC temporal data showing which forest pixels were lost after nearby clearings.

## Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | ConvNeXt-V2 Base (96→192→384→768), 4 stages with GRN |
| **Decoder** | UNet++ with nested dense skip connections |
| **Temporal** | Multi-head self-attention on bottleneck + learned skip-level temporal fusion |
| **Inference** | Siamese paired forward: `impact_delta = clamp(output_cf - output_f, 0, 1)` |
| **Parameters** | ~40.6M |
| **Input** | `[B, T, 6, 256, 256]` or `[B, 6, 256, 256]` |
| **Output** | `[B, 1, 256, 256]` — cascade deforestation impact delta [0, 1] |

## Input Channels (6)

| Channel | Description | Range |
|---------|-------------|-------|
| 0 | forest_cover — Canopy cover %, deforestation-aware | [0, 1] |
| 1 | recent_loss — Binary: clearing in this timestep | {0, 1} |
| 2 | gain — Binary forest gain (2000–2012) | {0, 1} |
| 3 | ndvi_proxy — `forest_cover * 0.8 + 0.2 * (1 - recent_loss)` | [0, 1] |
| 4 | canopy_change — Forest delta from previous timestep | [-1, 1] |
| 5 | **deforestation_mask** — binary: 1=cleared between T₁ and T₂ | {0, 1} |

## Target

Cascade deforestation delta: additional forest loss in surrounding still-forested pixels **caused by** the clearing event (from Hansen `lossyear`), with control-pixel baseline subtraction. Normalised by global target scale (p95).

## Loss Function

**CounterfactualDeltaLoss** — wraps a composite loss (Focal Charbonnier ε=1e-6, γ=2.0 + SSIM + edge-weighted MSE with 3× deforestation edge upweight) with a monotonicity penalty ensuring counterfactual output ≥ factual output. Combined with UNet++ deep supervision (aux_weight=0.3).

## Training

- **Siamese counterfactual**: paired forward pass (factual vs counterfactual) with shared weights
- **Data source**: Real Hansen GFC `lossyear` — fully real observed data (no physics proxy needed)
- **Spatial-only split**: 80% train tiles / 20% test+validate tiles, all years 1–23
- **Data augmentation**: random flips + 90° rotations + RadiometricJitter on channel [0] (brightness/contrast p=0.5)
- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.01) with decay-group exclusions for bias/norm layers
- **Scheduler**: 10% linear warmup → cosine annealing (min_lr floor)
- **Gradient accumulation**: 4× steps (effective batch = 64 on A100)
- **Deep supervision**: UNet++ auxiliary losses (weight=0.3), full-resolution re-computation
- **Early stopping**: patience=15
- **AMP**: enabled on CUDA
