# FireRiskNet — Fire Impact Model

## Overview

FireRiskNet predicts how deforestation **increases fire risk in surrounding forest**. It uses a Siamese counterfactual approach: the same encoder-decoder processes two versions of the landscape (with and without a clearing event), and the clamped difference reveals the causal fire impact delta.

Trained on real VIIRS per-year fire data when available, with physics-informed proxies as fallback.

## Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | ConvNeXt-V2 Base (96→192→384→768), 4 stages with GRN |
| **Decoder** | UNet++ with nested dense skip connections |
| **Temporal** | Multi-head self-attention on bottleneck + learned skip-level temporal fusion |
| **Inference** | Siamese paired forward: `impact_delta = clamp(output_cf - output_f, 0, 1)` |
| **Parameters** | ~40.6M |
| **Input** | `[B, T, 7, 256, 256]` or `[B, 7, 256, 256]` |
| **Output** | `[B, 1, 256, 256]` — fire impact delta, clamped [0, 1] |

## Input Channels (7)

### With real VIIRS data:

| Channel | Description | Range |
|---------|-------------|-------|
| 0 | forest_cover — Canopy cover %, deforestation-aware | [0, 1] |
| 1 | recent_loss — Binary: clearing in this timestep | {0, 1} |
| 2 | viirs_fire_count — Per-year fire detections (normalised) | [0, 1] |
| 3 | viirs_mean_frp — Mean fire radiative power (normalised) | [0, 1] |
| 4 | viirs_max_bright_ti4 — Max MIR brightness temperature | [0, 1] |
| 5 | viirs_max_bright_ti5 — Max TIR brightness temperature | [0, 1] |
| 6 | **deforestation_mask** — binary: 1=cleared between T₁ and T₂ | {0, 1} |

### Proxy fallback (no VIIRS):

| Channel | Description | Range |
|---------|-------------|-------|
| 0 | forest_cover | [0, 1] |
| 1 | recent_loss | {0, 1} |
| 2 | exposure — Edge-zone fire exposure proxy | [0, 1] |
| 3 | dryness — (1 - forest_cover) | [0, 1] |
| 4 | slope — SRTM, normalised | [0, 1] |
| 5 | elevation — SRTM, normalised | [0, 1] |
| 6 | deforestation_mask | {0, 1} |

## Target

Fire impact delta: increase in fire activity in surrounding forest **caused by** the clearing event, with control-pixel baseline subtraction to isolate causal signal. Normalised by global target scale (p95).

## Loss Function

**CounterfactualDeltaLoss** — wraps Edge-Weighted MSE (3× upweight near deforestation edges + gradient matching) with a monotonicity penalty ensuring counterfactual output ≥ factual output. Combined with UNet++ deep supervision (aux_weight=0.3).

## Training

- **Siamese counterfactual**: paired forward pass (factual vs counterfactual) with shared weights
- **Spatial-only split**: 80% train tiles / 20% test+validate tiles, all years 1–23
- **Data augmentation**: random flips + 90° rotations + RadiometricJitter (brightness/contrast p=0.5)
- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.01)
- **Scheduler**: 10% linear warmup → cosine annealing
- **Gradient accumulation**: 4× steps (effective batch = 64 on A100)
- **Deep supervision**: UNet++ auxiliary losses (weight=0.3)
- **Early stopping**: patience=10
- **AMP**: enabled on CUDA
