# FireRiskNet — Fire Impact Model

## Overview

FireRiskNet predicts how deforestation **increases fire risk in surrounding forest**. It uses a temporal counterfactual approach: given a deforestation mask showing which areas were cleared, the model predicts the fire impact delta in nearby still-forested areas.

Trained on real VIIRS per-year fire data when available, with physics-informed proxies as fallback.

## Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | ConvNeXt-V2 Base (96→192→384→768), 4 stages with GRN |
| **Decoder** | UNet++ with nested dense skip connections + DilatedContextModule (ASPP) |
| **Temporal** | Multi-head self-attention (4 heads) with positional encoding |
| **Parameters** | ~40M |
| **Input** | `[B, T, 7, 256, 256]` or `[B, 7, 256, 256]` |
| **Output** | `[B, 1, 256, 256]` — fire impact delta, ReLU+clamp [0, 1] |

## Input Channels (7)

| Channel | Description | Range |
|---------|-------------|-------|
| 0 | I1 — Visible reflectance (0.64 µm) | [0, 1] |
| 1 | I2 — NIR reflectance (0.86 µm) | [0, 1] |
| 2 | I3 — SWIR reflectance (1.61 µm) | [0, 1] |
| 3 | I4 — MIR brightness temp (3.74 µm) | [0, 1] |
| 4 | I5 — TIR brightness temp (11.45 µm) | [0, 1] |
| 5 | FRP — Fire radiative power | [0, 1] |
| 6 | **deforestation_mask** — binary: 1=cleared between T₁ and T₂ | {0, 1} |

## Target

Fire impact delta: increase in fire activity in surrounding forest **caused by** the clearing event, with control-pixel baseline subtraction to isolate causal signal.

## Loss Function

**Edge-Weighted MSE** — MSE + gradient matching + 3× upweight on pixels within 5px of deforestation edges where impact signal is strongest.

## Training

- **Sliding temporal windows**: randomly sampled (T₁, T₂) per chip, ~100× more training data
- **Single-patch augmentation**: 50% chance to show one clearing (bridges train/inference gap)
- **Data augmentation**: random flips, rotations
- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.01)
- **Scheduler**: 10% warmup → cosine annealing
- **Deep supervision**: UNet++ auxiliary losses (weight=0.3)
- **Early stopping**: patience=10
