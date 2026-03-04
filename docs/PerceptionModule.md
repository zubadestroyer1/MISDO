# Perception Module — Shared Backbone & Agent Decoders

## Overview

The **Perception Module** is the shared spatial-reasoning backbone that converts raw multi-modal satellite data into risk masks for each environmental domain. Two variants exist:

1. **MISDOPerception** — ConvNeXt-Tiny backbone with 4 lightweight U-Net decoder heads (legacy, 20-channel unified input)
2. **RealMISDOPerception** — Loads the 4 trained domain-specific models and stacks their outputs (production pipeline)

---

## MISDOPerception (Legacy / Unified Pipeline)

| Property | Value |
|---|---|
| **Backbone** | ConvNeXt-Tiny (3-stage) |
| **Decoder Heads** | 4 × DecoderHead (hydrology, biodiversity, climate, fire) |
| **Input** | `[B, 20, 256, 256]` — all 20 EO channels |
| **Output** | `[B, 4, 256, 256]` — stacked risk masks |

### ConvNeXt Backbone Architecture

```
Input [B, 20, 256, 256]
  │
  ▼
Stem: Conv2d(20→64, k=4, s=2) + GN          → [B, 64, 128, 128]
  │
  ▼
Stage 1: GN → Conv(64→128, k=2, s=2)
         ConvNeXtBlock(128) × 2               → [B, 128, 64, 64]
  │
  ▼
Stage 2: Conv(128→256, k=1)
         ConvNeXtBlock(256) × 2               → [B, 256, 64, 64]
```

### ConvNeXt Block

```
x ──┬── DWConv7×7(C, C, groups=C)  [depth-wise conv]
    │── GroupNorm(1, C)              [per-channel LayerNorm equivalent]
    │── Conv1×1(C, 4C)              [expand]
    │── GELU
    │── Conv1×1(4C, C)              [project]
    └── + (residual)
```

### Decoder Head (shared architecture, independent weights)

Each domain has its own DecoderHead mapping `[B, 256, 64, 64]` → `[B, 1, 256, 256]`:

```
ConvTranspose2d(256→128, k=2, s=2) + GN + GELU  → [B, 128, 128, 128]
ConvTranspose2d(128→64, k=2, s=2)  + GN + GELU  → [B, 64, 256, 256]
Conv2d(64→1, k=1) + Sigmoid                     → [B, 1, 256, 256]
```

---

## RealMISDOPerception (Production Pipeline)

| Property | Value |
|---|---|
| **Architecture** | 4 independent domain-specific models |
| **Input** | `Dict` of domain tensors (each with domain-specific channel count) |
| **Output** | `[B, 4, 256, 256]` — stacked risk masks |

This variant loads `FireRiskNet`, `ForestLossNet`, `HydroRiskNet`, and `SoilRiskNet` as independent sub-models, each with their own trained weights. This is the production configuration — each model processes only its domain-specific input.

### Weight Loading

```python
for name in ["fire", "forest", "hydro", "soil"]:
    path = f"weights/{name}_model.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    self.sub_models[name].load_state_dict(state)
```

---

## Why ConvNeXt for the Shared Backbone?

1. **Modern Architecture**: ConvNeXt achieves ResNet-level performance with pure convolutions — no attention, no transformers. This keeps inference fast on CPU/MPS.
2. **Depth-wise Convolutions**: 7×7 depthwise convs match the spatial correlation scale of satellite imagery.
3. **Shared Features**: A single backbone amortises feature extraction across 4 domains — spectral patterns (vegetation reflectance, thermal anomalies) are shared between fire and forest models.
4. **Decoupled Decoders**: Each domain head has independent weights, so domain-specific signals (fire hotspots vs. forest boundaries) are decoded separately.

---

## Why Domain-Specific Models in Production?

The legacy MISDOPerception requires all 20 channels simultaneously. In practice, each satellite source arrives at different times and resolutions. `RealMISDOPerception` with domain-specific sub-models allows:
1. **Independent updates**: Retrain one model without affecting others
2. **Domain-specific architectures**: Each model uses the optimal architecture for its data
3. **Flexible input**: Each model takes only its relevant channels
