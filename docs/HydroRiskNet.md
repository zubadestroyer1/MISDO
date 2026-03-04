# HydroRiskNet — SRTM/HydroSHEDS Water-Pollution Risk Model

## Overview

**HydroRiskNet** is a multi-scale Feature Pyramid Network (FPN) with attention-gated skip connections for erosion and runoff risk estimation. It processes SRTM digital elevation model (DEM) and HydroSHEDS flow-accumulation data to produce a continuous per-pixel water-pollution risk surface.

| Property | Value |
|---|---|
| **Architecture** | Multi-scale FPN Encoder + Attention-Gated U-Net Decoder |
| **Parameters** | ~3.8 M |
| **Input** | `[B, 5, 256, 256]` |
| **Output** | `[B, 1, 256, 256]` — sigmoid ∈ [0, 1] |
| **Data Source** | SRTM 90m DEM + HydroSHEDS |
| **Weight File** | `weights/hydro_model.pt` (~18 MB) |

---

## Input Channels

| Channel | Name | Description | Range |
|---|---|---|---|
| 0 | elevation | DEM height (normalised) | [0, 1] |
| 1 | slope | Terrain slope in degrees (normalised) | [0, 1] → 0–90° |
| 2 | aspect | Slope aspect (sin-encoded, shifted) | [0, 1] |
| 3 | flow_accumulation | Log-normalised upstream drainage area | [0, 1] |
| 4 | flow_direction | D8 flow direction (8-class, normalised) | [0, 1] |

The slope and flow-accumulation channels are the primary risk drivers: steep slopes with high accumulated drainage create erosion corridors that transport sediment and pollutants into waterways.

---

## Architecture

HydroRiskNet's architecture is motivated by the multi-scale nature of hydrological risk — small-scale gullies, medium-scale hillslope erosion, and large-scale watershed drainage all contribute to water pollution. The FPN captures features at all scales, while attention gates learn to suppress irrelevant spatial regions.

### Architecture Diagram

```
Input [B, 5, 256, 256]
  │
  ▼
┌──────────────────────────────────┐
│  Encoder 1: FPNBlock(5→64)      │  → e1 [B, 64, 256, 256]
│  Pool: Conv(s=2) → 128×128      │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  Encoder 2: FPNBlock(64→128)    │  → e2 [B, 128, 128, 128]
│  Pool: Conv(s=2) → 64×64        │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  Encoder 3: FPNBlock(128→256)   │  → e3 [B, 256, 64, 64]
│  Pool: Conv(s=2) → 32×32        │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  Bottleneck: FPNBlock(256→256)  │  → b [B, 256, 32, 32]
└──────────┬───────────────────────┘
           │
    ┌──────┼── FPN Lateral Connections ──┐
    │      │                             │
    ▼      ▼                             ▼
  lat1   lat2                          lat3
[64,256] [128,128]                   [256,64]
    │      │                             │
    │      │      ┌──────────────────────┘
    │      │      ▼
    │      │   Up3 → AttentionGate(d3, l3) → Concat → FPNBlock → d3 [128, 64, 64]
    │      │
    │      └──── Up2 → AttentionGate(d2, l2) → Concat → FPNBlock → d2 [64, 128, 128]
    │
    └──── Up1 → AttentionGate(d1, l1) → Concat → FPNBlock → d1 [32, 256, 256]
                                                               │
                                                               ▼
                                                    Head → Sigmoid → [1, 256, 256]
```

### Key Components

#### FPN Block (Feature Pyramid Block)

Two stacked 3×3 convolutions with GroupNorm and GELU:

```
x → Conv3×3 → GN → GELU → Conv3×3 → GN → GELU → y
```

The double-convolution pattern increases the effective receptive field while keeping individual kernel sizes small.

#### Attention Gate

The attention gate learns a spatial soft-mask that reweights skip-connection features:

$$\alpha_{s}^{l} = \sigma(\boldsymbol{\psi}^T \cdot \text{ReLU}(\mathbf{W}_g \cdot \mathbf{g} + \mathbf{W}_s \cdot \mathbf{s}))$$

$$\hat{\mathbf{s}} = \alpha_{s}^{l} \odot \mathbf{s}$$

where:
- $\mathbf{g}$ = gating signal from the decoder path (coarse, semantic-rich features)
- $\mathbf{s}$ = skip-connection from the encoder (fine, detail-rich features)
- $\mathbf{W}_g, \mathbf{W}_s$ = 1×1 convolutions projecting to a shared intermediate space
- $\boldsymbol{\psi}$ = 1×1 convolution mapping to sigmoid attention coefficients
- $\odot$ = element-wise multiplication

This mechanism learns to suppress encoder features in flat/low-risk regions and amplify features near steep slopes and flow channels — exactly the spatial selectivity needed for hydrological risk.

#### FPN Lateral Connections

1×1 convolutions project each encoder level to a common feature space, enabling the decoder to combine information from different spatial scales without channel-count mismatch.

---

## Mathematical Formulation

### Forward Pass

$$\mathbf{e}_1 = \text{FPN}_{5 \to 64}(\mathbf{X}) \in \mathbb{R}^{B \times 64 \times 256 \times 256}$$

$$\mathbf{e}_2 = \text{FPN}_{64 \to 128}(\text{Pool}(\mathbf{e}_1)) \in \mathbb{R}^{B \times 128 \times 128 \times 128}$$

$$\mathbf{e}_3 = \text{FPN}_{128 \to 256}(\text{Pool}(\mathbf{e}_2)) \in \mathbb{R}^{B \times 256 \times 64 \times 64}$$

$$\mathbf{b} = \text{FPN}_{256 \to 256}(\text{Pool}(\mathbf{e}_3)) \in \mathbb{R}^{B \times 256 \times 32 \times 32}$$

**Lateral projections:**

$$\mathbf{l}_k = \text{Conv}_{1 \times 1}(\mathbf{e}_k) \quad \text{for } k \in \{1, 2, 3\}$$

**Attention-gated decoding:**

$$\mathbf{d}_3 = \text{FPN}([\text{Up}(\mathbf{b}); \text{AG}(\text{Up}(\mathbf{b}), \mathbf{l}_3)])$$

$$\mathbf{d}_2 = \text{FPN}([\text{Up}(\mathbf{d}_3); \text{AG}(\text{Up}(\mathbf{d}_3), \mathbf{l}_2)])$$

$$\mathbf{d}_1 = \text{FPN}([\text{Up}(\mathbf{d}_2); \text{AG}(\text{Up}(\mathbf{d}_2), \mathbf{l}_1)])$$

$$\hat{\mathbf{y}} = \sigma(\text{Conv}_{1 \times 1}(\text{GELU}(\text{Conv}_{3 \times 3}(\mathbf{d}_1))))$$

---

## Training Approach

### Loss Function: Gradient MSE

Water-pollution risk is a **continuous risk surface**, not a binary segmentation mask. The loss combines per-pixel reconstruction accuracy with spatial gradient matching:

$$\mathcal{L} = \text{MSE}(\hat{\mathbf{y}}, \mathbf{y}) + \lambda \cdot \mathcal{L}_{\text{grad}}$$

where $\lambda = 0.3$ and:

$$\mathcal{L}_{\text{grad}} = \text{MSE}\left(\frac{\partial \hat{\mathbf{y}}}{\partial x}, \frac{\partial \mathbf{y}}{\partial x}\right) + \text{MSE}\left(\frac{\partial \hat{\mathbf{y}}}{\partial y}, \frac{\partial \mathbf{y}}{\partial y}\right)$$

The gradient term ensures the model captures the **spatial structure** of risk — sharp transitions at ridge lines, smooth gradients along valleys — not just pixel-level accuracy.

### Optimizer & Schedule

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam $(β_1=0.9, β_2=0.999)$ |
| Learning Rate | $8 \times 10^{-4}$ |
| Weight Decay | $1 \times 10^{-5}$ |
| Epochs | 30 |
| Batch Size | 4 |
| Scheduler | Cosine Annealing ($\eta_{\min} = 10^{-6}$) |
| Gradient Clipping | Max norm = 1.0 |

### Data Pipeline

Training uses synthetic terrain data from `SRTMHydroDataset`:
- **Elevation**: Multi-scale Perlin noise at 4 octaves (scales 64, 32, 16, 8)
- **Slope/Aspect**: Computed analytically via Sobel-like finite differences on the DEM
- **Flow Accumulation**: D8 steepest-descent routing algorithm
- **Target**: Continuous risk = `slope × 0.6 + flow_acc × 0.4` + proximity to high-flow channels

---

## Why This Architecture is Optimal

### Why FPN with Attention Gates?

1. **Multi-scale Nature of Hydrology**: Erosion risk depends on local slope (gully-scale), hillslope geometry (catchment-scale), and watershed-level drainage patterns. FPN naturally captures features at all three scales.
2. **Attention Selectivity**: Not all encoder features are relevant — flat valley floors have low risk but high flow accumulation. Attention gates learn to suppress irrelevant spatial regions, improving signal-to-noise in the skip connections.
3. **Continuous Output**: Unlike binary segmentation, water risk is continuous. Attention gates provide smooth, spatially-coherent weighting rather than hard boundaries.

### Why Gradient-MSE Loss?

1. **Risk Surfaces Are Smooth**: The sharp transitions at ridge lines and gradual gradients along valley floors are defining characteristics of hydrological risk. Standard MSE alone doesn't penalise blurred edges sufficiently.
2. **Gradient Matching**: By explicitly matching spatial derivatives, the model learns to preserve the physically-meaningful structure of the risk surface — essential for downstream aggregation in the MISDO pipeline.

### Why Lower Learning Rate (8e-4)?

The FPN + attention architecture has more complex gradient dynamics than simple encoder-decoders. The lower learning rate (vs. 1e-3 for other models) prevents attention gate instability during early training.

---

## Weights

| File | Size | Format |
|---|---|---|
| `weights/hydro_model.pt` | ~18 MB | PyTorch `state_dict` |

Load with:
```python
from models.hydro_model import HydroRiskNet
import torch

model = HydroRiskNet()
model.load_state_dict(torch.load("weights/hydro_model.pt", map_location="cpu", weights_only=True))
model.eval()
```
