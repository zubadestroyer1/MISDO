# SoilRiskNet — SMAP Soil Degradation Risk Model

## Overview

**SoilRiskNet** uses dilated (atrous) convolutions to capture broad spatial moisture patterns from NASA SMAP (Soil Moisture Active Passive) radiometer data. It features an ASPP (Atrous Spatial Pyramid Pooling) stem for multi-scale context extraction and a compact decoder for drought/degradation risk estimation.

| Property | Value |
|---|---|
| **Architecture** | ASPP Stem + Dilated-Conv Encoder + Compact U-Net Decoder |
| **Parameters** | ~2.1 M |
| **Input** | `[B, 4, 256, 256]` |
| **Output** | `[B, 1, 256, 256]` — sigmoid ∈ [0, 1] |
| **Data Source** | SMAP L3 Soil Moisture (36 km, enhanced 9 km) |
| **Weight File** | `weights/soil_model.pt` (~16 MB) |

---

## Input Channels

| Channel | Name | Description | Range |
|---|---|---|---|
| 0 | surface_soil_moisture | Volumetric moisture (m³/m³, normalised) | [0, 1] → 0–0.5 m³/m³ |
| 1 | vegetation_water_content | Plant water content (kg/m², normalised) | [0, 1] → 0–10 kg/m² |
| 2 | soil_temperature | Surface temperature (K, normalised) | [0, 1] → 240–330 K |
| 3 | freeze_thaw | Binary frozen-ground flag | {0, 1} |

Channel 0 (soil moisture) and Channel 2 (temperature) have a strong anti-correlation: hot, dry regions have low moisture and high temperature, creating the primary drought-risk signal. Channel 1 (vegetation water) provides biotic context — low vegetation water content amplifies degradation risk.

---

## Architecture

SoilRiskNet's architecture is specifically designed for the **coarse spatial resolution** of SMAP data (~9 km). Unlike optical imagery where fine texture matters, soil moisture patterns have broad spatial gradients spanning 50–200 km. Dilated convolutions expand the receptive field without downsampling, preserving the already-coarse spatial information.

### Architecture Diagram

```
Input [B, 4, 256, 256]
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  ASPP Stem (Multi-Scale Dilated Encoder)                    │
│  ┌───────┐  ┌───────┐  ┌───────┐                           │
│  │d=1    │  │d=2    │  │d=4    │  Three parallel branches   │
│  │4→24   │  │4→24   │  │4→24   │                           │
│  │ GN+GELU│ │ GN+GELU│ │ GN+GELU│                          │
│  └───┬───┘  └───┬───┘  └───┬───┘                           │
│      └─────┬────┴─────────┘                                 │
│            ▼ Concat → [B, 72, 256, 256]                     │
│            Conv1×1(72→64) + GN + GELU                       │  → s [B, 64, 256, 256]
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Encoder 1                          │
│  DilatedBlock(64, d=1)              │
│  DilatedBlock(64, d=2)              │
│  Pool: Conv(s=2) → 128×128          │  → e1 [B, 64, 256, 256]
└─────────────┬───────────────────────┘       e1_down [B, 64, 128, 128]
              │
              ▼
┌─────────────────────────────────────┐
│  Encoder 2                          │
│  Proj: Conv1×1(64→128)              │
│  DilatedBlock(128, d=1)             │
│  DilatedBlock(128, d=2)             │
│  Pool: Conv(s=2) → 64×64            │  → e2 [B, 128, 128, 128]
└─────────────┬───────────────────────┘       e2_down [B, 128, 64, 64]
              │
              ▼
┌─────────────────────────────────────┐
│  Encoder 3                          │
│  Proj: Conv1×1(128→256)             │
│  DilatedBlock(256, d=1)             │
│  DilatedBlock(256, d=4)             │  → e3 [B, 256, 64, 64]
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Decoder                                                    │
│  Up3(256→128) + Concat e2 → Conv(256→128) + GN + GELU      │ → d3 [128, 128, 128]
│  Up2(128→64)  + Concat e1 → Conv(128→64)  + GN + GELU      │ → d2 [64, 256, 256]
│  Head: Conv(64→32) + GELU + Conv(32→1) + Sigmoid            │ → out [1, 256, 256]
└─────────────────────────────────────────────────────────────┘
```

### Key Components

#### ASPP Stem (Atrous Spatial Pyramid Pooling)

Three parallel convolution branches with increasing dilation rates capture context at multiple spatial scales simultaneously:

| Branch | Dilation Rate | Effective Receptive Field |
|---|---|---|
| Branch 1 | d=1 | 3×3 = local texture |
| Branch 2 | d=2 | 5×5 = neighbourhood context |
| Branch 3 | d=4 | 9×9 = regional patterns |

Each branch outputs 24 channels. The three are concatenated (72 channels) and reduced to 64 via a 1×1 convolution. This is inspired by DeepLab v3+ (Chen et al., 2018).

#### Dilated Block

```
x ──┬── Conv3×3(C, C, dilation=d, padding=d) → GN → GELU
    │── Conv3×3(C, C, dilation=1, padding=1) → GN
    └── + (residual) → GELU
```

The first convolution uses the specified dilation rate for expanded receptive field; the second uses standard dilation (d=1) for local refinement. The residual connection preserves the original signal.

**Effective receptive field with dilation $d$:**

For a 3×3 kernel with dilation $d$, the effective kernel covers $(2d+1) \times (2d+1)$ spatial extent. With $d=4$, a single convolution layer "sees" a 9×9 region — matching SMAP's coarse resolution.

---

## Mathematical Formulation

### ASPP Forward Pass

Given input $\mathbf{X} \in \mathbb{R}^{B \times 4 \times H \times W}$:

$$\mathbf{b}_k = \text{GELU}(\text{GN}(\text{Conv}_{3 \times 3}^{d=d_k}(\mathbf{X}))) \in \mathbb{R}^{B \times 24 \times H \times W} \quad d_k \in \{1, 2, 4\}$$

$$\mathbf{s} = \text{GELU}(\text{GN}(\text{Conv}_{1 \times 1}([\mathbf{b}_1; \mathbf{b}_2; \mathbf{b}_4]))) \in \mathbb{R}^{B \times 64 \times H \times W}$$

### Dilated Block

For dilation rate $d$ and channel count $C$:

$$\text{DilatedBlock}_d(\mathbf{x}) = \text{GELU}\left(\text{GN}(\mathbf{W}_2^{d=1} * \text{GELU}(\text{GN}(\mathbf{W}_1^{d=d} * \mathbf{x}))) + \mathbf{x}\right)$$

The dilated convolution with padding $p = d$ preserves spatial dimensions:

$$(\mathbf{W}^{d} * \mathbf{x})[i, j] = \sum_{m, n} \mathbf{W}[m, n] \cdot \mathbf{x}[i + d \cdot m, j + d \cdot n]$$

### Full Forward

$$\hat{\mathbf{y}} = \sigma\left(\text{Decoder}\left(\text{Encoder}(\text{ASPP}(\mathbf{X}))\right)\right)$$

---

## Training Approach

### Loss Function: Smooth MSE

Soil degradation is a **continuous risk field** with smooth spatial structure (!). The loss combines pixel accuracy with a total-variation smoothness regulariser:

$$\mathcal{L} = \text{MSE}(\hat{\mathbf{y}}, \mathbf{y}) + \lambda \cdot \mathcal{L}_{\text{smooth}}$$

where $\lambda = 0.2$ and:

$$\mathcal{L}_{\text{smooth}} = \frac{1}{N}\sum_{i,j}\left(|\hat{y}_{i+1,j} - \hat{y}_{i,j}| + |\hat{y}_{i,j+1} - \hat{y}_{i,j}|\right)$$

This is **total variation (TV) regularisation** — it penalises high-frequency oscillations in the output, encouraging spatially smooth predictions that match the broad gradients of real soil moisture patterns. Unlike the gradient-MSE used for hydro (which preserves sharp edges), TV specifically smooths the output — appropriate because soil moisture transitions are physically gradual.

### Optimizer & Schedule

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam $(β_1=0.9, β_2=0.999)$ |
| Learning Rate | $1 \times 10^{-3}$ |
| Weight Decay | $1 \times 10^{-5}$ |
| Epochs | 30 |
| Batch Size | 4 |
| Scheduler | Cosine Annealing ($\eta_{\min} = 10^{-6}$) |
| Gradient Clipping | Max norm = 1.0 |

### Data Pipeline

Training uses synthetic SMAP data from `SMAPSoilDataset`:
- **Moisture**: Spatially correlated field (scale=32) + fine noise + Gaussian dry-patch perturbations
- **Vegetation Water**: Positively correlated with moisture (r ≈ 0.7)
- **Temperature**: Inversely correlated with moisture (wet=cool, dry=hot)
- **Freeze/Thaw**: Binary flag for regions with temperature < 240 K (normalised < 0.2)
- **Target**: `(1-moisture) × 0.4 + temperature × 0.35 + (1-veg_water) × 0.25`, normalised

---

## Why This Architecture is Optimal

### Why Dilated Convolutions?

1. **SMAP Resolution**: At 9-36 km native resolution, soil moisture patterns are inherently broad. Standard convolutions with stride-2 downsampling would destroy the limited spatial detail. Dilated convolutions expand the receptive field without losing resolution.
2. **Receptive Field**: With dilation rates [1, 2, 4], the ASPP stem has an effective receptive field of 9×9 pixels at the first layer — immediately capturing regional moisture patterns that standard 3×3 convolutions would need 4 layers to reach.
3. **Memory Efficiency**: No spatial downsampling in the early stages means feature maps stay at full resolution — important for preserving the already-coarse SMAP signal.

### Why ASPP Stem?

Soil moisture is influenced by processes at multiple scales:
- **Local** (d=1): Irrigation, urban heat islands, small water bodies
- **Neighbourhood** (d=2): Vegetation type boundaries, soil type transitions
- **Regional** (d=4): Climate zones, elevation-driven precipitation gradients

Parallel branches with different dilation rates capture all three scales simultaneously, providing a rich multi-scale context to the encoder from the very first layer.

### Why Smooth-MSE Loss?

1. **Physical Reality**: Soil moisture doesn't jump discontinuously — it transitions gradually over 10s of km. TV regularisation enforces this physical constraint.
2. **Noise Suppression**: Without smoothness, the model might overfit to pixel-level noise in the training data. TV acts as a spatial prior.
3. **Downstream Compatibility**: The aggregator applies Gaussian smoothing — a naturally smooth input from SoilRiskNet reduces compounding smoothing artifacts.

### Why ~2.1M Parameters (Smallest Model)?

Soil degradation risk has fewer degrees of spatial freedom than fire, forest, or hydro risk — broad moisture gradients are lower-information than pixel-level fire hotspots. A compact model avoids overfitting to these smooth patterns.

---

## Weights

| File | Size | Format |
|---|---|---|
| `weights/soil_model.pt` | ~16 MB | PyTorch `state_dict` |

Load with:
```python
from models.soil_model import SoilRiskNet
import torch

model = SoilRiskNet()
model.load_state_dict(torch.load("weights/soil_model.pt", map_location="cpu", weights_only=True))
model.eval()
```
