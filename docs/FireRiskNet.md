# FireRiskNet — VIIRS Active Fire Risk Model

## Overview

**FireRiskNet** is a fully convolutional neural network for wildfire detection and risk mapping from NASA VIIRS (Visible Infrared Imaging Radiometer Suite) satellite imagery. It ingests 6-band I-level radiance and fire radiative power data and produces a per-pixel fire-risk probability mask at 256×256 resolution.

| Property | Value |
|---|---|
| **Architecture** | ResNet-18 Encoder + U-Net Decoder |
| **Parameters** | ~2.5 M |
| **Input** | `[B, 6, 256, 256]` |
| **Output** | `[B, 1, 256, 256]` — sigmoid ∈ [0, 1] |
| **Data Source** | VIIRS VNP14IMG Active Fire product |
| **Weight File** | `weights/fire_model.pt` (~16 MB) |

---

## Input Channels

| Channel | Band | Description | Range |
|---|---|---|---|
| 0 | I1 | 0.64 µm visible reflectance | [0, 1] |
| 1 | I2 | 0.86 µm NIR reflectance | [0, 1] |
| 2 | I3 | 1.61 µm SWIR reflectance | [0, 1] |
| 3 | I4 | 3.74 µm MIR brightness temperature (normalised) | [0, 1] maps ~250–500 K |
| 4 | I5 | 11.45 µm TIR brightness temperature (normalised) | [0, 1] maps ~200–350 K |
| 5 | FRP | Fire Radiative Power (normalised) | [0, 1] |

The I4 (mid-infrared) band is the primary fire-detection channel — active fires emit strongly at 3.74 µm, producing a characteristic brightness temperature exceedance of 100+ K above background. FRP adds quantitative fire intensity information for severity-aware mapping.

---

## Architecture

FireRiskNet uses a **ResNet-18-style encoder** with a **symmetric U-Net decoder**, chosen because fire detection is fundamentally a semantic segmentation problem with severe class imbalance (fire pixels are sparse).

### Architecture Diagram

```
Input [B, 6, 256, 256]
  │
  ▼
┌─────────────────────────────────────┐
│  Encoder Stage 1                    │
│  Conv2d(6→64, k=7, s=2, p=3)       │
│  GroupNorm(8, 64) + GELU            │  → e1 [B, 64, 128, 128]
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Encoder Stage 2                    │
│  ResBlock(64)                       │
│  Conv2d(64→128, k=3, s=2, p=1)     │
│  GroupNorm(8, 128) + GELU           │  → e2 [B, 128, 64, 64]
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Encoder Stage 3                    │
│  ResBlock(128)                      │
│  Conv2d(128→256, k=3, s=2, p=1)    │
│  GroupNorm(8, 256) + GELU           │  → e3 [B, 256, 32, 32]
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Bottleneck                         │
│  ResBlock(256) × 2                  │  → b [B, 256, 32, 32]
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  Decoder Stage 3                                │
│  ConvTranspose2d(256→128, k=2, s=2)             │
│  Concat with e2 → [B, 256, 64, 64]             │
│  Conv2d(256→128) + GN + GELU + ResBlock(128)    │  → d3 [B, 128, 64, 64]
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  Decoder Stage 2                                │
│  ConvTranspose2d(128→64, k=2, s=2)              │
│  Concat with e1 → [B, 128, 128, 128]           │
│  Conv2d(128→64) + GN + GELU + ResBlock(64)      │  → d2 [B, 64, 128, 128]
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  Decoder Stage 1                                │
│  ConvTranspose2d(64→32, k=2, s=2)               │  → d1 [B, 32, 256, 256]
│  Head: Conv(32→16, k=3) + GELU + Conv(16→1, k=1)│
│  Sigmoid                                        │  → out [B, 1, 256, 256]
└─────────────────────────────────────────────────┘
```

### ResBlock (Core Building Block)

Each **ResBlock** consists of:

```
x ──┬── Conv2d(C, C, k=3, p=1) → GroupNorm(8, C) → GELU
    │── Conv2d(C, C, k=3, p=1) → GroupNorm(8, C)
    └── + (residual addition) → GELU
```

Key design choices:
- **GroupNorm instead of BatchNorm**: Stable with small batch sizes (B=4) common in satellite imagery training. GroupNorm normalises across groups of channels rather than across the batch, avoiding the batch-size sensitivity of BatchNorm.
- **GELU activation**: Smooth, non-monotonic activation that outperforms ReLU in residual networks — provides small gradient flow even for negative inputs, reducing dead-neuron issues.
- **Bias-free convolutions**: When followed by normalisation, convolutional bias is redundant — removing it saves parameters.

---

## Mathematical Formulation

### Forward Pass

Given input tensor $\mathbf{X} \in \mathbb{R}^{B \times 6 \times 256 \times 256}$:

**Encoder:**

$$\mathbf{e}_1 = \text{GELU}(\text{GN}(\text{Conv}_{7 \times 7}^{s=2}(\mathbf{X}))) \in \mathbb{R}^{B \times 64 \times 128 \times 128}$$

$$\mathbf{e}_2 = \text{GELU}(\text{GN}(\text{Conv}_{3 \times 3}^{s=2}(\text{ResBlock}(\mathbf{e}_1)))) \in \mathbb{R}^{B \times 128 \times 64 \times 64}$$

$$\mathbf{e}_3 = \text{GELU}(\text{GN}(\text{Conv}_{3 \times 3}^{s=2}(\text{ResBlock}(\mathbf{e}_2)))) \in \mathbb{R}^{B \times 256 \times 32 \times 32}$$

**Bottleneck:**

$$\mathbf{b} = \text{ResBlock}(\text{ResBlock}(\mathbf{e}_3)) \in \mathbb{R}^{B \times 256 \times 32 \times 32}$$

**Decoder with Skip Connections:**

$$\mathbf{d}_3 = \text{Dec}_3([\text{Up}_3(\mathbf{b}); \mathbf{e}_2]) \in \mathbb{R}^{B \times 128 \times 64 \times 64}$$

$$\mathbf{d}_2 = \text{Dec}_2([\text{Up}_2(\mathbf{d}_3); \mathbf{e}_1]) \in \mathbb{R}^{B \times 64 \times 128 \times 128}$$

**Output:**

$$\hat{\mathbf{y}} = \sigma(\text{Conv}_{1 \times 1}(\text{GELU}(\text{Conv}_{3 \times 3}(\text{Up}_1(\mathbf{d}_2))))) \in [0, 1]^{B \times 1 \times 256 \times 256}$$

where $[;]$ denotes channel-wise concatenation, $\text{Up}$ denotes transposed convolution upsampling, and $\sigma$ is the sigmoid function.

### ResBlock Math

For a ResBlock with channel count $C$:

$$\text{ResBlock}(\mathbf{x}) = \text{GELU}\left(\text{GN}(\mathbf{W}_2 * \text{GELU}(\text{GN}(\mathbf{W}_1 * \mathbf{x}))) + \mathbf{x}\right)$$

The residual connection ensures gradient flow through the identity path, enabling deeper networks without vanishing gradients.

---

## Training Approach

### Loss Function: Focal BCE

Fire detection suffers from extreme class imbalance — typically < 1% of pixels contain active fire. Standard BCE would be dominated by the majority (no-fire) class. The **Focal Loss** down-weights easy negatives:

$$\mathcal{L}_{\text{focal}} = -\alpha (1 - p_t)^\gamma \log(p_t)$$

where:
- $p_t = p$ if $y = 1$, else $p_t = 1 - p$
- $\alpha = 0.75$ — class weighting factor
- $\gamma = 2.0$ — focusing parameter

When $\gamma > 0$, the loss for well-classified examples ($p_t \to 1$) is reduced by factor $(1 - p_t)^\gamma$, focusing training on hard positives (fire pixels the model is uncertain about).

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

Training uses synthetic VIIRS data generated by `VIIRSFireDataset`:
- **Background**: Spatially-correlated vegetation reflectance fields for I1/I2/I3
- **Fires**: Gaussian cluster hotspots (2–8 per sample) with physically realistic MIR/TIR brightness temperature signatures
- **Fire Targets**: Binary mask from I4 brightness temperature thresholding ($T > 0.3$ normalised)
- **Burnt Scars**: Reduced visible/NIR reflectance near fire centres

---

## Why This Architecture is Optimal

### Why ResNet-18 + U-Net?

1. **Semantic Segmentation**: Fire detection is pixel-level classification — U-Net's encoder-decoder with skip connections is the gold standard for this task.
2. **Skip Connections**: Fire features span multiple scales — sub-pixel hot spots (I4/I5 anomalies) and landscape-level patterns (burn scar areas). Skip connections preserve both fine and coarse spatial information.
3. **ResNet Encoder**: Residual blocks enable deeper feature extraction without vanishing gradients. The ResNet-18 depth is sufficient for 6-channel input without overfitting.
4. **Lightweight**: ~2.5M parameters enables training and inference on Mac CPU/MPS — critical for a hackathon project that needs to run locally.

### Why 6 Input Channels?

The VIIRS I-band suite provides:
- **Spectral contrast**: I4 (MIR) brightness temperature exceedance above I5 (TIR) is the classic contextual fire detection signal (Giglio et al., 2003)
- **FRP**: Quantifies fire intensity, not just presence — enables severity-weighted risk gradients
- **Vegetation context**: I1/I2/I3 reflectance captures pre-fire land cover, aiding burn severity estimation

### Why Focal Loss?

Standard BCE on fire data with ~1% positive rate would converge to a degenerate "predict no-fire everywhere" solution. Focal loss with $\gamma=2$ reduces the gradient contribution from easy true-negatives by $(1-p)^2$, reallocating training signal to the sparse fire pixels.

---

## Weights

| File | Size | Format |
|---|---|---|
| `weights/fire_model.pt` | ~16 MB | PyTorch `state_dict` |

Load with:
```python
from models.fire_model import FireRiskNet
import torch

model = FireRiskNet()
model.load_state_dict(torch.load("weights/fire_model.pt", map_location="cpu", weights_only=True))
model.eval()
```
