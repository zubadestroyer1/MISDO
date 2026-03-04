# ForestLossNet — Hansen Global Forest Change Risk Model

## Overview

**ForestLossNet** is a convolutional neural network for deforestation detection and forest-loss risk mapping from Hansen Global Forest Change (GFC) data. It uses depthwise-separable convolutions (EfficientNet-style) for parameter efficiency and a pixel-shuffle decoder for artifact-free upsampling.

| Property | Value |
|---|---|
| **Architecture** | EfficientNet-style Encoder (MBConv) + Pixel-Shuffle Decoder |
| **Parameters** | ~3.2 M |
| **Input** | `[B, 5, 256, 256]` |
| **Output** | `[B, 1, 256, 256]` — sigmoid ∈ [0, 1] |
| **Data Source** | Hansen GFC v1.11 (Landsat-derived) |
| **Weight File** | `weights/forest_model.pt` (~5 MB) |

---

## Input Channels

| Channel | Name | Description | Range |
|---|---|---|---|
| 0 | treecover2000 | Percent canopy cover (year 2000 baseline) | [0, 1] → 0–100% |
| 1 | lossyear | Year of forest loss (normalised) | [0, 1] → 0–23 (2001–2023) |
| 2 | gain | Binary forest gain flag (2000–2012) | {0, 1} |
| 3 | red | Landsat red band composite | [0, 1] |
| 4 | NIR | Landsat NIR band composite | [0, 1] |

Channel 0 (treecover2000) provides the baseline canopy context — the model learns that deforestation is only meaningful where forests existed. The Landsat red/NIR composites capture spectral signatures of cleared vs. intact forest (NDVI relationship).

---

## Architecture

ForestLossNet uses an **EfficientNet-style encoder** with **MBConv (Mobile Bottleneck Convolution)** blocks and a **Pixel-Shuffle decoder**. This design prioritises parameter efficiency — depthwise-separable convolutions reduce FLOPs by ~8× compared to standard convolutions.

### Architecture Diagram

```
Input [B, 5, 256, 256]
  │
  ▼
┌─────────────────────────────────────┐
│  Stem                               │
│  Conv2d(5→48, k=3, s=2, p=1)       │
│  GroupNorm(8, 48) + GELU            │  → e1 [B, 48, 128, 128]
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Stage 1                            │
│  DepthwiseSeparableConv(48→96, s=2) │
│  MBConvBlock(96) × 2               │  → e2 [B, 96, 64, 64]
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Stage 2                            │
│  DepthwiseSeparableConv(96→192, s=2)│
│  MBConvBlock(192) × 2              │  → e3 [B, 192, 32, 32]
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Bottleneck                         │
│  MBConvBlock(192)                   │  → b [B, 192, 32, 32]
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  Decoder (Pixel Shuffle)                        │
│  PixelShuffleUp(192→96) → Concat e2 → MBConv   │  → d3 [B, 96, 64, 64]
│  PixelShuffleUp(96→48) → Concat e1 → MBConv    │  → d2 [B, 48, 128, 128]
│  PixelShuffleUp(48→24) → Head → Sigmoid         │  → out [B, 1, 256, 256]
└─────────────────────────────────────────────────┘
```

### Key Components

#### Depthwise-Separable Convolution

Decomposes a standard convolution into:
1. **Depthwise** (3×3, groups=C_in): Each input channel convolved independently
2. **Pointwise** (1×1): Cross-channel mixing

```
Standard Conv:  C_in × C_out × k² parameters
DW-Sep Conv:    C_in × k²  +  C_in × C_out parameters
Savings:        ~k² × (k²=9 for 3×3 → ~8× fewer params)
```

#### MBConv Block (Mobile Bottleneck)

```
x ──┬── Conv1×1(C→4C)    [expand]    → GroupNorm → GELU
    │── DWConv3×3(4C→4C)  [depthwise] → GroupNorm → GELU
    │── Conv1×1(4C→C)     [squeeze]   → GroupNorm
    └── + (residual) → GELU
```

The 4× expansion ratio in the intermediate space maximises the representational capacity while keeping the input/output dimensions compact.

#### Pixel-Shuffle Upsampling

Instead of strided transposed convolutions (which cause checkerboard artifacts), Pixel-Shuffle rearranges a $C \times 4$ channel tensor into a spatially $2\times$ larger tensor:

$$\text{PixelShuffle}(\mathbf{X} \in \mathbb{R}^{C \cdot r^2 \times H \times W}) = \mathbf{Y} \in \mathbb{R}^{C \times rH \times rW}$$

with $r = 2$. This is mathematically equivalent to sub-pixel convolution (Shi et al., 2016).

---

## Mathematical Formulation

### Forward Pass

Given $\mathbf{X} \in \mathbb{R}^{B \times 5 \times 256 \times 256}$:

**Encoder:**

$$\mathbf{e}_1 = \text{GELU}(\text{GN}(\text{Conv}_{3 \times 3}^{s=2}(\mathbf{X}))) \in \mathbb{R}^{B \times 48 \times 128 \times 128}$$

$$\mathbf{e}_2 = \text{MBConv}^2(\text{DWSep}_{s=2}(\mathbf{e}_1)) \in \mathbb{R}^{B \times 96 \times 64 \times 64}$$

$$\mathbf{e}_3 = \text{MBConv}^2(\text{DWSep}_{s=2}(\mathbf{e}_2)) \in \mathbb{R}^{B \times 192 \times 32 \times 32}$$

**MBConv Block (with expansion factor $E=4$):**

$$\text{MBConv}(\mathbf{x}) = \text{GELU}\left(\text{GN}(\mathbf{W}_{1 \times 1}^{\text{sq}} \cdot \text{GELU}(\text{GN}(\mathbf{W}_{3 \times 3}^{\text{dw}} \cdot \text{GELU}(\text{GN}(\mathbf{W}_{1 \times 1}^{\text{ex}} \cdot \mathbf{x}))))) + \mathbf{x}\right)$$

**Decoder with Pixel Shuffle:**

$$\mathbf{d}_3 = \text{MBConv}(\text{Conv}_{1 \times 1}([\text{PS}_\uparrow(\mathbf{b}); \mathbf{e}_2]))$$

$$\mathbf{d}_2 = \text{MBConv}(\text{Conv}_{1 \times 1}([\text{PS}_\uparrow(\mathbf{d}_3); \mathbf{e}_1]))$$

$$\hat{\mathbf{y}} = \sigma(\text{Conv}_{1 \times 1}(\text{GELU}(\text{Conv}_{3 \times 3}(\text{PS}_\uparrow(\mathbf{d}_2)))))$$

---

## Training Approach

### Loss Function: Dice + BCE

Forest-loss detection produces fragmented patches of varying size. The **Dice-BCE** loss combines pixel-level accuracy (BCE) with region-level overlap (Dice):

$$\mathcal{L} = (1 - w_d) \cdot \mathcal{L}_{\text{BCE}} + w_d \cdot \mathcal{L}_{\text{Dice}}$$

where $w_d = 0.5$ (equal weighting).

**BCE component:**

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

**Dice component:**

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_i \hat{y}_i y_i + \epsilon}{\sum_i \hat{y}_i + \sum_i y_i + \epsilon}$$

The Dice term directly optimises the F1-score (harmonic mean of precision and recall), which is the standard metric for segmentation quality. The smooth factor $\epsilon = 1.0$ prevents division by zero.

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

Training uses synthetic Hansen GFC data from `HansenGFCDataset`:
- **Forest Cover**: Multi-octave fractal noise generating realistic forest/non-forest boundaries
- **Deforestation Patches**: Random rectangular clearings (3–12 per sample) applied only where treecover > 30%
- **Landsat Composites**: Spectrally-coupled Red (higher in clearings) and NIR (higher in forest, NDVI-like)
- **Regrowth**: Early-loss areas (< 2012) get probabilistic gain flags

---

## Why This Architecture is Optimal

### Why EfficientNet-style (MBConv)?

1. **Parameter Efficiency**: Depthwise-separable convolutions reduce parameters by ~8× compared to standard convolutions. The model achieves ~3.2M params while having strong representational capacity — critical for the limited training data regime.
2. **Deforestation is Multi-scale**: Small clearings (10×10 px) and large plantation conversions (40×40 px) require multi-scale feature extraction. The inverted bottleneck (expand→depthwise→squeeze) captures both local texture and broader spatial patterns.
3. **Mobile-First Design**: MBConv blocks were designed for edge deployment — exactly matching our Mac CPU/MPS target.

### Why Pixel-Shuffle Decoder?

1. **Checkerboard-free**: Transposed convolutions produce visible grid artifacts. Pixel-shuffle rearranges channels into spatial dimensions without any learnable upsampling kernel overlap.
2. **Learnable**: Unlike bilinear interpolation, the preceding 1×1 convolution learns task-specific upsampling patterns.
3. **Compact**: Requires only a 1×1 convolution followed by channel rearrangement — fewer parameters than ConvTranspose2d.

### Why Dice-BCE?

Forest-loss patches are irregularly shaped and fragmented. Pure BCE doesn't penalise the model for missing entire patches (a pixel-level metric can still look good). The Dice component directly optimises the overlap between predicted and true loss regions, ensuring contiguous patch detection.

---

## Weights

| File | Size | Format |
|---|---|---|
| `weights/forest_model.pt` | ~5 MB | PyTorch `state_dict` |

Load with:
```python
from models.forest_model import ForestLossNet
import torch

model = ForestLossNet()
model.load_state_dict(torch.load("weights/forest_model.pt", map_location="cpu", weights_only=True))
model.eval()
```
