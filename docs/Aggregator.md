# Aggregator — Learnable Parameter-Conditioned Spatial Risk Fusion

## Overview

The **ConditionedAggregator** is a learned fusion module that combines the 4 domain-specific risk masks into a single **Final Harm Mask**. It uses a small 1×1 convolutional network to capture non-linear cross-domain interactions, followed by Gaussian spatial smoothing and hard boolean constraints for environmental no-go zones.

| Property | Value |
|---|---|
| **Type** | Learned (1×1 conv fusion network) |
| **Learnable Parameters** | ~833 |
| **Input** | Agent masks `[B, 4, 256, 256]` + user weights `[B, 4]` + optional slope/river tensors |
| **Output** | `[B, 1, 256, 256]` — Final Harm Mask ∈ [0, 1] |

---

## Pipeline

```
Agent_Masks  [B, 4, 256, 256]  ─┐
                                 ├→ Concat [B, 8, H, W] → Learned Fusion → Gaussian Blur → Hard Constraints
User_Weights [B, 4] broadcast  ─┤                                                        → Final_Harm_Mask
Slope        [B, 1, H, W]      ─┤                                                          [B, 1, 256, 256]
River_Prox   [B, 1, H, W]      ─┘
```

---

## Architecture

### Learned Fusion Network

User weights `[B, 4]` are broadcast to `[B, 4, H, W]` and concatenated with the 4 agent masks to form an 8-channel input. This is passed through a 3-layer 1×1 conv network:

```
Input [B, 8, H, W]  (4 masks + 4 weight channels)
  │
  ▼
Conv1×1(8→32)  + GELU
Conv1×1(32→16) + GELU
Conv1×1(16→1)  → Sigmoid → [B, 1, H, W]
```

**Why 1×1 convolutions?** Each pixel is fused independently — the network learns per-pixel non-linear interactions between domains (e.g., fire × low moisture = exponentially worse) without introducing spatial blur. Spatial coherence is handled downstream by the Gaussian smoothing.

**Why this captures cross-domain interactions:**
- Standard weighted sum: `harm = w1*fire + w2*forest + w3*hydro + w4*soil` (linear only)
- Learned fusion: `harm = f(fire, forest, hydro, soil, w1, w2, w3, w4)` where f is a non-linear function that can learn:
  - Multiplicative interactions (fire × drought risk)
  - Threshold effects (high risk only when multiple factors align)
  - Weight-dependent gating (suppress a domain when user weight is low)

### Initialization

Xavier uniform with gain=0.5 and zero biases ensure initial outputs near sigmoid(0) = 0.5. This prevents extreme harm estimates before training.

### Post-Fusion Processing

Same as before:
1. **Gaussian smoothing** (σ=4.0, k=21) for spatial coherence
2. **Hard constraints**: slope > 0.8 → no-go (set to 1.0), river proximity < 0.05 → no-go

---

## Mathematical Formulation

### Learned Fusion

Given agent masks $\mathbf{M} \in \mathbb{R}^{B \times 4 \times H \times W}$ and user weights $\mathbf{w} \in \mathbb{R}^{B \times 4}$:

$$\mathbf{w}_{\text{spatial}} = \text{broadcast}(\mathbf{w}) \in \mathbb{R}^{B \times 4 \times H \times W}$$

$$\mathbf{F} = [\mathbf{M}; \mathbf{w}_{\text{spatial}}] \in \mathbb{R}^{B \times 8 \times H \times W}$$

$$\mathbf{H}_{\text{raw}} = \sigma\left(\mathbf{W}_3 \cdot \text{GELU}\left(\mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \cdot \mathbf{F} + \mathbf{b}_1) + \mathbf{b}_2\right) + \mathbf{b}_3\right)$$

Where $\mathbf{W}_1 \in \mathbb{R}^{32 \times 8}$, $\mathbf{W}_2 \in \mathbb{R}^{16 \times 32}$, $\mathbf{W}_3 \in \mathbb{R}^{1 \times 16}$ (all 1×1 conv weights).

---

## Why Learned Fusion is Optimal

1. **Non-linear interactions**: Real environmental risks compound non-linearly. Fire risk in drought-stressed forest is far worse than the sum of fire and drought individually.
2. **Weight-conditioned**: The network sees user weights as input channels — it learns different fusion strategies for different stakeholder priorities.
3. **Lightweight**: 833 parameters — negligible overhead vs. the ~12M in domain models.
4. **Trainable end-to-end**: Can be fine-tuned with the RL environment's reward signal via backpropagation.

---

## Configuration

| Parameter | Value | Effect |
|---|---|---|
| `SLOPE_THRESHOLD` | 0.8 | Slopes > 72° are no-go zones |
| `RIVER_THRESHOLD` | 0.05 | Areas within ~5% proximity to rivers are protected |
| `GAUSSIAN_SIGMA` | 4.0 | Smoothing spread (pixels) |
| `GAUSSIAN_KERNEL_SIZE` | 21 | Kernel width (must be odd) |
