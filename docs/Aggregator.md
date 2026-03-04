# Aggregator — Parameter-Conditioned Spatial Risk Fusion

## Overview

The **ConditionedAggregator** is a deterministic fusion module that combines the 4 domain-specific risk masks into a single **Final Harm Mask**. It applies user-defined priority weights, Gaussian spatial smoothing for contiguous risk regions, and hard boolean constraints for environmental no-go zones.

| Property | Value |
|---|---|
| **Type** | Deterministic (no learnable parameters) |
| **Input** | Agent masks `[B, 4, 256, 256]` + user weights `[B, 4]` + optional slope/river tensors |
| **Output** | `[B, 1, 256, 256]` — Final Harm Mask ∈ [0, 1] |
| **Parameters** | 0 (buffer-only: pre-computed Gaussian kernel) |

---

## Pipeline

```
Agent_Masks  [B, 4, 256, 256]  ─┐
User_Weights [B, 4]             ─┤→ Weighted Sum → Gaussian Blur → Hard Constraints → Final_Harm_Mask
Slope        [B, 1, H, W]      ─┤                                                      [B, 1, 256, 256]
River_Prox   [B, 1, H, W]      ─┘
```

---

## Mathematical Formulation

### Step 1: Weighted Sum

Given agent masks $\mathbf{M} \in \mathbb{R}^{B \times 4 \times H \times W}$ and user weights $\mathbf{w} \in \mathbb{R}^{B \times 4}$:

$$\mathbf{H}_{\text{raw}}[b, 1, h, w] = \frac{\sum_{c=1}^{4} w_{b,c} \cdot M_{b,c,h,w}}{\sum_{c=1}^{4} w_{b,c}}$$

This is a normalised weighted average — if a user sets weights `[0.9, 0.1, 0.5, 0.2]`, fire risk (0.9) dominates while forest risk (0.1) has minimal influence.

### Step 2: Gaussian Smoothing

A 2D Gaussian kernel $\mathbf{G}$ is convolved with the raw harm mask to produce spatially contiguous risk regions:

$$\mathbf{H}_{\text{smooth}} = \mathbf{G}_{\sigma, k} * \mathbf{H}_{\text{raw}}$$

where:
- $\sigma = 4.0$ pixels
- $k = 21$ (kernel size, odd)
- Reflect padding is used to handle boundaries

The Gaussian kernel:

$$G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

This smoothing is critical for the RL environment — it prevents the agent from exploiting isolated low-risk pixels surrounded by high-risk areas.

### Step 3: Hard Boolean Constraints

Two deterministic no-go zones override the smoothed risk to maximum (1.0):

**Slope constraint:**

$$\mathbf{H}[h, w] = \begin{cases} 1.0 & \text{if slope}[h, w] > 0.8 \\ \mathbf{H}_{\text{smooth}}[h, w] & \text{otherwise} \end{cases}$$

**River proximity constraint:**

$$\mathbf{H}[h, w] = \begin{cases} 1.0 & \text{if river\_proximity}[h, w] < 0.05 \\ \mathbf{H}[h, w] & \text{otherwise} \end{cases}$$

These enforce physical forestry regulations: steep slopes (>~72°) are unsafe for logging equipment, and areas too close to rivers are protected riparian zones.

---

## Why This Design is Optimal

1. **Deterministic**: No training required — the aggregator's behaviour is fully specified by user weights and physical constraints. This ensures interpretability for decision-makers.
2. **Gaussian Smoothing**: Prevents "checkerboard" risk patterns that would confuse the RL agent. Real deforestation risk is spatially continuous.
3. **Hard Constraints**: Non-negotiable environmental protections are enforced post-smoothing, so no amount of low risk in surrounding areas can override them.
4. **User-Conditioned**: Different stakeholders (logging company vs. conservation agency) can set different weight vectors to get different harm masks — enabling multi-objective decision-making.

---

## Configuration

| Parameter | Value | Effect |
|---|---|---|
| `SLOPE_THRESHOLD` | 0.8 | Slopes > 72° are no-go zones |
| `RIVER_THRESHOLD` | 0.05 | Areas within ~5% proximity to rivers are protected |
| `GAUSSIAN_SIGMA` | 4.0 | Smoothing spread (pixels) |
| `GAUSSIAN_KERNEL_SIZE` | 21 | Kernel width (must be odd) |
