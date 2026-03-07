# Conditioned Aggregator — Hybrid Deterministic + Learnable Fusion

## Overview

The Conditioned Aggregator fuses 4 domain-specific **impact masks** into a unified harm mask, conditioned on user-provided priority weights. It uses a **hybrid** approach that guarantees weight sensitivity while allowing learned cross-domain interactions.

> **Note**: With the counterfactual model overhaul, each domain mask now represents the **predicted impact of deforestation** (not static risk). The aggregator combines these impact deltas to show where clearing would cause the most total environmental damage.

## Design Philosophy

The previous aggregator used a purely learned 1×1 conv network that was never trained, causing user weight sliders to have **no visible effect** on the output. The redesigned aggregator ensures user weights **always** produce visible changes.

## Pipeline

```
Agent_Masks  [B, 4, 256, 256]  ─┐
User_Weights [B, 4]             ─┤
                                 ├─→ ① Deterministic Weighted Sum
                                 │     weighted = Σ(wᵢ · maskᵢ) / Σ(wᵢ)
                                 │
                                 ├─→ ② Learned Correction (optional)
                                 │     correction = LearnedNet(masks) · sigmoid(scale)
                                 │     combined = weighted + correction
                                 │
                                 ├─→ ③ Gaussian Spatial Smoothing
                                 │     σ=3.0, kernel=17×17
                                 │
slope        [B, 1, H, W]      ─┤─→ ④ Hard Constraints
river_prox   [B, 1, H, W]      ─┘     slope > 0.8 → 1.0
                                       river_prox < 0.05 → 1.0

Output: Final_Harm_Mask [B, 1, 256, 256]
```

## Components

### 1. Deterministic Weighted Sum

Always active, always respects user weights:
```python
w_norm = user_weights / user_weights.sum()
harm = Σ(w_norm_i * mask_i)
```

### 2. Learned Correction (LearnedCorrection)

Optional cross-domain interaction network:
- `Conv1×1(4→32) + GELU + Conv1×1(32→16) + GELU + Conv1×1(16→1)`
- Scale parameter initialized to 0 — **outputs nothing without training**
- When trained, captures non-linear interactions (e.g., fire impact × soil impact → amplified harm)
- ~700 parameters

### 3. Gaussian Smoothing

Produces spatially contiguous risk regions (σ=3.0, 17×17 kernel).

### 4. Hard Constraints

Deterministic no-go zones:
- **Steep slopes** (normalised slope > 0.8) → forced to 1.0
- **Near rivers** (flow proximity < 0.05) → forced to 1.0

## Parameters

Total learnable parameters: ~700 (just the learned correction network).
