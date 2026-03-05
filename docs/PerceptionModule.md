# Perception Module — Domain Model Orchestration

## Overview

The Perception Module orchestrates all 4 domain-specific ConvNeXt-V2 + UNet++ models and optionally fuses their bottleneck features via cross-domain attention before decoding.

## Implementations

### RealMISDOPerception (primary)

Loads 4 trained domain models and runs them in a 3-phase pipeline:

```
Phase 1: Encode     — Run each model's ConvNeXt-V2 encoder
Phase 2: Fuse       — CrossDomainFusion exchanges bottleneck info
Phase 3: Decode     — Each model's UNet++ decodes from enriched features
```

**Input**: Dict of domain tensors `{"fire": [B,6,H,W], "forest": [B,5,H,W], "hydro": [B,5,H,W], "soil": [B,4,H,W]}`

**Output**: Stacked risk masks `[B, 4, 256, 256]`

### CrossDomainFusion

Lightweight feature fusion (~40K params) that exchanges information between the 4 domain bottlenecks:

1. Project each domain to shared 96-dim space (1×1 conv)
2. Concatenate all 4 projected features
3. Apply cross-attention (2-layer 1×1 conv network)
4. Per-domain gating (softmax across domains)
5. Add as residual back to each domain's bottleneck

Near-zero initialized — starts as identity, learns to contribute during training.

### MISDOPerception (legacy)

Legacy shared ConvNeXt-Tiny backbone with 4 decoder heads for backward compatibility.

**Input**: `[B, 20, 256, 256]` (20-channel mock data)

**Output**: `[B, 4, 256, 256]`

## Total Parameters

| Component | Parameters |
|-----------|-----------|
| 4× ConvNeXt-V2 + UNet++ models | ~136M total (~34M each) |
| CrossDomainFusion | ~40K |
| TemporalAttention (per model) | ~4.5M each |
| **Total** | ~136M |
