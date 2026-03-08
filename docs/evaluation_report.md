# MISDO — Model Evaluation Report

> **Evaluated on:** Held-out validation set (unseen spatial tiles, all years 1–23)
> **Samples per model:** 500 | **Threshold:** 0.5 | **Hardware:** NVIDIA A100 GPU

---

## Summary

| Model | Grade | AUROC | Pearson | SSIM | Dice | F1 | MSE |
|-------|-------|-------|---------|------|------|----|-----|
| **Hydro** 💧 | **A — Excellent** | 0.9994 | 0.9808 | 0.9684 | 0.3847 | 0.8862 | 0.000387 |
| **Fire** 🔥 | **B — Good** | 0.9871 | 0.5408 | 0.6493 | 0.1137 | 0.0315 | 0.002297 |
| **Forest** 🌲 | **B — Good** | 0.9202 | 0.4328 | 0.3149 | 0.1074 | 0.0692 | 0.009225 |
| **Soil** 🪨 | **D — Needs Improvement** | 0.5491 | 0.0000 | 0.6060 | 0.0000 | 0.0000 | 0.003783 |

---

## Hydro 💧 — Grade A (Score: 9/10)

> **Parameters:** 40,604,775 | **Architecture:** HydroRiskNet (non-temporal)

### Quality Notes
- ✅ Strong spatial correlation with target
- ✅ Reasonable segmentation overlap
- ✅ Good structural similarity

### Pixel Metrics

| Metric | Value |
|--------|-------|
| MSE | 0.000387 |
| MAE | 0.005705 |
| RMSE | 0.019664 |
| Pearson | 0.9808 |
| Dice | 0.3847 |
| IoU | 0.2381 |
| Precision | 0.8735 |
| Recall | 0.8993 |
| F1 | 0.8862 |
| AUROC | 0.9994 |

### Spatial Metrics

| Metric | Value |
|--------|-------|
| SSIM | 0.9684 ± 0.0817 |
| Gradient MSE | 0.000009 |

### Calibration

| Metric | Value |
|--------|-------|
| ECE | 0.0009 |

### Distribution

| | Mean | Std | Min | Max |
|---|------|-----|-----|-----|
| **Predictions** | 0.0274 | 0.0999 | 0.0 | 1.0 |
| **Targets** | 0.0278 | 0.1005 | 0.0 | 1.0 |
| **KS Statistic** | 0.1619 | | | |

---

## Fire 🔥 — Grade B (Score: 7/10)

> **Parameters:** 40,606,311 | **Architecture:** FireRiskNet (temporal, T=5)

### Quality Notes
- ⚠️ Low variance — predictions lack spatial diversity
- ✅ Strong spatial correlation with target
- ⚠️ Low segmentation overlap
- ✅ Good structural similarity

### Pixel Metrics

| Metric | Value |
|--------|-------|
| MSE | 0.002297 |
| MAE | 0.014038 |
| RMSE | 0.047923 |
| Pearson | 0.5408 |
| Dice | 0.1137 |
| IoU | 0.0603 |
| Precision | 0.6365 |
| Recall | 0.0162 |
| F1 | 0.0315 |
| AUROC | 0.9871 |

### Spatial Metrics

| Metric | Value |
|--------|-------|
| SSIM | 0.6493 ± 0.2587 |
| Gradient MSE | 0.000226 |

### Calibration

| Metric | Value |
|--------|-------|
| ECE | 0.0039 |

### Distribution

| | Mean | Std | Min | Max |
|---|------|-----|-----|-----|
| **Predictions** | 0.0133 | 0.0386 | 0.0 | 0.7181 |
| **Targets** | 0.0094 | 0.0559 | 0.0 | 0.9757 |
| **KS Statistic** | 0.1603 | | | |

---

## Forest 🌲 — Grade B (Score: 6/10)

> **Parameters:** 40,604,775 | **Architecture:** ForestLossNet (temporal, T=5)

### Quality Notes
- ⚠️ Moderate spatial correlation
- ⚠️ Low segmentation overlap
- ⚠️ Moderate structural similarity
- ⚠️ Prediction distribution diverges from target

### Pixel Metrics

| Metric | Value |
|--------|-------|
| MSE | 0.009225 |
| MAE | 0.051508 |
| RMSE | 0.096046 |
| Pearson | 0.4328 |
| Dice | 0.1074 |
| IoU | 0.0568 |
| Precision | 0.5847 |
| Recall | 0.0368 |
| F1 | 0.0692 |
| AUROC | 0.9202 |

### Spatial Metrics

| Metric | Value |
|--------|-------|
| SSIM | 0.3149 ± 0.2347 |
| Gradient MSE | 0.000504 |

### Calibration

| Metric | Value |
|--------|-------|
| ECE | 0.0226 |

### Distribution

| | Mean | Std | Min | Max |
|---|------|-----|-----|-----|
| **Predictions** | 0.0493 | 0.0630 | 0.0 | 0.8741 |
| **Targets** | 0.0266 | 0.1013 | 0.0 | 0.9932 |
| **KS Statistic** | 0.6331 | | | |

---

## Soil 🪨 — Grade D (Score: 2/10)

> **Parameters:** 40,603,239 | **Architecture:** SoilRiskNet (temporal, T=5)

> [!CAUTION]
> **Mode collapse detected.** The model outputs constant zero for all inputs. Early stopped at epoch 11/60.
> Root cause: target_scale=45.0 squashes all targets near zero, eliminating gradient signal.

### Quality Notes
- ⚠️ Mode collapse — predictions near-constant
- ❌ Weak spatial correlation
- ❌ Very low segmentation overlap
- ✅ Good structural similarity (trivially, since both pred and target are mostly zero)

### Pixel Metrics

| Metric | Value |
|--------|-------|
| MSE | 0.003783 |
| MAE | 0.012912 |
| RMSE | 0.061508 |
| Pearson | 0.0000 |
| Dice | 0.0000 |
| IoU | 0.0000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| AUROC | 0.5491 |

### Spatial Metrics

| Metric | Value |
|--------|-------|
| SSIM | 0.6060 ± 0.3979 |
| Gradient MSE | 0.000159 |

### Calibration

| Metric | Value |
|--------|-------|
| ECE | 0.0129 |

### Distribution

| | Mean | Std | Min | Max |
|---|------|-----|-----|-----|
| **Predictions** | 0.0000 | 0.0000 | 0.0 | 0.0 |
| **Targets** | 0.0129 | 0.0601 | 0.0 | 1.0 |
| **KS Statistic** | 0.2961 | | | |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Hardware | NVIDIA A100 80GB |
| Epochs | 60 (max) |
| Batch size | 16 |
| Gradient accumulation | 4× (effective batch = 64) |
| Optimizer | AdamW (lr=3e-4, weight_decay=0.01) |
| Scheduler | Cosine annealing with 6-epoch linear warmup |
| Loss | CounterfactualDeltaLoss + DeepSupervision (aux_weight=0.3) |
| AMP | Enabled |
| Early stopping | Patience = 10 epochs |
| Spatial split | 80% train tiles / 20% test+validate tiles |
| Temporal range | All years 1–23 |
| Augmentation | Random flips + 90° rotations |
| Gradient clipping | Max norm = 1.0 |

## Metric Definitions

| Metric | Description |
|--------|-------------|
| **AUROC** | Area under ROC curve — measures ranking quality (1.0 = perfect) |
| **Pearson** | Pearson correlation — spatial pattern similarity (-1 to 1) |
| **SSIM** | Structural similarity index — perceptual quality (0 to 1) |
| **Dice** | Soft Dice coefficient — overlap between prediction and target (0 to 1) |
| **IoU** | Intersection over Union (Jaccard index) |
| **ECE** | Expected Calibration Error — how well probabilities match reality (lower = better) |
| **KS Statistic** | Kolmogorov-Smirnov — distribution divergence (lower = better) |
| **F1 / P / R** | F1 score, Precision, Recall at binary threshold = 0.5 |
