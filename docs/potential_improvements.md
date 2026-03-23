# MISDO — Potential Improvements

Prioritised list of improvements to increase model accuracy and robustness.

---

## 🔴 High Priority (Training Impact)

### 1. Activate Monotonicity Penalty
The `CounterfactualDeltaLoss` has a built-in penalty ensuring counterfactual output ≥ factual output (deforestation shouldn't decrease risk). Currently dead code — factual/counterfactual raw outputs are never passed to the loss.

### 2. Replace Hydro Proxy Targets with Real Observations
**Current:** Hydro target is a deterministic formula (`exposed × slope × curvature × flow_acc`) — the model learns to replicate a hand-coded equation, not real-world erosion.

**Proposed:** Use **Landsat-derived NDVI change in riparian zones** as a real observation target:
- Landsat 5/7/8/9 (30m resolution) — perfect spatial match with Hansen chips
- Detect riparian vegetation loss downstream of deforestation events
- NDVI is a direct spectral product (no atmospheric correction needed)
- Available 1985–present with 16-day revisit

**Alternative (harder):** Landsat water turbidity via Normalized Difference Turbidity Index (NDTI), requiring water pixel detection, atmospheric correction, cloud masking, and watershed delineation. ~1-2 weeks effort.

### 3. Replace Soil Proxy Targets with SMAP Observations
**Current:** Soil target is `Σ(deforested × slope × years_since)` — a physics formula with no observational validation.

**Proposed:** Use **NASA SMAP L3/L4 soil moisture** as a real observation target:
- Free via NASA Earthdata (`earthaccess` Python package)
- L4 downscaled to 1km resolution (each 256×256 chip ≈ 1 SMAP pixel)
- Available 2015–present (daily)
- Frame as: `target = moisture_after_clearing - moisture_before_clearing`
- ~2-3 days integration effort

**Limitation:** Coarse resolution means chip-level scalar targets, not pixel-level maps. Could combine with DEM-derived susceptibility for spatial detail.

### 4. ~~Add Data Quality Checks~~ ✔ (Fixed)
~~No validation at load time for all-zero chips, NaN/Inf values, missing keys, or corrupted `.npz` files. Any of these silently corrupt training.~~ Implemented via `validate_chip()` in `real_datasets.py` — validates every chip at `__getitem__` time with retry on failure.

---

## 🟡 Moderate Priority (Accuracy / Robustness)

### 5. ~~Global Random Seeds~~ ✔ (Fixed)
~~Add `torch.manual_seed(42)` and `np.random.seed(42)` at training entry for reproducible runs.~~ Implemented in `train_real_models.py`.

### 6. ~~Fix Gradient Accumulation Edge Case~~ ✔ (Fixed)
~~Final mini-batch of each epoch has incorrect gradient scaling when fewer than `accumulation_steps` batches remain.~~ Fixed in `train_real_models.py`.

### 7. VIIRS Proxy Distribution Shift
Events years 1–11 (2001–2011) use crude proxy fire channels because VIIRS launched in 2012. Consider:
- Training the fire model only on 2012+ events
- Using separate proxy/real epochs with domain adaptation
- Weighting real-VIIRS samples higher

### 8. Global Fire Normalisation
Per-chip per-year fire rasters are normalised to [0,1] independently. A chip with 1 fire and a chip with 500 fires look identical. Compute a global normalisation scale similar to `compute_global_target_scale()`.

### 9. Conditional Temporal Module Instantiation
`TemporalAttention` and `TemporalSkipFusion` are created for all models, including non-temporal ones (Hydro, Soil). Skip instantiation when `temporal=False` to save ~1.5M params of GPU memory.

### 10. AUROC Computation
Manual AUROC uses left-endpoint summation — systematically underestimates. Replace with trapezoidal integration or use `sklearn.metrics.roc_auc_score`.

---

## 🟢 Low Priority (Polish / Optimisation)

### 11. Siamese Encoder Optimisation
`forward_paired_deep()` runs the full encoder twice. Since factual/counterfactual differ only in the deforestation mask channel, a shared-backbone + mask-injection approach could halve encoder compute.

### 12. SRTM Void Infill
Void values (-32768) are replaced with `valid_min`. Local spatial interpolation would be more accurate for mountainous tiles.

### 13. ~~FIRMS API Coverage~~ ✔ (Addressed)
~~VIIRS download queries day 1 of each month (10-day window), missing ~2/3 of annual fire detections.~~ **Resolved** by switching to FIRMS bulk archive CSV downloads via `--viirs-archive`. The bulk CSVs contain the complete year of fire detections with no gaps.

### 14. Input Normalisation Consistency
Some channels use global scales (slope: degrees/45°), others per-chip (VIIRS fire count). Standardise to global scales computed from dataset statistics.

### 15. Learn Protective Effects
`torch.clamp(out_cf - out_f, 0.0, 1.0)` zeros out negative deltas. Could model beneficial clearing effects (firebreaks, managed forestry) with a separate head.

### 16. End-to-End Pipeline Test with Real Data
`test_real_pipeline.py` uses synthetic datasets (2-tuple). Add a test path that uses `RealHansenDataset` etc. (3-tuple) to validate the full counterfactual pipeline.

### 17. Background Rate Stability
`_control_baseline()` defaults to 0.0 when fewer than 100 "far from clearing" pixels exist. For heavily deforested chips, use a regional baseline from neighbouring chips.

---

## 📊 Data Source Summary

| Model | Current Target | Proposed Target | Source | Resolution | Temporal |
|-------|---------------|-----------------|--------|------------|----------|
| Fire | Real (VIIRS + Hansen) | — (already real) | NASA FIRMS | 375m | 2012–now |
| Forest | Real (Hansen lossyear) | — (already real) | Hansen GFC | 30m | 2001–2023 |
| Hydro | Physics proxy formula | Landsat riparian NDVI change | USGS Landsat | 30m | 1985–now |
| Soil | Physics proxy formula | NASA SMAP soil moisture delta | NASA Earthdata | 1–9km | 2015–now |
