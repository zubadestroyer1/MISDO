# MISDO — Potential Improvements

Prioritised list of improvements to increase model accuracy and robustness.

---

## 🔴 High Priority (Training Impact)

### 1. Activate Monotonicity Penalty
The `CounterfactualDeltaLoss` has a built-in penalty ensuring counterfactual output ≥ factual output (deforestation shouldn't decrease risk). Currently dead code — factual/counterfactual raw outputs are never passed to the loss.

### 2. ~~Replace Hydro Proxy Targets with Real Observations~~ ✔ (Fixed)
~~**Current:** Hydro target is a deterministic formula...~~ Implemented Sentinel-2 NDSSI. The pipeline now fetches real pre/post deforestation MSI data to compute the NDSSI delta.

### 3. ~~Replace Soil Proxy Targets with SMAP Observations~~ ✔ (Fixed)
~~**Current:** Soil target is a physics formula...~~ Implemented TerraClimate soil moisture. The pipeline now fetches actual soil moisture data to weight the physically-modelled degradation.

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
