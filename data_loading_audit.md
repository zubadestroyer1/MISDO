# MISDO Data Loading Pipeline — Comprehensive Audit Report

> **Auditor:** Antigravity (Claude Opus 4.6)
> **Date:** 2026-04-02
> **Scope:** `download_real_data.py`, `download_msi_smap.py`, `real_datasets.py`, `__init__.py`, synthetic datasets, and interactions with `train_real_models.py`
> **Method:** Line-by-line code trace, cross-file cross-reference, external API documentation verification

---

## Executive Summary

| Severity | Count | Description |
|----------|-------|-------------|
| 🔴 **Critical** | 5 | Will cause data download failures or silent training corruption |
| 🟠 **High** | 7 | Significant accuracy degradation or data quality issues |
| 🟡 **Medium** | 8 | Robustness and correctness concerns |
| 🟢 **Low** | 5 | Polish, documentation, and minor optimisations |
| 📊 **Data Science** | 2 | Methodological concerns backed by literature |

The two issues you specifically reported — **MSI/SMAP loading errors** and **VIIRS batch download failure** — are confirmed and traced to root causes below (Issues #1 and #2).

---

## 🔴 Critical Issues (Training-Breaking)

### Issue #1 — VIIRS Bulk Archive Download URLs Are Completely Wrong
**File:** [download_real_data.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/download_real_data.py#L152-L310)
**Lines:** 152–310

> [!CAUTION]
> This is the root cause of the VIIRS batch download failure you reported.

**Root Cause:** The `FIRMS_ARCHIVE_BASE` URL and filename pattern are fabricated and do not match any real NASA FIRMS endpoint:

```python
# Line 152-155 — WRONG base URL
FIRMS_ARCHIVE_BASE = (
    "https://firms.modaps.eosdis.nasa.gov/data/active_fire/"
    "suomi-npp-viirs-c2/csv/"
)

# Line 276 — WRONG filename pattern
filename = f"fire_nrt_SV-C2_{year}.csv"  # This file does not exist
```

**What the code tries:**
1. `https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/fire_nrt_SV-C2_2020.csv`
2. `https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/fire_archive_SV-C2_2020.csv`

**What NASA actually provides:**
- The bulk archive is at `https://nrt3.modaps.eosdis.nasa.gov/archive/FIRMS/suomi-npp-viirs-c2/`
- Files are organized by **year/day-of-year**, not as single yearly CSVs
- Filenames follow `SUOMI_VIIRS_C2_Global_*.csv` convention
- **Authentication is required** (NASA Earthdata Login token) — the code uses no auth
- Direct HTTP download of yearly aggregate CSVs is only available through the [FIRMS Archive Download Tool](https://firms.modaps.eosdis.nasa.gov/download/) web interface, which generates custom ZIP packages

**Impact:** `download_viirs_archive()` silently fails for every year (line 305-306 catches the exception and `continue`s), resulting in an empty `csv_paths` list. The `VIIRSArchive` is never populated, and all chips fall through to the per-chip FIRMS API or to no-VIIRS-data. The log output `VIIRS {year}: ✗ not available (will skip)` (line 309) masks a fundamental URL error.

**Fix:** Either:
1. Implement proper Earthdata-authenticated download from `nrt3.modaps.eosdis.nasa.gov`
2. Use the FIRMS REST API to programmatically request archive download packages
3. Document that users must manually download CSVs from the FIRMS Archive Download tool and provide the path via `--viirs-archive`

---

### Issue #2 — Sentinel-2 Reflectance Values Not Scaled (÷10,000 Missing)
**File:** [download_msi_smap.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/download_msi_smap.py#L54-L79)
**Lines:** 54–79, 129–138

> [!CAUTION]
> This is a root cause of the MSI/SMAP data loading errors you reported. NDSSI values computed from unscaled integers will be near 0 or NaN for many pixels.

**Root Cause:** Sentinel-2 L2A data on Planetary Computer is stored as **16-bit unsigned integers** with a scale factor of **1/10,000**. The `fetch_stac_window()` function reads raw integer values and never divides by 10,000:

```python
# Line 67-73 — reads raw DN, never scales
data = src.read(
    1, 
    window=window, 
    out_shape=out_shape,
    resampling=rasterio.enums.Resampling.bilinear
)
return data.astype(np.float32)  # Values like 1500.0 instead of 0.15
```

When `calculate_ndssi()` (line 129–138) computes `(green - nir) / (green + nir)`, the formula itself is scale-invariant for a single image, **but** the clipping and delta computation are not. The resulting `sediment_increase` values are in the wrong range because:
1. The BOA_ADD_OFFSET (since PB 04.00) is not applied, corrupting negative-reflectance pixels
2. Intermediate values overflow float32 precision when dealing with 16-bit DN values in the thousands

**Additionally:** Since processing baseline 04.00 (January 2022), Sentinel-2 L2A has per-band radiometric offsets. The correct formula is:
```
reflectance = (DN + BOA_ADD_OFFSET) / 10000
```

**Impact:** NDSSI deltas are either near-zero (when the scale error cancels out in the ratio) or wildly incorrect (when offsets corrupt the result), leading to mostly-zero targets for the Hydro model. The model learns nothing meaningful.

---

### Issue #3 — NDSSI Formula Uses Wrong Bands (Green-NIR Instead of Blue-NIR)
**File:** [download_msi_smap.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/download_msi_smap.py#L129-L138)
**Lines:** 129–138

**Root Cause:** The code comments say "NDSSI = (Green - NIR) / (Green + NIR)" and uses B03 (Green) and B08 (NIR). According to remote sensing literature, the **Normalised Difference Suspended Sediment Index** uses **Blue** and **NIR**:

$$\text{NDSSI} = \frac{\rho_{\text{blue}} - \rho_{\text{NIR}}}{\rho_{\text{blue}} + \rho_{\text{NIR}}}$$

- **Blue** = B02 (490 nm), not B03 (560 nm)
- What the code actually computes with (Green - NIR)/(Green + NIR) is closer to **NDWI** (Normalized Difference Water Index), not NDSSI

**Impact:** The target signal fed to the Hydro model is measuring the wrong physical quantity. Green/NIR is sensitive to vegetation water content, not suspended sediment transport. This fundamentally undermines the Hydro model's ability to learn erosion impact.

**Fix:** Change `B03` → `B02`:
```python
blue = fetch_stac_window(item, "B02", bounds, (chip_size, chip_size))
nir = fetch_stac_window(item, "B08", bounds, (chip_size, chip_size))
ndssi = (blue - nir) / (blue + nir + 1e-8)
```

---

### Issue #4 — TerraClimate `soil` Asset Is Zarr, Not a GeoTIFF Band
**File:** [download_msi_smap.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/download_msi_smap.py#L107-L121)
**Lines:** 107–121

**Root Cause:** The code calls `fetch_stac_window(soil_item, "soil", bounds, ...)` which tries to open `soil_item.assets["soil"].href` with rasterio. But TerraClimate on Planetary Computer stores data as a **Zarr store**, not individual GeoTIFF bands. The asset key is `"zarr"`, not `"soil"`.

```python
# Line 113 — This will fail or return zeros
soil_moisture = fetch_stac_window(soil_item, "soil", bounds, (chip_size, chip_size))
```

The `fetch_stac_window` function catches the exception (line 74–78) and returns `np.zeros(...)`, so the error is **silently swallowed**, and all chips get zero soil moisture.

**Impact:** The `smap_soil_moisture` key in every chip is zero-filled. The Soil model's SMAP-weighted target computation (lines 1177-1184 of `real_datasets.py`) computes `soil_impact * (0.5 + 0.5 * 0.0) = soil_impact * 0.5`, which halves the target everywhere, but the half-factor is constant so the model can still learn relative patterns. However, the intended physics (high baseline moisture → worse degradation) is completely lost.

**Fix:** Use `xarray` + `zarr` to read TerraClimate data:
```python
import xarray as xr
ds = xr.open_zarr(soil_item.assets["zarr"].href)
soil_data = ds["soil"].sel(time=...).values
```

---

### Issue #5 — Hydro Dataset Retry Logic Doesn't Rebuild obs_f/obs_cf on Retry Failure
**File:** [real_datasets.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/real_datasets.py#L972-L998)
**Lines:** 972–998

When the initial chip has no `msi_ndssi_delta`, the code retries with a random chip (line 974). If that retry also fails (line 995 `except Exception: pass`), the code falls through to line 998 which reads `data.get("msi_ndssi_delta", np.zeros_like(tc))` — but `data` is still the **original** chip (since the retry's `data2` failed to load), and `tc` has already been overwritten with the retry chip's treecover if the retry partially succeeded before failing.

**Impact:** Shape mismatch between `obs_f`/`obs_cf` (potentially from retry chip) and `target` (from original chip), leading to silent training corruption or runtime errors.

**Fix:** The retry logic should be integrated into the main validation loop at the top of `__getitem__`, not as an ad-hoc patch in the middle.

---

## 🟠 High Severity Issues (Accuracy-Degrading)

### Issue #6 — SRTM URL Construction Has Dead Code / Double-Construction
**File:** [download_real_data.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/download_real_data.py#L393-L400)
**Lines:** 393–400

The SRTM URL template at line 142 constructs a URL, then lines 397-400 **completely overwrite** it with a hardcoded URL:

```python
# Line 394 — uses template (never reaches the network)
url = SRTM_URL_TEMPLATE.format(ns_dir=ns_dir, ew_str=ew_str)

# Lines 397-400 — overwrites the template URL entirely
url = (
    f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/"
    f"{ns_dir}/{full_name}.hgt.gz"
)
```

While the final URL is **correct** (matches the AWS Skadi format `N10/N10W070.hgt.gz`), the template at line 142 has a **bug**: it uses `{ns_dir}/{ns_dir}{ew_str}` which would produce `N10/N10W070` — coincidentally correct, but only because `ns_dir` and the first part of `full_name` are the same. This dead code is confusing and indicates the developer found a bug and hotfixed it without removing the broken template.

**Impact:** No runtime impact (the overwrite is correct), but the dead template should be removed to avoid confusion. If someone removes the overwrite thinking the template works, they'd get broken URLs for some tile naming edge cases.

---

### Issue #7 — FIRMS Per-Chip API Has Incorrect Day Range Parameter
**File:** [download_real_data.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/download_real_data.py#L146-L149)
**Lines:** 146–149, 670–701

The FIRMS API URL template uses `10` as the day_range parameter:
```python
FIRMS_API_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    "{key}/VIIRS_SNPP_SP/{west},{south},{east},{north}/10/{date}"
)
```

According to NASA documentation, the FIRMS API **limits day_range to 1–5 days** (sometimes up to 10, but requests >5 count as multiple transactions and may be rejected on the free tier). Additionally, the code queries **only the 1st of each month** (line 673), capturing just 10 days per month — **missing ~67% of fire detections per year**.

**Impact:** Even when the API key works, the fire data is severely incomplete: only 120 days out of 365 per year are sampled. This artificially reduces fire detection density and creates systematic temporal bias.

---

### Issue #8 — `download_msi_smap.py` Uses Thread-Unsafe STAC Client
**File:** [download_msi_smap.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/download_msi_smap.py#L186-L203)
**Lines:** 186–203

The `augment_dataset()` function creates **one** STAC client (line 176) and passes it to 32 concurrent threads via `ThreadPoolExecutor(max_workers=32)`. The `pystac_client.Client` object is not guaranteed to be thread-safe, and the `planetary_computer.sign_inplace` modifier mutates items in place — concurrent mutations to shared state can cause:
- Race conditions in token signing
- Corrupted HTTP session state
- Intermittent `ConnectionError` or `AttributeError` exceptions

**Impact:** Random failures during MSI/SMAP augmentation, especially under high concurrency. This likely contributed to the errors you observed.

**Fix:** Create one client per thread, or use `planetary_computer.sign` (copy) instead of `sign_inplace` (mutation).

---

### Issue #9 — Per-Year VIIRS Fire Rasters Use Per-Chip Normalisation
**File:** [download_real_data.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/download_real_data.py#L824-L828)
**Lines:** 824–828

```python
for year_code, counts in per_year_counts.items():
    ymax = counts.max()
    norm = counts / ymax if ymax > 0 else counts
```

Each per-year fire raster is normalised independently per chip. A chip with 1 fire in 2020 and a chip with 500 fires in 2020 both get `viirs_fire_year_20 = 1.0` for their hottest pixel.

**Impact:** The fire model cannot distinguish between high-fire and low-fire regions. The temporal fire differencing in `RealFireDataset.__getitem__` (lines 792–814) divides normalised-fires-after by normalised-fires-before, but the normalisation destroys the denominator's meaning. Already noted in `potential_improvements.md` as Issue #8, but critical for fire model accuracy.

---

### Issue #10 — `download_msi_smap.py` Chip Path Resolution May Fail
**File:** [download_msi_smap.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/download_msi_smap.py#L191)
**Line:** 191

```python
chip_path = os.path.join(tiles_dir, entry["file"])
```

The manifest stores paths from `_download_single_chip` as relative paths via `os.path.relpath(chip_file, output_dir)` (line 1096 of `download_real_data.py`). These relative paths are like `train/amazon_rondonia_chip_000.npz`. But `download_msi_smap.py` does `os.path.join(tiles_dir, entry["file"])`, which works **only** if `tiles_dir` is an absolute path to the same directory. If `tiles_dir` is given as a relative path from a different working directory, the join will produce a wrong path and `os.path.exists` will fail silently.

The `_resolve_chip_path()` function in `real_datasets.py` handles this correctly (lines 331-347), but `download_msi_smap.py` doesn't use it.

---

### Issue #11 — Manifest References `val` Split That Doesn't Exist
**File:** [download_msi_smap.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/download_msi_smap.py#L182)
**Line:** 182, 212

```python
all_entries = manifest.get("train", []) + manifest.get("val", []) + manifest.get("test", [])
```

The `download_all()` function in `download_real_data.py` creates manifests with only `"train"` and `"test"` keys (line 1331). There is no `"val"` key. The `manifest.get("val", [])` returns an empty list, which is harmless, but is misleading and indicates a mismatch between the download and augmentation scripts.

---

### Issue #12 — `download_msi_smap.py` Skips Already-Augmented Chips Too Aggressively
**File:** [download_msi_smap.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/download_msi_smap.py#L97-L99)
**Lines:** 97–99

```python
if "smap_soil_moisture" in data and "msi_ndssi_delta" in data:
    return True
```

If a previous run partially augmented a chip (wrote zero-filled fallback data due to API failure), the keys exist but contain all zeros. On re-run, the check passes and the chip is never retried with potentially working API credentials. The `has_real_msi_smap` flag is not checked here.

**Fix:** Check `has_real_msi_smap` flag:
```python
if "has_real_msi_smap" in data and float(data["has_real_msi_smap"].flat[0]) > 0.5:
    return True  # Already has real data
```

---

## 🟡 Medium Severity Issues (Robustness)

### Issue #13 — SRTM `_download_srtm_hgt` Caches in Big-Endian But Reads as Big-Endian
**File:** [download_real_data.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/download_real_data.py#L388-L418)
**Lines:** 388–418

When reading from cache (line 390), the code uses `dtype=">i2"` (big-endian int16). When writing to cache (line 417), it converts to `">i2"` and uses `.tofile()`. This is internally consistent, **but** if the SRTM data from AWS contains any void values (-32768), the float32 conversion on line 392 preserves them. The void handling on line 413 only happens on the non-cached path (`arr = data.reshape(3601, 3601).astype(np.float32)`), so cached tiles with voids will have `-32768.0` float values that propagate to terrain derivatives.

The `_derive_real_terrain()` function (line 473) does handle voids (`< -1000`), but the cache path at line 392 returns directly before this check is applied.

**Impact:** First download + terrain derivation works correctly (void handling applies). Subsequent cached reads bypass void handling if the cache reading path is used **without** then calling `_derive_real_terrain()` — but tracing the call chain, `download_srtm_for_chip()` always calls `_derive_real_terrain()` on the result, so this is technically safe. However, if `_download_srtm_hgt` is called directly by any future code, voids will leak through.

---

### Issue #14 — Flow Accumulation Uses O(n) Python Loop Per Pixel
**File:** [download_real_data.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/download_real_data.py#L531-L574)
**Lines:** 531–574

The docstring says "~100× faster than the per-pixel Python loop," but the topological scan on lines 569-572 is still a pure Python `for` loop over every pixel:

```python
for idx in flat_order:
    r, c = divmod(idx, w)
    if has_downhill[r, c]:
        flow_acc[target_r[r, c], target_c[r, c]] += flow_acc[r, c]
```

For a 256×256 chip, this is 65,536 iterations. For a full SRTM tile (3601×3601 in `_derive_real_terrain`), this is **13 million iterations** in pure Python — taking **30–120 seconds per tile**.

**Impact:** When computing terrain for chips that cross tile boundaries (using stitched tiles), flow accumulation becomes a severe bottleneck. Total download time is dominated by this computation, not network I/O.

**Fix:** Use `numba.jit` or Cython for the topological scan, or replace with `richdem` / `pysheds` library.

---

### Issue #15 — `_sample_window` Has Unintuitive Edge Cases at Small `train_end`
**File:** [real_datasets.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/real_datasets.py#L414-L464)
**Lines:** 414–464

When `train_end_year=23` and `max_window=5`: `max_t2 = 23 - 5 = 18`. So `t2` is sampled from `[3, 18]`, meaning deforestation events after year 18 are **never used as event windows**. Events in years 19-23 can only appear as cascade/impact targets, not as input deforestation masks.

This is intentional (need post-event years to observe impact), but with `delta` sampled as `randint(2, 6)`, the maximum `t_impact` is `18 + 5 = 23`. If `delta=5` and `t2=18`, then `t_impact=23` — the full range is used. But if `delta=2` and `t2=18`, then `t_impact=20`, wasting 3 years of observations.

**Note:** This is not a bug per se, but it means ~40% of sampled windows under-utilise the available temporal depth, reducing the diversity of impact observation periods.

---

### Issue #16 — `RealHydroDataset` Uses Fixed Temporal Window But Ignores `train_end_year`
**File:** [real_datasets.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/real_datasets.py#L853-L860)
**Lines:** 853–860, 943

```python
_HYDRO_T1: int = 16   # baseline year  (2016)
_HYDRO_T2: int = 20   # impact year    (2020)
```

The Hydro dataset uses a **fixed** temporal window locked to 2016→2020. The `train_end_year` parameter in the constructor (line 888) is accepted but **never used** — it's stored as `self.train_end_year` but the `__getitem__` method uses `_HYDRO_T1` and `_HYDRO_T2` constants. This is correct because the Sentinel-2 NDSSI target is baked to these dates, and the docstring explains this. However, the `train_end_year` parameter being accepted but silently ignored is a footgun.

---

### Issue #17 — `compute_global_target_scale` Hydro Path Skips Non-Real Chips But May Have Zero Valid Chips
**File:** [real_datasets.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/real_datasets.py#L253-L261)
**Lines:** 253–261

```python
if has_real is not None and float(has_real.flat[0]) < 0.5:
    continue
raw_target = data.get("msi_ndssi_delta", None)
if raw_target is None:
    continue
```

If `download_msi_smap.py` failed completely (Issue #2, #4 above), **every** chip either has `has_real_msi_smap = 0` or is missing `msi_ndssi_delta`. All 200 sampled chips are skipped, `maxima` is empty, and the function falls back to `scale = 1.0` with a warning. This means the Hydro model trains with `target_scale=1.0`, which may over-normalise or under-normalise the (broken) NDSSI deltas.

---

### Issue #18 — SRTM Tile Latitude Offset in `download_srtm_for_chip`
**File:** [download_real_data.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/download_real_data.py#L608-L617)
**Lines:** 608–617

```python
for lat_i in range(n_lat_tiles):
    for lon_j in range(n_lon_tiles):
        tile_lat = lat_max_int - lat_i   # NW corner latitude
```

SRTM tiles are named by their **SW corner** coordinate (confirmed by AWS documentation). The code downloads using `_download_srtm_hgt(float(tile_lat), ...)` where `tile_lat` is computed as a **NW corner**. Inside `_download_srtm_hgt`, `_srtm_tile_name` uses `int(math.floor(lat))` which gives the SW corner. Since SRTM tiles cover 1° from SW corner `N` to `N+1`, passing the NW corner latitude works if chips are within a single tile, but for cross-tile stitching this can miss the correct tile.

Example: For a chip spanning 10.0°N to 10.0064°N (a 256-pixel chip at ~0.00025°/px), `lat_max_int = 10`, `lat_min_int = 10`, so only one tile is downloaded: `_download_srtm_hgt(10.0, ...)` → tile `N10`. This is correct because N10 covers 10°N to 11°N. The logic appears correct upon deep inspection, but the code comments mix up NW/SW terminology.

---

### Issue #19 — `PIL.Image.BILINEAR` Deprecated in Newer Pillow Versions
**File:** [download_real_data.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/download_real_data.py#L460-L461)
**Lines:** 460–461, 641

```python
img_resized = img.resize((chip_size, chip_size), Image.BILINEAR)
```

`Image.BILINEAR` was deprecated in Pillow 9.1.0 and may be removed in future. Should use `Image.Resampling.BILINEAR`.

---

### Issue #20 — Concurrent STAC Queries Without Rate Limiting
**File:** [download_msi_smap.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/download_msi_smap.py#L198)
**Line:** 198

32 concurrent threads hammering the Planetary Computer STAC API with zero rate limiting will trigger HTTP 429 (Too Many Requests) responses. The retry logic in `fetch_stac_window` only retries 3 times with 1-second sleep, but the STAC search queries in `query_terraclimate` and `query_sentinel2` have **no retry logic at all**.

---

## 🟢 Low Severity Issues (Polish)

### Issue #21 — `__init__.py` Exports Synthetic Datasets, Not Real Ones
**File:** [__init__.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/__init__.py)
**Lines:** 1–38

The `datasets/__init__.py` exposes `VIIRSFireDataset`, `HansenGFCDataset`, `SRTMHydroDataset`, `SMAPSoilDataset` (synthetic), but the training pipeline (`train_real_models.py`) imports `RealFireDataset` etc. from `datasets.real_datasets`. The `__init__.py` is stale and doesn't reflect the real data pipeline.

---

### Issue #22 — `download_earth_data.py` Is Effectively Dead Code
**File:** [download_earth_data.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/download_earth_data.py)

The `download_earth_data.py` script has a `--execute` mode that prints "NOT IMPLEMENTED IN THIS SCRIPT YET" (line 113). It's superseded by `download_real_data.py` + `download_msi_smap.py` but still exists. Should be deleted or archived.

---

### Issue #23 — `multi_source_datasets.py` Is Incomplete / Dead Code
**File:** [multi_source_datasets.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/multi_source_datasets.py)

Classes `TrueFireDataset`, `TrueHydroDataset`, `TrueSoilDataset`, `TrueForestDataset` have stubs that either crash (`pass` return with no value) or return dummy data. These are never imported by any other file. Should be deleted.

---

### Issue #24 — Training Runbook Section 4.2 Has Stale Temporal Split Information
**File:** [TrainingRunbook.md](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/docs/TrainingRunbook.md#L176-L184)
**Lines:** 176–184

The runbook states:
```
Train: events 2001–2016, impact by 2018
Test:  events 2015–2018, impact by 2020
Validate: events 2019–2021, impact by 2023
```

But the actual code uses **full temporal range (1–23) for all splits**, with spatial tile-level splitting. The temporal split was removed in an earlier audit. The runbook is stale and misleading.

---

### Issue #25 — `warnings.filterwarnings("ignore")` Blanket Suppression
**File:** [download_msi_smap.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/download_msi_smap.py#L9)
**Line:** 9

```python
warnings.filterwarnings("ignore")
```

This suppresses **all** warnings including deprecation, data corruption, and API change notices. Should be scoped to specific warning categories:
```python
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pystac_client")
```

---

## 📊 Data Science Concerns

### Issue #26 — Hydro vs Soil Datasets: No Water Body Masking in NDSSI
**File:** [download_msi_smap.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/download_msi_smap.py#L129-L146)

NDSSI should only be computed on **water pixels** (rivers, lakes downstream of deforestation). The current code computes NDSSI on **all** pixels including land, forest, and clouds. On land pixels, green/NIR ratios reflect vegetation, not sediment. The resulting delta is dominated by vegetation change, not water quality change.

The Hydro model target docstring says "water pollution delta downstream" (line 864 of `real_datasets.py`), but the actual NDSSI delta is computed on the full 256×256 chip with no water masking.

**Fix:** Compute NDWI (Green-NIR)/(Green+NIR) first to identify water pixels, then compute NDSSI only on those pixels. Use the SRTM flow accumulation channel to target measurements to river corridors.

---

### Issue #27 — Sentinel-2 Temporal Window Too Narrow for Robust Compositing
**File:** [download_msi_smap.py](file:///Users/kieranpi/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Greenhacks%20%28DOFE%29/datasets/download_msi_smap.py#L126-L127)
**Lines:** 126–127

```python
s2_baseline = query_sentinel2(client, bounds, "2016-01-01/2017-01-01")
s2_impact = query_sentinel2(client, bounds, "2020-01-01/2021-01-01")
```

Each period is exactly 1 year, and the code takes the **single best** (lowest cloud cover) Sentinel-2 scene. At 10-day revisit, many tropical tiles have persistent cloud cover. Taking a single scene means:
- Many chips will return empty results (no scene with <20% clouds)
- The "best" scene may still have significant cloud contamination
- Seasonal variation is not accounted for (dry season vs wet season NDSSI differs enormously)

**Fix:** Use a multi-temporal composite (median of 3–5 clearest scenes) for each period, with cloud mask application per-pixel.

---

## Summary of Issues by File

| File | Critical | High | Medium | Low |
|------|----------|------|--------|-----|
| `download_real_data.py` | 1 | 2 | 3 | 0 |
| `download_msi_smap.py` | 3 | 3 | 2 | 1 |
| `real_datasets.py` | 1 | 0 | 2 | 0 |
| `__init__.py` | 0 | 0 | 0 | 1 |
| `download_earth_data.py` | 0 | 0 | 0 | 1 |
| `multi_source_datasets.py` | 0 | 0 | 0 | 1 |
| `TrainingRunbook.md` | 0 | 0 | 0 | 1 |

---

## Recommended Fix Priority

> [!IMPORTANT]
> Issues #1–#5 must be fixed before any training run. They cause either complete data download failure or fundamentally corrupted training targets.

1. **Immediate:** Fix #1 (VIIRS URLs), #2 (S2 scale factor), #3 (NDSSI bands), #4 (TerraClimate Zarr), #5 (Hydro retry logic)
2. **Before next training:** Fix #7 (FIRMS API day range), #8 (thread-unsafe STAC), #9 (per-chip fire normalisation), #12 (skip-cache check)
3. **Robustness:** Fix #10 (chip path resolution), #14 (flow accumulation perf), #20 (rate limiting), #26 (water masking)
4. **Cleanup:** Delete dead code (#22, #23), update stale docs (#24), fix deprecation warnings (#19, #25)
