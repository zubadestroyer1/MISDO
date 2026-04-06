#!/usr/bin/env python3
"""
VIIRS Download Pipeline — Real Smoke Tests
===========================================
Tests every component of the VIIRS download pipeline against real
NASA endpoints, without needing a FIRMS API key for most tests.

Run:  python tests/smoke_test_viirs.py
"""
import sys
import os
import time
import json
import urllib.request
import urllib.error
import tempfile
import traceback
import numpy as np

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"
SKIP = "⏭️  SKIP"

results = []

def record(name, status, detail=""):
    results.append((name, status, detail))
    print(f"  {status}  {name}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"         {line}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: FIRMS API endpoint reachability (no key needed)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  VIIRS Download Pipeline — Smoke Tests")
print("=" * 70)
print()

print("── Test 1: FIRMS API Endpoint Reachability ──")
try:
    # Hit the FIRMS data availability endpoint (no key needed for this check)
    url = "https://firms.modaps.eosdis.nasa.gov/api/data_availability/"
    req = urllib.request.Request(url, headers={"User-Agent": "MISDO-SmokeTest/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        status_code = resp.getcode()
        body = resp.read().decode("utf-8")[:500]
    if status_code == 200:
        record("FIRMS base endpoint reachable", PASS,
               f"HTTP {status_code}, response length={len(body)} chars")
    else:
        record("FIRMS base endpoint reachable", FAIL,
               f"HTTP {status_code}")
except Exception as e:
    record("FIRMS base endpoint reachable", FAIL, str(e))

# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: FIRMS API with no key (should return an error, not hang)
# ═══════════════════════════════════════════════════════════════════════════
print("── Test 2: FIRMS API Error Handling (invalid key) ──")
try:
    # Use the exact URL template from the code but with a dummy key
    from datasets.download_real_data import FIRMS_API_URL
    test_url = FIRMS_API_URL.format(
        key="INVALID_KEY_12345",
        west=-62.0, south=-10.5, east=-61.5, north=-10.0,
        date="2023-06-15",
    )
    record("URL template renders correctly", PASS, f"URL: {test_url}")

    req = urllib.request.Request(test_url, headers={"User-Agent": "MISDO-SmokeTest/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8")[:300]
            record("Invalid key response", WARN,
                   f"Got HTTP 200 with body: {body[:200]}...")
    except urllib.error.HTTPError as e:
        record("Invalid key rejected", PASS,
               f"HTTP {e.code} — server correctly rejects invalid key")
    except urllib.error.URLError as e:
        record("Invalid key request", FAIL,
               f"URL error (not HTTP error): {e.reason}")
except Exception as e:
    record("FIRMS API error handling", FAIL, traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: FIRMS API with env key (if available)
# ═══════════════════════════════════════════════════════════════════════════
print("── Test 3: FIRMS API with Real Key ──")
firms_key = os.environ.get("FIRMS_MAP_KEY")
if firms_key:
    try:
        from datasets.download_real_data import FIRMS_API_URL
        test_url = FIRMS_API_URL.format(
            key=firms_key,
            west=-62.0, south=-10.5, east=-61.5, north=-10.0,
            date="2023-06-15",
        )
        req = urllib.request.Request(test_url, headers={"User-Agent": "MISDO-SmokeTest/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            lines = body.strip().split("\n")
            if len(lines) >= 2:
                record("FIRMS API real query", PASS,
                       f"Got {len(lines)-1} fire detections\n"
                       f"Header: {lines[0][:100]}\n"
                       f"First row: {lines[1][:100]}")
            elif len(lines) == 1:
                record("FIRMS API real query", WARN,
                       f"Got header but no data rows (may be no fires on that date)\n"
                       f"Header: {lines[0][:100]}")
            else:
                record("FIRMS API real query", FAIL, "Empty response")
    except Exception as e:
        record("FIRMS API real query", FAIL, traceback.format_exc())
else:
    record("FIRMS API real query", SKIP,
           "No FIRMS_MAP_KEY env var. Set it to test: export FIRMS_MAP_KEY=your_key")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: _download_viirs_fires function (with real key)
# ═══════════════════════════════════════════════════════════════════════════
print("── Test 4: _download_viirs_fires Function ──")
if firms_key:
    try:
        from datasets.download_real_data import _download_viirs_fires
        t0 = time.time()
        fires = _download_viirs_fires(
            west=-62.0, south=-10.5, east=-61.5, north=-10.0,
            firms_key=firms_key,
            years=[2023],  # Single year to keep it fast
        )
        elapsed = time.time() - t0
        if fires:
            record("_download_viirs_fires()", PASS,
                   f"Got {len(fires)} detections in {elapsed:.1f}s\n"
                   f"Sample keys: {list(fires[0].keys())}\n"
                   f"Sample: {fires[0]}")
        else:
            record("_download_viirs_fires()", FAIL,
                   f"Returned None in {elapsed:.1f}s (silent failure — all 12 API calls failed)")
    except Exception as e:
        record("_download_viirs_fires()", FAIL, traceback.format_exc())
else:
    record("_download_viirs_fires()", SKIP, "No FIRMS_MAP_KEY")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: VIIRSArchive class with synthetic CSV
# ═══════════════════════════════════════════════════════════════════════════
print("── Test 5: VIIRSArchive (Bulk CSV Loader) ──")
try:
    from datasets.download_real_data import VIIRSArchive

    # Create a small synthetic FIRMS CSV
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("latitude,longitude,bright_ti4,bright_ti5,frp,confidence,acq_date\n")
        # 50 synthetic fire points in Amazon Rondônia
        rng = np.random.RandomState(42)
        for i in range(50):
            lat = -10.0 + rng.uniform(-0.5, 0.0)
            lon = -62.0 + rng.uniform(0.0, 0.5)
            ti4 = 330 + rng.uniform(0, 100)
            ti5 = 280 + rng.uniform(0, 50)
            frp = rng.uniform(5, 200)
            conf = rng.choice(["low", "nominal", "high"])
            date = f"2023-{rng.randint(1,13):02d}-{rng.randint(1,29):02d}"
            f.write(f"{lat:.4f},{lon:.4f},{ti4:.1f},{ti5:.1f},{frp:.1f},{conf},{date}\n")
        csv_path = f.name

    archive = VIIRSArchive([csv_path])
    record("VIIRSArchive loads CSV", PASS, f"Loaded from {csv_path}")

    # Query within bounds
    fires = archive.query(west=-62.5, south=-10.5, east=-61.5, north=-9.5)
    if fires and len(fires) > 0:
        record("VIIRSArchive.query() returns data", PASS,
               f"Found {len(fires)} fires in bbox")
    else:
        record("VIIRSArchive.query() returns data", FAIL,
               "No fires returned from synthetic data!")

    # Query outside bounds (should return None)
    empty = archive.query(west=100, south=40, east=101, north=41)
    if empty is None:
        record("VIIRSArchive.query() empty bbox", PASS, "Correctly returns None")
    else:
        record("VIIRSArchive.query() empty bbox", FAIL,
               f"Should be None, got {len(empty)} results")

    os.unlink(csv_path)

except Exception as e:
    record("VIIRSArchive", FAIL, traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: _rasterize_fires function
# ═══════════════════════════════════════════════════════════════════════════
print("── Test 6: _rasterize_fires (point → raster) ──")
try:
    from datasets.download_real_data import _rasterize_fires

    # Build synthetic fire list (same format as FIRMS API returns)
    test_fires = []
    rng = np.random.RandomState(42)
    for i in range(100):
        test_fires.append({
            "latitude": str(-10.0 + rng.uniform(-0.25, 0.0)),
            "longitude": str(-62.0 + rng.uniform(0.0, 0.25)),
            "bright_ti4": str(330 + rng.uniform(0, 100)),
            "bright_ti5": str(280 + rng.uniform(0, 50)),
            "frp": str(rng.uniform(5, 200)),
            "confidence": rng.choice(["low", "nominal", "high"]),
            "acq_date": f"2023-{rng.randint(1,13):02d}-{rng.randint(1,29):02d}",
        })

    result = _rasterize_fires(
        test_fires,
        west=-62.25, south=-10.25, east=-62.0, north=-10.0,
        chip_size=256,
    )

    expected_keys = {
        "viirs_fire_count", "viirs_mean_frp", "viirs_max_bright_ti4",
        "viirs_max_bright_ti5", "viirs_confidence", "viirs_persistence",
        "has_real_viirs",
    }
    # Also expect per-year keys
    per_year_keys = {f"viirs_fire_year_{y:02d}" for y in range(12, 24)}

    got_keys = set(result.keys())
    missing = expected_keys - got_keys
    extra_expected_year = per_year_keys & got_keys

    if not missing:
        detail_lines = []
        for k in sorted(result.keys()):
            v = result[k]
            detail_lines.append(
                f"{k:30s}  shape={str(v.shape):15s}  "
                f"min={v.min():.4f}  max={v.max():.4f}  "
                f"dtype={v.dtype}"
            )
        # Check for non-zero fire data
        fc = result["viirs_fire_count"]
        n_fire_pixels = (fc > 0).sum()
        detail_lines.append(f"\nFire pixels (non-zero): {n_fire_pixels}/{fc.size}")

        if n_fire_pixels > 0:
            record("_rasterize_fires()", PASS, "\n".join(detail_lines))
        else:
            record("_rasterize_fires()", WARN,
                   "All zeros — fires may have fallen outside bbox\n" +
                   "\n".join(detail_lines))
    else:
        record("_rasterize_fires()", FAIL, f"Missing keys: {missing}")

except Exception as e:
    record("_rasterize_fires()", FAIL, traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════════
# TEST 7: Existing chips — check for VIIRS data
# ═══════════════════════════════════════════════════════════════════════════
print("── Test 7: Existing Chip VIIRS Content ──")
try:
    tile_dir = os.path.join(ROOT, "datasets", "real_tiles")
    all_chips = []
    for split in ["train", "test"]:
        split_dir = os.path.join(tile_dir, split)
        if os.path.isdir(split_dir):
            for f in sorted(os.listdir(split_dir)):
                if f.endswith(".npz"):
                    all_chips.append(os.path.join(split_dir, f))

    if not all_chips:
        record("Existing chips audit", SKIP, "No .npz chips found")
    else:
        viirs_present = 0
        viirs_absent = 0
        srtm_real = 0
        srtm_proxy = 0
        detail_lines = [f"Auditing {len(all_chips)} chips:\n"]

        for chip_path in all_chips:
            data = np.load(chip_path, allow_pickle=True)
            keys = list(data.keys())
            has_viirs = "has_real_viirs" in keys
            viirs_keys = [k for k in keys if k.startswith("viirs_")]
            has_srtm = data["has_real_srtm"][0] > 0 if "has_real_srtm" in keys else False

            chip_name = os.path.basename(chip_path)
            parent = os.path.basename(os.path.dirname(chip_path))

            if has_viirs:
                viirs_present += 1
            else:
                viirs_absent += 1

            if has_srtm:
                srtm_real += 1
            else:
                srtm_proxy += 1

            viirs_str = f"VIIRS:✓ ({len(viirs_keys)} keys)" if has_viirs else "VIIRS:✗"
            srtm_str = "SRTM:real" if has_srtm else "SRTM:proxy"
            detail_lines.append(
                f"  {parent}/{chip_name:15s}  {viirs_str:25s}  {srtm_str:12s}  "
                f"keys={len(keys)}"
            )

        detail_lines.append(f"\nSummary:")
        detail_lines.append(f"  VIIRS present: {viirs_present}/{len(all_chips)}")
        detail_lines.append(f"  VIIRS absent:  {viirs_absent}/{len(all_chips)}")
        detail_lines.append(f"  SRTM real:     {srtm_real}/{len(all_chips)}")
        detail_lines.append(f"  SRTM proxy:    {srtm_proxy}/{len(all_chips)}")

        if viirs_present > 0:
            record("Existing chips VIIRS audit", PASS, "\n".join(detail_lines))
        else:
            record("Existing chips VIIRS audit", WARN,
                   "No existing chips contain real VIIRS data\n" +
                   "\n".join(detail_lines))

except Exception as e:
    record("Existing chips audit", FAIL, traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════════
# TEST 8: download_viirs_archive function (scan for cache)
# ═══════════════════════════════════════════════════════════════════════════
print("── Test 8: download_viirs_archive (Cache Scanner) ──")
try:
    from datasets.download_real_data import download_viirs_archive

    tile_dir = os.path.join(ROOT, "datasets", "real_tiles")
    csv_paths = download_viirs_archive(tile_dir)

    if csv_paths:
        record("download_viirs_archive()", PASS,
               f"Found {len(csv_paths)} CSV files")
    else:
        record("download_viirs_archive()", WARN,
               "No VIIRS CSV files in cache — bulk archive not configured.\n"
               "To fix: download CSVs from https://firms.modaps.eosdis.nasa.gov/download/\n"
               f"and place them in: {os.path.join(tile_dir, '.viirs_cache/')}")

except Exception as e:
    record("download_viirs_archive()", FAIL, traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════════
# TEST 9: End-to-end VIIRSArchive → rasterize pipeline
# ═══════════════════════════════════════════════════════════════════════════
print("── Test 9: End-to-End Bulk Archive → Rasterize ──")
try:
    from datasets.download_real_data import VIIRSArchive, _rasterize_fires

    # Create a realistic synthetic CSV with 500 fires across a tile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("latitude,longitude,bright_ti4,bright_ti5,frp,confidence,acq_date\n")
        rng = np.random.RandomState(123)
        for i in range(500):
            # Spread across Amazon tile 10S_070W (lat -10 to -20, lon -70 to -60)
            lat = rng.uniform(-15.0, -10.0)
            lon = rng.uniform(-70.0, -65.0)
            ti4 = 330 + rng.uniform(0, 150)
            ti5 = 280 + rng.uniform(0, 70)
            frp = rng.exponential(50)
            conf = rng.choice(["low", "nominal", "high"])
            year = rng.choice([2020, 2021, 2022, 2023])
            month = rng.randint(1, 13)
            day = rng.randint(1, 29)
            f.write(f"{lat:.6f},{lon:.6f},{ti4:.1f},{ti5:.1f},{frp:.1f},{conf},{year}-{month:02d}-{day:02d}\n")
        csv_path = f.name

    archive = VIIRSArchive([csv_path])

    # Simulate what _download_single_chip does for the archive path
    west, south, east, north = -67.0, -12.5, -66.5, -12.0
    buffer = 0.01
    fires = archive.query(
        west - buffer, south - buffer, east + buffer, north + buffer
    )

    if fires:
        rasters = _rasterize_fires(fires, west, south, east, north, chip_size=256)
        fc = rasters["viirs_fire_count"]
        n_fire_px = (fc > 0).sum()
        has_years = [k for k in rasters if k.startswith("viirs_fire_year_")]

        record("End-to-end archive → rasterize", PASS,
               f"Query returned {len(fires)} fires\n"
               f"Rasterized: {n_fire_px} fire pixels in 256x256 chip\n"
               f"Per-year channels: {len(has_years)}\n"
               f"has_real_viirs: {rasters['has_real_viirs']}")
    else:
        record("End-to-end archive → rasterize", WARN,
               "No fires in query bbox (synthetic data may not overlap)")

    os.unlink(csv_path)

except Exception as e:
    record("End-to-end archive → rasterize", FAIL, traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════════
# TEST 10: Coverage analysis of _download_viirs_fires
# ═══════════════════════════════════════════════════════════════════════════
print("── Test 10: Temporal Coverage Analysis ──")
try:
    from datasets.download_real_data import FIRMS_API_URL

    # Simulate the date iteration logic to count how many days are actually queried
    years_tested = list(range(2018, 2024))
    dates_queried = []
    for year in years_tested:
        for month in range(1, 13):
            date = f"{year}-{month:02d}-01"
            dates_queried.append(date)

    total_days = sum(366 if y % 4 == 0 else 365 for y in years_tested)
    # DAY_RANGE=1 in URL means only 1 day per request
    days_covered = len(dates_queried)
    coverage_pct = days_covered / total_days * 100

    detail = (
        f"Years: {years_tested[0]}–{years_tested[-1]}\n"
        f"Total days in range: {total_days}\n"
        f"Days actually queried: {days_covered} (1st of each month)\n"
        f"Temporal coverage: {coverage_pct:.1f}%\n"
        f"DAY_RANGE in URL: 1 (only 1 day per API call)\n"
        f"\n"
        f"API calls per chip: {len(dates_queried)}\n"
        f"For 300k chips: {len(dates_queried) * 300_000:,} total API calls\n"
        f"At 5000/10min rate limit: {len(dates_queried) * 300_000 / 5000 * 10 / 60:.0f} hours"
    )

    if coverage_pct < 10:
        record("Temporal coverage", FAIL, detail)
    else:
        record("Temporal coverage", PASS, detail)

except Exception as e:
    record("Temporal coverage analysis", FAIL, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  SUMMARY")
print("=" * 70)
n_pass = sum(1 for _, s, _ in results if s == PASS)
n_fail = sum(1 for _, s, _ in results if s == FAIL)
n_warn = sum(1 for _, s, _ in results if s == WARN)
n_skip = sum(1 for _, s, _ in results if s == SKIP)
total = len(results)

print(f"\n  {PASS} Passed: {n_pass}/{total}")
print(f"  {FAIL} Failed: {n_fail}/{total}")
print(f"  {WARN} Warnings: {n_warn}/{total}")
print(f"  {SKIP} Skipped: {n_skip}/{total}")

print(f"\nDetailed results:")
for name, status, detail in results:
    print(f"  {status}  {name}")

print()
sys.exit(1 if n_fail > 0 else 0)
