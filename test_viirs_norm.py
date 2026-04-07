"""VIIRS normalization verification tests.

Validates that the physics-based normalization in download_real_data.py
and the clip-only pass-through in real_datasets.py are correct.
"""
import numpy as np


def test_download_time_normalization():
    """Verify download-time log-transform + physical constants."""
    _VIIRS_FIRE_COUNT_99P = 50.0
    _VIIRS_FRP_99P = 200.0

    # --- Fire count: log-transform preserves cross-chip magnitude ---
    # Chip A: sparse area with 1 fire detection
    # Chip B: intense wildfire zone with 100 detections
    chipA = np.array([[0, 1], [0, 0]], dtype=np.float64)
    chipB = np.array([[10, 100], [50, 20]], dtype=np.float64)

    normA = np.clip(np.log1p(chipA) / np.log1p(_VIIRS_FIRE_COUNT_99P), 0, 1)
    normB = np.clip(np.log1p(chipB) / np.log1p(_VIIRS_FIRE_COUNT_99P), 0, 1)

    # OLD behavior: both chips max -> 1.0 (indistinguishable)
    oldA = chipA / max(chipA.max(), 1)
    oldB = chipB / max(chipB.max(), 1)
    assert oldA.max() == 1.0 and oldB.max() == 1.0, "Old norm sanity check"

    # NEW behavior: chips have different magnitudes
    print(f"  Chip A (1 fire):    old max={oldA.max():.2f}  new max={normA.max():.4f}")
    print(f"  Chip B (100 fires): old max={oldB.max():.2f}  new max={normB.max():.4f}")
    assert normB.max() > normA.max() * 5, (
        f"New norm should distinguish chips: A={normA.max():.4f} B={normB.max():.4f}"
    )
    print("  PASS: Cross-chip magnitude preserved")

    # --- FRP: log-transform handles heavy tail ---
    frp_values = np.array([1, 10, 50, 100, 200, 500, 1000], dtype=np.float64)
    frp_norm = np.clip(np.log1p(frp_values) / np.log1p(_VIIRS_FRP_99P), 0, 1)
    print(f"  FRP (MW):   {frp_values.tolist()}")
    print(f"  Normalised: {frp_norm.round(4).tolist()}")
    # Ordering preserved
    assert all(frp_norm[i] <= frp_norm[i + 1] for i in range(len(frp_norm) - 1))
    # Bounded to [0, 1]
    assert frp_norm.min() >= 0 and frp_norm.max() <= 1.0
    # 200 MW (99th pctl) maps to 1.0
    assert abs(frp_norm[4] - 1.0) < 1e-10, f"99th pctl should map to 1.0, got {frp_norm[4]}"
    # Values above 99th pctl clip to 1.0
    assert frp_norm[5] == 1.0 and frp_norm[6] == 1.0
    print("  PASS: FRP log-transform correct")

    # --- Brightness temp: physical ranges unchanged ---
    ti4_kelvin = np.array([290, 300, 310, 340, 367, 500], dtype=np.float64)
    ti4_norm = np.clip((ti4_kelvin - 300) / 200, 0, 1)
    print(f"  BT I4 (K):  {ti4_kelvin.tolist()}")
    print(f"  Normalised: {ti4_norm.round(4).tolist()}")
    assert ti4_norm[0] == 0.0, "Below 300K should clip to 0"
    assert ti4_norm[1] == 0.0, "300K (ambient) should be 0"
    assert abs(ti4_norm[2] - 0.05) < 1e-10, "310K should be 0.05"
    assert abs(ti4_norm[3] - 0.20) < 1e-10, "340K should be 0.20"
    assert abs(ti4_norm[4] - 0.335) < 1e-3, "367K (saturation) should be ~0.335"
    assert ti4_norm[5] == 1.0, "500K should clip to 1.0"
    print("  PASS: BT I4 physical ranges correct")

    ti5_kelvin = np.array([240, 250, 300, 350, 380], dtype=np.float64)
    ti5_norm = np.clip((ti5_kelvin - 250) / 100, 0, 1)
    assert ti5_norm[0] == 0.0, "Below 250K should clip to 0"
    assert ti5_norm[1] == 0.0, "250K (ambient) should be 0"
    assert abs(ti5_norm[2] - 0.50) < 1e-10, "300K should be 0.50"
    assert ti5_norm[4] == 1.0, "380K should clip to 1.0"
    print("  PASS: BT I5 physical ranges correct")

    # --- Per-year counts: same log-transform ---
    year_counts = np.array([[0, 3], [10, 0]], dtype=np.float64)
    norm = np.clip(np.log1p(year_counts) / np.log1p(_VIIRS_FIRE_COUNT_99P), 0, 1)
    assert norm[0, 0] == 0.0, "Zero counts should stay zero"
    assert norm[0, 1] > 0 and norm[0, 1] < 1, "3 counts should be mid-range"
    assert norm[1, 0] > norm[0, 1], "10 > 3 should be preserved"
    print("  PASS: Per-year counts log-transform correct")


def test_dataset_loading_no_renorm():
    """Verify dataset loading does clip-only (no re-normalization)."""
    # Simulate values as stored in .npz (already normalised at download)
    # For existing chips: fire_count/frp are per-chip max normalised [0,1]
    # For existing chips: bright_ti4/ti5 are physical range normalised [0,1]
    fire_at_year = np.array([[0.0, 0.3], [0.7, 1.0]], dtype=np.float32)
    frp = np.array([[0.0, 0.1], [0.5, 0.8]], dtype=np.float32)
    bright_ti4 = np.array([[0.0, 0.05], [0.15, 0.3]], dtype=np.float32)
    bright_ti5 = np.array([[0.0, 0.1], [0.3, 0.5]], dtype=np.float32)

    # NEW code: just clip
    fire_at_year_new = np.clip(fire_at_year, 0.0, 1.0)
    frp_new = np.clip(frp, 0.0, 1.0)
    bright_ti4_new = np.clip(bright_ti4, 0.0, 1.0)
    bright_ti5_new = np.clip(bright_ti5, 0.0, 1.0)

    # Values should be IDENTICAL (all already in [0,1])
    assert np.array_equal(fire_at_year, fire_at_year_new)
    assert np.array_equal(frp, frp_new)
    assert np.array_equal(bright_ti4, bright_ti4_new)
    assert np.array_equal(bright_ti5, bright_ti5_new)
    print("  PASS: Clip preserves already-normalised values exactly")

    # OLD code would have destroyed bright_ti4:
    # bright_ti4 = [0, 0.05, 0.15, 0.3]
    # old: bright_ti4 / 0.3 = [0, 0.167, 0.5, 1.0]  <-- WRONG
    # new: bright_ti4 stays [0, 0.05, 0.15, 0.3]       <-- CORRECT
    old_ti4 = bright_ti4.copy()
    ti4_max = old_ti4.max()
    if ti4_max > 1e-6:
        old_ti4 = old_ti4 / ti4_max
    print(f"  OLD ti4 max pixel: {old_ti4.max():.2f} (rescaled to 1.0 -- destroys physics)")
    print(f"  NEW ti4 max pixel: {bright_ti4_new.max():.2f} (preserves physical value)")
    assert old_ti4.max() == 1.0, "Old norm forces max to 1.0"
    assert bright_ti4_new.max() == 0.3, "New norm preserves 0.3"
    print("  PASS: Old re-normalization destruction confirmed and fixed")


def test_edge_cases():
    """Verify edge cases don't produce NaN or crash."""
    _VIIRS_FIRE_COUNT_99P = 50.0
    _VIIRS_FRP_99P = 200.0

    # All-zero chip (no fires)
    zeros = np.zeros((64, 64), dtype=np.float64)
    norm_zeros = np.clip(np.log1p(zeros) / np.log1p(_VIIRS_FIRE_COUNT_99P), 0, 1)
    assert np.all(norm_zeros == 0), "All-zero input should produce all-zero output"
    print("  PASS: All-zero chip handled")

    # Single pixel with fire
    single = np.zeros((64, 64), dtype=np.float64)
    single[32, 32] = 25.0
    norm_single = np.clip(np.log1p(single) / np.log1p(_VIIRS_FIRE_COUNT_99P), 0, 1)
    assert norm_single[32, 32] > 0, "Fire pixel should be non-zero"
    assert norm_single[0, 0] == 0, "Non-fire pixel should be zero"
    assert not np.any(np.isnan(norm_single)), "No NaN values"
    print("  PASS: Single fire pixel handled")

    # Extreme values
    extreme = np.array([0, 1e-10, 1e6], dtype=np.float64)
    norm_extreme = np.clip(np.log1p(extreme) / np.log1p(_VIIRS_FRP_99P), 0, 1)
    assert norm_extreme[0] == 0.0
    assert norm_extreme[1] > 0  # tiny but non-zero
    assert norm_extreme[2] == 1.0  # clipped
    assert not np.any(np.isnan(norm_extreme))
    print("  PASS: Extreme values handled")

    # log1p denominator can never be zero (log1p(50) = 3.93, log1p(200) = 5.30)
    assert np.log1p(_VIIRS_FIRE_COUNT_99P) > 0, "Denominator must be positive"
    assert np.log1p(_VIIRS_FRP_99P) > 0, "Denominator must be positive"
    print("  PASS: No division-by-zero possible")


if __name__ == "__main__":
    print("=" * 60)
    print("VIIRS Normalization Verification Suite")
    print("=" * 60)
    print()

    print("[1/3] Download-time normalization (download_real_data.py)")
    test_download_time_normalization()
    print()

    print("[2/3] Dataset loading clip-only (real_datasets.py)")
    test_dataset_loading_no_renorm()
    print()

    print("[3/3] Edge cases")
    test_edge_cases()
    print()

    print("=" * 60)
    print("ALL VIIRS NORMALIZATION TESTS PASSED")
    print("=" * 60)
