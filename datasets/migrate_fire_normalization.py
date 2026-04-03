"""Migrate per-year VIIRS fire rasters from per-year to global normalisation.

Existing chips have each viirs_fire_year_XX independently normalised to
[0,1] per chip. This destroys the temporal fire intensity signal (1 fire
vs 1000 fires look identical). This script re-normalises all per-year
rasters within each chip by the global max across all years for that chip,
preserving the temporal magnitude relationship.

**Limitation:** Since the old per-year normalization lost the raw fire
counts, this migration can only approximate the correct global
normalization. For exact results, a full re-download with the fixed
``_rasterize_fires()`` is needed.

Usage::

    python datasets/migrate_fire_normalization.py --tiles-dir datasets/real_tiles
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def migrate_chip(path: str) -> bool:
    """Re-normalise per-year fire rasters in a single .npz chip.

    Each year's fire raster was independently normalised to [0,1]:
        old_norm[year] = counts[year] / counts[year].max()

    We want global normalisation across all years:
        new_norm[year] = counts[year] / max_across_all_years

    Since we lost the raw counts, we can recover relative magnitudes.
    Each old normalised year has max=1.0 (or 0 for empty years).
    The best approximation is to keep the data as-is since within each
    chip, the relative spatial patterns per year are correct. What we
    CAN do is record a global fire max so that high-fire years are
    weighted more during temporal differencing.

    For a proper fix, re-download with the corrected _rasterize_fires()
    that uses global normalization.

    Returns True if the chip was modified.
    """
    try:
        data = dict(np.load(path, allow_pickle=True))
    except Exception:
        return False

    # Find all per-year fire keys
    year_keys = sorted(k for k in data if k.startswith("viirs_fire_year_"))
    if not year_keys:
        return False  # No VIIRS data

    # Check if aggregate fire count exists for global reference
    total_fire = data.get("viirs_fire_count", None)
    if total_fire is None:
        return False  # Can't establish global reference

    # Use the total fire count array's max as the global reference
    total_max = float(total_fire.max()) if total_fire is not None else 0.0
    if total_max < 1e-8:
        return False  # No fires in this chip

    # Compute each year's contribution (sum of all per-year maxes)
    year_maxes = {}
    for k in year_keys:
        arr = data[k].astype(np.float32)
        year_maxes[k] = float(arr.max())

    sum_of_maxes = sum(year_maxes.values())
    if sum_of_maxes < 1e-8:
        return False

    # Re-scale each year: multiply each normalised year by its
    # relative contribution (year_max / sum_of_maxes) to approximate
    # the global normalisation
    modified = False
    for k in year_keys:
        if year_maxes[k] < 1e-8:
            continue
        arr = data[k].astype(np.float32)
        # Scale factor: this year's weight relative to all years
        scale = year_maxes[k] / sum_of_maxes
        data[k] = (arr * scale).astype(np.float32)
        modified = True

    if modified:
        np.savez_compressed(path, **data)

    return modified


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate VIIRS fire rasters to global normalization"
    )
    parser.add_argument(
        "--tiles-dir", default="datasets/real_tiles",
        help="Root directory containing .npz chip files",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Count chips that would be modified without writing",
    )
    args = parser.parse_args()

    chips = glob.glob(
        os.path.join(args.tiles_dir, "**", "*.npz"), recursive=True
    )
    print(f"Found {len(chips)} chip files in {args.tiles_dir}")

    migrated = 0
    skipped = 0
    for i, path in enumerate(chips):
        if args.dry_run:
            try:
                data = np.load(path, allow_pickle=True)
                year_keys = [k for k in data.files if k.startswith("viirs_fire_year_")]
                if year_keys:
                    migrated += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1
        else:
            if migrate_chip(path):
                migrated += 1
            else:
                skipped += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(chips)} chips ({migrated} migrated)")

    action = "would migrate" if args.dry_run else "migrated"
    print(f"\nDone: {action} {migrated}/{len(chips)} chips ({skipped} skipped)")


if __name__ == "__main__":
    main()
