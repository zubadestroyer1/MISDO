"""
MISDO — MSI/SMAP Data Augmentation Pipeline
=============================================
Downloads real Sentinel-2 MSI and TerraClimate soil moisture data
from Microsoft Planetary Computer to augment existing Hansen GFC
chips with:
  - msi_ndssi_delta  : NDSSI change (Blue-NIR)/(Blue+NIR) [Hossain 2010]
  - smap_soil_moisture : TerraClimate monthly soil moisture (mm)
  - has_real_msi_smap  : flag indicating real data vs zero-filled fallback

Usage:
    python datasets/download_msi_smap.py --tiles-dir datasets/real_tiles
"""

import os
import json
import numpy as np
import argparse
import time
import warnings
from typing import List, Tuple, Dict, Any, Optional

# Suppress known noisy warnings from STAC/rasterio, not all warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pystac_client")
warnings.filterwarnings("ignore", category=FutureWarning, module="rasterio")
warnings.filterwarnings("ignore", category=UserWarning, module="rasterio")

try:
    from pystac_client import Client
    import planetary_computer
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds
except ImportError:
    print("Error: Required packages missing.")
    print("Run: pip install pystac-client planetary-computer rasterio")
    exit(1)

# TerraClimate requires xarray + zarr (via Planetary Computer Zarr store)
_HAS_XARRAY = True
try:
    import xarray as xr
except ImportError:
    _HAS_XARRAY = False
    print("Warning: xarray not found. TerraClimate soil moisture will use zeros.")
    print("Run: pip install xarray zarr adlfs")


# ── Sentinel-2 L2A Scaling Constants ──────────────────────────────
# S2 L2A stores reflectance as uint16 with scale factor 1/10000.
# Since Processing Baseline 04.00 (Jan 2022), ESA applies
# BOA_ADD_OFFSET = -1000 DN for all bands to avoid dark-scene clipping.
# Correct conversion: reflectance = (DN + offset) / 10000
_S2_QUANTIFICATION_VALUE = 10000.0
_S2_BOA_ADD_OFFSET = -1000.0  # Applied since PB 04.00 (all bands)


def setup_pc_client() -> Client:
    """Connect to Planetary Computer STAC API.

    Each thread should create its own client instance to avoid
    sharing mutable state (sessions, token caches).
    """
    return Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


def query_sentinel2(
    client: Client,
    bbox: List[float],
    time_range: str,
) -> List[Any]:
    """Query Sentinel-2 L2A for MSI, sorted by cloud cover (best first)."""
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=time_range,
        query={"eo:cloud_cover": {"lt": 20}},  # Less than 20% clouds
    )
    items = list(search.items())
    items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
    return items


def fetch_s2_band(
    item: Any,
    band_name: str,
    bbox: List[float],
    out_shape: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Fetch a Sentinel-2 L2A band and convert to surface reflectance [0, 1].

    Handles:
    - Raw uint16 DN → float32 reflectance via ÷10,000
    - BOA_ADD_OFFSET correction for Processing Baseline ≥04.00
    - Clipping to valid reflectance range [0, 1]

    Parameters
    ----------
    item : pystac.Item
        Signed STAC item from Planetary Computer.
    band_name : str
        Asset key (e.g. "B02", "B03", "B08").
    bbox : list of float
        [west, south, east, north] bounding box.
    out_shape : tuple of int
        Output spatial dimensions.

    Returns
    -------
    np.ndarray
        Surface reflectance array in [0, 1], shape ``out_shape``.
    """
    try:
        href = item.assets[band_name].href
    except KeyError:
        print(f"      Band {band_name} not found in STAC item assets")
        return np.zeros(out_shape, dtype=np.float32)

    for attempt in range(3):
        try:
            with rasterio.open(href) as src:
                # Reproject bbox from EPSG:4326 (lat/lon) to the raster's
                # native CRS (typically UTM).  Without this, from_bounds()
                # produces a near-zero window at a huge negative offset and
                # the read fails silently with "Read failed".
                native_bbox = transform_bounds(
                    "EPSG:4326", src.crs, *bbox
                )
                window = from_bounds(*native_bbox, src.transform)
                data = src.read(
                    1,
                    window=window,
                    out_shape=out_shape,
                    resampling=rasterio.enums.Resampling.bilinear,
                )
                data = data.astype(np.float32)

                # Apply Sentinel-2 L2A scaling to get surface reflectance.
                # PB ≥04.00 applies BOA_ADD_OFFSET = -1000 before quantification.
                # Detect PB version from the authoritative STAC metadata property
                # (s2:processing_baseline) rather than heuristic median check.
                pb_str = item.properties.get("s2:processing_baseline", "00.00")
                try:
                    pb_version = float(pb_str)
                except (ValueError, TypeError):
                    pb_version = 0.0
                if pb_version >= 4.0:
                    # PB ≥04.00: apply BOA_ADD_OFFSET then scale
                    data = (data + _S2_BOA_ADD_OFFSET) / _S2_QUANTIFICATION_VALUE
                else:
                    # Pre-PB04: no offset, just scale
                    data = data / _S2_QUANTIFICATION_VALUE

                return np.clip(data, 0.0, 1.0)
        except Exception as e:
            if attempt == 2:
                print(f"      Failed to fetch {band_name}: {e}")
            time.sleep(1 + attempt)  # Incremental backoff

    return np.zeros(out_shape, dtype=np.float32)


def fetch_terraclimate_soil(
    bbox: List[float],
    time_range: str,
    out_shape: Tuple[int, int] = (256, 256),
) -> Optional[np.ndarray]:
    """Fetch TerraClimate soil moisture from Planetary Computer Zarr store.

    TerraClimate is stored as a Zarr dataset on Azure, NOT as individual
    GeoTIFFs. Access requires xarray + zarr.

    Parameters
    ----------
    bbox : list of float
        [west, south, east, north].
    time_range : str
        ISO date range, e.g. "2018-01-01/2018-12-31".
    out_shape : tuple of int
        Output spatial dimensions (for resampling).

    Returns
    -------
    np.ndarray or None
        Soil moisture in mm (float32), resampled to out_shape.
        Returns None if xarray is unavailable or data fetch fails.
    """
    if not _HAS_XARRAY:
        return None

    try:
        # Access TerraClimate Zarr store via the collection-level asset.
        client = setup_pc_client()
        collection = client.get_collection("terraclimate")
        asset = collection.assets["zarr-https"]

        # The Planetary Computer returns a SAS-signed Azure Blob URL:
        #   https://<account>.blob.core.windows.net/<container>/<blob>?<sas>
        # We must parse this into components for adlfs because:
        #   - fsspec HTTPFileSystem appends chunk paths AFTER the SAS query
        #     string, producing URLs like ...?sas_token=xxx/.zattrs → 403
        #   - xr.open_zarr(raw_url) also fails for the same reason
        #   - adlfs.AzureBlobFileSystem with explicit SAS token handles
        #     per-chunk authentication correctly
        from urllib.parse import urlparse
        import adlfs

        parsed = urlparse(asset.href)
        account_name = parsed.hostname.split(".")[0]
        path_parts = parsed.path.lstrip("/").split("/", 1)
        container = path_parts[0]
        blob_path = path_parts[1] if len(path_parts) > 1 else ""
        sas_token = parsed.query

        store = adlfs.AzureBlobFileSystem(
            account_name=account_name,
            sas_token=sas_token,
        ).get_mapper(f"{container}/{blob_path}")

        ds = xr.open_zarr(store, consolidated=True)

        west, south, east, north = bbox
        start_date, end_date = time_range.split("/")

        # TerraClimate lat goes from +90 to -90 (descending)
        soil = (
            ds["soil"]
            .sel(
                time=slice(start_date, end_date),
                lat=slice(north, south),  # descending lat
                lon=slice(west, east),
            )
            .mean(dim="time", skipna=True)
            .values
        )

        if soil.size == 0:
            return None

        # Resample from native ~4km to chip size
        from PIL import Image
        resample_method = getattr(Image, 'Resampling', Image).BILINEAR
        soil_img = Image.fromarray(soil.astype(np.float32))
        soil_resized = np.array(
            soil_img.resize(out_shape[::-1], resample_method),
            dtype=np.float32,
        )

        # TerraClimate Zarr on Planetary Computer is pre-scaled (analysis-ready).
        # The scale_factor (0.1) is already applied during Zarr conversion.
        # Values are in physical units (mm). No additional scaling needed.
        # NOTE: Downstream code normalises per-chip (smap / smap.max()),
        # so this change does not affect model training behaviour.

        return np.clip(soil_resized, 0.0, None)  # Moisture can't be negative

    except Exception as e:
        print(f"      TerraClimate fetch failed: {e}")
        return None


def calculate_ndssi(
    item: Any,
    bbox: List[float],
    chip_size: int = 256,
) -> np.ndarray:
    """Calculate NDSSI from Sentinel-2 L2A bands.

    NDSSI = (Blue - NIR) / (Blue + NIR)  [Hossain et al., 2010]

    Uses B02 (Blue, 490nm) and B08 (NIR, 842nm). NOT B03 (Green) —
    Green/NIR computes NDWI (water content), not NDSSI (sediment).

    Higher NDSSI = clearer water; lower NDSSI = more sediment.

    Returns values in [-1, 1].
    """
    blue = fetch_s2_band(item, "B02", bbox, (chip_size, chip_size))
    nir = fetch_s2_band(item, "B08", bbox, (chip_size, chip_size))

    denominator = blue + nir + 1e-8  # Avoid division by zero
    ndssi = (blue - nir) / denominator
    return np.clip(ndssi, -1.0, 1.0)


def process_chip(client: Client, npz_path: str, chip_size: int = 256) -> bool:
    """Read a chip, query STAC, download MSI/SMAP, and augment the .npz.

    Parameters
    ----------
    client : pystac_client.Client
        Thread-local STAC client (do NOT share across threads).
    npz_path : str
        Path to the .npz chip file to augment.
    chip_size : int
        Spatial dimension of the chip (default 256).

    Returns
    -------
    bool
        True if the chip was successfully augmented or already had real data.
    """
    try:
        data = dict(np.load(npz_path))
    except Exception as e:
        print(f"  Error loading {npz_path}: {e}")
        return False

    if "bounds" not in data:
        print(f"  Skipping {npz_path} (no geographic bounds found in chip)")
        return False

    bounds = data["bounds"].tolist()  # [west, south, east, north]

    # ── Skip if already has real (non-fallback) data ──────────────
    # Check the flag, not just key existence — a previous failed run
    # may have written zero-filled arrays with has_real_msi_smap=0.
    if "has_real_msi_smap" in data:
        flag = float(data["has_real_msi_smap"].flat[0])
        if flag > 0.5:
            return True  # Already has real data, skip

    print(f"  Augmenting chip at bounds {bounds}...")

    has_real_soil = False
    has_real_msi = False

    # ── 1. Soil Component (TerraClimate via Zarr) ─────────────────
    soil_moisture = fetch_terraclimate_soil(
        bounds, "2018-01-01/2018-12-31", (chip_size, chip_size),
    )
    if soil_moisture is not None and soil_moisture.max() > 0:
        data["smap_soil_moisture"] = soil_moisture
        has_real_soil = True
    else:
        data["smap_soil_moisture"] = np.zeros(
            (chip_size, chip_size), dtype=np.float32,
        )

    # ── 2. Hydro Component (Sentinel-2 NDSSI Delta) ──────────────
    # Baseline: 2016, Impact: 2020 (aligned with _HYDRO_T1/_HYDRO_T2)
    s2_baseline = query_sentinel2(client, bounds, "2016-01-01/2017-01-01")
    s2_impact = query_sentinel2(client, bounds, "2020-01-01/2021-01-01")

    if s2_baseline and s2_impact:
        ndssi_before = calculate_ndssi(s2_baseline[0], bounds, chip_size)
        ndssi_after = calculate_ndssi(s2_impact[0], bounds, chip_size)

        # Sediment increase = NDSSI drop (lower NDSSI = more sediment)
        # Positive values = more sediment after clearing
        sediment_increase = np.clip(ndssi_before - ndssi_after, 0, 1)
        data["msi_ndssi_delta"] = sediment_increase
        data["msi_ndssi_baseline"] = ndssi_before
        has_real_msi = True
    else:
        data["msi_ndssi_delta"] = np.zeros(
            (chip_size, chip_size), dtype=np.float32,
        )
        data["msi_ndssi_baseline"] = np.zeros(
            (chip_size, chip_size), dtype=np.float32,
        )

    # Flag: only mark as real when BOTH sources returned actual data.
    data["has_real_msi_smap"] = np.array(
        [1.0 if (has_real_soil and has_real_msi) else 0.0],
        dtype=np.float32,
    )

    # Save back to disk
    np.savez_compressed(npz_path, **data)
    return True


def augment_dataset(tiles_dir: str) -> None:
    """Iterate through the real_tiles directory and augment chips with STAC data.

    Creates one STAC client per worker thread for thread safety.
    """
    manifest_path = os.path.join(tiles_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        return

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Combine train and test entries (manifest has no "val" key)
    all_entries = manifest.get("train", []) + manifest.get("test", [])

    print(f"Discovered {len(all_entries)} chips in manifest. "
          f"Augmenting with Sentinel-2 & TerraClimate...")

    import concurrent.futures
    import threading
    success_count = 0

    # Use 4 workers (reduced from 8) to avoid triggering Planetary Computer
    # rate limits (HTTP 429). Each worker creates its own STAC client
    # for thread safety — pystac_client.Client is NOT thread-safe.
    n_workers = 4

    # Thread-local storage: one STAC client per thread (not per chip)
    _thread_local = threading.local()

    def _get_thread_client():
        """Get or create a thread-local STAC client with retry."""
        if not hasattr(_thread_local, "client"):
            for attempt in range(5):
                try:
                    _thread_local.client = setup_pc_client()
                    return _thread_local.client
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"  STAC client init failed (attempt {attempt+1}/5): {e}")
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)
            # Final attempt — let it raise
            _thread_local.client = setup_pc_client()
        return _thread_local.client

    def process_wrapper(args):
        idx, entry = args
        chip_path = os.path.join(tiles_dir, entry["file"])
        print(f"[{idx + 1}/{len(all_entries)}] Processing {entry['file']}...")
        try:
            thread_client = _get_thread_client()
            return process_chip(thread_client, chip_path)
        except Exception as e:
            print(f"  Error processing {entry['file']}: {e}")
            # Reset the client so the next chip gets a fresh one
            _thread_local.client = None
            if hasattr(_thread_local, "client"):
                delattr(_thread_local, "client")
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(process_wrapper, enumerate(all_entries))
        for res in results:
            if res:
                success_count += 1

    print(f"\nSuccessfully augmented {success_count}/{len(all_entries)} chips "
          f"using {n_workers} parallel workers.")

    # ── Propagate has_real_msi_smap flag into manifest ────────────
    print("\nUpdating manifest with has_real_msi_smap flags...")
    updated = 0
    for split_name in ("train", "test"):
        entries = manifest.get(split_name, [])
        for entry in entries:
            chip_path = os.path.join(tiles_dir, entry["file"])
            if not os.path.exists(chip_path):
                continue
            try:
                chip_data = np.load(chip_path)
                flag_arr = chip_data.get("has_real_msi_smap", None)
                if flag_arr is not None:
                    entry["has_real_msi_smap"] = bool(
                        float(flag_arr.flat[0]) > 0.5
                    )
                    updated += 1
                else:
                    entry["has_real_msi_smap"] = False
            except Exception:
                entry["has_real_msi_smap"] = False

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Updated {updated} manifest entries with has_real_msi_smap flag.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment MISDO chips with real MSI & soil moisture data.",
    )
    parser.add_argument(
        "--tiles-dir",
        type=str,
        default="datasets/real_tiles",
        help="Path to the real_tiles directory containing manifest.json",
    )
    args = parser.parse_args()

    augment_dataset(args.tiles_dir)
