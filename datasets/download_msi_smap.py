import os
import json
import numpy as np
import argparse
import time
from typing import List, Tuple, Dict, Any
import warnings

warnings.filterwarnings("ignore")

try:
    from pystac_client import Client
    import planetary_computer
    import rasterio
    from rasterio.windows import from_bounds
except ImportError:
    print("Error: Required packages missing.")
    print("Run: pip install pystac-client planetary-computer rasterio")
    exit(1)


def setup_pc_client() -> Client:
    """Connect to Planetary Computer STAC API."""
    return Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


def query_terraclimate(client: Client, bbox: List[float], time_range: str) -> List[Any]:
    """Query TerraClimate for soil moisture."""
    search = client.search(
        collections=["terraclimate"],
        bbox=bbox,
        datetime=time_range,
    )
    return list(search.items())


def query_sentinel2(client: Client, bbox: List[float], time_range: str) -> List[Any]:
    """Query Sentinel-2 L2A for MSI."""
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=time_range,
        query={"eo:cloud_cover": {"lt": 20}}  # Less than 20% clouds
    )
    # Sort by cloud cover and take best
    items = list(search.items())
    items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
    return items


def fetch_stac_window(item: Any, asset_name: str, bbox: List[float], out_shape: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Fetch a specific window from a STAC item asset using rasterio."""
    href = item.assets[asset_name].href
    
    # Simple retry logic
    for attempt in range(3):
        try:
            with rasterio.open(href) as src:
                # Convert bbox to rasterio Window based on the dataset's transform
                window = from_bounds(*bbox, src.transform)
                
                # S2 bands can be higher res, so we read with out_shape to resample to 256x256
                # Resampling to out_shape handles the scaling natively in rasterio
                data = src.read(
                    1, 
                    window=window, 
                    out_shape=out_shape,
                    resampling=rasterio.enums.Resampling.bilinear
                )
                return data.astype(np.float32)
        except Exception as e:
            if attempt == 2:
                print(f"      Failed to fetch {asset_name}: {e}")
            time.sleep(1)
            
    return np.zeros(out_shape, dtype=np.float32)


def process_chip(client: Client, npz_path: str, chip_size: int = 256) -> bool:
    """Read a chip, query STAC, download MSI/SMAP, and augment the .npz."""
    try:
        data = dict(np.load(npz_path))
    except Exception as e:
        print(f"  Error loading {npz_path}: {e}")
        return False

    if "bounds" not in data:
        print(f"  Skipping {npz_path} (no geographic bounds found in chip)")
        return False
        
    bounds = data["bounds"].tolist()  # [west, south, east, north]
    west, south, east, north = bounds
    
    # Only process if we haven't already augmented this chip
    if "smap_soil_moisture" in data and "msi_ndssi_delta" in data:
        return True
        
    print(f"  Augmenting chip at bounds {bounds}...")

    # Track whether real data was successfully retrieved from both sources
    has_real_soil = False
    has_real_msi = False

    # ── 1. Soil Component (TerraClimate) ──
    # Request data for a specific year to represent the "baseline" soil state
    tc_items = query_terraclimate(client, bounds, "2018-01-01/2018-12-31")
    if tc_items:
        # Get soil moisture ('soil') from the best item
        soil_item = tc_items[0]
        soil_moisture = fetch_stac_window(soil_item, "soil", bounds, (chip_size, chip_size))
        
        # TerraClimate uses scale factor 0.1 for soil moisture (mm)
        soil_moisture = soil_moisture * 0.1
        data["smap_soil_moisture"] = soil_moisture
        has_real_soil = True
    else:
        # Fallback empty if no coverage
        data["smap_soil_moisture"] = np.zeros((chip_size, chip_size), dtype=np.float32)

    # ── 2. Hydro Component (Sentinel-2 MSI) ──
    # Request two periods: baseline (before) and impact (after)
    # Using red (B04) and near-infrared (B08) to compute NDSSI
    s2_baseline = query_sentinel2(client, bounds, "2016-01-01/2017-01-01")
    s2_impact = query_sentinel2(client, bounds, "2020-01-01/2021-01-01")
    
    def calculate_ndssi(item) -> np.ndarray:
        # NDSSI = (Green - NIR) / (Green + NIR)
        # Using B03 (Green) and B08 (NIR)
        green = fetch_stac_window(item, "B03", bounds, (chip_size, chip_size))
        nir = fetch_stac_window(item, "B08", bounds, (chip_size, chip_size))
        
        # Avoid division by zero
        denominator = green + nir + 1e-8
        ndssi = (green - nir) / denominator
        return np.clip(ndssi, -1.0, 1.0)
    
    if s2_baseline and s2_impact:
        ndssi_before = calculate_ndssi(s2_baseline[0])
        ndssi_after = calculate_ndssi(s2_impact[0])
        
        # True impact is change in suspended sediment (decrease in NDSSI often means more sediment)
        # Higher sediment -> lower NDSSI -> larger drop. We capture the absolute change or drop.
        sediment_increase = np.clip(ndssi_before - ndssi_after, 0, 1)
        data["msi_ndssi_delta"] = sediment_increase
        data["msi_ndssi_baseline"] = ndssi_before
        has_real_msi = True
    else:
        # Fallback empty
        data["msi_ndssi_delta"] = np.zeros((chip_size, chip_size), dtype=np.float32)
        data["msi_ndssi_baseline"] = np.zeros((chip_size, chip_size), dtype=np.float32)

    # Only flag as real when BOTH sources returned actual data.
    # Zero-filled fallback chips must NOT be counted in target scale
    # computation (compute_global_target_scale), otherwise they
    # artificially deflate the 95th-percentile normalisation constant.
    data["has_real_msi_smap"] = np.array(
        [1.0 if (has_real_soil and has_real_msi) else 0.0],
        dtype=np.float32,
    )
    
    # Save back to disk
    np.savez_compressed(npz_path, **data)
    return True


def augment_dataset(tiles_dir: str):
    """Iterate through the real_tiles directory and augment chips with STAC data."""
    manifest_path = os.path.join(tiles_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        return

    client = setup_pc_client()

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Combine train and val items
    all_entries = manifest.get("train", []) + manifest.get("val", []) + manifest.get("test", [])
    
    print(f"Discovered {len(all_entries)} chips in manifest. Augmenting with Sentinel-2 & SMAP (TerraClimate)...")
    
    import concurrent.futures
    success_count = 0
    
    def process_wrapper(args):
        idx, entry = args
        chip_path = os.path.join(tiles_dir, entry["file"])
        print(f"[{idx+1}/{len(all_entries)}] Processing {entry['file']}...")
        try:
            return process_chip(client, chip_path)
        except Exception:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        results = executor.map(process_wrapper, enumerate(all_entries))
        for res in results:
            if res:
                success_count += 1
            
    print(f"\nSuccessfully augmented {success_count}/{len(all_entries)} chips using 32 parallel workers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment MISDO chips with real MSI & SMAP data.")
    parser.add_argument("--tiles-dir", type=str, default="datasets/real_tiles",
                        help="Path to the real_tiles directory containing manifest.json")
    args = parser.parse_args()
    
    augment_dataset(args.tiles_dir)
