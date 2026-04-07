"""
MISDO — Global-Scale Real Data Downloader
===================================================
Downloads Hansen GFC v1.11 + SRTM elevation + VIIRS fire data from
ALL available forested tiles worldwide for maximum-accuracy training.

Data sources:
    - Hansen GFC v1.11 (Google Cloud Storage, public, no auth)
    - SRTM GL1 30m elevation (OpenTopography AWS, public, no auth)
    - VIIRS VNP14IMG fire detections (NASA FIRMS API, free MAP_KEY)

Scale:
    Hansen GFC covers 80°N–60°S × 180°W–180°E = 504 possible tiles.
    Of these, ~280–320 have meaningful forest (>5% treecover).
    Default: 1000 chips/tile × all forested tiles → 280k–320k chips.

Modes:
    global  — discover + download ALL forested tiles (recommended)
    curated — use 30 high-priority hand-picked tiles (faster)

Usage:
    # Full global download (recommended for A100 training)
    python datasets/download_real_data.py --mode global --parallel 16

    # Curated 30 tiles only (faster, still diverse)
    python datasets/download_real_data.py --mode curated --parallel 8

    # Specific latitude band (e.g. tropics only)
    python datasets/download_real_data.py --mode global --lat-range -10 10

    # Discovery: scan which tiles have forest (no download)
    python datasets/download_real_data.py --discover-tiles --parallel 16
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import struct
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ═══════════════════════════════════════════════════════════════════════════
# Global Tile Enumeration
# ═══════════════════════════════════════════════════════════════════════════

def _make_tile_code(lat: int, lon: int) -> str:
    """Convert integer lat/lon to Hansen tile code (e.g. 10, -70 → '10N_070W')."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):02d}{ns}_{abs(lon):03d}{ew}"


def _enumerate_all_tile_codes() -> List[str]:
    """Enumerate ALL 504 possible Hansen GFC tile positions.

    Hansen GFC v1.11 covers 80°N to 60°S (14 NW-corner latitudes)
    × 180°W to 170°E (36 columns) = 504 tiles.
    """
    codes = []
    for lat in range(80, -60, -10):   # 80, 70, 60, ..., -50
        for lon in range(-180, 180, 10):  # -180, -170, ..., 170
            codes.append(_make_tile_code(lat, lon))
    return codes


# ═══════════════════════════════════════════════════════════════════════════
# Curated Tiles — 30 High-Priority Named Tiles (backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════

CURATED_TILES: Dict[str, Dict] = {
    # ── Tropical Rainforest (8) ──
    "amazon_rondonia":    {"tile": "10S_070W", "biome": "tropical_rainforest", "region": "Rondônia, Brazil",        "description": "Arc of deforestation"},
    "amazon_para":        {"tile": "00N_050W", "biome": "tropical_rainforest", "region": "Pará, Brazil",            "description": "Eastern Amazon — cattle ranching frontier"},
    "amazon_mato_grosso": {"tile": "10S_060W", "biome": "tropical_rainforest", "region": "Mato Grosso, Brazil",     "description": "Transition zone — soy/cattle expansion"},
    "amazon_acre":        {"tile": "10S_080W", "biome": "tropical_rainforest", "region": "Acre, Brazil",            "description": "Western Amazon — logging and road expansion"},
    "congo_basin_west":   {"tile": "00N_020E", "biome": "tropical_rainforest", "region": "Congo Basin West, DRC",   "description": "Smallholder deforestation"},
    "congo_basin_east":   {"tile": "00N_030E", "biome": "tropical_rainforest", "region": "Congo Basin East, DRC",   "description": "Mining and subsistence agriculture"},
    "borneo_kalimantan":  {"tile": "00N_110E", "biome": "tropical_rainforest", "region": "Kalimantan, Indonesia",   "description": "Oil palm expansion"},
    "sumatra_riau":       {"tile": "00N_100E", "biome": "tropical_rainforest", "region": "Riau, Sumatra",           "description": "Peatland deforestation"},
    # ── Tropical Dry Forest (5) ──
    "guatemala_peten":     {"tile": "10N_090W", "biome": "tropical_dry", "region": "Petén, Guatemala",       "description": "Maya Biosphere — fire-driven"},
    "mexico_yucatan":      {"tile": "20N_090W", "biome": "tropical_dry", "region": "Yucatán, Mexico",        "description": "Seasonal dry forest"},
    "tanzania_eastern_arc":{"tile": "10S_030E", "biome": "tropical_dry", "region": "Eastern Arc, Tanzania",  "description": "Agricultural encroachment"},
    "mozambique_miombo":   {"tile": "20S_030E", "biome": "tropical_dry", "region": "Northern Mozambique",    "description": "Charcoal and clearing"},
    "cambodia_tonle_sap":  {"tile": "10N_100E", "biome": "tropical_dry", "region": "Tonle Sap, Cambodia",   "description": "Rubber and cashew conversion"},
    # ── Temperate Forest (5) ──
    "pacific_northwest":    {"tile": "50N_130W", "biome": "temperate", "region": "British Columbia, Canada", "description": "Old-growth + pine beetle"},
    "us_southeast":         {"tile": "30N_090W", "biome": "temperate", "region": "Southeast US",             "description": "Pine plantation forestry"},
    "romania_carpathians":  {"tile": "40N_020E", "biome": "temperate", "region": "Carpathians, Romania",     "description": "Old-growth — illegal logging"},
    "japan_hokkaido":       {"tile": "40N_140E", "biome": "temperate", "region": "Hokkaido, Japan",          "description": "Temperate mixed forest"},
    "chile_valdivian":      {"tile": "40S_070W", "biome": "temperate", "region": "Valdivian, Chile",         "description": "Southern temperate rainforest"},
    # ── Boreal Forest (4) ──
    "canada_ontario":  {"tile": "50N_090W", "biome": "boreal", "region": "Ontario, Canada",     "description": "Boreal shield — fire + forestry"},
    "canada_alberta":  {"tile": "50N_120W", "biome": "boreal", "region": "Alberta, Canada",     "description": "Boreal plains — oil sands"},
    "russia_siberia":  {"tile": "60N_090E", "biome": "boreal", "region": "Central Siberia",     "description": "Fire + permafrost thaw"},
    "sweden_norrland": {"tile": "60N_010E", "biome": "boreal", "region": "Norrland, Sweden",    "description": "Managed boreal forestry"},
    # ── Mangrove / Coastal (4) ──
    "myanmar_delta":        {"tile": "10N_090E", "biome": "mangrove", "region": "Irrawaddy Delta, Myanmar",   "description": "Mangrove to rice/aquaculture"},
    "nigeria_niger_delta":  {"tile": "00N_000E", "biome": "mangrove", "region": "Niger Delta, Nigeria",       "description": "Oil extraction + urbanisation"},
    "bangladesh_sundarbans":{"tile": "20N_080E", "biome": "mangrove", "region": "Sundarbans, Bangladesh",     "description": "Largest mangrove — sea level rise"},
    "brazil_maranhao_coast":{"tile": "00N_040W", "biome": "mangrove", "region": "Maranhão Coast, Brazil",     "description": "Atlantic mangroves"},
    # ── Savanna / Cerrado (4) ──
    "brazil_cerrado":      {"tile": "10S_050W", "biome": "savanna", "region": "Cerrado, Brazil",         "description": "Soy expansion frontier"},
    "brazil_cerrado_west": {"tile": "20S_060W", "biome": "savanna", "region": "Western Cerrado, Brazil", "description": "Cerrado-Amazon transition"},
    "zambia_miombo":       {"tile": "10S_020E", "biome": "savanna", "region": "Miombo, Zambia",          "description": "Charcoal + agriculture"},
    "australia_queensland": {"tile": "20S_140E", "biome": "savanna", "region": "Queensland, Australia",  "description": "Eucalypt woodland clearing"},
}

# Backward-compat alias
TILE_REGISTRY = CURATED_TILES

BIOME_GROUPS: Dict[str, List[str]] = {}
for _tn, _ti in CURATED_TILES.items():
    BIOME_GROUPS.setdefault(_ti["biome"], []).append(_tn)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

GFC_BASE_URL = (
    "https://storage.googleapis.com/earthenginepartners-hansen/"
    "GFC-2023-v1.11/"
)
GFC_LAYERS = ["treecover2000", "lossyear", "gain"]

# FIRMS API — per-chip fire queries (rate-limited, max 1 day per request)
# See: https://firms.modaps.eosdis.nasa.gov/api/area/
FIRMS_API_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    "{key}/VIIRS_SNPP_SP/{west},{south},{east},{north}/1/{date}"
)


class VIIRSArchive:
    """Fast spatial lookup for VIIRS fire detections from bulk CSV files.

    Loads FIRMS bulk archive CSVs into memory and builds a 1°×1° grid
    index for O(1) bounding-box queries.  Each query returns a list of
    fire dicts identical in format to ``_download_viirs_fires()``, so
    the existing ``_rasterize_fires()`` works unchanged.

    Parameters
    ----------
    csv_paths : list of str
        Paths to FIRMS VIIRS SNPP CSV files (any mix of yearly/monthly).
    """

    # Columns we need (FIRMS CSV uses these header names)
    _KEEP_COLS = [
        "latitude", "longitude", "bright_ti4", "bright_ti5",
        "frp", "confidence", "acq_date",
    ]

    def __init__(self, csv_paths: List[str]) -> None:
        import csv as csv_mod

        all_rows: List[Dict[str, str]] = []
        for path in csv_paths:
            with open(path, newline="") as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    # Keep only the columns we need
                    filtered = {
                        k: row.get(k, "") for k in self._KEEP_COLS
                        if k in row
                    }
                    if "latitude" in filtered and "longitude" in filtered:
                        all_rows.append(filtered)

        print(f"  VIIRSArchive: loaded {len(all_rows):,} fire detections "
              f"from {len(csv_paths)} file(s)")

        # Build 1°×1° spatial grid index for fast bbox queries
        self._grid: Dict[Tuple[int, int], List[Dict[str, str]]] = {}
        for row in all_rows:
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except (ValueError, KeyError):
                continue
            key = (int(math.floor(lat)), int(math.floor(lon)))
            self._grid.setdefault(key, []).append(row)

        n_cells = len(self._grid)
        print(f"  VIIRSArchive: indexed into {n_cells} grid cells")

    def query(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
    ) -> Optional[List[Dict[str, str]]]:
        """Return fire detections within a bounding box.

        Returns list of dicts (same format as ``_download_viirs_fires``),
        or None if no fires found.
        """
        # Determine which 1° grid cells overlap the bbox
        lat_min = int(math.floor(south))
        lat_max = int(math.floor(north))
        lon_min = int(math.floor(west))
        lon_max = int(math.floor(east))

        matches: List[Dict[str, str]] = []
        for lat_cell in range(lat_min, lat_max + 1):
            for lon_cell in range(lon_min, lon_max + 1):
                cell_rows = self._grid.get((lat_cell, lon_cell))
                if cell_rows is None:
                    continue
                for row in cell_rows:
                    try:
                        rlat = float(row["latitude"])
                        rlon = float(row["longitude"])
                    except (ValueError, KeyError):
                        continue
                    if south <= rlat <= north and west <= rlon <= east:
                        matches.append(row)

        return matches if matches else None


def download_viirs_archive(
    output_dir: str,
    years: Optional[List[int]] = None,
) -> List[str]:
    """Scan for user-provided FIRMS VIIRS bulk archive CSVs.

    NASA FIRMS does not expose yearly bulk CSV files via direct URLs.
    Users must download CSVs manually from the FIRMS Archive Download
    tool (https://firms.modaps.eosdis.nasa.gov/download/) and place
    them in ``{output_dir}/.viirs_cache/``.

    This function scans that cache directory for any ``.csv`` files
    and returns their paths.

    Parameters
    ----------
    output_dir : str
        Base output directory (e.g. ``datasets/real_tiles``).
    years : list of int, optional
        Ignored (kept for backwards compatibility).

    Returns
    -------
    list of str
        Paths to CSV files found in the cache directory.
    """
    import glob as glob_mod

    cache_dir = os.path.join(output_dir, ".viirs_cache")
    os.makedirs(cache_dir, exist_ok=True)

    csv_paths = sorted(glob_mod.glob(os.path.join(cache_dir, "*.csv")))

    if csv_paths:
        total_mb = sum(os.path.getsize(p) for p in csv_paths) / 1e6
        print(f"  VIIRS archive: found {len(csv_paths)} CSV file(s) "
              f"({total_mb:.1f} MB) in {cache_dir}")
    else:
        print(f"  VIIRS archive: no CSV files found in {cache_dir}")
        print(f"  To add VIIRS fire data:")
        print(f"    1. Go to https://firms.modaps.eosdis.nasa.gov/download/")
        print(f"    2. Select VIIRS S-NPP, your area of interest, and date range")
        print(f"    3. Download the CSV and place it in {cache_dir}")
        print(f"    4. Or use --viirs-archive /path/to/your/csvs/")

    return csv_paths


# ═══════════════════════════════════════════════════════════════════════════
# Geo-referencing — convert Hansen pixel coords to lat/lon
# ═══════════════════════════════════════════════════════════════════════════

def _parse_tile_code(tile_code: str) -> Tuple[float, float]:
    """Parse Hansen tile code (e.g. '10S_070W') to NW corner lat/lon.

    Hansen GFC tiles are 10° × 10°. The tile code specifies the
    latitude and longitude of the upper-left (northwest) corner.
    """
    parts = tile_code.split("_")
    lat_str, lon_str = parts[0], parts[1]

    lat = int(lat_str[:-1])
    if lat_str[-1] == "S":
        lat = -lat

    lon = int(lon_str[:-1])
    if lon_str[-1] == "W":
        lon = -lon

    return float(lat), float(lon)


def _chip_bounds(
    tile_code: str,
    px: int,
    py: int,
    chip_size: int = 256,
    tile_pixels: int = 40000,
) -> Tuple[float, float, float, float]:
    """Get geographic bounds (west, south, east, north) of a chip.

    Hansen GFC tiles are 10° × 10° at ~40,000 pixels → ~0.00025° per pixel.
    """
    nw_lat, nw_lon = _parse_tile_code(tile_code)
    deg_per_pixel = 10.0 / tile_pixels

    west = nw_lon + px * deg_per_pixel
    north = nw_lat - py * deg_per_pixel
    east = west + chip_size * deg_per_pixel
    south = north - chip_size * deg_per_pixel

    return west, south, east, north


# ═══════════════════════════════════════════════════════════════════════════
# SRTM Elevation Download (from AWS Terrain Tiles — free, no auth)
# ═══════════════════════════════════════════════════════════════════════════

def _srtm_tile_name(lat: float, lon: float) -> str:
    """Get SRTM tile filename for a given lat/lon.

    SRTM tiles are named by the SW corner integer coordinate.
    """
    lat_int = int(math.floor(lat))
    lon_int = int(math.floor(lon))

    ns = "N" if lat_int >= 0 else "S"
    ew = "E" if lon_int >= 0 else "W"

    return f"{ns}{abs(lat_int):02d}{ew}{abs(lon_int):03d}"


def _download_srtm_hgt(lat: float, lon: float, cache_dir: str) -> Optional[np.ndarray]:
    """Download a single SRTM .hgt tile and return as 3601×3601 array.

    Caches tiles to avoid re-downloading (each tile is ~2.8 MB gzipped).
    """
    name = _srtm_tile_name(lat, lon)
    ns_dir = name[:3]  # e.g. "N10" or "S05"
    ew_str = name[3:]  # e.g. "W070"
    full_name = name

    cache_file = os.path.join(cache_dir, f"{full_name}.hgt")
    if os.path.exists(cache_file):
        data = np.fromfile(cache_file, dtype=">i2")
        if data.size == 3601 * 3601:
            return data.reshape(3601, 3601).astype(np.float32)

    # AWS Skadi SRTM format: {lat_dir}/{lat_dir}{lon_dir}.hgt.gz
    # e.g. https://s3.amazonaws.com/elevation-tiles-prod/skadi/N10/N10W070.hgt.gz
    url = (
        f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/"
        f"{ns_dir}/{full_name}.hgt.gz"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MISDO/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            compressed = resp.read()

        raw_bytes = gzip.decompress(compressed)
        data = np.frombuffer(raw_bytes, dtype=">i2")

        if data.size != 3601 * 3601:
            return None

        arr = data.reshape(3601, 3601).astype(np.float32)

        # Cache for reuse
        os.makedirs(cache_dir, exist_ok=True)
        arr.astype(">i2").tofile(cache_file)

        return arr
    except Exception:
        return None




def _derive_real_terrain(elevation: np.ndarray) -> Dict[str, np.ndarray]:
    """Derive slope, aspect, flow accumulation from real SRTM elevation.

    Uses finite differences for gradient computation — same as
    standard GIS terrain analysis.
    """
    # Replace void values (-32768) with local valid minimum BEFORE normalisation.
    # If void pixels are included in min/max, the normalised elevation channel
    # is corrupted (e.g., mapped to ~0 for all real values).
    void_mask = elevation < -1000
    if void_mask.any():
        valid_min = elevation[~void_mask].min() if (~void_mask).any() else 0
        elevation = elevation.copy()
        elevation[void_mask] = valid_min

    # Normalise elevation to [0, 1] for model input (void-free)
    e_min, e_max = elevation.min(), elevation.max()
    if e_max - e_min > 1e-6:
        elev_norm = (elevation - e_min) / (e_max - e_min)
    else:
        elev_norm = np.zeros_like(elevation)

    # Gradient computation (central differences) with physical pixel spacing.
    # SRTM native resolution is ~30 m/pixel. Chips are resampled to 256×256
    # from a small region of the 3601×3601 tile; using the native 30 m spacing
    # gives physically meaningful dz/dx ratios for slope computation.
    pixel_spacing_m = 30.0
    dy, dx = np.gradient(elevation, pixel_spacing_m)

    # Slope (degrees, normalised to [0, 1] where 1 = 45°+)
    slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    slope_norm = np.clip(slope_deg / 45.0, 0, 1).astype(np.float32)

    # Aspect: compass bearing of steepest descent direction.
    # Convention: 0=N, 0.25=E, 0.5=S, 0.75=W (normalised to [0,1]).
    #
    # np.gradient returns (dy, dx) where:
    #   dy = dz/drow — row increases southward (NW-origin raster)
    #   dx = dz/dcol — column increases eastward
    #
    # Descent vector components:
    #   North component of descent = +dy  (positive dy means elevation
    #     rises going south, so descent is northward)
    #   East component of descent = -dx   (positive dx means elevation
    #     rises going east, so descent is westward → east component is -dx)
    #
    # Compass bearing (CW from North) = atan2(east, north)
    aspect_rad = np.arctan2(-dx, dy)
    aspect_norm = (np.mod(aspect_rad, 2 * np.pi) / (2 * np.pi)).astype(np.float32)

    # Flow accumulation (simple D8-based approximation)
    # Accumulate flow by tracing downhill from each cell
    flow_acc = _compute_flow_accumulation(elevation)
    flow_acc = np.log1p(flow_acc)
    fa_max = flow_acc.max()
    if fa_max > 1e-6:
        flow_acc = flow_acc / fa_max
    flow_acc = flow_acc.astype(np.float32)

    # Flow direction (D8 encoded, normalised)
    flow_dir = _compute_flow_direction(dx, dy)

    return {
        "srtm_elevation": elev_norm.astype(np.float32),
        "srtm_slope": slope_norm,
        "srtm_aspect": aspect_norm,
        "srtm_flow_acc": flow_acc,
        "srtm_flow_dir": flow_dir.astype(np.float32),
        "has_real_srtm": np.array([1.0], dtype=np.float32),
    }


def _compute_flow_direction(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Compute D8 flow direction from gradient components.

    Returns direction normalised to [0, 1] where values map to
    8 cardinal/intercardinal directions.
    """
    angle = np.arctan2(-dy, -dx)  # direction of steepest descent
    # Normalise to [0, 1]
    flow_dir = (angle + np.pi) / (2 * np.pi)
    return np.clip(flow_dir, 0, 1)


def _fill_single_sinks(elevation: np.ndarray) -> np.ndarray:
    """Fill single-pixel sinks by raising them to the lowest neighbour.

    Only fills sinks where ALL 8 neighbours are strictly higher
    (true single-cell depressions, typically DEM noise).
    Does NOT fill multi-cell depressions (which would require
    iterative priority-flood filling beyond chip-level scope).

    Parameters
    ----------
    elevation : ndarray (H, W)
        Raw elevation in metres.

    Returns
    -------
    ndarray (H, W)
        Elevation with single-pixel sinks filled.
    """
    h, w = elevation.shape
    pad = np.pad(elevation, 1, mode="edge")
    dr = np.array([-1, -1, -1,  0, 0,  1, 1, 1])
    dc = np.array([-1,  0,  1, -1, 1, -1, 0, 1])
    neighbours = np.stack([
        pad[1 + dri : 1 + dri + h, 1 + dci : 1 + dci + w]
        for dri, dci in zip(dr, dc)
    ], axis=0)
    min_neighbour = neighbours.min(axis=0)
    is_sink = elevation < min_neighbour
    filled = elevation.copy()
    filled[is_sink] = min_neighbour[is_sink]
    return filled


def _compute_flow_accumulation(elevation: np.ndarray) -> np.ndarray:
    """Vectorised D8 flow accumulation using NumPy.

    Pre-processes the DEM by filling single-pixel sinks (Jenson &
    Domingue 1988), then computes the steepest-descent neighbour for
    every cell in parallel and does a topological (highest-first)
    scan to accumulate flow.  ~100× faster than per-pixel Python.
    """
    # Fill single-pixel sinks to prevent flow trapping at DEM noise
    elevation = _fill_single_sinks(elevation)

    h, w = elevation.shape
    # Pad elevation to handle borders without conditionals
    pad = np.pad(elevation, 1, mode="edge")

    # D8 offsets (row_offset, col_offset) — 8 neighbours
    dr = np.array([-1, -1, -1,  0, 0,  1, 1, 1])
    dc = np.array([-1,  0,  1, -1, 1, -1, 0, 1])

    # Stack all 8 neighbour elevations: shape (8, h, w)
    neighbours = np.stack([
        pad[1 + dri : 1 + dri + h, 1 + dci : 1 + dci + w]
        for dri, dci in zip(dr, dc)
    ], axis=0)

    # Drop = centre - neighbour (positive = downhill)
    centre = elevation[np.newaxis, :, :]  # (1, h, w)
    drops = centre - neighbours           # (8, h, w)

    # Mask uphill / flat neighbours
    drops_masked = np.where(drops > 0, drops, -1.0)
    best_dir = np.argmax(drops_masked, axis=0)  # (h, w), index into 8 dirs
    has_downhill = drops_masked.max(axis=0) > 0  # (h, w), bool

    # Convert best_dir to target (row, col)
    target_r = np.arange(h)[:, None] + dr[best_dir]  # (h, w)
    target_c = np.arange(w)[None, :] + dc[best_dir]  # (h, w)

    # Topological scan: process cells highest-first
    flat_order = np.argsort(-elevation.ravel())
    flow_acc = np.ones((h, w), dtype=np.float64)

    for idx in flat_order:
        r, c = divmod(idx, w)
        if has_downhill[r, c]:
            tr, tc = int(target_r[r, c]), int(target_c[r, c])
            # Bounds check: prevent NumPy negative-index wraparound
            # at raster borders (border cells may point outside grid)
            if 0 <= tr < h and 0 <= tc < w:
                flow_acc[tr, tc] += flow_acc[r, c]

    return flow_acc


def download_srtm_for_chip(
    tile_code: str,
    px: int,
    py: int,
    chip_size: int,
    cache_dir: str,
) -> Optional[Dict[str, np.ndarray]]:
    """Download real SRTM elevation for a chip and derive terrain.

    Returns dict of terrain arrays, or None if SRTM unavailable.
    Handles chips that cross 1° SRTM tile boundaries by stitching
    up to 4 neighbouring tiles.
    """
    west, south, east, north = _chip_bounds(tile_code, px, py, chip_size)

    # Determine which SRTM 1° tiles are needed (chips may span 1-4 tiles)
    lat_min_int = int(math.floor(south))
    lat_max_int = int(math.floor(north))
    lon_min_int = int(math.floor(west))
    lon_max_int = int(math.floor(east))

    # Build a stitched elevation grid covering the full chip extent
    n_lat_tiles = lat_max_int - lat_min_int + 1
    n_lon_tiles = lon_max_int - lon_min_int + 1

    # Each SRTM tile is 3601×3601 (1 pixel overlap at edges)
    stitched_h = n_lat_tiles * 3600 + 1
    stitched_w = n_lon_tiles * 3600 + 1
    stitched = np.zeros((stitched_h, stitched_w), dtype=np.float32)
    any_tile_found = False

    for lat_i in range(n_lat_tiles):
        for lon_j in range(n_lon_tiles):
            tile_lat = lat_max_int - lat_i   # NW corner latitude
            tile_lon = lon_min_int + lon_j
            srtm = _download_srtm_hgt(float(tile_lat), float(tile_lon), cache_dir)
            if srtm is not None:
                any_tile_found = True
                r0 = lat_i * 3600
                c0 = lon_j * 3600
                stitched[r0:r0 + 3601, c0:c0 + 3601] = srtm

    if not any_tile_found:
        return None

    # Extract the chip region from the stitched grid
    px_per_deg = 3600
    col_start = int((west - lon_min_int) * px_per_deg)
    col_end = int((east - lon_min_int) * px_per_deg)
    row_start = int((lat_max_int + 1 - north) * px_per_deg)
    row_end = int((lat_max_int + 1 - south) * px_per_deg)

    col_start = max(0, min(col_start, stitched_w - 1))
    col_end = max(col_start + 1, min(col_end, stitched_w))
    row_start = max(0, min(row_start, stitched_h - 1))
    row_end = max(row_start + 1, min(row_end, stitched_h))

    crop = stitched[row_start:row_end, col_start:col_end]
    if crop.shape[0] < 2 or crop.shape[1] < 2:
        return None

    from PIL import Image
    img = Image.fromarray(crop)
    elevation = np.array(
        img.resize((chip_size, chip_size), getattr(Image, 'Resampling', Image).BILINEAR), dtype=np.float32
    )

    return _derive_real_terrain(elevation)


# ═══════════════════════════════════════════════════════════════════════════
# VIIRS Fire Download (from NASA FIRMS API — free MAP_KEY)
# ═══════════════════════════════════════════════════════════════════════════

def _download_viirs_fires(
    west: float,
    south: float,
    east: float,
    north: float,
    firms_key: str,
    years: List[int] = None,
) -> Optional[List[Dict]]:
    """Query NASA FIRMS API for VIIRS fire detections in a bounding box.

    Returns list of fire detection dicts, or None on failure.
    Each detection has: latitude, longitude, bright_ti4, bright_ti5,
    frp, confidence, acq_date.
    """
    if years is None:
        years = list(range(2018, 2024))

    all_fires = []

    for year in years:
        # FIRMS API: request 10-day windows across the year
        for month in range(1, 13):
            date = f"{year}-{month:02d}-01"
            url = FIRMS_API_URL.format(
                key=firms_key,
                west=round(west, 4),
                south=round(south, 4),
                east=round(east, 4),
                north=round(north, 4),
                date=date,
            )

            try:
                req = urllib.request.Request(url, headers={"User-Agent": "MISDO/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    csv_text = resp.read().decode("utf-8")

                lines = csv_text.strip().split("\n")
                if len(lines) < 2:
                    continue

                headers = lines[0].split(",")
                for line in lines[1:]:
                    values = line.split(",")
                    if len(values) >= len(headers):
                        fire = dict(zip(headers, values))
                        all_fires.append(fire)

            except Exception:
                continue

    return all_fires if all_fires else None


def _rasterize_fires(
    fires: List[Dict],
    west: float,
    south: float,
    east: float,
    north: float,
    chip_size: int = 256,
) -> Dict[str, np.ndarray]:
    """Rasterize VIIRS fire point detections into chip-sized grids.

    Creates aggregate channels PLUS per-year fire count rasters
    (viirs_fire_year_01 through viirs_fire_year_23) so temporal
    models can access fire data aligned to Hansen lossyear.

    Aggregate channels:
        viirs_fire_count      — total fire detections per pixel (normalised)
        viirs_mean_frp        — mean fire radiative power
        viirs_max_bright_ti4  — max brightness temperature I4 band
        viirs_max_bright_ti5  — max brightness temperature I5 band
        viirs_confidence      — mean detection confidence
        viirs_persistence     — fraction of time periods with fire
    """
    fire_count = np.zeros((chip_size, chip_size), dtype=np.float32)
    sum_frp = np.zeros((chip_size, chip_size), dtype=np.float32)
    max_ti4 = np.zeros((chip_size, chip_size), dtype=np.float32)
    max_ti5 = np.zeros((chip_size, chip_size), dtype=np.float32)
    sum_conf = np.zeros((chip_size, chip_size), dtype=np.float32)
    date_set = [{} for _ in range(chip_size * chip_size)]

    # Per-year fire counts for temporal alignment
    # Years 1-23 correspond to Hansen lossyear encoding (2001=1, ..., 2023=23)
    # VIIRS only available from 2012 (year 12) onwards
    per_year_counts: Dict[int, np.ndarray] = {}
    for y in range(12, 24):  # 2012-2023
        per_year_counts[y] = np.zeros((chip_size, chip_size), dtype=np.float32)

    lon_range = east - west
    lat_range = north - south

    for fire in fires:
        try:
            lat = float(fire.get("latitude", 0))
            lon = float(fire.get("longitude", 0))
            frp = float(fire.get("frp", 0))
            ti4 = float(fire.get("bright_ti4", 0))
            ti5 = float(fire.get("bright_ti5", 0))
            conf_str = fire.get("confidence", "nominal")
            acq_date = fire.get("acq_date", "")

            # Convert confidence labels to numeric
            if conf_str == "high":
                conf = 1.0
            elif conf_str == "nominal":
                conf = 0.5
            else:
                conf = float(conf_str) / 100.0 if conf_str.replace(".", "").isdigit() else 0.3

            # Map lat/lon to pixel coordinates
            col = int((lon - west) / lon_range * chip_size)
            row = int((north - lat) / lat_range * chip_size)

            if 0 <= row < chip_size and 0 <= col < chip_size:
                fire_count[row, col] += 1
                sum_frp[row, col] += frp
                if ti4 > max_ti4[row, col]:
                    max_ti4[row, col] = ti4
                if ti5 > max_ti5[row, col]:
                    max_ti5[row, col] = ti5
                sum_conf[row, col] += conf
                flat = row * chip_size + col
                date_set[flat][acq_date] = True

                # Bucket into per-year raster
                if len(acq_date) >= 4:
                    try:
                        fire_year = int(acq_date[:4]) - 2000  # 2018 → 18
                        if fire_year in per_year_counts:
                            per_year_counts[fire_year][row, col] += 1
                    except ValueError:
                        pass

        except (ValueError, KeyError):
            continue

    # Normalise aggregate channels
    valid = fire_count > 0
    mean_frp = np.zeros_like(fire_count)
    mean_conf = np.zeros_like(fire_count)
    mean_frp[valid] = sum_frp[valid] / fire_count[valid]
    mean_conf[valid] = sum_conf[valid] / fire_count[valid]

    # ── Physics-based normalization ──
    # Use log-transform + VIIRS sensor physical ranges so absolute
    # magnitudes are preserved across chips.  Per-chip max normalization
    # was stripping this signal (a chip with 1 campfire and a chip with
    # 500 wildfire detections both had their max rescaled to 1.0).
    #
    # Log-transform handles the heavy-tailed FRP distribution where
    # values span 3+ orders of magnitude (1 MW smouldering → 2500 MW
    # crown fire).  Source: NASA VIIRS Active Fire Product User Guide.
    _VIIRS_FIRE_COUNT_99P = 50.0   # annual detections per 375m pixel, 99th pctl
    _VIIRS_FRP_99P = 200.0          # MW, 99th percentile of vegetation fire FRP

    fire_count_norm = np.log1p(fire_count) / np.log1p(_VIIRS_FIRE_COUNT_99P)
    fire_count_norm = np.clip(fire_count_norm, 0, 1).astype(np.float32)

    mean_frp_norm = np.log1p(mean_frp) / np.log1p(_VIIRS_FRP_99P)
    mean_frp_norm = np.clip(mean_frp_norm, 0, 1).astype(np.float32)

    # Brightness temps: normalise using VIIRS physical BT ranges.
    # I4 (3.7 μm MIR): ambient ~300 K, fire threshold ~310 K, saturation ~367 K
    # I5 (11 μm TIR):  ambient ~250 K, valid range ~250–380 K
    max_ti4_norm = np.clip((max_ti4 - 300) / 200, 0, 1)
    max_ti5_norm = np.clip((max_ti5 - 250) / 100, 0, 1)

    # Fire persistence: fraction of unique dates with fire
    total_dates = len(set(d for ds in date_set for d in ds.keys())) or 1
    persistence = np.zeros_like(fire_count)
    for i in range(chip_size):
        for j in range(chip_size):
            flat = i * chip_size + j
            persistence[i, j] = len(date_set[flat]) / total_dates

    result: Dict[str, np.ndarray] = {
        "viirs_fire_count": fire_count_norm.astype(np.float32),
        "viirs_mean_frp": mean_frp_norm.astype(np.float32),
        "viirs_max_bright_ti4": max_ti4_norm.astype(np.float32),
        "viirs_max_bright_ti5": max_ti5_norm.astype(np.float32),
        "viirs_confidence": mean_conf.astype(np.float32),
        "viirs_persistence": persistence.astype(np.float32),
        "has_real_viirs": np.array([1.0], dtype=np.float32),
    }

    # Per-year fire rasters — log-transform with same physical constant
    # to preserve cross-year magnitude comparison (fire trend detection).
    for year_code, counts in per_year_counts.items():
        norm = np.log1p(counts) / np.log1p(_VIIRS_FIRE_COUNT_99P)
        norm = np.clip(norm, 0, 1)
        result[f"viirs_fire_year_{year_code:02d}"] = norm.astype(np.float32)

    return result


def download_viirs_for_chip(
    tile_code: str,
    px: int,
    py: int,
    chip_size: int,
    firms_key: str,
) -> Optional[Dict[str, np.ndarray]]:
    """Download and rasterize VIIRS fire data for a chip.

    Returns dict of fire channel arrays, or None if unavailable.
    """
    west, south, east, north = _chip_bounds(tile_code, px, py, chip_size)

    # Expand bounds slightly for VIIRS (375m resolution >> 30m Hansen)
    buffer = 0.01  # ~1 km buffer
    fires = _download_viirs_fires(
        west - buffer, south - buffer, east + buffer, north + buffer,
        firms_key,
    )

    if fires is None:
        return None

    return _rasterize_fires(fires, west, south, east, north, chip_size)


# ═══════════════════════════════════════════════════════════════════════════
# Proxy Terrain Fallback (used when SRTM unavailable)
# ═══════════════════════════════════════════════════════════════════════════

def _derive_proxy_terrain(treecover: np.ndarray) -> Dict[str, np.ndarray]:
    """Derive terrain proxy features from treecover spatial patterns.

    Fallback when real SRTM tiles are unavailable. Clearly flagged
    as proxy data via has_real_srtm=0.

    Uses Gaussian-smoothed treecover texture as a rough elevation
    analogue (ridges tend to have less canopy) and spatial gradients
    for slope / aspect proxies.
    """
    from scipy.ndimage import gaussian_filter

    tc_norm = treecover.astype(np.float32) / 100.0

    # Proxy elevation: smooth treecover texture (large sigma captures
    # landscape-scale variation; inverted so sparser canopy = higher)
    elev_proxy = gaussian_filter(1.0 - tc_norm, sigma=15)
    e_min, e_max = elev_proxy.min(), elev_proxy.max()
    if e_max - e_min > 1e-6:
        elev_proxy = (elev_proxy - e_min) / (e_max - e_min)
    else:
        elev_proxy = np.full_like(elev_proxy, 0.5)

    # Slope / aspect from elevation proxy gradients
    dy, dx = np.gradient(elev_proxy)
    slope = np.sqrt(dx**2 + dy**2)
    s_max = slope.max()
    slope = slope / (s_max + 1e-8)
    aspect = (np.mod(np.arctan2(-dx, dy), 2 * np.pi) / (2 * np.pi))

    # Flow accumulation from proxy elevation
    flow_acc = _compute_flow_accumulation(
        elev_proxy * 1000  # scale up so drops are meaningful
    )
    flow_acc = np.log1p(flow_acc)
    fa_max = flow_acc.max()
    flow_acc = (flow_acc / (fa_max + 1e-8)).astype(np.float32)

    # Flow direction from proxy gradients
    flow_dir = _compute_flow_direction(dx, dy)

    return {
        "srtm_elevation": elev_proxy.astype(np.float32),
        "srtm_slope": slope.astype(np.float32),
        "srtm_aspect": aspect.astype(np.float32),
        "srtm_flow_acc": flow_acc,
        "srtm_flow_dir": flow_dir.astype(np.float32),
        "has_real_srtm": np.array([0.0], dtype=np.float32),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Single Chip Download (Hansen + SRTM + VIIRS)
# ═══════════════════════════════════════════════════════════════════════════

def _download_single_chip(
    tile_id: str,
    tile_code: str,
    px: int,
    py: int,
    chip_size: int,
    output_file: str,
    srtm_cache_dir: str,
    firms_key: Optional[str] = None,
    viirs_archive: Optional['VIIRSArchive'] = None,
) -> Optional[Dict]:
    """Download a complete chip: Hansen GFC + SRTM elevation + VIIRS fire.

    VIIRS priority: bulk archive → API (firms_key) → skip (proxy fallback).
    Returns metadata dict on success, None on failure.
    """
    import rasterio
    from rasterio.windows import Window

    if os.path.exists(output_file):
        return {"file": output_file, "px": px, "py": py, "tile": tile_id, "cached": True}

    chip_data = {}
    try:
        # ── 1. Hansen GFC layers (with retry for transient HTTP errors) ──
        for layer in GFC_LAYERS:
            url = GFC_BASE_URL + f"Hansen_GFC-2023-v1.11_{layer}_{tile_code}.tif"
            for _attempt in range(3):
                try:
                    with rasterio.open(url) as src:
                        window = Window(px, py, chip_size, chip_size)
                        data = src.read(1, window=window)
                        if data.shape != (chip_size, chip_size):
                            return None
                        chip_data[layer] = data.astype(np.float32)
                    break
                except Exception:
                    if _attempt == 2:
                        raise
                    time.sleep(2 ** _attempt)

        # ── 2. SRTM elevation (real terrain) ──
        srtm_data = download_srtm_for_chip(
            tile_code, px, py, chip_size, srtm_cache_dir
        )
        if srtm_data is not None:
            chip_data.update(srtm_data)
        else:
            # Fallback to proxy terrain
            chip_data.update(_derive_proxy_terrain(chip_data["treecover2000"]))

        # ── 3. VIIRS fire detections ──
        # Priority: bulk archive → per-chip API → skip
        if viirs_archive is not None:
            west, south, east, north = _chip_bounds(tile_code, px, py, chip_size)
            buffer = 0.01  # ~1 km buffer (VIIRS 375m >> Hansen 30m)
            fires = viirs_archive.query(
                west - buffer, south - buffer, east + buffer, north + buffer
            )
            if fires:
                chip_data.update(
                    _rasterize_fires(fires, west, south, east, north, chip_size)
                )
                chip_data["has_real_viirs"] = np.array([1.0], dtype=np.float32)
        elif firms_key:
            viirs_data = download_viirs_for_chip(
                tile_code, px, py, chip_size, firms_key
            )
            if viirs_data is not None:
                chip_data.update(viirs_data)

        # ── 4. Store geographic bounds for spatial CV ──
        west, south, east, north = _chip_bounds(tile_code, px, py, chip_size)
        chip_data["bounds"] = np.array(
            [west, south, east, north], dtype=np.float32
        )

        np.savez_compressed(output_file, **chip_data)

        has_srtm = "✓" if chip_data.get("has_real_srtm", np.array([0]))[0] > 0 else "proxy"
        has_viirs = "✓" if "has_real_viirs" in chip_data else "—"

        return {
            "file": output_file, "px": px, "py": py, "tile": tile_id,
            "lat": round((south + north) / 2, 4),
            "lon": round((west + east) / 2, 4),
            "has_real_srtm": bool(chip_data.get("has_real_srtm", np.array([0]))[0] > 0),
            "has_real_viirs": "has_real_viirs" in chip_data,
        }

    except Exception as e:
        print(f"    ✗ Failed chip at ({px},{py}) on {tile_id}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Tile Download
# ═══════════════════════════════════════════════════════════════════════════

def download_tile(
    tile_name: str,
    tile_info: Dict,
    output_dir: str,
    split: str = "train",
    chips_per_tile: int = 500,
    chip_size: int = 256,
    min_forest_pct: float = 0.20,
    max_forest_pct: float = 0.95,
    srtm_cache_dir: str = "",
    firms_key: Optional[str] = None,
    viirs_archive: Optional['VIIRSArchive'] = None,
) -> List[Dict]:
    """Download chips from a single Hansen GFC tile with SRTM + VIIRS.

    ALL chips from this tile go into a SINGLE split to prevent
    spatial data leakage.  Tile-level splitting is done in download_all().

    Chips with forest cover above max_forest_pct are skipped because
    pristine-forest chips have near-zero deforestation labels (empty targets).
    """
    import rasterio
    from rasterio.windows import Window

    tile_code = tile_info["tile"]
    print(f"\n  [{tile_name}] {tile_info['region']} — {tile_info['description']}")
    print(f"    Tile: {tile_code}  |  Target: {chips_per_tile} chips  |  Split: {split}")
    srtm_status = "✓ Real SRTM" if srtm_cache_dir else "Proxy"
    viirs_status = ("✓ Real VIIRS (bulk archive)" if viirs_archive is not None
                     else ("✓ Real VIIRS (API)" if firms_key else "No VIIRS"))
    print(f"    Data: {srtm_status} | {viirs_status}")

    # Generate candidate positions
    np.random.seed(hash(tile_name) % 2**31)
    candidates = []
    for gx in range(1500, 38000, 800):
        for gy in range(1500, 38000, 800):
            jx = gx + np.random.randint(-300, 300)
            jy = gy + np.random.randint(-300, 300)
            candidates.append((jx, jy))
    np.random.shuffle(candidates)

    # Scan for forest cover
    tc_url = GFC_BASE_URL + f"Hansen_GFC-2023-v1.11_treecover2000_{tile_code}.tif"
    good_positions = []

    try:
        with rasterio.open(tc_url) as src:
            for px, py in candidates:
                if len(good_positions) >= chips_per_tile:
                    break
                try:
                    window = Window(px, py, chip_size, chip_size)
                    tc = src.read(1, window=window)
                    if tc.shape != (chip_size, chip_size):
                        continue
                    forest_pct = (tc > 30).mean()
                    if forest_pct > min_forest_pct and forest_pct < max_forest_pct:
                        good_positions.append((px, py, float(forest_pct)))
                except Exception:
                    continue
    except Exception as e:
        print(f"    ✗ Cannot access tile {tile_code}: {e}")
        return []

    if not good_positions:
        print(f"    ✗ No suitable chips found")
        return []

    print(f"    Found {len(good_positions)} chips → all assigned to '{split}'")

    entries = []
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    for idx, (px, py, fc) in enumerate(good_positions):
        chip_name = f"{tile_name}_chip_{idx:03d}"
        chip_file = os.path.join(split_dir, f"{chip_name}.npz")

        # Store path relative to output_dir for portability
        rel_path = os.path.relpath(chip_file, output_dir)

        result = _download_single_chip(
            tile_name, tile_code, px, py, chip_size, chip_file,
            srtm_cache_dir, firms_key, viirs_archive,
        )
        if result:
            result["file"] = rel_path    # overwrite with portable relative path
            result["forest_pct"] = round(fc, 3)
            result["biome"] = tile_info["biome"]
            result["region"] = tile_info["region"]
            entries.append(result)

    n_srtm = sum(1 for e in entries if e.get("has_real_srtm"))
    n_viirs = sum(1 for e in entries if e.get("has_real_viirs"))
    print(f"    ✓ Downloaded {len(entries)} chips to '{split}'"
          f"  (SRTM: {n_srtm}, VIIRS: {n_viirs})")

    return entries


# ═══════════════════════════════════════════════════════════════════════════
# Global Tile Discovery — Scan Hansen GCS for All Forested Tiles
# ═══════════════════════════════════════════════════════════════════════════

def _probe_tile_forest_pct(
    tile_code: str,
    sample_positions: int = 25,
    chip_size: int = 256,
    min_forest_threshold: int = 30,
) -> Optional[float]:
    """Quick-scan a tile for forest cover percentage.

    Reads a few small windows from the treecover2000 GeoTIFF to
    estimate whether the tile has meaningful forest.
    Returns estimated forest fraction, or None if tile doesn't exist.
    """
    import rasterio
    from rasterio.windows import Window

    url = GFC_BASE_URL + f"Hansen_GFC-2023-v1.11_treecover2000_{tile_code}.tif"

    try:
        with rasterio.open(url) as src:
            h, w = src.height, src.width
            rng = np.random.RandomState(hash(tile_code) % 2**31)

            forest_pixels = 0
            total_pixels = 0
            for _ in range(sample_positions):
                px = rng.randint(1000, max(1001, w - chip_size - 1000))
                py = rng.randint(1000, max(1001, h - chip_size - 1000))
                window = Window(px, py, chip_size, chip_size)
                try:
                    tc = src.read(1, window=window)
                    if tc.shape == (chip_size, chip_size):
                        forest_pixels += (tc > min_forest_threshold).sum()
                        total_pixels += tc.size
                except Exception:
                    continue

            if total_pixels == 0:
                return None
            return float(forest_pixels) / total_pixels

    except Exception:
        return None


def discover_forested_tiles(
    min_forest_pct: float = 0.05,
    parallel: int = 8,
    lat_range: Optional[Tuple[int, int]] = None,
    cache_file: Optional[str] = None,
) -> List[Dict]:
    """Discover ALL Hansen GFC tiles with meaningful forest cover.

    Scans the full 80°N–60°S × 180°W–180°E grid and returns tiles
    where the estimated forest percentage exceeds min_forest_pct.

    Parameters
    ----------
    min_forest_pct : float
        Minimum fraction of forested pixels to include a tile (default 5%).
    parallel : int
        Number of concurrent HTTP probes.
    lat_range : tuple of (min_lat, max_lat), optional
        Filter to specific latitude band (e.g. (-10, 10) for tropics).
    cache_file : str, optional
        Path to cache discovery results as JSON.

    Returns
    -------
    list of dicts with keys: tile_code, forest_pct, lat, lon
    """
    # Check cache first
    if cache_file and os.path.exists(cache_file):
        with open(cache_file) as f:
            cached = json.load(f)
        print(f"  Loaded {len(cached)} tiles from cache: {cache_file}")
        if lat_range is not None:
            min_lat, max_lat = lat_range
            cached = [
                t for t in cached
                if (t["lat"] - 10) < max_lat and t["lat"] > min_lat
            ]
        return cached

    all_codes = _enumerate_all_tile_codes()

    # Apply latitude filter if specified
    if lat_range is not None:
        min_lat, max_lat = lat_range
        filtered = []
        for code in all_codes:
            lat, lon = _parse_tile_code(code)
            tile_south = lat - 10
            if tile_south < max_lat and lat > min_lat:
                filtered.append(code)
        all_codes = filtered

    print(f"  Scanning {len(all_codes)} potential tiles for forest cover...")
    results = []
    scanned = 0

    def _scan_one(code: str) -> Optional[Dict]:
        pct = _probe_tile_forest_pct(code)
        if pct is not None and pct >= min_forest_pct:
            lat, lon = _parse_tile_code(code)
            return {
                "tile_code": code,
                "forest_pct": round(pct, 4),
                "lat": lat,
                "lon": lon,
            }
        return None

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(_scan_one, code): code for code in all_codes}
        for future in as_completed(futures):
            scanned += 1
            if scanned % 50 == 0:
                print(f"    Scanned {scanned}/{len(all_codes)} tiles, "
                      f"found {len(results)} forested...")
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception:
                continue

    results.sort(key=lambda x: -x["forest_pct"])
    print(f"  ✓ Found {len(results)} forested tiles out of {len(all_codes)} scanned")

    if cache_file:
        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Cached to: {cache_file}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Download All Tiles (supports both curated and discovered tiles)
# ═══════════════════════════════════════════════════════════════════════════

def download_all(
    output_dir: str,
    tile_list: Optional[List[Dict]] = None,
    chips_per_tile: int = 1000,
    parallel: int = 1,
    firms_key: Optional[str] = None,
    train_ratio: float = 0.8,
    viirs_archive_dir: Optional[str] = None,
) -> Dict:
    """Download chips from all tiles in tile_list.

    TILE-LEVEL SPLITTING: entire tiles are assigned to either 'train'
    or 'test' to prevent spatial data leakage.  No chips from the same
    tile ever appear in both splits.

    Parameters
    ----------
    tile_list : list of dicts
        Each dict must have at least 'tile_code'. May also have
        'forest_pct', 'biome', 'region', etc.
        If None, uses CURATED_TILES.
    train_ratio : float
        Fraction of tiles assigned to training (default 0.8).
    """
    # Build tile list from curated registry if not provided
    if tile_list is None:
        tile_list = [
            {"tile_code": v["tile"], "name": k, "biome": v["biome"],
             "region": v["region"], "description": v["description"]}
            for k, v in CURATED_TILES.items()
        ]

    srtm_cache_dir = os.path.join(output_dir, ".srtm_cache")
    os.makedirs(srtm_cache_dir, exist_ok=True)

    # ── Tile-level spatial split ──────────────────────────────────────
    # Shuffle tiles with a fixed seed and assign 80% to train, 20% to test.
    # This ensures ZERO spatial overlap between train and test.
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(tile_list))
    n_train_tiles = max(1, int(len(tile_list) * train_ratio))
    train_indices = set(indices[:n_train_tiles].tolist())

    # Assign split label to each tile
    tile_splits = []
    for i, ti in enumerate(tile_list):
        split = "train" if i in train_indices else "test"
        tile_splits.append(split)

    n_train_t = sum(1 for s in tile_splits if s == "train")
    n_test_t = sum(1 for s in tile_splits if s == "test")

    print(f"\nTile-level spatial split: {n_train_t} train tiles, "
          f"{n_test_t} test tiles (ratio={train_ratio})")
    print(f"Downloading {len(tile_list)} tiles × {chips_per_tile} chips/tile")
    print(f"Target: ~{len(tile_list) * chips_per_tile:,} total chips")
    print(f"SRTM: ✓ Real elevation (cached to {srtm_cache_dir})")
    # ── Load VIIRS bulk archive if provided ───────────────────────────
    viirs_archive: Optional[VIIRSArchive] = None
    if viirs_archive_dir is not None:
        import glob
        csv_files = sorted(glob.glob(os.path.join(viirs_archive_dir, "*.csv")))
        if csv_files:
            print(f"\nLoading VIIRS bulk archive from {viirs_archive_dir}...")
            viirs_archive = VIIRSArchive(csv_files)
        else:
            print(f"\n⚠ No CSV files found in {viirs_archive_dir} — falling back")

    viirs_source = (
        "✓ Bulk archive (no rate limit)" if viirs_archive is not None
        else ("✓ Real fire data (FIRMS API)" if firms_key
              else "✗ No VIIRS source — fire data skipped")
    )
    print(f"VIIRS: {viirs_source}\n")

    manifest: Dict[str, Any] = {"train": [], "test": [], "metadata": {
        "n_tiles": len(tile_list),
        "n_train_tiles": n_train_t,
        "n_test_tiles": n_test_t,
        "chips_per_tile": chips_per_tile,
        "has_real_srtm": True,
        "has_real_viirs": viirs_archive is not None or bool(firms_key),
        "viirs_source": ("bulk_archive" if viirs_archive is not None
                         else ("api" if firms_key else "none")),
        "split_strategy": "tile-level (no within-tile splitting)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }}

    os.makedirs(output_dir, exist_ok=True)

    def _download_one_tile(tile_info: Dict, split: str):
        tile_code = tile_info["tile_code"]
        tile_name = tile_info.get("name", f"tile_{tile_code}")
        # Build a tile_info dict compatible with download_tile()
        ti = {
            "tile": tile_code,
            "biome": tile_info.get("biome", "unknown"),
            "region": tile_info.get("region", tile_code),
            "description": tile_info.get("description", ""),
        }
        return split, download_tile(
            tile_name, ti, output_dir, split, chips_per_tile,
            256, 0.10, 0.95,
            srtm_cache_dir, firms_key, viirs_archive,
        )

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(_download_one_tile, ti, tile_splits[i]): ti
                for i, ti in enumerate(tile_list)
            }
            for future in as_completed(futures):
                ti = futures[future]
                try:
                    split, entries = future.result()
                    manifest[split].extend(entries)
                except Exception as e:
                    print(f"  ✗ Tile {ti.get('tile_code', '?')} failed: {e}")
    else:
        for i, ti in enumerate(tile_list):
            try:
                split, entries = _download_one_tile(ti, tile_splits[i])
                manifest[split].extend(entries)
            except Exception as e:
                print(f"  ✗ Tile {ti.get('tile_code', '?')} failed: {e}")

    # Save manifest
    manifest_file = os.path.join(output_dir, "manifest.json")
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MISDO — Global-scale real data downloader "
                    "(Hansen GFC + SRTM + VIIRS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full global download (recommended for A100 training)
  python datasets/download_real_data.py --mode global --parallel 16

  # Curated 30 tiles only
  python datasets/download_real_data.py --mode curated --parallel 8

  # Tropics only
  python datasets/download_real_data.py --mode global --lat-range -23 23

  # Scan which tiles have forest (no download)
  python datasets/download_real_data.py --discover-tiles --parallel 16
""",
    )
    parser.add_argument(
        "--mode", default="global",
        choices=["global", "curated"],
        help="'global' = discover + download ALL forested tiles (~300), "
             "'curated' = use 30 hand-picked tiles (default: global)",
    )
    parser.add_argument(
        "--lat-range", type=int, nargs=2, default=None,
        metavar=("MIN_LAT", "MAX_LAT"),
        help="Latitude range filter, e.g. --lat-range -23 23 for tropics",
    )
    parser.add_argument(
        "--min-forest-pct", type=float, default=0.05,
        help="Minimum forest fraction to include a tile (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--chips-per-tile", type=int, default=1000,
        help="Chips per tile (default: 1000)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: datasets/real_tiles)",
    )
    parser.add_argument(
        "--parallel", type=int, default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--firms-key", default=None,
        help="NASA FIRMS MAP_KEY for VIIRS fire data. "
             "Get one free at: https://firms.modaps.eosdis.nasa.gov/api/",
    )
    parser.add_argument(
        "--viirs-archive", default=None, metavar="PATH",
        help="Path to directory containing FIRMS bulk CSV files. "
             "Bypasses per-chip API calls (no rate limit). "
             "Download from: https://firms.modaps.eosdis.nasa.gov/download/",
    )
    parser.add_argument(
        "--discover-tiles", action="store_true",
        help="Only scan for forested tiles (no download). Saves cache file.",
    )
    parser.add_argument(
        "--list-tiles", action="store_true",
        help="List the 30 curated tiles and exit",
    )
    parser.add_argument(
        "--max-total-chips", type=int, default=None,
        help="Maximum total chips to download across all tiles. "
             "If set, reduces chips_per_tile to stay within this limit. "
             "Recommended: 30000 for curated (1-2 day training), "
             "100000 for global (~1 week training on A100).",
    )
    args = parser.parse_args()

    firms_key = args.firms_key or os.environ.get("FIRMS_MAP_KEY")
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "real_tiles"
    )
    cache_file = os.path.join(output_dir, ".tile_discovery_cache.json")

    # ── List curated tiles ──
    if args.list_tiles:
        print("\nCurated tiles (30):\n")
        for name, info in CURATED_TILES.items():
            print(f"  {name:25s}  {info['tile']:10s}  "
                  f"[{info['biome']:20s}]  {info['region']}")
        return

    # ── Discovery-only mode ──
    if args.discover_tiles:
        print("=" * 70)
        print("  MISDO — Global Tile Discovery")
        print("=" * 70)
        lat_range = tuple(args.lat_range) if args.lat_range else None
        tiles = discover_forested_tiles(
            min_forest_pct=args.min_forest_pct,
            parallel=args.parallel,
            lat_range=lat_range,
            cache_file=cache_file,
        )
        print(f"\n  Found {len(tiles)} forested tiles")
        print(f"  Top 20 by forest cover:")
        for t in tiles[:20]:
            print(f"    {t['tile_code']:10s}  forest={t['forest_pct']:.1%}  "
                  f"lat={t['lat']:+d}  lon={t['lon']:+d}")
        return

    # ── Build tile list ──
    lat_range = tuple(args.lat_range) if args.lat_range else None

    if args.mode == "global":
        print("=" * 70)
        print("  MISDO — Global-Scale Data Downloader")
        print("  Hansen GFC + SRTM Elevation + VIIRS Fire")
        print("=" * 70)
        print(f"\n  Mode: GLOBAL (all forested tiles)")
        print(f"  Min forest: {args.min_forest_pct:.0%}")
        if lat_range:
            print(f"  Latitude range: {lat_range[0]}° to {lat_range[1]}°")

        tile_list = discover_forested_tiles(
            min_forest_pct=args.min_forest_pct,
            parallel=args.parallel,
            lat_range=lat_range,
            cache_file=cache_file,
        )
    else:
        # Curated mode
        tile_list = [
            {"tile_code": v["tile"], "name": k, "biome": v["biome"],
             "region": v["region"], "description": v["description"]}
            for k, v in CURATED_TILES.items()
        ]
        print("=" * 70)
        print("  MISDO — Curated Data Downloader (30 tiles)")
        print("  Hansen GFC + SRTM Elevation + VIIRS Fire")
        print("=" * 70)

    if not tile_list:
        print("No forested tiles found. Try lowering --min-forest-pct.")
        return

    # ── Enforce max-total-chips bound ──
    chips_per_tile = args.chips_per_tile
    if args.max_total_chips is not None:
        max_per_tile = max(10, args.max_total_chips // len(tile_list))
        if max_per_tile < chips_per_tile:
            print(f"\n  ⚠ Reducing chips_per_tile from {chips_per_tile:,} to "
                  f"{max_per_tile:,} to stay within --max-total-chips={args.max_total_chips:,}")
            chips_per_tile = max_per_tile

    total_chips = len(tile_list) * chips_per_tile
    est_size_gb_low = total_chips * 0.5 / 1000  # ~0.5 MB per chip compressed
    est_size_gb_high = total_chips * 2.0 / 1000  # ~2.0 MB with SRTM+VIIRS

    print(f"\n  Tiles: {len(tile_list)}")
    print(f"  Chips per tile: {chips_per_tile:,}")
    print(f"  Estimated total: ~{total_chips:,} chips")
    print(f"  ⚠ Estimated disk usage: {est_size_gb_low:.0f}–{est_size_gb_high:.0f} GB")
    if total_chips > 100_000:
        print(f"  ⚠ WARNING: Large download ({total_chips:,} chips). "
              f"Use --max-total-chips to limit.")
    print(f"  Output: {output_dir}")
    print(f"  Parallel: {args.parallel}")
    print(f"  SRTM: ✓ Real elevation (30m, from AWS)")

    viirs_archive_dir = args.viirs_archive
    viirs_label = (
        f"✓ Bulk archive ({viirs_archive_dir})" if viirs_archive_dir
        else ("✓ Real fire (FIRMS API)" if firms_key
              else "✗ No key — pass --firms-key or --viirs-archive")
    )
    print(f"  VIIRS: {viirs_label}")

    t0 = time.time()
    manifest = download_all(
        output_dir, tile_list, chips_per_tile, args.parallel, firms_key,
        viirs_archive_dir=viirs_archive_dir,
    )
    elapsed = time.time() - t0

    n_srtm = sum(1 for e in manifest["train"] + manifest["test"]
                 if e.get("has_real_srtm"))
    n_viirs = sum(1 for e in manifest["train"] + manifest["test"]
                  if e.get("has_real_viirs"))
    total = len(manifest["train"]) + len(manifest["test"])

    print(f"\n{'=' * 70}")
    print(f"  Download complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Tiles processed: {len(tile_list)}")
    print(f"  Train chips: {len(manifest['train']):,}")
    print(f"  Test chips:  {len(manifest['test']):,}")
    print(f"  Real SRTM:   {n_srtm:,}/{total:,} chips")
    print(f"  Real VIIRS:  {n_viirs:,}/{total:,} chips")
    print(f"  Manifest: {os.path.join(output_dir, 'manifest.json')}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()