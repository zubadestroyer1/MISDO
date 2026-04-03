# MISDO — Training Runbook

> Complete step-by-step instructions for training the 4 counterfactual impact models on an A100 GPU cluster.

---

## Table of Contents

1. [Cluster Setup & Environment](#1-cluster-setup--environment)
2. [Download Training Data](#2-download-training-data)
3. [Verify Data Integrity](#3-verify-data-integrity)
4. [Training the Models](#4-training-the-models)
5. [Evaluate & Validate](#5-evaluate--validate)
6. [Save & Export Weights](#6-save--export-weights)
7. [Quick Reference Card](#7-quick-reference-card)

---

## 1. Cluster Setup & Environment

### 1.1 Provision the A100 Instance

| Setting | Recommended Value |
|---|---|
| **GPU** | A100 80GB (or A100 40GB) |
| **vCPUs** | 16–32 |
| **RAM** | 64–128 GB |
| **Storage** | 500 GB+ SSD (data is ~200 GB for global, ~15 GB for curated) |
| **OS** | Ubuntu 22.04 LTS with CUDA 12.x |

### 1.2 Clone & Install

```bash
git clone https://github.com/zubadestroyer1/MISDO.git
cd MISDO

# Create conda env
conda create -n misdo python=3.11 -y
conda activate misdo

# PyTorch + CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Project deps
pip install -r requirements.txt
```

### 1.3 Verify CUDA

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
# Expected: CUDA: True, Device: NVIDIA A100-SXM4-80GB
```

---

## 2. Download Training Data

### Option A — Curated Dataset (Recommended First Run)

30 hand-picked tiles spanning 6 biomes across all continents. ~15–30 GB, takes ~1–3 hours.

```bash
python datasets/download_real_data.py \
    --mode curated \
    --chips-per-tile 1000 \
    --parallel 8
```

### Option B — Global Dataset (Full Scale for A100)

Discovers and downloads all ~300 forested tiles worldwide. ~100–200 GB, takes 6–24 hours.

```bash
python datasets/download_real_data.py \
    --mode global \
    --chips-per-tile 1000 \
    --parallel 16
```

### Optional: Add VIIRS Fire Data

Real fire detections improve the fire model significantly.

**Option 1 — Bulk archive (recommended, no rate limit):**

Download VIIRS SNPP CSV archives from [NASA FIRMS Download](https://firms.modaps.eosdis.nasa.gov/download/) and place them in a directory:

```bash
python datasets/download_real_data.py \
    --mode curated \
    --chips-per-tile 1000 \
    --parallel 8 \
    --viirs-archive /path/to/firms_csvs/
```

**Option 2 — Per-chip API (slow, rate-limited):**

Get a free MAP_KEY from [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/area/):

```bash
python datasets/download_real_data.py \
    --mode curated \
    --chips-per-tile 1000 \
    --parallel 8 \
    --firms-key your_key_here
```

> [!WARNING]
> The per-chip API is heavily rate-limited (~10 req/min on free tier). For 30K chips this would take weeks. Use the bulk archive instead.

### Optional: Add MSI/SMAP Augmentation

Downloads Sentinel-2 MSI and NASA SMAP data to enrich Hydro and Soil model targets:

```bash
python datasets/download_msi_smap.py
```

### Optional: Tropics-Only Download

```bash
python datasets/download_real_data.py \
    --mode global \
    --lat-range -23 23 \
    --parallel 16
```

### Where Data Goes

| Path | Contents |
|---|---|
| `datasets/real_tiles/` | All `.npz` chip files |
| `datasets/real_tiles/manifest.json` | Train/test split index (80/20 spatial) |
| `datasets/real_tiles/.srtm_cache/` | Cached SRTM elevation tiles |
| `datasets/real_tiles/.viirs_cache/` | Cached VIIRS bulk archive CSVs (if downloaded) |

---

## 3. Verify Data Integrity

```bash
# Check manifest
python -c "
import json
with open('datasets/real_tiles/manifest.json') as f:
    m = json.load(f)
print(f'Train chips: {len(m[\"train\"]):,}')
print(f'Test chips:  {len(m[\"test\"]):,}')
"
```

> [!IMPORTANT]
> You need at least **~500 train chips** for basic learning. With sliding temporal windows (see below), 500 chips → ~50,000 training samples.

```bash
# Test the pipeline end-to-end
python test_real_pipeline.py
```

---

## 4. Training the Models

### 4.1 The Four Counterfactual Impact Models

| Model | Input | Target | Data Source |
|---|---|---|---|
| **Fire** (`FireRiskNet`) | 7ch temporal (VIIRS + defo mask) | Fire increase near clearing | VIIRS per-year rasters (real) |
| **Forest** (`ForestLossNet`) | 6ch temporal (Hansen + defo mask) | Cascade deforestation | Hansen `lossyear` (real) |
| **Hydro** (`HydroRiskNet`) | 7ch static (SRTM + MSI NDSSI + defo mask) | Erosion increase downstream | Sentinel-2 NDSSI + physics proxy |
| **Soil** (`SoilRiskNet`) | 7ch temporal (SMAP + terrain + defo mask) | Soil degradation increase | TerraClimate SMAP + real terrain |

All models use **Focal Charbonnier + SSIM + Edge-Weighted MSE Loss** which upweights pixels near deforestation edges (3× weight) and uses focal modulation (γ=2.0) to prioritise impact zones.

### 4.2 Split Strategy

| Split | Deforestation Events | Impact Observed By | Spatial Tiles |
|---|---|---|---|
| **Train** | Full range (2001–2023) | Full range | 80% "train" tiles |
| **Test** | Full range (2001–2023) | Full range | First half of 20% "test" tiles |
| **Validate** | Full range (2001–2023) | Full range | Second half of 20% "test" tiles |

**Pure Spatial Tile-Level Splitting**: To prevent spatial data leakage, tiles are strictly assigned to train or test. There is NO temporal splitting; models see the full 23-year range of deforestation events across all splits.

**Sliding temporal windows**: each `__getitem__` randomly samples (T₁, T₂) within the available valid range, multiplying effective dataset size ~100× compared to fixed windows.

### 4.3 Train All 4 Models (Recommended Command)

```bash
python train_real_models.py \
    --model all \
    --epochs 60 \
    --accumulation-steps 4 \
    --amp \
    --early-stop-patience 15
```

**What this does:**
- Trains all 4 models sequentially: Fire → Forest → Hydro → Soil
- **Batch size**: 16 on CUDA (effective = 16 × 4 = 64)
- **AMP**: Mixed precision for 2× speed
- **LR**: warmup → cosine annealing (Fire/Forest: 3e-4, Hydro: 2e-4, Soil: 2.5e-4)
- **Loss**: CounterfactualDeltaLoss wrapping Focal Charbonnier + SSIM + Edge-Weighted MSE
- **Deep supervision**: UNet++ auxiliary losses (weight=0.3), full-resolution re-computation
- **Radiometric jitter**: per-model channel-aware brightness/contrast perturbation (p=0.5)
- **Early stopping**: Halts if test loss doesn’t improve for 10 epochs

### 4.4 Train a Single Model

```bash
python train_real_models.py --model fire --epochs 80 --amp
python train_real_models.py --model forest --epochs 60 --amp
```

### 4.5 Train on Synthetic Data (No Download Required)

For quick testing without real satellite data:

```bash
python train_models.py --model all --epochs 30
```

### 4.6 Expected Training Times (A100 80GB)

| Dataset Size | Epochs | Est. Time per Model | Total (all 4) |
|---|---|---|---|
| Curated (24k chips, ~2.4M samples) | 60 | ~10–20 min | ~1–1.5 hours |
| Global (240k chips, ~24M samples) | 60 | ~2–4 hours | ~8–16 hours |

> [!TIP]
> Use `tmux` or `screen` to keep training running if you disconnect from SSH:
> ```bash
> tmux new -s training
> python train_real_models.py --model all --epochs 60 --amp
> # Ctrl+B then D to detach; tmux attach -t training to reattach
> ```

---

## 5. Evaluate & Validate

```bash
python evaluate_models.py \
    --tiles-dir datasets/real_tiles \
    --weights-dir weights
```

Results saved to `weights/evaluation_report.json`.

---

## 6. Save & Export Weights

Training automatically saves to `weights/`:

| File | Description |
|---|---|
| `weights/fire_model.pt` | Best fire impact model |
| `weights/forest_model.pt` | Best cascade deforestation model |
| `weights/hydro_model.pt` | Best erosion impact model |
| `weights/soil_model.pt` | Best soil degradation model |
| `weights/real_training_metrics.json` | Full training metrics |

```bash
# Back up
tar -czf misdo_weights_$(date +%Y%m%d).tar.gz weights/

# Download to local
scp -r user@a100-host:/path/to/MISDO/weights/ ./weights/
```

---

## 7. Quick Reference Card

```bash
# 1. Setup
git clone https://github.com/zubadestroyer1/MISDO.git && cd MISDO
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Download curated data
python datasets/download_real_data.py --mode curated --chips-per-tile 1000 --parallel 8

# 2b. (Optional) Add VIIRS fire data via bulk archive
python datasets/download_real_data.py --mode curated --parallel 8 --viirs-archive /path/to/firms_csvs/

# 2c. (Optional) Add MSI/SMAP augmentation
python datasets/download_msi_smap.py

# 3. Train all 4 counterfactual impact models
python train_real_models.py --model all --epochs 60 --amp --accumulation-steps 4

# 4. Evaluate
python evaluate_models.py --tiles-dir datasets/real_tiles --weights-dir weights

# 5. Back up
tar -czf misdo_weights_$(date +%Y%m%d).tar.gz weights/
```

### CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `all` | Which model: `fire`, `forest`, `hydro`, `soil`, `all` |
| `--epochs` | `60` | Max training epochs |
| `--accumulation-steps` | `4` | Gradient accumulation (effective batch = batch_size × this) |
| `--amp` | off | Enable mixed precision (always use on A100) |
| `--early-stop-patience` | `15` | Epochs without improvement before stopping |
| `--tiles-dir` | `datasets/real_tiles` | Path to downloaded data |
| `--weights-dir` | `weights` | Path to save model checkpoints |
| `--viirs-archive` | — | Path to FIRMS bulk CSV directory (bypasses API rate limit) |
| `--firms-key` | — | NASA FIRMS MAP_KEY for per-chip API (slow, rate-limited) |

### Troubleshooting

| Issue | Solution |
|---|---|
| `No real data found` | Run the download script first (Step 2) |
| `CUDA out of memory` | Reduce `--accumulation-steps` to 2 |
| `treecover2000 missing` | Hansen GCS server may be throttling — retry with fewer `--parallel` |
| `SRTM download failures` | Some polar tiles lack SRTM — these fall back to proxy terrain |
| `No VIIRS fire data` | Use `--viirs-archive` with bulk CSV files, or pass `--firms-key` |
| `Training loss not decreasing` | Verify data with `python test_real_pipeline.py` |
