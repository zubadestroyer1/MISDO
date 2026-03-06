# MISDO — A100 Training Runbook

> Complete step-by-step instructions for training all 4 domain-specific risk models on an A100 GPU cluster.

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

On your cloud provider (GCP, AWS, Lambda Labs, etc.):

| Setting | Recommended Value |
|---|---|
| **GPU** | A100 80GB (or A100 40GB) |
| **vCPUs** | 16–32 |
| **RAM** | 64–128 GB |
| **Storage** | 500 GB+ SSD (data is ~200 GB for global, ~15 GB for curated) |
| **OS** | Ubuntu 22.04 LTS with CUDA 12.x |

### 1.2 Clone the Repository

```bash
git clone https://github.com/zubadestroyer1/MISDO.git
cd MISDO
```

### 1.3 Install Dependencies

```bash
# Create a conda env (recommended)
conda create -n misdo python=3.11 -y
conda activate misdo

# Install PyTorch with CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

**`requirements.txt` includes:** `torch>=2.0`, `torchvision`, `gymnasium`, `stable-baselines3`, `einops`, `scipy`, `numpy`, `matplotlib`, `flask`, `timm`, `pystac-client`, `planetary-computer`, `rasterio`, `Pillow`

### 1.4 Verify CUDA is Working

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA: True, Device: NVIDIA A100-SXM4-80GB
```

---

## 2. Download Training Data

You have **two modes**: curated (30 tiles, fast) or global (300+ tiles, maximum accuracy).

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

Real fire detections improve the fire model significantly. Get a free MAP_KEY from [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/area/), then:

```bash
# Set as environment variable
export FIRMS_MAP_KEY="your_key_here"

# OR pass directly
python datasets/download_real_data.py \
    --mode curated \
    --chips-per-tile 1000 \
    --parallel 8 \
    --firms-key your_key_here
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
| `datasets/real_tiles/manifest.json` | Train/test split index |
| `datasets/real_tiles/.srtm_cache/` | Cached SRTM elevation tiles |
| `datasets/real_tiles/.tile_discovery_cache.json` | Tile discovery cache (global mode) |

---

## 3. Verify Data Integrity

### Check Manifest

```bash
python -c "
import json
with open('datasets/real_tiles/manifest.json') as f:
    m = json.load(f)
print(f'Train chips: {len(m[\"train\"]):,}')
print(f'Test chips:  {len(m[\"test\"]):,}')
print(f'Total:       {len(m[\"train\"]) + len(m[\"test\"]):,}')
"
```

> [!IMPORTANT]
> You need at least **~500 train chips** for basic learning. Curated 30 tiles with 1000 chips/tile yields ~24,000 train chips. Global mode yields ~240,000+.

### Test-Load a Single Chip

```bash
python -c "
import numpy as np
data = np.load('datasets/real_tiles/chip_00N_050W_0000.npz')
print('Layers:', list(data.keys()))
print('Shapes:', {k: data[k].shape for k in data.keys()})
"
```

### Run the Pipeline Smoke Test

```bash
python test_real_pipeline.py
```

---

## 4. Training the Models

### 4.1 The Four Models

| Model | File | Input Channels | Loss | Task |
|---|---|---|---|---|
| **Fire** (`FireRiskNet`) | `models/fire_model.py` | Temporal (T=5 frames) | Focal BCE | Fire risk probability |
| **Forest** (`ForestLossNet`) | `models/forest_model.py` | Temporal (T=5 frames) | Dice+BCE | Deforestation detection |
| **Hydro** (`HydroRiskNet`) | `models/hydro_model.py` | Static (SRTM terrain) | Gradient MSE | Water pollution risk |
| **Soil** (`SoilRiskNet`) | `models/soil_model.py` | Temporal (T=5 frames) | Smooth MSE | Soil degradation risk |

### 4.2 Temporal Split Strategy

| Split | Spatial Tiles | Years | Purpose |
|---|---|---|---|
| **Train** | "train" tiles (80%) | 2001–2018 | Model learning |
| **Test** | "train" tiles (80%) | 2019–2020 | Monitoring / checkpointing |
| **Validate** | "test" tiles (20%) | 2021–2023 | True hold-out evaluation |

### 4.3 Train All 4 Models (Recommended Command)

```bash
python train_real_models.py \
    --model all \
    --epochs 60 \
    --accumulation-steps 4 \
    --amp \
    --early-stop-patience 10
```

**What this does:**
- Trains all 4 models sequentially: Fire → Forest → Hydro → Soil
- **Batch size**: Auto-selects 16 on CUDA (effective batch = 16 × 4 = 64 with gradient accumulation)
- **AMP**: Mixed precision for 2× speed + lower memory
- **LR schedule**: 10% warmup → cosine annealing (lr=3e-4 for fire/forest, 1e-3 for hydro/soil)
- **Deep supervision**: UNet++ auxiliary losses (weight=0.3)
- **Early stopping**: Halts if test loss doesn't improve for 10 epochs
- **Checkpointing**: Saves best model by test loss

### 4.4 Train a Single Model

```bash
# Train only the fire model
python train_real_models.py --model fire --epochs 80 --amp

# Train only forest
python train_real_models.py --model forest --epochs 60 --amp
```

### 4.5 Custom Data Directory

If you put data somewhere else (e.g. a fast NVMe mount):

```bash
python train_real_models.py \
    --model all \
    --epochs 60 \
    --amp \
    --tiles-dir /mnt/nvme/real_tiles \
    --weights-dir /mnt/nvme/weights
```

### 4.6 Expected Training Times (A100 80GB)

| Dataset Size | Epochs | Est. Time per Model | Total (all 4) |
|---|---|---|---|
| Curated (24k chips) | 60 | ~10–20 min | ~1–1.5 hours |
| Global (240k chips) | 60 | ~2–4 hours | ~8–16 hours |

### 4.7 Monitoring Training Progress

The script logs every 5 epochs:
```
Epoch   5/60: train=0.234567  test=0.198765  best=0.198765  lr=3.00e-04  time=12.3s  patience=0/10
Epoch  10/60: train=0.189012  test=0.167890  best=0.167890  lr=2.85e-04  time=11.8s  patience=0/10
```

**Key things to watch:**
- `train` and `test` loss should both decrease
- `patience` counter — if it reaches 10, training stops early
- `time` — should be 5–15s per epoch on A100 for curated

> [!TIP]
> Use `tmux` or `screen` to keep training running if you disconnect from SSH:
> ```bash
> tmux new -s training
> python train_real_models.py --model all --epochs 60 --amp
> # Ctrl+B then D to detach
> # tmux attach -t training to reattach
> ```

---

## 5. Evaluate & Validate

After training completes, run the comprehensive evaluation:

```bash
python evaluate_models.py \
    --tiles-dir datasets/real_tiles \
    --weights-dir weights
```

This computes per-model:
- **Pixel metrics**: MSE, MAE, Pearson correlation, AUROC, F1, precision, recall
- **Spatial metrics**: SSIM, gradient matching
- **Distribution**: KS test, skewness, kurtosis
- **Calibration**: Expected calibration error

Results are saved to `weights/evaluation_report.json`.

---

## 6. Save & Export Weights

### 6.1 Where Weights Are Saved

Training automatically saves to the `weights/` directory:

| File | Description |
|---|---|
| `weights/fire_model.pt` | Best fire model checkpoint |
| `weights/forest_model.pt` | Best forest model checkpoint |
| `weights/hydro_model.pt` | Best hydro model checkpoint |
| `weights/soil_model.pt` | Best soil model checkpoint |
| `weights/real_training_metrics.json` | Full training metrics (losses, timing, config) |

### 6.2 Back Up Weights

```bash
# Tar up all weights + metrics
tar -czf misdo_weights_$(date +%Y%m%d).tar.gz weights/

# Copy to persistent storage / local machine
scp misdo_weights_*.tar.gz user@your-machine:~/backups/
```

### 6.3 Download Weights to Local Machine

```bash
# From your local machine
scp -r user@a100-host:/path/to/MISDO/weights/ ./weights/
```

### 6.4 Load Trained Weights in Code

```python
import torch
from models.fire_model import FireRiskNet

model = FireRiskNet()
model.load_state_dict(
    torch.load("weights/fire_model.pt", map_location="cpu", weights_only=True)
)
model.eval()
```

### 6.5 Push Weights to Git (Optional)

> [!CAUTION]
> Model weights are ~20–50 MB each. Consider using Git LFS or storing them separately.

```bash
git lfs install
git lfs track "weights/*.pt"
git add weights/ .gitattributes
git commit -m "Add trained model weights"
git push
```

---

## 7. Quick Reference Card

### Full Pipeline (Copy-Paste)

```bash
# 1. Setup
git clone https://github.com/zubadestroyer1/MISDO.git && cd MISDO
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Download data (curated — use --mode global for full scale)
python datasets/download_real_data.py --mode curated --chips-per-tile 1000 --parallel 8

# 3. Train all 4 models
python train_real_models.py --model all --epochs 60 --amp --accumulation-steps 4

# 4. Evaluate
python evaluate_models.py --tiles-dir datasets/real_tiles --weights-dir weights

# 5. Back up weights
tar -czf misdo_weights_$(date +%Y%m%d).tar.gz weights/
```

### CLI Flags Reference

| Flag | Default | Description |
|---|---|---|
| `--model` | `all` | Which model: `fire`, `forest`, `hydro`, `soil`, `all` |
| `--epochs` | `60` | Max training epochs |
| `--accumulation-steps` | `4` | Gradient accumulation (effective batch = batch_size × this) |
| `--amp` | off | Enable mixed precision (always use on A100) |
| `--early-stop-patience` | `10` | Epochs without improvement before stopping |
| `--tiles-dir` | `datasets/real_tiles` | Path to downloaded data |
| `--weights-dir` | `weights` | Path to save model checkpoints |

### Troubleshooting

| Issue | Solution |
|---|---|
| `No real data found` | Run the download script first (Step 2) |
| `CUDA out of memory` | Reduce `--accumulation-steps` to 2, or edit batch_size in `train_real_models.py` line 246 |
| `treecover2000 missing` | The Hansen GCS server may be throttling — retry with fewer `--parallel` workers |
| `SRTM download failures` | Some polar tiles lack SRTM coverage — these fall back to proxy terrain automatically |
| `No VIIRS fire data` | You need a FIRMS MAP_KEY — it's free at https://firms.modaps.eosdis.nasa.gov/api/ |
| `Training loss not decreasing` | Verify data with `python test_real_pipeline.py`, check for NaN in chips |
