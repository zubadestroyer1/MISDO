# MISDO — Multi-domain Intelligent Sustainable Deforestation Optimizer

> AI-driven environmental **impact analysis** and decision-support system for sustainable forestry planning.

MISDO uses a **temporal counterfactual approach** to predict what happens to surrounding forest when an area is cleared. It fuses satellite data from 4 remote sensing domains through domain-specific neural networks, aggregates their outputs into a unified harm mask, and uses reinforcement learning to find optimal harvest sequences that minimise environmental damage.

---

## How It Works

Instead of predicting static "risk maps," each model answers:

> *"If I clear THIS area, how much does fire / cascade deforestation / erosion / soil degradation **increase** in the surrounding forest?"*

This is learned from **real before-and-after satellite data** (Hansen GFC deforestation history + VIIRS fire detections), using temporal counterfactual targets with control-pixel baseline subtraction to isolate causal impact from background trends.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          MISDO Pipeline                                │
│                                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │  VIIRS   │  │  Hansen  │  │  SRTM/   │  │  SMAP    │  Satellite    │
│  │  Fire    │  │  GFC     │  │  Hydro   │  │  Soil    │  Data         │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
│       │              │              │              │                    │
│       ▼              ▼              ▼              ▼                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ FireImpac│  │ Forest   │  │ HydroImp │  │ SoilImpa │  ConvNeXt-V2  │
│  │ tNet     │  │ ImpactNet│  │ actNet   │  │ ctNet    │  + UNet++     │
│  │ ~40M     │  │ ~40M     │  │ ~40M     │  │ ~40M     │  + Dilated    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    Context    │
│       │              │              │              │                    │
│       └──────┬───────┴──────┬───────┘              │                    │
│              │              │                      │                    │
│              ▼              ▼                      ▼                    │
│         ┌─────────────────────────────────────────────┐                 │
│         │  Conditioned Aggregator                     │  Hybrid         │
│         │  Deterministic Weighted Sum + Learned       │  Fusion         │
│         │  Correction + Gaussian Smooth + Hard        │                 │
│         │  Constraints (slope, river)                 │                 │
│         └──────────────────┬──────────────────────────┘                 │
│                            │                                           │
│                            ▼                                           │
│                  ┌──────────────────┐                                   │
│                  │  PPO RL Agent    │  Sequential                       │
│                  │  (Harvest Plan)  │  Optimizer                        │
│                  └──────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Models

All models use a **ConvNeXt-V2 Base** encoder + **UNet++ decoder** + **DilatedContextModule** (ASPP) at bottleneck for long-range impact. Output uses ReLU + clamp(0,1) instead of sigmoid.

| Model | Params | Input | Target | Loss |
|---|---|---|---|---|
| [FireRiskNet](docs/FireRiskNet.md) | ~40M | 7ch (VIIRS + defo mask) | Fire increase near clearing | Edge-Weighted MSE |
| [ForestLossNet](docs/ForestLossNet.md) | ~40M | 6ch (Hansen + defo mask) | Cascade deforestation | Edge-Weighted MSE |
| [HydroRiskNet](docs/HydroRiskNet.md) | ~40M | 6ch (SRTM + defo mask) | Erosion increase downstream | Edge-Weighted MSE |
| [SoilRiskNet](docs/SoilRiskNet.md) | ~40M | 5ch (soil + defo mask) | Soil degradation increase | Edge-Weighted MSE |

### Key Architecture Features

- **ConvNeXt-V2 encoder** with Global Response Normalization (GRN) and stochastic depth
- **UNet++ decoder** with nested dense skip connections and deep supervision
- **DilatedContextModule** (ASPP-style) at bottleneck — multi-rate dilated convolutions for 1–5 km impact propagation
- **Multi-head temporal attention** for fusing multi-timestep inputs
- **Deforestation mask input channel** — model knows WHERE clearing happened
- **ReLU + clamp output** for sharper gradients on near-zero impact deltas

### Key Training Features

- **Sliding temporal windows** — ~100 samples per chip (vs 1 with fixed windows)
- **Control-pixel baseline subtraction** — isolates causal signal from background trends
- **LR**: warmup → cosine annealing (3e-4 for all models)
- **Loss**: CounterfactualDeltaLoss wrapping Edge-Weighted MSE (3× upweight near deforestation edges)
- **Deep supervision**: UNet++ auxiliary losses (weight=0.3)
- **Radiometric jitter**: mild brightness/contrast perturbation (p=0.5) to reduce overfitting to radiometric noise
- **Early stopping**: Halts if test loss doesn't improve for 10 epochs

See the [docs/](docs/) directory for detailed architecture documentation.

### Additional Components

- [Aggregator](docs/Aggregator.md) — Hybrid deterministic + learnable risk fusion with hard constraints
- [Perception Module](docs/PerceptionModule.md) — Domain model orchestration with cross-domain fusion
- [RL Optimizer](docs/RLOptimizer.md) — PPO-based sequential deforestation planning
- [Training Runbook](docs/TrainingRunbook.md) — Step-by-step A100 cluster training guide

---

## Enterprise Features

### Uncertainty Quantification

MC Dropout inference provides per-pixel confidence intervals on all impact predictions:
- Mean prediction, standard deviation, 90% credible intervals, and predictive entropy
- Encoder features computed once, decoder sampled N times (~20× faster than naive MC)

```python
from uncertainty import enable_mc_dropout, predict_with_uncertainty
enable_mc_dropout(model, p=0.1)
result = predict_with_uncertainty(model, x, n_samples=20)
# result['mean'], result['std'], result['confidence_lower'], result['confidence_upper']
```

### Explainability (GradCAM)

Visual attribution maps showing *why* each pixel has high or low predicted impact:

```python
from explainability import generate_attribution_report
report = generate_attribution_report(model, obs, channel_names=["I1", "I2", "I3", "I4", "I5", "FRP", "defo_mask"])
# report['attribution_map'], report['channel_importance']
```

### Spatial Cross-Validation

Prevents data leakage through spatial autocorrelation and temporal correlation:
- **Spatial blocking**: 1° blocks (~111 km) ensuring adjacent tiles never split across folds
- **Temporal holdout**: train events ≤2016 (impact ≤2018), test 2017–2018, validate 2019+

```python
from validation import SpatialBlockCV, TemporalHoldout
cv = SpatialBlockCV(n_folds=5, block_size_deg=1.0)
holdout = TemporalHoldout(train_end=18, val_end=20, test_end=23)
```

---

## Quickstart

### Requirements

```bash
pip install -r requirements.txt
```

### Train All Models (Synthetic Data)

```bash
python train_models.py --model all --epochs 30
```

### Train on Real Satellite Data

```bash
# 1. Download curated data (30 tiles, ~1–3 hours)
python datasets/download_real_data.py --mode curated --chips-per-tile 1000 --parallel 8

# 2. (Optional) Add MSI/SMAP augmentation for Hydro/Soil targets
python datasets/download_msi_smap.py

# 3. (Optional) Add VIIRS fire data via bulk archive (no rate limit)
python datasets/download_real_data.py --mode curated --parallel 8 \
    --viirs-archive /path/to/firms_csvs/

# 4. Train all 4 models with AMP + gradient accumulation
python train_real_models.py --model all --epochs 60 --amp --accumulation-steps 4

# 5. Evaluate trained models
python evaluate_models.py --tiles-dir datasets/real_tiles --weights-dir weights
```

> **📖 For full A100 cluster setup, global-scale data download, and production training instructions, see the [Training Runbook](docs/TrainingRunbook.md).**

### Start the Web Server

```bash
python server.py
```

---

## Project Structure

```
.
├── models/
│   ├── base_model.py         # Shared ConvNeXt-V2 + UNet++ base
│   ├── backbone.py           # ConvNeXt-V2 encoder
│   ├── decoders.py           # UNet++ decoder + DilatedContextModule
│   ├── fire_model.py         # FireRiskNet (7ch → impact)
│   ├── forest_model.py       # ForestLossNet (6ch → cascade)
│   ├── hydro_model.py        # HydroRiskNet (6ch → erosion)
│   ├── soil_model.py         # SoilRiskNet (5ch → degradation)
│   ├── temporal.py           # Multi-head temporal attention
│   └── fusion.py             # Cross-domain feature fusion
├── datasets/
│   ├── real_datasets.py      # Real counterfactual datasets (sliding windows)
│   ├── download_real_data.py # Satellite tile downloader (curated + global + VIIRS bulk)
│   ├── download_msi_smap.py  # MSI/SMAP augmentation downloader
│   ├── viirs_fire.py         # Synthetic fire impact data
│   ├── hansen_gfc.py         # Synthetic cascade deforestation data
│   ├── srtm_hydro.py         # Synthetic erosion impact data
│   └── smap_soil.py          # Synthetic soil degradation data
├── weights/                  # Trained model checkpoints
├── docs/                     # Architecture documentation
│   └── TrainingRunbook.md    # A100 training guide
├── static/                   # Web UI assets
├── losses.py                 # Loss functions (EdgeWeightedMSE, SmoothMSE, etc.)
├── uncertainty.py            # MC Dropout uncertainty quantification
├── explainability.py         # GradCAM attribution maps
├── validation.py             # Spatial cross-validation & temporal holdout
├── aggregator.py             # Hybrid impact fusion module
├── env.py                    # Gymnasium RL environment
├── impact.py                 # Cascading impact propagation (D8 routing)
├── evaluate_models.py        # Post-training model evaluation
├── train_models.py           # Domain model training (synthetic)
├── train_real_models.py      # Domain model training (real satellite)
├── server.py                 # Flask web server
└── requirements.txt          # Python dependencies
```

---

## License

Proprietary. All rights reserved.
