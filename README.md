# MISDO — Multi-domain Intelligent Sustainable Deforestation Optimizer

> AI-driven environmental risk analysis and decision-support system for sustainable forestry planning.

MISDO fuses satellite data from 4 remote sensing domains (VIIRS fire, Hansen forest change, SRTM hydrology, SMAP soil moisture) through domain-specific neural networks, aggregates their outputs into a unified harm mask, and uses reinforcement learning to find optimal harvest sequences that minimise environmental damage.

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
│  │ FireRisk │  │ Forest   │  │ HydroRisk│  │ SoilRisk │  ConvNeXt-V2  │
│  │ Net      │  │ LossNet  │  │ Net      │  │ Net      │  + UNet++     │
│  │ ~34M     │  │ ~34M     │  │ ~34M     │  │ ~34M     │               │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
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

All models use a **ConvNeXt-V2 Base** encoder with **UNet++** decoder (nested dense skip connections).

| Model | Architecture | Params | Input | Data Source | Loss |
|---|---|---|---|---|---|
| [FireRiskNet](docs/FireRiskNet.md) | ConvNeXt-V2 + UNet++ | ~34M | 6ch (VIIRS I-bands + FRP) | VIIRS VNP14IMG | Focal BCE |
| [ForestLossNet](docs/ForestLossNet.md) | ConvNeXt-V2 + UNet++ | ~34M | 5ch (treecover, lossyear, gain, red, NIR) | Hansen GFC | Dice + BCE |
| [HydroRiskNet](docs/HydroRiskNet.md) | ConvNeXt-V2 + UNet++ | ~34M | 5ch (elevation, slope, aspect, flow_acc, flow_dir) | SRTM/HydroSHEDS | Gradient MSE |
| [SoilRiskNet](docs/SoilRiskNet.md) | ConvNeXt-V2 + UNet++ | ~34M | 4ch (moisture, veg_water, temp, freeze_thaw) | SMAP L3 | Smooth MSE |

### Key Architecture Features

- **ConvNeXt-V2 encoder** with Global Response Normalization (GRN) and stochastic depth
- **UNet++ decoder** with nested dense skip connections and deep supervision
- **Multi-head temporal attention** for fusing multi-timestep inputs
- **Cross-domain feature fusion** exchanging information between domain bottlenecks
- **Hybrid aggregator** — deterministic weighted sum + learnable cross-domain correction

See the [docs/](docs/) directory for detailed architecture documentation.

### Additional Components

- [Aggregator](docs/Aggregator.md) — Hybrid deterministic + learnable risk fusion with hard constraints
- [Perception Module](docs/PerceptionModule.md) — Domain model orchestration with cross-domain fusion
- [RL Optimizer](docs/RLOptimizer.md) — PPO-based sequential deforestation planning
- [Training Runbook](docs/TrainingRunbook.md) — Step-by-step A100 cluster training guide

---

## Enterprise Features

### Uncertainty Quantification

MC Dropout inference provides per-pixel confidence intervals on all risk predictions:
- Mean prediction, standard deviation, 90% credible intervals, and predictive entropy
- Encoder features computed once, decoder sampled N times (~20× faster than naive MC)
- Essential for decision-makers who need to know prediction reliability

```python
from uncertainty import enable_mc_dropout, predict_with_uncertainty
enable_mc_dropout(model, p=0.1)
result = predict_with_uncertainty(model, x, n_samples=20)
# result['mean'], result['std'], result['confidence_lower'], result['confidence_upper']
```

### Explainability (GradCAM)

Visual attribution maps showing *why* each pixel is predicted as high or low risk:
- GradCAM on encoder bottleneck for region-level spatial explanations
- Per-channel importance breakdown (e.g., "73% attributed to deforestation edges")
- Audit-ready: provides evidence trail for regulatory compliance

```python
from explainability import generate_attribution_report
report = generate_attribution_report(model, obs, channel_names=["I1", "I2", "I3", "I4", "I5", "FRP"])
# report['attribution_map'], report['channel_importance']
```

### Spatial Cross-Validation

Prevents data leakage through spatial autocorrelation and temporal correlation:
- **Spatial blocking**: 1° blocks (~111 km) ensuring adjacent tiles never split across folds
- **Temporal holdout**: train ≤2018, validation 2019-2020, test 2021-2023
- Follows best practices from Ploton et al. (2020, Nature Communications)

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

# 2. Train all 4 models with AMP + gradient accumulation
python train_real_models.py --model all --epochs 60 --amp --accumulation-steps 4

# 3. Evaluate trained models
python evaluate_models.py --tiles-dir datasets/real_tiles --weights-dir weights
```

> **📖 For full A100 cluster setup, global-scale data download, and production training instructions, see the [Training Runbook](docs/TrainingRunbook.md).**

### Training Features

- **Consolidated NaN-safe loss functions** with prediction clamping for AMP
- **UNet++ deep supervision** auxiliary losses for improved convergence
- **LR warmup + cosine annealing** schedule
- **Gradient accumulation** for larger effective batch sizes (up to 64 on A100)
- **Automatic Mixed Precision (AMP)** for 2× throughput on CUDA
- **MPS/CUDA/CPU** automatic device detection with safe DataLoader settings
- **Data augmentation**: random flips, rotations, brightness jitter
- **Best checkpoint saving** on test loss with early stopping
- **Temporal split**: train ≤2018, test 2019–2020, validate 2021–2023

### Start the Web Server

```bash
python server.py
```

---

## Project Structure

```
.
├── models/
│   ├── __init__.py           # Model registry
│   ├── base_model.py         # Shared base model architecture
│   ├── backbone.py           # ConvNeXt-V2 encoder (shared)
│   ├── decoders.py           # UNet++ decoder (shared)
│   ├── fire_model.py         # FireRiskNet
│   ├── forest_model.py       # ForestLossNet
│   ├── hydro_model.py        # HydroRiskNet
│   ├── soil_model.py         # SoilRiskNet
│   ├── temporal.py           # Multi-head temporal attention
│   └── fusion.py             # Cross-domain feature fusion
├── datasets/
│   ├── __init__.py           # Dataset registry
│   ├── real_datasets.py      # Real Hansen GFC multi-temporal datasets
│   ├── download_real_data.py # Script for downloading real satellite tiles
│   ├── viirs_fire.py         # VIIRS synthetic data
│   ├── hansen_gfc.py         # Hansen GFC synthetic data
│   ├── srtm_hydro.py         # SRTM/HydroSHEDS synthetic data
│   └── smap_soil.py          # SMAP soil synthetic data
├── weights/                  # Trained model checkpoints
├── docs/                     # Architecture documentation
│   ├── TrainingRunbook.md    # A100 training guide (setup → deploy)
│   └── ...                   # Model & component docs
├── static/                   # Web UI assets
├── losses.py                 # Consolidated production-grade loss functions
├── uncertainty.py            # MC Dropout uncertainty quantification
├── explainability.py         # GradCAM attribution maps
├── validation.py             # Spatial cross-validation & temporal holdout
├── aggregator.py             # Hybrid risk fusion module
├── perception.py             # Domain model orchestration
├── env.py                    # Gymnasium RL environment
├── impact.py                 # Cascading impact propagation (D8 routing)
├── evaluate_models.py        # Post-training model evaluation
├── train.py                  # End-to-end PPO training
├── train_models.py           # Domain model training (synthetic data)
├── train_real_models.py      # Domain model training (real satellite data)
├── server.py                 # Flask web server
├── test_real_pipeline.py     # Pipeline integration tests
└── requirements.txt          # Python dependencies
```

---

## License

Proprietary. All rights reserved.
