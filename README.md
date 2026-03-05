# MISDO — Multi-domain Intelligent Sustainable Deforestation Optimizer

> **Greenhacks (DOFE)** — AI-driven environmental risk analysis for sustainable forestry planning.

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

---

## Quickstart

### Requirements

```bash
pip install -r requirements.txt
```

### Train All Models

```bash
python train_models.py --model all --epochs 30
```

### Train a Single Model

```bash
python train_models.py --model fire --epochs 50 --samples 128
```

### Training Features

- **Data augmentation**: random flips, rotations, brightness jitter
- **Train/val split**: 80/20 with separate seed ranges
- **Best checkpoint**: saves model with lowest validation loss
- **Early stopping**: stops if no improvement for 10 epochs
- **AdamW optimizer** with cosine annealing and weight decay

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
│   ├── download_earth_data.py# Script for downloading Earth Engine datasets
│   ├── multi_source_datasets.py# Multi-source real dataset loader
│   ├── viirs_fire.py         # VIIRS synthetic data
│   ├── hansen_gfc.py         # Hansen GFC synthetic data
│   ├── srtm_hydro.py         # SRTM/HydroSHEDS synthetic data
│   └── smap_soil.py          # SMAP soil synthetic data
├── weights/
│   ├── fire_model.pt         # Trained fire model weights
│   ├── forest_model.pt       # Trained forest model weights
│   ├── hydro_model.pt        # Trained hydro model weights
│   └── soil_model.pt         # Trained soil model weights
├── docs/                     # Architecture documentation
├── static/                   # Web UI assets
├── aggregator.py             # Hybrid risk fusion module
├── perception.py             # Domain model orchestration
├── env.py                    # Gymnasium RL environment
├── impact.py                 # Cascading impact propagation
├── train.py                  # End-to-end PPO training
├── train_models.py           # Domain model training script
├── server.py                 # Flask web server
├── test_real_pipeline.py     # Pipeline integration tests
└── requirements.txt          # Python dependencies
```

---

## License

This project was developed for the Greenhacks hackathon (DOFE).
