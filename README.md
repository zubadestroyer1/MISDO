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
│  │ FireRisk │  │ Forest   │  │ HydroRisk│  │ SoilRisk │  Domain       │
│  │ Net      │  │ LossNet  │  │ Net      │  │ Net      │  Models       │
│  │ ~2.5M    │  │ ~3.2M    │  │ ~3.8M    │  │ ~2.1M    │               │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
│       │              │              │              │                    │
│       └──────┬───────┴──────┬───────┘              │                    │
│              │              │                      │                    │
│              ▼              ▼                      ▼                    │
│         ┌─────────────────────────────────────────────┐                 │
│         │  Conditioned Aggregator                     │  Fusion         │
│         │  Weighted Sum + Gaussian Smooth + Hard      │                 │
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

| Model | Architecture | Params | Input | Data Source | Loss |
|---|---|---|---|---|---|
| [FireRiskNet](docs/FireRiskNet.md) | ResNet-18 + U-Net | ~2.5M | 6ch (VIIRS I-bands + FRP) | VIIRS VNP14IMG | Focal BCE |
| [ForestLossNet](docs/ForestLossNet.md) | EfficientNet (MBConv) + PixelShuffle | ~3.2M | 5ch (treecover, lossyear, gain, red, NIR) | Hansen GFC | Dice + BCE |
| [HydroRiskNet](docs/HydroRiskNet.md) | FPN + Attention Gates | ~3.8M | 5ch (elevation, slope, aspect, flow_acc, flow_dir) | SRTM/HydroSHEDS | Gradient MSE |
| [SoilRiskNet](docs/SoilRiskNet.md) | ASPP + Dilated Conv | ~2.1M | 4ch (moisture, veg_water, temp, freeze_thaw) | SMAP L3 | Smooth MSE |

See the [docs/](docs/) directory for detailed architecture diagrams, mathematical formulations, training approaches, and design rationale for each model.

### Additional Components

- [Aggregator](docs/Aggregator.md) — Parameter-conditioned spatial risk fusion with Gaussian smoothing and hard constraints
- [Perception Module](docs/PerceptionModule.md) — Shared ConvNeXt backbone and domain-specific model orchestration
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
python train_models.py --model fire --epochs 50
```

### Run the Full Pipeline

```bash
python train.py
```

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
│   ├── fire_model.py         # FireRiskNet
│   ├── forest_model.py       # ForestLossNet
│   ├── hydro_model.py        # HydroRiskNet
│   └── soil_model.py         # SoilRiskNet
├── datasets/
│   ├── __init__.py           # Dataset registry
│   ├── viirs_fire.py         # VIIRS synthetic data
│   ├── hansen_gfc.py         # Hansen GFC synthetic data
│   ├── srtm_hydro.py         # SRTM/HydroSHEDS synthetic data
│   └── smap_soil.py          # SMAP soil synthetic data
├── weights/
│   ├── fire_model.pt         # Trained fire model weights
│   ├── forest_model.pt       # Trained forest model weights
│   ├── hydro_model.pt        # Trained hydro model weights
│   └── soil_model.pt         # Trained soil model weights
├── docs/
│   ├── FireRiskNet.md        # Fire model documentation
│   ├── ForestLossNet.md      # Forest model documentation
│   ├── HydroRiskNet.md       # Hydro model documentation
│   ├── SoilRiskNet.md        # Soil model documentation
│   ├── Aggregator.md         # Aggregator documentation
│   ├── PerceptionModule.md   # Perception module documentation
│   └── RLOptimizer.md        # RL optimizer documentation
├── static/                   # Web UI assets
├── data.py                   # Data ingestion module
├── perception.py             # Shared backbone + decoder heads
├── aggregator.py             # Risk fusion module
├── env.py                    # Gymnasium RL environment
├── train.py                  # End-to-end PPO training
├── train_models.py           # Domain model training script
├── server.py                 # Flask web server
├── test_real_pipeline.py     # Pipeline integration tests
└── requirements.txt          # Python dependencies
```

---

## License

This project was developed for the Greenhacks hackathon (DOFE).
