"""
MISDO Models Package — Domain-Specific Risk Networks
=====================================================
Each sub-module provides a dataset-specific neural network that maps
raw satellite channels to a per-pixel risk mask [B, 1, 256, 256].

Models:
    FireRiskNet    — VIIRS active-fire detection      (6 input channels)
    ForestLossNet  — Hansen deforestation detection    (5 input channels)
    HydroRiskNet   — SRTM/HydroSHEDS water-pollution  (5 input channels)
    SoilRiskNet    — SMAP soil degradation             (4 input channels)
"""

from __future__ import annotations

from typing import Dict, Type

import torch.nn as nn

from .fire_model import FireRiskNet
from .forest_model import ForestLossNet
from .hydro_model import HydroRiskNet
from .soil_model import SoilRiskNet

# Re-export legacy perception classes (from perception.py at project root)
# These are lazy-loaded to avoid circular imports since perception.py
# imports from models.fire_model etc.
def _get_perception_classes():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from perception import MISDOPerception, RealMISDOPerception
    return MISDOPerception, RealMISDOPerception

def __getattr__(name):
    if name in ("MISDOPerception", "RealMISDOPerception"):
        MISDOPerception, RealMISDOPerception = _get_perception_classes()
        globals()["MISDOPerception"] = MISDOPerception
        globals()["RealMISDOPerception"] = RealMISDOPerception
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "fire": FireRiskNet,
    "forest": ForestLossNet,
    "hydro": HydroRiskNet,
    "soil": SoilRiskNet,
}


def get_model(name: str) -> nn.Module:
    """Instantiate a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]()
