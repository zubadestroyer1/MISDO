"""
MISDO Visualization Dashboard — Flask Backend
===============================================
Serves model outputs as base64-encoded heatmap PNGs via a REST API.

Endpoints:
    GET  /                  → Dashboard HTML
    GET  /api/agent-masks   → 4 individual risk heatmaps
    POST /api/aggregate     → Fused harm mask with custom weights
    GET  /api/env-state     → Current RL environment state
    POST /api/env-step      → Execute one harvest step
    POST /api/env-reset     → Reset RL environment
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, request, send_from_directory
from torch import Tensor

from data import MockEODataset
from models import MISDOPerception, RealMISDOPerception
from aggregator import ConditionedAggregator
from env import DeforestationEnv, SPATIAL
from impact import ImpactPropagation

# ═══════════════════════════════════════════════════════════════════════════
# App setup
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__, static_folder="static", static_url_path="/static")

# Global state (populated on startup)
_perception: Optional[nn.Module] = None  # MISDOPerception or RealMISDOPerception
_aggregator: Optional[ConditionedAggregator] = None
_obs_tensor: Optional[Tensor] = None       # [1, 20, 256, 256] (legacy only)
_domain_tensors: Optional[Dict[str, Tensor]] = None  # real domain inputs
_agent_masks: Optional[Tensor] = None      # [1, 4, 256, 256]
_env: Optional[DeforestationEnv] = None
_impact: Optional[ImpactPropagation] = None
_device: torch.device = torch.device("cpu")
_using_real_models: bool = False

# Hard constraint tensors extracted from real domain data
_slope_tensor: Optional[Tensor] = None       # [1, 1, 256, 256]
_river_prox_tensor: Optional[Tensor] = None  # [1, 1, 256, 256]
_flow_dir_tensor: Optional[Tensor] = None    # [1, 1, 256, 256] — D8 flow direction
_forest_mask_tensor: Optional[Tensor] = None # [1, 1, 256, 256] — 1=forested, 0=non-forest

# Head metadata — updated for real models
HEAD_NAMES: List[str] = ["Fire Risk", "Forest Loss", "Water Pollution", "Soil Degradation"]
HEAD_CMAPS: List[str] = ["inferno", "YlOrRd", "YlGnBu", "Oranges"]
HEAD_DESCRIPTIONS: List[str] = [
    "VIIRS active fire detection risk",
    "Hansen deforestation / forest loss risk",
    "SRTM/HydroSHEDS erosion & runoff risk",
    "SMAP soil degradation & drought risk",
]


def _extract_hard_constraints(domain_tensors: Dict[str, Tensor]) -> None:
    """Extract slope, river proximity, and forest presence from real domain data.

    SRTM hydro channels: 0=elevation, 1=slope, 2=aspect, 3=flow_accumulation, 4=flow_direction
    Hansen GFC channels: 0=treecover2000, 1=lossyear, 2=gain, 3=red, 4=NIR
    """
    global _slope_tensor, _river_prox_tensor, _flow_dir_tensor, _forest_mask_tensor

    hydro = domain_tensors["hydro"]  # [1, 5, 256, 256]
    _slope_tensor = hydro[:, 1:2, :, :].clone()           # [1, 1, 256, 256]
    _river_prox_tensor = hydro[:, 3:4, :, :].clone()      # [1, 1, 256, 256]
    _flow_dir_tensor = hydro[:, 4:5, :, :].clone()        # [1, 1, 256, 256] — D8 flow direction

    # Forest presence from Hansen treecover2000 (channel 0)
    # Pixels with >20% canopy cover are considered forest
    forest = domain_tensors["forest"]  # [1, 5, 256, 256]
    _forest_mask_tensor = (forest[:, 0:1, :, :] > 0.2).float()  # [1, 1, 256, 256]

    forest_pct = _forest_mask_tensor.mean().item() * 100
    print(f"  [Constraints] slope range: [{_slope_tensor.min():.3f}, {_slope_tensor.max():.3f}]")
    print(f"  [Constraints] river_prox range: [{_river_prox_tensor.min():.3f}, {_river_prox_tensor.max():.3f}]")
    print(f"  [Constraints] flow_dir range: [{_flow_dir_tensor.min():.3f}, {_flow_dir_tensor.max():.3f}]")
    print(f"  [Constraints] forest cover: {forest_pct:.1f}%")


def _init_models() -> None:
    """Run perception pipeline once on startup and cache results.

    Tries to load trained domain models first; falls back to legacy
    mock pipeline if weights are not found.
    """
    global _perception, _aggregator, _obs_tensor, _agent_masks, _env
    global _domain_tensors, _using_real_models

    import os
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    has_weights = all(
        os.path.exists(os.path.join(weights_dir, f"{n}_model.pt"))
        for n in ["fire", "forest", "hydro", "soil"]
    )

    if has_weights:
        _init_real_models(weights_dir)
    else:
        print("[INIT] No trained weights found — using legacy mock pipeline")
        _init_legacy_models()


def _init_real_models(weights_dir: str) -> None:
    """Load trained domain-specific models with synthetic input data."""
    global _perception, _aggregator, _obs_tensor, _agent_masks, _env
    global _domain_tensors, _using_real_models

    _using_real_models = True

    print("[INIT] Loading domain-specific synthetic data ...")
    from datasets.viirs_fire import VIIRSFireDataset
    from datasets.hansen_gfc import HansenGFCDataset
    from datasets.srtm_hydro import SRTMHydroDataset
    from datasets.smap_soil import SMAPSoilDataset

    fire_obs, _ = VIIRSFireDataset(1, 256, 0)[0]
    forest_obs, _ = HansenGFCDataset(1, 256, 0)[0]
    hydro_obs, _ = SRTMHydroDataset(1, 256, 0)[0]
    soil_obs, _ = SMAPSoilDataset(1, 256, 0)[0]

    _domain_tensors = {
        "fire": fire_obs.unsqueeze(0).to(_device),
        "forest": forest_obs.unsqueeze(0).to(_device),
        "hydro": hydro_obs.unsqueeze(0).to(_device),
        "soil": soil_obs.unsqueeze(0).to(_device),
    }

    # Extract hard constraint data from real SRTM domain
    _extract_hard_constraints(_domain_tensors)

    print("[INIT] Loading trained domain models ...")
    _perception = RealMISDOPerception(weights_dir=weights_dir).to(_device)
    _perception.eval()
    with torch.no_grad():
        _agent_masks = _perception(_domain_tensors)

    print("[INIT] Initializing Aggregator ...")
    _aggregator = ConditionedAggregator().to(_device)
    _aggregator.eval()

    default_weights = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=_device)
    with torch.no_grad():
        harm = _aggregator(
            _agent_masks, default_weights,
            slope=_slope_tensor, river_proximity=_river_prox_tensor,
            forest_mask=_forest_mask_tensor,
        )
    harm_np = harm.squeeze().cpu().numpy()

    print("[INIT] Initializing Impact Propagation ...")
    slope_np = _slope_tensor.squeeze().cpu().numpy() if _slope_tensor is not None else np.zeros((256, 256))
    flow_np = _river_prox_tensor.squeeze().cpu().numpy() if _river_prox_tensor is not None else np.zeros((256, 256))
    flow_dir_np = _flow_dir_tensor.squeeze().cpu().numpy() if _flow_dir_tensor is not None else None
    _impact = ImpactPropagation(flow_accumulation=flow_np, slope=slope_np, flow_direction=flow_dir_np)

    print("[INIT] Initializing RL Environment (with ImpactPropagation) ...")
    _env = DeforestationEnv(
        harm_mask=harm_np,
        flow_accumulation=flow_np,
        slope=slope_np,
        flow_direction=flow_dir_np,
    )
    _env.reset(seed=42)
    print("[INIT] Ready (REAL models).\n")


def _init_legacy_models() -> None:
    """Original mock-data pipeline."""
    global _perception, _aggregator, _obs_tensor, _agent_masks, _env

    print("[INIT] Loading synthetic EO data ...")
    dataset = MockEODataset(num_samples=1, seed=0)
    obs, _ = dataset[0]
    _obs_tensor = obs.unsqueeze(0).to(_device)

    print("[INIT] Running Perception backbone ...")
    _perception = MISDOPerception(in_channels=20).to(_device)
    _perception.eval()
    with torch.no_grad():
        _agent_masks = _perception(_obs_tensor)

    print("[INIT] Initializing Aggregator ...")
    _aggregator = ConditionedAggregator().to(_device)
    _aggregator.eval()

    default_weights = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=_device)
    with torch.no_grad():
        # Legacy: extract slope/river from mock 20-channel tensor
        slope = _obs_tensor[:, 14:15, :, :]       # SRTM Slope channel
        river = _obs_tensor[:, 18:19, :, :]        # Proximity — Distance to River
        harm = _aggregator(_agent_masks, default_weights, slope=slope, river_proximity=river)
    harm_np = harm.squeeze().cpu().numpy()

    # Extract terrain for ImpactPropagation
    slope_np = slope.squeeze().cpu().numpy() if slope is not None else np.zeros((256, 256))
    flow_np = _obs_tensor[:, 16:17, :, :].squeeze().cpu().numpy() if _obs_tensor is not None else np.zeros((256, 256))

    print("[INIT] Initializing RL Environment (with ImpactPropagation) ...")
    _env = DeforestationEnv(
        harm_mask=harm_np,
        flow_accumulation=flow_np,
        slope=slope_np,
    )
    _env.reset(seed=42)
    print("[INIT] Ready (legacy models).\n")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _array_to_base64(
    arr: np.ndarray,
    cmap: str = "inferno",
    vmin: float = 0.0,
    vmax: float = 1.0,
    title: str = "",
    cbar_label: str = "Risk",
) -> str:
    """Render a 2D array as a colormapped PNG and return base64 string.

    Includes axis labels (pixel → km at ~30 m resolution), a labelled
    colorbar explaining the colour scale, and readable white text.
    """
    PIXEL_M = 30  # ~30 m/pixel (Landsat/SRTM resolution)
    H, W = arr.shape

    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5), dpi=110)
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")

    # Axis ticks in km
    km_ticks_x = np.linspace(0, W - 1, 5)
    km_labels_x = [f"{int(t * PIXEL_M / 1000)}" for t in km_ticks_x]
    km_ticks_y = np.linspace(0, H - 1, 5)
    km_labels_y = [f"{int(t * PIXEL_M / 1000)}" for t in km_ticks_y]

    ax.set_xticks(km_ticks_x)
    ax.set_xticklabels(km_labels_x, fontsize=8, color="#cccccc")
    ax.set_yticks(km_ticks_y)
    ax.set_yticklabels(km_labels_y, fontsize=8, color="#cccccc")
    ax.set_xlabel("km", fontsize=9, color="#aaaaaa", labelpad=4)
    ax.set_ylabel("km", fontsize=9, color="#aaaaaa", labelpad=4)
    ax.tick_params(colors="#666666", length=3, width=0.5)

    if title:
        ax.set_title(title, fontsize=10, color="white", pad=6, fontweight="bold")

    # Colorbar with label
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label(cbar_label, fontsize=9, color="#cccccc", labelpad=6)
    cbar.ax.tick_params(labelsize=8, colors="#cccccc", length=3, width=0.5)
    cbar.outline.set_edgecolor("#444444")

    fig.tight_layout(pad=1.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#0f0f1a")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _env_state_images() -> Dict[str, Any]:
    """Capture current RL env state as base64 images + stats."""
    assert _env is not None
    obs = _env._get_obs()  # [5, 256, 256]

    harm_img = _array_to_base64(obs[0], cmap="inferno", title="Dynamic Harm Mask", cbar_label="Harm (0=safe, 1=danger)")
    forest_img = _array_to_base64(obs[1], cmap="Greens", vmin=0, vmax=1, title="Forest State", cbar_label="Cover (1=intact, 0=cleared)")
    infra_img = _array_to_base64(obs[2], cmap="Oranges", vmin=0, vmax=1, title="Infrastructure", cbar_label="Presence (1=road/cleared)")
    contagion_img = _array_to_base64(obs[3], cmap="Reds", vmin=0, vmax=1, title="Contagion Risk", cbar_label="Risk (0=none, 1=high)")
    pollution_img = _array_to_base64(obs[4], cmap="YlGnBu", vmin=0, vmax=1, title="Water Pollution", cbar_label="Pollution (0=clean, 1=severe)")

    forest_pct = float(obs[1].sum()) / (SPATIAL * SPATIAL) * 100
    infra_pct = float(obs[2].sum()) / (SPATIAL * SPATIAL) * 100

    return {
        "harm_mask": harm_img,
        "forest_state": forest_img,
        "infrastructure": infra_img,
        "contagion_risk": contagion_img,
        "pollution_risk": pollution_img,
        "stats": {
            "forest_remaining_pct": round(forest_pct, 2),
            "infrastructure_pct": round(infra_pct, 2),
            "harvest_count": _env._harvest_count,
            "contagion_mean": round(float(obs[3].mean()), 4),
            "pollution_mean": round(float(obs[4].mean()), 4),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/agent-masks", methods=["GET"])
def get_agent_masks():
    """Return 4 individual agent risk heatmaps."""
    assert _agent_masks is not None
    masks_np = _agent_masks.squeeze(0).cpu().numpy()  # [4, 256, 256]

    # Colorbar labels per domain
    HEAD_CBAR_LABELS: List[str] = [
        "Fire probability",
        "Loss probability",
        "Runoff risk",
        "Degradation risk",
    ]

    result: List[Dict[str, Any]] = []
    for i, (name, cmap, desc, cbar_lbl) in enumerate(
        zip(HEAD_NAMES, HEAD_CMAPS, HEAD_DESCRIPTIONS, HEAD_CBAR_LABELS)
    ):
        arr = masks_np[i]
        result.append({
            "name": name,
            "description": desc,
            "image": _array_to_base64(arr, cmap=cmap, title=name, cbar_label=cbar_lbl),
            "stats": {
                "min": round(float(arr.min()), 4),
                "max": round(float(arr.max()), 4),
                "mean": round(float(arr.mean()), 4),
            },
        })

    # Reference image: raw forest canopy cover (no model applied)
    ref_image = None
    if _using_real_models and _forest_mask_tensor is not None:
        ref_arr = _forest_mask_tensor.squeeze().cpu().numpy()
        ref_image = _array_to_base64(
            ref_arr, cmap="Greens", vmin=0, vmax=1,
            title="Reference: Forest Canopy",
            cbar_label="Canopy cover (1=forest)",
        )

    return jsonify({"masks": result, "reference_image": ref_image})


@app.route("/api/aggregate", methods=["POST"])
def aggregate():
    """Run aggregator with user-provided weights and return fused harm mask."""
    assert _agent_masks is not None and _aggregator is not None

    data = request.get_json(force=True)
    weights_list: List[float] = data.get("weights", [0.5, 0.5, 0.5, 0.5])

    weights = torch.tensor([weights_list], dtype=torch.float32, device=_device)

    with torch.no_grad():
        if _using_real_models:
            harm: Tensor = _aggregator(
                _agent_masks, weights,
                slope=_slope_tensor, river_proximity=_river_prox_tensor,
                forest_mask=_forest_mask_tensor,
            )
        else:
            slope = _obs_tensor[:, 14:15, :, :] if _obs_tensor is not None else None
            river = _obs_tensor[:, 18:19, :, :] if _obs_tensor is not None else None
            harm = _aggregator(_agent_masks, weights, slope=slope, river_proximity=river)

    harm_np = harm.squeeze().cpu().numpy()  # [256, 256]

    # Compute safety mask and recommended harvest zones
    with torch.no_grad():
        safety_data = _aggregator.compute_safety_mask(
            harm, forest_mask=_forest_mask_tensor,
            safety_threshold=0.3,
        )
    safety_np = safety_data['safety_mask'].squeeze().cpu().numpy()
    recommended_np = safety_data['recommended'].squeeze().cpu().numpy()

    # ── Issue 2 fix: FULLY reset the RL environment with new harm mask ──
    assert _env is not None
    _env._base_harm_mask = harm_np.copy()
    _env.reset(seed=42)  # Full reset: forest, infra, impact layers

    no_go_pct = float((harm_np >= 0.999).sum()) / (SPATIAL * SPATIAL) * 100
    rec_pct = float(recommended_np.sum()) / (SPATIAL * SPATIAL) * 100

    return jsonify({
        "harm_image": _array_to_base64(harm_np, cmap="inferno", title="Fused Harm Mask", cbar_label="Harm (0=safe, 1=danger)"),
        "safety_image": _array_to_base64(safety_np, cmap="RdYlGn", title="Safety Gradient", cbar_label="Safety (1=safest, 0=danger)"),
        "recommended_image": _array_to_base64(recommended_np, cmap="Greens", vmin=0, vmax=1, title="Recommended Harvest", cbar_label="Harvestable (1=yes)"),
        "stats": {
            "min": round(float(harm_np.min()), 4),
            "max": round(float(harm_np.max()), 4),
            "mean": round(float(harm_np.mean()), 4),
            "no_go_pct": round(no_go_pct, 2),
            "recommended_pct": round(rec_pct, 2),
        },
        # Legacy key for backward compat with existing frontend
        "image": _array_to_base64(harm_np, cmap="inferno", title="Fused Harm Mask", cbar_label="Harm (0=safe, 1=danger)"),
    })


@app.route("/api/env-state", methods=["GET"])
def get_env_state():
    """Return current RL environment state."""
    return jsonify(_env_state_images())


@app.route("/api/env-step", methods=["POST"])
def env_step():
    """Execute one harvest step."""
    assert _env is not None
    data = request.get_json(force=True)
    row: int = int(data.get("row", 0))
    col: int = int(data.get("col", 0))

    action = row * SPATIAL + col
    _, reward, terminated, _, info = _env.step(action)

    state = _env_state_images()
    state["step_result"] = {
        "reward": round(float(reward), 2),
        "valid": info.get("valid", False),
        "terminated": terminated,
        "row": row,
        "col": col,
    }

    # Use impact scores computed inside env.step() (avoids redundant recalculation)
    if info.get('valid', False) and 'impact_scores' in info:
        state['impact'] = info['impact_scores']

    return jsonify(state)


@app.route("/api/env-reset", methods=["POST"])
def env_reset():
    """Reset the RL environment."""
    assert _env is not None
    _env.reset(seed=42)
    state = _env_state_images()
    state["message"] = "Environment reset successfully"
    return jsonify(state)


@app.route("/api/change-location", methods=["POST"])
def change_location():
    """Switch to a new region by regenerating data with a different seed.

    This simulates analyzing a different forest region by using the
    domain-specific data generators with the location's seed.
    """
    global _obs_tensor, _domain_tensors, _agent_masks, _env

    data = request.get_json(force=True)
    seed: int = int(data.get("seed", 42))
    location: str = data.get("location", "amazon")

    print(f"[LOCATION] Switching to: {location} (seed={seed})")

    try:
        if _using_real_models and _perception is not None:
            # Re-generate domain data with new seed
            from datasets.viirs_fire import VIIRSFireDataset
            from datasets.hansen_gfc import HansenGFCDataset
            from datasets.srtm_hydro import SRTMHydroDataset
            from datasets.smap_soil import SMAPSoilDataset

            fire_obs, _ = VIIRSFireDataset(1, 256, seed)[0]
            forest_obs, _ = HansenGFCDataset(1, 256, seed)[0]
            hydro_obs, _ = SRTMHydroDataset(1, 256, seed)[0]
            soil_obs, _ = SMAPSoilDataset(1, 256, seed)[0]

            _domain_tensors = {
                "fire": fire_obs.unsqueeze(0).to(_device),
                "forest": forest_obs.unsqueeze(0).to(_device),
                "hydro": hydro_obs.unsqueeze(0).to(_device),
                "soil": soil_obs.unsqueeze(0).to(_device),
            }

            # Update hard constraints from new region's hydro data
            _extract_hard_constraints(_domain_tensors)

            # Re-run models
            _perception.eval()
            with torch.no_grad():
                _agent_masks = _perception(_domain_tensors)
        else:
            # Legacy: just regenerate mock data
            from data import MockEODataset
            obs, _ = MockEODataset(num_samples=1, seed=seed)[0]
            _obs_tensor = obs.unsqueeze(0).to(_device)

            _perception.eval()
            with torch.no_grad():
                _agent_masks = _perception(_obs_tensor)

        # Update aggregator → harm mask → env
        default_weights = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=_device)
        with torch.no_grad():
            if _using_real_models:
                harm = _aggregator(
                    _agent_masks, default_weights,
                    slope=_slope_tensor, river_proximity=_river_prox_tensor,
                )
            else:
                slope = _obs_tensor[:, 14:15, :, :] if _obs_tensor is not None else None
                river = _obs_tensor[:, 18:19, :, :] if _obs_tensor is not None else None
                harm = _aggregator(
                    _agent_masks, default_weights,
                    slope=slope, river_proximity=river,
                )
        harm_np = harm.squeeze().cpu().numpy()

        # Extract terrain data for ImpactPropagation
        if _using_real_models and _slope_tensor is not None and _river_prox_tensor is not None:
            loc_slope = _slope_tensor.squeeze().cpu().numpy()
            loc_flow = _river_prox_tensor.squeeze().cpu().numpy()
            loc_flow_dir = _flow_dir_tensor.squeeze().cpu().numpy() if _flow_dir_tensor is not None else None
        elif _obs_tensor is not None:
            loc_slope = _obs_tensor[:, 14:15, :, :].squeeze().cpu().numpy()
            loc_flow = _obs_tensor[:, 16:17, :, :].squeeze().cpu().numpy()
            loc_flow_dir = None  # legacy mock data has no explicit flow direction
        else:
            loc_slope = np.zeros((SPATIAL, SPATIAL), dtype=np.float32)
            loc_flow = np.zeros((SPATIAL, SPATIAL), dtype=np.float32)
            loc_flow_dir = None

        _env = DeforestationEnv(
            harm_mask=harm_np,
            flow_accumulation=loc_flow,
            slope=loc_slope,
            flow_direction=loc_flow_dir,
        )
        _env.reset(seed=seed)

        print(f"[LOCATION] Ready: {location}")
        return jsonify({"status": "ok", "location": location})

    except Exception as e:
        print(f"[LOCATION] Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _init_models()
    print("Starting MISDO Dashboard at http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)

