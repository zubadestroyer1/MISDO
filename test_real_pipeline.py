"""
MISDO — End-to-End Real Pipeline Test
=======================================
Tests the full pipeline: domain data → trained models → aggregator → RL env.

Usage:
    python test_real_pipeline.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, os.path.dirname(__file__))

from datasets.viirs_fire import VIIRSFireDataset
from datasets.hansen_gfc import HansenGFCDataset
from datasets.srtm_hydro import SRTMHydroDataset
from datasets.smap_soil import SMAPSoilDataset
from models.fire_model import FireRiskNet
from models.forest_model import ForestLossNet
from models.hydro_model import HydroRiskNet
from models.soil_model import SoilRiskNet
from aggregator import ConditionedAggregator
from env import DeforestationEnv, SPATIAL, BLOCK_SIZE


def main() -> None:
    print("=" * 70)
    print("  MISDO — End-to-End Real Pipeline Test")
    print("=" * 70)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")

    weights_dir = os.path.join(os.path.dirname(__file__) or ".", "weights")
    has_weights = all(
        os.path.exists(os.path.join(weights_dir, f"{n}_model.pt"))
        for n in ["fire", "forest", "hydro", "soil"]
    )

    if not has_weights:
        print("\n⚠ No trained weights found!")
        print("  Please run: python train_models.py --model all")
        print("  Proceeding with random-init models for structural test...\n")

    # ─── Step 1: Generate domain data ────────────────────────────────────
    print("\n[1/5] Generating domain-specific synthetic data ...")
    t0 = time.time()

    datasets_info = {
        "fire": {"class": VIIRSFireDataset, "channels": 6},
        "forest": {"class": HansenGFCDataset, "channels": 5},
        "hydro": {"class": SRTMHydroDataset, "channels": 5},
        "soil": {"class": SMAPSoilDataset, "channels": 4},
    }

    domain_inputs = {}
    for name, info in datasets_info.items():
        ds = info["class"](num_samples=1, spatial_size=256, seed=42)
        obs, target = ds[0]
        domain_inputs[name] = obs.unsqueeze(0).to(device)
        print(f"  {name:8s}: obs {obs.shape}  "
              f"range [{obs.min():.3f}, {obs.max():.3f}]  "
              f"target {target.shape}")

    print(f"  Data generation: {time.time() - t0:.2f}s")

    # ─── Step 2: Load and run each model ─────────────────────────────────
    print("\n[2/5] Running domain-specific models ...")
    t0 = time.time()

    model_classes = {
        "fire": FireRiskNet,
        "forest": ForestLossNet,
        "hydro": HydroRiskNet,
        "soil": SoilRiskNet,
    }

    agent_masks_list = []
    model_stats = {}

    for name, ModelClass in model_classes.items():
        model = ModelClass().to(device)
        params = sum(p.numel() for p in model.parameters())

        # Load weights if available
        weight_path = os.path.join(weights_dir, f"{name}_model.pt")
        if os.path.exists(weight_path):
            state = torch.load(weight_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            trained = "✓ trained"
        else:
            trained = "✗ random"

        model.eval()
        with torch.no_grad():
            mask = model(domain_inputs[name])  # [1, 1, 256, 256]

        agent_masks_list.append(mask)
        model_stats[name] = {
            "params": params,
            "trained": trained,
            "min": mask.min().item(),
            "max": mask.max().item(),
            "mean": mask.mean().item(),
            "std": mask.std().item(),
        }
        print(f"  {name:8s}: {trained}  params={params:>10,}  "
              f"mask=[{mask.min():.4f}, {mask.max():.4f}]  "
              f"mean={mask.mean():.4f}")

    agent_masks = torch.cat(agent_masks_list, dim=1)  # [1, 4, 256, 256]
    print(f"  Stacked agent_masks: {agent_masks.shape}")
    print(f"  Inference time: {time.time() - t0:.2f}s")

    # ─── Step 3: Aggregator ──────────────────────────────────────────────
    print("\n[3/5] Running conditioned aggregator ...")
    t0 = time.time()

    aggregator = ConditionedAggregator().to(device)
    aggregator.eval()

    # Test with multiple weight configurations
    weight_configs = {
        "balanced": [0.5, 0.5, 0.5, 0.5],
        "fire_priority": [0.9, 0.2, 0.3, 0.1],
        "water_priority": [0.2, 0.3, 0.9, 0.2],
        "forest_priority": [0.1, 0.9, 0.3, 0.3],
    }

    # Extract hard-constraint data from the real SRTM hydro domain
    # SRTM channels: 0=elevation, 1=slope, 2=aspect, 3=flow_acc, 4=flow_dir
    hydro_tensor = domain_inputs["hydro"]  # [1, 5, 256, 256]
    slope_tensor = hydro_tensor[:, 1:2, :, :]     # [1, 1, 256, 256]
    river_tensor = hydro_tensor[:, 3:4, :, :]     # [1, 1, 256, 256]

    harm_masks = {}
    for config_name, weights in weight_configs.items():
        user_weights = torch.tensor([weights], device=device)
        with torch.no_grad():
            harm = aggregator(
                agent_masks, user_weights,
                slope=slope_tensor, river_proximity=river_tensor,
            )
        harm_np = harm.squeeze().cpu().numpy()
        harm_masks[config_name] = harm_np

        no_go_pct = float((harm_np >= 0.999).sum()) / (SPATIAL * SPATIAL) * 100
        print(f"  {config_name:16s}: harm=[{harm_np.min():.4f}, {harm_np.max():.4f}]  "
              f"mean={harm_np.mean():.4f}  no_go={no_go_pct:.1f}%")

    print(f"  Aggregation time: {time.time() - t0:.2f}s")

    # ─── Step 4: RL Environment ──────────────────────────────────────────
    print("\n[4/5] Running RL environment test (balanced weights) ...")
    t0 = time.time()

    harm_mask_np = harm_masks["balanced"]
    env = DeforestationEnv(harm_mask=harm_mask_np)
    obs, info = env.reset(seed=42)
    print(f"  Initial obs: {obs.shape}")

    # Run 100 steps
    total_reward = 0.0
    valid_harvests = 0
    invalid_attempts = 0
    rewards_list = []

    for step in range(100):
        # Simple heuristic: harvest blocks near the road (low column)
        row = (step * 11) % SPATIAL
        col = min(step // 10 * BLOCK_SIZE + 1, SPATIAL - 1)
        action = row * SPATIAL + col

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rewards_list.append(reward)

        if info.get("valid", False):
            valid_harvests += 1
        else:
            invalid_attempts += 1

        if terminated:
            print(f"  Episode terminated at step {step + 1} (quota met)")
            break

    print(f"  Steps: {step + 1}")
    print(f"  Valid harvests: {valid_harvests}")
    print(f"  Invalid attempts: {invalid_attempts}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Avg reward/step: {total_reward / (step + 1):.2f}")
    print(f"  Final forest: {obs[1].sum() / (SPATIAL * SPATIAL) * 100:.1f}%")
    print(f"  RL test time: {time.time() - t0:.2f}s")

    # ─── Step 5: Summary ─────────────────────────────────────────────────
    from data import MockEODataset
    print(f"\n{'=' * 70}")
    print(f"  TEST RESULTS SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n  {'Model':<12} {'Status':<12} {'Params':>10} {'Mean Risk':>10} {'Std':>8}")
    print(f"  {'-' * 52}")
    for name, stats in model_stats.items():
        print(f"  {name:<12} {stats['trained']:<12} {stats['params']:>10,} "
              f"{stats['mean']:>10.4f} {stats['std']:>8.4f}")

    total_params = sum(s["params"] for s in model_stats.values())
    print(f"\n  Total model parameters: {total_params:,}")

    print(f"\n  Aggregator harm mask statistics (balanced weights):")
    harm = harm_masks["balanced"]
    print(f"    Range: [{harm.min():.4f}, {harm.max():.4f}]")
    print(f"    Mean:  {harm.mean():.4f}")
    print(f"    Std:   {harm.std():.4f}")
    pcts = [25, 50, 75, 90, 95, 99]
    percentiles = np.percentile(harm.flatten(), pcts)
    print(f"    Percentiles: " + "  ".join(
        f"p{p}={v:.4f}" for p, v in zip(pcts, percentiles)
    ))

    print(f"\n  RL Environment:")
    print(f"    Total reward: {total_reward:.2f}")
    print(f"    Valid harvests: {valid_harvests} / {step + 1} attempts")
    print(f"    Forest remaining: {obs[1].sum() / (SPATIAL * SPATIAL) * 100:.1f}%")

    print(f"\n{'=' * 70}")
    print(f"  ✓ End-to-end pipeline test PASSED")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
