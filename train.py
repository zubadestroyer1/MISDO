"""
MISDO Module 5 — End-to-End Execution & PPO Training
======================================================
Wires all modules together:
    MockEODataset → MISDOPerception → ConditionedAggregator →
    DeforestationEnv → PPO (Stable-Baselines3)

Includes a custom CNN ``BaseFeaturesExtractor`` for the [3, 256, 256] obs.

Architecture Note — Custom CNN Feature Extractor
-------------------------------------------------
The PPO policy receives a [3, 256, 256] observation consisting of:
    - Layer 0: Final_Harm_Mask  (pre-computed spatial risk)
    - Layer 1: Forest_State     (binary, evolves over the episode)
    - Layer 2: Infrastructure   (binary, evolves over the episode)

All heavy spatial reasoning (multi-modal fusion, non-linear aggregation,
hard constraints) has already been performed by the Perception backbone and
Aggregator upstream.  The RL policy therefore only needs to learn *where to
cut next* given these three summary layers.

A lightweight 3-layer strided-conv CNN (3 → 32 → 64 → 128) progressively
compresses the 256×256 spatial dimensions down to a 1-D feature vector.
This keeps the policy network small and fast to train, while still giving
PPO enough spatial context to reason about contiguity and harm gradients.
"""

from __future__ import annotations

import sys
from typing import Dict, Type

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import Tensor

# Local modules
from data import MockEODataset
from perception import MISDOPerception
from aggregator import ConditionedAggregator
from env import DeforestationEnv, SPATIAL


# ═══════════════════════════════════════════════════════════════════════════
# Custom CNN Feature Extractor for SB3
# ═══════════════════════════════════════════════════════════════════════════

class MISDOFeatureExtractor(BaseFeaturesExtractor):
    """3-layer strided-conv CNN that maps [3, 256, 256] → 1-D feature vector.

    Architecture:
        Conv(3→32,  k=8, s=4) → ReLU → 64×64
        Conv(32→64, k=4, s=2) → ReLU → 32×32
        Conv(64→128,k=3, s=2) → ReLU → 16×16
        Flatten → Linear → features_dim
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim)

        n_channels: int = observation_space.shape[0]  # 3

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            # Shape: [B, 32, 63, 63]   (floor((256-8)/4)+1 = 63)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            # Shape: [B, 64, 30, 30]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            # Shape: [B, 128, 14, 14]
            nn.Flatten(),
        )

        # Compute the flattened size dynamically
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flat: int = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flat, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Tensor) -> Tensor:
        x = self.cnn(observations)       # Shape: [B, n_flat]
        return self.linear(x)            # Shape: [B, features_dim]


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline: Dataset → Perception → Aggregator → Harm Mask
# ═══════════════════════════════════════════════════════════════════════════

def generate_harm_mask(
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Runs one batch through the perception+aggregation pipeline and
    returns a single [256, 256] harm mask as a NumPy array for the RL env.
    """
    print("[1/3] Generating synthetic EO data ...")
    dataset = MockEODataset(num_samples=1, seed=0)
    obs_tensor, _ = dataset[0]
    obs_tensor = obs_tensor.unsqueeze(0).to(device)
    # obs_tensor Shape: [1, 20, 256, 256]

    print("[2/3] Running Perception backbone + decoder heads ...")
    perception = MISDOPerception(in_channels=20).to(device)
    perception.eval()
    with torch.no_grad():
        agent_masks: Tensor = perception(obs_tensor)
    # agent_masks Shape: [1, 4, 256, 256]
    print(f"       Agent_Masks shape: {agent_masks.shape}  "
          f"min={agent_masks.min():.3f}  max={agent_masks.max():.3f}")

    print("[3/3] Aggregating with user weights + hard constraints ...")
    aggregator = ConditionedAggregator().to(device)
    aggregator.eval()

    user_weights = torch.tensor([[0.9, 0.1, 0.5, 0.2]], device=device)
    # user_weights Shape: [1, 4]

    # Extract slope and river proximity from the 20-channel mock tensor
    slope = obs_tensor[:, 14:15, :, :]       # SRTM Slope channel
    river = obs_tensor[:, 18:19, :, :]        # Proximity — Distance to River

    with torch.no_grad():
        harm_mask: Tensor = aggregator(
            agent_masks, user_weights,
            slope=slope, river_proximity=river,
        )
    # harm_mask Shape: [1, 1, 256, 256]
    print(f"       Final_Harm_Mask shape: {harm_mask.shape}  "
          f"min={harm_mask.min():.3f}  max={harm_mask.max():.3f}")

    return harm_mask.squeeze(0).squeeze(0).cpu().numpy()  # [256, 256]


# ═══════════════════════════════════════════════════════════════════════════
# Main: end-to-end training
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Step A: compute harm mask ---
    harm_mask_np: np.ndarray = generate_harm_mask(device=device)
    # harm_mask_np Shape: [256, 256]

    # --- Step B: create Gymnasium environment ---
    print("\n[ENV] Initialising DeforestationEnv ...")
    env = DeforestationEnv(harm_mask=harm_mask_np)
    obs, info = env.reset(seed=42)
    print(f"       Obs shape: {obs.shape}")  # [3, 256, 256]

    # --- Step C: initialise PPO with custom feature extractor ---
    print("\n[PPO] Creating PPO agent with MISDOFeatureExtractor ...")
    policy_kwargs: Dict[str, object] = dict(
        features_extractor_class=MISDOFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        device=device,
    )

    # --- Step D: train for 1,000 timesteps ---
    print("\n[TRAIN] Starting PPO training for 1,000 timesteps ...\n")
    model.learn(total_timesteps=1_000)
    print("\n[DONE] Training complete.")

    # --- Step E: quick evaluation ---
    print("\n[EVAL] Running one evaluation episode (max 500 steps) ...")
    obs, info = env.reset(seed=123)
    total_reward: float = 0.0
    steps: int = 0
    max_eval_steps: int = 500
    done: bool = False

    while not done and steps < max_eval_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        steps += 1
        done = terminated or truncated

    print(f"       Eval episode: {steps} steps, total reward = {total_reward:.2f}")
    valid_harvests: int = info.get("harvest_count", 0)
    print(f"       Valid harvests: {valid_harvests}")
    if steps >= max_eval_steps:
        print("       (capped at max eval steps — expected for untrained agents)")


if __name__ == "__main__":
    main()
