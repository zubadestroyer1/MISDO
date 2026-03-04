"""
MISDO Module 4 — RL Sequential Optimizer (Gymnasium Environment)
=================================================================
Deforestation formulated as a Markov Decision Process (MDP).

Observation Space : Box([3, 256, 256])
    Layer 0 — Final_Harm_Mask   (from Aggregator, static within episode)
    Layer 1 — Forest_State      (1=intact, 0=harvested, mutates each step)
    Layer 2 — Infrastructure    (1=road/cleared, 0=untouched, mutates)

Action Space : Discrete(256 * 256) = Discrete(65536)
    Flattened (row, col) coordinate of the harvest block centre.

Episode ends after 50 valid harvests (quota met).

Reward includes:
    - Base reward for valid contiguous harvests
    - Harm penalty (sum of harm mask in harvested block)
    - Fragmentation penalty (splitting forest into disconnected components)
    - Core-area bonus (harvesting at edges preserves interior habitat)
    - Dynamic edge effects (1.2× harm multiplier on newly exposed forest edges)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.ndimage import binary_dilation, label


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPATIAL: int = 256
BLOCK_SIZE: int = 10           # 10 × 10 harvest block
MAX_HARVESTS: int = 50         # episode quota
HALF_BLOCK: int = BLOCK_SIZE // 2

# Reward components
REWARD_BASE: float = 10.0      # base reward for a valid contiguous harvest
REWARD_REJECT: float = -100.0  # penalty for non-contiguous action

# Edge-effect parameters
DILATION_PX: int = 2
EDGE_MULTIPLIER: float = 1.2

# Fragmentation penalty parameters
FRAGMENTATION_WEIGHT: float = 3.0   # penalty per new fragment created
CORE_AREA_BONUS: float = 1.5        # bonus for harvesting at forest edges


def _count_forest_components(forest: np.ndarray) -> int:
    """Count the number of connected forest components using 8-connectivity."""
    struct = np.ones((3, 3), dtype=bool)  # 8-connected
    _, n_components = label(forest > 0.5, structure=struct)
    return n_components


def _compute_edge_fraction(forest: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> float:
    """Compute what fraction of harvested pixels are on the forest edge.

    Edge pixels are forest pixels adjacent to non-forest (infrastructure,
    already harvested, or grid boundary). High edge fraction = good
    (harvesting at edges preserves interior habitat).

    Returns a value in [0, 1].
    """
    block = forest[r0:r1, c0:c1]
    if block.sum() < 1:
        return 0.0

    # Pad the full forest grid's sub-region with 0 (boundary = non-forest)
    # Check each forest pixel in the block for non-forest neighbours
    padded = np.pad(forest, 1, mode="constant", constant_values=0)
    # Adjust indices for padding
    pr0, pr1, pc0, pc1 = r0 + 1, r1 + 1, c0 + 1, c1 + 1

    edge_count = 0
    total_forest = 0

    for r in range(pr0, pr1):
        for c in range(pc0, pc1):
            if padded[r, c] > 0.5:
                total_forest += 1
                # Check 4-connected neighbours
                neighbours = [
                    padded[r - 1, c], padded[r + 1, c],
                    padded[r, c - 1], padded[r, c + 1],
                ]
                if any(n < 0.5 for n in neighbours):
                    edge_count += 1

    return edge_count / max(total_forest, 1)


class DeforestationEnv(gym.Env):
    """Gymnasium environment for sequential deforestation planning.

    Parameters
    ----------
    harm_mask : np.ndarray [256, 256]
        Pre-computed Final_Harm_Mask from the Aggregator (values in [0, 1]).
    """

    metadata: Dict[str, Any] = {"render_modes": ["human"]}

    def __init__(self, harm_mask: np.ndarray) -> None:
        super().__init__()
        assert harm_mask.shape == (SPATIAL, SPATIAL), (
            f"harm_mask must be ({SPATIAL}, {SPATIAL}), got {harm_mask.shape}"
        )

        self._base_harm_mask: np.ndarray = harm_mask.astype(np.float32).copy()

        # Gym spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,  # harm mask can exceed 1.0 after edge multiplier
            shape=(3, SPATIAL, SPATIAL),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(SPATIAL * SPATIAL)

        # State arrays (initialized in reset)
        self._harm_mask: np.ndarray = np.zeros((SPATIAL, SPATIAL), dtype=np.float32)
        self._forest: np.ndarray = np.zeros((SPATIAL, SPATIAL), dtype=np.float32)
        self._infra: np.ndarray = np.zeros((SPATIAL, SPATIAL), dtype=np.float32)
        self._harvest_count: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self._harm_mask = self._base_harm_mask.copy()

        # Forest state — fully forested
        self._forest = np.ones((SPATIAL, SPATIAL), dtype=np.float32)

        # Infrastructure — vertical access road on the left edge
        self._infra = np.zeros((SPATIAL, SPATIAL), dtype=np.float32)
        self._infra[:, 0] = 1.0  # column 0

        self._harvest_count = 0

        return self._get_obs(), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        row: int = action // SPATIAL
        col: int = action % SPATIAL

        # --- Harvest block bounds (clipped to grid) ---
        r0 = max(row - HALF_BLOCK, 0)
        r1 = min(row + HALF_BLOCK, SPATIAL)
        c0 = max(col - HALF_BLOCK, 0)
        c1 = min(col + HALF_BLOCK, SPATIAL)

        # --- Contiguity check ---
        infra_overlap: float = float(self._infra[r0:r1, c0:c1].sum())
        if infra_overlap == 0.0:
            # Block does not touch any existing infrastructure → reject
            return self._get_obs(), REWARD_REJECT, False, False, {"valid": False}

        # --- Pre-harvest measurements ---
        pre_forest = self._forest.copy()
        n_components_before = _count_forest_components(self._forest)

        # Edge fraction: how much of the harvest block is on the forest edge
        edge_fraction = _compute_edge_fraction(self._forest, r0, r1, c0, c1)

        # --- Apply harvest ---
        self._forest[r0:r1, c0:c1] = 0.0
        self._infra[r0:r1, c0:c1] = 1.0

        # --- Post-harvest measurements ---
        n_components_after = _count_forest_components(self._forest)
        new_fragments = max(0, n_components_after - n_components_before)

        # --- Dynamic edge effects ---
        newly_cleared: np.ndarray = (pre_forest == 1.0) & (self._forest == 0.0)
        struct = np.ones((2 * DILATION_PX + 1, 2 * DILATION_PX + 1), dtype=bool)
        edge_zone: np.ndarray = binary_dilation(
            newly_cleared, structure=struct, iterations=1
        )
        # Only apply to pixels that are STILL forested
        edge_zone = edge_zone & (self._forest == 1.0)
        self._harm_mask[edge_zone] *= EDGE_MULTIPLIER

        # --- Reward ---
        block_harm: float = float(self._harm_mask[r0:r1, c0:c1].sum())
        fragmentation_penalty: float = FRAGMENTATION_WEIGHT * new_fragments
        core_area_bonus: float = CORE_AREA_BONUS * edge_fraction

        reward: float = (
            REWARD_BASE
            - block_harm
            - fragmentation_penalty
            + core_area_bonus
        )

        self._harvest_count += 1
        terminated: bool = self._harvest_count >= MAX_HARVESTS

        return (
            self._get_obs(),
            reward,
            terminated,
            False,
            {
                "valid": True,
                "harvest_count": self._harvest_count,
                "new_fragments": new_fragments,
                "edge_fraction": round(edge_fraction, 3),
                "fragmentation_penalty": round(fragmentation_penalty, 2),
                "core_area_bonus": round(core_area_bonus, 2),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        obs = np.stack(
            [self._harm_mask, self._forest, self._infra], axis=0
        )  # Shape: [3, 256, 256]
        return obs.astype(np.float32)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dummy_harm = np.random.rand(SPATIAL, SPATIAL).astype(np.float32) * 0.5
    env = DeforestationEnv(harm_mask=dummy_harm)
    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}")  # [3, 256, 256]

    # Take an action on the left edge (adjacent to road)
    action = 5 * SPATIAL + 1  # row=5, col=1 (next to col-0 road)
    obs, rew, term, trunc, info = env.step(action)
    print(f"Step reward: {rew:.2f}  valid: {info.get('valid')}")
    print(f"  new_fragments: {info.get('new_fragments')}")
    print(f"  edge_fraction: {info.get('edge_fraction')}")
    print(f"  fragmentation_penalty: {info.get('fragmentation_penalty')}")
    print(f"  core_area_bonus: {info.get('core_area_bonus')}")
    print(f"Obs shape after step: {obs.shape}")

    # Run a few more steps to see fragmentation in action
    total_reward = 0.0
    total_frags = 0
    for i in range(10):
        row = (i + 1) * 20
        col = 1
        action = row * SPATIAL + col
        obs, rew, term, trunc, info = env.step(action)
        if info.get("valid"):
            total_reward += rew
            total_frags += info.get("new_fragments", 0)
            print(f"  Step {i+2}: rew={rew:.2f}  frags={info.get('new_fragments')}  "
                  f"edge={info.get('edge_fraction')}")

    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Total new fragments: {total_frags}")
