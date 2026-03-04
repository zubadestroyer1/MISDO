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
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.ndimage import binary_dilation


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

        # --- Apply harvest ---
        # Snapshot the harvested area before clearing (for edge-effect mask)
        pre_forest = self._forest.copy()

        self._forest[r0:r1, c0:c1] = 0.0
        self._infra[r0:r1, c0:c1] = 1.0

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
        reward: float = REWARD_BASE - block_harm

        self._harvest_count += 1
        terminated: bool = self._harvest_count >= MAX_HARVESTS

        return (
            self._get_obs(),
            reward,
            terminated,
            False,
            {"valid": True, "harvest_count": self._harvest_count},
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
    print(f"Obs shape after step: {obs.shape}")
