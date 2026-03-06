"""
MISDO Module 4 — RL Sequential Optimizer (Gymnasium Environment)
=================================================================
Deforestation formulated as a Markov Decision Process (MDP).

Observation Space : Box([5, 256, 256])
    Layer 0 — Dynamic_Harm_Mask  (updated each step via ImpactPropagation)
    Layer 1 — Forest_State       (1=intact, 0=harvested, mutates each step)
    Layer 2 — Infrastructure     (1=road/cleared, 0=untouched, mutates)
    Layer 3 — Contagion_Risk     (forest loss contagion from ImpactPropagation)
    Layer 4 — Pollution_Risk     (downstream water pollution from ImpactPropagation)

Action Space : Discrete(256 * 256) = Discrete(65536)
    Flattened (row, col) coordinate of the harvest block centre.

Episode ends after 50 valid harvests (quota met).

Reward includes:
    - Base reward for valid contiguous harvests
    - Harm penalty (direct harm in harvested block)
    - Contagion penalty (risk of secondary deforestation created by this harvest)
    - Pollution penalty (downstream water quality impact)
    - Fragmentation penalty (splitting forest into disconnected components)
    - Core-area bonus (harvesting at edges preserves interior habitat)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.ndimage import binary_dilation, label

from impact import ImpactPropagation


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

# Impact-based penalty weights (applied to mean over grid, not raw sum)
CONTAGION_PENALTY_WEIGHT: float = 2.0   # penalty for contagion risk (mean-normalised)
POLLUTION_PENALTY_WEIGHT: float = 3.0   # penalty for downstream pollution (mean-normalised)

# Fragmentation penalty parameters
FRAGMENTATION_WEIGHT: float = 3.0   # penalty per new fragment created
CORE_AREA_BONUS: float = 1.5        # bonus for harvesting at forest edges

# Number of observation channels
OBS_CHANNELS: int = 5


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

    Uses vectorised NumPy operations (no Python for-loops over pixels).

    Returns a value in [0, 1].
    """
    block = forest[r0:r1, c0:c1]
    forest_pixels = block > 0.5
    total_forest = forest_pixels.sum()
    if total_forest < 1:
        return 0.0

    # Pad entire grid with 0 (boundary = non-forest)
    padded = np.pad(forest, 1, mode="constant", constant_values=0)
    # Extract the padded region corresponding to our block
    pr0, pr1, pc0, pc1 = r0 + 1, r1 + 1, c0 + 1, c1 + 1
    region = padded[pr0:pr1, pc0:pc1]

    # Check 4-connected neighbours using shifted slices
    has_non_forest_neighbour = (
        (padded[pr0 - 1:pr1 - 1, pc0:pc1] < 0.5) |  # above
        (padded[pr0 + 1:pr1 + 1, pc0:pc1] < 0.5) |  # below
        (padded[pr0:pr1, pc0 - 1:pc1 - 1] < 0.5) |  # left
        (padded[pr0:pr1, pc0 + 1:pc1 + 1] < 0.5)     # right
    )

    edge_count = (forest_pixels & has_non_forest_neighbour).sum()
    return float(edge_count) / float(total_forest)


class DeforestationEnv(gym.Env):
    """Gymnasium environment for sequential deforestation planning.

    Uses ImpactPropagation to compute cascading environmental damage
    (forest contagion + downstream water pollution) at every step,
    replacing the previous naive edge multiplier.

    Parameters
    ----------
    harm_mask : np.ndarray [256, 256]
        Pre-computed Final_Harm_Mask from the Aggregator (values in [0, 1]).
    flow_accumulation : np.ndarray [256, 256] or None
        Flow accumulation network (from SRTM/terrain). If None, defaults
        to zeros (no hydrological cascade).
    slope : np.ndarray [256, 256] or None
        Terrain slope in [0, 1]. If None, defaults to zeros.
    """

    metadata: Dict[str, Any] = {"render_modes": ["human"]}

    def __init__(
        self,
        harm_mask: np.ndarray,
        flow_accumulation: Optional[np.ndarray] = None,
        slope: Optional[np.ndarray] = None,
        flow_direction: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        assert harm_mask.shape == (SPATIAL, SPATIAL), (
            f"harm_mask must be ({SPATIAL}, {SPATIAL}), got {harm_mask.shape}"
        )

        self._base_harm_mask: np.ndarray = harm_mask.astype(np.float32).copy()

        # --- Terrain data for ImpactPropagation ---
        if flow_accumulation is None:
            flow_accumulation = np.zeros((SPATIAL, SPATIAL), dtype=np.float32)
        if slope is None:
            slope = np.zeros((SPATIAL, SPATIAL), dtype=np.float32)

        self._flow_acc = flow_accumulation.astype(np.float32)
        self._slope = slope.astype(np.float32)

        # --- Instantiate ImpactPropagation ---
        self._impact = ImpactPropagation(
            flow_accumulation=self._flow_acc,
            slope=self._slope,
            flow_direction=flow_direction,
        )

        # Gym spaces — 5 channels now (harm, forest, infra, contagion, pollution)
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(OBS_CHANNELS, SPATIAL, SPATIAL),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(SPATIAL * SPATIAL)

        # State arrays (initialized in reset)
        self._harm_mask: np.ndarray = np.zeros((SPATIAL, SPATIAL), dtype=np.float32)
        self._forest: np.ndarray = np.zeros((SPATIAL, SPATIAL), dtype=np.float32)
        self._infra: np.ndarray = np.zeros((SPATIAL, SPATIAL), dtype=np.float32)
        self._contagion: np.ndarray = np.zeros((SPATIAL, SPATIAL), dtype=np.float32)
        self._pollution: np.ndarray = np.zeros((SPATIAL, SPATIAL), dtype=np.float32)
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

        # Impact layers start at zero (no clearing has happened yet)
        self._contagion = np.zeros((SPATIAL, SPATIAL), dtype=np.float32)
        self._pollution = np.zeros((SPATIAL, SPATIAL), dtype=np.float32)

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

        # --- Cascading Impact via ImpactPropagation ---
        newly_cleared: np.ndarray = (pre_forest > 0.5) & (self._forest < 0.5)
        newly_cleared_f = newly_cleared.astype(np.float32)

        impact = self._impact.compute_cumulative_impact(
            harm_mask=self._harm_mask,
            forest_state=self._forest,
            newly_cleared=newly_cleared_f,
        )

        # ImpactPropagation returns PyTorch Tensors — convert to NumPy
        total_impact = impact["total_impact"].cpu().numpy() if hasattr(impact["total_impact"], 'cpu') else impact["total_impact"]
        step_contagion = impact["contagion"].cpu().numpy() if hasattr(impact["contagion"], 'cpu') else impact["contagion"]
        step_pollution = impact["pollution"].cpu().numpy() if hasattr(impact["pollution"], 'cpu') else impact["pollution"]

        # Update the dynamic harm mask with total impact
        self._harm_mask = total_impact

        # Store contagion and pollution layers for observation
        # Accumulate: each step's contagion/pollution adds to history
        self._contagion = np.clip(
            self._contagion + step_contagion, 0, 1
        )
        self._pollution = np.clip(
            self._pollution + step_pollution, 0, 1
        )

        # --- Reward ---
        block_area = float((r1 - r0) * (c1 - c0))
        # Normalise to mean so it's on the same scale as cascading penalties
        block_harm: float = float(self._harm_mask[r0:r1, c0:c1].mean())
        fragmentation_penalty: float = FRAGMENTATION_WEIGHT * new_fragments
        core_area_bonus: float = CORE_AREA_BONUS * edge_fraction

        # Cascading impact penalties (mean over grid for scale-invariance)
        mean_contagion: float = float(step_contagion.mean())
        mean_pollution: float = float(step_pollution.mean())
        contagion_penalty: float = CONTAGION_PENALTY_WEIGHT * mean_contagion
        pollution_penalty: float = POLLUTION_PENALTY_WEIGHT * mean_pollution

        reward: float = (
            REWARD_BASE
            - block_harm
            - fragmentation_penalty
            + core_area_bonus
            - contagion_penalty
            - pollution_penalty
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
                "contagion_penalty": round(contagion_penalty, 4),
                "pollution_penalty": round(pollution_penalty, 4),
                "impact_scores": self._impact.compute_total_ecosystem_score(
                    self._harm_mask, self._forest, newly_cleared_f,
                ),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        obs = np.stack(
            [self._harm_mask, self._forest, self._infra,
             self._contagion, self._pollution],
            axis=0,
        )  # Shape: [5, 256, 256]
        return obs.astype(np.float32)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dummy_harm = np.random.rand(SPATIAL, SPATIAL).astype(np.float32) * 0.5
    dummy_flow = np.random.rand(SPATIAL, SPATIAL).astype(np.float32) * 0.5
    dummy_slope = np.random.rand(SPATIAL, SPATIAL).astype(np.float32) * 0.3
    env = DeforestationEnv(
        harm_mask=dummy_harm,
        flow_accumulation=dummy_flow,
        slope=dummy_slope,
    )
    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}")  # [5, 256, 256]

    # Take an action on the left edge (adjacent to road)
    action = 5 * SPATIAL + 1  # row=5, col=1 (next to col-0 road)
    obs, rew, term, trunc, info = env.step(action)
    print(f"Step reward: {rew:.2f}  valid: {info.get('valid')}")
    print(f"  new_fragments: {info.get('new_fragments')}")
    print(f"  edge_fraction: {info.get('edge_fraction')}")
    print(f"  fragmentation_penalty: {info.get('fragmentation_penalty')}")
    print(f"  core_area_bonus: {info.get('core_area_bonus')}")
    print(f"  contagion_penalty: {info.get('contagion_penalty')}")
    print(f"  pollution_penalty: {info.get('pollution_penalty')}")
    print(f"Obs shape after step: {obs.shape}")

    # Run a few more steps to see cascading impact
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
                  f"edge={info.get('edge_fraction')}  "
                  f"contagion={info.get('contagion_penalty'):.4f}  "
                  f"pollution={info.get('pollution_penalty'):.4f}")

    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Total new fragments: {total_frags}")
