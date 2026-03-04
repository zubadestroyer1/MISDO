"""
MISDO — Impact Propagation Module
====================================
Models how deforestation at one location causes cascading environmental
damage across the landscape:

1. Forest Loss Contagion: spatial diffusion of deforestation risk
2. Hydrological Cascade: upstream deforestation → downstream water pollution
3. Cumulative Impact Score: total ecosystem-wide harm from a harvest

This module sits between the Aggregator and the RL Environment,
transforming the static harm mask into a dynamic impact surface.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, label, binary_dilation
from typing import Dict, Optional


class ImpactPropagation:
    """Computes cascading environmental impact from deforestation events.

    Parameters
    ----------
    flow_accumulation : np.ndarray [H, W]
        Flow accumulation network (from SRTM/terrain). Higher values =
        more upstream area draining through that pixel.
    slope : np.ndarray [H, W]
        Terrain slope in [0, 1].
    """

    def __init__(
        self,
        flow_accumulation: np.ndarray,
        slope: np.ndarray,
        contagion_sigma: float = 8.0,
        contagion_decay: float = 0.3,
        hydro_sigma: float = 12.0,
        hydro_weight: float = 0.4,
        erosion_weight: float = 0.3,
    ) -> None:
        self.flow_acc = flow_accumulation.astype(np.float32)
        self.slope = slope.astype(np.float32)
        self.contagion_sigma = contagion_sigma
        self.contagion_decay = contagion_decay
        self.hydro_sigma = hydro_sigma
        self.hydro_weight = hydro_weight
        self.erosion_weight = erosion_weight

    def compute_forest_contagion(
        self,
        forest_state: np.ndarray,
        newly_cleared: np.ndarray,
    ) -> np.ndarray:
        """Compute forest loss contagion from newly cleared areas.

        Areas near recent clearing have elevated deforestation risk
        (edge drying, wind exposure, road access effects).

        Parameters
        ----------
        forest_state : np.ndarray [H, W]
            Current forest state (1=intact, 0=cleared).
        newly_cleared : np.ndarray [H, W]
            Boolean mask of newly cleared pixels.

        Returns
        -------
        contagion_risk : np.ndarray [H, W]
            Additional deforestation risk from proximity to clearings.
        """
        # Diffuse clearing signal spatially
        clearing_signal = newly_cleared.astype(np.float32)
        diffused = gaussian_filter(clearing_signal, sigma=self.contagion_sigma)

        # Only applies to remaining forested areas
        contagion = diffused * forest_state * self.contagion_decay

        return np.clip(contagion, 0, 1).astype(np.float32)

    def compute_hydro_cascade(
        self,
        forest_state: np.ndarray,
        newly_cleared: np.ndarray,
    ) -> np.ndarray:
        """Compute downstream water pollution from upstream deforestation.

        Deforested areas contribute sediment/nutrient runoff that
        propagates downstream through the flow accumulation network.

        Parameters
        ----------
        forest_state : np.ndarray [H, W]
            Current forest state.
        newly_cleared : np.ndarray [H, W]
            Boolean mask of newly cleared pixels.

        Returns
        -------
        pollution : np.ndarray [H, W]
            Downstream water pollution intensity.
        """
        # Erosion source: cleared areas on slopes
        erosion_source = newly_cleared.astype(np.float32) * self.slope

        # Propagate downstream via flow accumulation
        # (approximate: convolve erosion source with flow-weighted kernel)
        weighted_erosion = erosion_source * (1.0 + self.flow_acc * 3.0)
        pollution = gaussian_filter(weighted_erosion, sigma=self.hydro_sigma)

        # Normalise
        max_val = pollution.max()
        if max_val > 0:
            pollution = pollution / max_val

        return (pollution * self.hydro_weight).astype(np.float32)

    def compute_cumulative_impact(
        self,
        harm_mask: np.ndarray,
        forest_state: np.ndarray,
        newly_cleared: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute total ecosystem impact from a deforestation event.

        Combines:
        1. Direct harm (from aggregated risk models)
        2. Forest loss contagion (spatial diffusion)
        3. Hydrological cascade (downstream pollution)

        Parameters
        ----------
        harm_mask : np.ndarray [H, W]
            Base harm mask from aggregator.
        forest_state : np.ndarray [H, W]
            Current forest state.
        newly_cleared : np.ndarray [H, W]
            Boolean mask of newly cleared pixels.

        Returns
        -------
        Dict with keys:
            'total_impact': combined impact surface [H, W]
            'contagion': forest loss contagion [H, W]
            'pollution': water pollution cascade [H, W]
            'direct_harm': original harm mask [H, W]
        """
        contagion = self.compute_forest_contagion(forest_state, newly_cleared)
        pollution = self.compute_hydro_cascade(forest_state, newly_cleared)

        # Combine: direct + contagion + pollution
        total = harm_mask + contagion + pollution
        total = np.clip(total, 0, 1)

        return {
            "total_impact": total.astype(np.float32),
            "contagion": contagion.astype(np.float32),
            "pollution": pollution.astype(np.float32),
            "direct_harm": harm_mask.astype(np.float32),
        }

    def compute_total_ecosystem_score(
        self,
        harm_mask: np.ndarray,
        forest_state: np.ndarray,
        newly_cleared: np.ndarray,
    ) -> Dict[str, float]:
        """Compute scalar ecosystem health metrics.

        Returns
        -------
        Dict with:
            'direct_harm_sum': sum of harm in cleared area
            'contagion_risk': total risk of secondary deforestation
            'downstream_pollution': total downstream water quality impact
            'fragmentation_score': number of forest fragments created
            'total_ecosystem_cost': weighted sum of all impacts
        """
        impact = self.compute_cumulative_impact(
            harm_mask, forest_state, newly_cleared
        )

        direct = float(harm_mask[newly_cleared > 0.5].sum())
        contagion_total = float(impact["contagion"].sum())
        pollution_total = float(impact["pollution"].sum())

        # Fragmentation
        struct = np.ones((3, 3), dtype=bool)
        _, n_frags = label(forest_state > 0.5, structure=struct)

        total_cost = (
            direct
            + contagion_total * 0.5
            + pollution_total * 0.8
            + n_frags * 2.0
        )

        return {
            "direct_harm_sum": round(direct, 4),
            "contagion_risk": round(contagion_total, 4),
            "downstream_pollution": round(pollution_total, 4),
            "fragmentation_score": n_frags,
            "total_ecosystem_cost": round(total_cost, 4),
        }


if __name__ == "__main__":
    H, W = 256, 256
    flow = np.random.rand(H, W).astype(np.float32) * 0.5
    slope = np.random.rand(H, W).astype(np.float32) * 0.3

    ip = ImpactPropagation(flow, slope)

    forest = np.ones((H, W), dtype=np.float32)
    harm = np.random.rand(H, W).astype(np.float32) * 0.5

    # Simulate a clearing
    cleared = np.zeros((H, W), dtype=np.float32)
    cleared[100:110, 100:110] = 1.0
    forest[100:110, 100:110] = 0.0

    impact = ip.compute_cumulative_impact(harm, forest, cleared)
    print(f"Direct harm range: [{impact['direct_harm'].min():.3f}, {impact['direct_harm'].max():.3f}]")
    print(f"Contagion range: [{impact['contagion'].min():.3f}, {impact['contagion'].max():.3f}]")
    print(f"Pollution range: [{impact['pollution'].min():.3f}, {impact['pollution'].max():.3f}]")
    print(f"Total impact range: [{impact['total_impact'].min():.3f}, {impact['total_impact'].max():.3f}]")

    scores = ip.compute_total_ecosystem_score(harm, forest, cleared)
    for k, v in scores.items():
        print(f"  {k}: {v}")
