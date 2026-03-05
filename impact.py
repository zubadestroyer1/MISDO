"""
MISDO — Impact Propagation Module (PyTorch)
===============================================
Models how deforestation at one location causes cascading environmental
damage across the landscape:

1. Forest Loss Contagion: spatial diffusion of deforestation risk
2. Hydrological Cascade: upstream deforestation → downstream water pollution
3. Cumulative Impact Score: total ecosystem-wide harm from a harvest

Ported from NumPy to PyTorch for GPU acceleration.  All operations
use batched tensor ops — no Python for-loops over pixels.

This module sits between the Aggregator and the RL Environment,
transforming the static harm mask into a dynamic impact surface.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def _make_gaussian_kernel_2d(sigma: float, truncate: float = 4.0) -> Tensor:
    """Create a 2D Gaussian kernel matching scipy.ndimage.gaussian_filter."""
    radius = int(truncate * sigma + 0.5)
    ks = 2 * radius + 1
    ax = torch.arange(ks, dtype=torch.float32) - radius
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]


def _gaussian_blur(x: Tensor, kernel: Tensor) -> Tensor:
    """Apply Gaussian blur to a 2D tensor [H, W] using a precomputed kernel."""
    pad = kernel.shape[-1] // 2
    x_4d = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    kernel = kernel.to(x.device, x.dtype)
    out = F.conv2d(F.pad(x_4d, (pad, pad, pad, pad), mode="reflect"), kernel)
    return out.squeeze(0).squeeze(0)  # [H, W]


# D8 direction offsets indexed 0-8 (0 = no flow / pit)
_D8_DY = torch.tensor([0, -1, -1, -1, 0, 0, 1, 1, 1], dtype=torch.long)
_D8_DX = torch.tensor([0, -1, 0, 1, -1, 1, -1, 0, 1], dtype=torch.long)


class ImpactPropagation:
    """Computes cascading environmental impact from deforestation events.

    All computations use PyTorch tensors for GPU compatibility.

    Parameters
    ----------
    flow_accumulation : Tensor or ndarray [H, W]
        Flow accumulation network (from SRTM/terrain). Higher values =
        more upstream area draining through that pixel.
    slope : Tensor or ndarray [H, W]
        Terrain slope in [0, 1].
    flow_direction : Tensor or ndarray [H, W] or None
        D8 flow direction codes normalised to [0, 1] (divide by 8).
    device : str or torch.device
        Device to place tensors on.
    """

    def __init__(
        self,
        flow_accumulation,
        slope,
        flow_direction=None,
        contagion_sigma: float = 8.0,
        contagion_decay: float = 0.3,
        hydro_sigma: float = 12.0,
        hydro_weight: float = 0.4,
        erosion_weight: float = 0.3,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.flow_acc = self._to_tensor(flow_accumulation)
        self.slope = self._to_tensor(slope)

        # Decode normalised flow direction to integer D8 codes
        if flow_direction is not None:
            fd = self._to_tensor(flow_direction)
            self.flow_dir = torch.round(fd * 8.0).to(torch.long)
        else:
            self.flow_dir = None

        self.contagion_sigma = contagion_sigma
        self.contagion_decay = contagion_decay
        self.hydro_sigma = hydro_sigma
        self.hydro_weight = hydro_weight
        self.erosion_weight = erosion_weight

        # Precompute Gaussian kernels
        self._contagion_kernel = _make_gaussian_kernel_2d(contagion_sigma).to(self.device)
        self._hydro_kernel = _make_gaussian_kernel_2d(hydro_sigma).to(self.device)

    def _to_tensor(self, x) -> Tensor:
        """Convert ndarray or tensor to float32 on self.device."""
        if isinstance(x, Tensor):
            return x.to(device=self.device, dtype=torch.float32)
        import numpy as np
        return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)

    def compute_forest_contagion(
        self,
        forest_state: Tensor,
        newly_cleared: Tensor,
    ) -> Tensor:
        """Compute forest loss contagion from newly cleared areas.

        Areas near recent clearing have elevated deforestation risk
        (edge drying, wind exposure, road access effects).

        Returns
        -------
        contagion_risk : Tensor [H, W] in [0, 1]
        """
        forest_state = self._to_tensor(forest_state)
        newly_cleared = self._to_tensor(newly_cleared)

        diffused = _gaussian_blur(newly_cleared, self._contagion_kernel)
        contagion = diffused * forest_state * self.contagion_decay
        return contagion.clamp(0, 1)

    def compute_hydro_cascade(
        self,
        forest_state: Tensor,
        newly_cleared: Tensor,
    ) -> Tensor:
        """Compute downstream water pollution from upstream deforestation.

        Uses batched D8 flow direction routing (no Python for-loops over
        pixels). Falls back to Gaussian diffusion if no flow direction.

        Returns
        -------
        pollution : Tensor [H, W]
        """
        forest_state = self._to_tensor(forest_state)
        newly_cleared = self._to_tensor(newly_cleared)

        # Erosion source: cleared areas on slopes
        erosion_source = newly_cleared * self.slope

        if erosion_source.max() < 1e-8:
            return torch.zeros_like(erosion_source)

        H, W = erosion_source.shape
        pollution = erosion_source.clone()

        if self.flow_dir is not None:
            # ── Batched D8 routing (GPU-friendly) ──
            # Each iteration moves pollution one pixel downstream along D8.
            # All 8 direction codes are processed in parallel via masking.
            n_steps = max(H, W) // 4
            decay = 0.85

            for _ in range(n_steps):
                new_pollution = torch.zeros_like(pollution)
                for code in range(1, 9):
                    dy = _D8_DY[code].item()
                    dx = _D8_DX[code].item()
                    dir_mask = (self.flow_dir == code).float()
                    contribution = pollution * dir_mask * decay
                    # Shift to downstream neighbour using torch.roll
                    shifted = torch.roll(torch.roll(contribution, dy, 0), dx, 1)
                    new_pollution = new_pollution + shifted
                pollution = pollution + new_pollution
        else:
            # ── Fallback: Gaussian diffusion ──
            pollution = _gaussian_blur(pollution, self._hydro_kernel)

        # Normalise to [0, 1]
        max_val = pollution.max()
        if max_val > 0:
            pollution = pollution / max_val

        return pollution * self.hydro_weight

    def compute_cumulative_impact(
        self,
        harm_mask,
        forest_state,
        newly_cleared,
    ) -> Dict[str, Tensor]:
        """Compute total ecosystem impact from a deforestation event.

        Accepts both Tensors and ndarrays (auto-converted).

        Returns
        -------
        Dict with keys:
            'total_impact': combined impact surface [H, W]
            'contagion': forest loss contagion [H, W]
            'pollution': water pollution cascade [H, W]
            'direct_harm': original harm mask [H, W]
        """
        harm_mask = self._to_tensor(harm_mask)
        forest_state = self._to_tensor(forest_state)
        newly_cleared = self._to_tensor(newly_cleared)

        contagion = self.compute_forest_contagion(forest_state, newly_cleared)
        pollution = self.compute_hydro_cascade(forest_state, newly_cleared)

        total = (harm_mask + contagion + pollution).clamp(0, 1)

        return {
            "total_impact": total,
            "contagion": contagion,
            "pollution": pollution,
            "direct_harm": harm_mask,
        }

    def compute_total_ecosystem_score(
        self,
        harm_mask,
        forest_state,
        newly_cleared,
    ) -> Dict[str, float]:
        """Compute scalar ecosystem health metrics.

        Returns
        -------
        Dict with:
            'direct_harm_sum': sum of harm in cleared area
            'contagion_risk': total risk of secondary deforestation
            'downstream_pollution': total downstream water quality impact
            'fragmentation_score': number of forest fragments
            'total_ecosystem_cost': weighted sum of all impacts
        """
        harm_mask = self._to_tensor(harm_mask)
        forest_state = self._to_tensor(forest_state)
        newly_cleared = self._to_tensor(newly_cleared)

        impact = self.compute_cumulative_impact(harm_mask, forest_state, newly_cleared)

        direct = float(harm_mask[newly_cleared > 0.5].sum())
        contagion_total = float(impact["contagion"].sum())
        pollution_total = float(impact["pollution"].sum())

        # Fragmentation: count connected components using a flood-fill approach
        # (pure PyTorch — no scipy dependency)
        n_frags = self._count_components(forest_state > 0.5)

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

    @staticmethod
    def _count_components(mask: Tensor) -> int:
        """Count connected components in a boolean mask (8-connectivity).

        Uses iterative morphological flood fill — pure PyTorch, no scipy.
        Slower than scipy.ndimage.label but runs on GPU.
        """
        # For moderate sizes, fall back to scipy if available for speed
        try:
            from scipy.ndimage import label as scipy_label
            import numpy as np
            struct = np.ones((3, 3), dtype=bool)
            _, n = scipy_label(mask.cpu().numpy(), structure=struct)
            return int(n)
        except ImportError:
            pass

        # Pure-torch fallback: iterative dilation-based component counting
        remaining = mask.clone().bool()
        count = 0
        kernel = torch.ones(1, 1, 3, 3, device=mask.device, dtype=torch.float32)
        while remaining.any():
            # Pick first True pixel as seed
            idx = remaining.nonzero(as_tuple=False)[0]
            seed = torch.zeros_like(remaining, dtype=torch.float32)
            seed[idx[0], idx[1]] = 1.0
            # Iteratively dilate to flood fill
            prev_sum = 0.0
            while True:
                dilated = F.conv2d(
                    seed.unsqueeze(0).unsqueeze(0),
                    kernel,
                    padding=1,
                )
                seed = ((dilated.squeeze() > 0) & remaining).float()
                cur_sum = seed.sum().item()
                if cur_sum == prev_sum:
                    break
                prev_sum = cur_sum
            remaining = remaining & ~(seed > 0)
            count += 1
        return count


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    H, W = 256, 256
    flow = torch.rand(H, W) * 0.5
    slope = torch.rand(H, W) * 0.3

    ip = ImpactPropagation(flow, slope)

    forest = torch.ones(H, W)
    harm = torch.rand(H, W) * 0.5

    # Simulate a clearing
    cleared = torch.zeros(H, W)
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

    # Test numpy interop (backward compat with env.py)
    import numpy as np
    np_harm = np.random.rand(H, W).astype(np.float32) * 0.5
    np_forest = np.ones((H, W), dtype=np.float32)
    np_cleared = np.zeros((H, W), dtype=np.float32)
    np_cleared[50:60, 50:60] = 1.0
    np_forest[50:60, 50:60] = 0.0

    impact2 = ip.compute_cumulative_impact(np_harm, np_forest, np_cleared)
    print(f"\nNumPy interop OK: {type(impact2['total_impact'])}")
