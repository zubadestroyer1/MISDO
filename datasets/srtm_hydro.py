"""
SRTM + HydroSHEDS Dataset
===========================
Generates synthetic elevation and hydrological data matching SRTM DEM (90 m)
and HydroSHEDS flow-accumulation products.

Channels (5):
    0: elevation (DEM, normalised)               — [0, 1]
    1: slope (degrees, normalised to [0, 1])      — [0, 1]  (maps 0–90°)
    2: aspect (circular, sin-encoded)             — [-1, 1] → shifted to [0, 1]
    3: flow_accumulation (log-normalised)         — [0, 1]
    4: flow_direction (8-class, normalised)       — [0, 1]

Target: Water-pollution risk [1, 256, 256] — continuous in [0, 1], derived from
        high-slope × high-flow-accumulation = erosion/runoff risk.
"""

from __future__ import annotations

from typing import Tuple
import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


def _perlin_noise_2d(S: int, g: torch.Generator, scale: int = 32) -> Tensor:
    """Generate smooth 2D noise via bilinear upsampling of low-res random grid."""
    grid_size = max(S // scale, 2)
    low_res = torch.randn(1, 1, grid_size, grid_size, generator=g)
    high_res = F.interpolate(low_res, size=(S, S), mode="bilinear", align_corners=False)
    return high_res.squeeze()  # [S, S]


def _compute_slope_aspect(elevation: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute slope and aspect from elevation using Sobel-like finite differences.

    Returns slope in [0, 1] and sin(aspect) shifted to [0, 1].
    """
    # Pad edges for gradient computation
    padded = F.pad(elevation.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    # Finite differences (approximation of ∂z/∂x and ∂z/∂y)
    dzdx = (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]) / 2.0
    dzdy = (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]) / 2.0
    dzdx = dzdx.squeeze()  # [S, S]
    dzdy = dzdy.squeeze()

    # Slope (normalised: 0=flat, 1=vertical)
    slope = torch.sqrt(dzdx ** 2 + dzdy ** 2)
    slope = (slope / (slope.max() + 1e-8)).clamp(0, 1)

    # Aspect (sin component, shifted to [0, 1])
    aspect = torch.atan2(dzdy, dzdx)  # [-π, π]
    aspect_sin = (torch.sin(aspect) + 1) / 2  # [0, 1]

    return slope, aspect_sin


def _compute_flow_accumulation(elevation: Tensor) -> Tuple[Tensor, Tensor]:
    """Approximate flow accumulation via iterative steepest-descent routing.

    Returns log-normalised flow accumulation and encoded flow direction.
    """
    S = elevation.shape[0]
    flow_acc = torch.ones(S, S)
    flow_dir = torch.zeros(S, S)

    # 8-connected neighbour offsets
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]

    elev_np = elevation.numpy()

    # Simple D8 flow direction
    for r in range(1, S - 1):
        for c in range(1, S - 1):
            min_elev = elev_np[r, c]
            min_dir = 0
            for d_idx, (dr, dc) in enumerate(offsets):
                nr, nc = r + dr, c + dc
                if elev_np[nr, nc] < min_elev:
                    min_elev = elev_np[nr, nc]
                    min_dir = d_idx + 1
            flow_dir[r, c] = min_dir / 8.0  # normalise to [0, 1]

    # Simplified flow accumulation: iterate downstream
    # Sort cells by elevation (high to low) and accumulate
    flat_elev = elevation.flatten()
    sorted_idx = torch.argsort(flat_elev, descending=True)

    for flat_i in sorted_idx[:S * S // 2]:  # process top half for speed
        r = (flat_i // S).item()
        c = (flat_i % S).item()
        if r < 1 or r >= S - 1 or c < 1 or c >= S - 1:
            continue
        d = int(flow_dir[r, c].item() * 8)
        if d > 0 and d <= 8:
            dr, dc = offsets[d - 1]
            nr, nc = r + dr, c + dc
            flow_acc[nr, nc] += flow_acc[r, c]

    # Log-normalise
    flow_acc_log = torch.log1p(flow_acc)
    flow_acc_log = flow_acc_log / (flow_acc_log.max() + 1e-8)

    return flow_acc_log, flow_dir


class SRTMHydroDataset(Dataset):
    """Synthetic SRTM + HydroSHEDS terrain and hydrology dataset.

    Generates terrain using multi-scale Perlin noise, then derives slope,
    aspect, flow direction, and flow accumulation analytically.
    """

    NUM_CHANNELS: int = 5

    def __init__(
        self,
        num_samples: int = 64,
        spatial_size: int = 256,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.spatial_size = spatial_size
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        S = self.spatial_size
        g = torch.Generator().manual_seed(self.seed + idx)

        # --- Generate terrain via multi-scale noise ---
        elevation = torch.zeros(S, S)
        for scale in [64, 32, 16, 8]:
            weight = scale / 64.0
            elevation += _perlin_noise_2d(S, g, scale=scale) * weight
        # Normalise to [0, 1]
        elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 1e-8)

        # --- Derive slope and aspect ---
        slope, aspect_sin = _compute_slope_aspect(elevation)

        # --- Derive flow accumulation and direction ---
        flow_acc, flow_dir = _compute_flow_accumulation(elevation)

        observation = torch.stack([
            elevation,
            slope,
            aspect_sin,
            flow_acc,
            flow_dir,
        ], dim=0)  # [5, S, S]

        # --- Target: water-pollution / erosion risk ---
        # High slope + high flow accumulation = high erosion/runoff risk
        risk = (slope * 0.6 + flow_acc * 0.4)
        risk = risk / (risk.max() + 1e-8)
        # Add proximity to high-flow channels as contamination corridors
        channel_proximity = (flow_acc > 0.5).float()
        channel_proximity = F.max_pool2d(
            channel_proximity.unsqueeze(0).unsqueeze(0),
            kernel_size=7, stride=1, padding=3,
        ).squeeze()
        risk = (risk * 0.7 + channel_proximity * slope * 0.3).clamp(0, 1)

        target = risk.unsqueeze(0)  # [1, S, S]

        return observation, target


if __name__ == "__main__":
    ds = SRTMHydroDataset(num_samples=2, spatial_size=64, seed=0)
    obs, tgt = ds[0]
    print(f"SRTM obs: {obs.shape}  min={obs.min():.3f}  max={obs.max():.3f}")
    print(f"SRTM tgt: {tgt.shape}  mean_risk={tgt.mean():.3f}")
