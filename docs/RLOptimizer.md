# RL Optimizer — PPO Sequential Deforestation Planner

## Overview

The **RL Optimizer** formulates sustainable deforestation as a Markov Decision Process (MDP) solved by Proximal Policy Optimization (PPO). Given the fused harm mask from the aggregator, the PPO agent learns *where to harvest next* to minimise environmental harm while meeting timber quotas.

| Property | Value |
|---|---|
| **Algorithm** | PPO (Proximal Policy Optimization) |
| **Policy** | CNNPolicy + MISDOFeatureExtractor |
| **Observation** | `[3, 256, 256]` — harm mask, forest state, infrastructure |
| **Action** | `Discrete(65536)` — flattened (row, col) coordinate |
| **Episode** | 50 valid harvests (10×10 blocks) |

---

## MDP Formulation

### State Space

The observation is a 3-layer spatial tensor `[3, 256, 256]`:

| Layer | Name | Description | Dynamics |
|---|---|---|---|
| 0 | Final_Harm_Mask | Aggregated environmental risk | Static + edge effects |
| 1 | Forest_State | 1=intact, 0=harvested | Mutates each step |
| 2 | Infrastructure | 1=road/cleared, 0=untouched | Grows each step |

### Action Space

`Discrete(256 × 256) = Discrete(65536)` — each action selects a pixel coordinate as the centre of a 10×10 harvest block.

### Transition Dynamics

When action $(r, c)$ is executed:
1. **Contiguity check**: Block must overlap existing infrastructure (roads/clearings). Otherwise → rejected.
2. **Harvest**: Set `forest_state[r-5:r+5, c-5:c+5] = 0`
3. **Infrastructure update**: Set `infrastructure[r-5:r+5, c-5:c+5] = 1`
4. **Edge effects**: Newly exposed forest edges within 2px get harm multiplied by 1.2×

### Reward Structure

$$R = \begin{cases} R_{\text{base}} - \sum_{(i,j) \in \text{block}} H[i,j] & \text{if contiguous (valid harvest)} \\ -100 & \text{if not contiguous (rejected)} \end{cases}$$

where $R_{\text{base}} = 10.0$ and $H$ is the harm mask.

The agent is incentivised to:
- Harvest in low-harm areas (low $\sum H$)
- Maintain contiguity with existing infrastructure
- Avoid triggering excessive edge effects

### Episode Termination

After 50 valid harvests (quota met), the episode terminates.

---

## Policy Architecture: MISDOFeatureExtractor

A lightweight 3-layer strided-conv CNN that maps the 3-layer observation to a feature vector:

```
Input [B, 3, 256, 256]
  │
  ▼
Conv2d(3→32,   k=8, s=4)  + ReLU    → [B, 32, 63, 63]
Conv2d(32→64,  k=4, s=2)  + ReLU    → [B, 64, 30, 30]
Conv2d(64→128, k=3, s=2)  + ReLU    → [B, 128, 14, 14]
  │
  ▼
Flatten → [B, 25088]
Linear(25088→256) + ReLU             → [B, 256]
```

### Why This Design?

All heavy spatial reasoning (multi-modal fusion, non-linear aggregation, hard constraints) is done upstream by the perception and aggregator modules. The RL policy only needs to learn *spatial planning* — where to cut next given three clean summary layers. A lightweight CNN is sufficient.

---

## PPO Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Learning Rate | $3 \times 10^{-4}$ | Standard for PPO with CNN policies |
| n_steps | 128 | Rollout buffer length |
| batch_size | 64 | Mini-batch for gradient updates |
| n_epochs | 4 | SGD passes per rollout |
| γ (discount) | 0.99 | Long-horizon planning (50-step episodes) |
| λ (GAE) | 0.95 | Bias-variance tradeoff in advantage estimation |
| clip_range | 0.2 | PPO clipping for stable updates |

### PPO Objective

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the probability ratio
- $\hat{A}_t$ is the Generalised Advantage Estimation (GAE)
- $\epsilon = 0.2$ is the clipping range

GAE combines TD residuals across multiple timesteps:

$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}$$

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

---

## Environment Dynamics

### Edge Effects

When forest is cleared, adjacent remaining forest suffers increased environmental harm (wind exposure, moisture loss, microclimate disruption):

$$H'[i, j] = \begin{cases} H[i,j] \times 1.2 & \text{if } (i,j) \in \text{EdgeZone} \wedge \text{Forest}[i,j] = 1 \\ H[i,j] & \text{otherwise} \end{cases}$$

The edge zone is computed via binary dilation (2px, 5×5 structuring element) of newly cleared areas.

### Contiguity Constraint

Harvests must touch existing infrastructure. This models real-world forestry: access roads are needed for equipment. The initial infrastructure is a single vertical road on the left edge (column 0).

---

## Why PPO is Optimal

1. **Discrete spatial actions**: With 65,536 possible actions per step, PPO's stochastic policy naturally explores the action space via entropy bonus.
2. **Sparse rewards**: Only valid harvests yield positive reward; PPO's advantage estimation handles the credit assignment across long episodes.
3. **Stable training**: PPO's clipping prevents catastrophic policy updates — critical when the reward landscape is shaped by complex environmental interactions.
4. **Interpretable**: The learned policy can be visualised as a spatial decision sequence — valuable for forestry stakeholders.
