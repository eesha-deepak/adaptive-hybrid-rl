# Adaptive Hybrid Reinforcement Learning

Implementation of adaptive hybrid RL agents that switch between model-based planning (PETS) and model-free control (SAC) based on ensemble disagreement.

## Overview

This repository contains code for our research on when and how to combine model-based and model-free reinforcement learning. We implement a hybrid agent that uses calibrated ensemble disagreement to decide when to trust learned dynamics models versus falling back to robust model-free policies.

**Key results:**
- Contact frequency predicts model-based RL failure (100× difference between smooth and contact-rich environments)
- Adaptive switching adjusts model usage from 92% (smooth tasks) to 23% (contact-rich tasks)
- Soft switching (weighted blending) outperforms hard switching (binary selection) by eliminating action discontinuities

## Installation

```bash
git clone https://github.com/eesha-deepak/adaptive-hybrid-rl.git
cd adaptive-hybrid-rl
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- Gymnasium
- NumPy, Matplotlib

## Repository Structure

```
hybrid-rl/
├── hybrid/              # Hybrid agent implementation
│   ├── agent.py        # Main HybridAgent class
│   ├── gate.py         # Switching gate with threshold calibration
│   ├── planner_pets.py # PETS model-based planner
│   └── replay_buffer.py # Shared replay buffer
├── sac.py              # SAC implementation
├── train_all_envs.py   # Training script
└──analyze_environments.py # Environment characterization
```

## Usage

### Training Arguments

```bash
python train_all_envs.py [OPTIONS]
```

**Options:**
- `--env`: Environment name (e.g., `Swimmer-v5`, `Hopper-v5`)
- `--mode`: Training mode (`sac`, `pets`, `hybrid`)
- `--switching`: Switching type (`hard`, `soft`) - only for hybrid mode
- `--steps`: Number of training steps (default: 100000)
- `--threshold`: Initial disagreement threshold (default: 0.08)
- `--seed`: Random seed (default: 0)

### Example Commands

```bash
# Hard switching on InvertedPendulum
python train_all_envs.py --env InvertedPendulum-v5 --mode hybrid --switching hard

# Soft switching on Hopper
python train_all_envs.py --env Hopper-v5 --mode hybrid --switching soft --steps 100000

# SAC baseline on multiple environments
python train_all_envs.py --env Swimmer-v5 Hopper-v5 Ant-v5 --mode sac
```

## Hyperparameters

### SAC
- Learning rate: 3e-4
- Batch size: 256
- Target entropy: -0.5 × action_dim
- UTD ratio: 1→5 (adaptive ramp)
- Replay buffer: 1M transitions

### PETS
- Ensemble size: 5 models
- Hidden layers: [200, 200, 200]
- Planning horizon: 15 steps
- CEM candidates: 500
- CEM elite fraction: 0.1

### Hybrid
- Calibration quantile: 0.9
- Threshold range: 0.05-0.25 (environment-dependent)
- Calibration frequency: Every 5000 steps

## Environments Tested

We evaluated on 9 continuous control environments:

**Smooth dynamics:**
- Pendulum-v1
- InvertedPendulum-v5
- Reacher-v5
- Swimmer-v5
- Pushing2D-v1

**Contact-rich locomotion:**
- Hopper-v5
- Walker2d-v5
- HalfCheetah-v5
- Ant-v5

## Environment Characterization

Compute contact frequency and other metrics:
```bash
python analyze_environments.py --env Swimmer-v5 --episodes 100
```

Metrics computed:
- **Contact frequency**: Proportion of velocity discontinuities
- **Smoothness score**: Inverse local Lipschitz constant
- **Reward variance**: Landscape complexity

## Implementation Notes

### Key Design Decisions

1. **Disagreement on SAC actions**: We compute ensemble disagreement using the model-free policy's actions to avoid feedback loops where model-based decisions make states appear more predictable.

2. **Unified replay buffer**: Both agents share the same replay buffer to prevent "stale buffer" problems where agents only learn from states where they were active.

3. **Threshold calibration**: Per-environment calibration using empirical disagreement distributions is critical. Generic thresholds lead to degenerate behavior (100% MB or 0% MB usage).

4. **Hard vs. soft switching**: Hard switching provides interpretability but causes action discontinuities. Soft switching eliminates discontinuities through weighted blending but requires 2× computation.
