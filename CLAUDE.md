# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ACT (Action Chunking with Transformers) is a robotics research implementation for bimanual manipulation tasks forked from tonyzhaozh/act. The project uses transformer-based models with VAE encoding to learn and reproduce complex manipulation behaviors from demonstrations.

You can use Deepwiki MCP to query about the Repo.

- **Language**: Python 3.8+
- **Framework**: PyTorch with MuJoCo + DM_Control for simulation
- **Real robot**: Requires [ALOHA](https://github.com/tonyzhaozh/aloha) hardware integration
- **Project website**: https://tonyzhao.github.io/aloha/

## Development Commands

### Environment Setup

```bash
uv sync
```

### Record Simulation Episodes

Generate scripted demonstration data for training:

```bash
uv run record_sim_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --dataset_dir <data save dir> \
    --num_episodes 50
```

Add `--onscreen_render` to see real-time rendering.

Available tasks: `sim_transfer_cube_scripted`, `sim_insertion_scripted`

### Train ACT Model

```bash
uv run imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir <ckpt dir> \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --seed 0
```

### Evaluate Policy

Add `--eval` flag to the training command. This loads the best validation checkpoint:

```bash
uv run imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir <ckpt dir> \
    --policy_class ACT ... \
    --eval
```

Additional evaluation flags:
- `--temporal_agg` - Enable temporal ensembling for smoother actions
- `--onscreen_render` - Show real-time rendering
- Videos are saved to `<ckpt_dir>`

### Visualize Episodes

```bash
uv run visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0
```

## Architecture

The codebase follows a layered architecture:

### Model Layer ([detr/](detr/))

- **[detr/models/detr_vae.py](detr/models/detr_vae.py)** - DETRVAE: Core transformer model using VAE encoding for action sequences
  - Encoder: Processes action sequences + robot state to learn latent representations
  - Decoder: Generates future actions conditioned on latent code and observations
  - Uses sinusoidal position embeddings and transformer encoder-decoder structure

- **[detr/models/cnnmlp.py](detr/models/cnnmlp.py)** - CNNMLP: Baseline CNN+MLP model

### Policy Layer ([policy.py](policy.py))

- **ACTPolicy** - Wrapper for DETRVAE model
  - Handles training with L1 loss + KL divergence loss
  - Manages inference with action chunking
  - Loads/saves checkpoints

- **CNNMLPPolicy** - Wrapper for baseline model

### Environment Layer

- **[sim_env.py](sim_env.py)** - Joint space control environments
  - Action space: `[left_arm_qpos(6), left_gripper(1), right_arm_qpos(6), right_gripper(1)]`
  - Observation space: Joint positions, velocities, camera images
  - Uses ViperX robot arm with custom gripper

- **[ee_sim_env.py](ee_sim_env.py)** - End-effector space control environments

### Training Layer ([imitate_episodes.py](imitate_episodes.py))

- Data loading and batching from HDF5 datasets
- Model training loop with validation
- Policy evaluation with episode rollouts

### Data Layer

- **[utils.py](utils.py)** - EpisodicDataset class for loading HDF5 episode data
  - Random timestep sampling for training batches
  - Data normalization utilities

- **[constants.py](constants.py)** - Task configurations and robot parameters
  - SIM_TASK_CONFIGS defines dataset paths, episode lengths, camera names
  - Robot-specific constants (gripper normalization, joint limits)

### Visualization Layer

- **[visualize_episodes.py](visualize_episodes.py)** - Generate videos from HDF5 datasets
- **[scripted_policy.py](scripted_policy.py)** - Scripted policies for data generation

## Key Design Patterns

### Action Chunking
ACT treats action prediction as a sequence-to-sequence problem. The model predicts a "chunk" of future actions (e.g., 100 timesteps) at once, which are then executed sequentially. This enables planning ahead and smoother trajectories.

### VAE Encoding
Action sequences are encoded into a latent space using a VAE. The latent code captures high-level action primitives, enabling the model to learn reusable behaviors.

### Episode Data Format
Episodes are stored as HDF5 files with normalized actions/observations. The dataset format is consistent across simulation and real-world data, enabling sim-to-real transfer.

## Training Tips

From the [ACT tuning tips document](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing):

- If policy is jerky or pauses mid-episode, train longer
- Success rate and smoothness can improve significantly after loss plateaus
- For real-world data, train for at least 5000 epochs or 3-4x the length after loss plateau

Expected success rates:
- Transfer cube: ~90%
- Insertion: ~50%
