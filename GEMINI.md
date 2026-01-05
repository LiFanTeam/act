# ACT: Action Chunking with Transformers

## Project Overview
ACT (Action Chunking with Transformers) is an imitation learning algorithm for robotic manipulation. This repository contains the implementation along with simulated environments (Transfer Cube and Bimanual Insertion). It leverages a modified DETR (Detection Transformer) architecture to predict action sequences from visual observations.

## Key Files & Directories
- **`imitate_episodes.py`**: Main script for training and evaluating the ACT policy.
- **`record_sim_episodes.py`**: Script to generate scripted/human demonstration data in simulation.
- **`visualize_episodes.py`**: Tool to visualize episodes from HDF5 datasets.
- **`policy.py`**: Defines the ACT policy architecture.
- **`detr/`**: Contains the modified DETR model definitions.
- **`sim_env.py` / `ee_sim_env.py`**: Mujoco + DM_Control simulation environments (Joint space vs. End-Effector space).
- **`constants.py`**: Shared constants and configuration.
- **`assets/`**: 3D assets (XML, STL) for the simulation environments.

## Setup & Installation

### Option 1: Conda (Recommended in README)
The project provides a `conda_env.yaml` for setting up the environment.

```bash
conda env create -f conda_env.yaml
conda activate aloha
cd detr && pip install -e .
```

### Option 2: UV (Modern Python Tooling)
The project also includes `pyproject.toml` and `uv.lock`, allowing for faster setup with `uv`.

```bash
uv sync
source .venv/bin/activate
cd detr && pip install -e .
```

## Usage

### 1. Data Generation (Simulation)
Generate scripted demonstration episodes for a task (e.g., `sim_transfer_cube_scripted`).

```bash
python record_sim_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --dataset_dir data/sim_transfer_cube_scripted \
    --num_episodes 50
```
*   Add `--onscreen_render` to watch the collection in real-time.

### 2. Visualization
Visualize a collected episode to verify data quality.

```bash
python visualize_episodes.py \
    --dataset_dir data/sim_transfer_cube_scripted \
    --episode_idx 0
```

### 3. Training
Train the ACT policy. Adjust parameters as needed.

```bash
python imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir checkpoint/my_experiment \
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

### 4. Evaluation
Evaluate a trained policy by adding the `--eval` flag. This typically loads the best checkpoint from `ckpt_dir`.

```bash
python imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir checkpoint/my_experiment \
    --policy_class ACT \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --seed 0 \
    --eval
```
*   **Note**: `temporal_agg` (Temporal Ensembling) can be enabled during evaluation for smoother control.

## Development Notes
- **Dependencies**: The project relies heavily on `mujoco`, `dm_control`, and `torch`. Ensure these are correctly installed and compatible with your CUDA version.
- **Configuration**: Task-specific configurations (dataset directory, episode length, etc.) are found in `constants.py` (for sim) or `aloha_scripts/constants.py` (for real robot, if applicable).
- **Code Style**: The codebase uses standard Python conventions. No explicit linter/formatter config was observed, so follow the existing style.
