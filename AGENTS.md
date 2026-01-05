# Repository Guidelines

This repository contains code for training and evaluating Action Chunking with Transformers (ACT) policies in DM_Control simulated environments. The main focus is on imitation learning from scripted demonstrations for robotic manipulation tasks.


## User Instruction
- Use Chinese to communicate with user.
- Provide clear and concise explanations, the user is learning about the repository and algorithms.

## Project Structure & Module Organization
- `imitate_episodes.py`: entry point for training and evaluation runs.
- `policy.py`, `detr/`: ACT and CNN-MLP policy definitions plus DETR-based model components.
- `sim_env.py` / `ee_sim_env.py`: DM_Control environments (joint-space vs end-effector control); `scripted_policy.py` holds scripted demos.
- `record_sim_episodes.py`, `visualize_episodes.py`: data collection and visualization utilities; `constants.py` and `utils.py` share configuration helpers.
- `assets/`: MuJoCo XML/STL assets; `data/`: datasets and generated rollouts; `checkpoint/`: saved models/videos; `docs/`: learning guides.
- Tooling: `pyproject.toml` + `uv.lock` for uv-based installs; `conda_env.yaml` for the conda workflow.

## Build, Test, and Development Commands
- Install deps with uv (fast path): `uv sync` (uses CUDA 11.8 torch index when available).
- Collect scripted rollouts: `uv run record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir data/sim_transfer_cube_scripted --num_episodes 10` (add `--onscreen_render` to watch).
- Inspect data: `uv run visualize_episodes.py --dataset_dir data/sim_transfer_cube_scripted --episode_idx 0`.
- Train ACT: `uv run imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir checkpoint/cube_act --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0`; append `--eval` to load the best checkpoint, `--temporal_agg` for ensembling.
- Conda fallback: `conda env create -f conda_env.yaml && conda activate aloha`, then run the same scripts with `python`.
- Quick smoke check after edits: `uv run python -m compileall .` to ensure files import cleanly.

## Coding Style & Naming Conventions
- Python 3.10+; follow PEP 8 with 4-space indents and ~88–100 character lines.
- Functions/variables use `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE_CASE` (see `constants.py`).
- Keep tensor device/dtype handling explicit; mirror image normalization patterns from `policy.py` when adding new inputs.
- Document non-obvious tensor shapes, seeding, and environment flags; update both `pyproject.toml`/`uv.lock` and `conda_env.yaml` when adding dependencies.
- Use augment context engine mcp to fetch code snippets for better codespace understanding.

## Testing Guidelines
- No formal unit test suite; validate changes by running a short `record_sim_episodes.py` + `visualize_episodes.py` cycle and a small `imitate_episodes.py --num_epochs 1 --batch_size 2` smoke.
- For model or environment changes, log training/validation metrics and verify checkpoints land under `checkpoint/<experiment>`; attach brief notes or video paths when rollout behavior changes.

## Commit & Pull Request Guidelines
- Use short, descriptive commit titles (imperative/verb-led where possible); keep subjects ≤72 characters and include scope (e.g., `record_sim: add cube scripted tweak`). English or Chinese is fine if clear.
- PRs should include a summary of behavior changes, commands run, before/after metrics or screenshots/video paths, any new flags/defaults, and linked issues. Call out backward-incompatible config or dependency changes explicitly.
