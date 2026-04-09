# Stack Cube Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 缩短 stack demo 的无效尾帧，进一步平滑 scripted 轨迹，并用更大的数据集和更长训练跑一轮新实验。

**Architecture:** 保持现有 `sim_stack_cube_scripted` 任务与 ACT 模型结构不变，只调整任务配置、scripted policy 动作生成与实验运行参数。录制端继续复用 EE rollout + joint replay 流程，训练端继续用双视角 ACT。

**Tech Stack:** Python 3.10, dm_control, MuJoCo, PyTorch, uv

---

### Task 1: Lock New Stack Config In Tests

**Files:**
- Modify: `tests/test_stack_cube.py`
- Modify: `constants.py`

- [ ] **Step 1: Add failing assertions for the new dataset size and episode length**

- [ ] **Step 2: Run `uv run python -m unittest tests.test_stack_cube` and confirm failure**

- [ ] **Step 3: Update `sim_stack_cube_scripted` config to `num_episodes=100` and `episode_len=260`**

- [ ] **Step 4: Re-run `uv run python -m unittest tests.test_stack_cube` and confirm pass**

### Task 2: Tighten Scripted Motion Smoothness

**Files:**
- Modify: `scripted_policy.py`
- Modify: `tests/test_stack_cube.py`

- [ ] **Step 1: Add a failing regression test that checks stack policy emits bounded per-step position/gripper changes**

- [ ] **Step 2: Run the targeted test and confirm it fails for the current policy**

- [ ] **Step 3: Add minimal smoothing to stack policy targets / outputs so phase transitions stop producing large spikes**

- [ ] **Step 4: Re-run the targeted test and confirm it passes**

### Task 3: Re-verify Dataset Quality

**Files:**
- Output: `data/sim_stack_cube_scripted`

- [ ] **Step 1: Record 100 scripted episodes with the new config**

- [ ] **Step 2: Run a small diagnostic script to measure action deltas and effective active horizon**

- [ ] **Step 3: Confirm the tail is materially shorter than the previous ~278 static frames**

### Task 4: Run Updated ACT Experiment

**Files:**
- Output: `checkpoint/stack_act_260_100ep`

- [ ] **Step 1: Train dual-camera ACT for 600 epochs on the new dataset**

- [ ] **Step 2: Evaluate the best checkpoint with `--temporal_agg`**

- [ ] **Step 3: Compare against `checkpoint/stack_act_opt_v2` and `checkpoint/stack_act_2cam`**

- [ ] **Step 4: Summarize whether shortening episodes plus smoother demos improved final success or only mid-stage rewards**
