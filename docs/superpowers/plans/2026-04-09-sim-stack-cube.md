# Sim Stack Cube Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 ACT 仓库中新增 `sim_stack_cube_scripted` 任务，并完整跑通场景搭建、脚本数据录制、ACT 训练、评估与结果分析。

**Architecture:** 复用现有 `transfer_cube` / `insertion` 的双环境结构，在 MuJoCo XML、`sim_env.py`、`ee_sim_env.py`、`scripted_policy.py`、`record_sim_episodes.py`、`utils.py`、`constants.py`、`imitate_episodes.py` 中新增 stack task 支持。脚本策略采用半闭环状态机，在关键阶段根据当前物体 pose 动态更新目标末端位姿。

**Tech Stack:** Python 3.10+, dm_control, MuJoCo, PyTorch, h5py, uv

---

### Task 1: Add Stack Cube Scene Assets

**Files:**
- Create: `assets/bimanual_viperx_stack_cube.xml`
- Create: `assets/bimanual_viperx_ee_stack_cube.xml`
- Reference: `assets/bimanual_viperx_transfer_cube.xml`
- Reference: `assets/bimanual_viperx_ee_transfer_cube.xml`

- [ ] **Step 1: Add a regression check placeholder**

Run: `uv run python - <<'PY'\nfrom dm_control import mujoco\nmujoco.Physics.from_xml_path('assets/bimanual_viperx_stack_cube.xml')\nPY`

Expected: FAIL because the XML file does not exist yet.

- [ ] **Step 2: Create the joint-control stack XML**

Create `assets/bimanual_viperx_stack_cube.xml` by cloning the transfer cube scene and adding a second free-body cube named `blue_box` on the right side.

- [ ] **Step 3: Create the EE-control stack XML**

Create `assets/bimanual_viperx_ee_stack_cube.xml` by cloning the EE transfer cube scene and adding the same second cube plus matching keyframe object state.

- [ ] **Step 4: Verify both XML files load**

Run: `uv run python - <<'PY'\nfrom dm_control import mujoco\nfor path in ['assets/bimanual_viperx_stack_cube.xml', 'assets/bimanual_viperx_ee_stack_cube.xml']:\n    physics = mujoco.Physics.from_xml_path(path)\n    print(path, physics.data.qpos.shape[0])\nPY`

Expected: both files load successfully and report a valid `qpos` size.

### Task 2: Register Stack Task and Sampling Utilities

**Files:**
- Modify: `constants.py`
- Modify: `utils.py`

- [ ] **Step 1: Write a failing sampling test**

Run: `uv run python - <<'PY'\nfrom utils import sample_stack_pose\nprint(sample_stack_pose())\nPY`

Expected: FAIL because `sample_stack_pose` does not exist.

- [ ] **Step 2: Register the new task config**

Add `sim_stack_cube_scripted` to `SIM_TASK_CONFIGS` with dataset path, episode length, number of episodes, and camera names aligned with transfer cube defaults.

- [ ] **Step 3: Implement stack pose sampling**

Add `sample_stack_pose()` that returns concatenated red and blue free-joint poses sampled in left/right workspace ranges.

- [ ] **Step 4: Verify the task config and sampler**

Run: `uv run python - <<'PY'\nfrom constants import SIM_TASK_CONFIGS\nfrom utils import sample_stack_pose\nprint(SIM_TASK_CONFIGS['sim_stack_cube_scripted'])\npose = sample_stack_pose()\nprint(pose.shape, pose)\nPY`

Expected: task config prints correctly and the sampled pose has shape `(14,)`.

### Task 3: Implement Stack Environments

**Files:**
- Modify: `sim_env.py`
- Modify: `ee_sim_env.py`

- [ ] **Step 1: Write a failing environment instantiation test**

Run: `uv run python - <<'PY'\nfrom sim_env import make_sim_env\nfrom ee_sim_env import make_ee_sim_env\nmake_sim_env('sim_stack_cube_scripted')\nmake_ee_sim_env('sim_stack_cube_scripted')\nPY`

Expected: FAIL because stack env branches are not implemented yet.

- [ ] **Step 2: Implement stack task support in joint sim**

Add XML loading, reset logic, env-state extraction, and reward computation for `StackCubeTask`.

- [ ] **Step 3: Implement stack task support in EE sim**

Add XML loading, reset logic, env-state extraction, and reward computation for `StackCubeEETask`.

- [ ] **Step 4: Verify environment reset and observation contract**

Run: `uv run python - <<'PY'\nfrom sim_env import make_sim_env\nfrom ee_sim_env import make_ee_sim_env\nfor maker in [make_sim_env, make_ee_sim_env]:\n    env = maker('sim_stack_cube_scripted')\n    ts = env.reset()\n    print(maker.__name__, ts.observation['qpos'].shape, ts.observation['env_state'].shape)\nPY`

Expected: both environments reset successfully with `qpos.shape == (14,)` and `env_state.shape == (14,)`.

### Task 4: Implement Semi-Closed-Loop Stack Policy

**Files:**
- Modify: `scripted_policy.py`

- [ ] **Step 1: Write a failing policy import test**

Run: `uv run python - <<'PY'\nfrom scripted_policy import StackCubePolicy\nprint(StackCubePolicy)\nPY`

Expected: FAIL because `StackCubePolicy` does not exist.

- [ ] **Step 2: Implement the policy state machine**

Add a policy that uses per-step target updates for left-arm support and right-arm stacking rather than precomputed full trajectories.

- [ ] **Step 3: Wire policy testing entry points**

Extend `test_policy` so the stack task uses `make_ee_sim_env('sim_stack_cube_scripted')` and `StackCubePolicy`.

- [ ] **Step 4: Verify a short scripted rollout**

Run: `uv run python - <<'PY'\nfrom constants import SIM_TASK_CONFIGS\nfrom ee_sim_env import make_ee_sim_env\nfrom scripted_policy import StackCubePolicy\nenv = make_ee_sim_env('sim_stack_cube_scripted')\nts = env.reset()\npolicy = StackCubePolicy(False)\nfor _ in range(min(50, SIM_TASK_CONFIGS['sim_stack_cube_scripted']['episode_len'])):\n    ts = env.step(policy(ts))\nprint('reward', ts.reward)\nPY`

Expected: rollout runs without crashing and produces numeric rewards.

### Task 5: Integrate Data Recording and Evaluation

**Files:**
- Modify: `record_sim_episodes.py`
- Modify: `imitate_episodes.py`

- [ ] **Step 1: Write a failing recorder selection test**

Run: `uv run python - <<'PY'\nfrom record_sim_episodes import main\nargs = {'task_name': 'sim_stack_cube_scripted', 'dataset_dir': 'data/tmp_stack', 'num_episodes': 1, 'onscreen_render': False}\nmain(args)\nPY`

Expected: FAIL because the new task is not selectable yet.

- [ ] **Step 2: Add stack policy selection to data recording**

Update the task-to-policy mapping for `sim_stack_cube_scripted`.

- [ ] **Step 3: Add stack reset sampling to evaluation**

Update evaluation rollout setup in `imitate_episodes.py` to use `sample_stack_pose()` when evaluating the new task.

- [ ] **Step 4: Verify one-episode recording works**

Run: `rm -rf data/tmp_stack && uv run record_sim_episodes.py --task_name sim_stack_cube_scripted --dataset_dir data/tmp_stack --num_episodes 1`

Expected: one HDF5 episode is produced successfully.

### Task 6: Smoke-Test Training Pipeline

**Files:**
- Modify if needed: any file touched above

- [ ] **Step 1: Validate the recorded dataset format**

Run: `uv run python - <<'PY'\nimport h5py\nroot = h5py.File('data/tmp_stack/episode_0.hdf5', 'r')\nprint(root['/observations/qpos'].shape)\nprint(root['/action'].shape)\nprint(root['/observations/images/top'].shape)\nroot.close()\nPY`

Expected: shapes are valid and consistent with ACT expectations.

- [ ] **Step 2: Run a one-epoch ACT smoke test**

Run: `rm -rf checkpoint/stack_smoke && uv run imitate_episodes.py --task_name sim_stack_cube_scripted --ckpt_dir checkpoint/stack_smoke --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 2 --dim_feedforward 3200 --num_epochs 1 --lr 1e-5 --seed 0`

Expected: training completes, saves stats and a checkpoint.

- [ ] **Step 3: Run smoke evaluation**

Run: `uv run imitate_episodes.py --task_name sim_stack_cube_scripted --ckpt_dir checkpoint/stack_smoke --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 2 --dim_feedforward 3200 --num_epochs 1 --lr 1e-5 --seed 0 --eval`

Expected: evaluation completes and writes a result file.

### Task 7: Run Full Experiment and Collect Evidence

**Files:**
- Output: `data/sim_stack_cube_scripted`
- Output: `checkpoint/stack_act`

- [ ] **Step 1: Record the full demonstration dataset**

Run: `rm -rf data/sim_stack_cube_scripted && uv run record_sim_episodes.py --task_name sim_stack_cube_scripted --dataset_dir data/sim_stack_cube_scripted --num_episodes 50`

Expected: 50 episodes saved with a stable scripted success rate.

- [ ] **Step 2: Train the ACT policy**

Run: `rm -rf checkpoint/stack_act && uv run imitate_episodes.py --task_name sim_stack_cube_scripted --ckpt_dir checkpoint/stack_act --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0`

Expected: training finishes and writes `policy_best.ckpt`.

- [ ] **Step 3: Evaluate the trained policy**

Run: `uv run imitate_episodes.py --task_name sim_stack_cube_scripted --ckpt_dir checkpoint/stack_act --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --eval`

Expected: evaluation produces success-rate statistics and rollout videos.

### Task 8: Summarize Results and Failure Modes

**Files:**
- Output: logs, checkpoints, result files, videos

- [ ] **Step 1: Inspect the full experiment artifacts**

Run: `ls checkpoint/stack_act && cat checkpoint/stack_act/result_policy_best.txt`

Expected: checkpoint, result summary, and videos exist.

- [ ] **Step 2: Review representative episodes**

Run: `uv run visualize_episodes.py --dataset_dir data/sim_stack_cube_scripted --episode_idx 0`

Expected: the saved episode can be visualized without format issues.

- [ ] **Step 3: Produce a concise analysis**

Summarize scripted policy stability, training behavior, evaluation success rate, and dominant failure modes from saved outputs.
