# ACT 训练场景分析

本文档分析项目中自带的两个训练场景的定义和特性。

---

## 场景一：`sim_transfer_cube_scripted` (传输立方体)

**策略类**: `PickAndTransferPolicy` ([scripted_policy.py:67-104](../scripted_policy.py#L67-L104))

**配置**: [constants.py:6-11](../constants.py#L6-L11)

### 任务目标

双臂机器人协作，将立方体从右侧机械臂抓取并传递给左侧机械臂。

### 执行流程

| 时间步 | 右臂动作 | 左臂动作 |
|--------|------------------------------|------------------------------|
| 0-90 | 休眠 | 休眠 |
| 90-130 | 接近立方体上方 → 下移 | 保持不动 |
| 130-170 | 下移到位 → 闭合夹爪抓取 | 保持不动 |
| 170-220 | 抓取后移动到交接位置 | 移向交接位置并准备接收 |
| 220-310 | 到达交接位置 → 张开夹爪释放 | 到达交接位置 → 闭合夹爪抓取 |
| 310-360 | 右移撤离 | 左移撤离 |
| 360-400 | 保持姿态 | 保持姿态 |

### 关键代码位置

- 右臂轨迹定义: [scripted_policy.py:94-104](../scripted_policy.py#L94-L104)
- 左臂轨迹定义: [scripted_policy.py:85-92](../scripted_policy.py#L85-L92)
- 交接点位置: `meet_xyz = [0, 0.5, 0.25]`
- 抓取姿态: 右臂夹爪倾斜 -60°

### 关键特性

| 特性 | 说明 |
|------|------|
| **双臂协作** | 需要精确的空间和时间协调 |
| **交接点固定** | 交接位置固定在 `meet_xyz = [0, 0.5, 0.25]` |
| **抓取姿态** | 右臂夹爪相对于初始姿态倾斜 -60°（便于抓取立方体） |
| **相机配置** | 单一顶部视角 (`camera_names: ['top']`) |
| **Episode 长度** | 400 步 |
| **预期成功率** | ~90% |

---

## 场景二：`sim_insertion_scripted` (插入任务)

**策略类**: `InsertionPolicy` ([scripted_policy.py:107-149](../scripted_policy.py#L107-L149))

**配置**: [constants.py:20-25](../constants.py#L20-L25)

### 任务目标

双臂机器人协作完成 peg-in-hole（插孔）任务，左臂持插座（socket），右臂持插销（peg）进行精密对接。

### 执行流程

| 时间步 | 右臂动作 (持 peg) | 左臂动作 (持 socket) |
|--------|------------------------------|------------------------------|
| 0-120 | 休眠 | 休眠 |
| 120-170 | 接近 peg 上方 → 下移抓取 | 接近 socket 上方 → 下移抓取 |
| 170-220 | 闭合夹爪抓取 peg | 闭合夹爪抓取 socket |
| 220-285 | 移动到交接位置 | 移动到交接位置 |
| 285-340 | 精细调整插入位置 | 精细调整接收位置 |
| 340-400 | 保持插入姿态 | 保持接收姿态 |

### 关键代码位置

- 右臂轨迹定义: [scripted_policy.py:140-149](../scripted_policy.py#L140-L149)
- 左臂轨迹定义: [scripted_policy.py:130-138](../scripted_policy.py#L130-L138)
- 交接点位置: `meet_xyz = [0, 0.5, 0.15]`
- 高度差补偿: `lift_right = 0.00715`

### 关键特性

| 特性 | 说明 |
|------|------|
| **高精度要求** | 需要亚毫米级的位置精度 |
| **非对称姿态** | 右臂夹爪倾斜 -60°，左臂夹爪倾斜 +60° |
| **高度差补偿** | 右臂有 `lift_right = 0.00715` 的垂直偏移 |
| **交接点位置** | `meet_xyz = [0, 0.5, 0.15]` |
| **相机配置** | 单一顶部视角 (`camera_names: ['top']`) |
| **Episode 长度** | 400 步 |
| **预期成功率** | ~50% |

---

## 两个场景对比

| 特性 | Transfer Cube | Insertion |
|------|---------------|-----------|
| **任务类型** | 物品传递 | 精密装配 (peg-in-hole) |
| **协作复杂度** | 中等（抓取-传递-释放） | 高（需要对齐插孔） |
| **精度要求** | 较低 | 很高 |
| **预期成功率** | ~90% | ~50% |
| **Episode 长度** | 400 | 400 |
| **双臂姿态对称性** | 接近对称 | 非对称（不同倾角） |
| **难度级别** | 入门级 | 进阶级 |

---

## 场景通用架构

### BasePolicy 基类

两个场景策略都继承自 `BasePolicy` ([scripted_policy.py:12-64](../scripted_policy.py#L12-L64))，提供以下通用功能：

```python
class BasePolicy:
    def __init__(self, inject_noise=False)
    def generate_trajectory(self, ts_first)  # 子类实现
    def interpolate(curr_waypoint, next_waypoint, t)  # 航点插值
    def __call__(ts)  # 执行策略
```

### 轨迹生成机制

1. **Waypoint 系统**: 使用关键航点定义轨迹
2. **线性插值**: 航点之间进行位置、姿态和夹爪的线性插值
3. **开环执行**: 轨迹在第一步生成后全程开环执行，不根据观测调整

### Waypoint 数据结构

```python
{
    "t": 时间步,
    "xyz": [x, y, z]  # 位置
    "quat": [w, x, y, z]  # 四元数姿态
    "gripper": 0或1  # 0=闭合, 1=张开
}
```

### 噪声注入

可选 `inject_noise=True` 参数添加位置扰动（±0.01m），用于增强策略鲁棒性。

---

## 使用方式

### 生成训练数据

```bash
# 传输立方体场景
uv run record_sim_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --dataset_dir <data save dir> \
    --num_episodes 50

# 插入任务场景
uv run record_sim_episodes.py \
    --task_name sim_insertion_scripted \
    --dataset_dir <data save dir> \
    --num_episodes 50
```

### 训练模型

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

### 评估策略

```bash
uv run imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir <ckpt dir> \
    --policy_class ACT ... \
    --eval
```

---

## 训练建议

根据 [ACT tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing)：

- 如果策略动作抖动或中途暂停，需要训练更长时间
- 成功率和平滑度可能在损失平台期后显著提升
- 对于真实数据，至少训练 5000 epochs 或损失平台期后 3-4 倍时长
- Transfer Cube 预期成功率 ~90%
- Insertion 预期成功率 ~50%
