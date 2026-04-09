# Sim Stack Cube Design

**Goal:** 在 ACT 仓库中新增一个双臂协作的简化堆叠任务 `sim_stack_cube_scripted`：左臂先抓取红色方块并放置到中央后持续轻扶，右臂抓取蓝色方块并完成堆叠，最终可用于 demonstration 录制、ACT 训练与评估。

**Chosen Approach:** 采用半闭环脚本策略。环境继续复用仓库现有的 joint-control 与 ee-control 双环境结构，但 scripted policy 不再一次性生成全程开环轨迹，而是在关键阶段基于当前 `env_state` 动态更新目标位姿，重点增强左臂扶持红块与右臂慢速下放蓝块时的稳定性。

## Scope

- 新增堆叠任务 XML 场景
- 新增 `sim_env.py` / `ee_sim_env.py` 的 stack task
- 新增半闭环 `StackCubePolicy`
- 新增堆叠任务的 pose 采样、录制、训练、评估接入
- 完成从场景搭建、调试、跑实验到结果分析的全流程

不做的内容：

- 不改 ACT 网络结构
- 不引入额外传感器或新的 observation 字段到训练输入
- 不做完整的“双臂都参与搬运和放置多个阶段”的高复杂版本

## Scene Design

新增两份 MuJoCo XML：

- `assets/bimanual_viperx_stack_cube.xml`
- `assets/bimanual_viperx_ee_stack_cube.xml`

设计要求：

- 保留现有双臂桌面场景结构
- 引入两个自由体方块：
  - `red_box`：左侧初始区域，供左臂抓取
  - `blue_box`：右侧初始区域，供右臂抓取
- 两个方块尺寸沿用 transfer cube：`0.02 0.02 0.02`
- 中央预留堆叠目标区，不单独建静态托盘，直接放置在桌面
- 两个方块都使用 `free joint`
- 蓝块与红块颜色明显区分，方便相机学习

## Environment Design

### Task Registration

新增任务：

- `sim_stack_cube_scripted`

沿用和原仓库一致的任务配置入口 `SIM_TASK_CONFIGS`。

### State and Reset

`env_state` 采用两个方块 pose 的拼接：

- `red_box_pose`: 7 维
- `blue_box_pose`: 7 维
- 合计 14 维

reset 时：

- 左右臂复位到标准初始姿态
- 红块采样到左侧可抓取区域
- 蓝块采样到右侧可抓取区域

在 joint sim 中，replay 时使用 episode 首帧保存的 14 维物体 pose 保证与 ee rollout 一致。

### Reward

采用分阶段奖励，保留与原任务一致的 `max_reward = 4`：

- `1`: 左臂成功接触/抓住红块
- `2`: 红块已被搬到中央并稳定放置，左臂仍允许轻扶
- `3`: 右臂成功抓住蓝块
- `4`: 蓝块成功堆叠到红块上且右臂已释放

成功判定采用“中等严格度”：

- 蓝块中心高度高于红块中心高度加一个方块高度附近的阈值
- 蓝块与红块在 `xy` 平面的距离小于阈值
- 右臂不再接触蓝块
- 左臂允许继续接触红块

## Policy Design

新增 `StackCubePolicy`，采用半闭环状态机。

### Left Arm

- 接近红块
- 抓取红块
- 抬起红块
- 移动到中央
- 放置红块
- 轻扶红块并小范围跟踪其位置

### Right Arm

- 等待左臂进入稳定扶持阶段
- 接近蓝块
- 抓取蓝块
- 抬起蓝块
- 对齐红块上方
- 慢速下降
- 释放蓝块
- 撤离

### Semi-Closed-Loop Logic

关键阶段每步读取：

- 当前红块 pose
- 当前蓝块 pose
- 左右末端 mocap pose

右臂堆叠目标基于红块当前位置动态更新，左臂扶持姿态允许做厘米级跟踪修正。控制器采用小步长笛卡尔伺服而非整段轨迹插值。

## Data and Training

录制链路保持原样：

- EE 环境用脚本策略 rollout
- 将得到的 joint trajectory replay 到 joint 环境
- 记录图像、`qpos`、`qvel`、`action`

训练侧保持 ACT 默认结构：

- `state_dim` 仍为 14，对应机器人 `qpos`
- 图像输入与 `transfer_cube` 一致
- 第一轮超参直接复用 `sim_transfer_cube_scripted`

`env_state` 仅用于环境与脚本策略，不直接作为 ACT 输入。

## Debug and Validation Strategy

按以下顺序调试：

1. XML 场景可加载，两个方块显示正确
2. `sim_env` / `ee_sim_env` 可 `reset` 和 `step`
3. 只调左臂搬运红块
4. 固定红块中央后只调右臂堆蓝块
5. 合并完整双臂状态机
6. 小规模录制 2 到 5 条 episode 检查 replay 成功率
7. 跑 `num_epochs=1` smoke training
8. 再录 50 条正式数据并训练评估

## Risks

- 方块接触参数导致堆叠时弹飞
- 左臂轻扶过紧或过松
- 右臂下放速度过快
- EE rollout 与 joint replay 行为不一致
- 随机化范围过大导致 scripted policy 成功率下降

## Acceptance Criteria

- 新任务场景可以稳定加载
- 两个环境版本均可运行
- scripted policy 在 EE rollout 中达到高成功率
- replay 后仍保持稳定成功率
- demonstration 数据可用于 ACT 训练
- 训练、评估和视频导出链路完整跑通
- 输出成功/失败案例与原因分析
