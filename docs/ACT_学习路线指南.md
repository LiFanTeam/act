# ACT: Action Chunking with Transformers 学习路线指南

## 概述

ACT（Action Chunking with Transformers）是一个用于机器人模仿学习的深度学习框架，它使用Transformer架构来预测动作序列。本指南将帮助您从零开始学习ACT，包括环境搭建、数据收集、模型训练和评估。

## 学习路线概览

### 阶段一：基础准备（1-2天）
1. 理解ACT的基本概念和原理
2. 搭建开发环境
3. 熟悉项目结构

### 阶段二：环境搭建（1-2天）
1. 安装依赖包
2. 配置模拟环境
3. 验证安装

### 阶段三：数据收集与处理（2-3天）
1. 生成演示数据
2. 数据可视化
3. 理解数据格式

### 阶段四：模型训练（3-5天）
1. 训练第一个ACT模型
2. 监控训练过程
3. 理解超参数

### 阶段五：评估与调优（2-3天）
1. 评估模型性能
2. 调优超参数
3. 理解常见问题

### 阶段六：进阶应用（可选，3-7天）
1. 应用到真实机器人
2. 自定义任务
3. 模型改进

---

## 详细学习步骤

### 阶段一：基础准备

#### 1.1 理解ACT核心概念
- **Transformer架构**：理解自注意力机制和编码器-解码器结构
- **动作分块（Action Chunking）**：一次性预测多个时间步的动作
- **条件变分自编码器（CVAE）**：用于动作生成的概率模型
- **模仿学习**：从演示数据中学习策略

#### 1.2 学习资源
- 阅读论文：[Action Chunking with Transformers](https://arxiv.org/abs/2305.03053)
- 观看项目演示视频
- 阅读官方文档和代码注释

#### 1.3 熟悉项目结构
```
act/
├── imitate_episodes.py      # 主训练和评估脚本
├── policy.py                # 策略适配器
├── detr/                    # 模型定义（基于DETR修改）
├── sim_env.py               # 模拟环境（关节空间控制）
├── ee_sim_env.py            # 模拟环境（末端执行器空间控制）
├── scripted_policy.py       # 脚本策略
├── record_sim_episodes.py   # 数据收集脚本
├── visualize_episodes.py    # 数据可视化
├── constants.py             # 常量定义
├── utils.py                 # 工具函数
├── assets/                  # 机器人模型和场景
└── conda_env.yaml           # 环境配置文件
```

### 阶段二：环境搭建

#### 2.1 使用Conda创建环境
```bash
# 方法1：使用conda_env.yaml（推荐）
conda env create -f conda_env.yaml
conda activate aloha

# 方法2：手动安装（如果yaml文件有问题）
conda create -n aloha python=3.9
conda activate aloha
pip install torch==2.0.0 torchvision==0.15.0
pip install pyquaternion pyyaml rospkg pexpect
pip install mujoco==2.3.3 dm_control==1.0.9
pip install opencv-python matplotlib einops packaging h5py ipython
```

#### 2.2 安装ACT模型包
```bash
cd act/detr
pip install -e .
```

#### 2.3 验证安装
```bash
# 检查Python环境
python -c "import torch; print(torch.__version__)"
python -c "import mujoco; print(mujoco.__version__)"

# 检查ACT包
python -c "from detr.main import build_ACT_model_and_optimizer; print('ACT包安装成功')"
```

### 阶段三：数据收集与处理

#### 3.1 理解数据格式
ACT使用HDF5格式存储演示数据，包含：
- `observations/qpos`：机器人关节位置
- `observations/images`：相机图像
- `actions`：动作序列
- `rewards`：奖励信号

#### 3.2 生成演示数据
```bash
# 生成转移立方体任务的演示数据
python record_sim_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --dataset_dir ./data/transfer_cube \
    --num_episodes 10 \
    --onscreen_render  # 可选：实时渲染

# 生成插入任务的演示数据
python record_sim_episodes.py \
    --task_name sim_insertion_scripted \
    --dataset_dir ./data/insertion \
    --num_episodes 10
```

#### 3.3 可视化数据
```bash
# 查看第一个演示视频
python visualize_episodes.py \
    --dataset_dir ./data/transfer_cube \
    --episode_idx 0
```

#### 3.4 理解数据加载过程
查看`utils.py`中的`EpisodicDataset`类，理解：
- 如何加载HDF5数据
- 数据标准化处理
- 批次生成逻辑

### 阶段四：模型训练

#### 4.1 理解ACT模型架构
ACT模型包含以下组件：
1. **骨干网络（ResNet）**：提取图像特征
2. **Transformer编码器**：编码状态和动作序列
3. **Transformer解码器**：生成动作预测
4. **VAE潜在空间**：学习动作分布

#### 4.2 第一次训练
```bash
# 训练转移立方体任务
python imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir ./checkpoints/transfer_cube \
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

#### 4.3 关键超参数说明
- `--kl_weight`：KL散度权重，控制正则化强度
- `--chunk_size`：动作分块大小，一次性预测的动作数
- `--hidden_dim`：Transformer隐藏层维度
- `--dim_feedforward`：前馈网络维度
- `--num_epochs`：训练轮数
- `--lr`：学习率

#### 4.4 监控训练过程
训练过程中会输出：
- 训练损失和验证损失
- KL散度值
- L1损失值
- 每100轮保存一次检查点

#### 4.5 查看训练曲线
训练结束后，在检查点目录中会生成：
- `train_val_loss_seed_0.png`：损失曲线
- `train_val_l1_seed_0.png`：L1损失曲线
- `train_val_kl_seed_0.png`：KL散度曲线

### 阶段五：评估与调优

#### 5.1 评估模型
```bash
# 使用最佳检查点进行评估
python imitate_episodes.py \
    --eval \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir ./checkpoints/transfer_cube \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --seed 0 \
    --onscreen_render  # 可选：实时渲染
```

#### 5.2 理解评估指标
- **成功率**：达到最大奖励的回合比例
- **平均回报**：所有回合的平均奖励
- **视频记录**：每个评估回合的视频

#### 5.3 常见问题与调优

##### 问题1：动作抖动或不连贯
**解决方案**：
- 增加训练轮数（5000+轮）
- 降低学习率（1e-6）
- 增加KL权重（20-50）

##### 问题2：模型在任务中间停顿
**解决方案**：
- 增加`chunk_size`（200-500）
- 启用时间聚合（`--temporal_agg`）
- 增加数据量

##### 问题3：过拟合
**解决方案**：
- 增加数据增强
- 使用dropout
- 早停策略

#### 5.4 调优示例
```bash
# 更长时间的训练
python imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir ./checkpoints/transfer_cube_tuned \
    --policy_class ACT \
    --kl_weight 20 \
    --chunk_size 200 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 5000 \
    --lr 5e-6 \
    --seed 0 \
    --temporal_agg  # 启用时间聚合
```

### 阶段六：进阶应用

#### 6.1 应用到真实机器人
需要安装[ALOHA项目](https://github.com/tonyzhaozh/aloha)：
1. 搭建ALOHA硬件系统
2. 收集真实机器人演示数据
3. 调整超参数（需要更长时间训练）

#### 6.2 自定义任务
1. **修改环境**：在`sim_env.py`中添加新任务
2. **修改脚本策略**：在`scripted_policy.py`中实现新策略
3. **调整常量**：在`constants.py`中配置新任务参数

#### 6.3 模型改进思路
1. **多模态输入**：添加触觉、力觉传感器
2. **分层策略**：结合高层规划和底层控制
3. **在线适应**：在部署时进行微调
4. **数据增强**：增加模拟到真实的迁移能力

## 实践项目建议

### 项目1：基础掌握（1-2周）
- 在模拟环境中训练ACT完成转移立方体任务
- 达到90%以上的成功率
- 理解所有超参数的影响

### 项目2：任务扩展（2-3周）
- 实现一个新的简单任务（如推箱子）
- 收集演示数据并训练模型
- 分析模型在新任务上的表现

### 项目3：性能优化（3-4周）
- 尝试不同的骨干网络（ResNet34, ResNet50）
- 实现数据增强策略
- 对比ACT与CNNMLP的性能差异

### 项目4：真实部署（4-8周，需要硬件）
- 在ALOHA机器人上部署ACT
- 收集真实世界演示数据
- 解决模拟到真实的迁移问题

## 学习资源

### 官方资源
- [项目网站](https://tonyzhaozh.github.io/aloha/)
- [GitHub仓库](https://github.com/tonyzhaozh/act)
- [调优指南](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit)

### 相关论文
1. [Action Chunking with Transformers](https://arxiv.org/abs/2305.03053)
2. [DETR: End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
3. [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)

### 学习社区
- GitHub Issues：报告问题和寻求帮助
- Robotics相关论坛和社区
- 学术会议（CoRL, ICRA, RSS）

## 故障排除

### 常见错误1：Mujoco许可证问题
```bash
# 设置Mujoco许可证密钥
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
```

### 常见错误2：CUDA内存不足
```bash
# 减小批次大小
--batch_size 4

# 减小图像分辨率
# 修改constants.py中的图像尺寸
```

### 常见错误3：依赖包版本冲突
```bash
# 创建干净的环境
conda env remove -n aloha
conda env create -f conda_env.yaml
```

## 总结

学习ACT需要循序渐进，从理解基本概念开始，逐步掌握环境搭建、数据收集、模型训练和评估调优。建议按照本指南的六个阶段逐步学习，每个阶段都要动手实践并理解背后的原理。

记住：机器人学习是一个实践性很强的领域，多动手、多调试、多思考是成功的关键。遇到问题时，仔细阅读错误信息、查阅文档、分析代码，这些都是宝贵的学习机会。

祝您学习顺利！

---
*最后更新：2026年1月*
*文档维护：ACT学习社区*