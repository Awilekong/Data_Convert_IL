# record_sim_episodes.py 输出数据集格式文档

## 概述

`record_sim_episodes.py` 脚本用于在仿真环境中生成演示数据，输出格式为 **HDF5 文件**，每个 episode 保存为一个独立的 `.hdf5` 文件。

---

## 数据集目录结构

```
<dataset_dir>/
├── episode_0.hdf5
├── episode_1.hdf5
├── episode_2.hdf5
└── ...
```

- **文件命名**: `episode_{episode_idx}.hdf5`
- **每个文件**: 包含一个完整 episode 的所有时间步数据

---

## HDF5 文件内部结构

每个 HDF5 文件包含以下层次结构：

```
episode_*.hdf5
├── (attributes)
│   └── sim: bool = True                    # 标记为仿真数据
│
├── /observations/                           # 观测数据组
│   ├── images/                              # 图像数据组
│   │   ├── <cam_name_1>: (T, 480, 640, 3)  # 相机1图像，uint8
│   │   ├── <cam_name_2>: (T, 480, 640, 3)  # 相机2图像，uint8
│   │   └── ...                              # 更多相机
│   │
│   ├── qpos: (T, 14)                        # 关节位置，float64
│   └── qvel: (T, 14)                        # 关节速度，float64
│
└── /action: (T, 14)                         # 动作序列，float64
```

---

## 数据字段详细说明

### 1. 全局属性 (Attributes)

| 属性名 | 类型 | 值 | 说明 |
|--------|------|-----|------|
| `sim` | bool | `True` | 标识这是仿真生成的数据 |

---

### 2. 观测数据 `/observations/`

#### 2.1 图像数据 `/observations/images/{cam_name}`

| 字段 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| `{cam_name}` | `(T, 480, 640, 3)` | `uint8` | RGB 图像序列 |

- **T**: episode 的时间步数量 (max_timesteps)
- **480×640**: 图像分辨率 (H×W)
- **3**: RGB 三通道
- **相机名称**: 由 `SIM_TASK_CONFIGS[task_name]['camera_names']` 配置决定
- **存储格式**: 
  - 每个时间步一个 chunk: `chunks=(1, 480, 640, 3)`
  - 未压缩 (注释中有压缩选项但未启用)

**示例相机名称**:
- `angle` (常用的渲染相机)
- 其他相机取决于任务配置

---

#### 2.2 关节位置 `/observations/qpos`

| 字段 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| `qpos` | `(T, 14)` | `float64` | 双臂关节位置 |

**维度分解** (14维):
- **索引 0-6**: 左臂 7 个关节位置
  - 关节 0-5: 机械臂关节角度
  - 关节 6: 左夹爪位置 (归一化后的值)
- **索引 7-13**: 右臂 7 个关节位置
  - 关节 7-12: 机械臂关节角度
  - 关节 13: 右夹爪位置 (归一化后的值)

**重要**: 夹爪值已经过 `PUPPET_GRIPPER_POSITION_NORMALIZE_FN` 处理

---

#### 2.3 关节速度 `/observations/qvel`

| 字段 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| `qvel` | `(T, 14)` | `float64` | 双臂关节速度 |

**维度分解** (与 qpos 对应):
- **索引 0-6**: 左臂 7 个关节速度
- **索引 7-13**: 右臂 7 个关节速度

---

### 3. 动作数据 `/action`

| 字段 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| `action` | `(T, 14)` | `float64` | 关节空间动作序列 |

**维度分解** (14维):
- **索引 0-6**: 左臂目标关节位置
  - 关节 0-5: 目标关节角度
  - 关节 6: 目标夹爪位置
- **索引 7-13**: 右臂目标关节位置
  - 关节 7-12: 目标关节角度
  - 关节 13: 目标夹爪位置

**关键特性**:
- Action 是从 EE space policy 生成的关节轨迹
- 经过重放验证 (replay) 确保可执行性
- 夹爪值已替换为控制命令 (`gripper_ctrl`)

---

## 时间对齐关系

```
时间步 t:
  observation[t]:
    - qpos[t]        # t 时刻的关节位置
    - qvel[t]        # t 时刻的关节速度
    - images[t]      # t 时刻的相机图像
  
  action[t]:         # 在 t 时刻执行的动作（下一时刻目标位置）
```

**因果关系**:
- `action[t]` 在 `observation[t]` 之后执行
- 执行 `action[t]` 后会转移到 `observation[t+1]`

---

## 数据生成流程

1. **Phase 1 - EE Space Policy Rollout**:
   - 在 `ee_sim_env` 中执行末端空间策略
   - 收集关节轨迹 `qpos` 和夹爪控制 `gripper_ctrl`
   - 替换夹爪位置为归一化的控制命令

2. **Phase 2 - Joint Space Replay**:
   - 在 `sim_env` 中重放关节轨迹
   - 记录所有观测数据 (qpos, qvel, images)
   - 验证任务成功性

3. **Phase 3 - Data Saving**:
   - 截断数据确保一致性 (去除最后的过渡帧)
   - 保存到 HDF5 文件

---

## 数据长度说明

- **原始 episode 长度**: `episode_len` (由 `SIM_TASK_CONFIGS` 定义)
- **实际保存长度**: `max_timesteps = episode_len`
  - 因为重放会增加 1 帧，脚本会截断最后一帧
  - 确保 `len(action) == len(observation) == T`

---

## 读取数据示例

```python
import h5py
import numpy as np

# 读取单个 episode
with h5py.File('episode_0.hdf5', 'r') as f:
    # 检查是否为仿真数据
    is_sim = f.attrs['sim']
    
    # 读取观测
    qpos = f['/observations/qpos'][:]      # (T, 14)
    qvel = f['/observations/qvel'][:]      # (T, 14)
    
    # 读取图像 (假设相机名为 'angle')
    images = f['/observations/images/angle'][:]  # (T, 480, 640, 3)
    
    # 读取动作
    actions = f['/action'][:]              # (T, 14)
    
    print(f"Episode length: {len(qpos)} timesteps")
    print(f"Image shape: {images.shape}")
```

---

## 配置依赖

以下配置信息来自外部文件，影响数据集格式:

| 配置项 | 来源 | 影响 |
|--------|------|------|
| `episode_len` | `SIM_TASK_CONFIGS[task_name]` | 决定 episode 长度 |
| `camera_names` | `SIM_TASK_CONFIGS[task_name]` | 决定保存哪些相机 |
| `PUPPET_GRIPPER_POSITION_NORMALIZE_FN` | `constants.py` | 夹爪值归一化函数 |

---

## 支持的任务类型

根据脚本中的策略选择:

| 任务名称 | 策略类 | 说明 |
|----------|--------|------|
| `sim_transfer_cube_scripted` | `PickAndTransferPolicy` | 抓取和转移立方体 |
| `sim_insertion_scripted` | `InsertionPolicy` | 插入任务 (如 peg-in-hole) |

---

## 注意事项

1. **数据完整性**: 最后一帧被截断以保持一致性
2. **夹爪处理**: 夹爪值不是原始传感器读数，而是归一化的控制命令
3. **坐标系**: 数据在关节空间 (joint space)，不是笛卡尔空间
4. **成功标记**: 脚本会打印每个 episode 的成功/失败状态，但不保存在 HDF5 中
5. **图像格式**: RGB, uint8, 无压缩
6. **浮点精度**: qpos/qvel/action 使用 float64

---

## 数据转换建议

如果需要转换为其他格式 (如 LeRobot)，需要注意:

1. **Action Space**: 当前是 joint position (绝对位置)
2. **State**: qpos 可直接作为 state
3. **图像处理**: 可能需要 resize 到训练分辨率
4. **Gripper**: 可能需要二值化 (0/1) 或重新归一化
5. **时序对齐**: 确认 action-observation 的因果关系
6. **双臂配置**: 需要正确处理 14 维的双臂数据结构

---

## 文件大小估算

单个 episode 文件大小取决于:
- **T** (时间步数)
- **N_cameras** (相机数量)
- **图像数据**: `T × N_cameras × 480 × 640 × 3 bytes` (主要部分)
- **数值数据**: `T × (14 + 14 + 14) × 8 bytes` (可忽略)

**示例**: 
- T=400 步, 2 个相机
- 图像: 400 × 2 × 480 × 640 × 3 ≈ 295 MB
- 总计: ~300 MB/episode

---

生成时间: 2025-11-20
脚本版本: record_sim_episodes.py
