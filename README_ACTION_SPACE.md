# Franka 数据转换器 - 动作空间配置

## 功能概述

脚本现在支持 **4 种动作空间类型**，可以灵活配置生成不同格式的训练数据。

## 四种动作空间

| 动作空间 | State | Action | 适用场景 |
|---------|-------|--------|---------|
| `JOINT_POSITION_GLOBAL` | 当前关节位置 (7维) | 下一帧关节位置 (7维) | 位置控制，输出目标位置 |
| `JOINT_POSITION_DELTA` | 当前关节位置 (7维) | 关节位置增量 (7维) | 增量控制，更平滑 (默认) |
| `EE_POSE_GLOBAL` | 当前末端位姿 (6维) | 下一帧末端位姿 (6维) | 末端位置控制 |
| `EE_POSE_DELTA` | 当前末端位姿 (6维) | 末端位姿增量 (6维) | 末端增量控制 |

*注: 
- 关节空间: 7 个关节 + 1 gripper = **8 维**
- 末端空间: 3 位置 + 3 旋转向量 + 1 gripper = **7 维***

## 快速使用

### 方法 1: 修改配置类默认值

编辑 `franka_to_lerobot.py` 第 47 行:

```python
@dataclass
class Config:
    # ...
    action_space: ActionSpace = ActionSpace.JOINT_POSITION_DELTA  # 修改这里
```

### 方法 2: 在 main() 中修改

编辑 `franka_to_lerobot.py` 第 383-386 行:

```python
def main():
    config = Config()
    config.action_space = ActionSpace.EE_POSE_DELTA  # 修改这里
    # ...
```

### 方法 3: 命令行参数 (需要额外实现)

```bash
python franka_to_lerobot.py --action-space ee_pose_delta
```

## 输出数据集命名

脚本会自动根据动作空间添加后缀到 `repo_id`:

```
franka/peg_in_hole → franka/peg_in_hole_joint_position_delta
```

## 数据格式

### 关节空间 (JOINT_POSITION_*)

```python
State:  [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper]
Action: [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper]
```

### 末端位姿空间 (EE_POSE_*)

```python
State:  [x, y, z, rot_x, rot_y, rot_z, gripper]
Action: [x, y, z, rot_x, rot_y, rot_z, gripper]
```

其中:
- `(x, y, z)` 是笛卡尔坐标
- `(rot_x, rot_y, rot_z)` 是**旋转向量** (rotation vector / axis-angle):
  - 方向表示旋转轴
  - 长度 (范数) 表示旋转角度 (弧度)
  - 这是机器人学中末端执行器控制的标准表示
  
> **注**: Franka 数据中存储的是四元数 `(qw, qx, qy, qz)`，脚本会自动转换为旋转向量

## 选择建议

| 任务类型 | 推荐动作空间 | 理由 |
|---------|------------|------|
| 精确插孔 | `JOINT_POSITION_DELTA` | 关节控制更精确，增量更稳定 |
| 抓取放置 | `EE_POSE_DELTA` | 末端控制更直观 |
| 复杂轨迹 | `JOINT_POSITION_GLOBAL` | 全局位置，路径规划更清晰 |
| 力控任务 | `EE_POSE_DELTA` | 增量控制，力反馈更好 |

## 测试脚本

运行测试验证所有动作空间:

```bash
python test_action_spaces.py
```

## 批量生成

生成所有 4 种动作空间的数据集:

```python
from franka_to_lerobot import Config, ActionSpace

action_spaces = [
    ActionSpace.JOINT_POSITION_GLOBAL,
    ActionSpace.JOINT_POSITION_DELTA,
    ActionSpace.EE_POSE_GLOBAL,
    ActionSpace.EE_POSE_DELTA,
]

for action_space in action_spaces:
    config = Config()
    config.action_space = action_space
    # 运行转换...
```

## 代码修改要点

1. **新增 Enum**: `ActionSpace` 定义 4 种类型
2. **Config 类**: 添加 `action_space` 字段
3. **DataProcessor**: 根据 action_space 处理不同数据
4. **LeRobotConverter**: 动态生成特征定义
5. **自动命名**: repo_id 自动添加后缀

## 常见问题

**Q: 如何知道该用哪种动作空间？**  
A: 先用默认的 `JOINT_POSITION_DELTA`，如果训练不稳定，尝试其他类型。

**Q: Global 和 Delta 的区别？**  
A: Global 输出绝对位置/姿态，Delta 输出相对变化量。Delta 通常更平滑。

**Q: 可以同时生成多种吗？**  
A: 可以，多次运行脚本并修改 `action_space` 即可，输出会保存到不同目录。

**Q: 数据维度都是 8 吗？**  
A: 不是。关节空间是 8 维 (7 关节 + 1 gripper)，末端空间是 7 维 (3 位置 + 3 旋转向量 + 1 gripper)。

**Q: 为什么末端空间用旋转向量而不是四元数？**  
A: 旋转向量是机器人学中末端控制的标准表示，维度更低 (3D vs 4D)，更适合增量控制。原始 Franka 数据中的四元数会被自动转换。

**Q: 旋转向量的范围是多少？**  
A: 范数 (模长) 在 0 到 π 之间，表示旋转角度。方向表示旋转轴。

## 相关文件

- `franka_to_lerobot.py` - 主转换脚本
- `ACTION_SPACE_GUIDE.py` - 详细使用指南
- `test_action_spaces.py` - 测试脚本
- `README_ACTION_SPACE.md` - 本文档
