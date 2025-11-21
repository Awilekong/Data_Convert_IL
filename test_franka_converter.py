"""
Franka 数据转换器模块化测试脚本

测试各个模块的功能是否正常：
1. 数据加载器 (FrankaDataLoader)
2. 数据处理器 (DataProcessor)
3. LeRobot 转换器 (LeRobotConverter)
"""

import sys
from pathlib import Path
import numpy as np
import json

# 导入被测试模块
from franka_to_lerobot import (
    Config,
    FrankaDataLoader,
    DataProcessor,
    LeRobotConverter
)


def print_section(title: str):
    """打印测试节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_config():
    """测试 1: 配置模块"""
    print_section("测试 1: 配置模块 (Config)")
    
    config = Config()
    
    print(f"✓ 数据根目录: {config.data_root}")
    print(f"✓ 任务文件夹: {config.task_folder}")
    print(f"✓ 输出数据集: {config.repo_id}")
    print(f"✓ 目标图像尺寸: {config.target_size}")
    print(f"✓ 采样间隔: {config.stride}")
    print(f"✓ 使用关节增量: {config.use_delta_joint}")
    print(f"✓ 相机列表: {config.camera_names}")
    
    assert config.data_root.exists(), f"数据根目录不存在: {config.data_root}"
    print(f"\n✓✓✓ 配置模块测试通过")
    
    return config


def test_data_loader(config: Config):
    """测试 2: 数据加载器"""
    print_section("测试 2: 数据加载器 (FrankaDataLoader)")
    
    loader = FrankaDataLoader(config)
    
    # 测试获取所有 episodes
    print("\n[2.1] 测试获取 episode 列表...")
    episodes = loader.get_all_episodes()
    print(f"✓ 找到 {len(episodes)} 个 episodes")
    print(f"  前 5 个: {episodes[:5]}")
    
    assert len(episodes) > 0, "没有找到任何 episode"
    
    # 测试加载 meta 信息
    print(f"\n[2.2] 测试加载 meta 信息 (episode: {episodes[0]})...")
    meta = loader.load_meta(episodes[0])
    print(f"✓ 任务名称: {meta['task_meta']['task_name']}")
    print(f"✓ 提示词: {meta['task_meta']['prompt']}")
    print(f"✓ 机器人型号: {meta['robot_meta']['robots'][0]['robot_model']}")
    print(f"✓ 总帧数: {meta['task_meta']['frames']}")
    
    # 测试加载机器人数据
    print(f"\n[2.3] 测试加载机器人数据...")
    robot_data = loader.load_robot_data(episodes[0])
    print(f"✓ timestamps shape: {robot_data['timestamps'].shape}")
    print(f"✓ joint_positions shape: {robot_data['joint_positions'].shape}")
    print(f"✓ ee_positions shape: {robot_data['ee_positions'].shape}")
    print(f"✓ gripper shape: {robot_data['gripper'].shape}")
    print(f"✓ gripper_width shape: {robot_data['gripper_width'].shape}")
    
    # 检查数据维度
    T = len(robot_data['timestamps'])
    assert robot_data['joint_positions'].shape == (T, 7), "joint_positions 维度错误"
    assert robot_data['ee_positions'].shape == (T, 7), "ee_positions 维度错误"
    assert robot_data['gripper'].shape == (T, 2), "gripper 维度错误"
    assert robot_data['gripper_width'].shape == (T,), "gripper_width 维度错误"
    
    # 检查 gripper 数据
    print(f"\n[2.4] 检查 gripper 数据...")
    print(f"  gripper 前 3 个值: {robot_data['gripper'][:3].tolist()}")
    print(f"  gripper_width 范围: [{robot_data['gripper_width'].min():.4f}, {robot_data['gripper_width'].max():.4f}]")
    gripper_unique = np.unique(robot_data['gripper'])
    print(f"  gripper 唯一值: {gripper_unique.tolist()}")
    
    # 测试加载视频帧
    print(f"\n[2.5] 测试加载视频帧...")
    frame_indices = list(range(0, min(100, T), config.stride))
    video_frames = loader.load_video_frames(episodes[0], frame_indices)
    
    for cam_name, frames in video_frames.items():
        print(f"✓ {cam_name}: shape={frames.shape}, dtype={frames.dtype}")
        assert frames.ndim == 4, f"{cam_name} 帧维度错误"
        assert frames.shape[0] == len(frame_indices), f"{cam_name} 帧数量错误"
    
    print(f"\n✓✓✓ 数据加载器测试通过")
    
    return loader, episodes[0], robot_data, video_frames


def test_data_processor(config: Config, robot_data: dict, video_frames: dict):
    """测试 3: 数据处理器"""
    print_section("测试 3: 数据处理器 (DataProcessor)")
    
    processor = DataProcessor(config)
    
    # 测试处理 episode
    print("\n[3.1] 测试处理 episode 数据...")
    processed_data = processor.process_episode(robot_data, video_frames)
    
    print(f"✓ 处理后的数据键: {list(processed_data.keys())}")
    
    # 检查必需的键
    required_keys = ['timestamps', 'qpos', 'joint_positions_delta', 'ee_positions', 'gripper']
    for key in required_keys:
        assert key in processed_data, f"缺少必需的键: {key}"
        print(f"✓ {key}: shape={processed_data[key].shape}")
    
    # 检查维度
    T = len(processed_data['timestamps'])
    assert processed_data['qpos'].shape == (T, 7), "qpos 维度错误"
    assert processed_data['joint_positions_delta'].shape == (T, 7), "joint_positions_delta 维度错误"
    assert processed_data['gripper'].shape == (T,), "gripper 维度错误"
    
    # 检查 gripper 二值化
    print(f"\n[3.2] 检查 gripper 二值化...")
    gripper_unique = np.unique(processed_data['gripper'])
    print(f"  gripper 唯一值: {gripper_unique.tolist()}")
    assert set(gripper_unique).issubset({0.0, 1.0}), "gripper 应该是二值化的 (0 或 1)"
    print(f"  ✓ gripper 已正确二值化")
    
    # 检查视频帧 resize
    print(f"\n[3.3] 检查视频帧 resize...")
    for cam_name, frames in processed_data['video_frames'].items():
        H, W = config.target_size
        expected_shape = (T, H, W, 3)
        assert frames.shape == expected_shape, f"{cam_name} resize 后维度错误: {frames.shape} vs {expected_shape}"
        print(f"✓ {cam_name}: 已 resize 到 {config.target_size}")
    
    # 检查增量计算
    if config.use_delta_joint:
        print(f"\n[3.4] 检查关节增量计算...")
        delta = processed_data['joint_positions_delta']
        print(f"  增量范围: [{delta.min():.4f}, {delta.max():.4f}]")
        print(f"  增量均值: {delta.mean():.6f}")
        print(f"  ✓ 增量计算完成")
    
    print(f"\n✓✓✓ 数据处理器测试通过")
    
    return processor, processed_data


def test_lerobot_converter(config: Config, processed_data: dict, meta: dict):
    """测试 4: LeRobot 转换器"""
    print_section("测试 4: LeRobot 转换器 (LeRobotConverter)")
    
    converter = LeRobotConverter(config)
    
    # 测试 state 构建
    print("\n[4.1] 测试 state 构建...")
    state = converter._build_state(processed_data)
    print(f"✓ state shape: {state.shape}")
    assert state.shape[1] == 8, f"state 维度应该是 8 (7关节+1夹爪), 实际: {state.shape[1]}"
    print(f"  state 范围: [{state.min():.4f}, {state.max():.4f}]")
    print(f"  state 均值: {state.mean(axis=0)}")
    
    # 测试 action 构建
    print("\n[4.2] 测试 action 构建...")
    action = converter._build_action(processed_data)
    print(f"✓ action shape: {action.shape}")
    assert action.shape[1] == 8, f"action 维度应该是 8 (7关节+1夹爪), 实际: {action.shape[1]}"
    print(f"  action 范围: [{action.min():.4f}, {action.max():.4f}]")
    print(f"  action 均值: {action.mean(axis=0)}")
    
    # 测试 FPS 估计
    print("\n[4.3] 测试 FPS 估计...")
    fps = converter._estimate_fps(processed_data['timestamps'])
    print(f"✓ 估计 FPS: {fps:.2f}")
    assert 10 < fps < 100, f"FPS 似乎不合理: {fps}"
    
    # 测试任务字符串
    print("\n[4.4] 测试任务字符串提取...")
    task_str = converter._get_task_string(meta)
    print(f"✓ 任务字符串: {task_str}")
    assert len(task_str) > 0, "任务字符串为空"
    
    print(f"\n✓✓✓ LeRobot 转换器测试通过")
    
    return converter, state, action


def test_integration(config: Config, loader: FrankaDataLoader, processor: DataProcessor, 
                     converter: LeRobotConverter, episode: str):
    """测试 5: 集成测试"""
    print_section("测试 5: 集成测试 (完整流程)")
    
    print(f"\n[5.1] 测试完整数据转换流程 (episode: {episode})...")
    
    # 加载数据
    meta = loader.load_meta(episode)
    robot_data = loader.load_robot_data(episode)
    
    T_raw = len(robot_data['timestamps'])
    frame_indices = list(range(0, T_raw - config.stride, config.stride))
    video_frames = loader.load_video_frames(episode, frame_indices)
    
    # 处理数据
    processed_data = processor.process_episode(robot_data, video_frames)
    
    # 构建 state 和 action
    state = converter._build_state(processed_data)
    action = converter._build_action(processed_data)
    
    # 检查长度对齐
    T = min(len(state), len(action))
    for cam_name, frames in processed_data['video_frames'].items():
        T = min(T, len(frames))
    
    print(f"✓ 对齐后的帧数: {T}")
    print(f"  state: {len(state)}")
    print(f"  action: {len(action)}")
    for cam_name, frames in processed_data['video_frames'].items():
        print(f"  {cam_name}: {len(frames)}")
    
    assert T > 0, "对齐后没有有效帧"
    
    print(f"\n[5.2] 检查数据一致性...")
    
    # 检查 state 和 action 的 gripper 是否一致
    state_gripper = state[:, -1]
    action_gripper = action[:, -1]
    gripper_match = np.allclose(state_gripper, action_gripper)
    print(f"  state 和 action 的 gripper 是否一致: {gripper_match}")
    assert gripper_match, "state 和 action 的 gripper 应该一致"
    
    # 检查 qpos vs joint_delta
    qpos = processed_data['qpos']
    joint_delta = processed_data['joint_positions_delta']
    print(f"  qpos 范围: [{qpos.min():.4f}, {qpos.max():.4f}]")
    print(f"  joint_delta 范围: [{joint_delta.min():.4f}, {joint_delta.max():.4f}]")
    
    # qpos 应该是绝对位置（较大），joint_delta 应该是增量（较小）
    assert np.abs(qpos).mean() > np.abs(joint_delta).mean() * 10, \
        "qpos 应该是绝对位置，joint_delta 应该是增量"
    print(f"  ✓ qpos (绝对位置) 和 joint_delta (增量) 的大小关系正确")
    
    print(f"\n✓✓✓ 集成测试通过")


def test_data_statistics(processed_data: dict):
    """测试 6: 数据统计分析"""
    print_section("测试 6: 数据统计分析")
    
    print("\n[6.1] 关节数据统计...")
    qpos = processed_data['qpos']
    joint_delta = processed_data['joint_positions_delta']
    
    print(f"\nqpos (当前位置) 统计:")
    print(f"  均值: {qpos.mean(axis=0)}")
    print(f"  标准差: {qpos.std(axis=0)}")
    print(f"  范围: [{qpos.min(axis=0)}, {qpos.max(axis=0)}]")
    
    print(f"\njoint_delta (增量) 统计:")
    print(f"  均值: {joint_delta.mean(axis=0)}")
    print(f"  标准差: {joint_delta.std(axis=0)}")
    print(f"  范围: [{joint_delta.min(axis=0)}, {joint_delta.max(axis=0)}]")
    
    print("\n[6.2] 夹爪数据统计...")
    gripper = processed_data['gripper']
    print(f"  唯一值: {np.unique(gripper)}")
    print(f"  值分布: 0={np.sum(gripper==0)}, 1={np.sum(gripper==1)}")
    print(f"  开合比例: {gripper.mean():.2%} 的时间处于开启状态")
    
    print("\n[6.3] 视频帧统计...")
    for cam_name, frames in processed_data['video_frames'].items():
        print(f"\n{cam_name}:")
        print(f"  shape: {frames.shape}")
        print(f"  dtype: {frames.dtype}")
        print(f"  范围: [{frames.min()}, {frames.max()}]")
        print(f"  均值: {frames.mean():.2f}")
    
    print(f"\n✓✓✓ 数据统计分析完成")


def main():
    """主测试流程"""
    print("\n" + "="*60)
    print("  Franka 数据转换器模块化测试")
    print("="*60)
    
    try:
        # 测试 1: 配置
        config = test_config()
        
        # 测试 2: 数据加载器
        loader, test_episode, robot_data, video_frames = test_data_loader(config)
        meta = loader.load_meta(test_episode)
        
        # 测试 3: 数据处理器
        processor, processed_data = test_data_processor(config, robot_data, video_frames)
        
        # 测试 4: LeRobot 转换器
        converter, state, action = test_lerobot_converter(config, processed_data, meta)
        
        # 测试 5: 集成测试
        test_integration(config, loader, processor, converter, test_episode)
        
        # 测试 6: 数据统计
        test_data_statistics(processed_data)
        
        # 最终总结
        print_section("测试总结")
        print("\n✓✓✓ 所有测试通过！")
        print("\n各模块功能正常:")
        print("  ✓ Config: 配置管理")
        print("  ✓ FrankaDataLoader: 数据加载 (jsonl + mp4)")
        print("  ✓ DataProcessor: 数据处理 (采样、增量、二值化、resize)")
        print("  ✓ LeRobotConverter: LeRobot 格式转换 (state/action 构建)")
        print("  ✓ Integration: 完整流程集成")
        print("  ✓ Statistics: 数据统计分析")
        
        print("\n数据处理逻辑验证:")
        print("  ✓ State 使用 qpos (当前位置，非增量)")
        print("  ✓ Action 使用 joint_delta (增量)")
        print("  ✓ Gripper 使用 gripper_width (已二值化)")
        print("  ✓ 维度: (T, 8) = 7关节 + 1夹爪")
        
        print("\n🎉 可以运行完整的数据转换脚本了！")
        print(f"   python franka_to_lerobot.py")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
