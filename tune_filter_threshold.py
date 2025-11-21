#!/usr/bin/env python3
"""
帧过滤阈值调优工具

用于分析数据集的运动幅度分布，帮助选择合适的帧过滤阈值。
"""

import argparse
import numpy as np
from pathlib import Path
from franka_to_lerobot import Config, ActionSpace, FrankaDataLoader


def analyze_motion_distribution(config: Config, num_episodes: int = 10):
    """分析多个 episode 的运动幅度分布"""
    loader = FrankaDataLoader(config)
    episodes = loader.get_all_episodes()[:num_episodes]
    
    print(f"分析 {len(episodes)} 个 episodes 的运动幅度分布...\n")
    
    all_magnitudes = []
    
    for episode in episodes:
        robot_data = loader.load_robot_data(episode)
        
        # 根据动作空间选择数据
        if config.action_space in [ActionSpace.JOINT_POSITION_GLOBAL, ActionSpace.JOINT_POSITION_DELTA]:
            state_data = robot_data['joint_positions']
        else:  # EE_POSE
            from franka_to_lerobot import quaternion_to_rotation_vector
            ee_positions = robot_data['ee_positions']
            ee_xyz = ee_positions[:, :3]
            ee_quat = ee_positions[:, 3:]
            ee_rotvec = quaternion_to_rotation_vector(ee_quat)
            state_data = np.concatenate([ee_xyz, ee_rotvec], axis=1)
        
        # 计算相邻帧的变化幅度
        for i in range(len(state_data) - 1):
            delta = state_data[i + 1] - state_data[i]
            magnitude = np.linalg.norm(delta)
            all_magnitudes.append(magnitude)
    
    all_magnitudes = np.array(all_magnitudes)
    
    return all_magnitudes


def print_statistics(magnitudes: np.ndarray):
    """打印统计信息"""
    print("=" * 70)
    print("运动幅度统计")
    print("=" * 70)
    print(f"总数据点: {len(magnitudes):,}")
    print(f"\n基础统计:")
    print(f"  最小值:   {magnitudes.min():.8f}")
    print(f"  最大值:   {magnitudes.max():.6f}")
    print(f"  均值:     {magnitudes.mean():.6f}")
    print(f"  中位数:   {np.median(magnitudes):.8f}")
    print(f"  标准差:   {magnitudes.std():.6f}")
    
    print(f"\n分位数:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(magnitudes, p)
        print(f"  {p:2d}%: {val:.8f}")
    
    # 检测静止帧（几乎为0的帧）
    near_zero = (magnitudes < 1e-6).sum()
    print(f"\n静止帧统计:")
    print(f"  完全静止 (<1e-6): {near_zero:,} ({near_zero/len(magnitudes)*100:.2f}%)")


def test_thresholds(magnitudes: np.ndarray, thresholds: list):
    """测试不同阈值的过滤效果"""
    print("\n" + "=" * 70)
    print("不同阈值的过滤效果")
    print("=" * 70)
    print(f"{'阈值':<12} {'保留帧数':<15} {'保留比例':<12} {'平均幅度':<15}")
    print("-" * 70)
    
    for threshold in thresholds:
        kept_mask = magnitudes >= threshold
        n_kept = kept_mask.sum()
        ratio = n_kept / len(magnitudes) * 100
        avg_magnitude = magnitudes[kept_mask].mean() if n_kept > 0 else 0
        
        print(f"{threshold:<12.6f} {n_kept:<15,} {ratio:<12.1f}% {avg_magnitude:<15.6f}")


def recommend_threshold(magnitudes: np.ndarray):
    """推荐合适的阈值"""
    print("\n" + "=" * 70)
    print("阈值推荐")
    print("=" * 70)
    
    # 计算一些关键点
    p1 = np.percentile(magnitudes, 1)
    p5 = np.percentile(magnitudes, 5)
    p10 = np.percentile(magnitudes, 10)
    median = np.median(magnitudes)
    mean = magnitudes.mean()
    
    recommendations = [
        ("极保守 (99% 保留)", p1, "只过滤完全静止的帧"),
        ("保守 (95% 保留)", p5, "过滤静止和极微小抖动"),
        ("中等 (90% 保留)", p10, "过滤静止和小幅抖动"),
        ("标准 (中位数)", median, "保留有意义的运动"),
        ("激进 (均值)", mean, "只保留明显运动"),
    ]
    
    print(f"\n{'策略':<20} {'推荐阈值':<15} {'说明':<30}")
    print("-" * 70)
    for name, threshold, desc in recommendations:
        kept_ratio = (magnitudes >= threshold).sum() / len(magnitudes) * 100
        print(f"{name:<20} {threshold:<15.6f} {desc:<30}")
        print(f"{'':20} → 保留 {kept_ratio:.1f}% 的帧")
    
    print("\n💡 使用建议:")
    print(f"  • 默认建议: {p5:.6f} (过滤静止帧，保留 ~95%)")
    print(f"  • 如果数据量充足: {p10:.6f} (过滤小幅抖动，保留 ~90%)")
    print(f"  • 如果需要精简数据: {median:.6f} (只保留有意义运动，保留 ~50%)")


def plot_distribution(magnitudes: np.ndarray, output_path: str = None):
    """绘制分布图"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 直方图
        axes[0, 0].hist(magnitudes, bins=100, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('运动幅度')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title('运动幅度分布 (线性)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 对数直方图
        axes[0, 1].hist(magnitudes, bins=100, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('运动幅度')
        axes[0, 1].set_ylabel('频数 (对数)')
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_title('运动幅度分布 (对数)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 累积分布
        sorted_mags = np.sort(magnitudes)
        cumulative = np.arange(1, len(sorted_mags) + 1) / len(sorted_mags) * 100
        axes[1, 0].plot(sorted_mags, cumulative, linewidth=2)
        axes[1, 0].set_xlabel('运动幅度阈值')
        axes[1, 0].set_ylabel('保留帧的比例 (%)')
        axes[1, 0].set_title('累积分布函数 (CDF)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(left=0)
        
        # 4. 箱线图
        axes[1, 1].boxplot(magnitudes, vert=True)
        axes[1, 1].set_ylabel('运动幅度')
        axes[1, 1].set_title('箱线图')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n📊 图表已保存到: {output_path}")
        else:
            plt.show()
            
    except ImportError:
        print("\n⚠️  未安装 matplotlib，跳过绘图")
        print("   安装方法: pip install matplotlib")


def main():
    parser = argparse.ArgumentParser(
        description="帧过滤阈值调优工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基础分析
  python tune_filter_threshold.py
  
  # 分析更多 episodes
  python tune_filter_threshold.py --num-episodes 50
  
  # 测试自定义阈值
  python tune_filter_threshold.py --thresholds 0.0001 0.001 0.005 0.01 0.05
  
  # 使用末端空间
  python tune_filter_threshold.py --action-space ee_pose_delta
  
  # 保存分布图
  python tune_filter_threshold.py --plot motion_distribution.png
        """
    )
    
    parser.add_argument(
        '--num-episodes', '-n',
        type=int,
        default=10,
        help='分析的 episode 数量 (默认: 10)'
    )
    
    parser.add_argument(
        '--action-space', '-a',
        type=str,
        default='joint_position_delta',
        choices=['joint_position_delta', 'joint_position_global', 
                 'ee_pose_delta', 'ee_pose_global'],
        help='动作空间类型 (默认: joint_position_delta)'
    )
    
    parser.add_argument(
        '--thresholds', '-t',
        type=float,
        nargs='+',
        default=[0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05],
        help='要测试的阈值列表 (默认: 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.05)'
    )
    
    parser.add_argument(
        '--plot', '-p',
        type=str,
        help='保存分布图的路径 (例如: motion_distribution.png)'
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default='/home/megvii/ws_zpw/data/2025_11_18',
        help='数据根目录'
    )
    
    parser.add_argument(
        '--task-folder',
        type=str,
        default='peg_in_hole1',
        help='任务文件夹名称'
    )
    
    args = parser.parse_args()
    
    # 配置
    config = Config()
    config.data_root = Path(args.data_root)
    config.task_folder = args.task_folder
    config.action_space = ActionSpace(args.action_space)
    
    print("=" * 70)
    print("帧过滤阈值调优工具")
    print("=" * 70)
    print(f"数据路径: {config.data_root / config.task_folder}")
    print(f"动作空间: {config.action_space.value}")
    print(f"分析数量: {args.num_episodes} episodes")
    print()
    
    # 分析数据
    magnitudes = analyze_motion_distribution(config, args.num_episodes)
    
    # 打印统计信息
    print_statistics(magnitudes)
    
    # 测试不同阈值
    test_thresholds(magnitudes, sorted(args.thresholds))
    
    # 推荐阈值
    recommend_threshold(magnitudes)
    
    # 绘图
    if args.plot:
        plot_distribution(magnitudes, args.plot)
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)
    print("\n💡 下一步:")
    print("  1. 根据推荐阈值，在 Config 中设置:")
    print("     config.frame_filter_threshold = <your_value>")
    print("  2. 启用帧过滤:")
    print("     config.enable_frame_filtering = True")
    print("  3. 运行转换脚本:")
    print("     python franka_to_lerobot.py")
    print()


if __name__ == "__main__":
    main()
