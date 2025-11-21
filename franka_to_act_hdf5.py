"""
Franka 单臂数据转换为仿真 HDF5 格式 (单臂版本)

功能:
    - 从 Franka 机器人采集的原始数据(jsonl + mp4)转换为 HDF5 数据集格式
    - 支持多相机(main_realsense, side_realsense, handeye_realsense)
    - 动作空间: Global Joint Position (下一帧的关节位置)
    - 支持 stride 采样、基于末端位姿的帧过滤、夹爪二值化
    - 单臂数据: 7维 (6个关节 + 1个夹爪)
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import jsonlines
import numpy as np
import tqdm

# 线程控制，避免过度并行
# os.environ.setdefault("OPENCV_NUM_THREADS", "1")
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


# ===== 配置 =====

@dataclass
class Config:
    """全局配置"""
    # 数据路径
    data_root: Path = Path("/home/megvii/ws_zpw/data/2025_11_18")
    task_folder: str = "peg_in_hole1"
    
    # 输出配置
    output_dir: Path = Path("/home/megvii/ws_zpw/data/2025_11_18/sim_format_dataset")
    output_name: str = "peg_in_hole_sim"  # 输出文件夹名称
    
    # 图像配置 - 保持原始分辨率 480x640
    target_size: Tuple[int, int] = (480, 640)  # (H, W) - 与 record_sim_episodes.py 一致
    
    # 数据处理
    stride: int = 1  # 采样间隔
    
    # 帧过滤配置
    enable_frame_filtering: bool = True  # 是否启用帧过滤
    frame_filter_threshold: float = 1e-10  # 帧过滤阈值：joint position 变化幅度
    min_frames_per_episode: int = 10  # 每个 episode 最少保留的帧数
    
    # None 表示转换所有，否则转换前n个
    max_episodes: Optional[int] = None
    
    # 相机配置
    camera_names: List[str] = None  # None表示自动检测
    
    def __post_init__(self):
        if self.camera_names is None:
            self.camera_names = ["main_realsense_rgb", "side_realsense_rgb", "handeye_realsense_rgb"]


# ===== 数据读取模块 (复用 franka_to_lerobot.py) =====

class FrankaDataLoader:
    """Franka 数据加载器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_root = config.data_root / config.task_folder
        
    def get_all_episodes(self) -> List[str]:
        """获取所有 episode 时间文件夹"""
        if not self.data_root.exists():
            raise ValueError(f"Task folder not found: {self.data_root}")
        
        episodes = sorted([d.name for d in self.data_root.iterdir() if d.is_dir()])
        
        if self.config.max_episodes:
            episodes = episodes[:self.config.max_episodes]
            
        return episodes
    
    def load_meta(self, episode: str) -> Dict:
        """加载 meta.json"""
        meta_path = self.data_root / episode / "v1" / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found: {meta_path}")
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_robot_data(self, episode: str) -> Dict[str, np.ndarray]:
        """从 jsonl 加载机器人数据"""
        jsonl_path = self.data_root / episode / "v1" / "data" / "Franka_4_arms_arm.jsonl"
        
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Robot data not found: {jsonl_path}")
        
        with jsonlines.open(jsonl_path) as reader:
            data = list(reader)
        
        # 提取字段
        timestamps = np.array([d['timestamp'] for d in data], dtype=np.float32)
        joint_positions = np.array([d['joint_positions'] for d in data], dtype=np.float32)  # (T, 7)
        ee_positions = np.array([d['ee_positions'] for d in data], dtype=np.float32)  # (T, 7) - xyz + quat
        gripper_width = np.array([d['gripper_width'][0] for d in data], dtype=np.float32)  # (T,)
        
        return {
            'timestamps': timestamps,
            'joint_positions': joint_positions,  # (T, 7) - Franka 有 7 个关节
            'ee_positions': ee_positions,  # (T, 7) - 末端位姿 xyz(3) + quaternion(4)
            'gripper_width': gripper_width,  # (T,) - 夹爪宽度
        }
    
    def load_video_frames(self, episode: str, frame_indices: List[int]) -> Dict[str, np.ndarray]:
        """加载指定帧的图像（并行加载多个相机）"""
        video_dir = self.data_root / episode / "v1" / "videos"
        
        def load_camera(cam_name):
            video_path = video_dir / f"{cam_name}.mp4"
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")
            return cam_name, self._extract_frames(str(video_path), frame_indices)
        
        # 并行加载所有相机
        frames = {}
        with ThreadPoolExecutor(max_workers=len(self.config.camera_names)) as executor:
            results = executor.map(load_camera, self.config.camera_names)
            for cam_name, cam_frames in results:
                frames[cam_name] = cam_frames
        
        return frames
    
    def _extract_frames(self, video_path: str, frame_indices: List[int]) -> np.ndarray:
        """从视频提取指定索引的帧，返回 (T, H, W, 3) RGB uint8"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                print(f"Warning: Failed to read frame {idx} from {video_path}")
        
        cap.release()
        return np.array(frames, dtype=np.uint8)


# ===== 数据处理模块 =====

class DataProcessor:
    """数据处理器 - 单臂数据处理"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def process_episode(self, robot_data: Dict[str, np.ndarray], 
                       video_frames: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """处理单个 episode 的数据
        
        处理流程:
        1. Stride 采样
        2. 【可选】基于末端位姿变化的帧过滤
        3. 构建 qpos (7维: 6个关节 + 1个夹爪)
        4. 计算 qvel (关节速度估计)
        5. 计算 action (global joint position - 下一帧的关节位置)
        """
        stride = self.config.stride
        
        # === 步骤1: Stride 采样 ===
        T_raw = len(robot_data['timestamps'])
        stride_indices = list(range(0, T_raw - stride, stride))
        
        # 检查视频帧数量是否匹配
        first_cam_frames = list(video_frames.values())[0]
        if len(first_cam_frames) != len(stride_indices):
            min_len = min(len(stride_indices), len(first_cam_frames))
            stride_indices = stride_indices[:min_len]
        
        # 提取数据
        joint_positions = robot_data['joint_positions']  # (T_raw, 7)
        ee_positions = robot_data['ee_positions']  # (T_raw, 7)
        stride_joint_positions = joint_positions[stride_indices]  # (T_stride, 7)
        stride_ee_positions = ee_positions[stride_indices]  # (T_stride, 7)
        
        # === 步骤2: 【可选】基于末端位姿的帧过滤 ===
        if self.config.enable_frame_filtering:
            keep_mask = self._filter_frames_by_ee_motion(stride_ee_positions)
            final_indices = [stride_indices[i] for i in range(len(stride_indices)) if keep_mask[i]]
            final_joint_positions = stride_joint_positions[keep_mask]
        else:
            final_indices = stride_indices
            final_joint_positions = stride_joint_positions
        
        # === 步骤3: 构建单臂 qpos (8维: 7个关节 + 1个夹爪) ===
        qpos_8 = self._build_qpos(final_joint_positions, robot_data['gripper_width'][final_indices])
        
        # === 步骤4: 计算 qvel (关节速度) ===
        qvel_8 = self._compute_qvel(qpos_8, robot_data['timestamps'][final_indices])
        
        # === 步骤5: 计算 action (global joint position - 下一帧的关节位置) ===
        action_8 = self._compute_global_action(qpos_8)
        
        # === 步骤6: 处理视频帧 ===
        processed_video_frames = {}
        for cam_name, frames in video_frames.items():
            if self.config.enable_frame_filtering:
                video_indices = [stride_indices.index(idx) for idx in final_indices]
                processed_video_frames[cam_name] = self._resize_frames(frames[video_indices])
            else:
                processed_video_frames[cam_name] = self._resize_frames(frames[:len(final_indices)])
        
        return {
            'qpos': qpos_8,  # (T, 8)
            'qvel': qvel_8,  # (T, 8)
            'action': action_8,  # (T, 8)
            'video_frames': processed_video_frames,  # Dict[cam_name, (T, H, W, 3)]
        }
    
    def _build_qpos(self, joint_positions: np.ndarray, gripper_width: np.ndarray) -> np.ndarray:
        """构建单臂 qpos (8维)
        
        格式: [joint_0:6, gripper] - Franka有7个关节，但我们只使用前6个+夹爪=7维
        实际上Franka有7个关节，所以应该是8维: [joint_0:6, gripper]
        
        Args:
            joint_positions: (T, 7) 关节位置 (Franka的7个关节)
            gripper_width: (T,) 夹爪宽度
        
        Returns:
            qpos_8: (T, 8) 单臂关节位置 (7个关节 + 1个夹爪)
        """
        T = len(joint_positions)
        qpos_8 = np.zeros((T, 8), dtype=np.float64)
        
        # 使用全部7个关节位置
        qpos_8[:, 0:7] = joint_positions[:, 0:7]
        # 第8维是夹爪 (二值化)
        qpos_8[:, 7] = self._binarize_gripper(gripper_width)
        
        return qpos_8
    
    def _compute_qvel(self, qpos: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """计算关节速度 (数值微分)
        
        Args:
            qpos: (T, 8) 关节位置
            timestamps: (T,) 时间戳
        
        Returns:
            qvel: (T, 8) 关节速度
        """
        T = len(qpos)
        qvel = np.zeros_like(qpos, dtype=np.float64)
        
        for i in range(T - 1):
            dt = timestamps[i + 1] - timestamps[i]
            if dt > 0:
                qvel[i] = (qpos[i + 1] - qpos[i]) / dt
            else:
                qvel[i] = 0.0
        
        # 最后一帧速度设为0
        qvel[-1] = 0.0
        
        return qvel
    
    def _compute_global_action(self, qpos: np.ndarray) -> np.ndarray:
        """计算 global action (下一帧的关节位置)
        
        Args:
            qpos: (T, 8) 当前关节位置
        
        Returns:
            action: (T, 8) 下一帧的关节位置
        """
        action = np.zeros_like(qpos, dtype=np.float64)
        action[:-1] = qpos[1:]  # 前 T-1 帧的 action 是下一帧的 qpos
        action[-1] = qpos[-1]   # 最后一帧保持不变
        return action
    
    def _filter_frames_by_ee_motion(self, ee_positions: np.ndarray) -> np.ndarray:
        """根据末端位姿变化幅度过滤关键帧
        
        参考 franka_to_lerobot.py 的实现，使用完整的末端位姿（xyz + rotation_vector）
        
        Args:
            ee_positions: (T, 7) 末端位姿 xyz(3) + quaternion(4)
        
        Returns:
            keep_mask: (T,) 布尔数组
        """
        from scipy.spatial.transform import Rotation as R
        
        T = len(ee_positions)
        if T < 2:
            return np.ones(T, dtype=bool)
        
        # 转换为 xyz + rotation_vector 格式以便计算变化幅度
        ee_xyz = ee_positions[:, :3]  # (T, 3)
        ee_quat = ee_positions[:, 3:]  # (T, 4) - [qw, qx, qy, qz]
        
        # 转换四元数为旋转向量
        quaternions_scipy = np.concatenate([
            ee_quat[:, 1:4],  # qx, qy, qz
            ee_quat[:, 0:1]   # qw
        ], axis=-1)
        rotations = R.from_quat(quaternions_scipy)
        ee_rotvec = rotations.as_rotvec()  # (T, 3)
        
        # 组合为完整的末端位姿表示
        ee_pose = np.concatenate([ee_xyz, ee_rotvec], axis=1)  # (T, 6)
        
        # 计算相邻帧之间的末端位姿变化幅度（包括位置和旋转）
        motion_magnitudes = np.zeros(T)
        for i in range(T - 1):
            delta = ee_pose[i + 1] - ee_pose[i]
            motion_magnitudes[i] = np.linalg.norm(delta)
        motion_magnitudes[-1] = 0.0
        
        # 根据阈值过滤
        threshold = self.config.frame_filter_threshold
        keep_mask = motion_magnitudes >= threshold
        
        # 确保至少保留 min_frames_per_episode 帧
        n_valid = keep_mask.sum()
        min_frames = self.config.min_frames_per_episode
        
        if n_valid < min_frames:
            top_indices = np.argsort(motion_magnitudes)[-min_frames:]
            keep_mask = np.zeros(T, dtype=bool)
            keep_mask[top_indices] = True
        else:
            keep_mask[0] = True
            keep_mask[-1] = True
        
        # 记录过滤信息
        n_kept = keep_mask.sum()
        filter_ratio = n_kept / T if T > 0 else 0
        
        if n_kept < T:
            print(f"  [Frame Filter (EE Motion)] {T} -> {n_kept} frames "
                  f"(kept {filter_ratio:.1%}, threshold={threshold:.6f})")
        
        return keep_mask
    
    def _binarize_gripper(self, gripper: np.ndarray) -> np.ndarray:
        """夹爪二值化 - 基于数据分析确定阈值
        
        将夹爪宽度转换为归一化的 0-1 值
        - 0: 合 (closed) < 25mm
        - 1: 开 (open) >= 25mm
        """
        threshold = 0.025  # 25mm
        return np.where(gripper < threshold, 0.0, 1.0).astype(np.float64)
    
    def _resize_frames(self, frames: np.ndarray) -> np.ndarray:
        """Resize 视频帧到目标尺寸"""
        H, W = self.config.target_size
        resized = []
        for frame in frames:
            resized_frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            resized.append(resized_frame)
        return np.array(resized, dtype=np.uint8)


# ===== HDF5 转换模块 =====

class HDF5Converter:
    """HDF5 数据集转换器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = config.output_dir / config.output_name
        
    def initialize(self):
        """初始化输出目录"""
        if self.output_dir.exists():
            import shutil
            print(f"Warning: Output directory exists, removing: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
    
    def save_episode(self, episode_idx: int, processed_data: Dict[str, np.ndarray], 
                    meta: Dict) -> None:
        """保存一个 episode 为 HDF5 文件
        
        文件格式 (单臂版本):
        - /observations/images/{cam_name}: (T, 480, 640, 3) uint8
        - /observations/qpos: (T, 8) float64 - 7个关节 + 1个夹爪
        - /observations/qvel: (T, 8) float64 - 7个关节 + 1个夹爪
        - /action: (T, 8) float64 - 7个关节 + 1个夹爪
        - attributes: sim=True
        """
        qpos = processed_data['qpos']
        qvel = processed_data['qvel']
        action = processed_data['action']
        video_frames = processed_data['video_frames']
        
        # 验证数据长度
        T = len(qpos)
        assert len(qvel) == T, f"qvel length mismatch: {len(qvel)} != {T}"
        assert len(action) == T, f"action length mismatch: {len(action)} != {T}"
        for cam_name, frames in video_frames.items():
            assert len(frames) == T, f"{cam_name} frames length mismatch: {len(frames)} != {T}"
        
        # 保存为 HDF5
        t0 = time.time()
        dataset_path = self.output_dir / f'episode_{episode_idx}.hdf5'
        
        with h5py.File(dataset_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            # 设置属性
            root.attrs['sim'] = False  # 标记为真机数据
            
            # 创建 observations 组
            obs = root.create_group('observations')
            
            # 创建 images 组
            image = obs.create_group('images')
            H, W = self.config.target_size
            for cam_name, frames in video_frames.items():
                _ = image.create_dataset(
                    cam_name, 
                    (T, H, W, 3), 
                    dtype='uint8',
                    chunks=(1, H, W, 3),
                    data=frames
                )
            
            # 创建 qpos 和 qvel
            obs.create_dataset('qpos', (T, 8), dtype='float64', data=qpos)
            obs.create_dataset('qvel', (T, 8), dtype='float64', data=qvel)
            
            # 创建 action
            root.create_dataset('action', (T, 8), dtype='float64', data=action)
        
        print(f'  Saved episode_{episode_idx}.hdf5 (T={T}, {time.time() - t0:.1f}s)')
    
    def finalize(self, num_episodes: int, success_count: int):
        """完成转换，打印统计信息"""
        print(f"\n{'='*60}")
        print(f"Dataset saved to: {self.output_dir}")
        print(f"Total episodes: {num_episodes}")
        print(f"Success rate: {success_count}/{num_episodes}")
        print(f"{'='*60}")


# ===== 主流程 =====

def main():
    """主函数"""
    config = Config()
    
    # 可选: 修改配置
    # config.stride = 2
    # config.enable_frame_filtering = True
    # config.frame_filter_threshold = 0.0001
    # config.max_episodes = 10
    
    print(f"{'='*60}")
    print(f"Franka to Simulation HDF5 Format Converter")
    print(f"{'='*60}")
    print(f"Data root: {config.data_root / config.task_folder}")
    print(f"Output: {config.output_dir / config.output_name}")
    print(f"Action space: Global Joint Position")
    print(f"Stride: {config.stride}")
    print(f"Frame filtering: {config.enable_frame_filtering}")
    if config.enable_frame_filtering:
        print(f"  Threshold: {config.frame_filter_threshold}")
        print(f"  Min frames: {config.min_frames_per_episode}")
    print(f"{'='*60}\n")
    
    # 初始化模块
    loader = FrankaDataLoader(config)
    processor = DataProcessor(config)
    converter = HDF5Converter(config)
    
    # 初始化输出目录
    converter.initialize()
    
    # 获取所有 episodes
    episodes = loader.get_all_episodes()
    print(f"Found {len(episodes)} episodes\n")
    
    if not episodes:
        raise ValueError("No episodes found")
    
    # 转换所有 episodes
    success_count = 0
    for episode_idx, episode in enumerate(tqdm.tqdm(episodes, desc="Converting episodes")):
        try:
            # 加载元数据
            meta = loader.load_meta(episode)
            
            # 加载机器人数据
            robot_data = loader.load_robot_data(episode)
            
            # 计算帧索引
            T_raw = len(robot_data['timestamps'])
            frame_indices = list(range(0, T_raw - config.stride, config.stride))
            
            # 加载视频帧
            video_frames = loader.load_video_frames(episode, frame_indices)
            
            # 处理数据
            processed_data = processor.process_episode(robot_data, video_frames)
            
            # 保存为 HDF5
            converter.save_episode(episode_idx, processed_data, meta)
            
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing episode {episode_idx} ({episode}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 完成
    converter.finalize(len(episodes), success_count)


if __name__ == "__main__":
    main()
