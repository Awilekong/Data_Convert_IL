"""
Franka 单臂数据转换为 Zarr 格式 (类似 pusht_real)

功能:
    - 从 Franka 机器人采集的原始数据(jsonl + mp4)转换为 Zarr 数据集格式
    - 支持多相机(main_realsense, side_realsense, handeye_realsense)
    - 动作空间: End-Effector Pose (绝对末端位姿, 6维 xyz+rotation_vector) + 夹爪(1维)
    - 状态空间: 关节位置(7维) + 夹爪(1维) = 8维
    - 支持 stride 采样、基于末端位姿的帧过滤、夹爪二值化
    - 所有 episodes 合并为单个 zarr 数据集
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import jsonlines
import numpy as np
import tqdm
import zarr
from scipy.spatial.transform import Rotation as R


# ===== 配置 =====

@dataclass
class Config:
    """全局配置"""
    # 数据路径
    data_root: Path = Path("/home/megvii/ws_zpw/data/2025_11_18")
    task_folder: str = "peg_in_hole1"
    
    # 输出配置
    output_dir: Path = Path("/home/megvii/ws_zpw/data/2025_11_18/zarr_dataset")
    output_name: str = "peg_in_hole_zarr"  # 输出文件夹名称
    
    # 图像配置 - 保持原始分辨率 480x640
    target_size: Tuple[int, int] = (480, 640)  # (H, W)
    video_fps: int = 30  # 视频帧率
    
    # 数据处理
    stride: int = 1  # 采样间隔
    
    # 帧过滤配置
    enable_frame_filtering: bool = True  # 是否启用帧过滤
    frame_filter_threshold: float = 1e-10  # 帧过滤阈值：joint position 变化幅度
    min_frames_per_episode: int = 10  # 每个 episode 最少保留的帧数
    
    # None 表示转换所有，否则转换前n个
    max_episodes: Optional[int] = None
    
    # 相机配置 - 映射到数字编号
    camera_names: List[str] = None  # None表示自动检测
    camera_mapping: Dict[str, int] = None  # 相机名称到编号的映射
    
    def __post_init__(self):
        if self.camera_names is None:
            self.camera_names = ["main_realsense_rgb", "side_realsense_rgb", "handeye_realsense_rgb"]
        if self.camera_mapping is None:
            # main_realsense -> 0.mp4, side_realsense -> 2.mp4, handeye_realsense -> 1.mp4
            self.camera_mapping = {
                "main_realsense_rgb": 0,
                "side_realsense_rgb": 2,
                "handeye_realsense_rgb": 1,
            }


# ===== 数据读取模块 (复用 franka_to_act_hdf5.py) =====

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
        timestamps = np.array([d['timestamp'] for d in data], dtype=np.float64)
        joint_positions = np.array([d['joint_positions'] for d in data], dtype=np.float64)  # (T, 7)
        ee_positions = np.array([d['ee_positions'] for d in data], dtype=np.float64)  # (T, 7) - xyz + quat
        gripper_width = np.array([d['gripper_width'][0] for d in data], dtype=np.float64)  # (T,)
        
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
    """数据处理器 - 转换为 Zarr 格式所需的数据"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def process_episode(self, robot_data: Dict[str, np.ndarray], 
                       video_frames: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """处理单个 episode 的数据
        
        处理流程:
        1. Stride 采样
        2. 【可选】基于末端位姿变化的帧过滤
        3. 构建 robot_joint (8维: 7个关节 + 1个夹爪)
        4. 计算 robot_joint_vel (8维)
        5. 构建 robot_eef_pose (6维: xyz + rotation_vector)
        6. 计算 robot_eef_pose_vel (6维)
        7. 构建 action (7维: 6维eef_pose + 1维夹爪，取下一帧的值)
        8. 构建 stage (全0)
        9. 处理视频帧
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
            final_ee_positions = stride_ee_positions[keep_mask]
        else:
            final_indices = stride_indices
            final_joint_positions = stride_joint_positions
            final_ee_positions = stride_ee_positions
        
        # 提取其他数据
        final_timestamps = robot_data['timestamps'][final_indices]
        final_gripper_width = robot_data['gripper_width'][final_indices]
        
        # === 步骤3: 构建 robot_joint (8维) ===
        robot_joint_8 = self._build_robot_joint(final_joint_positions, final_gripper_width)
        
        # === 步骤4: 计算 robot_joint_vel (8维) ===
        robot_joint_vel_8 = self._compute_joint_vel(robot_joint_8, final_timestamps)
        
        # === 步骤5: 构建 robot_eef_pose (6维: xyz + rotation_vector) ===
        robot_eef_pose_6 = self._convert_ee_to_6d(final_ee_positions)
        
        # === 步骤6: 计算 robot_eef_pose_vel (6维) ===
        robot_eef_pose_vel_6 = self._compute_ee_vel(robot_eef_pose_6, final_timestamps)
        
        # === 步骤7: 构建 action (7维: 6维eef + 1维夹爪) ===
        action_7 = self._compute_action(robot_eef_pose_6, final_gripper_width)
        
        # === 步骤8: 构建 stage (全0) ===
        T = len(final_indices)
        stage = np.zeros(T, dtype=np.int64)
        
        # === 步骤9: 处理视频帧 ===
        processed_video_frames = {}
        for cam_name, frames in video_frames.items():
            if self.config.enable_frame_filtering:
                video_indices = [stride_indices.index(idx) for idx in final_indices]
                processed_video_frames[cam_name] = self._resize_frames(frames[video_indices])
            else:
                processed_video_frames[cam_name] = self._resize_frames(frames[:len(final_indices)])
        
        return {
            'robot_joint': robot_joint_8,  # (T, 8)
            'robot_joint_vel': robot_joint_vel_8,  # (T, 8)
            'robot_eef_pose': robot_eef_pose_6,  # (T, 6)
            'robot_eef_pose_vel': robot_eef_pose_vel_6,  # (T, 6)
            'action': action_7,  # (T, 7) - 6维eef + 1维夹爪
            'stage': stage,  # (T,)
            'timestamp': final_timestamps,  # (T,)
            'video_frames': processed_video_frames,  # Dict[cam_name, (T, H, W, 3)]
        }
    
    def _build_robot_joint(self, joint_positions: np.ndarray, gripper_width: np.ndarray) -> np.ndarray:
        """构建 robot_joint (8维: 7关节 + 1夹爪)
        
        Args:
            joint_positions: (T, 7) 关节位置
            gripper_width: (T,) 夹爪宽度
        
        Returns:
            robot_joint: (T, 8)
        """
        T = len(joint_positions)
        robot_joint = np.zeros((T, 8), dtype=np.float64)
        robot_joint[:, 0:7] = joint_positions[:, 0:7]  # 7个关节
        robot_joint[:, 7] = self._binarize_gripper(gripper_width)  # 夹爪二值化
        return robot_joint
    
    def _compute_joint_vel(self, robot_joint: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """计算关节速度 (数值微分)"""
        T = len(robot_joint)
        joint_vel = np.zeros_like(robot_joint, dtype=np.float64)
        
        for i in range(T - 1):
            dt = timestamps[i + 1] - timestamps[i]
            if dt > 0:
                joint_vel[i] = (robot_joint[i + 1] - robot_joint[i]) / dt
            else:
                joint_vel[i] = 0.0
        
        joint_vel[-1] = 0.0  # 最后一帧速度设为0
        return joint_vel
    
    def _convert_ee_to_6d(self, ee_positions: np.ndarray) -> np.ndarray:
        """将末端位姿从 xyz+quaternion(7维) 转换为 xyz+rotation_vector(6维)
        
        Args:
            ee_positions: (T, 7) - xyz(3) + quaternion(4) [qw, qx, qy, qz]
        
        Returns:
            eef_pose_6d: (T, 6) - xyz(3) + rotation_vector(3)
        """
        T = len(ee_positions)
        eef_pose_6d = np.zeros((T, 6), dtype=np.float64)
        
        # xyz 部分
        eef_pose_6d[:, 0:3] = ee_positions[:, 0:3]
        
        # quaternion -> rotation_vector
        ee_quat = ee_positions[:, 3:]  # (T, 4) - [qw, qx, qy, qz]
        # scipy 使用 [qx, qy, qz, qw] 格式
        quaternions_scipy = np.concatenate([
            ee_quat[:, 1:4],  # qx, qy, qz
            ee_quat[:, 0:1]   # qw
        ], axis=-1)
        rotations = R.from_quat(quaternions_scipy)
        eef_pose_6d[:, 3:6] = rotations.as_rotvec()
        
        return eef_pose_6d
    
    def _compute_ee_vel(self, eef_pose: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """计算末端速度 (数值微分)"""
        T = len(eef_pose)
        ee_vel = np.zeros_like(eef_pose, dtype=np.float64)
        
        for i in range(T - 1):
            dt = timestamps[i + 1] - timestamps[i]
            if dt > 0:
                ee_vel[i] = (eef_pose[i + 1] - eef_pose[i]) / dt
            else:
                ee_vel[i] = 0.0
        
        ee_vel[-1] = 0.0  # 最后一帧速度设为0
        return ee_vel
    
    def _compute_action(self, eef_pose: np.ndarray, gripper_width: np.ndarray) -> np.ndarray:
        """计算 action (7维: 6维末端位姿 + 1维夹爪)
        
        Action 取下一帧的末端位姿（绝对值）+ 下一帧的夹爪状态
        
        Args:
            eef_pose: (T, 6) 末端位姿
            gripper_width: (T,) 夹爪宽度
        
        Returns:
            action: (T, 7) - 6维eef + 1维夹爪
        """
        T = len(eef_pose)
        action = np.zeros((T, 7), dtype=np.float64)
        
        # 末端位姿部分：使用下一帧
        action[:-1, 0:6] = eef_pose[1:]  # 下一帧的末端位姿
        action[-1, 0:6] = eef_pose[-1]   # 最后一帧保持不变
        
        # 夹爪部分：使用下一帧的夹爪状态
        gripper_binary = self._binarize_gripper(gripper_width)
        action[:-1, 6] = gripper_binary[1:]  # 下一帧的夹爪
        action[-1, 6] = gripper_binary[-1]   # 最后一帧保持不变
        
        return action
    
    def _filter_frames_by_ee_motion(self, ee_positions: np.ndarray) -> np.ndarray:
        """根据末端位姿变化幅度过滤关键帧
        
        参考 franka_to_act_hdf5.py 的实现
        
        Args:
            ee_positions: (T, 7) 末端位姿 xyz(3) + quaternion(4)
        
        Returns:
            keep_mask: (T,) 布尔数组
        """
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
        
        # 计算相邻帧之间的末端位姿变化幅度
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
            print(f"  [Frame Filter] {T} -> {n_kept} frames "
                  f"(kept {filter_ratio:.1%}, threshold={threshold:.6f})")
        
        return keep_mask
    
    def _binarize_gripper(self, gripper: np.ndarray) -> np.ndarray:
        """夹爪二值化
        
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


# ===== Zarr 转换模块 =====

class ZarrConverter:
    """Zarr 数据集转换器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = config.output_dir / config.output_name
        self.zarr_path = self.output_dir / "replay_buffer.zarr"
        self.videos_dir = self.output_dir / "videos"
        
        # 用于累积所有 episode 的数据
        self.all_data = {
            'action': [],
            'robot_eef_pose': [],
            'robot_eef_pose_vel': [],
            'robot_joint': [],
            'robot_joint_vel': [],
            'stage': [],
            'timestamp': [],
        }
        self.episode_ends = []
        self.total_frames = 0
        
    def initialize(self):
        """初始化输出目录"""
        if self.output_dir.exists():
            import shutil
            print(f"Warning: Output directory exists, removing: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
    
    def accumulate_episode(self, episode_idx: int, processed_data: Dict[str, np.ndarray]):
        """累积一个 episode 的数据"""
        print(f'  [Accumulate] Starting episode_{episode_idx}...')
        
        # 累积数据
        for key in ['action', 'robot_eef_pose', 'robot_eef_pose_vel', 
                    'robot_joint', 'robot_joint_vel', 'stage', 'timestamp']:
            self.all_data[key].append(processed_data[key])
        
        # 更新 episode_ends
        T = len(processed_data['timestamp'])
        self.total_frames += T
        self.episode_ends.append(self.total_frames)
        
        print(f'  [Accumulate] Saving videos for episode_{episode_idx}...')
        # 保存视频
        self._save_videos(episode_idx, processed_data['video_frames'])
        
        print(f'  [Accumulate] Done episode_{episode_idx} (T={T}, total_frames={self.total_frames})')
    
    def _save_videos(self, episode_idx: int, video_frames: Dict[str, np.ndarray]):
        """保存视频文件
        
        Args:
            episode_idx: episode 索引
            video_frames: Dict[cam_name, (T, H, W, 3)]
        """
        # 创建 episode 子目录
        episode_video_dir = self.videos_dir / str(episode_idx)
        episode_video_dir.mkdir(parents=True, exist_ok=True)
        
        # 遍历相机
        for cam_name, frames in video_frames.items():
            # 获取相机编号
            cam_idx = self.config.camera_mapping.get(cam_name, 0)
            video_path = episode_video_dir / f"{cam_idx}.mp4"
            
            print(f'    [Video] Writing {cam_name} -> {cam_idx}.mp4 ({frames.shape[0]} frames)...')
            # 写入视频
            self._write_video(str(video_path), frames)
            print(f'    [Video] Done {cam_idx}.mp4')
    
    def _write_video(self, video_path: str, frames: np.ndarray):
        """写入视频文件
        
        Args:
            video_path: 视频路径
            frames: (T, H, W, 3) RGB uint8
        """
        T, H, W, C = frames.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, self.config.video_fps, (W, H))
        
        for i in range(T):
            frame_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
    
    def finalize(self, num_episodes: int, success_count: int):
        """完成转换，保存 Zarr 数据集"""
        print(f"\n{'='*60}")
        print(f"Finalizing Zarr dataset...")
        print(f"{'='*60}")
        
        # 合并所有数据
        merged_data = {}
        for key, data_list in self.all_data.items():
            if len(data_list) > 0:
                merged_data[key] = np.concatenate(data_list, axis=0)
            else:
                merged_data[key] = np.array([])
        
        episode_ends_array = np.array(self.episode_ends, dtype=np.int64)
        
        # 打印统计信息
        print(f"Total episodes: {num_episodes}")
        print(f"Success episodes: {success_count}")
        print(f"Total timesteps: {self.total_frames}")
        print(f"Data shapes:")
        for key, arr in merged_data.items():
            print(f"  {key}: {arr.shape}")
        print(f"  episode_ends: {episode_ends_array.shape}")
        
        # 创建 Zarr 数据集
        print(f"\nCreating Zarr dataset at {self.zarr_path}...")
        root = zarr.open(str(self.zarr_path), mode='w')
        
        # 创建 data 组
        data_group = root.create_group('data')
        
        # 保存数据，使用压缩
        compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=2)
        chunk_size = 170  # 参考 pusht_real 的 chunk 设置
        
        # action: (N, 7)
        data_group.create_dataset(
            'action',
            data=merged_data['action'],
            chunks=(chunk_size, merged_data['action'].shape[1]),
            compressor=compressor,
            dtype=np.float64
        )
        
        # robot_eef_pose: (N, 6)
        data_group.create_dataset(
            'robot_eef_pose',
            data=merged_data['robot_eef_pose'],
            chunks=(chunk_size, merged_data['robot_eef_pose'].shape[1]),
            compressor=compressor,
            dtype=np.float64
        )
        
        # robot_eef_pose_vel: (N, 6)
        data_group.create_dataset(
            'robot_eef_pose_vel',
            data=merged_data['robot_eef_pose_vel'],
            chunks=(chunk_size, merged_data['robot_eef_pose_vel'].shape[1]),
            compressor=compressor,
            dtype=np.float64
        )
        
        # robot_joint: (N, 8)
        data_group.create_dataset(
            'robot_joint',
            data=merged_data['robot_joint'],
            chunks=(chunk_size, merged_data['robot_joint'].shape[1]),
            compressor=compressor,
            dtype=np.float64
        )
        
        # robot_joint_vel: (N, 8)
        data_group.create_dataset(
            'robot_joint_vel',
            data=merged_data['robot_joint_vel'],
            chunks=(chunk_size, merged_data['robot_joint_vel'].shape[1]),
            compressor=compressor,
            dtype=np.float64
        )
        
        # stage: (N,)
        data_group.create_dataset(
            'stage',
            data=merged_data['stage'],
            chunks=(chunk_size,),
            compressor=compressor,
            dtype=np.int64
        )
        
        # timestamp: (N,)
        data_group.create_dataset(
            'timestamp',
            data=merged_data['timestamp'],
            chunks=(chunk_size,),
            compressor=compressor,
            dtype=np.float64
        )
        
        # 创建 meta 组
        meta_group = root.create_group('meta')
        
        # episode_ends: (num_episodes,)
        meta_group.create_dataset(
            'episode_ends',
            data=episode_ends_array,
            chunks=(len(episode_ends_array),),
            compressor=None,
            dtype=np.int64
        )
        
        print(f"\nZarr dataset saved successfully!")
        print(f"{'='*60}")
        print(f"Dataset location: {self.output_dir}")
        print(f"  - Zarr data: {self.zarr_path}")
        print(f"  - Videos: {self.videos_dir}")
        print(f"{'='*60}")


# ===== 主流程 =====

def main():
    """主函数"""
    config = Config()
    
    # 可选: 修改配置
    # config.stride = 2
    # config.enable_frame_filtering = True
    # config.frame_filter_threshold = 0.0001
    # config.max_episodes = 10  # None 表示转换所有 episodes
    
    print(f"{'='*60}")
    print(f"Franka to Zarr Format Converter")
    print(f"{'='*60}")
    print(f"Data root: {config.data_root / config.task_folder}")
    print(f"Output: {config.output_dir / config.output_name}")
    print(f"Action space: End-Effector Pose (6D) + Gripper (1D)")
    print(f"State space: Joint Position (7D) + Gripper (1D)")
    print(f"Stride: {config.stride}")
    print(f"Frame filtering: {config.enable_frame_filtering}")
    if config.enable_frame_filtering:
        print(f"  Threshold: {config.frame_filter_threshold}")
        print(f"  Min frames: {config.min_frames_per_episode}")
    print(f"Camera mapping: {config.camera_mapping}")
    print(f"{'='*60}\n")
    
    # 初始化模块
    loader = FrankaDataLoader(config)
    processor = DataProcessor(config)
    converter = ZarrConverter(config)
    
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
            
            # 累积到 Zarr 数据集
            converter.accumulate_episode(episode_idx, processed_data)
            
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing episode {episode_idx} ({episode}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 完成转换，保存 Zarr
    converter.finalize(len(episodes), success_count)


if __name__ == "__main__":
    main()
