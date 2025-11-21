# ===== 导入 & 全局参数 =====
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import json
import jsonlines
import numpy as np
import torch
import cv2
import tqdm
import h5py  # 只用于类型提示，实际不需hdf5

# LeRobot：默认把数据写到 ~/.cache/lerobot/datasets/<repo_id>
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset

# 建议：限制底层库线程，避免多核机器上“过度并行”导致抖动（删除也不影响正确性）
os.environ.setdefault("OPENCV_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# 输入根目录：包含日期/任务/时间/v1/data/*.jsonl 和 videos/*.mp4
ALOHA_ROOT_DIR: Path = Path("/root/pi0_work/data_process/aloha_data_process/data_aloha4")

# 要处理的日期列表
DATA_DATES: List[str] = ["2025_08_08"]

# 输出数据集名称（会创建到 ~/.cache/lerobot/datasets/<repo_id>）
REPO_ID: str = "your_org/aloha_full_raw"

# 目标图像分辨率（H, W）：写入前统一到 224×224（常见训练配置）
TARGET_HW: Tuple[int, int] = (224, 224)

# （可选）只处理前 N 个 episode（调试用）；None 表示处理全部
MAX_EPISODES: Optional[int] = None

# ALOHA 臂索引（用于文件命名，如 aloha_4_arms_left.jsonl）
ALOHA_IDX: int = 4

# 数据处理设置（从第一个代码继承）
DATA_PROCESS_SETTING: int = 1
DATA_STRIDE_X: int = 1


# ===== 数据读取 & 处理模块（从第一个和第二个代码整合）=====

def get_raw_data(jsonl_path: str) -> Tuple[np.ndarray, ...]:
    """从 jsonl 读取原始数据，返回 timestamps, joint_positions, joint_vel, qpos, ee_pose_quaternion, ee_pose_rpy, gripper"""
    with jsonlines.open(jsonl_path) as reader:
        data = list(reader)

    timestamps = np.array([entry['timestamp'] for entry in data])
    joint_positions = np.array([entry['joint_positions'] for entry in data])
    joint_vel = np.array([entry['joint_vel'] for entry in data])
    qpos = np.array([entry['qpos'] for entry in data])
    ee_pose_quaternion = np.array([entry['ee_pose_quaternion'] for entry in data])
    ee_pose_rpy = np.array([entry['ee_pose_rpy'] for entry in data])
    gripper = np.array([entry['gripper'] for entry in data])

    return timestamps, joint_positions, joint_vel, qpos, ee_pose_quaternion, ee_pose_rpy, gripper


def extract_frames_by_index(video_path: str, frame_idx: List[int], resize: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """按索引提取视频帧，返回 (T, H, W, 3) RGB uint8"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frames = []
    for i in frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize:
                frame = cv2.resize(frame, resize)
            frames.append(frame)
    cap.release()
    
    return np.array(frames)


def calculate_stride(data: np.ndarray, x: int, use_delta: bool = True) -> np.ndarray:
    """数据增量或取样"""
    data_processed = []
    for i in range(0, len(data) - x, x):
        if use_delta:
            delta = data[i + x] - data[i]
            data_processed.append(delta)
        else:
            data_processed.append(data[i])
    return np.array(data_processed)


def process_gripper(gripper_data: np.ndarray, threshold: float = 0.005) -> np.ndarray:
    """gripper 开关阈值处理"""
    return np.where(np.abs(gripper_data) < threshold, 0, 1)


def process_data_for_episode(
    left_jsonl: str, right_jsonl: str, cam_high_mp4: str, cam_left_wrist_mp4: str, cam_right_wrist_mp4: str,
    setting: int = DATA_PROCESS_SETTING, x: int = DATA_STRIDE_X
) -> Dict[str, np.ndarray]:
    """处理单个 episode 的数据：读取 jsonl，应用 stride/delta，提取帧，返回 processed_data 字典"""
    timestamps_left, joint_positions_left, joint_vel_left, qpos_left, ee_pose_quaternion_left, ee_pose_rpy_left, gripper_left = get_raw_data(left_jsonl)
    timestamps_right, joint_positions_right, joint_vel_right, qpos_right, ee_pose_quaternion_right, ee_pose_rpy_right, gripper_right = get_raw_data(right_jsonl)
    
    if setting == 1:
        # 获取视频帧索引（每 x 帧）
        frame_idx = list(range(0, len(timestamps_left) - x, x))

        timestamps_left = calculate_stride(timestamps_left, x, use_delta=False)
        joint_positions_left = calculate_stride(joint_positions_left, x, use_delta=True)
        joint_vel_left = calculate_stride(joint_vel_left, x, use_delta=False)
        qpos_left = calculate_stride(qpos_left, x, use_delta=False)
        ee_pose_quaternion_left = calculate_stride(ee_pose_quaternion_left, x, use_delta=False)
        ee_pose_rpy_left = calculate_stride(ee_pose_rpy_left, x, use_delta=True)
        gripper_left = process_gripper(calculate_stride(gripper_left, x, use_delta=False))

        timestamps_right = calculate_stride(timestamps_right, x, use_delta=False)
        joint_positions_right = calculate_stride(joint_positions_right, x, use_delta=True)
        joint_vel_right = calculate_stride(joint_vel_right, x, use_delta=False)
        qpos_right = calculate_stride(qpos_right, x, use_delta=False)
        ee_pose_quaternion_right = calculate_stride(ee_pose_quaternion_right, x, use_delta=False)
        ee_pose_rpy_right = calculate_stride(ee_pose_rpy_right, x, use_delta=True)
        gripper_right = process_gripper(calculate_stride(gripper_right, x, use_delta=False))

    # 提取视频帧
    cam_high_frames = extract_frames_by_index(cam_high_mp4, frame_idx)
    cam_left_wrist_frames = extract_frames_by_index(cam_left_wrist_mp4, frame_idx)
    cam_right_wrist_frames = extract_frames_by_index(cam_right_wrist_mp4, frame_idx)

    # 组织成字典
    processed_data = {
        'timestamps_left': timestamps_left,
        'joint_positions_left': joint_positions_left,
        'joint_vel_left': joint_vel_left,
        'qpos_left': qpos_left,
        'ee_pose_quaternion_left': ee_pose_quaternion_left,
        'ee_pose_rpy_left': ee_pose_rpy_left,
        'gripper_left': gripper_left,

        'timestamps_right': timestamps_right,
        'joint_positions_right': joint_positions_right,
        'joint_vel_right': joint_vel_right,
        'qpos_right': qpos_right,
        'ee_pose_quaternion_right': ee_pose_quaternion_right,
        'ee_pose_rpy_right': ee_pose_rpy_right,
        'gripper_right': gripper_right,

        'video_frames': {
            'cam_high': cam_high_frames,
            'cam_left_wrist': cam_left_wrist_frames,
            'cam_right_wrist': cam_right_wrist_frames
        }
    }

    return processed_data


# ===== 目录遍历 & 元信息模块（从第二个代码继承）=====

def get_all_tasks_and_times(date: str) -> Dict[str, List[str]]:
    """获取指定日期下的所有任务和时间文件夹"""
    date_dir = ALOHA_ROOT_DIR / date
    if not date_dir.exists():
        raise ValueError(f"Date directory for {date} not found!")

    tasks_and_times = {}
    for task_dir in date_dir.iterdir():
        if task_dir.is_dir():
            times = [time_dir.name for time_dir in task_dir.iterdir() if time_dir.is_dir()]
            tasks_and_times[task_dir.name] = times
    return tasks_and_times


def extract_task_info(date: str, task: str, time: str) -> Dict[str, str]:
    """从 meta.json 提取 task_name, prompt, robot_model"""
    meta_path = ALOHA_ROOT_DIR / date / task / time / 'v1' / 'meta.json'
    if not meta_path.exists():
        raise ValueError(f"meta.json not found for {date}/{task}/{time}")

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

    return {
        "task_name": meta_data["task_meta"]["task_name"],
        "prompt": meta_data["task_meta"]["prompt"],
        "robot_model": meta_data["robot_meta"]["robots"][0]["robot_model"]
    }


def get_episode_paths(date: str, task: str, time: str, arms_idx: int = ALOHA_IDX) -> Dict[str, str]:
    """获取单个 episode 的文件路径：jsonl 和 mp4"""
    data_dir = ALOHA_ROOT_DIR / date / task / time / 'v1' / 'data'
    cam_dir = ALOHA_ROOT_DIR / date / task / time / 'v1' / 'videos'

    if not data_dir.exists() or not cam_dir.exists():
        raise ValueError(f"Data or videos directory not found for {date}/{task}/{time}")

    paths = {
        'left_jsonl': str(data_dir / f'aloha_{arms_idx}_arms_left.jsonl'),
        'right_jsonl': str(data_dir / f'aloha_{arms_idx}_arms_right.jsonl'),
        'cam_high_mp4': str(cam_dir / 'cam_high_rgb.mp4'),
        'cam_left_wrist_mp4': str(cam_dir / 'cam_wrist_left_rgb.mp4'),
        'cam_right_wrist_mp4': str(cam_dir / 'cam_wrist_right_rgb.mp4'),
    }

    for path in paths.values():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    return paths


# ===== LeRobot 转换模块（从第三个代码继承 & 适配）=====

def detect_cameras(processed_data: Dict[str, np.ndarray]) -> List[str]:
    """从 processed_data['video_frames'] 键中获取相机名"""
    return list(processed_data.get('video_frames', {}).keys())


def estimate_fps_from_timestamps(processed_data: Dict[str, np.ndarray]) -> float:
    """从 timestamps_left (或 right) 估计 fps：中位数间隔倒数"""
    if 'timestamps_left' in processed_data:
        ts = processed_data['timestamps_left']
    elif 'timestamps_right' in processed_data:
        ts = processed_data['timestamps_right']
    else:
        return 30.0
    if len(ts) < 2:
        return 30.0
    dt = float(np.median(np.diff(ts)))
    return 1.0 / dt if dt > 0 else 30.0


def resize_to_chw(img_hw3: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """HxWx3 uint8 → resize 到 (Ht, Wt) → CHW (3, Ht, Wt)"""
    Ht, Wt = target_hw
    img = cv2.resize(img_hw3, (Wt, Ht), interpolation=cv2.INTER_AREA)
    return np.transpose(img, (2, 0, 1))  # CHW


def pick_qpos(processed_data: Dict[str, np.ndarray], side: str) -> np.ndarray:
    """提取 qpos_<side> (T,6) float32"""
    key = f"qpos_{side}"
    if key not in processed_data:
        raise KeyError(f"Missing {key} for state")
    return processed_data[key].astype(np.float32)


def pick_joint_positions(processed_data: Dict[str, np.ndarray], side: str) -> np.ndarray:
    """提取 joint_positions_<side> (T,6) float32（已是增量）"""
    key = f"joint_positions_{side}"
    if key not in processed_data:
        raise KeyError(f"Missing {key} for action")
    return processed_data[key].astype(np.float32)


def read_gripper(processed_data: Dict[str, np.ndarray], side: str) -> np.ndarray:
    """提取 gripper_<side> (T,) float32"""
    key = f"gripper_{side}"
    if key not in processed_data:
        raise KeyError(f"Missing {key}")
    return processed_data[key].astype(np.float32)


def build_state14_from_qpos(processed_data: Dict[str, np.ndarray]) -> np.ndarray:
    """构建 state: right(qpos6 + grip) + left(qpos6 + grip) → (T,14)"""
    qpos_r = pick_qpos(processed_data, "right")
    qpos_l = pick_qpos(processed_data, "left")
    grip_r = read_gripper(processed_data, "right")
    grip_l = read_gripper(processed_data, "left")
    right_7 = np.concatenate([qpos_r, grip_r[:, None]], axis=1)
    left_7 = np.concatenate([qpos_l, grip_l[:, None]], axis=1)
    return np.concatenate([right_7, left_7], axis=1).astype(np.float32)


def build_action14_from_jointpos(processed_data: Dict[str, np.ndarray]) -> np.ndarray:
    """构建 action: right(jpos6_delta + grip) + left(jpos6_delta + grip) → (T,14)"""
    jpos_r = pick_joint_positions(processed_data, "right")
    jpos_l = pick_joint_positions(processed_data, "left")
    grip_r = read_gripper(processed_data, "right")
    grip_l = read_gripper(processed_data, "left")
    right_7 = np.concatenate([jpos_r, grip_r[:, None]], axis=1)
    left_7 = np.concatenate([jpos_l, grip_l[:, None]], axis=1)
    return np.concatenate([right_7, left_7], axis=1).astype(np.float32)


def create_dataset_skeleton(first_processed_data: Dict[str, np.ndarray], first_task_info: Dict[str, str]) -> LeRobotDataset:
    """用第一个 episode 数据创建 LeRobot 骨架：自动探测相机 & 估 fps"""
    cam_names = detect_cameras(first_processed_data)
    if not cam_names:
        raise RuntimeError("No RGB cameras in video_frames")

    fps_to_use = estimate_fps_from_timestamps(first_processed_data)

    # features 定义
    motors = 14
    features: Dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (motors,),
            "names": [[
                "right_0","right_1","right_2","right_3","right_4","right_5","right_gripper",
                "left_0","left_1","left_2","left_3","left_4","left_5","left_gripper"
            ]],
        },
        "action": {
            "dtype": "float32",
            "shape": (motors,),
            "names": [[
                "right_0","right_1","right_2","right_3","right_4","right_5","right_gripper",
                "left_0","left_1","left_2","left_3","left_4","left_5","left_gripper"
            ]],
        },
    }
    C, Ht, Wt = 3, TARGET_HW[0], TARGET_HW[1]
    for cam in cam_names:
        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": (C, Ht, Wt),
            "names": ["channels", "height", "width"],
        }

    # 清理旧输出
    out_root = LEROBOT_HOME / REPO_ID
    if out_root.exists():
        import shutil
        shutil.rmtree(out_root)

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=fps_to_use,
        robot_type="aloha",
        features=features,
        use_videos=False,
        tolerance_s=1e-4,
        image_writer_processes=4,
        image_writer_threads=4,
        video_backend=None,
    )
    return dataset


def convert_one_episode(dataset: LeRobotDataset, processed_data: Dict[str, np.ndarray], task_info: Dict[str, str]) -> None:
    """把单个 processed_data 转成 LeRobot episode"""
    state14 = build_state14_from_qpos(processed_data)
    action14 = build_action14_from_jointpos(processed_data)

    cam_names = detect_cameras(processed_data)
    if not cam_names:
        raise RuntimeError("No RGB cameras in video_frames")
    video_frames = processed_data['video_frames']

    # 对齐长度
    T = min(state14.shape[0], action14.shape[0])
    for cam_data in video_frames.values():
        T = min(T, cam_data.shape[0])

    # 任务字符串
    task_name = task_info.get("task_name", "UNKNOWN_TASK")
    task_prompt = task_info.get("prompt", "")
    robot_model = task_info.get("robot_model", "")
    task_str = f"{task_name} | {task_prompt} | robot={robot_model}".strip()

    # 逐帧写入
    for t in range(T):
        frame = {
            "observation.state": torch.from_numpy(state14[t]),
            "action": torch.from_numpy(action14[t]),
            "task": task_str,
        }
        for cam in cam_names:
            img_hw3 = video_frames[cam][t]  # (H,W,3) uint8
            img_chw = resize_to_chw(img_hw3, TARGET_HW)
            frame[f"observation.images.{cam}"] = img_chw

        dataset.add_frame(frame)

    dataset.save_episode()


# ===== 主函数 =====

def main():
    # 1) 收集所有 episode 元组 (date, task, time)
    all_episodes: List[Tuple[str, str, str]] = []
    for date in DATA_DATES:
        tasks_and_times = get_all_tasks_and_times(date)
        for task, times in tasks_and_times.items():
            for time in times:
                all_episodes.append((date, task, time))

    if MAX_EPISODES is not None:
        all_episodes = all_episodes[:MAX_EPISODES]

    if not all_episodes:
        raise ValueError("No episodes found in specified dates")

    # 2) 用第一个 episode 创建数据集骨架（需先处理数据）
    first_date, first_task, first_time = all_episodes[0]
    first_task_info = extract_task_info(first_date, first_task, first_time)
    first_paths = get_episode_paths(first_date, first_task, first_time)
    first_processed = process_data_for_episode(
        first_paths['left_jsonl'], first_paths['right_jsonl'],
        first_paths['cam_high_mp4'], first_paths['cam_left_wrist_mp4'], first_paths['cam_right_wrist_mp4']
    )
    dataset = create_dataset_skeleton(first_processed, first_task_info)

    # 3) 逐 episode 处理 & 转换
    for date, task, time in tqdm.tqdm(all_episodes, desc="Converting episodes"):
        task_info = extract_task_info(date, task, time)
        paths = get_episode_paths(date, task, time)
        processed_data = process_data_for_episode(
            paths['left_jsonl'], paths['right_jsonl'],
            paths['cam_high_mp4'], paths['cam_left_wrist_mp4'], paths['cam_right_wrist_mp4']
        )
        convert_one_episode(dataset, processed_data, task_info)

    # 4) 收尾
    dataset.consolidate()
    out_dir = LEROBOT_HOME / REPO_ID
    print(f"Done. Saved LeRobot dataset to: {out_dir}")


if __name__ == "__main__":
    main()