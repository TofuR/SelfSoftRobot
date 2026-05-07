"""collect_utils.py — 数据采集工具函数。

动作策略、数据保存、文件命名、配置加载。
被 collect.py 调用，也可被其他脚本独立使用。
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config.params import load_config, get_camera_params


# =============================================================================
# 配置加载
# =============================================================================

def load_defaults():
    """从 simulation.json + camera.json 加载采集默认参数。"""
    sim = load_config("simulation")
    cam = get_camera_params()
    col = sim["collection"]
    act = sim["action_space"]
    return {
        "dt": sim["time"]["dt"],
        "action_min": act["min_val"],
        "action_max": act["max_val"],
        "step_size": act["step_size"],
        "num_sequences": col["num_sequences"],
        "actions_per_seq": col["actions_per_seq"],
        "steps_per_action": col["steps_per_action"],
        "record_interval": col["record_interval"],
        "warmup_steps": col.get("warmup_steps", 2000),
        "camera": {
            "focal": float(cam["focal"]),
            "H": cam["H"],
            "W": cam["W"],
            "camera_eye": np.array(cam["eye"]),
            "camera_center": np.array(cam["center"]),
            "camera_up": np.array(cam["up"]),
        },
    }


# =============================================================================
# 动作策略
# =============================================================================

class ActionSchedule:
    """为每个维度独立生成动作序列。

    每个维度可选模式：
      zero   — 始终为 0（canonical 场景）
      random — 平滑随机游走（sequence 场景）
      hold   — 采样一个随机值后整段保持（batch 场景）
      file   — 从 npz 文件读取该维度序列

    用法:
        schedule = ActionSchedule(["random", "zero"], n_actions=50, ...)
        actions = schedule.generate()  # (50, 2)

        schedule = ActionSchedule(["file", "zero"], file_path="traj.npz")
        actions = schedule.generate()
    """

    VALID_MODES = {"zero", "random", "hold", "file"}

    def __init__(self, dim_modes, n_actions=50,
                 min_val=-0.005, max_val=0.005, step_size=0.001,
                 file_path=None, file_key="actions"):
        for m in dim_modes:
            if m not in self.VALID_MODES:
                raise ValueError(f"Invalid mode '{m}', choose from {self.VALID_MODES}")

        self.dim_modes = list(dim_modes)
        self.n_dims = len(dim_modes)
        self.n_actions = n_actions
        self.min_val = min_val
        self.max_val = max_val
        self.step_size = step_size
        self.file_path = file_path
        self.file_key = file_key

        # 预加载 file 数据
        self._file_data = None
        if file_path is not None:
            data = np.load(file_path)
            self._file_data = data[file_key]

    def generate(self):
        """生成动作序列。

        Returns:
            (n_actions, n_dims) 数组。
        """
        actions = np.zeros((self.n_actions, self.n_dims))

        for d, mode in enumerate(self.dim_modes):
            actions[:, d] = self._generate_dim(d, mode)

        return actions

    def _generate_dim(self, dim_idx, mode):
        """为单个维度生成动作序列。"""
        if mode == "zero":
            return np.zeros(self.n_actions)

        if mode == "random":
            return self._random_walk()

        if mode == "hold":
            val = np.random.uniform(self.min_val, self.max_val)
            return np.full(self.n_actions, val)

        if mode == "file":
            if self._file_data is None:
                raise ValueError("file mode requires --action-file")
            col = self._file_data[:, dim_idx]
            if len(col) < self.n_actions:
                # 不足则重复
                reps = (self.n_actions // len(col)) + 1
                col = np.tile(col, reps)
            return col[:self.n_actions]

    def _random_walk(self):
        """单维度平滑随机游走。"""
        values = np.zeros(self.n_actions)
        current = 0.0
        for i in range(self.n_actions):
            current += np.random.uniform(-self.step_size, self.step_size)
            current = np.clip(current, self.min_val, self.max_val)
            values[i] = current
        return values

    @property
    def mode_tag(self):
        """短标签如 'rr', 'rz', 'zz', 'hh'。"""
        return "".join(m[0] for m in self.dim_modes)


# =============================================================================
# 数据保存
# =============================================================================

def save_collection(path, images, actions, dt, camera,
                    positions=None, radii=None, depth_maps=None):
    """保存采集数据到 npz，始终嵌入相机参数。"""
    data = {
        "images": np.array(images),
        "actions": np.array(actions),
        "dt": dt,
        "focal": camera["focal"],
        "H": camera["H"],
        "W": camera["W"],
        "camera_eye": camera["camera_eye"],
        "camera_center": camera["camera_center"],
        "camera_up": camera["camera_up"],
    }
    if positions is not None:
        data["positions"] = np.array(positions)
    if radii is not None:
        data["radii"] = np.array(radii)
    if depth_maps is not None:
        data["depth_maps"] = np.array(depth_maps, dtype=np.float32)
    np.savez_compressed(path, **data)


# =============================================================================
# 文件命名
# =============================================================================

def make_filename(seq_idx, mode_tag, has_3d, timestamp=None):
    """生成自描述文件名。

    示例:
      seq_000_rz_1748000000.npz        # random + zero
      seq_000_rr_3d_1748000000.npz     # 两维 random + 3D
      seq_000_zz_1748000000.npz        # 两维 zero (canonical)
      seq_000_rh_1748000000.npz        # random + hold
    """
    ts = timestamp or int(time.time())
    tag = f"seq_{seq_idx:03d}_{mode_tag}"
    if has_3d:
        tag += "_3d"
    return f"{tag}_{ts}.npz"


def infer_save_dir(mode_tag, has_3d, user_dir=None):
    """根据动作模式和是否 3D 推断保存目录。

    示例:
      data/seq_rr/          # 两维 random
      data/seq_rr_3d/       # 两维 random + 3D
      data/seq_zz/          # 两维 zero (canonical)
      data/seq_rz/          # random + zero
    """
    if user_dir:
        return user_dir
    suffix = "_3d" if has_3d else ""
    return f"data/seq_{mode_tag}{suffix}"
