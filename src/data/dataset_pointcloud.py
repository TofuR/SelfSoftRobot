"""dataset_pointcloud.py — 点云数据集，用于 Flow Matching 训练。

每个样本返回 2 元组:
  (action_window, gt_pointcloud)
  - action_window:  (seq_len, action_dim) 归一化动作历史
  - gt_pointcloud:  (n_surface_points, 3) 归一化后的表面点云（~[-1, 1]³）

每次 __getitem__ 重新随机采样表面点，作为数据增强。
表面采样逻辑复用 SkeletonSDFDataset._sample_surface()。
"""

import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    """点云数据集。

    从 NPZ 文件加载 PyElastica 仿真数据（positions + radii），
    在每次访问时随机采样表面点云作为训练目标。

    Args:
        data_dir: 数据目录路径，包含 .npz 文件。
        seq_len: 时序窗口长度。
        n_surface_points: 每帧采样的表面点数量。
    """

    def __init__(self, data_dir, seq_len=20, n_surface_points=1000):
        self.seq_len = seq_len
        self.n_surface_points = n_surface_points
        self.samples = []
        self.data_cache = []

        file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not file_list:
            raise FileNotFoundError(f"No .npz files in {data_dir}")

        # 动作归一化因子
        all_acts = []
        for f in file_list:
            d = np.load(f)
            if 'actions' in d:
                all_acts.append(d['actions'])
        self.norm_factor = (
            float(np.max(np.abs(np.concatenate(all_acts)))) if all_acts else 1.0
        )

        # 缓存数据
        self.action_dim = None
        for f_path in file_list:
            raw = np.load(f_path)
            if 'positions' not in raw:
                continue
            actions = raw['actions'] / self.norm_factor
            positions = raw['positions'].astype(np.float32)  # (T, 3, 31)
            radii = raw['radii'].astype(np.float32) if 'radii' in raw else None
            if self.action_dim is None:
                self.action_dim = actions.shape[1]
            self.data_cache.append({
                'actions': actions,
                'positions': positions,
                'radii': radii,
                'length': len(positions),
            })

        # 构建 sample index: (seq_id, timestep)
        for seq_id, item in enumerate(self.data_cache):
            T = item['length']
            for t in range(self.seq_len - 1, T):
                self.samples.append((seq_id, t))

        print(f"PointCloudDataset: {len(self.samples)} samples, "
              f"action_dim={self.action_dim}, n_seqs={len(self.data_cache)}")

        # 计算点云归一化参数（per-axis center + scale → 映射到 [-1, 1]³）
        self._compute_normalization()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_id, t = self.samples[idx]
        data = self.data_cache[seq_id]

        action_window = self._get_action_window(data, t)
        positions = data['positions'][t]  # (3, N_nodes)
        radii = data['radii'][t] if data['radii'] is not None else None
        avg_radius = float(np.mean(radii)) if radii is not None else 0.015

        # 采样表面点云（复用 SkeletonSDFDataset 的采样逻辑）
        surface_pts, _ = _sample_surface(positions, avg_radius, self.n_surface_points)

        # 如果采样点不足，用最后一个点填充
        if len(surface_pts) < self.n_surface_points:
            pad = np.tile(
                surface_pts[-1:], (self.n_surface_points - len(surface_pts), 1))
            surface_pts = np.concatenate([surface_pts, pad], axis=0)
        # 截断到目标数量
        surface_pts = surface_pts[:self.n_surface_points]

        # 归一化到 [-1, 1]³
        surface_pts = (surface_pts - self.pc_center) / self.pc_scale

        return (
            torch.from_numpy(action_window).float(),
            torch.from_numpy(surface_pts).float(),
        )

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _compute_normalization(self):
        """从所有数据计算 per-axis center + scale，映射到 [-1, 1]³。"""
        all_pts = []
        for item in self.data_cache:
            positions = item['positions']  # (T, 3, N)
            radii_val = item['radii']
            avg_radius = float(np.mean(radii_val)) if radii_val is not None else 0.015
            # 从均匀间隔的帧采样表面点来估计范围
            T = item['length']
            for t_idx in range(0, T, max(1, T // 5)):
                pts, _ = _sample_surface(positions[t_idx], avg_radius, 200)
                if len(pts) > 0:
                    all_pts.append(pts)

        if all_pts:
            all_pts = np.concatenate(all_pts, axis=0)  # (M, 3)
            pc_min = all_pts.min(axis=0)
            pc_max = all_pts.max(axis=0)
            self.pc_center = ((pc_max + pc_min) / 2.0).astype(np.float32)  # (3,)
            self.pc_scale = np.maximum(
                ((pc_max - pc_min) / 2.0).astype(np.float32), 1e-6)         # (3,)
        else:
            self.pc_center = np.zeros(3, dtype=np.float32)
            self.pc_scale = np.ones(3, dtype=np.float32)

        print(f"  Normalization: center={self.pc_center}, scale={self.pc_scale}")

    def get_normalization_params(self):
        """返回归一化参数（供模型推理时反归一化使用）。

        Returns:
            (pc_center, pc_scale): 各为 (3,) numpy array。
        """
        return self.pc_center, self.pc_scale

    def _get_action_window(self, data, t):
        """获取以 t 结尾的时序动作窗口，不足时 zero-pad。"""
        start = t - self.seq_len + 1
        end = t + 1
        if start >= 0:
            return data['actions'][start:end].copy()
        pad = np.zeros((-start, self.action_dim), dtype=data['actions'].dtype)
        return np.concatenate([pad, data['actions'][0:end]], axis=0)


# ── 表面采样（复用 SkeletonSDFDataset._sample_surface 的逻辑）──────────


def _sample_surface(positions, radius, n_points):
    """在杆体表面均匀采样点及其法向量。

    与 SkeletonSDFDataset._sample_surface 相同的实现，
    提取为独立函数以便复用。

    Args:
        positions: (3, N) 杆体节点坐标。
        radius: 杆体半径。
        n_points: 目标采样点数。

    Returns:
        pts: (n_points, 3), normals: (n_points, 3)
    """
    N = positions.shape[1]
    n_segs = N - 1
    n_per_seg = max(1, n_points // n_segs)

    pts_list, normals_list = [], []
    for i in range(n_segs):
        p1, p2 = positions[:, i], positions[:, i + 1]
        seg_vec = p2 - p1
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-8:
            continue
        tangent = seg_vec / seg_len

        # 构建法平面正交基
        ref = (np.array([0.0, 1.0, 0.0]) if abs(tangent[1]) < 0.99
               else np.array([1.0, 0.0, 0.0]))
        perp1 = np.cross(tangent, ref)
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(tangent, perp1)

        # 沿线段插值 + 圆周采样
        n_seg = min(n_per_seg, n_points - len(pts_list))
        if n_seg <= 0:
            break
        t_param = np.random.rand(n_seg)
        centers = p1[:, None] * (1 - t_param[None, :]) + p2[:, None] * t_param[None, :]
        angles = np.random.rand(n_seg) * 2 * np.pi
        offsets = radius * (
            np.cos(angles)[:, None] * perp1[None, :] +
            np.sin(angles)[:, None] * perp2[None, :])
        pts_list.append(centers.T + offsets)
        normals_list.append(
            np.cos(angles)[:, None] * perp1[None, :] +
            np.sin(angles)[:, None] * perp2[None, :])

    if not pts_list:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    return (np.concatenate(pts_list, axis=0).astype(np.float32),
            np.concatenate(normals_list, axis=0).astype(np.float32))
