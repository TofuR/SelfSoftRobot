"""SDF Dataset — 从仿真数据生成 3D 查询点 + SDF 监督信号。

每帧数据包含:
  - positions (T, 3, N_nodes): 杆体节点 3D 坐标
  - radii (T, N_nodes-1): 节点半径
  - actions (T, action_dim): 驱动扭矩

生成训练样本:
  - on-surface 点: 在杆体表面采样，SDF = 0
  - near-surface 点: 表面附近偏置采样，有精确 SDF 值
  - off-surface 点: 扩大空间均匀采样，有精确 SDF 值
  - 坐标归一化到 [-1, 1]^3
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class SDFDataset(Dataset):
    """3D SDF 监督数据集。"""

    def __init__(
        self,
        data_dir,
        seq_len=20,
        n_surface=300,
        n_near_surface=200,
        n_off_surface=200,
    ):
        self.seq_len = seq_len
        self.n_surface = n_surface
        self.n_near_surface = n_near_surface
        self.n_off_surface = n_off_surface
        self.samples = []
        self.data_cache = []

        file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not file_list:
            raise FileNotFoundError(f"No .npz files in {data_dir}")

        all_acts = []
        for f in file_list:
            d = np.load(f)
            if 'actions' in d:
                all_acts.append(d['actions'])
        if all_acts:
            all_acts = np.concatenate(all_acts, axis=0)
            self.norm_factor = np.max(np.abs(all_acts)) or 1.0
        else:
            self.norm_factor = 1.0

        # 计算全局坐标范围用于归一化
        all_pos = []
        for f in file_list:
            d = np.load(f)
            if 'positions' in d:
                all_pos.append(d['positions'])
        if all_pos:
            all_pos = np.concatenate(all_pos, axis=0)  # (total_T, 3, N)
            pos_min = all_pos.reshape(3, -1).min(axis=1)
            pos_max = all_pos.reshape(3, -1).max(axis=1)
            center = (pos_min + pos_max) / 2
            half_extent = (pos_max - pos_min).max() / 2 * 1.1
            self.coord_center = center.astype(np.float32)
            self.coord_scale = float(half_extent)
        else:
            self.coord_center = np.zeros(3, dtype=np.float32)
            self.coord_scale = 1.0

        self.action_dim = None
        for f_path in file_list:
            raw = np.load(f_path)
            actions = raw['actions'] / self.norm_factor

            if 'positions' not in raw:
                continue

            positions = raw['positions'].astype(np.float32)
            radii = raw['radii'].astype(np.float32) if 'radii' in raw else None

            if self.action_dim is None:
                self.action_dim = actions.shape[1]

            entry = {
                'actions': actions,
                'positions': positions,
                'radii': radii,
                'length': len(positions),
            }
            self.data_cache.append(entry)

        for seq_id, item in enumerate(self.data_cache):
            T = item['length']
            for t in range(self.seq_len - 1, T):
                self.samples.append((seq_id, t))

        print(f"SDFDataset: {len(self.samples)} samples, action_dim={self.action_dim}")
        print(f"  coord_center={self.coord_center}, coord_scale={self.coord_scale:.4f}")

    def _get_action_window(self, data, t):
        start = t - self.seq_len + 1
        end = t + 1
        if start >= 0:
            return data['actions'][start:end].copy()
        pad = np.zeros((-start, self.action_dim), dtype=data['actions'].dtype)
        return np.concatenate([pad, data['actions'][0:end]], axis=0)

    def _normalize(self, coords):
        """ coords: (..., 3) → 归一化到 [-1, 1]^3 """
        return (coords - self.coord_center[:, None]) / self.coord_scale

    def _unnormalize(self, coords):
        """ coords: (..., 3) in [-1, 1] → 原始坐标 """
        return coords * self.coord_scale + self.coord_center[:, None]

    def _compute_sdf_to_rod(self, points, positions, radii):
        """计算点到 Cosserat 杆体的精确有符号距离。

        Args:
            points: (3, M) 查询点
            positions: (3, N) 杆体节点坐标
            radii: (N-1,) 各段半径

        Returns:
            sdf: (M,) 有符号距离（外正内负）
            normals: (3, M) 表面法向量
        """
        N = positions.shape[1]
        avg_radius = float(np.mean(radii)) if radii is not None else 0.015
        M = points.shape[1]

        min_dist = np.full(M, 1e6, dtype=np.float32)
        normals = np.zeros((3, M), dtype=np.float32)

        for i in range(N - 1):
            seg_start = positions[:, i]
            seg_end = positions[:, i + 1]
            seg_vec = seg_end - seg_start
            seg_len = np.linalg.norm(seg_vec)
            if seg_len < 1e-8:
                continue
            seg_dir = seg_vec / seg_len

            v = points - seg_start[:, None]
            t_proj = np.clip(np.dot(seg_dir, v), 0, seg_len)
            closest = seg_start[:, None] + seg_dir[:, None] * t_proj[None, :]

            diff = points - closest
            dist = np.linalg.norm(diff, axis=0)

            closer = dist < min_dist
            min_dist[closer] = dist[closer]
            normal_dir = diff[:, closer]
            norm = np.linalg.norm(normal_dir, axis=0, keepdims=True) + 1e-8
            normals[:, closer] = normal_dir / norm

        sdf = min_dist - avg_radius
        return sdf, normals

    def _compute_sdf_and_normals(self, positions, radii):
        """生成 on-surface + near-surface + off-surface 采样点。"""
        N = positions.shape[1]
        avg_radius = float(np.mean(radii)) if radii is not None else 0.015

        # ── On-surface: 在杆体表面采样, SDF = 0 ──
        n_surf = self.n_surface
        seg_idx = np.random.randint(0, N - 1, size=n_surf)
        t_param = np.random.rand(n_surf)
        axis_pts = positions[:, seg_idx] * (1 - t_param) + \
                   positions[:, np.minimum(seg_idx + 1, N - 1)] * t_param

        theta = np.random.rand(n_surf) * 2 * np.pi
        surf_normals = np.zeros((3, n_surf), dtype=np.float32)
        surf_normals[0] = np.cos(theta)
        surf_normals[1] = np.sin(theta)
        surface_pts = axis_pts + avg_radius * surf_normals
        sdf_surface = np.zeros(n_surf, dtype=np.float32)

        # ── Near-surface: 表面附近偏置采样 ──
        n_near = self.n_near_surface
        seg_idx_ns = np.random.randint(0, N - 1, size=n_near)
        t_param_ns = np.random.rand(n_near)
        axis_pts_ns = positions[:, seg_idx_ns] * (1 - t_param_ns) + \
                      positions[:, np.minimum(seg_idx_ns + 1, N - 1)] * t_param_ns

        theta_ns = np.random.rand(n_near) * 2 * np.pi
        direction_ns = np.zeros((3, n_near), dtype=np.float32)
        direction_ns[0] = np.cos(theta_ns)
        direction_ns[1] = np.sin(theta_ns)

        offset_dist = (np.random.rand(n_near) * 6 - 3) * avg_radius
        near_pts = axis_pts_ns + (avg_radius + offset_dist) * direction_ns

        sdf_near, normals_near = self._compute_sdf_to_rod(near_pts, positions, radii)

        # ── Off-surface: [-1, 1]^3 均匀采样（归一化空间） ──
        n_off = self.n_off_surface
        off_pts_norm = np.random.uniform(-1, 1, size=(3, n_off)).astype(np.float32)
        off_pts = self._unnormalize(off_pts_norm)

        sdf_off, normals_off = self._compute_sdf_to_rod(off_pts, positions, radii)

        # ── 合并 ──
        coords = np.concatenate([surface_pts, near_pts, off_pts], axis=1)
        sdf = np.concatenate([sdf_surface, sdf_near, sdf_off], axis=0)
        normals_all = np.concatenate([surf_normals, normals_near, normals_off], axis=1)

        coords = self._normalize(coords)

        return coords.T, sdf, normals_all.T

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_id, t = self.samples[idx]
        data = self.data_cache[seq_id]

        action_window = self._get_action_window(data, t)
        positions = data['positions'][t]
        radii = data['radii'][t] if data['radii'] is not None else None

        coords, sdf, normals = self._compute_sdf_and_normals(
            positions, radii)

        return (
            torch.from_numpy(action_window).float(),
            torch.from_numpy(coords).float(),
            torch.from_numpy(sdf).float(),
            torch.from_numpy(normals).float(),
        )
