"""SDF Dataset — 从仿真数据生成 3D 查询点 + SDF 监督信号。

每帧数据包含:
  - positions (3, N_nodes): 杆体节点 3D 坐标
  - radii (N_nodes,): 节点半径
  - actions (action_dim,): 驱动扭矩

生成训练样本:
  - on-surface 点: 在杆体表面采样，SDF = 0
  - off-surface 点: 随机采样，通过到杆体轴线的距离计算 SDF
  - 法向量: 从杆体几何计算
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
        n_off_surface=300,
        off_surface_range=3.0,
    ):
        self.seq_len = seq_len
        self.n_surface = n_surface
        self.n_off_surface = n_off_surface
        self.off_surface_range = off_surface_range
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

    def _get_action_window(self, data, t):
        start = t - self.seq_len + 1
        end = t + 1
        if start >= 0:
            return data['actions'][start:end].copy()
        pad = np.zeros((-start, self.action_dim), dtype=data['actions'].dtype)
        return np.concatenate([pad, data['actions'][0:end]], axis=0)

    def _compute_sdf_and_normals(self, positions, radii, n_surface, n_off):
        N = positions.shape[1]
        avg_radius = float(np.mean(radii)) if radii is not None else 0.015

        # On-surface: 在杆体表面采样
        seg_idx = np.random.randint(0, N - 1, size=n_surface)
        t_param = np.random.rand(n_surface)
        axis_pts = positions[:, seg_idx] * (1 - t_param) + \
                   positions[:, np.minimum(seg_idx + 1, N - 1)] * t_param

        theta = np.random.rand(n_surface) * 2 * np.pi
        normals_xy = np.zeros((3, n_surface), dtype=np.float32)
        normals_xy[0] = np.cos(theta)
        normals_xy[1] = np.sin(theta)
        surface_pts = axis_pts + avg_radius * normals_xy
        sdf_surface = np.zeros(n_surface, dtype=np.float32)

        # Off-surface: 随机采样空间点
        p_min = positions.min(axis=1) - avg_radius * self.off_surface_range
        p_max = positions.max(axis=1) + avg_radius * self.off_surface_range
        off_pts = np.random.uniform(p_min[:, None], p_max[:, None], size=(3, n_off)).astype(np.float32)

        sdf_off = np.full(n_off, avg_radius * self.off_surface_range, dtype=np.float32)
        normals_off = np.zeros((3, n_off), dtype=np.float32)

        for i in range(N - 1):
            seg_start = positions[:, i]
            seg_end = positions[:, i + 1]
            seg_vec = seg_end - seg_start
            seg_len = np.linalg.norm(seg_vec)
            if seg_len < 1e-8:
                continue
            seg_dir = seg_vec / seg_len

            v = off_pts - seg_start[:, None]
            t_proj = np.clip(np.dot(seg_dir, v), 0, seg_len)
            closest = seg_start[:, None] + seg_dir[:, None] * t_proj[None, :]

            diff = off_pts - closest
            dist = np.linalg.norm(diff, axis=0)

            closer = dist < sdf_off
            sdf_off[closer] = dist[closer]
            normal_dir = diff[:, closer]
            normal_dir = normal_dir / (np.linalg.norm(normal_dir, axis=0, keepdims=True) + 1e-8)
            normals_off[:, closer] = normal_dir

        sdf_off = sdf_off - avg_radius

        coords = np.concatenate([surface_pts, off_pts], axis=1).T
        sdf = np.concatenate([sdf_surface, sdf_off], axis=0)
        normals = np.concatenate([normals_xy, normals_off], axis=1).T

        return coords, sdf, normals

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_id, t = self.samples[idx]
        data = self.data_cache[seq_id]

        action_window = self._get_action_window(data, t)
        positions = data['positions'][t]
        radii = data['radii'][t] if data['radii'] is not None else None

        coords, sdf, normals = self._compute_sdf_and_normals(
            positions, radii, self.n_surface, self.n_off_surface)

        return (
            torch.from_numpy(action_window).float(),
            torch.from_numpy(coords).float(),
            torch.from_numpy(sdf).float(),
            torch.from_numpy(normals).float(),
        )
