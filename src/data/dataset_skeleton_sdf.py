"""SkeletonSDF 训练数据集 — 骨架监督 + SDF 采样。

每个样本返回 5 元组:
  (action_window, coords, gt_sdf, gt_normals, gt_positions)
  - action_window: (seq_len, action_dim) 归一化动作历史
  - coords:        (M, 3) 原始坐标空间的查询点
  - gt_sdf:        (M,) GT 有符号距离值
  - gt_normals:    (M, 3) 表面法向量（仅表面点有效）
  - gt_positions:  (n_fine, 3) GT 骨架节点坐标（原始空间）
"""

import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset


class SkeletonSDFDataset(Dataset):

    def __init__(self, data_dir, seq_len=20,
                 n_surface=500, n_near_surface=500, n_off_surface=500):
        self.seq_len = seq_len
        self.n_surface = n_surface
        self.n_near_surface = n_near_surface
        self.n_off_surface = n_off_surface
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
        self.norm_factor = float(np.max(np.abs(np.concatenate(all_acts)))) if all_acts else 1.0

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

        print(f"SkeletonSDFDataset: {len(self.samples)} samples, "
              f"action_dim={self.action_dim}, n_seqs={len(self.data_cache)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_id, t = self.samples[idx]
        data = self.data_cache[seq_id]

        action_window = self._get_action_window(data, t)
        positions = data['positions'][t]        # (3, 31)
        radii = data['radii'][t] if data['radii'] is not None else None

        coords, sdf, normals = self._sample_sdf_points(positions, radii)
        gt_positions = positions.T.copy()       # (31, 3)

        return (
            torch.from_numpy(action_window).float(),
            torch.from_numpy(coords).float(),
            torch.from_numpy(sdf).float(),
            torch.from_numpy(normals).float(),
            torch.from_numpy(gt_positions).float(),
        )

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _get_action_window(self, data, t):
        start = t - self.seq_len + 1
        end = t + 1
        if start >= 0:
            return data['actions'][start:end].copy()
        pad = np.zeros((-start, self.action_dim), dtype=data['actions'].dtype)
        return np.concatenate([pad, data['actions'][0:end]], axis=0)

    def _sample_sdf_points(self, positions, radii):
        """采样 SDF 训练点: 表面 + 近表面 + 远表面。"""
        avg_radius = float(np.mean(radii)) if radii is not None else 0.015

        # 1. 表面点: 沿杆体表面圆采样, SDF = 0
        surf_coords, surf_normals = self._sample_surface(
            positions, avg_radius, self.n_surface)
        n_surf = len(surf_coords)
        sdf_surf = np.zeros(n_surf, dtype=np.float32)

        # 2. 近表面点: 表面 + 随机偏移
        near_coords = surf_coords + (
            np.random.randn(*surf_coords.shape).astype(np.float32) * avg_radius * 2)
        sdf_near, normals_near = self._sdf_to_rod(near_coords.T, positions, avg_radius)

        # 3. 远表面点: 均匀空间采样
        margin = avg_radius * 5
        pos_min = positions.min(axis=1, keepdims=True) - margin
        pos_max = positions.max(axis=1, keepdims=True) + margin
        off_coords = np.random.uniform(
            pos_min, pos_max, size=(3, self.n_off_surface)).astype(np.float32).T
        sdf_off, normals_off = self._sdf_to_rod(off_coords.T, positions, avg_radius)

        coords = np.concatenate([surf_coords, near_coords, off_coords], axis=0)
        sdf = np.concatenate([sdf_surf, sdf_near, sdf_off])
        normals = np.concatenate([surf_normals, normals_near, normals_off], axis=0)

        return coords, sdf, normals

    @staticmethod
    def _sample_surface(positions, radius, n_points):
        """在杆体表面均匀采样点及其法向量。

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
            ref = np.array([0.0, 1.0, 0.0]) if abs(tangent[1]) < 0.99 \
                else np.array([1.0, 0.0, 0.0])
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

    @staticmethod
    def _sdf_to_rod(points, positions, radius):
        """计算点到杆体的精确 SDF 和法向量。

        Args:
            points: (3, M) 查询点
            positions: (3, N) 杆体节点
            radius: 管半径

        Returns:
            sdf: (M,), normals: (M, 3)
        """
        M = points.shape[1]
        min_dist = np.full(M, 1e6, dtype=np.float32)
        normals = np.zeros((M, 3), dtype=np.float32)

        for i in range(positions.shape[1] - 1):
            seg_s = positions[:, i]
            seg_e = positions[:, i + 1]
            seg_vec = seg_e - seg_s
            seg_len = np.linalg.norm(seg_vec)
            if seg_len < 1e-8:
                continue
            seg_dir = seg_vec / seg_len

            v = points - seg_s[:, None]
            t_proj = np.clip(seg_dir @ v, 0, seg_len)
            closest = seg_s[:, None] + seg_dir[:, None] * t_proj[None, :]
            diff = points - closest
            dist = np.linalg.norm(diff, axis=0)

            closer = dist < min_dist
            min_dist[closer] = dist[closer]
            norm = np.linalg.norm(diff[:, closer], axis=0, keepdims=True) + 1e-8
            normals[closer] = (diff[:, closer] / norm).T

        return min_dist - radius, normals
