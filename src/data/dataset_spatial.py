"""dataset_spatial.py — 空间序列数据集，用于 SpatialSequenceModel 训练。

每个样本返回 dict batch:
  action_window:     (seq_len, action_dim) 归一化动作历史
  action_window_next: (seq_len, action_dim) 下一帧动作历史（smooth loss 用）
  gt_skeleton:       (n_nodes, 3) 归一化后的中心线坐标
  gt_radii:          (n_nodes,) 各节点半径

与 PointCloudDataset 共享归一化参数和数据加载逻辑，
但返回中心线节点坐标而非采样表面点云。
"""

import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset


class SpatialSequenceDataset(Dataset):
    """空间序列数据集。

    从 NPZ 文件加载 PyElastica 仿真数据（positions + radii），
    返回中心线节点坐标作为训练目标。

    Args:
        data_dir: 数据目录路径，包含 .npz 文件。
        seq_len: 时序窗口长度。
        pairs: 是否返回相邻帧（smooth loss）。
    """

    def __init__(self, data_dir, seq_len=20, pairs=True):
        self.seq_len = seq_len
        self.pairs = pairs
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
            positions = raw['positions'].astype(np.float32)  # (T, 3, N)
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
            end = T - 1 if pairs else T
            for t in range(self.seq_len - 1, end):
                self.samples.append((seq_id, t))

        print(f"SpatialSequenceDataset: {len(self.samples)} samples, "
              f"action_dim={self.action_dim}, n_seqs={len(self.data_cache)}")

        # 计算归一化参数（基于中心线坐标范围）
        self._compute_normalization()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_id, t = self.samples[idx]
        data = self.data_cache[seq_id]

        action_window = self._get_action_window(data, t)

        # GT 中心线: (3, N) → (N, 3)，然后归一化
        positions = data['positions'][t].astype(np.float32)  # (3, N)
        skeleton = positions.T  # (N, 3)
        skeleton = (skeleton - self.pc_center) / self.pc_scale

        # GT 半径
        radii = data['radii']
        if radii is not None:
            radii = radii[t].astype(np.float32)  # (N-1,)
            # 从 (N-1,) 扩展到 (N,)（最后一个复用前一个）
            if len(radii) == skeleton.shape[0] - 1:
                radii = np.append(radii, radii[-1])
        else:
            radii = np.full(skeleton.shape[0], 0.015, dtype=np.float32)

        result = {
            "action_window": torch.from_numpy(action_window).float(),
            "gt_skeleton": torch.from_numpy(skeleton).float(),
            "gt_radii": torch.from_numpy(radii).float(),
            "action_window_next": None,
        }

        # 相邻帧（smooth loss 用）
        if self.pairs and t + 1 < data['length']:
            result["action_window_next"] = torch.from_numpy(
                self._get_action_window(data, t + 1)).float()

        return result

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _compute_normalization(self):
        """从所有中心线坐标计算 per-axis center + scale。"""
        all_skeletons = []
        for item in self.data_cache:
            positions = item['positions']  # (T, 3, N)
            T = item['length']
            for t_idx in range(0, T, max(1, T // 5)):
                skel = positions[t_idx].T  # (N, 3)
                all_skeletons.append(skel)

        if all_skeletons:
            all_pts = np.concatenate(all_skeletons, axis=0)  # (M, 3)
            pc_min = all_pts.min(axis=0)
            pc_max = all_pts.max(axis=0)
            self.pc_center = ((pc_max + pc_min) / 2.0).astype(np.float32)
            self.pc_scale = np.maximum(
                ((pc_max - pc_min) / 2.0).astype(np.float32), 1e-6)
        else:
            self.pc_center = np.zeros(3, dtype=np.float32)
            self.pc_scale = np.ones(3, dtype=np.float32)

        print(f"  Normalization: center={self.pc_center}, scale={self.pc_scale}")

    def get_normalization_params(self):
        """返回归一化参数。"""
        return self.pc_center, self.pc_scale

    def _get_action_window(self, data, t):
        """获取以 t 结尾的时序动作窗口，不足时 zero-pad。"""
        start = t - self.seq_len + 1
        end = t + 1
        if start >= 0:
            return data['actions'][start:end].copy()
        pad = np.zeros((-start, self.action_dim), dtype=data['actions'].dtype)
        return np.concatenate([pad, data['actions'][0:end]], axis=0)


def spatial_collate_fn(batch):
    """将 dict 样本列表合并为 batched dict。"""
    keys = batch[0].keys()
    result = {}
    for k in keys:
        vals = [item[k] for item in batch]
        if vals[0] is None:
            result[k] = None
        else:
            result[k] = torch.stack(vals, dim=0)
    return result


class StateTransitionDataset(SpatialSequenceDataset):
    """闭环状态转移数据集：在 SpatialSequenceDataset 基础上额外返回前一步骨架。

    用于 StateTransitionSpatialModel，模型学习状态转移 s_t = F(s_{t-1}, a_t, z_{t-1})，
    需要前一步的 GT 中心线作为输入（teacher forcing）。

    额外返回（相对父类）:
      prev_gt_skeleton:      (n_nodes, 3) 归一化后的 positions[t-1]（前一步中心线）
      prev_prev_gt_skeleton: (n_nodes, 3) 归一化后的 positions[t-2]（用于速度 v = s_{t-1} - s_{t-2}）

    边界说明:
      父类样本循环 t ∈ [seq_len-1, end]，其中 end = T-1（pairs=True）或 T。
      因此 t-1 ≥ seq_len-2 ≥ 0，t-2 ≥ seq_len-3，positions[t-1]/[t-2] 恒为同一 episode
      内的有效前驱帧，**无需 zero-pad**，也无跨 episode 污染。

    归一化:
      前一步骨架复用父类在同一数据集上计算的 pc_center / pc_scale（同一空间坐标系，
      仅时间步更早），保证与 gt_skeleton 处于同一归一化空间。
    """

    def __getitem__(self, idx):
        # 复用父类：返回 action_window, gt_skeleton(t), gt_radii, action_window_next
        result = super().__getitem__(idx)

        seq_id, t = self.samples[idx]
        data = self.data_cache[seq_id]

        # 前一步骨架 positions[t-1]: (3, N) → (N, 3) → 归一化（与父类 gt_skeleton 一致）
        prev_skeleton = data['positions'][t - 1].astype(np.float32).T
        prev_skeleton = (prev_skeleton - self.pc_center) / self.pc_scale

        # 前两步骨架 positions[t-2]: 用于计算速度 v = s_{t-1} - s_{t-2}
        prev_prev_skeleton = data['positions'][t - 2].astype(np.float32).T
        prev_prev_skeleton = (prev_prev_skeleton - self.pc_center) / self.pc_scale

        result["prev_gt_skeleton"] = torch.from_numpy(prev_skeleton).float()
        result["prev_prev_gt_skeleton"] = torch.from_numpy(prev_prev_skeleton).float()
        return result
