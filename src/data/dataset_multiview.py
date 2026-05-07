"""dataset_multiview.py — 多视角数据集。

从双视角 npz 文件加载图像、动作、预计算的 2D 骨架。
数据格式要求:
  npz:
    images_front: (T, H, W)
    images_side:  (T, H, W)
    actions:      (T, 2)
    positions:    (T, 3, 31)   可选，仅用于评估
    camera_eye_front, camera_center_front, camera_up_front
    camera_eye_side, camera_center_side, camera_up_side
    focal, H, W
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.skeleton_2d import batch_extract_skeleton_2d


class MultiViewDataset(Dataset):
    """双视角序列数据集，带预计算 2D 骨架。"""

    def __init__(self, data_dir, seq_len=20, return_3d=False, n_skeleton_points=31):
        self.seq_len = seq_len
        self.return_3d = return_3d
        self.n_skeleton_points = n_skeleton_points
        self.samples = []

        file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not file_list:
            raise FileNotFoundError(f"No .npz files in {data_dir}")

        # 归一化系数
        all_acts = []
        for f in file_list:
            d = np.load(f)
            all_acts.append(d['actions'])
        all_acts = np.concatenate(all_acts, axis=0)
        self.norm_factor = np.max(np.abs(all_acts))
        if self.norm_factor == 0:
            self.norm_factor = 1.0

        # 加载数据并预计算 2D 骨架
        self.data_cache = []
        self.has_3d = False
        cam_example = None

        for f_path in file_list:
            raw = np.load(f_path)
            actions = raw['actions'] / self.norm_factor
            images_front = raw['images_front'].astype(np.float32)
            images_side = raw['images_side'].astype(np.float32)

            entry = {
                'images_front': images_front,
                'images_side': images_side,
                'actions': actions,
                'length': len(images_front),
            }

            if 'positions' in raw:
                self.has_3d = True
                entry['positions'] = raw['positions'].astype(np.float32)

            # 预计算 2D 骨架
            entry['skeleton_2d_front'] = batch_extract_skeleton_2d(
                images_front, n_skeleton_points)
            entry['skeleton_2d_side'] = batch_extract_skeleton_2d(
                images_side, n_skeleton_points)

            self.data_cache.append(entry)

            if cam_example is None:
                cam_example = raw

        # 相机参数（假设所有文件一致）
        self.H = int(cam_example['H'])
        self.W = int(cam_example['W'])
        self.focal = float(cam_example['focal'])
        self.action_dim = int(cam_example['actions'].shape[1])

        self.cameras = [
            {
                'eye': tuple(cam_example['camera_eye_front'].tolist()),
                'center': tuple(cam_example['camera_center_front'].tolist()),
                'up': tuple(cam_example['camera_up_front'].tolist()),
                'focal': self.focal,
                'H': self.H, 'W': self.W,
            },
            {
                'eye': tuple(cam_example['camera_eye_side'].tolist()),
                'center': tuple(cam_example['camera_center_side'].tolist()),
                'up': tuple(cam_example['camera_up_side'].tolist()),
                'focal': self.focal,
                'H': self.H, 'W': self.W,
            },
        ]

        # 构建样本索引
        for seq_id, item in enumerate(self.data_cache):
            T = item['length']
            for t in range(T):
                self.samples.append((seq_id, t))

        print(f"MultiViewDataset: {len(self.samples)} samples, "
              f"H={self.H}, W={self.W}, focal={self.focal:.1f}")

    def _get_action_window(self, data, t):
        start = t - self.seq_len + 1
        end = t + 1
        if start >= 0:
            return data['actions'][start:end].copy()
        pad = np.zeros((-start, self.action_dim), dtype=data['actions'].dtype)
        return np.concatenate([pad, data['actions'][0:end]], axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_id, t = self.samples[idx]
        data = self.data_cache[seq_id]

        action_window = torch.from_numpy(self._get_action_window(data, t)).float()
        img_front = torch.from_numpy(data['images_front'][t]).float().reshape(-1)
        img_side = torch.from_numpy(data['images_side'][t]).float().reshape(-1)
        skel_2d_front = torch.from_numpy(data['skeleton_2d_front'][t]).float()
        skel_2d_side = torch.from_numpy(data['skeleton_2d_side'][t]).float()

        result = (action_window, img_front, img_side, skel_2d_front, skel_2d_side)

        if self.return_3d and self.has_3d:
            result += (torch.from_numpy(data['positions'][t]).float(),)

        return result
