"""dataset_multiview_depth.py — 多视角 + 深度数据集。

支持新旧 npz 格式:
  新格式（优先）:
    images:          (N, V, H, W)
    depths:          (N, V, H, W)  可选
    camera_params:   (V, 10)
    view_names:      ['front', 'side', ...]
  旧格式（自动回退）:
    images_front, images_side, ...
    depth_maps_front, depth_maps_side, ...
    camera_eye_front, ...

返回每个样本:
    action_window, images_list[V], depths_list[V]（可选）, positions（可选）
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.camera_system import MultiCameraSystem


class MultiViewDepthDataset(Dataset):
    """多视角 + 深度序列数据集。

    Args:
        data_dir: .npz 文件目录。
        seq_len: 时序窗口长度。
        return_depth: 是否返回深度图。
        return_3d: 是否返回 3D positions。
        return_pairs: 是否返回相邻帧对。
    """

    def __init__(self, data_dir, seq_len=20, return_depth=False,
                 return_3d=False, return_pairs=False):
        self.seq_len = seq_len
        self.return_depth = return_depth
        self.return_3d = return_3d
        self.return_pairs = return_pairs
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

        # 加载并缓存数据
        self.data_cache = []
        self.has_3d = False
        self.has_depth = False
        cam_system = None

        for f_path in file_list:
            raw = np.load(f_path, allow_pickle=True)
            actions = raw['actions'] / self.norm_factor

            # 读取多视角 images
            if 'images' in raw and raw['images'].ndim == 4:
                # 新格式: (N, V, H, W)
                images = raw['images'].astype(np.float32)
            else:
                # 旧格式: 从 _front/_side 拼接
                views = []
                for suffix in ['front', 'side', 'top', 'back']:
                    key = f'images_{suffix}'
                    if key in raw:
                        views.append(raw[key].astype(np.float32))
                if not views:
                    # 单视角: (N, H, W) → (N, 1, H, W)
                    images = raw['images'].astype(np.float32)[:, None, :, :]
                else:
                    images = np.stack(views, axis=1)

            entry = {
                'images': images,
                'actions': actions,
                'length': len(actions),
            }

            # 深度图
            if 'depths' in raw and raw['depths'].ndim == 4:
                entry['depths'] = raw['depths'].astype(np.float32)
                self.has_depth = True
            elif return_depth:
                # 旧格式深度
                dep_views = []
                for suffix in ['front', 'side', 'top', 'back']:
                    key = f'depth_maps_{suffix}'
                    if key in raw:
                        dep_views.append(raw[key].astype(np.float32))
                if dep_views:
                    entry['depths'] = np.stack(dep_views, axis=1)
                    self.has_depth = True

            # 3D positions
            if 'positions' in raw:
                self.has_3d = True
                entry['positions'] = raw['positions'].astype(np.float32)

            self.data_cache.append(entry)

            # 相机系统（从第一个文件构建）
            if cam_system is None:
                cam_system = MultiCameraSystem.from_npz(raw)

        self.cam_system = cam_system
        self.n_views = cam_system.n_views
        self.H = cam_system.cameras[0]['H']
        self.W = cam_system.cameras[0]['W']
        self.action_dim = actions.shape[1]

        # 构建样本索引
        for seq_id, item in enumerate(self.data_cache):
            T = item['length']
            end = T - 1 if return_pairs else T
            for t in range(end):
                self.samples.append((seq_id, t))

        print(f"MultiViewDepthDataset: {len(self.samples)} samples, "
              f"{self.n_views} views, H={self.H}, W={self.W}, "
              f"depth={self.has_depth}, 3d={self.has_3d}")

    def _get_action_window(self, data, t):
        start = t - self.seq_len + 1
        if start >= 0:
            return data['actions'][start:t + 1].copy()
        pad = np.zeros((-start, self.action_dim), dtype=data['actions'].dtype)
        return np.concatenate([pad, data['actions'][0:t + 1]], axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_id, t = self.samples[idx]
        data = self.data_cache[seq_id]

        action_window = torch.from_numpy(self._get_action_window(data, t)).float()

        # 多视角 images: list of V 个 (H*W,) tensor
        images = data['images'][t]  # (V, H, W)
        images_list = [torch.from_numpy(images[v]).float().reshape(-1)
                       for v in range(self.n_views)]

        result = [action_window, images_list]

        # 深度图
        if self.return_depth and self.has_depth and 'depths' in data:
            depths = data['depths'][t]  # (V, H, W)
            depths_list = [torch.from_numpy(depths[v]).float().reshape(-1)
                           for v in range(self.n_views)]
            result.append(depths_list)
        else:
            result.append(None)

        # 3D positions
        if self.return_3d and self.has_3d:
            result.append(torch.from_numpy(data['positions'][t]).float())
        else:
            result.append(None)

        # 相邻帧对
        if self.return_pairs:
            t1 = min(t + 1, data['length'] - 1)
            action_window_next = torch.from_numpy(
                self._get_action_window(data, t1)).float()
            images_next = data['images'][t1]
            images_next_list = [torch.from_numpy(images_next[v]).float().reshape(-1)
                                for v in range(self.n_views)]
            result.extend([action_window_next, images_next_list])

        return tuple(result)
