import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class SoftSequenceDataset(Dataset):
    """通用序列数据集：从 .npz 文件加载动作-图像序列。

    Args:
        data_dir: .npz 文件目录。
        seq_len: 时序窗口长度。
        file_list: 指定文件列表（可选）。
        norm_factor: 动作归一化系数（可选，自动计算）。
        target_size: 可选 resize 目标尺寸。
        return_pairs: 是否返回相邻帧对（用于 smoothness loss）。
    """

    def __init__(self, data_dir, seq_len=10, file_list=None, norm_factor=None,
                 target_size=None, return_pairs=False, return_3d=False, return_depth=False):
        self.seq_len = seq_len
        self.target_size = target_size
        self.return_pairs = return_pairs
        self.return_3d = return_3d
        self.return_depth = return_depth
        self.samples = []

        if file_list is None:
            file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

        # 归一化计算
        if norm_factor is None:
            all_acts = []
            for f in file_list:
                try:
                    d = np.load(f)
                    all_acts.append(d['actions'])
                except Exception as e:
                    print(f"Error loading {f}: {e}")
            if all_acts:
                all_acts = np.concatenate(all_acts, axis=0)
                self.norm_factor = np.max(np.abs(all_acts))
            else:
                self.norm_factor = 1.0
            if self.norm_factor == 0:
                self.norm_factor = 1.0
            print(f"Norm Factor: {self.norm_factor}")
        else:
            self.norm_factor = norm_factor

        self.has_3d = False
        self.has_depth = False
        self.data_cache = []
        for f_path in file_list:
            raw = np.load(f_path)
            actions = raw['actions'] / self.norm_factor

            imgs_raw = raw['images']
            if self.target_size is not None:
                imgs_processed = []
                for img in imgs_raw:
                    if img.max() > 1.0:
                        img = img.astype(np.float32) / 255.0
                    img_resized = cv2.resize(img, (self.target_size, self.target_size),
                                             interpolation=cv2.INTER_AREA)
                    imgs_processed.append(img_resized)
                images = np.stack(imgs_processed, axis=0)[:, np.newaxis, :, :]
            else:
                images = imgs_raw

            entry = {
                'images': images,
                'actions': actions,
                'length': len(images),
            }

            if 'positions' in raw:
                self.has_3d = True
                entry['positions'] = raw['positions'].astype(np.float32)

            if 'depth_maps' in raw:
                self.has_depth = True
                entry['depth_maps'] = raw['depth_maps'].astype(np.float32)

            self.data_cache.append(entry)

        # 构建样本索引
        for seq_id, item in enumerate(self.data_cache):
            T = item['length']
            if self.return_pairs:
                # 需要 t 和 t+1 都在范围内
                for t in range(T - 1):
                    self.samples.append((seq_id, t))
            else:
                for t in range(T):
                    self.samples.append((seq_id, t))

        self.H, self.W = self.data_cache[0]['images'].shape[1:3]
        self.action_dim = self.data_cache[0]['actions'].shape[1]
        self.focal = float(raw.get('focal', 130.0))

        # 读取数据自带的相机参数（可选，让训练器优先使用数据中的参数）
        self.camera_eye = tuple(raw['camera_eye'].tolist()) if 'camera_eye' in raw else None
        self.camera_center = tuple(raw['camera_center'].tolist()) if 'camera_center' in raw else None
        self.camera_up = tuple(raw['camera_up'].tolist()) if 'camera_up' in raw else None

    def _get_action_window(self, data, t):
        """获取时间步 t 的动作窗口，不足部分零填充。"""
        start = t - self.seq_len + 1
        end = t + 1
        if start >= 0:
            return data['actions'][start:end].copy()
        else:
            pad = np.zeros((-start, self.action_dim), dtype=data['actions'].dtype)
            return np.concatenate([pad, data['actions'][0:end]], axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_id, t = self.samples[idx]
        data = self.data_cache[seq_id]
        seq = self._get_action_window(data, t)

        def _get_depth(data, t):
            if self.has_depth and 'depth_maps' in data:
                return torch.from_numpy(data['depth_maps'][t]).float().reshape(-1)
            return None

        # 构建返回值
        if self.return_pairs:
            seq_next = self._get_action_window(data, t + 1)
            target_img = torch.from_numpy(data['images'][t]).float().reshape(-1)
            target_img_next = torch.from_numpy(data['images'][t + 1]).float().reshape(-1)
            result = (
                torch.from_numpy(seq).float(),
                torch.from_numpy(seq_next).float(),
                target_img, target_img_next,
            )
            if self.return_3d and self.has_3d:
                result += (
                    torch.from_numpy(data['positions'][t]).float(),
                    torch.from_numpy(data['positions'][t + 1]).float(),
                )
            if self.return_depth:
                depth_t = _get_depth(data, t)
                depth_t1 = _get_depth(data, t + 1)
                if depth_t is not None and depth_t1 is not None:
                    result += (depth_t, depth_t1)
            return result

        if self.target_size:
            image_seq = data['images'][t - self.seq_len + 1:t + 1]
            result = (torch.from_numpy(image_seq).float(),
                      torch.from_numpy(seq).float())
        else:
            result = (torch.from_numpy(seq).float(),
                      torch.from_numpy(data['images'][t]).float().reshape(-1))

        if self.return_3d and self.has_3d:
            result += (torch.from_numpy(data['positions'][t]).float(),)

        if self.return_depth:
            depth = _get_depth(data, t)
            if depth is not None:
                result += (depth,)

        return result
    
    def get_raw_actions(self, seq_id=0):
        return self.data_cache[seq_id]['actions'] * self.norm_factor

    def get_camera_params(self):
        """返回数据自带的相机参数，无则返回 None（让训练器回退到 config）。"""
        if self.camera_eye is not None:
            return {
                'eye': self.camera_eye,
                'center': self.camera_center,
                'up': self.camera_up,
            }
        return None


def load_soft_data(data_dir):
    """加载并合并目录下所有 .npz 数据文件。

    Args:
        data_dir: 包含 `images` 与 `actions` 字段的 .npz 文件目录。

    Returns:
        images: 图像数组，形状 (N, H, W) 或 (N, H, W, C)。
        actions: 动作数组，形状 (N, action_dim)。
        focal: 焦距（float）。
    """
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    print(f"Found {len(files)} data files. Loading...")

    all_images = []
    all_actions = []
    focal = None

    for f in sorted(files):
        data = np.load(f)
        all_images.append(data['images'])
        all_actions.append(data['actions'])
        if focal is None and 'focal' in data:
            focal = float(data['focal'])

    images = np.concatenate(all_images, axis=0)
    actions = np.concatenate(all_actions, axis=0)

    if focal is None or focal == 1.0:
        height, width = images.shape[1:3]
        focal = 0.5 * width / np.tan(0.5 * 30 * np.pi / 180)
        print(f"Warning: Using calculated focal length: {focal}")

    return images, actions, float(focal)