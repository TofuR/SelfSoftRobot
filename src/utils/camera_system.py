"""camera_system.py — 多相机系统管理。

统一管理多个相机的参数、射线生成、投影/反投影。
用于多视角训练和深度监督。
"""

import numpy as np
import torch
from src.utils.camera import get_rays


class MultiCameraSystem:
    """管理多个相机的参数和射线生成。

    每个相机配置为 dict，包含:
        eye: (3,) 相机位置
        center: (3,) 注视点
        up: (3,) 上方向
        focal: float 焦距
        H, W: int 图像尺寸

    相机索引与数据中 images[:, view_idx, ...] 对齐。
    """

    def __init__(self, camera_configs):
        self.cameras = camera_configs
        self.n_views = len(camera_configs)
        self._validate()

    def _validate(self):
        required_keys = {'eye', 'center', 'up', 'focal', 'H', 'W'}
        for i, cam in enumerate(self.cameras):
            missing = required_keys - set(cam.keys())
            if missing:
                raise ValueError(f"Camera {i} missing keys: {missing}")

    @classmethod
    def from_npz(cls, npz_data):
        """从 npz 数据自动构建（兼容旧格式和新的数组格式）。

        旧格式: camera_eye_front, camera_eye_side, ...（后缀命名）
        新格式: camera_params (V, param_dim) 数组格式
        """
        cameras = []

        # 优先尝试新格式: camera_params 数组
        if 'camera_params' in npz_data:
            params = npz_data['camera_params']
            for i in range(len(params)):
                p = params[i]
                cameras.append({
                    'eye': tuple(p[0:3].tolist()),
                    'center': tuple(p[3:6].tolist()),
                    'up': tuple(p[6:9].tolist()),
                    'focal': float(p[9]),
                    'H': int(npz_data.get('H', 100)),
                    'W': int(npz_data.get('W', 100)),
                })
            return cls(cameras)

        # 回退: 从 view_names 读取
        view_names = npz_data.get('view_names', None)
        if view_names is not None:
            for name in view_names:
                key = name if isinstance(name, str) else str(name)
                cameras.append({
                    'eye': tuple(npz_data[f'camera_eye_{key}'].tolist()),
                    'center': tuple(npz_data[f'camera_center_{key}'].tolist()),
                    'up': tuple(npz_data[f'camera_up_{key}'].tolist()),
                    'focal': float(npz_data['focal']),
                    'H': int(npz_data['H']),
                    'W': int(npz_data['W']),
                })
            return cls(cameras)

        # 回退: 旧格式，探测 _front, _side, ... 后缀
        suffixes = []
        for suffix in ['front', 'side', 'top', 'back']:
            if f'camera_eye_{suffix}' in npz_data:
                suffixes.append(suffix)

        if not suffixes:
            raise ValueError("Cannot find camera params in npz data")

        for suffix in suffixes:
            cameras.append({
                'eye': tuple(npz_data[f'camera_eye_{suffix}'].tolist()),
                'center': tuple(npz_data[f'camera_center_{suffix}'].tolist()),
                'up': tuple(npz_data[f'camera_up_{suffix}'].tolist()),
                'focal': float(npz_data['focal']),
                'H': int(npz_data['H']),
                'W': int(npz_data['W']),
            })
        return cls(cameras)

    def get_rays(self, view_idx, device='cpu'):
        """获取指定视角的所有像素射线。

        Returns:
            rays_o: (H*W, 3)
            rays_d: (H*W, 3)
        """
        cam = self.cameras[view_idx]
        return get_rays(cam['H'], cam['W'], cam['focal'],
                        cam['eye'], cam['center'], cam['up'], device=device)

    def get_all_rays(self, device='cpu'):
        """获取所有视角的射线。

        Returns:
            list of (rays_o, rays_d) tuples, 长度 n_views
        """
        return [self.get_rays(i, device) for i in range(self.n_views)]

    def project(self, points_3d, view_idx, device='cpu'):
        """将 3D 点投影到指定相机的 2D 像素坐标。

        Args:
            points_3d: (N, 3) 3D 点
            view_idx: 相机索引

        Returns:
            pixels: (N, 2) 像素坐标 (x, y)
            depths: (N,) 各点在该相机下的深度
        """
        cam = self.cameras[view_idx]
        eye = torch.tensor(cam['eye'], dtype=torch.float32, device=device)
        center = torch.tensor(cam['center'], dtype=torch.float32, device=device)
        up = torch.tensor(cam['up'], dtype=torch.float32, device=device)

        if isinstance(points_3d, np.ndarray):
            points_3d = torch.from_numpy(points_3d).float().to(device)

        view_dir = center - eye
        view_dir = view_dir / torch.norm(view_dir)
        right = torch.linalg.cross(view_dir, up)
        right = right / torch.norm(right)
        true_up = torch.linalg.cross(right, view_dir)
        true_up = true_up / torch.norm(true_up)

        pts_rel = points_3d - eye
        depths = (pts_rel * view_dir).sum(dim=-1)
        x_cam = (pts_rel * right).sum(dim=-1)
        y_cam = (pts_rel * true_up).sum(dim=-1)

        focal = float(cam['focal'])
        H, W = cam['H'], cam['W']
        px = x_cam * focal / (depths + 1e-8) + W * 0.5
        py = -y_cam * focal / (depths + 1e-8) + H * 0.5

        return torch.stack([px, py], dim=-1), depths

    def unproject(self, pixels, depths, view_idx, device='cpu'):
        """从指定相机的 2D 像素 + 深度反投影到 3D。

        Args:
            pixels: (N, 2) 像素坐标
            depths: (N,) 深度值
            view_idx: 相机索引

        Returns:
            points_3d: (N, 3) 3D 点
        """
        cam = self.cameras[view_idx]
        rays_o, rays_d = self.get_rays(view_idx, device=device)

        if isinstance(pixels, np.ndarray):
            pixels = torch.from_numpy(pixels).long()
        if isinstance(depths, np.ndarray):
            depths = torch.from_numpy(depths).float().to(device)

        H, W = cam['H'], cam['W']
        pixel_idx = (pixels[:, 1] * W + pixels[:, 0]).clamp(0, H * W - 1)

        origins = rays_o[pixel_idx]
        directions = rays_d[pixel_idx]
        return origins + directions * depths.unsqueeze(-1)

    def get_camera_params_array(self):
        """将相机参数序列化为 (V, 10) 数组，用于保存到 npz。

        每行: [eye(3), center(3), up(3), focal(1)]
        """
        params = []
        for cam in self.cameras:
            params.append([*cam['eye'], *cam['center'], *cam['up'], cam['focal']])
        return np.array(params, dtype=np.float32)

    def summary(self):
        """打印相机配置摘要。"""
        print(f"MultiCameraSystem: {self.n_views} views")
        for i, cam in enumerate(self.cameras):
            print(f"  [{i}] eye={cam['eye']}, focal={cam['focal']:.1f}, "
                  f"size={cam['H']}x{cam['W']}")
