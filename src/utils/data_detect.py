"""数据集自动探测工具。

从 npz 文件中自动探测 action_dim、节点数、视角数等参数，
避免在每个训练脚本中重复定义。"""

import glob
import os

import numpy as np


def detect_action_dim(data_dir):
    """从数据目录的第一个 npz 文件探测 action 维度。

    Args:
        data_dir: npz 文件所在目录。

    Returns:
        int: action 维度（如 2）。

    Raises:
        FileNotFoundError: 目录中没有 npz 文件。
        ValueError: npz 中没有 'actions' 字段。
    """
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No data in {data_dir}")
    sample = np.load(npz_files[0])
    if 'actions' in sample:
        return sample['actions'].shape[-1]
    raise ValueError(f"No 'actions' field in {npz_files[0]}")


def detect_n_nodes(data_dir):
    """从数据目录的第一个 npz 文件探测中心线节点数。

    Args:
        data_dir: npz 文件所在目录。

    Returns:
        int: 节点数（如 31）。

    Raises:
        FileNotFoundError: 目录中没有 npz 文件。
        ValueError: npz 中没有 'positions' 字段。
    """
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No data in {data_dir}")
    sample = np.load(npz_files[0])
    if 'positions' in sample:
        # positions shape: (T, 3, n_nodes)
        return sample['positions'].shape[-1]
    raise ValueError(f"No 'positions' field in {npz_files[0]}")


def detect_n_views(data_dir):
    """从数据目录的第一个 npz 文件探测视角数。

    Args:
        data_dir: npz 文件所在目录。

    Returns:
        int: 视角数（如 2 或 6）。
    """
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No data in {data_dir}")
    sample = np.load(npz_files[0])
    images = sample.get('images')
    if images is not None and images.ndim == 4:
        # images shape: (T, n_views, H, W)
        return images.shape[1]
    return 2
