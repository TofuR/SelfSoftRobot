"""dataset_factory.py — 数据集创建与 batch collation 工厂。

根据 PhaseSpec.dataset_type 自动创建对应的数据集实例，
并将各数据集不同的 tuple 返回格式统一为 dict batch。

支持的 dataset_type:
  "sequence"        → SoftSequenceDataset（2D 图像 + 动作）
  "multiview_depth" → MultiViewDepthDataset（多视角 + 深度）
  "sdf"             → SDFDataset（3D SDF 监督）
  "skeleton_sdf"    → SkeletonSDFDataset（骨架 + SDF）
  "pointcloud"      → PointCloudDataset（点云，用于 Flow Matching）

统一 dict batch 格式:
  {
      "action_window":       Tensor (B, K, D),
      "action_window_next":  Tensor | None,        # pairs 模式
      "images":              list[Tensor] | Tensor, # 单视角=Tensor, 多视角=list
      "depths":              list[Tensor] | None,
      "gt_positions":        Tensor | None,         # (B, N, 3) 3D 骨架 GT
      "coords":              Tensor | None,         # (B, M, 3) SDF 查询点
      "gt_sdf":              Tensor | None,         # (B, M)
      "gt_normals":          Tensor | None,         # (B, M, 3)
      "gt_pointcloud":       Tensor | None,         # (B, N, 3) GT 点云（pointcloud 模式）
  }
"""

import torch
from torch.utils.data import Dataset

from src.training.spec import PhaseSpec


def create_dataset(dataset_type: str, data_dir: str, config: dict,
                   phase_spec: PhaseSpec) -> Dataset:
    """根据 dataset_type 创建数据集实例。

    Args:
        dataset_type: "sequence" | "multiview_depth" | "sdf" | "skeleton_sdf"
        data_dir: 数据目录路径
        config: 训练配置 dict
        phase_spec: 当前阶段配置（从中读取 dataset_kwargs 等）

    Returns:
        Dataset 实例
    """
    temp_cfg = config.get("temporal", {})
    kwargs = dict(phase_spec.dataset_kwargs)
    seq_len = temp_cfg["window_size"]

    if dataset_type == "sequence":
        from src.data.dataset import SoftSequenceDataset
        return SoftSequenceDataset(
            data_dir,
            seq_len=seq_len,
            return_pairs=kwargs.get("return_pairs", False),
            return_3d=kwargs.get("return_3d", False),
            return_depth=kwargs.get("return_depth", False),
        )

    elif dataset_type == "multiview_depth":
        from src.data.dataset_multiview_depth import MultiViewDepthDataset
        active = phase_spec.active_losses
        return MultiViewDepthDataset(
            data_dir,
            seq_len=seq_len,
            return_depth="depth" in active,
            return_pairs="smooth" in active,
            return_3d=kwargs.get("return_3d", False),
        )

    elif dataset_type == "sdf":
        from src.data.dataset_sdf import SDFDataset
        sdf_cfg = config["sdf"]
        return SDFDataset(
            data_dir,
            seq_len=seq_len,
            n_surface=sdf_cfg["n_surface"],
            n_near_surface=sdf_cfg["n_near_surface"],
            n_off_surface=sdf_cfg["n_off_surface"],
        )

    elif dataset_type == "skeleton_sdf":
        from src.data.dataset_skeleton_sdf import SkeletonSDFDataset
        sdf_cfg = config["sdf"]
        return SkeletonSDFDataset(
            data_dir,
            seq_len=seq_len,
            n_surface=sdf_cfg["n_surface"],
            n_near_surface=sdf_cfg["n_near_surface"],
            n_off_surface=sdf_cfg["n_off_surface"],
        )

    elif dataset_type == "pointcloud":
        from src.data.dataset_pointcloud import PointCloudDataset
        pc_cfg = config.get("pointcloud", {})
        return PointCloudDataset(
            data_dir,
            seq_len=seq_len,
            n_surface_points=pc_cfg.get("n_surface_points", 1000),
        )

    elif dataset_type == "spatial_sequence":
        from src.data.dataset_spatial import SpatialSequenceDataset
        return SpatialSequenceDataset(
            data_dir,
            seq_len=seq_len,
            pairs="smooth" in phase_spec.active_losses,
            action_channels=config.get("action_view", {}).get(
                "model_action_channels"),
        )

    elif dataset_type == "state_transition":
        # 闭环状态转移：额外返回前一步骨架（单帧模式）或连续序列（episode 模式）。
        # 继承 SpatialSequenceDataset，复用通用 collate（spatial_collate_fn）。
        from src.data.dataset_spatial import StateTransitionDataset
        return StateTransitionDataset(
            data_dir,
            seq_len=seq_len,
            pairs="smooth" in phase_spec.active_losses,
            episode_mode=getattr(phase_spec, "use_episode_mode", False),
            episode_len=getattr(phase_spec, "episode_len", 20),
            action_channels=config.get("action_view", {}).get(
                "model_action_channels"),
        )

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


def get_collate_fn(dataset_type: str, dataset: Dataset):
    """返回将 tuple batch 转为统一 dict batch 的 collate 函数。

    Args:
        dataset_type: 同 create_dataset
        dataset: 已创建的数据集实例（某些 collate 需要元信息）

    Returns:
        callable: list[tuple] → dict[str, Tensor]
    """
    if dataset_type == "sequence":
        return _collate_sequence(dataset)
    elif dataset_type == "multiview_depth":
        return _collate_multiview_depth(dataset)
    elif dataset_type == "sdf":
        return _collate_sdf
    elif dataset_type == "skeleton_sdf":
        return _collate_skeleton_sdf
    elif dataset_type == "pointcloud":
        return _collate_pointcloud
    elif dataset_type == "spatial_sequence":
        from src.data.dataset_spatial import spatial_collate_fn
        return spatial_collate_fn
    elif dataset_type == "state_transition":
        # 闭环状态转移：dict 样本新增 prev_gt_skeleton/prev_prev_gt_skeleton 键，
        # spatial_collate_fn 是通用 dict 合并，自动 stack 新键，无需单独 collate。
        from src.data.dataset_spatial import spatial_collate_fn
        return spatial_collate_fn
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


# ── Collate 函数 ──────────────────────────────────────────────────────


def _collate_sequence(dataset):
    """SoftSequenceDataset 的 collate 函数。

    返回格式取决于数据集的 return_pairs / return_3d 标志。
    """
    return_pairs = dataset.return_pairs
    return_3d = getattr(dataset, 'return_3d', False)

    def collate(batch):
        action_windows = torch.stack([b[0] for b in batch])

        result = {"action_window": action_windows}

        if return_pairs:
            # (seq_t, seq_t1, img_t, img_t1, [pos_t, pos_t1])
            result["action_window_next"] = torch.stack([b[1] for b in batch])
            result["images"] = torch.stack([b[2] for b in batch])
            idx = 3
        else:
            # (action_window, image, [positions], [depth])
            result["images"] = torch.stack([b[1] for b in batch])
            idx = 2

        # 3D positions
        if return_3d and len(batch[0]) > idx:
            result["gt_positions"] = torch.stack([b[idx] for b in batch])
            idx += 1
        else:
            result["gt_positions"] = None

        # depths
        if getattr(dataset, 'return_depth', False) and len(batch[0]) > idx:
            result["depths"] = torch.stack([b[idx] for b in batch])
            idx += 1
        else:
            result["depths"] = None
        result["coords"] = None
        result["gt_sdf"] = None
        result["gt_normals"] = None

        return result

    return collate


def _collate_multiview_depth(dataset):
    """MultiViewDepthDataset 的 collate 函数。"""
    n_views = dataset.n_views
    has_3d = getattr(dataset, 'return_3d', False) and getattr(dataset, 'has_3d', False)
    has_pairs = dataset.return_pairs

    def collate(batch):
        action_windows = torch.stack([b[0] for b in batch])

        result = {"action_window": action_windows}

        # b[1] = images_list (list of V tensors)
        images_per_view = []
        for v in range(n_views):
            images_per_view.append(torch.stack([b[1][v] for b in batch]))
        result["images"] = images_per_view

        # b[2] = depths_list or None
        if batch[0][2] is not None:
            depths_per_view = []
            for v in range(n_views):
                depths_per_view.append(torch.stack([b[2][v] for b in batch]))
            result["depths"] = depths_per_view
        else:
            result["depths"] = None

        # b[3] = positions or None
        if has_3d and len(batch[0]) > 3 and batch[0][3] is not None:
            result["gt_positions"] = torch.stack([b[3] for b in batch])
        else:
            result["gt_positions"] = None

        # b[4:] = pairs (action_window_next, images_next_list)
        if has_pairs and len(batch[0]) > 4:
            result["action_window_next"] = torch.stack([b[4] for b in batch])
        else:
            result["action_window_next"] = None

        result["coords"] = None
        result["gt_sdf"] = None
        result["gt_normals"] = None

        return result

    return collate


def _collate_sdf(batch):
    """SDFDataset 的 collate 函数。 (action_window, coords, sdf, normals)"""
    return {
        "action_window": torch.stack([b[0] for b in batch]),
        "images": None,
        "depths": None,
        "action_window_next": None,
        "gt_positions": None,
        "coords": torch.stack([b[1] for b in batch]),
        "gt_sdf": torch.stack([b[2] for b in batch]),
        "gt_normals": torch.stack([b[3] for b in batch]),
    }


def _collate_skeleton_sdf(batch):
    """SkeletonSDFDataset 的 collate 函数。 (action, coords, sdf, normals, positions)"""
    return {
        "action_window": torch.stack([b[0] for b in batch]),
        "images": None,
        "depths": None,
        "action_window_next": None,
        "coords": torch.stack([b[1] for b in batch]),
        "gt_sdf": torch.stack([b[2] for b in batch]),
        "gt_normals": torch.stack([b[3] for b in batch]),
        "gt_positions": torch.stack([b[4] for b in batch]),
    }


def _collate_pointcloud(batch):
    """PointCloudDataset 的 collate 函数。 (action_window, gt_pointcloud)"""
    return {
        "action_window": torch.stack([b[0] for b in batch]),
        "gt_pointcloud": torch.stack([b[1] for b in batch]),
        "images": None,
        "depths": None,
        "action_window_next": None,
        "gt_positions": None,
        "coords": None,
        "gt_sdf": None,
        "gt_normals": None,
    }
