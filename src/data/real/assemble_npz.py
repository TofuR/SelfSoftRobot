"""assemble_npz.py — 把实物处理结果组装成仿真 schema 的 .npz。

输出字段对齐 scripts/data_collection/collect.py 多视角格式，使
SoftSequenceDataset / UnifiedTrainer 直接消费（零侵入复用）：
  images (T,V,H,W), actions (T,A), positions (T,3,J), radii (T,J),
  camera_params (V,10), focal, H, W, dt, view_names (+ 可选 masks /
  commanded_actions / ndi_tip_anchor)。

positions 由三角化 3D 骨架 (T,J,3) 转置成 (T,3,J)，与仿真 positions 同 layout。
"""

import numpy as np


def build_real_npz(images, masks, skeletons_3d, actions, camera_params,
                   dt, view_names, focal=None, radii=None,
                   commanded_actions=None, ndi_tip_anchor=None,
                   positions_2d=None, visibility=None,
                   reprojection_error=None, position_confidence=None,
                   source_mask=None, frame_times=None,
                   projection_matrices=None):
    """组装仿真 schema .npz 字典。

    Args:
        images: (N,V,H,W) 或 (N,V,H,W,3) uint8/float。
        masks: (N,V,H,W) 二值分割掩码。
        skeletons_3d: (T,J,3) 三角化世界系骨架（替代 sim positions）。
        actions: (N,A) 实测气压（单腔道 A=2 存 [p,0]）。
        camera_params: (V,10) [eye,center,up,focal]。
        dt: 帧间隔（秒）。
        view_names: list[V] str。
        focal: 显式焦距；None 则取 camera_params[0,9]。
        radii: (T,J) 或 None（None→常数 0.015 m）。
        commanded_actions: (N,A) 可选（迟滞分析）。
        ndi_tip_anchor: (N,3) 可选（NDI 末端独立验证）。

    Returns:
        dict（供 np.savez_compressed）。
    """
    images = np.asarray(images)
    N, V = images.shape[:2]
    H, W = images.shape[2], images.shape[3]
    if images.ndim == 5:                       # 取灰度单通道，对齐 sim images (N,V,H,W)
        images = images[..., 0]

    cp = np.asarray(camera_params, np.float32)
    if focal is None:
        focal = float(cp[0, 9])

    sk3 = np.asarray(skeletons_3d, np.float32)         # (T,J,3)
    positions = np.transpose(sk3, (0, 2, 1))           # (T,3,J) 同 sim layout
    if radii is None:
        radii = np.full((N, sk3.shape[1]), 0.015, np.float32)

    data = {
        "images": images.astype(np.float32),
        "actions": np.asarray(actions, np.float32),
        "dt": float(dt),
        "focal": float(focal),
        "H": int(H), "W": int(W),
        "camera_params": cp,
        "view_names": np.asarray(view_names),
        "positions": positions,
        "radii": np.asarray(radii, np.float32),
        "masks": np.asarray(masks, np.float32),
    }
    if commanded_actions is not None:
        data["commanded_actions"] = np.asarray(commanded_actions, np.float32)
    if ndi_tip_anchor is not None:
        data["ndi_tip_anchor"] = np.asarray(ndi_tip_anchor, np.float32)
    optional = {
        "positions_2d": (positions_2d, np.float32),
        "visibility": (visibility, np.bool_),
        "reprojection_error": (reprojection_error, np.float32),
        "position_confidence": (position_confidence, np.float32),
        "source_mask": (source_mask, np.uint8),
        "frame_times": (frame_times, np.float64),
        "projection_matrices": (projection_matrices, np.float32),
    }
    for key, (value, dtype) in optional.items():
        if value is not None:
            data[key] = np.asarray(value, dtype=dtype)
    return data


def save_real_npz(path, **kwargs):
    """组装并保存。kwargs 见 build_real_npz。"""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **build_real_npz(**kwargs))
    return path
