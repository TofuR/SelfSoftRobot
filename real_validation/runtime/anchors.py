"""锚点构建:从离线 NPZ 或实时相机帧建立模型状态锚点(合并原 live/offline/anchor_utils)。

本模块是 re-export 门面 —— live_anchor / offline_anchor / anchor_utils 三个源文件
保留在 runtime/ 内,这里只聚合公共函数,供 GUI 与 scripts/run_avoidance 使用:

    from real_validation.runtime.anchors import anchor_from_npz
    from real_validation.runtime.anchors import anchor_from_camera_frame
"""
from .anchor_utils import float_rows, model_normalization, normalize_rows  # noqa: F401
from .live_anchor import anchor_from_camera_frame  # noqa: F401
from .offline_anchor import anchor_from_npz  # noqa: F401

__all__ = [
    "anchor_from_camera_frame",
    "anchor_from_npz",
    "float_rows",
    "model_normalization",
    "normalize_rows",
]
