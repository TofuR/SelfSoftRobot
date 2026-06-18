"""实物图像 → 仿真 schema .npz 处理管线。

阶段（见 capture_to_npz.py 串联）:
  io_video → segmentation.segment_views → skeleton_2d.extract_skeleton_2d
  → triangulation.triangulate_skeletons → assemble_npz.build_real_npz

复用现有 src/utils/skeleton_2d（2D 骨架）与 src/calibration（标定）。
"""

from src.data.real import io_video, segmentation, triangulation, assemble_npz

__all__ = ["io_video", "segmentation", "triangulation", "assemble_npz"]
