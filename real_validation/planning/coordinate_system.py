"""集中管理平面齐次坐标转换；3D 标定后续使用独立实现。"""

from __future__ import annotations

import numpy as np


class PlanarTransform:
    def __init__(self, matrix, source_frame: str, target_frame: str):
        self.matrix = np.asarray(matrix, dtype=np.float64)
        if self.matrix.shape != (3, 3) or not np.isfinite(self.matrix).all():
            raise ValueError("平面齐次变换必须是有限的 3×3 矩阵")
        if abs(np.linalg.det(self.matrix)) < 1e-12:
            raise ValueError("平面齐次变换不可逆")
        self.source_frame = source_frame
        self.target_frame = target_frame

    def apply(self, points):
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("points 必须为 N×2")
        homogeneous = np.column_stack((values, np.ones(len(values))))
        mapped = homogeneous @ self.matrix.T
        if np.any(np.abs(mapped[:, 2]) < 1e-12):
            raise ValueError("点映射到无穷远")
        return mapped[:, :2] / mapped[:, 2:3]

    def inverse(self) -> "PlanarTransform":
        return PlanarTransform(np.linalg.inv(self.matrix), self.target_frame,
                               self.source_frame)

    def roundtrip_error(self, points) -> float:
        values = np.asarray(points, dtype=np.float64)
        restored = self.inverse().apply(self.apply(values))
        return float(np.linalg.norm(restored - values, axis=1).max(initial=0.0))
