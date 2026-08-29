"""LocalFrameModeBank — 局部标架模态位移场（平面合同）。

作用: 把"模态位移"表达在参考骨架的局部标架（切向 T、面内法向 N）上，
而非全局固定坐标——修复 v1"全局常模态在大曲率下方向错误"的问题
（设计文档 §2.2: 模态随臂的当前构形旋转）:

    mode_i(node) = α_{i,node} · N(node) + β_{i,node} · T(node)

  - T: 相邻节点差分（中心差分，端点单侧），单位化。
  - N: 面内法向 (−T_y, T_x, 0) —— 平面合同（branch: planar-constrained），
    实物骨架 (col,row,0) / robot_planar_mm_v1 均 z=0，模态 z 分量为 0。

α、β 可学习（每模态每节点各一标量），符号自由度由 α/β 吸收（配合
PlayBank 的非负权重、MaxwellBank 的带符号权重）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalFrameModeBank(nn.Module):
    """可学习局部标架模态组。

    Args:
        n_modes: 模态个数（play 的 J 或 Maxwell 的 M）。
        n_nodes: 骨架节点数 N。
        init_std: α/β 初始化标准差（小随机，打破对称）。
    """

    def __init__(self, n_modes: int, n_nodes: int, init_std: float = 0.02):
        super().__init__()
        assert n_modes >= 1 and n_nodes >= 2
        self.n_modes = n_modes
        self.n_nodes = n_nodes
        self.alpha = nn.Parameter(
            init_std * torch.randn(n_modes, n_nodes))   # 法向分量
        self.beta = nn.Parameter(
            init_std * torch.randn(n_modes, n_nodes))   # 切向分量

    def _local_frame(self, skeleton: torch.Tensor):
        """参考骨架 (B, N, 3) → (tangent, normal)，各 (B, N, 3)。"""
        seg = skeleton[:, 1:] - skeleton[:, :-1]          # (B, N-1, 3)
        tangent = torch.zeros_like(skeleton)
        tangent[:, 1:-1] = 0.5 * (seg[:, 1:] + seg[:, :-1])
        tangent[:, 0] = seg[:, 0]
        tangent[:, -1] = seg[:, -1]
        tangent = F.normalize(tangent, dim=-1, eps=1e-6)
        # 平面合同: 面内法向 = 切向绕 z 轴旋转 90°
        normal = torch.stack(
            [-tangent[..., 1], tangent[..., 0],
             torch.zeros_like(tangent[..., 0])], dim=-1)
        return tangent, normal

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        """在参考骨架的局部标架下生成模态位移场。

        Args:
            skeleton: (B, N, 3) 参考骨架（通常为当前静态预测 s̃_t）。
        Returns:
            (n_modes, B, N, 3) 模态位移场，z 分量为 0。
        """
        tangent, normal = self._local_frame(skeleton)     # (B, N, 3) 各
        M = self.n_modes
        alpha = self.alpha.view(M, 1, self.n_nodes, 1)
        beta = self.beta.view(M, 1, self.n_nodes, 1)
        modes = alpha * normal.unsqueeze(0) + beta * tangent.unsqueeze(0)
        return modes                                        # (M, B, N, 3)
