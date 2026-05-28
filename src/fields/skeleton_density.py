"""骨架条件密度场：查询点密度取决于骨架局部柱坐标 (dist, t_axial)。"""

import torch
import torch.nn as nn
from src.models.layers import PositionalEncoder, MLPDecoder
from src.heads.skeleton_heads import point_to_skeleton_coords


class SkeletonConditionedDensity(nn.Module):
    """骨架条件密度场。

    将查询点的 3D 绝对坐标替换为骨架局部柱坐标：
      dist    — 到最近骨架线段的径向距离
      t_axial — 沿骨架的归一化轴向参数 in [0, 1]
      theta   — 环向角度（当前不使用，留供未来扩展）

    Args:
        n_freqs: 距离位置编码频率数。
        d_filter: MLP 隐层维度。
        axial_n_freqs: 轴向参数位置编码频率数。
    """

    def __init__(self, n_freqs=6, d_filter=128, axial_n_freqs=8):
        super().__init__()
        self.dist_encoder = PositionalEncoder(d_input=1, n_freqs=n_freqs, log_space=True)
        self.axial_encoder = PositionalEncoder(d_input=1, n_freqs=axial_n_freqs, log_space=True)

        dist_enc_dim = 1 * (1 + 2 * n_freqs)
        axial_enc_dim = 1 * (1 + 2 * axial_n_freqs)
        input_dim = dist_enc_dim + axial_enc_dim

        self.decoder = MLPDecoder(input_dim=input_dim, d_filter=d_filter, output_size=2)
        with torch.no_grad():
            self.decoder.net[-1].bias[1] = 0.0

        self._last_theta = None

    def forward(self, query_points, skeleton):
        """查询骨架条件密度。

        Args:
            query_points: (N, n_samples, 3)
            skeleton: (B, n_fine, 3)

        Returns:
            (B*N, n_samples, 2) [visibility, density]
        """
        N, n_samples, _ = query_points.shape
        B = skeleton.shape[0]

        pts = query_points.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * N, n_samples, 3)
        skel_expanded = skeleton.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, -1, 3)
        seg_start = skel_expanded[:, :-1, :]
        seg_end = skel_expanded[:, 1:, :]

        dist, t_axial, theta = point_to_skeleton_coords(pts, seg_start, seg_end)

        # 保存 theta 供未来检查（不参与网络计算）
        self._last_theta = theta.detach()

        dist_enc = self.dist_encoder(dist.unsqueeze(-1))
        axial_enc = self.axial_encoder(t_axial.unsqueeze(-1))
        latent = torch.cat([dist_enc, axial_enc], dim=-1)

        return self.decoder(latent)
