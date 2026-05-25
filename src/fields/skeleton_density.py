"""骨架条件密度场：查询点密度取决于到骨架曲线的距离。"""

import torch
import torch.nn as nn
from src.models.layers import PositionalEncoder, MLPDecoder
from src.heads.skeleton_heads import point_to_segment_distance


class SkeletonConditionedDensity(nn.Module):
    """骨架条件密度场。

    Args:
        n_freqs: 距离位置编码频率数。
        d_filter: MLP 隐层维度。
    """

    def __init__(self, n_freqs=6, d_filter=128):
        super().__init__()
        self.dist_encoder = PositionalEncoder(d_input=1, n_freqs=n_freqs, log_space=True)
        self.pos_encoder = PositionalEncoder(d_input=3, n_freqs=4, log_space=True)

        dist_enc_dim = 1 * (1 + 2 * n_freqs)
        pos_enc_dim = 3 * (1 + 2 * 4)
        input_dim = dist_enc_dim + pos_enc_dim

        self.decoder = MLPDecoder(input_dim=input_dim, d_filter=d_filter, output_size=2)
        with torch.no_grad():
            self.decoder.net[-1].bias[1] = 0.0

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
        n_seg = skeleton.shape[1] - 1

        pts = query_points.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * N, n_samples, 3)
        skel_expanded = skeleton.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, -1, 3)
        seg_start = skel_expanded[:, :-1, :]
        seg_end = skel_expanded[:, 1:, :]

        dist = point_to_segment_distance(pts, seg_start, seg_end)
        dist = dist.unsqueeze(-1)

        dist_enc = self.dist_encoder(dist)
        pos_enc = self.pos_encoder(pts)
        latent = torch.cat([dist_enc, pos_enc], dim=-1)

        return self.decoder(latent)
