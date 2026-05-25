"""静态 canonical 场：空间坐标 → [visibility, density]。

不依赖任何动作信息，纯粹表示机器人在零动作下的 3D 形态。
"""

import torch
import torch.nn as nn
from src.models.layers import PositionalEncoder, MLPDecoder


class CanonicalField(nn.Module):
    """静态 canonical 场。

    Args:
        d_filter: MLP 隐层维度。
        n_freqs: 位置编码频率数。
    """

    def __init__(self, d_filter=128, n_freqs=10):
        super().__init__()
        self.pos_encoder = PositionalEncoder(d_input=3, n_freqs=n_freqs, log_space=True)
        pos_enc_dim = 3 * (1 + 2 * n_freqs)
        self.decoder = MLPDecoder(input_dim=pos_enc_dim, d_filter=d_filter, output_size=2)

        # density bias = 0.0，配合 softplus 激活
        with torch.no_grad():
            self.decoder.net[-1].bias[1] = 0.0

    def forward(self, points):
        """查询 canonical 场。

        Args:
            points: (N, n_samples, 3)

        Returns:
            output: (N, n_samples, 2) [visibility, density]
        """
        return self.decoder(self.pos_encoder(points))
