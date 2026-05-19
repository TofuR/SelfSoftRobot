"""mixins.py — 可复用的模型方法混入。

TemporalMixin: 时序编码相关方法（需要 self.temporal）
SkeletonMixin: 骨架预测与损失相关方法（需要 self.temporal, self.skeleton_head）
"""

import torch

from .skeleton_heads import downsample_skeleton


class TemporalMixin:
    """时序编码混入。使用方必须有 self.temporal (MultiScaleEMA 实例)。"""

    def encode(self, action_window):
        return self.temporal(action_window)

    def compute_smoothness(self, action_windows_t, action_windows_t1):
        return self.temporal.compute_smoothness(action_windows_t, action_windows_t1)

    def get_learned_decays(self):
        return self.temporal.decays.detach().cpu().numpy()


class SkeletonMixin:
    """骨架预测与损失混入。使用方必须有 self.skeleton_head, self.skeleton_mode, self.n_fine。"""

    def predict_skeleton(self, action_window):
        return self.skeleton_head(self.encode(action_window))

    def compute_skeleton_loss(self, pred_dict, gt_positions):
        losses = {}
        losses['fine'] = ((pred_dict['fine'] - gt_positions) ** 2).mean()
        gt_medium = downsample_skeleton(gt_positions, pred_dict['medium'].shape[-2])
        losses['medium'] = ((pred_dict['medium'] - gt_medium) ** 2).mean()
        gt_coarse = downsample_skeleton(gt_positions, pred_dict['coarse'].shape[-2])
        losses['coarse'] = ((pred_dict['coarse'] - gt_coarse) ** 2).mean()
        return losses

    def skeleton_config(self):
        cfg = {"skeleton_mode": self.skeleton_mode, "n_fine": self.n_fine}
        if hasattr(self, 'rod_radius'):
            cfg["rod_radius"] = self.rod_radius
        if self.skeleton_mode == "fourier":
            cfg["fourier_n_freq"] = self.skeleton_head.n_freq
        elif self.skeleton_mode == "bspline":
            cfg["bspline_n_ctrl"] = self.skeleton_head.n_ctrl
        elif self.skeleton_mode == "catmullrom":
            cfg["catmullrom_n_ctrl"] = self.skeleton_head.n_ctrl
        return cfg
