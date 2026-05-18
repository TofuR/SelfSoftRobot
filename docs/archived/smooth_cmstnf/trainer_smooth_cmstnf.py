"""SmoothCMSTNFTrainer — 正则化光滑变形的两阶段训练。"""

import numpy as np
from .two_phase_trainer import TwoPhaseTrainer
from src.models.model_smooth_cmstnf import SmoothCMSTNFModel


class SmoothCMSTNFTrainer(TwoPhaseTrainer):
    def _model_name(self):
        return "Smooth_CMSTNF"

    def _create_model(self, action_dim):
        return SmoothCMSTNFModel(
            action_dim=action_dim,
            window_size=self.temp_cfg["window_size"],
            n_scales=self.temp_cfg["n_scales"],
            hidden_dim=self.temp_cfg["hidden_dim"],
            d_filter=self.model_cfg["d_filter"],
            n_freqs=self.model_cfg["n_freqs"],
            deform_n_freqs=self.canon_cfg["deform_n_freqs"],
        )

    def _extra_phase2_losses(self, model, pts, seq_t, seq_t1, global_step):
        losses = {}
        w_jac = self.canon_cfg.get("w_jacobian", 0.01)
        w_tgrad = self.canon_cfg.get("w_temporal_grad", 0.01)

        # Jacobian 每 5 步计算一次（省时间）
        if global_step % 5 == 0:
            jac = model.compute_jacobian_penalty(pts, seq_t)
            losses['jac'] = (jac, w_jac)

        # 时序梯度惩罚
        tgrad = model.compute_temporal_gradient_penalty(pts, seq_t, seq_t1)
        losses['tgrad'] = (tgrad, w_tgrad)

        return losses

    def _save_extra_params(self, model, log_dir):
        np.savetxt(log_dir + "/learned_decays.txt", model.get_learned_decays())
        print(f"    Decays: {model.get_learned_decays()}")
