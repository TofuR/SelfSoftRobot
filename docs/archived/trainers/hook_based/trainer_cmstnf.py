"""CMSTNFTrainer — 标准 Canonical + Deformation 训练。"""

from .two_phase_trainer import TwoPhaseTrainer
from src.models.model_cmstnf import CMSTNFModel


class CMSTNFTrainer(TwoPhaseTrainer):
    def _model_name(self):
        return "CMSTNF"

    def _create_model(self, action_dim):
        return CMSTNFModel(
            action_dim=action_dim,
            window_size=self.temp_cfg["window_size"],
            n_scales=self.temp_cfg["n_scales"],
            hidden_dim=self.temp_cfg["hidden_dim"],
            d_filter=self.model_cfg["d_filter"],
            n_freqs=self.model_cfg["n_freqs"],
            deform_n_freqs=self.canon_cfg["deform_n_freqs"],
        )

    def _save_extra_params(self, model, log_dir):
        import numpy as np
        np.savetxt(log_dir + "/learned_decays.txt", model.get_learned_decays())
        print(f"    Decays: {model.get_learned_decays()}")
