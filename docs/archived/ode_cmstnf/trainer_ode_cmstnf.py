"""ODECMSTNFTrainer — Neural ODE 时序编码的两阶段训练。"""

import json
from .two_phase_trainer import TwoPhaseTrainer
from src.models.model_ode_cmstnf import ODECMSTNFModel


class ODECMSTNFTrainer(TwoPhaseTrainer):
    def _model_name(self):
        return "ODE_CMSTNF"

    def _create_model(self, action_dim):
        return ODECMSTNFModel(
            action_dim=action_dim,
            window_size=self.temp_cfg["window_size"],
            hidden_dim=self.temp_cfg["hidden_dim"],
            d_filter=self.model_cfg["d_filter"],
            n_freqs=self.model_cfg["n_freqs"],
            deform_n_freqs=self.canon_cfg["deform_n_freqs"],
        )

    def _save_extra_params(self, model, log_dir):
        params = model.get_ode_params()
        with open(log_dir + "/ode_params.json", 'w') as f:
            json.dump({k: v.tolist() if hasattr(v, 'tolist') else v
                       for k, v in params.items()}, f, indent=2)
        print(f"    ODE params: dt={params['dt']:.4f}, "
              f"k[:4]={params['stiffness'][:4]}, c[:4]={params['damping'][:4]}")
