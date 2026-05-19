"""MS-SCNF Trainer — 两阶段训练：Phase 1 骨架回归 + Phase 2 联合训练。

Phase 1: 仅 SkeletonHead，3D L2 loss（GT 来自仿真器 position_collection）。
Phase 2: 联合 SkeletonHead + DensityField，3D skeleton loss + 2D rendering loss。

继承 TwoPhaseTrainer，通过钩子定制具体训练逻辑。
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .two_phase_trainer import TwoPhaseTrainer
from .metrics_3d import mean_node_error, endpoint_error, curve_smoothness
from src.models.model_ms_scnf import MSSCNFModel
from src.utils.rendering import OM_rendering, sample_stratified
from src.data.dataset import SoftSequenceDataset


class MSSCNFTrainer(TwoPhaseTrainer):
    """MS-SCNF 两阶段训练器。"""

    def __init__(self, device, config=None):
        super().__init__(device, config=config)
        self.ms_cfg = self.train_cfg.get("ms_scnf", {})

    def _model_name(self):
        return "MS_SCNF"

    def _create_model(self, action_dim):
        return MSSCNFModel(
            action_dim=action_dim,
            window_size=self.temp_cfg["window_size"],
            n_scales=self.temp_cfg["n_scales"],
            hidden_dim=self.temp_cfg["hidden_dim"],
            d_filter=self.model_cfg["d_filter"],
            n_freqs=self.model_cfg["n_freqs"],
            n_coarse=self.ms_cfg.get("n_coarse", 4),
            n_medium=self.ms_cfg.get("n_medium", 10),
            n_fine=self.ms_cfg.get("n_fine", 31),
            deform_n_freqs=self.canon_cfg["deform_n_freqs"],
            skeleton_mode=self.ms_cfg.get("skeleton_mode", "point"),
            fourier_n_freq=self.ms_cfg.get("fourier_n_freq", 8),
            bspline_n_ctrl=self.ms_cfg.get("bspline_n_ctrl", 10),
            catmullrom_n_ctrl=self.ms_cfg.get("catmullrom_n_ctrl", 10),
        )

    # ── Phase 1 钩子覆盖（骨架回归，无渲染）──────────────────────────────

    def _phase1_dataset(self, data_dir):
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not all_files:
            raise FileNotFoundError(
                f"No 3D sequence data in {data_dir}. "
                "Run: python scripts/data_collection/collect_sequence_3d.py")

        sample_data = np.load(all_files[0])
        if 'positions' not in sample_data:
            raise ValueError(
                f"Data in {data_dir} lacks 'positions' field. "
                "Use collect_sequence_3d.py to generate 3D data.")

        return SoftSequenceDataset(
            data_dir, seq_len=self.temp_cfg["window_size"],
            file_list=all_files, return_3d=True,
        )

    def _phase1_freeze(self, model):
        for p in model.density.parameters():
            p.requires_grad = False

    def _phase1_train_step(self, model, batch):
        seq = batch[0].to(self.device)
        positions = batch[2].to(self.device)
        gt_skeleton = positions.permute(0, 2, 1)

        state = model.encode(seq)
        pred_dict = model.skeleton_head(state)
        losses = model.compute_skeleton_loss(pred_dict, gt_skeleton)

        w_fine = self.ms_cfg.get("w_skeleton_fine", 1.0)
        w_medium = self.ms_cfg.get("w_skeleton_medium", 0.3)
        w_coarse = self.ms_cfg.get("w_skeleton_coarse", 0.1)

        loss = w_fine * losses['fine'] + w_medium * losses['medium'] + w_coarse * losses['coarse']
        info = {'fine': losses['fine'].item(), 'medium': losses['medium'].item(),
                'coarse': losses['coarse'].item()}
        return loss, info

    def _phase1_save(self, model, path):
        torch.save({
            'temporal': model.temporal.state_dict(),
            'skeleton_head': model.skeleton_head.state_dict(),
        }, path)

    def _phase1_validate(self, model, ds, epoch, log_dir):
        pass

    # ── Phase 2 钩子覆盖（skeleton + rendering 联合训练）──────────────────

    def _phase2_dataset(self, data_dir):
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not all_files:
            raise FileNotFoundError(f"No 3D sequence data in {data_dir}")

        split = max(1, int(0.8 * len(all_files)))
        train_files, val_files = all_files[:split], all_files[split:]

        train_ds = SoftSequenceDataset(
            data_dir, seq_len=self.temp_cfg["window_size"],
            file_list=train_files, return_pairs=True, return_3d=True,
        )
        val_ds = SoftSequenceDataset(
            data_dir, seq_len=self.temp_cfg["window_size"],
            file_list=val_files, norm_factor=train_ds.norm_factor,
        )
        return train_ds, val_ds

    def _phase2_load_phase1(self, model, path):
        if path and os.path.exists(path):
            state = torch.load(path, map_location=self.device, weights_only=True)
            model.temporal.load_state_dict(state['temporal'])
            model.skeleton_head.load_state_dict(state['skeleton_head'])
            print(f"    Loaded skeleton weights: {path}")
        else:
            print("    WARNING: No skeleton weights!")

    def _phase2_freeze(self, model):
        pass  # Phase 2 训练所有参数

    def _phase2_train_step(self, model, batch, global_step):
        seq_t, seq_t1, img_t, img_t1, pos_t, pos_t1 = batch
        seq_t = seq_t.to(self.device)
        seq_t1 = seq_t1.to(self.device)
        img_t = img_t.to(self.device)
        pos_t = pos_t.to(self.device)

        B = img_t.shape[0]
        gt_skeleton = pos_t.permute(0, 2, 1)

        w_skeleton = self.ms_cfg.get("w_skeleton_fine", 1.0)
        w_render = self.ms_cfg.get("w_render", 0.5)
        w_smooth = self.loss_cfg.get("smoothness", 0.01)

        # 1. Skeleton loss
        state_t = model.encode(seq_t)
        pred_dict = model.skeleton_head(state_t)
        skel_losses = model.compute_skeleton_loss(pred_dict, gt_skeleton)
        loss_skel = skel_losses['fine']

        # 2. Rendering loss
        sel, rays_o_sel, rays_d_sel = self.sample_fg_rays(img_t)
        pts, _ = sample_stratified(rays_o_sel, rays_d_sel, self.near, self.far, self.n_samples)

        rgb_map = self.render_points(lambda p: model(p, seq_t), pts)
        pred_t = rgb_map.reshape(B, -1)
        loss_render = torch.nn.functional.mse_loss(pred_t, img_t[:, sel])

        # 3. Smoothness loss
        loss_smooth = model.compute_smoothness(seq_t, seq_t1)

        loss = w_skeleton * loss_skel + w_render * loss_render + w_smooth * loss_smooth
        info = {'skel': loss_skel.item(), 'render': loss_render.item(),
                'smooth': loss_smooth.item()}
        return loss, info

    # ── 统一入口覆盖 ─────────────────────────────────────────────────────

    def train(self, data_dir="data/sequence_data_3d"):
        print(f"\n>>> {self._model_name()}: Phase 1 (Skeleton) → Phase 2 (Joint)")
        print(f"    Data: {data_dir}\n")
        exp_dir, skeleton_path = self.train_phase1(data_dir=data_dir)
        self.train_phase2(exp_dir, skeleton_path, data_dir=data_dir)
