"""SkeletonSDF 两阶段训练器。

Phase 1: 骨架预热 — 只训练时序编码器 + 骨架头
Phase 2: 联合训练 — 骨架 + SDF (含 Eikonal + Normal loss)

继承 TwoPhaseTrainer，覆盖所有 Phase 1/2 钩子。
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .two_phase_trainer import TwoPhaseTrainer
from src.data.dataset_skeleton_sdf import SkeletonSDFDataset
from src.models.model_skeleton_sdf import SkeletonSDFModel


def skeleton_smoothness(skeleton):
    """骨架二阶差分正则 (B, N, 3) -> scalar。"""
    if skeleton.shape[1] < 3:
        return torch.tensor(0.0, device=skeleton.device)
    return ((skeleton[:, 2:] - 2 * skeleton[:, 1:-1] + skeleton[:, :-2]) ** 2).mean()


class SkeletonSDFTrainer(TwoPhaseTrainer):
    """骨架 + SDF 两阶段训练器。"""

    def __init__(self, device, config,
                 skeleton_mode="bspline", rod_radius=0.015,
                 phase1_epochs=50):
        super().__init__(device, config=config)
        self.skeleton_mode = skeleton_mode
        self.rod_radius = rod_radius
        self._phase1_ep = phase1_epochs

        ms = self.train_cfg.get("ms_scnf", {})
        sdf = self.train_cfg.get("sdf", {})

        # 骨架 loss 权重
        self.w_skel_fine = ms.get("w_skeleton_fine", 1.0)
        self.w_skel_medium = ms.get("w_skeleton_medium", 0.3)
        self.w_skel_coarse = ms.get("w_skeleton_coarse", 0.1)
        self.w_skel_smooth = ms.get("w_smooth", 0.01)

        # SDF loss 权重
        self.w_sdf = sdf.get("w_sdf", 3e3)
        self.w_normal = sdf.get("w_normal", 10.0)
        self.w_eikonal = sdf.get("w_eikonal", 50.0)

        # SDF 采样参数
        self.n_surface = sdf.get("n_surface", 500)
        self.n_near = sdf.get("n_near_surface", 500)
        self.n_off = sdf.get("n_off_surface", 500)

    # ── 模型构造 ──

    def _model_name(self):
        return "SkeletonSDF"

    def _create_model(self, action_dim):
        return SkeletonSDFModel(
            action_dim=action_dim,
            window_size=self.temp_cfg.get("window_size", 20),
            n_scales=self.temp_cfg.get("n_scales", 4),
            hidden_dim=self.temp_cfg.get("hidden_dim", 128),
            skeleton_mode=self.skeleton_mode,
            rod_radius=self.rod_radius,
        )

    # ── 配置钩子 ──

    def _phase1_epochs(self):
        return self._phase1_ep

    def _phase2_epochs(self):
        return self.train_cfg.get("optimization", {}).get("n_epochs", 500) - self._phase1_ep

    def _phase1_lr(self):
        return self.train_cfg.get("optimization", {}).get("lr", 5e-5)

    def _phase2_lr(self):
        return self._phase1_lr() * 0.5

    # ── Phase 1 钩子（骨架预热，无渲染）──────────────────────────────────

    def _phase1_dataset(self, data_dir):
        return SkeletonSDFDataset(
            data_dir, seq_len=self.temp_cfg.get("window_size", 20),
            n_surface=self.n_surface,
            n_near_surface=self.n_near,
            n_off_surface=self.n_off,
        )

    def _phase1_freeze(self, model):
        if model.sdf_residual:
            for p in model.sdf_net.parameters():
                p.requires_grad = False
            for p in model.state_proj.parameters():
                p.requires_grad = False

    def _phase1_train_step(self, model, batch):
        action_window, _, _, _, gt_positions = batch
        action_window = action_window.to(self.device)
        gt_positions = gt_positions.to(self.device).squeeze(0)

        pred_dict = model.predict_skeleton(action_window)
        skel = model.compute_skeleton_loss(pred_dict, gt_positions.unsqueeze(0))

        loss_skel = (skel['fine'] * self.w_skel_fine +
                     skel['medium'] * self.w_skel_medium +
                     skel['coarse'] * self.w_skel_coarse)
        loss_smooth = skeleton_smoothness(pred_dict['fine']) * self.w_skel_smooth

        loss = loss_skel + loss_smooth
        info = {'skel': loss_skel.item(), 'smooth': loss_smooth.item()}
        return loss, info

    def _phase1_save(self, model, path):
        torch.save({
            'temporal': model.temporal.state_dict(),
            'skeleton_head': model.skeleton_head.state_dict(),
        }, path)

    def _phase1_validate(self, model, ds, epoch, log_dir):
        pass

    # ── Phase 2 钩子（骨架 + SDF 联合训练）───────────────────────────────

    def _phase2_dataset(self, data_dir):
        ds = SkeletonSDFDataset(
            data_dir, seq_len=self.temp_cfg.get("window_size", 20),
            n_surface=self.n_surface,
            n_near_surface=self.n_near,
            n_off_surface=self.n_off,
        )
        return ds, ds  # 无单独验证集，训练集同时用于验证

    def _phase2_load_phase1(self, model, path):
        if path and os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            model.temporal.load_state_dict(ckpt['temporal'])
            model.skeleton_head.load_state_dict(ckpt['skeleton_head'])
            print(f"    Loaded skeleton weights: {path}")

    def _phase2_freeze(self, model):
        pass  # Phase 2 训练所有参数

    def _phase2_train_step(self, model, batch, global_step):
        action_window, coords, gt_sdf, gt_normals, gt_positions = batch
        action_window = action_window.to(self.device)
        coords = coords.to(self.device).squeeze(0).requires_grad_(True)
        gt_sdf = gt_sdf.to(self.device).squeeze(0)
        gt_normals = gt_normals.to(self.device).squeeze(0)
        gt_positions = gt_positions.to(self.device).squeeze(0)

        # 骨架 loss
        pred_dict = model.predict_skeleton(action_window)
        skel = model.compute_skeleton_loss(pred_dict, gt_positions.unsqueeze(0))
        loss_skel = (skel['fine'] * self.w_skel_fine +
                     skel['medium'] * self.w_skel_medium +
                     skel['coarse'] * self.w_skel_coarse)
        loss_smooth = skeleton_smoothness(pred_dict['fine']) * self.w_skel_smooth
        loss_skeleton = loss_skel + loss_smooth

        # SDF loss (共享梯度)
        query = coords.unsqueeze(0)
        pred_sdf = model(query, action_window).squeeze(-1)

        loss_sdf_l1 = torch.abs(pred_sdf - gt_sdf).mean() * self.w_sdf

        gradient = torch.autograd.grad(
            pred_sdf.sum(), coords, create_graph=True,
        )[0]

        is_surface = (gt_sdf.abs() < 1e-6).float()
        n_surf = is_surface.sum()
        loss_normal = torch.tensor(0.0, device=self.device)
        if n_surf > 0 and gt_normals.abs().sum() > 0:
            cos_sim = F.cosine_similarity(gradient, gt_normals, dim=-1)
            loss_normal = (is_surface * (1 - cos_sim)).sum() / (n_surf + 1e-8) * self.w_normal

        loss_eikonal = ((gradient.norm(dim=-1) - 1) ** 2).mean() * self.w_eikonal

        loss_sdf = loss_sdf_l1 + loss_normal + loss_eikonal
        total = loss_skeleton + loss_sdf

        info = {
            'skel': loss_skeleton.item(),
            'sdf': loss_sdf_l1.item(),
            'normal': loss_normal.item(),
            'eikonal': loss_eikonal.item(),
        }
        return total, info

    def _phase2_validate(self, model, val_ds, epoch, log_dir):
        return None  # 无渲染验证，train_phase2 会用训练 loss 作为 monitor

    # ── 统一入口覆盖 ──

    def train(self, data_dir="data/sequence_data_3d"):
        print(f"\n>>> {self._model_name()}: Phase 1 (Skeleton) → Phase 2 (Skeleton+SDF)")
        print(f"    Data: {data_dir}\n")
        exp_dir, skel_path = self.train_phase1(data_dir=data_dir)
        self.train_phase2(exp_dir, skel_path, data_dir=data_dir)
