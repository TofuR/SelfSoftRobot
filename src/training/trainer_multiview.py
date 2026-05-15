"""trainer_multiview.py — 多视角 + 深度监督训练器（方案 A）。

在现有模型（CMSTNF/MS-SCNF/MSTNF）基础上，不做模型架构修改，
加入多视角 rendering loss 和深度 loss。

核心流程:
  每个 training step:
    for each view_i in views:
      采样射线 → 查询模型 → 体渲染 → MSE(rendered, gt)
      如果有深度: L1(rendered_depth, gt_depth)
    loss = Σ view_losses / V + w_depth * depth_loss + w_smooth * smoothness_loss
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.camera_system import MultiCameraSystem
from src.utils.rendering import (
    OM_rendering, OM_rendering_with_depth,
    sample_stratified, sample_depth_guided,
)
from src.utils.experiment import create_experiment
from config.params import load_config


class MultiViewTrainer:
    """多视角 + 深度监督训练器。

    与 BaseTrainer 平行设计，不继承以避免单视角耦合。
    直接接收任意模型（只要 forward 接口兼容）。

    Args:
        model: 神经场模型，需支持 model(pts, action_window) → raw
        cam_system: MultiCameraSystem 实例
        device: torch device
        config: 训练配置 dict（默认从 training.json 读取）
    """

    def __init__(self, model, cam_system, device, config=None):
        self.model = model
        self.cam_system = cam_system
        self.device = device
        self.train_cfg = config or load_config("training")
        self.mv_cfg = self.train_cfg.get("multiview", {})

        self.n_views = cam_system.n_views
        self.H = cam_system.cameras[0]['H']
        self.W = cam_system.cameras[0]['W']

        # 采样参数
        self.n_rays_per_view = self.mv_cfg.get("n_rays_per_view", 512)
        self.n_samples = self.mv_cfg.get("n_samples", 64)
        self.fg_ratio = self.mv_cfg.get("fg_ratio", 0.5)
        self.near = self.mv_cfg.get("near", 0.5)
        self.far = self.mv_cfg.get("far", 2.5)

        # Loss 权重
        self.w_depth = self.mv_cfg.get("w_depth", 0.1)
        self.w_recon = self.mv_cfg.get("w_recon_per_view", 1.0)
        self.w_smooth = self.train_cfg.get("loss_weights", {}).get("smoothness", 0.1)

        # 深度引导采样
        self.use_depth_guided = self.mv_cfg.get("use_depth_guided_sampling", False)
        self.n_coarse = self.mv_cfg.get("n_coarse", 32)
        self.n_fine = self.mv_cfg.get("n_fine", 32)

        # 预计算所有视角的射线
        self.all_rays_o = []
        self.all_rays_d = []
        for v in range(self.n_views):
            rays_o, rays_d = cam_system.get_rays(v, device=device)
            self.all_rays_o.append(rays_o)
            self.all_rays_d.append(rays_d)

    def _sample_rays_for_view(self, view_idx, target_img, n_rays=None):
        """为指定视角采样前景+背景混合射线。"""
        n_rays = n_rays or self.n_rays_per_view
        rays_o = self.all_rays_o[view_idx]
        rays_d = self.all_rays_d[view_idx]
        N_total = rays_o.shape[0]

        fg_mask = target_img > 0.1
        fg_idx = torch.where(fg_mask)[0]
        n_fg = int(n_rays * self.fg_ratio)
        n_bg = n_rays - n_fg

        if len(fg_idx) > 0 and n_fg > 0:
            chosen_fg = fg_idx[torch.randint(len(fg_idx), (n_fg,), device=self.device)]
            chosen_bg = torch.randint(N_total, (n_bg,), device=self.device)
            sel = torch.cat([chosen_fg, chosen_bg])
        else:
            sel = torch.randint(N_total, (n_rays,), device=self.device)

        return sel, rays_o[sel], rays_d[sel]

    def _render_view(self, action_window, view_idx, target_img,
                     target_depth=None, chunk_size=4096):
        """对单视角执行渲染 + 计算 loss。"""
        sel_idx, rays_o, rays_d = self._sample_rays_for_view(view_idx, target_img)
        target_pixels = target_img[sel_idx]

        if self.use_depth_guided and target_depth is not None:
            # Coarse pass
            pts_coarse, z_vals_coarse = sample_stratified(
                rays_o, rays_d, self.near, self.far, self.n_coarse)
            raw_coarse = self._query_model_chunked(pts_coarse, action_window, chunk_size)
            _, _, weights_coarse = OM_rendering_with_depth(raw_coarse, z_vals_coarse)

            # Fine pass
            pts, z_vals = sample_depth_guided(
                rays_o, rays_d, z_vals_coarse, weights_coarse,
                self.n_fine, self.near, self.far)
        else:
            pts, z_vals = sample_stratified(
                rays_o, rays_d, self.near, self.far, self.n_samples)

        raw = self._query_model_chunked(pts, action_window, chunk_size)

        if target_depth is not None:
            rendered, rendered_depth, _ = OM_rendering_with_depth(raw, z_vals)
            target_depth_sel = target_depth[sel_idx]
            valid_mask = target_depth_sel > 0
            if valid_mask.any():
                loss_depth = F.l1_loss(rendered_depth[valid_mask],
                                       target_depth_sel[valid_mask])
            else:
                loss_depth = torch.tensor(0.0, device=self.device)
        else:
            rendered, _ = OM_rendering(raw)
            loss_depth = torch.tensor(0.0, device=self.device)

        loss_recon = F.mse_loss(rendered, target_pixels)
        return loss_recon, loss_depth, rendered

    def _query_model_chunked(self, pts, action_window, chunk_size):
        """分块查询模型，避免 OOM。"""
        parts = []
        for i in range(0, pts.shape[0], chunk_size):
            chunk_pts = pts[i:i + chunk_size]
            raw = self.model(chunk_pts, action_window)
            parts.append(raw)
        return torch.cat(parts, dim=0)

    def train_step(self, action_window, images_list, depths_list=None,
                   action_window_next=None, images_next_list=None):
        """单个训练 step（支持 batched 输入）。

        遍历 batch 维度逐样本渲染，累加 loss 用于 backward。

        Args:
            action_window: (B, K, D) 当前帧 action
            images_list: list of V 个 (B, H*W) 目标图像
            depths_list: list of V 个 (B, H*W) 深度图（可选）
            action_window_next: (B, K, D) 下一帧 action
            images_next_list: list of V 个 (B, H*W) 下一帧图像

        Returns:
            dict of losses (averaged over batch × views)
        """
        action_window = action_window.to(self.device)
        B = action_window.shape[0]

        total_recon = torch.tensor(0.0, device=self.device)
        total_depth = torch.tensor(0.0, device=self.device)

        for b in range(B):
            aw_b = action_window[b:b + 1]  # (1, K, D)
            imgs_b = [img[b].to(self.device) for img in images_list]
            deps_b = None
            if depths_list:
                deps_b = [(d[b].to(self.device) if d is not None else None)
                          for d in depths_list]

            for v in range(self.n_views):
                dep_v = deps_b[v] if deps_b else None
                loss_recon, loss_depth, _ = self._render_view(
                    aw_b, v, imgs_b[v], target_depth=dep_v)
                total_recon = total_recon + loss_recon
                total_depth = total_depth + loss_depth

        total_recon = total_recon / (B * self.n_views) * self.w_recon
        total_depth = total_depth / (B * self.n_views) * self.w_depth

        losses = {'recon': total_recon, 'depth': total_depth}

        # 时序 smoothness
        if action_window_next is not None and hasattr(self.model, 'temporal'):
            action_window_next = action_window_next.to(self.device)
            state_t = self.model.temporal(action_window)
            state_t1 = self.model.temporal(action_window_next)
            losses['smooth'] = F.mse_loss(state_t, state_t1) * self.w_smooth

        losses['total'] = sum(losses.values())
        return losses

    def validate(self, val_loader, n_samples=20):
        """在验证集上计算平均 loss。"""
        self.model.eval()
        total_loss = 0.0
        n_eval = min(n_samples, len(val_loader))

        with torch.no_grad():
            indices = np.random.choice(len(val_loader), n_eval, replace=False)
            for idx in indices:
                batch = val_loader[idx]
                action_window = batch[0].unsqueeze(0)  # (1, K, D)
                images_list = [img.unsqueeze(0) for img in batch[1]]  # (1, H*W)
                depths_list = None
                if len(batch) > 2 and batch[2] is not None:
                    depths_list = [d.unsqueeze(0) for d in batch[2]]
                losses = self.train_step(action_window, images_list,
                                         depths_list=depths_list)
                total_loss += losses['total'].item()

        self.model.train()
        return total_loss / n_eval
