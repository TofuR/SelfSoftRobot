"""view_strategy.py — 视角策略：管理射线采样、渲染、loss 聚合。

SingleViewStrategy  — 单视角，提取自 BaseTrainer
MultiViewStrategy   — 多视角，提取自 MultiViewTrainer
  可选 with_reprojection / with_consistency 启用跨视角约束
"""

from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

from src.utils.camera import get_rays
from src.utils.camera_system import MultiCameraSystem
from src.utils.rendering import (
    OM_rendering, OM_rendering_with_depth, sample_stratified,
)


def _query_chunked(forward_fn, pts, action_window, chunk_size=4096):
    """分块查询模型，避免 OOM。"""
    parts = []
    for i in range(0, pts.shape[0], chunk_size):
        raw = forward_fn(pts[i:i + chunk_size], action_window)
        parts.append(raw)
    return torch.cat(parts, dim=0)


class ViewStrategy(ABC):
    """视角策略基类。"""

    @abstractmethod
    def setup(self, device, config):
        """初始化射线等资源。"""

    @abstractmethod
    def compute_losses(self, forward_fn, action_window, images,
                       depths=None, active_losses=None) -> dict:
        """采样射线 → 查询模型 → 渲染 → 返回 losses dict。"""


class SingleViewStrategy(ViewStrategy):
    """单视角策略。"""

    def __init__(self, H, W, focal, camera_pose):
        self.H, self.W = H, W
        self.focal = focal
        self.camera_pose = camera_pose

    def setup(self, device, config):
        self.device = device
        mv_cfg = config.get("multiview", {})
        self.n_rays = mv_cfg.get("n_rays_per_view", 1024)
        self.n_samples = mv_cfg.get("n_samples", 64)
        self.fg_ratio = mv_cfg.get("fg_ratio", 0.5)
        self.near = mv_cfg.get("near", 0.5)
        self.far = mv_cfg.get("far", 2.5)
        self.w_recon = mv_cfg.get("w_recon_per_view", 1.0)
        self.w_depth = mv_cfg.get("w_depth", 0.1)

        focal_t = torch.tensor(self.focal).float().to(device)
        eye = self.camera_pose['eye']
        center = self.camera_pose['center']
        up = self.camera_pose['up']
        self.rays_o, self.rays_d = get_rays(
            self.H, self.W, focal_t, eye, center, up, device=device)

    def _sample_rays(self, target_img, n_rays=None):
        n_rays = n_rays or self.n_rays
        N_total = self.rays_o.shape[0]
        target_img = target_img.to(self.device)
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
        return sel, self.rays_o[sel], self.rays_d[sel]

    def compute_losses(self, forward_fn, action_window, images,
                       depths=None, active_losses=None):
        active = active_losses or ["recon", "depth"]
        if images.dim() == 1:
            images = images.unsqueeze(0)
        B = images.shape[0]
        total_recon = torch.tensor(0.0, device=self.device)
        total_depth = torch.tensor(0.0, device=self.device)

        for b in range(B):
            aw_b = action_window[b:b + 1] if action_window.dim() == 3 else action_window
            img_b = images[b]
            dep_b = depths[b] if depths is not None else None

            sel, rays_o_sel, rays_d_sel = self._sample_rays(img_b)
            target_pixels = img_b[sel]
            pts, z_vals = sample_stratified(rays_o_sel, rays_d_sel, self.near, self.far, self.n_samples)
            raw = _query_chunked(forward_fn, pts, aw_b)

            if dep_b is not None and "depth" in active:
                rendered, rendered_depth, _ = OM_rendering_with_depth(raw, z_vals)
                target_depth_sel = dep_b[sel]
                valid = target_depth_sel > 0
                if valid.any():
                    total_depth = total_depth + F.l1_loss(
                        rendered_depth[valid], target_depth_sel[valid])
            else:
                rendered, _ = OM_rendering(raw)

            if "recon" in active:
                total_recon = total_recon + F.mse_loss(rendered, target_pixels)

        losses = {}
        if "recon" in active:
            losses['recon'] = total_recon / B * self.w_recon
        if "depth" in active:
            losses['depth'] = total_depth / B * self.w_depth
        return losses


class MultiViewStrategy(ViewStrategy):
    """多视角策略，支持可选的跨视角一致性约束。"""

    def __init__(self, cam_system: MultiCameraSystem,
                 with_depth=False, with_consistency=False, with_reprojection=False):
        self.cam_system = cam_system
        self.n_views = cam_system.n_views
        self.with_depth = with_depth
        self.with_consistency = with_consistency
        self.with_reprojection = with_reprojection

    def setup(self, device, config):
        self.device = device
        mv_cfg = config.get("multiview", {})
        self.n_rays_per_view = mv_cfg.get("n_rays_per_view", 512)
        self.n_samples = mv_cfg.get("n_samples", 64)
        self.fg_ratio = mv_cfg.get("fg_ratio", 0.5)
        self.near = mv_cfg.get("near", 0.5)
        self.far = mv_cfg.get("far", 2.5)
        self.w_recon = mv_cfg.get("w_recon_per_view", 1.0)
        self.w_depth = mv_cfg.get("w_depth", 0.1)
        self.w_consist = mv_cfg.get("w_consist", 0.05)
        self.w_reproj = mv_cfg.get("w_reproj", 0.1)
        self.n_reproj_points = mv_cfg.get("n_reproj_points", 256)
        self.alpha_threshold = mv_cfg.get("alpha_threshold", 0.5)

        self.H = self.cam_system.cameras[0]['H']
        self.W = self.cam_system.cameras[0]['W']

        self.all_rays_o = []
        self.all_rays_d = []
        for v in range(self.n_views):
            rays_o, rays_d = self.cam_system.get_rays(v, device=device)
            self.all_rays_o.append(rays_o)
            self.all_rays_d.append(rays_d)

    def _sample_rays_for_view(self, view_idx, target_img, n_rays=None):
        n_rays = n_rays or self.n_rays_per_view
        rays_o = self.all_rays_o[view_idx]
        rays_d = self.all_rays_d[view_idx]
        N_total = rays_o.shape[0]
        target_img = target_img.to(self.device)
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

    def _render_view(self, forward_fn, action_window, view_idx, target_img,
                     target_depth=None):
        target_img = target_img.to(self.device)
        if target_depth is not None:
            target_depth = target_depth.to(self.device)
        sel_idx, rays_o, rays_d = self._sample_rays_for_view(view_idx, target_img)
        target_pixels = target_img[sel_idx]
        pts, z_vals = sample_stratified(rays_o, rays_d, self.near, self.far, self.n_samples)
        raw = _query_chunked(forward_fn, pts, action_window)

        if target_depth is not None:
            rendered, rendered_depth, _ = OM_rendering_with_depth(raw, z_vals)
            target_depth_sel = target_depth[sel_idx]
            valid = target_depth_sel > 0
            if valid.any():
                loss_depth = F.l1_loss(rendered_depth[valid], target_depth_sel[valid])
            else:
                loss_depth = torch.tensor(0.0, device=self.device)
        else:
            rendered, _ = OM_rendering(raw)
            loss_depth = torch.tensor(0.0, device=self.device)

        loss_recon = F.mse_loss(rendered, target_pixels)
        return loss_recon, loss_depth

    def _compute_reprojection_loss(self, forward_fn, action_window, images_list):
        if self.n_views < 2:
            return torch.tensor(0.0, device=self.device)
        view_A, view_B = 0, min(1, self.n_views - 1)
        target_img_A = images_list[view_A].to(self.device)
        target_img_B = images_list[view_B].to(self.device)

        sel_A, rays_o_sel, rays_d_sel = self._sample_rays_for_view(view_A, target_img_A)
        pts, z_vals = sample_stratified(rays_o_sel, rays_d_sel, self.near, self.far, self.n_samples)
        raw = _query_chunked(forward_fn, pts, action_window)
        _, depth_A, weights_A = OM_rendering_with_depth(raw, z_vals)

        alpha_sum = weights_A.sum(dim=-1)
        confident_mask = alpha_sum > self.alpha_threshold
        if confident_mask.sum() < 10:
            return torch.tensor(0.0, device=self.device)

        confident_idx = torch.where(confident_mask)[0]
        n_pts = min(self.n_reproj_points, confident_idx.shape[0])
        chosen = confident_idx[torch.randperm(confident_idx.shape[0], device=self.device)[:n_pts]]
        sel_depth = depth_A[chosen]
        sel_pixels = sel_A[chosen]

        H, W = self.H, self.W
        px_x = sel_pixels % W
        px_y = sel_pixels // W
        pixels_2d = torch.stack([px_x, px_y], dim=-1)
        points_3d = self.cam_system.unproject(pixels_2d, sel_depth, view_A, device=self.device)

        pixels_B, depths_B = self.cam_system.project(points_3d, view_B, device=self.device)
        valid = ((pixels_B[:, 0] >= 0) & (pixels_B[:, 0] < W) &
                 (pixels_B[:, 1] >= 0) & (pixels_B[:, 1] < H) & (depths_B > 0))
        if valid.sum() < 5:
            return torch.tensor(0.0, device=self.device)

        pixels_B_valid = pixels_B[valid]
        pixel_idx_B = (pixels_B_valid[:, 1].long() * W + pixels_B_valid[:, 0].long())
        pixel_idx_B = pixel_idx_B.clamp(0, H * W - 1)

        rays_o_B = self.all_rays_o[view_B]
        rays_d_B = self.all_rays_d[view_B]
        pts_B, z_vals_B = sample_stratified(
            rays_o_B[pixel_idx_B], rays_d_B[pixel_idx_B],
            self.near, self.far, self.n_samples)
        raw_B = _query_chunked(forward_fn, pts_B, action_window)
        rendered_B, _, _ = OM_rendering_with_depth(raw_B, z_vals_B)

        gt_B = target_img_B[pixel_idx_B]
        return F.mse_loss(rendered_B, gt_B)

    def _compute_consistency_loss(self, forward_fn, action_window, images_list):
        if self.n_views < 2:
            return torch.tensor(0.0, device=self.device)
        view_A, view_B = 0, min(1, self.n_views - 1)
        target_img_A = images_list[view_A].to(self.device)

        rays_o_A = self.all_rays_o[view_A]
        rays_d_A = self.all_rays_d[view_A]
        fg_mask = target_img_A > 0.1
        fg_idx = torch.where(fg_mask)[0]
        if fg_idx.shape[0] < 10:
            return torch.tensor(0.0, device=self.device)
        n_pts = min(self.n_reproj_points, self.n_rays_per_view)
        chosen = fg_idx[torch.randperm(fg_idx.shape[0], device=self.device)[:n_pts]]

        pts_A, z_vals_A = sample_stratified(
            rays_o_A[chosen], rays_d_A[chosen], self.near, self.far, self.n_samples)
        raw_A = _query_chunked(forward_fn, pts_A, action_window)
        _, _, weights_A = OM_rendering_with_depth(raw_A, z_vals_A)

        target_img_B = images_list[view_B].to(self.device)
        fg_mask_B = target_img_B > 0.1
        fg_idx_B = torch.where(fg_mask_B)[0]
        if fg_idx_B.shape[0] < 10:
            return torch.tensor(0.0, device=self.device)
        n_B = min(n_pts, fg_idx_B.shape[0])
        chosen_B = fg_idx_B[torch.randperm(fg_idx_B.shape[0], device=self.device)[:n_B]]

        pts_B, z_vals_B = sample_stratified(
            self.all_rays_o[view_B][chosen_B], self.all_rays_d[view_B][chosen_B],
            self.near, self.far, self.n_samples)
        raw_B = _query_chunked(forward_fn, pts_B, action_window)
        _, _, weights_B = OM_rendering_with_depth(raw_B, z_vals_B)

        alpha_A = weights_A.mean(dim=-1)
        alpha_B = weights_B.mean(dim=-1)
        n_cmp = min(alpha_A.shape[0], alpha_B.shape[0])
        if n_cmp < 5:
            return torch.tensor(0.0, device=self.device)

        return (F.mse_loss(alpha_A[:n_cmp].mean(), alpha_B[:n_cmp].mean()) +
                F.l1_loss(alpha_A[:n_cmp].std(), alpha_B[:n_cmp].std()))

    def compute_losses(self, forward_fn, action_window, images_list,
                       depths_list=None, active_losses=None,
                       gt_skeleton=None):
        active = active_losses or ["recon"]
        total_recon = torch.tensor(0.0, device=self.device)
        total_depth = torch.tensor(0.0, device=self.device)

        B = images_list[0].shape[0] if images_list[0].dim() == 2 else 1

        for b in range(B):
            aw_b = action_window[b:b + 1]

            # 按 batch 元素包装 forward_fn，注入对应的 GT skeleton
            fn_b = forward_fn
            if gt_skeleton is not None:
                gt_b = gt_skeleton[b:b + 1]
                _orig = forward_fn
                def fn_b(pts, aw, _gt=gt_b, _fn=_orig):
                    return _fn(pts, aw, gt_skeleton=_gt)

            imgs_b = [img[b].to(self.device) if img.dim() == 2 else img.to(self.device)
                      for img in images_list]
            deps_b = None
            if depths_list:
                deps_b = [(d[b].to(self.device) if d is not None and d.dim() == 2 else
                           d.to(self.device) if d is not None else None)
                          for d in depths_list]

            for v in range(self.n_views):
                dep_v = deps_b[v] if deps_b else None
                loss_recon, loss_depth = self._render_view(
                    fn_b, aw_b, v, imgs_b[v], target_depth=dep_v)
                total_recon = total_recon + loss_recon
                total_depth = total_depth + loss_depth

        losses = {}
        if "recon" in active:
            losses['recon'] = total_recon / (B * self.n_views) * self.w_recon
        if "depth" in active:
            losses['depth'] = total_depth / (B * self.n_views) * self.w_depth

        # 跨视角约束（仅对第一个样本）
        aw_first = action_window[0:1]
        imgs_first = [(img[0].to(self.device) if img[0].dim() == 1
                        else img.to(self.device))
                       for img in images_list]
        fn_first = forward_fn
        if gt_skeleton is not None:
            gt_first = gt_skeleton[0:1]
            _orig = forward_fn
            def fn_first(pts, aw, _gt=gt_first, _fn=_orig):
                return _fn(pts, aw, gt_skeleton=_gt)

        if "reproj" in active and self.with_reprojection:
            losses['reproj'] = self._compute_reprojection_loss(
                fn_first, aw_first, imgs_first) * self.w_reproj

        if "consist" in active and self.with_consistency:
            losses['consist'] = self._compute_consistency_loss(
                fn_first, aw_first, imgs_first) * self.w_consist

        return losses
