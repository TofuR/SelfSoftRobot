"""trainer_multiview_consistency.py — 多视角一致性约束训练器（方案 B）。

在 Plan A（多视角渲染 + 深度监督）基础上，增加两个跨视角几何约束：

  约束 3 — 跨视角 density 一致性:
    同一 3D 区域从不同视角射线交汇处查询，density 分布应一致。
    实现方式：利用各视角渲染 depth → unproject 到 3D → 沿另一视角射线查询 → 比较密度统计量。

  约束 4 — 重投影一致性:
    视角 A 渲染 depth → unproject 到 3D → project 到视角 B → 比较渲染结果与 GT。

继承 MultiViewTrainer，复用 Plan A 的全部渲染和采样逻辑。
"""

import torch
import torch.nn.functional as F

from src.training.trainer_multiview import MultiViewTrainer
from src.utils.rendering import OM_rendering_with_depth, sample_stratified
from config.params import load_config


class MultiViewConsistencyTrainer(MultiViewTrainer):
    """多视角一致性约束训练器（方案 B）。

    在 Plan A 的多视角渲染 + 深度监督基础上，追加跨视角一致性 loss。
    需要至少 2 个视角才能计算跨视角约束。

    Args:
        model: 神经场模型，需支持 model(pts, action_window) → raw
        cam_system: MultiCameraSystem 实例
        device: torch device
        config: 训练配置 dict
    """

    def __init__(self, model, cam_system, device, config=None):
        super().__init__(model, cam_system, device, config)

        cfg = self.mv_cfg
        self.w_consist = cfg.get("w_consist", 0.05)
        self.w_reproj = cfg.get("w_reproj", 0.1)
        self.n_reproj_points = cfg.get("n_reproj_points", 256)
        self.alpha_threshold = cfg.get("alpha_threshold", 0.5)

        if self.n_views < 2:
            raise ValueError("MultiViewConsistencyTrainer 需要至少 2 个视角")

    # ── 重投影一致性 loss ──────────────────────────────────────────

    def _compute_reprojection_loss(self, action_window, images_list,
                                   target_img_A=None, target_img_B=None):
        """计算跨视角重投影一致性 loss。

        流程：
          1. 对视角 A 做完整 rendering（少量射线），得到 rendered depth + weights
          2. 选高置信度区域 → unproject 到 3D 点
          3. project 到视角 B → 在视角 B 的投影位置采样射线
          4. 沿射线查询模型 → 比较 rendered 值与 GT

        Args:
            action_window: (1, K, D)
            images_list: list of V 个 (H*W,) 目标图像
            target_img_A: (H*W,) 视角 A 图像（若为 None 则从 images_list 取）
            target_img_B: (H*W,) 视角 B 图像

        Returns:
            loss_reproj: scalar tensor
        """
        view_A, view_B = 0, min(1, self.n_views - 1)

        # 1. 对视角 A 采样射线并渲染
        rays_o_A = self.all_rays_o[view_A]
        rays_d_A = self.all_rays_d[view_A]
        n_rays = min(self.n_rays_per_view, rays_o_A.shape[0])

        if target_img_A is None:
            target_img_A = images_list[view_A]
        if target_img_B is None:
            target_img_B = images_list[view_B]

        # 前景 + 背景混合采样
        sel_A, rays_o_sel, rays_d_sel = self._sample_rays_for_view(
            view_A, target_img_A, n_rays=n_rays)

        pts, z_vals = sample_stratified(
            rays_o_sel, rays_d_sel, self.near, self.far, self.n_samples)
        raw = self._query_model_chunked(pts, action_window, chunk_size=4096)

        rendered_A, depth_A, weights_A = OM_rendering_with_depth(raw, z_vals)

        # 2. 选高置信度区域
        alpha_sum = weights_A.sum(dim=-1)  # (n_rays,)
        confident_mask = alpha_sum > self.alpha_threshold

        if confident_mask.sum() < 10:
            return torch.tensor(0.0, device=self.device)

        confident_idx = torch.where(confident_mask)[0]
        n_pts = min(self.n_reproj_points, confident_idx.shape[0])
        chosen = confident_idx[torch.randperm(confident_idx.shape[0], device=self.device)[:n_pts]]

        sel_depth = depth_A[chosen]  # (n_pts,)
        sel_pixels = sel_A[chosen]   # (n_pts,) pixel indices

        # 3. unproject 到 3D
        H, W = self.H, self.W
        px_x = sel_pixels % W
        px_y = sel_pixels // W
        pixels_2d = torch.stack([px_x, px_y], dim=-1)  # (n_pts, 2)

        points_3d = self.cam_system.unproject(pixels_2d, sel_depth, view_A, device=self.device)

        # 4. project 到视角 B
        pixels_B, depths_B = self.cam_system.project(points_3d, view_B, device=self.device)

        # 检查投影点是否在图像范围内
        valid = ((pixels_B[:, 0] >= 0) & (pixels_B[:, 0] < W) &
                 (pixels_B[:, 1] >= 0) & (pixels_B[:, 1] < H) &
                 (depths_B > 0))
        if valid.sum() < 5:
            return torch.tensor(0.0, device=self.device)

        pixels_B_valid = pixels_B[valid]
        pixel_idx_B = (pixels_B_valid[:, 1].long() * W + pixels_B_valid[:, 0].long())
        pixel_idx_B = pixel_idx_B.clamp(0, H * W - 1)

        # 5. 在视角 B 的投影位置采样射线并渲染
        rays_o_B = self.all_rays_o[view_B]
        rays_d_B = self.all_rays_d[view_B]

        rays_o_B_sel = rays_o_B[pixel_idx_B]  # (n_valid, 3)
        rays_d_B_sel = rays_d_B[pixel_idx_B]

        pts_B, z_vals_B = sample_stratified(
            rays_o_B_sel, rays_d_B_sel, self.near, self.far, self.n_samples)
        raw_B = self._query_model_chunked(pts_B, action_window, chunk_size=4096)
        rendered_B, _, _ = OM_rendering_with_depth(raw_B, z_vals_B)

        # 6. GT 对比
        gt_B = target_img_B[pixel_idx_B]
        loss_reproj = F.mse_loss(rendered_B, gt_B)

        return loss_reproj

    # ── 跨视角 density 一致性 loss ────────────────────────────────

    def _compute_consistency_loss(self, action_window, images_list,
                                  target_img_A=None):
        """跨视角 density 一致性 loss。

        对同一 3D 区域，从不同视角渲染的 density 统计量应一致。
        实现方式：从视角 A 的前景区域取 3D 点 → 在视角 B 沿最近射线渲染 →
        比较两者的 density 预测方差。

        Args:
            action_window: (1, K, D)
            images_list: list of V 个 (H*W,) 目标图像

        Returns:
            loss_consist: scalar tensor
        """
        if self.n_views < 2:
            return torch.tensor(0.0, device=self.device)

        view_A, view_B = 0, min(1, self.n_views - 1)

        if target_img_A is None:
            target_img_A = images_list[view_A]

        # 1. 从视角 A 前景区域采样射线
        rays_o_A = self.all_rays_o[view_A]
        rays_d_A = self.all_rays_d[view_A]
        n_pts = min(self.n_reproj_points, self.n_rays_per_view)

        fg_mask = target_img_A > 0.1
        fg_idx = torch.where(fg_mask)[0]
        if fg_idx.shape[0] < 10:
            return torch.tensor(0.0, device=self.device)

        chosen = fg_idx[torch.randperm(fg_idx.shape[0], device=self.device)[:n_pts]]
        rays_o_sel = rays_o_A[chosen]
        rays_d_sel = rays_d_A[chosen]

        # 2. 渲染视角 A，取 rendered depth
        pts_A, z_vals_A = sample_stratified(
            rays_o_sel, rays_d_sel, self.near, self.far, self.n_samples)
        raw_A = self._query_model_chunked(pts_A, action_window, chunk_size=4096)
        _, depth_A, weights_A = OM_rendering_with_depth(raw_A, z_vals_A)

        # 只保留有效深度（前景）
        valid = weights_A.sum(dim=-1) > self.alpha_threshold
        if valid.sum() < 5:
            return torch.tensor(0.0, device=self.device)

        depth_A_valid = depth_A[valid]

        # 3. unproject 到 3D 点
        sel_valid = chosen[valid]
        H, W = self.H, self.W
        px_x = sel_valid % W
        px_y = sel_valid // W
        pixels_2d = torch.stack([px_x, px_y], dim=-1)
        points_3d = self.cam_system.unproject(pixels_2d, depth_A_valid, view_A, device=self.device)

        # 4. 对同一 3D 点，从视角 B 找最近射线并查询 density
        rays_o_B = self.all_rays_o[view_B]
        rays_d_B = self.all_rays_d[view_B]

        # 简化：直接沿视角 B 的射线方向采样经过这些 3D 点附近的射线
        # 对每个 3D 点，计算在视角 B 射线上的投影参数 t，取最近的射线
        # 为了效率，随机采样视角 B 的前景射线
        target_img_B = images_list[view_B]
        fg_mask_B = target_img_B > 0.1
        fg_idx_B = torch.where(fg_mask_B)[0]
        if fg_idx_B.shape[0] < 10:
            return torch.tensor(0.0, device=self.device)

        n_B = min(n_pts, fg_idx_B.shape[0])
        chosen_B = fg_idx_B[torch.randperm(fg_idx_B.shape[0], device=self.device)[:n_B]]

        # 5. 沿视角 B 的射线渲染
        pts_B, z_vals_B = sample_stratified(
            rays_o_B[chosen_B], rays_d_B[chosen_B],
            self.near, self.far, self.n_samples)
        raw_B = self._query_model_chunked(pts_B, action_window, chunk_size=4096)
        _, _, weights_B = OM_rendering_with_depth(raw_B, z_vals_B)

        # 6. 比较两视角的 density 统计量：alpha 分布的均值和方差
        alpha_A = weights_A[valid].mean(dim=-1)  # (n_valid_A,)
        alpha_B = weights_B.mean(dim=-1)          # (n_valid_B,)

        # 取最小公共数量比较
        n_cmp = min(alpha_A.shape[0], alpha_B.shape[0])
        if n_cmp < 5:
            return torch.tensor(0.0, device=self.device)

        alpha_A_cmp = alpha_A[:n_cmp]
        alpha_B_cmp = alpha_B[:n_cmp]

        # 一致性：两组 alpha 分布的均值差异
        loss_consist = F.mse_loss(alpha_A_cmp.mean(), alpha_B_cmp.mean()) + \
                       F.l1_loss(alpha_A_cmp.std(), alpha_B_cmp.std())

        return loss_consist

    # ── 重写 train_step ─────────────────────────────────────────────

    def train_step(self, action_window, images_list, depths_list=None,
                   action_window_next=None, images_next_list=None):
        """单个训练 step，在 Plan A 基础上追加一致性和重投影 loss。

        Returns:
            dict of losses: total, recon, depth, consist, reproj, smooth
        """
        action_window = action_window.to(self.device)
        B = action_window.shape[0]

        total_recon = torch.tensor(0.0, device=self.device)
        total_depth = torch.tensor(0.0, device=self.device)

        for b in range(B):
            aw_b = action_window[b:b + 1]
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

        # ── 跨视角约束（仅对 batch 中第一个样本计算，节省开销） ──
        aw_first = action_window[0:1]
        imgs_first = [img[0].to(self.device) for img in images_list]

        loss_reproj = self._compute_reprojection_loss(
            aw_first, imgs_first)
        loss_consist = self._compute_consistency_loss(
            aw_first, imgs_first)

        losses['reproj'] = loss_reproj * self.w_reproj
        losses['consist'] = loss_consist * self.w_consist

        # 时序 smoothness
        if action_window_next is not None and hasattr(self.model, 'temporal'):
            action_window_next = action_window_next.to(self.device)
            state_t = self.model.temporal(action_window)
            state_t1 = self.model.temporal(action_window_next)
            losses['smooth'] = F.mse_loss(state_t, state_t1) * self.w_smooth

        losses['total'] = sum(losses.values())
        return losses
