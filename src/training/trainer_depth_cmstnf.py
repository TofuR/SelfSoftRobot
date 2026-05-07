"""Depth-supervised CMSTNF Trainer — 在 CMSTNF 基础上添加深度监督损失和深度引导采样。

继承 TwoPhaseTrainer，Phase 1 不变（canonical field 训练不需要深度），
Phase 2 额外添加:
  - 深度渲染: 从 NeRF density field 计算期望深度
  - 深度损失: L1(|E[d] - d_gt|) 作为额外监督信号
  - 深度引导采样: coarse-to-fine 两阶段采样，集中在物体表面附近
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .two_phase_trainer import TwoPhaseTrainer
from src.models.model_cmstnf import CMSTNFModel
from src.utils.rendering import (
    OM_rendering, OM_rendering_with_depth, sample_stratified, sample_depth_guided,
)
from src.data.dataset import SoftSequenceDataset


class DepthCMSTNFTrainer(TwoPhaseTrainer):
    """Depth-supervised CMSTNF: 在 CMSTNF 两阶段训练基础上添加深度监督。"""

    def __init__(self, device, depth_weight=0.1, use_guided_sampling=True):
        super().__init__(device)
        self.depth_weight = depth_weight
        self.use_guided_sampling = use_guided_sampling

    def _model_name(self):
        return "Depth-CMSTNF"

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
        np.savetxt(log_dir + "/learned_decays.txt", model.get_learned_decays())
        print(f"    Decays: {model.get_learned_decays()}")

    def render_points_with_depth(self, forward_fn, pts, z_vals, chunk_size=4096):
        """渲染像素值 + 期望深度。

        Args:
            forward_fn: 模型前向函数。
            pts: 采样点 (N_rays, N_samples, 3)。
            z_vals: 采样深度 (N_rays, N_samples)。
            chunk_size: 分块大小。

        Returns:
            rgb_map: (N_rays,) 渲染像素值。
            depth_map: (N_rays,) 期望深度。
        """
        rgb_parts = []
        depth_parts = []

        for i in range(0, pts.shape[0], chunk_size):
            pts_chunk = pts[i:i + chunk_size]
            z_chunk = z_vals[i:i + chunk_size]
            raw = forward_fn(pts_chunk)
            rgb_chunk, depth_chunk, _ = OM_rendering_with_depth(raw, z_chunk)
            rgb_parts.append(rgb_chunk)
            depth_parts.append(depth_chunk)

        return torch.cat(rgb_parts, dim=0), torch.cat(depth_parts, dim=0)

    def sample_rays_depth_guided(self, model, seq, target_img, n_coarse=32, n_fine=32):
        """深度引导的两阶段采样。

        Args:
            model: CMSTNF 模型。
            seq: 动作窗口 (B, K, D)。
            target_img: 目标图像 (B, N_pixels)。
            n_coarse: 第一阶段采样数。
            n_fine: 第二阶段精细采样数。

        Returns:
            pts: 采样点。
            z_vals: 采样深度。
            sel: 选择的射线索引。
        """
        sel, rays_o_sel, rays_d_sel = self.sample_fg_rays(target_img)
        pts_coarse, z_vals_coarse = sample_stratified(
            rays_o_sel, rays_d_sel, self.near, self.far, n_coarse)

        with torch.no_grad():
            raw_coarse = self.render_points(lambda p: model(p, seq), pts_coarse)
            alpha = 1.0 - torch.exp(-torch.nn.functional.softplus(raw_coarse[..., 1]))
            transmittance = torch.cumprod(
                torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha[..., :-1] + 1e-10], dim=-1),
                dim=-1,
            )
            weights_coarse = transmittance * alpha

        pts_fine, z_vals_fine = sample_depth_guided(
            rays_o_sel, rays_d_sel, z_vals_coarse, weights_coarse,
            n_fine, self.near, self.far)

        return pts_fine, z_vals_fine, sel

    def train_phase2(self, exp_dir, canonical_path, data_dir="data/sequence_data"):
        """Phase 2 with depth supervision."""
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not all_files:
            raise FileNotFoundError(f"No sequence data in {data_dir}")

        split = max(1, int(0.8 * len(all_files)))
        train_files, val_files = all_files[:split], all_files[split:]

        has_depth_data = False
        sample_data = np.load(all_files[0])
        if 'depth_maps' in sample_data:
            has_depth_data = True
            print(f"    Depth maps detected in data, enabling depth supervision")

        train_ds = SoftSequenceDataset(
            data_dir, seq_len=self.temp_cfg["window_size"],
            file_list=train_files, return_pairs=True,
            return_depth=has_depth_data,
        )
        val_ds = SoftSequenceDataset(
            data_dir, seq_len=self.temp_cfg["window_size"],
            file_list=val_files, norm_factor=train_ds.norm_factor,
        )
        train_loader = DataLoader(train_ds, batch_size=self.opt_cfg["batch_size"],
                                  shuffle=True, num_workers=4)

        self.setup_camera(train_ds.H, train_ds.W, train_ds.focal,
                          camera_pose=train_ds.get_camera_params())
        action_dim = train_ds.action_dim

        model = self._create_model(action_dim).to(self.device)

        if canonical_path and os.path.exists(canonical_path):
            state = torch.load(canonical_path, map_location=self.device, weights_only=True)
            model.canonical.load_state_dict(state)
            print(f"    Loaded canonical: {canonical_path}")
        else:
            print("    WARNING: No canonical weights!")

        model.freeze_canonical()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=self.canon_cfg["deform_lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.opt_cfg["scheduler_patience"])

        w_recon = self.loss_cfg["recon_current"]
        w_recon_next = self.loss_cfg["recon_next"]
        w_smooth = self.loss_cfg["smoothness"]
        w_depth = self.depth_weight
        n_epochs = self.canon_cfg["phase2_epochs"]
        save_rate = self.log_cfg["save_rate"]

        phase2_dir = self.make_phase_dirs(exp_dir, "phase2")

        n_trainable = sum(p.numel() for p in trainable_params)
        print(f"\n{'='*60}")
        print(f">>> Phase 2: {self._model_name()}, {n_epochs} epochs")
        print(f"    Data: {data_dir}, Trainable: {n_trainable:,}")
        print(f"    Depth weight: {w_depth}, Guided sampling: {self.use_guided_sampling}")
        print(f"    Log: {phase2_dir}")
        print(f"{'='*60}")

        val_actions = val_ds.get_raw_actions(seq_id=0)

        def val_forward(val_seq):
            def fn(pts_chunk):
                return model(pts_chunk, val_seq)
            pts, _ = sample_stratified(self.rays_o, self.rays_d, self.near, self.far,
                                       self.n_samples, perturb=False)
            return self.render_points(fn, pts)

        best_val_loss = float("inf")
        global_step = 0

        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0
            epoch_depth_loss = 0
            depth_count = 0
            pbar = tqdm(train_loader, desc=f"[Phase2] Epoch {epoch}/{n_epochs}")

            for batch in pbar:
                if has_depth_data and len(batch) == 6:
                    seq_t, seq_t1, img_t, img_t1, depth_t, depth_t1 = batch
                else:
                    seq_t, seq_t1, img_t, img_t1 = batch[:4]
                    depth_t = depth_t1 = None

                seq_t = seq_t.to(self.device)
                seq_t1 = seq_t1.to(self.device)
                img_t = img_t.to(self.device)
                img_t1 = img_t1.to(self.device)
                if depth_t is not None:
                    depth_t = depth_t.to(self.device)
                    depth_t1 = depth_t1.to(self.device)

                B = img_t.shape[0]

                if self.use_guided_sampling and has_depth_data:
                    pts, z_vals, sel = self.sample_rays_depth_guided(
                        model, seq_t, img_t, n_coarse=32, n_fine=32)
                else:
                    sel, rays_o_sel, rays_d_sel = self.sample_fg_rays(img_t)
                    pts, z_vals = sample_stratified(
                        rays_o_sel, rays_d_sel, self.near, self.far, self.n_samples)

                rgb_map, depth_pred = self.render_points_with_depth(
                    lambda p: model(p, seq_t), pts, z_vals)
                pred_t = rgb_map.reshape(B, -1)
                loss_recon = torch.nn.functional.mse_loss(pred_t, img_t[:, sel])

                rgb_map2 = self.render_points(
                    lambda p: model(p, seq_t1), pts)
                pred_t1 = rgb_map2.reshape(B, -1)
                loss_recon_next = torch.nn.functional.mse_loss(pred_t1, img_t1[:, sel])

                loss_smooth = model.compute_smoothness(seq_t, seq_t1)

                loss = w_recon * loss_recon + w_recon_next * loss_recon_next + w_smooth * loss_smooth

                if has_depth_data and depth_t is not None:
                    depth_gt = depth_t[:, sel]
                    fg_mask = depth_gt > 0.01
                    if fg_mask.any():
                        loss_depth = torch.nn.functional.l1_loss(
                            depth_pred.reshape(B, -1)[fg_mask],
                            depth_gt[fg_mask])
                        loss = loss + w_depth * loss_depth
                        epoch_depth_loss += loss_depth.item()
                        depth_count += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                postfix = {'loss': f'{loss.item():.5f}'}
                if depth_count > 0:
                    postfix['d_loss'] = f'{epoch_depth_loss/depth_count:.5f}'
                pbar.set_postfix(postfix)
                global_step += 1

            model.eval()
            val_loss = self.validate_and_gif(
                val_forward, val_ds, epoch, phase2_dir, action_curves=val_actions)
            scheduler.step(val_loss)

            avg_train = epoch_loss / max(len(train_loader), 1)
            depth_str = f" | Depth: {epoch_depth_loss/max(depth_count,1):.5f}" if depth_count > 0 else ""
            print(f"  Epoch {epoch} | Train: {avg_train:.5f} | Val: {val_loss:.5f}{depth_str}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(phase2_dir, "model", "best_model.pt"))

            if global_step % save_rate == 0:
                torch.save(model.state_dict(),
                           os.path.join(phase2_dir, "model", f"model_{global_step:05d}.pt"))

        np.savetxt(os.path.join(phase2_dir, "action_norm_factor.txt"), [train_ds.norm_factor])
        self._save_extra_params(model, phase2_dir)
        print(f">>> Phase 2 done! Best val: {best_val_loss:.5f}")

    def train(self, data_dir="data/sequence_data", canonical_data_dir="data/canonical_data"):
        print(f"\n>>> {self._model_name()}: Phase 1 → Phase 2 (with depth supervision)")
        print(f"    Canonical data: {canonical_data_dir}")
        print(f"    Sequence data:  {data_dir}\n")
        exp_dir, canonical_path = self.train_phase1(data_dir=canonical_data_dir)
        self.train_phase2(exp_dir, canonical_path, data_dir=data_dir)
