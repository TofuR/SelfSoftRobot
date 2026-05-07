"""RGBDTrainer — RGB-D Neural Field 单阶段端到端训练。

Loss 组合:
  L_img     : 图像重建 MSE
  L_depth   : 深度监督 L1（前景区域）
  L_smooth  : 时序平滑
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import BaseTrainer
from src.models.model_rgbd import RGBDNeuralField
from src.utils.rendering import (
    OM_rendering, OM_rendering_with_depth, sample_stratified, sample_depth_guided,
)
from src.data.dataset import SoftSequenceDataset


class RGBDTrainer(BaseTrainer):
    """RGB-D Neural Field 单阶段训练器。"""

    def __init__(self, device, depth_weight=1.0, normal_weight=0.1,
                 smooth_weight=0.01, use_guided_sampling=True):
        super().__init__(device)
        self.depth_weight = depth_weight
        self.normal_weight = normal_weight
        self.smooth_weight = smooth_weight
        self.use_guided_sampling = use_guided_sampling

    def _create_model(self, action_dim, H, W):
        return RGBDNeuralField(
            action_dim=action_dim,
            depth_feat_dim=64,
            hidden_dim=64,
            d_filter=self.train_cfg["model"].get("d_filter", 128),
            n_freqs=self.train_cfg["model"].get("n_freqs", 10),
            window_size=self.train_cfg["temporal"].get("window_size", 10),
        ).to(self.device)

    def render_points_with_depth(self, forward_fn, pts, z_vals, chunk_size=4096):
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

    def train(self, data_dir="data/sequence_data", n_epochs=100):
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not all_files:
            raise FileNotFoundError(f"No data in {data_dir}")

        split = max(1, int(0.8 * len(all_files)))
        train_files, val_files = all_files[:split], all_files[split:]

        has_depth_data = False
        sample_data = np.load(all_files[0])
        if 'depth_maps' in sample_data:
            has_depth_data = True
            print(f"    Depth maps detected, enabling RGB-D training")

        train_ds = SoftSequenceDataset(
            data_dir, seq_len=self.train_cfg["temporal"].get("window_size", 10),
            file_list=train_files, return_pairs=True, return_depth=has_depth_data,
        )
        val_ds = SoftSequenceDataset(
            data_dir, seq_len=self.train_cfg["temporal"].get("window_size", 10),
            file_list=val_files, norm_factor=train_ds.norm_factor,
        )
        train_loader = DataLoader(train_ds, batch_size=self.train_cfg["optimization"]["batch_size"],
                                  shuffle=True, num_workers=4)

        self.setup_camera(train_ds.H, train_ds.W, train_ds.focal,
                          camera_pose=train_ds.get_camera_params())
        action_dim = train_ds.action_dim

        model = self._create_model(action_dim, train_ds.H, train_ds.W)
        n_params = sum(p.numel() for p in model.parameters())

        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_cfg["optimization"]["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.train_cfg["optimization"].get("scheduler_patience", 5))

        config_dict = {
            "model": "RGBDNeuralField",
            "params": n_params,
            "depth_weight": self.depth_weight,
            "guided_sampling": self.use_guided_sampling,
            "has_depth_data": has_depth_data,
        }
        exp_dir = self.create_experiment(
            os.path.join("train_log", "train_rgbd"), config_dict)
        model_dir = os.path.join(exp_dir, "model")
        vis_dir = os.path.join(exp_dir, "vis")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f">>> RGB-D Neural Field, {n_epochs} epochs")
        print(f"    Data: {data_dir}, Params: {n_params:,}")
        print(f"    Depth weight: {self.depth_weight}")
        print(f"    Log: {exp_dir}")
        print(f"{'='*60}")

        val_actions = val_ds.get_raw_actions(seq_id=0)

        def val_forward(val_seq):
            def fn(pts_chunk):
                return model(pts_chunk, val_seq)
            pts, _ = sample_stratified(self.rays_o, self.rays_d, self.near, self.far,
                                       self.n_samples, perturb=False)
            return self.render_points(fn, pts)

        best_val_loss = float("inf")

        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0
            epoch_depth_loss = 0
            depth_count = 0
            pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{n_epochs}")

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

                B = img_t.shape[0]
                H, W = self.H, self.W

                depth_input = None
                if has_depth_data and depth_t is not None:
                    depth_t_dev = depth_t.to(self.device)
                    depth_input = depth_t_dev.reshape(B, 1, H, W)

                sel, rays_o_sel, rays_d_sel = self.sample_fg_rays(img_t)

                if self.use_guided_sampling and has_depth_data:
                    pts_coarse, z_vals_coarse = sample_stratified(
                        rays_o_sel, rays_d_sel, self.near, self.far, 32)
                    with torch.no_grad():
                        raw_coarse = self.render_points(
                            lambda p: model(p, seq_t, depth_input), pts_coarse)
                        alpha = 1.0 - torch.exp(-torch.nn.functional.softplus(raw_coarse[..., 1]))
                        transmittance = torch.cumprod(
                            torch.cat([torch.ones_like(alpha[..., :1]),
                                       1.0 - alpha[..., :-1] + 1e-10], dim=-1), dim=-1)
                        weights_coarse = transmittance * alpha
                    pts, z_vals = sample_depth_guided(
                        rays_o_sel, rays_d_sel, z_vals_coarse, weights_coarse,
                        32, self.near, self.far)
                else:
                    pts, z_vals = sample_stratified(
                        rays_o_sel, rays_d_sel, self.near, self.far, self.n_samples)

                rgb_map, depth_pred = self.render_points_with_depth(
                    lambda p: model(p, seq_t, depth_input), pts, z_vals)
                pred_t = rgb_map.reshape(B, -1)
                loss_recon = torch.nn.functional.mse_loss(pred_t, img_t[:, sel])

                depth_input_t1 = None
                if has_depth_data and depth_t1 is not None:
                    depth_t1_dev = depth_t1.to(self.device)
                    depth_input_t1 = depth_t1_dev.reshape(B, 1, H, W)

                rgb_map2 = self.render_points(
                    lambda p: model(p, seq_t1, depth_input_t1), pts)
                pred_t1 = rgb_map2.reshape(B, -1)
                loss_recon_next = torch.nn.functional.mse_loss(pred_t1, img_t1[:, sel])

                loss_smooth = model.compute_smoothness(seq_t, seq_t1)

                loss = loss_recon + 0.5 * loss_recon_next + self.smooth_weight * loss_smooth

                if has_depth_data and depth_t is not None:
                    depth_gt = depth_t_dev.reshape(B, -1)[:, sel]
                    fg_mask = depth_gt > 0.01
                    if fg_mask.any():
                        loss_depth = torch.nn.functional.l1_loss(
                            depth_pred.reshape(B, -1)[fg_mask], depth_gt[fg_mask])
                        loss = loss + self.depth_weight * loss_depth
                        epoch_depth_loss += loss_depth.item()
                        depth_count += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                postfix = {'loss': f'{loss.item():.5f}'}
                if depth_count > 0:
                    postfix['d'] = f'{epoch_depth_loss/depth_count:.4f}'
                pbar.set_postfix(postfix)

            model.eval()
            val_loss = self.validate_and_gif(
                val_forward, val_ds, epoch, exp_dir, action_curves=val_actions)
            scheduler.step(val_loss)

            avg_train = epoch_loss / max(len(train_loader), 1)
            depth_str = f" | Depth: {epoch_depth_loss/max(depth_count,1):.5f}" if depth_count > 0 else ""
            print(f"  Epoch {epoch} | Train: {avg_train:.5f} | Val: {val_loss:.5f}{depth_str}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))

            if epoch % 10 == 0:
                torch.save(model.state_dict(),
                           os.path.join(model_dir, f"model_epoch_{epoch:03d}.pt"))

        np.savetxt(os.path.join(exp_dir, "action_norm_factor.txt"), [train_ds.norm_factor])
        print(f">>> Training done! Best val: {best_val_loss:.5f}")
