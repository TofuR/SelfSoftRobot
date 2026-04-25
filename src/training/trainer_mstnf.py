"""MSTNFTrainer — MSTNF 单阶段训练。"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import BaseTrainer
from src.models.model_mstnf import MSTNFModel
from src.utils.rendering import OM_rendering, sample_stratified
from src.data.dataset import SoftSequenceDataset


class MSTNFTrainer(BaseTrainer):
    def __init__(self, device):
        super().__init__(device)
        self.temp_cfg = self.train_cfg["temporal"]
        self.loss_cfg = self.train_cfg["loss_weights"]
        self.log_cfg = self.train_cfg["logging"]

    def _forward(self, rays_o, rays_d, model, action_window,
                 target_img=None, n_rays_sample=1024, fg_ratio=0.5):
        B, K, D = action_window.shape
        physics_state = model.encode_temporal(action_window)
        current_action = action_window[:, -1, :]

        if target_img is not None and n_rays_sample < rays_o.shape[0]:
            sel, rays_o_sel, rays_d_sel = self.sample_fg_rays(target_img, n_rays_sample, fg_ratio)
        else:
            sel = None
            rays_o_sel, rays_d_sel = rays_o, rays_d

        pts, _ = sample_stratified(rays_o_sel, rays_d_sel, self.near, self.far, self.n_samples)
        N_rays = pts.shape[0]

        state_exp = physics_state.unsqueeze(1).expand(-1, N_rays, -1).reshape(-1, physics_state.shape[-1])
        action_exp = current_action.unsqueeze(1).expand(-1, N_rays, -1).reshape(-1, D)
        pts_exp = pts.unsqueeze(0).expand(B, -1, -1, -1).reshape(-1, self.n_samples, 3)

        chunk_size = 4096
        raw_parts = []
        for i in range(0, pts_exp.shape[0], chunk_size):
            raw_parts.append(model.decode_spatial(
                pts_exp[i:i + chunk_size], state_exp[i:i + chunk_size], action_exp[i:i + chunk_size]))
        raw = torch.cat(raw_parts, dim=0).reshape(B, N_rays, self.n_samples, 2)

        rgb_map, _ = OM_rendering(raw.reshape(-1, self.n_samples, 2))
        return rgb_map.reshape(B, -1), sel

    def _val_forward(self, model, val_seq):
        physics_state = model.encode_temporal(val_seq)
        current_action = val_seq[:, -1, :]

        def fn(pts_chunk):
            N = pts_chunk.shape[0]
            return model.decode_spatial(
                pts_chunk, physics_state.expand(N, -1), current_action.expand(N, -1))

        img = self.render_full_image(fn)
        return torch.tensor(img, device=self.device, dtype=torch.float32).reshape(-1)

    def train(self, data_dir="data/sequence_data"):
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not all_files:
            raise FileNotFoundError(f"No data in {data_dir}")

        split = max(1, int(0.8 * len(all_files)))
        train_files, val_files = all_files[:split], all_files[split:]

        train_ds = SoftSequenceDataset(
            data_dir, seq_len=self.temp_cfg["window_size"],
            file_list=train_files, return_pairs=True)
        val_ds = SoftSequenceDataset(
            data_dir, seq_len=self.temp_cfg["window_size"],
            file_list=val_files, norm_factor=train_ds.norm_factor)
        train_loader = DataLoader(train_ds, batch_size=self.opt_cfg["batch_size"],
                                  shuffle=True, num_workers=4)

        self.setup_camera(train_ds.H, train_ds.W, train_ds.focal)
        action_dim = train_ds.action_dim

        model = MSTNFModel(
            action_dim=action_dim,
            window_size=self.temp_cfg["window_size"],
            n_scales=self.temp_cfg["n_scales"],
            hidden_dim=self.temp_cfg["hidden_dim"]).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.opt_cfg["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.opt_cfg["scheduler_patience"])

        w_recon = self.loss_cfg["recon_current"]
        w_recon_next = self.loss_cfg["recon_next"]
        w_smooth = self.loss_cfg["smoothness"]
        n_epochs = self.opt_cfg["n_epochs"]
        save_rate = self.log_cfg["save_rate"]

        config_dict = {
            "model": "MSTNFModel", "action_dim": action_dim, "data": data_dir,
            "window_size": self.temp_cfg["window_size"],
            "n_scales": self.temp_cfg["n_scales"],
            "hidden_dim": self.temp_cfg["hidden_dim"],
            "training": {"lr": self.opt_cfg["lr"], "batch_size": self.opt_cfg["batch_size"],
                         "n_epochs": n_epochs},
            "loss_weights": self.loss_cfg,
        }
        LOG_DIR = self.create_experiment(os.path.join("train_log", "train_mstnf"), config_dict)

        print(f"\n>>> MSTNF Training: {n_epochs} epochs, {len(train_ds)} samples, data: {data_dir}")

        val_actions = val_ds.get_raw_actions(seq_id=0)
        best_val_loss = float("inf")
        global_step = 0

        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")

            for seq_t, seq_t1, img_t, img_t1 in pbar:
                seq_t = seq_t.to(self.device)
                seq_t1 = seq_t1.to(self.device)
                img_t = img_t.to(self.device)
                img_t1 = img_t1.to(self.device)

                pred_t, idx = self._forward(self.rays_o, self.rays_d, model, seq_t, target_img=img_t)
                gt = img_t[:, idx] if idx is not None else img_t
                loss_recon = torch.nn.functional.mse_loss(pred_t, gt)

                pred_t1, _ = self._forward(self.rays_o, self.rays_d, model, seq_t1, target_img=img_t1)
                gt1 = img_t1[:, idx] if idx is not None else img_t1
                loss_recon_next = torch.nn.functional.mse_loss(pred_t1, gt1)

                loss_smooth = model.compute_smoothness(seq_t, seq_t1)
                loss = w_recon * loss_recon + w_recon_next * loss_recon_next + w_smooth * loss_smooth

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.5f}'})
                global_step += 1

                if global_step % save_rate == 0:
                    torch.save(model.state_dict(),
                               os.path.join(LOG_DIR, "model", f"model_{global_step:05d}.pt"))

            model.eval()
            val_forward_fn = lambda vs: self._val_forward(model, vs)
            val_loss = self.validate_and_gif(val_forward_fn, val_ds, epoch, LOG_DIR, val_actions)
            scheduler.step(val_loss)

            decays = model.get_learned_decays()
            avg_train = epoch_loss / max(len(train_loader), 1)
            print(f"Epoch {epoch} | Train: {avg_train:.5f} | Val: {val_loss:.5f} | "
                  f"Decays: [{', '.join(f'{d:.3f}' for d in decays)}]")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(LOG_DIR, "model", "best_model.pt"))

        np.savetxt(os.path.join(LOG_DIR, "action_norm_factor.txt"), [train_ds.norm_factor])
        np.savetxt(os.path.join(LOG_DIR, "learned_decays.txt"), model.get_learned_decays())
        print("Training Finished.")
