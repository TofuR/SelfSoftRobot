"""SDF Trainer — 3D 点云监督训练 TemporalSDFModel。

Loss 组合（来自 Chen 2022）:
  1. SDF constraint:   表面点 SDF=0
  2. Interior constraint: 内部点惩罚
  3. Normal constraint: 法向量一致性
  4. Grad constraint (Eikonal): SDF 梯度模=1
  5. Temporal smoothness: EMA 状态连续性
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.model_sdf import TemporalSDFModel
from src.data.dataset_sdf import SDFDataset
from src.config.params import load_config


def sdf_gradient(pred_sdf, coords):
    grad = torch.autograd.grad(
        outputs=pred_sdf,
        inputs=coords,
        grad_outputs=torch.ones_like(pred_sdf),
        create_graph=True,
    )[0]
    return grad


class SDFTrainer:
    """3D SDF 监督训练器。"""

    def __init__(self, device, w_sdf=3e3, w_inter=1e2, w_normal=1e2,
                 w_grad=5e1, w_smooth=1e-2):
        self.device = device
        self.w_sdf = w_sdf
        self.w_inter = w_inter
        self.w_normal = w_normal
        self.w_grad = w_grad
        self.w_smooth = w_smooth
        self.train_cfg = load_config("training")

    def _create_model(self, action_dim):
        temporal_cfg = self.train_cfg.get("temporal", {})
        return TemporalSDFModel(
            action_dim=action_dim,
            window_size=temporal_cfg.get("window_size", 20),
            n_scales=temporal_cfg.get("n_scales", 4),
            hidden_dim=temporal_cfg.get("hidden_dim", 128),
            sdf_hidden=256,
        ).to(self.device)

    def compute_loss(self, model, coords, action_window, gt_sdf, gt_normals):
        pred_sdf = model(coords, action_window)
        gt_sdf = gt_sdf.reshape(-1, 1)
        gt_normals = gt_normals.reshape(-1, 3)

        gradient = sdf_gradient(pred_sdf, coords)

        sdf_constraint = torch.where(
            gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
        loss_sdf = torch.abs(sdf_constraint).mean() * self.w_sdf

        inter_constraint = torch.where(
            gt_sdf != -1,
            torch.zeros_like(pred_sdf),
            torch.exp(-1e2 * torch.abs(pred_sdf)))
        loss_inter = inter_constraint.mean() * self.w_inter

        normal_constraint = torch.where(
            gt_sdf != -1,
            1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
            torch.zeros_like(gradient[..., :1]))
        loss_normal = normal_constraint.mean() * self.w_normal

        grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)
        loss_grad = grad_constraint.mean() * self.w_grad

        total = loss_sdf + loss_inter + loss_normal + loss_grad

        loss_dict = {
            'sdf': loss_sdf.item(),
            'inter': loss_inter.item(),
            'normal': loss_normal.item(),
            'eikonal': loss_grad.item(),
        }
        return total, loss_dict

    def train(self, data_dir="data/seq_rr_3d", n_epochs=500):
        train_ds = SDFDataset(
            data_dir, seq_len=self.train_cfg.get("temporal", {}).get("window_size", 20),
            n_surface=300, n_off_surface=300)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.train_cfg.get("optimization", {}).get("batch_size", 4),
            shuffle=True, num_workers=4)

        action_dim = train_ds.action_dim
        model = self._create_model(action_dim)
        n_params = sum(p.numel() for p in model.parameters())

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.train_cfg.get("optimization", {}).get("lr", 5e-5))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100000], gamma=0.5)

        log_dir = f"train_log/train_sdf/exp_{n_epochs}ep"
        os.makedirs(os.path.join(log_dir, "model"), exist_ok=True)

        print(f"\n{'='*60}")
        print(f">>> SDF 3D Supervised Training, {n_epochs} epochs")
        print(f"    Data: {data_dir}, Params: {n_params:,}")
        print(f"    Losses: sdf={self.w_sdf}, inter={self.w_inter}, "
              f"normal={self.w_normal}, eikonal={self.w_grad}")
        print(f"    Log: {log_dir}")
        print(f"{'='*60}")

        best_loss = float("inf")

        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0
            loss_sums = {}
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")
            for action_window, coords, gt_sdf, gt_normals in pbar:
                action_window = action_window.to(self.device)
                coords = coords.to(self.device).reshape(-1, 3).requires_grad_(True)
                gt_sdf = gt_sdf.to(self.device).reshape(-1)
                gt_normals = gt_normals.to(self.device).reshape(-1, 3)

                loss, loss_dict = self.compute_loss(
                    model, coords, action_window, gt_sdf, gt_normals)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                for k, v in loss_dict.items():
                    loss_sums[k] = loss_sums.get(k, 0) + v
                n_batches += 1

                avg_losses = {k: f"{v/n_batches:.4f}" for k, v in loss_sums.items()}
                pbar.set_postfix(avg_losses)

            avg_epoch = epoch_loss / max(n_batches, 1)

            if avg_epoch < best_loss:
                best_loss = avg_epoch
                torch.save(model.state_dict(),
                           os.path.join(log_dir, "model", "best_model.pt"))
                np.savetxt(os.path.join(log_dir, "model", "decays.txt"),
                           model.get_learned_decays())

            if epoch % 50 == 0:
                torch.save(model.state_dict(),
                           os.path.join(log_dir, "model", f"model_epoch_{epoch:04d}.pt"))

            loss_str = " | ".join(f"{k}:{v/n_batches:.2f}" for k, v in loss_sums.items())
            print(f"  Epoch {epoch} | Total: {avg_epoch:.2f} | {loss_str}")

        np.savetxt(os.path.join(log_dir, "action_norm_factor.txt"),
                   [train_ds.norm_factor])
        print(f"\n>>> Done! Best loss: {best_loss:.2f}")
