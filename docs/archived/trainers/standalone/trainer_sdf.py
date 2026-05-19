"""SDF Trainer — 3D 点云监督训练 TemporalSDFModel。

Loss 组合:
  1. SDF regression:  |pred_sdf - gt_sdf| L1 回归（表面点=0，off-surface 点=真实距离）
  2. Normal constraint: 法向量一致性（仅表面点）
  3. Grad constraint (Eikonal): SDF 梯度模=1
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.model_sdf import TemporalSDFModel
from src.data.dataset_sdf import SDFDataset
from src.utils.experiment import create_experiment


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

    def __init__(self, device, config):
        self.device = device
        self.config = config
        sdf_cfg = config.get("sdf", {})
        self.w_sdf = sdf_cfg.get("w_sdf", 3e3)
        self.w_normal = sdf_cfg.get("w_normal", 1e2)
        self.w_grad = sdf_cfg.get("w_grad", 5e1)

    def _create_model(self, action_dim):
        temporal_cfg = self.config.get("temporal", {})
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

        # SDF L1 回归: 所有点（表面=0, off-surface=真实有符号距离）
        loss_sdf = torch.abs(pred_sdf - gt_sdf).mean() * self.w_sdf

        # 法向量 loss: 仅表面点 (gt_sdf == 0)
        is_surface = (gt_sdf.abs() < 1e-6).float()
        if is_surface.sum() > 0:
            cos_sim = F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None]
            loss_normal = (is_surface * (1 - cos_sim)).sum() / (is_surface.sum() + 1e-8) * self.w_normal
        else:
            loss_normal = torch.tensor(0.0, device=self.device)

        # Eikonal: SIREN w0=30 使梯度被放大 ~27000 倍，与 eikonal 不兼容，默认关闭
        if self.w_grad > 0:
            loss_grad = ((gradient.norm(dim=-1) - 1) ** 2).mean() * self.w_grad
        else:
            loss_grad = torch.tensor(0.0, device=self.device)

        total = loss_sdf + loss_normal + loss_grad

        loss_dict = {
            'sdf': loss_sdf.item(),
            'normal': loss_normal.item(),
            'eikonal': loss_grad.item(),
        }
        return total, loss_dict

    def train(self, data_dir="data/seq_rr_3d"):
        opt_cfg = self.config.get("optimization", {})
        temporal_cfg = self.config.get("temporal", {})
        n_epochs = opt_cfg.get("n_epochs", 500)
        lr = opt_cfg.get("lr", 5e-5)
        window_size = temporal_cfg.get("window_size", 20)

        train_ds = SDFDataset(
            data_dir, seq_len=window_size,
            n_surface=self.config.get("sdf", {}).get("n_surface", 300),
            n_near_surface=self.config.get("sdf", {}).get("n_near_surface", 200),
            n_off_surface=self.config.get("sdf", {}).get("n_off_surface", 200))
        train_loader = DataLoader(
            train_ds,
            batch_size=1,
            shuffle=True, num_workers=4)

        action_dim = train_ds.action_dim
        model = self._create_model(action_dim)
        n_params = sum(p.numel() for p in model.parameters())

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100000], gamma=0.5)

        log_config = dict(self.config)
        log_config.update({
            "data_dir": data_dir,
            "n_params": n_params,
            "action_dim": action_dim,
        })
        log_dir = create_experiment("train_log/train_sdf", log_config)

        print(f"\n{'='*60}")
        print(f">>> SDF 3D Supervised Training, {n_epochs} epochs")
        print(f"    Data: {data_dir}, Params: {n_params:,}")
        print(f"    LR: {lr}, Window: {window_size}")
        print(f"    Losses: sdf={self.w_sdf}, "
              f"normal={self.w_normal}, eikonal={self.w_grad}")
        print(f"    Log: {log_dir}")
        print(f"{'='*60}")

        # 初始诊断：检查模型输出和梯度范围
        with torch.no_grad():
            test_pts = torch.randn(100, 3, device=self.device)
            test_aw = torch.randn(1, 20, train_ds.action_dim, device=self.device) * 0.1
            test_out = model(test_pts, test_aw)
            print(f"  [诊断] 初始输出范围: [{test_out.min():.6f}, {test_out.max():.6f}]")
            print("  [诊断] GT SDF 范围: ~[-0.01, 0.6] (归一化坐标)")

        best_loss = float("inf")

        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0
            loss_sums = {}
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")
            for action_window, coords, gt_sdf, gt_normals in pbar:
                action_window = action_window.to(self.device)
                coords = coords.to(self.device).squeeze(0).requires_grad_(True)
                gt_sdf = gt_sdf.to(self.device).squeeze(0)
                gt_normals = gt_normals.to(self.device).squeeze(0)

                loss, loss_dict = self.compute_loss(
                    model, coords, action_window, gt_sdf, gt_normals)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                for k, v in loss_dict.items():
                    loss_sums[k] = loss_sums.get(k, 0) + v
                n_batches += 1

                avg_losses = {k: f"{v/n_batches:.4f}" for k, v in loss_sums.items()}
                pbar.set_postfix(avg_losses)

            avg_epoch = epoch_loss / max(n_batches, 1)
            scheduler.step()

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
