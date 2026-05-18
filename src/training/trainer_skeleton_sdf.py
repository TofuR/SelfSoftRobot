"""SkeletonSDF 两阶段训练器。

Phase 1: 骨架预热 — 只训练时序编码器 + 骨架头
Phase 2: 联合训练 — 骨架 + SDF (含 Eikonal + Normal loss)

Loss 组成:
  Phase 1: skeleton L2 (多尺度) + 曲线平滑正则
  Phase 2: Phase 1 losses + SDF L1 + Normal + Eikonal (共享梯度)
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset_skeleton_sdf import SkeletonSDFDataset
from src.models.model_skeleton_sdf import SkeletonSDFModel
from src.utils.experiment import create_experiment


def skeleton_smoothness(skeleton):
    """骨架二阶差分正则 (B, N, 3) -> scalar。"""
    if skeleton.shape[1] < 3:
        return torch.tensor(0.0, device=skeleton.device)
    return ((skeleton[:, 2:] - 2 * skeleton[:, 1:-1] + skeleton[:, :-2]) ** 2).mean()


class SkeletonSDFTrainer:

    def __init__(self, device, config,
                 skeleton_mode="bspline", rod_radius=0.015,
                 phase1_epochs=50):
        self.device = device
        self.config = config
        self.skeleton_mode = skeleton_mode
        self.rod_radius = rod_radius
        self.phase1_epochs = phase1_epochs

        opt = config.get("optimization", {})
        temporal = config.get("temporal", {})
        ms = config.get("ms_scnf", {})
        sdf = config.get("sdf", {})

        self.lr = opt.get("lr", 5e-5)
        self.n_epochs = opt.get("n_epochs", 500)
        self.window_size = temporal.get("window_size", 20)
        self.n_scales = temporal.get("n_scales", 4)
        self.hidden_dim = temporal.get("hidden_dim", 128)
        self.batch_size = opt.get("batch_size", 1)

        # 骨架 loss 权重
        self.w_skel_fine = ms.get("w_skeleton_fine", 1.0)
        self.w_skel_medium = ms.get("w_skeleton_medium", 0.3)
        self.w_skel_coarse = ms.get("w_skeleton_coarse", 0.1)
        self.w_smooth = ms.get("w_smooth", 0.01)

        # SDF loss 权重
        self.w_sdf = sdf.get("w_sdf", 3e3)
        self.w_normal = sdf.get("w_normal", 10.0)
        self.w_eikonal = sdf.get("w_eikonal", 50.0)

        # SDF 采样参数
        self.n_surface = sdf.get("n_surface", 500)
        self.n_near = sdf.get("n_near_surface", 500)
        self.n_off = sdf.get("n_off_surface", 500)

    # ── 模型构造 ──────────────────────────────────────────────────────────

    def _create_model(self, action_dim):
        return SkeletonSDFModel(
            action_dim=action_dim,
            window_size=self.window_size,
            n_scales=self.n_scales,
            hidden_dim=self.hidden_dim,
            skeleton_mode=self.skeleton_mode,
            rod_radius=self.rod_radius,
        ).to(self.device)

    # ── Loss 计算 ─────────────────────────────────────────────────────────

    def _skeleton_losses(self, model, action_window, gt_positions):
        """多尺度骨架 L2 + 平滑正则。"""
        pred_dict = model.predict_skeleton(action_window)
        skel = model.compute_skeleton_loss(pred_dict, gt_positions.unsqueeze(0))

        loss = (skel['fine'] * self.w_skel_fine +
                skel['medium'] * self.w_skel_medium +
                skel['coarse'] * self.w_skel_coarse)
        smooth = skeleton_smoothness(pred_dict['fine']) * self.w_smooth

        return loss + smooth, {
            'skel': loss.item(),
            'smooth': smooth.item(),
        }

    def _sdf_losses(self, model, coords, action_window, gt_sdf, gt_normals):
        """SDF L1 + Normal + Eikonal (共享梯度，只算一次 autograd.grad)。

        Args:
            coords: (M, 3) 需要 requires_grad
            action_window: (B, K, D)
            gt_sdf: (M,)
            gt_normals: (M, 3)

        Returns:
            (total_loss, loss_dict)
        """
        query = coords.unsqueeze(0)                          # (1, M, 3)
        pred_sdf = model(query, action_window).squeeze(-1)   # (M,)

        # SDF L1 回归
        loss_sdf = torch.abs(pred_sdf - gt_sdf).mean() * self.w_sdf

        # 共享梯度: 一次 autograd.grad 服务 Normal + Eikonal
        gradient = torch.autograd.grad(
            pred_sdf.sum(), coords, create_graph=True,
        )[0]  # (M, 3)

        # Normal loss (仅表面点)
        loss_normal = torch.tensor(0.0, device=self.device)
        is_surface = (gt_sdf.abs() < 1e-6).float()
        n_surf = is_surface.sum()
        if n_surf > 0 and gt_normals.abs().sum() > 0:
            cos_sim = F.cosine_similarity(gradient, gt_normals, dim=-1)
            loss_normal = (is_surface * (1 - cos_sim)).sum() / (n_surf + 1e-8) * self.w_normal

        # Eikonal loss (所有查询点)
        loss_eikonal = ((gradient.norm(dim=-1) - 1) ** 2).mean() * self.w_eikonal

        total = loss_sdf + loss_normal + loss_eikonal
        return total, {
            'sdf': loss_sdf.item(),
            'normal': loss_normal.item(),
            'eikonal': loss_eikonal.item(),
        }

    # ── 数据集 ────────────────────────────────────────────────────────────

    def _make_dataset(self, data_dir):
        return SkeletonSDFDataset(
            data_dir, seq_len=self.window_size,
            n_surface=self.n_surface,
            n_near_surface=self.n_near,
            n_off_surface=self.n_off,
        )

    # ── Phase 1: 骨架预热 ─────────────────────────────────────────────────

    def train_phase1(self, data_dir, exp_dir=None):
        """Phase 1: 只训练骨架头，冻结 SIREN 残差网络。

        Returns:
            (exp_dir, skeleton_weights_path)
        """
        ds = self._make_dataset(data_dir)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

        model = self._create_model(ds.action_dim)

        # 冻结 SIREN 残差网络
        if model.sdf_residual:
            for p in model.sdf_net.parameters():
                p.requires_grad = False
            for p in model.state_proj.parameters():
                p.requires_grad = False

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        if exp_dir is None:
            log_config = self._log_config(model, ds, data_dir)
            exp_dir = create_experiment("train_log/train_skeleton_sdf", log_config)
        phase1_dir = os.path.join(exp_dir, "phase1")
        os.makedirs(os.path.join(phase1_dir, "model"), exist_ok=True)

        best_loss = float("inf")
        print(f"\n{'='*60}")
        print(f">>> Phase 1: 骨架预热 ({self.skeleton_mode})")
        print(f"    Epochs: {self.phase1_epochs}, Params: {self._count_params(model):,}")
        print(f"    Data: {data_dir} ({len(ds)} samples)")
        print(f"    Log: {exp_dir}")
        print(f"{'='*60}")

        for epoch in range(1, self.phase1_epochs + 1):
            model.train()
            epoch_loss, n_batches = 0.0, 0
            loss_sums = {}

            for action_window, _, _, _, gt_positions in tqdm(
                    loader, desc=f"P1 E{epoch}/{self.phase1_epochs}"):
                action_window = action_window.to(self.device)
                gt_positions = gt_positions.to(self.device).squeeze(0)

                loss, losses = self._skeleton_losses(model, action_window, gt_positions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                for k, v in losses.items():
                    loss_sums[k] = loss_sums.get(k, 0) + v

            self._print_epoch("Phase1", epoch, epoch_loss, n_batches, loss_sums)

            avg = epoch_loss / max(n_batches, 1)
            if avg < best_loss:
                best_loss = avg
                self._save_skeleton(model, os.path.join(phase1_dir, "model", "skeleton_best.pt"))

        final_path = os.path.join(phase1_dir, "model", "skeleton_final.pt")
        self._save_skeleton(model, final_path)
        return exp_dir, final_path

    # ── Phase 2: 联合训练 ─────────────────────────────────────────────────

    def train_phase2(self, data_dir, exp_dir, skeleton_path):
        """Phase 2: 全参数训练，骨架 + SDF + Eikonal + Normal。"""
        ds = self._make_dataset(data_dir)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

        model = self._create_model(ds.action_dim)

        # 加载 Phase 1 骨架权重
        ckpt = torch.load(skeleton_path, map_location=self.device)
        model.temporal.load_state_dict(ckpt['temporal'])
        model.skeleton_head.load_state_dict(ckpt['skeleton_head'])

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr * 0.5)
        phase2_epochs = self.n_epochs - self.phase1_epochs
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[phase2_epochs // 2], gamma=0.5)

        phase2_dir = os.path.join(exp_dir, "phase2")
        os.makedirs(os.path.join(phase2_dir, "model"), exist_ok=True)

        best_loss = float("inf")
        print(f"\n{'='*60}")
        print(f">>> Phase 2: 联合训练 (骨架 + SDF)")
        print(f"    Epochs: {phase2_epochs}, Loaded: {skeleton_path}")
        print(f"    Loss: skel={self.w_skel_fine}/{self.w_skel_medium}/{self.w_skel_coarse}, "
              f"sdf={self.w_sdf}, normal={self.w_normal}, eikonal={self.w_eikonal}")
        print(f"{'='*60}")

        for epoch in range(1, phase2_epochs + 1):
            model.train()
            epoch_loss, n_batches = 0.0, 0
            loss_sums = {}

            for action_window, coords, gt_sdf, gt_normals, gt_positions in tqdm(
                    loader, desc=f"P2 E{epoch}/{phase2_epochs}"):
                action_window = action_window.to(self.device)
                coords = coords.to(self.device).squeeze(0).requires_grad_(True)
                gt_sdf = gt_sdf.to(self.device).squeeze(0)
                gt_normals = gt_normals.to(self.device).squeeze(0)
                gt_positions = gt_positions.to(self.device).squeeze(0)

                # 骨架 loss
                loss_skel, losses_skel = self._skeleton_losses(model, action_window, gt_positions)

                # SDF loss (共享梯度)
                loss_sdf, losses_sdf = self._sdf_losses(
                    model, coords, action_window, gt_sdf, gt_normals)

                total = loss_skel + loss_sdf

                optimizer.zero_grad()
                total.backward()
                optimizer.step()

                epoch_loss += total.item()
                n_batches += 1
                for k, v in {**losses_skel, **losses_sdf}.items():
                    loss_sums[k] = loss_sums.get(k, 0) + v

            scheduler.step()
            self._print_epoch("Phase2", epoch, epoch_loss, n_batches, loss_sums)

            avg = epoch_loss / max(n_batches, 1)
            if avg < best_loss:
                best_loss = avg
                torch.save(model.state_dict(),
                           os.path.join(phase2_dir, "model", "best_model.pt"))

            if epoch % 50 == 0:
                torch.save(model.state_dict(),
                           os.path.join(phase2_dir, "model", f"model_epoch_{epoch:04d}.pt"))

        # 保存归一化因子
        np.savetxt(os.path.join(phase2_dir, "action_norm_factor.txt"), [ds.norm_factor])
        print(f"\n>>> Phase 2 done! Best loss: {best_loss:.4f}")

    # ── 统一入口 ──────────────────────────────────────────────────────────

    def train(self, data_dir):
        """Phase 1 -> Phase 2 统一训练入口。"""
        exp_dir, skel_path = self.train_phase1(data_dir)
        self.train_phase2(data_dir, exp_dir, skel_path)

    # ── 工具方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _save_skeleton(model, path):
        torch.save({
            'temporal': model.temporal.state_dict(),
            'skeleton_head': model.skeleton_head.state_dict(),
        }, path)

    @staticmethod
    def _count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _log_config(self, model, dataset, data_dir):
        return {
            **self.config,
            "data_dir": data_dir,
            "skeleton_mode": self.skeleton_mode,
            "rod_radius": self.rod_radius,
            "phase1_epochs": self.phase1_epochs,
            "action_dim": dataset.action_dim,
            "n_params": self._count_params(model),
            "w_skel_fine": self.w_skel_fine,
            "w_skel_medium": self.w_skel_medium,
            "w_skel_coarse": self.w_skel_coarse,
            "w_smooth": self.w_smooth,
            "w_sdf": self.w_sdf,
            "w_normal": self.w_normal,
            "w_eikonal": self.w_eikonal,
        }

    @staticmethod
    def _print_epoch(phase, epoch, epoch_loss, n_batches, loss_sums):
        avg = epoch_loss / max(n_batches, 1)
        parts = " | ".join(f"{k}:{v/n_batches:.4f}" for k, v in loss_sums.items())
        print(f"  [{phase}] Epoch {epoch} | Total: {avg:.4f} | {parts}")
