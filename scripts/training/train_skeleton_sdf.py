"""train_skeleton_sdf.py — 方案 B: 混合 Bezier 骨架 + SDF 截面 训练入口。

方案 B 核心思路:
  1. 参数化骨架 (Bezier/B-spline/Fourier) → 保证拓扑连通，不会断裂
  2. 管状 SDF 先验 (dist_to_skeleton - radius) + SIREN 残差 → 完整 3D 形状
  3. 两阶段训练: Phase 1 骨架预热, Phase 2 联合

训练信号:
  - 骨架 loss: 多尺度 (coarse/medium/fine) L2 + 曲线平滑正则
  - SDF loss: L1 回归 (表面=0, off-surface=真实距离)
  - Normal loss: 表面法向量一致性
  - 渲染 loss: 体渲染一致性 (可选)

用法:
    # 默认: GPU 1, bspline 骨架, 两阶段训练
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_skeleton_sdf.py

    # 指定骨架模式
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_skeleton_sdf.py --skeleton_mode fourier

    # 覆盖 loss 权重
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_skeleton_sdf.py \
        --w_skeleton_fine 1.0 --w_sdf 3000 --w_smooth 0.01
"""

import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config.params import load_config
from src.models.model_skeleton_sdf import SkeletonSDFModel
from src.models.skeleton_heads import downsample_skeleton
from src.utils.experiment import create_experiment


# ── Dataset ──────────────────────────────────────────────────────────────


class SkeletonSDFDataset(Dataset):
    """为 SkeletonSDF 训练设计的数据集。

    与 SDFDataset 的区别:
      - 额外返回 GT positions (3, 31) 用于骨架监督
      - 保留 SDF 采样点 + 法向量

    每个样本: (action_window, coords, gt_sdf, gt_normals, gt_positions)
    """

    def __init__(self, data_dir, seq_len=20, n_surface=500,
                 n_near_surface=500, n_off_surface=500):
        import glob

        self.seq_len = seq_len
        self.n_surface = n_surface
        self.n_near_surface = n_near_surface
        self.n_off_surface = n_off_surface
        self.samples = []
        self.data_cache = []

        file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not file_list:
            raise FileNotFoundError(f"No .npz files in {data_dir}")

        # 坐标归一化参数
        all_pos = []
        for f in file_list:
            d = np.load(f)
            if 'positions' in d:
                all_pos.append(d['positions'])
        if all_pos:
            all_pos_flat = np.concatenate(all_pos, axis=0)
            pos_min = all_pos_flat.reshape(3, -1).min(axis=1)
            pos_max = all_pos_flat.reshape(3, -1).max(axis=1)
            self.coord_center = ((pos_min + pos_max) / 2).astype(np.float32)
            self.coord_scale = float((pos_max - pos_min).max() / 2 * 1.1)
        else:
            self.coord_center = np.zeros(3, dtype=np.float32)
            self.coord_scale = 1.0

        # 动作归一化
        all_acts = []
        for f in file_list:
            d = np.load(f)
            if 'actions' in d:
                all_acts.append(d['actions'])
        self.norm_factor = np.max(np.abs(np.concatenate(all_acts))) if all_acts else 1.0

        # 缓存数据
        self.action_dim = None
        for f_path in file_list:
            raw = np.load(f_path)
            actions = raw['actions'] / self.norm_factor
            if 'positions' not in raw:
                continue
            positions = raw['positions'].astype(np.float32)  # (T, 3, 31)
            radii = raw['radii'].astype(np.float32) if 'radii' in raw else None
            if self.action_dim is None:
                self.action_dim = actions.shape[1]
            self.data_cache.append({
                'actions': actions,
                'positions': positions,
                'radii': radii,
                'length': len(positions),
            })

        # 构建 sample index: (seq_id, timestep)
        for seq_id, item in enumerate(self.data_cache):
            T = item['length']
            for t in range(self.seq_len - 1, T):
                self.samples.append((seq_id, t))

        print(f"SkeletonSDFDataset: {len(self.samples)} samples, "
              f"action_dim={self.action_dim}, n_seqs={len(self.data_cache)}")

    def _get_action_window(self, data, t):
        start = t - self.seq_len + 1
        end = t + 1
        if start >= 0:
            return data['actions'][start:end].copy()
        pad = np.zeros((-start, self.action_dim), dtype=data['actions'].dtype)
        return np.concatenate([pad, data['actions'][0:end]], axis=0)

    def _normalize(self, coords):
        """(3, M) -> 归一化到 [-1, 1]^3"""
        return (coords - self.coord_center[:, None]) / self.coord_scale

    def _compute_sdf_and_normals(self, positions, radii):
        """生成 SDF 训练点: on-surface + near-surface + off-surface。

        Args:
            positions: (3, N) 杆体节点坐标
            radii: (N-1,) 各段半径
        """
        N = positions.shape[1]
        avg_radius = float(np.mean(radii)) if radii is not None else 0.015

        # On-surface: 杆体表面采样, SDF = 0
        n_surf = self.n_surface
        seg_idx = np.random.randint(0, N - 1, size=n_surf)
        t_param = np.random.rand(n_surf)
        axis_pts = positions[:, seg_idx] * (1 - t_param) + \
                   positions[:, np.minimum(seg_idx + 1, N - 1)] * t_param
        theta = np.random.rand(n_surf) * 2 * np.pi

        # 逐段构建法向量 (tangent -> perp1, perp2)
        seg_dir = positions[:, np.minimum(seg_idx + 1, N - 1)] - positions[:, seg_idx]
        seg_dir_norm = np.linalg.norm(seg_dir, axis=0, keepdims=True) + 1e-8
        tangent = seg_dir / seg_dir_norm
        # tangent: (3, n_surf), 每列是一个单位切向量
        surf_normals = np.zeros((3, n_surf), dtype=np.float32)
        for i in range(n_surf):
            t_vec = tangent[:, i]
            if abs(t_vec[1]) < 0.99:
                ref = np.array([0.0, 1.0, 0.0])
            else:
                ref = np.array([1.0, 0.0, 0.0])
            p1 = np.cross(t_vec, ref)
            p1 /= np.linalg.norm(p1) + 1e-8
            p2 = np.cross(t_vec, p1)
            surf_normals[:, i] = np.cos(theta[i]) * p1 + np.sin(theta[i]) * p2

        surface_pts = axis_pts + avg_radius * surf_normals
        sdf_surface = np.zeros(n_surf, dtype=np.float32)

        # Near-surface
        n_near = self.n_near_surface
        seg_idx_ns = np.random.randint(0, N - 1, size=n_near)
        t_param_ns = np.random.rand(n_near)
        axis_pts_ns = positions[:, seg_idx_ns] * (1 - t_param_ns) + \
                      positions[:, np.minimum(seg_idx_ns + 1, N - 1)] * t_param_ns
        theta_ns = np.random.rand(n_near) * 2 * np.pi
        # 近表面法向量: 逐段构建
        dir_ns = np.zeros((3, n_near), dtype=np.float32)
        seg_dir_ns = positions[:, np.minimum(seg_idx_ns + 1, N - 1)] - positions[:, seg_idx_ns]
        seg_dir_ns_norm = np.linalg.norm(seg_dir_ns, axis=0, keepdims=True) + 1e-8
        tangent_ns = seg_dir_ns / seg_dir_ns_norm
        for i in range(n_near):
            t_vec = tangent_ns[:, i]
            if abs(t_vec[1]) < 0.99:
                ref = np.array([0.0, 1.0, 0.0])
            else:
                ref = np.array([1.0, 0.0, 0.0])
            p1 = np.cross(t_vec, ref)
            p1 /= np.linalg.norm(p1) + 1e-8
            p2 = np.cross(t_vec, p1)
            dir_ns[:, i] = np.cos(theta_ns[i]) * p1 + np.sin(theta_ns[i]) * p2

        offset = (np.random.rand(n_near) * 6 - 3) * avg_radius
        near_pts = axis_pts_ns + (avg_radius + offset) * dir_ns

        sdf_near, normals_near = self._sdf_to_rod(near_pts, positions, avg_radius)

        # Off-surface: 在 positions bounding box 附近均匀采样（原始坐标空间）
        n_off = self.n_off_surface
        pos_min = positions.min(axis=1, keepdims=True) - avg_radius * 5
        pos_max = positions.max(axis=1, keepdims=True) + avg_radius * 5
        off_pts = np.random.uniform(pos_min, pos_max, size=(3, n_off)).astype(np.float32)
        sdf_off, normals_off = self._sdf_to_rod(off_pts, positions, avg_radius)

        # 合并 — 全部在原始坐标空间
        coords = np.concatenate([surface_pts, near_pts, off_pts], axis=1)
        sdf = np.concatenate([sdf_surface, sdf_near, sdf_off])
        normals_all = np.concatenate([surf_normals, normals_near, normals_off], axis=1)

        return coords.T, sdf, normals_all.T

    @staticmethod
    def _sdf_to_rod(points, positions, avg_radius):
        """点到杆体的精确 SDF。points/positions: (3, M)/(3, N)。"""
        M, N = points.shape[1], positions.shape[1]
        min_dist = np.full(M, 1e6, dtype=np.float32)
        normals = np.zeros((3, M), dtype=np.float32)

        for i in range(N - 1):
            seg_s = positions[:, i]
            seg_e = positions[:, i + 1]
            seg_vec = seg_e - seg_s
            seg_len = np.linalg.norm(seg_vec)
            if seg_len < 1e-8:
                continue
            seg_dir = seg_vec / seg_len
            v = points - seg_s[:, None]
            t_proj = np.clip(np.dot(seg_dir, v), 0, seg_len)
            closest = seg_s[:, None] + seg_dir[:, None] * t_proj[None, :]
            diff = points - closest
            dist = np.linalg.norm(diff, axis=0)
            closer = dist < min_dist
            min_dist[closer] = dist[closer]
            norm = np.linalg.norm(diff[:, closer], axis=0, keepdims=True) + 1e-8
            normals[:, closer] = diff[:, closer] / norm

        return min_dist - avg_radius, normals

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_id, t = self.samples[idx]
        data = self.data_cache[seq_id]

        action_window = self._get_action_window(data, t)
        positions = data['positions'][t]  # (3, 31)
        radii = data['radii'][t] if data['radii'] is not None else None

        coords, sdf, normals = self._compute_sdf_and_normals(positions, radii)

        # positions 转置为 (31, 3) 用于骨架 loss
        gt_positions = positions.T.copy()  # (31, 3)

        return (
            torch.from_numpy(action_window).float(),
            torch.from_numpy(coords).float(),        # (M, 3) 归一化坐标
            torch.from_numpy(sdf).float(),            # (M,)
            torch.from_numpy(normals).float(),         # (M, 3)
            torch.from_numpy(gt_positions).float(),    # (31, 3) 原始坐标
        )


# ── Training ─────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="SkeletonSDF Training (方案 B)")
    parser.add_argument("--data_dir", type=str, default="data/seq_rz_c2_sk")
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--phase1_epochs", type=int, default=50,
                        help="Phase 1: 骨架预热 epoch 数")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--skeleton_mode", type=str, default=None,
                        choices=["point", "fourier", "bspline", "catmullrom"])
    parser.add_argument("--w_sdf", type=float, default=None)
    parser.add_argument("--w_skeleton_fine", type=float, default=1.0)
    parser.add_argument("--w_skeleton_medium", type=float, default=0.3)
    parser.add_argument("--w_skeleton_coarse", type=float, default=0.1)
    parser.add_argument("--w_smooth", type=float, default=0.01,
                        help="骨架曲线平滑正则权重 (二阶差分)")
    parser.add_argument("--w_normal", type=float, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--n_surface", type=int, default=None)
    parser.add_argument("--n_near_surface", type=int, default=None)
    parser.add_argument("--n_off_surface", type=int, default=None)
    parser.add_argument("--rod_radius", type=float, default=0.015)
    return parser.parse_args()


def resolve_config(defaults, overrides):
    cfg = dict(defaults)
    for key, val in overrides.items():
        if val is not None:
            parts = key.split('.')
            d = cfg
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = val
    return cfg


def skeleton_smoothness(skeleton):
    """骨架二阶差分正则 (B, N, 3) -> scalar。"""
    if skeleton.shape[1] < 3:
        return torch.tensor(0.0, device=skeleton.device)
    return ((skeleton[:, 2:] - 2 * skeleton[:, 1:-1] + skeleton[:, :-2]) ** 2).mean()


def train():
    args = parse_args()
    defaults = load_config("training")
    config = resolve_config(defaults, {
        "optimization.lr": args.lr,
        "optimization.n_epochs": args.n_epochs,
        "temporal.window_size": args.window_size,
        "sdf.w_sdf": args.w_sdf,
        "sdf.w_normal": args.w_normal,
        "sdf.n_surface": args.n_surface,
        "sdf.n_near_surface": args.n_near_surface,
        "sdf.n_off_surface": args.n_off_surface,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt_cfg = config.get("optimization", {})
    temporal_cfg = config.get("temporal", {})
    sdf_cfg = config.get("sdf", {})
    ms_scnf_cfg = config.get("ms_scnf", {})

    total_epochs = opt_cfg.get("n_epochs", 500)
    phase1_epochs = args.phase1_epochs
    lr = opt_cfg.get("lr", 5e-5)
    window_size = temporal_cfg.get("window_size", 20)
    skeleton_mode = args.skeleton_mode or ms_scnf_cfg.get("skeleton_mode", "bspline")

    w_sdf = sdf_cfg.get("w_sdf", 3e3)
    w_normal = sdf_cfg.get("w_normal", 10.0)
    w_skel_fine = args.w_skeleton_fine
    w_skel_medium = args.w_skeleton_medium
    w_skel_coarse = args.w_skeleton_coarse
    w_smooth = args.w_smooth

    # Dataset
    train_ds = SkeletonSDFDataset(
        args.data_dir, seq_len=window_size,
        n_surface=sdf_cfg.get("n_surface", 500),
        n_near_surface=sdf_cfg.get("n_near_surface", 500),
        n_off_surface=sdf_cfg.get("n_off_surface", 500),
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4,
                              pin_memory=True)

    # Model
    model = SkeletonSDFModel(
        action_dim=train_ds.action_dim,
        window_size=window_size,
        n_scales=temporal_cfg.get("n_scales", 4),
        hidden_dim=temporal_cfg.get("hidden_dim", 128),
        skeleton_mode=skeleton_mode,
        rod_radius=args.rod_radius,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[total_epochs // 2], gamma=0.5)

    log_config = dict(config)
    log_config.update({
        "data_dir": args.data_dir,
        "n_params": n_params,
        "action_dim": train_ds.action_dim,
        "skeleton_mode": skeleton_mode,
        "rod_radius": args.rod_radius,
        "phase1_epochs": phase1_epochs,
        "w_skel_fine": w_skel_fine,
        "w_skel_medium": w_skel_medium,
        "w_skel_coarse": w_skel_coarse,
        "w_smooth": w_smooth,
    })
    log_dir = create_experiment("train_log/train_skeleton_sdf", log_config)

    print(f"\n{'='*60}")
    print(f">>> 方案 B: 混合 {skeleton_mode} 骨架 + SDF 截面")
    print(f"    Phase 1: {phase1_epochs} epochs (骨架预热)")
    print(f"    Phase 2: {total_epochs - phase1_epochs} epochs (联合训练)")
    print(f"    Model params: {n_params:,}")
    print(f"    Data: {args.data_dir} ({len(train_ds)} samples)")
    print(f"    Loss weights: sdf={w_sdf}, skel_fine={w_skel_fine}, "
          f"skel_med={w_skel_medium}, skel_coarse={w_skel_coarse}, "
          f"smooth={w_smooth}, normal={w_normal}")
    print(f"    Log: {log_dir}")
    print(f"{'='*60}")

    best_loss = float("inf")

    for epoch in range(1, total_epochs + 1):
        model.train()
        epoch_loss = 0
        loss_sums = {}
        n_batches = 0
        is_phase1 = (epoch <= phase1_epochs)

        pbar = tqdm(train_loader,
                    desc=f"Phase{'1' if is_phase1 else '2'} E{epoch}/{total_epochs}")
        for action_window, coords, gt_sdf, gt_normals, gt_positions in pbar:
            action_window = action_window.to(device)
            coords = coords.to(device).squeeze(0).requires_grad_(True)
            gt_sdf = gt_sdf.to(device).squeeze(0)
            gt_normals = gt_normals.to(device).squeeze(0)
            gt_positions = gt_positions.to(device).squeeze(0)  # (31, 3)

            # ── 骨架预测 + loss ──
            pred_dict = model.predict_skeleton(action_window)
            skel_losses = model.compute_skeleton_loss(pred_dict, gt_positions.unsqueeze(0))

            loss_skel = (skel_losses['fine'] * w_skel_fine +
                         skel_losses['medium'] * w_skel_medium +
                         skel_losses['coarse'] * w_skel_coarse)

            # 曲线平滑正则
            loss_smooth = skeleton_smoothness(pred_dict['fine']) * w_smooth

            total = loss_skel + loss_smooth

            # ── SDF loss (Phase 2) ──
            loss_sdf_val = torch.tensor(0.0, device=device)
            loss_normal_val = torch.tensor(0.0, device=device)

            if not is_phase1:
                query = coords.unsqueeze(0)  # (1, M, 3)
                pred_sdf = model(query, action_window).squeeze(0)  # (M, 1)
                pred_sdf_flat = pred_sdf.squeeze(-1)

                loss_sdf_val = torch.abs(pred_sdf_flat - gt_sdf).mean() * w_sdf

                # Normal loss (表面点)
                is_surface = (gt_sdf.abs() < 1e-6).float()
                if is_surface.sum() > 0 and gt_normals.abs().sum() > 0:
                    gradient = torch.autograd.grad(
                        pred_sdf.sum(), coords, create_graph=True,
                    )[0]
                    cos_sim = F.cosine_similarity(gradient, gt_normals, dim=-1)
                    loss_normal_val = (is_surface * (1 - cos_sim)).sum() / \
                                      (is_surface.sum() + 1e-8) * w_normal

                total = total + loss_sdf_val + loss_normal_val

            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += total.item()
            for k, v in [('skel', loss_skel.item()),
                         ('smooth', loss_smooth.item()),
                         ('sdf', loss_sdf_val.item()),
                         ('normal', loss_normal_val.item())]:
                loss_sums[k] = loss_sums.get(k, 0) + v
            n_batches += 1

            avg_losses = {k: f"{v/n_batches:.2f}" for k, v in loss_sums.items()}
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

        loss_str = " | ".join(f"{k}:{v/n_batches:.4f}" for k, v in loss_sums.items())
        phase_tag = "Phase1" if is_phase1 else "Phase2"
        print(f"  [{phase_tag}] Epoch {epoch} | Total: {avg_epoch:.4f} | {loss_str}")

    np.savetxt(os.path.join(log_dir, "action_norm_factor.txt"),
               [train_ds.norm_factor])
    print(f"\n>>> Done! Best loss: {best_loss:.4f}, Log: {log_dir}")


if __name__ == "__main__":
    train()
