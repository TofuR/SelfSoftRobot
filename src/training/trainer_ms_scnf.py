"""MS-SCNF Trainer — 两阶段训练：Phase 1 骨架回归 + Phase 2 联合训练。

Phase 1: 仅 SkeletonHead，3D L2 loss（GT 来自仿真器 position_collection）。
Phase 2: 联合 SkeletonHead + DensityField，3D skeleton loss + 2D rendering loss。
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import BaseTrainer
from .metrics_3d import mean_node_error, endpoint_error, curve_smoothness
from src.models.model_ms_scnf import MSSCNFModel
from src.utils.rendering import OM_rendering, sample_stratified
from src.data.dataset import SoftSequenceDataset


class MSSCNFTrainer(BaseTrainer):
    """MS-SCNF 两阶段训练器。"""

    def __init__(self, device, config=None):
        super().__init__(device, config=config)
        self.temp_cfg = self.train_cfg["temporal"]
        self.canon_cfg = self.train_cfg["canonical"]
        self.loss_cfg = self.train_cfg["loss_weights"]
        self.log_cfg = self.train_cfg["logging"]
        self.ms_cfg = self.train_cfg.get("ms_scnf", {})

    def _model_name(self):
        return "MS_SCNF"

    def _create_model(self, action_dim):
        return MSSCNFModel(
            action_dim=action_dim,
            window_size=self.temp_cfg["window_size"],
            n_scales=self.temp_cfg["n_scales"],
            hidden_dim=self.temp_cfg["hidden_dim"],
            d_filter=self.model_cfg["d_filter"],
            n_freqs=self.model_cfg["n_freqs"],
            n_coarse=self.ms_cfg.get("n_coarse", 4),
            n_medium=self.ms_cfg.get("n_medium", 10),
            n_fine=self.ms_cfg.get("n_fine", 31),
            deform_n_freqs=self.canon_cfg["deform_n_freqs"],
        )

    # =========================================================================
    # Phase 1: Skeleton Regression (3D loss only)
    # =========================================================================

    def train_phase1(self, exp_dir=None, data_dir="data/sequence_data_3d"):
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not all_files:
            raise FileNotFoundError(
                f"No 3D sequence data in {data_dir}. "
                "Run: python scripts/data_collection/collect_sequence_3d.py")

        # 检测是否有 3D 标注
        sample_data = np.load(all_files[0])
        if 'positions' not in sample_data:
            raise ValueError(
                f"Data in {data_dir} lacks 'positions' field. "
                "Use collect_sequence_3d.py to generate 3D data.")

        ds = SoftSequenceDataset(
            data_dir, seq_len=self.temp_cfg["window_size"],
            file_list=all_files, return_3d=True,
        )
        loader = DataLoader(ds, batch_size=self.opt_cfg["batch_size"],
                            shuffle=True, num_workers=2)

        action_dim = ds.action_dim
        n_nodes = sample_data['positions'].shape[-1]  # 应为 3 (x,y,z per node) → 但 positions 是 (T,3,31)
        # positions shape: (T, 3, 31) → GT skeleton 形状 (B, 31, 3)
        n_fine = self.ms_cfg.get("n_fine", 31)

        model = self._create_model(action_dim).to(self.device)
        # Phase 1 只训练 temporal + skeleton_head
        for p in model.density.parameters():
            p.requires_grad = False

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=self.opt_cfg["lr"])
        n_epochs = self.canon_cfg["phase1_epochs"]

        w_fine = self.ms_cfg.get("w_skeleton_fine", 1.0)
        w_medium = self.ms_cfg.get("w_skeleton_medium", 0.3)
        w_coarse = self.ms_cfg.get("w_skeleton_coarse", 0.1)

        if exp_dir is None:
            config_dict = {
                "model": self._model_name(),
                "phase1": {"data": data_dir, "lr": self.opt_cfg["lr"],
                           "n_epochs": n_epochs},
            }
            exp_dir = self.create_experiment(
                os.path.join("train_log", f"train_{self._model_name().lower()}"), config_dict)
        phase1_dir = self.make_phase_dirs(exp_dir, "phase1")

        n_trainable = sum(p.numel() for p in trainable)
        print(f"\n{'='*60}")
        print(f">>> Phase 1: Skeleton Regression, {n_epochs} epochs")
        print(f"    Data: {data_dir} ({len(all_files)} files)")
        print(f"    Nodes: {n_fine}, Trainable: {n_trainable:,}")
        print(f"    Log: {phase1_dir}")
        print(f"{'='*60}")

        best_loss = float("inf")

        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0
            metrics = {'fine': 0, 'medium': 0, 'coarse': 0}
            pbar = tqdm(loader, desc=f"[Phase1] Epoch {epoch}/{n_epochs}")

            for batch in pbar:
                # return_3d=True: (seq, img, positions)
                seq = batch[0].to(self.device)
                positions = batch[2].to(self.device)  # (B, 3, N_nodes)
                # 转置为 (B, N_nodes, 3)
                gt_skeleton = positions.permute(0, 2, 1)
                B = seq.shape[0]

                state = model.encode(seq)
                pred_dict = model.skeleton_head(state)
                losses = model.compute_skeleton_loss(pred_dict, gt_skeleton)

                loss = (w_fine * losses['fine']
                        + w_medium * losses['medium']
                        + w_coarse * losses['coarse'])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                for k in metrics:
                    metrics[k] += losses[k].item()
                pbar.set_postfix({'loss': f'{loss.item():.5f}'})

            avg_loss = epoch_loss / max(len(loader), 1)
            n = max(len(loader), 1)
            metrics_str = " | ".join(f"{k}: {v/n:.5f}" for k, v in metrics.items())

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'temporal': model.temporal.state_dict(),
                    'skeleton_head': model.skeleton_head.state_dict(),
                }, os.path.join(phase1_dir, "model", "skeleton_best.pt"))

            print(f"  Epoch {epoch} | Loss: {avg_loss:.5f} | {metrics_str}")

        # 保存最终权重
        torch.save({
            'temporal': model.temporal.state_dict(),
            'skeleton_head': model.skeleton_head.state_dict(),
        }, os.path.join(phase1_dir, "model", "skeleton_final.pt"))

        skeleton_path = os.path.join(phase1_dir, "model", "skeleton_best.pt")
        print(f">>> Phase 1 done! Best: {best_loss:.5f}, Weights: {skeleton_path}")
        del model
        return exp_dir, skeleton_path

    # =========================================================================
    # Phase 2: Joint Training (3D + 2D)
    # =========================================================================

    def train_phase2(self, exp_dir, skeleton_path, data_dir="data/sequence_data_3d"):
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not all_files:
            raise FileNotFoundError(f"No 3D sequence data in {data_dir}")

        split = max(1, int(0.8 * len(all_files)))
        train_files, val_files = all_files[:split], all_files[split:]

        train_ds = SoftSequenceDataset(
            data_dir, seq_len=self.temp_cfg["window_size"],
            file_list=train_files, return_pairs=True, return_3d=True,
        )
        val_ds = SoftSequenceDataset(
            data_dir, seq_len=self.temp_cfg["window_size"],
            file_list=val_files, norm_factor=train_ds.norm_factor,
        )
        train_loader = DataLoader(train_ds, batch_size=self.opt_cfg["batch_size"],
                                  shuffle=True, num_workers=4)

        self.setup_camera(train_ds.H, train_ds.W, train_ds.focal, camera_pose=train_ds.get_camera_params())
        action_dim = train_ds.action_dim

        model = self._create_model(action_dim).to(self.device)

        # 加载 Phase 1 骨架权重
        if skeleton_path and os.path.exists(skeleton_path):
            state = torch.load(skeleton_path, map_location=self.device, weights_only=True)
            model.temporal.load_state_dict(state['temporal'])
            model.skeleton_head.load_state_dict(state['skeleton_head'])
            print(f"    Loaded skeleton weights: {skeleton_path}")

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=self.canon_cfg["deform_lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.opt_cfg["scheduler_patience"])

        w_skeleton = self.ms_cfg.get("w_skeleton_fine", 1.0)
        w_render = self.ms_cfg.get("w_render", 0.5)
        w_smooth = self.loss_cfg["smoothness"]
        n_epochs = self.canon_cfg["phase2_epochs"]
        save_rate = self.log_cfg["save_rate"]

        phase2_dir = self.make_phase_dirs(exp_dir, "phase2")

        n_trainable = sum(p.numel() for p in trainable)
        print(f"\n{'='*60}")
        print(f">>> Phase 2: {self._model_name()} Joint Training, {n_epochs} epochs")
        print(f"    Data: {data_dir}, Trainable: {n_trainable:,}")
        print(f"    Log: {phase2_dir}")
        print(f"{'='*60}")

        best_val_loss = float("inf")
        global_step = 0

        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0
            loss_info = {'skel': 0, 'render': 0, 'smooth': 0}
            pbar = tqdm(train_loader, desc=f"[Phase2] Epoch {epoch}/{n_epochs}")

            for batch in pbar:
                # return_pairs + return_3d: (seq_t, seq_t1, img_t, img_t1, pos_t, pos_t1)
                seq_t = batch[0].to(self.device)
                seq_t1 = batch[1].to(self.device)
                img_t = batch[2].to(self.device)
                img_t1 = batch[3].to(self.device)
                pos_t = batch[4].to(self.device)  # (B, 3, N_nodes)

                B = img_t.shape[0]
                gt_skeleton = pos_t.permute(0, 2, 1)  # (B, N_nodes, 3)

                # 1. Skeleton loss
                state_t = model.encode(seq_t)
                pred_dict = model.skeleton_head(state_t)
                skel_losses = model.compute_skeleton_loss(pred_dict, gt_skeleton)
                loss_skel = skel_losses['fine']

                # 2. Rendering loss
                sel, rays_o_sel, rays_d_sel = self.sample_fg_rays(img_t)
                pts, _ = sample_stratified(rays_o_sel, rays_d_sel, self.near, self.far, self.n_samples)

                rgb_map = self.render_points(lambda p: model(p, seq_t), pts)
                pred_t = rgb_map.reshape(B, -1)
                loss_render = torch.nn.functional.mse_loss(pred_t, img_t[:, sel])

                # 3. Smoothness loss
                loss_smooth = model.compute_smoothness(seq_t, seq_t1)

                loss = w_skeleton * loss_skel + w_render * loss_render + w_smooth * loss_smooth

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                loss_info['skel'] += loss_skel.item()
                loss_info['render'] += loss_render.item()
                loss_info['smooth'] += loss_smooth.item()
                pbar.set_postfix({'loss': f'{loss.item():.5f}'})
                global_step += 1

            # 验证（2D rendering based）
            model.eval()

            def val_forward(val_seq):
                def fn(pts_chunk):
                    return model(pts_chunk, val_seq)
                pts, _ = sample_stratified(self.rays_o, self.rays_d, self.near, self.far,
                                           self.n_samples, perturb=False)
                return self.render_points(fn, pts)

            val_actions = val_ds.get_raw_actions(seq_id=0)
            val_loss = self.validate_and_gif(
                val_forward, val_ds, epoch, phase2_dir, action_curves=val_actions)
            scheduler.step(val_loss)

            n = max(len(train_loader), 1)
            info_str = " | ".join(f"{k}: {v/n:.5f}" for k, v in loss_info.items())
            avg_train = epoch_loss / n
            print(f"  Epoch {epoch} | Train: {avg_train:.5f} | Val: {val_loss:.5f} | {info_str}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(phase2_dir, "model", "best_model.pt"))

            if global_step % save_rate == 0:
                torch.save(model.state_dict(),
                           os.path.join(phase2_dir, "model", f"model_{global_step:05d}.pt"))

        np.savetxt(os.path.join(phase2_dir, "action_norm_factor.txt"), [train_ds.norm_factor])
        print(f">>> Phase 2 done! Best val: {best_val_loss:.5f}")

    # =========================================================================
    # 统一入口
    # =========================================================================

    def train(self, data_dir="data/sequence_data_3d"):
        print(f"\n>>> {self._model_name()}: Phase 1 (Skeleton) → Phase 2 (Joint)")
        print(f"    Data: {data_dir}\n")
        exp_dir, skeleton_path = self.train_phase1(data_dir=data_dir)
        self.train_phase2(exp_dir, skeleton_path, data_dir=data_dir)
