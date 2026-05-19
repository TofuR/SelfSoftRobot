"""TwoPhaseTrainer — 两阶段训练的可扩展骨架。

子类通过钩子方法定制 Phase 1 和 Phase 2 的具体逻辑。
默认实现对应 CMSTNF 的 canonical → deformation 训练流程。

子类必须覆盖:
  _create_model(action_dim)  → 返回具体模型
  _model_name()              → 返回模型名字字符串

Phase 1 钩子（默认：canonical 渲染训练）:
  _phase1_dataset(data_dir)      → Dataset
  _phase1_freeze(model)          → 冻结 Phase 1 不训练的模块
  _phase1_train_step(model, batch) → (loss, info_dict)
  _phase1_save(model, path)      → 保存 Phase 1 权重
  _phase1_validate(model, ds, epoch, log_dir) → 可选验证

Phase 2 钩子（默认：recon + smooth 渲染训练）:
  _phase2_dataset(data_dir)      → (train_ds, val_ds)
  _phase2_load_phase1(model, path) → 加载 Phase 1 权重
  _phase2_freeze(model)          → 冻结 Phase 1 训练过的模块
  _phase2_train_step(model, batch, global_step) → (loss, info_dict)
  _phase2_validate(model, val_ds, epoch, log_dir) → val_loss 或 None

配置钩子:
  _phase1_epochs(), _phase2_epochs(), _phase1_lr(), _phase2_lr()

可选:
  _save_extra_params(model, log_dir)
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import BaseTrainer
from src.utils.rendering import sample_stratified
from src.data.dataset import SoftSequenceDataset


class TwoPhaseTrainer(BaseTrainer):
    """两阶段训练：Phase 1 + Phase 2，通过钩子定制具体逻辑。"""

    def __init__(self, device, config=None):
        super().__init__(device, config=config)
        self.temp_cfg = self.train_cfg.get("temporal", {})
        self.canon_cfg = self.train_cfg.get("canonical", {})
        self.loss_cfg = self.train_cfg.get("loss_weights", {})
        self.log_cfg = self.train_cfg.get("logging", {})

    # ── 必须覆盖 ──────────────────────────────────────────────────────────

    def _create_model(self, action_dim):
        raise NotImplementedError

    def _model_name(self):
        raise NotImplementedError

    # ── 配置钩子 ──────────────────────────────────────────────────────────

    def _phase1_epochs(self):
        return self.canon_cfg.get("phase1_epochs", 50)

    def _phase2_epochs(self):
        return self.canon_cfg.get("phase2_epochs", 200)

    def _phase1_lr(self):
        return self.opt_cfg.get("lr", 5e-4)

    def _phase2_lr(self):
        return self.canon_cfg.get("deform_lr", 5e-4)

    # ── Phase 1 钩子（默认：canonical 渲染训练）────────────────────────────

    def _phase1_dataset(self, data_dir):
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not all_files:
            raise FileNotFoundError(
                f"No data in {data_dir}. "
                "Run: python scripts/data_collection/collect_canonical.py")
        return SoftSequenceDataset(data_dir, seq_len=1, file_list=all_files)

    def _phase1_freeze(self, model):
        for p in model.deform.parameters():
            p.requires_grad = False

    def _phase1_train_step(self, model, batch):
        if len(batch) == 2:
            _, img = batch
        else:
            img = batch[1]
        img = img.to(self.device)
        B = img.shape[0]

        sel, rays_o_sel, rays_d_sel = self.sample_fg_rays(img)
        pts, _ = sample_stratified(rays_o_sel, rays_d_sel, self.near, self.far, self.n_samples)

        rgb_map = self.render_points(model.forward_canonical, pts)
        pred = rgb_map.unsqueeze(0).expand(B, -1)
        gt = img[:, sel]

        loss = torch.nn.functional.mse_loss(pred, gt)
        return loss, {}

    def _phase1_save(self, model, path):
        torch.save(model.canonical.state_dict(), path)

    def _phase1_validate(self, model, ds, epoch, log_dir):
        model.eval()
        with torch.no_grad():
            pred_img = self.render_full_image(model.forward_canonical, perturb=False)
            sample = ds[0]
            gt_img = sample[1].reshape(self.H, self.W).numpy()
            self.save_canonical_comparison(
                pred_img, gt_img,
                os.path.join(log_dir, "vis", f"phase1_epoch_{epoch:02d}.png"))

    # ── Phase 2 钩子（默认：recon + smooth 渲染训练）───────────────────────

    def _phase2_dataset(self, data_dir):
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not all_files:
            raise FileNotFoundError(f"No sequence data in {data_dir}")

        split = max(1, int(0.8 * len(all_files)))
        train_files, val_files = all_files[:split], all_files[split:]

        train_ds = SoftSequenceDataset(
            data_dir, seq_len=self.temp_cfg["window_size"],
            file_list=train_files, return_pairs=True,
        )
        val_ds = SoftSequenceDataset(
            data_dir, seq_len=self.temp_cfg["window_size"],
            file_list=val_files, norm_factor=train_ds.norm_factor,
        )
        return train_ds, val_ds

    def _phase2_load_phase1(self, model, path):
        if path and os.path.exists(path):
            state = torch.load(path, map_location=self.device, weights_only=True)
            model.canonical.load_state_dict(state)
            print(f"    Loaded Phase 1 weights: {path}")
        else:
            print("    WARNING: No Phase 1 weights!")

    def _phase2_freeze(self, model):
        model.freeze_canonical()

    def _phase2_train_step(self, model, batch, global_step):
        seq_t, seq_t1, img_t, img_t1 = batch
        seq_t = seq_t.to(self.device)
        seq_t1 = seq_t1.to(self.device)
        img_t = img_t.to(self.device)
        img_t1 = img_t1.to(self.device)
        B = img_t.shape[0]

        sel, rays_o_sel, rays_d_sel = self.sample_fg_rays(img_t)
        pts, _ = sample_stratified(rays_o_sel, rays_d_sel, self.near, self.far, self.n_samples)

        w_recon = self.loss_cfg.get("recon_current", 1.0)
        w_recon_next = self.loss_cfg.get("recon_next", 0.5)
        w_smooth = self.loss_cfg.get("smoothness", 0.01)

        rgb_map = self.render_points(lambda p: model(p, seq_t), pts)
        pred_t = rgb_map.reshape(B, -1)
        loss_recon = torch.nn.functional.mse_loss(pred_t, img_t[:, sel])

        rgb_map2 = self.render_points(lambda p: model(p, seq_t1), pts)
        pred_t1 = rgb_map2.reshape(B, -1)
        loss_recon_next = torch.nn.functional.mse_loss(pred_t1, img_t1[:, sel])

        loss_smooth = model.compute_smoothness(seq_t, seq_t1)

        loss = w_recon * loss_recon + w_recon_next * loss_recon_next + w_smooth * loss_smooth
        info = {'recon': loss_recon.item(), 'next': loss_recon_next.item(),
                'smooth': loss_smooth.item()}
        return loss, info

    def _phase2_validate(self, model, val_ds, epoch, log_dir):
        val_actions = val_ds.get_raw_actions(seq_id=0)

        def val_forward(val_seq):
            def fn(pts_chunk):
                return model(pts_chunk, val_seq)
            pts, _ = sample_stratified(self.rays_o, self.rays_d, self.near, self.far,
                                       self.n_samples, perturb=False)
            return self.render_points(fn, pts)

        return self.validate_and_gif(
            val_forward, val_ds, epoch, log_dir, action_curves=val_actions)

    def _save_extra_params(self, model, log_dir):
        pass

    # =========================================================================
    # Phase 1: Training Loop
    # =========================================================================

    def train_phase1(self, exp_dir=None, data_dir="data/canonical_data"):
        ds = self._phase1_dataset(data_dir)

        try:
            self.setup_camera(ds.H, ds.W, ds.focal, camera_pose=ds.get_camera_params())
        except AttributeError:
            pass

        loader = DataLoader(ds, batch_size=self.opt_cfg.get("batch_size", 4),
                            shuffle=True, num_workers=2)

        model = self._create_model(ds.action_dim).to(self.device)
        self._phase1_freeze(model)

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=self._phase1_lr())
        n_epochs = self._phase1_epochs()

        if exp_dir is None:
            config_dict = {
                "model": self._model_name(),
                "phase1": {"data": data_dir, "lr": self._phase1_lr(),
                           "n_epochs": n_epochs, "image_size": [self.H, self.W]},
            }
            exp_dir = self.create_experiment(
                os.path.join("train_log", f"train_{self._model_name().lower()}"), config_dict)
        phase1_dir = self.make_phase_dirs(exp_dir, "phase1")

        n_trainable = sum(p.numel() for p in trainable)
        print(f"\n{'='*60}")
        print(f">>> Phase 1: {self._model_name()}, {n_epochs} epochs")
        print(f"    Data: {data_dir} ({len(ds)} samples)")
        if self.H:
            print(f"    Image: {self.H}x{self.W}")
        print(f"    Trainable: {n_trainable:,}")
        print(f"    Log: {phase1_dir}")
        print(f"{'='*60}")

        best_loss = float("inf")

        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0
            epoch_info = {}
            pbar = tqdm(loader, desc=f"[Phase1] Epoch {epoch}/{n_epochs}")

            for batch in pbar:
                loss, info = self._phase1_train_step(model, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                for k, v in info.items():
                    epoch_info[k] = epoch_info.get(k, 0) + (v if isinstance(v, (int, float)) else v)
                pbar.set_postfix({'loss': f'{loss.item():.5f}'})

            avg_loss = epoch_loss / max(len(loader), 1)

            if avg_loss < best_loss:
                best_loss = avg_loss
                self._phase1_save(model, os.path.join(phase1_dir, "model", "phase1_best.pt"))

            if epoch % 5 == 0 or epoch == n_epochs:
                try:
                    self._phase1_validate(model, ds, epoch, phase1_dir)
                except (AttributeError, RuntimeError):
                    pass

            info_str = " | ".join(
                f"{k}: {v / max(len(loader), 1):.5f}" for k, v in epoch_info.items())
            print(f"  Epoch {epoch} | Loss: {avg_loss:.5f}"
                  + (f" | {info_str}" if info_str else ""))

        self._phase1_save(model, os.path.join(phase1_dir, "model", "phase1_final.pt"))

        phase1_path = os.path.join(phase1_dir, "model", "phase1_best.pt")
        print(f">>> Phase 1 done! Best: {best_loss:.5f}, Weights: {phase1_path}")
        del model
        return exp_dir, phase1_path

    # =========================================================================
    # Phase 2: Training Loop
    # =========================================================================

    def train_phase2(self, exp_dir, phase1_path, data_dir="data/sequence_data"):
        train_ds, val_ds = self._phase2_dataset(data_dir)
        train_loader = DataLoader(train_ds, batch_size=self.opt_cfg.get("batch_size", 4),
                                  shuffle=True, num_workers=4)

        try:
            self.setup_camera(train_ds.H, train_ds.W, train_ds.focal,
                              camera_pose=train_ds.get_camera_params())
        except AttributeError:
            pass

        model = self._create_model(train_ds.action_dim).to(self.device)
        self._phase2_load_phase1(model, phase1_path)
        self._phase2_freeze(model)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=self._phase2_lr())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.opt_cfg.get("scheduler_patience", 5))

        n_epochs = self._phase2_epochs()
        save_rate = self.log_cfg.get("save_rate", 1000)

        phase2_dir = self.make_phase_dirs(exp_dir, "phase2")

        n_trainable = sum(p.numel() for p in trainable_params)
        print(f"\n{'='*60}")
        print(f">>> Phase 2: {self._model_name()}, {n_epochs} epochs")
        print(f"    Data: {data_dir}, Trainable: {n_trainable:,}")
        print(f"    Log: {phase2_dir}")
        print(f"{'='*60}")

        best_monitor = float("inf")
        global_step = 0

        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0
            epoch_info = {}
            pbar = tqdm(train_loader, desc=f"[Phase2] Epoch {epoch}/{n_epochs}")

            for batch in pbar:
                loss, info = self._phase2_train_step(model, batch, global_step)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                for k, v in info.items():
                    epoch_info[k] = epoch_info.get(k, 0) + (v if isinstance(v, (int, float)) else v)
                pbar.set_postfix({'loss': f'{loss.item():.5f}'})
                global_step += 1

            avg_train = epoch_loss / max(len(train_loader), 1)

            # 验证
            val_loss = None
            try:
                val_loss = self._phase2_validate(model, val_ds, epoch, phase2_dir)
            except (AttributeError, RuntimeError):
                pass

            monitor = val_loss if val_loss is not None else avg_train
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitor)
            else:
                scheduler.step()

            val_str = f" | Val: {val_loss:.5f}" if val_loss is not None else ""
            info_str = " | ".join(
                f"{k}: {v / max(len(train_loader), 1):.4f}" for k, v in epoch_info.items())
            print(f"  Epoch {epoch} | Train: {avg_train:.5f}{val_str}"
                  + (f" | {info_str}" if info_str else ""))

            if monitor < best_monitor:
                best_monitor = monitor
                torch.save(model.state_dict(), os.path.join(phase2_dir, "model", "best_model.pt"))

            if global_step % save_rate == 0:
                torch.save(model.state_dict(),
                           os.path.join(phase2_dir, "model", f"model_{global_step:05d}.pt"))

        try:
            np.savetxt(os.path.join(phase2_dir, "action_norm_factor.txt"), [train_ds.norm_factor])
        except AttributeError:
            pass
        self._save_extra_params(model, phase2_dir)
        print(f">>> Phase 2 done! Best: {best_monitor:.5f}")

    # =========================================================================
    # 统一入口
    # =========================================================================

    def train(self, data_dir="data/sequence_data", canonical_data_dir="data/canonical_data"):
        print(f"\n>>> {self._model_name()}: Phase 1 → Phase 2")
        print(f"    Phase 1 data: {canonical_data_dir}")
        print(f"    Phase 2 data: {data_dir}\n")
        exp_dir, phase1_path = self.train_phase1(data_dir=canonical_data_dir)
        self.train_phase2(exp_dir, phase1_path, data_dir=data_dir)
