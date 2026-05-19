"""TwoPhaseTrainer — 两阶段训练的可扩展骨架。

继承 BaseTrainer，实现「先训练静态场、再训练变形场」的两阶段训练流程。
子类通过覆盖钩子方法来定制各阶段的具体逻辑（数据集、损失函数、冻结策略等）。

默认实现对应 CMSTNF 的 canonical → deformation 训练流程：
  - Phase 1: 冻结 deformation 模块，仅训练 canonical 场（静态形状）
  - Phase 2: 冻结 canonical 模块，训练 deformation 场（时序变形）

继承关系:
  BaseTrainer → TwoPhaseTrainer → CMSTNFTrainer / SkeletonSDFTrainer / MSSCNFTrainer

子类必须覆盖:
  _create_model(action_dim)  → 返回具体模型实例
  _model_name()              → 返回模型名字字符串（用于日志目录命名）

Phase 1 钩子（默认：canonical 渲染训练）:
  _phase1_dataset(data_dir)         → Dataset（单帧，用于学习静态形状）
  _phase1_freeze(model)             → 冻结 Phase 1 不需要训练的模块
  _phase1_train_step(model, batch)  → (loss, info_dict)（单步训练）
  _phase1_save(model, path)         → 保存 Phase 1 权重
  _phase1_validate(model, ds, epoch, log_dir) → 可选验证与可视化

Phase 2 钩子（默认：recon + smooth 渲染训练）:
  _phase2_dataset(data_dir)             → (train_ds, val_ds)（序列数据，含训练/验证拆分）
  _phase2_load_phase1(model, path)      → 加载 Phase 1 权重到模型
  _phase2_freeze(model)                 → 冻结 Phase 1 训练过的模块（防止灾难性遗忘）
  _phase2_train_step(model, batch, global_step) → (loss, info_dict)（含多损失加权）
  _phase2_validate(model, val_ds, epoch, log_dir) → val_loss 或 None

配置钩子（从 training.json 读取，子类可覆盖以改变默认值）:
  _phase1_epochs()  → Phase 1 训练轮数
  _phase2_epochs()  → Phase 2 训练轮数
  _phase1_lr()      → Phase 1 学习率
  _phase2_lr()      → Phase 2 学习率

可选覆盖:
  _save_extra_params(model, log_dir) → 保存额外参数（如 norm_factor 之外的信息）

典型用法:
  class MyTrainer(TwoPhaseTrainer):
      def _create_model(self, action_dim):
          return MyModel(action_dim=action_dim, ...)

      def _model_name(self):
          return "MyModel"

  trainer = MyTrainer(device="cuda:0")
  trainer.train(data_dir="data/sequence_data", canonical_data_dir="data/canonical_data")
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
    """两阶段训练的模板基类，通过钩子方法实现可扩展性。

    Phase 1 训练静态场（如 canonical field），Phase 2 在其基础上训练变形场。
    子类只需覆盖与具体模型相关的钩子，训练循环、日志、checkpoint 等由本类统一管理。

    配置来源: training.json 中的 temporal / canonical / loss_weights / logging 子段。
    """

    def __init__(self, device, config=None):
        """初始化训练器。

        Args:
            device: torch.device 或 str，训练设备（如 "cuda:0"）。
            config: 可选的训练配置 dict。默认从 training.json 加载。
                    结构需包含: model, optimization, temporal, canonical, loss_weights, logging。
        """
        super().__init__(device, config=config)
        self.temp_cfg = self.train_cfg.get("temporal", {})
        self.canon_cfg = self.train_cfg.get("canonical", {})
        self.loss_cfg = self.train_cfg.get("loss_weights", {})
        self.log_cfg = self.train_cfg.get("logging", {})

    # ── 必须覆盖 ──────────────────────────────────────────────────────────

    def _create_model(self, action_dim):
        """创建并返回具体模型实例。子类必须覆盖。

        Args:
            action_dim (int): 驱动参数的维度（由数据集自动检测）。

        Returns:
            nn.Module: 具体模型，需包含 canonical / deform 等子模块（供默认钩子使用）。
        """
        raise NotImplementedError

    def _model_name(self):
        """返回模型名称字符串，用于日志目录命名。子类必须覆盖。

        Returns:
            str: 模型名称，如 "CMSTNF", "SkeletonSDF"。
        """
        raise NotImplementedError

    # ── 配置钩子 ──────────────────────────────────────────────────────────

    def _phase1_epochs(self):
        """Phase 1 训练轮数。从 canonical.phase1_epochs 读取，默认 50。"""
        return self.canon_cfg.get("phase1_epochs", 50)

    def _phase2_epochs(self):
        """Phase 2 训练轮数。从 canonical.phase2_epochs 读取，默认 200。"""
        return self.canon_cfg.get("phase2_epochs", 200)

    def _phase1_lr(self):
        """Phase 1 学习率。从 optimization.lr 读取，默认 5e-4。"""
        return self.opt_cfg.get("lr", 5e-4)

    def _phase2_lr(self):
        """Phase 2 学习率。从 canonical.deform_lr 读取，默认 5e-4。"""
        return self.canon_cfg.get("deform_lr", 5e-4)

    # ── Phase 1 钩子（默认：canonical 渲染训练）────────────────────────────

    def _phase1_dataset(self, data_dir):
        """构建 Phase 1 训练数据集。

        默认加载 data_dir 下所有 .npz 文件，创建 seq_len=1 的 SoftSequenceDataset
        （即每帧独立，不使用时序窗口）。

        Args:
            data_dir (str): canonical 数据目录路径，包含 .npz 文件。

        Returns:
            SoftSequenceDataset: 单帧数据集。

        Raises:
            FileNotFoundError: 目录为空或不存在时。
        """
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not all_files:
            raise FileNotFoundError(
                f"No data in {data_dir}. "
                "Run: python scripts/data_collection/collect_canonical.py")
        return SoftSequenceDataset(data_dir, seq_len=1, file_list=all_files)

    def _phase1_freeze(self, model):
        """冻结 Phase 1 不需要训练的模块。

        默认冻结 model.deform 的所有参数（Phase 1 只训练 canonical 场）。

        Args:
            model: 模型实例，需有 deform 属性。
        """
        for p in model.deform.parameters():
            p.requires_grad = False

    def _phase1_train_step(self, model, batch):
        """Phase 1 单步训练：canonical 场渲染损失。

        流程:
          1. 从 batch 中提取图像
          2. 采样前景/背景射线
          3. 沿射线采样 3D 点，通过 model.forward_canonical 渲染 RGB
          4. 计算渲染结果与 GT 像素的 MSE 损失

        Args:
            model: 模型实例，需有 forward_canonical(pts) 方法。
            batch: DataLoader 返回的批次数据，包含 (actions, image) 或仅 image。

        Returns:
            tuple: (loss_tensor, info_dict)。info_dict 为空字典（Phase 1 无额外指标）。
        """
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
        """保存 Phase 1 训练的权重。

        默认只保存 model.canonical 的 state_dict。

        Args:
            model: 模型实例，需有 canonical 属性。
            path (str): 保存路径（如 phase1_best.pt）。
        """
        torch.save(model.canonical.state_dict(), path)

    def _phase1_validate(self, model, ds, epoch, log_dir):
        """Phase 1 验证：渲染完整图像并与 GT 对比保存。

        Args:
            model: 模型实例。
            ds: 验证数据集。
            epoch (int): 当前 epoch 编号。
            log_dir (str): 日志目录，可视化图片保存到 log_dir/vis/ 下。
        """
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
        """构建 Phase 2 训练与验证数据集。

        加载 data_dir 下的 .npz 文件，按 80/20 拆分为训练集和验证集。
        数据集使用时序窗口（window_size）并返回相邻帧对（return_pairs=True）。

        Args:
            data_dir (str): 序列数据目录路径。

        Returns:
            tuple: (train_ds, val_ds)，两个 SoftSequenceDataset 实例。
                   val_ds 复用 train_ds 的 norm_factor 以保证归一化一致。

        Raises:
            FileNotFoundError: 目录为空或不存在时。
        """
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
        """将 Phase 1 训练好的 canonical 权重加载到模型中。

        Args:
            model: 模型实例，需有 canonical 属性。
            path (str): Phase 1 权重文件路径（如 phase1_best.pt）。
                        为 None 或文件不存在时打印警告。
        """
        if path and os.path.exists(path):
            state = torch.load(path, map_location=self.device, weights_only=True)
            model.canonical.load_state_dict(state)
            print(f"    Loaded Phase 1 weights: {path}")
        else:
            print("    WARNING: No Phase 1 weights!")

    def _phase2_freeze(self, model):
        """冻结 Phase 1 训练过的模块，Phase 2 只训练新增模块。

        默认调用 model.freeze_canonical()。模型需实现该方法。

        Args:
            model: 模型实例，需有 freeze_canonical() 方法。
        """
        model.freeze_canonical()

    def _phase2_train_step(self, model, batch, global_step):
        """Phase 2 单步训练：当前帧重建 + 下一帧预测 + 平滑正则。

        损失组成:
          - loss_recon:      当前时刻 action → 渲染 vs GT 的 MSE（权重 w_recon）
          - loss_recon_next: 下一时刻 action → 渲染 vs GT 的 MSE（权重 w_recon_next）
          - loss_smooth:     相邻帧间的变形平滑正则（权重 w_smooth）

        权重从 training.json 的 loss_weights 段读取。

        Args:
            model: 模型实例，需支持 model(pts, action_seq) 前向调用。
            batch: DataLoader 返回的批次，包含 (seq_t, seq_t1, img_t, img_t1)。
            global_step (int): 全局训练步数（可用于调度器或日志）。

        Returns:
            tuple: (total_loss, info_dict)。
                   info_dict 包含 {'recon': float, 'next': float, 'smooth': float}。
        """
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

        # 当前时刻重建损失
        rgb_map = self.render_points(lambda p: model(p, seq_t), pts)
        pred_t = rgb_map.reshape(B, -1)
        loss_recon = torch.nn.functional.mse_loss(pred_t, img_t[:, sel])

        # 下一时刻预测损失
        rgb_map2 = self.render_points(lambda p: model(p, seq_t1), pts)
        pred_t1 = rgb_map2.reshape(B, -1)
        loss_recon_next = torch.nn.functional.mse_loss(pred_t1, img_t1[:, sel])

        # 平滑正则损失
        loss_smooth = model.compute_smoothness(seq_t, seq_t1)

        loss = w_recon * loss_recon + w_recon_next * loss_recon_next + w_smooth * loss_smooth
        info = {'recon': loss_recon.item(), 'next': loss_recon_next.item(),
                'smooth': loss_smooth.item()}
        return loss, info

    def _phase2_validate(self, model, val_ds, epoch, log_dir):
        """Phase 2 验证：渲染验证序列并生成 GIF 可视化。

        Args:
            model: 模型实例。
            val_ds: 验证数据集（SoftSequenceDataset）。
            epoch (int): 当前 epoch 编号。
            log_dir (str): 日志目录，GIF 保存到 log_dir/ 下。

        Returns:
            float or None: 平均验证损失。验证失败时返回 None。
        """
        val_actions = val_ds.get_raw_actions(seq_id=0)

        def val_forward(val_seq):
            """给定 action 序列，渲染完整图像。"""
            def fn(pts_chunk):
                return model(pts_chunk, val_seq)
            pts, _ = sample_stratified(self.rays_o, self.rays_d, self.near, self.far,
                                       self.n_samples, perturb=False)
            return self.render_points(fn, pts)

        return self.validate_and_gif(
            val_forward, val_ds, epoch, log_dir, action_curves=val_actions)

    def _save_extra_params(self, model, log_dir):
        """保存模型额外参数的钩子。子类可覆盖以保存 norm_factor 之外的信息。

        Args:
            model: 模型实例。
            log_dir (str): Phase 2 日志目录。
        """
        pass

    # =========================================================================
    # Phase 1: Training Loop
    # =========================================================================

    def train_phase1(self, exp_dir=None, data_dir="data/canonical_data"):
        """执行 Phase 1 训练：学习静态 canonical 场。

        流程:
          1. 加载数据集并设置相机参数
          2. 创建模型，冻结非训练模块
          3. 创建实验日志目录
          4. 训练循环：每 epoch 调用 _phase1_train_step，记录 loss
          5. 每 5 个 epoch 执行验证与可视化
          6. 保存最佳和最终的 Phase 1 权重

        Args:
            exp_dir (str, optional): 实验目录路径。为 None 时自动创建
                                     train_log/train_<model_name>/exp_<date>_<n>/。
            data_dir (str): canonical 数据目录，默认 "data/canonical_data"。

        Returns:
            tuple: (exp_dir, phase1_path)。
                   - exp_dir: 实验根目录（Phase 2 需要用）
                   - phase1_path: 最佳 Phase 1 权重路径
        """
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

        # 自动创建实验目录（含时间戳）
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
        """执行 Phase 2 训练：在 Phase 1 canonical 场基础上训练变形场。

        流程:
          1. 加载序列数据集（80/20 训练/验证拆分）
          2. 创建模型，加载 Phase 1 权重，冻结 canonical 模块
          3. 使用 ReduceLROnPlateau 学习率调度
          4. 训练循环：每 epoch 调用 _phase2_train_step，执行验证，调整学习率
          5. 保存最佳模型和定期 checkpoint
          6. 保存归一化因子和额外参数

        Args:
            exp_dir (str): Phase 1 返回的实验根目录。
            phase1_path (str): Phase 1 最佳权重路径。
            data_dir (str): 序列数据目录，默认 "data/sequence_data"。
        """
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

            # 保存最佳模型（基于验证损失或训练损失的较小值）
            if monitor < best_monitor:
                best_monitor = monitor
                torch.save(model.state_dict(), os.path.join(phase2_dir, "model", "best_model.pt"))

            # 定期保存 checkpoint（基于 global_step）
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
        """统一入口：依次执行 Phase 1 和 Phase 2 训练。

        Phase 1 的输出（实验目录和权重路径）自动传递给 Phase 2。

        Args:
            data_dir (str): Phase 2 序列数据目录，默认 "data/sequence_data"。
            canonical_data_dir (str): Phase 1 canonical 数据目录，默认 "data/canonical_data"。
        """
        print(f"\n>>> {self._model_name()}: Phase 1 → Phase 2")
        print(f"    Phase 1 data: {canonical_data_dir}")
        print(f"    Phase 2 data: {data_dir}\n")
        exp_dir, phase1_path = self.train_phase1(data_dir=canonical_data_dir)
        self.train_phase2(exp_dir, phase1_path, data_dir=data_dir)
