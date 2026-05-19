"""trainer_unified.py — 组合 PhaseStrategy + ViewStrategy 的通用训练器。

支持三种监督模式:
  "rendering"  — ViewStrategy 处理射线采样 + 体渲染 (recon, depth, reproj, consist)
  "direct_3d"  — 模型直接查询 3D 坐标 (sdf, normal, eikonal)
  "skeleton"   — 模型预测骨架，直接与 GT 对比

Loss 分两层:
  渲染层 (ViewStrategy): recon, depth, reproj, consist
  模型层 (model.compute_losses): smooth, skeleton, sdf, normal, eikonal, ...

用法:
    trainer = UnifiedTrainer(model, view_strategy=None, config=config)
    trainer.train(data_dirs, n_epochs_per_phase={...})
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.phase_strategy import PhaseStrategy
from src.training.view_strategy import ViewStrategy
from src.training.dataset_factory import create_dataset, get_collate_fn
from src.utils.experiment import create_experiment
from config.params import load_config


class UnifiedTrainer:
    """组合 PhaseStrategy + ViewStrategy 的通用训练器。

    Args:
        model: 神经场模型，必须有 training_spec 类属性和 compute_losses() 方法
        view_strategy: ViewStrategy 实例（rendering 模式必须提供，direct_3d/skeleton 模式传 None）
        config: 训练配置 dict
    """

    def __init__(self, model, view_strategy: ViewStrategy = None, config=None):
        self.model = model
        self.phase = PhaseStrategy(model)
        self.views = view_strategy
        self.config = config or load_config("training")

    def _get_loss_weight(self, loss_name, default=1.0):
        """从 config 中获取 loss 权重。"""
        lw = self.config.get("loss_weights", {})
        return lw.get(loss_name, lw.get(f"w_{loss_name}", default))

    def _compute_losses(self, forward_fn, batch, phase_spec):
        """根据 phase_spec 计算 loss。

        渲染层: supervision_mode == "rendering" 时由 ViewStrategy 处理
        模型层: 所有模式都调用 model.compute_losses()

        Args:
            forward_fn: 当前 phase 的 forward 函数
            batch: 统一 dict batch
            phase_spec: 当前阶段配置

        Returns:
            dict: loss 名到标量 Tensor 的映射，含 "total"
        """
        active = set(phase_spec.active_losses)
        losses = {}

        # ── 1. 渲染层: recon, depth, reproj, consist ──
        if phase_spec.supervision_mode == "rendering" and self.views:
            action_window = batch["action_window"].to(self.device)
            images = batch.get("images")
            depths = batch.get("depths")
            view_losses = self.views.compute_losses(
                forward_fn, action_window, images, depths, active)
            losses.update(view_losses)

        # ── 2. 模型层: smooth, skeleton, sdf, normal, eikonal, ... ──
        model_losses = self.model.compute_losses(batch, phase_spec)
        for name, val in model_losses.items():
            w = self._get_loss_weight(name, 1.0)
            losses[name] = val * w

        losses["total"] = sum(losses.values())
        return losses

    def _save_phase_modules(self, phase_dir, phase_spec):
        """保存 PhaseSpec.save_modules 中声明的子模块权重。"""
        if not phase_spec.save_modules:
            return None
        save_dict = {}
        for mod_name in phase_spec.save_modules:
            module = getattr(self.model, mod_name)
            save_dict[mod_name] = module.state_dict()
        path = os.path.join(phase_dir, "model", "phase_modules.pt")
        torch.save(save_dict, path)
        return path

    def _load_phase_modules(self, phase_spec, saved_modules_by_phase):
        """加载前面阶段保存的子模块权重。"""
        for mod_name, prev_phase_name in phase_spec.load_modules.items():
            prev_data = saved_modules_by_phase.get(prev_phase_name)
            if prev_data and mod_name in prev_data:
                module = getattr(self.model, mod_name)
                module.load_state_dict(prev_data[mod_name])
                print(f"    Loaded {mod_name} from phase '{prev_phase_name}'")

    def _create_loader(self, data_dir, phase_spec):
        """根据 phase_spec 创建 DataLoader。"""
        ds = create_dataset(
            phase_spec.dataset_type, data_dir, self.config, phase_spec)
        collate_fn = get_collate_fn(phase_spec.dataset_type, ds)
        batch_size = self.config.get("optimization", {}).get("batch_size", 4)
        return DataLoader(ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, collate_fn=collate_fn), ds

    def _setup_views_from_dataset(self, ds):
        """从数据集获取相机参数并初始化 ViewStrategy（如果需要）。"""
        if self.views is None:
            return

        if hasattr(ds, 'cam_system'):
            for v in range(ds.cam_system.n_views):
                self.views.setup(self.device, self.config)
                return
        elif hasattr(ds, 'get_camera_params'):
            params = ds.get_camera_params()
            if params is not None and hasattr(self.views, 'setup'):
                self.views.setup(self.device, self.config)

        self.views.setup(self.device, self.config)

    def train(self, data_dirs, exp_dir=None, n_epochs_per_phase=None):
        """统一训练入口。

        Args:
            data_dirs: dict, key 为 data_mode ("canonical"/"sequence"), value 为路径
            exp_dir: 实验日志目录
            n_epochs_per_phase: dict, key 为 phase name, value 为 epoch 数
        """
        self.device = next(self.model.parameters()).device
        if self.views:
            self.views.setup(self.device, self.config)

        opt_cfg = self.config["optimization"]

        if exp_dir is None:
            exp_dir = create_experiment("train_log/train_unified", {
                "model": type(self.model).__name__,
                "phases": [p.name for p in self.phase.spec.phases],
            })

        saved_modules_by_phase = {}

        for phase_idx, phase_spec in self.phase.iterate_phases():
            data_dir = data_dirs.get(phase_spec.data_mode)
            if data_dir is None:
                print(f"  Skipping phase '{phase_spec.name}': no data for '{phase_spec.data_mode}'")
                continue

            # 加载前面阶段保存的权重
            self._load_phase_modules(phase_spec, saved_modules_by_phase)

            lr = phase_spec.lr or opt_cfg["lr"]
            trainable = self.phase.get_trainable_params()
            n_trainable = sum(p.numel() for p in trainable)
            optimizer = torch.optim.Adam(trainable, lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=opt_cfg.get("scheduler_patience", 5))

            n_epochs = (n_epochs_per_phase or {}).get(
                phase_spec.name, opt_cfg["n_epochs"])

            print(f"\n{'='*50}")
            print(f"Phase: {phase_spec.name} ({phase_idx+1}/{len(self.phase.spec.phases)})")
            print(f"  Data: {data_dir}")
            print(f"  Epochs: {n_epochs}, LR: {lr}, Trainable: {n_trainable:,}")
            print(f"  Supervision: {phase_spec.supervision_mode}")
            print(f"  Active losses: {phase_spec.active_losses}")
            print(f"  Forward: {phase_spec.forward_attr}")
            print(f"{'='*50}")

            loader, ds = self._create_loader(data_dir, phase_spec)

            # 为 rendering 模式设置 ViewStrategy
            if phase_spec.supervision_mode == "rendering" and self.views:
                self._setup_views_from_dataset(ds)

            best_val = float("inf")
            phase_dir = os.path.join(exp_dir, f"phase_{phase_spec.name}")
            os.makedirs(os.path.join(phase_dir, "model"), exist_ok=True)

            for epoch in range(1, n_epochs + 1):
                self.model.train()
                epoch_loss = 0
                epoch_details = {}
                n_batches = 0

                pbar = tqdm(loader, desc=f"[{phase_spec.name}] Epoch {epoch}/{n_epochs}")
                for batch in pbar:
                    forward_fn = self.phase.get_forward_fn()
                    losses = self._compute_losses(forward_fn, batch, phase_spec)

                    optimizer.zero_grad()
                    losses["total"].backward()
                    optimizer.step()

                    epoch_loss += losses["total"].item()
                    for k, v in losses.items():
                        if k != "total":
                            epoch_details[k] = epoch_details.get(k, 0) + (v.item() if isinstance(v, torch.Tensor) else v)
                    n_batches += 1

                    pbar.set_postfix({k: f"{v.item():.4f}" for k, v in losses.items() if k != "total"})

                avg = epoch_loss / max(n_batches, 1)
                detail_str = ", ".join(
                    f"{k}={v / max(n_batches, 1):.4f}" for k, v in epoch_details.items())
                print(f"  Epoch {epoch} | Loss: {avg:.5f} | {detail_str}")

                if avg < best_val:
                    best_val = avg
                    torch.save(self.model.state_dict(),
                               os.path.join(phase_dir, "model", "best_model.pt"))

                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg)
                else:
                    scheduler.step()

            # 保存 Phase 权重
            save_path = self._save_phase_modules(phase_dir, phase_spec)
            if save_path:
                saved_modules_by_phase[phase_spec.name] = {
                    mod_name: getattr(self.model, mod_name).state_dict()
                    for mod_name in phase_spec.save_modules
                }

            torch.save(self.model.state_dict(),
                       os.path.join(phase_dir, "model", "final_model.pt"))

        print(f"\n训练完成! 日志: {exp_dir}")
        return exp_dir
