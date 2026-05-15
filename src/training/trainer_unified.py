"""trainer_unified.py — 组合 PhaseStrategy + ViewStrategy 的通用训练器。

任意模型 × 任意阶段策略 × 任意视角策略，通过单一 trainer 训练。
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.phase_strategy import PhaseStrategy
from src.training.view_strategy import ViewStrategy
from src.utils.experiment import create_experiment
from config.params import load_config


class UnifiedTrainer:
    """组合 PhaseStrategy + ViewStrategy 的通用训练器。

    Args:
        model: 神经场模型，必须有 training_spec 类属性
        view_strategy: ViewStrategy 实例
        config: 训练配置 dict
        extra_losses_fn: 可选回调 fn(model, batch, phase_spec, active_losses) → dict
    """

    def __init__(self, model, view_strategy: ViewStrategy, config=None,
                 extra_losses_fn=None):
        self.model = model
        self.phase = PhaseStrategy(model)
        self.views = view_strategy
        self.config = config or load_config("training")
        self.extra_losses_fn = extra_losses_fn

    def _compute_smoothness(self, action_window, action_window_next):
        """计算时序平滑 loss。"""
        action_window = action_window.to(self.device)
        action_window_next = action_window_next.to(self.device)
        if hasattr(self.model, 'temporal'):
            state_t = self.model.temporal(action_window)
            state_t1 = self.model.temporal(action_window_next)
        elif hasattr(self.model, 'deform') and hasattr(self.model.deform, 'temporal'):
            state_t = self.model.deform.temporal(action_window)
            state_t1 = self.model.deform.temporal(action_window_next)
        else:
            return torch.tensor(0.0, device=self.device)
        return F.mse_loss(state_t, state_t1)

    def _compute_losses(self, forward_fn, batch, phase_spec):
        """根据 phase_spec.active_losses 选择性计算 loss。"""
        active = phase_spec.active_losses
        losses = {}

        action_window = batch['action_window'].to(self.device)
        images = batch['images']
        depths = batch.get('depths', None)
        action_window_next = batch.get('action_window_next', None)

        # 1. 渲染相关 loss
        view_losses = self.views.compute_losses(
            forward_fn, action_window, images, depths, active)
        losses.update(view_losses)

        # 2. smoothness
        w_smooth = self.config.get("loss_weights", {}).get("smoothness", 0.1)
        if "smooth" in active and self.phase.spec.supports_smoothness:
            if action_window_next is not None:
                losses["smooth"] = self._compute_smoothness(
                    action_window, action_window_next) * w_smooth

        # 3. 模型特定 loss
        if self.extra_losses_fn:
            extra = self.extra_losses_fn(self.model, batch, phase_spec, active)
            losses.update(extra)

        losses["total"] = sum(losses.values())
        return losses

    def train(self, data_dirs, exp_dir=None, n_epochs_per_phase=None):
        """统一训练入口。

        Args:
            data_dirs: dict, key 为 data_mode ("canonical"/"sequence"), value 为路径
            exp_dir: 实验日志目录
            n_epochs_per_phase: dict, key 为 phase name, value 为 epoch 数
        """
        self.device = next(self.model.parameters()).device
        self.views.setup(self.device, self.config)

        opt_cfg = self.config["optimization"]

        if exp_dir is None:
            exp_dir = create_experiment("train_log/train_unified", {
                "model": type(self.model).__name__,
                "phases": [p.name for p in self.phase.spec.phases],
            })

        for phase_idx, phase_spec in self.phase.iterate_phases():
            data_dir = data_dirs.get(phase_spec.data_mode)
            if data_dir is None:
                print(f"  Skipping phase '{phase_spec.name}': no data for '{phase_spec.data_mode}'")
                continue

            lr = phase_spec.lr or opt_cfg["lr"]
            trainable = self.phase.get_trainable_params()
            n_trainable = sum(p.numel() for p in trainable)
            optimizer = torch.optim.Adam(trainable, lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=opt_cfg["scheduler_patience"])

            n_epochs = (n_epochs_per_phase or {}).get(
                phase_spec.name, opt_cfg["n_epochs"])

            print(f"\n{'='*50}")
            print(f"Phase: {phase_spec.name} ({phase_idx+1}/{len(self.phase.spec.phases)})")
            print(f"  Data: {data_dir}")
            print(f"  Epochs: {n_epochs}, LR: {lr}, Trainable: {n_trainable:,}")
            print(f"  Active losses: {phase_spec.active_losses}")
            print(f"  Forward: {phase_spec.forward_attr}")
            print(f"{'='*50}")

            loader = self._create_loader(data_dir, phase_spec)

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
                            epoch_details[k] = epoch_details.get(k, 0) + v.item()
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

            torch.save(self.model.state_dict(),
                       os.path.join(phase_dir, "model", "final_model.pt"))

        print(f"\n训练完成! 日志: {exp_dir}")
        return exp_dir

    def _create_loader(self, data_dir, phase_spec):
        """根据 phase_spec.data_mode 创建 DataLoader。"""
        from src.data.dataset_multiview_depth import MultiViewDepthDataset

        temp_cfg = self.config["temporal"]
        batch_size = self.config["optimization"]["batch_size"]

        if phase_spec.data_mode == "canonical":
            ds = MultiViewDepthDataset(
                data_dir, seq_len=1, return_depth=False, return_pairs=False)
        else:
            ds = MultiViewDepthDataset(
                data_dir, seq_len=temp_cfg["window_size"],
                return_depth="depth" in phase_spec.active_losses,
                return_pairs="smooth" in phase_spec.active_losses)

        n_views = ds.n_views

        def collate_to_dict(batch):
            """将 tuple batch 转为 dict 格式供 _compute_losses 使用。"""
            action_windows = torch.stack([b[0] for b in batch])
            images_list = [torch.stack([b[1][v] for b in batch]) for v in range(n_views)]

            depths_list = None
            if batch[0][2] is not None:
                depths_list = [torch.stack([b[2][v] for b in batch]) for v in range(n_views)]

            action_window_next = None
            if len(batch[0]) > 4 and batch[0][4] is not None:
                action_window_next = torch.stack([b[4] for b in batch])

            return {
                'action_window': action_windows,
                'images': images_list,
                'depths': depths_list,
                'action_window_next': action_window_next,
            }

        return DataLoader(ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, collate_fn=collate_to_dict)
