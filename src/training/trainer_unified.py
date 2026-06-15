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

import csv
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.phase_strategy import PhaseStrategy
from src.rendering.view_strategy import ViewStrategy
from src.training.dataset_factory import create_dataset, get_collate_fn
from src.utils.experiment import create_experiment
from src.evaluation.shape_evaluation import evaluate_shape_during_training, evaluate_skeleton_during_training
from config.params import load_config
from src.evaluation.surface_sampling import sample_gt_surface, model_output_to_pointcloud
from src.evaluation.shape_metrics import chamfer_distance, f_score, hausdorff_distance


class UnifiedTrainer:
    """组合 PhaseStrategy + ViewStrategy 的通用训练器。

    Args:
        model: 神经场模型，必须有 training_spec 类属性和 compute_losses() 方法
        view_strategy: ViewStrategy 实例（rendering 模式必须提供，direct_3d/skeleton 模式传 None）
        config: 训练配置 dict
    """

    def __init__(self, model, view_strategy: ViewStrategy = None, config=None,
                 model_tag=None):
        self.model = model
        self.model_tag = model_tag or type(model).__name__.lower().replace('model', '')
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

        # 如果 use_gt_skeleton，准备 GT skeleton（按 batch 元素切片传给 view_strategy）
        gt_skeleton = None
        if getattr(phase_spec, 'use_gt_skeleton', False) and "gt_positions" in batch:
            gt_skeleton = batch["gt_positions"].to(self.device)
            if gt_skeleton.shape[-1] != 3 and gt_skeleton.shape[1] == 3:
                gt_skeleton = gt_skeleton.permute(0, 2, 1)

        # ── 1. 渲染层: recon, depth, reproj, consist ──
        if phase_spec.supervision_mode == "rendering" and self.views:
            action_window = batch["action_window"].to(self.device)
            images = batch.get("images")
            depths = batch.get("depths")
            view_losses = self.views.compute_losses(
                forward_fn, action_window, images, depths, active,
                gt_skeleton=gt_skeleton)
            losses.update(view_losses)

        # ── 2. 模型层: smooth, skeleton, sdf, normal, eikonal, ... ──
        model_losses = self.model.compute_losses(batch, phase_spec)
        for name, val in model_losses.items():
            w = self._get_loss_weight(name, 1.0)
            losses[name] = val * w

        losses["total"] = sum(v for k, v in losses.items() if not k.endswith("_monitor"))
        return losses

    def _compute_sequence_losses(self, batch, phase_spec):
        """Stage 1 序列级损失：episode 内逐步 rollout + scheduled sampling + z 跨帧演化。

        与逐帧 _compute_losses 的区别：
          - z 在序列内逐步演化（经 model.forward_sequence），真正成为迟滞潜变量
          - 梯度穿过 T 步（BPTT），训练 z 的转移动力学
          - scheduled sampling：每步按 teacher_forcing_ratio 决定下一步的 prev_skeleton
            取 GT（teacher forcing）还是模型上一步预测（闭环），弥合 train/inference gap

        batch（episode 模式）:
          action_windows: (B, T, seq_len, D)
          gt_skeletons:   (B, T, N, 3)
          init_skeleton:  (B, N, 3)
        """
        import random
        device = self.device
        action_windows = batch["action_windows"].to(device)   # (B, T, K, D)
        gt_skeletons = batch["gt_skeletons"].to(device)        # (B, T, N, 3)
        init_skeleton = batch["init_skeleton"].to(device)      # (B, N, 3)

        T = action_windows.shape[1]
        tf_ratio = getattr(phase_spec, "teacher_forcing_ratio", 0.5)

        # 构建 scheduled-sampling 的 teacher_states：逐步决定该步 prev 用 GT 还是空（闭环）。
        # model.forward_sequence 接收 teacher_states：非 None 时下一步 prev = GT。
        # 为实现 per-step mixing，这里按步生成 teacher mask 序列：mask[t]=True 表示
        # "第 t 步预测后，下一步的 prev 用 GT"。
        losses = {}
        total = 0.0
        z_t = self.model.init_z_from_action(action_windows[:, 0])
        s_prev = init_skeleton
        s_prev_prev = init_skeleton
        preds = []

        for t in range(T):
            out = self.model.forward(
                action_windows[:, t], s_prev, s_prev_prev, z_t)
            s_pred = out["skeleton"]
            z_t = out["latent_z"]
            preds.append(s_pred)

            # scheduled sampling：决定下一步的 prev_skeleton
            use_teacher = random.random() < tf_ratio
            s_prev_prev = s_prev
            s_prev = gt_skeletons[:, t] if use_teacher else s_pred

        pred_seq = torch.stack(preds, dim=1)  # (B, T, N, 3)

        # 逐步 MSE（skeleton）+ 空间平滑（用 torch 原生，与本文件不含 F 的风格一致）
        if "skeleton" in phase_spec.active_losses:
            losses["skeleton"] = ((pred_seq - gt_skeletons) ** 2).mean()
        if "spatial_smooth" in phase_spec.active_losses:
            pd = pred_seq[:, :, 1:, :] - pred_seq[:, :, :-1, :]
            gd = gt_skeletons[:, :, 1:, :] - gt_skeletons[:, :, :-1, :]
            losses["spatial_smooth"] = ((pd - gd) ** 2).mean()

        losses["total"] = sum(losses.values())
        return losses

    def _build_exp_config(self, data_dirs, n_epochs_per_phase):
        """构建完整的实验配置 dict，用于保存到 config.json。"""
        opt_cfg = self.config.get("optimization", {})
        model = self.model
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        phases_info = []
        for p in self.phase.spec.phases:
            p_info = {
                "name": p.name,
                "forward_attr": p.forward_attr,
                "supervision_mode": p.supervision_mode,
                "dataset_type": p.dataset_type,
                "data_mode": p.data_mode,
                "active_losses": p.active_losses,
                "freeze_modules": p.freeze_modules,
                "use_gt_skeleton": getattr(p, 'use_gt_skeleton', False),
                "trained": False,
            }
            if p.lr:
                p_info["lr"] = p.lr
            if p.save_modules:
                p_info["save_modules"] = p.save_modules
            if p.load_modules:
                p_info["load_modules"] = p.load_modules
            phases_info.append(p_info)

        config = {
            "model": type(model).__name__,
            "model_tag": self.model_tag,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "action_dim": getattr(model, 'action_dim', None),
            "window_size": self.config.get("temporal", {}).get("window_size"),
            "hidden_dim": self.config.get("temporal", {}).get("hidden_dim"),
            "n_scales": self.config.get("temporal", {}).get("n_scales"),
            "device": str(self.device),
            "phases": phases_info,
            "training": {
                "lr": opt_cfg.get("lr"),
                "batch_size": opt_cfg.get("batch_size"),
                "n_epochs": opt_cfg.get("n_epochs"),
                "optimizer": "Adam",
                "scheduler": "ReduceLROnPlateau",
                "scheduler_patience": opt_cfg["scheduler_patience"],
            },
            "loss_weights": self.config.get("loss_weights", {}),
            "data_dirs": {k: str(v) for k, v in data_dirs.items()},
            "view_strategy": type(self.views).__name__ if self.views else None,
        }

        # 模型特有参数
        for attr in ('skeleton_mode', 'rod_radius', 'd_filter', 'n_freqs',
                     'n_fine', 'n_medium', 'n_coarse', 'deform_n_freqs',
                     'fourier_n_freq', 'bspline_n_ctrl', 'catmullrom_n_ctrl',
                     'encoder_type'):
            val = getattr(model, attr, None)
            if val is not None:
                config[attr] = val

        # 多视角参数
        mv_cfg = self.config.get("multiview", {})
        if mv_cfg:
            config["multiview"] = mv_cfg

        # SDF 参数
        sdf_cfg = self.config.get("sdf", {})
        if sdf_cfg:
            config["sdf"] = sdf_cfg

        return config

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

    def _update_config_phase_trained(self, exp_dir, phase_name, best_loss):
        """Phase 完成后更新 config.json，标记 trained=true 并记录最终 loss。"""
        import json
        config_path = os.path.join(exp_dir, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for p in cfg.get("phases", []):
            if p["name"] == phase_name:
                p["trained"] = True
                p["best_loss"] = best_loss
                break
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

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
        opt_cfg = self.config["optimization"]
        return DataLoader(ds, batch_size=opt_cfg["batch_size"], shuffle=True,
                          num_workers=opt_cfg["num_workers"], collate_fn=collate_fn), ds

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

    def train(self, data_dirs, exp_dir=None, n_epochs_per_phase=None,
              skip_phases=None):
        """统一训练入口。

        Args:
            data_dirs: dict, key 为 data_mode ("canonical"/"sequence"), value 为路径
            exp_dir: 实验日志目录
            n_epochs_per_phase: dict, key 为 phase name, value 为 epoch 数
            skip_phases: list[str], 要跳过的阶段名列表
        """
        self.device = next(self.model.parameters()).device
        if self.views:
            self.views.setup(self.device, self.config)

        opt_cfg = self.config["optimization"]

        if exp_dir is None:
            exp_config = self._build_exp_config(data_dirs, n_epochs_per_phase)
            exp_dir = create_experiment(f"train_log/{self.model_tag}", exp_config)

        saved_modules_by_phase = {}

        for phase_idx, phase_spec in self.phase.iterate_phases():
            if skip_phases and phase_spec.name in skip_phases:
                print(f"  Skipping phase '{phase_spec.name}'")
                continue

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
                optimizer, patience=opt_cfg["scheduler_patience"])

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

            # 初始化 loss log CSV（第一个 epoch 后根据实际 loss 名写 header）
            csv_path = os.path.join(phase_dir, "loss_log.csv")
            csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            csv_header_written = False

            for epoch in range(1, n_epochs + 1):
                self.model.train()
                self._current_epoch = epoch
                epoch_loss = 0
                epoch_details = {}
                n_batches = 0

                # reproj/consist 课程式 warmup
                warmup_epochs = self.config.get("multiview", {}).get("warmup_epochs", 0)
                self._warmup_factor = min(1.0, epoch / max(warmup_epochs, 1)) if warmup_epochs > 0 else 1.0

                pbar = tqdm(loader, desc=f"[{phase_spec.name}] Epoch {epoch}/{n_epochs}")
                for batch in pbar:
                    # Stage 1 episode 模式走序列级损失（z 跨帧演化 + scheduled sampling）
                    if getattr(phase_spec, 'use_episode_mode', False):
                        losses = self._compute_sequence_losses(batch, phase_spec)
                    else:
                        forward_fn = self.phase.get_forward_fn()
                        losses = self._compute_losses(forward_fn, batch, phase_spec)

                    # warmup 缩放跨视角 loss
                    if self._warmup_factor < 1.0:
                        for key in ("reproj", "consist"):
                            if key in losses:
                                losses[key] = losses[key] * self._warmup_factor
                        losses["total"] = sum(v for k, v in losses.items() if k != "total")

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

                # 写入 loss log CSV
                current_lr = optimizer.param_groups[0]["lr"]
                loss_names = sorted(epoch_details.keys())
                if not csv_header_written:
                    header = ["epoch", "total"] + loss_names + ["lr"]
                    csv_writer = csv.DictWriter(csv_file, fieldnames=header)
                    csv_writer.writeheader()
                    csv_header_written = True
                csv_row = {"epoch": epoch, "total": f"{avg:.6f}", "lr": f"{current_lr:.8f}"}
                for k, v in epoch_details.items():
                    csv_row[k] = f"{v / max(n_batches, 1):.6f}"
                csv_writer.writerow(csv_row)
                csv_file.flush()

                # 每 10 epoch 或最后一个 epoch 打印编码器参数
                temporal = getattr(self.model, "temporal", None)
                if temporal is not None and (epoch % 10 == 0 or epoch == 1 or epoch == n_epochs):
                    # GammaLaguerre 编码器：打印 k（阶次/峰值延迟）和 λ（衰减率）
                    if hasattr(temporal, "ks") and hasattr(temporal, "lambdas"):
                        ks = temporal.ks.detach().cpu().numpy()
                        lambdas = temporal.lambdas.detach().cpu().numpy()
                        print(f"    Gamma kernels:")
                        for i in range(len(ks)):
                            peak = (ks[i] - 1) / max(-np.log(max(lambdas[i], 0.01)), 0.01)
                            print(f"      [{i}] k={ks[i]:.2f}, λ={lambdas[i]:.3f}, peak≈{peak:.1f} frames")
                    elif hasattr(temporal, "alphas") and hasattr(temporal, "_compute_gl_weights"):
                        # FractionalMemory
                        a = temporal.alphas.detach().cpu().numpy()
                        print(f"    alphas: {[round(x, 4) for x in a]}")
                    elif hasattr(temporal, "cls_token"):
                        # TemporalTransformer
                        cls_norm = temporal.cls_token.norm().item()
                        print(f"    Transformer: CLS norm={cls_norm:.4f}, heads={temporal.n_heads}, layers={temporal.n_layers}")
                    elif hasattr(temporal, "tcn_layers"):
                        # TemporalTCN
                        n_l = len(temporal.tcn_layers)
                        print(f"    TCN: {n_l} dilated conv layers")
                    elif hasattr(temporal, "decays"):
                        # MultiScaleEMA or TemporalGRU (synthetic decays)
                        d = temporal.decays.detach().cpu().numpy()
                        enc_name = type(temporal).__name__
                        print(f"    {enc_name}: decays={[round(x, 4) for x in d]}")

                if avg < best_val:
                    best_val = avg
                    torch.save(self.model.state_dict(),
                               os.path.join(phase_dir, "model", "best_model.pt"))

                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg)
                else:
                    scheduler.step()

                # 形状评估（每 eval_interval epoch）
                eval_interval = self.config.get("evaluation", {}).get("eval_interval", 0)
                if eval_interval > 0 and (epoch % eval_interval == 0 or epoch == n_epochs):
                    evaluate_shape_during_training(
                        self.model, self.model_tag, self.config,
                        self.device, phase_spec.name, data_dir, epoch, exp_dir)
                    evaluate_skeleton_during_training(
                        self.model, self.model_tag, self.config,
                        self.device, phase_spec.name, data_dir, epoch, exp_dir)

            # 保存 Phase 权重
            csv_file.close()
            save_path = self._save_phase_modules(phase_dir, phase_spec)
            if save_path:
                saved_modules_by_phase[phase_spec.name] = {
                    mod_name: getattr(self.model, mod_name).state_dict()
                    for mod_name in phase_spec.save_modules
                }

            torch.save(self.model.state_dict(),
                       os.path.join(phase_dir, "model", "final_model.pt"))

            # 更新 config.json 标记该 phase 已训练
            self._update_config_phase_trained(exp_dir, phase_spec.name, best_val)

        print(f"\n训练完成! 日志: {exp_dir}")
        return exp_dir
