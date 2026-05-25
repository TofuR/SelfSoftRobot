"""mixins.py — 可复用的模型方法混入（Mixin）。

什么是 Mixin？
  一种通过多重继承"混入"共享方法的技巧。模型类只需要声明继承 TemporalMixin
  或 SkeletonMixin，就自动获得对应的方法，无需重复实现。

使用前提：
  TemporalMixin — 模型必须有 self.temporal (MultiScaleEMA 实例)
  SkeletonMixin — 模型必须有 self.skeleton_head (SkeletonHead 实例)
                   以及 self.skeleton_mode (str), self.n_fine (int)

使用方式：
  class MyModel(nn.Module, TemporalMixin, SkeletonMixin):
      def __init__(self, ...):
          self.temporal = MultiScaleEMA(...)       # TemporalMixin 依赖
          self.skeleton_head = create_skeleton_head(...)  # SkeletonMixin 依赖
          self.skeleton_mode = "bspline"
          self.n_fine = 31

  model = MyModel(action_dim=2)
  state = model.encode(action_window)          # 来自 TemporalMixin
  loss = model.compute_smoothness(seq_t, seq_t1)  # 来自 TemporalMixin
  pred = model.predict_skeleton(action_window)  # 来自 SkeletonMixin
"""

import torch

from src.heads.skeleton_heads import downsample_skeleton


class TemporalMixin:
    """时序编码相关方法混入。

    提供方法，供 Trainer 调用：
      encode()            — 将动作窗口编码为物理状态向量
      compute_smoothness() — 计算相邻两帧的时序平滑损失
      get_learned_decays() — 返回 EMA 学到的衰减率（用于分析/可视化）
      compute_losses()     — 默认的模型层 loss 计算（处理 "smooth"）

    前置条件：self.temporal 必须是 MultiScaleEMA 实例。
    """

    def encode(self, action_window):
        """将动作窗口编码为物理状态向量。

        委托给 self.temporal (MultiScaleEMA)，计算多尺度 EMA 特征后经 MLP 映射。

        Args:
            action_window: (B, window_size, action_dim) 动作序列窗口。
                           B=批次大小, window_size=时序窗口长度, action_dim=动作维度

        Returns:
            physics_state: (B, hidden_dim) 物理状态向量。
        """
        return self.temporal(action_window)

    def compute_smoothness(self, action_windows_t, action_windows_t1):
        """计算相邻两帧物理状态之间的时序平滑损失 (L2)。

        约束：相邻时间步的物理状态不应突变，保证运动的连续性。

        Args:
            action_windows_t:  (B, window_size, action_dim) 时间步 t 的动作窗口
            action_windows_t1: (B, window_size, action_dim) 时间步 t+1 的动作窗口

        Returns:
            smooth_loss: 标量，mean((state_t1 - state_t) ** 2)
        """
        return self.temporal.compute_smoothness(action_windows_t, action_windows_t1)

    def get_learned_decays(self):
        """返回 EMA 学到的衰减率（用于分析/可视化）。

        Returns:
            numpy array, shape (n_scales,), 值在 (0, 1) 范围内。
            例如 [0.20, 0.45, 0.70, 0.95] 分别对应快/中/慢/极慢衰减。
        """
        return self.temporal.decays.detach().cpu().numpy()

    def compute_losses(self, batch: dict, phase_spec) -> dict:
        """默认的模型层 loss 计算。处理 "smooth" loss。

        子类可覆盖此方法以添加更多模型特定 loss（skeleton, sdf 等）。
        建议子类调用 super().compute_losses(batch, phase_spec) 后追加自己的 loss。

        Args:
            batch: 统一 dict batch（由 dataset_factory 的 collate 函数生成）。
            phase_spec: 当前 PhaseSpec，通过 active_losses 判断要计算哪些 loss。

        Returns:
            dict[str, torch.Tensor]: loss 名到标量 Tensor 的映射。
        """
        losses = {}
        active = set(phase_spec.active_losses)

        if "smooth" in active and "action_window_next" in batch and batch["action_window_next"] is not None:
            aw_t = batch["action_window"].to(self.temporal.decays.device)
            aw_t1 = batch["action_window_next"].to(self.temporal.decays.device)
            losses["smooth"] = self.compute_smoothness(aw_t, aw_t1)

        return losses


class SkeletonMixin:
    """骨架预测与损失相关方法混入。

    提供三个方法：
      predict_skeleton()     — 从动作窗口预测 3D 骨架曲线（多尺度）
      compute_skeleton_loss() — 计算预测骨架与 GT 的多尺度 L2 损失
      skeleton_config()      — 返回骨架配置 dict（用于保存到实验日志）

    前置条件：
      self.temporal      — MultiScaleEMA 实例（通过 TemporalMixin.encode 使用）
      self.skeleton_head  — SkeletonHead 实例（通过 create_skeleton_head 创建）
      self.skeleton_mode  — str, 骨架模式 ("point"/"fourier"/"bspline"/"catmullrom")
      self.n_fine         — int, 精细骨架节点数

    继承顺序：通常同时继承 TemporalMixin，因为 predict_skeleton 内部调用 encode()。
    """

    def predict_skeleton(self, action_window):
        """预测多尺度 3D 骨架曲线。

        流程：action_window → encode() → physics_state → skeleton_head → 多尺度骨架

        Args:
            action_window: (B, window_size, action_dim) 动作序列窗口

        Returns:
            dict, 包含三个尺度的骨架坐标：
              'fine':   (B, n_fine, 3)   — 精细骨架（如 31 个节点）
              'medium': (B, n_medium, 3) — 中等骨架（如 10 个节点）
              'coarse': (B, n_coarse, 3) — 粗糙骨架（如 4 个节点）
        """
        return self.skeleton_head(self.encode(action_window))

    def compute_skeleton_loss(self, pred_dict, gt_positions):
        """计算多尺度骨架 L2 损失。

        将 GT 骨架下采样到与预测匹配的尺度，然后计算每个尺度的 MSE。

        Args:
            pred_dict:  predict_skeleton() 的返回值，包含 'fine'/'medium'/'coarse'
            gt_positions: (B, 3, N) 或 (B, N, 3) GT 骨架坐标（仿真器提供）

        Returns:
            dict, 包含三个尺度的损失：
              'fine':   标量, MSE(pred_fine, gt_fine)
              'medium': 标量, MSE(pred_medium, gt 下采样到 n_medium)
              'coarse': 标量, MSE(pred_coarse, gt 下采样到 n_coarse)
        """
        losses = {}
        losses['fine'] = ((pred_dict['fine'] - gt_positions) ** 2).mean()
        gt_medium = downsample_skeleton(gt_positions, pred_dict['medium'].shape[-2])
        losses['medium'] = ((pred_dict['medium'] - gt_medium) ** 2).mean()
        gt_coarse = downsample_skeleton(gt_positions, pred_dict['coarse'].shape[-2])
        losses['coarse'] = ((pred_dict['coarse'] - gt_coarse) ** 2).mean()
        return losses

    def skeleton_config(self):
        """返回骨架配置 dict，用于保存到实验日志。

        Returns:
            dict, 至少包含:
              'skeleton_mode': 骨架模式名称
              'n_fine': 精细节点数
            可选包含:
              'rod_radius': 杆件半径（如果 hasattr(self, 'rod_radius')）
              'fourier_n_freq' / 'bspline_n_ctrl' / 'catmullrom_n_ctrl':
                  模式特有的参数
        """
        cfg = {"skeleton_mode": self.skeleton_mode, "n_fine": self.n_fine}
        if hasattr(self, 'rod_radius'):
            cfg["rod_radius"] = self.rod_radius
        if self.skeleton_mode == "fourier":
            cfg["fourier_n_freq"] = self.skeleton_head.n_freq
        elif self.skeleton_mode == "bspline":
            cfg["bspline_n_ctrl"] = self.skeleton_head.n_ctrl
        elif self.skeleton_mode == "catmullrom":
            cfg["catmullrom_n_ctrl"] = self.skeleton_head.n_ctrl
        return cfg
