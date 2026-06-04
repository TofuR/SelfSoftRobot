"""spec.py — 模型训练需求声明。

模型通过类属性 training_spec 声明自己需要几个训练阶段、每个阶段冻结什么、
用什么 forward 方法、启用哪些 loss、使用什么监督模式和数据集。

三种监督模式:
  "rendering"  — 射线采样 → 体渲染 → 像素对比（recon, depth）
  "direct_3d"  — 3D 坐标查询 → 值对比（SDF, normal, eikonal）
  "skeleton"   — action → 预测骨架 → 骨架对比（无空间查询）
  "pointcloud" — action → velocity field ODE → 点云 → FM/CD loss

Loss 分两层:
  渲染层 (ViewStrategy): recon, depth, reproj, consist
  模型层 (model.compute_losses): smooth, skeleton, sdf, normal, eikonal, ...
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhaseSpec:
    """单个训练阶段配置。

    Attributes:
        name: 阶段名称，用于日志和权重保存（如 "canonical", "deformation"）
        freeze_modules: 该阶段冻结的子模块名列表（如 ["deform", "density"]）
        forward_attr: 该阶段使用的 forward 方法名（如 "forward_canonical"）
        data_mode: 数据类型 — "canonical"（单帧静态）| "sequence"（时序）
        dataset_type: 数据集类型 — "sequence" | "multiview_depth" | "sdf" | "skeleton_sdf"
        supervision_mode: 监督模式 — "rendering" | "direct_3d" | "skeleton"
        lr: 学习率覆盖（None 表示用 config 默认）
        active_losses: 该阶段启用的 loss 名列表
        dataset_kwargs: 传给数据集构造器的额外参数
        save_modules: 阶段结束时保存的子模块名列表
        load_modules: 阶段开始时从前面阶段加载的子模块 {"module_name": "prev_phase_name"}
    """
    name: str
    freeze_modules: list[str] = field(default_factory=list)
    forward_attr: str = "forward"
    data_mode: str = "sequence"
    dataset_type: str = "sequence"
    supervision_mode: str = "rendering"
    lr: Optional[float] = None
    active_losses: list[str] = field(default_factory=lambda: ["recon", "smooth"])
    dataset_kwargs: dict = field(default_factory=dict)
    save_modules: list[str] = field(default_factory=list)
    load_modules: dict[str, str] = field(default_factory=dict)
    use_gt_skeleton: bool = False


@dataclass
class TrainingSpec:
    """模型的完整训练需求声明。

    Attributes:
        phases: 训练阶段列表（单元素=单阶段，多元素=多阶段）
        supports_smoothness: 模型是否支持 smoothness loss
    """
    phases: list[PhaseSpec]
    supports_smoothness: bool = True

    @property
    def is_two_phase(self) -> bool:
        return len(self.phases) > 1

    @property
    def needs_canonical_data(self) -> bool:
        return any(p.data_mode == "canonical" for p in self.phases)
