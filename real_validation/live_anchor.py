"""从实时相机帧建立模型状态锚点(免标定 2D 路线)。

链:分割 → 骨架(tip_fix)→ 质量门(拒帧不进模型)→ 归一化 → Anchor。

与 offline_anchor.anchor_from_npz 的区别:offline 输入是已落盘 npz(像素骨架直接读);
live 输入是单帧 BGR,必须现场做分割+骨架+质量门。两者输出同样的 Anchor 契约
(state/action_history/prev_state/quality/state_space/action_units)。

单位约定:action_history 必须是**归一化域**(npz 的 [0,1],或 kPa 经 units.kPa_to_model
换算后),宽度 = action_dim。action_units = "model_normalized"。
"""

from __future__ import annotations

import numpy as np

from .models import Anchor
from .perception.quality import QualityThresholds, assess_frame
from .perception.segmentation import segment_white_on_blue
from .perception.skeleton import extract_skeleton_2d

# 分割/骨架所需 cv2/scipy 由 perception 子模块内部处理;本模块只依赖 numpy 与上述调用。


def _model_history_steps(model) -> int:
    """模型动作历史窗口长度(先 temporal.window_size,再 history_steps,兜底 40)。"""
    temporal = getattr(model, "temporal", None)
    window = getattr(temporal, "window_size", None)
    if window:
        return int(window)
    return int(getattr(model, "history_steps", 40) or 40)


def anchor_from_camera_frame(
        bgr, *, background_gray, segment_params: dict, n_nodes: int, model,
        action_history, area_median_px: float,
        prev_skeleton=None, frame_age_s: float | None = None,
        registration_displacement_px: float | None = None,
        frame_ref: str = "", state_space: str = "model_normalized",
        action_units: str = "model_normalized", source: str = "camera_live",
        zero_pad_history: bool = False):
    """单帧 BGR → (Anchor, FrameQuality, skeleton_px)。

    质量门 verdict == "reject" 时返回 (None, quality, skeleton_px)—— 调用方不得上锚。
    skeleton_px 是 (n_nodes,2) [col,row],供 GUI 叠加显示。

    area_median_px 必须显式提供(quality.QualityThresholds 无默认值;来自
    deploy_manifest.mask_area_median_px)。

    zero_pad_history:action_history 为空/不足时,是否零填充到完整 H 步(模型
    history_steps 从 model 的 config 推断)。⚠️ 模型训练从没见过零填充窗口,
    零填充起步是 OOD(预测可能不准),只在操作员明确接受时开启(GUI 需标注)。
    """
    mask = segment_white_on_blue(bgr, background_gray, **segment_params)
    skeleton, info = extract_skeleton_2d(mask, n_nodes, tip_fix=True, return_info=True)
    thresholds = QualityThresholds(float(area_median_px))
    quality = assess_frame(mask, skeleton, info, thresholds,
                           prev_skeleton=prev_skeleton, frame_age_s=frame_age_s,
                           registration_displacement_px=registration_displacement_px)

    if quality.verdict == "reject":
        return None, quality, skeleton

    history = tuple(tuple(float(v) for v in action) for action in action_history)
    if history and any(len(action) != len(history[0]) for action in history):
        raise ValueError("action_history 必须是 (H, action_dim) 且宽度一致")
    history_steps = _model_history_steps(model)
    action_dim = getattr(model, "action_dim", None)
    if action_dim is None and history:
        action_dim = len(history[0])
    if not history:
        if not zero_pad_history:
            raise ValueError("action_history 为空;开启 zero_pad_history 可用全 0 历史起步")
        if action_dim is None:
            raise ValueError("zero_pad_history 需要模型暴露 action_dim")
        history = ((0.0,) * action_dim,) * history_steps
    elif zero_pad_history and len(history) < history_steps:
        # 部分历史 + 零填充到完整 H(运行几步后累积的真实历史 + 前缀零)
        pad = ((0.0,) * len(history[0]),) * (history_steps - len(history))
        history = pad + history

    # 归一化(与 offline_anchor 同款;live 只取平面 [:2],pc_scale[2]=1e-6 退化)
    from .anchor_utils import float_rows, model_normalization, normalize_rows
    center, scale = model_normalization(model)
    dims = slice(0, 2)
    normalized = normalize_rows(skeleton, center, scale, dims=dims)
    prev_norm = (None if prev_skeleton is None
                 else normalize_rows(prev_skeleton, center, scale, dims=dims))

    anchor = Anchor(
        state=float_rows(normalized),
        action_history=history,
        prev_state=(None if prev_norm is None else float_rows(prev_norm)),
        frame_id="model_normalized",
        frame_ref=frame_ref,
        state_space=state_space,
        action_units=action_units,
        source=source,
        quality={**quality.flags, "verdict": quality.verdict, "kind": "camera_live"},
    )
    return anchor, quality, skeleton
