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


def anchor_from_camera_frame(
        bgr, *, background_gray, segment_params: dict, n_nodes: int, model,
        action_history, area_median_px: float,
        prev_skeleton=None, frame_age_s: float | None = None,
        registration_displacement_px: float | None = None,
        frame_ref: str = "", state_space: str = "model_normalized",
        action_units: str = "model_normalized", source: str = "camera_live"):
    """单帧 BGR → (Anchor, FrameQuality, skeleton_px)。

    质量门 verdict == "reject" 时返回 (None, quality, skeleton_px)—— 调用方不得上锚。
    skeleton_px 是 (n_nodes,2) [col,row],供 GUI 叠加显示。

    area_median_px 必须显式提供(quality.QualityThresholds 无默认值;来自
    deploy_manifest.mask_area_median_px)。
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
    if not history or any(len(action) != len(history[0]) for action in history):
        raise ValueError("action_history 必须是 (H, action_dim) 且宽度一致")

    # 归一化(与 offline_anchor 同款):(skeleton[:, :2] - pc_center[:2]) / pc_scale[:2]
    try:
        center = model.pc_center.detach().cpu().numpy().reshape(3)
        scale = model.pc_scale.detach().cpu().numpy().reshape(3)
    except (AttributeError, ValueError) as error:
        raise ValueError("模型缺少可用 pc_center/pc_scale") from error
    if np.any(np.abs(scale[:2]) < 1e-12):
        raise ValueError("模型 pc_scale 平面尺度含零")
    normalized = (skeleton[:, :2] - center[:2]) / scale[:2]

    anchor = Anchor(
        state=tuple(tuple(float(v) for v in node) for node in normalized),
        action_history=history,
        prev_state=(None if prev_skeleton is None
                    else tuple(tuple(float(v) for v in node)
                               for node in (np.asarray(prev_skeleton, dtype=np.float64)
                                            - center[:2]) / scale[:2])),
        frame_id="model_normalized",
        frame_ref=frame_ref,
        state_space=state_space,
        action_units=action_units,
        source=source,
        quality={**quality.flags, "verdict": quality.verdict, "kind": "camera_live"},
    )
    return anchor, quality, skeleton
