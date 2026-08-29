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

from ..contracts.models import Anchor
from ..perception.quality import QualityThresholds, assess_frame
from ..perception.segmentation import (segment_backlight, segment_white_on_blue,
                                       trim_wide_base_attachment)
from ..perception.skeleton import extract_centerline_2d, extract_skeleton_2d

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
        zero_pad_history: bool = False,
        segmentation_method: str = "white_on_blue",
        state_coordinate_frame: str = "camera_pixel_v1",
        robot_diameter_mm: float | None = None,
        frame_transform=None, roi_xywh=None,
        skeleton_method: str = "row_centroid",
        segment_lengths=(1.0, 1.0), base_anchor_camera_xy=None,
        quality_params: dict | None = None,
        validation_setup_id: str | None = None,
        base_side: str = "none"):
    """单帧 BGR → (Anchor, FrameQuality, skeleton_px)。

    质量门 verdict == "reject" 时返回 (None, quality, skeleton_px)—— 调用方不得上锚。
    skeleton_px 是 (n_nodes,2) [col,row],供 GUI 叠加显示。

    ``area_median_px`` 由本次验证配置提供。首次现场 Anchor 可传 ``None``，以当前
    通过几何检查的 mask 面积建立 run 内参考；后续 Anchor 复用该参考。

    zero_pad_history:action_history 为空/不足时,是否零填充到完整 H 步(模型
    history_steps 从 model 的 config 推断)。⚠️ 模型训练从没见过零填充窗口,
    零填充起步是 OOD(预测可能不准),只在操作员明确接受时开启(GUI 需标注)。
    """
    from ..perception.roi import (camera_to_roi_local, crop_frame,
                                  roi_local_to_camera)
    frame = np.asarray(bgr)
    if roi_xywh is None:
        roi_xywh = (0, 0, frame.shape[1], frame.shape[0])
    local_frame = crop_frame(frame, roi_xywh)
    local_background = (None if background_gray is None else
                        crop_frame(background_gray, roi_xywh))
    if segmentation_method == "backlight":
        gray = local_frame.mean(axis=2).astype(np.uint8)
        mask = segment_backlight(gray, thresh=int(segment_params.get("thresh", 60)))
    elif segmentation_method == "white_on_blue":
        if local_background is None:
            raise ValueError("white_on_blue 在线锚定缺少参考背景")
        mask = segment_white_on_blue(local_frame, local_background, **segment_params)
        mask = trim_wide_base_attachment(mask, base_side=base_side)
    else:
        raise ValueError(f"在线锚定尚不支持分割方法 {segmentation_method}")
    base_anchor_local = (None if base_anchor_camera_xy is None else
                         camera_to_roi_local([base_anchor_camera_xy], roi_xywh)[0])
    if skeleton_method == "row_centroid":
        skeleton_local, info = extract_skeleton_2d(
            mask, n_nodes, tip_fix=True, return_info=True)
    else:
        skeleton_local, info = extract_centerline_2d(
            mask, n_nodes, method=skeleton_method,
            segment_lengths=segment_lengths,
            base_anchor_xy=base_anchor_local,
            endpoint_fix=True, return_info=True)
        info = {
            **info,
            "tip_fix_requested": bool(info.get("endpoint_fix_requested", True)),
            "tip_fix_applied": bool(info.get("tip_endpoint_fix_applied", False)),
            "tip_fix_reason": str(info.get("tip_endpoint_fix_reason", "")),
            "n_valid_rows": int(info.get("n_medial_pixels", 0)),
        }
    skeleton = roi_local_to_camera(skeleton_local, roi_xywh)

    model_skeleton = skeleton
    transform = frame_transform
    if state_coordinate_frame == "robot_planar_mm_v1":
        from ..perception.coordinates import (
            SkeletonFrameTransform,
            estimate_body_diameter_px,
            estimate_skeleton_frame,
        )
        diameter_mm = float(robot_diameter_mm or 0.0)
        if diameter_mm <= 0:
            raise ValueError("robot_planar_mm_v1 在线锚定需要 robot_diameter_mm")
        if transform is not None and not isinstance(transform, SkeletonFrameTransform):
            transform = SkeletonFrameTransform.from_dict(transform)
        if transform is None:
            diameter_px = estimate_body_diameter_px(mask, skeleton_local)
            transform = estimate_skeleton_frame(
                skeleton, robot_diameter_mm=diameter_mm,
                robot_diameter_px=diameter_px,
                source="live_anchor_base_tangent_and_body_width")
        model_skeleton = transform.camera_to_model(skeleton)
    elif state_coordinate_frame != "camera_pixel_v1":
        raise ValueError(f"在线锚定不支持状态坐标 {state_coordinate_frame}")

    reference_area = float(area_median_px or np.count_nonzero(mask))
    thresholds = QualityThresholds(reference_area, **(quality_params or {}))
    previous_local = (None if prev_skeleton is None else
                      camera_to_roi_local(prev_skeleton, roi_xywh))
    quality = assess_frame(mask, skeleton_local, info, thresholds,
                           prev_skeleton=previous_local, frame_age_s=frame_age_s,
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

    # 模型坐标归一化；相机像素骨架继续作为独立显示输出。
    from .anchor_utils import float_rows, model_normalization, normalize_rows
    center, scale = model_normalization(model)
    dims = slice(0, 2)
    normalized = normalize_rows(model_skeleton, center, scale, dims=dims)
    if prev_skeleton is None:
        prev_norm = None
    else:
        prev_model = (transform.camera_to_model(prev_skeleton)
                      if transform is not None else prev_skeleton)
        prev_norm = normalize_rows(prev_model, center, scale, dims=dims)

    coordinate_quality = {
        "state_coordinate_frame": state_coordinate_frame,
        "state_length_unit": "mm" if state_coordinate_frame == "robot_planar_mm_v1" else "px",
        "roi_xywh": [int(value) for value in roi_xywh],
        "skeleton_method": skeleton_method,
    }
    if validation_setup_id:
        coordinate_quality["validation_setup_id"] = validation_setup_id
    if transform is not None:
        coordinate_quality["skeleton_frame_transform"] = transform.to_dict()

    anchor = Anchor(
        state=float_rows(normalized),
        action_history=history,
        prev_state=(None if prev_norm is None else float_rows(prev_norm)),
        frame_id="model_normalized",
        frame_ref=frame_ref,
        state_space=state_space,
        action_units=action_units,
        node_order="base_to_tip",
        source=source,
        quality={**quality.flags, **coordinate_quality,
                 "verdict": quality.verdict, "kind": "camera_live"},
    )
    return anchor, quality, skeleton
