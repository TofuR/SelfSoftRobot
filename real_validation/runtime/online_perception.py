"""验证实验配置驱动的在线分割、骨架与可审计产物。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts.io import atomic_write_json
from ..contracts.validation_setup import ValidationSetup
from ..perception.roi import (camera_to_roi_local, crop_frame,
                              roi_local_to_camera)
from ..perception.segmentation import (segment_backlight,
                                       segment_white_on_blue_stages,
                                       trim_wide_base_attachment)
from ..perception.skeleton import extract_centerline_2d


def gray_image(bgr) -> np.ndarray:
    return np.mean(np.asarray(bgr, dtype=np.float64), axis=2).astype(np.uint8)


def load_background(setup: ValidationSetup, base_dir=None) -> np.ndarray | None:
    if setup.segmentation_method == "backlight":
        return None
    from ..perception.background import load_median_background
    path = Path(setup.background_path)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return load_median_background(path)


def segmentation_stages_with_setup(frame, setup: ValidationSetup,
                                   background_gray=None) -> dict[str, np.ndarray]:
    local = crop_frame(frame, setup.roi_xywh)
    if setup.segmentation_method == "backlight":
        final = segment_backlight(
            gray_image(local), int(setup.segment_params.get("thresh", 60)))
        return {"final": final}
    if background_gray is None:
        background_gray = load_background(setup)
    background = crop_frame(background_gray, setup.roi_xywh)
    stages = segment_white_on_blue_stages(
        local, background, **setup.segment_params)
    stages["pretrim"] = stages["final"].copy()
    stages["final"] = trim_wide_base_attachment(
        stages["pretrim"], base_side=setup.base_side)
    return stages


def segment_with_setup(frame, setup: ValidationSetup,
                       background_gray=None) -> np.ndarray:
    return segmentation_stages_with_setup(
        frame, setup, background_gray)["final"]


def extract_skeleton_with_setup(frame, setup: ValidationSetup, *, n_nodes: int,
                                background_gray=None):
    mask = segment_with_setup(frame, setup, background_gray)
    skeleton_local, info = extract_centerline_2d(
        mask, n_nodes, method="skeletonize",
        segment_lengths=setup.segment_lengths,
        endpoint_fix=True, return_info=True)
    skeleton_camera = roi_local_to_camera(skeleton_local, setup.roi_xywh)
    return skeleton_camera, mask, info


def save_observation_artifacts(run_dir, label: str, frame,
                               setup: ValidationSetup, skeleton_camera,
                               quality, *, background_gray=None) -> Path | None:
    from ..perception._compat import cv2
    if cv2 is None:
        return None
    directory = Path(run_dir) / "perception" / "observations" / label
    directory.mkdir(parents=True, exist_ok=True)
    values = np.asarray(frame)
    stages = segmentation_stages_with_setup(values, setup, background_gray)
    mask = stages["final"]
    roi_frame = crop_frame(values, setup.roi_xywh)
    skeleton_local = np.rint(camera_to_roi_local(
        skeleton_camera, setup.roi_xywh)).astype(np.int32)
    overlay = roi_frame.copy()
    camera_roi = values.copy()
    x, y, width, height = setup.roi_xywh
    cv2.rectangle(camera_roi, (x, y), (x + width - 1, y + height - 1),
                  (0, 200, 0), 2)
    if len(skeleton_local) > 1:
        cv2.polylines(overlay, [skeleton_local], False, (0, 255, 255), 2,
                      lineType=cv2.LINE_AA)
    for point in skeleton_local:
        cv2.circle(overlay, tuple(point), 3, (0, 255, 255), -1)
    cv2.imwrite(str(directory / "camera_frame.png"), values)
    cv2.imwrite(str(directory / "camera_roi_overlay.png"), camera_roi)
    cv2.imwrite(str(directory / "roi_frame.png"), roi_frame)
    cv2.imwrite(str(directory / "mask.png"), (mask > 0).astype(np.uint8) * 255)
    cv2.imwrite(str(directory / "skeleton_overlay.png"), overlay)
    stages_dir = directory / "segmentation_stages"
    stages_dir.mkdir(exist_ok=True)
    for name, stage in stages.items():
        cv2.imwrite(str(stages_dir / f"{name}.png"),
                    (np.asarray(stage) > 0).astype(np.uint8) * 255)
    atomic_write_json(directory / "quality.json", {
        "verdict": quality.verdict, "reasons": list(quality.reasons),
        "flags": quality.flags, "validation_setup_id": setup.setup_id,
        "roi_xywh": list(setup.roi_xywh),
    })
    return directory
