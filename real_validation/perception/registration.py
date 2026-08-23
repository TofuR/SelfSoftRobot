"""一次在线实验内的相机稳定性注册。

``robot_planar_mm_v1`` 在首次 Anchor 时吸收当前相机的平移、旋转和统一尺度，随后固定
使用该正反变换。本模块比较 Anchor 参考帧与后续 live 帧，检测实验进行中相机是否发生
移动；新相机安装通过新实验和新 Anchor 建立新的机器人坐标变换。

本模块只做检测，输出两个数字，门控用 displacement_px：
  fit_residual_px  内点重投影误差中位数 —— 拟合质量
  displacement_px  H 作用到图像四角的最大位移 —— 位姿到底移了多远

失败时 displacement_px 是 NaN，homography 是 None；消费者检查 `ok` 后再更新质量门。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np

from ._compat import _CV2_ERR, cv2

REG_OK = "ok"
REG_TOO_FEW_FEATURES = "too_few_features"
REG_TOO_FEW_MATCHES = "too_few_matches"
REG_HOMOGRAPHY_FAILED = "homography_failed"
REG_DISPLACED = "displaced"

@dataclass(frozen=True)
class RegistrationResult:
    homography: tuple[tuple[float, ...], ...] | None
    fit_residual_px: float
    displacement_px: float
    n_inliers: int
    n_matches: int
    reference_sha256: str
    ok: bool
    reason: str

    def to_dict(self) -> dict:
        return {"schema_version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict) -> "RegistrationResult":
        data = dict(value)
        data.pop("schema_version", None)
        homography = data["homography"]
        data["homography"] = (None if homography is None
                              else tuple(tuple(float(v) for v in row)
                                         for row in homography))
        return cls(**data)


def _failure(reason: str, n_matches: int = 0, n_inliers: int = 0,
             reference_sha256: str = "") -> RegistrationResult:
    return RegistrationResult(
        homography=None, fit_residual_px=float("nan"),
        displacement_px=float("nan"), n_inliers=n_inliers, n_matches=n_matches,
        reference_sha256=reference_sha256, ok=False, reason=reason)


def _corner_displacement(homography, width: int, height: int) -> float:
    corners = np.float32([[0, 0], [width - 1, 0],
                          [width - 1, height - 1], [0, height - 1]]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    return float(np.linalg.norm(mapped - corners.reshape(-1, 2), axis=1).max())


def estimate_registration(reference_gray, live_gray, *, reference_sha256: str = "",
                          max_displacement_px: float = 2.0,
                          min_inliers: int = 12) -> RegistrationResult:
    """ORB + BFMatcher(Hamming, crossCheck) + RANSAC homography。

    只用 opencv-python 主包（ORB 不在 contrib）。
    """
    if cv2 is None:
        raise RuntimeError(f"需要 opencv：{_CV2_ERR}")
    reference = np.asarray(reference_gray)
    live = np.asarray(live_gray)
    if reference.shape != live.shape:
        raise ValueError(f"参考帧与 live 帧尺寸不同：{reference.shape} != {live.shape}")
    height, width = reference.shape[:2]

    orb = cv2.ORB_create(nfeatures=2000)
    key_ref, desc_ref = orb.detectAndCompute(reference, None)
    key_live, desc_live = orb.detectAndCompute(live, None)
    if desc_ref is None or desc_live is None or len(key_ref) < min_inliers \
            or len(key_live) < min_inliers:
        return _failure(REG_TOO_FEW_FEATURES, reference_sha256=reference_sha256)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_ref, desc_live)
    if len(matches) < min_inliers:
        return _failure(REG_TOO_FEW_MATCHES, n_matches=len(matches),
                        reference_sha256=reference_sha256)

    source = np.float32([key_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    target = np.float32([key_live[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    if homography is None or mask is None:
        return _failure(REG_HOMOGRAPHY_FAILED, n_matches=len(matches),
                        reference_sha256=reference_sha256)
    inliers = int(mask.ravel().sum())
    if inliers < min_inliers:
        return _failure(REG_TOO_FEW_MATCHES, n_matches=len(matches),
                        n_inliers=inliers, reference_sha256=reference_sha256)

    projected = cv2.perspectiveTransform(source, homography)
    errors = np.linalg.norm(projected.reshape(-1, 2) - target.reshape(-1, 2), axis=1)
    fit_residual = float(np.median(errors[mask.ravel().astype(bool)]))
    displacement = _corner_displacement(homography, width, height)
    ok = math.isfinite(displacement) and displacement <= max_displacement_px
    return RegistrationResult(
        homography=tuple(tuple(float(v) for v in row) for row in homography),
        fit_residual_px=fit_residual, displacement_px=displacement,
        n_inliers=inliers, n_matches=len(matches),
        reference_sha256=reference_sha256,
        ok=bool(ok), reason=REG_OK if ok else REG_DISPLACED)


def save_registration(result: RegistrationResult, path) -> None:
    """写 registration.json（NaN 会被 json 拒绝，故显式转 None）。"""
    payload = result.to_dict()
    for key in ("fit_residual_px", "displacement_px"):
        if not math.isfinite(payload[key]):
            payload[key] = None
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load_registration(path) -> RegistrationResult:
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    for key in ("fit_residual_px", "displacement_px"):
        if payload.get(key) is None:
            payload[key] = float("nan")
    return RegistrationResult.from_dict(payload)
