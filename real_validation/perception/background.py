"""静态背景的加载 / 重建 / 漂移检测。

中值背景是 white_on_blue 分割的第 2 步（背景差）所依赖的量，且它逐像素绑定相机
位姿 —— 相机一动，absdiff 全图激活、分割崩溃。因此这里同时提供漂移检测。
"""

import numpy as np

from ._compat import _CV2_ERR, cv2


def load_median_background(path):
    """读取 bg_median.png → (H,W) uint8 灰度。"""
    if cv2 is None:
        raise RuntimeError(f"需要 opencv：{_CV2_ERR}")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"无法读取背景图：{path}")
    return image


def build_median_background_from_frames(frames, n_bg: int = 500):
    """(T,H,W) 灰度或 (T,H,W,3) BGR 帧序列 → per-pixel 中值背景 (H,W) uint8。

    与 segmentation.build_median_background 同算法，但吃内存中的序列而非目录，
    供在线"开机重建背景"使用。机器人移动占每像素 <50% 时间 → 中值 ≈ 静态背景。
    """
    array = np.asarray(frames)
    if array.ndim == 4:
        if cv2 is None:
            raise RuntimeError(f"需要 opencv：{_CV2_ERR}")
        array = np.stack([cv2.cvtColor(item, cv2.COLOR_BGR2GRAY) for item in array])
    if array.ndim != 3 or len(array) == 0:
        raise ValueError("frames 必须是 (T,H,W) 灰度或 (T,H,W,3) BGR，且 T>0")
    index = np.linspace(0, len(array) - 1, min(n_bg, len(array))).astype(int)
    return np.median(array[index], axis=0).astype(np.uint8)


def background_drift(reference_gray, live_gray) -> float:
    """两张背景灰度图的逐像素绝对差中位数（灰阶）。

    用中位数而非均值：对局部遮挡（手、异物）稳健，对全局位移敏感。
    """
    reference = np.asarray(reference_gray, dtype=np.int16)
    live = np.asarray(live_gray, dtype=np.int16)
    if reference.shape != live.shape:
        raise ValueError(f"背景尺寸不同：{reference.shape} != {live.shape}")
    return float(np.median(np.abs(reference - live)))
