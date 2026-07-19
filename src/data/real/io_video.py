"""io_video.py — 多相机视频/图像加载 + 跨视角配对 + 去畸变。

支持两种输入：
  - 已按视角分好的图像目录（每视角一个 dir，按文件名排序跨目录配对）
  - 视频文件（每视角一个 video，按帧序号配对——需硬件/同 fps 同步）

输出 (V, N, H, W, 3) BGR uint8。可选 undistort callable（由标定 K,dist 构造）
对每帧去畸变，使后续主点=图像中心、单一 focal 的假设成立。
"""

import glob
import os

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    cv2 = None
    _CV2_ERR = exc


def make_undistorter(K, dist, H, W):
    """返回 undistort(img)->img 的去畸变函数（用预计算映射加速）。"""
    if cv2 is None:
        raise RuntimeError(f"需要 opencv：{_CV2_ERR}")
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, K, (W, H), cv2.CV_16UC2)

    def undistort(img):
        return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    return undistort


def load_image_views(view_dirs, undistort=None, max_frames=None):
    """每视角一个图像目录 → (V, N, H, W, 3) BGR + 帧名。

    Args:
        view_dirs: list[V] 目录路径；每目录按文件名排序，跨目录按下标配对。
        undistort: 可选 callable(img)->img。
        max_frames: 最多取前 N 帧。
    """
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [[path for path in sorted(glob.glob(os.path.join(d, "*")))
              if os.path.splitext(path)[1].lower() in extensions]
             for d in view_dirs]
    if any(not paths for paths in files):
        empty = [view_dirs[i] for i, paths in enumerate(files) if not paths]
        raise FileNotFoundError(f"没有可读图像视角目录：{empty}")
    n = min(len(f) for f in files) if files else 0
    if max_frames:
        n = min(n, max_frames)
    seqs = []
    for v in range(len(files)):
        frames = []
        for i in range(n):
            img = cv2.imread(files[v][i]) if cv2 else None
            if img is None:
                raise ValueError(f"无法读取图像：{files[v][i]}")
            if undistort is not None:
                img = undistort(img)
            frames.append(img)
        seqs.append(frames)
    n = min(len(s) for s in seqs) if seqs else 0
    names = [os.path.basename(files[0][i]) for i in range(n)]
    arr = np.stack([np.stack(s[:n], 0) for s in seqs], 0)  # (V,N,H,W,3)
    return arr, names


def load_video_views(video_paths, undistort=None, max_frames=None):
    """每视角一个视频 → (V, N, H, W, 3) BGR（按帧序号配对）。"""
    if cv2 is None:
        raise RuntimeError(f"需要 opencv：{_CV2_ERR}")
    seqs = []
    for vp in video_paths:
        cap = cv2.VideoCapture(vp)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if undistort is not None:
                fr = undistort(fr)
            frames.append(fr)
            if max_frames and len(frames) >= max_frames:
                break
        cap.release()
        seqs.append(frames)
    n = min(len(s) for s in seqs) if seqs else 0
    arr = np.stack([np.stack(s[:n], 0) for s in seqs], 0)
    return arr
