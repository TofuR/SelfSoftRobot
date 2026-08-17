"""命令行感知探针：抓帧 → 分割 → 骨架 → 质量门控 → 叠加图 + 逐算子耗时。

不需要 GUI、不需要 checkpoint，却把在线感知链每个算子都跑通，并给出采集协议所需的
参数（实际分割参数、单帧耗时、坏帧率）。

用法（--source 必填；**不内置任何仓库路径默认值**，否则在 PC 上会指向不存在的目录）：

  # 离线：用已采集的一段帧（开发机没有相机时的唯一途径）
  python perception_probe.py --source dir \\
      --frames-dir <seq>/cam0 --background <derived>/bg_median.png \\
      [--segment-params <derived>/segment_meta.json] \\
      [--reference <derived>/bg_median.png] --n-points 15 --frames 12 --out <out>

  # 在线：从 RealSense 实时取流（经 hardware.camera.RealSenseCam，需要 pyrealsense2）
  python perception_probe.py --source live --background <bg.png> --frames 12 --out <out>

产物：overlay.png（叠加网格）/ timing.json（逐算子 mean+p90）/ quality.jsonl（逐帧标志）
      / registration.json（若给了 --reference）
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # 支持复制目录后直接 ``python tools/perception_probe.py``
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    __package__ = "real_validation.tools"

import numpy as np

from ..perception.background import background_drift, load_median_background
from ..perception.quality import QualityThresholds, assess_frame
from ..perception.registration import estimate_registration, save_registration
from ..perception.segmentation import segment_white_on_blue
from ..perception.skeleton import extract_skeleton_2d

_SKELETON_COLOR = (255, 255, 0)   # BGR 青
_MASK_COLOR = (0, 0, 255)         # BGR 红


def _percentile(values, ratio: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), ratio)) if values else 0.0


def _stats(values) -> dict:
    return {"mean": float(np.mean(values)) if values else 0.0,
            "p90": _percentile(values, 90.0),
            "max": float(np.max(values)) if values else 0.0}


def list_frames(frames_dir) -> list[str]:
    """masks_repaired/ 之类目录里含子目录 → 必须 glob '*.png'，不能 os.listdir。"""
    files = sorted(glob.glob(os.path.join(str(frames_dir), "*.png")))
    if not files:
        raise FileNotFoundError(f"目录里没有 PNG：{frames_dir}")
    return files


def load_segment_params(path) -> dict:
    """从 derived/<seq>/segment_meta.json 读真实分割参数。

    ⚠️ 必须读这个文件而不是用代码默认值：批产用的是 val=100，而
    segment_white_on_blue 的默认是 val=120。
    """
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    params = payload.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"{path} 缺少 params 对象")
    return {key: params[key] for key in
            ("sat", "val", "diff", "dil", "open_k", "close_k",
             "min_area_frac", "min_h_frac") if key in params}


def draw_overlay(bgr, mask, skeleton, label: str):
    """mask 半透明红 + 骨架青线 + 末端圈 + 左上角文字（纯 cv2，无 matplotlib）。"""
    import cv2
    canvas = bgr.copy()
    tint = np.zeros_like(canvas)
    tint[mask > 0] = _MASK_COLOR
    canvas = cv2.addWeighted(tint, 0.22, canvas, 0.78, 0.0)
    points = np.asarray(skeleton, dtype=np.int32).reshape(-1, 1, 2)
    if len(points) >= 2 and np.abs(skeleton).max() > 0:
        cv2.polylines(canvas, [points], False, _SKELETON_COLOR, 1, cv2.LINE_AA)
        for point in points.reshape(-1, 2):
            cv2.circle(canvas, tuple(int(v) for v in point), 2, _SKELETON_COLOR, -1)
        cv2.circle(canvas, tuple(int(v) for v in points[0, 0]), 5, (0, 255, 0), 1)
    cv2.putText(canvas, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def grid(tiles, columns: int = 4):
    """把若干同尺寸图拼成网格（不足处填黑）。"""
    if not tiles:
        raise ValueError("没有可拼接的图")
    height, width = tiles[0].shape[:2]
    rows = (len(tiles) + columns - 1) // columns
    canvas = np.zeros((rows * height, columns * width, 3), np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        canvas[row * height:(row + 1) * height, column * width:(column + 1) * width] = tile
    return canvas


def run_probe(frames_bgr, background_gray, *, segment_params: dict, n_points: int,
              thresholds: QualityThresholds, reference_gray=None,
              frame_age_s: float | None = None) -> dict:
    """对一批 BGR 帧跑完整在线链，返回 {timing, quality, overlay, registration}。"""
    timing = {"segment_ms": [], "skeleton_ms": [], "quality_ms": [], "total_ms": []}
    quality_records = []
    tiles = []
    previous_skeleton = None

    # 位姿注册在 loop **之前**对首帧算一次（探测的是"live 像素 == 训练期像素"这个
    # 恒等映射，首帧即可判定）。结果喂给每一帧的 quality 门控：ok 时传位移值，
    # 失败时传 NaN —— assess_frame 据此加 registration_displaced 并 reject 整批。
    registration = None
    registration_displacement_px = None
    if reference_gray is not None and frames_bgr:
        import cv2
        live_gray = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
        registration = estimate_registration(
            reference_gray, live_gray,
            max_displacement_px=thresholds.max_registration_displacement_px)
        registration_displacement_px = (
            registration.displacement_px if registration.ok else float("nan"))

    for index, bgr in enumerate(frames_bgr):
        start = time.perf_counter()
        mark = time.perf_counter()
        mask = segment_white_on_blue(bgr, background_gray, **segment_params)
        timing["segment_ms"].append((time.perf_counter() - mark) * 1e3)

        mark = time.perf_counter()
        skeleton, info = extract_skeleton_2d(mask, n_points, tip_fix=True, return_info=True)
        timing["skeleton_ms"].append((time.perf_counter() - mark) * 1e3)

        mark = time.perf_counter()
        quality = assess_frame(mask, skeleton, info, thresholds,
                               prev_skeleton=previous_skeleton,
                               frame_age_s=frame_age_s,
                               registration_displacement_px=registration_displacement_px)
        timing["quality_ms"].append((time.perf_counter() - mark) * 1e3)
        timing["total_ms"].append((time.perf_counter() - start) * 1e3)

        record = {"frame": index, **quality.flags,
                  "reasons": list(quality.reasons)}
        quality_records.append(record)
        tiles.append(draw_overlay(bgr, mask, skeleton,
                                  f"#{index} {quality.verdict}"))
        if quality.verdict != "reject":
            previous_skeleton = skeleton

    verdicts = [record["verdict"] for record in quality_records]
    return {
        "timing": {key: _stats(values) for key, values in timing.items()} |
                  {"n_frames": len(frames_bgr)},
        "quality": quality_records,
        "verdict_counts": {name: verdicts.count(name)
                           for name in ("ok", "degraded", "reject")},
        "overlay": grid(tiles),
        "registration": registration,
    }


def _load_frames_from_dir(frames_dir, count: int, start: int = 0):
    import cv2
    files = list_frames(frames_dir)
    # 取**连续** [start, start+count) 帧，不能均匀采样：node_step_high 判据的语义是
    # "上一被接受(相邻时刻 ≈ 0.2s)帧 → 当前帧位移 <4px"，只有连续帧符合；对 10214 帧
    # linspace 采样会让相邻采样帧间隔 ~850 帧、骨架位移天然巨大，全部误报。
    # start 允许跳过采集开头的质量差段（真实序列里有坏帧）。
    end = min(start + count, len(files))
    picked = files[start:end]
    frames = []
    for path in picked:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取帧：{path}")
        frames.append(image)
    return frames


def _load_frames_from_camera(count: int, warmup: int = 5):
    """从 RealSense 取 count 帧。

    经 hardware.camera.RealSenseCam(驱动已内部移植),不再依赖 real_capture/ 并排
    存在 + pyrealsense2。延迟 import:只有 --source live 模式需要 Qt + 相机驱动。
    """
    from PyQt5.QtCore import QCoreApplication
    from ..hardware.camera import create_realsense_cam

    app = QCoreApplication.instance() or QCoreApplication([])
    cam = create_realsense_cam(width=640, height=480, fps=30)
    frames: list[np.ndarray] = []
    errors: list[str] = []

    def _on_frame(img, _t_monotonic):
        frames.append(img)

    def _on_error(message):
        errors.append(message)

    cam.frame_ready.connect(_on_frame)
    cam.error.connect(_on_error)
    cam.start()
    try:
        target = count + warmup
        deadline = time.monotonic() + 30.0
        while len(frames) < target and time.monotonic() < deadline:
            app.processEvents()   # 把 QThread 的 queued signal 分发给上面的 Python slot
            if errors:
                raise RuntimeError(errors[0])
            time.sleep(0.01)
        if len(frames) < target:
            raise RuntimeError(f"相机只返回 {len(frames)} 帧(需 {target})")
        return frames[warmup:warmup + count]
    finally:
        cam.stop()


def main() -> int:
    import cv2
    parser = argparse.ArgumentParser(description="在线感知链探针（无 GUI、无 checkpoint）")
    parser.add_argument("--source", required=True, choices=("dir", "live"),
                        help="dir=用已采集的一段帧；live=从 RealSense 实时取流")
    parser.add_argument("--frames-dir", help="--source dir 时必填：含 *.png 的目录")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="--source dir 时起始帧号（默认 0）；采集开头常有质量差段，"
                             "可跳到干净窗口")
    parser.add_argument("--background", required=True, help="中值背景灰度图 PNG")
    parser.add_argument("--segment-params",
                        help="derived/<seq>/segment_meta.json；不给则用代码默认（val=120，"
                             "与批产的 val=100 不同，仅供快速冒烟）")
    parser.add_argument("--reference", help="位姿注册的基准灰度图；不给则跳过注册")
    parser.add_argument("--n-points", type=int, default=15)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--frame-age-s", type=float, default=None,
                        help="每帧相对采集时刻的帧龄(秒)；不给则跳过 frame_age 判据，"
                             "给且 > max_frame_age_s(默认 0.5) 时该帧判 reject")
    parser.add_argument("--area-median-px", type=float, default=None,
                        help="mask 面积中位数；不给则用本批帧自身的中位数（仅冒烟用，"
                             "正式验收必须从 manifest 提供）")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.source == "dir":
        if not args.frames_dir:
            parser.error("--source dir 需要 --frames-dir")
        frames = _load_frames_from_dir(args.frames_dir, args.frames, args.start_frame)
    else:
        frames = _load_frames_from_camera(args.frames)

    background = load_median_background(args.background)
    segment_params = (load_segment_params(args.segment_params)
                      if args.segment_params else {})
    reference = (load_median_background(args.reference) if args.reference else None)

    area_median = args.area_median_px
    if area_median is None:
        areas = [float(segment_white_on_blue(frame, background, **segment_params).sum())
                 for frame in frames]
        positive = [value for value in areas if value > 0]
        area_median = float(np.median(positive)) if positive else 1.0
        print(f"[probe] --area-median-px 未提供，用本批中位数 {area_median:.0f} px"
              f"（仅冒烟；正式验收须从 deploy_manifest 提供）")

    thresholds = QualityThresholds(area_median)
    result = run_probe(frames, background, segment_params=segment_params,
                       n_points=args.n_points, thresholds=thresholds,
                       reference_gray=reference,
                       frame_age_s=args.frame_age_s)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "overlay.png"), result["overlay"])
    with open(out / "timing.json", "w", encoding="utf-8") as stream:
        json.dump(result["timing"], stream, ensure_ascii=False, indent=2, allow_nan=False)
    with open(out / "quality.jsonl", "w", encoding="utf-8") as stream:
        for record in result["quality"]:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    if result["registration"] is not None:
        save_registration(result["registration"], out / "registration.json")
        print(f"[probe] 配准 displacement={result['registration'].displacement_px:.2f} px "
              f"ok={result['registration'].ok} reason={result['registration'].reason}")
    if reference is not None:
        print(f"[probe] 背景漂移中位数="
              f"{background_drift(reference, background):.2f} 灰阶")

    print(f"[probe] {result['timing']['n_frames']} 帧  "
          f"total p90={result['timing']['total_ms']['p90']:.1f} ms  "
          f"(segment {result['timing']['segment_ms']['mean']:.1f} / "
          f"skeleton {result['timing']['skeleton_ms']['mean']:.1f} / "
          f"quality {result['timing']['quality_ms']['mean']:.1f} ms 均值)")
    print(f"[probe] 判决分布 {result['verdict_counts']}")
    print(f"[probe] 产物写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
