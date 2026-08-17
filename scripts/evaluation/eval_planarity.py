"""用 NDI 末端三维轨迹评估二维采集序列的离面漂移。

NDI 只用于独立质控/评价，不生成训练骨架，也不进入模型或 Planner。运动平面由单位法向量
和面上一点定义；未显式给定平面点时，使用零驱动基线样本的 XYZ 中位数。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_vec3(text: str, name: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(value.strip()) for value in text.split(","))
    except ValueError as error:
        raise ValueError(f"{name} 必须是逗号分隔的 x,y,z") from error
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} 必须是三个有限数")
    return values


def load_ndi_xyz(path: str | Path, ndi_index: int = 0,
                 min_quality: float | None = None) -> tuple[np.ndarray, int]:
    """读取 real_capture ``ndi.csv``，返回有效 XYZ 和总数据行数。"""
    if ndi_index < 0:
        raise ValueError("ndi_index 不能为负")
    target = Path(path)
    with target.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = [f"ndi{ndi_index}_{axis}" for axis in "xyz"]
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in required):
            raise ValueError(f"{target} 缺少列 {required}")
        quality_name = f"ndi{ndi_index}_quality"
        points = []
        total = 0
        for row in reader:
            total += 1
            try:
                point = tuple(float(row[name]) for name in required)
                quality = (float(row[quality_name])
                           if quality_name in row and row[quality_name] not in (None, "")
                           else float("nan"))
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in point):
                continue
            if min_quality is not None and (
                    not math.isfinite(quality) or quality < float(min_quality)):
                continue
            points.append(point)
    return np.asarray(points, dtype=np.float64).reshape(-1, 3), total


def evaluate_planarity(points_xyz, plane_normal, threshold_mm: float, *,
                       plane_point=None, baseline_samples: int = 30,
                       pass_stat: str = "p95") -> dict:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("至少需要一个有效的 NDI XYZ 样本")
    if not np.isfinite(points).all():
        raise ValueError("NDI XYZ 含 NaN/Inf")
    normal = np.asarray(plane_normal, dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        raise ValueError("plane_normal 必须是三个有限数")
    magnitude = float(np.linalg.norm(normal))
    if magnitude <= 1e-12:
        raise ValueError("plane_normal 不能是零向量")
    normal = normal / magnitude
    if not math.isfinite(float(threshold_mm)) or float(threshold_mm) < 0.0:
        raise ValueError("threshold_mm 必须是非负有限数")
    if pass_stat not in {"p95", "max"}:
        raise ValueError("pass_stat 只能是 p95 或 max")

    if plane_point is None:
        count = min(len(points), int(baseline_samples))
        if count <= 0:
            raise ValueError("baseline_samples 必须为正数")
        point = np.median(points[:count], axis=0)
        point_source = f"baseline_median_first_{count}_valid_samples"
    else:
        point = np.asarray(plane_point, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("plane_point 必须是三个有限数")
        count = 0
        point_source = "operator"

    distances = np.abs((points - point) @ normal)
    p50, p95 = np.percentile(distances, (50, 95))
    maximum = float(np.max(distances))
    values = {"p50": float(p50), "p95": float(p95), "max": maximum}
    return {
        "schema_version": 1,
        "metric": "ndi_tip_absolute_out_of_plane_distance_mm",
        "plane_normal": normal.tolist(),
        "plane_point_mm": point.tolist(),
        "plane_point_source": point_source,
        "baseline_samples_used": count,
        "planarity_threshold_mm": float(threshold_mm),
        "planarity_pass_stat": pass_stat,
        "planarity_tip_abs_mm_p50": values["p50"],
        "planarity_tip_abs_mm_p95": values["p95"],
        "planarity_tip_abs_mm_max": values["max"],
        "valid_samples": int(len(points)),
        "planarity_pass": bool(values[pass_stat] <= float(threshold_mm)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NDI 末端离面漂移质控（不进入训练/Planner）")
    parser.add_argument("--seq", required=True, help="real_capture 原始序列目录")
    parser.add_argument("--ndi-csv", default=None, help="默认 <seq>/ndi.csv")
    parser.add_argument("--ndi-index", type=int, default=0)
    parser.add_argument("--plane-normal", required=True, help="运动平面法向量 nx,ny,nz")
    parser.add_argument("--plane-point", default=None,
                        help="可选面上一点 x,y,z(mm)；默认取基线中位数")
    parser.add_argument("--baseline-samples", type=int, default=30,
                        help="估计 plane point 的开头有效样本数(默认30)")
    parser.add_argument("--threshold-mm", type=float, required=True,
                        help="离面通过阈值(mm)，应由零驱动噪声/重复实验确定")
    parser.add_argument("--threshold-source", default="operator",
                        help="阈值来源说明，写入报告供审计")
    parser.add_argument("--pass-stat", choices=("p95", "max"), default="p95")
    parser.add_argument("--min-quality", type=float, default=None,
                        help="可选 NDI quality 下限")
    parser.add_argument("--out", default=None, help="默认 <seq>/planarity_qc.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seq = Path(args.seq).resolve()
    ndi_csv = Path(args.ndi_csv).resolve() if args.ndi_csv else seq / "ndi.csv"
    points, total = load_ndi_xyz(
        ndi_csv, ndi_index=args.ndi_index, min_quality=args.min_quality)
    report = evaluate_planarity(
        points, parse_vec3(args.plane_normal, "plane_normal"), args.threshold_mm,
        plane_point=(parse_vec3(args.plane_point, "plane_point")
                     if args.plane_point else None),
        baseline_samples=args.baseline_samples, pass_stat=args.pass_stat)
    report.update({
        "source_ndi_csv": str(ndi_csv),
        "ndi_index": int(args.ndi_index),
        "total_samples": int(total),
        "rejected_samples": int(total - len(points)),
        "min_quality": args.min_quality,
        "planarity_threshold_source": str(args.threshold_source),
        "ndi_role": "hidden_evaluation_and_qc_only",
    })
    output = Path(args.out).resolve() if args.out else seq / "planarity_qc.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    status = "PASS" if report["planarity_pass"] else "FAIL"
    print(f"{status}: p95={report['planarity_tip_abs_mm_p95']:.4g} mm, "
          f"max={report['planarity_tip_abs_mm_max']:.4g} mm, "
          f"valid={report['valid_samples']}/{total} -> {output}")


if __name__ == "__main__":
    main()
