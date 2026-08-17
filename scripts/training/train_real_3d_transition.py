"""真实多视角15节点三维 transition 训练入口。

本脚本只做真实3D数据合同检查，并复用 ``train_transition.py`` 的模型与训练器。
默认开启置信度加权3D监督和多视角骨架重投影 loss。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)


REQUIRED_3D_FIELDS = (
    "positions", "actions", "positions_2d", "visibility",
    "position_confidence", "projection_matrices", "H", "W",
)


def validate_real_3d_data(data_dir, expected_nodes=15):
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"没有 NPZ: {data_dir}")
    summary = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in REQUIRED_3D_FIELDS if key not in data]
            if missing:
                raise ValueError(f"{path} 缺少真实3D训练字段: {missing}")
            positions = data["positions"]
            if positions.ndim != 3 or positions.shape[1] != 3:
                raise ValueError(f"{path} positions 应为 (T,3,N)，得到 {positions.shape}")
            if positions.shape[2] != int(expected_nodes):
                raise ValueError(
                    f"{path} 节点数 {positions.shape[2]} != {expected_nodes}")
            confidence = data["position_confidence"]
            if confidence.shape != (positions.shape[0], positions.shape[2]):
                raise ValueError(f"{path} position_confidence shape 不匹配")
            if (not np.isfinite(confidence).all() or np.any(confidence < 0)
                    or np.any(confidence > 1)):
                raise ValueError(f"{path} position_confidence 必须是 [0,1] 有限值")
            positions_2d = data["positions_2d"]
            visibility = data["visibility"]
            matrices = data["projection_matrices"]
            if matrices.ndim != 3 or matrices.shape[1:] != (3, 4):
                raise ValueError(f"{path} projection_matrices 应为 (V,3,4)")
            expected_2d = (positions.shape[0], matrices.shape[0],
                           positions.shape[2], 2)
            if positions_2d.shape != expected_2d:
                raise ValueError(
                    f"{path} positions_2d 应为 {expected_2d}，得到 {positions_2d.shape}")
            if visibility.shape != expected_2d[:-1]:
                raise ValueError(f"{path} visibility shape 不匹配")
            if not np.isfinite(matrices).all():
                raise ValueError(f"{path} projection_matrices 含 NaN/Inf")
            if not np.isfinite(positions).all():
                raise ValueError(f"{path} positions 含 NaN/Inf；先用 --clean-nan 转换")
            valid_ratio = float((confidence > 0).mean())
            summary.append((os.path.basename(path), positions.shape[0], valid_ratio))
    return summary


def main(argv=None):
    forwarded = list(argv if argv is not None else sys.argv[1:])
    if any(flag in forwarded for flag in ("-h", "--help")):
        from scripts.training.train_transition import main as train_main
        train_main(forwarded)
        return

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--data_dir", default="data/real_3d/train")
    pre.add_argument("--n_nodes", type=int, default=15)
    known, remaining = pre.parse_known_args(argv)
    summary = validate_real_3d_data(known.data_dir, known.n_nodes)
    print("真实3D数据合同检查通过:")
    for name, frames, ratio in summary:
        print(f"  {name}: {frames} frames, measured-node ratio={ratio:.1%}")

    if "--w_reprojection" not in forwarded:
        forwarded += ["--w_reprojection", "0.1"]
    if "--n_nodes" not in forwarded:
        forwarded += ["--n_nodes", str(known.n_nodes)]
    from scripts.training.train_transition import main as train_main
    train_main(forwarded)


if __name__ == "__main__":
    main()
