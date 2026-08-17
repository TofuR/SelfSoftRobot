"""从已有实验生成 deploy_manifest.json。

3 源 join,必须在服务器上跑(PC 上没有 real_capture/data/raw):
  1. checkpoint + config.json           → 网络形状 + data_dirs
  2. raw/<seq>/meta.json + frame_times  → action_scale_kpa(经 action_max_per_channel)/ train_dt
  3. eval_horizon/horizon_summary.json  → k_safe_table_px

Usage:
  python scripts/utils/build_deploy_manifest.py \
      --exp-dir train_log/open_loop_transition/exp_20260714_8 \
      --raw-seq real_capture/data/raw/seq_20260627_163921 \
      [--horizon-summary <exp>/eval_horizon/horizon_summary.json] \
      [--out <exp>/deploy_manifest.json]
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from scripts.real.masks_to_transition_npz import (
    EQUALITY_TOLERANCE_KPA,
    action_max_per_channel,
    normalize_channel_equalities,
    validate_action_equalities,
    validate_equality_action_maxes,
)
from real_validation.contracts.io import file_sha256
from real_validation.contracts.models import validate_hardware_action_contract


def find_checkpoint(exp_dir):
    """exp 根 → phase_*/model/best_model.pt。"""
    candidates = sorted(glob.glob(os.path.join(exp_dir, "phase_*", "model", "best_model.pt")))
    if not candidates:
        raise FileNotFoundError(f"{exp_dir} 下没有 phase_*/model/best_model.pt")
    return candidates[0]


def find_config(checkpoint):
    """checkpoint → 向上找 config.json(最多 3 级)。"""
    current = os.path.dirname(checkpoint)
    for _ in range(3):
        candidate = os.path.join(current, "config.json")
        if os.path.isfile(candidate):
            return candidate
        current = os.path.dirname(current)
    raise FileNotFoundError(f"{checkpoint} 附近找不到 config.json")


def measure_train_dt(raw_seq):
    """frame_times.txt → (measured_s, std_s)。禁止硬写 0.203125(仓库无任何文件记录该数)。"""
    times_path = os.path.join(raw_seq, "frame_times.txt")
    with open(times_path) as stream:
        times = np.array([float(line) for line in stream if line.strip()])
    diffs = np.diff(times)
    return float(diffs.mean()), float(diffs.std())


def load_actions_kpa(raw_seq, channels):
    """actions6.csv → 每通道原始 kPa(跳表头)。"""
    csv_path = os.path.join(raw_seq, "actions6.csv")
    raw = np.atleast_2d(np.genfromtxt(csv_path, delimiter=",", dtype=float))
    while raw.shape[0] and np.isnan(raw[0]).all():
        raw = raw[1:]
    cols = [int(c) + 1 for c in channels]   # +1:第 0 列是 t_sec
    return raw[:, cols].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="生成 deploy_manifest.json(3 源 join)")
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--raw-seq", required=True)
    parser.add_argument("--horizon-summary")
    parser.add_argument("--channels", default=None,
                        help="逗号分隔的驱动通道,如 0,1,2(3 腔道);缺省从 meta.hi6>0 推断")
    parser.add_argument("--out")
    args = parser.parse_args()

    checkpoint = find_checkpoint(args.exp_dir)
    config_path = find_config(checkpoint)
    with open(config_path) as stream:
        config = json.load(stream)

    meta_path = os.path.join(args.raw_seq, "meta.json")
    with open(meta_path) as stream:
        meta = json.load(stream)
    # 驱动通道:优先 CLI，其次使用训练 config 中由 Dataset 记录的动作视图合同。
    action_view = config.get("action_view", {})
    if args.channels:
        channels = [int(c.strip()) for c in args.channels.split(",") if c.strip()]
    elif action_view.get("model_action_channels") is not None:
        channels = [int(value) for value in action_view["model_action_channels"]]
    else:
        hi6 = meta.get("hi6", [])
        channels = [i for i, v in enumerate(hi6) if float(v) > 0] or \
                   [int(meta.get("active_channel", 0))]
    equalities = normalize_channel_equalities(meta.get("channel_equalities", ()))
    equality_tolerance = float(meta.get(
        "channel_equality_tolerance_kpa", EQUALITY_TOLERANCE_KPA))
    action_dim = int(config.get("action_dim", 1))
    if equalities:
        stored_equalities = tuple(tuple(int(v) for v in pair) for pair in
                                  action_view.get("channel_equalities", ()))
        if stored_equalities and stored_equalities != equalities:
            raise ValueError(
                "训练 config 与采集 meta 的 channel_equalities 不一致")
        if action_dim != len(channels):
            raise ValueError(
                f"训练 action_dim={action_dim} 与模型动作通道 {channels} 数量不一致")
    # 等值关系必须在原始六维 kPa 上验证；模型归一化尺度只取独立通道。
    raw_actions6 = load_actions_kpa(args.raw_seq, range(6))
    validate_action_equalities(raw_actions6, range(6), equalities, equality_tolerance)
    raw_maxes6 = action_max_per_channel(args.raw_seq, range(6), raw_actions6)
    validate_equality_action_maxes(
        raw_maxes6, range(6), equalities, equality_tolerance)
    maxes = raw_maxes6[np.asarray(channels, dtype=np.int64)]
    expansion = validate_hardware_action_contract(
        action_dim, channels, equalities, action_view.get("action_expansion6", ()))
    dt_mean, dt_std = measure_train_dt(args.raw_seq)

    data_dirs = config.get("data_dirs", {}).get("sequence", "")
    # 判断完整路径(data_dirs 以 .../train 结尾,os.path.basename 会取到 "train" 而丢掉
    # 目录名里的 _sam2/_rep 后缀 —— 必须对整个路径判断)
    if "_sam2" in data_dirs:
        mask_source, provenance = "sam2", "path_suffix"
    elif "_rep" in data_dirs:
        mask_source, provenance = "masks_repaired", "path_suffix"
    else:
        mask_source, provenance = "white_on_blue", "path_suffix"

    segment_params = None
    if mask_source == "white_on_blue":
        seg_meta = os.path.join(os.path.dirname(args.raw_seq), "derived",
                                os.path.basename(args.raw_seq), "segment_meta.json")
        if os.path.isfile(seg_meta):
            with open(seg_meta) as stream:
                segment_params = json.load(stream).get("params")

    k_safe_table_px = None
    if args.horizon_summary and os.path.isfile(args.horizon_summary):
        with open(args.horizon_summary) as stream:
            summary = json.load(stream)
        for entry in summary.get("summaries", []):
            if entry.get("model_type") == "open_loop":
                k_safe_table_px = {
                    "5px": entry.get("Kmax_px_5"),
                    "10px": entry.get("Kmax_px_10"),
                    "20px": entry.get("Kmax_px_20"),
                }
                k_safe_table_px = {k: int(v) for k, v in k_safe_table_px.items() if v is not None}
                break

    manifest = {
        "schema_version": 1,
        "checkpoint_sha256": file_sha256(checkpoint),
        "action_scale_kpa": [float(v) for v in maxes],
        "channel_map": channels,
        "channel_equalities": [list(pair) for pair in equalities],
        "action_expansion6": list(expansion),
        "train_dt_nominal_s": float(meta.get("action_interval_s", 0.2)),
        "train_dt_measured_s": dt_mean,
        "train_dt_std_s": dt_std,
        "mask_source": mask_source,
        "mask_source_provenance": provenance,
        "segment_params": segment_params,
        "camera": None,
        "reference_frame": None,
        "reference_frame_sha256": None,
        "mask_area_median_px": None,
        "registration_residual_max_px": 2.0,
        "k_safe_table_px": k_safe_table_px,
        "train_sequences": [os.path.basename(args.raw_seq)],
        "n_nodes": int(config.get("n_nodes", 15)),
        "window_size": int(config.get("window_size", 40)),
        "z_dim": int(config.get("z_dim", 16)),
        "episode_len": int(config.get("episode_len", 40)),
        "action_dim": action_dim,
        "encoder_type": str(config.get("encoder_type", "fractional")),
        "hidden_dim": int(config.get("hidden_dim", 128)),
        "n_scales": int(config.get("n_scales", 4)),
    }

    out = args.out or os.path.join(os.path.dirname(checkpoint), "..", "..", "deploy_manifest.json")
    with open(out, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(f"manifest 写入 {out}")
    print(f"  action_scale_kpa={manifest['action_scale_kpa']}  train_dt={dt_mean:.4f}±{dt_std:.4f}"
          f"  mask_source={mask_source}  k_safe={k_safe_table_px}")


if __name__ == "__main__":
    main()
