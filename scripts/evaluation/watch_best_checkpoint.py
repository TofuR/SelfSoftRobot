"""在训练期间按epoch里程碑评价当前快照，并保存定量结果和图片。

训练器原子写入周期归档 ``model_epoch_XXXX.pt``，本脚本直接评价该
epoch 快照，以验证集全节点均误选出 ``best_eval_model.pt``。评价作为独立
进程并行运行；训练和评价可以使用同一张或不同的 GPU。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time


def build_parser():
    parser = argparse.ArgumentParser(description="周期评价训练中的best checkpoint")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--mode", required=True, choices=("gt", "open_loop"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--window-len", type=int, default=40)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--parent-pid", type=int, default=None)
    parser.add_argument("--calibration-file", default=None)
    parser.add_argument("--no-ndi", action="store_true")
    parser.add_argument("--overlay", action="store_true",
                        help="同时生成真实照片+mask+GT/预测骨架叠图")
    parser.add_argument("--cam0", default=None)
    parser.add_argument("--masks", default=None)
    return parser


def process_alive(pid):
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def phase_name(mode):
    return "gt_transition" if mode == "gt" else "open_loop_transition"


def validation_node_mean(per_frame_csv):
    """读取评价器的机器可读输出，返回实际预测帧的全节点均误。"""
    with open(per_frame_csv, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        metric = next((name for name in (reader.fieldnames or ())
                       if name.startswith("node_mean_") and
                       not name.endswith("_est_mm")), None)
        if metric is None:
            raise ValueError(f"per_frame.csv 缺少 node_mean_<unit>: {per_frame_csv}")
        values = [float(row[metric]) for row in reader
                  if row.get("is_prediction") == "1" and row.get(metric)]
    if not values:
        raise ValueError(f"per_frame.csv 没有可统计的预测帧: {per_frame_csv}")
    return sum(values) / len(values), metric.removeprefix("node_mean_")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.interval <= 0 or args.max_steps <= 0 or args.poll_seconds <= 0:
        raise ValueError("interval、max-steps和poll-seconds必须为正数")
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    phase_dir = os.path.join(
        os.path.abspath(args.experiment_dir), f"phase_{phase_name(args.mode)}")
    final_model = os.path.join(phase_dir, "model", "final_model.pt")
    best_eval_model = os.path.join(phase_dir, "model", "best_eval_model.pt")
    archive_dir = os.path.join(phase_dir, "checkpoints")
    os.makedirs(args.out_root, exist_ok=True)
    evaluated = set()
    best = None

    while True:
        archives = []
        if os.path.isdir(archive_dir):
            for name in os.listdir(archive_dir):
                if not (name.startswith("model_epoch_") and name.endswith(".pt")):
                    continue
                try:
                    epoch = int(name[len("model_epoch_"):-3])
                except ValueError:
                    continue
                if epoch % args.interval == 0:
                    archives.append(epoch)

        for epoch in sorted(set(archives) - evaluated):
            checkpoint = os.path.join(
                archive_dir, f"model_epoch_{epoch:04d}.pt")
            if not os.path.isfile(checkpoint):
                continue
            out_dir = os.path.join(args.out_root, f"epoch_{epoch:04d}")
            command = [
                sys.executable, "scripts/evaluation/eval_real_quant.py",
                "--checkpoint", checkpoint,
                "--data_dir", args.data_dir,
                "--mode", args.mode,
                "--max-steps", str(args.max_steps),
                "--out", out_dir,
            ]
            if args.mode == "open_loop":
                command.extend(("--window-len", str(args.window_len)))
            if args.calibration_file:
                command.extend(("--calibration-file", args.calibration_file))
            if args.no_ndi:
                command.append("--no-ndi")
            print(f">>> periodic snapshot eval epoch={epoch}: {' '.join(command)}",
                  flush=True)
            started = time.time()
            subprocess.run(command, cwd=project_root, check=True)
            if args.overlay:
                overlay_command = [
                    sys.executable,
                    "scripts/evaluation/visualize_real_overlay.py",
                    "--checkpoint", checkpoint,
                    "--data_dir", args.data_dir,
                    "--mode", args.mode,
                    "--window-len", str(args.window_len),
                    "--max-steps", str(args.max_steps),
                    "--out", os.path.join(out_dir, "overlay"),
                ]
                if args.mode == "open_loop":
                    overlay_command.append("--with-onestep")
                if args.cam0:
                    overlay_command.extend(("--cam0", args.cam0))
                if args.masks:
                    overlay_command.extend(("--masks", args.masks))
                print(f">>> periodic overlay epoch={epoch}: "
                      f"{' '.join(overlay_command)}", flush=True)
                subprocess.run(overlay_command, cwd=project_root, check=True)
            score, unit = validation_node_mean(
                os.path.join(out_dir, "per_frame.csv"))
            if best is None or score < best["node_mean"]:
                shutil.copy2(checkpoint, best_eval_model)
                best = {
                    "mode": args.mode,
                    "epoch": epoch,
                    "checkpoint": os.path.abspath(checkpoint),
                    "selected_checkpoint": os.path.abspath(best_eval_model),
                    "selection_metric": f"validation_node_mean_{unit}",
                    "node_mean": score,
                    "output": os.path.abspath(out_dir),
                }
                with open(os.path.join(args.out_root, "best.json"), "w",
                          encoding="utf-8") as stream:
                    json.dump(best, stream, indent=2, ensure_ascii=False)
            evaluated.add(epoch)
            latest = {
                "mode": args.mode,
                "epoch": epoch,
                "checkpoint": checkpoint,
                "output": os.path.abspath(out_dir),
                "elapsed_sec": time.time() - started,
                "evaluated_epochs": sorted(evaluated),
                "best": best,
            }
            with open(os.path.join(args.out_root, "latest.json"), "w",
                      encoding="utf-8") as stream:
                json.dump(latest, stream, indent=2, ensure_ascii=False)

        if os.path.isfile(final_model) and set(archives).issubset(evaluated):
            print(f">>> periodic evaluator complete: {sorted(evaluated)}", flush=True)
            return 0
        if not process_alive(args.parent_pid) and not os.path.isfile(final_model):
            print(">>> training process stopped before final_model; evaluator exits",
                  flush=True)
            return 2
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
