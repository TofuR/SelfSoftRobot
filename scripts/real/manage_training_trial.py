"""创建并完成真实状态转移训练试次。

每个试次拥有完整的阶段权重、日志、评价和配置。流水线通过本入口创建标准目录，
完成后生成 artifacts.json，供人工查看和部署工具读取。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.experiment import create_experiment, save_config  # noqa: E402


LAYOUT = {
    "gt_stage": "stages/gt",
    "open_loop_stage": "stages/open_loop",
    "gt_periodic": "evaluations/gt/periodic",
    "gt_best": "evaluations/gt/best",
    "open_loop_periodic": "evaluations/open_loop/periodic",
    "open_loop_best": "evaluations/open_loop/best",
    "diagnostics": "diagnostics",
}


def _timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(tmp_path, path)


def _prepare_requested_dir(path):
    path = os.path.normpath(path)
    if os.path.isdir(path):
        if os.listdir(path):
            raise FileExistsError(f"试次目录已包含产物: {path}")
    else:
        os.makedirs(path)
    return path


def build_trial_config(args, trial_dir):
    train_npz_files = sorted(glob.glob(os.path.join(args.train_dir, "*.npz")))
    val_npz_files = sorted(glob.glob(os.path.join(args.val_dir, "*.npz")))
    return {
        "schema_version": 1,
        "trial": {
            "id": os.path.basename(trial_dir),
            "created_at": _timestamp(),
            "sequence_tag": args.sequence_tag,
            "capture_sequence": args.capture_sequence,
        },
        "data": {
            "train_dir": args.train_dir,
            "val_dir": args.val_dir,
            "train_npz": args.train_npz,
            "train_npz_files": train_npz_files,
            "val_npz_files": val_npz_files,
            "dataset_manifest": getattr(args, "dataset_manifest", None),
            "cam0_dir": args.cam0_dir,
            "masks_dir": args.masks_dir,
        },
        "resources": {
            "train_gpu": args.gpu_id,
            "evaluation_gpu": args.eval_gpu_id,
            "num_workers": args.num_workers,
        },
        "evaluation": {
            "ndi_available": bool(int(getattr(args, "ndi_available", 0))),
            "ndi_csv": getattr(args, "ndi_csv", None),
            "frame_times": getattr(args, "frame_times", None),
            "ndi_role": "independent_endpoint_diagnostic",
        },
        "training": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "save_interval": args.save_interval,
            "periodic_eval_interval": args.periodic_eval_interval,
            "periodic_max_steps": args.periodic_max_steps,
            "window_size": args.window_size,
            "episode_len": args.episode_len,
            "stages": {
                "gt": {
                    "epochs": args.gt_epochs,
                    "mode": "gt",
                },
                "open_loop": {
                    "epochs": args.open_loop_epochs,
                    "mode": "open_loop",
                    "initialization": "gt_best",
                    "tf_ratio": 1.0,
                    "tf_anneal_epochs": args.tf_anneal_epochs,
                    "tf_min": 0.0,
                    "tf_schedule": "staircase",
                },
            },
        },
        "layout": dict(LAYOUT),
    }


def create_trial(args):
    if args.trial_dir:
        trial_dir = _prepare_requested_dir(args.trial_dir)
    else:
        trial_dir = create_experiment(
            args.base_dir, prefix="trial", announce=False)
    for relative in LAYOUT.values():
        os.makedirs(os.path.join(trial_dir, relative), exist_ok=True)
    save_config(trial_dir, build_trial_config(args, trial_dir))
    return trial_dir


def _require_files(trial_dir, paths):
    missing = [
        relative for relative in paths
        if not os.path.isfile(os.path.join(trial_dir, relative))
    ]
    if missing:
        raise FileNotFoundError(
            "试次归档缺少必要产物: " + ", ".join(missing))


def finalize_trial(trial_dir):
    trial_dir = os.path.normpath(trial_dir)
    required = [
        "config.json",
        "commands.sh",
        "stages/gt/config.json",
        "stages/gt/phase_gt_transition/model/best_model.pt",
        "stages/open_loop/config.json",
        "stages/open_loop/phase_open_loop_transition/model/best_model.pt",
        "evaluations/gt/best/quantitative/summary.txt",
        "evaluations/gt/best/overlay/summary.txt",
        "evaluations/gt/best/overlay/montage.png",
        "evaluations/open_loop/best/quantitative/summary.txt",
        "evaluations/open_loop/best/overlay/summary.txt",
        "evaluations/open_loop/best/overlay/montage.png",
    ]
    _require_files(trial_dir, required)
    with open(os.path.join(trial_dir, "config.json"), encoding="utf-8") as stream:
        config = json.load(stream)
    artifacts = {
        "schema_version": 1,
        "trial_id": config["trial"]["id"],
        "completed_at": _timestamp(),
        "stages": {
            "gt": {
                "experiment_dir": LAYOUT["gt_stage"],
                "config": "stages/gt/config.json",
                "best_checkpoint":
                    "stages/gt/phase_gt_transition/model/best_model.pt",
                "loss_log":
                    "stages/gt/phase_gt_transition/loss_log.csv",
                "training_log": "stages/gt/train.log",
                "periodic_evaluations": LAYOUT["gt_periodic"],
                "best_quantitative":
                    "evaluations/gt/best/quantitative",
                "best_overlay": "evaluations/gt/best/overlay",
            },
            "open_loop": {
                "experiment_dir": LAYOUT["open_loop_stage"],
                "config": "stages/open_loop/config.json",
                "best_checkpoint":
                    "stages/open_loop/phase_open_loop_transition/model/best_model.pt",
                "loss_log":
                    "stages/open_loop/phase_open_loop_transition/loss_log.csv",
                "training_log": "stages/open_loop/train.log",
                "periodic_evaluations": LAYOUT["open_loop_periodic"],
                "best_quantitative":
                    "evaluations/open_loop/best/quantitative",
                "best_overlay": "evaluations/open_loop/best/overlay",
            },
        },
        "diagnostics": LAYOUT["diagnostics"],
        "commands": "commands.sh",
        "status": "status.txt",
    }
    artifacts_path = os.path.join(trial_dir, "artifacts.json")
    _write_json(artifacts_path, artifacts)
    return artifacts_path


def validate_dataset_manifest(path):
    with open(path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    quality = manifest.get("quality_control", {})
    if quality.get("automated_checks_passed") is not True \
            or quality.get("training_ready") is not True:
        failed = [item.get("name") for item in quality.get("checks", [])
                  if item.get("required_for_training", True) and
                  item.get("passed") is not True]
        raise ValueError(
            f"dataset_manifest尚未满足训练合同: {path}; failed={failed}")
    return {
        "dataset_id": manifest.get("dataset_id"),
        "training_ready": True,
        "checks": len(quality.get("checks", [])),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="真实训练试次目录管理")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="创建标准试次目录和config.json")
    create.add_argument("--base-dir", required=True)
    create.add_argument("--trial-dir", default=None)
    create.add_argument("--sequence-tag", required=True)
    create.add_argument("--capture-sequence", required=True)
    create.add_argument("--train-dir", required=True)
    create.add_argument("--val-dir", required=True)
    create.add_argument("--train-npz", required=True)
    create.add_argument("--dataset-manifest", default=None)
    create.add_argument("--cam0-dir", required=True)
    create.add_argument("--masks-dir", required=True)
    create.add_argument("--ndi-csv", default=None)
    create.add_argument("--frame-times", default=None)
    create.add_argument("--ndi-available", choices=("0", "1"), default="0")
    create.add_argument("--gpu-id", required=True)
    create.add_argument("--eval-gpu-id", required=True)
    create.add_argument("--gt-epochs", type=int, required=True)
    create.add_argument("--open-loop-epochs", type=int, required=True)
    create.add_argument("--batch-size", type=int, required=True)
    create.add_argument("--num-workers", type=int, required=True)
    create.add_argument("--save-interval", type=int, required=True)
    create.add_argument("--periodic-eval-interval", type=int, required=True)
    create.add_argument("--periodic-max-steps", type=int, required=True)
    create.add_argument("--seed", type=int, required=True)
    create.add_argument("--window-size", type=int, required=True)
    create.add_argument("--episode-len", type=int, required=True)
    create.add_argument("--tf-anneal-epochs", type=int, required=True)

    finalize = subparsers.add_parser(
        "finalize", help="校验标准产物并写artifacts.json")
    finalize.add_argument("--trial-dir", required=True)
    validate = subparsers.add_parser(
        "validate-dataset", help="校验自动前处理数据清单")
    validate.add_argument("--dataset-manifest", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "create":
        print(create_trial(args))
    elif args.command == "finalize":
        print(finalize_trial(args.trial_dir))
    else:
        print(json.dumps(validate_dataset_manifest(
            args.dataset_manifest), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
