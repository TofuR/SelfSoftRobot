"""创建并完成真实状态转移训练试次。

每个试次拥有完整的阶段权重、日志、评价和配置。流水线通过本入口创建标准目录，
完成后生成 artifacts.json，供人工查看和部署工具读取。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
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

FINAL_OUTPUT_DIRS = (
    "evaluations/gt/best/quantitative",
    "evaluations/gt/best/overlay",
    "evaluations/open_loop/best/quantitative",
    "evaluations/open_loop/best/overlay",
)

SEQUENCE_TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
CAPTURE_SEQUENCE_PATTERN = re.compile(r"^(seq_\d{8}_\d{6})(?:_|$)")
PROCESSING_SUFFIX_PATTERN = re.compile(
    r"(?:_n\d+)?(?:_sam2)?(?:_robot)?(?:_mm)?$")


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


def validate_sequence_tag(value):
    """Validate the stable grouping label used immediately below real_pipeline."""
    if not value or not SEQUENCE_TAG_PATTERN.fullmatch(value):
        raise ValueError(
            "sequence_tag仅支持字母、数字、点、下划线和连字符，且必须以字母或数字开头: "
            f"{value!r}")
    return value


def infer_sequence_tag(train_dir, dataset_manifest=None):
    """Infer a concise acquisition label while retaining preprocessing in config.

    A single-sequence manifest maps directly to its capture sequence.  A combined
    dataset uses its dataset id with the standard processing contract suffix
    removed.  NPZ names provide a final fallback for older datasets.
    """
    manifest = None
    if dataset_manifest and os.path.isfile(dataset_manifest):
        with open(dataset_manifest, encoding="utf-8") as stream:
            manifest = json.load(stream)
    if manifest:
        source = manifest.get("source", {})
        sequence = source.get("sequence")
        if isinstance(sequence, str) and sequence:
            return validate_sequence_tag(sequence)
        dataset_id = manifest.get("dataset_id")
        if isinstance(dataset_id, str) and dataset_id:
            concise = PROCESSING_SUFFIX_PATTERN.sub("", dataset_id)
            return validate_sequence_tag(concise)

    for npz_path in sorted(glob.glob(os.path.join(train_dir, "*.npz"))):
        match = CAPTURE_SEQUENCE_PATTERN.match(os.path.basename(npz_path))
        if match:
            return validate_sequence_tag(match.group(1))

    dataset_dir = os.path.basename(os.path.dirname(os.path.normpath(train_dir)))
    concise = PROCESSING_SUFFIX_PATTERN.sub("", dataset_dir)
    return validate_sequence_tag(concise)


def prepare_trial_layout(trial_dir):
    """Create every directory consumed directly by a stage process or logger."""
    trial_dir = os.path.normpath(trial_dir)
    for relative in (*LAYOUT.values(), *FINAL_OUTPUT_DIRS):
        os.makedirs(os.path.join(trial_dir, relative), exist_ok=True)
    return trial_dir


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
                    "learning_rate": args.gt_lr,
                    "dense_step_weight": args.dense_step_weight,
                    "scheduler_patience": args.gt_scheduler_patience,
                },
                "open_loop": {
                    "epochs": args.open_loop_epochs,
                    "mode": "open_loop",
                    "initialization": "gt_best",
                    "learning_rate": args.open_loop_lr,
                    "tf_ratio": args.tf_ratio,
                    "tf_anneal_epochs": args.tf_anneal_epochs,
                    "tf_min": args.tf_min,
                    "tf_schedule": args.tf_schedule,
                    "dense_step_weight": args.dense_step_weight,
                    "scheduler_patience": args.open_loop_scheduler_patience,
                },
            },
        },
        "layout": dict(LAYOUT),
    }


def create_trial(args):
    validate_sequence_tag(args.sequence_tag)
    if args.trial_dir:
        trial_dir = _prepare_requested_dir(args.trial_dir)
    else:
        trial_dir = create_experiment(
            args.base_dir, prefix="trial", announce=False)
    prepare_trial_layout(trial_dir)
    save_config(trial_dir, build_trial_config(args, trial_dir))
    return trial_dir


def validate_open_loop_start(trial_dir, train_dir, val_dir,
                             training_settings=None):
    """Validate a GT-complete trial before starting its OpenLoop stage."""
    trial_dir = os.path.normpath(trial_dir)
    config_path = os.path.join(trial_dir, "config.json")
    gt_checkpoint = os.path.join(
        trial_dir, "stages/gt/phase_gt_transition/model/best_model.pt")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"试次缺少config.json: {trial_dir}")
    if not os.path.isfile(gt_checkpoint):
        raise FileNotFoundError(f"试次缺少GT best权重: {gt_checkpoint}")
    with open(config_path, encoding="utf-8") as stream:
        config = json.load(stream)
    expected_train = os.path.normpath(config["data"]["train_dir"])
    expected_val = os.path.normpath(config["data"]["val_dir"])
    actual_train = os.path.normpath(train_dir)
    actual_val = os.path.normpath(val_dir)
    if (actual_train, actual_val) != (expected_train, expected_val):
        raise ValueError(
            "OpenLoop数据目录与试次配置不一致: "
            f"expected=({expected_train}, {expected_val}) "
            f"actual=({actual_train}, {actual_val})")
    if training_settings is not None:
        expected_training = config["training"]
        expected = {
            "open_loop_epochs": expected_training["stages"]["open_loop"]["epochs"],
            "batch_size": expected_training["batch_size"],
            "num_workers": config["resources"]["num_workers"],
            "save_interval": expected_training["save_interval"],
            "periodic_eval_interval": expected_training["periodic_eval_interval"],
            "periodic_max_steps": expected_training["periodic_max_steps"],
            "seed": expected_training["seed"],
            "window_size": expected_training["window_size"],
            "episode_len": expected_training["episode_len"],
            "tf_anneal_epochs": expected_training["stages"]["open_loop"][
                "tf_anneal_epochs"],
            "tf_ratio": expected_training["stages"]["open_loop"]["tf_ratio"],
            "tf_min": expected_training["stages"]["open_loop"]["tf_min"],
            "tf_schedule": expected_training["stages"]["open_loop"][
                "tf_schedule"],
            "dense_step_weight": expected_training["stages"]["open_loop"][
                "dense_step_weight"],
            "open_loop_lr": expected_training["stages"]["open_loop"][
                "learning_rate"],
            "open_loop_scheduler_patience": expected_training["stages"][
                "open_loop"]["scheduler_patience"],
        }
        mismatches = {
            name: (expected[name], training_settings[name])
            for name in expected
            if expected[name] != training_settings[name]
        }
        if mismatches:
            raise ValueError(f"OpenLoop训练参数与试次配置不一致: {mismatches}")
    open_loop_outputs = (
        "stages/open_loop/config.json",
        "stages/open_loop/phase_open_loop_transition/loss_log.csv",
        "stages/open_loop/phase_open_loop_transition/model/best_model.pt",
    )
    existing = [relative for relative in open_loop_outputs
                if os.path.exists(os.path.join(trial_dir, relative))]
    if existing:
        raise FileExistsError(
            "OpenLoop阶段已有训练产物: " + ", ".join(existing))
    prepare_trial_layout(trial_dir)
    return {
        "trial_dir": trial_dir,
        "gt_checkpoint": gt_checkpoint,
        "train_dir": expected_train,
        "val_dir": expected_val,
    }


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
    def selected_checkpoint(stage):
        evaluated = (
            f"stages/{stage}/phase_{'gt' if stage == 'gt' else 'open_loop'}_transition/"
            "model/best_eval_model.pt")
        if os.path.isfile(os.path.join(trial_dir, evaluated)):
            return evaluated
        return (
            f"stages/{stage}/phase_{'gt' if stage == 'gt' else 'open_loop'}_transition/"
            "model/best_model.pt")

    artifacts = {
        "schema_version": 1,
        "trial_id": config["trial"]["id"],
        "completed_at": _timestamp(),
        "stages": {
            "gt": {
                "experiment_dir": LAYOUT["gt_stage"],
                "config": "stages/gt/config.json",
                "best_checkpoint": selected_checkpoint("gt"),
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
                "best_checkpoint": selected_checkpoint("open_loop"),
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
    create.add_argument("--tf-ratio", type=float, required=True)
    create.add_argument("--tf-min", type=float, required=True)
    create.add_argument(
        "--tf-schedule", choices=("linear", "staircase"), required=True)
    create.add_argument(
        "--dense-step-weight", choices=("uniform", "linear"), required=True)
    create.add_argument("--gt-lr", type=float, default=None)
    create.add_argument("--open-loop-lr", type=float, default=None)
    create.add_argument("--gt-scheduler-patience", type=int, default=None)
    create.add_argument(
        "--open-loop-scheduler-patience", type=int, default=None)

    finalize = subparsers.add_parser(
        "finalize", help="校验标准产物并写artifacts.json")
    finalize.add_argument("--trial-dir", required=True)
    validate = subparsers.add_parser(
        "validate-dataset", help="校验自动前处理数据清单")
    validate.add_argument("--dataset-manifest", required=True)
    open_loop = subparsers.add_parser(
        "validate-open-loop-start", help="校验GT试次并准备OpenLoop阶段")
    open_loop.add_argument("--trial-dir", required=True)
    open_loop.add_argument("--train-dir", required=True)
    open_loop.add_argument("--val-dir", required=True)
    open_loop.add_argument("--open-loop-epochs", type=int, required=True)
    open_loop.add_argument("--batch-size", type=int, required=True)
    open_loop.add_argument("--num-workers", type=int, required=True)
    open_loop.add_argument("--save-interval", type=int, required=True)
    open_loop.add_argument("--periodic-eval-interval", type=int, required=True)
    open_loop.add_argument("--periodic-max-steps", type=int, required=True)
    open_loop.add_argument("--seed", type=int, required=True)
    open_loop.add_argument("--window-size", type=int, required=True)
    open_loop.add_argument("--episode-len", type=int, required=True)
    open_loop.add_argument("--tf-anneal-epochs", type=int, required=True)
    open_loop.add_argument("--tf-ratio", type=float, required=True)
    open_loop.add_argument("--tf-min", type=float, required=True)
    open_loop.add_argument(
        "--tf-schedule", choices=("linear", "staircase"), required=True)
    open_loop.add_argument(
        "--dense-step-weight", choices=("uniform", "linear"), required=True)
    open_loop.add_argument("--open-loop-lr", type=float, default=None)
    open_loop.add_argument(
        "--open-loop-scheduler-patience", type=int, default=None)
    infer = subparsers.add_parser(
        "infer-sequence-tag", help="从数据清单推导简洁采集序列标签")
    infer.add_argument("--train-dir", required=True)
    infer.add_argument("--dataset-manifest", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "create":
        print(create_trial(args))
    elif args.command == "finalize":
        print(finalize_trial(args.trial_dir))
    elif args.command == "validate-dataset":
        print(json.dumps(validate_dataset_manifest(
            args.dataset_manifest), ensure_ascii=False))
    elif args.command == "validate-open-loop-start":
        print(json.dumps(validate_open_loop_start(
            args.trial_dir, args.train_dir, args.val_dir,
            training_settings={
                "open_loop_epochs": args.open_loop_epochs,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "save_interval": args.save_interval,
                "periodic_eval_interval": args.periodic_eval_interval,
                "periodic_max_steps": args.periodic_max_steps,
                "seed": args.seed,
                "window_size": args.window_size,
                "episode_len": args.episode_len,
                "tf_anneal_epochs": args.tf_anneal_epochs,
                "tf_ratio": args.tf_ratio,
                "tf_min": args.tf_min,
                "tf_schedule": args.tf_schedule,
                "dense_step_weight": args.dense_step_weight,
                "open_loop_lr": args.open_loop_lr,
                "open_loop_scheduler_patience":
                    args.open_loop_scheduler_patience,
            }), ensure_ascii=False))
    else:
        print(infer_sequence_tag(args.train_dir, args.dataset_manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
