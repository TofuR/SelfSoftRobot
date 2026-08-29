from argparse import Namespace
from datetime import datetime
import json
from pathlib import Path

import pytest

from scripts.real.manage_training_trial import create_trial, finalize_trial
from src.utils.experiment import create_experiment


def _trial_args(base_dir, trial_dir=None):
    return Namespace(
        base_dir=str(base_dir),
        trial_dir=None if trial_dir is None else str(trial_dir),
        sequence_tag="seq_demo_n15",
        capture_sequence="seq_demo",
        train_dir="data/real_seq/seq_demo_n15/train",
        val_dir="data/real_seq/seq_demo_n15/val",
        train_npz="data/real_seq/seq_demo_n15/train/seq_demo_train.npz",
        cam0_dir="real_capture/data/raw/seq_demo/cam0",
        masks_dir="sam2/masks/seq_demo_full",
        gpu_id="2",
        eval_gpu_id="2",
        gt_epochs=60,
        open_loop_epochs=240,
        batch_size=128,
        num_workers=4,
        save_interval=5,
        periodic_eval_interval=10,
        periodic_max_steps=500,
        seed=20260821,
        window_size=40,
        episode_len=40,
        tf_anneal_epochs=40,
    )


def test_create_experiment_uses_timestamp_and_daily_sequence(tmp_path):
    now = datetime(2026, 8, 22, 9, 7, 5)
    first = create_experiment(
        tmp_path, prefix="trial", now=now, announce=False)
    second = create_experiment(
        tmp_path, prefix="trial", now=now, announce=False)

    assert Path(first).name == "trial_20260822_000"
    assert Path(second).name == "trial_20260822_001"


def test_create_trial_writes_root_config_and_layout(tmp_path):
    trial_dir = Path(create_trial(_trial_args(tmp_path)))

    assert trial_dir.name.startswith("trial_")
    config = json.loads((trial_dir / "config.json").read_text())
    assert config["trial"]["id"] == trial_dir.name
    assert config["training"]["batch_size"] == 128
    assert config["training"]["stages"]["open_loop"]["epochs"] == 240
    assert (trial_dir / "stages/gt").is_dir()
    assert (trial_dir / "evaluations/open_loop/best").is_dir()


def test_requested_trial_dir_must_be_empty(tmp_path):
    trial_dir = tmp_path / "named_trial"
    trial_dir.mkdir()
    (trial_dir / "existing.txt").write_text("occupied")

    with pytest.raises(FileExistsError):
        create_trial(_trial_args(tmp_path, trial_dir))


def test_finalize_writes_artifact_index(tmp_path):
    trial_dir = Path(create_trial(_trial_args(tmp_path)))
    required = [
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
    for relative in required:
        path = trial_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")

    artifact_path = Path(finalize_trial(str(trial_dir)))
    artifacts = json.loads(artifact_path.read_text())

    assert artifacts["trial_id"] == trial_dir.name
    assert artifacts["stages"]["gt"]["best_checkpoint"].startswith("stages/gt/")
    assert artifacts["stages"]["open_loop"]["best_overlay"] == (
        "evaluations/open_loop/best/overlay")


def test_finalize_prefers_validation_selected_checkpoint(tmp_path):
    trial_dir = Path(create_trial(_trial_args(tmp_path)))
    required = [
        "commands.sh",
        "stages/gt/config.json",
        "stages/gt/phase_gt_transition/model/best_model.pt",
        "stages/open_loop/config.json",
        "stages/open_loop/phase_open_loop_transition/model/best_model.pt",
        "stages/open_loop/phase_open_loop_transition/model/best_eval_model.pt",
        "evaluations/gt/best/quantitative/summary.txt",
        "evaluations/gt/best/overlay/summary.txt",
        "evaluations/gt/best/overlay/montage.png",
        "evaluations/open_loop/best/quantitative/summary.txt",
        "evaluations/open_loop/best/overlay/summary.txt",
        "evaluations/open_loop/best/overlay/montage.png",
    ]
    for relative in required:
        path = trial_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")

    artifacts = json.loads(Path(finalize_trial(str(trial_dir))).read_text())

    assert artifacts["stages"]["open_loop"]["best_checkpoint"].endswith(
        "model/best_eval_model.pt")


def test_finalize_reports_missing_final_artifacts(tmp_path):
    trial_dir = Path(create_trial(_trial_args(tmp_path)))
    (trial_dir / "commands.sh").write_text("#!/usr/bin/env bash\n")

    with pytest.raises(FileNotFoundError, match="best_model.pt"):
        finalize_trial(str(trial_dir))
