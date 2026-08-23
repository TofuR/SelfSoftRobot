from pathlib import Path

import numpy as np

from scripts.evaluation.eval_real_quant import (
    sequence_from_npz as quant_sequence_from_npz,
    split_frame_offset,
)
from scripts.evaluation.visualize_real_overlay import (
    auto_offset,
    sequence_from_npz as overlay_sequence_from_npz,
)


def _write_npz(path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        positions=np.zeros((frames, 3, 15), dtype=np.float32),
        actions=np.zeros((frames, 6), dtype=np.float32),
    )


def test_multirun_val_uses_matching_train_length(tmp_path):
    train = tmp_path / "train"
    val = tmp_path / "val"
    run_a_train = train / "seq_a_train.npz"
    run_b_train = train / "seq_b_train.npz"
    run_b_val = val / "seq_b_val.npz"
    _write_npz(run_a_train, 11)
    _write_npz(run_b_train, 29)
    _write_npz(run_b_val, 7)

    assert split_frame_offset(str(val), str(run_b_val)) == 29
    assert auto_offset(str(val), str(run_b_val)) == 29


def test_sequence_name_comes_from_selected_npz():
    path = "/data/combined/val/seq_20260819_182519_val.npz"
    expected = "seq_20260819_182519"
    assert quant_sequence_from_npz(path) == expected
    assert overlay_sequence_from_npz(path) == expected
