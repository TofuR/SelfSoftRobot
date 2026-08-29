from pathlib import Path

import numpy as np

from scripts.evaluation.eval_real_quant import (
    sequence_from_npz as quant_sequence_from_npz,
    split_frame_offset,
)
from scripts.evaluation.visualize_real_overlay import (
    auto_offset,
    run_rollout,
    sequence_from_npz as overlay_sequence_from_npz,
)


class _IdentityOpenLoopModel:
    """最小 rollout stub：输出上一形态，用于检查锚点语义。"""

    def __init__(self):
        import torch
        self.pc_center = torch.zeros(3)
        self.pc_scale = torch.ones(3)

    def init_z_from_action(self, action):
        import torch
        return torch.zeros((action.shape[0], 1), device=action.device)

    def forward(self, action, previous, previous_previous, latent):
        return {"skeleton": previous, "latent_z": latent}


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


def test_openloop_overlay_rollout_preserves_observed_anchor():
    import torch

    positions = np.zeros((3, 3, 2), dtype=np.float32)
    positions[0, 0] = (2.0, 4.0)
    positions[0, 1] = (3.0, 5.0)
    pred, center, scale = run_rollout(
        _IdentityOpenLoopModel(), "open_loop",
        np.zeros((3, 1), dtype=np.float32), positions,
        window_size=2, norm_factor=1.0, device=torch.device("cpu"), K=2)

    expected = positions[0].T
    np.testing.assert_allclose(pred[0] * scale + center, expected)
