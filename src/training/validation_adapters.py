"""Explicit evaluator adapters consumed by :class:`UnifiedTrainer`."""

from __future__ import annotations

from src.evaluation.real_transition_validation import evaluate_native_node_metrics


def transition_validation_adapter(
        *, model, phase_spec, data_dir, epoch, exp_dir, device, config):
    """Evaluate transition validation rows using deployment-aligned semantics."""
    del epoch, exp_dir
    mode = "open_loop" if "open_loop" in phase_spec.name else "gt"
    eval_config = config.get("evaluation", {})
    return evaluate_native_node_metrics(
        model,
        data_dir,
        config,
        device,
        mode=mode,
        window_len=getattr(phase_spec, "episode_len", None),
        max_steps=eval_config.get("transition_validation_max_steps"),
        seq_idx=eval_config.get("transition_validation_seq_idx", 0),
    )
