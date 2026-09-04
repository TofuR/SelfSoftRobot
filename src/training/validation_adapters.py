"""Explicit evaluator adapters consumed by :class:`UnifiedTrainer`."""

from __future__ import annotations

from src.evaluation.real_transition_validation import evaluate_native_node_metrics


def transition_validation_adapter(
        *, model, phase_spec, data_dir, epoch, exp_dir, device, config):
    """Evaluate transition validation rows using deployment-aligned semantics."""
    del epoch, exp_dir
    is_hereditary = "hereditary" in phase_spec.name
    if "ishsm" in phase_spec.name:
        protocol = config.get("evaluation", {}).get(
            "ishsm_validation_protocol", "single_anchor")
        if protocol == "single_anchor":
            mode = "ishsm"
        elif protocol == "periodic_40":
            mode = "ishsm_periodic"
        else:
            raise ValueError(
                f"未知 ISHSM validation protocol: {protocol!r}")
    else:
        mode = "open_loop" if "open_loop" in phase_spec.name else "gt"
    eval_config = config.get("evaluation", {})
    return evaluate_native_node_metrics(
        model,
        data_dir,
        config,
        device,
        mode=mode,
        window_len=(config.get("evaluation", {}).get(
            "ishsm_reanchor_interval", 40)
            if mode == "ishsm_periodic" else
            getattr(phase_spec, "episode_len", None)),
        max_steps=eval_config.get("transition_validation_max_steps"),
        seq_idx=(None if mode == "ishsm" or is_hereditary else
                 eval_config.get("transition_validation_seq_idx", 0)),
    )
