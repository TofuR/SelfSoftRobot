#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU_INDEX RUN_DIR [extra train_transition args ...]" >&2
  exit 2
fi

gpu_index="$1"
run_dir="$2"
shift 2
ishsm_cpu_threads="${ISHSM_CPU_THREADS:-4}"

if [[ -e "$run_dir" ]]; then
  echo "refusing to overwrite existing run directory: $run_dir" >&2
  exit 3
fi

mkdir -p "$run_dir"
{
  printf 'CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q MKL_NUM_THREADS=%q MPLCONFIGDIR=%q python %q' \
    "$gpu_index" "$ishsm_cpu_threads" "$ishsm_cpu_threads" \
    "/tmp/mplconfig-${run_dir//\//-}" \
    "scripts/training/train_transition.py"
  printf ' %q' \
    --mode ishsm \
    --data_dir workspace/data/derived/ishsm_v1_20260902_000/fit \
    --val_dir workspace/data/derived/ishsm_v1_20260902_000/dev \
    --experiment-dir "$run_dir" \
    --n_epochs 100 --batch_size 64 --num_workers 2 \
    --episode_len 40 --window_size 40 --dt 0.1 \
    --n_bend_modes 8 --section_intervals 7,7 \
    --tau_min 0.3 --tau_max 2.0 \
    --h0_reference monotone_spline --h0_knots 5 --h0_fit_steps 500 \
    --h0_fit_objective geometry --h0_geometry_weight 1.0 \
    --h0_endpoint_weight 0.25 --endpoint_loss_weight 1.0 \
    --bend_loss_weight 0.05 \
    --disable_persistent_state --persistence_init 0.1 \
    --ishsm_observation_update innovation \
    --ishsm_observation_gain_init 0.25 \
    --ishsm_training_reanchor_intervals 0,5,10,20 \
    --ishsm_tau_parameterization shared_bending \
    --ishsm_validation_protocol single_anchor --ishsm_reanchor_interval 40 \
    --validation_interval 2 --validation_max_steps 800 \
    --validation_warmup 2 --early_stopping_patience 8 \
    --scheduler_patience 3 --save_interval 2 --eval_interval 0 \
    --seed 42 "$@"
  printf '\n'
} > "$run_dir/command.txt"

date --iso-8601=seconds > "$run_dir/STARTED_AT"
set +e
CUDA_VISIBLE_DEVICES="$gpu_index" \
OMP_NUM_THREADS="$ishsm_cpu_threads" \
MKL_NUM_THREADS="$ishsm_cpu_threads" \
MPLCONFIGDIR="/tmp/mplconfig-${run_dir//\//-}" \
python scripts/training/train_transition.py \
  --mode ishsm \
  --data_dir workspace/data/derived/ishsm_v1_20260902_000/fit \
  --val_dir workspace/data/derived/ishsm_v1_20260902_000/dev \
  --experiment-dir "$run_dir" \
  --n_epochs 100 --batch_size 64 --num_workers 2 \
  --episode_len 40 --window_size 40 --dt 0.1 \
  --n_bend_modes 8 --section_intervals 7,7 \
  --tau_min 0.3 --tau_max 2.0 \
  --h0_reference monotone_spline --h0_knots 5 --h0_fit_steps 500 \
  --h0_fit_objective geometry --h0_geometry_weight 1.0 \
  --h0_endpoint_weight 0.25 --endpoint_loss_weight 1.0 \
  --bend_loss_weight 0.05 \
  --disable_persistent_state --persistence_init 0.1 \
  --ishsm_observation_update innovation \
  --ishsm_observation_gain_init 0.25 \
  --ishsm_training_reanchor_intervals 0,5,10,20 \
  --ishsm_tau_parameterization shared_bending \
  --ishsm_validation_protocol single_anchor --ishsm_reanchor_interval 40 \
  --validation_interval 2 --validation_max_steps 800 \
  --validation_warmup 2 --early_stopping_patience 8 \
  --scheduler_patience 3 --save_interval 2 --eval_interval 0 \
  --seed 42 "$@" 2>&1 | tee "$run_dir/session.log"
status=${PIPESTATUS[0]}
set -e

printf '%s\n' "$status" > "$run_dir/exit_code.txt"
date --iso-8601=seconds > "$run_dir/FINISHED_AT"
if [[ $status -eq 0 ]]; then
  touch "$run_dir/TRAINING_COMPLETE"
else
  touch "$run_dir/TRAINING_FAILED"
fi
exit "$status"
