#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "Usage: $0 GPU EXPERIMENT_DIR LABEL BURNIN_MODE TAU_MAX N_PLAY RESIDUAL_MAX SEED" >&2
  exit 2
fi

physical_gpu=$1
exp_dir=${2%/}
label=$3
burnin_mode=$4
tau_max=$5
n_play=$6
residual_max=$7
seed=$8
data_dir="data/real_seq/seq_20260819_10hz_n15_sam2_robot_mm/train"

if [[ -e "$exp_dir" ]]; then
  echo "Refusing to reuse existing experiment directory: $exp_dir" >&2
  exit 3
fi

mkdir -p "$(dirname "$exp_dir")"
mkdir "$exp_dir"

export CUDA_VISIBLE_DEVICES="$physical_gpu"
export MPLCONFIGDIR="/tmp/selfsoftrobot-mpl-$label"
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
mkdir -p "$MPLCONFIGDIR"

{
  echo "label=$label"
  echo "physical_gpu=$physical_gpu"
  echo "experiment_dir=$exp_dir"
  echo "burnin_mode=$burnin_mode"
  echo "tau_max=$tau_max"
  echo "n_play=$n_play"
  echo "n_maxwell=6"
  echo "residual_scale_max=$residual_max"
  echo "seed=$seed"
  echo "n_epochs=500"
  echo "batch_size=4"
  echo "lr=0.001"
  echo "episode_len=40"
  echo "dt=0.1"
} > "$exp_dir/run_manifest.txt"

python scripts/training/train_transition.py \
  --mode hereditary \
  --data_dir "$data_dir" \
  --episode_len 40 \
  --dt 0.1 \
  --n_play "$n_play" \
  --n_maxwell 6 \
  --tau_max "$tau_max" \
  --burnin_mode "$burnin_mode" \
  --residual_scale_max "$residual_max" \
  --n_epochs 500 \
  --batch_size 4 \
  --num_workers 4 \
  --lr 0.001 \
  --seed "$seed" \
  --experiment-dir "$exp_dir" \
  2>&1 | tee "$exp_dir/session.log"

scripts/experiments/evaluate_hereditary_v2_checkpoint.sh "$exp_dir" \
  2>&1 | tee "$exp_dir/evaluation_bundle.log"

touch "$exp_dir/RUN_COMPLETE"
echo "Training and evaluation complete: $exp_dir"
