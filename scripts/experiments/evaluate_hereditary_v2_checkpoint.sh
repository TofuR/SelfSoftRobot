#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 EXPERIMENT_DIR" >&2
  exit 2
fi

exp_dir=${1%/}
checkpoint="$exp_dir/phase_hereditary/model/best_model.pt"
data_dir="data/real_seq/seq_20260819_10hz_n15_sam2_robot_mm/val"
eval_root="$exp_dir/evaluations"

if [[ ! -s "$checkpoint" ]]; then
  echo "Missing checkpoint: $checkpoint" >&2
  exit 3
fi
if [[ -e "$eval_root/COMPLETE" ]]; then
  echo "Evaluation already complete: $eval_root" >&2
  exit 4
fi

mkdir -p "$eval_root/continuous" "$eval_root/window40" "$eval_root/spectrum"

python scripts/evaluation/eval_real_quant.py \
  --checkpoint "$checkpoint" \
  --data_dir "$data_dir" \
  --mode gt \
  --no-ndi \
  --out "$eval_root/continuous" \
  2>&1 | tee "$eval_root/continuous.log"

python scripts/evaluation/eval_real_quant.py \
  --checkpoint "$checkpoint" \
  --data_dir "$data_dir" \
  --mode open_loop \
  --window-len 40 \
  --no-ndi \
  --out "$eval_root/window40" \
  2>&1 | tee "$eval_root/window40.log"

python scripts/evaluation/plot_hysteresis_spectrum.py \
  --checkpoint "$checkpoint" \
  --out "$eval_root/spectrum" \
  2>&1 | tee "$eval_root/spectrum.log"

python - "$checkpoint" <<'PY' | tee "$eval_root/residual_scale.txt"
import sys

from src.utils.model_loader import load_model

checkpoint = sys.argv[1]
info = load_model(checkpoint, device="cpu")
model = info["model"]
print(f"checkpoint = {checkpoint}")
print(f"residual_scale = {model.residual_scale.item():.9g}")
print(f"residual_scale_max = {model.residual_scale_max:.9g}")
print(f"upper_bound_ratio = {model.residual_scale.item() / model.residual_scale_max:.9g}")
PY

touch "$eval_root/COMPLETE"
echo "Evaluation complete: $eval_root"
