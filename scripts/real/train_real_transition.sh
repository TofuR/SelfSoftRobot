#!/usr/bin/env bash
# 真实序列训练流水线：一个试次目录收纳 GT、OpenLoop、周期评价和最终评价。
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "用法: $0 DATA_TRAIN_DIR [DATA_VAL_DIR]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA_TRAIN_DIR="$1"
DATA_VAL_DIR="${2:-$(dirname "$DATA_TRAIN_DIR")/val}"
GPU_ID="${GPU_ID:-0}"
EVAL_GPU_ID="${EVAL_GPU_ID:-$GPU_ID}"
GT_EPOCHS="${GT_EPOCHS:-60}"
OPEN_LOOP_EPOCHS="${OPEN_LOOP_EPOCHS:-240}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
PERIODIC_EVAL_INTERVAL="${PERIODIC_EVAL_INTERVAL:-10}"
PERIODIC_MAX_STEPS="${PERIODIC_MAX_STEPS:-500}"
SEED="${SEED:-20260821}"
WINDOW_SIZE="${WINDOW_SIZE:-40}"
EPISODE_LEN="${EPISODE_LEN:-40}"
TF_ANNEAL_EPOCHS="${TF_ANNEAL_EPOCHS:-40}"
START_STAGE="${START_STAGE:-gt}"
REQUESTED_RUN_DIR="${RUN_DIR:-}"
SEQ_TAG="$(basename "$(dirname "$DATA_TRAIN_DIR")")"
TRAIN_NPZ="$(find "$DATA_TRAIN_DIR" -maxdepth 1 -type f -name '*.npz' | sort | head -1)"
VAL_NPZ="$(find "$DATA_VAL_DIR" -maxdepth 1 -type f -name '*.npz' | sort | head -1)"

if [[ -z "$TRAIN_NPZ" ]]; then
  echo "训练目录中没有NPZ: $DATA_TRAIN_DIR" >&2
  exit 2
fi

if [[ -n "$VAL_NPZ" ]]; then
  DEFAULT_CAPTURE_SEQ="$(basename "$VAL_NPZ" _val.npz)"
else
  DEFAULT_CAPTURE_SEQ="$(basename "$TRAIN_NPZ" _train.npz)"
fi
CAPTURE_SEQ="${CAPTURE_SEQ:-$DEFAULT_CAPTURE_SEQ}"
CAM0_DIR="${CAM0_DIR:-real_capture/data/raw/${CAPTURE_SEQ}/cam0}"
MASKS_DIR="${MASKS_DIR:-sam2/masks/${CAPTURE_SEQ}_full}"
TRIAL_BASE="train_log/real_pipeline/${SEQ_TAG}"
DATASET_MANIFEST="${DATASET_MANIFEST:-$(dirname "$DATA_TRAIN_DIR")/dataset_manifest.json}"
NDI_CSV="real_capture/data/raw/${CAPTURE_SEQ}/ndi.csv"
FRAME_TIMES_FILE="real_capture/data/raw/${CAPTURE_SEQ}/frame_times.txt"
HAS_NDI=0
if [[ -f "$NDI_CSV" && -f "$FRAME_TIMES_FILE" ]]; then
  HAS_NDI=1
fi

if (( SAVE_INTERVAL <= 0 || PERIODIC_EVAL_INTERVAL <= 0 || PERIODIC_MAX_STEPS <= 0 )); then
  echo "SAVE_INTERVAL、PERIODIC_EVAL_INTERVAL和PERIODIC_MAX_STEPS必须为正数" >&2
  exit 2
fi
if [[ "$START_STAGE" != "gt" && "$START_STAGE" != "open_loop" ]]; then
  echo "START_STAGE必须是gt或open_loop" >&2
  exit 2
fi
if (( PERIODIC_EVAL_INTERVAL % SAVE_INTERVAL != 0 )); then
  echo "PERIODIC_EVAL_INTERVAL必须是SAVE_INTERVAL的整数倍" >&2
  exit 2
fi
if [[ "$START_STAGE" == "gt" && -f "$DATASET_MANIFEST" ]]; then
  python scripts/real/manage_training_trial.py validate-dataset \
    --dataset-manifest "$DATASET_MANIFEST"
fi

CREATE_TRIAL_CMD=(python scripts/real/manage_training_trial.py create
  --base-dir "$TRIAL_BASE"
  --sequence-tag "$SEQ_TAG"
  --capture-sequence "$CAPTURE_SEQ"
  --train-dir "$DATA_TRAIN_DIR"
  --val-dir "$DATA_VAL_DIR"
  --train-npz "$TRAIN_NPZ"
  --cam0-dir "$CAM0_DIR"
  --masks-dir "$MASKS_DIR"
  --ndi-csv "$NDI_CSV"
  --frame-times "$FRAME_TIMES_FILE"
  --ndi-available "$HAS_NDI"
  --gpu-id "$GPU_ID"
  --eval-gpu-id "$EVAL_GPU_ID"
  --gt-epochs "$GT_EPOCHS"
  --open-loop-epochs "$OPEN_LOOP_EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --save-interval "$SAVE_INTERVAL"
  --periodic-eval-interval "$PERIODIC_EVAL_INTERVAL"
  --periodic-max-steps "$PERIODIC_MAX_STEPS"
  --seed "$SEED"
  --window-size "$WINDOW_SIZE"
  --episode-len "$EPISODE_LEN"
  --tf-anneal-epochs "$TF_ANNEAL_EPOCHS")
if [[ -f "$DATASET_MANIFEST" ]]; then
  CREATE_TRIAL_CMD+=(--dataset-manifest "$DATASET_MANIFEST")
fi
if [[ "$START_STAGE" == "gt" ]]; then
  if [[ -n "$REQUESTED_RUN_DIR" ]]; then
    CREATE_TRIAL_CMD+=(--trial-dir "$REQUESTED_RUN_DIR")
  fi
  RUN_DIR="$("${CREATE_TRIAL_CMD[@]}")"
else
  if [[ -z "$REQUESTED_RUN_DIR" ]]; then
    echo "START_STAGE=open_loop时必须设置RUN_DIR" >&2
    exit 2
  fi
  RUN_DIR="$REQUESTED_RUN_DIR"
  python scripts/real/manage_training_trial.py validate-open-loop-start \
    --trial-dir "$RUN_DIR" --train-dir "$DATA_TRAIN_DIR" --val-dir "$DATA_VAL_DIR" \
    --open-loop-epochs "$OPEN_LOOP_EPOCHS" --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" --save-interval "$SAVE_INTERVAL" \
    --periodic-eval-interval "$PERIODIC_EVAL_INTERVAL" \
    --periodic-max-steps "$PERIODIC_MAX_STEPS" --seed "$SEED" \
    --window-size "$WINDOW_SIZE" --episode-len "$EPISODE_LEN" \
    --tf-anneal-epochs "$TF_ANNEAL_EPOCHS"
fi

GT_EXP_DIR="$RUN_DIR/stages/gt"
OPEN_LOOP_EXP_DIR="$RUN_DIR/stages/open_loop"
GT_PERIODIC_DIR="$RUN_DIR/evaluations/gt/periodic"
OPEN_LOOP_PERIODIC_DIR="$RUN_DIR/evaluations/open_loop/periodic"
GT_BEST_QUANT_DIR="$RUN_DIR/evaluations/gt/best/quantitative"
GT_BEST_OVERLAY_DIR="$RUN_DIR/evaluations/gt/best/overlay"
OPEN_LOOP_BEST_QUANT_DIR="$RUN_DIR/evaluations/open_loop/best/quantitative"
OPEN_LOOP_BEST_OVERLAY_DIR="$RUN_DIR/evaluations/open_loop/best/overlay"
CALIBRATION_FILE="${CALIBRATION_FILE:-$RUN_DIR/diagnostics/state_to_ndi_same_sequence.npz}"
STATUS_FILE="$RUN_DIR/status.txt"
COMMAND_FILE="$RUN_DIR/commands.sh"

if [[ "$START_STAGE" == "open_loop" && "$HAS_NDI" -eq 1 \
      && ! -f "$CALIBRATION_FILE" ]]; then
  echo "OpenLoop评价需要已完成的NDI标定文件: $CALIBRATION_FILE" >&2
  exit 2
fi

on_exit() {
  code=$?
  if [[ -n "${ACTIVE_WATCHER_PID:-}" ]]; then
    kill "$ACTIVE_WATCHER_PID" 2>/dev/null || true
  fi
  if [[ $code -ne 0 ]]; then
    printf 'FAILED exit=%s at %s\n' "$code" "$(date --iso-8601=seconds)" >> "$STATUS_FILE"
  fi
}
trap on_exit EXIT

COMMON_ARGS=(
  --data_dir "$DATA_TRAIN_DIR"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --window_size "$WINDOW_SIZE"
  --episode_len "$EPISODE_LEN"
  --eval_interval 0
  --save_interval "$SAVE_INTERVAL"
  --seed "$SEED"
)
OVERLAY_ARGS=(--overlay)
if [[ -d "$CAM0_DIR" ]]; then OVERLAY_ARGS+=(--cam0 "$CAM0_DIR"); fi
if [[ -d "$MASKS_DIR" ]]; then OVERLAY_ARGS+=(--masks "$MASKS_DIR"); fi

record_command() {
  local gpu="$1"
  shift
  printf 'CUDA_VISIBLE_DEVICES=%q MPLCONFIGDIR=/tmp/selfsoftrobot-mpl PYTHONUNBUFFERED=1 ' "$gpu" >> "$COMMAND_FILE"
  printf '%q ' "$@" >> "$COMMAND_FILE"
  printf '\n' >> "$COMMAND_FILE"
}

if [[ "$START_STAGE" == "gt" || ! -f "$COMMAND_FILE" ]]; then
  {
    echo '#!/usr/bin/env bash'
    echo '# 本文件记录该试次实际执行的训练与评价命令。'
    echo 'set -euo pipefail'
  } > "$COMMAND_FILE"
fi
chmod +x "$COMMAND_FILE"

GT_CKPT="$GT_EXP_DIR/phase_gt_transition/model/best_model.pt"
GT_EVAL_CKPT="$GT_EXP_DIR/phase_gt_transition/model/best_eval_model.pt"
if [[ -f "$GT_EVAL_CKPT" ]]; then
  GT_CKPT="$GT_EVAL_CKPT"
fi
if [[ "$START_STAGE" == "gt" ]]; then
  GT_CMD=(python scripts/training/train_transition.py --mode gt
    --n_epochs "$GT_EPOCHS" --experiment-dir "$GT_EXP_DIR" "${COMMON_ARGS[@]}")
  record_command "$GPU_ID" "${GT_CMD[@]}"

  printf 'RUNNING gt started=%s train_gpu=%s eval_gpu=%s\n' \
    "$(date --iso-8601=seconds)" "$GPU_ID" "$EVAL_GPU_ID" > "$STATUS_FILE"
  (set -o pipefail; CUDA_VISIBLE_DEVICES="$GPU_ID" MPLCONFIGDIR=/tmp/selfsoftrobot-mpl \
    PYTHONUNBUFFERED=1 "${GT_CMD[@]}" 2>&1 | tee "$GT_EXP_DIR/train.log") &
  GT_TRAIN_PID=$!
  GT_WATCH_CMD=(python scripts/evaluation/watch_best_checkpoint.py
    --experiment-dir "$GT_EXP_DIR" --mode gt --data-dir "$DATA_VAL_DIR"
    --out-root "$GT_PERIODIC_DIR" --interval "$PERIODIC_EVAL_INTERVAL"
    --max-steps "$PERIODIC_MAX_STEPS" --parent-pid "$GT_TRAIN_PID" --no-ndi
    "${OVERLAY_ARGS[@]}")
  record_command "$EVAL_GPU_ID" "${GT_WATCH_CMD[@]}"
  (set -o pipefail; CUDA_VISIBLE_DEVICES="$EVAL_GPU_ID" MPLCONFIGDIR=/tmp/selfsoftrobot-mpl \
    PYTHONUNBUFFERED=1 "${GT_WATCH_CMD[@]}" 2>&1 | tee "$GT_PERIODIC_DIR/watch.log") &
  GT_WATCH_PID=$!
  ACTIVE_WATCHER_PID="$GT_WATCH_PID"
  wait "$GT_TRAIN_PID"
  wait "$GT_WATCH_PID"
  ACTIVE_WATCHER_PID=""
  if [[ -f "$GT_EVAL_CKPT" ]]; then
    GT_CKPT="$GT_EVAL_CKPT"
  fi

  GT_EVAL_CMD=(python scripts/evaluation/eval_real_quant.py
    --checkpoint "$GT_CKPT" --data_dir "$DATA_VAL_DIR" --mode gt
    --out "$GT_BEST_QUANT_DIR")
  if (( HAS_NDI )); then
    GT_EVAL_CMD+=(--save-calibration "$CALIBRATION_FILE")
  else
    GT_EVAL_CMD+=(--no-ndi)
  fi
  record_command "$EVAL_GPU_ID" "${GT_EVAL_CMD[@]}"
  printf 'RUNNING gt_best_evaluation started=%s checkpoint=%s\n' \
    "$(date --iso-8601=seconds)" "$GT_CKPT" >> "$STATUS_FILE"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU_ID" MPLCONFIGDIR=/tmp/selfsoftrobot-mpl \
    PYTHONUNBUFFERED=1 "${GT_EVAL_CMD[@]}" 2>&1 | tee "$GT_BEST_QUANT_DIR/run.log"

  GT_OVERLAY_CMD=(python scripts/evaluation/visualize_real_overlay.py
    --checkpoint "$GT_CKPT" --data_dir "$DATA_VAL_DIR" --mode gt
    --out "$GT_BEST_OVERLAY_DIR")
  if [[ -d "$CAM0_DIR" ]]; then GT_OVERLAY_CMD+=(--cam0 "$CAM0_DIR"); fi
  if [[ -d "$MASKS_DIR" ]]; then GT_OVERLAY_CMD+=(--masks "$MASKS_DIR"); fi
  record_command "$EVAL_GPU_ID" "${GT_OVERLAY_CMD[@]}"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU_ID" MPLCONFIGDIR=/tmp/selfsoftrobot-mpl \
    PYTHONUNBUFFERED=1 "${GT_OVERLAY_CMD[@]}" 2>&1 | tee "$GT_BEST_OVERLAY_DIR/run.log"
fi

OPEN_LOOP_CMD=(python scripts/training/train_transition.py --mode open_loop
  --n_epochs "$OPEN_LOOP_EPOCHS" --experiment-dir "$OPEN_LOOP_EXP_DIR"
  --init_from "$GT_CKPT" --tf_ratio 1.0
  --tf_anneal_epochs "$TF_ANNEAL_EPOCHS" --tf_min 0.0
  --tf_schedule staircase "${COMMON_ARGS[@]}")
record_command "$GPU_ID" "${OPEN_LOOP_CMD[@]}"

printf 'RUNNING open_loop started=%s initialization=%s\n' \
  "$(date --iso-8601=seconds)" "$GT_CKPT" >> "$STATUS_FILE"
(set -o pipefail; CUDA_VISIBLE_DEVICES="$GPU_ID" MPLCONFIGDIR=/tmp/selfsoftrobot-mpl \
  PYTHONUNBUFFERED=1 "${OPEN_LOOP_CMD[@]}" 2>&1 | tee "$OPEN_LOOP_EXP_DIR/train.log") &
OPEN_LOOP_TRAIN_PID=$!
OPEN_LOOP_CKPT="$OPEN_LOOP_EXP_DIR/phase_open_loop_transition/model/best_model.pt"
OPEN_LOOP_WATCH_CMD=(python scripts/evaluation/watch_best_checkpoint.py
  --experiment-dir "$OPEN_LOOP_EXP_DIR" --mode open_loop --data-dir "$DATA_VAL_DIR"
  --out-root "$OPEN_LOOP_PERIODIC_DIR" --interval "$PERIODIC_EVAL_INTERVAL"
  --max-steps "$PERIODIC_MAX_STEPS" --window-len "$EPISODE_LEN"
  --parent-pid "$OPEN_LOOP_TRAIN_PID"
  "${OVERLAY_ARGS[@]}")
if (( HAS_NDI )); then
  OPEN_LOOP_WATCH_CMD+=(--calibration-file "$CALIBRATION_FILE")
else
  OPEN_LOOP_WATCH_CMD+=(--no-ndi)
fi
record_command "$EVAL_GPU_ID" "${OPEN_LOOP_WATCH_CMD[@]}"
(set -o pipefail; CUDA_VISIBLE_DEVICES="$EVAL_GPU_ID" MPLCONFIGDIR=/tmp/selfsoftrobot-mpl \
  PYTHONUNBUFFERED=1 "${OPEN_LOOP_WATCH_CMD[@]}" 2>&1 \
  | tee "$OPEN_LOOP_PERIODIC_DIR/watch.log") &
OPEN_LOOP_WATCH_PID=$!
ACTIVE_WATCHER_PID="$OPEN_LOOP_WATCH_PID"
wait "$OPEN_LOOP_TRAIN_PID"
wait "$OPEN_LOOP_WATCH_PID"
ACTIVE_WATCHER_PID=""
OPEN_LOOP_EVAL_CKPT="$OPEN_LOOP_EXP_DIR/phase_open_loop_transition/model/best_eval_model.pt"
if [[ -f "$OPEN_LOOP_EVAL_CKPT" ]]; then
  OPEN_LOOP_CKPT="$OPEN_LOOP_EVAL_CKPT"
fi

OPEN_LOOP_EVAL_CMD=(python scripts/evaluation/eval_real_quant.py
  --checkpoint "$OPEN_LOOP_CKPT" --data_dir "$DATA_VAL_DIR"
  --mode open_loop --window-len "$EPISODE_LEN"
  --out "$OPEN_LOOP_BEST_QUANT_DIR")
if (( HAS_NDI )); then
  OPEN_LOOP_EVAL_CMD+=(--calibration-file "$CALIBRATION_FILE")
else
  OPEN_LOOP_EVAL_CMD+=(--no-ndi)
fi
record_command "$EVAL_GPU_ID" "${OPEN_LOOP_EVAL_CMD[@]}"
printf 'RUNNING open_loop_best_evaluation started=%s checkpoint=%s\n' \
  "$(date --iso-8601=seconds)" "$OPEN_LOOP_CKPT" >> "$STATUS_FILE"
CUDA_VISIBLE_DEVICES="$EVAL_GPU_ID" MPLCONFIGDIR=/tmp/selfsoftrobot-mpl \
  PYTHONUNBUFFERED=1 "${OPEN_LOOP_EVAL_CMD[@]}" 2>&1 | tee "$OPEN_LOOP_BEST_QUANT_DIR/run.log"

OPEN_LOOP_OVERLAY_CMD=(python scripts/evaluation/visualize_real_overlay.py
  --checkpoint "$OPEN_LOOP_CKPT" --data_dir "$DATA_VAL_DIR"
  --mode open_loop --window-len "$EPISODE_LEN" --with-onestep
  --out "$OPEN_LOOP_BEST_OVERLAY_DIR")
if [[ -d "$CAM0_DIR" ]]; then OPEN_LOOP_OVERLAY_CMD+=(--cam0 "$CAM0_DIR"); fi
if [[ -d "$MASKS_DIR" ]]; then OPEN_LOOP_OVERLAY_CMD+=(--masks "$MASKS_DIR"); fi
record_command "$EVAL_GPU_ID" "${OPEN_LOOP_OVERLAY_CMD[@]}"
CUDA_VISIBLE_DEVICES="$EVAL_GPU_ID" MPLCONFIGDIR=/tmp/selfsoftrobot-mpl \
  PYTHONUNBUFFERED=1 "${OPEN_LOOP_OVERLAY_CMD[@]}" 2>&1 | tee "$OPEN_LOOP_BEST_OVERLAY_DIR/run.log"

FINALIZE_CMD=(python scripts/real/manage_training_trial.py finalize --trial-dir "$RUN_DIR")
record_command "$EVAL_GPU_ID" "${FINALIZE_CMD[@]}"
"${FINALIZE_CMD[@]}"
printf 'COMPLETE finished=%s\n' "$(date --iso-8601=seconds)" >> "$STATUS_FILE"
trap - EXIT
echo "全部完成：$RUN_DIR"
