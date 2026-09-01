#!/usr/bin/env bash
# 真实序列训练流水线：一个试次目录收纳 GT、OpenLoop、内部验证和最终评价。
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
TEST_MAX_STEPS="${TEST_MAX_STEPS:-}"
SEED="${SEED:-20260821}"
WINDOW_SIZE="${WINDOW_SIZE:-40}"
EPISODE_LEN="${EPISODE_LEN:-40}"
TF_ANNEAL_EPOCHS="${TF_ANNEAL_EPOCHS:-40}"
TF_RATIO="${TF_RATIO:-1.0}"
TF_MIN="${TF_MIN:-0.0}"
TF_SCHEDULE="${TF_SCHEDULE:-linear}"
DENSE_STEP_WEIGHT="${DENSE_STEP_WEIGHT:-uniform}"
GT_LR="${GT_LR:-}"
OPEN_LOOP_LR="${OPEN_LOOP_LR:-}"
GT_SCHEDULER_PATIENCE="${GT_SCHEDULER_PATIENCE:-}"
OPEN_LOOP_SCHEDULER_PATIENCE="${OPEN_LOOP_SCHEDULER_PATIENCE:-}"
START_STAGE="${START_STAGE:-gt}"
PIPELINE_PREFLIGHT_ONLY="${PIPELINE_PREFLIGHT_ONLY:-0}"
REQUESTED_RUN_DIR="${RUN_DIR:-}"
TRAIN_NPZ="$(find "$DATA_TRAIN_DIR" -maxdepth 1 -type f -name '*.npz' | sort | head -1)"
VAL_NPZ="$(find "$DATA_VAL_DIR" -maxdepth 1 -type f -name '*.npz' | sort | head -1)"
if [[ -n "${DATASET_MANIFEST:-}" ]]; then
  DATASET_MANIFEST="$DATASET_MANIFEST"
elif [[ "$(basename "$(dirname "$DATA_TRAIN_DIR")")" == "splits" ]]; then
  DATASET_MANIFEST="$(dirname "$(dirname "$DATA_TRAIN_DIR")")/dataset_manifest.json"
else
  DATASET_MANIFEST="$(dirname "$DATA_TRAIN_DIR")/dataset_manifest.json"
fi
DATA_TEST_DIR="${DATA_TEST_DIR:-}"
TEST_SEQUENCE="${TEST_SEQUENCE:-}"

if [[ -z "$TRAIN_NPZ" ]]; then
  echo "训练目录中没有NPZ: $DATA_TRAIN_DIR" >&2
  exit 2
fi
if [[ -z "$VAL_NPZ" ]]; then
  echo "验证目录中没有NPZ: $DATA_VAL_DIR" >&2
  exit 2
fi

DEFAULT_CAPTURE_SEQ="$(basename "$VAL_NPZ" _val.npz)"
if [[ -n "${CAPTURE_SEQ:-}" ]]; then
  CAPTURE_SEQ="$CAPTURE_SEQ"
elif [[ -f "$DATASET_MANIFEST" ]] && \
    RESOLVED_CAPTURE_SEQ="$(python scripts/real/manage_training_trial.py \
      resolve-dataset-role --dataset-manifest "$DATASET_MANIFEST" \
      --role val --field sequence_id 2>/dev/null)"; then
  CAPTURE_SEQ="$RESOLVED_CAPTURE_SEQ"
else
  CAPTURE_SEQ="$DEFAULT_CAPTURE_SEQ"
fi
CAMERA="${CAMERA:-cam0}"
if [[ -n "${SEQUENCE_TAG:-}" ]]; then
  SEQ_TAG="$SEQUENCE_TAG"
else
  INFER_TAG_CMD=(python scripts/real/manage_training_trial.py infer-sequence-tag
    --train-dir "$DATA_TRAIN_DIR")
  if [[ -f "$DATASET_MANIFEST" ]]; then
    INFER_TAG_CMD+=(--dataset-manifest "$DATASET_MANIFEST")
  fi
  SEQ_TAG="$("${INFER_TAG_CMD[@]}")"
fi
DATASET_ID="$(basename "$(dirname "$DATA_TRAIN_DIR")")"
if [[ -f "$DATASET_MANIFEST" ]] && \
    RESOLVED_DATASET_ID="$(python scripts/real/manage_training_trial.py \
      resolve-dataset-role --dataset-manifest "$DATASET_MANIFEST" \
      --role train --field dataset_id 2>/dev/null)"; then
  DATASET_ID="$RESOLVED_DATASET_ID"
fi
resolve_real_path_for() {
  local sequence_id="$1"
  local field="$2"
  python scripts/real/manage_training_trial.py resolve-real-path \
    --sequence-id "$sequence_id" --dataset-id "$DATASET_ID" \
    --sequence-tag "$SEQ_TAG" --camera "$CAMERA" --field "$field"
}
resolve_real_path() {
  resolve_real_path_for "$CAPTURE_SEQ" "$1"
}
CAM0_DIR="${CAM0_DIR:-$(resolve_real_path camera_dir)}"
MASKS_DIR="${MASKS_DIR:-$(resolve_real_path masks_dir)}"
TRIAL_BASE="${TRIAL_BASE:-$(resolve_real_path trial_base)}"
NDI_CSV="${NDI_CSV:-$(resolve_real_path ndi_csv)}"
FRAME_TIMES_FILE="${FRAME_TIMES_FILE:-$(resolve_real_path frame_times)}"
if [[ -f "$DATASET_MANIFEST" ]]; then
  if [[ -z "$DATA_TEST_DIR" ]]; then
    if RESOLVED_TEST_DIR="$(python scripts/real/manage_training_trial.py \
        resolve-dataset-role --dataset-manifest "$DATASET_MANIFEST" \
        --role test --field dir 2>/dev/null)"; then
      DATA_TEST_DIR="$RESOLVED_TEST_DIR"
    fi
  fi
  if [[ -z "$TEST_SEQUENCE" ]]; then
    if RESOLVED_TEST_SEQUENCE="$(python scripts/real/manage_training_trial.py \
        resolve-dataset-role --dataset-manifest "$DATASET_MANIFEST" \
        --role test --field sequence_id 2>/dev/null)"; then
      TEST_SEQUENCE="$RESOLVED_TEST_SEQUENCE"
    fi
  fi
fi
HAS_FROZEN_TEST=0
TEST_CAM0_DIR=""
TEST_MASKS_DIR=""
if [[ -n "$DATA_TEST_DIR" || -n "$TEST_SEQUENCE" ]]; then
  if [[ -z "$DATA_TEST_DIR" || -z "$TEST_SEQUENCE" ]]; then
    echo "frozen test 需要同时解析 DATA_TEST_DIR 和 TEST_SEQUENCE" >&2
    exit 2
  fi
  TEST_NPZ_COUNT="$(find "$DATA_TEST_DIR" -maxdepth 1 -type f -name '*.npz' | wc -l)"
  if [[ "$TEST_NPZ_COUNT" -ne 1 ]]; then
    echo "正式 frozen test 目录必须恰有一个NPZ: $DATA_TEST_DIR count=$TEST_NPZ_COUNT" >&2
    exit 2
  fi
  HAS_FROZEN_TEST=1
  TEST_CAM0_DIR="$(resolve_real_path_for "$TEST_SEQUENCE" camera_dir)"
  TEST_MASKS_DIR="$(resolve_real_path_for "$TEST_SEQUENCE" masks_dir)"
fi
HAS_NDI=0
if [[ -f "$NDI_CSV" && -f "$FRAME_TIMES_FILE" ]]; then
  HAS_NDI=1
fi

if (( SAVE_INTERVAL <= 0 || PERIODIC_EVAL_INTERVAL <= 0 || PERIODIC_MAX_STEPS <= 0 )); then
  echo "SAVE_INTERVAL、验证间隔和验证最大步数必须为正数" >&2
  exit 2
fi
if [[ -n "$TEST_MAX_STEPS" ]] && (( TEST_MAX_STEPS <= 0 )); then
  echo "设置 TEST_MAX_STEPS 时必须为正数；默认空值评价完整 frozen test" >&2
  exit 2
fi
if [[ "$START_STAGE" != "gt" && "$START_STAGE" != "open_loop" ]]; then
  echo "START_STAGE必须是gt或open_loop" >&2
  exit 2
fi
if [[ "$PIPELINE_PREFLIGHT_ONLY" != "0" && "$PIPELINE_PREFLIGHT_ONLY" != "1" ]]; then
  echo "PIPELINE_PREFLIGHT_ONLY必须是0或1" >&2
  exit 2
fi
if [[ "$START_STAGE" == "gt" && -f "$DATASET_MANIFEST" ]]; then
  python scripts/real/manage_training_trial.py validate-dataset \
    --dataset-manifest "$DATASET_MANIFEST"
fi
if [[ "$PIPELINE_PREFLIGHT_ONLY" == "1" ]]; then
  printf 'dataset_id=%s\nsequence_tag=%s\ncapture_sequence=%s\n' \
    "$DATASET_ID" "$SEQ_TAG" "$CAPTURE_SEQ"
  printf 'train_dir=%s\nval_dir=%s\ndataset_manifest=%s\n' \
    "$DATA_TRAIN_DIR" "$DATA_VAL_DIR" "$DATASET_MANIFEST"
  printf 'frozen_test=%s\ntest_dir=%s\ntest_sequence=%s\n' \
    "$HAS_FROZEN_TEST" "$DATA_TEST_DIR" "$TEST_SEQUENCE"
  exit 0
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
  --tf-anneal-epochs "$TF_ANNEAL_EPOCHS"
  --tf-ratio "$TF_RATIO"
  --tf-min "$TF_MIN"
  --tf-schedule "$TF_SCHEDULE"
  --dense-step-weight "$DENSE_STEP_WEIGHT")
if [[ -n "$GT_LR" ]]; then
  CREATE_TRIAL_CMD+=(--gt-lr "$GT_LR")
fi
if [[ -n "$OPEN_LOOP_LR" ]]; then
  CREATE_TRIAL_CMD+=(--open-loop-lr "$OPEN_LOOP_LR")
fi
if [[ -n "$GT_SCHEDULER_PATIENCE" ]]; then
  CREATE_TRIAL_CMD+=(--gt-scheduler-patience "$GT_SCHEDULER_PATIENCE")
fi
if [[ -n "$OPEN_LOOP_SCHEDULER_PATIENCE" ]]; then
  CREATE_TRIAL_CMD+=(--open-loop-scheduler-patience
    "$OPEN_LOOP_SCHEDULER_PATIENCE")
fi
if [[ -f "$DATASET_MANIFEST" ]]; then
  CREATE_TRIAL_CMD+=(--dataset-manifest "$DATASET_MANIFEST")
fi
if (( HAS_FROZEN_TEST )); then
  CREATE_TRIAL_CMD+=(--test-dir "$DATA_TEST_DIR" --test-sequence "$TEST_SEQUENCE")
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
  VALIDATE_OPEN_LOOP_CMD=(python scripts/real/manage_training_trial.py
    validate-open-loop-start --trial-dir "$RUN_DIR"
    --train-dir "$DATA_TRAIN_DIR" --val-dir "$DATA_VAL_DIR"
    --open-loop-epochs "$OPEN_LOOP_EPOCHS" --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS" --save-interval "$SAVE_INTERVAL"
    --periodic-eval-interval "$PERIODIC_EVAL_INTERVAL"
    --periodic-max-steps "$PERIODIC_MAX_STEPS" --seed "$SEED"
    --window-size "$WINDOW_SIZE" --episode-len "$EPISODE_LEN"
    --tf-anneal-epochs "$TF_ANNEAL_EPOCHS" --tf-ratio "$TF_RATIO"
    --tf-min "$TF_MIN" --tf-schedule "$TF_SCHEDULE"
    --dense-step-weight "$DENSE_STEP_WEIGHT")
  if [[ -n "$OPEN_LOOP_LR" ]]; then
    VALIDATE_OPEN_LOOP_CMD+=(--open-loop-lr "$OPEN_LOOP_LR")
  fi
  if [[ -n "$OPEN_LOOP_SCHEDULER_PATIENCE" ]]; then
    VALIDATE_OPEN_LOOP_CMD+=(--open-loop-scheduler-patience
      "$OPEN_LOOP_SCHEDULER_PATIENCE")
  fi
  "${VALIDATE_OPEN_LOOP_CMD[@]}"
fi

GT_EXP_DIR="$RUN_DIR/stages/gt"
OPEN_LOOP_EXP_DIR="$RUN_DIR/stages/open_loop"
GT_BEST_QUANT_DIR="$RUN_DIR/evaluations/gt/best/quantitative"
GT_BEST_OVERLAY_DIR="$RUN_DIR/evaluations/gt/best/overlay"
OPEN_LOOP_BEST_QUANT_DIR="$RUN_DIR/evaluations/open_loop/best/quantitative"
OPEN_LOOP_BEST_OVERLAY_DIR="$RUN_DIR/evaluations/open_loop/best/overlay"
GT_TEST_QUANT_DIR="$RUN_DIR/evaluations/test/gt/quantitative"
GT_TEST_OVERLAY_DIR="$RUN_DIR/evaluations/test/gt/overlay"
OPEN_LOOP_TEST_QUANT_DIR="$RUN_DIR/evaluations/test/open_loop/quantitative"
OPEN_LOOP_TEST_OVERLAY_DIR="$RUN_DIR/evaluations/test/open_loop/overlay"
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
  if [[ $code -ne 0 ]]; then
    printf 'FAILED exit=%s at %s\n' "$code" "$(date --iso-8601=seconds)" >> "$STATUS_FILE"
  fi
}
trap on_exit EXIT

COMMON_ARGS=(
  --data_dir "$DATA_TRAIN_DIR"
  --val_dir "$DATA_VAL_DIR"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --window_size "$WINDOW_SIZE"
  --episode_len "$EPISODE_LEN"
  --eval_interval 0
  --save_interval "$SAVE_INTERVAL"
  --validation_interval "$PERIODIC_EVAL_INTERVAL"
  --validation_max_steps "$PERIODIC_MAX_STEPS"
  --seed "$SEED"
)

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
    --n_epochs "$GT_EPOCHS" --experiment-dir "$GT_EXP_DIR"
    --dense_step_weight "$DENSE_STEP_WEIGHT" "${COMMON_ARGS[@]}")
  if [[ -n "$GT_LR" ]]; then GT_CMD+=(--lr "$GT_LR"); fi
  if [[ -n "$GT_SCHEDULER_PATIENCE" ]]; then
    GT_CMD+=(--scheduler_patience "$GT_SCHEDULER_PATIENCE")
  fi
  record_command "$GPU_ID" "${GT_CMD[@]}"

  printf 'RUNNING gt started=%s train_gpu=%s eval_gpu=%s selection=engine_validation\n' \
    "$(date --iso-8601=seconds)" "$GPU_ID" "$EVAL_GPU_ID" > "$STATUS_FILE"
  CUDA_VISIBLE_DEVICES="$GPU_ID" MPLCONFIGDIR=/tmp/selfsoftrobot-mpl \
    PYTHONUNBUFFERED=1 "${GT_CMD[@]}" 2>&1 | tee "$GT_EXP_DIR/train.log"
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
  --init_from "$GT_CKPT" --tf_ratio "$TF_RATIO"
  --tf_anneal_epochs "$TF_ANNEAL_EPOCHS" --tf_min "$TF_MIN"
  --tf_schedule "$TF_SCHEDULE" --dense_step_weight "$DENSE_STEP_WEIGHT"
  "${COMMON_ARGS[@]}")
if [[ -n "$OPEN_LOOP_LR" ]]; then OPEN_LOOP_CMD+=(--lr "$OPEN_LOOP_LR"); fi
if [[ -n "$OPEN_LOOP_SCHEDULER_PATIENCE" ]]; then
  OPEN_LOOP_CMD+=(--scheduler_patience "$OPEN_LOOP_SCHEDULER_PATIENCE")
fi
record_command "$GPU_ID" "${OPEN_LOOP_CMD[@]}"

printf 'RUNNING open_loop started=%s initialization=%s\n' \
  "$(date --iso-8601=seconds)" "$GT_CKPT" >> "$STATUS_FILE"
CUDA_VISIBLE_DEVICES="$GPU_ID" MPLCONFIGDIR=/tmp/selfsoftrobot-mpl \
  PYTHONUNBUFFERED=1 "${OPEN_LOOP_CMD[@]}" 2>&1 | tee "$OPEN_LOOP_EXP_DIR/train.log"
OPEN_LOOP_CKPT="$OPEN_LOOP_EXP_DIR/phase_open_loop_transition/model/best_model.pt"
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

if (( HAS_FROZEN_TEST )); then
  printf 'RUNNING frozen_test started=%s dataset_role=test sequence=%s\n' \
    "$(date --iso-8601=seconds)" "$TEST_SEQUENCE" >> "$STATUS_FILE"

  GT_TEST_QUANT_CMD=(python scripts/evaluation/eval_real_quant.py
    --checkpoint "$GT_CKPT" --data_dir "$DATA_TEST_DIR" --mode gt
    --no-ndi --out "$GT_TEST_QUANT_DIR")
  if [[ -n "$TEST_MAX_STEPS" ]]; then
    GT_TEST_QUANT_CMD+=(--max-steps "$TEST_MAX_STEPS")
  fi
  record_command "$EVAL_GPU_ID" "${GT_TEST_QUANT_CMD[@]}"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU_ID" MPLCONFIGDIR=/tmp/selfsoftrobot-mpl \
    PYTHONUNBUFFERED=1 "${GT_TEST_QUANT_CMD[@]}" 2>&1 | tee "$GT_TEST_QUANT_DIR/run.log"

  GT_TEST_OVERLAY_CMD=(python scripts/evaluation/visualize_real_overlay.py
    --checkpoint "$GT_CKPT" --data_dir "$DATA_TEST_DIR" --mode gt
    --frame-offset 0
    --cam0 "$TEST_CAM0_DIR" --masks "$TEST_MASKS_DIR"
    --out "$GT_TEST_OVERLAY_DIR")
  if [[ -n "$TEST_MAX_STEPS" ]]; then
    GT_TEST_OVERLAY_CMD+=(--max-steps "$TEST_MAX_STEPS")
  fi
  record_command "$EVAL_GPU_ID" "${GT_TEST_OVERLAY_CMD[@]}"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU_ID" MPLCONFIGDIR=/tmp/selfsoftrobot-mpl \
    PYTHONUNBUFFERED=1 "${GT_TEST_OVERLAY_CMD[@]}" 2>&1 | tee "$GT_TEST_OVERLAY_DIR/run.log"

  OPEN_LOOP_TEST_QUANT_CMD=(python scripts/evaluation/eval_real_quant.py
    --checkpoint "$OPEN_LOOP_CKPT" --data_dir "$DATA_TEST_DIR"
    --mode open_loop --window-len "$EPISODE_LEN"
    --no-ndi --out "$OPEN_LOOP_TEST_QUANT_DIR")
  if [[ -n "$TEST_MAX_STEPS" ]]; then
    OPEN_LOOP_TEST_QUANT_CMD+=(--max-steps "$TEST_MAX_STEPS")
  fi
  record_command "$EVAL_GPU_ID" "${OPEN_LOOP_TEST_QUANT_CMD[@]}"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU_ID" MPLCONFIGDIR=/tmp/selfsoftrobot-mpl \
    PYTHONUNBUFFERED=1 "${OPEN_LOOP_TEST_QUANT_CMD[@]}" 2>&1 | tee "$OPEN_LOOP_TEST_QUANT_DIR/run.log"

  OPEN_LOOP_TEST_OVERLAY_CMD=(python scripts/evaluation/visualize_real_overlay.py
    --checkpoint "$OPEN_LOOP_CKPT" --data_dir "$DATA_TEST_DIR"
    --mode open_loop --window-len "$EPISODE_LEN" --with-onestep
    --frame-offset 0
    --cam0 "$TEST_CAM0_DIR" --masks "$TEST_MASKS_DIR"
    --out "$OPEN_LOOP_TEST_OVERLAY_DIR")
  if [[ -n "$TEST_MAX_STEPS" ]]; then
    OPEN_LOOP_TEST_OVERLAY_CMD+=(--max-steps "$TEST_MAX_STEPS")
  fi
  record_command "$EVAL_GPU_ID" "${OPEN_LOOP_TEST_OVERLAY_CMD[@]}"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU_ID" MPLCONFIGDIR=/tmp/selfsoftrobot-mpl \
    PYTHONUNBUFFERED=1 "${OPEN_LOOP_TEST_OVERLAY_CMD[@]}" 2>&1 | tee "$OPEN_LOOP_TEST_OVERLAY_DIR/run.log"
fi

FINALIZE_CMD=(python scripts/real/manage_training_trial.py finalize --trial-dir "$RUN_DIR")
record_command "$EVAL_GPU_ID" "${FINALIZE_CMD[@]}"
"${FINALIZE_CMD[@]}"
printf 'COMPLETE finished=%s\n' "$(date --iso-8601=seconds)" >> "$STATUS_FILE"
trap - EXIT
echo "全部完成：$RUN_DIR"
