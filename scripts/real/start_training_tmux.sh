#!/usr/bin/env bash
# 在可重连的 tmux 交互会话中启动真实数据训练流水线。
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "用法: $0 SESSION_NAME DATA_TRAIN_DIR [DATA_VAL_DIR]" >&2
  exit 2
fi

SESSION_NAME="$1"
shift
if [[ ! "$SESSION_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "SESSION_NAME仅支持字母、数字、点、下划线和连字符" >&2
  exit 2
fi
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux会话已存在: $SESSION_NAME" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAMES=(
  GPU_ID EVAL_GPU_ID GT_EPOCHS OPEN_LOOP_EPOCHS BATCH_SIZE NUM_WORKERS
  SAVE_INTERVAL PERIODIC_EVAL_INTERVAL PERIODIC_MAX_STEPS SEED WINDOW_SIZE
  EPISODE_LEN TF_ANNEAL_EPOCHS TF_RATIO TF_MIN TF_SCHEDULE DENSE_STEP_WEIGHT
  GT_LR OPEN_LOOP_LR GT_SCHEDULER_PATIENCE OPEN_LOOP_SCHEDULER_PATIENCE
  START_STAGE RUN_DIR SEQUENCE_TAG CAPTURE_SEQ CAM0_DIR MASKS_DIR DATASET_MANIFEST
)
COMMAND=(env)
for name in "${ENV_NAMES[@]}"; do
  if [[ -v "$name" ]]; then
    COMMAND+=("$name=${!name}")
  fi
done
COMMAND+=(bash scripts/real/train_real_transition.sh "$@")

printf -v ROOT_QUOTED '%q' "$PROJECT_ROOT"
printf -v COMMAND_QUOTED '%q ' "${COMMAND[@]}"
PANE_COMMAND="cd ${ROOT_QUOTED} && ${COMMAND_QUOTED}; training_exit=\$?; printf '\nTRAINING_EXIT=%s finished=%s\n' \"\$training_exit\" \"\$(date --iso-8601=seconds)\""

tmux new-session -d -s "$SESSION_NAME" -n pipeline
tmux set-option -t "$SESSION_NAME" remain-on-exit on
tmux send-keys -l -t "$SESSION_NAME:pipeline.0" "$PANE_COMMAND"
tmux send-keys -t "$SESSION_NAME:pipeline.0" C-m

echo "训练已启动: $SESSION_NAME"
echo "进入会话: tmux attach -t $SESSION_NAME"
echo "查看末尾: tmux capture-pane -pt $SESSION_NAME:pipeline.0 -S -100"
