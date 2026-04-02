#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/aiscuser/UniG2U"
MODE="${1:-smoke}"
MODEL_REPO_ID="${MODEL_REPO_ID:-meituan-longcat/LongCat-Next}"
MODEL_PATH="${MODEL_PATH:-/home/aiscuser/.cache/huggingface/hub/models--meituan-longcat--LongCat-Next/snapshots/522f2020e5ed353429cc403b72491ba1899ef0e6}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0,1,2,3,4,5,6,7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/logs}"
SMOKE_TASK="${SMOKE_TASK:-chartqa100}"
SMOKE_LIMIT="${SMOKE_LIMIT:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LONGCAT_DTYPE="${LONGCAT_DTYPE:-bfloat16}"
MAX_MEMORY="${MAX_MEMORY:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"

resolve_model_source() {
  if [[ -d "$MODEL_PATH" ]]; then
    if find "$MODEL_PATH" -maxdepth 1 \( -name '*.safetensors' -o -name '*.bin' -o -name '*.index.json' \) | grep -q .; then
      echo "$MODEL_PATH"
      return
    fi
    echo "[warn] Local MODEL_PATH exists but does not contain weights: $MODEL_PATH" >&2
    echo "[warn] Falling back to repo id: $MODEL_REPO_ID" >&2
    echo "$MODEL_REPO_ID"
    return
  fi

  if [[ -n "$MODEL_PATH" && "$MODEL_PATH" != "$MODEL_REPO_ID" ]]; then
    echo "[warn] MODEL_PATH not found: $MODEL_PATH" >&2
    echo "[warn] Falling back to repo id: $MODEL_REPO_ID" >&2
  fi
  echo "$MODEL_REPO_ID"
}

MODEL_SOURCE="$(resolve_model_source)"

if [[ ! -d "$ROOT" ]]; then
  echo "Missing UniG2U repo: $ROOT"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29314}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

COMMON_ARGS="pretrained=${MODEL_SOURCE},device_map=auto,dtype=${LONGCAT_DTYPE}"
if [[ -n "$MAX_MEMORY" ]]; then
  COMMON_ARGS="${COMMON_ARGS},max_memory=${MAX_MEMORY}"
fi
if [[ -n "$MAX_NEW_TOKENS" ]]; then
  COMMON_ARGS="${COMMON_ARGS},max_new_tokens=${MAX_NEW_TOKENS}"
fi
STANDARD_ARGS="${COMMON_ARGS},mode=understanding"
COT_ARGS="${COMMON_ARGS},save_intermediate=true"

cd "$ROOT"

case "$MODE" in
  smoke)
    python -m lmms_eval \
      --model longcat_next \
      --model_args "$STANDARD_ARGS" \
      --tasks "$SMOKE_TASK" \
      --batch_size "$BATCH_SIZE" \
      --limit "$SMOKE_LIMIT" \
      --log_samples \
      --output_path "$OUTPUT_ROOT/longcat_next_smoke/${SMOKE_TASK}"
    ;;
  full)
    echo "Running standard UniG2U suite for LongCat-Next"
    echo "Note: some tasks require judge API env such as OPENAI_API_KEY or az login."
    export BATCH_SIZE
    bash script/eval_all.sh \
      --model longcat_next \
      --model_args "$STANDARD_ARGS"
    ;;
  cot)
    echo "Running Visual CoT smoke suite for LongCat-Next"
    echo "Current script/eval_all_cot.sh keeps --limit 1 for each task."
    export BATCH_SIZE
    bash script/eval_all_cot.sh \
      --model longcat_next_visual_cot \
      --model_args "$COT_ARGS"
    ;;
  *)
    echo "Usage: bash /home/aiscuser/UniG2U/examples/test_longcat_next.sh [smoke|full|cot]"
    exit 1
    ;;
esac
