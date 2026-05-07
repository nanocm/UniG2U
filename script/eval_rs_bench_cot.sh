#!/bin/bash
# Run the full RS Bench Visual CoT evaluation suite sequentially.
# Usage: bash script/eval_rs_bench_cot.sh --model bagel_visual_cot --model_args "pretrained=ByteDance-Seed/BAGEL-7B-MoT,save_intermediate=true"
set -e

MODEL=""
MODEL_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2";      shift 2 ;;
        --model_args) MODEL_ARGS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Usage: bash script/eval_rs_bench_cot.sh --model <model> --model_args <args>"
    exit 1
fi

OUTPUT_BASE="./logs/${MODEL}_rs"
mkdir -p "$OUTPUT_BASE"
BATCH_SIZE="${BATCH_SIZE:-1}"

export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29314

TASKS=(
    rs_scene_classification_visual_cot
    rs_object_counting_visual_cot
    rs_change_detection_visual_cot
    rs_spatial_reasoning_visual_cot
    rs_map_reading_visual_cot
)

for TASK in "${TASKS[@]}"; do
    echo "========================================"
    echo "Running: $TASK"
    echo "========================================"
    python -m lmms_eval \
        --model "$MODEL" \
        --model_args "$MODEL_ARGS" \
        --tasks "$TASK" \
        --batch_size "$BATCH_SIZE" \
        --log_samples \
        --output_path "${OUTPUT_BASE}/${TASK}"

    echo "Done: $TASK"
    echo
done

echo "All RS Bench CoT tasks completed. Results in ${OUTPUT_BASE}/"

# Aggregate all results.
echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
python script/aggregate_results.py --output-base "$OUTPUT_BASE" --mode rs_cot
