#!/bin/bash
# Run the full RS Bench (GeoG2U) standard evaluation suite.
#
# Usage:
#   bash script/eval_rs_bench.sh --model qwen2_5_vl --model_args "pretrained=Qwen/Qwen2.5-VL-3B-Instruct,device_map=auto"
#   bash script/eval_rs_bench.sh --model qwen2_5_vl --model_args "..." --num_gpus 4
#
# Required env vars:
#   GEOG2U_IMAGE_ROOT  - path to satellite images (contains AF/, Asia/, etc.)
# Optional:
#   GEOG2U_MANNUAL_ROOT - path to mannual change detection images (default: raw_data/mannual)
#   BATCH_SIZE          - batch size (default: 1)
set -e

MODEL=""
MODEL_ARGS=""
NUM_GPUS=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2";      shift 2 ;;
        --model_args) MODEL_ARGS="$2"; shift 2 ;;
        --num_gpus)   NUM_GPUS="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Usage: bash script/eval_rs_bench.sh --model <model> --model_args <args> [--num_gpus N]"
    exit 1
fi

if [[ -z "$GEOG2U_IMAGE_ROOT" ]]; then
    echo "ERROR: GEOG2U_IMAGE_ROOT not set. Export it before running."
    echo "  export GEOG2U_IMAGE_ROOT=/path/to/GeoG2U"
    exit 1
fi

OUTPUT_BASE="./logs/${MODEL}_rs"
mkdir -p "$OUTPUT_BASE"
BATCH_SIZE="${BATCH_SIZE:-1}"

# Build launch command based on GPU count
if [[ "$NUM_GPUS" -gt 1 ]]; then
    LAUNCH_CMD="accelerate launch --num_processes $NUM_GPUS -m lmms_eval"
else
    export WORLD_SIZE=1
    export RANK=0
    export LOCAL_RANK=0
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=29314
    LAUNCH_CMD="python -m lmms_eval"
fi

# Single-image tasks (16)
TASKS=(
    rs_object_classification
    rs_object_color
    rs_object_state
    rs_object_background
    rs_scene_classification
    rs_environmental_reasoning
    rs_regional_existence
    rs_counting
    rs_spatial_relationship
    rs_route_planning
    rs_boundary_extraction
    rs_cross_tile_adjacency
    rs_buffer_analysis
    rs_anomaly_detection
    rs_counterfactual_editing
    rs_landuse_plan_judgment
)

# Change detection tasks (4, dual-image)
CD_TASKS=(
    rs_urban_expansion_cd
    rs_forest_cover_cd
    rs_water_body_cd
    rs_farmland_cd
)

for TASK in "${TASKS[@]}" "${CD_TASKS[@]}"; do
    echo "========================================"
    echo "Running: $TASK (${NUM_GPUS} GPU(s))"
    echo "========================================"
    $LAUNCH_CMD \
        --model "$MODEL" \
        --model_args "$MODEL_ARGS" \
        --tasks "$TASK" \
        --batch_size "$BATCH_SIZE" \
        --log_samples \
        --output_path "${OUTPUT_BASE}/${TASK}"

    echo "Done: $TASK"
    echo
done

echo "All RS Bench tasks completed. Results in ${OUTPUT_BASE}/"

echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
python script/aggregate_results.py --output-base "$OUTPUT_BASE" --mode rs_standard
