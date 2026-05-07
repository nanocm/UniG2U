"""Convert raw GeoG2U data to parquet files for lmms_eval.

Reads from:
  - raw_data/exports/*/samples.jsonl   (16 tasks, single-image MCQ)
  - raw_data/mannual/VQA(E)_subtasks/*.json  (4 tasks, dual-image MCQ)

Writes to:
  - lmms_eval/tasks/rs_bench/data/<task_name>.parquet

Each parquet row contains:
  id, source_image_name, question, options, answer, task_name, category

Images are NOT embedded — doc_to_visual loads them at runtime via GEOG2U_IMAGE_ROOT.

Usage (local):
  python lmms_eval/tasks/rs_bench/convert_raw_to_parquet.py --raw-root raw_data

Usage (server):
  python lmms_eval/tasks/rs_bench/convert_raw_to_parquet.py --raw-root /path/to/raw_data
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# ── Category assignments ──────────────────────────────────────────────────────
TASK_CATEGORY = {
    "01_object_classification": "Object Recognition",
    "02_object_color": "Object Recognition",
    "03_counting": "Counting & Spatial",
    "04_scene_classification": "Scene Understanding",
    "05_spatial_relationship": "Counting & Spatial",
    "06_object_state": "Object Recognition",
    "07_environmental_reasoning": "Scene Understanding",
    "08_route_planning": "Spatial Analysis",
    "09_anomaly_detection": "Anomaly & Counterfactual",
    "10_object_background": "Object Recognition",
    "11_regional_existence": "Scene Understanding",
    "12_boundary_extraction": "Spatial Analysis",
    "13_cross-tile_adjacency": "Spatial Analysis",
    "14_buffer_analysis": "Spatial Analysis",
    "15_urban_expansion_cd": "Change Detection",
    "16_forest_cover_cd": "Change Detection",
    "17_water_body_cd": "Change Detection",
    "18_farmland_cd": "Change Detection",
    "19_counterfactual_editing": "Anomaly & Counterfactual",
    "20_landuse_plan_judgment": "Anomaly & Counterfactual",
}

# Friendly task names for the benchmark
TASK_NAMES = {
    "01_object_classification": "rs_object_classification",
    "02_object_color": "rs_object_color",
    "03_counting": "rs_counting",
    "04_scene_classification": "rs_scene_classification",
    "05_spatial_relationship": "rs_spatial_relationship",
    "06_object_state": "rs_object_state",
    "07_environmental_reasoning": "rs_environmental_reasoning",
    "08_route_planning": "rs_route_planning",
    "09_anomaly_detection": "rs_anomaly_detection",
    "10_object_background": "rs_object_background",
    "11_regional_existence": "rs_regional_existence",
    "12_boundary_extraction": "rs_boundary_extraction",
    "13_cross-tile_adjacency": "rs_cross_tile_adjacency",
    "14_buffer_analysis": "rs_buffer_analysis",
    "15_urban_expansion_cd": "rs_urban_expansion_cd",
    "16_forest_cover_cd": "rs_forest_cover_cd",
    "17_water_body_cd": "rs_water_body_cd",
    "18_farmland_cd": "rs_farmland_cd",
    "19_counterfactual_editing": "rs_counterfactual_editing",
    "20_landuse_plan_judgment": "rs_landuse_plan_judgment",
}

# mannual json → task key mapping
MANNUAL_FILES = {
    "15_Urban_expansion_change_detection.json": "15_urban_expansion_cd",
    "16_Forest_cover_change_detection.json": "16_forest_cover_cd",
    "17_Water_body_change_detection.json": "17_water_body_cd",
    "18_Farmland_change_detection.json": "18_farmland_cd",
}


def convert_exports_task(samples_path: Path, task_key: str) -> list[dict]:
    """Convert a single exports task from samples.jsonl."""
    records = []
    with samples_path.open() as f:
        for line in f:
            d = json.loads(line)
            qa = d["sample"]["qa"]
            meta = d["sample"]["source_spatial_meta"]
            source_image_name = meta.get("source_image_name")

            # task 13 has source_image_name=None, skip those
            if source_image_name is None:
                continue

            records.append(
                {
                    "id": d["sample_id"],
                    "source_image_name": source_image_name,
                    "question": qa["question"],
                    "options": qa["options"],
                    "answer": qa["answer"],
                    "task_name": TASK_NAMES[task_key],
                    "category": TASK_CATEGORY[task_key],
                }
            )
    return records


def convert_mannual_task(json_path: Path, task_key: str) -> list[dict]:
    """Convert a mannual change detection task from JSON."""
    with json_path.open() as f:
        data = json.load(f)

    records = []
    for i, d in enumerate(data):
        # mannual format: "answer" is the options list, "Ground truth" is the correct letter
        records.append(
            {
                "id": f"{task_key}_{i:04d}",
                "image1": d["Image1"],  # relative path like "Google_Earth/1-changing1.jpg"
                "image2": d["Image2"],
                "question": d["question"],
                "options": d["answer"],  # this is the options list
                "answer": d["Ground truth"],  # this is the correct letter
                "task_name": TASK_NAMES[task_key],
                "category": TASK_CATEGORY[task_key],
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description="Convert raw GeoG2U data to parquet")
    parser.add_argument("--raw-root", type=str, default="raw_data", help="Path to raw_data directory")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = 0

    # ── exports tasks ─────────────────────────────────────────────────────────
    exports_dir = raw_root / "exports"
    for task_dir in sorted(exports_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        samples_file = task_dir / "samples.jsonl"
        if not samples_file.exists():
            continue

        task_key = task_dir.name
        if task_key not in TASK_NAMES:
            print(f"  SKIP: {task_key} (not in task list)")
            continue

        records = convert_exports_task(samples_file, task_key)
        if not records:
            print(f"  SKIP: {task_key} (no valid samples)")
            continue

        task_name = TASK_NAMES[task_key]
        df = pd.DataFrame(records)
        out_path = output_dir / f"{task_name}.parquet"
        df.to_parquet(out_path, index=False)
        total_samples += len(df)
        print(f"  {task_name}: {len(df)} samples -> {out_path.name}")

    # ── mannual tasks ─────────────────────────────────────────────────────────
    mannual_dir = raw_root / "mannual" / "VQA(E)_subtasks"
    if mannual_dir.exists():
        for json_name, task_key in MANNUAL_FILES.items():
            json_path = mannual_dir / json_name
            if not json_path.exists():
                print(f"  SKIP: {json_name} (not found)")
                continue

            records = convert_mannual_task(json_path, task_key)
            task_name = TASK_NAMES[task_key]
            df = pd.DataFrame(records)
            out_path = output_dir / f"{task_name}.parquet"
            df.to_parquet(out_path, index=False)
            total_samples += len(df)
            print(f"  {task_name}: {len(df)} samples -> {out_path.name}")

    print(f"\nTotal: {total_samples} samples across {len(TASK_NAMES)} tasks")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
