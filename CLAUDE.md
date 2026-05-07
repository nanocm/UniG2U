# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UniG2U is a multimodal LLM evaluation harness, forked from [lmms-eval v0.5](https://github.com/EvolvingLMMs-Lab/lmms-eval). It extends upstream with 11 custom benchmark task families (30 fine-grained subtasks), Visual CoT (chain-of-thought) model wrappers, and one-shot evaluation scripts for the UniG2U benchmark suite.

## Common Commands

```bash
# Install (editable, all extras)
uv pip install -e ".[all]"

# Run single evaluation
uv run python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args "pretrained=Qwen/Qwen2.5-VL-3B-Instruct,device_map=auto" \
  --tasks mmmu,mme \
  --batch_size 1 --log_samples --output_path ./logs/qwen

# Run full UniG2U benchmark suite
bash script/eval_all.sh --model qwen2_5_vl --model_args "pretrained=Qwen/Qwen2.5-VL-3B-Instruct"

# Run CoT/Visual CoT variant
bash script/eval_all_cot.sh --model bagel_visual_cot --model_args "pretrained=ByteDance-Seed/BAGEL-7B-MoT,save_intermediate=true"

# Lint and format
uv run ruff format .
uv run ruff check . --fix

# Type check
uv run pyright
```

## Package Management

- **ONLY use uv, NEVER pip**
- Add deps: `uv add <package>` (updates pyproject.toml + uv.lock)
- Remove deps: `uv remove <package>`
- Run tools: `uv run <tool>`
- FORBIDDEN: `uv pip install`, `@latest` syntax

## Architecture

### Evaluation Pipeline

```
CLI (__main__.py:cli_evaluate)
  → evaluator.simple_evaluate()
    → TaskManager: auto-discovers YAML configs from lmms_eval/tasks/
    → get_model(): dynamically imports model class from lmms_eval/models/
    → Builds Instance objects (requests per document)
    → Dispatches to model.generate_until() or model.loglikelihood()
    → Scores results via per-task process_results functions
    → Writes results*.json to --output_path
  → script/aggregate_results.py: produces summary.json + benchmark_summary.json
```

### Two Model Paradigms

- **Simple** (`lmms_eval/models/simple/`): `generate_until(requests)` receives `(contexts, gen_kwargs, doc_to_visual, doc_id, task, split)` per instance
- **Chat** (`lmms_eval/models/chat/`): `generate_until(requests)` receives `(doc_to_messages, gen_kwargs, doc_id, task, split)` with interleaved multimodal messages

Models are registered in `lmms_eval/models/__init__.py` via `AVAILABLE_SIMPLE_MODELS` and `AVAILABLE_CHAT_TEMPLATE_MODELS` dicts. `get_model(name)` resolves to a dotted class path and imports dynamically.

### Visual CoT Pipeline (Two-Stage Inference)

1. **Stage 1**: Given original image + generation prompt → produce an auxiliary image (annotated diagram, highlighted chart, geometry construction lines, etc.)
2. **Stage 2**: Given [original image, auxiliary image] + question → produce final answer

Prompt format uses `[GEN_PROMPT]...[/GEN_PROMPT][QUESTION]...[/QUESTION]` tags. See `lmms_eval/tasks/VISUAL_COT_PROMPTS.md` for all prompt designs.

### Task Definitions

Tasks are YAML configs under `lmms_eval/tasks/<task_name>/`. Key fields:
- `dataset_path` / `dataset_kwargs`: HuggingFace dataset (UniG2U tasks use parquet files from `hf://datasets/kkv233/unig2u_dataset/`)
- `doc_to_visual`, `doc_to_text`, `doc_to_target`: functions extracting images, prompts, ground truth
- `process_results`: per-task scoring function
- `lmms_eval_specific_kwargs`: per-model prompt overrides (keys like `default`, `qwen_vl`, `bagel_visual_cot`)

### LLM Judge (for Scoring)

Some tasks (auxsolidmath_easy, geometry3k, babyvision, phyx_simple) use an LLM judge. Backend auto-selected:
- `OPENAI_API_KEY` set → OpenAI (model: `OPENAI_JUDGE_MODEL`, default `gpt-4o`)
- `AZURE_OPENAI_COMPAT_BASE_URL` / `OPENAI_BASE_URL` set → Azure endpoint
- Neither → Azure TRAPI via `AzureCliCredential`

## UniG2U Benchmark Tasks (11 Families, 30 Subtasks)

| Task | Subtasks | Category |
|------|----------|----------|
| auxsolidmath_easy | 1 | Geometry Reasoning |
| chartqa100 | 1 | Chart & Table Reasoning |
| geometry3k | 1 | Geometry Reasoning |
| babyvision | fine_grained, visual_tracking | Perception Reasoning / Puzzles |
| illusionbench_arshia_test | 6 (icon/logo/in × shape/scene) | Perception Reasoning |
| mmsi | 5 (attribute_appr/meas, motion_cam/obj, msr) | Spatial Intelligence |
| phyx_simple | mechanics100, optics100 | Physics Reasoning |
| realunify | attentional_focusing, mental_reconstruction, mental_tracking | Real-world Apps / Puzzles |
| uni_mmmu | jigsaw100, maze100, sliding54 | Puzzles and Games |
| vsp | google_map, collision | Real-world Applications |
| VisualPuzzles | algorithmic, analogical, deductive, inductive, spatial | Perception Reasoning |

Each task has a `_visual_cot` or `_cot` variant for the CoT pipeline.

## Key Directories

- `lmms_eval/tasks/`: benchmark YAML configs, prompt and metric utilities
- `lmms_eval/models/simple/`: simple-paradigm model wrappers (including Visual CoT)
- `lmms_eval/models/chat/`: chat-paradigm model wrappers
- `lmms_eval/models/__init__.py`: model registry
- `lmms_eval/evaluator.py`: core evaluation loop
- `lmms_eval/api/task.py`: TaskConfig and task base classes
- `script/eval_all.sh` / `eval_all_cot.sh`: one-shot benchmark runners
- `script/aggregate_results.py`: post-processing and benchmark aggregation

## Code Style

- Line length: black uses 240, ruff uses 88 — follow ruff for new code
- Pre-commit: `black --line-length=240` + `isort --profile black`
- PEP 8 naming: `snake_case` functions/variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
- Python >=3.9, pinned `transformers==4.57.0`

## Commit Conventions

- NEVER mention co-authored-by or the tool used
- For bug/feature reports: `git commit --trailer "Reported-by:<name>"`
- For GitHub issues: `git commit --trailer "Github-Issue:#<number>"`

## Environment Variables

```bash
export OPENAI_API_KEY="..."       # For LLM judge (OpenAI backend)
export OPENAI_JUDGE_MODEL="gpt-4o"  # Optional judge model override
export HF_HOME="<path>"          # HuggingFace cache directory
export HF_TOKEN="..."            # HuggingFace access token
```

## External Model Dependencies

Some models require external repos to be cloned separately:
- `uniworld` / `uniworld_visual_cot`: expects `UniWorld/UniWorld-V1/`
- `mio` / `emu3`: expects `MIO/` repo
