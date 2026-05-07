"""Utility functions for RS Bench (GeoG2U) tasks.

Images are loaded from disk at runtime via the GEOG2U_IMAGE_ROOT environment
variable. Parquet files only store the image file name (not the bytes).

Set GEOG2U_IMAGE_ROOT before running evaluation:
  export GEOG2U_IMAGE_ROOT=/path/to/GeoG2U   # contains AF/, Asia/, Europe/, ...

For change detection tasks (15-18), mannual images are loaded from
GEOG2U_MANNUAL_ROOT (defaults to raw_data/mannual).
"""

import logging
import os
import random
import re

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # RS images can exceed 300M pixels

eval_logger = logging.getLogger("lmms-eval")

IMAGE_ROOT = os.environ.get("GEOG2U_IMAGE_ROOT", "")
MANNUAL_ROOT = os.environ.get("GEOG2U_MANNUAL_ROOT", "raw_data/mannual")

MCQ_DIRECT_PROMPT = "Answer with the option's letter from the given choices directly."


# ---------------------------------------------------------------------------
# doc_to_visual
# ---------------------------------------------------------------------------


def rs_doc_to_visual(doc):
    """Load the source satellite image from disk. Returns [PIL.Image].

    Image path is resolved as: {IMAGE_ROOT}/{continent}/{source_image_name}
    where continent is the prefix before the first underscore.
    """
    name = doc["source_image_name"]
    continent = name.split("_")[0]
    path = os.path.join(IMAGE_ROOT, continent, name)
    return [Image.open(path).convert("RGB")]


def rs_doc_to_visual_change(doc):
    """Load two images for change detection tasks. Returns [PIL.Image, PIL.Image].

    Image paths are relative to GEOG2U_MANNUAL_ROOT:
      e.g. "Google_Earth/1-changing1.jpg" -> {MANNUAL_ROOT}/Google_Earth/1-changing1.jpg
    """
    path1 = os.path.join(MANNUAL_ROOT, doc["image1"])
    path2 = os.path.join(MANNUAL_ROOT, doc["image2"])
    return [Image.open(path1).convert("RGB"), Image.open(path2).convert("RGB")]


# ---------------------------------------------------------------------------
# doc_to_text
# ---------------------------------------------------------------------------


def _format_options(options) -> str:
    """Format MCQ options. Handles both 'A. xxx' and plain 'xxx' formats."""
    labels = "ABCDEFGHIJ"
    lines = []
    for i, opt in enumerate(options):
        opt_str = str(opt).strip()
        # If option already starts with a letter prefix like "A. " or "A ", skip adding one
        if len(opt_str) >= 2 and opt_str[0] in labels and opt_str[1] in ".) ":
            lines.append(f"({opt_str[0]}) {opt_str[2:].strip()}")
        else:
            lines.append(f"({labels[i]}) {opt_str}")
    return "\n".join(lines)


def rs_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Build MCQ prompt for standard (direct) evaluation."""
    question = doc["question"].strip()
    options = doc.get("options")

    if options is not None and len(options) > 0:
        question = f"Question: {question}\nOptions:\n{_format_options(options)}"

    if lmms_eval_specific_kwargs:
        pre = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post = lmms_eval_specific_kwargs.get("post_prompt", "")
        question = f"{pre}{question}{post}"

    return question


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


def parse_mcq_response(response: str, all_choices: list[str] | None = None) -> str:
    """Parse MCQ answer letter from model response."""
    if all_choices is None:
        all_choices = ["A", "B", "C", "D"]

    response = response.strip()

    # Pattern 1: "Answer: (A)" or "Answer: A"
    for pattern in [r"Answer:\s*\(([A-Da-d])\)", r"Answer:\s*([A-Da-d])\b"]:
        matches = re.findall(pattern, response)
        if matches:
            for m in reversed(matches):
                if m.upper() in all_choices:
                    return m.upper()

    # Pattern 2: Standalone "(A)" style
    matches = re.findall(r"\(([A-Da-d])\)", response)
    if matches:
        for m in reversed(matches):
            if m.upper() in all_choices:
                return m.upper()

    # Pattern 3: Single letter response
    cleaned = response.strip().upper()
    if len(cleaned) <= 3 and cleaned in all_choices:
        return cleaned

    # Pattern 4: Letter followed by period or parenthesis
    for pattern in [r"\b([A-Da-d])\.", r"\b([A-Da-d])\)"]:
        matches = re.findall(pattern, response)
        if matches:
            for m in reversed(matches):
                if m.upper() in all_choices:
                    return m.upper()

    return random.choice(all_choices)


# ---------------------------------------------------------------------------
# process_results & aggregation
# ---------------------------------------------------------------------------


def rs_process_results(doc, results):
    """Score MCQ response. Returns dict with accuracy metric."""
    pred = parse_mcq_response(results[0])
    target = doc["answer"].strip().upper()
    score = 1.0 if pred == target else 0.0
    return {"accuracy": score}


def rs_aggregate_results(results):
    """Compute mean accuracy over all samples."""
    if not results:
        return 0.0
    return sum(results) / len(results)
