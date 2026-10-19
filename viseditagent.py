"""Run VisEditAgent: plan, generate candidates, render, judge, and refine once."""

import argparse
import base64
import json
import mimetypes
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm


# CONFIG

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "image"
OUTPUT_CSV = ROOT / "viseditagent_predictions_eval.csv"
RENDER_DIR = ROOT / "renders_viseditagent"

AGENT_MODEL = "gpt-5-mini"
JUDGE_MODEL = "gpt-4o"

TOP_K_CANDIDATES = 5
REFINE_ONCE = True

# Keep temperature as None for GPT-5/GPT-5-mini compatibility.
# Diversity comes from strategy-specific prompts.
GENERATION_TEMPERATURE = None
GENERATION_TOP_P = None
PLANNING_TEMPERATURE = None
REFINEMENT_TEMPERATURE = None

MAX_OUTPUT_TOKENS_CODE = 4096
MAX_OUTPUT_TOKENS_PLAN = 1200
MAX_OUTPUT_TOKENS_JUDGE = 1200

EXECUTION_TIMEOUT = 10
OPENAI_TIMEOUT = 90.0
JUDGE_TIMEOUT = 60.0
OPENAI_MAX_RETRIES = 0
JUDGE_MAX_RETRIES = 1


SAVE_EVERY = 5
LIMIT = None
RESUME = True
SKIP_POSITIONS = []
SLEEP_AFTER_ERROR = 1
STORE_CANDIDATE_CODES = False


REPAIR_INTENTS = {"Correctness Repair", "Quality Improvement"}
STYLE_ADAPTATION_INTENTS = {"Style Adaptation"}
STYLE_AWARE_ERROR_REPAIR_INTENTS = {"Style-Aware Error Repair"}
ALLOWED_SCORES = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]


def make_openai_client(timeout=OPENAI_TIMEOUT):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key, timeout=timeout, max_retries=OPENAI_MAX_RETRIES)


def is_valid_text(x):
    if pd.isna(x):
        return False
    x = str(x).strip()
    return x != "" and x.lower() not in {"nan", "none", "null"}


def safe_name(x):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(x))


def clean_code(text):
    if text is None:
        return ""
    text = str(text).strip()
    match = re.search(
        r"```(?:python|py|javascript|js|json|vega|vega-lite)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        text = match.group(1).strip()
    if len(text) >= 2 and ((text[0] == text[-1] == "'") or (text[0] == text[-1] == '"')):
        try:
            import ast
            value = ast.literal_eval(text)
            if isinstance(value, str):
                text = value
        except Exception:
            text = text[1:-1]
    if "\\n" in text and text.count("\n") < 2:
        try:
            text = text.encode("utf-8").decode("unicode_escape")
        except Exception:
            text = text.replace("\\n", "\n")
    text = re.sub(r"^\s*%matplotlib\s+\w+\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def truncate_text(text, max_chars=7000):
    text = "" if text is None else str(text)
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n# ... TRUNCATED ...\n\n" + text[-half:]


def round_to_half(x):
    try:
        x = float(x)
    except Exception:
        return 0.0
    x = max(0.0, min(5.0, x))
    return round(x * 2) / 2


def clamp_similarity(x):
    try:
        x = int(round(float(x)))
    except Exception:
        return 0
    return max(0, min(100, x))


def find_image_path(image_dir, img_id):
    image_dir = Path(image_dir)
    img_id = str(img_id).strip()
    if Path(img_id).exists():
        return str(Path(img_id))
    direct_path = image_dir / img_id
    if direct_path.exists():
        return str(direct_path)
    stem = Path(img_id).stem
    stems = [stem]
    match = re.fullmatch(r"(.*?)(\d+)", stem)
    if match:
        stems.append(f"{match[1]}{int(match[2]):04d}")
    for candidate in stems:
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            image_path = image_dir / f"{candidate}{ext}"
            if image_path.is_file():
                return str(image_path)
    return ""


def image_to_data_url(image_path):
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/png"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def call_vlm_with_images(prompt, image_paths, model, instructions, max_output_tokens,
                         temperature=None, top_p=None, timeout=OPENAI_TIMEOUT):
    client = make_openai_client(timeout=timeout)
    content = [{"type": "input_text", "text": prompt}]
    for image_path in image_paths:
        if image_path and Path(image_path).exists():
            content.append({"type": "input_image", "image_url": image_to_data_url(image_path), "detail": "high"})
    kwargs = {
        "model": model,
        "max_output_tokens": max_output_tokens,
        "instructions": instructions,
        "input": [{"role": "user", "content": content}],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    response = client.responses.create(**kwargs)
    if response.status != "completed" or not response.output_text.strip():
        raise ValueError(f"No complete response (status={response.status})")
    return response.output_text


def row_value(row, col, default=""):
    try:
        value = row.get(col, default)
    except Exception:
        value = default
    return "" if pd.isna(value) else str(value)


def row_context(row):
    optional_parts = []
    for col in ["Question", "Table Data", "Source", "problem_type", "reference_chart_type", "overall_complexity"]:
        value = row_value(row, col, "")
        if is_valid_text(value):
            optional_parts.append(f"{col}: {value}")
    return "\n".join(optional_parts)


def intent_mode_text(intent):
    if intent == "Correctness Repair":
        return "Fix the visual or semantic error while preserving the intended data meaning. Do not reproduce handwritten marks or the original bug."
    if intent == "Quality Improvement":
        return "Improve readability or presentation while preserving the same data meaning and avoiding unnecessary changes."
    if intent == "Style Adaptation":
        return "Use the base data/code meaning, but match the reference image style as closely as possible."
    if intent == "Constraint Satisfaction":
        return "Enforce the explicit user requirement while preserving data meaning and readability."
    if intent == "Consistency Harmonization":
        return "Fix inconsistent visual elements such as colors, legends, scales, labels, layers, or category mappings."
    if intent == "Refactor / Transformation":
        return "Restructure the code or visualization as requested while preserving intended semantics and matching the reference when applicable."
    if intent == "Robustness / Generalization":
        return "Make the chart handle changing data, long labels, missing values, duplicates, large ranges, or variable input sizes."
    if intent == "Style-Aware Error Repair":
        return "Match or preserve the desired style while avoiding the marked or anticipated visual errors."
    return "Infer the required visualization edit from the instruction and image while preserving relevant data semantics."


def get_intent_strategies(intent, k=TOP_K_CANDIDATES):
    strategies_by_intent = {
        "Correctness Repair": [
            "Minimal targeted bug fix: change only the code needed to fix the described visual/semantic error.",
            "Data/encoding repair: prioritize correct data types, grouping, sorting, aggregation, mappings, and axis encodings.",
            "Layout-preserving repair: fix the issue while keeping the original chart type, style, layout, and labels as much as possible.",
            "Library-idiomatic repair: rewrite the broken plotting/specification logic using standard Matplotlib or Vega-Lite practices.",
            "Robust repair: fix the issue and handle likely edge cases such as duplicates, missing values, long labels, or changing ranges.",
        ],
        "Quality Improvement": [
            "Label and axis readability improvement: fix tick labels, rotations, font sizes, clipping, and axis formatting.",
            "Layout and spacing improvement: adjust figure size, margins, tight/constrained layout, legend placement, and whitespace.",
            "Visual clarity improvement: improve colors, contrast, gridlines, titles, and annotations without changing data meaning.",
            "Minimal presentation fix: make the smallest visual changes needed and preserve the original chart.",
            "Robust readability fix: ensure readability with more categories, long labels, or crowded data.",
        ],
        "Style Adaptation": [
            "Color and mark-style matching: prioritize reference colors, line/bar/point styles, and theme.",
            "Layout and aspect-ratio matching: prioritize chart size, spacing, margins, legend placement, and axis layout.",
            "Typography and annotation matching: prioritize title, labels, fonts, annotations, and text placement.",
            "Grid/background/axis styling match: prioritize gridlines, background, borders, ticks, and axis appearance.",
            "Full style recreation with base data: combine style elements from the reference while preserving data semantics.",
        ],
        "Constraint Satisfaction": [
            "Direct constraint implementation: enforce the explicit requirement exactly as stated.",
            "Constraint plus readability: satisfy the requirement while keeping labels, axes, legends, and annotations clear.",
            "Constraint plus semantic preservation: satisfy the requirement while avoiding changes to data meaning or aggregation.",
            "Constraint plus robust handling: implement the requirement in a way that handles future data changes.",
            "Minimal constraint fix: make the smallest safe code change that satisfies the stated constraint.",
        ],
        "Consistency Harmonization": [
            "Color consistency: ensure the same categories/series use the same colors across marks, layers, or legends.",
            "Legend consistency: remove duplicate legends, align legend labels with plotted marks, and fix legend placement.",
            "Scale and axis consistency: harmonize shared scales, axis ranges, tick labels, and units across chart elements.",
            "Label/category consistency: align labels, field names, annotations, and categories with the plotted data.",
            "Layer/encoding consistency: make layered or grouped charts use consistent encodings and configurations.",
        ],
        "Refactor / Transformation": [
            "Simple refactor: simplify and clean the code while preserving the intended rendered output.",
            "Chart-structure transformation: transform the chart representation to match the instruction or reference image.",
            "Data-pipeline transformation: cleanly restructure data preparation, aggregation, filtering, or reshaping.",
            "Maintainable implementation: rewrite the chart in a clearer, reusable, library-idiomatic way.",
            "Reference-matching transformation: transform the visualization structure while preserving data semantics and matching the target.",
        ],
        "Robustness / Generalization": [
            "Missing/invalid value robustness: handle NaNs, non-numeric values, missing categories, or empty data safely.",
            "Label/category scalability: handle long labels, many categories, duplicates, and crowded axes dynamically.",
            "Range/scale robustness: handle large or changing values, outliers, and axis formatting.",
            "Dynamic layout robustness: adapt figure size, legend placement, margins, and text layout to variable data.",
            "Generalized data pipeline: avoid hard-coded assumptions and compute chart values dynamically from the data.",
        ],
        "Style-Aware Error Repair": [
            "Clean target recreation: match the target chart style/output while removing marked or anticipated errors.",
            "Style plus label/layout repair: match style while avoiding overlap, clipping, and poor spacing.",
            "Style plus axis/legend repair: match style while fixing axes, legends, scales, or annotations.",
            "Robust styled output: match the target style with dynamic layout and safe handling of data changes.",
            "Conservative style-aware fix: preserve the intended style with minimal changes and avoid visual artifacts.",
        ],
    }
    default = [
        "Minimal targeted edit that follows the instruction.",
        "Data/semantic-preserving edit that matches the image.",
        "Layout/readability-focused edit.",
        "Reference-image visual matching edit.",
        "Robust/generalized edit.",
    ]
    strategies = strategies_by_intent.get(str(intent).strip(), default)
    out = []
    while len(out) < k:
        out.extend(strategies)
    return out[:k]


def build_plan_prompt(row):
    intent = row_value(row, "intent")
    instruction = row_value(row, "Instruction")
    lang = row_value(row, "Lang")
    code = truncate_text(row_value(row, "code"), 9000)
    context = row_context(row)
    return f"""
You are planning a visualization code edit. The chart image is provided separately.

Intent: {intent}
Language / Library: {lang}
{context}

User instruction:
{instruction}

Base code:
{code}

Produce a concise edit plan identifying:
1. The concrete issue or target change.
2. Relevant code/data/encoding/layout elements.
3. What data semantics must be preserved.
4. What the final rendered chart should look like.
5. Risks to avoid.

Return only the plan, not code.
""".strip()


def build_candidate_prompt(row, plan, strategy, candidate_id, k):
    intent = row_value(row, "intent")
    instruction = row_value(row, "Instruction")
    lang = row_value(row, "Lang")
    code = truncate_text(row_value(row, "code"), 12000)
    context = row_context(row)
    mode = intent_mode_text(intent)
    return f"""
You are an expert visualization code editor. The chart image is provided separately.

Generate candidate {candidate_id} of {k}. All candidates must solve the SAME intent and issue.

Intent: {intent}
Intent-specific rule: {mode}
Candidate strategy: {strategy}

Edit plan:
{plan}

Language / Library: {lang}
{context}

User instruction:
{instruction}

Base code:
{code}

Rules:
- Return only complete revised code. No explanation. No markdown fences.
- If Python/Matplotlib, return complete Python code.
- If Vega-Lite, return complete valid Vega-Lite JSON.
- Preserve relevant data semantics unless required.
- Match the corrected/reference chart closely in chart type, encodings, colors, fonts, title, labels, axes, legend, annotations, layout, spacing, and aspect ratio.
- For Correctness Repair and Quality Improvement, fix the issue while keeping the clean corrected chart visually similar.
- For all other intents, treat the image as the target reference unless the instruction says otherwise.
- Ignore handwritten marks unless they indicate issues to fix; do not reproduce marks.

Return only the complete revised code.
""".strip()


def build_refinement_prompt(row, plan, best_code, best_judge, best_exec, strategy):
    intent = row_value(row, "intent")
    instruction = row_value(row, "Instruction")
    lang = row_value(row, "Lang")
    base_code = truncate_text(row_value(row, "code"), 8000)
    best_code = truncate_text(best_code, 12000)
    context = row_context(row)
    mode = intent_mode_text(intent)
    metrics_text = json.dumps(best_judge, indent=2)
    stderr = truncate_text(best_exec.get("stderr", ""), 3000)
    stdout = truncate_text(best_exec.get("stdout", ""), 1000)
    return f"""
You are refining a visualization code edit.

Images:
- Image 1 is the original/reference/buggy chart from the task.
- If Image 2 is provided, it is the rendered output from the current best candidate.

Intent: {intent}
Intent-specific rule: {mode}
Instruction: {instruction}
Language / Library: {lang}
{context}

Original edit plan:
{plan}

Candidate strategy:
{strategy}

Base code:
{base_code}

Current best candidate code:
{best_code}

Execution feedback:
returncode: {best_exec.get('returncode', '')}
stderr:
{stderr}
stdout:
{stdout}

Judge feedback / scores:
{metrics_text}

Refine the candidate code to better satisfy the instruction and visual target.
Fix execution errors if any. If it already executes, improve task accuracy, readability/clarity, visual quality, and visual similarity.
Avoid over-editing or changing data semantics unnecessarily.

Return only complete refined code. No explanation. No markdown fences.
""".strip()

# RENDERING / EXECUTION

def try_parse_json(text):
    text = clean_code(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return None


def is_vegalite_json(obj):
    if not isinstance(obj, dict):
        return False
    schema = str(obj.get("$schema", "")).lower()
    return ("vega-lite" in schema) or ("mark" in obj and "encoding" in obj)


def execute_vegalite_one_by_one(pred_code, output_png):
    spec = try_parse_json(pred_code)
    if not is_vegalite_json(spec):
        return 0, "Not Vega-Lite JSON."
    try:
        import vl_convert as vlc
    except Exception as e:
        return 0, "Vega-Lite JSON detected, but vl-convert-python is missing. Error: " + repr(e)
    try:
        png_bytes = vlc.vegalite_to_png(spec)
        Path(output_png).write_bytes(png_bytes)
        ok = int(Path(output_png).exists() and Path(output_png).stat().st_size > 0)
        return ok, ""
    except Exception as e:
        return 0, repr(e)


def make_matplotlib_script(pred_code, output_png):
    pred_code = clean_code(pred_code)
    output_png = str(Path(output_png).resolve())
    lines = []
    lines.append("import os")
    lines.append('os.environ["MPLBACKEND"] = "Agg"')
    lines.append("import matplotlib")
    lines.append('matplotlib.use("Agg")')
    lines.append("import matplotlib.pyplot as plt")
    lines.append("plt.show = lambda *args, **kwargs: None")
    lines.append(f"OUTPUT_PNG = {repr(output_png)}")
    lines.append("")
    lines.append("# ---------- GENERATED CODE START ----------")
    lines.append(pred_code)
    lines.append("# ---------- GENERATED CODE END ----------")
    lines.append("")
    lines.append("if len(plt.get_fignums()) > 0:")
    lines.append('    plt.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=150)')
    lines.append("")
    return "\n".join(lines)


def execute_matplotlib_one_by_one(pred_code, output_dir, timeout=EXECUTION_TIMEOUT):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / "rendered.png"
    script_path = output_dir / "candidate.py"
    script_path.write_text(make_matplotlib_script(pred_code, output_png), encoding="utf-8")
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONWARNINGS"] = "ignore"
    try:
        result = subprocess.run(
            [sys.executable, "candidate.py"],
            cwd=str(output_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
        )
        ok = int(result.returncode == 0 and output_png.exists() and output_png.stat().st_size > 0)
        return {
            "executable": ok,
            "rendered_image_path": str(output_png) if output_png.exists() else "",
            "stdout": result.stdout[-5000:],
            "stderr": result.stderr[-5000:],
            "returncode": result.returncode,
            "render_backend": "matplotlib",
        }
    except subprocess.TimeoutExpired as e:
        return {
            "executable": 0,
            "rendered_image_path": "",
            "stdout": "",
            "stderr": "TimeoutExpired: " + repr(e),
            "returncode": -1,
            "render_backend": "matplotlib",
        }
    except Exception as e:
        return {
            "executable": 0,
            "rendered_image_path": "",
            "stdout": "",
            "stderr": repr(e),
            "returncode": -1,
            "render_backend": "matplotlib",
        }


def execute_one_example(pred_code, output_dir, timeout=EXECUTION_TIMEOUT):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / "rendered.png"
    output_png.unlink(missing_ok=True)
    pred_code = clean_code(pred_code)
    spec = try_parse_json(pred_code)
    if is_vegalite_json(spec):
        ok, err = execute_vegalite_one_by_one(pred_code, output_png)
        return {
            "executable": int(ok),
            "rendered_image_path": str(output_png) if output_png.exists() else "",
            "stdout": "",
            "stderr": err,
            "returncode": 0 if ok else -1,
            "render_backend": "vegalite",
        }
    return execute_matplotlib_one_by_one(pred_code, output_dir, timeout=timeout)


# VLM JUDGE

RUBRIC_TEXT = """
Use only these scores: 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5.

Metric 1: Task Accuracy, 0-5
For Correctness Repair and Quality Improvement:
- 5 = Fully fixes the issue shown in the image and instruction, preserves data meaning, and introduces no meaningful new problems.
- 0 = Invalid, irrelevant, unreadable, or does not address the requested fix at all.
For all other intents:
- 5 = Closely matches the reference/target image and instruction, preserving data meaning unless the instruction asks otherwise.
- 0 = Invalid, irrelevant, unreadable, or does not match the reference/instruction at all.

Metric 2: Clarity and Readability, 0-5
- 5 = Labels, titles, axes, ticks, legends, fonts, and annotations are clear, readable, well-positioned, and non-overlapping.
- 0 = No valid readable chart, or chart elements are unreadable, clipped, or overlapped so badly that the chart cannot be interpreted.

Metric 3: Visual Quality, 0-5
- 5 = Polished, professional, visually clean, accessible, uses appropriate colors/scales/chart type, and communicates the intended information clearly.
- 0 = No valid visualization, irrelevant image, misleading visual design, or extremely poor visual quality.

Metric 4: Visual Similarity, 0-100
For Correctness Repair and Quality Improvement:
- Compare the rendered chart to the clean corrected version of the original buggy/marked chart.
- The rendered chart should remain visually similar to the original chart, but should fix the issue described in the instruction.
- Do not penalize the rendered chart for not reproducing handwritten marks or the original bug.

For Style Adaptation:
- Compare style similarity: colors, fonts, background, gridlines, mark style, layout, and overall appearance.
- Data values may differ if the task uses base data, but the visual style should match the reference image closely.

For Style-Aware Error Repair:
- The reference image is the target chart style/output and may show marked potential errors.
- The rendered chart should recreate the clean target chart while avoiding the marked or anticipated issues.
- Do not reward reproducing handwritten marks or buggy artifacts.

For all other intents:
- The reference image is the target chart.
- The rendered chart should match it as closely as possible in chart type, data, encodings, colors, labels, title, axes, legend, annotations, layout, and overall appearance.

Visual Similarity scale:
100 = near-perfect match under the intent-specific rules.
90-99 = almost identical, only tiny differences.
80-89 = strong match, minor differences.
60-79 = partially correct but noticeable differences.
40-59 = weak match.
1-39 = barely related.
0 = completely different, invalid, or no usable chart.

Important:
- If executable = 0, all scores must be 0.
- Be strict.
- Ignore handwritten annotations, circles, arrows, scribbles, highlights, and notes unless they describe what should be fixed.
- Do not reward a nice-looking chart if it fails the instruction.
- Do not reward a chart that changes data meaning unless requested.
"""


def judge_schema():
    return {
        "type": "json_schema",
        "name": "viseditbench_eval",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "task_accuracy": {"type": "number", "enum": ALLOWED_SCORES},
                "clarity_readability": {"type": "number", "enum": ALLOWED_SCORES},
                "visual_quality": {"type": "number", "enum": ALLOWED_SCORES},
                "visual_similarity_score": {"type": "integer", "minimum": 0, "maximum": 100},
                "rationale": {"type": "string"},
            },
            "required": [
                "task_accuracy",
                "clarity_readability",
                "visual_quality",
                "visual_similarity_score",
                "rationale",
            ],
        },
    }


def build_judge_prompt(row, executable, pred_code):
    intent = row_value(row, "intent")
    instruction = row_value(row, "Instruction")
    input_code = truncate_text(row_value(row, "code"), 7000)
    pred_code = truncate_text(pred_code, 7000)
    if intent in REPAIR_INTENTS:
        image_context = """
You will see two images:
Image 1: the original buggy or marked chart.
Image 2: the chart rendered from the generated code.

For this task, judge whether Image 2 fixes the issue shown in Image 1 and described in the instruction.
For visual similarity, compare Image 2 to the clean corrected version of Image 1: it should stay visually similar to Image 1, but should not preserve the bug or handwritten/error markings.
"""
    elif intent in STYLE_ADAPTATION_INTENTS:
        image_context = """
You will see two images:
Image 1: the reference style chart.
Image 2: the chart rendered from the generated code.

For this task, judge whether Image 2 matches the style of Image 1.
The exact data values may differ if the task uses base data, but the visual style should match closely.
"""
    elif intent in STYLE_AWARE_ERROR_REPAIR_INTENTS:
        image_context = """
You will see two images:
Image 1: the target chart style/output, possibly with marked potential errors.
Image 2: the chart rendered from the generated code.

For this task, judge whether Image 2 recreates the clean target chart while avoiding the marked or anticipated issues.
Do not penalize Image 2 for not reproducing handwritten/error markings.
"""
    else:
        image_context = """
You will see two images:
Image 1: the reference or target chart.
Image 2: the chart rendered from the generated code.

For this task, judge whether Image 2 matches the reference image and follows the instruction.
For visual similarity, Image 2 should match Image 1 as closely as possible in chart type, data, encodings, colors, labels, title, axes, legend, annotations, layout, and overall appearance.
"""
    return f"""
You are a strict expert evaluator for visualization code editing.

Intent:
{intent}

Instruction:
{instruction}

Execution status:
executable = {executable}

{image_context}

Input code:
```python
{input_code}
```

Generated code:
```python
{pred_code}
```

{RUBRIC_TEXT}

Return JSON only with:
- task_accuracy
- clarity_readability
- visual_quality
- visual_similarity_score
- rationale
""".strip()


def judge_with_vlm(row, original_image_path, rendered_image_path, executable, pred_code, judge_model=JUDGE_MODEL, max_retries=JUDGE_MAX_RETRIES):
    if executable == 0:
        return {"task_accuracy": 0.0, "clarity_readability": 0.0, "visual_quality": 0.0, "visual_similarity_score": 0, "rationale": "Not executable or no rendered image."}
    if not original_image_path or not Path(original_image_path).exists():
        return {"task_accuracy": 0.0, "clarity_readability": 0.0, "visual_quality": 0.0, "visual_similarity_score": 0, "rationale": "Original/reference image path missing."}
    if not rendered_image_path or not Path(rendered_image_path).exists():
        return {"task_accuracy": 0.0, "clarity_readability": 0.0, "visual_quality": 0.0, "visual_similarity_score": 0, "rationale": "Rendered image path missing."}

    client = make_openai_client(timeout=JUDGE_TIMEOUT)
    prompt = build_judge_prompt(row, executable, pred_code)
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=judge_model,
                temperature=0.0,
                max_output_tokens=MAX_OUTPUT_TOKENS_JUDGE,
                instructions="Return only valid JSON. Use only scores in increments of 0.5 for 0-5 metrics.",
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_to_data_url(original_image_path), "detail": "high"},
                        {"type": "input_image", "image_url": image_to_data_url(rendered_image_path), "detail": "high"},
                    ],
                }],
                text={"format": judge_schema()},
            )
            result = json.loads(response.output_text)
            return {
                "task_accuracy": round_to_half(result.get("task_accuracy", 0.0)),
                "clarity_readability": round_to_half(result.get("clarity_readability", 0.0)),
                "visual_quality": round_to_half(result.get("visual_quality", 0.0)),
                "visual_similarity_score": clamp_similarity(result.get("visual_similarity_score", 0)),
                "rationale": str(result.get("rationale", "")),
            }
        except Exception as e:
            if attempt == max_retries - 1:
                return {"task_accuracy": 0.0, "clarity_readability": 0.0, "visual_quality": 0.0, "visual_similarity_score": 0, "rationale": "Judge failed: " + repr(e)}
            time.sleep(SLEEP_AFTER_ERROR)
    return {"task_accuracy": 0.0, "clarity_readability": 0.0, "visual_quality": 0.0, "visual_similarity_score": 0, "rationale": "Judge failed unexpectedly."}


def candidate_rank_score(executable, judge_result):
    if int(executable) != 1:
        return 0.0
    task_pct = float(judge_result.get("task_accuracy", 0.0)) / 5.0 * 100.0
    clarity_pct = float(judge_result.get("clarity_readability", 0.0)) / 5.0 * 100.0
    quality_pct = float(judge_result.get("visual_quality", 0.0)) / 5.0 * 100.0
    sim_pct = float(judge_result.get("visual_similarity_score", 0.0))
    return round(0.35 * task_pct + 0.20 * clarity_pct + 0.20 * quality_pct + 0.25 * sim_pct, 4)


def compute_pass(executable, judge_result):
    task_accuracy = round_to_half(judge_result.get("task_accuracy", 0.0))
    clarity_readability = round_to_half(judge_result.get("clarity_readability", 0.0))
    visual_quality = round_to_half(judge_result.get("visual_quality", 0.0))
    visual_similarity_score = clamp_similarity(judge_result.get("visual_similarity_score", 0))
    return int(
        int(executable) == 1
        and task_accuracy >= 4.5
        and clarity_readability >= 4.0
        and visual_quality >= 4.0
        and visual_similarity_score > 90
    )

# AGENT CORE

def generate_plan(row, image_path):
    prompt = build_plan_prompt(row)
    text = call_vlm_with_images(
        prompt=prompt,
        image_paths=[image_path],
        model=AGENT_MODEL,
        instructions="You are an expert visualization code editor. Return only a concise edit plan.",
        max_output_tokens=MAX_OUTPUT_TOKENS_PLAN,
        temperature=PLANNING_TEMPERATURE,
        top_p=GENERATION_TOP_P,
        timeout=OPENAI_TIMEOUT,
    )
    return str(text).strip()


def generate_candidate(row, image_path, plan, strategy, candidate_id, k):
    prompt = build_candidate_prompt(row, plan, strategy, candidate_id, k)
    text = call_vlm_with_images(
        prompt=prompt,
        image_paths=[image_path],
        model=AGENT_MODEL,
        instructions="You are an expert visualization code editor. Return only complete corrected code. No markdown. No explanation.",
        max_output_tokens=MAX_OUTPUT_TOKENS_CODE,
        temperature=GENERATION_TEMPERATURE,
        top_p=GENERATION_TOP_P,
        timeout=OPENAI_TIMEOUT,
    )
    return clean_code(text)


def refine_candidate(row, original_image_path, rendered_image_path, plan, best_code, best_judge, best_exec, strategy):
    prompt = build_refinement_prompt(row, plan, best_code, best_judge, best_exec, strategy)
    image_paths = [original_image_path]
    if rendered_image_path and Path(rendered_image_path).exists():
        image_paths.append(rendered_image_path)
    text = call_vlm_with_images(
        prompt=prompt,
        image_paths=image_paths,
        model=AGENT_MODEL,
        instructions="You are an expert visualization code editor. Return only complete refined code. No markdown. No explanation.",
        max_output_tokens=MAX_OUTPUT_TOKENS_CODE,
        temperature=REFINEMENT_TEMPERATURE,
        top_p=GENERATION_TOP_P,
        timeout=OPENAI_TIMEOUT,
    )
    return clean_code(text)


def summarize_candidate(candidate, store_code=STORE_CANDIDATE_CODES):
    out = {
        "candidate_id": candidate.get("candidate_id"),
        "stage": candidate.get("stage"),
        "strategy": candidate.get("strategy"),
        "rank_score": candidate.get("rank_score"),
        "executable": candidate.get("executable"),
        "render_backend": candidate.get("render_backend"),
        "rendered_image_path": candidate.get("rendered_image_path"),
        "execution_returncode": candidate.get("execution_returncode"),
        "execution_stderr": truncate_text(candidate.get("execution_stderr", ""), 1200),
        "task_accuracy": candidate.get("task_accuracy"),
        "clarity_readability": candidate.get("clarity_readability"),
        "visual_quality": candidate.get("visual_quality"),
        "visual_similarity_score": candidate.get("visual_similarity_score"),
        "passed": candidate.get("passed"),
        "judge_rationale": truncate_text(candidate.get("judge_rationale", ""), 1200),
    }
    if store_code:
        out["pred_code"] = candidate.get("pred_code", "")
    return out


def evaluate_candidate(row, candidate_code, original_image_path, output_dir, stage, candidate_id, strategy, timeout):
    exec_result = execute_one_example(candidate_code, output_dir=output_dir, timeout=timeout)
    executable = int(exec_result.get("executable", 0))
    rendered_image_path = exec_result.get("rendered_image_path", "")
    judge_result = judge_with_vlm(
        row=row,
        original_image_path=original_image_path,
        rendered_image_path=rendered_image_path,
        executable=executable,
        pred_code=candidate_code,
        judge_model=JUDGE_MODEL,
    )
    rank_score = candidate_rank_score(executable, judge_result)
    passed = compute_pass(executable, judge_result)
    return {
        "candidate_id": candidate_id,
        "stage": stage,
        "strategy": strategy,
        "pred_code": candidate_code,
        "rank_score": rank_score,
        "executable": executable,
        "render_backend": exec_result.get("render_backend", ""),
        "rendered_image_path": rendered_image_path,
        "execution_returncode": exec_result.get("returncode", -1),
        "execution_stdout": exec_result.get("stdout", ""),
        "execution_stderr": exec_result.get("stderr", ""),
        "task_accuracy": round_to_half(judge_result.get("task_accuracy", 0.0)),
        "clarity_readability": round_to_half(judge_result.get("clarity_readability", 0.0)),
        "visual_quality": round_to_half(judge_result.get("visual_quality", 0.0)),
        "visual_similarity_score": clamp_similarity(judge_result.get("visual_similarity_score", 0)),
        "passed": passed,
        "judge_rationale": judge_result.get("rationale", ""),
    }


def make_failed_agent_row(row, error_message, image_path=""):
    result_row = dict(row)
    result_row.update({
        "agent_model": AGENT_MODEL,
        "judge_model": JUDGE_MODEL,
        "used_image_path": image_path,
        "agent_plan": "",
        "candidate_summary_json": "[]",
        "best_candidate_id": "",
        "best_strategy": "",
        "best_stage": "error",
        "pred_code": "",
        "generated_code": "",
        "executable": 0,
        "render_backend": "error",
        "rendered_image_path": "",
        "execution_returncode": -1,
        "execution_stdout": "",
        "execution_stderr": error_message,
        "task_accuracy": 0.0,
        "clarity_readability": 0.0,
        "visual_quality": 0.0,
        "visual_similarity_score": 0,
        "rank_score": 0.0,
        "passed": 0,
        "judge_rationale": "Row skipped due to error: " + error_message,
    })
    return result_row


def run_viseditagent_one_row(row, image_dir, render_dir, timeout=EXECUTION_TIMEOUT, k=TOP_K_CANDIDATES):
    img_id = row_value(row, "IMG_ID")
    image_path = row_value(row, "image_path") or row_value(row, "used_image_path") or find_image_path(image_dir, img_id)
    if not image_path or not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found for IMG_ID={img_id}")

    example_dir = Path(render_dir).resolve() / safe_name(img_id)
    example_dir.mkdir(parents=True, exist_ok=True)

    plan = generate_plan(row, image_path)
    strategies = get_intent_strategies(row_value(row, "intent"), k=k)
    candidates = []

    for i, strategy in enumerate(strategies, start=1):
        candidate_dir = example_dir / f"candidate_{i}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        try:
            candidate_code = generate_candidate(row, image_path, plan, strategy, i, k)
            candidate = evaluate_candidate(row, candidate_code, image_path, candidate_dir, "candidate", i, strategy, timeout)
        except Exception as e:
            candidate = {
                "candidate_id": i,
                "stage": "candidate",
                "strategy": strategy,
                "pred_code": "",
                "rank_score": 0.0,
                "executable": 0,
                "render_backend": "error",
                "rendered_image_path": "",
                "execution_returncode": -1,
                "execution_stdout": "",
                "execution_stderr": repr(e),
                "task_accuracy": 0.0,
                "clarity_readability": 0.0,
                "visual_quality": 0.0,
                "visual_similarity_score": 0,
                "passed": 0,
                "judge_rationale": "Candidate failed: " + repr(e),
            }
        candidates.append(candidate)

    if not candidates:
        raise RuntimeError("No candidates were generated.")

    best = max(candidates, key=lambda x: x.get("rank_score", 0.0))
    best_before_refine = best

    if REFINE_ONCE:
        refined_dir = example_dir / "refined"
        refined_dir.mkdir(parents=True, exist_ok=True)
        try:
            best_exec = {
                "returncode": best.get("execution_returncode", -1),
                "stdout": best.get("execution_stdout", ""),
                "stderr": best.get("execution_stderr", ""),
            }
            best_judge = {
                "task_accuracy": best.get("task_accuracy", 0.0),
                "clarity_readability": best.get("clarity_readability", 0.0),
                "visual_quality": best.get("visual_quality", 0.0),
                "visual_similarity_score": best.get("visual_similarity_score", 0),
                "rationale": best.get("judge_rationale", ""),
            }
            refined_code = refine_candidate(
                row=row,
                original_image_path=image_path,
                rendered_image_path=best.get("rendered_image_path", ""),
                plan=plan,
                best_code=best.get("pred_code", ""),
                best_judge=best_judge,
                best_exec=best_exec,
                strategy=best.get("strategy", ""),
            )
            refined_candidate = evaluate_candidate(row, refined_code, image_path, refined_dir, "refined", "refined", "refine best candidate once", timeout)
            candidates.append(refined_candidate)
            if refined_candidate.get("rank_score", 0.0) >= best.get("rank_score", 0.0):
                best = refined_candidate
        except Exception as e:
            candidates.append({
                "candidate_id": "refined",
                "stage": "refined",
                "strategy": "refine best candidate once",
                "pred_code": "",
                "rank_score": 0.0,
                "executable": 0,
                "render_backend": "error",
                "rendered_image_path": "",
                "execution_returncode": -1,
                "execution_stdout": "",
                "execution_stderr": repr(e),
                "task_accuracy": 0.0,
                "clarity_readability": 0.0,
                "visual_quality": 0.0,
                "visual_similarity_score": 0,
                "passed": 0,
                "judge_rationale": "Refinement failed: " + repr(e),
            })

    result_row = dict(row)
    result_row.update({
        "agent_model": AGENT_MODEL,
        "judge_model": JUDGE_MODEL,
        "agent_k": k,
        "agent_refine_once": int(REFINE_ONCE),
        "used_image_path": image_path,
        "agent_plan": plan,
        "candidate_summary_json": json.dumps([summarize_candidate(c) for c in candidates], ensure_ascii=False),
        "best_candidate_id": best.get("candidate_id", ""),
        "best_strategy": best.get("strategy", ""),
        "best_stage": best.get("stage", ""),
        "best_before_refine_candidate_id": best_before_refine.get("candidate_id", ""),
        "best_before_refine_score": best_before_refine.get("rank_score", 0.0),
        "pred_code": best.get("pred_code", ""),
        "generated_code": best.get("pred_code", ""),
        "executable": int(best.get("executable", 0)),
        "render_backend": best.get("render_backend", ""),
        "rendered_image_path": best.get("rendered_image_path", ""),
        "execution_returncode": best.get("execution_returncode", -1),
        "execution_stdout": best.get("execution_stdout", ""),
        "execution_stderr": best.get("execution_stderr", ""),
        "task_accuracy": round_to_half(best.get("task_accuracy", 0.0)),
        "clarity_readability": round_to_half(best.get("clarity_readability", 0.0)),
        "visual_quality": round_to_half(best.get("visual_quality", 0.0)),
        "visual_similarity_score": clamp_similarity(best.get("visual_similarity_score", 0)),
        "rank_score": best.get("rank_score", 0.0),
        "passed": int(best.get("passed", 0)),
        "judge_rationale": best.get("judge_rationale", ""),
    })
    return result_row


def run_viseditagent(df, image_dir=IMAGE_DIR, render_dir=RENDER_DIR, output_csv=OUTPUT_CSV,
                     limit=LIMIT, save_every=SAVE_EVERY, resume=RESUME,
                     skip_positions=SKIP_POSITIONS, timeout=EXECUTION_TIMEOUT, k=TOP_K_CANDIDATES):
    required_cols = {"IMG_ID", "code", "Instruction", "intent"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work_df = df.copy()
    if skip_positions:
        work_df = work_df.drop(work_df.index[skip_positions]).copy()
    if limit is not None:
        work_df = work_df.head(limit).copy()

    output_path = Path(output_csv) if output_csv else None
    previous_df = None
    done_ids = set()
    if resume and output_path is not None and output_path.exists():
        previous_df = pd.read_csv(output_path)
        if "IMG_ID" in previous_df.columns:
            done_ids = set(previous_df["IMG_ID"].astype(str))
            work_df = work_df[~work_df["IMG_ID"].astype(str).isin(done_ids)].copy()
            print(f"Resume mode: found {len(done_ids)} completed rows. Remaining: {len(work_df)}")

    Path(render_dir).mkdir(parents=True, exist_ok=True)
    results = []

    for _, row in tqdm(work_df.iterrows(), total=len(work_df), desc=f"Running VisEditAgent ({AGENT_MODEL})"):
        image_path_for_error = ""
        try:
            img_id = row_value(row, "IMG_ID")
            image_path_for_error = row_value(row, "image_path") or row_value(row, "used_image_path") or find_image_path(image_dir, img_id)
            result_row = run_viseditagent_one_row(row, image_dir=image_dir, render_dir=render_dir, timeout=timeout, k=k)
        except Exception as e:
            result_row = make_failed_agent_row(row, repr(e), image_path=image_path_for_error)
            time.sleep(SLEEP_AFTER_ERROR)
        results.append(result_row)

        if output_path is not None and save_every is not None and len(results) % save_every == 0:
            current_df = pd.DataFrame(results)
            if previous_df is not None:
                save_df = pd.concat([previous_df, current_df], ignore_index=True).drop_duplicates(subset=["IMG_ID"], keep="last")
            else:
                save_df = current_df
            save_df.to_csv(output_path, index=False)

    new_df = pd.DataFrame(results)
    if previous_df is not None:
        final_df = pd.concat([previous_df, new_df], ignore_index=True).drop_duplicates(subset=["IMG_ID"], keep="last")
    else:
        final_df = new_df

    if output_path is not None:
        final_df.to_csv(output_path, index=False)

    print_summary(final_df)
    return final_df


def print_summary(eval_df):
    if len(eval_df) == 0:
        print("No evaluated rows.")
        return
    print("\n===== Overall Summary =====")
    print(f"Examples: {len(eval_df)}")
    for col in ["executable", "task_accuracy", "clarity_readability", "visual_quality", "visual_similarity_score", "passed"]:
        if col in eval_df.columns:
            print(f"{col:25s}: {eval_df[col].mean():.4f}")

    if "intent" in eval_df.columns:
        print("\n===== By Intent =====")
        summary = (
            eval_df.groupby("intent")
            .agg(
                n=("IMG_ID", "count"),
                executability=("executable", "mean"),
                task_accuracy=("task_accuracy", "mean"),
                clarity_readability=("clarity_readability", "mean"),
                visual_quality=("visual_quality", "mean"),
                visual_similarity_score=("visual_similarity_score", "mean"),
                pass_rate=("passed", "mean"),
            )
            .reset_index()
        )
        print(summary.to_string(index=False))

    if "overall_complexity" in eval_df.columns:
        print("\n===== By Complexity =====")
        complexity_summary = (
            eval_df.groupby("overall_complexity")
            .agg(
                n=("IMG_ID", "count"),
                executability=("executable", "mean"),
                task_accuracy=("task_accuracy", "mean"),
                clarity_readability=("clarity_readability", "mean"),
                visual_quality=("visual_quality", "mean"),
                visual_similarity_score=("visual_similarity_score", "mean"),
                pass_rate=("passed", "mean"),
            )
            .reset_index()
        )
        print(complexity_summary.to_string(index=False))


def main():
    global AGENT_MODEL, JUDGE_MODEL, REFINE_ONCE
    global GENERATION_TEMPERATURE, PLANNING_TEMPERATURE, REFINEMENT_TEMPERATURE
    global GENERATION_TOP_P, MAX_OUTPUT_TOKENS_CODE

    parser = argparse.ArgumentParser(
        description="Run VisEditAgent. Executes generated code locally.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--agent-model", default=AGENT_MODEL, help="Planning and editing model.")
    parser.add_argument("--judge-model", default=JUDGE_MODEL, help="Visual judge model.")
    parser.add_argument("--candidates", type=int, default=TOP_K_CANDIDATES, help="Number of candidates.")
    parser.add_argument("--temperature", type=float, help="Temperature for planning, generation, and refinement.")
    parser.add_argument("--top-p", type=float, help="Top-p for planning, generation, and refinement.")
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS_CODE,
                        help="Code generation and refinement token limit.")
    parser.add_argument("--timeout", type=int, default=EXECUTION_TIMEOUT, help="Python execution timeout in seconds.")
    parser.add_argument("--limit", type=int, help="Run only the first N examples.")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, help="Save every N examples.")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV, help="Predictions and evaluation CSV.")
    parser.add_argument("--render-dir", type=Path, default=RENDER_DIR, help="Candidate renders.")
    parser.add_argument("--no-refine", action="store_true", help="Disable the single refinement step.")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh and replace the output CSV.")
    args = parser.parse_args()
    for name in ("candidates", "max_output_tokens", "timeout", "limit", "save_every"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        parser.error("--top-p must be greater than 0 and at most 1")
    if args.output.resolve() == (ROOT / "dataset.csv").resolve():
        parser.error("--output must differ from dataset.csv")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in your terminal before running")

    AGENT_MODEL = args.agent_model
    JUDGE_MODEL = args.judge_model
    REFINE_ONCE = not args.no_refine
    GENERATION_TEMPERATURE = args.temperature
    PLANNING_TEMPERATURE = args.temperature
    REFINEMENT_TEMPERATURE = args.temperature
    GENERATION_TOP_P = args.top_p
    MAX_OUTPUT_TOKENS_CODE = args.max_output_tokens

    df = pd.read_csv(ROOT / "dataset.csv", keep_default_na=False)
    run_viseditagent(
        df=df,
        image_dir=IMAGE_DIR,
        render_dir=args.render_dir,
        output_csv=args.output,
        limit=args.limit,
        save_every=args.save_every,
        resume=not args.no_resume,
        timeout=args.timeout,
        k=args.candidates,
    )
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
