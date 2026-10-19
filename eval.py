"""Evaluate VisEditBench predictions with execution and multimodal visual scoring."""

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


REPAIR_INTENTS = {"Correctness Repair", "Quality Improvement"}
STYLE_ADAPTATION_INTENTS = {"Style Adaptation"}
STYLE_AWARE_ERROR_REPAIR_INTENTS = {"Style-Aware Error Repair"}

EXECUTION_TIMEOUT = 10
JUDGE_TIMEOUT = 60.0
JUDGE_MAX_RETRIES = 1
OPENAI_MAX_RETRIES = 0

ALLOWED_SCORES = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

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
- Data values may differ if the task uses the base data, but the style should match the reference image closely.

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

Intermediate guide for 0-5 metrics:
5.0 = Exceptional; fully satisfies every applicable criterion.
4.5 = Very good; satisfies the requested edit with no issues.
4.0 = Good; small issues but clearly successful.
3.5 = Decent; mostly usable but with noticeable issues.
3.0 = Average; important issues affect clarity/correctness.
2.5 = Below average; several issues may mislead or confuse.
2.0 = Poor; significant issues.
1.5 = Very poor; major problems.
1.0 = Barely usable or mostly wrong.
0.5 = Almost completely failed but has a tiny relevant element.
0.0 = Failed.

Important:
- If executable = 0, all scores must be 0.
- Be strict.
- Ignore handwritten annotations, circles, arrows, scribbles, highlights, and notes unless they describe what should be fixed.
- Do not reward a nice-looking chart if it fails the instruction.
- Do not reward a chart that changes data meaning unless requested.
"""


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

    # Handles cells that are accidentally stored as quoted repr strings.
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

    return text.strip()


def safe_name(x):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(x))


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


def image_to_data_url(image_path):
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def truncate_text(text, max_chars=5000):
    text = "" if text is None else str(text)
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n# ... TRUNCATED ...\n\n" + text[-half:]


# Vega-Lite execution

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
    """
    Render direct Vega-Lite JSON.
    Returns executable, stderr.
    """
    spec = try_parse_json(pred_code)

    if not is_vegalite_json(spec):
        return 0, "Not Vega-Lite JSON."

    try:
        import vl_convert as vlc
    except Exception as e:
        return 0, "Vega-Lite JSON detected, but vl-convert-python is missing. Install with pip install vl-convert-python. Error: " + repr(e)

    try:
        png_bytes = vlc.vegalite_to_png(spec)
        Path(output_png).write_bytes(png_bytes)
        ok = int(Path(output_png).exists() and Path(output_png).stat().st_size > 0)
        return ok, ""
    except Exception as e:
        return 0, repr(e)


# Matplotlib execution

def make_matplotlib_script(pred_code, output_png):
    """
    Very simple script:
        - set Agg backend
        - run model code
        - save active matplotlib figure to ABSOLUTE output_png
    """
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
    lines.append("# Save active Matplotlib figure.")
    lines.append("if len(plt.get_fignums()) > 0:")
    lines.append('    plt.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=150)')
    lines.append("")

    return "\n".join(lines)


def execute_matplotlib_one_by_one(pred_code, output_dir, timeout=30):
    """
    Execute one Matplotlib/Python code output.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_png = output_dir / "rendered.png"
    script_path = output_dir / "candidate.py"

    script_text = make_matplotlib_script(pred_code, output_png)
    script_path.write_text(script_text, encoding="utf-8")

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

        ok = int(
            result.returncode == 0
            and output_png.exists()
            and output_png.stat().st_size > 0
        )

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


def execute_one_example(pred_code, output_dir, timeout=30):
    """
    Simple order:
        1. Try Vega-Lite JSON.
        2. If not Vega-Lite, run Matplotlib/Python.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / "rendered.png"

    pred_code = clean_code(pred_code)
    output_png.unlink(missing_ok=True)

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


# GPT-4o judge

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


def build_judge_prompt(row, executable):
    intent = str(row["intent"])
    instruction = str(row["Instruction"])
    input_code = truncate_text(str(row.get("input_code", row.get("code", ""))))
    pred_code = truncate_text(str(row.get("pred_code", "")))

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

def judge_with_gpt4o(row, original_image_path, rendered_image_path, executable, judge_model="gpt-4o", max_retries=JUDGE_MAX_RETRIES,
                     temperature=0.0, max_output_tokens=1200):
    if executable == 0:
        return {
            "task_accuracy": 0.0,
            "clarity_readability": 0.0,
            "visual_quality": 0.0,
            "visual_similarity_score": 0,
            "rationale": "Not executable or no rendered image.",
        }

    if not original_image_path or not Path(original_image_path).exists():
        return {
            "task_accuracy": 0.0,
            "clarity_readability": 0.0,
            "visual_quality": 0.0,
            "visual_similarity_score": 0,
            "rationale": "Original/reference image path missing.",
        }

    if not rendered_image_path or not Path(rendered_image_path).exists():
        return {
            "task_accuracy": 0.0,
            "clarity_readability": 0.0,
            "visual_quality": 0.0,
            "visual_similarity_score": 0,
            "rationale": "Rendered image path missing.",
        }

    client = OpenAI(timeout=JUDGE_TIMEOUT,
                        max_retries=OPENAI_MAX_RETRIES)


    prompt = build_judge_prompt(row, executable)
    original_image_url = image_to_data_url(original_image_path)
    rendered_image_url = image_to_data_url(rendered_image_path)

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=judge_model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                instructions="Return only valid JSON. Use only scores in increments of 0.5 for 0-5 metrics.",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": original_image_url, "detail": "high"},
                            {"type": "input_image", "image_url": rendered_image_url, "detail": "high"},
                        ],
                    }
                ],
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
                return {
                    "task_accuracy": 0.0,
                    "clarity_readability": 0.0,
                    "visual_quality": 0.0,
                    "visual_similarity_score": 0,
                    "rationale": "Judge failed: " + repr(e),
                }
            time.sleep(2 ** attempt)

    return {
        "task_accuracy": 0.0,
        "clarity_readability": 0.0,
        "visual_quality": 0.0,
        "visual_similarity_score": 0,
        "rationale": "Judge failed unexpectedly.",
    }


def make_failed_result_row(row, error_message):
    result_row = dict(row)

    result_row.update({
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
        "passed": 0,
        "judge_rationale": "Row skipped due to error: " + error_message,
    })

    return result_row

# Main evaluation

def run_evaluation(
    pred_df,
    render_dir,
    output_csv=None,
    judge_model="gpt-4o",
    timeout=30,
    limit=None,
    save_every=10,
    temperature=0.0,
    max_output_tokens=1200,
):
    required_cols = {"IMG_ID", "intent", "Instruction", "image_path", "pred_code"}
    missing = required_cols - set(pred_df.columns)
    if missing:
        raise ValueError(f"Missing columns in prediction dataframe: {missing}")

    if limit is not None:
        pred_df = pred_df.head(limit).copy()
    else:
        pred_df = pred_df.copy()

    render_dir = Path(render_dir).resolve()
    render_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for _, row in tqdm(pred_df.iterrows(), total=len(pred_df), desc="Evaluating"):
        try: 
            img_id = str(row["IMG_ID"])
            example_dir = render_dir / safe_name(img_id)
            example_dir.mkdir(parents=True, exist_ok=True)
    
            pred_code = str(row.get("pred_code", ""))
            original_image_path = str(row.get("image_path", ""))
    
            if not pred_code.strip():
                exec_result = {
                    "executable": 0,
                    "rendered_image_path": "",
                    "stdout": "",
                    "stderr": "Empty pred_code.",
                    "returncode": -1,
                    "render_backend": "none",
                }
            else:
                exec_result = execute_one_example(
                    pred_code=pred_code,
                    output_dir=example_dir,
                    timeout=timeout,
                )
    
            executable = int(exec_result["executable"])
            rendered_image_path = exec_result.get("rendered_image_path", "")
    
            judge_result = judge_with_gpt4o(
                row=row,
                original_image_path=original_image_path,
                rendered_image_path=rendered_image_path,
                executable=executable,
                judge_model=judge_model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
    
            task_accuracy = round_to_half(judge_result["task_accuracy"])
            clarity_readability = round_to_half(judge_result["clarity_readability"])
            visual_quality = round_to_half(judge_result["visual_quality"])
            visual_similarity_score = clamp_similarity(judge_result["visual_similarity_score"])
    
            passed = int(
                executable == 1
                and task_accuracy >= 4.5
                and clarity_readability >= 4.0
                and visual_quality >= 4.0
                and visual_similarity_score > 90
            )
    
            result_row = dict(row)
            result_row.update(
                {
                    "executable": executable,
                    "render_backend": exec_result.get("render_backend", ""),
                    "rendered_image_path": rendered_image_path,
                    "execution_returncode": exec_result.get("returncode", -1),
                    "execution_stdout": exec_result.get("stdout", ""),
                    "execution_stderr": exec_result.get("stderr", ""),
                    "task_accuracy": task_accuracy,
                    "clarity_readability": clarity_readability,
                    "visual_quality": visual_quality,
                    "visual_similarity_score": visual_similarity_score,
                    "passed": passed,
                    "judge_rationale": judge_result.get("rationale", ""),
                }
            )
    
            results.append(result_row)
    
            if output_csv is not None and save_every is not None and len(results) % save_every == 0:
                pd.DataFrame(results).to_csv(output_csv, index=False)
        except Exception as e:
            results.append(make_failed_result_row(row, repr(e)))

    eval_df = pd.DataFrame(results)

    if output_csv is not None:
        eval_df.to_csv(output_csv, index=False)

    print_summary(eval_df)

    return eval_df



def print_summary(eval_df):
    print("\n===== Overall Summary =====")
    print(f"Examples: {len(eval_df)}")
    if eval_df.empty:
        return
    print(f"Executability:            {eval_df['executable'].mean():.4f}")
    print(f"Task Accuracy:            {eval_df['task_accuracy'].mean():.4f}")
    print(f"Clarity Readability:      {eval_df['clarity_readability'].mean():.4f}")
    print(f"Visual Quality:           {eval_df['visual_quality'].mean():.4f}")
    print(f"Visual Similarity Score:  {eval_df['visual_similarity_score'].mean():.4f}")
    print(f"Pass Rate:                {eval_df['passed'].mean():.4f}")

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



ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VisEditBench predictions. Executes generated code locally.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions", type=Path, default=ROOT / "predictions.csv",
                        help="Inference output CSV.")
    parser.add_argument("--render-dir", type=Path, default=ROOT / "renders", help="Rendered charts.")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation.csv", help="Evaluation CSV.")
    parser.add_argument("--judge-model", default="gpt-4o", help="Visual evaluator model.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Judge temperature.")
    parser.add_argument("--max-output-tokens", type=int, default=1200, help="Judge output token limit.")
    parser.add_argument("--timeout", type=int, default=10, help="Python execution timeout in seconds.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N examples.")
    parser.add_argument("--save-every", type=int, default=10, help="Save every N examples.")
    args = parser.parse_args()
    for name in ("max_output_tokens", "timeout", "limit", "save_every"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")
    if args.output.resolve() == args.predictions.resolve():
        parser.error("--output must differ from --predictions")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in your terminal before evaluation")

    pred_df = pd.read_csv(args.predictions, keep_default_na=False)
    for source, target in (("generated_code", "pred_code"),
                           ("used_image_path", "image_path"), ("code", "input_code")):
        if target not in pred_df.columns and source in pred_df.columns:
            pred_df[target] = pred_df[source]

    run_evaluation(
        pred_df=pred_df,
        render_dir=args.render_dir,
        output_csv=args.output,
        judge_model=args.judge_model,
        timeout=args.timeout,
        limit=args.limit,
        save_every=args.save_every,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    print(f"Saved evaluation to {args.output}")


if __name__ == "__main__":
    main()
