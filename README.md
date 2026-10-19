# VisEditBench

Dataset and Python scripts for **VisEditBench: Can Vision-Language Models Edit Visualization Code from Multimodal Feedback?**

[Paper](https://arxiv.org/abs/2608.10408)

VisEditBench contains 1,395 visualization code-editing tasks covering feedback-guided repair and reference-guided restyling. Each task combines existing visualization code, an image, an editing instruction, and metadata. The scripts support inference, rendered-output evaluation, and the VisEditAgent baseline through the OpenAI Responses API.

## Repository

```text
dataset.csv       Task prompts, source code, instructions, and metadata
image/            Task images referenced by IMG_ID
inference.py      Generate revised visualization code
eval.py           Render and evaluate predictions
viseditagent.py   Plan, generate candidates, render, judge, select, and refine
```

Input paths default to `dataset.csv` and `image/` beside the scripts. No notebook is required.

## Model support

GPT-4o is an example model for running inference and evaluation; the workflows
are not restricted to the GPT family. Select an inference model with `--model`,
an evaluation model with `--judge-model`, and a VisEditAgent editing model with
`--agent-model`. The agent and judge can use different models.

The current scripts use the OpenAI Python SDK and Responses API. Other models
can be used through a compatible endpoint that supports image inputs. Judge
models must also support the structured JSON output schema used by the scripts.
Set `OPENAI_BASE_URL` and `OPENAI_API_KEY` for a compatible provider when needed;
a Chat Completions-only endpoint is not sufficient. Available sampling parameters
depend on the selected model and provider.

```sh
python inference.py --model YOUR_INFERENCE_MODEL
python eval.py --judge-model YOUR_JUDGE_MODEL
python viseditagent.py --agent-model YOUR_AGENT_MODEL --judge-model YOUR_JUDGE_MODEL
```

The defaults below are runnable configuration examples, not restrictions on
which compatible models can be evaluated or used as agents and judges.

## Download the images

First, clone or download this repository. Then download the task images from
[Google Drive](https://drive.google.com/drive/folders/1y5q_Ew8nUGFhd6SSpjCMQyArJiInMhMy?usp=drive_link)
and place the image files in the repository's `image/` folder, beside `dataset.csv`.
If the download is a ZIP archive, extract it first. Keep the original filenames
and ensure the images are directly inside `image/`, rather than a nested folder.

## Setup

Use Python 3.10 or newer. Install dependencies in your Python environment:

```sh
python -m pip install openai pandas tqdm matplotlib pillow vl-convert-python
```

`vl-convert-python` is needed for direct Vega-Lite JSON rendering. Generated Python code may require additional packages depending on the task.

Set your own API key in the terminal where you run the scripts. Never commit it.

**Windows Command Prompt:**

```cmd
set "OPENAI_API_KEY=YOUR_API_KEY"
```

**PowerShell:**

```powershell
$env:OPENAI_API_KEY = "YOUR_API_KEY"
```

**Linux/macOS:**

```sh
export OPENAI_API_KEY="YOUR_API_KEY"
```

## Inference

Test two examples:

```sh
python inference.py --limit 2
```

Run the complete dataset:

```sh
python inference.py
```

Defaults are model `gpt-5`, maximum output tokens `4096`, and image detail `high`. Temperature and top-p are omitted unless supplied. Use only parameters supported by your selected model. For example:

```sh
python inference.py --model gpt-4o --temperature 0.2 --max-output-tokens 4096 --limit 2
```

The script uses the dataset's `prompt` column and resolves images from `IMG_ID`, including zero-padded filenames. Output is saved to `predictions.csv`, preserving dataset columns and adding `generated_code`, `inference_error`, and `used_image_path`. It saves every 25 examples and at completion or interruption. Running again replaces this file.

## Evaluation

Evaluate two predictions or the complete predictions file:

```sh
python eval.py --limit 2
python eval.py
```

The evaluator renders generated Matplotlib/Python code or direct Vega-Lite JSON, then compares the rendered chart with the task image and instruction. It reports:

| Metric | Range |
|---|---|
| Executability | 0 or 1 |
| Task accuracy | 0–5, in 0.5-point increments |
| Clarity/readability | 0–5, in 0.5-point increments |
| Visual quality | 0–5, in 0.5-point increments |
| Visual similarity | 0–100 |

An example receives `passed = 1` only when execution succeeds, task accuracy is **≥ 4.5**, clarity/readability and visual quality are each **≥ 4.0**, and visual similarity is **> 90**. Otherwise, `passed = 0`. Execution failures receive zero visual scores. Pass rate is the proportion of evaluated examples with `passed = 1`.

Defaults are judge `gpt-4o`, temperature `0.0`, maximum output tokens `1200`, and Python execution timeout `10` seconds. Results go to `evaluation.csv`; renders go to `renders/`. Override paths and judge model as needed:

```sh
python eval.py --predictions predictions.csv --output evaluation.csv --render-dir renders --judge-model gpt-4o
```

Check `execution_stderr` for rendering failures and `judge_rationale` for scoring feedback. Judge failures are recorded with zero visual scores. Running again replaces the evaluation output.

## VisEditAgent

Test one task:

```sh
python viseditagent.py --limit 1
```

The agent plans the edit, generates five intent-specific candidates, executes and renders each, and selects the strongest using a weighted judge score. It refines that candidate once with execution and visual feedback, then retains the refined candidate if its score is equal to or higher than the original.

Selection weights are task accuracy **35%**, clarity/readability **20%**, visual quality **20%**, and visual similarity **25%**, with all metrics normalized to 100. Defaults are agent model `gpt-5-mini`, judge `gpt-4o`, five candidates, and one refinement round.

```sh
python viseditagent.py --agent-model gpt-5-mini --candidates 5
```

Output is saved to `viseditagent_predictions_eval.csv`, with plans, candidate summaries, selected code, evaluation scores, and a single `passed` column using the same thresholds as `eval.py`. Candidate renders are saved under `renders_viseditagent/`.

Resume is enabled by default and skips IDs already present in the output, including failed rows. Use `--no-resume` to start fresh when changing models or settings. Use `--no-refine` to disable refinement. Agent temperature and top-p are omitted unless supplied.

Evaluate the selected outputs with:

```sh
python eval.py --predictions viseditagent_predictions_eval.csv --output agent_evaluation.csv --render-dir agent_final_renders
```

## Dataset and execution notes

`dataset.csv` includes `IMG_ID`, `code`, `Lang`, `Instruction`, `intent`, `prompt`, and additional metadata. Inference requires `IMG_ID` and `prompt`; the agent requires `IMG_ID`, `code`, `Instruction`, and `intent`. Both retain task metadata in their outputs.

Evaluation accepts the inference columns `generated_code` and `used_image_path`, or the equivalent `pred_code` and `image_path`. Keep image files accessible at the paths saved by inference.

Evaluation and the agent execute generated Python code locally. Use an isolated environment for untrusted code. The Python timeout does not cover direct Vega-Lite rendering. API calls use your account and incur provider charges.

Display all supported arguments:

```sh
python inference.py --help
python eval.py --help
python viseditagent.py --help
```

## Citation

If you use VisEditBench in your research, please cite:

```bibtex
@article{rahman2026viseditbench,
  title={VisEditBench: Can Vision-Language Models Edit Visualization Code from Multimodal Feedback?},
  author={Rahman, Mizanur and Azimlu, Arshia and Rahman, Shadikur and Laskar, Md Tahmid Rahman and Bhuiyan, Amran and Joty, Shafiq and Prince, Enamul Hoque},
  journal={arXiv preprint arXiv:2608.10408},
  year={2026}
}
```

## Contact

For questions, contact [mizanurr@yorku.ca](mailto:mizanurr@yorku.ca).
