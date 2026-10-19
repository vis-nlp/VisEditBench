"""Run VisEditBench inference. Install: pip install openai pandas tqdm"""

import argparse
import base64
import mimetypes
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset.csv"
IMAGE_DIR = ROOT / "image"
OUTPUT_CSV = ROOT / "predictions.csv"

SAVE_EVERY = 25


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VisEditBench inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="gpt-5", help="Model name.")
    parser.add_argument("--temperature", type=float, help="Sampling temperature; omitted if unset.")
    parser.add_argument("--top-p", type=float, help="Nucleus sampling; omitted if unset.")
    parser.add_argument("--max-output-tokens", type=int, default=4096, help="Output token limit.")
    parser.add_argument("--image-detail", choices=("auto", "low", "high"), default="high",
                        help="Input image detail.")
    parser.add_argument("--limit", type=int, help="Run only the first N examples.")
    args = parser.parse_args()
    if not args.model.strip():
        parser.error("--model cannot be empty")
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        parser.error("--top-p must be greater than 0 and at most 1")
    for name in ("max_output_tokens", "limit"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def find_image_path(img_id):
    name = str(img_id).strip()
    direct = IMAGE_DIR / name
    if direct.is_file():
        return direct

    stem = Path(name).stem
    stems = [stem]
    match = re.fullmatch(r"(.*?)(\d+)", stem)
    if match:
        stems.append(f"{match[1]}{int(match[2]):04d}")

    for candidate in stems:
        for extension in (".png", ".jpg", ".jpeg", ".webp"):
            path = IMAGE_DIR / f"{candidate}{extension}"
            if path.is_file():
                return path
    raise FileNotFoundError(f"Image not found for IMG_ID={name}")


def clean_code(text):
    text = text.strip()
    match = re.fullmatch(r"```[^\n`]*\n(.*?)\n?```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text


def generate_code(client, prompt, image_path, args):
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    image = {
        "type": "input_image",
        "image_url": f"data:{mime_type};base64,{encoded}",
    }
    if args.image_detail is not None:
        image["detail"] = args.image_detail

    parameters = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_output_tokens": args.max_output_tokens,
    }
    response = client.responses.create(
        model=args.model,
        instructions=(
            "You are an expert visualization code editor. "
            "Return only complete corrected code. No markdown. No explanation."
        ),
        input=[{
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}, image],
        }],
        **{key: value for key, value in parameters.items() if value is not None},
    )
    if response.status != "completed" or not response.output_text.strip():
        raise ValueError(f"No complete code response (status={response.status})")
    return clean_code(response.output_text)


def main():
    args = parse_args()
    df = pd.read_csv(DATASET, keep_default_na=False)
    if not {"IMG_ID", "prompt"}.issubset(df.columns):
        raise ValueError("dataset.csv must contain IMG_ID and prompt columns.")
    if args.limit is not None:
        df = df.head(args.limit).copy()
    for column in ("generated_code", "inference_error", "used_image_path"):
        df[column] = ""

    with OpenAI() as client:  # Reads OPENAI_API_KEY from the environment.
        try:
            for count, (idx, row) in enumerate(
                tqdm(df.iterrows(), total=len(df), desc="Running inference"), start=1
            ):
                try:
                    if not row["prompt"].strip():
                        raise ValueError("Empty prompt")
                    image_path = find_image_path(row["IMG_ID"])
                    df.at[idx, "used_image_path"] = str(image_path)
                    df.at[idx, "generated_code"] = generate_code(client, row["prompt"], image_path, args)
                except Exception as exc:
                    df.at[idx, "inference_error"] = str(exc)
                if count % SAVE_EVERY == 0:
                    df.to_csv(OUTPUT_CSV, index=False)
        finally:
            df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
