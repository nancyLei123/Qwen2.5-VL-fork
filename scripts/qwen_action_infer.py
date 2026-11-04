#!/usr/bin/env python3
"""Run Qwen video inference on prepared 2-second clips.

This expects the outputs of `process_aria_trial.py`, especially `clip_metrics.csv`.
It keeps the action set and zero-shot prompt close to the thesis setup, while still
being flexible enough to swap in different checkpoints later.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QWEN_UTILS_SRC = ROOT / "qwen-vl-utils" / "src"
if str(QWEN_UTILS_SRC) not in sys.path:
    sys.path.insert(0, str(QWEN_UTILS_SRC))

from qwen_vl_utils import process_vision_info  # type: ignore  # noqa: E402
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # type: ignore  # noqa: E402


ACTIONS = ["sitting", "standing", "walking", "go upstairs", "go downstairs"]
THESIS_RULES_TEXT = (
    "If 0.15 < vertical_displacement < 0.2 and horizontal_displacement < 1.45: "
    "go upstairs. If vertical_displacement < -0.15 and 0.2 < horizontal_displacement < 1.45: "
    "go downstairs. If horizontal_displacement > 0.5 and mean_speed > 0.5: walking. "
    "If mean_speed < 0.15 and horizontal_displacement < 0.25: sitting. Else: standing."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-csv", required=True, help="clip_metrics.csv from process_aria_trial.py")
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Hugging Face model id or local checkpoint path.",
    )
    parser.add_argument("--output-csv", required=True, help="Where to write qwen predictions.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--use-slam-prompt",
        action="store_true",
        help="Inject the SLAM summary and thesis rule text into the prompt.",
    )
    return parser.parse_args()


def normalize_prediction(text: str) -> str:
    lowered = text.strip().lower()
    for action in ACTIONS:
        if action in lowered:
            return action
    cleaned = re.sub(r"[^a-z ]", "", lowered).strip()
    if cleaned in ACTIONS:
        return cleaned
    return lowered


def build_prompt(row: pd.Series, use_slam_prompt: bool) -> str:
    if use_slam_prompt:
        return (
            "Analyze the video clip and classify the action being performed. "
            f"Choose ONLY ONE action from this list: {', '.join(ACTIONS)}. "
            f"Use the SLAM data {row['slam_text']} with the following criteria {THESIS_RULES_TEXT} "
            "to aid in your choice. Respond with just the action name, nothing else."
        )
    return (
        "Analyze the video clip and classify the action being performed. "
        f"Choose ONLY ONE action from this list: {', '.join(ACTIONS)}. "
        "Respond with just the action name, nothing else."
    )


def main() -> None:
    args = parse_args()
    clips_df = pd.read_csv(args.clips_csv)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    rows = []
    for row in clips_df.itertuples(index=False):
        clip_path = Path(row.clip_path)
        if not clip_path.exists():
            continue
        prompt = build_prompt(pd.Series(row._asdict()), args.use_slam_prompt)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"video": str(clip_path)},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=video_kwargs["fps"],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        rows.append(
            {
                "clip_idx": int(row.clip_idx),
                "clip_path": str(clip_path),
                "qwen_prediction": normalize_prediction(output_text),
                "raw_response": output_text,
            }
        )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote predictions to: {output_path}")


if __name__ == "__main__":
    main()
