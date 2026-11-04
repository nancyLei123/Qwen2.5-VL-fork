#!/usr/bin/env python3
"""Convert local trial outputs into slam_data_analysis.ipynb-compatible JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-metrics-csv", required=True, help="Path to clip_metrics.csv")
    parser.add_argument(
        "--action-labels-csv",
        help=(
            "Optional CSV with columns video_id,action_label or clip_index,action_label. "
            "If provided, action_label will be merged for accuracy analysis."
        ),
    )
    parser.add_argument("--output-json", required=True, help="Path to write notebook-compatible JSON")
    parser.add_argument(
        "--prediction-field",
        default="rule_prediction",
        choices=["rule_prediction", "hybrid_prediction", "qwen_prediction"],
        help="Which local prediction field to export as predicted_label.",
    )
    return parser.parse_args()


def load_action_labels(labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    if {"clip_index", "action_label"}.issubset(df.columns):
        return df[["clip_index", "action_label"]].copy()
    if {"video_id", "action_label"}.issubset(df.columns):
        out = df[["video_id", "action_label"]].copy()
        out["clip_index"] = (
            out["video_id"].astype(str).str.extract(r"clip(\d+)")[0].astype(int).map(lambda x: f"clip_{x:04d}")
        )
        return out[["clip_index", "action_label"]]
    raise ValueError(
        f"{labels_csv} must contain either clip_index,action_label or video_id,action_label columns"
    )


def main() -> None:
    args = parse_args()
    clip_df = pd.read_csv(args.clip_metrics_csv)

    export_df = pd.DataFrame()
    export_df["clip_index"] = clip_df["clip_idx"].map(lambda x: f"clip_{int(x):04d}")
    export_df["clip_start_utc_ns"] = (clip_df["start_epoch_s"] * 1_000_000_000).round().astype("int64")
    export_df["clip_end_utc_ns"] = (clip_df["end_epoch_s"] * 1_000_000_000).round().astype("int64")
    export_df["slam_text_summary"] = clip_df["slam_text"]
    export_df["mean_speed_mps"] = clip_df["mean_linear_speed_mps"]
    export_df["speed_variance"] = clip_df["speed_variance"] if "speed_variance" in clip_df.columns else pd.NA
    export_df["mean_angular_speed"] = clip_df["mean_angular_speed_rps"]
    export_df["angular_speed_variance"] = (
        clip_df["angular_speed_variance"] if "angular_speed_variance" in clip_df.columns else pd.NA
    )
    export_df["vertical_displacement_m"] = clip_df["vertical_displacement_m"]
    export_df["horizontal_displacement_m"] = clip_df["horizontal_displacement_m"]
    export_df["gravity_variance"] = clip_df["gravity_variance"] if "gravity_variance" in clip_df.columns else pd.NA
    export_df["mean_z_m"] = clip_df["mean_z_m"] if "mean_z_m" in clip_df.columns else pd.NA

    if args.prediction_field == "rule_prediction":
        export_df["predicted_label"] = clip_df["rule_prediction"]
    elif args.prediction_field == "hybrid_prediction":
        export_df["predicted_label"] = clip_df["hybrid_prediction"]
    else:
        export_df["predicted_label"] = clip_df["qwen_prediction"]

    if "qwen_prediction" in clip_df.columns:
        export_df["qwen_prediction"] = clip_df["qwen_prediction"]
    if "hybrid_prediction" in clip_df.columns:
        export_df["hybrid_prediction"] = clip_df["hybrid_prediction"]

    if args.action_labels_csv:
        labels_df = load_action_labels(Path(args.action_labels_csv))
        export_df = export_df.merge(labels_df, on="clip_index", how="left")

    export_df = export_df.astype(object)
    export_df = export_df.where(pd.notnull(export_df), None)
    records = export_df.to_dict(orient="records")
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))
    print(f"Wrote notebook-compatible JSON to: {output_path}")


if __name__ == "__main__":
    main()
