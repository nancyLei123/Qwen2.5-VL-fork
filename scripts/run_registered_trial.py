#!/usr/bin/env python3
"""Run clip-level analysis for a trial using its trial_manifest.json."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-id", required=True, help="Trial id, e.g. trial1")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        help="Optional explicit path to a trial_manifest.json file.",
    )
    parser.add_argument(
        "--qwen-predictions-csv",
        type=Path,
        help="Optional clip_idx,qwen_prediction CSV for hybrid fusion.",
    )
    parser.add_argument(
        "--hybrid-override-labels",
        nargs="*",
        default=["sitting"],
        help="Labels for which Qwen predictions override the rule-based prediction.",
    )
    return parser.parse_args()


def required_path(section: dict[str, object], key: str) -> Path:
    record = section[key]
    return Path(str(record["path"]))


def main() -> None:
    args = parse_args()
    default_manifest = (
        Path(__file__).resolve().parents[1] / "runs" / args.trial_id / "trial_manifest.json"
    )
    manifest_path = (args.manifest_path or default_manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    trial_id = str(manifest.get("trial_id", args.trial_id))
    recording_stem = str(manifest.get("recording_stem", ""))
    raw = manifest["raw_data"]
    mps = manifest["mps_outputs"]
    extracted = manifest["extracted_media"]
    analysis = manifest["analysis_outputs"]

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "process_aria_trial.py"),
        "--video-path",
        str(required_path(extracted, "video_mp4")),
        "--vrs-json-path",
        str(required_path(raw, "vrs_json")),
        "--trajectory-csv",
        str(required_path(mps, "closed_loop_trajectory_csv")),
        "--points-csv",
        str(required_path(mps, "semidense_points_csv_gz")),
        "--eye-gaze-csv",
        str(required_path(mps, "eye_gaze_csv")),
        "--output-dir",
        str(required_path(analysis, "run_dir")),
        "--trial-id",
        trial_id,
        "--recording-stem",
        recording_stem,
    ]

    if args.qwen_predictions_csv:
        cmd.extend(["--qwen-predictions-csv", str(args.qwen_predictions_csv.resolve())])
        if args.hybrid_override_labels:
            cmd.extend(["--hybrid-override-labels", *args.hybrid_override_labels])

    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    selector_cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "build_pd_insighter_trial_selector.py"),
    ]
    print("Running:")
    print(" ".join(selector_cmd))
    subprocess.run(selector_cmd, check=True)


if __name__ == "__main__":
    main()
