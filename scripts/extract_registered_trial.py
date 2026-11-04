#!/usr/bin/env python3
"""Extract the RGB MP4 for a registered trial using its manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


GEN1_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-id", required=True, help="Trial id, e.g. trial2")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        help="Optional explicit path to a trial_manifest.json file.",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Only extract the companion audio track, without rewriting the MP4.",
    )
    parser.add_argument(
        "--video-only",
        action="store_true",
        help="Only extract the MP4/thumbnail, without extracting audio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_manifest = Path(__file__).resolve().parents[1] / "runs" / args.trial_id / "trial_manifest.json"
    manifest_path = (args.manifest_path or default_manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    vrs_path = Path(manifest["raw_data"]["vrs"]["path"])
    vrs_repo = Path(manifest["repos"]["vrs_pre_processing"])
    extractor = vrs_repo / "scripts" / "video_extractor.py"

    # Explicit roots also support several recordings sharing one raw-data folder.
    cmd = [sys.executable, str(extractor), "--vrs_file", str(vrs_path),
           "--input_dir", str(vrs_path.resolve().parent),
           "--output_root", str(manifest["extracted_media"]["trial_dir"]["path"]),
           "--devignette", "false"]
    if args.audio_only and args.video_only:
        raise ValueError("Use at most one of --audio-only or --video-only.")
    if args.audio_only:
        cmd.extend(["--extract-video", "false", "--extract-audio", "true"])
    elif args.video_only:
        cmd.extend(["--extract-video", "true", "--extract-audio", "false"])
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=vrs_repo)


if __name__ == "__main__":
    main()
