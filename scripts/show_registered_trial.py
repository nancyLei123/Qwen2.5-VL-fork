#!/usr/bin/env python3
"""Print the important paths for a registered trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-id", required=True, help="Trial id, e.g. trial2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(__file__).resolve().parents[1] / "runs" / args.trial_id / "trial_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())

    print(f"Trial: {manifest['trial_id']}")
    print(f"Recording stem: {manifest['recording_stem']}")
    print(f"VRS: {manifest['raw_data']['vrs']['path']}")
    print(f"VRS JSON: {manifest['raw_data']['vrs_json']['path']}")
    print(f"Extracted MP4: {manifest['extracted_media']['video_mp4']['path']}")
    if "audio_wav" in manifest["extracted_media"]:
        print(f"Extracted audio: {manifest['extracted_media']['audio_wav']['path']}")
    print(f"MPS dir: {manifest['mps_outputs']['mps_dir']['path']}")
    print(f"Run dir: {manifest['analysis_outputs']['run_dir']['path']}")


if __name__ == "__main__":
    main()
