#!/usr/bin/env python3
"""Refresh existing PD-Insighter dashboards from saved trial outputs.

This avoids re-running the full clip extraction + plotting pipeline when we only
want to refresh the HTML wrapper, such as after changing labels, titles, or UI.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from process_aria_trial import (
    create_clip_audio_files,
    create_clip_media_files,
    discover_audio_companion,
    write_interactive_dashboard,
)


QWEN_REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = QWEN_REPO / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        nargs="*",
        help="Optional subset of trial ids to refresh, e.g. trial1 trial6 trial7.",
    )
    return parser.parse_args()


def iter_trial_dirs(trials: list[str] | None) -> list[Path]:
    if trials:
        return [RUNS_DIR / trial_id for trial_id in trials]
    return sorted(path.parent for path in RUNS_DIR.glob("*/trial_manifest.json"))


def refresh_trial(run_dir: Path) -> bool:
    manifest_path = run_dir / "trial_manifest.json"
    clip_metrics_path = run_dir / "clip_metrics.csv"
    rule_segments_path = run_dir / "rule_segments.csv"
    gif_dir = run_dir / "clip_gifs"
    clip_audio_dir = run_dir / "clip_audio"
    clip_media_dir = run_dir / "clip_media"
    output_html = run_dir / "pd_insighter_rule_interactive.html"

    if not (manifest_path.exists() and clip_metrics_path.exists() and rule_segments_path.exists() and gif_dir.exists()):
        return False

    manifest = json.loads(manifest_path.read_text())
    clip_df = pd.read_csv(clip_metrics_path)
    segments_df = pd.read_csv(rule_segments_path)
    extracted_media = manifest.get("extracted_media", {})
    video_path_value = extracted_media.get("video_mp4", {}).get("path")
    audio_path = None
    audio_offset_s = 0.0
    clip_audio_map = {}
    if video_path_value:
        audio_path, audio_offset_s = discover_audio_companion(Path(str(video_path_value)))
        if audio_path:
            clip_audio_map = create_clip_audio_files(clip_df, clip_audio_dir, audio_path, audio_offset_s)
    clip_media_map = create_clip_media_files(clip_df, clip_media_dir, clip_audio_map)
    write_interactive_dashboard(
        clip_df,
        segments_df,
        output_html,
        gif_dir,
        trial_id=str(manifest.get("trial_id", run_dir.name)),
        recording_stem=str(manifest.get("recording_stem", run_dir.name)),
        video_path=Path(str(video_path_value)) if video_path_value else None,
        audio_path=audio_path,
        audio_offset_s=audio_offset_s,
        clip_audio_map=clip_audio_map,
        clip_media_map=clip_media_map,
    )
    print(f"Refreshed dashboard: {output_html}")
    return True


def main() -> None:
    args = parse_args()
    refreshed = 0
    for run_dir in iter_trial_dirs(args.trials):
        if refresh_trial(run_dir):
            refreshed += 1

    print(f"Refreshed {refreshed} dashboard(s).")
    selector_cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "build_pd_insighter_trial_selector.py"),
    ]
    subprocess.run(selector_cmd, check=True)


if __name__ == "__main__":
    main()
