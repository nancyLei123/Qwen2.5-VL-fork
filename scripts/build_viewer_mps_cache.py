#!/usr/bin/env python3
"""Build reusable Rerun cache files for one or more trials."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from open_viewer_mps_trials import (
    cache_ready_marker_for_trial,
    cache_path_for_trial,
    discover_trials,
    resolve_trial_resources,
)


GEN1_ROOT = Path(__file__).resolve().parents[2]
QWEN_REPO = GEN1_ROOT / "Qwen2.5-VL-fork"
PROJECTARIA_PYTHON = GEN1_ROOT / "projectaria_tools" / ".venv" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        nargs="*",
        help="Optional subset of trials, e.g. trial1 trial6 trial7.",
    )
    parser.add_argument(
        "--include-eye-gaze",
        action="store_true",
        help="Include general eye gaze in the cached viewer recording when available.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild caches even if the .rrd file already exists.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep building later trials even if one cache build fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be built without writing cache files.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--vrs", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--trajectory", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--points", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--eye-gaze", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--output-rrd", type=str, help=argparse.SUPPRESS)
    return parser.parse_args()

def cache_is_ready(trial_id: str, include_eye_gaze: bool) -> bool:
    cache_path = cache_path_for_trial(trial_id, include_eye_gaze)
    ready_marker = cache_ready_marker_for_trial(trial_id, include_eye_gaze)
    return cache_path.exists() and ready_marker.exists()


def run_worker(args: argparse.Namespace) -> int:
    import rerun as rr

    from projectaria_tools.tools.viewer_mps.rerun_viewer_mps import log_mps_to_rerun

    output_rrd = Path(args.output_rrd)
    output_rrd.parent.mkdir(parents=True, exist_ok=True)
    temp_rrd = output_rrd.with_suffix(output_rrd.suffix + ".part")
    ready_marker = Path(str(output_rrd) + ".ready.json")
    if temp_rrd.exists():
        temp_rrd.unlink()
    if ready_marker.exists():
        ready_marker.unlink()

    try:
        rr.init("MPS Data Viewer", spawn=False)
        rr.save(temp_rrd)

        eye_gaze_file = args.eye_gaze or None
        log_mps_to_rerun(
            vrs_path=args.vrs,
            trajectory_files=[args.trajectory],
            points_files=[args.points],
            eye_gaze_file=eye_gaze_file,
            wrist_and_palm_poses_file=None,
            hand_tracking_results_file=None,
            should_rectify_image=True,
            down_sampling_factor=4,
            jpeg_quality=75,
            rrd_output_path="",
        )
        rr.rerun_shutdown()
        os.replace(temp_rrd, output_rrd)
        ready_marker.write_text(
            json.dumps(
                {
                    "built_at": datetime.now(timezone.utc).isoformat(),
                    "vrs": args.vrs,
                    "trajectory": args.trajectory,
                    "points": args.points,
                    "eye_gaze": args.eye_gaze or None,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        if temp_rrd.exists():
            temp_rrd.unlink()
    print(f"Saved cache: {output_rrd}")
    return 0


def build_worker_command(
    trial_id: str,
    include_eye_gaze: bool,
    output_rrd: Path,
) -> list[str] | None:
    resources = resolve_trial_resources(trial_id)
    if resources is None:
        return None

    cmd = [
        str(PROJECTARIA_PYTHON),
        str(Path(__file__).resolve()),
        "--worker",
        "--vrs",
        str(resources["vrs"]),
        "--trajectory",
        str(resources["trajectory"]),
        "--points",
        str(resources["points"]),
        "--output-rrd",
        str(output_rrd),
    ]
    if include_eye_gaze and resources["eye_gaze"].exists():
        cmd.extend(["--eye-gaze", str(resources["eye_gaze"])])
    return cmd


def main() -> int:
    args = parse_args()
    if args.worker:
        return run_worker(args)

    built = 0
    skipped: list[str] = []
    for trial_id in discover_trials(args.trials):
        output_rrd = cache_path_for_trial(trial_id, include_eye_gaze=args.include_eye_gaze)
        if cache_is_ready(trial_id, include_eye_gaze=args.include_eye_gaze) and not args.force_rebuild:
            print(f"Skipping {trial_id}: cache already exists at {output_rrd}")
            skipped.append(trial_id)
            continue

        cmd = build_worker_command(
            trial_id,
            include_eye_gaze=args.include_eye_gaze,
            output_rrd=output_rrd,
        )
        if cmd is None:
            print(f"Skipping {trial_id}: usable VRS/SLAM inputs not found")
            skipped.append(trial_id)
            continue

        print(f"\n=== {trial_id} ===")
        print("Building cache:")
        print(" ".join(cmd))
        if args.dry_run:
            continue

        result = subprocess.run(cmd)
        if result.returncode != 0:
            if not args.continue_on_error:
                return result.returncode
            print(f"Cache build failed for {trial_id} with exit code {result.returncode}")
            continue
        built += 1

    print(f"\nBuilt {built} viewer cache file(s).")
    if skipped:
        print("Skipped:")
        for trial_id in skipped:
            print(f"  - {trial_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
