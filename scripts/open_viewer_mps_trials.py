#!/usr/bin/env python3
"""Launch cached Rerun viewer files or viewer_mps for trials with SLAM outputs."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


GEN1_ROOT = Path(__file__).resolve().parents[2]
VRS_REPO = GEN1_ROOT / "VRS_Pre_Processing"
QWEN_REPO = GEN1_ROOT / "Qwen2.5-VL-fork"
PROJECTARIA_BIN = GEN1_ROOT / "projectaria_tools" / ".venv" / "bin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        nargs="*",
        help="Optional subset of trials to launch, e.g. trial1 trial6 trial7.",
    )
    parser.add_argument(
        "--include-eye-gaze",
        action="store_true",
        help="Also pass the MPS eye gaze CSV to viewer_mps when available.",
    )
    parser.add_argument(
        "--stagger-seconds",
        type=float,
        default=1.0,
        help="Delay between launches so the viewer has time to register recordings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without launching viewer_mps.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore existing cached .rrd files and always launch viewer_mps from raw inputs.",
    )
    parser.add_argument(
        "--build-cache-missing",
        action="store_true",
        help="If a trial has no cached .rrd file yet, build it first and then open the cached file.",
    )
    parser.add_argument(
        "--force-rebuild-cache",
        action="store_true",
        help="Rebuild cached .rrd files before launching them.",
    )
    return parser.parse_args()


def trial_sort_key(trial_id: str) -> tuple[int, str]:
    suffix = trial_id.removeprefix("trial")
    try:
        return int(suffix), trial_id
    except ValueError:
        return 10**9, trial_id


def discover_trials(explicit_trials: list[str] | None) -> list[str]:
    if explicit_trials:
        return explicit_trials
    trial_ids = [path.name for path in (VRS_REPO / "data").glob("trial*") if path.is_dir()]
    return sorted(trial_ids, key=trial_sort_key)


def resolve_trial_resources(trial_id: str) -> dict[str, Path] | None:
    trial_dir = VRS_REPO / "data" / trial_id
    vrs_files = sorted(path for path in trial_dir.glob("*.vrs") if path.is_file())
    if len(vrs_files) != 1:
        return None

    vrs_path = vrs_files[0]
    recording_stem = vrs_path.name.removesuffix(".vrs")
    mps_dir = trial_dir / f"mps_{recording_stem}_vrs"
    trajectory_path = mps_dir / "slam" / "closed_loop_trajectory.csv"
    points_path = mps_dir / "slam" / "semidense_points.csv.gz"
    eye_gaze_path = mps_dir / "eye_gaze" / "general_eye_gaze.csv"

    if not trajectory_path.exists() or not points_path.exists():
        return None

    return {
        "trial_dir": trial_dir,
        "vrs": vrs_path,
        "trajectory": trajectory_path,
        "points": points_path,
        "eye_gaze": eye_gaze_path,
    }


def cache_path_for_trial(trial_id: str, include_eye_gaze: bool) -> Path:
    suffix = "_eye_gaze" if include_eye_gaze else ""
    return QWEN_REPO / "runs" / trial_id / f"viewer_mps_cache{suffix}.rrd"


def cache_ready_marker_for_trial(trial_id: str, include_eye_gaze: bool) -> Path:
    return Path(str(cache_path_for_trial(trial_id, include_eye_gaze)) + ".ready.json")


def cache_is_ready(trial_id: str, include_eye_gaze: bool) -> bool:
    cache_path = cache_path_for_trial(trial_id, include_eye_gaze)
    ready_marker = cache_ready_marker_for_trial(trial_id, include_eye_gaze)
    return cache_path.exists() and ready_marker.exists()


def build_command(resources: dict[str, Path], include_eye_gaze: bool) -> list[str]:
    cmd = [
        str(PROJECTARIA_BIN / "viewer_mps"),
        "--vrs",
        str(resources["vrs"]),
        "--trajectory",
        str(resources["trajectory"]),
        "--points",
        str(resources["points"]),
    ]
    if include_eye_gaze and resources["eye_gaze"].exists():
        cmd.extend(["--eyegaze", str(resources["eye_gaze"])])
    return cmd


def build_cache(
    resources: dict[str, Path],
    trial_id: str,
    include_eye_gaze: bool,
    force_rebuild: bool,
) -> None:
    cmd = [
        str(PROJECTARIA_BIN / "python"),
        str(QWEN_REPO / "scripts" / "build_viewer_mps_cache.py"),
        "--trials",
        trial_id,
    ]
    if include_eye_gaze:
        cmd.append("--include-eye-gaze")
    if force_rebuild:
        cmd.append("--force-rebuild")
    print("Building cache:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=GEN1_ROOT)


def build_cached_launch_command(trial_id: str, include_eye_gaze: bool) -> list[str]:
    return [str(PROJECTARIA_BIN / "rerun"), str(cache_path_for_trial(trial_id, include_eye_gaze))]


def main() -> None:
    args = parse_args()

    env = os.environ.copy()
    env["PATH"] = f"{PROJECTARIA_BIN}:{env.get('PATH', '')}"

    launched = 0
    skipped: list[str] = []
    for trial_id in discover_trials(args.trials):
        resources = resolve_trial_resources(trial_id)
        if resources is None:
            skipped.append(trial_id)
            continue

        cache_path = cache_path_for_trial(trial_id, include_eye_gaze=args.include_eye_gaze)
        should_use_cache = not args.no_cache
        if should_use_cache and (
            args.force_rebuild_cache
            or (args.build_cache_missing and not cache_is_ready(trial_id, include_eye_gaze=args.include_eye_gaze))
        ):
            if args.dry_run:
                print(f"\n=== {trial_id} ===")
                print(f"Would build cache: {cache_path}")
            else:
                build_cache(
                    resources,
                    trial_id,
                    include_eye_gaze=args.include_eye_gaze,
                    force_rebuild=args.force_rebuild_cache,
                )

        if should_use_cache and cache_is_ready(trial_id, include_eye_gaze=args.include_eye_gaze):
            cmd = build_cached_launch_command(trial_id, include_eye_gaze=args.include_eye_gaze)
            launch_mode = "cached .rrd"
        else:
            cmd = build_command(resources, include_eye_gaze=args.include_eye_gaze)
            launch_mode = "raw viewer_mps"
        print(f"\n=== {trial_id} ===")
        print(f"Launching ({launch_mode}):")
        print(" ".join(cmd))

        if not args.dry_run:
            proc = subprocess.Popen(
                cmd,
                cwd=GEN1_ROOT,
                env=env,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"PID: {proc.pid}")
            launched += 1
            if args.stagger_seconds > 0:
                time.sleep(args.stagger_seconds)

    print(f"\nLaunched {launched} viewer_mps process(es).")
    if skipped:
        print("Skipped trials (missing usable SLAM inputs):")
        for trial_id in skipped:
            print(f"  - {trial_id}")


if __name__ == "__main__":
    main()
