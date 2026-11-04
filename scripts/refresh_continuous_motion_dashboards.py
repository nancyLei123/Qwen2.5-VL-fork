#!/usr/bin/env python3
"""Inject continuous motion results into existing dashboards without rebuilding video."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from build_downward_dual_view_dashboards import (
    CANONICAL_TRIAL_ENTRY_NAME,
    RUNS_DIR,
    STRIDE_REVIEW_DASHBOARD_NAME,
    TRANSITION_REVIEW_DASHBOARD_NAME,
    TRANSITION_REVIEW_TRIAL_EXCLUDE,
    build_dashboard_html,
    build_stride_review_html,
    build_time_spent_records_from_segments,
    build_transition_review_html,
    load_continuous_motion_data,
    load_transition_state_segments,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        nargs="*",
        help="Trial ids. Defaults to all dashboards with continuous output.",
    )
    return parser.parse_args()


def discover_trials() -> list[str]:
    return [
        path.parents[1].name
        for path in sorted(RUNS_DIR.glob("*/raw_parameter_diagnostics/continuous_motion_segments.json"))
        if path.parents[1].name != "Ntrial10"
    ]


def refresh_trial(trial_id: str) -> Path | None:
    run_dir = RUNS_DIR / trial_id
    manifest_path = run_dir / "session_dashboard_manifest.json"
    if not manifest_path.exists():
        print(f"{trial_id}: skipped; session dashboard manifest is missing", flush=True)
        return None

    import json

    manifest = json.loads(manifest_path.read_text())
    duration_s = float(manifest.get("session_duration_s") or 0.0)
    continuous_motion = load_continuous_motion_data(trial_id, duration_s)
    if not continuous_motion["available"]:
        print(f"{trial_id}: skipped; valid full-duration continuous timeline is missing", flush=True)
        return None

    manifest["continuous_model_segments"] = continuous_motion["segments"]
    manifest["continuous_time_spent_records"] = build_time_spent_records_from_segments(
        continuous_motion["segments"]
    )
    manifest["continuous_motion"] = {
        key: value for key, value in continuous_motion.items() if key != "segments"
    }
    manual_segments, manual_source = load_transition_state_segments(trial_id, duration_s)
    manifest["manual_timeline_segments"] = manual_segments
    manifest["manual_timeline_source"] = manual_source
    save_json(manifest_path, manifest)
    output_path = run_dir / CANONICAL_TRIAL_ENTRY_NAME
    output_path.write_text(build_dashboard_html(manifest))
    (run_dir / STRIDE_REVIEW_DASHBOARD_NAME).write_text(build_stride_review_html(manifest))
    transition_review = manifest.get("transition_review") or {}
    if transition_review.get("available") and trial_id not in TRANSITION_REVIEW_TRIAL_EXCLUDE:
        (run_dir / TRANSITION_REVIEW_DASHBOARD_NAME).write_text(build_transition_review_html(manifest))
    print(f"{trial_id}: refreshed continuous session dashboard", flush=True)
    return output_path


def main() -> None:
    args = parse_args()
    trials = args.trials or discover_trials()
    refreshed = [trial_id for trial_id in trials if refresh_trial(trial_id) is not None]
    if refreshed:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "build_full_session_manual_labeling.py"),
                "--trials",
                *refreshed,
            ],
            check=True,
        )
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "build_pd_insighter_trial_selector.py")],
        check=True,
    )
    print(f"Refreshed {len(refreshed)} continuous dashboard(s).")


if __name__ == "__main__":
    main()
