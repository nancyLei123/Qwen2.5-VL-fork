#!/usr/bin/env python3
"""Refresh transition/stride signal data and review HTML without rebuilding media."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_downward_dual_view_dashboards import (
    CANONICAL_TRIAL_ENTRY_NAME,
    RUNS_DIR,
    STRIDE_REVIEW_DASHBOARD_NAME,
    TRANSITION_REVIEW_DASHBOARD_NAME,
    build_dashboard_html,
    build_stride_review_html,
    build_stride_signal_data,
    build_transition_review_data,
    build_transition_review_html,
    estimate_alignment_offset_s,
    load_json,
    save_json,
)


TRAJECTORY_COLUMNS = {
    "tracking_timestamp_us",
    "tx_world_device",
    "ty_world_device",
    "tz_world_device",
    "device_linear_velocity_x_device",
    "device_linear_velocity_y_device",
    "device_linear_velocity_z_device",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", nargs="+", required=True)
    return parser.parse_args()


def existing_manifest_path(manifest: dict, trial_id: str, key: str, pattern: str) -> Path | None:
    raw_value = str(manifest.get("mps_outputs", {}).get(key, {}).get("path") or "").strip()
    if raw_value:
        raw_path = Path(raw_value)
        if raw_path.exists():
            return raw_path
    trial_dir = Path(str(manifest.get("raw_data", {}).get("trial_dir", {}).get("path") or ""))
    matches = sorted(trial_dir.glob(pattern)) if trial_dir.exists() else []
    if matches:
        return matches[0]
    print(f"{trial_id}: missing {key}")
    return None


def refresh_trial(trial_id: str) -> bool:
    run_dir = RUNS_DIR / trial_id
    payload_path = run_dir / "session_dashboard_manifest.json"
    trial_manifest_path = run_dir / "trial_manifest.json"
    sync_path = run_dir / "downward_sync" / "sync_suggestion.json"
    if not all(path.exists() for path in [payload_path, trial_manifest_path, sync_path]):
        print(f"{trial_id}: missing dashboard, trial, or sync manifest")
        return False

    payload = load_json(payload_path)
    trial_manifest = load_json(trial_manifest_path)
    sync_data = load_json(sync_path)
    trajectory_path = existing_manifest_path(
        trial_manifest,
        trial_id,
        "closed_loop_trajectory_csv",
        "mps_*/slam/closed_loop_trajectory.csv",
    )
    if trajectory_path is None:
        return False
    eye_gaze_path = existing_manifest_path(
        trial_manifest,
        trial_id,
        "eye_gaze_csv",
        "mps_*/eye_gaze/general_eye_gaze.csv",
    )

    print(f"{trial_id}: reading SLAM trajectory {trajectory_path}")
    trajectory_df = pd.read_csv(
        trajectory_path,
        usecols=lambda column: column in TRAJECTORY_COLUMNS,
    )
    alignment_offset_s = estimate_alignment_offset_s(
        trajectory_df,
        eye_gaze_path if eye_gaze_path and eye_gaze_path.exists() else None,
    )
    session_alignment_offset_s = alignment_offset_s - float(sync_data.get("aria_trim_start_s", 0.0))
    if payload.get("session_start_tracking_us") is not None:
        session_alignment_offset_s = (
            float(trajectory_df["tracking_timestamp_us"].iloc[0]) - float(payload["session_start_tracking_us"])
        ) / 1e6
    clip_df = pd.DataFrame(payload.get("clip_records") or [])
    transition_review = build_transition_review_data(
        trial_id,
        clip_df,
        trajectory_df,
        session_alignment_offset_s,
        float(payload.get("session_duration_s") or 0.0),
    )
    payload["transition_review"] = transition_review
    payload["stride_signal"] = build_stride_signal_data(
        trial_id,
        trajectory_df,
        session_alignment_offset_s,
        float(payload.get("session_duration_s") or 0.0),
    )
    if transition_review.get("available"):
        payload["transition_review_href"] = TRANSITION_REVIEW_DASHBOARD_NAME

    save_json(payload_path, payload)
    (run_dir / STRIDE_REVIEW_DASHBOARD_NAME).write_text(build_stride_review_html(payload))
    (run_dir / TRANSITION_REVIEW_DASHBOARD_NAME).write_text(build_transition_review_html(payload))
    (run_dir / CANONICAL_TRIAL_ENTRY_NAME).write_text(build_dashboard_html(payload))
    counts = transition_review.get("summary_counts") or {}
    print(
        f"{trial_id}: refreshed {counts.get('total', 0)} transitions "
        f"({counts.get('common', 0)} common, {counts.get('less_likely', 0)} less likely, "
        f"{counts.get('needs_review', 0)} needs review)"
    )
    return True


def main() -> None:
    args = parse_args()
    refreshed = sum(refresh_trial(trial_id) for trial_id in args.trials)
    print(f"Refreshed {refreshed} transition/stride signal page set(s).")


if __name__ == "__main__":
    main()
