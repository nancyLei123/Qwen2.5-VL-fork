#!/usr/bin/env python3
"""Connect saved simulated-PD results across views without recutting media or labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from build_parkinson_motion_review import publish_parkinson_review
from build_downward_dual_view_dashboards import (
    RUNS_DIR, build_dashboard_html, build_stride_review_html, build_transition_review_html,
    build_transition_review_data, build_transition_candidates, build_time_spent_records_from_segments,
    load_continuous_motion_data, load_stride_review_data, load_transition_state_segments,
    collect_ntrial_dashboard_options, color_for_label, estimate_alignment_offset_s,
)
from build_full_session_manual_labeling import build_html as manual_html, load_payload as manual_payload
from pd_insighter_environment_navigation import SIMULATED_TRIALS


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def protected_files(root: Path) -> dict[Path, str]:
    paths = [p for folder in ("manual_labels", "parkinson_motion_review", "spoken_action_labels")
             for p in (root / folder).glob("*.json")
             if "manual" in p.name or folder == "spoken_action_labels"]
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def refresh(trial: str, prepare_only=False):
    if trial not in SIMULATED_TRIALS:
        raise ValueError("This refresh is restricted to the three simulated-PD trials.")
    root = RUNS_DIR / trial
    protected = protected_files(root)
    backup = RUNS_DIR / "_simulation_review_backup_20261029" / trial
    for source in [root / "trial_manifest.json", root / "session_dashboard_manifest.json", *protected]:
        target = backup / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    manifest_path = root / "trial_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["trial_type"] = "parkinson_simulated"
    if trial == "Ntrial10":
        manifest["slam_time_alignment"] = "legacy_session_clock"
    save_json(manifest_path, manifest)
    session_path = root / "session_dashboard_manifest.json"
    session = json.loads(session_path.read_text())
    pd_payload = json.loads((root / f"parkinson_motion_review/{trial}_parkinson_review_payload.json").read_text())
    assert abs(pd_payload["duration_s"] - session["session_duration_s"]) < .01
    pd_payload["media"] = {"downward": session["downward_session_video_uri"], "aria": session["aria_session_video_uri"]}
    publish_parkinson_review(pd_payload)
    if not prepare_only:
        duration = session["session_duration_s"]
        continuous = load_continuous_motion_data(trial, duration)
        if not continuous["available"]:
            raise ValueError(f"{trial}: saved continuous classification has gaps")
        segments = [{**s, "color": color_for_label(s["label"]), "clip_start_idx": int(s["start_s"] // 2),
                     "clip_end_idx": int(max(0, s["end_s"] - 1e-6) // 2)} for s in continuous["segments"]]
        session["trial_type"] = "parkinson_simulated"
        session["continuous_model_segments"] = segments
        session["model_segments"] = segments
        session["continuous_motion"] = {k: v for k, v in continuous.items() if k != "segments"}
        session["time_spent_records"] = session["continuous_time_spent_records"] = build_time_spent_records_from_segments(segments)
        for clip in session["clip_records"]:
            best = max(segments, key=lambda s: max(0, min(clip["end_s"], s["end_s"]) - max(clip["start_s"], s["start_s"])))
            for key in ("rule_prediction", "hybrid_prediction"):
                clip[key] = best["label"]
            clip["color"] = best["color"]
            clip["model_prediction_source"] = "parkinson_high_v1"
        clips = pd.DataFrame(session["clip_records"])
        session["manual_timeline_segments"], session["manual_timeline_source"] = load_transition_state_segments(trial, duration)
        session["stride_review"] = load_stride_review_data(trial, root / "index.html")
        session["transition_candidates"] = build_transition_candidates(clips, segments, session_duration_s=duration)
        trajectory = pd.read_csv(manifest["mps_outputs"]["closed_loop_trajectory_csv"]["path"])
        if session.get("session_start_tracking_us") is not None:
            alignment = (trajectory["tracking_timestamp_us"].iloc[0] - session["session_start_tracking_us"]) / 1e6
        else:
            build = json.loads((root / "downward_master_session/aria_session_video.build.json").read_text())
            alignment = estimate_alignment_offset_s(trajectory, Path(manifest["mps_outputs"]["eye_gaze_csv"]["path"])) - build["video_trim_start_s"]
        session["transition_review"] = build_transition_review_data(trial, clips, trajectory, alignment, duration, automatic_segments=segments)
        session["transition_review_href"] = "pd_insighter_transition_review.html"
        session["parkinson_review_href"] = "pd_insighter_parkinson_review.html"
        session["trial_jump_options"] = collect_ntrial_dashboard_options(trial, root / "index.html")
        save_json(session_path, session)
        (root / "index.html").write_text(build_dashboard_html(session))
        (root / "pd_insighter_stride_review.html").write_text(build_stride_review_html(session))
        (root / "pd_insighter_transition_review.html").write_text(build_transition_review_html(session))
        (root / "pd_insighter_manual_labeling.html").write_text(manual_html(manual_payload(trial)))
    assert protected_files(root) == protected, f"{trial}: an annotation changed during refresh"
    print(f"{trial}: {'prepared saved classifications' if prepare_only else 'refreshed all five views'}; {len(protected)} annotation files unchanged", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", nargs="+", default=["Ntrial10", "Parki", "Parki2"])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    for trial in args.trials:
        refresh(trial, args.prepare_only)
