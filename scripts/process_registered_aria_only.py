#!/usr/bin/env python3
"""Run existing simulated-motion analyses for a registered Aria-only trial."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import shutil

import pandas as pd

from build_downward_dual_view_dashboards import (
    build_trial, build_dashboard_html, build_stride_review_html,
    build_transition_review_html, load_stride_review_data,
)

ROOT = Path(__file__).resolve().parents[1]


def run(trial, refresh_only=False):
    directory = ROOT / "runs" / trial
    path = directory / "trial_manifest.json"
    manifest = json.loads(path.read_text())
    if (directory / "downward_sync/sync_suggestion.json").exists():
        raise ValueError("Use the existing dual-view pipeline for a paired trial.")
    manifest["media_mode"] = "aria_only"
    for group in ("raw_data", "mps_outputs", "extracted_media", "analysis_outputs"):
        for item in manifest[group].values():
            if isinstance(item, dict) and "path" in item:
                item["exists"] = Path(item["path"]).exists()
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not refresh_only and not build_trial(trial):
        raise RuntimeError(f"Dashboard prerequisites missing: {trial}")
    if not refresh_only:
        subprocess.run([sys.executable, str(ROOT / "scripts/generate_step_stride_diagnostics.py"),
                        "--trials", trial, "--out-dir", str(directory / "stride_step_diagnostics"),
                        "--no-contact-sheets"], check=True)
    session_path = directory / "session_dashboard_manifest.json"
    payload = json.loads(session_path.read_text())
    from build_parkinson_motion_review import publish_parkinson_review
    review_path = directory / "parkinson_motion_review" / f"{trial}_parkinson_review_payload.json"
    publish_parkinson_review(json.loads(review_path.read_text()))
    payload["stride_review"] = load_stride_review_data(trial, directory / "index.html")
    session_path.write_text(json.dumps(payload, indent=2) + "\n")
    for name, builder in [("index.html", build_dashboard_html),
                          ("pd_insighter_stride_review.html", build_stride_review_html),
                          ("pd_insighter_transition_review.html", build_transition_review_html)]:
        (directory / name).write_text(builder(payload))
    subprocess.run([sys.executable, str(ROOT / "scripts/build_full_session_manual_labeling.py"),
                    "--trials", trial], check=True)
    timeline = json.loads((directory / "raw_parameter_diagnostics/continuous_motion_segments.json").read_text())
    shutil.copyfile(directory / "downward_master_session/downward_session_clip_metrics.csv", directory / "clip_metrics.csv")
    (directory / "timeline_predictions.json").write_text(json.dumps(timeline["segments"], indent=2) + "\n")
    pd.DataFrame(timeline["segments"]).to_csv(directory / "rule_segments.csv", index=False)
    summary = {"trial_id": trial, "trial_type": manifest["trial_type"], "media_mode": "aria_only",
               "session_duration_s": payload["session_duration_s"], "classification_method": timeline["version"],
               "classification_validated": False, "accuracy": None,
               "classification_periods": len(timeline["segments"]),
               "step_proxy_event_count": len(payload["stride_review"].get("events", [])),
               "limitations": ["Simulated PD, not patient diagnosis.", "No paired downward video or manual action/foot-contact labels provided.",
                               "SLAM head-motion stride proxies are not independently measured foot strides.",
                               "See environment_reconstruction/reconstruction_summary.json for actual completed 3D coverage."]}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    from process_aria_trial import plot_displacement_graphs, plot_point_cloud
    plots = directory / "plots"
    plots.mkdir(exist_ok=True)
    if not (plots / "horizontal_displacement_by_clip.png").exists():
        plot_displacement_graphs(pd.read_csv(directory / "clip_metrics.csv"), plots)
    if not (plots / "point_cloud_3d.png").exists():
        plot_point_cloud(Path(manifest["mps_outputs"]["semidense_points_csv_gz"]["path"]),
                         Path(manifest["mps_outputs"]["closed_loop_trajectory_csv"]["path"]),
                         plots / "point_cloud_3d.png", 50000)
    for group in ("raw_data", "mps_outputs", "extracted_media", "analysis_outputs"):
        for item in manifest[group].values():
            if isinstance(item, dict) and "path" in item:
                item["exists"] = Path(item["path"]).exists()
    manifest["analysis_outputs"]["interactive_dashboard_html"] = {"path": str(directory / "index.html"), "exists": True}
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run([sys.executable, str(ROOT / "scripts/build_pd_insighter_trial_selector.py")], check=True)
    print(f"Completed Aria-only analysis: {trial}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", nargs="+", required=True)
    parser.add_argument("--refresh-only", action="store_true")
    args = parser.parse_args()
    for trial in args.trials:
        run(trial, args.refresh_only)
