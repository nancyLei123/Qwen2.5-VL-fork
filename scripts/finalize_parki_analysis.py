#!/usr/bin/env python3
"""Export supplementary local results for the two simulated-PD recordings."""
import json
from pathlib import Path

import pandas as pd

from process_aria_trial import plot_displacement_graphs, plot_point_cloud

RUNS = Path(__file__).resolve().parents[1] / "runs"


def main():
    for trial in ("Parki", "Parki2"):
        root = RUNS / trial
        manifest_path = root / "trial_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["trial_type"] == "parkinson_simulated"
        session = json.loads((root / "session_dashboard_manifest.json").read_text())
        timeline = json.loads((root / "raw_parameter_diagnostics/continuous_motion_segments.json").read_text())
        sync = json.loads((root / "downward_sync/sync_suggestion.json").read_text())
        verified = json.loads((root / "downward_sync/sync_verification.json").read_text())
        metrics = pd.read_csv(root / "downward_master_session/downward_session_clip_metrics.csv")
        # Summary windows are retained for compatibility; classification boundaries remain continuous.
        metrics.to_csv(root / "clip_metrics.csv", index=False)
        segments = pd.DataFrame(timeline["segments"])
        segments.to_csv(root / "rule_segments.csv", index=False)
        (root / "timeline_predictions.json").write_text(json.dumps(timeline["segments"], indent=2) + "\n")
        plots = root / "plots"
        plots.mkdir(exist_ok=True)
        plot_displacement_graphs(metrics, plots)
        cloud = plots / "point_cloud_trajectory.png"
        if not cloud.exists():
            plot_point_cloud(Path(manifest["mps_outputs"]["semidense_points_csv_gz"]["path"]),
                             Path(manifest["mps_outputs"]["closed_loop_trajectory_csv"]["path"]), cloud, 100000)
        events = pd.read_csv(root / "stride_step_diagnostics/data/step_stride_event_estimates.csv")
        summary = {
            "trial_id": trial, "trial_type": "parkinson_simulated",
            "classification_method": timeline["version"], "classification_validated": False,
            "session_duration_s": float(segments["end_s"].max()),
            "classification_periods": len(segments),
            "automatic_label_duration_s": segments.groupby("label")["duration_s"].sum().to_dict(),
            "video_offset_s": sync["estimated_offset_s"],
            "waveform_sync_verified": verified["valid"],
            "estimated_audio_drift_s": verified["estimated_drift_over_overlap_s"],
            "step_proxy_event_count": len(events),
            "time_origin": "Synchronized video start; Aria device timestamps align SLAM to video.",
            "accuracy": None,
            "limitations": ["No manual action/foot-contact validation yet.",
                            "Simulated Parkinson behavior, not patient diagnosis.",
                            "Stair and posture labels require review; head motion can confuse the existing classifier.",
                            "No photorealistic 3D reconstruction or new MPS submission performed."],
        }
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        for group in ("raw_data", "mps_outputs", "extracted_media", "analysis_outputs"):
            for entry in manifest.get(group, {}).values():
                if isinstance(entry, dict) and "path" in entry:
                    entry["exists"] = Path(entry["path"]).exists()
        manifest["analysis_outputs"]["interactive_dashboard_html"] = {"path": str(root / "index.html"), "exists": True}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
