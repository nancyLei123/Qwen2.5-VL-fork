#!/usr/bin/env python3
"""Create or refresh a standard multi-repo manifest for one Aria trial.

This keeps the workspace organized across the two repos:
- VRS_Pre_Processing: raw VRS, MPS outputs, extracted media
- Qwen2.5-VL-fork: clip-level analysis, plots, dashboards, model outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GEN1_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VRS_REPO = GEN1_ROOT / "VRS_Pre_Processing"
DEFAULT_QWEN_REPO = GEN1_ROOT / "Qwen2.5-VL-fork"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-id", required=True, help="Stable trial folder name, e.g. trial2")
    parser.add_argument("--data-dir", type=Path, help="Existing raw/MPS directory; leave source recordings in place.")
    parser.add_argument("--trial-type", choices=["normal", "parkinson_simulated", "unspecified"], default="unspecified")
    parser.add_argument(
        "--recording-stem",
        required=True,
        help=(
            "Recording basename without extension, e.g. 2_Walking_Stairs. "
            "The script will derive .vrs, .vrs.json, and MPS folder names from it."
        ),
    )
    parser.add_argument(
        "--vrs-repo-root",
        type=Path,
        default=DEFAULT_VRS_REPO,
        help="Path to the VRS_Pre_Processing repo.",
    )
    parser.add_argument(
        "--qwen-repo-root",
        type=Path,
        default=DEFAULT_QWEN_REPO,
        help="Path to the Qwen2.5-VL-fork repo.",
    )
    parser.add_argument(
        "--extract-flags",
        default="undistort_True_colorcorrect_True_devignette_False",
        help="Expected extractor flag folder for the extracted video location.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def entry(path: Path) -> dict[str, object]:
    return {"path": str(path), "exists": path.exists()}


def main() -> None:
    args = parse_args()
    trial_id = args.trial_id
    recording_stem = args.recording_stem
    vrs_repo = args.vrs_repo_root.resolve()
    qwen_repo = args.qwen_repo_root.resolve()

    trial_data_dir = ensure_dir(args.data_dir.resolve() if args.data_dir else vrs_repo / "data" / trial_id)
    extracted_trial_dir = ensure_dir(vrs_repo / "data" / "extracted_videos" / trial_id)
    run_dir = ensure_dir(qwen_repo / "runs" / trial_id)
    ensure_dir(run_dir / "clips")
    ensure_dir(run_dir / "clip_gifs")
    ensure_dir(run_dir / "clip_audio")
    ensure_dir(run_dir / "plots")
    ensure_dir(run_dir / "labels")
    ensure_dir(run_dir / "model_results")

    recording_filename = f"{recording_stem}.vrs"
    vrs_path = trial_data_dir / recording_filename
    vrs_json_path = trial_data_dir / f"{recording_filename}.json"
    mps_dir = trial_data_dir / f"mps_{recording_stem}_vrs"

    extracted_video_dir = extracted_trial_dir / args.extract_flags / recording_filename
    extracted_video_path = extracted_video_dir / f"{recording_filename}.mp4"
    extracted_thumbnail_path = extracted_video_dir / f"{recording_filename}.png"
    extracted_metadata_path = extracted_video_dir / "metadata.json"
    extracted_audio_path = extracted_video_dir / "audio_track.wav"
    extracted_audio_metadata_path = extracted_video_dir / "audio_metadata.json"

    manifest = {
        "trial_id": trial_id,
        "trial_type": args.trial_type,
        "recording_stem": recording_stem,
        "workspace_root": str(GEN1_ROOT),
        "repos": {
            "vrs_pre_processing": str(vrs_repo),
            "qwen_repo": str(qwen_repo),
        },
        "raw_data": {
            "trial_dir": entry(trial_data_dir),
            "vrs": entry(vrs_path),
            "vrs_json": entry(vrs_json_path),
        },
        "mps_outputs": {
            "mps_dir": entry(mps_dir),
            "health_check_json": entry(mps_dir / "vrs_health_check.json"),
            "eye_gaze_csv": entry(mps_dir / "eye_gaze" / "general_eye_gaze.csv"),
            "hand_tracking_csv": entry(mps_dir / "hand_tracking" / "hand_tracking_results.csv"),
            "closed_loop_trajectory_csv": entry(mps_dir / "slam" / "closed_loop_trajectory.csv"),
            "open_loop_trajectory_csv": entry(mps_dir / "slam" / "open_loop_trajectory.csv"),
            "semidense_points_csv_gz": entry(mps_dir / "slam" / "semidense_points.csv.gz"),
            "semidense_observations_csv_gz": entry(mps_dir / "slam" / "semidense_observations.csv.gz"),
        },
        "extracted_media": {
            "trial_dir": entry(extracted_trial_dir),
            "video_dir": entry(extracted_video_dir),
            "video_mp4": entry(extracted_video_path),
            "thumbnail_png": entry(extracted_thumbnail_path),
            "metadata_json": entry(extracted_metadata_path),
            "audio_wav": entry(extracted_audio_path),
            "audio_metadata_json": entry(extracted_audio_metadata_path),
        },
        "analysis_outputs": {
            "run_dir": entry(run_dir),
            "clips_dir": entry(run_dir / "clips"),
            "clip_gifs_dir": entry(run_dir / "clip_gifs"),
            "clip_audio_dir": entry(run_dir / "clip_audio"),
            "plots_dir": entry(run_dir / "plots"),
            "labels_dir": entry(run_dir / "labels"),
            "model_results_dir": entry(run_dir / "model_results"),
            "clip_metrics_csv": entry(run_dir / "clip_metrics.csv"),
            "qwen_manifest_json": entry(run_dir / "qwen_manifest.json"),
            "timeline_predictions_json": entry(run_dir / "timeline_predictions.json"),
            "rule_segments_csv": entry(run_dir / "rule_segments.csv"),
            "summary_json": entry(run_dir / "summary.json"),
            "notebook_compatible_json": entry(run_dir / f"{trial_id}_notebook_compatible.json"),
            "interactive_dashboard_html": entry(run_dir / "pd_insighter_rule_interactive.html"),
        },
        "recommended_commands": {
            "extract_video": (
                "cd {vrs_repo} && "
                "{python} scripts/video_extractor.py --vrs_file {vrs_path}"
            ).format(
                vrs_repo=vrs_repo,
                python=GEN1_ROOT / "projectaria_tools" / ".venv" / "bin" / "python",
                vrs_path=vrs_path,
            ),
            "run_mps": (
                "export SSL_CERT_FILE=/etc/ssl/cert.pem && "
                f"{GEN1_ROOT / 'projectaria_tools' / '.venv' / 'bin' / 'aria_mps'} "
                f"single -i {vrs_path}"
            ),
            "run_analysis": (
                "cd {qwen_repo} && "
                "{python} scripts/process_aria_trial.py "
                "--video-path {video_path} "
                "--vrs-json-path {vrs_json} "
                "--trajectory-csv {trajectory_csv} "
                "--points-csv {points_csv} "
                "--eye-gaze-csv {eye_gaze_csv} "
                "--output-dir {run_dir}"
            ).format(
                qwen_repo=qwen_repo,
                python=GEN1_ROOT / "projectaria_tools" / ".venv" / "bin" / "python",
                video_path=extracted_video_path,
                vrs_json=vrs_json_path,
                trajectory_csv=mps_dir / "slam" / "closed_loop_trajectory.csv",
                points_csv=mps_dir / "slam" / "semidense_points.csv.gz",
                eye_gaze_csv=mps_dir / "eye_gaze" / "general_eye_gaze.csv",
                run_dir=run_dir,
            ),
        },
    }

    manifest_path = run_dir / "trial_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest to: {manifest_path}")
    print(f"Trial data directory: {trial_data_dir}")
    print(f"Analysis run directory: {run_dir}")


if __name__ == "__main__":
    main()
