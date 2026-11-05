#!/usr/bin/env python3
"""Publish exact-frame playback for registered trials, including unfinished models."""
import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path

import numpy as np

from build_aria_scene_playback import camera_pose, three_pose, publish_ui
from export_aria_gaussian_dataset import find_inputs, load_trajectory
from pd_insighter_environment_navigation import refresh_navigation

ROOT = Path(__file__).resolve().parents[1]


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def build_poses(trial):
    from projectaria_tools.core import data_provider, mps
    from projectaria_tools.core.sensor_data import TimeDomain
    from process_aria_trial import load_video_info

    run = ROOT / "runs" / trial
    output = run / "environment_reconstruction"
    output.mkdir(parents=True, exist_ok=True)
    inputs = find_inputs(trial)
    manifest = json.loads((run / "trial_manifest.json").read_text())
    session = json.loads((run / "session_dashboard_manifest.json").read_text())
    source = Path(manifest["extracted_media"]["video_mp4"]["path"])
    metadata = json.loads(source.with_name("metadata.json").read_text())
    audio = json.loads(source.with_name("audio_metadata.json").read_text())
    video = run / session["aria_session_video_uri"].split("?")[0]
    fps, frames, _, _ = load_video_info(video)
    provider = data_provider.create_vrs_data_provider(str(inputs["vrs"]))
    sid = provider.get_stream_id_from_label("camera-rgb")
    timestamps = np.asarray(provider.get_timestamps_ns(sid, TimeDomain.DEVICE_TIME), dtype=np.int64)
    if frames != len(timestamps) or frames != metadata["written_frames"]:
        raise ValueError("Video and VRS frame counts differ; do not assume synchronization.")
    if abs(fps - metadata["fps"]) > .001 or not np.all(np.diff(timestamps) > 0):
        raise ValueError("Frame rate or timestamp validation failed.")
    if int(timestamps[0]) != audio["video_start_device_ns"]:
        raise ValueError("VRS RGB origin differs from audio/extraction metadata.")
    trajectory = load_trajectory(inputs["trajectory"])
    calibrations = mps.read_online_calibration(str(inputs["calibration"]))
    calibration_ns = np.asarray([round(c.tracking_timestamp.total_seconds() * 1e9) for c in calibrations])
    cameras = [c.get_camera_calib("camera-rgb") for c in calibrations]
    poses, status = [], []
    for i, stamp in enumerate(timestamps):
        right = min(int(np.searchsorted(calibration_ns, stamp)), len(calibration_ns) - 1)
        index = min((max(0, right - 1), right), key=lambda k: abs(int(calibration_ns[k]) - int(stamp)))
        gap = abs(int(calibration_ns[index]) - int(stamp))
        matrix = camera_pose(trajectory, int(stamp), cameras[index], gap) if cameras[index] else None
        poses.append(np.round(three_pose(matrix), 9).tolist() if matrix is not None else None)
        status.append("mps_supported" if matrix is not None else "unavailable")
        if i % 5000 == 0:
            print(f"{trial}: calibrated poses {i}/{frames}", flush=True)
    valid = np.asarray([p[:3] for p in poses if p is not None])
    if not len(valid):
        raise ValueError("No supported camera poses")
    recording_times = (timestamps - timestamps[0]) / 1e9
    payload = {
        "trial_id": trial, "video_url": os.path.relpath(video, output), "video_has_audio": True,
        "fps": fps, "duration_s": frames / fps, "recording_times": recording_times.round(9).tolist(),
        "poses": poses, "pose_status": status, "scenes": [], "dashboard_offset_s": 0,
        "dashboard_duration_s": session["session_duration_s"], "labels_url": "../session_dashboard_manifest.json",
        "bounds": [valid.min(axis=0).tolist(), valid.max(axis=0).tolist()],
        "provenance": {
            **{key: str(value) for key, value in inputs.items()}, "graph_uid": trajectory["graph_uid"],
            "frame_count": frames, "valid_pose_count": len(valid), "mps_supported_pose_count": len(valid),
            "estimated_pose_count": 0, "pose_recovery_policy": "No extrapolation or inferred camera poses. Missing poses remain gaps.",
            "time_mapping": "MP4 frame index -> exact VRS RGB capture time -> calibrated camera time offset -> closed-loop MPS pose",
            "video_minus_device_at_last_frame_s": float((frames - 1) / fps - recording_times[-1]),
            "orientation": "T_world_device @ online T_device_RGB @ upright Rz(-90) @ diag(1,-1,-1,1)",
        },
    }
    save(output / "scene_playback_data.json", payload)
    from build_scene_spatial_overview import export
    export(trial, inputs["points"])


def publish(trial):
    output = ROOT / "runs" / trial / "environment_reconstruction"
    path = output / "scene_playback_data.json"
    data = json.loads(path.read_text())
    times = np.asarray(data["recording_times"])
    scenes, links = [], []
    for source in sorted((output / "gaussian_submaps").glob("*/dataset_manifest.json")):
        dataset = json.loads(source.read_text())
        if dataset["trial_id"] != trial or dataset["world_frame"]["graph_uid"] != data["provenance"]["graph_uid"]:
            raise ValueError("Cannot mix different trial coordinate systems")
        trained_path = source.parent / "trained_7000_fullres/training_summary.json"
        if not trained_path.exists():
            links.append(f"<li>{html.escape(source.parent.name)}: training not complete</li>")
            continue
        trained = json.loads(trained_path.read_text())
        splat = trained_path.parent / trained["viewer_splat_filename"]
        if trained["trial_id"] != trial or not splat.is_file():
            raise ValueError("Invalid completed model")
        sampled = [f["recording_s"] for f in dataset["frames"]["items"]]
        start, end = min(sampled), max(sampled) + 1e-6
        indices = np.flatnonzero((times >= start) & (times < end))
        if not len(indices):
            continue
        details = os.path.relpath(trained_path.parent / "index.html", output)
        scene = {"id": source.parent.name, "title": f"Home submap {len(scenes) + 1}",
                 "start_s": start, "end_s": end, "video_start_s": int(indices[0]) / data["fps"],
                 "video_end_s": (int(indices[-1]) + 1) / data["fps"],
                 "ply": os.path.relpath(splat, output), "iterations": trained["iterations"],
                 "camera": dataset["camera"], "validation_psnr_db": trained["validation_psnr_db"],
                 "validation_recording_s": trained["scene_view"]["recording_s"],
                 "details_url": details, "training_start_s": dataset["segment"]["start_s"],
                 "training_end_s": dataset["segment"]["end_s"]}
        scenes.append(scene)
        links.append(f'<li><a href="../{html.escape(details)}">{start:.1f}-{end:.1f} s: {trained["iterations"]:,} iterations, held-out PSNR {trained["validation_psnr_db"]:.2f} dB</a></li>')
    data["scenes"] = sorted(scenes, key=lambda s: s["start_s"])
    covered = np.zeros(len(times), dtype=bool)
    for s in scenes:
        covered |= (times >= s["start_s"]) & (times < s["end_s"])
    supported = np.asarray([p is not None for p in data["poses"]])
    initial = np.flatnonzero(covered & supported)
    if len(initial):
        data["initial_video_time_s"] = int(initial[0]) / data["fps"]
    data["coverage_summary"] = {"scope_s": data["duration_s"], "model_seconds": int(covered.sum()) / data["fps"]}
    save(path, data)
    summary = {"trial_id": trial, "updated_at": datetime.now(timezone.utc).isoformat(),
               "status": "partial_trained_submaps" if scenes else "poses_and_spatial_map_ready_models_pending",
               "coverage": data["coverage_summary"], "provenance": data["provenance"],
               "limitations": "Simulated PD; unvalidated motion estimates. Static surroundings only, no recovered body. Time coverage is not surface completeness; moving objects may leave artifacts.",
               "outputs": {"browser_video_mp4": str((output / data["video_url"]).resolve())}}
    save(output / "reconstruction_summary.json", summary)
    publish_ui(trial)
    page = output / "playback.html"
    document = page.read_text()
    document = document.replace("Browser preview uses SH2 color; offline quality comparisons use the trainer's full SH3 renderer.", "Browser preview and these new models use SH2 color. The static reconstruction is experimental.")
    page.write_text(document)
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=playback.html"><a href="playback.html">Synchronized environment playback</a>')
    submaps = output / "gaussian_submaps"
    submaps.mkdir(exist_ok=True)
    (submaps / "index.html").write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{trial} reconstruction</title><style>body{{font:17px Georgia;max-width:960px;margin:50px auto;padding:20px;background:#f3f6ef;color:#24382d}}li{{margin:18px 0}}a{{color:#167852}}</style><a href="../playback.html">Back to synchronized playback</a><h1>{trial}: reconstruction submaps</h1><p>Completed models only are shown as available on the timeline. Reload after training progresses. Coverage does not imply complete surfaces.</p><ul>{"".join(links) or "<li>Preparing calibrated views; no trained model yet.</li>"}</ul><a href="../reconstruction_jobs.json">Processing status</a></html>')
    refresh_navigation([trial])
    print(f"{trial}: published {len(scenes)} trained submaps; model-time coverage {covered.mean():.1%}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--publish-only", action="store_true")
    args = parser.parse_args()
    if not args.publish_only:
        build_poses(args.trial_id)
    publish(args.trial_id)
