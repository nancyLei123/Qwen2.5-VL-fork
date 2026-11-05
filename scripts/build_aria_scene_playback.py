#!/usr/bin/env python3
"""Publish a frame-timed Aria/Gaussian player without retraining any model."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from export_aria_gaussian_dataset import interpolate_pose, load_trajectory
from aria_playback_pose_recovery import recover_short_gap, support_reason
from pd_insighter_environment_navigation import inject_environment_navigation

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = Path(__file__).parent / "templates"
PLAYER_MODULES = ("scene_playback.js", "scene_playback_core.mjs", "scene_playback_quality.mjs",
                  "scene_route_navigation.mjs", "scene_camera_view.mjs", "scene_spatial_overview.mjs",
                  "scene_pose_figure.mjs", "scene_gait_fusion.mjs", "scene_view_footprint.mjs",
                  "scene_surface_visibility.mjs")


def camera_pose(trajectory, capture_ns, camera, calibration_gap_ns):
    """Match the exporter, including camera time offset and upright RGB basis."""
    if calibration_gap_ns > 60_000_000:
        return None
    query_ns = capture_ns - round(camera.get_time_offset_sec_device_camera() * 1e9)
    try:
        world_device, _ = interpolate_pose(
            trajectory["times_ns"], trajectory["translations"],
            trajectory["quaternions"], query_ns,
        )
    except ValueError:
        return None
    right = int(np.searchsorted(trajectory["times_ns"], query_ns))
    quality = trajectory["frame"].quality_score.iloc[max(0, right - 1):right + 1]
    if quality.empty or not np.isfinite(quality).all() or quality.min() < 0.9:
        return None
    world_camera = world_device @ camera.get_transform_device_camera().to_matrix()
    upright = Rotation.from_euler("z", -90, degrees=True).as_matrix()
    world_camera[:3, :3] = world_camera[:3, :3] @ upright
    return world_camera


def three_pose(world_camera):
    # COLMAP: right/down/forward. Three.js: right/up/backward.
    rotation = world_camera[:3, :3] @ np.diag([1, -1, -1])
    return [*world_camera[:3, 3], *Rotation.from_matrix(rotation).as_quat()]


def verify_frame_mapping(timestamps, metadata, summary):
    if len(timestamps) != metadata["written_frames"]:
        raise ValueError("VRS and extracted video frame counts differ; cannot assume index alignment.")
    if len(timestamps) != summary["video"]["browser_asset"]["frames"]:
        raise ValueError("Browser video has a different frame count.")
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError("RGB timestamps must be strictly increasing.")
    if int(timestamps[0]) != summary["sync"]["video_start_device_ns"]:
        raise ValueError("RGB origin differs from the existing video sync metadata.")
    if summary["sync"]["video_trim_start_s"] != 0:
        raise ValueError("This player requires the complete untrimmed browser video.")
    if metadata["fps"] != summary["video"]["browser_asset"]["fps"]:
        raise ValueError("Browser and extraction frame rates differ.")


def convert_segments(segments, offset, duration):
    converted = []
    for segment in segments:
        start = max(0, float(segment["start_s"]) - offset)
        end = min(duration, float(segment["end_s"]) - offset)
        if end <= start:
            continue
        converted.append({
            "start_s": start, "end_s": end, "label": segment["label"],
            "color": segment.get("color", "#8a948f"),
            "source": segment.get("source", "unspecified"),
        })
    return converted


def publish_player_assets(output):
    """Deploy the complete viewer bundle together, including its camera controls."""
    for filename in PLAYER_MODULES:
        shutil.copyfile(TEMPLATES / filename, output / filename)
    # Local vendor asset: no CDN, research data upload, or reconstruction rerun.
    from plotly.offline.offline import get_plotlyjs
    (output / "spatial_plotly.min.js").write_text(get_plotlyjs())


def publish_ui(trial_id):
    """Refresh player controls without recomputing or modifying research data."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", trial_id):
        raise ValueError("Invalid trial ID.")
    output = ROOT / "runs" / trial_id / "environment_reconstruction"
    payload = json.loads((output / "scene_playback_data.json").read_text())
    if payload["trial_id"] != trial_id:
        raise ValueError("Playback data belongs to a different trial.")
    template = (TEMPLATES / "pd_insighter_scene_playback.html").read_text()
    document = inject_environment_navigation(template.replace("__TRIAL_ID__", trial_id), output.parent, output / "playback.html")
    (output / "playback.html").write_text(document)
    publish_player_assets(output)
    return output / "playback.html"


def build(trial_id):
    from projectaria_tools.core import data_provider, mps
    from projectaria_tools.core.sensor_data import TimeDomain

    if trial_id != "Ntrial3":
        raise ValueError("Currently validated for Ntrial3 only.")
    run = ROOT / "runs" / trial_id
    output = run / "environment_reconstruction"
    summary = json.loads((output / "reconstruction_summary.json").read_text())
    dataset = json.loads((output / "gaussian_submaps/sitting_context/dataset_manifest.json").read_text())
    inputs = dataset["inputs"]
    video = Path(summary["outputs"]["browser_video_mp4"])
    if not video.is_file():
        raise FileNotFoundError(video)
    metadata = json.loads(Path(summary["inputs"]["aria_video"]).with_name("metadata.json").read_text())
    provider = data_provider.create_vrs_data_provider(inputs["vrs"])
    sid = provider.get_stream_id_from_label("camera-rgb")
    timestamps = np.asarray(provider.get_timestamps_ns(sid, TimeDomain.DEVICE_TIME), dtype=np.int64)
    verify_frame_mapping(timestamps, metadata, summary)
    for index in (0, len(timestamps) // 2, len(timestamps) - 1):
        _, record = provider.get_image_data_by_index(sid, index)
        if record.capture_timestamp_ns != timestamps[index]:
            raise ValueError("VRS DEVICE_TIME does not match the RGB exposure timestamps.")
    trajectory = load_trajectory(Path(inputs["trajectory"]))
    if trajectory["graph_uid"] != dataset["world_frame"]["graph_uid"]:
        raise ValueError("Trajectory and model use different world frames.")
    calibrations = mps.read_online_calibration(inputs["calibration"])
    calibration_ns = np.asarray([
        round(item.tracking_timestamp.total_seconds() * 1e9) for item in calibrations
    ], dtype=np.int64)
    cameras = [item.get_camera_calib("camera-rgb") for item in calibrations]
    poses, pose_status, pose_audit = [], [], []
    for frame_index, stamp in enumerate(timestamps):
        index = int(np.argmin(np.abs(calibration_ns - stamp)))
        gap = abs(int(calibration_ns[index]) - int(stamp))
        camera = cameras[index]
        matrix = camera_pose(trajectory, int(stamp), camera, gap) if camera else None
        status = "mps_supported" if matrix is not None else "unavailable"
        if matrix is None:
            query = int(stamp) - round(camera.get_time_offset_sec_device_camera() * 1e9) if camera else int(stamp)
            audit = {"frame": frame_index, "video_s": frame_index / float(metadata["fps"]),
                     "capture_ns": int(stamp), "calibration_gap_ms": gap / 1e6,
                     "reason": support_reason(trajectory, query), "status": status}
            if camera and gap <= 60_000_000:
                device, recovery = recover_short_gap(trajectory, query)
                audit.update(recovery)
                if device is not None:
                    matrix = device @ camera.get_transform_device_camera().to_matrix()
                    matrix[:3, :3] = matrix[:3, :3] @ Rotation.from_euler("z", -90, degrees=True).as_matrix()
                    status = "interpolated_short_gap"
            else:
                audit["calibration_rejection"] = "no_nearby_online_RGB_calibration"
            audit["status"] = status
            pose_audit.append(audit)
        poses.append(np.round(three_pose(matrix), 9).tolist() if matrix is not None else None)
        pose_status.append(status)
    fps = float(metadata["fps"])
    recording_times = (timestamps - timestamps[0]) / 1e9
    duration = len(timestamps) / fps
    specs = [
        ("sitting_context", "Sitting-context submap", "gaussian_submaps/sitting_context", "trained_7000_fullres"),
        ("early_proof", "Earlier corridor experiment", "gaussian_splatting_trial", "trained_proof"),
        ("upstairs_context", "Upstairs-context submap", "gaussian_submaps/upstairs_context", "trained_7000_fullres"),
        ("corridor_walking", "Corridor submap", "gaussian_submaps/corridor_walking", "trained_7000_fullres"),
        ("downstairs_context", "Downstairs-context submap", "gaussian_submaps/downstairs_context", "trained_7000_fullres"),
    ]
    # Additional gap models are opt-in artifacts; existing trained models remain untouched.
    for scene_id, title in [("connecting_walk_gap", "Connecting-walk gap submap"),
                            ("final_walk_gap", "Final stairwell gap submap")]:
        dataset_dir = f"gaussian_submaps/{scene_id}"
        if (output / dataset_dir / "trained_7000_fullres/training_summary.json").is_file():
            specs.append((scene_id, title, dataset_dir, "trained_7000_fullres"))
    scenes = []
    for scene_id, title, dataset_dir, trained_dir in specs:
        source = output / dataset_dir
        manifest = json.loads((source / "dataset_manifest.json").read_text())
        trained = json.loads((source / trained_dir / "training_summary.json").read_text())
        if manifest["trial_id"] != trial_id or manifest["world_frame"]["graph_uid"] != trajectory["graph_uid"]:
            raise ValueError("Model belongs to another trial/world frame.")
        if trained["trial_id"] != trial_id:
            raise ValueError("Training result belongs to another trial.")
        splat = source / trained_dir / trained["viewer_splat_filename"]
        if not splat.is_file():
            raise FileNotFoundError(splat)
        start = manifest["segment"]["start_s"]
        end = manifest["segment"]["end_s"]
        if scene_id == "connecting_walk_gap":
            start, end = 60.0, 78.0
        elif scene_id == "final_walk_gap":
            start = 216.0
            # Include the last exported view, but never advertise unsampled tail coverage.
            end = max(item["recording_s"] for item in manifest["frames"]["items"]) + 1e-6
        if scene_id.endswith("_gap"):
            sampled = [item["recording_s"] for item in manifest["frames"]["items"]]
            if start < min(sampled) or end > max(sampled) + 1.1e-6:
                raise ValueError("Gap-model playback bounds extend beyond exported camera views.")
        covered_frames = np.flatnonzero((recording_times >= start) & (recording_times < end))
        if not len(covered_frames):
            raise ValueError("Scene has no corresponding video frames.")
        scenes.append({
            "id": scene_id, "title": title, "start_s": start, "end_s": end,
            "video_start_s": int(covered_frames[0]) / fps,
            "video_end_s": (int(covered_frames[-1]) + 1) / fps,
            "ply": os.path.relpath(splat, output), "iterations": trained["iterations"],
            "camera": manifest["camera"], "validation_psnr_db": trained["validation_psnr_db"],
            "validation_recording_s": trained["scene_view"]["recording_s"],
            "details_url": f"{dataset_dir}/{trained_dir}/index.html",
            "training_start_s": manifest["segment"]["start_s"],
            "training_end_s": manifest["segment"]["end_s"],
        })
    scenes.sort(key=lambda item: item["start_s"])
    valid_positions = np.array([pose[:3] for pose in poses if pose is not None])
    payload = {
        "trial_id": trial_id, "video_url": os.path.relpath(video, output),
        "fps": fps, "duration_s": duration, "recording_times": recording_times.round(9).tolist(),
        "poses": poses, "pose_status": pose_status, "pose_recovery_audit": pose_audit,
        "scenes": scenes, "dashboard_offset_s": summary["sync"]["dashboard_time_offset_s"],
        "dashboard_duration_s": summary["sync"]["dashboard_duration_s"],
        "labels_url": "../session_dashboard_manifest.json",
        "bounds": [valid_positions.min(axis=0).tolist(), valid_positions.max(axis=0).tolist()],
        "provenance": {
            "vrs": inputs["vrs"], "trajectory": inputs["trajectory"], "calibration": inputs["calibration"],
            "graph_uid": trajectory["graph_uid"], "frame_count": len(timestamps),
            "valid_pose_count": sum(pose is not None for pose in poses),
            "mps_supported_pose_count": pose_status.count("mps_supported"),
            "estimated_pose_count": pose_status.count("interpolated_short_gap"),
            "pose_recovery_policy": "Playback only: <=50ms low-quality gaps, two reliable anchors, bounded motion and local held-out consistency check. No endpoint extrapolation. Estimates explicitly marked; never used for model training.",
            "time_mapping": "MP4 frame index -> exact VRS RGB capture time -> camera time offset -> closed-loop pose",
            "video_minus_device_at_last_frame_s": float((len(timestamps) - 1) / fps - recording_times[-1]),
            "orientation": "T_world_device @ online T_device_RGB @ upright Rz(-90) @ diag(1,-1,-1,1)",
            "label_time_mapping": "dashboard time = MP4 time + dashboard_offset_s (existing label clock)",
            "model_coverage_clock": "VRS recording seconds from the first RGB capture",
        },
    }
    (output / "scene_playback_data.json").write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    publish_ui(trial_id)
    print(json.dumps({"page": str(output / "playback.html"), "frames": len(timestamps),
                      "poses": payload["provenance"]["valid_pose_count"], "scenes": len(scenes),
                      "clock_correction_s": payload["provenance"]["video_minus_device_at_last_frame_s"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-id", default="Ntrial3")
    parser.add_argument("--ui-only", action="store_true", help="Refresh HTML/JS only; preserve all playback data and models.")
    args = parser.parse_args()
    if args.ui_only:
        print(publish_ui(args.trial_id))
    else:
        build(args.trial_id)
