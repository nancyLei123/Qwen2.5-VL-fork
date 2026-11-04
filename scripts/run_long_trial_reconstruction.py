#!/usr/bin/env python3
"""Local, additive long-trial reconstruction with the existing dashboard clock."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np

from build_aria_scene_playback import camera_pose, three_pose
from compare_aria_gaussian_quality import (
    DEFAULT_BINARY, base_command, color_points, evaluate, execute, frame_records,
    make_dataset, save_json, sha256, split_frames, verify_sources,
)
from export_aria_gaussian_dataset import export_dataset, load_export_context

ROOT = Path(__file__).resolve().parents[1]
SPECS = (
    {"id": "workroom_transition", "title": "Workroom / rising and leaving", "start": 460, "end": 490},
    {"id": "corridor_walk", "title": "Corridor / walking context", "start": 490, "end": 530},
    {"id": "stair_descent", "title": "Stairwell / downstairs context", "start": 1365, "end": 1402},
)
FIRST15_SPECS = (
    {"id": "first15_workroom_before", "title": "Workroom / before leaving", "start": 0, "end": 460, "candidate_fps": 1.0, "max_frame_gap_s": 4.0},
    {"id": "first15_sink_visit", "title": "Corridor and washbasin visit", "start": 530, "end": 570},
    {"id": "first15_return_walk", "title": "Returning to the workroom", "start": 570, "end": 605},
    {"id": "first15_workroom_after", "title": "Workroom / after returning", "start": 605, "end": 900, "candidate_fps": 1.0, "max_frame_gap_s": 4.0},
)
FIRST30_SPECS = (
    {"id": "first30_workroom_continued", "title": "Workroom / continued sitting", "start": 900, "end": 1235, "candidate_fps": 1.0, "max_frame_gap_s": 4.0},
    {"id": "first30_workroom_exit", "title": "Leaving the workroom", "start": 1235, "end": 1265},
    {"id": "first30_upper_corridor", "title": "Upper corridor / outward walk", "start": 1265, "end": 1315},
    {"id": "first30_stair_approach", "title": "Upper corridor / approaching stairs", "start": 1315, "end": 1365},
    {"id": "first30_lower_landing", "title": "Lower landing and corridor", "start": 1402, "end": 1435},
    {"id": "first30_lower_corridor", "title": "Lower corridor / approaching wall display", "start": 1435, "end": 1470},
    {"id": "first30_wall_display_arrival", "title": "Wall display / arrival", "start": 1470, "end": 1545, "candidate_fps": 2.0, "max_frame_gap_s": 2.0},
    {"id": "first30_wall_display_standing", "title": "Wall display / standing context", "start": 1545, "end": 1800, "candidate_fps": 1.0, "max_frame_gap_s": 4.0},
)


def batch_files(batch):
    if batch not in ("pilot", "first15", "first30"):
        raise ValueError(f"Unknown reconstruction batch: {batch}")
    prefix = "" if batch == "pilot" else f"{batch}_"
    return f"{prefix}reconstruction_plan.json", f"{prefix}progress.json"


def probe(path):
    command = ["/opt/homebrew/bin/ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration", "-of", "json", str(path)]
    stream = json.loads(subprocess.check_output(command))["streams"][0]
    numerator, denominator = map(float, stream["avg_frame_rate"].split("/"))
    return {**stream, "fps": numerator / denominator, "frames": int(stream["nb_frames"])}


def verify_alignment(timestamps, metadata, video, audit):
    if len(timestamps) != metadata["written_frames"] or np.any(np.diff(timestamps) <= 0):
        raise ValueError("VRS/extracted video frame mapping is not verified")
    fps, trim = video["fps"], audit["source_trim_frames"]
    if fps != metadata["fps"] or fps != audit["fps"] or trim < 0:
        raise ValueError("Frame rate or trim mapping changed")
    if trim + video["frames"] > len(timestamps):
        raise ValueError("Dashboard extends beyond readable VRS frames")
    checks = audit["frame_checks"]
    if len(checks) < 3 or any(c["best_delta_frames"] != 0 for c in checks):
        raise ValueError("Source/dashboard image audit does not support the trim mapping")
    if any(c["mean_rgb8_difference_by_delta"]["0"] > 5 for c in checks):
        raise ValueError("Dashboard/source images differ too much")
    return timestamps[trim:trim + video["frames"]]


def nearest_indices(reference, queries):
    right = np.searchsorted(reference, queries).clip(0, len(reference) - 1)
    left = (right - 1).clip(0)
    return np.where(np.abs(reference[left] - queries) <= np.abs(reference[right] - queries), left, right)


def check_protected(output):
    audit = json.loads((output / "video_alignment_audit.json").read_text())
    for path, digest in audit["protected_sha256"].items():
        if sha256(path) != digest:
            raise ValueError(f"Protected label/analysis file changed: {path}")
    return audit


def prepare(output, batch="pilot"):
    from projectaria_tools.core.sensor_data import TimeDomain
    audit = check_protected(output)
    context = load_export_context("60minutes_1")
    inputs, trajectory, calibrations, calibration_ns, provider = context
    sid = provider.get_stream_id_from_label("camera-rgb")
    timestamps = np.asarray(provider.get_timestamps_ns(sid, TimeDomain.DEVICE_TIME), dtype=np.int64)
    metadata = json.loads(Path(audit["source_video"]).with_name("metadata.json").read_text())
    video = probe(audit["video"])
    aligned = verify_alignment(timestamps, metadata, video, audit)
    for index in (0, len(timestamps) // 2, len(timestamps) - 1):
        image, record = provider.get_image_data_by_index(sid, index)
        if not image.is_valid() or record.capture_timestamp_ns != timestamps[index]:
            raise ValueError("VRS image decode/timestamp verification failed")
    plan = {"trial_id": "60minutes_1", "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {k: str(v) for k, v in inputs.items()}, "scenes": [],
            "scope": {"pilot": "First three motion-centered submaps; not a whole-session reconstruction.",
                      "first15": "Fill the first 15 dashboard minutes; preserve all pilot models.",
                      "first30": "Fill untrained intervals in dashboard minutes 15-30; preserve all seven existing models."}[batch],
            "raw_tail_note": "VRS reader reported one truncated final RGB record; it was excluded by the reader. All 103358 extracted frames match readable VRS timestamps; the dashboard ends before the raw tail.",
            "source_stats": {k: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for k, p in inputs.items()}}
    plan_name, progress_name = batch_files(batch)
    for spec in {"pilot": SPECS, "first15": FIRST15_SPECS, "first30": FIRST30_SPECS}[batch]:
        start = audit["source_trim_frames"] + round(spec["start"] * video["fps"])
        end = audit["source_trim_frames"] + round(spec["end"] * video["fps"])
        plan["scenes"].append({**spec, "start_s": float((timestamps[start] - timestamps[0]) / 1e9),
                                "end_s": float((timestamps[end] - timestamps[0]) / 1e9)})
    plan_path = output / plan_name
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        for key in ("trial_id", "inputs", "source_stats", "scenes"):
            if previous[key] != plan[key]:
                raise ValueError(f"Existing reconstruction plan changed ({key}); preserve it and use a new output directory")
        plan = previous
    else:
        save_json(plan_path, plan)
    # Preserve the entire dashboard timeline, including unsupported pose/model intervals.
    base_path = output / "scene_playback_base.json"
    if base_path.exists():
        cached = json.loads(base_path.read_text())
        if (cached["trial_id"] != plan["trial_id"] or cached["fps"] != video["fps"]
                or cached["provenance"]["source_trim_frames"] != audit["source_trim_frames"]
                or len(cached["poses"]) != len(aligned)
                or not np.allclose(cached["recording_times"], (aligned - timestamps[0]) / 1e9, rtol=0, atol=1e-8)):
            raise ValueError("Cached playback timing changed; refusing to reuse its poses")
    if not base_path.exists():
        poses = []
        indices = nearest_indices(calibration_ns, aligned)
        for i, (stamp, index) in enumerate(zip(aligned, indices)):
            camera = calibrations[int(index)].get_camera_calib("camera-rgb")
            gap = abs(int(calibration_ns[index]) - int(stamp))
            matrix = camera_pose(trajectory, int(stamp), camera, gap) if camera else None
            poses.append(np.round(three_pose(matrix), 9).tolist() if matrix is not None else None)
            if i % 10000 == 0:
                print(f"Playback poses {i}/{len(aligned)}", flush=True)
        positions = np.array([p[:3] for p in poses if p is not None])
        if len(positions) == 0:
            raise ValueError("No supported poses")
        payload = {
            "trial_id": "60minutes_1", "video_url": os.path.relpath(audit["video"], output),
            "video_has_audio": True, "fps": video["fps"], "duration_s": video["frames"] / video["fps"],
            "recording_times": ((aligned - timestamps[0]) / 1e9).round(9).tolist(),
            "poses": poses, "pose_status": ["mps_supported" if p else "unavailable" for p in poses],
            "dashboard_offset_s": 0.0, "dashboard_duration_s": video["frames"] / video["fps"],
            "labels_url": "../session_dashboard_manifest.json", "scenes": [],
            "bounds": [positions.min(axis=0).tolist(), positions.max(axis=0).tolist()],
            "provenance": {"graph_uid": trajectory["graph_uid"], "inputs": plan["inputs"],
                           "frame_count": len(aligned), "valid_pose_count": len(positions),
                           "source_trim_frames": audit["source_trim_frames"],
                           "time_mapping": "Dashboard frame i -> source frame i+1265 -> exact VRS exposure -> online camera time offset -> MPS pose. Labels remain on the dashboard clock.",
                           "pose_recovery_policy": "Strict supported MPS poses only; no extrapolation or fabricated recovery.",
                           "orientation": "T_world_device @ online T_device_RGB @ upright Rz(-90) @ diag(1,-1,-1,1)",
                           "raw_tail_note": plan["raw_tail_note"]}}
        save_json(base_path, payload)
    for spec in plan["scenes"]:
        dataset = output / "gaussian_submaps" / spec["id"]
        if not (dataset / "dataset_manifest.json").exists():
            print(f"Exporting {spec['id']}: dashboard {spec['start']}-{spec['end']} s", flush=True)
            args = SimpleNamespace(trial_id="60minutes_1", start_s=spec["start_s"], end_s=spec["end_s"],
                                   candidate_fps=spec.get("candidate_fps", 5.0), width=1024, height=1024, focal_px=512.0,
                                   min_translation_m=0.06, min_rotation_deg=4.0, max_frame_gap_s=spec.get("max_frame_gap_s", 0.75),
                                   max_point_path_distance_m=8.0, max_points=200000, min_color_views=2,
                                   inv_dist_std_max=0.005, dist_std_max=0.01, output_dir=dataset)
            export_dataset(args, context)
        make_training_split(dataset)
    check_protected(output)
    save_json(output / progress_name, {"state": "prepared", "scenes": [s["id"] for s in plan["scenes"]]})


def make_training_split(source):
    destination = source / "validation_run"
    split_path = destination / "split.json"
    if split_path.exists():
        verify_sources(json.loads(split_path.read_text()))
        return
    manifest = json.loads((source / "dataset_manifest.json").read_text())
    train, validation, guard = split_frames(manifest["frames"]["items"])
    points = np.loadtxt(source / "sparse/0/points3D.txt", usecols=(1, 2, 3), dtype=np.float32)
    camera = manifest["camera"]
    colors, counts = color_points(points, frame_records(train, source),
                                  tuple(camera[k] for k in ("fx", "fy", "cx", "cy")),
                                  camera["width"], camera["height"])
    points, colors = points[counts >= 2], colors[counts >= 2]
    if len(points) < 1000:
        raise ValueError("Too few training-supported initializer points")
    make_dataset(destination / "train", manifest, train, source, points, colors)
    make_dataset(destination / "eval", manifest, manifest["frames"]["items"], source, points, colors)
    protected = [source / "dataset_manifest.json", source / "sparse/0/points3D.txt", *sorted((source / "images").glob("*.jpg"))]
    save_json(split_path, {"scene": source.name, "source": str(source), "training": train,
                          "validation": validation, "guard": guard, "guard_s": 0.25,
                          "camera": camera, "world_frame": manifest["world_frame"],
                          "initializer_points": len(points),
                          "source_sha256": {str(p): sha256(p) for p in protected},
                          "limits": "Five RGB views excluded from optimization and recoloring; shared MPS geometry and prior all-view point-support selection remain. Conditional appearance check, not an independent end-to-end or clinical evaluation."})


def train(output, batch="pilot"):
    check_protected(output)
    plan_name, progress_name = batch_files(batch)
    plan = json.loads((output / plan_name).read_text())
    for scene in plan["scenes"]:
        root = output / "gaussian_submaps" / scene["id"] / "validation_run"
        split = json.loads((root / "split.json").read_text())
        verify_sources(split)
        target = root / "trained_7000"
        target.mkdir(exist_ok=True)
        ply = target / "splat_7000.ply"
        record_path = target / "run.json"
        if not record_path.exists():
            if ply.exists() or (target / "training.log").exists():
                raise ValueError(f"Incomplete existing run preserved for inspection: {target}")
            command = base_command(DEFAULT_BINARY, root / "train", 7000) + [
                "--refine-every", "500", "--densify-from", "500", "--densify-until", "3500", "-o", str(ply)]
            runtime, _ = execute(command, target / "training.log", output / progress_name, scene["id"], 7000)
            save_json(record_path, {"command": command, "runtime_s": runtime,
                                   "binary_sha256": sha256(DEFAULT_BINARY), "model_sha256": sha256(ply)})
        record = json.loads(record_path.read_text())
        if sha256(ply) != record["model_sha256"] or sha256(DEFAULT_BINARY) != record["binary_sha256"]:
            raise ValueError("Model or trainer changed")
        score = evaluate(DEFAULT_BINARY, root, split, ply, 7000, target / "evaluation", output / progress_name)
        save_json(target / "training_summary.json", {"trial_id": plan["trial_id"], "iterations": 7000,
                  "backend": "Apple MPS", "split": str(root / "split.json"), "metrics": score,
                  "runtime_s": record["runtime_s"], "status": "trained_pending_visual_review"})
        verify_sources(split)
    check_protected(output)
    save_json(output / progress_name, {"state": "complete", "stage": "training and five-view appearance checks", "scenes": [s["id"] for s in plan["scenes"]]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train"), required=True)
    parser.add_argument("--batch", choices=("pilot", "first15", "first30"), default="pilot")
    args = parser.parse_args()
    folder = ROOT / "runs/60minutes_1/environment_reconstruction"
    if shutil.disk_usage(folder).free < 15 * 1024**3:
        raise RuntimeError("Less than 15 GiB free")
    (prepare if args.stage == "prepare" else train)(folder, args.batch)
