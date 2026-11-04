#!/usr/bin/env python3
"""Create an allowlisted, checksummed local Ntrial3 reconstruction snapshot."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import struct
import zipfile


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
RUN = ROOT / "runs/Ntrial3"
ENV = RUN / "environment_reconstruction"
ASSETS = ROOT / "docs/ntrial3_reconstruction_package"
CHUNK = 8 * 1024 * 1024


def digest_stream(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(CHUNK), b""):
        result.update(chunk)
    return result.hexdigest()


def digest_file(path):
    with path.open("rb") as stream:
        return digest_stream(stream)


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def select_files():
    selected = {}

    def add(path, category, destination=None):
        if not path.is_file():
            raise FileNotFoundError(path)
        if not path.resolve().is_relative_to(WORKSPACE):
            raise ValueError(f"Source resolves outside research workspace: {path}")
        relative = destination or path.relative_to(WORKSPACE).as_posix()
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(relative)
        if relative in selected and selected[relative][0] != path:
            raise ValueError(f"Destination collision: {relative}")
        selected[relative] = (path, category)

    def tree(path, category):
        if not path.is_dir():
            raise FileNotFoundError(path)
        for item in sorted(path.rglob("*")):
            if item.is_file() and not any(p.startswith(".") for p in item.relative_to(path).parts):
                add(item, category)

    raw = WORKSPACE / "VRS_Pre_Processing/data/Ntrial3"
    add(raw / "NTrial_3.vrs", "raw_recording")
    add(raw / "NTrial_3.vrs.json", "raw_recording")
    mps = raw / "mps_NTrial_3_vrs"
    for name in ("closed_loop_trajectory.csv", "open_loop_trajectory.csv",
                 "online_calibration.jsonl", "semidense_points.csv.gz",
                 "semidense_observations.csv.gz", "summary.json"):
        add(mps / "slam" / name, "mps_slam")
    add(mps / "vrs_health_check.json", "mps_slam")
    tree(mps / "eye_gaze", "optional_context")
    tree(mps / "hand_tracking", "optional_context")

    playback = read_json(ENV / "scene_playback_data.json")
    datasets = []
    for scene in playback["scenes"]:
        model = ENV / scene["ply"]
        dataset = model.parent.parent
        manifest = read_json(dataset / "dataset_manifest.json")
        tree(dataset / "images", "calibrated_training_inputs")
        tree(dataset / "sparse", "calibrated_training_inputs")
        for name in ("dataset_manifest.json", "README.md", "contact_sheet.jpg"):
            add(dataset / name, "calibrated_training_inputs")
        add(model, "reference_models")
        add(model.parent / "training_summary.json", "reference_models")
        add(model.parent / "training.log", "reference_models")
        for camera_file in model.parent.glob("cameras_*.json"):
            add(camera_file, "reference_models")
        datasets.append({
            "id": scene["id"], "title": scene["title"],
            "path": dataset.relative_to(WORKSPACE).as_posix(),
            "training_segment_recording_s": manifest["segment"],
            "playback_segment_recording_s": [scene["start_s"], scene["end_s"]],
            "frames": manifest["frames"]["selected"],
            "initializer_points": manifest["points"]["written"],
            "camera": manifest["camera"], "world_frame": manifest["world_frame"],
            "reference_model": model.relative_to(WORKSPACE).as_posix(),
        })

    quality = ENV / "quality_comparison_20261022"
    for scene in ("sitting_context", "corridor_walking", "upstairs_context"):
        base = quality / scene
        add(base / "split.json", "comparison_inputs")
        tree(base / "train", "comparison_inputs")
        tree(base / "eval", "comparison_inputs")
        for recipe in ("baseline_7000", "long_30000"):
            folder = base / recipe
            for name in ("run.json", "splat.ply", "training.log"):
                add(folder / name, "comparison_results")
            evaluation = folder / ("eval_7000" if recipe == "baseline_7000" else "eval_30000")
            tree(evaluation, "comparison_results")
    for name in ("results.json", "findings.json", "report.md", "playback_choices.json"):
        add(quality / name, "comparison_results")

    media = WORKSPACE / "VRS_Pre_Processing/data/extracted_videos/Ntrial3/undistort_True_colorcorrect_True_devignette_False/NTrial_3.vrs"
    for name in ("NTrial_3.vrs.mp4", "metadata.json", "audio_track.wav", "audio_metadata.json"):
        add(media / name, "reference_media")
    add(WORKSPACE / "downward videos/NTrial3.MP4", "optional_downward_video")
    for name in ("browser_aria_video.mp4", "scene_playback_data.json", "reconstruction_summary.json",
                 "filtered_mps_point_cloud.ply", "experimental_surface_mesh.glb"):
        add(ENV / name, "playback_and_geometry_reference")
    add(ENV / "gaussian_submaps/submaps_manifest.json", "playback_and_geometry_reference")
    add(ENV / "coverage_recovery_audit/comparison.json", "playback_and_geometry_reference")
    tree(RUN / "manual_labels", "optional_context")
    tree(RUN / "spoken_action_labels", "optional_context")
    for name in ("sync_suggestion.json", "aligned_envelopes.png", "aligned_peaks.png", "correlation_curve.png"):
        add(RUN / "downward_sync" / name, "optional_context")
    for name in ("aria_session_video.build.json", "downward_session_video.build.json"):
        add(RUN / "downward_master_session" / name, "optional_context")
    add(RUN / "time_information.json", "optional_context")
    for name in ("export_aria_gaussian_dataset.py", "train_aria_gaussian_scene.py",
                 "compare_aria_gaussian_quality.py", "aria_playback_pose_recovery.py"):
        add(ROOT / "scripts" / name, "source_snapshot")
    add(ASSETS / "README.md", "documentation", "README.md")
    add(ASSETS / "verify_package.py", "documentation", "verify_package.py")
    return selected, datasets, playback


def validate_datasets(datasets):
    from PIL import Image
    import numpy as np
    from scipy.spatial.transform import Rotation

    checks = []
    for item in datasets:
        folder = WORKSPACE / item["path"]
        manifest = read_json(folder / "dataset_manifest.json")
        frames = manifest["frames"]["items"]
        names = {frame["name"] for frame in frames}
        assert len(names) == len(frames) == item["frames"]
        assert names == {p.name for p in (folder / "images").glob("*.jpg")}
        stamps = [frame["capture_timestamp_ns"] for frame in frames]
        assert all(b > a for a, b in zip(stamps, stamps[1:]))
        for frame in frames:
            with Image.open(folder / "images" / frame["name"]) as image:
                assert image.size == (manifest["camera"]["width"], manifest["camera"]["height"])
                image.verify()
            transform = np.asarray(frame["T_world_camera"])
            assert np.isfinite(transform).all()
            rotation = transform[:3, :3]
            assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            assert np.isclose(np.linalg.det(rotation), 1, atol=1e-6)
            q = frame["colmap_qvec_wxyz"]
            assert np.allclose(Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(), rotation.T, atol=1e-6)
            assert np.allclose(frame["colmap_tvec"], -rotation.T @ transform[:3, 3], atol=1e-6)
        for filename, expected in (("cameras.bin", 1), ("images.bin", len(frames)), ("points3D.bin", item["initializer_points"])):
            with (folder / "sparse/0" / filename).open("rb") as stream:
                assert struct.unpack("<Q", stream.read(8))[0] == expected
        checks.append({"id": item["id"], "frames_checked": len(frames), "passed": True,
                       "checks": ["image_integrity_and_dimensions", "monotonic_capture_timestamps",
                                  "camera_pose_inverse_convention", "COLMAP_binary_record_counts"]})
    return checks


def inspect_vrs(playback):
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.sensor_data import TimeDomain

    source = WORKSPACE / "VRS_Pre_Processing/data/Ntrial3/NTrial_3.vrs"
    provider = data_provider.create_vrs_data_provider(str(source))
    if provider is None:
        raise RuntimeError("Cannot open the raw VRS")
    streams = [{"stream_id": str(s), "label": provider.get_label_from_stream_id(s),
                "record_count": provider.get_num_data(s)} for s in provider.get_all_streams()]
    rgb = provider.get_stream_id_from_label("camera-rgb")
    times = provider.get_timestamps_ns(rgb, TimeDomain.DEVICE_TIME)
    assert len(times) == len(playback["recording_times"]) == len(playback["poses"])
    assert all(b > a for a, b in zip(times, times[1:]))
    assert all(abs((stamp - times[0]) / 1e9 - rec) < 1e-6
               for stamp, rec in zip(times, playback["recording_times"]))
    decoded = []
    for index in (0, len(times) // 2, len(times) - 1):
        image, metadata = provider.get_image_data_by_index(rgb, index)
        assert metadata.capture_timestamp_ns == times[index]
        decoded.append({"frame_index": index, "shape": list(image.to_numpy_array().shape)})
    return {"streams": streams, "rgb_sample_decodes": decoded,
            "first_rgb_capture_ns": times[0], "last_rgb_capture_ns": times[-1],
            "all_playback_recording_times_match_vrs": True,
            "scope": "Stream metadata, complete RGB timestamp mapping, and three sampled image decodes; not a full sensor decode."}


def validate_splits():
    checks = []
    for scene in ("sitting_context", "corridor_walking", "upstairs_context"):
        folder = ENV / "quality_comparison_20261022" / scene
        split = read_json(folder / "split.json")
        groups = {key: {item["name"] for item in split[key]} for key in ("training", "validation", "guard")}
        assert groups["training"].isdisjoint(groups["validation"] | groups["guard"])
        assert groups["validation"].isdisjoint(groups["guard"])
        assert groups["training"] == {p.name for p in (folder / "train/images").glob("*.jpg")}
        assert set.union(*groups.values()) == {p.name for p in (folder / "eval/images").glob("*.jpg")}
        for frame in split["training"]:
            assert all(abs(frame["recording_s"] - held["recording_s"]) > split["guard_s"]
                       for held in split["validation"])
        checks.append({"scene": scene, "training_frames": len(groups["training"]),
                       "validation_frames": len(groups["validation"]), "guard_frames": len(groups["guard"]),
                       "membership_and_temporal_guard_checked": True})
    return checks


def add_generated(root, entries, relative, value):
    path = root / relative
    write_json(path, value)
    entries.append({"path": relative, "bytes": path.stat().st_size,
                    "sha256": digest_file(path), "category": "generated_metadata", "source": "generated"})


def build(output, selected, datasets, playback):
    archive = output.with_suffix(".zip")
    if output.exists() or archive.exists() or archive.with_suffix(".zip.partial").exists():
        raise FileExistsError("Choose a new output name; existing packages are never overwritten.")
    total = sum(p.stat().st_size for p, _ in selected.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < total * 2.2:
        raise RuntimeError("Insufficient room for both the folder and ZIP with safety margin.")
    checks = validate_datasets(datasets)
    raw_checks = inspect_vrs(playback)
    split_checks = validate_splits()
    output.mkdir()
    entries = []
    source_checks = []
    copied = 0
    for index, (relative, (source, category)) in enumerate(sorted(selected.items()), 1):
        before = source.stat()
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        checksum = hashlib.sha256()
        with source.open("rb") as reader, target.open("xb") as writer:
            for chunk in iter(lambda: reader.read(CHUNK), b""):
                checksum.update(chunk)
                writer.write(chunk)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Source changed during copy: {source}")
        assert target.stat().st_size == before.st_size
        entries.append({"path": relative, "bytes": before.st_size, "sha256": checksum.hexdigest(),
                        "category": category, "source": source.relative_to(WORKSPACE).as_posix(),
                        "source_was_symlink": source.is_symlink()})
        source_checks.append((source, before.st_size, before.st_mtime_ns))
        copied += before.st_size
        if index % 250 == 0 or before.st_size > 100_000_000 or index == len(selected):
            print(f"Copied {index}/{len(selected)} files, {copied / 1e9:.2f}/{total / 1e9:.2f} GB", flush=True)

    dashboard = read_json(RUN / "session_dashboard_manifest.json")
    context_fields = ("trial_id", "recording_stem", "session_duration_s", "session_fps",
                      "continuous_model_segments", "manual_timeline_segments", "manual_timeline_source")
    context = {key: dashboard.get(key) for key in context_fields}
    context["source_sha256"] = digest_file(RUN / "session_dashboard_manifest.json")
    context["time_mapping"] = playback["provenance"]["label_time_mapping"]
    context["dashboard_offset_s"] = playback["dashboard_offset_s"]
    add_generated(output, entries, "reference_context/action_labels.json", context)

    package_index = {
        "trial_id": "Ntrial3", "created_utc": datetime.now(timezone.utc).isoformat(),
        "raw_vrs": "VRS_Pre_Processing/data/Ntrial3/NTrial_3.vrs",
        "mps": "VRS_Pre_Processing/data/Ntrial3/mps_NTrial_3_vrs",
        "datasets": datasets,
        "comparison_root": "Qwen2.5-VL-fork/runs/Ntrial3/environment_reconstruction/quality_comparison_20261022",
        "playback_data": "Qwen2.5-VL-fork/runs/Ntrial3/environment_reconstruction/scene_playback_data.json",
        "action_context": "reference_context/action_labels.json",
        "pose_status_counts": dict(Counter(playback["pose_status"])),
        "duration_s": playback["duration_s"], "dataset_validation": checks,
        "raw_vrs_check": raw_checks, "comparison_split_checks": split_checks,
        "note": "Paths in this index are relative to the package root. Original manifests/logs retain historical paths."
    }
    add_generated(output, entries, "PACKAGE_INDEX.json", package_index)
    categories = defaultdict(lambda: {"files": 0, "bytes": 0})
    for entry in entries:
        categories[entry["category"]]["files"] += 1
        categories[entry["category"]]["bytes"] += entry["bytes"]
    catalog = ["# NTrial3 Data Catalog", "", "## Ready to train datasets", "",
               "Times are exact VRS-recording seconds, not the trimmed dashboard clock.", "",
               "| Scene | Training interval (s) | RGB frames | Initial points | Dataset path |",
               "|---|---:|---:|---:|---|"]
    for scene in datasets:
        segment = scene["training_segment_recording_s"]
        catalog.append(f"| {scene['title']} | {segment['start_s']}-{segment['end_s']} | {scene['frames']} | {scene['initializer_points']:,} | `{scene['path']}` |")
    catalog += ["", "## Category sizes", "", "Sizes below count file bytes, before ZIP compression.", ""]
    for category, count in sorted(categories.items()):
        catalog.append(f"- {category}: {count['files']:,} files, {count['bytes']/1e9:.3f} GB.")
    catalog += ["", "## What is not a training input", "",
                "Action labels, gaze, hands, downward video, playback poses, and old model outputs are optional context/reference.",
                "The raw VRS, closed-loop trajectory, calibration and landmarks are the core re-export inputs.",
                "Existing 7k/1.5k models remain separate from the newer comparison models; none were substituted.", ""]
    (output / "DATA_CATALOG.md").write_text("\n".join(catalog))
    catalog_path = output / "DATA_CATALOG.md"
    entries.append({"path": "DATA_CATALOG.md", "bytes": catalog_path.stat().st_size,
                    "sha256": digest_file(catalog_path), "category": "documentation", "source": "generated"})
    inventory = {"algorithm": "SHA-256", "trial_id": "Ntrial3", "files": entries,
                 "payload_file_count": len(entries), "payload_bytes": sum(e["bytes"] for e in entries),
                 "note": "Inventory excludes itself; the external archive checksum covers it."}
    write_json(output / "FILE_INVENTORY.json", inventory)
    partial = archive.with_suffix(".zip.partial")
    expected = {entry["path"]: entry["sha256"] for entry in entries}
    expected["FILE_INVENTORY.json"] = digest_file(output / "FILE_INVENTORY.json")
    print("Creating ZIP64 archive; existing compressed recordings/media are stored losslessly...", flush=True)
    compressed_suffixes = {".vrs", ".mp4", ".jpg", ".jpeg", ".png", ".gz", ".zip"}
    with zipfile.ZipFile(partial, "x", allowZip64=True, compresslevel=1) as bundle:
        for index, relative in enumerate(sorted(expected), 1):
            path = output / relative
            method = zipfile.ZIP_STORED if path.suffix.lower() in compressed_suffixes else zipfile.ZIP_DEFLATED
            bundle.write(path, f"{output.name}/{relative}", compress_type=method, compresslevel=1)
            if index % 500 == 0:
                print(f"Archived {index}/{len(expected)} files", flush=True)
    print("Re-reading every archived file to verify SHA-256 and ZIP integrity...", flush=True)
    with zipfile.ZipFile(partial) as bundle:
        assert len(bundle.namelist()) == len(expected)
        for relative, digest in expected.items():
            with bundle.open(f"{output.name}/{relative}") as stream:
                if digest_stream(stream) != digest:
                    raise RuntimeError(f"Archive checksum mismatch: {relative}")
    for source, size, modified in source_checks:
        now = source.stat()
        if (now.st_size, now.st_mtime_ns) != (size, modified):
            raise RuntimeError(f"Source changed before verification finished: {source}")
    partial.rename(archive)
    zip_hash = digest_file(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{zip_hash}  {archive.name}\n")
    report = {"status": "verified", "archive": str(archive), "folder": str(output),
              "archive_bytes": archive.stat().st_size, "payload_bytes": inventory["payload_bytes"],
              "archived_files": len(expected), "sha256": zip_hash,
              "dataset_validation": checks, "source_sizes_and_mtimes_unchanged": True,
              "archive_payload_sha256_matches_source_snapshot": True, "uploaded": False}
    write_json(archive.with_suffix(".verification.json"), report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", action="store_true", help="Otherwise print a read-only plan.")
    args = parser.parse_args()
    selected, datasets, playback = select_files()
    totals = defaultdict(int)
    for source, category in selected.values():
        totals[category] += source.stat().st_size
    print(json.dumps({"files": len(selected), "source_bytes": sum(totals.values()),
                      "categories_bytes": totals, "datasets": [d["id"] for d in datasets]}, indent=2), flush=True)
    if args.build:
        build(args.output.resolve(), selected, datasets, playback)


if __name__ == "__main__":
    main()
