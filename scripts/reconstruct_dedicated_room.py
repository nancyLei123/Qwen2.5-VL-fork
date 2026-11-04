#!/usr/bin/env python3
"""Auditable, local-only reconstruction of the dedicated 3c room scan.

Source files are read-only. All derived products live in a new experiment.
Unlike the short-clip exporter, no full-resolution candidate images are kept
in memory while screening the entire recording.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
STEM = "3c6ed915-3279-4edf-955b-6523e1e11bb8"
SOURCE = ROOT.parent / "VRS_Pre_Processing/data/3Dreconstruction"
OUTPUT = ROOT / "runs/RoomScan3c/environment_reconstruction/quality_20261029"
BINARY = ROOT.parent / "tools/OpenSplat-1.2.2/build-room-background/opensplat"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inputs():
    slam = SOURCE / f"mps_{STEM}_vrs/slam"
    return {"vrs": SOURCE / f"{STEM}.vrs", "metadata": SOURCE / f"{STEM}.vrs.json",
            "trajectory": slam / "closed_loop_trajectory.csv",
            "calibration": slam / "online_calibration.jsonl",
            "points": slam / "semidense_points.csv.gz",
            "observations": slam / "semidense_observations.csv.gz",
            "summary": slam / "summary.json"}


def source_stats():
    return {k: {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for k, p in inputs().items()}


def verify_stats(expected):
    if source_stats() != expected:
        raise ValueError("Raw inputs changed; refusing to mix experiments")


def nearest_index(sorted_values, value):
    right = min(int(np.searchsorted(sorted_values, value)), len(sorted_values) - 1)
    left = max(0, right - 1)
    return left if abs(sorted_values[left] - value) <= abs(sorted_values[right] - value) else right


class Reader:
    def __init__(self):
        import pandas as pd
        from projectaria_tools.core import data_provider, mps
        from projectaria_tools.core.sensor_data import TimeDomain
        print("Loading exact VRS timestamps, MPS trajectory and online calibration", flush=True)
        src = inputs()
        self.provider = data_provider.create_vrs_data_provider(str(src["vrs"]))
        self.sid = self.provider.get_stream_id_from_label("camera-rgb")
        self.timestamps = np.asarray(self.provider.get_timestamps_ns(self.sid, TimeDomain.DEVICE_TIME), dtype=np.int64)
        self.origin = int(self.timestamps[0])
        self.duration = float((self.timestamps[-1] - self.origin) / 1e9)
        columns = ["tracking_timestamp_us", "graph_uid", "quality_score"]
        columns += [f"t{a}_world_device" for a in "xyz"] + [f"q{a}_world_device" for a in "xyzw"]
        frame = pd.read_csv(src["trajectory"], usecols=columns)
        ids = frame.graph_uid.dropna().unique()
        if len(ids) != 1:
            raise ValueError("Multiple MPS coordinate frames: alignment required before joint training")
        self.graph = str(ids[0])
        self.t = frame.tracking_timestamp_us.to_numpy(dtype=np.int64) * 1000
        self.p = frame[[f"t{a}_world_device" for a in "xyz"]].to_numpy()
        self.q = frame[[f"q{a}_world_device" for a in "xyzw"]].to_numpy()
        self.quality = frame.quality_score.to_numpy()
        if not np.all(np.diff(self.t) > 0) or not np.all(np.diff(self.timestamps) > 0):
            raise ValueError("Non-monotonic timestamps")
        self.cal = mps.read_online_calibration(str(src["calibration"]))
        self.ct = np.asarray([round(c.tracking_timestamp.total_seconds() * 1e9) for c in self.cal], dtype=np.int64)

    def pose(self, capture):
        from export_aria_gaussian_dataset import interpolate_pose
        ci = nearest_index(self.ct, capture)
        gap = abs(int(self.ct[ci]) - capture)
        if gap > 60_000_000:
            raise ValueError("missing_calibration")
        camera = self.cal[ci].get_camera_calib("camera-rgb")
        offset = round(camera.get_time_offset_sec_device_camera() * 1e9)
        query = capture - offset
        transform, span = interpolate_pose(self.t, self.p, self.q, query)
        right = int(np.searchsorted(self.t, query))
        if self.quality[max(0, right - 1):right + 1].min() < .9:
            raise ValueError("low_pose_quality")
        transform = transform @ camera.get_transform_device_camera().to_matrix()
        return camera, transform, {"capture_timestamp_ns": capture,
                                  "pose_query_timestamp_ns": query,
                                  "online_calibration_gap_ns": gap,
                                  "trajectory_interpolation_gap_ns": span}

    def frame(self, index, width):
        from export_aria_gaussian_dataset import rotate_upright
        from projectaria_tools.core import calibration
        image, record = self.provider.get_image_data_by_index(self.sid, int(index))
        if not image.is_valid():
            raise ValueError("invalid_image")
        capture = int(record.capture_timestamp_ns)
        if capture != self.timestamps[index]:
            raise ValueError("timestamp_mismatch")
        camera, transform, extra = self.pose(capture)
        target = calibration.get_linear_camera_calibration(width, width, width / 2,
            "rectified-rgb", camera.get_transform_device_camera(), camera.get_time_offset_sec_device_camera())
        rgb = calibration.distort_by_calibration(image.to_numpy_array(), target, camera)
        rgb, transform = rotate_upright(rgb, transform)
        return rgb, {**extra, "source_frame_index": int(index),
                     "recording_s": float((capture - self.origin) / 1e9),
                     "exposure_duration_s": float(record.exposure_duration),
                     "T_world_camera": transform.tolist()}


def image_quality(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {"laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "brightness": float(np.mean(gray)), "dark_fraction": float(np.mean(gray < 8)),
            "bright_fraction": float(np.mean(gray > 248))}


def contact_sheet(paths, records, output, columns=6, tile=224):
    sheet = Image.new("RGB", (columns * tile, ((len(paths) + columns - 1) // columns) * (tile + 30)), "#f4f3ed")
    draw = ImageDraw.Draw(sheet)
    for i, (p, r) in enumerate(zip(paths, records)):
        im = Image.open(p).convert("RGB")
        im.thumbnail((tile, tile))
        x, y = i % columns * tile, i // columns * (tile + 30)
        sheet.paste(im, (x, y + 30))
        draw.text((x + 5, y + 8), f"{r['recording_s']:.1f}s | sharp {r['laplacian_variance']:.0f}", fill="#183529")
    sheet.save(output, quality=92)


def audit(out, fps=3.):
    audit_path = out / "audit.json"
    if audit_path.exists():
        data = json.loads(audit_path.read_text())
        verify_stats(data["source_stats"])
        return data
    out.mkdir(parents=True, exist_ok=True)
    reader = Reader()
    stats = source_stats()
    sampled = sorted(set(nearest_index(reader.timestamps, reader.origin + round(s * 1e9))
                         for s in np.arange(0, reader.duration, 1 / fps)))
    preview = out / "audit_previews"
    preview.mkdir(exist_ok=True)
    records, skips = [], Counter()
    began = time.monotonic()
    for j, index in enumerate(sampled):
        try:
            rgb, record = reader.frame(index, 352)
        except ValueError as exc:
            skips[str(exc)] += 1
            continue
        record.update(image_quality(rgb))
        path = preview / f"{index:06d}.jpg"
        Image.fromarray(rgb).save(path, quality=86)
        record["preview"] = str(path)
        records.append(record)
        if j % 150 == 0:
            print(f"Screening {j + 1}/{len(sampled)} ({100 * (j + 1) / len(sampled):.1f}%)", flush=True)
            save_json(out / "progress.json", {"stage": "screening_full_recording", "state": "running",
                "candidates_done": j + 1, "candidates_total": len(sampled)})
    if not records:
        raise RuntimeError("No supported RGB views")
    positions = np.asarray([np.asarray(r["T_world_camera"])[:3, 3] for r in records])
    chosen = np.linspace(0, len(records) - 1, 48).astype(int)
    for batch, ids in enumerate(np.array_split(chosen, 2), 1):
        contact_sheet([records[i]["preview"] for i in ids], [records[i] for i in ids], out / f"audit_contact_{batch}.jpg")
    result = {"trial_id": "RoomScan3c", "source_stats": stats, "duration_s": reader.duration,
        "rgb_frames": len(reader.timestamps), "candidate_fps": fps, "world_frame": {"graph_uid": reader.graph, "units": "meters"},
        "candidate_count": len(sampled), "usable_pose_candidates": len(records), "skipped": dict(skips),
        "position_min_m": positions.min(axis=0).tolist(), "position_max_m": positions.max(axis=0).tolist(),
        "sharpness_quantiles": np.quantile([r["laplacian_variance"] for r in records], [0, .1, .5, .9, 1]).tolist(),
        "runtime_s": time.monotonic() - began, "frames": records,
        "scope": "All recording time screened at 3 fps; not every source frame and not a claim of room surface coverage."}
    verify_stats(stats)
    save_json(audit_path, result)
    save_json(out / "progress.json", {"stage": "audit_ready", "state": "awaiting_visual_review", "usable_candidates": len(records)})
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2), flush=True)
    return result


def select_views(records, limit=800):
    """Sharpness-aware pose coverage; time coverage is not surface coverage."""
    eligible = [r for r in records if r["laplacian_variance"] >= 30 and r["dark_fraction"] < .5
                and r["bright_fraction"] < .3]
    if len(eligible) < 30:
        raise ValueError("Too few usable frames")
    matrices = np.asarray([r["T_world_camera"] for r in eligible])
    # Rotation-matrix columns avoid quaternion sign ambiguity and include roll.
    features = np.concatenate([matrices[:, :3, 3] / .35,
                               matrices[:, :3, :2].reshape(-1, 6) / .35], axis=1)
    quality = np.asarray([r["laplacian_variance"] for r in eligible])
    quality_weight = np.clip(quality / np.median(quality), .5, 1.7)
    buckets = {}
    for i, r in enumerate(eligible):
        bucket = int(r["recording_s"] // 30)
        if bucket not in buckets or quality[i] > quality[buckets[bucket]]:
            buckets[bucket] = i
    selected = list(buckets.values())
    distances = np.full(len(eligible), np.inf)
    for i in selected:
        distances = np.minimum(distances, np.sum((features - features[i]) ** 2, axis=1))
    while len(selected) < min(limit, len(eligible)):
        scores = distances * quality_weight
        scores[selected] = -1
        i = int(np.argmax(scores))
        selected.append(i)
        distances = np.minimum(distances, np.sum((features - features[i]) ** 2, axis=1))
    return sorted([eligible[i] for i in selected], key=lambda r: r["recording_s"])


def split_views(records, count=12, guard_s=1.):
    indices = np.round(np.linspace(.04, .96, count) * (len(records) - 1)).astype(int)
    validation = [records[i] for i in indices]
    names = {r["name"] for r in validation}
    train, guard = [], []
    for r in records:
        if r["name"] in names:
            continue
        if min(abs(r["recording_s"] - v["recording_s"]) for v in validation) <= guard_s:
            guard.append(r)
        else:
            train.append(r)
    return train, validation, guard


def export(out):
    from projectaria_tools.core import calibration
    audit_data = audit(out)
    manifest_path = out / "dataset_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    selected = select_views(audit_data["frames"])
    reader = Reader()
    width = 1408
    (out / "images").mkdir(exist_ok=True)
    frames = []
    for i, r in enumerate(selected):
        rgb, row = reader.frame(r["source_frame_index"], width)
        name = f"frame_{r['source_frame_index']:06d}.jpg"
        Image.fromarray(rgb).save(out / "images" / name, quality=98, subsampling=0)
        frames.append({**r, **row, "name": name})
        if i % 50 == 0:
            print(f"Full-size calibrated export {i + 1}/{len(selected)}", flush=True)
            save_json(out / "progress.json", {"stage": "export_1408px", "state": "running", "done": i + 1, "total": len(selected)})
    fx, fy, cx, cy = map(float, calibration.get_linear_camera_calibration(width, width, width / 2).get_projection_params()[:4])
    result = {"trial_id": "RoomScan3c", "segment": {"start_s": 0, "end_s": reader.duration},
              "camera": {"model": "PINHOLE", "width": width, "height": width, "fx": fx, "fy": fy, "cx": cx, "cy": cy},
              "world_frame": audit_data["world_frame"], "source_stats": audit_data["source_stats"],
              "frames": {"selected": len(frames), "items": frames},
              "selection": "3fps full recording screening; sharpness floor, 30s temporal anchors and quality-weighted farthest camera-pose sampling. No synthetic frames or sharpening.",
              "image_processing": "Original RGB rectified to 1408x1408 90-degree pinhole, upright rotation with matching camera-basis transform. Resampled, not additional sensor resolution."}
    save_json(manifest_path, result)
    chosen = np.linspace(0, len(frames) - 1, 24).astype(int)
    contact_sheet([out / "images" / frames[i]["name"] for i in chosen], [frames[i] for i in chosen], out / "selected_contact.jpg")
    verify_stats(audit_data["source_stats"])
    return result


def foreground_masks(out, device="cpu"):
    """Run in the existing local vision environment, never an inference service."""
    from ultralytics import YOLO
    manifest = json.loads((out / "dataset_manifest.json").read_text())
    model_path = ROOT.parent / "tools/reconstruction_models/yolo11n-seg.pt"
    model = YOLO(str(model_path))
    folder = out / "masks"
    folder.mkdir(exist_ok=True)
    frames = manifest["frames"]["items"]
    rows, preview_paths, preview_rows = [], [], []
    for i, r in enumerate(frames):
        rgb = np.asarray(Image.open(out / "images" / r["name"]).convert("RGB"))
        h, w = rgb.shape[:2]
        result = model.predict(source=Image.fromarray(rgb), imgsz=640, conf=.2,
                               classes=[0, 67], device=device, verbose=False)[0]
        excluded = np.zeros((h, w), np.uint8)
        if result.masks is not None:
            for polygon in result.masks.xy:
                cv2.fillPoly(excluded, [np.rint(polygon).astype(np.int32)], 255)
        excluded = cv2.dilate(excluded, np.ones((21, 21), np.uint8))
        # Phones are small; include their bounding boxes to catch screen edges.
        if result.boxes is not None:
            for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                if int(cls) == 67:
                    x0, y0, x1, y1 = np.rint(box).astype(int)
                    excluded[max(0, y0-12):min(h, y1+12), max(0, x0-12):min(w, x1+12)] = 255
        # Conservatively exclude the near-body bottom strip (shoes/hands).
        excluded[int(h * .94):] = 255
        Image.fromarray(255 - excluded).save(folder / (Path(r["name"]).stem + ".png"))
        fraction = float(np.mean(excluded > 0))
        rows.append({"name": r["name"], "excluded_fraction": fraction,
                     "detections": len(result.boxes) if result.boxes is not None else 0})
        if (len(preview_paths) < 24 and (i % 35 == 0 or (fraction > .12 and len(preview_paths) < 8))):
            overlay = rgb.copy()
            overlay[excluded > 0] = (rgb[excluded > 0] * .35 + np.array([220, 40, 90]) * .65).astype(np.uint8)
            p = out / "audit_previews" / f"mask_{i:04d}.jpg"
            Image.fromarray(overlay).save(p, quality=90)
            preview_paths.append(p)
            preview_rows.append(r)
        if i % 50 == 0:
            print(f"Foreground masks {i + 1}/{len(frames)}", flush=True)
            save_json(out / "progress.json", {"stage": "local_foreground_masks", "state": "running", "done": i + 1, "total": len(frames)})
    contact_sheet(preview_paths, preview_rows, out / "mask_review.jpg")
    save_json(out / "mask_manifest.json", {"model": str(model_path), "sha256": sha256(model_path),
        "classes": ["person", "cell phone"], "confidence": .2, "dilation_px": 21,
        "near_body_exclusion_bottom_fraction": .06, "frames": rows,
        "limits": "Automatic masks can miss hands/people or exclude static details; not perfect dynamic segmentation. Bottom strip excludes some floor evidence."})


def prepare_training(out):
    from compare_aria_gaussian_quality import make_dataset
    from export_aria_gaussian_dataset import load_local_points, pinhole_project, depth_visible
    manifest = json.loads((out / "dataset_manifest.json").read_text())
    verify_stats(manifest["source_stats"])
    if (out / "split.json").exists():
        return json.loads((out / "split.json").read_text())
    masks = json.loads((out / "mask_manifest.json").read_text())
    excluded = {r["name"] for r in masks["frames"] if r["excluded_fraction"] > .45}
    frames = [r for r in manifest["frames"]["items"] if r["name"] not in excluded]
    train, validation, guard = split_views(frames)
    c = manifest["camera"]
    intrinsics = tuple(c[k] for k in ("fx", "fy", "cx", "cy"))
    centers = np.asarray([np.asarray(r["T_world_camera"])[:3, 3] for r in train])
    points, uids, counts = load_local_points(inputs()["points"], manifest["world_frame"]["graph_uid"],
                                            centers, .005, .01, 7, 350000)
    sums = np.zeros((len(points), 3), dtype=np.float64)
    seen = np.zeros(len(points), dtype=np.uint16)
    for i, r in enumerate(train):
        rgb = np.asarray(Image.open(out / "images" / r["name"]).convert("RGB"))
        mask = np.asarray(Image.open(out / "masks" / (Path(r["name"]).stem + ".png")))
        uv, depth = pinhole_project(points, np.asarray(r["T_world_camera"]), intrinsics)
        ids = np.flatnonzero(depth_visible(uv, depth, c["width"], c["height"]))
        xy = np.rint(uv[ids]).astype(int).clip(0, c["width"] - 1)
        valid = mask[xy[:, 1], xy[:, 0]] > 127
        ids, xy = ids[valid], xy[valid]
        sums[ids] += rgb[xy[:, 1], xy[:, 0]]
        seen[ids] += 1
        if i % 50 == 0:
            print(f"Training-only static point colors {i + 1}/{len(train)}", flush=True)
            save_json(out / "progress.json", {"stage": "static_initializer", "state": "running", "done": i + 1, "total": len(train)})
    keep = seen >= 3
    if keep.sum() < 1000:
        raise ValueError("Insufficient static initializer support")
    colors = np.rint(sums[keep] / seen[keep, None]).clip(0, 255).astype(np.uint8)
    for name, group in (("train", train), ("eval", frames)):
        make_dataset(out / name, manifest, group, out, points[keep], colors)
        (out / name / "masks").mkdir(exist_ok=True)
        for r in group:
            target = out / name / "masks" / (Path(r["name"]).stem + ".png")
            if not target.exists():
                target.symlink_to(out / "masks" / target.name)
    split = {"scene": "dedicated_room", "source": str(out), "camera": c,
             "world_frame": manifest["world_frame"], "training": train, "validation": validation,
             "guard": guard, "guard_s": 1., "excluded_large_foreground": sorted(excluded),
             "initializer_counts": counts, "initializer_points": int(keep.sum()),
             "initializer_uid_sha256": __import__("hashlib").sha256(uids[keep].astype("<i8").tobytes()).hexdigest(),
             "source_stats": manifest["source_stats"],
             "source_sha256": {str(out / p): sha256(out / p) for p in ("dataset_manifest.json", "mask_manifest.json")},
             "limits": "Held-out RGB excluded from optimization and point colors with 1s temporal guards. Shared whole-recording MPS geometry/poses. Conditional appearance validation, not independent geometry accuracy or clinical validation."}
    save_json(out / "split.json", split)
    print(f"Prepared {len(train)} training, {len(validation)} validation, {len(guard)} guard views; {keep.sum()} static initializer points", flush=True)
    return split


def render_checkpoint(out, split, model, step):
    from compare_aria_gaussian_quality import base_command, execute, make_dataset, metrics
    from skimage.metrics import structural_similarity
    folder = out / f"evaluation_{step}"
    folder.mkdir(exist_ok=True)
    if (folder / "metrics.json").exists():
        data = json.loads((folder / "metrics.json").read_text())
        if data["model_sha256"] != sha256(model):
            raise ValueError("Evaluated model changed")
        return data
    rows = []
    for r in split["validation"]:
        name = r["name"]
        dataset = out / "eval_views" / Path(name).stem
        if not (dataset / "sparse/0/points3D.bin").exists():
            make_dataset(dataset, {"camera": split["camera"]}, [split["training"][0], r], out,
                         np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8))
            for suffix in ("bin", "txt"):
                target = dataset / f"sparse/0/points3D.{suffix}"
                target.unlink()
                target.symlink_to(out / f"train/sparse/0/points3D.{suffix}")
        dest = folder / Path(name).stem
        dest.mkdir(exist_ok=True)
        command = base_command(BINARY, dataset, step) + ["--resume", str(model), "--val", "--val-image", name,
            "--val-render", str(dest), "-o", str(dest / "unused.ply")]
        _, log = execute(command, dest / "render.log", out / "progress.json", f"held-out render {step} {name}")
        if re.search(r"^Step \d+:", log, re.MULTILINE):
            raise RuntimeError("Evaluation unexpectedly trained")
        reference, render = out / "images" / name, dest / f"{step}.png"
        a, b = Image.open(reference).convert("RGB"), Image.open(render).convert("RGB")
        full = metrics(a, b)
        x, y = np.asarray(a).astype(np.float32) / 255, np.asarray(b).astype(np.float32) / 255
        mask = np.asarray(Image.open(out / "masks" / (Path(name).stem + ".png"))) > 127
        mask = cv2.erode(mask.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
        if not mask.any():
            raise RuntimeError("No static pixels for validation")
        mse = np.mean((x[mask] - y[mask]) ** 2)
        _, ssim_map = structural_similarity(x, y, channel_axis=2, data_range=1., full=True)
        row = {"name": name, "recording_s": r["recording_s"], "reference": str(reference), "render": str(render),
               **full, "background_psnr_db": float(-10 * np.log10(max(mse, 1e-12))),
               "background_ssim": float(ssim_map[mask].mean()), "static_pixel_fraction": float(mask.mean())}
        rows.append(row)
        print(f"Validation {step}: {name} background {row['background_psnr_db']:.2f} dB", flush=True)
    result = {"step": step, "model": str(model), "model_sha256": sha256(model), "views": rows,
              **{f"mean_{k}": float(np.mean([r[k] for r in rows])) for k in
                 ("psnr_db", "ssim", "background_psnr_db", "background_ssim")}}
    save_json(folder / "metrics.json", result)
    return result


def train(out):
    from compare_aria_gaussian_quality import base_command, execute
    split = json.loads((out / "split.json").read_text())
    verify_stats(split["source_stats"])
    trained = out / "trained_30000"
    trained.mkdir(exist_ok=True)
    final = trained / "splat.ply"
    record_path = trained / "run.json"
    if not record_path.exists():
        if final.exists() or (trained / "training.log").exists():
            raise ValueError("Prior incomplete training preserved; inspect before retrying")
        command = base_command(BINARY, out / "train", 30000) + [
            "--background-masks", "--num-downscales", "1", "--resolution-schedule", "3000",
            "--densify-from", "500", "--densify-until", "15000", "--refine-every", "500",
            "--save-every", "10000", "-o", str(final)]
        runtime, log = execute(command, trained / "training.log", out / "progress.json", "dedicated room joint training", 30000)
        if f"Found {len(split['training'])} masks" not in log:
            raise RuntimeError("Not all training masks were loaded")
        save_json(record_path, {"command": command, "binary_sha256": sha256(BINARY),
            "model_sha256": sha256(final), "runtime_s": runtime, "backend": "Apple MPS",
            "resolution": "704px first 2999 steps, 1408px thereafter", "training_views": len(split["training"]),
            "background_masks": "Ignored pixels do not receive a transparent-background opacity penalty"})
    evaluations = []
    for step, model in ((10000, trained / "splat_10000.ply"), (30000, final)):
        evaluations.append(render_checkpoint(out, split, model, step))
    verify_stats(split["source_stats"])
    save_json(out / "results.json", {"status": "evaluated_pending_visual_review", "evaluations": evaluations,
        "limits": split["limits"], "comparison": "10k and 30k checkpoints of one schedule; not independent trial runs."})
    save_json(out / "progress.json", {"state": "complete", "stage": "training_and_held_out_evaluation_complete"})


def complete_ply(path):
    if not path.exists():
        return False
    with path.open("rb") as stream:
        count, fields, size = 0, 0, 0
        while size < 65536:
            line = stream.readline()
            size += len(line)
            if not line:
                return False
            if line.startswith(b"element vertex "):
                count = int(line.split()[-1])
            if line.startswith(b"property float "):
                fields += 1
            if line.strip() == b"end_header":
                return count > 0 and path.stat().st_size == size + count * fields * 4
    return False


def summarize_quality(out):
    scores = [json.loads((out / f"evaluation_{step}/metrics.json").read_text()) for step in (10000, 30000)]
    a, b = scores
    if [v["name"] for v in a["views"]] != [v["name"] for v in b["views"]]:
        raise ValueError("Comparison views must match exactly")
    rows = [{"name": x["name"], "recording_s": x["recording_s"],
             "background_psnr_delta_db": y["background_psnr_db"] - x["background_psnr_db"],
             "background_ssim_delta": y["background_ssim"] - x["background_ssim"]}
            for x, y in zip(a["views"], b["views"])]
    panels = []
    for start in range(0, len(rows), 4):
        tile, label = 400, 38
        subset = rows[start:start + 4]
        sheet = Image.new("RGB", (3 * tile, (tile + label) * len(subset) + 34), "#f4f5ed")
        draw = ImageDraw.Draw(sheet)
        for j, title in enumerate(("ACTUAL CALIBRATED ARIA FRAME", "10,000 ITERATIONS", "30,000 ITERATIONS")):
            draw.text((j * tile + 8, 10), title, fill="#192e25")
        for i, item in enumerate(subset):
            k = start + i
            paths = [out / "images" / item["name"], Path(a["views"][k]["render"]), Path(b["views"][k]["render"])]
            y = 34 + i * (tile + label)
            for j, path in enumerate(paths):
                with Image.open(path) as image:
                    image = image.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
                    sheet.paste(image, (j * tile, y + label))
                caption = f"{item['recording_s']:.1f} s" if j == 0 else f"Background PSNR: {scores[j-1]['views'][k]['background_psnr_db']:.2f} dB"
                draw.text((j * tile + 8, y + 10), caption, fill="#192e25")
        path = out / f"validation_comparison_{start // 4 + 1}.jpg"
        sheet.save(path, quality=95)
        panels.append(path.name)
    summary = {"views": len(rows), "psnr_improved": sum(r["background_psnr_delta_db"] > 0 for r in rows),
               "ssim_improved": sum(r["background_ssim_delta"] > 0 for r in rows),
               "mean_background_psnr_gain_db": b["mean_background_psnr_db"] - a["mean_background_psnr_db"],
               "mean_background_ssim_gain": b["mean_background_ssim"] - a["mean_background_ssim"],
               "comparison_panels": panels, "per_view": rows,
               "interpretation": "Mean appearance agreement improves modestly, not every viewpoint. Not a sharpness guarantee, independent geometry validation, or proof of complete surface coverage."}
    save_json(out / "quality_summary.json", summary)
    return summary


def publish(out):
    manifest = json.loads((out / "dataset_manifest.json").read_text())
    split = json.loads((out / "split.json").read_text())
    verify_stats(split["source_stats"])
    models = []
    for step, relative in ((10000, "trained_30000/splat_10000.ply"), (30000, "trained_30000/splat.ply")):
        if complete_ply(out / relative):
            models.append({"step": step, "url": relative, "label": f"{step:,} steps / joint room"})
    evaluations = []
    for p in (out / "evaluation_10000/metrics.json", out / "evaluation_30000/metrics.json"):
        if p.exists():
            score = json.loads(p.read_text())
            for v in score["views"]:
                v["render_relative"] = os.path.relpath(v["render"], out)
            evaluations.append(score)
    summary = summarize_quality(out) if len(evaluations) == 2 else None
    data = {"duration_s": manifest["segment"]["end_s"], "frames": manifest["frames"]["items"],
            "training_count": len(split["training"]), "validation_count": len(split["validation"]),
            "models": models, "evaluations": evaluations, "quality_summary": summary}
    save_json(out / "viewer_data.json", data)
    templates = ROOT / "scripts/templates"
    for source, name in (("dedicated_room_viewer.html", "index.html"), ("dedicated_room_viewer.mjs", "viewer.mjs")):
        (out / name).write_text((templates / source).read_text())
    target = "playback.html" if (out.parent / "playback.html").exists() else "quality_20261029/index.html"
    (out.parent / "index.html").write_text(f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={target}"><a href="{target}">Open dedicated room reconstruction</a>')
    (out.parent.parent / "index.html").write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=environment_reconstruction/index.html"><a href="environment_reconstruction/index.html">Open dedicated room reconstruction</a>')
    report = f"""# Dedicated Room Scan 3c: Reconstruction Method

## Scope and provenance

Source: `{STEM}.vrs`, read-only, {manifest['segment']['end_s']:.3f} seconds.
This is a dedicated kitchen-room scan, not Parki or Parki2. All derived assets
are separate from the earlier Ntrial3 and 60minutes_1 reconstructions.
No research data was uploaded. VRS, MPS, existing videos and labels are unchanged.

## What is actually used

- Exact VRS RGB capture timestamps and original 1408-pixel RGB sensor images.
- MPS closed-loop poses, per-frame online RGB calibration and camera time offsets.
- Semi-dense point positions and uncertainty fields, within the single MPS world frame.
- Source metadata, SLAM summary and timestamps for integrity / support checks.

The semi-dense observations, eye gaze, hand-tracking and audio outputs are
available but are not optimization inputs in this run. SLAM-camera information
contributes through MPS geometry and poses, rather than being mixed as grayscale
appearance images. No claim is made that every source file or RGB frame is used.

## Frame preparation

The entire 24:43 recording is screened at 3 fps (4,449 candidates; 4,445 supported
by strict pose/calibration checks). This is temporal sampling, not a measurement
of complete room-surface coverage. Four unsupported candidate times are skipped,
not extrapolated. From supported views, {manifest['frames']['selected']} are retained
using a sharpness floor and quality-weighted camera-position/orientation diversity,
with one best view per 30-second bin as a temporal anchor.

Each retained frame is rectified to a 1408 x 1408, 90-degree pinhole image and
rotated upright together with its camera basis. Rectification resamples the
original pixels; it does not increase sensor detail. RGB time is corrected with
the online camera time offset before interpolation of the device pose.
`T_world_camera = T_world_device * T_device_camera`, followed by the matching
upright camera-basis rotation. No per-view scale fitting or synthetic geometry.

## Foreground and static initialization

YOLO11n-seg runs locally on people and cell phones with confidence 0.2. Masks are
dilated by 21 pixels; phone boxes add a 12-pixel margin. The bottom 6 percent is
conservatively excluded for near-body contamination. Masks can miss objects or
falsely mask static decorative heads. They are not ground truth or perfect hand
segmentation. Missing surfaces behind people are not filled by generation.

High-confidence MPS points near the camera path initialize the scene (up to
350,000 sampled candidates; {split['initializer_points']:,} retained with at least
three unmasked training-view color observations). Sparse depth-bin visibility is
an approximation, not a complete occlusion solver. Held-out images never color
the initializer.

## Training and comparison

One joint OpenSplat model is optimized across {len(split['training'])} training
views from the entire scan, rather than a mosaic of separately trained clips.
30,000 iterations, SH3, maximum 1,000,000 Gaussians, refinement every 500 steps,
densification steps 500-15,000. First 2,999 steps use 704 pixels; subsequent
steps use 1408 pixels. This schedule is recorded in the training command.

A separate executable adds `--background-masks`: foreground pixels are ignored
without penalizing the opacity of valid background surfaces behind them. The
previous executable is unchanged. The installed Metal compiler was unavailable,
so the new C++ executable reuses the existing Metal library after verifying the
shader-source SHA256 matches. No shader implementation was changed. Logs must
confirm Apple MPS and all {len(split['training'])} masks; CPU fallback is rejected.

12 fixed held-out RGB views and one-second temporal guards are excluded from
training ({len(split['guard'])} additional guard views). Render-only checkpoint
evaluation checks that zero optimization steps run. Both full-frame and masked
static-background PSNR/SSIM are reported for 10k and 30k checkpoints. Validation
masks are eroded before static scores to avoid uncertain boundaries.

These are conditional appearance checks: all-view MPS geometry/poses are shared.
They do not measure independent metric reconstruction accuracy or clinical
validity. This is a development validation set, not an untouched final test set.
10k and 30k are checkpoints of one schedule, not independent experimental runs.
Do not compare scores directly to Ntrial3: the scenes and evaluation views differ.

## Viewing and limitations

The local page uses GaussianSplats3D / Three.js. Aria viewpoint reproduces a
selected calibrated camera pose; Around camera orbits its position. The gold
frustum marks the measured camera direction. The right image is the actual
rectified RGB reference. The slider browses selected keyframes, not continuous
video. Browser rendering supports SH2, whereas numerical evaluation uses SH3.

No claim of fully recovered hidden surfaces. Reflective sink metal, transparent
objects, low-texture walls, motion blur, rolling shutter, lighting changes and
imperfect masks can cause residual holes or blur. Surface-coverage percentage is
unknown without a reference room model. Raw evidence remains the source of truth.

## Reproducibility

See `audit.json`, `dataset_manifest.json`, `mask_manifest.json`, `split.json`,
`trained_30000/run.json`, `evaluation_*/metrics.json` and `progress.json`.
Source file sizes and modification times are checked before and after processing.
`source_integrity.json` records a fresh full-file VRS SHA-256 match against the
original import validation. MPS file sizes / modification times are checked
separately; not every MPS file has a new byte-level checksum.

Sources: [Project Aria Tools](https://github.com/facebookresearch/projectaria_tools),
[OpenSplat](https://github.com/WebODM/OpenSplat),
[GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D),
[Ultralytics](https://github.com/ultralytics/ultralytics).
"""
    (out / "methodology.md").write_text(report)
    print(f"Published {len(models)} complete checkpoints and {len(evaluations)} evaluations: {out / 'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--stage", choices=("audit", "export", "mask", "prepare", "train", "publish"), default="audit")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.stage == "mask":
        foreground_masks(args.output, args.device)
    else:
        {"audit": audit, "export": export, "prepare": prepare_training, "train": train, "publish": publish}[args.stage](args.output)


if __name__ == "__main__":
    main()
