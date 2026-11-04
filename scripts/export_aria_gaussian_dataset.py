#!/usr/bin/env python3
"""Export an Aria VRS/MPS segment as a metric COLMAP dataset for 3DGS.

The exporter does not estimate a new trajectory. It preserves MPS closed-loop
poses and metric world coordinates, rectifies original RGB frames to one
pinhole camera, and writes the text COLMAP model consumed by OpenSplat and
other Gaussian-splatting trainers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "VRS_Pre_Processing" / "data"
RUNS_ROOT = ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-id", default="Ntrial3")
    parser.add_argument("--start-s", type=float, default=24.0)
    parser.add_argument("--end-s", type=float, default=60.0)
    parser.add_argument("--candidate-fps", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--focal-px", type=float, default=512.0)
    parser.add_argument("--min-translation-m", type=float, default=0.06)
    parser.add_argument("--min-rotation-deg", type=float, default=4.0)
    parser.add_argument("--max-frame-gap-s", type=float, default=0.75)
    parser.add_argument("--max-point-path-distance-m", type=float, default=8.0)
    parser.add_argument("--max-points", type=int, default=250_000)
    parser.add_argument("--min-color-views", type=int, default=2)
    parser.add_argument("--inv-dist-std-max", type=float, default=0.005)
    parser.add_argument("--dist-std-max", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def interpolate_pose(
    times_ns: np.ndarray,
    translations: np.ndarray,
    quaternions_xyzw: np.ndarray,
    query_ns: int,
    max_gap_ns: int = 20_000_000,
) -> tuple[np.ndarray, int]:
    """Interpolate within trajectory support without crossing a large gap."""
    if query_ns < times_ns[0] or query_ns > times_ns[-1]:
        raise ValueError("Image timestamp is outside the supported trajectory.")
    right = int(np.searchsorted(times_ns, query_ns, side="left"))
    left = max(0, right - 1)
    span = int(times_ns[right] - times_ns[left])
    if span > max_gap_ns:
        raise ValueError("Image timestamp crosses a large pose gap.")
    fraction = (query_ns - int(times_ns[left])) / span if span else 0.0
    transform = np.eye(4)
    transform[:3, 3] = (
        translations[left] * (1.0 - fraction) + translations[right] * fraction
    )
    if left == right:
        transform[:3, :3] = Rotation.from_quat(quaternions_xyzw[left]).as_matrix()
    else:
        transform[:3, :3] = Slerp(
            [0.0, 1.0], Rotation.from_quat(quaternions_xyzw[[left, right]])
        )([fraction]).as_matrix()[0]
    return transform, span


def camera_to_colmap(transform_world_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert camera-to-world T_W_C into COLMAP world-to-camera qvec/tvec."""
    rotation_world_camera = transform_world_camera[:3, :3]
    rotation_camera_world = rotation_world_camera.T
    translation = -rotation_camera_world @ transform_world_camera[:3, 3]
    q_xyzw = Rotation.from_matrix(rotation_camera_world).as_quat()
    q_wxyz = q_xyzw[[3, 0, 1, 2]]
    if q_wxyz[0] < 0:
        q_wxyz *= -1
    q_wxyz /= np.linalg.norm(q_wxyz)
    return q_wxyz, translation


def relative_motion(previous: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    distance = float(np.linalg.norm(current[:3, 3] - previous[:3, 3]))
    relative_rotation = previous[:3, :3].T @ current[:3, :3]
    angle = float(np.degrees(Rotation.from_matrix(relative_rotation).magnitude()))
    return distance, angle


def select_frame_indices(
    poses: list[np.ndarray],
    timestamps_ns: list[int],
    min_translation_m: float,
    min_rotation_deg: float,
    max_frame_gap_s: float,
) -> list[int]:
    """Keep overlap while avoiding near-duplicate frames from stationary views."""
    if not poses:
        return []
    selected = [0]
    for index in range(1, len(poses)):
        last = selected[-1]
        distance, angle = relative_motion(poses[last], poses[index])
        elapsed = (timestamps_ns[index] - timestamps_ns[last]) / 1e9
        if (
            distance >= min_translation_m
            or angle >= min_rotation_deg
            or elapsed >= max_frame_gap_s
        ):
            selected.append(index)
    if selected[-1] != len(poses) - 1:
        selected.append(len(poses) - 1)
    return selected


def pinhole_project(
    points_world: np.ndarray,
    transform_world_camera: np.ndarray,
    intrinsics: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into a pinhole image using row-vector NumPy math."""
    points_camera = (
        points_world - transform_world_camera[:3, 3]
    ) @ transform_world_camera[:3, :3]
    z = points_camera[:, 2]
    fx, fy, cx, cy = intrinsics
    uv = np.full((len(points_world), 2), np.nan, dtype=np.float64)
    front = z > 0
    uv[front, 0] = fx * points_camera[front, 0] / z[front] + cx
    uv[front, 1] = fy * points_camera[front, 1] / z[front] + cy
    return uv, z


def depth_visible(
    uv: np.ndarray,
    depth: np.ndarray,
    width: int,
    height: int,
    bin_px: int = 4,
    tolerance_m: float = 0.05,
) -> np.ndarray:
    """Reject points clearly behind another sparse point in the same image tile."""
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 0.35)
    valid &= (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    ids = np.flatnonzero(valid)
    visible = np.zeros(len(uv), dtype=bool)
    if not len(ids):
        return visible
    columns = math.ceil(width / bin_px)
    rows = math.ceil(height / bin_px)
    cells = np.floor(uv[ids] / bin_px).astype(int)
    nearest = np.full((rows, columns), np.inf, dtype=np.float32)
    np.minimum.at(nearest, (cells[:, 1], cells[:, 0]), depth[ids])
    nearest = minimum_filter(nearest, size=3, mode="constant", cval=np.inf)
    local_depth = nearest[cells[:, 1], cells[:, 0]]
    visible[ids] = depth[ids] <= local_depth + np.maximum(
        tolerance_m, depth[ids] * 0.01
    )
    return visible


def image_sharpness(image: np.ndarray) -> float:
    gray = image.astype(np.float32).mean(axis=2)
    horizontal = np.abs(np.diff(gray, axis=1)).mean()
    vertical = np.abs(np.diff(gray, axis=0)).mean()
    return float(horizontal + vertical)


def rotate_upright(
    image: np.ndarray, transform_world_camera: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate Gen1 RGB pixels clockwise and rotate the camera basis with them."""
    upright = np.rot90(image, k=3)
    camera_basis = np.eye(4)
    camera_basis[:3, :3] = Rotation.from_euler("z", -90, degrees=True).as_matrix()
    return upright, transform_world_camera @ camera_basis


def find_inputs(trial_id: str) -> dict[str, Path]:
    trial_root = DATA_ROOT / trial_id
    vrs_files = sorted(trial_root.glob("*.vrs"))
    trajectory_files = sorted(trial_root.glob("mps_*/slam/closed_loop_trajectory.csv"))
    calibration_files = sorted(trial_root.glob("mps_*/slam/online_calibration.jsonl"))
    points_files = sorted(trial_root.glob("mps_*/slam/semidense_points.csv.gz"))
    groups = {
        "VRS": vrs_files,
        "closed-loop trajectory": trajectory_files,
        "online calibration": calibration_files,
        "semi-dense points": points_files,
    }
    for label, files in groups.items():
        if len(files) != 1:
            raise FileNotFoundError(
                f"Expected one {label} for {trial_id}; found {len(files)}: {files}"
            )
    return {
        "vrs": vrs_files[0],
        "trajectory": trajectory_files[0],
        "calibration": calibration_files[0],
        "points": points_files[0],
    }


def load_trajectory(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    graph_ids = frame.graph_uid.dropna().unique().tolist()
    if len(graph_ids) != 1:
        raise ValueError("The trajectory must contain exactly one graph_uid/world frame.")
    times = frame.tracking_timestamp_us.to_numpy(dtype=np.int64) * 1000
    if not (np.diff(times) > 0).all():
        raise ValueError("Trajectory timestamps must be strictly increasing.")
    translations = frame[[f"t{axis}_world_device" for axis in "xyz"]].to_numpy()
    quaternions = frame[[f"q{axis}_world_device" for axis in "xyzw"]].to_numpy()
    return {
        "frame": frame,
        "graph_uid": str(graph_ids[0]),
        "times_ns": times,
        "translations": translations,
        "quaternions": quaternions,
    }


def make_contact_sheet(frame_paths: list[Path], labels: list[str], output: Path) -> None:
    sample_count = min(12, len(frame_paths))
    chosen = np.linspace(0, len(frame_paths) - 1, sample_count, dtype=int)
    tile_width, tile_height = 320, 350
    columns = 4
    rows = math.ceil(sample_count / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "#f3efe4")
    draw = ImageDraw.Draw(sheet)
    for tile, frame_index in enumerate(chosen):
        image = Image.open(frame_paths[frame_index]).convert("RGB")
        image.thumbnail((tile_width, tile_width))
        x = (tile % columns) * tile_width
        y = (tile // columns) * tile_height
        sheet.paste(image, (x, y + 30))
        draw.text((x + 8, y + 8), labels[frame_index], fill="#17231c")
    sheet.save(output, quality=92)


def load_local_points(
    path: Path,
    graph_uid: str,
    camera_centers: np.ndarray,
    inv_dist_std_max: float,
    dist_std_max: float,
    max_path_distance_m: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    columns = [
        "uid",
        "graph_uid",
        "px_world",
        "py_world",
        "pz_world",
        "inv_dist_std",
        "dist_std",
    ]
    frame = pd.read_csv(path, usecols=columns)
    raw_count = len(frame)
    same_graph = frame.graph_uid == graph_uid
    finite = np.isfinite(
        frame[["px_world", "py_world", "pz_world", "inv_dist_std", "dist_std"]]
    ).all(axis=1)
    quality = (
        same_graph
        & finite
        & (frame.inv_dist_std <= inv_dist_std_max)
        & (frame.dist_std <= dist_std_max)
    )
    quality_frame = frame.loc[quality].copy()
    points = quality_frame[["px_world", "py_world", "pz_world"]].to_numpy(
        dtype=np.float32
    )
    local = cKDTree(camera_centers).query(points, k=1, workers=-1)[0]
    local_mask = local <= max_path_distance_m
    local_frame = quality_frame.loc[local_mask].copy()
    local_count = len(local_frame)
    if local_count > max_points:
        rng = np.random.default_rng(20261017)
        chosen = np.sort(rng.choice(local_count, max_points, replace=False))
        local_frame = local_frame.iloc[chosen]
    points = local_frame[["px_world", "py_world", "pz_world"]].to_numpy(
        dtype=np.float32
    )
    uids = local_frame.uid.to_numpy(dtype=np.int64)
    if len(points) < 1_000:
        raise RuntimeError("Fewer than 1,000 quality points remain near this camera path.")
    return points, uids, {
        "raw": int(raw_count),
        "quality": int(len(quality_frame)),
        "local_before_cap": int(local_count),
        "sampled_for_initialization": int(len(points)),
    }


def color_points(
    points: np.ndarray,
    frames: list[dict[str, Any]],
    intrinsics: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    color_sum = np.zeros((len(points), 3), dtype=np.float64)
    view_count = np.zeros(len(points), dtype=np.uint16)
    for index, frame in enumerate(frames):
        image = np.asarray(Image.open(frame["path"]).convert("RGB"))
        uv, depth = pinhole_project(points, frame["transform_world_camera"], intrinsics)
        keep = depth_visible(uv, depth, width, height)
        ids = np.flatnonzero(keep)
        if len(ids):
            pixels = np.rint(uv[ids]).astype(int)
            pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
            pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
            color_sum[ids] += image[pixels[:, 1], pixels[:, 0]]
            view_count[ids] += 1
        if index % 20 == 0 or index == len(frames) - 1:
            print(
                f"Colored initializer from frame {index + 1}/{len(frames)}; "
                f"{int((view_count > 0).sum()):,} points seen.",
                flush=True,
            )
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    seen = view_count > 0
    colors[seen] = np.rint(color_sum[seen] / view_count[seen, None]).clip(
        0, 255
    ).astype(np.uint8)
    return colors, view_count


def write_colmap_model(
    output: Path,
    width: int,
    height: int,
    intrinsics: tuple[float, float, float, float],
    frames: list[dict[str, Any]],
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    sparse = output / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    fx, fy, cx, cy = intrinsics
    (sparse / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {width} {height} {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}\n"
    )
    image_lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] is intentionally empty; MPS supplies the metric initializer.",
    ]
    for image_id, frame in enumerate(frames, start=1):
        qvec, tvec = camera_to_colmap(frame["transform_world_camera"])
        values = [*qvec.tolist(), *tvec.tolist()]
        numeric = " ".join(f"{value:.17g}" for value in values)
        image_lines.extend([f"{image_id} {numeric} 1 {frame['name']}", ""])
    (sparse / "images.txt").write_text("\n".join(image_lines) + "\n")
    point_lines = [
        "# 3D point list with one line of data per point:",
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]",
        "# Empty tracks are valid for this MPS metric initializer.",
    ]
    for point_id, (point, color) in enumerate(zip(points, colors), start=1):
        point_lines.append(
            f"{point_id} {point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
            f"{int(color[0])} {int(color[1])} {int(color[2])} 0"
        )
    (sparse / "points3D.txt").write_text("\n".join(point_lines) + "\n")

    with (sparse / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", 1))
        # COLMAP camera model ID 1 is PINHOLE.
        stream.write(struct.pack("<IiQQ", 1, 1, width, height))
        stream.write(struct.pack("<dddd", fx, fy, cx, cy))
    with (sparse / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(frames)))
        for image_id, frame in enumerate(frames, start=1):
            qvec, tvec = camera_to_colmap(frame["transform_world_camera"])
            stream.write(struct.pack("<I", image_id))
            stream.write(struct.pack("<dddd", *qvec.tolist()))
            stream.write(struct.pack("<ddd", *tvec.tolist()))
            stream.write(struct.pack("<I", 1))
            stream.write(frame["name"].encode("utf-8") + b"\0")
            stream.write(struct.pack("<Q", 0))
    with (sparse / "points3D.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(points)))
        for point_id, (point, color) in enumerate(zip(points, colors), start=1):
            stream.write(struct.pack("<Q", point_id))
            stream.write(struct.pack("<ddd", *map(float, point)))
            stream.write(struct.pack("<BBB", *map(int, color)))
            stream.write(struct.pack("<dQ", 0.0, 0))


def write_readme(output: Path, summary: dict[str, Any]) -> None:
    segment = summary["segment"]
    text = f"""# {summary['trial_id']} metric Gaussian-splatting dataset

This is a first **local, motion-centered** scene dataset for PD Insighter. It
uses original VRS RGB pixels and the MPS closed-loop trajectory; it does not
infer timing or scale from a transcoded video.

## Segment

- Recording time: {segment['start_s']:.1f}-{segment['end_s']:.1f} s
- Purpose: walking segment with camera translation and overlapping views
- Rectified RGB frames: {summary['frames']['selected']}
- Pinhole image: {summary['camera']['width']} x {summary['camera']['height']} px
- Initial MPS points with at least {summary['points']['minimum_color_views']} RGB views: {summary['points']['written']:,}

## Coordinate chain

For every exact RGB capture timestamp, the exporter subtracts the online
camera time offset, interpolates `T_World_Device` from the closed-loop
trajectory, and applies online `T_Device_RGB`. The resulting camera-to-world
pose is inverted only when written in COLMAP's world-to-camera convention.
The MPS world frame and meter scale are preserved.

## Structure

```text
images/                 rectified pinhole RGB frames
sparse/0/*.bin          OpenSplat-compatible COLMAP binary model
sparse/0/*.txt          auditable text copy of the same model
dataset_manifest.json   provenance, checks and exact transforms
contact_sheet.jpg       visual coverage inspection
```

## Important limitation

This folder is a validated **training input**, not yet a completed Gaussian
scene. Sparse occlusion checks and averaged initializer colors do not replace
3DGS optimization. Hands, people, moving objects, rolling shutter and lighting
change still require review. The next gate is to train OpenSplat, inspect novel
views and geometry alignment, and only then embed a splat viewer in PD
Insighter.
"""
    (output / "README.md").write_text(text)


def load_export_context(trial_id):
    """Reuse expensive VRS indexing and MPS loading across local submaps."""
    from projectaria_tools.core import data_provider, mps

    inputs = find_inputs(trial_id)
    print("Loading trajectory and online calibration...", flush=True)
    trajectory = load_trajectory(inputs["trajectory"])
    calibrations = mps.read_online_calibration(str(inputs["calibration"]))
    calibration_ns = np.array(
        [round(item.tracking_timestamp.total_seconds() * 1e9) for item in calibrations],
        dtype=np.int64,
    )
    if not len(calibration_ns) or not np.all(np.diff(calibration_ns) > 0):
        raise ValueError("Online calibration timestamps must be strictly increasing")
    provider = data_provider.create_vrs_data_provider(str(inputs["vrs"]))
    if provider is None:
        raise RuntimeError("VRS provider could not open the recording.")
    return inputs, trajectory, calibrations, calibration_ns, provider


def export_dataset(args, context=None) -> None:
    from projectaria_tools.core import calibration, data_provider, mps
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

    if not 0 <= args.start_s < args.end_s:
        raise ValueError("Require 0 <= start-s < end-s.")
    if args.candidate_fps <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("Frame rate and output dimensions must be positive.")
    if args.width != args.height:
        raise ValueError("The upright Gen1 RGB export currently requires a square image.")
    inputs = find_inputs(args.trial_id)
    output = args.output_dir or (
        RUNS_ROOT
        / args.trial_id
        / "environment_reconstruction"
        / "gaussian_splatting_trial"
    )
    output = output.resolve()
    if (output / "dataset_manifest.json").exists():
        raise FileExistsError(f"Preserving existing export; choose a new output directory: {output}")
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    context = context or load_export_context(args.trial_id)
    context_inputs, trajectory, calibrations, calibration_ns, provider = context
    if context_inputs != inputs:
        raise ValueError("Export context belongs to different trial inputs")
    rgb_stream = provider.get_stream_id_from_label("camera-rgb")
    _, first_record = provider.get_image_data_by_index(rgb_stream, 0)
    origin_ns = int(first_record.capture_timestamp_ns)

    candidates: list[dict[str, Any]] = []
    seen_capture_times: set[int] = set()
    skipped_candidates = {
        "duplicate_capture_timestamp": 0,
        "missing_nearby_calibration": 0,
        "unsupported_pose": 0,
        "low_pose_quality": 0,
    }
    requested_times = np.arange(
        args.start_s, args.end_s + 0.5 / args.candidate_fps, 1.0 / args.candidate_fps
    )
    for requested_s in requested_times:
        query_ns = origin_ns + round(float(requested_s) * 1e9)
        image_data, record = provider.get_image_data_by_time_ns(
            rgb_stream,
            query_ns,
            TimeDomain.DEVICE_TIME,
            TimeQueryOptions.CLOSEST,
        )
        capture_ns = int(record.capture_timestamp_ns)
        if capture_ns in seen_capture_times:
            skipped_candidates["duplicate_capture_timestamp"] += 1
            continue
        seen_capture_times.add(capture_ns)
        image = image_data.to_numpy_array()
        if not image_data.is_valid() or image.shape[:2] != (1408, 1408):
            raise ValueError("An RGB frame is invalid or has unexpected dimensions.")
        calibration_index = int(np.argmin(np.abs(calibration_ns - capture_ns)))
        calibration_gap = abs(int(calibration_ns[calibration_index]) - capture_ns)
        if calibration_gap > 60_000_000:
            skipped_candidates["missing_nearby_calibration"] += 1
            continue
        source_camera = calibrations[calibration_index].get_camera_calib("camera-rgb")
        if source_camera is None:
            raise ValueError("Online calibration has no RGB camera.")
        corrected_ns = capture_ns - round(
            source_camera.get_time_offset_sec_device_camera() * 1e9
        )
        try:
            transform_world_device, pose_gap = interpolate_pose(
                trajectory["times_ns"],
                trajectory["translations"],
                trajectory["quaternions"],
                corrected_ns,
            )
        except ValueError:
            skipped_candidates["unsupported_pose"] += 1
            continue
        pose_index = int(np.searchsorted(trajectory["times_ns"], corrected_ns))
        quality_slice = trajectory["frame"].quality_score.iloc[
            max(0, pose_index - 1) : pose_index + 1
        ]
        if len(quality_slice) == 0 or float(quality_slice.min()) < 0.9:
            skipped_candidates["low_pose_quality"] += 1
            continue
        transform_world_camera = (
            transform_world_device
            @ source_camera.get_transform_device_camera().to_matrix()
        )
        candidates.append(
            {
                "capture_ns": capture_ns,
                "corrected_ns": corrected_ns,
                "recording_s": (capture_ns - origin_ns) / 1e9,
                "requested_s": float(requested_s),
                "calibration_index": calibration_index,
                "calibration_gap_ns": calibration_gap,
                "pose_gap_ns": pose_gap,
                "exposure_duration_s": float(record.exposure_duration),
                "transform_world_camera": transform_world_camera,
                "source_camera": source_camera,
                "raw_image": image,
            }
        )

    selected_indices = select_frame_indices(
        [frame["transform_world_camera"] for frame in candidates],
        [frame["capture_ns"] for frame in candidates],
        args.min_translation_m,
        args.min_rotation_deg,
        args.max_frame_gap_s,
    )
    selected = [candidates[index] for index in selected_indices]
    if len(selected) < 12:
        raise RuntimeError("Frame selection retained fewer than 12 overlapping views.")

    written_frames: list[dict[str, Any]] = []
    for index, frame in enumerate(selected, start=1):
        destination_camera = calibration.get_linear_camera_calibration(
            args.width,
            args.height,
            args.focal_px,
            "rectified-rgb",
            frame["source_camera"].get_transform_device_camera(),
            frame["source_camera"].get_time_offset_sec_device_camera(),
        )
        rectified = calibration.distort_by_calibration(
            frame.pop("raw_image"), destination_camera, frame.pop("source_camera")
        )
        rectified, frame["transform_world_camera"] = rotate_upright(
            rectified, frame["transform_world_camera"]
        )
        name = f"frame_{index:05d}.jpg"
        path = images_dir / name
        Image.fromarray(rectified).save(path, quality=94, subsampling=0)
        frame.update(
            {
                "name": name,
                "path": path,
                "sharpness": image_sharpness(rectified),
            }
        )
        written_frames.append(frame)
        if index % 20 == 0 or index == len(selected):
            print(f"Rectified frame {index}/{len(selected)}.", flush=True)

    projection_params = calibration.get_linear_camera_calibration(
        args.width, args.height, args.focal_px
    ).get_projection_params()
    intrinsics = tuple(float(value) for value in projection_params[:4])
    camera_centers = np.array(
        [frame["transform_world_camera"][:3, 3] for frame in written_frames]
    )
    points, point_uids, point_counts = load_local_points(
        inputs["points"],
        trajectory["graph_uid"],
        camera_centers,
        args.inv_dist_std_max,
        args.dist_std_max,
        args.max_point_path_distance_m,
        args.max_points,
    )
    colors, view_count = color_points(
        points, written_frames, intrinsics, args.width, args.height
    )
    keep = view_count >= args.min_color_views
    if int(keep.sum()) < 1_000:
        raise RuntimeError(
            "Fewer than 1,000 points are visible in the requested number of RGB views."
        )
    points, colors, point_uids, view_count = (
        points[keep],
        colors[keep],
        point_uids[keep],
        view_count[keep],
    )
    write_colmap_model(
        output, args.width, args.height, intrinsics, written_frames, points, colors
    )
    make_contact_sheet(
        [frame["path"] for frame in written_frames],
        [f"{frame['recording_s']:.3f} s" for frame in written_frames],
        output / "contact_sheet.jpg",
    )

    distances = np.linalg.norm(np.diff(camera_centers, axis=0), axis=1)
    rotations = [
        relative_motion(
            written_frames[index - 1]["transform_world_camera"],
            written_frames[index]["transform_world_camera"],
        )[1]
        for index in range(1, len(written_frames))
    ]
    manifest_frames = []
    for image_id, frame in enumerate(written_frames, start=1):
        qvec, tvec = camera_to_colmap(frame["transform_world_camera"])
        manifest_frames.append(
            {
                "image_id": image_id,
                "name": frame["name"],
                "requested_s": frame["requested_s"],
                "recording_s": frame["recording_s"],
                "capture_timestamp_ns": frame["capture_ns"],
                "pose_query_timestamp_ns": frame["corrected_ns"],
                "camera_time_offset_ns": frame["capture_ns"] - frame["corrected_ns"],
                "online_calibration_gap_ns": frame["calibration_gap_ns"],
                "trajectory_interpolation_gap_ns": frame["pose_gap_ns"],
                "exposure_duration_s": frame["exposure_duration_s"],
                "sharpness": frame["sharpness"],
                "T_world_camera": frame["transform_world_camera"].tolist(),
                "colmap_qvec_wxyz": qvec.tolist(),
                "colmap_tvec": tvec.tolist(),
            }
        )
    summary = {
        "status": "validated_training_input_not_yet_a_trained_gaussian_scene",
        "trial_id": args.trial_id,
        "segment": {"start_s": args.start_s, "end_s": args.end_s},
        "inputs": {key: str(path.resolve()) for key, path in inputs.items()},
        "world_frame": {
            "graph_uid": trajectory["graph_uid"],
            "units": "meters",
            "pose_source": "MPS closed-loop trajectory",
        },
        "camera": {
            "model": "PINHOLE",
            "width": args.width,
            "height": args.height,
            "fx": intrinsics[0],
            "fy": intrinsics[1],
            "cx": intrinsics[2],
            "cy": intrinsics[3],
            "source_model": "per-frame online calibrated Aria RGB fisheye",
            "orientation": "clockwise 90 degree pixel rotation with matched camera-basis rotation",
        },
        "frames": {
            "candidate": len(candidates),
            "skipped_candidates": skipped_candidates,
            "selected": len(written_frames),
            "candidate_fps": args.candidate_fps,
            "selection": {
                "minimum_translation_m": args.min_translation_m,
                "minimum_rotation_deg": args.min_rotation_deg,
                "maximum_gap_s": args.max_frame_gap_s,
            },
            "adjacent_translation_m": {
                "median": float(np.median(distances)),
                "p90": float(np.quantile(distances, 0.9)),
                "maximum": float(distances.max()),
            },
            "adjacent_rotation_deg": {
                "median": float(np.median(rotations)),
                "p90": float(np.quantile(rotations, 0.9)),
                "maximum": float(np.max(rotations)),
            },
            "sharpness": {
                "minimum": float(min(frame["sharpness"] for frame in written_frames)),
                "median": float(
                    np.median([frame["sharpness"] for frame in written_frames])
                ),
            },
            "items": manifest_frames,
        },
        "points": {
            **point_counts,
            "minimum_color_views": args.min_color_views,
            "written": int(len(points)),
            "color_view_count_median": float(np.median(view_count)),
            "source_uids_sha256": hashlib.sha256(
                point_uids.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
        },
        "checks": {
            "strictly_increasing_capture_times": bool(
                np.all(np.diff([frame["capture_ns"] for frame in written_frames]) > 0)
            ),
            "single_mps_world_frame": True,
            "quaternion_norm_max_error": float(
                max(
                    abs(np.linalg.norm(item["colmap_qvec_wxyz"]) - 1.0)
                    for item in manifest_frames
                )
            ),
            "all_images_exist": all(frame["path"].exists() for frame in written_frames),
        },
        "software": {
            "projectaria-tools": importlib.metadata.version("projectaria-tools"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "limitations": [
            "No 3D Gaussian optimization has been run yet.",
            "Sparse depth-bin visibility is not a dense occlusion solution.",
            "Moving people, hands and transient objects are not segmented in this first export.",
            "Frames use a global pose at exposure time; RGB rolling shutter is not corrected row by row.",
            "Initializer colors are multi-view averages and are not the final rendered appearance.",
        ],
    }
    (output / "dataset_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_readme(output, summary)
    print(f"Wrote metric Gaussian-splatting dataset: {output}")
    print(
        f"Frames {len(written_frames)} | points {len(points):,} | "
        f"median adjacent baseline {np.median(distances):.3f} m"
    )


def main():
    export_dataset(parse_args())


if __name__ == "__main__":
    main()
