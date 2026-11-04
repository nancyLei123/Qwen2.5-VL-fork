#!/usr/bin/env python3
"""Build an auditable 3D environment prototype from Aria MPS SLAM outputs.

The output intentionally separates measured MPS points from the experimental
surface inferred from their local density. The interactive page synchronizes
the Aria video with the closed-loop head trajectory so scene geometry can be
checked against the recording instead of accepted from appearance alone.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import trimesh
from plotly.offline.offline import get_plotlyjs
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


WORKSPACE_ROOT = Path("/Users/leijianan/gen1_glasses")
REPO_ROOT = WORKSPACE_ROOT / "Qwen2.5-VL-fork"
DATA_ROOT = WORKSPACE_ROOT / "VRS_Pre_Processing" / "data"
RUNS_ROOT = REPO_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an exploratory Aria environment reconstruction."
    )
    parser.add_argument("--trial-id", required=True, help="Run folder name, e.g. Ntrial3")
    parser.add_argument("--points", type=Path, help="Override semidense_points.csv.gz")
    parser.add_argument("--trajectory", type=Path, help="Override closed_loop_trajectory.csv")
    parser.add_argument("--aria-video", type=Path, help="Override synchronized Aria MP4")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--inv-dist-std-max", type=float, default=0.005)
    parser.add_argument("--dist-std-max", type=float, default=0.01)
    parser.add_argument(
        "--max-path-distance-m",
        type=float,
        default=8.0,
        help="Reject points farther than this from the sampled device path.",
    )
    parser.add_argument("--voxel-size-m", type=float, default=0.14)
    parser.add_argument("--density-sigma", type=float, default=1.0)
    parser.add_argument("--density-level", type=float, default=0.15)
    parser.add_argument("--marching-step", type=int, default=2)
    parser.add_argument("--min-component-faces", type=int, default=500)
    parser.add_argument("--display-points", type=int, default=80_000)
    return parser.parse_args()


def _manifest_paths(trial_id: str) -> dict[str, Path]:
    manifest_path = RUNS_ROOT / trial_id / "trial_manifest.json"
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text())

    def get(*keys: str) -> Path | None:
        value: Any = payload
        for key in keys:
            value = value.get(key, {}) if isinstance(value, dict) else {}
        if isinstance(value, dict):
            value = value.get("path")
        return Path(value) if isinstance(value, str) and value else None

    paths = {
        "points": get("mps_outputs", "semidense_points_csv_gz"),
        "trajectory": get("mps_outputs", "closed_loop_trajectory_csv"),
        "aria_video": get("extracted_media", "video_mp4"),
    }
    return {key: value for key, value in paths.items() if value is not None}


def _find_single(patterns: list[str], label: str) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in DATA_ROOT.glob(pattern) if path.is_file())
    unique = sorted(set(matches))
    if len(unique) != 1:
        rendered = "\n  ".join(str(path) for path in unique[:20]) or "(none)"
        raise FileNotFoundError(f"Expected one {label}; found {len(unique)}:\n  {rendered}")
    return unique[0]


def resolve_inputs(args: argparse.Namespace) -> dict[str, Path]:
    manifest = _manifest_paths(args.trial_id)
    points = args.points or manifest.get("points")
    trajectory = args.trajectory or manifest.get("trajectory")
    if points is None or not points.exists():
        points = _find_single(
            [
                f"{args.trial_id}/mps_*/slam/semidense_points.csv.gz",
                f"{args.trial_id}/**/slam/semidense_points.csv.gz",
            ],
            "semi-dense point cloud",
        )
    if trajectory is None or not trajectory.exists():
        trajectory = _find_single(
            [
                f"{args.trial_id}/mps_*/slam/closed_loop_trajectory.csv",
                f"{args.trial_id}/**/slam/closed_loop_trajectory.csv",
            ],
            "closed-loop trajectory",
        )
    aria_video = args.aria_video or manifest.get("aria_video")
    if aria_video is None or not aria_video.exists():
        aria_video = RUNS_ROOT / args.trial_id / "downward_master_session" / "aria_session_video.mp4"
    if not aria_video.exists():
        raise FileNotFoundError(f"Synchronized Aria video not found: {aria_video}")
    output_dir = args.output_dir or RUNS_ROOT / args.trial_id / "environment_reconstruction"
    return {
        "points": points.resolve(),
        "trajectory": trajectory.resolve(),
        "aria_video": aria_video.resolve(),
        "output_dir": output_dir.resolve(),
    }


def read_video_metadata(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open Aria video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    codec_value = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
    codec = "".join(chr((codec_value >> (8 * index)) & 0xFF) for index in range(4))
    capture.release()
    duration = frames / fps if fps > 0 else 0.0
    return {
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
        "codec": codec,
        "duration_s": duration,
    }


def ensure_browser_video(source: Path, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    source_metadata = read_video_metadata(source)
    if str(source_metadata["codec"]).lower() in {"h264", "avc1"}:
        return source, {"transcoded": False, "source": source_metadata}

    target = output_dir / "browser_aria_video.mp4"
    if target.exists():
        target_metadata = read_video_metadata(target)
        if (
            target_metadata["frames"] == source_metadata["frames"]
            and abs(float(target_metadata["fps"]) - float(source_metadata["fps"])) < 0.01
            and target.stat().st_mtime_ns >= source.stat().st_mtime_ns
        ):
            return target, {
                "transcoded": True,
                "source": source_metadata,
                "browser_asset": target_metadata,
                "note": "OpenCV browser copy has no audio track; original media is unchanged.",
            }

    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*"avc1"),
        float(source_metadata["fps"]),
        (int(source_metadata["width"]), int(source_metadata["height"])),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not create the browser-compatible H.264 video.")
    written = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
        if written % 1_000 == 0:
            print(f"Transcoded {written:,}/{source_metadata['frames']:,} Aria frames...")
    capture.release()
    writer.release()
    if written != int(source_metadata["frames"]):
        target.unlink(missing_ok=True)
        raise RuntimeError(
            f"H.264 conversion stopped at {written:,}/{source_metadata['frames']:,} frames."
        )
    target_metadata = read_video_metadata(target)
    return target, {
        "transcoded": True,
        "source": source_metadata,
        "browser_asset": target_metadata,
        "note": "OpenCV conversion preserves video frames but does not copy an audio track.",
    }


def load_and_filter_points(
    points_path: Path,
    trajectory_xyz: np.ndarray,
    inv_dist_std_max: float,
    dist_std_max: float,
    max_path_distance_m: float,
    expected_graph_uid: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    columns = ["px_world", "py_world", "pz_world", "inv_dist_std", "dist_std"]
    frame = pd.read_csv(points_path, usecols=columns + ["graph_uid"])
    raw_count = len(frame)
    graph_ids = frame["graph_uid"].dropna().unique().tolist()
    if len(graph_ids) != 1 or (expected_graph_uid and graph_ids != [expected_graph_uid]):
        raise ValueError("Points and trajectory must share one graph_uid/world frame.")
    finite = np.isfinite(frame[columns]).all(axis=1)
    quality = (
        finite
        & (frame["inv_dist_std"] <= inv_dist_std_max)
        & (frame["dist_std"] <= dist_std_max)
    )
    quality_points = frame.loc[quality, ["px_world", "py_world", "pz_world"]].to_numpy(
        dtype=np.float32
    )
    del frame
    if len(quality_points) < 1_000:
        raise RuntimeError("Quality filtering retained too few points for reconstruction.")

    path_sample_step = max(1, len(trajectory_xyz) // 1_000)
    sampled_path = trajectory_xyz[::path_sample_step]
    distances = cKDTree(sampled_path).query(quality_points, k=1, workers=-1)[0]
    path_mask = distances <= max_path_distance_m
    filtered = quality_points[path_mask]
    if len(filtered) < 1_000:
        raise RuntimeError("Path filtering retained too few points for reconstruction.")

    # Optional spatial cropping is a heuristic, not proof that removed points are wrong.
    lower = np.quantile(filtered, 0.001, axis=0)
    upper = np.quantile(filtered, 0.999, axis=0)
    bounds_mask = np.all((filtered >= lower) & (filtered <= upper), axis=1)
    filtered = filtered[bounds_mask]
    if len(filtered) < 1_000:
        raise RuntimeError("Spatial filtering retained too few points for reconstruction.")

    summary = {
        "raw_point_count": int(raw_count),
        "graph_uid": graph_ids[0],
        "quality_filtered_count": int(len(quality_points)),
        "path_filtered_count": int(path_mask.sum()),
        "final_point_count": int(len(filtered)),
        "retained_percent": round(100.0 * len(filtered) / raw_count, 3),
        "bounds_min_m": lower.round(5).tolist(),
        "bounds_max_m": upper.round(5).tolist(),
        "nearest_path_distance_m": {
            "median": round(float(np.median(distances)), 4),
            "p95": round(float(np.quantile(distances, 0.95)), 4),
            "max": round(float(np.max(distances)), 4),
        },
    }
    return filtered, summary


def build_density_mesh(
    points: np.ndarray,
    voxel_size_m: float,
    density_sigma: float,
    density_level: float,
    marching_step: int,
    min_component_faces: int,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    lower = np.floor(np.min(points, axis=0) / voxel_size_m) * voxel_size_m - voxel_size_m
    upper = np.ceil(np.max(points, axis=0) / voxel_size_m) * voxel_size_m + voxel_size_m
    edges = [
        np.arange(lower[axis], upper[axis] + voxel_size_m, voxel_size_m)
        for axis in range(3)
    ]
    shape = tuple(len(edge) - 1 for edge in edges)
    voxel_count = int(np.prod(shape))
    if voxel_count > 120_000_000:
        raise RuntimeError(
            f"Voxel grid would contain {voxel_count:,} cells. Increase --voxel-size-m."
        )
    density, _ = np.histogramdd(points, bins=edges)
    density = gaussian_filter(density.astype(np.float32), sigma=density_sigma)
    if density_level >= float(np.max(density)):
        raise RuntimeError("Density level is above the reconstructed volume maximum.")
    vertices, faces, _, _ = marching_cubes(
        density,
        level=density_level,
        spacing=(voxel_size_m, voxel_size_m, voxel_size_m),
        step_size=marching_step,
        allow_degenerate=False,
    )
    vertices += lower + voxel_size_m / 2.0
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    components = trimesh.graph.connected_components(
        mesh.face_adjacency, min_len=min_component_faces
    )
    if components:
        keep_faces = np.concatenate(components)
        face_mask = np.zeros(len(mesh.faces), dtype=bool)
        face_mask[keep_faces] = True
        mesh.update_faces(face_mask)
        mesh.remove_unreferenced_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals(multibody=True)

    summary = {
        "voxel_size_m": voxel_size_m,
        "voxel_grid_shape": list(shape),
        "voxel_count": voxel_count,
        "density_sigma_voxels": density_sigma,
        "density_isosurface_level": density_level,
        "marching_cubes_step": marching_step,
        "minimum_component_faces": min_component_faces,
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "retained_components": int(len(components)),
        "watertight": bool(mesh.is_watertight),
    }
    return mesh, summary


def _height_colors(values: np.ndarray, bounds: tuple[float, float] | None = None) -> np.ndarray:
    low, high = bounds if bounds else (float(np.min(values)), float(np.max(values)))
    normalized = np.clip((values - low) / max(high - low, 1e-9), 0.0, 1.0)
    colors = plt.get_cmap("viridis")(normalized)
    return (colors * 255).astype(np.uint8)


def export_geometry(
    output_dir: Path, points: np.ndarray, mesh: trimesh.Trimesh
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    color_bounds = (float(np.min(points[:, 2])), float(np.max(points[:, 2])))
    point_colors = _height_colors(points[:, 2], color_bounds)
    point_cloud = trimesh.points.PointCloud(points, colors=point_colors)
    point_path = output_dir / "filtered_mps_point_cloud.ply"
    point_cloud.export(point_path)

    mesh.visual.vertex_colors = _height_colors(np.asarray(mesh.vertices)[:, 2], color_bounds)
    mesh_ply_path = output_dir / "experimental_surface_mesh.ply"
    mesh_glb_path = output_dir / "experimental_surface_mesh.glb"
    mesh.export(mesh_ply_path)
    mesh.export(mesh_glb_path)
    return {
        "point_cloud_ply": point_path,
        "mesh_ply": mesh_ply_path,
        "mesh_glb": mesh_glb_path,
    }


def _sample_evenly(array: np.ndarray, count: int) -> np.ndarray:
    if len(array) <= count:
        return array
    indices = np.linspace(0, len(array) - 1, count, dtype=np.int64)
    return array[indices]


def resolve_video_timing(
    video_path: Path, trajectory: pd.DataFrame, trial_dir: Path
) -> dict[str, Any]:
    """Use device timestamps, never stretch poses to fit a video's duration."""
    extraction_video = video_path
    trim_s = 0.0
    build_path = video_path.with_suffix(".build.json")
    if build_path.exists():
        build = json.loads(build_path.read_text())
        extraction_video = Path(build.get("source_video_path", video_path))
        trim_s = float(build.get("video_trim_start_s", 0.0))
    metadata_path = extraction_video.parent / "audio_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    video_start_ns = metadata.get("video_start_device_ns")
    first_us = int(trajectory["tracking_timestamp_us"].iloc[0])
    last_us = int(trajectory["tracking_timestamp_us"].iloc[-1])
    verified = isinstance(video_start_ns, int) and video_start_ns > 0
    first_video_s = (first_us * 1_000 - video_start_ns) / 1e9 - trim_s if verified else None
    last_video_s = (last_us * 1_000 - video_start_ns) / 1e9 - trim_s if verified else None
    dashboard_build = trial_dir / "downward_master_session" / "aria_session_video.build.json"
    dashboard = json.loads(dashboard_build.read_text()) if dashboard_build.exists() else {}
    dashboard_trim = dashboard.get("video_trim_start_s")
    session_manifest = trial_dir / "session_dashboard_manifest.json"
    session = json.loads(session_manifest.read_text()) if session_manifest.exists() else {}
    return {
        "mode": "device_timestamp_offset" if verified else "unverified",
        "method": "(SLAM device timestamp - RGB start device timestamp) minus video trim",
        "video_start_device_ns": video_start_ns if verified else None,
        "metadata_path": str(metadata_path) if verified else None,
        "video_trim_start_s": trim_s,
        "first_pose_video_s": first_video_s,
        "last_pose_video_s": last_video_s,
        "dashboard_time_offset_s": trim_s - float(dashboard_trim) if dashboard_trim is not None else None,
        "dashboard_duration_s": session.get("session_duration_s"),
        "note": "MP4 playback uses nominal frame rate; exact texture needs each original RGB exposure timestamp. No duration scaling is applied.",
    }


def display_trajectory(trajectory: pd.DataFrame, first_pose_video_s: float | None) -> pd.DataFrame:
    indices = np.unique(np.linspace(0, len(trajectory) - 1, min(5_000, len(trajectory)), dtype=int))
    path = trajectory.iloc[indices].copy()
    path["video_s"] = (
        (path["tracking_timestamp_us"] - trajectory["tracking_timestamp_us"].iloc[0]) / 1e6
        + (first_pose_video_s if first_pose_video_s is not None else 0.0)
    )
    return path


def write_preview(
    path: Path, points: np.ndarray, trajectory_xyz: np.ndarray, trial_id: str
) -> None:
    display = _sample_evenly(points, 55_000)
    trajectory = _sample_evenly(trajectory_xyz, 3_000)
    figure = plt.figure(figsize=(15, 9), facecolor="#f2efe7")
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        display[:, 0],
        display[:, 1],
        display[:, 2],
        c=display[:, 2],
        cmap="viridis",
        s=0.35,
        alpha=0.25,
        linewidths=0,
    )
    axis.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        color="#e05236",
        linewidth=2.2,
        label="closed-loop head path",
    )
    axis.scatter(*trajectory[0], color="#18745d", s=70, label="start")
    axis.scatter(*trajectory[-1], color="#15251f", s=70, label="end")
    axis.set_xlabel("world X (m)")
    axis.set_ylabel("world Y (m)")
    axis.set_zlabel("world Z / gravity-up (m)")
    axis.set_box_aspect(np.maximum(np.ptp(points, axis=0), 1.0))
    axis.set_title(
        f"{trial_id}: quality-filtered MPS environment evidence",
        fontsize=18,
        fontweight="bold",
    )
    axis.legend(loc="upper left")
    axis.view_init(elev=30, azim=-58)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def build_figure(
    display_points: np.ndarray,
    mesh: trimesh.Trimesh,
    trajectory: pd.DataFrame,
    first_pose_video_s: float | None = None,
    color_bounds: tuple[float, float] | None = None,
) -> go.Figure:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    path = display_trajectory(trajectory, first_pose_video_s)
    path_time = path["video_s"].to_numpy()
    color_low, color_high = color_bounds or (float(display_points[:, 2].min()), float(display_points[:, 2].max()))

    figure = go.Figure()
    figure.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=vertices[:, 2],
            colorscale="Viridis",
            cmin=color_low,
            cmax=color_high,
            opacity=0.28,
            visible=False,
            showlegend=True,
            showscale=False,
            flatshading=False,
            name="Experimental surface",
            hovertemplate="surface<br>x %{x:.2f} m<br>y %{y:.2f} m<br>z %{z:.2f} m<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=display_points[:, 0],
            y=display_points[:, 1],
            z=display_points[:, 2],
            mode="markers",
            marker={
                "size": 1.25,
                "opacity": 0.38,
                "color": display_points[:, 2],
                "colorscale": "Viridis",
                "cmin": color_low,
                "cmax": color_high,
                "showscale": True,
                "colorbar": {"title": "Height<br>(m)", "thickness": 14},
            },
            name="Measured MPS points",
            hovertemplate="measured point<br>x %{x:.2f} m<br>y %{y:.2f} m<br>z %{z:.2f} m<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=path["tx_world_device"],
            y=path["ty_world_device"],
            z=path["tz_world_device"],
            customdata=path_time,
            mode="lines",
            line={"color": "#e05236", "width": 6},
            name="Head trajectory",
            hovertemplate="video %{customdata:.1f} s<br>x %{x:.2f} m<br>y %{y:.2f} m<br>z %{z:.2f} m<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[path["tx_world_device"].iloc[0]],
            y=[path["ty_world_device"].iloc[0]],
            z=[path["tz_world_device"].iloc[0]],
            mode="markers",
            marker={"size": 7, "color": "#f7c94c", "line": {"color": "#17211c", "width": 2}},
            name="Video position",
            visible=False,
            hovertemplate="current video position<extra></extra>",
        )
    )
    figure.update_layout(
        paper_bgcolor="#f8f5ec",
        plot_bgcolor="#f8f5ec",
        margin={"l": 20, "r": 10, "t": 10, "b": 20},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(248,245,236,0.85)",
            "itemclick": False,
            "itemdoubleclick": False,
        },
        scene={
            "xaxis": {"title": "World X (m)", "backgroundcolor": "#f8f5ec"},
            "yaxis": {"title": "World Y (m)", "backgroundcolor": "#f8f5ec"},
            "zaxis": {"title": "World Z / up (m)", "backgroundcolor": "#f8f5ec"},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.45, "y": -1.55, "z": 1.05}},
        },
        uirevision="keep-camera",
    )
    return figure


def write_interactive_page(
    output_path: Path,
    trial_id: str,
    display_points: np.ndarray,
    mesh: trimesh.Trimesh,
    trajectory: pd.DataFrame,
    video_path: Path,
    summary: dict[str, Any],
) -> None:
    sync = summary["sync"]
    color_bounds = (
        summary["filtering"]["bounds_min_m"][2],
        summary["filtering"]["bounds_max_m"][2],
    )
    figure = build_figure(display_points, mesh, trajectory, sync["first_pose_video_s"], color_bounds)
    path = display_trajectory(trajectory, sync["first_pose_video_s"])
    manifest_path = RUNS_ROOT / trial_id / "session_dashboard_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    payload = {
        "path": {
            "x": path["tx_world_device"].round(5).tolist(),
            "y": path["ty_world_device"].round(5).tolist(),
            "z": path["tz_world_device"].round(5).tolist(),
            "t": path["video_s"].round(6).tolist(),
        },
        "sync": sync,
        "summary": summary,
        "motionSegments": manifest.get("continuous_model_segments") or manifest.get("model_segments") or [],
        "videoDuration": float(summary["video"].get("browser_asset", summary["video"]["source"])["duration_s"]),
    }
    options = ['<option value="../../pd_insighter_trial_selector.html">All trials</option>']
    for trial_dir in sorted(RUNS_ROOT.iterdir()):
        target = trial_dir / "environment_reconstruction" / "index.html"
        if target.is_file():
            selected = " selected" if trial_dir.name == trial_id else ""
            href = html.escape(os.path.relpath(target, output_path.parent), quote=True)
            options.append(f'<option value="{href}"{selected}>{html.escape(trial_dir.name)}</option>')
    template = (Path(__file__).parent / "templates" / "pd_insighter_environment_review.html").read_text()
    rgb_summary_path = output_path.parent / "rgb_vertex_color_trial" / "validation_summary.json"
    rgb_section = ""
    if rgb_summary_path.exists():
        rgb = json.loads(rgb_summary_path.read_text())
        if rgb.get("trial_id") != trial_id or rgb.get("status") != "local_vertex_color_experiment":
            raise ValueError("RGB experiment metadata does not belong to this trial.")
        check = rgb["slam_reprojection"]
        window = "-".join(f"{x:g}" for x in rgb["recording_window_s"])
        rgb_section = f'''<section class="card rgb-experiment">
          <div><h2>Local RGB Vertex-Color Test</h2>
          <p>Original recording {window} s: {rgb['rgb_frame_count']} raw VRS views, online calibration and exact device timing.</p>
          <p><strong>{rgb['coloring']['colored_points']:,} colored landmarks; {rgb['mesh']['vertices']:,} vertices and {rgb['mesh']['faces']:,} triangles.</strong>
          SLAM reprojection median {check['median_px']:.2f} px; 95th percentile {check['p95_px']:.2f} px.</p>
          <p class="notice">A small, partial appearance test, not a textured room. RGB is visually inspected, not independently ground-truth validated. Sparse coverage is retained rather than filled in.</p>
          <div class="downloads"><a class="button" href="rgb_vertex_color_trial/index.html">Inspect RGB experiment</a>
          <a class="button" href="rgb_vertex_color_trial/local_rgb_vertex_mesh.glb">Vertex-color GLB</a>
          <a class="button" href="rgb_vertex_color_trial/validation_summary.json">Validation results</a></div></div>
          <a href="rgb_vertex_color_trial/index.html"><img src="rgb_vertex_color_trial/rgb_projection_overlay.jpg" alt="RGB frame with calibrated landmark projections and foreground exclusion"></a>
        </section>'''
    gaussian_summary_path = output_path.parent / "gaussian_splatting_trial" / "trained_proof" / "training_summary.json"
    gaussian_section = ""
    if gaussian_summary_path.exists():
        gaussian = json.loads(gaussian_summary_path.read_text())
        if (
            gaussian.get("trial_id") != trial_id
            or gaussian.get("status") != "proof_of_pipeline_not_final_clinical_scene"
        ):
            raise ValueError("Gaussian proof metadata does not belong to this trial.")
        segment = gaussian["dataset"]["segment"]
        gaussian_training = gaussian.get("training_parameters", {})
        densify_until = gaussian_training.get("densify_until")
        densification_note = (
            f"limited CPU densification through iteration {densify_until}"
            if densify_until is not None
            else "a conservative CPU training configuration"
        )
        gaussian_section = f'''<section class="card gaussian-experiment">
          <div><h2>Metric Gaussian Scene: First Validated Pass</h2>
          <p>Original recording {segment['start_s']:g}-{segment['end_s']:g} s: {gaussian['dataset']['frames']['selected']} calibrated VRS views and {gaussian['dataset']['points']['written']:,} MPS initializer points.</p>
          <p><strong>Held-out PSNR {gaussian['validation_psnr_db']:.2f} dB; validation loss {gaussian['validation_loss']:.3f}.</strong> Major room surfaces and viewpoint alignment are recovered from the remaining views.</p>
          <p class="notice">Pipeline proof only: {gaussian['iterations']} iterations at {gaussian['downscale_factor']:g}x lower resolution with {densification_note}. Residual blur is expected; this is not final clinical scene quality.</p>
          <div class="downloads"><a class="button" href="gaussian_splatting_trial/trained_proof/index.html">Inspect held-out comparison</a>
          <a class="button" href="gaussian_splatting_trial/trained_proof/splat_{gaussian['iterations']}.ply">Gaussian PLY</a>
          <a class="button" href="gaussian_splatting_trial/trained_proof/training_summary.json">Training results</a></div></div>
          <iframe class="splat-frame" src="gaussian_splatting_trial/trained_proof/viewer.html" title="Interactive trained Gaussian scene"></iframe>
        </section>'''
    submap_manifest_path = output_path.parent / "gaussian_submaps" / "submaps_manifest.json"
    submap_section = ""
    if submap_manifest_path.exists():
        submaps = json.loads(submap_manifest_path.read_text())
        if submaps.get("trial_id") != trial_id:
            raise ValueError("Gaussian submap metadata does not belong to this trial.")
        rows = []
        completed = 0
        for item in submaps["submaps"]:
            result_path = (
                Path(item["dataset_dir"])
                / "trained_7000_fullres"
                / "training_summary.json"
            )
            result_label = "Pending"
            psnr_label = "--"
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                if result.get("backend") == "MPS" and result.get("iterations") == 7000:
                    completed += 1
                    result_label = "MPS complete"
                    psnr_label = f"{result['validation_psnr_db']:.2f} dB"
            rows.append(
                f"<tr><td>{html.escape(item['name'].replace('_', ' ').title())}</td>"
                f"<td>{item['start_s']:g}-{item['end_s']:g} s</td>"
                f"<td>{item['frames_selected']:,}</td><td>{item['initializer_points']:,}</td>"
                f"<td>{psnr_label}</td><td>{result_label}</td></tr>"
            )
        rows_html = "".join(rows)
        submap_section = f'''<section class="card submap-experiment">
          <div><h2>Motion-Centered Gaussian Submaps</h2>
          <p>Instead of forcing the full recording into one appearance model, Ntrial3 is divided into sitting context, upstairs, corridor walking, and downstairs context. Every submap stays in the same metric MPS world frame.</p>
          <p class="notice"><strong>Training status:</strong> {completed} of {len(submaps['submaps'])} submaps completed 7,000 full-resolution iterations on Apple MPS. The gallery contains each held-out render, interactive Gaussian model and validation record.</p>
          <div class="downloads"><a class="button" href="gaussian_submaps/index.html">Open submap gallery</a><a class="button" href="gaussian_submaps/submaps_manifest.json">Submap manifest</a></div></div>
          <table><thead><tr><th>Submap</th><th>Recording time</th><th>Views</th><th>Points</th><th>Held-out PSNR</th><th>Status</th></tr></thead><tbody>{rows_html}</tbody></table>
        </section>'''
    playback_section = ""
    if (output_path.parent / "playback.html").is_file():
        playback_section = '''<section class="card"><h2>Synchronized Video + 3D Playback</h2>
          <p>Drag one timeline to follow the Aria video, calibrated camera position and orientation, and action labels. Switch between the recorded viewpoint and free 3D inspection.</p>
          <div class="downloads"><a class="button" href="playback.html">Open synchronized 3D playback</a></div>
          <p>Only trained intervals show a Gaussian scene. Uncovered intervals remain explicit; original videos and labels are unchanged.</p></section>'''
    replacements = {
        "__TRIAL_ID__": html.escape(trial_id),
        "__VIDEO_HREF__": html.escape(os.path.relpath(video_path, output_path.parent), quote=True),
        "__TRIAL_OPTIONS__": "".join(options),
        "__RGB_EXPERIMENT__": rgb_section,
        "__GAUSSIAN_EXPERIMENT__": gaussian_section,
        "__GAUSSIAN_SUBMAPS__": submap_section,
        "__SCENE_PLAYBACK__": playback_section,
        "__PLOTLY_SCRIPT__": get_plotlyjs(),
        "__FIGURE__": pio.to_json(figure, pretty=False).replace("</", "<\\/"),
        "__DATA__": json.dumps(payload, separators=(",", ":")).replace("</", "<\\/"),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    output_path.write_text(template)


def main() -> None:
    args = parse_args()
    paths = resolve_inputs(args)
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_columns = [
        "graph_uid",
        "tracking_timestamp_us",
        "tx_world_device",
        "ty_world_device",
        "tz_world_device",
        "quality_score",
    ]
    trajectory = pd.read_csv(paths["trajectory"], usecols=trajectory_columns)
    trajectory = trajectory.replace([np.inf, -np.inf], np.nan).dropna()
    graph_ids = trajectory["graph_uid"].unique().tolist()
    if len(graph_ids) != 1:
        raise ValueError("A trial reconstruction requires one trajectory graph_uid/world frame.")
    if not trajectory["tracking_timestamp_us"].is_monotonic_increasing:
        raise ValueError("Trajectory timestamps must be ordered.")
    trajectory_xyz = trajectory[
        ["tx_world_device", "ty_world_device", "tz_world_device"]
    ].to_numpy(dtype=np.float32)
    trajectory_duration = float(
        (trajectory["tracking_timestamp_us"].iloc[-1] - trajectory["tracking_timestamp_us"].iloc[0])
        / 1e6
    )

    points, filtering_summary = load_and_filter_points(
        paths["points"],
        trajectory_xyz,
        args.inv_dist_std_max,
        args.dist_std_max,
        args.max_path_distance_m,
        expected_graph_uid=graph_ids[0],
    )
    mesh, mesh_summary = build_density_mesh(
        points,
        args.voxel_size_m,
        args.density_sigma,
        args.density_level,
        args.marching_step,
        args.min_component_faces,
    )
    geometry_paths = export_geometry(output_dir, points, mesh)
    browser_video_path, video_summary = ensure_browser_video(paths["aria_video"], output_dir)
    display_points = _sample_evenly(points, args.display_points)

    preview_path = output_dir / "environment_reconstruction_preview.png"
    write_preview(preview_path, points, trajectory_xyz, args.trial_id)
    sync = resolve_video_timing(paths["aria_video"], trajectory, RUNS_ROOT / args.trial_id)
    vertex_distances = cKDTree(points).query(np.asarray(mesh.vertices), workers=-1)[0]
    mesh_summary["vertex_to_measured_point_distance_m"] = {
        "median": round(float(np.median(vertex_distances)), 4),
        "p95": round(float(np.quantile(vertex_distances, 0.95)), 4),
        "interpretation": "Agreement with input points, not independent geometric accuracy.",
    }
    mesh_summary["effective_extraction_spacing_m"] = args.voxel_size_m * args.marching_step
    summary = {
        "trial_id": args.trial_id,
        "status": "research_prototype",
        "coordinate_definition": {
            "frame": "MPS gravity-aligned world frame",
            "x_y": "horizontal world-plane axes; neither is assumed to be travel-forward",
            "z": "gravity-up height in meters",
        },
        "inputs": {key: str(value) for key, value in paths.items() if key != "output_dir"},
        "filtering": {
            **filtering_summary,
            "inv_dist_std_max": args.inv_dist_std_max,
            "dist_std_max_m": args.dist_std_max,
            "max_distance_from_sampled_path_m": args.max_path_distance_m,
            "robust_spatial_quantiles": [0.001, 0.999],
        },
        "surface": mesh_summary,
        "method": mesh_summary,
        "trajectory": {
            "graph_uid": graph_ids[0],
            "sample_count": int(len(trajectory)),
            "duration_s": round(trajectory_duration, 4),
            "quality_score_mean": round(float(trajectory["quality_score"].mean()), 6),
            "bounds_min_m": np.min(trajectory_xyz, axis=0).round(5).tolist(),
            "bounds_max_m": np.max(trajectory_xyz, axis=0).round(5).tolist(),
        },
        "video": video_summary,
        "sync": sync,
        "outputs": {
            **{key: str(value) for key, value in geometry_paths.items()},
            "browser_video_mp4": str(browser_video_path),
            "preview_png": str(preview_path),
            "interactive_html": str(output_dir / "index.html"),
        },
        "limitations": [
            "The MPS point cloud is semi-dense and biased toward high-gradient static scene features.",
            "The inferred surface is a downstream density isosurface, not a Meta-provided mesh.",
            "Thin, reflective, moving, or weakly textured objects may be absent or distorted.",
            "Height colors are analytical colors and are not projected RGB texture.",
            "Path-distance and quantile crops are heuristics and may discard valid scene points.",
            "Missing points do not establish that space is empty or safe to walk through.",
        ],
    }
    summary_path = output_dir / "reconstruction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    write_interactive_page(
        output_dir / "index.html",
        args.trial_id,
        display_points,
        mesh,
        trajectory,
        browser_video_path,
        summary,
    )
    from pd_insighter_environment_navigation import refresh_navigation
    refresh_navigation([args.trial_id])
    print(f"Environment reconstruction written to: {output_dir}")
    print(f"Interactive page: {output_dir / 'index.html'}")
    print(
        f"Retained {len(points):,}/{filtering_summary['raw_point_count']:,} measured points; "
        f"surface has {len(mesh.vertices):,} vertices and {len(mesh.faces):,} faces."
    )


if __name__ == "__main__":
    main()
