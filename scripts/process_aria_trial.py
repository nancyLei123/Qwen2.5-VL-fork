#!/usr/bin/env python3
"""Prepare Aria Gen1 trial data for thesis-style action recognition analysis.

This script bridges the gap between raw MPS outputs and the thesis pipeline:
1. Split an extracted MP4 into strict 2-second clips.
2. Aggregate SLAM features per clip.
3. Apply the rule-based action classifier from the thesis.
4. Export timeline-ready CSV/JSON files.
5. Generate quick-look point-cloud and displacement plots.

It is intentionally lightweight and only relies on packages already available in
the local Project Aria tools environment: pandas, numpy, matplotlib, scipy, cv2.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import html
import os
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ACTIONS = ["sitting", "standing", "walking", "go upstairs", "go downstairs"]
THESIS_RULES_TEXT = (
    "If 0.15 < vertical_displacement < 0.2 and horizontal_displacement < 1.45: "
    "go upstairs. If vertical_displacement < -0.15 and 0.2 < horizontal_displacement < 1.45: "
    "go downstairs. If horizontal_displacement > 0.5 and mean_speed > 0.5: walking. "
    "If mean_speed < 0.15 and horizontal_displacement < 0.25: sitting. Else: standing."
)


@dataclass
class ClipWindow:
    clip_idx: int
    start_s: float
    end_s: float
    start_epoch_s: float
    end_epoch_s: float
    clip_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-path", required=True, help="Extracted MP4 path.")
    parser.add_argument("--vrs-json-path", required=True, help="Original .vrs.json metadata path.")
    parser.add_argument("--trajectory-csv", required=True, help="MPS closed_loop_trajectory.csv path.")
    parser.add_argument(
        "--points-csv",
        required=True,
        help="MPS semidense_points.csv.gz path.",
    )
    parser.add_argument(
        "--eye-gaze-csv",
        help="Optional general_eye_gaze.csv path for better SLAM/video alignment.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for generated clips, tables, and plots.",
    )
    parser.add_argument("--trial-id", help="Optional human-readable trial id, e.g. trial6.")
    parser.add_argument(
        "--recording-stem",
        help="Optional human-readable recording stem, e.g. 42_go_upstairs_-_head_motions.",
    )
    parser.add_argument("--clip-seconds", type=float, default=2.0, help="Clip duration in seconds.")
    parser.add_argument(
        "--point-sample-size",
        type=int,
        default=50000,
        help="Max number of semidense points to plot in the 3D preview.",
    )
    parser.add_argument(
        "--qwen-predictions-csv",
        help="Optional CSV containing clip_idx,qwen_prediction for hybrid fusion.",
    )
    parser.add_argument(
        "--hybrid-override-labels",
        nargs="*",
        default=["sitting"],
        help="Labels for which Qwen predictions override the rule-based label.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_video_info(video_path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, frame_count, width, height


def split_video_into_clips(
    video_path: Path,
    clips_dir: Path,
    clip_seconds: float,
    *,
    start_offset_s: float = 0.0,
    max_duration_s: float | None = None,
) -> list[ClipWindow]:
    clips_dir.mkdir(parents=True, exist_ok=True)
    fps, frame_count, width, height = load_video_info(video_path)
    frames_per_clip = int(round(clip_seconds * fps))
    source_start_frame = min(frame_count, max(0, int(round(start_offset_s * fps))))
    available_frames = max(0, frame_count - source_start_frame)
    if max_duration_s is not None:
        available_frames = min(available_frames, max(0, int(round(max_duration_s * fps))))
    clip_count = int(available_frames // frames_per_clip)

    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    windows: list[ClipWindow] = []

    for clip_idx in range(clip_count):
        start_frame = clip_idx * frames_per_clip
        end_frame = start_frame + frames_per_clip
        clip_path = clips_dir / f"clip_{clip_idx:04d}.mp4"
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_start_frame + start_frame)
        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
        for _ in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        writer.release()
        windows.append(
            ClipWindow(
                clip_idx=clip_idx,
                start_s=clip_idx * clip_seconds,
                end_s=(clip_idx + 1) * clip_seconds,
                start_epoch_s=0.0,
                end_epoch_s=0.0,
                clip_path=clip_path,
            )
        )

    cap.release()
    return windows


def create_clip_gif(clip_path: Path, gif_path: Path, max_width: int = 420, max_frames: int = 20) -> bool:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return False

    step = max(1, total_frames // max_frames)
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        idx += 1
    cap.release()

    if not frames:
        return False

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, frames, duration=0.12, loop=0)
    return True


def load_start_time(vrs_json_path: Path) -> float:
    metadata = json.loads(vrs_json_path.read_text())
    return float(metadata["start_time"])


def discover_audio_companion(video_path: Path) -> tuple[Path | None, float]:
    audio_path = video_path.parent / "audio_track.wav"
    audio_metadata_path = video_path.parent / "audio_metadata.json"
    if not audio_path.exists():
        return None, 0.0

    audio_offset_s = 0.0
    if audio_metadata_path.exists():
        try:
            metadata = json.loads(audio_metadata_path.read_text())
            audio_offset_s = float(metadata.get("video_audio_offset_s", 0.0))
        except (ValueError, TypeError):
            audio_offset_s = 0.0
    return audio_path, audio_offset_s


def relative_media_href(page_path: Path, target_path: Path | None) -> str | None:
    if target_path is None or not target_path.exists():
        return None
    return Path(os.path.relpath(target_path.resolve(), start=page_path.resolve().parent)).as_posix()


def resolve_media_href(page_path: Path, target: Path | str | None) -> str | None:
    if target is None:
        return None
    if isinstance(target, str):
        return target
    return relative_media_href(page_path, target)


def trial_sort_key_for_dashboard(trial_id: str) -> tuple[int, str]:
    if trial_id.startswith("trial"):
        suffix = trial_id.removeprefix("trial")
    elif trial_id.startswith("Ntrial"):
        suffix = trial_id.removeprefix("Ntrial")
    else:
        suffix = trial_id
    try:
        return int(suffix), trial_id
    except ValueError:
        return 10**9, trial_id


def versioned_relative_dashboard_href(output_path: Path, target_path: Path) -> str:
    href = Path(os.path.relpath(target_path.resolve(), start=output_path.resolve().parent)).as_posix()
    try:
        version = int(target_path.stat().st_mtime)
    except FileNotFoundError:
        return href
    joiner = "&" if "?" in href else "?"
    return f"{href}{joiner}v={version}"


def collect_dashboard_trial_options(output_path: Path) -> list[dict[str, str]]:
    runs_dir = output_path.parent.parent
    selector_path = runs_dir / "pd_insighter_trial_selector.html"
    prefer_clip_review = output_path.name == "pd_insighter_clip_review.html"
    items: list[dict[str, str]] = []
    for manifest_path in sorted(runs_dir.glob("*/trial_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            continue
        run_dir = manifest_path.parent
        trial_id = str(manifest.get("trial_id", run_dir.name))
        if not trial_id.startswith("Ntrial"):
            continue
        if prefer_clip_review:
            dashboard_path = run_dir / "pd_insighter_clip_review.html"
        else:
            dashboard_path = run_dir / "index.html"
        if not dashboard_path.exists():
            continue
        recording_stem = str(manifest.get("recording_stem", run_dir.name))
        items.append(
            {
                "trial_id": trial_id,
                "label": f"{trial_id} - {recording_stem}",
                "href": versioned_relative_dashboard_href(output_path, dashboard_path),
            }
        )
    items = sorted(items, key=lambda item: trial_sort_key_for_dashboard(item["trial_id"]))
    selector_href = versioned_relative_dashboard_href(output_path, selector_path) if selector_path.exists() else "../pd_insighter_trial_selector.html"
    return [{"trial_id": "__selector__", "label": "Back to Trial Selector", "href": selector_href}] + items


def create_clip_audio_files(
    clip_df: pd.DataFrame,
    clip_audio_dir: Path,
    audio_path: Path | None,
    audio_offset_s: float,
) -> dict[int, Path]:
    if audio_path is None or not audio_path.exists():
        return {}

    clip_audio_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[int, Path] = {}

    with wave.open(str(audio_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channel_count = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        raw_bytes = wav_file.readframes(frame_count)

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sample_width)
    if dtype is None:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width}")

    audio_samples = np.frombuffer(raw_bytes, dtype=dtype)
    if channel_count > 1:
        audio_samples = audio_samples.reshape(-1, channel_count)
    else:
        audio_samples = audio_samples.reshape(-1, 1)

    total_samples = audio_samples.shape[0]
    for row in clip_df.to_dict(orient="records"):
        clip_idx = int(row["clip_idx"])
        clip_start_s = max(0.0, float(row["start_s"]) + audio_offset_s)
        clip_end_s = max(clip_start_s, float(row["end_s"]) + audio_offset_s)
        start_index = min(total_samples, max(0, int(round(clip_start_s * sample_rate))))
        end_index = min(total_samples, max(start_index, int(round(clip_end_s * sample_rate))))
        clip_samples = audio_samples[start_index:end_index]
        output_path = clip_audio_dir / f"clip_{clip_idx:04d}.wav"
        with wave.open(str(output_path), "wb") as clip_wav:
            clip_wav.setnchannels(channel_count)
            clip_wav.setsampwidth(sample_width)
            clip_wav.setframerate(sample_rate)
            clip_wav.writeframes(clip_samples.tobytes())
        output_paths[clip_idx] = output_path

    return output_paths


def create_subclip_media_files_ffmpeg(
    source_video_path: Path,
    output_dir: Path,
    windows: list[ClipWindow],
    *,
    source_start_offset_s: float = 0.0,
) -> dict[int, Path]:
    ffmpeg_exe = discover_ffmpeg_executable()
    if not ffmpeg_exe:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[int, Path] = {}

    for window in windows:
        output_path = output_dir / f"clip_{window.clip_idx:04d}.mp4"
        if output_path.exists() and output_path.stat().st_mtime >= source_video_path.stat().st_mtime:
            output_paths[window.clip_idx] = output_path
            continue

        temp_output_path = output_path.with_suffix(".tmp.mp4")
        start_time_s = max(0.0, source_start_offset_s + window.start_s)
        duration_s = max(0.05, window.end_s - window.start_s)
        command = [
            ffmpeg_exe,
            "-y",
            "-ss",
            f"{start_time_s:.6f}",
            "-i",
            str(source_video_path),
            "-t",
            f"{duration_s:.6f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-vf",
            "scale=1280:-2:force_original_aspect_ratio=decrease",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(temp_output_path),
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            if temp_output_path.exists():
                temp_output_path.unlink()
            continue

        if not temp_output_path.exists():
            continue

        temp_output_path.replace(output_path)
        output_paths[window.clip_idx] = output_path

    return output_paths


def discover_ffmpeg_executable() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def create_clip_media_files(
    clip_df: pd.DataFrame,
    clip_media_dir: Path,
    clip_audio_map: dict[int, Path] | None,
) -> dict[int, Path]:
    if not clip_audio_map:
        return {}

    ffmpeg_exe = discover_ffmpeg_executable()
    if not ffmpeg_exe:
        return {}

    clip_media_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[int, Path] = {}

    for row in clip_df.to_dict(orient="records"):
        clip_idx = int(row["clip_idx"])
        clip_video_path = Path(str(row["clip_path"]))
        clip_audio_path = clip_audio_map.get(clip_idx)
        if clip_audio_path is None or not clip_video_path.exists() or not clip_audio_path.exists():
            continue

        output_path = clip_media_dir / f"clip_{clip_idx:04d}.mp4"
        if output_path.exists():
            output_mtime = output_path.stat().st_mtime
            newest_input = max(clip_video_path.stat().st_mtime, clip_audio_path.stat().st_mtime)
            if output_mtime >= newest_input:
                output_paths[clip_idx] = output_path
                continue

        temp_output_path = output_path.with_suffix(".tmp.mp4")
        base_cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(clip_video_path),
            "-i",
            str(clip_audio_path),
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
        ]
        command_variants = [
            base_cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(temp_output_path)],
            base_cmd + ["-c:v", "copy", str(temp_output_path)],
        ]

        muxed = False
        for command in command_variants:
            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                muxed = True
                break
            except subprocess.CalledProcessError:
                if temp_output_path.exists():
                    temp_output_path.unlink()

        if not muxed or not temp_output_path.exists():
            continue

        temp_output_path.replace(output_path)
        output_paths[clip_idx] = output_path

    return output_paths


def estimate_alignment_offset_s(trajectory_df: pd.DataFrame, eye_gaze_csv: Path | None) -> float:
    if eye_gaze_csv is None:
        return 0.0
    eye_df = pd.read_csv(eye_gaze_csv)
    if eye_df.empty or trajectory_df.empty:
        return 0.0
    return max(
        0.0,
        (float(trajectory_df["tracking_timestamp_us"].iloc[0]) - float(eye_df["tracking_timestamp_us"].iloc[0]))
        / 1_000_000.0,
    )


def horizontal_displacement(dx: float, dy: float) -> float:
    return float(math.hypot(dx, dy))


def angular_speed(row: pd.Series) -> float:
    return float(
        math.sqrt(
            row["angular_velocity_x_device"] ** 2
            + row["angular_velocity_y_device"] ** 2
            + row["angular_velocity_z_device"] ** 2
        )
    )


def linear_speed(row: pd.Series) -> float:
    return float(
        math.sqrt(
            row["device_linear_velocity_x_device"] ** 2
            + row["device_linear_velocity_y_device"] ** 2
            + row["device_linear_velocity_z_device"] ** 2
        )
    )


def calibrate_posture_z_band(rows: list[dict[str, object]]) -> float | None:
    low_motion_z = sorted(
        float(row["mean_z_m"])
        for row in rows
        if row.get("sample_count", 0)
        and pd.notna(row.get("mean_z_m"))
        and float(row.get("mean_linear_speed_mps", np.nan)) < 0.15
        and float(row.get("horizontal_displacement_m", np.nan)) < 0.25
    )
    if len(low_motion_z) < 8:
        return None

    gaps = [low_motion_z[idx + 1] - low_motion_z[idx] for idx in range(len(low_motion_z) - 1)]
    gap_index = max(range(len(gaps)), key=lambda idx: gaps[idx])
    gap_size = gaps[gap_index]
    lower_cluster = low_motion_z[: gap_index + 1]
    upper_cluster = low_motion_z[gap_index + 1 :]

    if gap_size < 0.12:
        return None
    if len(lower_cluster) < 5 or len(upper_cluster) < 5:
        return None
    if len(lower_cluster) / len(low_motion_z) < 0.15 or len(upper_cluster) / len(low_motion_z) < 0.15:
        return None

    return float((low_motion_z[gap_index] + low_motion_z[gap_index + 1]) / 2.0)


def classify_rule(horizontal: float, vertical: float, mean_speed: float, mean_z_m: float, posture_z_band_m: float | None) -> str:
    if 0.15 < vertical < 0.2 and horizontal < 1.45:
        return "go upstairs"
    if vertical < -0.15 and 0.2 < horizontal < 1.45:
        return "go downstairs"
    if horizontal > 0.5 and mean_speed > 0.5:
        return "walking"
    if mean_speed < 0.15 and horizontal < 0.25:
        if posture_z_band_m is not None and pd.notna(mean_z_m):
            return "sitting" if mean_z_m <= posture_z_band_m else "standing"
        return "sitting"
    return "standing"


def refine_downstairs_with_post_height_context(
    rows: list[dict[str, object]],
    posture_z_band_m: float | None,
) -> None:
    if posture_z_band_m is None:
        return

    for idx, row in enumerate(rows):
        if row.get("rule_prediction") != "go downstairs":
            continue

        future_rows = [
            rows[next_idx]
            for next_idx in (idx + 1, idx + 2)
            if next_idx < len(rows) and rows[next_idx].get("sample_count", 0)
        ]
        if not future_rows:
            continue

        settled_rows = [
            future_row
            for future_row in future_rows
            if float(future_row.get("mean_linear_speed_mps", np.nan)) < 0.15
            and float(future_row.get("horizontal_displacement_m", np.nan)) < 0.2
            and float(future_row.get("mean_z_m", np.nan)) <= posture_z_band_m
        ]
        continued_descent_rows = [
            future_row
            for future_row in future_rows
            if float(future_row.get("vertical_displacement_m", np.nan)) < -0.12
            and float(future_row.get("horizontal_displacement_m", np.nan)) > 0.2
            and float(future_row.get("mean_linear_speed_mps", np.nan)) > 0.2
        ]
        if not settled_rows or continued_descent_rows:
            continue

        settled_mean_z = float(np.mean([float(future_row["mean_z_m"]) for future_row in settled_rows]))
        current_mean_z = float(row.get("mean_z_m", np.nan))
        row["rule_prediction"] = "standing" if (current_mean_z - settled_mean_z) > 0.15 else "sitting"


def build_clip_metrics(
    windows: list[ClipWindow],
    start_epoch_s: float,
    trajectory_df: pd.DataFrame,
    alignment_offset_s: float,
) -> pd.DataFrame:
    traj = trajectory_df.copy()
    first_ts = float(traj["tracking_timestamp_us"].iloc[0])
    traj["video_rel_s"] = (traj["tracking_timestamp_us"] - first_ts) / 1_000_000.0 + alignment_offset_s
    traj["linear_speed_mps"] = traj.apply(linear_speed, axis=1)
    traj["angular_speed_rps"] = traj.apply(angular_speed, axis=1)

    rows: list[dict[str, object]] = []
    for window in windows:
        clip_start_s = window.start_s
        clip_end_s = window.end_s
        clip_df = traj[(traj["video_rel_s"] >= clip_start_s) & (traj["video_rel_s"] < clip_end_s)].copy()
        if clip_df.empty:
            rows.append(
                {
                    "clip_idx": window.clip_idx,
                    "clip_path": str(window.clip_path),
                    "start_s": clip_start_s,
                    "end_s": clip_end_s,
                    "start_epoch_s": start_epoch_s + clip_start_s,
                    "end_epoch_s": start_epoch_s + clip_end_s,
                    "sample_count": 0,
                    "horizontal_displacement_m": np.nan,
                    "vertical_displacement_m": np.nan,
                    "mean_linear_speed_mps": np.nan,
                    "mean_angular_speed_rps": np.nan,
                    "rule_prediction": "missing",
                    "slam_text": "",
                }
            )
            continue

        x0, y0, z0 = (
            float(clip_df["tx_world_device"].iloc[0]),
            float(clip_df["ty_world_device"].iloc[0]),
            float(clip_df["tz_world_device"].iloc[0]),
        )
        x1, y1, z1 = (
            float(clip_df["tx_world_device"].iloc[-1]),
            float(clip_df["ty_world_device"].iloc[-1]),
            float(clip_df["tz_world_device"].iloc[-1]),
        )
        horizontal = horizontal_displacement(x1 - x0, y1 - y0)
        vertical = float(z1 - z0)
        mean_speed = float(clip_df["linear_speed_mps"].mean())
        speed_variance = float(clip_df["linear_speed_mps"].var(ddof=0))
        mean_angular = float(clip_df["angular_speed_rps"].mean())
        angular_speed_variance = float(clip_df["angular_speed_rps"].var(ddof=0))
        gravity_flat = clip_df[["gravity_x_world", "gravity_y_world", "gravity_z_world"]].to_numpy().reshape(-1)
        gravity_variance = float(np.var(gravity_flat))
        mean_z_m = float(clip_df["tz_world_device"].mean())
        slam_text = (
            f"average linear velocity={mean_speed:.3f} m/s, "
            f"net horizontal displacement={horizontal:.3f} m, "
            f"net vertical displacement={vertical:.3f} m, "
            f"average angular velocity={mean_angular:.3f} rad/s"
        )
        rows.append(
            {
                "clip_idx": window.clip_idx,
                "clip_path": str(window.clip_path),
                "start_s": clip_start_s,
                "end_s": clip_end_s,
                "start_epoch_s": start_epoch_s + clip_start_s,
                "end_epoch_s": start_epoch_s + clip_end_s,
                "sample_count": int(len(clip_df)),
                "horizontal_displacement_m": horizontal,
                "vertical_displacement_m": vertical,
                "mean_linear_speed_mps": mean_speed,
                "speed_variance": speed_variance,
                "mean_angular_speed_rps": mean_angular,
                "angular_speed_variance": angular_speed_variance,
                "gravity_variance": gravity_variance,
                "mean_z_m": mean_z_m,
                "rule_prediction": "",
                "slam_text": slam_text,
            }
        )

    posture_z_band_m = calibrate_posture_z_band(rows)
    for row in rows:
        if not row.get("sample_count"):
            continue
        row["rule_prediction"] = classify_rule(
            float(row["horizontal_displacement_m"]),
            float(row["vertical_displacement_m"]),
            float(row["mean_linear_speed_mps"]),
            float(row["mean_z_m"]),
            posture_z_band_m,
        )
    refine_downstairs_with_post_height_context(rows, posture_z_band_m)

    return pd.DataFrame(rows)


def apply_hybrid_predictions(
    clip_df: pd.DataFrame,
    qwen_predictions_csv: Path | None,
    override_labels: Iterable[str],
) -> pd.DataFrame:
    clip_df = clip_df.copy()
    clip_df["qwen_prediction"] = pd.NA
    clip_df["hybrid_prediction"] = clip_df["rule_prediction"]
    if qwen_predictions_csv is None:
        return clip_df

    qwen_df = pd.read_csv(qwen_predictions_csv)
    required_columns = {"clip_idx", "qwen_prediction"}
    missing = required_columns.difference(qwen_df.columns)
    if missing:
        raise ValueError(f"{qwen_predictions_csv} is missing columns: {sorted(missing)}")

    merged = clip_df.merge(qwen_df[["clip_idx", "qwen_prediction"]], on="clip_idx", how="left", suffixes=("", "_new"))
    if "qwen_prediction_new" in merged.columns:
        merged["qwen_prediction"] = merged["qwen_prediction_new"]
        merged = merged.drop(columns=["qwen_prediction_new"])
    override_labels_set = {label.strip().lower() for label in override_labels}

    def choose_label(row: pd.Series) -> str:
        qwen_label = row["qwen_prediction"]
        if isinstance(qwen_label, str) and qwen_label.strip().lower() in override_labels_set:
            return qwen_label.strip().lower()
        return row["rule_prediction"]

    merged["hybrid_prediction"] = merged.apply(choose_label, axis=1)
    return merged


def write_qwen_manifest(clip_df: pd.DataFrame, output_path: Path) -> None:
    records = []
    for row in clip_df.itertuples(index=False):
        prompt = (
            "Analyze the video clip and classify the action being performed. "
            f"Choose ONLY ONE action from this list: {', '.join(ACTIONS)}. "
            f"Use the SLAM data {row.slam_text} with the following criteria {THESIS_RULES_TEXT} "
            "to aid in your choice. Respond with just the action name, nothing else."
        )
        records.append(
            {
                "clip_idx": int(row.clip_idx),
                "video": str(row.clip_path),
                "prompt": prompt,
                "slam_text": row.slam_text,
                "rule_prediction": row.rule_prediction,
            }
        )
    output_path.write_text(json.dumps(records, indent=2))


def write_timeline_json(clip_df: pd.DataFrame, output_path: Path) -> None:
    records = []
    for row in clip_df.itertuples(index=False):
        records.append(
            {
                "clip_idx": int(row.clip_idx),
                "label": row.hybrid_prediction if isinstance(row.hybrid_prediction, str) else row.rule_prediction,
                "rule_prediction": row.rule_prediction,
                "horizontal_displacement_m": float(row.horizontal_displacement_m),
                "vertical_displacement_m": float(row.vertical_displacement_m),
                "mean_linear_speed_mps": float(row.mean_linear_speed_mps),
                "speed_variance": float(row.speed_variance),
                "mean_angular_speed_rps": float(row.mean_angular_speed_rps),
                "angular_speed_variance": float(row.angular_speed_variance),
                "gravity_variance": float(row.gravity_variance),
                "mean_z_m": float(row.mean_z_m),
                "qwen_prediction": row.qwen_prediction if isinstance(row.qwen_prediction, str) else None,
                "start_s": float(row.start_s),
                "end_s": float(row.end_s),
                "start_epoch_s": float(row.start_epoch_s),
                "end_epoch_s": float(row.end_epoch_s),
                "clip_path": row.clip_path,
            }
        )
    output_path.write_text(json.dumps(records, indent=2))


def summarize_segments(clip_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    valid_df = clip_df[clip_df[label_col].notna()].copy()
    if valid_df.empty:
        return pd.DataFrame()
    valid_df = valid_df.sort_values("clip_idx").reset_index(drop=True)
    segment_ids = (valid_df[label_col] != valid_df[label_col].shift(1)).cumsum()
    rows = []
    for _, seg_df in valid_df.groupby(segment_ids, sort=False):
        rows.append(
            {
                "label": seg_df[label_col].iloc[0],
                "start_s": float(seg_df["start_s"].iloc[0]),
                "end_s": float(seg_df["end_s"].iloc[-1]),
                "duration_s": float(seg_df["end_s"].iloc[-1] - seg_df["start_s"].iloc[0]),
                "start_epoch_s": float(seg_df["start_epoch_s"].iloc[0]),
                "end_epoch_s": float(seg_df["end_epoch_s"].iloc[-1]),
                "clip_start_idx": int(seg_df["clip_idx"].iloc[0]),
                "clip_end_idx": int(seg_df["clip_idx"].iloc[-1]),
                "mean_speed_mps": float(seg_df["mean_linear_speed_mps"].mean()),
                "mean_angular_speed_rps": float(seg_df["mean_angular_speed_rps"].mean()),
                "horizontal_displacement_m": float(seg_df["horizontal_displacement_m"].sum()),
                "vertical_displacement_m": float(seg_df["vertical_displacement_m"].sum()),
            }
        )
    return pd.DataFrame(rows)


def extract_clip_preview(clip_path: Path, output_path: Path) -> bool:
    cap = cv2.VideoCapture(str(clip_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    cv2.imwrite(str(output_path), frame)
    return True


def plot_pdinsighter_time_spent(
    clip_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    output_path: Path,
    label_col: str,
) -> None:
    valid_df = clip_df[clip_df[label_col].notna()].copy()
    if valid_df.empty:
        return
    color_map = {
        "sitting": "#e6ab02",
        "walking": "#d95f02",
        "standing": "#7570b3",
        "go downstairs": "#666666",
        "go upstairs": "#1b9e77",
    }
    labels_order = ["sitting", "walking", "standing", "go downstairs", "go upstairs"]
    total_duration_s = float(valid_df["end_s"].max() - valid_df["start_s"].min())
    duration_by_label = valid_df.groupby(label_col)["end_s"].count().mul(
        valid_df["end_s"].iloc[0] - valid_df["start_s"].iloc[0] if len(valid_df) == 1 else valid_df["end_s"].iloc[0] - valid_df["start_s"].iloc[0]
    )
    clip_duration_s = float(valid_df["end_s"].iloc[0] - valid_df["start_s"].iloc[0])
    duration_by_label = valid_df.groupby(label_col).size() * clip_duration_s

    fig = plt.figure(figsize=(14, 7), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[4.5, 1.4], height_ratios=[3, 1.6], wspace=0.18, hspace=0.22)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_timeline = fig.add_subplot(gs[1, 0])
    ax_detail = fig.add_subplot(gs[:, 1])

    labels_present = [label for label in labels_order if label in duration_by_label.index]
    mins = [duration_by_label[label] / 60.0 for label in labels_present]
    colors = [color_map[label] for label in labels_present]
    ax_bar.bar(labels_present, mins, color=colors)
    ax_bar.set_title("Time Spent in Data Capture (min)", fontsize=18, weight="bold")
    ax_bar.set_ylabel("Minutes")
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.tick_params(axis="x", rotation=0)

    action_rows = ["sitting", "go upstairs", "go downstairs", "walking", "standing"]
    y_map = {label: i for i, label in enumerate(action_rows)}
    for seg in segments_df.itertuples(index=False):
        label = getattr(seg, "label")
        if label not in y_map:
            continue
        ax_timeline.barh(
            y_map[label],
            seg.duration_s,
            left=seg.start_s,
            height=0.6,
            color=color_map.get(label, "#999999"),
            edgecolor="white",
        )
    ax_timeline.set_yticks(list(y_map.values()))
    ax_timeline.set_yticklabels(action_rows)
    ax_timeline.set_xlabel("Time (s)")
    ax_timeline.set_title("SLAM Rule-Based Timeline", fontsize=14, weight="bold")
    ax_timeline.spines[["top", "right"]].set_visible(False)

    dominant_label = duration_by_label.idxmax()
    pct_sitting = 100.0 * duration_by_label.get("sitting", 0.0) / total_duration_s if total_duration_s else 0.0
    ax_detail.axis("off")
    detail_text = (
        "Detailed View\n\n"
        f"Duration of Time: {int(total_duration_s // 60):02d}:{int(total_duration_s % 60):02d}\n\n"
        f"Dominant Label: {dominant_label.title()}\n\n"
        f"% Time Sitting: {pct_sitting:.0f}%\n\n"
        f"Total Clips: {len(valid_df)}\n\n"
        f"Automatic Classification Source: {label_col}"
    )
    ax_detail.text(
        0.05, 0.95, detail_text,
        va="top", ha="left", fontsize=13,
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#f7f7f7", edgecolor="#dddddd")
    )

    fig.suptitle("PD-Insighter Style Time Spent per Action View", fontsize=18, y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pdinsighter_event_view(
    clip_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    clips_dir: Path,
    output_path: Path,
    label_col: str,
) -> None:
    if segments_df.empty:
        return
    color_map = {
        "sitting": "#e6ab02",
        "walking": "#d95f02",
        "standing": "#7570b3",
        "go downstairs": "#666666",
        "go upstairs": "#1b9e77",
    }
    selected = segments_df.sort_values("duration_s", ascending=False).iloc[0]
    total_duration_s = float(clip_df["end_s"].max())

    preview_path = clips_dir / f"segment_preview_{int(selected['clip_start_idx']):04d}.png"
    extract_clip_preview(clips_dir / f"clip_{int(selected['clip_start_idx']):04d}.mp4", preview_path)

    fig = plt.figure(figsize=(14, 6.5), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.4], wspace=0.18)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_detail = fig.add_subplot(gs[0, 1])

    y_map = {label: i for i, label in enumerate(["go upstairs", "standing", "walking", "go downstairs", "sitting"])}
    for seg in segments_df.itertuples(index=False):
        label = getattr(seg, "label")
        if label not in y_map:
            continue
        alpha = 1.0 if seg.clip_start_idx == selected["clip_start_idx"] else 0.7
        ax_main.barh(
            y_map[label],
            seg.duration_s,
            left=seg.start_s,
            height=0.55,
            color=color_map.get(label, "#999999"),
            edgecolor="white",
            alpha=alpha,
        )

    ax_main.axvspan(selected["start_s"], selected["end_s"], color="#000000", alpha=0.08)
    ax_main.set_yticks(list(y_map.values()))
    ax_main.set_yticklabels(list(y_map.keys()))
    ax_main.set_xlim(0, total_duration_s)
    ax_main.set_xlabel("Time (s)")
    ax_main.set_title("Event Data", fontsize=18, weight="bold")
    ax_main.spines[["top", "right"]].set_visible(False)

    ax_main.plot([selected["start_s"], selected["end_s"]], [-1.0, -1.0], color="black", linewidth=3)
    ax_main.scatter([selected["start_s"], selected["end_s"]], [-1.0, -1.0], color="black", s=80, zorder=5)
    ax_main.text(selected["start_s"], -1.35, f"{selected['start_s']:.0f}s", ha="center", va="top")
    ax_main.text(selected["end_s"], -1.35, f"{selected['end_s']:.0f}s", ha="center", va="top")
    ax_main.set_ylim(-1.8, max(y_map.values()) + 0.8)

    ax_detail.axis("off")
    detail_text = (
        "Detailed View\n\n"
        f"Duration of Time: {int(selected['duration_s'] // 60):02d}:{int(selected['duration_s'] % 60):02d}\n\n"
        f"Automatic Classification: {selected['label'].title()}\n\n"
        f"Mean Speed: {selected['mean_speed_mps']:.3f} m/s\n"
        f"Mean Angular Speed: {selected['mean_angular_speed_rps']:.3f} rad/s\n\n"
        f"Horiz Disp: {selected['horizontal_displacement_m']:.3f} m\n"
        f"Vert Disp: {selected['vertical_displacement_m']:.3f} m"
    )
    ax_detail.text(
        0.05, 0.95, detail_text,
        va="top", ha="left", fontsize=12,
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#f7f7f7", edgecolor="#dddddd")
    )
    if preview_path.exists():
        img = cv2.cvtColor(cv2.imread(str(preview_path)), cv2.COLOR_BGR2RGB)
        ax_inset = ax_detail.inset_axes([0.1, 0.03, 0.8, 0.38])
        ax_inset.imshow(img)
        ax_inset.set_title("Representative Clip", fontsize=11)
        ax_inset.axis("off")

    fig.suptitle("PD-Insighter Style Individual Clip View", fontsize=18, y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_interactive_dashboard(
    clip_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    output_path: Path,
    gif_dir: Path,
    *,
    trial_id: str | None = None,
    recording_stem: str | None = None,
    video_path: Path | None = None,
    audio_path: Path | None = None,
    audio_offset_s: float = 0.0,
    clip_audio_map: dict[int, Path] | None = None,
    clip_media_map: dict[int, Path] | None = None,
    secondary_clip_media_map: dict[int, Path] | None = None,
    primary_view_label: str = "Aria View",
    secondary_view_label: str | None = None,
    primary_default_rotation_deg: int = 90,
    secondary_default_rotation_deg: int = 0,
    primary_full_video_uri: Path | str | None = None,
    secondary_full_video_uri: Path | str | None = None,
    primary_video_trim_start_s: float = 0.0,
    secondary_video_trim_start_s: float = 0.0,
    trial_selector_href: str | None = None,
    secondary_mode_href: str | None = None,
    secondary_mode_label: str | None = None,
) -> None:
    valid_df = clip_df.copy().sort_values("clip_idx").reset_index(drop=True)
    if valid_df.empty:
        return

    trial_label = trial_id or output_path.parent.name
    recording_label = recording_stem or Path(valid_df["clip_path"].iloc[0]).parent.parent.name
    escaped_trial_label = html.escape(trial_label)
    escaped_recording_label = html.escape(recording_label)
    escaped_title = html.escape(f"{trial_label} - {recording_label}")
    color_map = {
        "sitting": "#e6ab02",
        "walking": "#d95f02",
        "standing": "#7570b3",
        "go downstairs": "#666666",
        "downstairs": "#666666",
        "go upstairs": "#1b9e77",
        "upstairs": "#1b9e77",
        "missing": "#bbbbbb",
    }
    total_duration_s = float(valid_df["end_s"].max())
    clip_duration_s = float(valid_df["end_s"].iloc[0] - valid_df["start_s"].iloc[0])
    recording_start_epoch_s = float(valid_df["start_epoch_s"].min())

    clip_records = []
    for row in valid_df.to_dict(orient="records"):
        label = row.get("hybrid_prediction") or row.get("rule_prediction") or "missing"
        gif_path = gif_dir / f"clip_{int(row['clip_idx']):04d}.gif"
        clip_idx = int(row["clip_idx"])
        clip_media_path = clip_media_map.get(clip_idx) if clip_media_map else None
        clip_secondary_media_path = secondary_clip_media_map.get(clip_idx) if secondary_clip_media_map else None
        clip_records.append(
            {
                "clip_idx": clip_idx,
                "clip_path": str(row["clip_path"]),
                "clip_video_uri": resolve_media_href(output_path, clip_media_path) or relative_media_href(output_path, Path(str(row["clip_path"]))),
                "clip_secondary_video_uri": resolve_media_href(output_path, clip_secondary_media_path),
                "full_video_uri": resolve_media_href(output_path, primary_full_video_uri),
                "full_secondary_video_uri": resolve_media_href(output_path, secondary_full_video_uri),
                "clip_gif_uri": relative_media_href(output_path, gif_path),
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
                "duration_s": float(row["end_s"] - row["start_s"]),
                "source_start_s": float(primary_video_trim_start_s + row["start_s"]),
                "source_end_s": float(primary_video_trim_start_s + row["end_s"]),
                "secondary_source_start_s": float(secondary_video_trim_start_s + row["start_s"]),
                "secondary_source_end_s": float(secondary_video_trim_start_s + row["end_s"]),
                "label": label,
                "rule_prediction": row.get("rule_prediction"),
                "old_rule_prediction": row.get("old_rule_prediction") or row.get("rule_prediction") or "",
                "old_hybrid_prediction": row.get("old_hybrid_prediction") or "",
                "improved_rule_prediction": row.get("improved_rule_prediction") or "",
                "criteria_v2_prediction": row.get("criteria_v2_prediction") or "",
                "criteria_v2_strict_match": row.get("criteria_v2_strict_match") or "",
                "criteria_v2_transition_aware_match": row.get("criteria_v2_transition_aware_match") or "",
                "criteria_v3_prediction": row.get("criteria_v3_prediction") or "",
                "criteria_v3_transition_reason": row.get("criteria_v3_transition_reason") or "",
                "criteria_v3_strict_match": row.get("criteria_v3_strict_match") or "",
                "criteria_v3_transition_aware_match": row.get("criteria_v3_transition_aware_match") or "",
                "criteria_v4_prediction": row.get("criteria_v4_prediction") or "",
                "criteria_v4_context_state": row.get("criteria_v4_context_state") or "",
                "criteria_v4_reason": row.get("criteria_v4_reason") or "",
                "criteria_v4_transition_evidence": row.get("criteria_v4_transition_evidence") or "",
                "criteria_v4_strict_match": row.get("criteria_v4_strict_match") or "",
                "criteria_v4_transition_aware_match": row.get("criteria_v4_transition_aware_match") or "",
                "criteria_target_label": row.get("criteria_target_label") or "",
                "model_prediction_source": row.get("model_prediction_source") or "",
                "horizontal_displacement_m": float(row["horizontal_displacement_m"]),
                "vertical_displacement_m": float(row["vertical_displacement_m"]),
                "mean_linear_speed_mps": float(row["mean_linear_speed_mps"]),
                "speed_variance": float(row["speed_variance"]),
                "mean_angular_speed_rps": float(row["mean_angular_speed_rps"]),
                "angular_speed_variance": float(row["angular_speed_variance"]),
                "gravity_variance": float(row["gravity_variance"]),
                "mean_z_m": float(row["mean_z_m"]),
                "height_slope_mps": float(row.get("height_slope_mps") or 0.0),
                "horizontal_path_length_m": float(row.get("horizontal_path_length_m") or 0.0),
                "window_median_height_m": float(row.get("window_median_height_m") or row.get("mean_z_m") or 0.0),
                "spoken_words_detected_in_clip": row.get("spoken_words_detected_in_clip") or "",
                "spoken_word_event_count": int(float(row.get("spoken_word_event_count") or 0)),
                "spoken_word_times_s": row.get("spoken_word_times_s") or "",
                "assumed_actions_in_clip": row.get("assumed_actions_in_clip") or "",
                "primary_assumed_action": row.get("primary_assumed_action") or "",
                "is_transition_clip": row.get("is_transition_clip") or "",
                "state_change_times_in_clip_s": row.get("state_change_times_in_clip_s") or "",
                "color": color_map.get(str(label).strip().lower(), "#999999"),
            }
        )

    seg_records = []
    for row in segments_df.to_dict(orient="records"):
        label = row.get("label") or "missing"
        seg_records.append(
            {
                "label": label,
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
                "duration_s": float(row["duration_s"]),
                "clip_start_idx": int(row["clip_start_idx"]),
                "clip_end_idx": int(row["clip_end_idx"]),
                "color": color_map.get(str(label).strip().lower(), "#999999"),
            }
        )

    duration_by_label = valid_df.groupby("rule_prediction").size() * clip_duration_s
    bar_records = [
        {
            "label": label,
            "minutes": float(duration / 60.0),
            "seconds": float(duration),
            "color": color_map.get(str(label).strip().lower(), "#999999"),
        }
        for label, duration in duration_by_label.items()
    ]
    escaped_primary_view_label = html.escape(primary_view_label)
    escaped_secondary_view_label = html.escape(secondary_view_label or "")
    has_secondary_view = any(record.get("clip_secondary_video_uri") for record in clip_records)
    trial_nav_options = collect_dashboard_trial_options(output_path)
    trial_nav_json = json.dumps(trial_nav_options)
    effective_trial_selector_href = html.escape(trial_selector_href or "../pd_insighter_trial_selector.html", quote=True)
    secondary_mode_link_html = (
        f'<a class="utility-link" href="{html.escape(secondary_mode_href, quote=True)}">{html.escape(secondary_mode_label or "Open Alternate Review")}</a>'
        if secondary_mode_href
        else ""
    )
    secondary_view_blurb = (
        f"{escaped_primary_view_label} drives the controls and audio. "
        f"{escaped_secondary_view_label} follows automatically."
        if has_secondary_view
        else "This player loads the 2-second clip that contains the currently selected time. Clicking a segment starts this clip automatically."
    )
    primary_controls_attr = "" if has_secondary_view else "controls"
    secondary_controls_attr = ""

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{escaped_title}</title>
  <style>
    :root {{
      --bg: #eef2f1;
      --card: rgba(255,255,255,0.9);
      --text: #14201c;
      --muted: #61716a;
      --border: rgba(20, 32, 28, 0.09);
      --accent: #1c8c6a;
      --accent-soft: rgba(28, 140, 106, 0.12);
      --shadow: 0 18px 44px rgba(18, 28, 24, 0.08);
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(28,140,106,0.12), transparent 24%),
        linear-gradient(180deg, #f4f8f7 0%, var(--bg) 100%);
      color: var(--text);
      margin: 0;
      padding: 28px;
    }}
    h1, h2, h3 {{ margin: 0; }}
    .page {{
      display: grid;
      gap: 22px;
      max-width: 1680px;
      margin: 0 auto;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      backdrop-filter: blur(10px);
      border-radius: 24px;
      padding: 22px;
      box-shadow: var(--shadow);
    }}
    .hero {{
      display: grid;
      gap: 12px;
    }}
    .eyebrow {{
      font-size: 13px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--accent);
      font-weight: 700;
    }}
    .hero h1 {{
      font-size: clamp(32px, 4vw, 52px);
      line-height: 0.98;
      max-width: 12ch;
    }}
    .hero-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 9px 13px;
      background: rgba(255,255,255,0.78);
      font-size: 14px;
      color: var(--text);
    }}
    .utility-bar {{
      position: sticky;
      top: 12px;
      z-index: 10;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 14px 16px;
    }}
    .utility-links {{
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .utility-link {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 1px solid rgba(20,32,28,0.08);
      border-radius: 999px;
      padding: 8px 14px;
      background: rgba(255,255,255,0.92);
      color: var(--text);
      text-decoration: none;
      font-size: 13px;
      font-weight: 600;
    }}
    .utility-select {{
      min-width: 280px;
      max-width: 100%;
      border: 1px solid rgba(20,32,28,0.08);
      border-radius: 999px;
      padding: 9px 14px;
      background: rgba(255,255,255,0.96);
      color: var(--text);
      font-size: 13px;
      font-weight: 600;
    }}
    .navigator-card {{
      display: grid;
      gap: 16px;
    }}
    .section-heading {{
      display: flex;
      flex-wrap: wrap;
      align-items: end;
      justify-content: space-between;
      gap: 10px;
    }}
    .section-heading p {{
      margin: 0;
      color: var(--muted);
      max-width: 78ch;
      font-size: 14px;
    }}
    .section-heading.compact {{
      align-items: center;
    }}
    .insight-grid {{
      display: grid;
      grid-template-columns: 3.4fr 1.2fr;
      gap: 18px;
    }}
    .bar-wrap {{
      display: flex;
      align-items: end;
      justify-content: space-around;
      gap: 14px;
      min-height: 280px;
      margin-top: 16px;
    }}
    .bar-col {{
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 10px;
    }}
    .bar {{
      width: 100%;
      max-width: 96px;
      border-radius: 10px 10px 0 0;
      min-height: 4px;
    }}
    .bar-label {{
      font-size: 14px;
      text-align: center;
      color: var(--text);
    }}
    .media-card {{
      display: grid;
      gap: 18px;
    }}
    .timeline-rows {{
      margin-top: 6px;
      display: grid;
      gap: 10px;
    }}
    .timeline-row {{
      display: grid;
      grid-template-columns: 120px 1fr;
      align-items: center;
      gap: 10px;
    }}
    .timeline-track {{
      position: relative;
      background: rgba(20,32,28,0.06);
      height: 18px;
      border-radius: 999px;
      overflow: hidden;
      border: 1px solid rgba(20,32,28,0.05);
    }}
    .timeline-segment {{
      position: absolute;
      top: 0;
      bottom: 0;
      border-radius: 999px;
      cursor: pointer;
      opacity: 0.95;
    }}
    .timeline-segment:hover {{
      outline: 2px solid rgba(20,32,28,0.85);
      z-index: 2;
    }}
    .detail-stack {{
      display: grid;
      gap: 14px;
      font-size: 15px;
      line-height: 1.45;
    }}
    .stat {{
      color: var(--muted);
    }}
    .event-panel {{
      display: grid;
      gap: 14px;
    }}
    .event-track {{
      position: relative;
      background: rgba(20,32,28,0.08);
      height: 34px;
      border-radius: 999px;
      overflow: hidden;
      border: 1px solid rgba(20,32,28,0.07);
      box-shadow: inset 0 1px 3px rgba(0,0,0,0.04);
    }}
    .event-segment {{
      position: absolute;
      top: 0;
      bottom: 0;
      cursor: pointer;
      opacity: 0.94;
    }}
    .event-segment.active {{
      outline: 3px solid rgba(20,32,28,0.9);
      z-index: 3;
    }}
    .scrubber-stack {{
      display: grid;
      gap: 10px;
      padding: 12px 14px;
      border-radius: 18px;
      background: rgba(255,255,255,0.68);
      border: 1px solid rgba(20,32,28,0.05);
    }}
    .scrubber {{
      width: 100%;
      accent-color: var(--accent);
    }}
    .time-readout {{
      font-variant-numeric: tabular-nums;
      color: var(--muted);
    }}
    .clip-view-grid {{
      display: grid;
      gap: 18px;
    }}
    .clip-view-grid.dual {{
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    }}
    .clip-view-card {{
      display: grid;
      gap: 10px;
      padding: 16px;
      border-radius: 20px;
      background: rgba(247, 250, 249, 0.88);
      border: 1px solid rgba(20,32,28,0.05);
    }}
    .clip-view-card h3 {{
      margin: 0;
      font-size: 17px;
    }}
    .clip-view-card .small {{
      margin-top: -4px;
    }}
    .clip-frame {{
      width: 100%;
      min-height: 430px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(180deg, #0b0d10, #10151b);
      border-radius: 18px;
      overflow: hidden;
      padding: 12px;
      box-sizing: border-box;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
    }}
    .clip-frame video,
    .clip-frame img {{
      width: 100%;
      max-height: 620px;
      object-fit: contain;
      border-radius: 12px;
      background: #000;
    }}
    .clip-controls {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .clip-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    .clip-controls button {{
      border: 1px solid rgba(20,32,28,0.08);
      background: rgba(255,255,255,0.92);
      color: var(--text);
      border-radius: 999px;
      padding: 8px 13px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
      transition: transform 0.15s ease, background 0.15s ease, box-shadow 0.15s ease;
    }}
    .clip-controls button:hover {{
      background: #ffffff;
      box-shadow: 0 8px 18px rgba(18,28,24,0.08);
      transform: translateY(-1px);
    }}
    #playPauseClipButton {{
      background: linear-gradient(180deg, #1c8c6a, #13624a);
      color: white;
      border-color: transparent;
    }}
    .clip-frame video,
    .clip-frame img {{
      transform-origin: center center;
      transition: transform 0.15s ease;
    }}
    .audio-heading {{
      display: grid;
      gap: 5px;
    }}
    .audio-heading h3 {{
      margin: 0;
      font-size: 22px;
    }}
    .mini-note {{
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }}
    .mono {{
      font-variant-numeric: tabular-nums;
    }}
    .small {{
      font-size: 13px;
      color: var(--muted);
    }}
    .clip-metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
    }}
    .metric-card {{
      border-radius: 16px;
      padding: 11px 12px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(20,32,28,0.05);
      min-height: 58px;
    }}
    .metric-card b {{
      display: block;
      font-size: 11px;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 5px;
    }}
    .metric-card span {{
      display: block;
      font-size: 15px;
      line-height: 1.25;
      color: var(--text);
      word-break: break-word;
    }}
    .annotation-panel {{
      display: grid;
      gap: 12px;
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(255,255,255,0.68);
      border: 1px solid rgba(20,32,28,0.05);
    }}
    .annotation-status-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
    }}
    .annotation-form {{
      display: grid;
      gap: 10px;
    }}
    .annotation-form-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      align-items: end;
    }}
    .annotation-field {{
      display: grid;
      gap: 6px;
    }}
    .annotation-field span {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 700;
    }}
    .annotation-field select {{
      border: 1px solid rgba(20,32,28,0.08);
      border-radius: 12px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.96);
      color: var(--text);
      font-size: 14px;
      font-weight: 600;
    }}
    .annotation-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    .annotation-actions button {{
      border: 1px solid rgba(20,32,28,0.08);
      background: rgba(255,255,255,0.92);
      color: var(--text);
      border-radius: 999px;
      padding: 8px 13px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
    }}
    .annotation-actions button.primary {{
      background: linear-gradient(180deg, #1c8c6a, #13624a);
      color: white;
      border-color: transparent;
    }}
    .annotation-note {{
      font-size: 12px;
      color: var(--muted);
      line-height: 1.4;
    }}
    .summary-kpis {{
      display: grid;
      gap: 12px;
    }}
    .summary-kpi {{
      border-radius: 18px;
      padding: 14px 15px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(20,32,28,0.05);
    }}
    .summary-kpi b {{
      display: block;
      font-size: 12px;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .summary-kpi span {{
      display: block;
      font-size: 18px;
      line-height: 1.25;
    }}
    @media (max-width: 1100px) {{
      .clip-view-grid.dual,
      .insight-grid {{
        grid-template-columns: 1fr;
      }}
      .clip-metrics-grid {{
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }}
    }}
    @media (max-width: 760px) {{
      body {{
        padding: 16px;
      }}
      .card {{
        padding: 18px;
      }}
      .clip-frame {{
        min-height: 280px;
      }}
      .clip-metrics-grid {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
    @media (max-width: 560px) {{
      .clip-metrics-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="card utility-bar">
      <div class="utility-links">
        <a class="utility-link" href="{effective_trial_selector_href}">Trial Selector</a>
        {secondary_mode_link_html}
        <div class="small">Jump to another processed trial without leaving the dashboard workflow.</div>
      </div>
      <select id="trialJumpSelect" class="utility-select"></select>
    </section>
    <section class="card hero">
      <div class="eyebrow">PD-Insighter Style Interactive View</div>
      <h1>{escaped_recording_label}</h1>
      <div class="hero-meta">
        <span class="pill"><b>Trial</b> {escaped_trial_label}</span>
        <span class="pill"><b>Recording</b> {escaped_recording_label}</span>
        <span class="pill"><b>Automatic Classification Source</b> SLAM Rule-Based</span>
      </div>
    </section>
    <div class="insight-grid">
      <section class="card">
        <h2>Time Spent in Data Capture (min)</h2>
        <div id="barWrap" class="bar-wrap"></div>
        <div id="timelineRows" class="timeline-rows"></div>
      </section>
      <aside class="card">
        <h2>Detailed View</h2>
        <div class="summary-kpis" id="summaryPanel"></div>
      </aside>
    </div>
    <section class="card navigator-card">
      <div class="section-heading">
        <div>
          <h2>Session Navigator</h2>
          <p>Click directly on the color bar at the moment you want to inspect, or drag the slider for fine control. The synchronized two-view clip section below updates to the exact 2-second window around that chosen point.</p>
        </div>
      </div>
      <div class="event-panel">
        <div id="eventTrack" class="event-track"></div>
        <div class="scrubber-stack">
          <input id="scrubber" class="scrubber" type="range" min="0" max="{total_duration_s:.6f}" step="{clip_duration_s/10:.6f}" value="0" />
          <div class="time-readout mono" id="timeReadout"></div>
        </div>
      </div>
    </section>
    <section class="card media-card">
      <div class="section-heading compact">
        <div class="audio-heading">
          <h2>Synchronized Clip Views</h2>
          <div class="small">{secondary_view_blurb}</div>
        </div>
        <div class="clip-actions">
          {'<button id="playPauseClipButton" type="button">Play Both</button>' if has_secondary_view else ""}
          {'<button id="restartClipButton" type="button">Restart Clip</button>' if has_secondary_view else ""}
          <button id="rotatePrimaryClipButton" type="button">Rotate {escaped_primary_view_label} 90°</button>
          {f'<button id="rotateSecondaryClipButton" type="button">Rotate {escaped_secondary_view_label} 90°</button>' if has_secondary_view else ""}
        </div>
      </div>
      <div id="clipViewGrid" class="clip-view-grid{' dual' if has_secondary_view else ''}">
        <div class="clip-view-card">
          <h3>{escaped_primary_view_label}</h3>
          <div class="small">Audio-enabled master view for the selected 2-second window.</div>
          <div class="clip-frame">
            <video id="clipVideo" {primary_controls_attr} preload="metadata"></video>
          </div>
        </div>
        <div id="secondaryClipCard" class="clip-view-card" style="display: {'grid' if has_secondary_view else 'none'};">
          <h3>{escaped_secondary_view_label}</h3>
          <div class="small">Synchronized companion view of the same selected clip window.</div>
          <div class="clip-frame">
            <video id="secondaryClipVideo" {secondary_controls_attr} preload="metadata" muted playsinline></video>
          </div>
        </div>
      </div>
      <div class="clip-controls">
        <div class="mini-note">{escaped_secondary_view_label if has_secondary_view else escaped_primary_view_label} preview defaults to {secondary_default_rotation_deg if has_secondary_view else primary_default_rotation_deg}° rotation.</div>
      </div>
      <div id="clipPanel" class="clip-metrics-grid"></div>
      <div class="annotation-panel">
        <div id="annotationStatus" class="annotation-status-grid"></div>
        <div class="annotation-form">
          <div class="annotation-form-row">
            <label class="annotation-field">
              <span>Label Mode</span>
              <select id="manualModeSelect">
                <option value="">Keep auto spoken action</option>
                <option value="single">Single motion</option>
                <option value="transition">Transition</option>
              </select>
            </label>
            <label id="singleActionWrap" class="annotation-field" style="display:none;">
              <span>Motion</span>
              <select id="singleActionSelect"></select>
            </label>
            <label id="transitionCountWrap" class="annotation-field" style="display:none;">
              <span>Transition Count</span>
              <select id="transitionCountSelect">
                <option value="2">2 motions</option>
                <option value="3">3 motions</option>
                <option value="4">4 motions</option>
                <option value="5">5 motions</option>
              </select>
            </label>
          </div>
          <div id="transitionActionsWrap" class="annotation-form-row" style="display:none;"></div>
          <div class="annotation-actions">
            <button id="saveManualLabelButton" class="primary" type="button">Save Manual Label</button>
            <button id="resetManualLabelButton" type="button">Reset to Auto</button>
            <div id="annotationServerNote" class="annotation-note">Manual labeling loads when the annotation API is available.</div>
          </div>
        </div>
      </div>
    </section>
  </div>
  <script>
    const totalDuration = {total_duration_s:.6f};
    const bars = {json.dumps(bar_records)};
    const clips = {json.dumps(clip_records)};
    const segments = {json.dumps(seg_records)};
    const labelOrder = ["sitting","walking","standing","go downstairs","go upstairs"];

    const barWrap = document.getElementById("barWrap");
    const timelineRows = document.getElementById("timelineRows");
    const summaryPanel = document.getElementById("summaryPanel");
    const clipPanel = document.getElementById("clipPanel");
    const trialJumpSelect = document.getElementById("trialJumpSelect");
    const eventTrack = document.getElementById("eventTrack");
    const scrubber = document.getElementById("scrubber");
    const timeReadout = document.getElementById("timeReadout");
    const clipVideo = document.getElementById("clipVideo");
    const secondaryClipCard = document.getElementById("secondaryClipCard");
    const secondaryClipVideo = document.getElementById("secondaryClipVideo");
    const playPauseClipButton = document.getElementById("playPauseClipButton");
    const restartClipButton = document.getElementById("restartClipButton");
    const rotatePrimaryClipButton = document.getElementById("rotatePrimaryClipButton");
    const rotateSecondaryClipButton = document.getElementById("rotateSecondaryClipButton");
    const annotationStatus = document.getElementById("annotationStatus");
    const manualModeSelect = document.getElementById("manualModeSelect");
    const singleActionWrap = document.getElementById("singleActionWrap");
    const singleActionSelect = document.getElementById("singleActionSelect");
    const transitionCountWrap = document.getElementById("transitionCountWrap");
    const transitionCountSelect = document.getElementById("transitionCountSelect");
    const transitionActionsWrap = document.getElementById("transitionActionsWrap");
    const saveManualLabelButton = document.getElementById("saveManualLabelButton");
    const resetManualLabelButton = document.getElementById("resetManualLabelButton");
    const annotationServerNote = document.getElementById("annotationServerNote");
    let primaryClipRotationDeg = {primary_default_rotation_deg};
    let secondaryClipRotationDeg = {secondary_default_rotation_deg};
    let currentSelectedTime = 0;
    let activeClipIdx = null;
    let activeClipVideoSrc = "";
    let activeSecondaryClipVideoSrc = "";
    let pendingPrimarySeekS = 0;
    let pendingSecondarySeekS = 0;
    let activePrimaryClipEndS = null;
    let activeSecondaryClipEndS = null;
    let pendingClipAutoplay = false;
    const hasSecondaryView = {str(has_secondary_view).lower()};
    const trialJumpOptions = {trial_nav_json};
    const trialId = "{escaped_trial_label}";
    const actionOptions = ["walking", "sitting", "standing", "upstairs", "downstairs"];
    const annotationApiUrl = `/api/manual-labels?trial_id=${{encodeURIComponent(trialId)}}`;
    const annotationSaveUrl = "/api/manual-labels";
    const annotationState = {{
      apiAvailable: false,
      annotations: {{}},
      summary: null,
    }};
    const annotationEditorState = {{
      clipIdx: null,
      dirty: false,
      autosaveTimer: null,
      saveInFlight: false,
    }};

    function fmtTime(seconds) {{
      const s = Math.max(0, Number(seconds) || 0);
      const mins = Math.floor(s / 60);
      const secs = Math.floor(s % 60);
      return mins > 0 ? `${{mins}}:${{String(secs).padStart(2, "0")}}` : `${{secs}}s`;
    }}

    function clamp(value, min, max) {{
      return Math.min(Math.max(value, min), max);
    }}

    function fmtClock(epochSeconds) {{
      const d = new Date(epochSeconds * 1000);
      return d.toLocaleString();
    }}

    function escapeHtml(value) {{
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }}

    function normalizeModelLabel(label) {{
      const transitionActions = transitionLabelToActions(label);
      if (transitionActions.length) return canonicalTransitionLabel(transitionActions);
      return normalizeActionLabel(label);
    }}

    function getModelPredictionLabel(clip) {{
      return clip.criteria_v4_prediction || clip.criteria_v3_prediction || clip.criteria_v2_prediction || clip.label || clip.rule_prediction || "";
    }}

    function normalizeActionLabel(label) {{
      const raw = String(label || "").trim().toLowerCase();
      if (raw === "sit") return "sitting";
      if (raw === "stand") return "standing";
      if (raw === "walk") return "walking";
      if (raw === "go upstairs") return "upstairs";
      if (raw === "go downstairs") return "downstairs";
      return raw;
    }}

    function transitionLabelToActions(label) {{
      const raw = String(label || "").trim().toLowerCase()
        .replace(/_/g, "-")
        .replace(/\\s+to\\s+/g, "-to-")
        .replace(/\\s*->\\s*/g, "-to-");
      if (!raw) return [];
      const parts = raw.includes("-to-")
        ? raw.split("-to-")
        : raw.split(/\\s*(?:\\||-|,|\\/)\\s*/);
      const actions = [];
      for (const part of parts) {{
        const action = normalizeActionLabel(part);
        if (actionOptions.includes(action) && !actions.includes(action)) actions.push(action);
      }}
      return actions.length > 1 ? actions : [];
    }}

    function canonicalTransitionLabel(actions) {{
      const ordered = [];
      for (const action of actions || []) {{
        const normalized = normalizeActionLabel(action);
        if (normalized && actionOptions.includes(normalized) && !ordered.includes(normalized)) {{
          ordered.push(normalized);
        }}
      }}
      if (!ordered.length) return "";
      if (ordered.length === 1) return ordered[0];
      const key = `${{ordered[0]}}-${{ordered[1]}}`;
      const clinicalTransitions = {{
        "sitting-standing": "sit-to-stand",
        "standing-sitting": "stand-to-sit",
        "standing-walking": "stand-to-walk",
        "walking-standing": "walk-to-stand",
      }};
      return clinicalTransitions[key] || ordered.join("-to-");
    }}

    function parseActionDisplay(value) {{
      const raw = String(value || "").trim().toLowerCase();
      const transitionActions = transitionLabelToActions(raw);
      if (transitionActions.length) return transitionActions;
      return raw
        .split(/\\s*(?:\\||-|,|\\/)\\s*/)
        .map(item => normalizeActionLabel(item))
        .filter(item => actionOptions.includes(item));
    }}

    function formatActionDisplay(actions) {{
      const ordered = [];
      for (const action of (actions || [])) {{
        if (action && !ordered.includes(action)) ordered.push(action);
      }}
      return ordered.join("-");
    }}

    function getManualAnnotationForClip(clipIdx) {{
      return annotationState.annotations[String(clipIdx)] || null;
    }}

    function annotationPayloadKey(mode, actions) {{
      return JSON.stringify({{
        mode: mode || "",
        actions: Array.isArray(actions) ? actions.filter(Boolean) : [],
      }});
    }}

    function getAutoActionParts(clip) {{
      return parseActionDisplay(clip.assumed_actions_in_clip || clip.primary_assumed_action || "");
    }}

    function getEffectiveActions(clip) {{
      const annotation = getManualAnnotationForClip(clip.clip_idx);
      if (annotation && Array.isArray(annotation.actions) && annotation.actions.length) {{
        return annotation.actions.slice();
      }}
      return getAutoActionParts(clip);
    }}

    function getManualCorrectionDisplay(clip) {{
      const annotation = getManualAnnotationForClip(clip.clip_idx);
      if (!annotation || !Array.isArray(annotation.actions)) return "";
      return formatActionDisplay(annotation.actions);
    }}

    function getEffectiveLabelDisplay(clip) {{
      return formatActionDisplay(getEffectiveActions(clip));
    }}

    function getModelMatchFlags(clip) {{
      const effectiveActions = getEffectiveActions(clip);
      const normalizedPrediction = normalizeModelLabel(getModelPredictionLabel(clip));
      if (!effectiveActions.length) {{
        return {{ strict: null, transitionAware: null }};
      }}
      const normalizedActions = effectiveActions.map(action => normalizeActionLabel(action)).filter(action => actionOptions.includes(action));
      const target = canonicalTransitionLabel(normalizedActions);
      return {{
        strict: normalizedPrediction === target,
        transitionAware: normalizedPrediction === target,
      }};
    }}

    function fmtAccuracy(value) {{
      if (value == null || Number.isNaN(Number(value))) return "N/A";
      return `${{(Number(value) * 100).toFixed(1)}}%`;
    }}

    function buildActionOptionsHtml(selectedValue = "") {{
      return actionOptions
        .map(action => `<option value="${{action}}"${{selectedValue === action ? " selected" : ""}}>${{action}}</option>`)
        .join("");
    }}

    function renderTransitionSelectors(count, selectedActions = []) {{
      transitionActionsWrap.innerHTML = "";
      for (let index = 0; index < count; index += 1) {{
        const label = document.createElement("label");
        label.className = "annotation-field";
        label.innerHTML = `<span>Motion ${{index + 1}}</span><select data-transition-index="${{index}}">${{buildActionOptionsHtml(selectedActions[index] || "")}}</select>`;
        transitionActionsWrap.appendChild(label);
      }}
    }}

    function syncAnnotationInputsToClip(clip, force = false) {{
      if (!clip) return;
      if (!force && annotationEditorState.dirty) return;
      const annotation = getManualAnnotationForClip(clip.clip_idx);
      manualModeSelect.value = annotation ? annotation.mode : "";
      if (annotation && annotation.mode === "single") {{
        singleActionSelect.value = (annotation.actions && annotation.actions[0]) || actionOptions[0];
      }} else {{
        singleActionSelect.value = actionOptions[0];
      }}
      const transitionCount = annotation && annotation.mode === "transition"
        ? Math.max(2, Math.min(5, (annotation.actions || []).length))
        : 2;
      transitionCountSelect.value = String(transitionCount);
      renderTransitionSelectors(transitionCount, annotation && annotation.mode === "transition" ? annotation.actions : []);
      updateAnnotationVisibility();
      annotationEditorState.clipIdx = clip.clip_idx;
      annotationEditorState.dirty = false;
    }}

    function updateAnnotationVisibility() {{
      const mode = manualModeSelect.value;
      singleActionWrap.style.display = mode === "single" ? "grid" : "none";
      transitionCountWrap.style.display = mode === "transition" ? "grid" : "none";
      transitionActionsWrap.style.display = mode === "transition" ? "grid" : "none";
      if (mode === "transition") {{
        const count = Number(transitionCountSelect.value || 2);
        const existingValues = Array.from(transitionActionsWrap.querySelectorAll("select")).map(el => el.value);
        renderTransitionSelectors(count, existingValues);
      }}
    }}

    function collectManualAnnotationInput() {{
      const mode = manualModeSelect.value;
      if (!mode) {{
        return {{ mode: "", actions: [] }};
      }}
      if (mode === "single") {{
        return {{ mode: "single", actions: [singleActionSelect.value].filter(Boolean) }};
      }}
      const actions = Array.from(transitionActionsWrap.querySelectorAll("select"))
        .map(el => el.value)
        .filter(Boolean);
      return {{ mode: "transition", actions }};
    }}

    function clearAnnotationAutosaveTimer() {{
      if (annotationEditorState.autosaveTimer) {{
        clearTimeout(annotationEditorState.autosaveTimer);
        annotationEditorState.autosaveTimer = null;
      }}
    }}

    function bindAnnotationEditorToSelectedClip() {{
      const clip = getClipAtTime(currentSelectedTime);
      if (!clip) return null;
      annotationEditorState.clipIdx = clip.clip_idx;
      return clip;
    }}

    function markAnnotationEditorDirty() {{
      const clip = bindAnnotationEditorToSelectedClip();
      if (!clip) return null;
      annotationEditorState.dirty = true;
      return clip;
    }}

    function queueAnnotationAutosave() {{
      if (!annotationState.apiAvailable) return;
      clearAnnotationAutosaveTimer();
      annotationEditorState.autosaveTimer = window.setTimeout(() => {{
        saveManualAnnotationForCurrentClip(false, {{ autosave: true }});
      }}, 500);
    }}

    function pauseClipReviewForAnnotation() {{
      clipVideo.pause();
      pauseSecondaryClipVideo();
      setPlayButtonState(false);
    }}

    function handleAnnotationInputChange() {{
      pauseClipReviewForAnnotation();
      const clip = markAnnotationEditorDirty();
      if (!clip) return;
      annotationServerNote.textContent = `Pending manual label for clip #${{clip.clip_idx}}...`;
      queueAnnotationAutosave();
    }}

    function renderAnnotationStatus(clip) {{
      const autoAction = clip.assumed_actions_in_clip || "Not detected";
      const manualCorrection = getManualCorrectionDisplay(clip) || "None";
      const effectiveLabel = getEffectiveLabelDisplay(clip) || "Not detected";
      const matchFlags = getModelMatchFlags(clip);
      const agreementText = matchFlags.transitionAware == null
        ? "N/A"
        : (matchFlags.transitionAware ? "Match" : "Mismatch");
      annotationStatus.innerHTML = `
        <div class="metric-card"><b>Auto Spoken Action</b><span>${{escapeHtml(autoAction)}}</span></div>
        <div class="metric-card"><b>Manual Correction</b><span>${{escapeHtml(manualCorrection)}}</span></div>
        <div class="metric-card"><b>Effective Manual Label</b><span>${{escapeHtml(effectiveLabel)}}</span></div>
        <div class="metric-card"><b>Model Agreement</b><span>${{escapeHtml(agreementText)}}</span></div>
      `;
    }}

    function setCurrentTimeSafe(mediaEl, time) {{
      if (!mediaEl || !Number.isFinite(time)) return;
      try {{
        if (!Number.isFinite(mediaEl.currentTime) || Math.abs(mediaEl.currentTime - time) > 0.12) {{
          mediaEl.currentTime = time;
        }}
      }} catch (err) {{}}
    }}

    function seekClipVideo(targetSeconds) {{
      const duration = Number.isFinite(clipVideo.duration) ? clipVideo.duration : null;
      const target = duration == null ? targetSeconds : clamp(targetSeconds, 0, Math.max(0, duration - 0.05));
      setCurrentTimeSafe(clipVideo, target);
    }}

    function seekSecondaryClipVideo(targetSeconds) {{
      if (!secondaryClipVideo || !secondaryClipVideo.src) return;
      const duration = Number.isFinite(secondaryClipVideo.duration) ? secondaryClipVideo.duration : null;
      const target = duration == null ? targetSeconds : clamp(targetSeconds, 0, Math.max(0, duration - 0.05));
      setCurrentTimeSafe(secondaryClipVideo, target);
    }}

    async function playClipVideo() {{
      try {{
        await clipVideo.play();
      }} catch (err) {{}}
    }}

    async function playSecondaryClipVideo() {{
      if (!secondaryClipVideo || !secondaryClipVideo.src) return;
      try {{
        await secondaryClipVideo.play();
      }} catch (err) {{}}
    }}

    function pauseSecondaryClipVideo() {{
      if (!secondaryClipVideo) return;
      try {{
        secondaryClipVideo.pause();
      }} catch (err) {{}}
    }}

    function setPlayButtonState(isPlaying) {{
      if (!playPauseClipButton) return;
      playPauseClipButton.textContent = isPlaying ? "Pause Both" : "Play Both";
    }}

    function applyClipSeekAndMaybePlay() {{
      seekClipVideo(pendingPrimarySeekS);
      seekSecondaryClipVideo(pendingSecondarySeekS);
      if (pendingClipAutoplay) {{
        playClipVideo();
        playSecondaryClipVideo();
      }}
      pendingClipAutoplay = false;
    }}

    function syncSecondaryToPrimary(playToo = false) {{
      if (!secondaryClipVideo || !secondaryClipVideo.src) return;
      const primaryOffset = Math.max(0, (clipVideo.currentTime || 0) - pendingPrimarySeekS);
      seekSecondaryClipVideo(pendingSecondarySeekS + primaryOffset);
      try {{
        secondaryClipVideo.playbackRate = clipVideo.playbackRate || 1;
      }} catch (err) {{}}
      if (playToo || !clipVideo.paused) {{
        playSecondaryClipVideo();
      }}
    }}

    async function toggleClipPlayback() {{
      if (!hasSecondaryView) return;
      if (clipVideo.paused) {{
        await playClipVideo();
        await playSecondaryClipVideo();
        setPlayButtonState(true);
      }} else {{
        clipVideo.pause();
        pauseSecondaryClipVideo();
        setPlayButtonState(false);
      }}
    }}

    function restartClipWindow() {{
      clipVideo.pause();
      pauseSecondaryClipVideo();
      seekClipVideo(pendingPrimarySeekS);
      seekSecondaryClipVideo(pendingSecondarySeekS);
      setPlayButtonState(false);
    }}

    function buildBars() {{
      const maxMinutes = Math.max(...bars.map(b => b.minutes), 0.001);
      bars.sort((a, b) => labelOrder.indexOf(a.label) - labelOrder.indexOf(b.label));
      for (const item of bars) {{
        const col = document.createElement("div");
        col.className = "bar-col";
        const value = document.createElement("div");
        value.className = "small mono";
        value.textContent = `${{item.minutes.toFixed(2)}} min`;
        const bar = document.createElement("div");
        bar.className = "bar";
        bar.style.background = item.color;
        bar.style.height = `${{Math.max(8, 220 * (item.minutes / maxMinutes))}}px`;
        const label = document.createElement("div");
        label.className = "bar-label";
        label.textContent = item.label;
        col.append(value, bar, label);
        barWrap.appendChild(col);
      }}
    }}

    function buildRows() {{
      for (const label of labelOrder) {{
        const rowSegs = segments.filter(seg => seg.label === label);
        if (!rowSegs.length) continue;
        const row = document.createElement("div");
        row.className = "timeline-row";
        const text = document.createElement("div");
        text.textContent = label;
        const track = document.createElement("div");
        track.className = "timeline-track";
        for (const seg of rowSegs) {{
          const el = document.createElement("div");
          el.className = "timeline-segment";
          el.style.left = `${{(seg.start_s / totalDuration) * 100}}%`;
          el.style.width = `${{(seg.duration_s / totalDuration) * 100}}%`;
          el.style.background = seg.color;
          el.title = `${{label}} • ${{fmtTime(seg.start_s)}} - ${{fmtTime(seg.end_s)}}`;
          el.addEventListener("click", (event) => {{
            const rect = el.getBoundingClientRect();
            const ratio = clamp((event.clientX - rect.left) / Math.max(rect.width, 1), 0, 1);
            const clickedTime = seg.start_s + ratio * seg.duration_s;
            selectTime(clickedTime, false);
          }});
          track.appendChild(el);
        }}
        row.append(text, track);
        timelineRows.appendChild(row);
      }}
    }}

    function buildEventTrack() {{
      for (const seg of segments) {{
        const el = document.createElement("div");
        el.className = "event-segment";
        el.dataset.start = seg.start_s;
        el.dataset.end = seg.end_s;
        el.style.left = `${{(seg.start_s / totalDuration) * 100}}%`;
        el.style.width = `${{(seg.duration_s / totalDuration) * 100}}%`;
        el.style.background = seg.color;
        el.title = `${{seg.label}} • ${{fmtTime(seg.start_s)}} - ${{fmtTime(seg.end_s)}}`;
        el.addEventListener("click", (event) => {{
          const rect = el.getBoundingClientRect();
          const ratio = clamp((event.clientX - rect.left) / Math.max(rect.width, 1), 0, 1);
          const clickedTime = seg.start_s + ratio * seg.duration_s;
          selectTime(clickedTime, false);
        }});
        eventTrack.appendChild(el);
      }}
    }}

    function getClipAtTime(t) {{
      return clips.find(c => t >= c.start_s && t < c.end_s) || clips[clips.length - 1];
    }}

    function buildTrialJump() {{
      if (!trialJumpSelect) return;
      for (const item of trialJumpOptions) {{
        const option = document.createElement("option");
        option.value = item.href;
        option.textContent = item.label;
        if (item.trial_id === "{escaped_trial_label}") {{
          option.selected = true;
        }}
        trialJumpSelect.appendChild(option);
      }}
      trialJumpSelect.addEventListener("change", (event) => {{
        const href = event.target.value;
        if (href) {{
          window.location.href = href;
        }}
      }});
    }}

    async function loadAnnotationState() {{
      try {{
        const response = await fetch(annotationApiUrl, {{ cache: "no-store" }});
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        const payload = await response.json();
        annotationState.apiAvailable = true;
        annotationState.annotations = payload.annotations || {{}};
        annotationState.summary = payload.summary || null;
        annotationServerNote.textContent = payload.summary && payload.summary.workbook_path
          ? `Saved labels update: ${{payload.summary.workbook_path}}`
          : "Manual labeling is connected.";
      }} catch (error) {{
        annotationState.apiAvailable = false;
        annotationState.annotations = {{}};
        annotationState.summary = null;
        annotationServerNote.textContent = "Manual labeling save is unavailable on this static page. Open the dashboard through the PD-Insighter annotation server.";
      }}
    }}

    async function saveManualAnnotationForCurrentClip(resetToAuto = false, options = {{}}) {{
      const clip = annotationEditorState.clipIdx != null
        ? clips.find(item => item.clip_idx === annotationEditorState.clipIdx)
        : getClipAtTime(currentSelectedTime);
      if (!annotationState.apiAvailable || !clip) {{
        annotationServerNote.textContent = "Manual labeling save is unavailable on this page.";
        return;
      }}
      clearAnnotationAutosaveTimer();
      const payload = resetToAuto
        ? {{ trial_id: trialId, clip_idx: clip.clip_idx, mode: "", actions: [] }}
        : (() => {{
            const collected = collectManualAnnotationInput();
            return {{ trial_id: trialId, clip_idx: clip.clip_idx, mode: collected.mode, actions: collected.actions }};
          }})();

      try {{
        annotationEditorState.saveInFlight = true;
        saveManualLabelButton.disabled = true;
        resetManualLabelButton.disabled = true;
        annotationServerNote.textContent = resetToAuto
          ? `Resetting clip #${{clip.clip_idx}} to auto...`
          : `Saving manual label for clip #${{clip.clip_idx}}...`;
        const response = await fetch(annotationSaveUrl, {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payload),
        }});
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        const updated = await response.json();
        annotationState.apiAvailable = true;
        annotationState.annotations = updated.annotations || {{}};
        annotationState.summary = updated.summary || null;
        annotationEditorState.clipIdx = clip.clip_idx;
        annotationEditorState.dirty = false;
        annotationServerNote.textContent = updated.summary && updated.summary.workbook_path
          ? `Saved to Excel: ${{updated.summary.workbook_path}}`
          : "Manual label saved.";
        updateSummary(currentSelectedTime);
        updateClipPanel(currentSelectedTime, false);
      }} catch (error) {{
        annotationServerNote.textContent = `Save failed: ${{error}}`;
      }} finally {{
        annotationEditorState.saveInFlight = false;
        saveManualLabelButton.disabled = false;
        resetManualLabelButton.disabled = false;
      }}
    }}

    function getSegmentAtTime(t) {{
      return segments.find(s => t >= s.start_s && t < s.end_s) || segments[segments.length - 1];
    }}

    function updateSummary(t) {{
      const clip = getClipAtTime(t);
      const segment = getSegmentAtTime(t);
      const sitting = bars.find(b => b.label === "sitting");
      const autoSpeechAction = clip.assumed_actions_in_clip || "Not detected";
      const autoSpeechWords = clip.spoken_words_detected_in_clip || "None in this clip";
      const effectiveManualLabel = getEffectiveLabelDisplay(clip) || "Not detected";
      const summary = annotationState.summary || {{}};
      summaryPanel.innerHTML = `
        <div class="summary-kpi"><b>Total Session</b><span class="mono">${{fmtTime(totalDuration)}}</span></div>
        <div class="summary-kpi"><b>Trial</b><span>{escaped_trial_label}</span></div>
        <div class="summary-kpi"><b>Recording</b><span>{escaped_recording_label}</span></div>
        <div class="summary-kpi"><b>Selected Window</b><span class="mono">${{fmtTime(clip.start_s)}} - ${{fmtTime(clip.end_s)}}</span></div>
        <div class="summary-kpi"><b>Criteria v3 Classification</b><span>${{escapeHtml(getModelPredictionLabel(clip) || "missing")}}</span></div>
        <div class="summary-kpi"><b>Original Rule Classification</b><span>${{escapeHtml(clip.old_rule_prediction || clip.rule_prediction || "missing")}}</span></div>
        <div class="summary-kpi"><b>Auto Spoken Action</b><span>${{escapeHtml(autoSpeechAction)}}</span></div>
        <div class="summary-kpi"><b>Effective Manual Label</b><span>${{escapeHtml(effectiveManualLabel)}}</span></div>
        <div class="summary-kpi"><b>Auto Spoken Words</b><span>${{escapeHtml(autoSpeechWords)}}</span></div>
        <div class="summary-kpi"><b>Current Event</b><span>${{escapeHtml(segment.label)}}</span></div>
        <div class="summary-kpi"><b>Transition-Inclusive Accuracy</b><span>${{fmtAccuracy(summary.transition_aware_accuracy)}}</span></div>
        <div class="summary-kpi"><b>Exact Label Accuracy</b><span>${{fmtAccuracy(summary.strict_accuracy)}}</span></div>
        <div class="summary-kpi"><b>Stable-Only Accuracy</b><span>${{fmtAccuracy(summary.stable_only_accuracy)}}</span></div>
        <div class="summary-kpi"><b>Manual Overrides</b><span>${{summary.manual_override_count ?? 0}}</span></div>
        <div class="summary-kpi"><b>% Time Sitting</b><span>${{sitting ? ((sitting.seconds / totalDuration) * 100).toFixed(0) : "0"}}%</span></div>
        <div class="summary-kpi"><b>Recording Start</b><span>${{fmtClock({recording_start_epoch_s:.6f})}}</span></div>
      `;
    }}

    function updateClipPanel(t, autoplay = false) {{
      const clip = getClipAtTime(t);
      const clipOffsetS = clamp(Number(t) - clip.start_s, 0, Math.max(0, clip.duration_s - 0.05));
      const autoSpeechWords = clip.spoken_words_detected_in_clip || "None";
      const autoSpeechAction = clip.assumed_actions_in_clip || "Not detected";
      const transitionSuffix = clip.state_change_times_in_clip_s ? " • " + clip.state_change_times_in_clip_s + "s" : "";
      const transitionText = clip.is_transition_clip ? "Yes" + transitionSuffix : "No";
      const effectiveManualLabel = getEffectiveLabelDisplay(clip) || "Not detected";
      const matchFlags = getModelMatchFlags(clip);
      const agreementText = matchFlags.transitionAware == null
        ? "N/A"
        : (matchFlags.transitionAware ? "Match" : "Mismatch");
      clipPanel.innerHTML = `
        <div class="metric-card"><b>Clip</b><span class="mono">#${{clip.clip_idx}}</span></div>
        <div class="metric-card"><b>Criteria v3 Classification</b><span>${{escapeHtml(getModelPredictionLabel(clip) || "missing")}}</span></div>
        <div class="metric-card"><b>Original Rule Classification</b><span>${{escapeHtml(clip.old_rule_prediction || clip.rule_prediction || "missing")}}</span></div>
        <div class="metric-card"><b>Window</b><span class="mono">${{fmtTime(clip.start_s)}} - ${{fmtTime(clip.end_s)}}</span></div>
        <div class="metric-card"><b>Offset</b><span class="mono">${{fmtTime(clipOffsetS)}}</span></div>
        <div class="metric-card"><b>Disp (H / V)</b><span>${{clip.horizontal_displacement_m.toFixed(3)}} m / ${{clip.vertical_displacement_m.toFixed(3)}} m</span></div>
        <div class="metric-card"><b>Motion</b><span>${{clip.mean_linear_speed_mps.toFixed(3)}} m/s · ${{clip.mean_angular_speed_rps.toFixed(3)}} rad/s · z ${{clip.mean_z_m.toFixed(3)}} m</span></div>
        <div class="metric-card"><b>Path / Height Slope</b><span>${{(clip.horizontal_path_length_m || 0).toFixed(3)}} m · ${{(clip.height_slope_mps || 0).toFixed(3)}} m/s</span></div>
        <div class="metric-card"><b>Auto Spoken Action</b><span>${{escapeHtml(autoSpeechAction)}}</span></div>
        <div class="metric-card"><b>Words Heard</b><span>${{escapeHtml(autoSpeechWords)}}</span></div>
        <div class="metric-card"><b>Effective Manual Label</b><span>${{escapeHtml(effectiveManualLabel)}}</span></div>
        <div class="metric-card"><b>Model vs Manual</b><span>${{escapeHtml(agreementText)}}</span></div>
        <div class="metric-card"><b>Word Times</b><span class="mono">${{escapeHtml(clip.spoken_word_times_s || "—")}}</span></div>
        <div class="metric-card"><b>Speech Transition</b><span>${{escapeHtml(transitionText)}}</span></div>
      `;
      renderAnnotationStatus(clip);
      syncAnnotationInputsToClip(clip);
      const usePrimaryFullSource = !!clip.full_video_uri;
      const useSecondaryFullSource = !!clip.full_secondary_video_uri;
      pendingPrimarySeekS = usePrimaryFullSource ? clip.source_start_s + clipOffsetS : clipOffsetS;
      pendingSecondarySeekS = useSecondaryFullSource ? clip.secondary_source_start_s + clipOffsetS : clipOffsetS;
      activePrimaryClipEndS = usePrimaryFullSource ? clip.source_end_s : clip.duration_s;
      activeSecondaryClipEndS = useSecondaryFullSource ? clip.secondary_source_end_s : clip.duration_s;
      pendingClipAutoplay = autoplay;
      const nextPrimarySrc = clip.full_video_uri || clip.clip_video_uri || "";
      if (activeClipIdx !== clip.clip_idx || activeClipVideoSrc !== nextPrimarySrc) {{
        activeClipVideoSrc = nextPrimarySrc;
        clipVideo.pause();
        clipVideo.src = activeClipVideoSrc;
        clipVideo.load();
      }} else {{
        applyClipSeekAndMaybePlay();
      }}
      if (secondaryClipCard && secondaryClipVideo) {{
        const nextSecondarySrc = clip.full_secondary_video_uri || clip.clip_secondary_video_uri || "";
        secondaryClipCard.style.display = nextSecondarySrc ? "grid" : "none";
        if (nextSecondarySrc) {{
          if (activeClipIdx !== clip.clip_idx || activeSecondaryClipVideoSrc !== nextSecondarySrc) {{
            activeSecondaryClipVideoSrc = nextSecondarySrc;
            pauseSecondaryClipVideo();
            secondaryClipVideo.src = nextSecondarySrc;
            secondaryClipVideo.load();
          }}
        }} else {{
          activeSecondaryClipVideoSrc = "";
          pauseSecondaryClipVideo();
          secondaryClipVideo.removeAttribute("src");
          secondaryClipVideo.load();
        }}
      }}
      clipVideo.style.transform = `rotate(${{primaryClipRotationDeg}}deg)`;
      if (secondaryClipVideo) {{
        secondaryClipVideo.style.transform = `rotate(${{secondaryClipRotationDeg}}deg)`;
      }}
      setPlayButtonState(false);
      activeClipIdx = clip.clip_idx;
    }}

    function highlightSegment(t) {{
      const active = getSegmentAtTime(t);
      document.querySelectorAll(".event-segment").forEach(el => {{
        const start = Number(el.dataset.start);
        const end = Number(el.dataset.end);
        el.classList.toggle("active", t >= start && t < end);
      }});
    }}

    function selectTime(t, autoplayClip = false) {{
      const time = Math.min(Math.max(0, Number(t) || 0), totalDuration);
      currentSelectedTime = time;
      scrubber.value = time;
      timeReadout.textContent = `Selected time: ${{fmtTime(time)}} / ${{fmtTime(totalDuration)}}`;
      updateSummary(time);
      updateClipPanel(time, autoplayClip);
      highlightSegment(time);
    }}

    scrubber.addEventListener("input", (e) => selectTime(e.target.value, false));
    if (playPauseClipButton) {{
      playPauseClipButton.addEventListener("click", () => {{
        toggleClipPlayback();
      }});
    }}
    if (restartClipButton) {{
      restartClipButton.addEventListener("click", () => {{
        restartClipWindow();
      }});
    }}
    rotatePrimaryClipButton.addEventListener("click", () => {{
      primaryClipRotationDeg = (primaryClipRotationDeg + 90) % 360;
      clipVideo.style.transform = `rotate(${{primaryClipRotationDeg}}deg)`;
    }});
    if (rotateSecondaryClipButton) {{
      rotateSecondaryClipButton.addEventListener("click", () => {{
        secondaryClipRotationDeg = (secondaryClipRotationDeg + 90) % 360;
        secondaryClipVideo.style.transform = `rotate(${{secondaryClipRotationDeg}}deg)`;
      }});
    }}
    clipVideo.addEventListener("loadedmetadata", () => {{
      applyClipSeekAndMaybePlay();
    }});
    clipVideo.addEventListener("canplay", () => {{
      if (pendingClipAutoplay) {{
        applyClipSeekAndMaybePlay();
      }}
    }});
    clipVideo.addEventListener("play", () => {{
      syncSecondaryToPrimary(true);
      setPlayButtonState(true);
    }});
    clipVideo.addEventListener("pause", () => {{
      pauseSecondaryClipVideo();
      setPlayButtonState(false);
    }});
    clipVideo.addEventListener("ended", () => {{
      pauseSecondaryClipVideo();
      setPlayButtonState(false);
    }});
    clipVideo.addEventListener("seeking", () => {{
      syncSecondaryToPrimary(false);
    }});
    clipVideo.addEventListener("timeupdate", () => {{
      if (Number.isFinite(activePrimaryClipEndS) && clipVideo.currentTime >= Math.max(0, activePrimaryClipEndS - 0.02)) {{
        clipVideo.pause();
        seekClipVideo(pendingPrimarySeekS);
        pauseSecondaryClipVideo();
        seekSecondaryClipVideo(pendingSecondarySeekS);
        setPlayButtonState(false);
        return;
      }}
      if (secondaryClipVideo && secondaryClipVideo.src) {{
        const expected = pendingSecondarySeekS + Math.max(0, (clipVideo.currentTime || 0) - pendingPrimarySeekS);
        if (!Number.isFinite(secondaryClipVideo.currentTime) || Math.abs((secondaryClipVideo.currentTime || 0) - expected) > 0.08) {{
          seekSecondaryClipVideo(expected);
        }}
      }}
    }});
    clipVideo.addEventListener("ratechange", () => {{
      if (secondaryClipVideo) {{
        try {{
          secondaryClipVideo.playbackRate = clipVideo.playbackRate || 1;
        }} catch (err) {{}}
      }}
    }});
    if (secondaryClipVideo) {{
      secondaryClipVideo.addEventListener("loadedmetadata", () => {{
        seekSecondaryClipVideo(pendingSecondarySeekS || 0);
      }});
      secondaryClipVideo.addEventListener("canplay", () => {{
        if (!clipVideo.paused) {{
          syncSecondaryToPrimary(true);
        }}
      }});
      secondaryClipVideo.addEventListener("timeupdate", () => {{
        if (Number.isFinite(activeSecondaryClipEndS) && secondaryClipVideo.currentTime >= Math.max(0, activeSecondaryClipEndS - 0.02)) {{
          pauseSecondaryClipVideo();
          seekSecondaryClipVideo(pendingSecondarySeekS);
        }}
      }});
    }}
    singleActionSelect.innerHTML = buildActionOptionsHtml(actionOptions[0]);
    transitionCountSelect.addEventListener("change", () => updateAnnotationVisibility());
    manualModeSelect.addEventListener("change", () => {{
      updateAnnotationVisibility();
      handleAnnotationInputChange();
    }});
    transitionCountSelect.addEventListener("change", () => {{
      updateAnnotationVisibility();
      handleAnnotationInputChange();
    }});
    singleActionSelect.addEventListener("change", handleAnnotationInputChange);
    transitionActionsWrap.addEventListener("change", handleAnnotationInputChange);
    saveManualLabelButton.addEventListener("click", () => {{
      saveManualAnnotationForCurrentClip(false);
    }});
    resetManualLabelButton.addEventListener("click", () => {{
      saveManualAnnotationForCurrentClip(true);
    }});
    (async () => {{
      await loadAnnotationState();
      buildTrialJump();
      buildBars();
      buildRows();
      buildEventTrack();
      selectTime(0, false);
    }})();
  </script>
</body>
</html>
"""
    output_path.write_text(html_text)


def plot_point_cloud(points_csv: Path, trajectory_csv: Path, output_path: Path, point_sample_size: int) -> None:
    points_df = pd.read_csv(points_csv, compression="gzip")
    traj_df = pd.read_csv(trajectory_csv)

    if len(points_df) > point_sample_size:
        points_df = points_df.sample(point_sample_size, random_state=0)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points_df["px_world"],
        points_df["py_world"],
        points_df["pz_world"],
        s=0.5,
        alpha=0.2,
        c=points_df["pz_world"],
        cmap="viridis",
    )
    ax.plot(
        traj_df["tx_world_device"],
        traj_df["ty_world_device"],
        traj_df["tz_world_device"],
        color="crimson",
        linewidth=1.0,
    )
    ax.set_title("Semi-dense Point Cloud with Closed-loop Trajectory")
    ax.set_xlabel("X world (m)")
    ax.set_ylabel("Y world (m)")
    ax.set_zlabel("Z world (m)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_displacement_graphs(clip_df: pd.DataFrame, output_dir: Path) -> None:
    valid_df = clip_df[clip_df["rule_prediction"] != "missing"].copy()
    if valid_df.empty:
        return

    color_map = {
        "sitting": "#377eb8",
        "standing": "#984ea3",
        "walking": "#4daf4a",
        "go upstairs": "#ff7f00",
        "go downstairs": "#e41a1c",
    }
    colors = [color_map.get(label, "#777777") for label in valid_df["hybrid_prediction"].fillna(valid_df["rule_prediction"])]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(valid_df["clip_idx"], valid_df["horizontal_displacement_m"], color=colors)
    ax.set_title("Per-clip Horizontal Displacement")
    ax.set_xlabel("Clip index")
    ax.set_ylabel("Horizontal displacement (m)")
    fig.tight_layout()
    fig.savefig(output_dir / "horizontal_displacement_by_clip.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(valid_df["clip_idx"], valid_df["vertical_displacement_m"], color=colors)
    ax.set_title("Per-clip Vertical Displacement")
    ax.set_xlabel("Clip index")
    ax.set_ylabel("Vertical displacement (m)")
    fig.tight_layout()
    fig.savefig(output_dir / "vertical_displacement_by_clip.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        valid_df["horizontal_displacement_m"],
        valid_df["vertical_displacement_m"],
        c=colors,
        s=40,
        alpha=0.8,
    )
    ax.set_title("Horizontal vs Vertical Displacement")
    ax.set_xlabel("Horizontal displacement (m)")
    ax.set_ylabel("Vertical displacement (m)")
    fig.tight_layout()
    fig.savefig(output_dir / "horizontal_vs_vertical_scatter.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        valid_df["horizontal_displacement_m"],
        valid_df["mean_linear_speed_mps"],
        c=colors,
        s=40,
        alpha=0.8,
    )
    ax.set_title("Horizontal Displacement vs Mean Linear Speed")
    ax.set_xlabel("Horizontal displacement (m)")
    ax.set_ylabel("Mean linear speed (m/s)")
    fig.tight_layout()
    fig.savefig(output_dir / "horizontal_vs_speed_scatter.png", dpi=180)
    plt.close(fig)


def write_summary(clip_df: pd.DataFrame, output_path: Path, alignment_offset_s: float) -> None:
    summary = {
        "clip_count": int(len(clip_df)),
        "valid_slam_clip_count": int((clip_df["rule_prediction"] != "missing").sum()),
        "alignment_offset_s": alignment_offset_s,
        "rule_label_counts": clip_df["rule_prediction"].value_counts(dropna=False).to_dict(),
        "hybrid_label_counts": clip_df["hybrid_prediction"].value_counts(dropna=False).to_dict(),
        "mean_speed_mps_range": [
            float(clip_df["mean_linear_speed_mps"].min()),
            float(clip_df["mean_linear_speed_mps"].max()),
        ],
    }
    output_path.write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    video_path = Path(args.video_path).resolve()
    vrs_json_path = Path(args.vrs_json_path).resolve()
    trajectory_csv = Path(args.trajectory_csv).resolve()
    points_csv = Path(args.points_csv).resolve()
    eye_gaze_csv = Path(args.eye_gaze_csv).resolve() if args.eye_gaze_csv else None
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    clips_dir = ensure_dir(output_dir / "clips")
    plots_dir = ensure_dir(output_dir / "plots")
    gif_dir = ensure_dir(output_dir / "clip_gifs")
    clip_audio_dir = ensure_dir(output_dir / "clip_audio")
    clip_media_dir = ensure_dir(output_dir / "clip_media")
    audio_path, audio_offset_s = discover_audio_companion(video_path)

    windows = split_video_into_clips(video_path, clips_dir, args.clip_seconds)
    for window in windows:
        create_clip_gif(window.clip_path, gif_dir / f"clip_{window.clip_idx:04d}.gif")
    start_epoch_s = load_start_time(vrs_json_path)
    trajectory_df = pd.read_csv(trajectory_csv)
    alignment_offset_s = estimate_alignment_offset_s(trajectory_df, eye_gaze_csv)
    clip_df = build_clip_metrics(windows, start_epoch_s, trajectory_df, alignment_offset_s)
    clip_df = apply_hybrid_predictions(
        clip_df,
        Path(args.qwen_predictions_csv).resolve() if args.qwen_predictions_csv else None,
        args.hybrid_override_labels,
    )
    clip_audio_map = create_clip_audio_files(clip_df, clip_audio_dir, audio_path, audio_offset_s)
    clip_media_map = create_clip_media_files(clip_df, clip_media_dir, clip_audio_map)

    clip_df.to_csv(output_dir / "clip_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    write_qwen_manifest(clip_df, output_dir / "qwen_manifest.json")
    write_timeline_json(clip_df, output_dir / "timeline_predictions.json")
    write_summary(clip_df, output_dir / "summary.json", alignment_offset_s)
    rule_segments_df = summarize_segments(clip_df, "rule_prediction")
    rule_segments_df.to_csv(output_dir / "rule_segments.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    plot_point_cloud(points_csv, trajectory_csv, plots_dir / "point_cloud_3d.png", args.point_sample_size)
    plot_displacement_graphs(clip_df, plots_dir)
    plot_pdinsighter_time_spent(clip_df, rule_segments_df, plots_dir / "pd_insighter_time_spent_rule.png", "rule_prediction")
    plot_pdinsighter_event_view(clip_df, rule_segments_df, clips_dir, plots_dir / "pd_insighter_event_view_rule.png", "rule_prediction")
    write_interactive_dashboard(
        clip_df,
        rule_segments_df,
        output_dir / "pd_insighter_rule_interactive.html",
        gif_dir,
        trial_id=args.trial_id,
        recording_stem=args.recording_stem,
        video_path=video_path,
        audio_path=audio_path,
        audio_offset_s=audio_offset_s,
        clip_audio_map=clip_audio_map,
        clip_media_map=clip_media_map,
    )

    print(f"Wrote clip metrics to: {output_dir / 'clip_metrics.csv'}")
    print(f"Wrote Qwen manifest to: {output_dir / 'qwen_manifest.json'}")
    print(f"Wrote timeline JSON to: {output_dir / 'timeline_predictions.json'}")
    print(f"Wrote plots to: {plots_dir}")


if __name__ == "__main__":
    main()
