#!/usr/bin/env python3
"""Build recording-time and cross-device clock checks for PD-Insighter trials."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import wave
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np


QWEN_REPO = Path(__file__).resolve().parents[1]
GEN1_ROOT = QWEN_REPO.parent
RUNS_DIR = QWEN_REPO / "runs"
DOWNWARD_ROOT = GEN1_ROOT / "downward videos"
LOCAL_TIMEZONE = "America/New_York"
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", nargs="+", required=True)
    parser.add_argument("--refresh", action="store_true", help="Recalculate cached drift estimates.")
    parser.add_argument(
        "--update-dashboard",
        action="store_true",
        help="Add the results to each existing session manifest and regenerate index.html.",
    )
    return parser.parse_args()


def discover_ffmpeg_executable() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _safe_stat_mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _source_signature(trial_id: str, manifest: dict[str, Any], sync_data: dict[str, Any]) -> dict[str, Any]:
    vrs_json_path = Path(str(manifest["raw_data"]["vrs_json"]["path"]))
    downward_path = Path(str(sync_data.get("downward_video") or ""))
    sync_path = RUNS_DIR / trial_id / "downward_sync" / "sync_suggestion.json"
    vrs_metadata = json.loads(vrs_json_path.read_text())
    return {
        "vrs_start_time": float(vrs_metadata.get("start_time") or 0.0),
        "audio_offset_s": float(sync_data.get("estimated_offset_s") or 0.0),
        "shared_duration_s": float(sync_data.get("shared_duration_s") or 0.0),
        "downward_video": str(downward_path),
        "downward_mtime_ns": _safe_stat_mtime_ns(downward_path),
        "sync_mtime_ns": _safe_stat_mtime_ns(sync_path),
        "downward_creation_timezone": manifest.get("downward_media", {}).get("creation_time_timezone"),
    }


def _probe_creation_time(video_path: Path, ffmpeg_exe: str) -> tuple[str | None, datetime | None]:
    if not video_path.exists():
        return None, None
    result = subprocess.run(
        [ffmpeg_exe, "-hide_banner", "-i", str(video_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(
        r"creation_time\s*:\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)Z",
        result.stderr,
    )
    if not match:
        return None, None
    raw_value = f"{match.group(1)}Z"
    return raw_value, datetime.fromisoformat(match.group(1))


def _creation_time_in_local_zone(raw_value: str, manifest: dict[str, Any]) -> datetime:
    # Older camera files used a false Z suffix; only convert explicitly verified UTC exports.
    value = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    if manifest.get("downward_media", {}).get("creation_time_timezone") == "UTC":
        return value.astimezone(ZoneInfo(LOCAL_TIMEZONE)).replace(tzinfo=None)
    return value.replace(tzinfo=None)


def _filename_timestamp_candidates() -> list[tuple[datetime, Path]]:
    candidates: list[tuple[datetime, Path]] = []
    if not DOWNWARD_ROOT.exists():
        return candidates
    pattern = re.compile(r"(20\d{6})_(\d{6})")
    for path in DOWNWARD_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".mov"}:
            continue
        for date_part, time_part in pattern.findall(path.name):
            try:
                candidates.append((datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S"), path))
            except ValueError:
                continue
    return candidates


def _nearest_filename_timestamp(expected_start: datetime) -> tuple[datetime | None, Path | None]:
    candidates = _filename_timestamp_candidates()
    if not candidates:
        return None, None
    timestamp, path = min(candidates, key=lambda item: abs((item[0] - expected_start).total_seconds()))
    if abs((timestamp - expected_start).total_seconds()) > 300.0:
        return None, None
    return timestamp, path


def _format_date(value: datetime) -> str:
    return value.strftime("%B %d, %Y").replace(" 0", " ")


def _format_time(value: datetime, timezone_label: str = "") -> str:
    text = value.strftime("%I:%M:%S %p").lstrip("0")
    return f"{text} {timezone_label}".strip()


def _format_minute(value: datetime, timezone_label: str = "") -> str:
    text = value.strftime("%I:%M %p").lstrip("0")
    return f"{text} {timezone_label}".strip()


def _read_wav_envelope(path: Path, frame_ms: float = 20.0) -> tuple[np.ndarray, float, float]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        total_frames = wav_file.getnframes()
        hop = max(1, int(round(sample_rate * frame_ms / 1000.0)))
        chunk_frames = hop * 5000
        rms_parts: list[np.ndarray] = []
        carry = np.array([], dtype=np.float32)
        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        dtype = dtype_map.get(sample_width)
        if dtype is None:
            raise RuntimeError(f"Unsupported WAV sample width: {sample_width}")

        while True:
            raw = wav_file.readframes(chunk_frames)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
            if channels > 1:
                samples = samples.reshape(-1, channels).mean(axis=1)
            if carry.size:
                samples = np.concatenate([carry, samples])
            usable = (len(samples) // hop) * hop
            if usable:
                frames = samples[:usable].reshape(-1, hop)
                rms_parts.append(np.sqrt(np.mean(np.square(frames), axis=1) + 1e-8))
            carry = samples[usable:]

    rms = np.concatenate(rms_parts) if rms_parts else np.array([], dtype=np.float32)
    if not len(rms):
        return rms, hop / sample_rate, total_frames / sample_rate
    smooth = np.convolve(rms, np.ones(5) / 5.0, mode="same")
    emphasis = np.maximum(0.0, smooth - np.convolve(smooth, np.ones(25) / 25.0, mode="same"))
    centered = emphasis - np.mean(emphasis)
    envelope = centered / (np.std(centered) or 1.0)
    return envelope.astype(np.float32), hop / sample_rate, total_frames / sample_rate


def _normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 10 or len(right) != len(left):
        return float("nan")
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denom = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(left_centered, right_centered) / denom)


def _estimate_local_audio_offset(
    aria_env: np.ndarray,
    downward_env: np.ndarray,
    frame_s: float,
    downward_window_start_s: float,
    window_duration_s: float,
    global_offset_s: float,
    search_radius_s: float,
) -> tuple[float | None, float | None, bool]:
    downward_start = int(round(downward_window_start_s / frame_s))
    frame_count = int(round(window_duration_s / frame_s))
    if downward_start < 0 or downward_start + frame_count > len(downward_env):
        return None, None, False
    downward_part = downward_env[downward_start : downward_start + frame_count]
    radius_frames = max(1, int(round(search_radius_s / frame_s)))
    global_frames = int(round(global_offset_s / frame_s))
    best_score = -math.inf
    best_delta_frames: int | None = None
    for delta_frames in range(-radius_frames, radius_frames + 1):
        aria_start = downward_start + global_frames + delta_frames
        if aria_start < 0 or aria_start + frame_count > len(aria_env):
            continue
        score = _normalized_correlation(
            aria_env[aria_start : aria_start + frame_count],
            downward_part,
        )
        if math.isfinite(score) and score > best_score:
            best_score = score
            best_delta_frames = delta_frames
    if best_delta_frames is None:
        return None, None, False
    at_boundary = abs(best_delta_frames) >= radius_frames
    return global_offset_s + best_delta_frames * frame_s, best_score, at_boundary


def estimate_audio_drift(trial_id: str, sync_data: dict[str, Any]) -> dict[str, Any]:
    shared_duration_s = float(sync_data.get("shared_duration_s") or 0.0)
    if shared_duration_s < 300.0:
        return {
            "status": "not_measured_short_session",
            "minimum_duration_s": 300.0,
            "shared_duration_s": round(shared_duration_s, 3),
            "summary": "Not measured because the synchronized overlap is shorter than 5 minutes.",
        }

    sync_dir = RUNS_DIR / trial_id / "downward_sync"
    aria_wav = sync_dir / "aria_sync_audio.wav"
    downward_wav = sync_dir / "downward_sync_audio.wav"
    if not aria_wav.exists() or not downward_wav.exists():
        return {
            "status": "unavailable",
            "summary": "Drift could not be measured because one or both synchronized WAV files are missing.",
        }

    aria_env, aria_frame_s, aria_duration_s = _read_wav_envelope(aria_wav)
    downward_env, downward_frame_s, downward_duration_s = _read_wav_envelope(downward_wav)
    if not math.isclose(aria_frame_s, downward_frame_s, rel_tol=1e-6, abs_tol=1e-9):
        return {"status": "unavailable", "summary": "The two audio envelopes use different time steps."}

    frame_s = aria_frame_s
    global_offset_s = float(sync_data.get("estimated_offset_s") or 0.0)
    downward_trim_start_s = float(sync_data.get("downward_trim_start_s") or max(0.0, -global_offset_s))
    if shared_duration_s >= 1800:
        window_s, sample_count, search_radius_s = 240.0, 7, 4.0
    elif shared_duration_s >= 900:
        window_s, sample_count, search_radius_s = 180.0, 6, 3.0
    else:
        window_s, sample_count, search_radius_s = 90.0, 5, 2.0
    centers = np.linspace(window_s / 2.0, shared_duration_s - window_s / 2.0, sample_count)
    local_samples: list[dict[str, Any]] = []
    for center_s in centers:
        session_window_start_s = float(center_s - window_s / 2.0)
        offset_s, correlation, at_boundary = _estimate_local_audio_offset(
            aria_env,
            downward_env,
            frame_s,
            downward_trim_start_s + session_window_start_s,
            window_s,
            global_offset_s,
            search_radius_s,
        )
        if offset_s is None or correlation is None:
            continue
        local_samples.append(
            {
                "session_center_s": round(float(center_s), 3),
                "offset_s": round(float(offset_s), 4),
                "correlation": round(float(correlation), 4),
                "search_boundary": bool(at_boundary),
            }
        )

    if len(local_samples) < 3:
        return {
            "status": "unavailable",
            "summary": "Too few reliable audio windows were available to estimate drift.",
        }

    times = np.array([item["session_center_s"] for item in local_samples], dtype=float)
    offsets = np.array([item["offset_s"] for item in local_samples], dtype=float)
    correlations = np.array([item["correlation"] for item in local_samples], dtype=float)
    weights = np.clip(correlations, 0.05, 1.0)
    slope, intercept = np.polyfit(times, offsets, 1, w=weights)
    fitted = intercept + slope * times
    fit_rmse_s = float(np.sqrt(np.mean(np.square(offsets - fitted))))
    drift_s = float(slope * shared_duration_s)
    early_offset_s = float(intercept)
    late_offset_s = float(intercept + slope * shared_duration_s)
    median_correlation = float(np.median(correlations))
    boundary_hit = any(bool(item["search_boundary"]) for item in local_samples)
    if median_correlation >= 0.55 and fit_rmse_s <= 0.08 and not boundary_hit:
        confidence = "high"
    elif median_correlation >= 0.25 and fit_rmse_s <= 0.25 and not boundary_hit:
        confidence = "medium"
    else:
        confidence = "low"

    direction = "Aria becomes later relative to downward" if drift_s > 0 else "Aria becomes earlier relative to downward"
    if abs(drift_s) < 0.05:
        direction = "No meaningful drift detected"
    return {
        "status": "measured",
        "method": "Local audio-envelope cross-correlation across the synchronized session",
        "shared_duration_s": round(shared_duration_s, 3),
        "audio_duration_s": {
            "aria": round(aria_duration_s, 3),
            "downward": round(downward_duration_s, 3),
        },
        "window_duration_s": window_s,
        "sample_count": len(local_samples),
        "early_offset_s": round(early_offset_s, 4),
        "late_offset_s": round(late_offset_s, 4),
        "drift_s": round(drift_s, 4),
        "drift_ms_per_min": round(slope * 60.0 * 1000.0, 3),
        "clock_rate_difference_ppm": round(slope * 1_000_000.0, 2),
        "median_correlation": round(median_correlation, 4),
        "fit_rmse_s": round(fit_rmse_s, 4),
        "confidence": confidence,
        "direction": direction,
        "local_offsets": local_samples,
        "summary": f"{drift_s:+.2f} s over {shared_duration_s / 60.0:.1f} min ({confidence} confidence).",
    }


def build_time_information(
    trial_id: str,
    manifest: dict[str, Any],
    sync_data: dict[str, Any],
    *,
    ffmpeg_exe: str | None = None,
) -> dict[str, Any]:
    ffmpeg_exe = ffmpeg_exe or discover_ffmpeg_executable()
    timezone = ZoneInfo(LOCAL_TIMEZONE)
    vrs_json_path = Path(str(manifest["raw_data"]["vrs_json"]["path"]))
    vrs_metadata = json.loads(vrs_json_path.read_text())
    aria_epoch_s = float(vrs_metadata["start_time"])
    aria_start = datetime.fromtimestamp(aria_epoch_s, timezone)
    aria_naive = aria_start.replace(tzinfo=None)
    timezone_label = aria_start.tzname() or LOCAL_TIMEZONE
    audio_offset_s = float(sync_data.get("estimated_offset_s") or 0.0)
    expected_downward_start = aria_naive + timedelta(seconds=audio_offset_s)

    downward_video = Path(str(sync_data.get("downward_video") or ""))
    raw_creation_time, downward_reported_start = _probe_creation_time(downward_video, ffmpeg_exe)
    if raw_creation_time:
        downward_reported_start = _creation_time_in_local_zone(raw_creation_time, manifest)
    downward_source = "embedded_mp4_creation_time" if downward_reported_start else None
    downward_source_path = downward_video if downward_reported_start else None
    if downward_reported_start is None:
        filename_start, filename_path = _nearest_filename_timestamp(expected_downward_start)
        if filename_start is not None and filename_path is not None:
            downward_reported_start = filename_start
            downward_source = "original_chunk_filename"
            downward_source_path = filename_path

    metadata_offset_s: float | None = None
    clock_agreement_error_s: float | None = None
    if downward_reported_start is not None:
        metadata_offset_s = (downward_reported_start - aria_naive).total_seconds()
        clock_agreement_error_s = metadata_offset_s - audio_offset_s
        absolute_error = abs(clock_agreement_error_s)
        if absolute_error <= 1.5:
            agreement_status = "consistent"
            agreement_label = "Clock values agree with audio"
        elif absolute_error <= 3.0:
            agreement_status = "close"
            agreement_label = "Clock values are close to audio"
        else:
            agreement_status = "review"
            agreement_label = "Clock values need review"
        downward_display_start = downward_reported_start
    else:
        agreement_status = "not_independently_verifiable"
        agreement_label = "Downward start derived from audio"
        downward_display_start = expected_downward_start

    top_candidate = (sync_data.get("top_candidates") or [{}])[0]
    peak_f1 = top_candidate.get("peak_f1_score")
    drift = estimate_audio_drift(trial_id, sync_data)
    audio_offset_direction = (
        f"Aria started {abs(audio_offset_s):.2f} s before downward"
        if audio_offset_s >= 0
        else f"Downward started {abs(audio_offset_s):.2f} s before Aria"
    )
    if clock_agreement_error_s is None:
        agreement_detail = "No independent downward clock timestamp survived video combination; start time is audio-derived."
    else:
        agreement_detail = (
            f"The clock-derived start difference and audio-derived start difference differ by "
            f"{abs(clock_agreement_error_s):.2f} s."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "trial_id": trial_id,
        "timezone": LOCAL_TIMEZONE,
        "recording_date": _format_date(aria_start),
        "recording_start_minute": f"{_format_date(aria_start)}, {_format_minute(aria_start, timezone_label)}",
        "aria": {
            "start_epoch_s": aria_epoch_s,
            "start_iso": aria_start.isoformat(),
            "start_display": f"{_format_date(aria_start)}, {_format_time(aria_start, timezone_label)}",
            "start_minute_display": f"{_format_date(aria_start)}, {_format_minute(aria_start, timezone_label)}",
            "source": "Aria VRS JSON start_time",
            "source_file": str(vrs_json_path),
            "ntp_enabled": bool(vrs_metadata.get("ntp_time_enabled")),
            "timecode_enabled": bool(vrs_metadata.get("timecode_enabled")),
            "external_clock_verified": False,
        },
        "downward": {
            "start_display": f"{_format_date(downward_display_start)}, {_format_time(downward_display_start, timezone_label)}",
            "start_minute_display": f"{_format_date(downward_display_start)}, {_format_minute(downward_display_start, timezone_label)}",
            "start_source": downward_source or "audio_alignment_derived",
            "source_file": str(downward_source_path) if downward_source_path else str(downward_video),
            "reported_creation_time_raw": raw_creation_time,
            "is_independent_clock_value": downward_reported_start is not None,
            "timezone_note": (
                ("The verified UTC creation time is converted to America/New_York; it agrees with the original filename."
                 if manifest.get("downward_media", {}).get("creation_time_timezone") == "UTC"
                 else "The recorder writes local wall-clock numbers with a Z suffix. They are compared as local time, not converted from UTC.")
                if raw_creation_time
                else "No embedded creation time is present in the combined MP4."
            ),
        },
        "alignment": {
            "audio_offset_s": round(audio_offset_s, 4),
            "metadata_offset_s": round(metadata_offset_s, 4) if metadata_offset_s is not None else None,
            "clock_agreement_error_s": (
                round(clock_agreement_error_s, 4) if clock_agreement_error_s is not None else None
            ),
            "status": agreement_status,
            "label": agreement_label,
            "start_order": audio_offset_direction,
            "detail": agreement_detail,
            "method": "Audio-envelope and salient-peak matching",
            "peak_f1_score": round(float(peak_f1), 4) if peak_f1 is not None else None,
            "shared_duration_s": round(float(sync_data.get("shared_duration_s") or 0.0), 3),
        },
        "drift": drift,
        "verification_note": (
            "This verifies agreement between the two recording clocks and their shared audio. It does not verify either device "
            "against an external reference clock because Aria NTP/timecode was disabled."
        ),
        "source_signature": _source_signature(trial_id, manifest, sync_data),
    }


def load_or_build_time_information(
    trial_id: str,
    manifest: dict[str, Any],
    sync_data: dict[str, Any],
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    output_path = RUNS_DIR / trial_id / "time_information.json"
    signature = _source_signature(trial_id, manifest, sync_data)
    if output_path.exists() and not refresh:
        cached = json.loads(output_path.read_text())
        if cached.get("schema_version") == SCHEMA_VERSION and cached.get("source_signature") == signature:
            return cached
    result = build_time_information(trial_id, manifest, sync_data)
    output_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    ffmpeg_exe = discover_ffmpeg_executable()
    for trial_id in args.trials:
        run_dir = RUNS_DIR / trial_id
        manifest_path = run_dir / "trial_manifest.json"
        sync_path = run_dir / "downward_sync" / "sync_suggestion.json"
        if not manifest_path.exists() or not sync_path.exists():
            print(f"Skipped {trial_id}: manifest or sync suggestion missing.")
            continue
        manifest = json.loads(manifest_path.read_text())
        sync_data = json.loads(sync_path.read_text())
        result = load_or_build_time_information(trial_id, manifest, sync_data, refresh=args.refresh)
        if args.update_dashboard:
            dashboard_manifest_path = run_dir / "session_dashboard_manifest.json"
            if dashboard_manifest_path.exists():
                dashboard_payload = json.loads(dashboard_manifest_path.read_text())
                dashboard_payload["time_information"] = result
                dashboard_manifest_path.write_text(json.dumps(dashboard_payload, indent=2))
                from build_downward_dual_view_dashboards import build_dashboard_html

                (run_dir / "index.html").write_text(build_dashboard_html(dashboard_payload))
        print(f"{trial_id}: {result['recording_start_minute']} | {result['alignment']['label']}")


if __name__ == "__main__":
    main()
