#!/usr/bin/env python3
"""Signals for the automatic symptom detector, all on the dashboard session clock.

Everything is read from a trial's saved outputs under ``runs/<trial>/``; nothing there is
modified. The session clock is the one used by Full Session, Manual Labels and Symptom
Annotation: seconds from the first frame of the session videos.

Sources, in order of how directly they measure movement:

``head_track`` (measured)
    Aria head position from MPS SLAM. Preferred: the per-video-frame camera poses saved for
    the Environment view (``environment_reconstruction/scene_playback_data.json``, 30 Hz,
    position + orientation + per-frame tracking status). Fallback: the ~15 Hz x/y/z samples in
    ``session_dashboard_manifest.json`` (no orientation).
``downward_video`` (measured image motion, indirect for the body)
    Dense optical flow of the downward session video at its native 15 fps, reduced to a few
    numbers per frame. Only inside the recorded window from ``downward_attachment``; padding
    and dark frames are masked.
``audio`` (supporting only)
    A broadband onset envelope of the Aria audio. No speech recognition, no content analysis.
``activity`` / ``step_events`` (existing estimates)
    The continuous activity classification and the SLAM head-sway step proxies already saved
    by the research pipeline. These are inferred, not measured.

All features are resampled to one grid (``GRID_HZ``). Samples without valid data are NaN and
``*_ok`` masks say where a source can be trusted.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import numpy as np
from scipy import signal as sps
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.spatial.transform import Rotation

GRID_HZ = 20.0
STEP_BAND_HZ = (0.6, 3.2)       # head bob / leg movement of stepping, from slow shuffles to brisk walking
LOCOMOTION_LABELS = ("walking", "upstairs", "downstairs")
ACTIVITY_CODES = {"unknown": 0, "sitting": 1, "standing": 2, "walking": 3, "upstairs": 4, "downstairs": 5}
VIDEO_FLOW_SIZE = (160, 90)     # optical flow is computed on a small greyscale copy
VIDEO_NATIVE_FPS = 15.0
DARK_FRAME_LUMA = 12.0          # mean luma below this is padding or a covered lens
AUDIO_ENVELOPE_HZ = 100.0
MAX_POSE_GAP_S = 0.5            # longer gaps in tracking are not interpolated


@dataclass
class TrialSignals:
    trial_id: str
    duration_s: float
    t: np.ndarray                                   # grid times (s), session clock
    features: dict[str, np.ndarray] = field(default_factory=dict)
    activity: np.ndarray | None = None              # ACTIVITY_CODES per grid sample
    activity_segments: list[dict[str, Any]] = field(default_factory=list)
    step_event_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    downward_coverage_s: tuple[float, float] | None = None
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def rate_hz(self) -> float:
        return GRID_HZ

    def index(self, seconds: float) -> int:
        return int(np.clip(round(seconds * GRID_HZ), 0, len(self.t) - 1))


# ----------------------------------------------------------------------------------------------
# small numeric helpers


def grid(duration_s: float) -> np.ndarray:
    return np.arange(0.0, max(duration_s, 1.0 / GRID_HZ), 1.0 / GRID_HZ)


def nan_interp(target_t: np.ndarray, t: np.ndarray, values: np.ndarray, max_gap_s: float) -> np.ndarray:
    """Linear interpolation that leaves NaN where the surrounding samples are too far apart."""
    good = np.isfinite(values)
    out = np.full(len(target_t), np.nan)
    if good.sum() < 2:
        return out
    tg, vg = t[good], values[good]
    out = np.interp(target_t, tg, vg, left=np.nan, right=np.nan)
    right = np.clip(np.searchsorted(tg, target_t), 1, len(tg) - 1)
    gap = tg[right] - tg[right - 1]
    out[gap > max_gap_s] = np.nan
    return out


def rolling_mean(values: np.ndarray, seconds: float, rate_hz: float = GRID_HZ) -> np.ndarray:
    """Centered moving average that ignores NaN (NaN only where the whole window is NaN)."""
    size = max(1, int(round(seconds * rate_hz)) | 1)
    good = np.isfinite(values)
    total = uniform_filter1d(np.where(good, values, 0.0), size, mode="nearest")
    count = uniform_filter1d(good.astype(float), size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 1e-9, total / count, np.nan)


def rolling_rms(values: np.ndarray, seconds: float, rate_hz: float = GRID_HZ) -> np.ndarray:
    return np.sqrt(rolling_mean(np.square(values), seconds, rate_hz))


def bandpass(values: np.ndarray, band: tuple[float, float], rate_hz: float = GRID_HZ) -> np.ndarray:
    """Zero-phase band-pass. NaN samples are bridged for filtering and restored as NaN."""
    good = np.isfinite(values)
    if good.sum() < 30:
        return np.full(len(values), np.nan)
    filled = np.interp(np.arange(len(values)), np.flatnonzero(good), values[good])
    high = min(band[1], rate_hz / 2 * 0.95)
    sos = sps.butter(2, (band[0], high), btype="bandpass", fs=rate_hz, output="sos")
    out = sps.sosfiltfilt(sos, filled)
    out[~good] = np.nan
    return out


def uri_path(trial_dir: Path, uri: str) -> Path | None:
    relative = unquote(urlsplit(str(uri or "")).path)
    return trial_dir / relative if relative else None


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


# ----------------------------------------------------------------------------------------------
# head track


def load_head_track(trial_dir: Path, session: dict[str, Any]) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    """Head positions on the session clock: ``{"t", "xyz", "quat" (or absent)}`` and a source report."""
    playback = read_json(trial_dir / "environment_reconstruction" / "scene_playback_data.json")
    poses = playback.get("poses")
    if isinstance(poses, list) and len(poses) > 10 and float(playback.get("fps") or 0) > 0:
        fps = float(playback["fps"])
        offset = float(playback.get("dashboard_offset_s") or 0.0)
        status = playback.get("pose_status") or []
        # Frame i of the Aria session video shows pose i, so the video frame time is the session time.
        t = np.arange(len(poses)) / fps + offset
        xyz = np.full((len(poses), 3), np.nan)
        quat = np.full((len(poses), 4), np.nan)
        estimated = 0
        for i, pose in enumerate(poses):
            if pose is None or len(pose) < 7:
                continue
            if i < len(status) and status[i] not in ("mps_supported", "interpolated_short_gap"):
                continue
            estimated += int(i < len(status) and status[i] == "interpolated_short_gap")
            xyz[i], quat[i] = pose[:3], pose[3:7]
        valid = np.isfinite(xyz[:, 0])
        report = {
            "available": bool(valid.sum() > 10), "kind": "measured",
            "source": "environment_reconstruction/scene_playback_data.json (MPS SLAM camera pose per video frame)",
            "rate_hz": fps, "has_orientation": True,
            "valid_fraction": round(float(valid.mean()), 4), "estimated_pose_count": estimated,
        }
        if report["available"]:
            return {"t": t, "xyz": xyz, "quat": quat}, report
    samples = (session.get("transition_review") or {}).get("samples") or []
    if len(samples) > 10 and all(k in samples[0] for k in ("t", "x", "y", "z")):
        t = np.array([float(s["t"]) for s in samples])
        xyz = np.array([[float(s["x"]), float(s["y"]), float(s["z"])] for s in samples])
        rate = float(1.0 / np.median(np.diff(t))) if len(t) > 2 else 0.0
        return {"t": t, "xyz": xyz}, {
            "available": True, "kind": "measured",
            "source": "session_dashboard_manifest.json transition_review.samples (block-mean MPS SLAM position)",
            "rate_hz": round(rate, 2), "has_orientation": False, "valid_fraction": 1.0,
            "limitations": ["No head orientation, so turning in place cannot be recognised."],
        }
    return None, {"available": False, "kind": "measured", "reason": "No saved head trajectory for this trial."}


def head_features(track: dict[str, np.ndarray], t: np.ndarray) -> dict[str, np.ndarray]:
    """Head-motion features on the grid. Distances in metres, speeds in m/s, angles in degrees."""
    out: dict[str, np.ndarray] = {}
    pos = np.stack([nan_interp(t, track["t"], track["xyz"][:, k], MAX_POSE_GAP_S) for k in range(3)], axis=1)
    ok = np.isfinite(pos).all(axis=1)
    out["head_ok"] = ok
    smooth = np.stack([rolling_mean(pos[:, k], 0.35) for k in range(3)], axis=1)
    smooth[~ok] = np.nan
    velocity = np.gradient(smooth, 1.0 / GRID_HZ, axis=0)
    out["speed"] = rolling_mean(np.hypot(velocity[:, 0], velocity[:, 1]), 0.5)

    # Net horizontal travel over a window: ignores sway and stepping in place.
    out["progress_speed"] = net_travel_speed(smooth, 2.0)
    out["progress_fast"] = net_travel_speed(smooth, 1.0)

    # How far the head gets around within 8 s. Shifting in a chair or at a counter stays small;
    # walking, however slow, covers ground (or height, on stairs).
    out["travel_extent"], out["vertical_extent"] = travel_extent(pos, ok, 8.0)

    # Vertical head bob in the stepping band, its envelope, and a step-rate estimate from its peaks.
    bob = bandpass(pos[:, 2], STEP_BAND_HZ)
    out["bob"] = bob
    out["bob_amp"] = rolling_rms(bob, 2.0) * np.sqrt(2.0)
    out["step_rate_hz"], out["bob_peak_times"] = step_rate(bob, t)

    # Horizontal movement that is not travel: residual around a 1.5 s smoothed path.
    trend = np.stack([rolling_mean(pos[:, k], 1.5) for k in range(2)], axis=1)
    out["sway_amp"] = rolling_rms(np.hypot(pos[:, 0] - trend[:, 0], pos[:, 1] - trend[:, 1]), 2.0)

    # Slow height trend separates sitting down / standing up and stairs from level movement.
    z_slow = rolling_mean(pos[:, 2], 1.0)
    out["height"] = z_slow
    out["vertical_speed"] = rolling_mean(np.gradient(z_slow, 1.0 / GRID_HZ), 1.0)

    # Turning: change of the direction of travel, and (when orientation exists) head yaw rate.
    vx, vy = rolling_mean(velocity[:, 0], 1.0), rolling_mean(velocity[:, 1], 1.0)
    cross = vx[:-1] * vy[1:] - vy[:-1] * vx[1:]
    dot = vx[:-1] * vx[1:] + vy[:-1] * vy[1:]
    path_turn = np.append(np.degrees(np.arctan2(cross, dot)) * GRID_HZ, np.nan)
    path_turn[~(out["speed"] > 0.08)] = np.nan          # direction of travel is undefined when barely moving
    out["path_turn_rate"] = rolling_mean(np.abs(path_turn), 1.0)
    if "quat" in track:
        out["yaw_rate"] = rolling_mean(np.abs(yaw_rate(track, t)), 1.0)
    else:
        out["yaw_rate"] = np.full(len(t), np.nan)
    for name, values in out.items():
        if name not in ("head_ok", "bob_peak_times") and values.shape == t.shape:
            values[~ok] = np.nan
    return out


def net_travel_speed(smooth: np.ndarray, window_s: float) -> np.ndarray:
    """Straight-line horizontal distance covered over a centered window, divided by its length."""
    half = int(round(window_s / 2 * GRID_HZ))
    net = np.full(len(smooth), np.nan)
    if len(smooth) > 2 * half:
        delta = smooth[2 * half:, :2] - smooth[:-2 * half, :2]
        net[half:-half] = np.hypot(delta[:, 0], delta[:, 1]) / window_s
        net[:half], net[-half:] = net[half], net[-half - 1]
    return net


def travel_extent(pos: np.ndarray, ok: np.ndarray, window_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Bounding-box diagonal of the horizontal positions, and the height range, in a centered window."""
    size = int(round(window_s * GRID_HZ)) | 1
    if not ok.any():
        nan = np.full(len(pos), np.nan)
        return nan, nan.copy()
    filled = np.stack([np.interp(np.arange(len(pos)), np.flatnonzero(ok), pos[ok, k]) for k in range(3)], axis=1)
    spans = [maximum_filter1d(filled[:, k], size, mode="nearest") - minimum_filter1d(filled[:, k], size, mode="nearest") for k in range(3)]
    return np.hypot(spans[0], spans[1]), spans[2]


def yaw_rate(track: dict[str, np.ndarray], t: np.ndarray) -> np.ndarray:
    """Signed rotation rate about the vertical world axis (deg/s), independent of camera axis conventions."""
    quat, tt = track["quat"], track["t"]
    good = np.isfinite(quat).all(axis=1)
    rate = np.full(len(tt), np.nan)
    pairs = np.flatnonzero(good[1:] & good[:-1])
    if len(pairs):
        relative = Rotation.from_quat(quat[pairs + 1]) * Rotation.from_quat(quat[pairs]).inv()
        rate[pairs + 1] = np.degrees(relative.as_rotvec()[:, 2]) / (tt[pairs + 1] - tt[pairs])
    return nan_interp(t, tt, rate, MAX_POSE_GAP_S)


def step_rate(bob: np.ndarray, t: np.ndarray, window_s: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Step-like events per second from peaks of the vertical head bob (one peak per step).

    A peak must stand out by at least 3 mm and by 0.35 of the local bob amplitude. This is a head-motion
    estimate of stepping rhythm, not a foot-contact measurement.
    """
    rate = np.full(len(t), np.nan)
    good = np.isfinite(bob)
    if good.sum() < 40:
        return rate, np.zeros(0)
    filled = np.where(good, bob, 0.0)
    local = rolling_rms(filled, 3.0) * np.sqrt(2.0)
    peaks, props = sps.find_peaks(filled, distance=int(0.28 * GRID_HZ), prominence=0.003)
    keep = props["prominences"] >= 0.35 * np.nan_to_num(local[peaks], nan=np.inf)
    peaks = peaks[keep & good[peaks]]
    marks = np.zeros(len(t))
    marks[peaks] = 1.0
    size = int(round(window_s * GRID_HZ)) | 1
    rate = uniform_filter1d(marks, size, mode="constant") * GRID_HZ
    rate[~good] = np.nan
    return rate, t[peaks]


# ----------------------------------------------------------------------------------------------
# downward video


def video_cache_key(path: Path, coverage: tuple[float, float] | None) -> str:
    stat = path.stat()
    text = f"{path.name}|{stat.st_size}|{int(stat.st_mtime)}|{coverage}|{VIDEO_FLOW_SIZE}|v2"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def video_motion(path: Path, coverage: tuple[float, float] | None, duration_s: float,
                 cache_dir: Path | None = None) -> dict[str, np.ndarray] | None:
    """Per-frame image-motion numbers from the downward video at 15 fps.

    ``flow`` is the mean optical-flow magnitude (pixels per frame at 160x90), ``flow_hi`` its 90th
    percentile, ``flow_moving`` the fraction of the image moving by more than 0.5 px, and ``luma``
    the mean brightness. Frames outside ``coverage`` are not decoded.
    """
    import cv2  # imported here so the rest of the detector works without OpenCV

    cache = cache_dir / f"downward_flow_{video_cache_key(path, coverage)}.npz" if cache_dir else None
    if cache and cache.is_file():
        with np.load(cache) as data:
            return {k: data[k] for k in data.files}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    stride = max(1, int(round(fps / VIDEO_NATIVE_FPS)))
    start_s, end_s = coverage if coverage else (0.0, duration_s)
    first = int(np.ceil(start_s * fps / stride) * stride)
    last = int(np.floor(min(end_s, duration_s) * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    times, flow, flow_hi, moving, luma = [], [], [], [], []
    previous = None
    index = first
    while index < last:
        ok, frame = capture.read()
        if not ok:
            break
        if (index - first) % stride == 0:
            grey = cv2.cvtColor(cv2.resize(frame, VIDEO_FLOW_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            brightness = float(grey.mean())
            if previous is not None:
                field_ = cv2.calcOpticalFlowFarneback(previous, grey, None, 0.5, 2, 13, 2, 5, 1.1, 0)
                magnitude = np.hypot(field_[..., 0], field_[..., 1])
                times.append(index / fps)
                flow.append(float(magnitude.mean()))
                flow_hi.append(float(np.percentile(magnitude, 90)))
                moving.append(float((magnitude > 0.5).mean()))
                luma.append(brightness)
            previous = grey
        index += 1
    capture.release()
    result = {"t": np.array(times), "flow": np.array(flow), "flow_hi": np.array(flow_hi),
              "flow_moving": np.array(moving), "luma": np.array(luma)}
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **result)
    return result


def video_features(motion: dict[str, np.ndarray], t: np.ndarray, coverage: tuple[float, float] | None) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    bright = motion["luma"] >= DARK_FRAME_LUMA
    for name in ("flow", "flow_hi", "flow_moving"):
        values = np.where(bright, motion[name], np.nan)
        out[name] = nan_interp(t, motion["t"], values, 0.4)
    ok = np.isfinite(out["flow"])
    if coverage:
        ok &= (t >= coverage[0] + 0.2) & (t <= coverage[1] - 0.2)
    out["video_ok"] = ok
    for name in ("flow", "flow_hi", "flow_moving"):
        out[name][~ok] = np.nan
    out["flow_smooth"] = rolling_mean(out["flow"], 1.0)
    out["flow_step_amp"] = rolling_rms(bandpass(out["flow"], STEP_BAND_HZ), 2.0) * np.sqrt(2.0)
    return out


# ----------------------------------------------------------------------------------------------
# audio (supporting evidence only)


def find_audio(trial_dir: Path, session: dict[str, Any], cache_dir: Path | None) -> tuple[Path | None, str]:
    """A mono WAV of the Aria audio on the session clock, or ``None``."""
    sync = trial_dir / "downward_sync" / "aria_sync_audio.wav"
    if sync.is_file():
        return sync, "downward_sync/aria_sync_audio.wav"
    video = uri_path(trial_dir, session.get("aria_session_video_uri", ""))
    ffmpeg = shutil.which("ffmpeg")
    if video and video.is_file() and ffmpeg and cache_dir:
        target = cache_dir / f"aria_audio_{video.stat().st_size}.wav"
        if not target.is_file():
            cache_dir.mkdir(parents=True, exist_ok=True)
            command = [ffmpeg, "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(target)]
            if subprocess.run(command, capture_output=True).returncode != 0 or not target.is_file():
                return None, "The Aria session video has no audio track that ffmpeg could extract."
        return target, "audio track of the Aria session video (extracted with ffmpeg)"
    if video and video.is_file() and ffmpeg:
        return None, "No saved Aria audio; it is extracted from the session video only when results are saved."
    return None, "No saved Aria audio (and no ffmpeg to extract it from the session video)."


def audio_envelope(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Times, an onset envelope and a loudness envelope at 100 Hz. Only signal energy is used."""
    with wave.open(str(path)) as wav:
        rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
    if width != 2 or rate < 4000:
        return None
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    hop = int(round(rate / AUDIO_ENVELOPE_HZ))
    _, times, spectrum = sps.stft(samples, fs=rate, nperseg=hop * 4, noverlap=hop * 3, boundary=None, padded=False)
    freqs = np.fft.rfftfreq(hop * 4, 1.0 / rate)
    band = (freqs >= 60) & (freqs <= 2500)
    magnitude = np.log1p(200.0 * np.abs(spectrum[band]))
    flux = np.maximum(np.diff(magnitude, axis=1, prepend=magnitude[:, :1]), 0.0).mean(axis=0)
    loudness = magnitude.mean(axis=0)
    return times, flux, loudness


def audio_features(envelope: tuple[np.ndarray, np.ndarray, np.ndarray], t: np.ndarray) -> dict[str, np.ndarray]:
    times, flux, loudness = envelope
    baseline = sps.medfilt(flux, 201)
    spread = np.median(np.abs(flux - baseline)) * 1.4826 + 1e-9
    peaks, _ = sps.find_peaks(flux - baseline, height=4.0 * spread, distance=int(0.2 * AUDIO_ENVELOPE_HZ))
    marks = np.zeros(len(t))
    index = np.clip(np.round(times[peaks] * GRID_HZ).astype(int), 0, len(t) - 1)
    np.add.at(marks, index, 1.0)
    size = int(round(4.0 * GRID_HZ)) | 1
    out = {"audio_onset_rate": uniform_filter1d(marks, size, mode="constant") * GRID_HZ,
           "audio_loudness": np.interp(t, times, loudness, left=np.nan, right=np.nan)}
    covered = (t >= times[0]) & (t <= times[-1])
    out["audio_ok"] = covered
    out["audio_onset_rate"][~covered] = np.nan
    out["audio_onset_times"] = times[peaks]
    return out


# ----------------------------------------------------------------------------------------------
# existing research outputs


def activity_context(session: dict[str, Any], t: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    segments = session.get("continuous_model_segments") or session.get("model_segments") or []
    codes = np.zeros(len(t), dtype=np.int8)
    clean = []
    for segment in segments:
        if not isinstance(segment, dict) or "start_s" not in segment or "end_s" not in segment:
            continue
        label = str(segment.get("label", "")).strip().lower()
        start, end = float(segment["start_s"]), float(segment["end_s"])
        codes[(t >= start) & (t < end)] = ACTIVITY_CODES.get(label, 0)
        clean.append({"label": label, "start_s": start, "end_s": end})
    return codes, clean


def saved_step_events(session: dict[str, Any]) -> np.ndarray:
    events = (session.get("stride_review") or {}).get("events") or []
    times = [float(e["session_s"]) for e in events if isinstance(e, dict) and e.get("session_s") not in (None, "")]
    return np.array(sorted(times))


def downward_coverage(trial_dir: Path, session: dict[str, Any], duration_s: float) -> tuple[tuple[float, float] | None, Path | None, str]:
    """Recorded window of the downward video on the session clock, its file, and a note."""
    trial = read_json(trial_dir / "trial_manifest.json")
    mode = session.get("media_mode") or trial.get("media_mode") or "dual_view"
    path = uri_path(trial_dir, session.get("downward_session_video_uri", ""))
    if mode == "aria_only" or path is None:
        return None, None, "This trial has no downward camera recording."
    if not path.is_file():
        return None, None, "The downward session video file is missing."
    attachment = session.get("downward_attachment") or trial.get("downward_attachment")
    if attachment:
        start, end = float(attachment["start_s"]), float(attachment["end_s"])
        return (max(0.0, start), min(duration_s, end)), path, "Recorded window from downward_attachment; the rest is padding."
    return (0.0, duration_s), path, "The downward video covers the whole session."


# ----------------------------------------------------------------------------------------------


def load_trial(runs_dir: Path, trial_id: str, *, use_video: bool = True, use_audio: bool = True,
               cache_dir: Path | None = None) -> TrialSignals:
    trial_dir = runs_dir / trial_id
    session = read_json(trial_dir / "session_dashboard_manifest.json")
    if not session:
        raise FileNotFoundError(f"{trial_id} has no session_dashboard_manifest.json")
    duration = float(session.get("session_duration_s") or 0)
    if duration <= 0:
        raise ValueError(f"{trial_id}: the manifest has no session duration")
    t = grid(duration)
    signals = TrialSignals(trial_id=trial_id, duration_s=duration, t=t)

    track, report = load_head_track(trial_dir, session)
    signals.sources["head_track"] = report
    if track is not None:
        signals.features.update(head_features(track, t))
    else:
        signals.features["head_ok"] = np.zeros(len(t), dtype=bool)

    coverage, video_path, note = downward_coverage(trial_dir, session, duration)
    signals.downward_coverage_s = coverage
    video_report: dict[str, Any] = {"available": False, "kind": "measured image motion", "note": note,
                                    "coverage_s": list(coverage) if coverage else None}
    if use_video and video_path is not None and coverage is not None:
        try:
            motion = video_motion(video_path, coverage, duration, cache_dir)
        except ImportError:
            motion, video_report["note"] = None, "OpenCV is not installed, so the downward video was not analysed."
        if motion is not None and len(motion["t"]) > 10:
            signals.features.update(video_features(motion, t, coverage))
            video_report.update(available=True, rate_hz=VIDEO_NATIVE_FPS,
                                usable_fraction_of_session=round(float(signals.features["video_ok"].mean()), 4),
                                source="downward session video, dense optical flow at 160x90")
    elif not use_video:
        video_report["note"] = "Video analysis was switched off for this run."
    signals.features.setdefault("video_ok", np.zeros(len(t), dtype=bool))
    signals.sources["downward_video"] = video_report

    audio_report: dict[str, Any] = {"available": False, "kind": "supporting"}
    if use_audio:
        audio_path, audio_note = find_audio(trial_dir, session, cache_dir)
        audio_report["note"] = audio_note
        envelope = audio_envelope(audio_path) if audio_path else None
        if envelope is not None:
            signals.features.update(audio_features(envelope, t))
            audio_report.update(available=True, source=audio_note, rate_hz=AUDIO_ENVELOPE_HZ)
    else:
        audio_report["note"] = "Audio analysis was switched off for this run."
    signals.features.setdefault("audio_ok", np.zeros(len(t), dtype=bool))
    signals.sources["audio"] = audio_report

    signals.activity, signals.activity_segments = activity_context(session, t)
    signals.sources["activity"] = {
        "available": bool(signals.activity_segments), "kind": "inferred",
        "source": "continuous activity classification in session_dashboard_manifest.json",
        "method": str((session.get("continuous_motion") or {}).get("version") or ""),
    }
    signals.step_event_times = saved_step_events(session)
    signals.sources["step_events"] = {
        "available": bool(len(signals.step_event_times)), "kind": "inferred",
        "source": "stride_review.events (SLAM head-sway step proxy, not foot contacts)",
        "count": int(len(signals.step_event_times)),
    }
    return signals
