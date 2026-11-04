#!/usr/bin/env python3
"""Suggest an Aria/downward-recorder sync offset from audio patterns.

This script:
1. Locates the extracted Aria audio for a processed trial.
2. Extracts a mono 16 kHz WAV from the downward-recorder MP4.
3. Computes low-rate audio envelopes from both recordings.
4. Uses cross-correlation to estimate the best offset.
5. Writes plots and a JSON suggestion so the user can verify the result.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GEN1_ROOT = Path(__file__).resolve().parents[2]
QWEN_REPO = GEN1_ROOT / "Qwen2.5-VL-fork"
RUNS_DIR = QWEN_REPO / "runs"
DOWNWARD_ROOT = GEN1_ROOT / "downward videos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-id", required=True, help="Trial id, e.g. Ntrial1")
    parser.add_argument(
        "--downward-video",
        type=Path,
        help="Optional explicit downward-recorder video path. If omitted, try to auto-match from the 'downward videos' folder.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Temporary sync WAV sample rate.",
    )
    parser.add_argument(
        "--frame-ms",
        type=float,
        default=20.0,
        help="Envelope frame size in milliseconds.",
    )
    parser.add_argument(
        "--peak-percentile",
        type=float,
        default=88.0,
        help="Percentile threshold for salient peak detection on the envelope.",
    )
    parser.add_argument(
        "--peak-min-separation-s",
        type=float,
        default=0.35,
        help="Minimum spacing between detected salient peaks.",
    )
    parser.add_argument(
        "--peak-match-tolerance-s",
        type=float,
        default=0.18,
        help="Tolerance window for counting two peaks as a match.",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=24,
        help="How many top correlation candidates to rescore with peak matching.",
    )
    return parser.parse_args()


def load_manifest(trial_id: str) -> dict[str, object]:
    manifest_path = RUNS_DIR / trial_id / "trial_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text())


def discover_ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - environment-specific fallback
        raise RuntimeError("Could not find bundled ffmpeg executable.") from exc


def discover_aria_audio(trial_id: str, manifest: dict[str, object]) -> Path:
    extracted_video = Path(str(manifest["extracted_media"]["video_mp4"]["path"]))
    audio_path = extracted_video.parent / "audio_track.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Aria extracted audio not found: {audio_path}")
    return audio_path


def infer_downward_video(trial_id: str) -> Path:
    suffix = trial_id
    if suffix.lower().startswith("ntrial"):
        suffix = "NTrial" + suffix[6:]
    candidates = [
        DOWNWARD_ROOT / f"{suffix}.MP4",
        DOWNWARD_ROOT / f"{suffix}.mp4",
        DOWNWARD_ROOT / f"{suffix}.Mov",
        DOWNWARD_ROOT / f"{suffix}.MOV",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not auto-match a downward video for {trial_id} in {DOWNWARD_ROOT}. "
        "Use --downward-video explicitly."
    )


def convert_audio_to_sync_wav(input_path: Path, output_path: Path, sample_rate: int, ffmpeg_exe: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sample_width)
    if dtype is None:
        raise RuntimeError(f"Unsupported sample width for {path}: {sample_width}")

    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    peak = np.max(np.abs(samples)) or 1.0
    return samples / peak, sample_rate


def build_envelope(samples: np.ndarray, sample_rate: int, frame_ms: float) -> tuple[np.ndarray, float]:
    hop = max(1, int(round(sample_rate * (frame_ms / 1000.0))))
    usable = (len(samples) // hop) * hop
    samples = samples[:usable]
    frames = samples.reshape(-1, hop)
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-8)
    smooth = np.convolve(rms, np.ones(5) / 5.0, mode="same")
    emphasis = np.maximum(0.0, smooth - np.convolve(smooth, np.ones(25) / 25.0, mode="same"))
    centered = emphasis - np.mean(emphasis)
    scale = np.std(centered) or 1.0
    return centered / scale, hop / sample_rate


def estimate_offset_s(aria_env: np.ndarray, downward_env: np.ndarray, frame_s: float) -> tuple[float, np.ndarray, np.ndarray]:
    correlation = np.correlate(aria_env, downward_env, mode="full")
    lags = np.arange(-len(downward_env) + 1, len(aria_env))
    best_idx = int(np.argmax(correlation))
    return float(lags[best_idx] * frame_s), lags.astype(np.float64) * frame_s, correlation


def top_offset_candidates(lag_seconds: np.ndarray, correlation: np.ndarray, top_k: int = 5) -> list[dict[str, float]]:
    if len(correlation) == 0:
        return []
    indices = np.argpartition(correlation, -top_k)[-top_k:]
    ranked = indices[np.argsort(correlation[indices])[::-1]]
    return [
        {
            "offset_s": float(lag_seconds[idx]),
            "score": float(correlation[idx]),
        }
        for idx in ranked
    ]


def smooth_signal(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal
    return np.convolve(signal, np.ones(window) / window, mode="same")


def detect_salient_peaks(envelope: np.ndarray, frame_s: float, percentile: float, min_separation_s: float) -> np.ndarray:
    if len(envelope) < 3:
        return np.array([], dtype=np.float64)
    threshold = float(np.percentile(envelope, percentile))
    candidate_indices = np.where(
        (envelope[1:-1] > envelope[:-2]) & (envelope[1:-1] >= envelope[2:]) & (envelope[1:-1] >= threshold)
    )[0] + 1
    if len(candidate_indices) == 0:
        return np.array([], dtype=np.float64)

    min_gap_frames = max(1, int(round(min_separation_s / frame_s)))
    order = candidate_indices[np.argsort(envelope[candidate_indices])[::-1]]
    kept: list[int] = []
    for idx in order:
        if all(abs(idx - existing) >= min_gap_frames for existing in kept):
            kept.append(int(idx))
    kept.sort()
    return np.array(kept, dtype=np.float64) * frame_s


def build_peak_train(peak_times_s: np.ndarray, frame_s: float, length: int) -> np.ndarray:
    signal = np.zeros(length, dtype=np.float32)
    if len(peak_times_s) == 0:
        return signal
    indices = np.clip(np.round(peak_times_s / frame_s).astype(int), 0, max(0, length - 1))
    signal[indices] = 1.0
    return smooth_signal(signal, 5)


def match_peak_times(
    aria_peak_times_s: np.ndarray,
    downward_peak_times_s: np.ndarray,
    offset_s: float,
    tolerance_s: float,
    aria_duration_s: float,
    downward_duration_s: float,
) -> dict[str, object]:
    shifted_aria = aria_peak_times_s - offset_s
    overlap_start = max(0.0, -offset_s)
    overlap_end = min(downward_duration_s, aria_duration_s - offset_s)
    if overlap_end <= overlap_start:
        return {
            "matched_count": 0,
            "aria_overlap_count": 0,
            "downward_overlap_count": 0,
            "f1_score": 0.0,
            "median_abs_error_s": None,
            "match_times_s": [],
        }

    aria_overlap = shifted_aria[(shifted_aria >= overlap_start) & (shifted_aria <= overlap_end)]
    downward_overlap = downward_peak_times_s[
        (downward_peak_times_s >= overlap_start) & (downward_peak_times_s <= overlap_end)
    ]
    if len(aria_overlap) == 0 or len(downward_overlap) == 0:
        return {
            "matched_count": 0,
            "aria_overlap_count": int(len(aria_overlap)),
            "downward_overlap_count": int(len(downward_overlap)),
            "f1_score": 0.0,
            "median_abs_error_s": None,
            "match_times_s": [],
        }

    used_downward = np.zeros(len(downward_overlap), dtype=bool)
    match_errors: list[float] = []
    match_times: list[float] = []
    for t in aria_overlap:
        insert_at = int(np.searchsorted(downward_overlap, t))
        candidate_indices = []
        if insert_at < len(downward_overlap):
            candidate_indices.append(insert_at)
        if insert_at > 0:
            candidate_indices.append(insert_at - 1)
        best_idx = None
        best_err = None
        for idx in candidate_indices:
            if used_downward[idx]:
                continue
            err = abs(float(downward_overlap[idx] - t))
            if err <= tolerance_s and (best_err is None or err < best_err):
                best_idx = idx
                best_err = err
        if best_idx is not None and best_err is not None:
            used_downward[best_idx] = True
            match_errors.append(best_err)
            match_times.append(float(t))

    matched = len(match_errors)
    denom = len(aria_overlap) + len(downward_overlap)
    f1_score = (2.0 * matched / denom) if denom else 0.0
    median_abs_error_s = float(np.median(match_errors)) if match_errors else None
    return {
        "matched_count": matched,
        "aria_overlap_count": int(len(aria_overlap)),
        "downward_overlap_count": int(len(downward_overlap)),
        "f1_score": f1_score,
        "median_abs_error_s": median_abs_error_s,
        "match_times_s": match_times,
    }


def score_candidates_with_peaks(
    raw_candidates: list[dict[str, float]],
    peak_candidates: list[dict[str, float]],
    aria_peak_times_s: np.ndarray,
    downward_peak_times_s: np.ndarray,
    tolerance_s: float,
    aria_duration_s: float,
    downward_duration_s: float,
) -> list[dict[str, float | int | None]]:
    merged: dict[float, dict[str, float | int | None]] = {}
    for source_name, source_candidates in [("raw_corr", raw_candidates), ("peak_corr", peak_candidates)]:
        for item in source_candidates:
            offset_s = round(float(item["offset_s"]), 3)
            if offset_s not in merged:
                merged[offset_s] = {
                    "offset_s": offset_s,
                    "raw_score": None,
                    "peak_corr_score": None,
                }
            score_key = "raw_score" if source_name == "raw_corr" else "peak_corr_score"
            merged[offset_s][score_key] = float(item["score"])

    rescored: list[dict[str, float | int | None]] = []
    for offset_s, base in merged.items():
        peak_match = match_peak_times(
            aria_peak_times_s,
            downward_peak_times_s,
            offset_s,
            tolerance_s,
            aria_duration_s,
            downward_duration_s,
        )
        raw_score = float(base["raw_score"]) if base["raw_score"] is not None else 0.0
        peak_corr_score = float(base["peak_corr_score"]) if base["peak_corr_score"] is not None else 0.0
        combined_score = peak_match["f1_score"] * 1000.0 + peak_corr_score * 0.05 + raw_score * 0.0005
        rescored.append(
            {
                "offset_s": offset_s,
                "combined_score": combined_score,
                "peak_f1_score": float(peak_match["f1_score"]),
                "matched_peak_count": int(peak_match["matched_count"]),
                "aria_overlap_peak_count": int(peak_match["aria_overlap_count"]),
                "downward_overlap_peak_count": int(peak_match["downward_overlap_count"]),
                "median_peak_error_s": peak_match["median_abs_error_s"],
                "raw_score": base["raw_score"],
                "peak_corr_score": base["peak_corr_score"],
            }
        )

    return sorted(
        rescored,
        key=lambda item: (
            float(item["combined_score"]),
            float(item["peak_f1_score"]),
            float(item["peak_corr_score"] or 0.0),
            float(item["raw_score"] or 0.0),
        ),
        reverse=True,
    )


def overlay_for_plot(
    aria_env: np.ndarray,
    downward_env: np.ndarray,
    frame_s: float,
    offset_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lag_frames = int(round(offset_s / frame_s))
    if lag_frames >= 0:
        aria_start = lag_frames
        downward_start = 0
    else:
        aria_start = 0
        downward_start = -lag_frames
    length = min(len(aria_env) - aria_start, len(downward_env) - downward_start)
    if length <= 0:
        return np.array([]), np.array([]), np.array([])
    t = np.arange(length) * frame_s
    return (
        t,
        aria_env[aria_start : aria_start + length],
        downward_env[downward_start : downward_start + length],
    )


def plot_sync_overlay(output_path: Path, time_s: np.ndarray, aria_env: np.ndarray, downward_env: np.ndarray, offset_s: float) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    if len(time_s):
        ax.plot(time_s, aria_env, label="Aria envelope", linewidth=1.2)
        ax.plot(time_s, downward_env, label="Downward envelope", linewidth=1.0, alpha=0.85)
    ax.set_title(f"Aligned Audio Envelopes (offset = {offset_s:+.3f} s)")
    ax.set_xlabel("Aligned time (s)")
    ax.set_ylabel("Normalized envelope")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_correlation(output_path: Path, lag_seconds: np.ndarray, correlation: np.ndarray, offset_s: float) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(lag_seconds, correlation, linewidth=1.0, color="#1f77b4")
    ax.axvline(offset_s, color="crimson", linestyle="--", linewidth=1.5, label=f"Best offset {offset_s:+.3f}s")
    ax.set_title("Audio Cross-Correlation")
    ax.set_xlabel("Offset (s)")
    ax.set_ylabel("Correlation score")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_peak_overlay(
    output_path: Path,
    time_s: np.ndarray,
    aria_env: np.ndarray,
    downward_env: np.ndarray,
    aria_peak_times_s: np.ndarray,
    downward_peak_times_s: np.ndarray,
    offset_s: float,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    if len(time_s):
        ax.plot(time_s, aria_env, label="Aria envelope", linewidth=1.2)
        ax.plot(time_s, downward_env, label="Downward envelope", linewidth=1.0, alpha=0.85)
    shifted_aria_peaks = aria_peak_times_s - offset_s
    if len(shifted_aria_peaks):
        ax.vlines(shifted_aria_peaks, ymin=np.min(aria_env) - 0.2, ymax=np.max(aria_env) + 0.2, colors="#1f77b4", alpha=0.18, linewidth=1)
    if len(downward_peak_times_s):
        ax.vlines(downward_peak_times_s, ymin=np.min(downward_env) - 0.2, ymax=np.max(downward_env) + 0.2, colors="#ff7f0e", alpha=0.15, linewidth=1)
    ax.set_title(f"Peak-Aware Overlay (offset = {offset_s:+.3f} s)")
    ax.set_xlabel("Aligned time (s)")
    ax.set_ylabel("Normalized envelope")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.trial_id)
    aria_audio = discover_aria_audio(args.trial_id, manifest)
    downward_video = args.downward_video.resolve() if args.downward_video else infer_downward_video(args.trial_id)

    output_dir = RUNS_DIR / args.trial_id / "downward_sync"
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = discover_ffmpeg_executable()
    aria_sync_wav = output_dir / "aria_sync_audio.wav"
    downward_sync_wav = output_dir / "downward_sync_audio.wav"

    convert_audio_to_sync_wav(aria_audio, aria_sync_wav, args.sample_rate, ffmpeg_exe)
    convert_audio_to_sync_wav(downward_video, downward_sync_wav, args.sample_rate, ffmpeg_exe)

    aria_samples, aria_sr = load_wav_mono(aria_sync_wav)
    downward_samples, downward_sr = load_wav_mono(downward_sync_wav)
    if aria_sr != downward_sr:
        raise RuntimeError("Resampled audio sample rates do not match.")

    aria_env, frame_s = build_envelope(aria_samples, aria_sr, args.frame_ms)
    downward_env, downward_frame_s = build_envelope(downward_samples, downward_sr, args.frame_ms)
    if not math.isclose(frame_s, downward_frame_s, rel_tol=1e-6, abs_tol=1e-9):
        raise RuntimeError("Envelope frame sizes do not match.")

    raw_offset_s, lag_seconds, correlation = estimate_offset_s(aria_env, downward_env, frame_s)
    raw_candidates = top_offset_candidates(lag_seconds, correlation, top_k=args.candidate_count)

    aria_peak_times_s = detect_salient_peaks(
        aria_env,
        frame_s,
        percentile=args.peak_percentile,
        min_separation_s=args.peak_min_separation_s,
    )
    downward_peak_times_s = detect_salient_peaks(
        downward_env,
        frame_s,
        percentile=args.peak_percentile,
        min_separation_s=args.peak_min_separation_s,
    )
    aria_duration_s = len(aria_samples) / aria_sr
    downward_duration_s = len(downward_samples) / downward_sr
    peak_train_aria = build_peak_train(aria_peak_times_s, frame_s, len(aria_env))
    peak_train_downward = build_peak_train(downward_peak_times_s, frame_s, len(downward_env))
    peak_offset_s, peak_lag_seconds, peak_correlation = estimate_offset_s(peak_train_aria, peak_train_downward, frame_s)
    peak_candidates = top_offset_candidates(peak_lag_seconds, peak_correlation, top_k=args.candidate_count)
    rescored_candidates = score_candidates_with_peaks(
        raw_candidates,
        peak_candidates,
        aria_peak_times_s,
        downward_peak_times_s,
        tolerance_s=args.peak_match_tolerance_s,
        aria_duration_s=aria_duration_s,
        downward_duration_s=downward_duration_s,
    )
    best_candidate = rescored_candidates[0] if rescored_candidates else {"offset_s": raw_offset_s}
    offset_s = float(best_candidate["offset_s"])

    aria_trim_start_s = max(0.0, offset_s)
    downward_trim_start_s = max(0.0, -offset_s)
    shared_duration_s = min(
        aria_duration_s - aria_trim_start_s,
        downward_duration_s - downward_trim_start_s,
    )

    overlay_t, overlay_aria, overlay_downward = overlay_for_plot(aria_env, downward_env, frame_s, offset_s)
    plot_sync_overlay(output_dir / "aligned_envelopes.png", overlay_t, overlay_aria, overlay_downward, offset_s)
    plot_correlation(output_dir / "correlation_curve.png", lag_seconds, correlation, offset_s)
    plot_peak_overlay(
        output_dir / "aligned_peaks.png",
        overlay_t,
        overlay_aria,
        overlay_downward,
        aria_peak_times_s,
        downward_peak_times_s,
        offset_s,
    )

    suggestion = {
        "trial_id": args.trial_id,
        "downward_video": str(downward_video),
        "aria_audio": str(aria_audio),
        "sample_rate_hz": aria_sr,
        "frame_s": frame_s,
        "raw_envelope_offset_s": raw_offset_s,
        "peak_train_offset_s": peak_offset_s,
        "estimated_offset_s": offset_s,
        "interpretation": (
            "Positive offset means the Aria recording started earlier and should be trimmed at the front by this amount. "
            "Negative offset means the downward recording started earlier."
        ),
        "aria_trim_start_s": aria_trim_start_s,
        "downward_trim_start_s": downward_trim_start_s,
        "shared_duration_s": max(0.0, shared_duration_s),
        "salient_peak_counts": {
            "aria": int(len(aria_peak_times_s)),
            "downward": int(len(downward_peak_times_s)),
        },
        "top_candidates": rescored_candidates[:10],
        "verification_files": {
            "aligned_envelopes_png": str(output_dir / "aligned_envelopes.png"),
            "aligned_peaks_png": str(output_dir / "aligned_peaks.png"),
            "correlation_curve_png": str(output_dir / "correlation_curve.png"),
            "aria_sync_wav": str(aria_sync_wav),
            "downward_sync_wav": str(downward_sync_wav),
        },
    }
    suggestion_path = output_dir / "sync_suggestion.json"
    suggestion_path.write_text(json.dumps(suggestion, indent=2))

    print(f"Wrote sync suggestion to: {suggestion_path}")
    print(json.dumps(suggestion, indent=2))


if __name__ == "__main__":
    main()
