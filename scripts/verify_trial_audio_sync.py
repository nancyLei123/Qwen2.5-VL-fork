#!/usr/bin/env python3
"""Verify an audio sync suggestion in independent windows before building media."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import butter, correlate, resample_poly, sosfiltfilt
from suggest_downward_audio_sync import load_wav_mono

RUNS = Path(__file__).resolve().parents[1] / "runs"


def local_offset(aria, downward, rate, start, duration, approximate):
    down_start = round(start * rate)
    n = round(duration * rate)
    ref = downward[down_start:down_start + n]
    aria_start = max(0, round((start + approximate - 1.5) * rate))
    search = aria[aria_start:min(len(aria), round((start + approximate + duration + 1.5) * rate))]
    if len(ref) != n or len(search) < n or np.std(ref) < 1e-6:
        return None
    ref = ref - ref.mean()
    corr = correlate(search, ref, mode="valid", method="fft")
    sums = np.r_[0.0, np.cumsum(search)]
    powers = np.r_[0.0, np.cumsum(search ** 2)]
    energy = powers[n:] - powers[:-n] - (sums[n:] - sums[:-n]) ** 2 / n
    scores = corr / np.sqrt(np.maximum(energy * np.dot(ref, ref), 1e-20))
    peak = int(np.argmax(np.abs(scores)))
    return {"downward_start_s": start, "window_duration_s": duration,
            "audio_offset_s": (aria_start + peak - down_start) / rate,
            "absolute_waveform_correlation": float(abs(scores[peak]))}


def verify(trial):
    run = RUNS / trial
    manifest = json.loads((run / "trial_manifest.json").read_text())
    path = run / "downward_sync/sync_suggestion.json"
    suggestion = json.loads(path.read_text())
    original = path.with_name("sync_suggestion.initial.json")
    if not original.exists():
        original.write_text(path.read_text())
    initial = json.loads(original.read_text())
    aria, sr = load_wav_mono(Path(suggestion["verification_files"]["aria_sync_wav"]))
    down, dsr = load_wav_mono(Path(suggestion["verification_files"]["downward_sync_wav"]))
    assert sr == dsr == 16000
    rate = 8000
    band = butter(4, [300, 3000], btype="bandpass", fs=rate, output="sos")
    aria = sosfiltfilt(band, resample_poly(aria, 1, 2))
    down = sosfiltfilt(band, resample_poly(down, 1, 2))
    approximate = float(initial["estimated_offset_s"])
    duration = len(down) / rate
    windows = [local_offset(aria, down, rate, float(t), 20.0, approximate)
               for t in np.linspace(max(5.0, -approximate + 3), duration - 25, 7)]
    windows = [row for row in windows if row]
    strong = [row for row in windows if row["absolute_waveform_correlation"] >= 0.12]
    report = {"trial_id": trial, "method": "Independent 20-second bandpassed waveform correlation windows",
              "windows": windows, "valid": False}
    if len(strong) >= 4:
        offsets = np.array([row["audio_offset_s"] for row in strong])
        times = np.array([row["downward_start_s"] + 10 for row in strong])
        slope, intercept = np.polyfit(times, offsets, 1)
        spread = float(np.ptp(offsets))
        report.update(audio_offset_s=float(np.median(offsets)), offset_spread_s=spread,
                      estimated_drift_over_overlap_s=float(slope * duration), valid=spread < 0.15)
    report_path = path.with_name("sync_verification.json")
    report_path.write_text(json.dumps(report, indent=2))
    if not report["valid"]:
        raise RuntimeError(f"{trial}: sync needs review; see {report_path}")
    video = Path(manifest["extracted_media"]["video_mp4"]["path"])
    audio_meta = json.loads((video.parent / "audio_metadata.json").read_text())
    video_meta = json.loads((video.parent / "metadata.json").read_text())
    probe = json.loads(subprocess.check_output([
        shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,start_time,duration:format=duration", "-of", "json", suggestion["downward_video"]
    ], text=True))
    down_audio = next(row for row in probe["streams"] if row["codec_type"] == "audio")
    down_video = next(row for row in probe["streams"] if row["codec_type"] == "video")
    down_audio_start = float(down_audio.get("start_time", 0)) - float(down_video.get("start_time", 0))
    # The audio WAV and RGB frame zero are not necessarily simultaneous.
    offset = report["audio_offset_s"] - float(audio_meta["video_audio_offset_s"]) - down_audio_start
    aria_trim, down_trim = max(0.0, offset), max(0.0, -offset)
    aria_duration = float(video_meta["written_frames"]) / float(video_meta["fps"])
    suggestion.update(estimated_offset_s=offset, audio_offset_s=report["audio_offset_s"],
                      aria_trim_start_s=aria_trim, downward_trim_start_s=down_trim,
                      shared_duration_s=min(aria_duration - aria_trim, float(down_video["duration"]) - down_trim),
                      downward_audio_start_offset_s=down_audio_start,
                      sync_verified=True, sync_verification_file=str(report_path),
                      interpretation="Aria RGB video time = downward video time + estimated_offset_s; corrected for audio/RGB start offset.")
    path.write_text(json.dumps(suggestion, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"{trial}: verified RGB offset {offset:.6f}s; overlap {suggestion['shared_duration_s']:.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", nargs="+", required=True)
    for trial in parser.parse_args().trials:
        verify(trial)
