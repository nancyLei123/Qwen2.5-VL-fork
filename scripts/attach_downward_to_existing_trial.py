#!/usr/bin/env python3
"""Attach verified downward footage without shifting an existing Aria timeline."""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def probe(path):
    return json.loads(subprocess.check_output([
        shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe", "-v", "error",
        "-show_streams", "-show_format", "-of", "json", str(path)], text=True))


def coverage_plan(offset, source_duration, session_duration, fps=30):
    if not all(math.isfinite(v) for v in (offset, source_duration, session_duration, fps)):
        raise ValueError("Non-finite media timing")
    if min(source_duration, session_duration, fps) <= 0:
        raise ValueError("Invalid media duration or frame rate")
    start = max(0.0, round(offset * fps) / fps)
    trim = max(0.0, -offset)
    available = min(source_duration - trim, session_duration - start)
    if available <= 1 / fps:
        raise ValueError("No usable overlap with the existing session")
    end = min(session_duration, start + round(available * fps) / fps)
    return {"offset_s": offset, "source_trim_s": trim, "start_s": start,
            "end_s": end, "session_duration_s": session_duration, "output_fps": fps,
            "source_duration_s": source_duration, "available_duration_s": end - start,
            "padding_before_s": start, "padding_after_s": session_duration - end,
            "applied_offset_s": start - trim, "quantization_error_s": start - trim - offset}


def protected_files(directory):
    patterns = ("manual_labels/**/*", "labels/**/*", "stride_step_diagnostics/**/*",
                "raw_parameter_diagnostics/**/*", "timeline_predictions.json", "rule_segments.csv",
                "clip_metrics.csv", "environment_reconstruction/scene_playback_data.json",
                "parkinson_motion_review/*segments*.json", "parkinson_motion_review/*.csv")
    return {str(p.relative_to(directory)): digest(p) for pattern in patterns
            for p in directory.glob(pattern) if p.is_file()}


def refresh_attached_views(trial):
    """Re-render saved results; never rerun classification or reset annotation stores."""
    from build_downward_dual_view_dashboards import (
        build_dashboard_html, build_stride_review_html, build_transition_review_html)
    from build_full_session_manual_labeling import build_html as manual_html, load_payload
    from build_parkinson_motion_review import build_html as parkinson_html
    directory = RUNS / trial
    session = load(directory / "session_dashboard_manifest.json")
    for name, builder in (("index.html", build_dashboard_html),
                          ("pd_insighter_stride_review.html", build_stride_review_html),
                          ("pd_insighter_transition_review.html", build_transition_review_html)):
        (directory / name).write_text(builder(session))
    (directory / "pd_insighter_manual_labeling.html").write_text(manual_html(load_payload(trial)))
    path = directory / "parkinson_motion_review" / f"{trial}_parkinson_review_payload.json"
    if path.is_file():
        payload = load(path)
        payload["media"]["downward"] = session["downward_session_video_uri"]
        save(path, payload)
        document = parkinson_html(payload)
        if trial != "Ntrial10":
            document = document.replace("/api/parkinson-manual-segments", "/api/session-manual-segments")
        (directory / "pd_insighter_parkinson_review.html").write_text(document)


def encode_aligned(source, target, plan):
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    start, trim = plan["start_s"], plan["source_trim_s"]
    duration, fps = plan["session_duration_s"], plan["output_fps"]
    # Add black (never cloned) frames outside observed coverage. No speed change.
    vf = (f"trim=start={trim}:duration={plan['available_duration_s']},setpts=PTS-STARTPTS,"
          f"fps={fps},tpad=start_mode=add:start={round(start * fps)}:stop_mode=add:stop=-1,"
          f"trim=duration={duration},format=yuv420p")
    af = (f"atrim=start={trim},asetpts=PTS-STARTPTS,adelay={round(start * 1000)}:all=1,"
          f"apad,atrim=duration={duration}")
    temporary = target.with_name(target.stem + ".building.mp4")
    subprocess.run([ffmpeg, "-v", "error", "-xerror", "-y", "-i", str(source),
                    "-map", "0:v:0", "-map", "0:a:0", "-vf", vf, "-af", af,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-threads", "2",
                    "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-t", str(duration),
                    str(temporary)], check=True)
    info = probe(temporary)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    if abs(float(video["duration"]) - duration) > 1.5 / fps:
        raise ValueError("Aligned video duration does not match the Aria session")
    temporary.replace(target)


def attach(trial, source_download):
    directory = RUNS / trial
    manifest_path = directory / "trial_manifest.json"
    session_path = directory / "session_dashboard_manifest.json"
    manifest, session = load(manifest_path), load(session_path)
    if manifest.get("media_mode") != "aria_only" and not manifest.get("downward_attachment"):
        raise ValueError("This attachment workflow is for an existing Aria-only timeline")
    sync = load(directory / "downward_sync/sync_suggestion.json")
    verification = load(directory / "downward_sync/sync_verification.json")
    if not sync.get("sync_verified") or not verification.get("valid"):
        raise ValueError("Independent audio verification must pass before attachment")
    source = Path(sync["downward_video"])
    source_hash = digest(source)
    if digest(source_download) != source_hash:
        raise ValueError("Copied source does not match the original download")
    source_info = probe(source)
    video = next(s for s in source_info["streams"] if s["codec_type"] == "video")
    plan = coverage_plan(float(sync["estimated_offset_s"]), float(video["duration"]),
                         float(session["session_duration_s"]))
    before = protected_files(directory)
    backup = directory / "downward_sync/before_attachment"
    backup.mkdir(exist_ok=True)
    for p in [manifest_path, session_path, directory / "summary.json", *directory.glob("*.html"),
              directory / "parkinson_motion_review" / f"{trial}_parkinson_review_payload.json"]:
        target = backup / p.name
        if p.is_file() and not target.exists():
            shutil.copy2(p, target)
    target = directory / "downward_master_session/downward_session_video.mp4"
    build_path = target.with_suffix(".build.json")
    build = {"type": "aria_clock_downward_attachment_v1", "source_sha256": source_hash, **plan}
    if not (target.is_file() and build_path.is_file() and load(build_path) == build):
        encode_aligned(source, target, plan)
        save(build_path, build)
    attachment = {**plan, "source_path": str(source), "original_download": str(source_download),
                  "source_sha256": source_hash, "source_width": video["width"], "source_height": video["height"],
                  "source_fps": video["r_frame_rate"], "rotation_quarter_turns": 0,
                  "timeline": "Existing Aria session preserved, zero remains original RGB frame zero",
                  "padding": "Black frames and silence outside recording coverage; not observed footage",
                  "verification": verification}
    manifest.update(media_mode="dual_view", downward_attachment=attachment)
    manifest.setdefault("raw_data", {})["downward_video"] = {"path": str(source), "exists": True}
    session.update(media_mode="dual_view", downward_attachment=attachment,
                   downward_session_video_uri=f"downward_master_session/downward_session_video.mp4?v={int(target.stat().st_mtime)}",
                   downward_session_duration_s=plan["session_duration_s"],
                   shared_overlap_duration_s=plan["available_duration_s"])
    save(manifest_path, manifest)
    save(session_path, session)
    summary_path = directory / "summary.json"
    if summary_path.exists():
        summary = load(summary_path)
        summary.update(media_mode="dual_view", downward_coverage_s=[plan["start_s"], plan["end_s"]])
        summary["limitations"] = [s for s in summary.get("limitations", []) if not s.startswith("No paired downward video")]
        note = "Partial downward footage is now aligned; foot contacts and action labels still require manual validation."
        summary["limitations"] = list(dict.fromkeys([*summary["limitations"], note]))
        save(summary_path, summary)
    refresh_attached_views(trial)
    after = protected_files(directory)
    if before != after:
        raise RuntimeError("Protected analysis or annotation files changed unexpectedly")
    save(directory / "downward_sync/attachment_report.json", {
        "trial_id": trial, **attachment, "original_copy_verified": True,
        "protected_files_unchanged": len(before), "aligned_video": str(target)})
    print(json.dumps({"trial": trial, "coverage_s": [plan["start_s"], plan["end_s"]],
                      "session_s": plan["session_duration_s"], "protected_files_unchanged": len(before)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--original-download", type=Path, required=True)
    args = parser.parse_args()
    attach(args.trial, args.original_download)
