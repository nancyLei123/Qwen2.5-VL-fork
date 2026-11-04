#!/usr/bin/env python3
"""Inspect an existing dashboard video without modifying source media or labels."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_frame(capture, index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"Cannot read video frame {index}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def inspect(trial, times, prefix=""):
    run = ROOT / "runs" / trial
    output = run / "environment_reconstruction"
    output.mkdir(exist_ok=True)
    config = json.loads((run / "downward_master_session/aria_session_video.build.json").read_text())
    source = Path(config["source_video_path"])
    video = run / "downward_master_session/aria_session_video.mp4"
    original, dashboard = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(video))
    fps = dashboard.get(cv2.CAP_PROP_FPS)
    source_fps = original.get(cv2.CAP_PROP_FPS)
    if not fps or abs(fps - source_fps) > 1e-6:
        raise ValueError("Unexpected video FPS mapping")
    trim_frames = config["video_trim_start_s"] * fps
    if abs(trim_frames - round(trim_frames)) > 0.001:
        raise ValueError("Trim is not frame aligned")
    trim_frames = round(trim_frames)
    manifest = json.loads((run / "session_dashboard_manifest.json").read_text())
    previews, audits = [], []
    for seconds in times:
        frame_index = round(seconds * fps)
        frame = read_frame(dashboard, frame_index)
        small = cv2.resize(frame, (192, 192)).astype(np.float32)
        errors = {}
        for delta in range(-2, 3):
            candidate = read_frame(original, trim_frames + frame_index + delta)
            errors[str(delta)] = float(np.abs(small - cv2.resize(candidate, (192, 192)).astype(np.float32)).mean())
        label = next((s["label"] for s in manifest.get("manual_timeline_segments", [])
                      if s["start_s"] <= seconds < s["end_s"]), "unlabeled")
        audits.append({"dashboard_s": seconds, "dashboard_frame": frame_index,
                       "expected_source_frame": trim_frames + frame_index,
                       "mean_rgb8_difference_by_delta": errors,
                       "best_delta_frames": int(min(errors, key=errors.get)), "manual_label": label})
        previews.append(Image.fromarray(frame).resize((300, 300)))
    original.release()
    dashboard.release()
    sheet = Image.new("RGB", (1200, 330 * ((len(previews) + 3) // 4)), "#f4f5ef")
    draw = ImageDraw.Draw(sheet)
    for index, (preview, audit) in enumerate(zip(previews, audits)):
        x, y = index % 4 * 300, index // 4 * 330
        sheet.paste(preview, (x, y + 30))
        draw.text((x + 8, y + 8), f"Dashboard {audit['dashboard_s']} s | {audit['manual_label']}", fill="#17231c")
    sheet.save(output / f"{prefix}candidate_preview.jpg", quality=93)
    protected = [run / "session_dashboard_manifest.json", *sorted((run / "manual_labels").glob("*.json"))]
    audit = {"trial_id": trial, "video": str(video), "source_video": str(source),
             "fps": fps, "source_trim_frames": trim_frames, "source_trim_s": trim_frames / fps,
             "frame_checks": audits, "protected_sha256": {str(p): sha256(p) for p in protected}}
    path = output / f"{prefix}video_alignment_audit.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["protected_sha256"] != audit["protected_sha256"]:
            raise ValueError("Protected labels or manifest changed; inspect before overwriting audit")
    path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(output / f"{prefix}candidate_preview.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--times", type=float, nargs="+", required=True)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()
    if any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in args.prefix):
        parser.error("prefix must be a simple lowercase filename prefix")
    inspect(args.trial_id, args.times, args.prefix)
