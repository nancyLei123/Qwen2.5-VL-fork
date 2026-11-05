#!/usr/bin/env python3
"""Build a small offline RoomScan3c result viewer without changing its sources."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

REPO = Path(__file__).resolve().parents[1]
ENV = REPO / "runs/RoomScan3c/environment_reconstruction"
QUALITY = ENV / "quality_20261029"
VENDOR = REPO / "vendor/gaussian-viewer/node_modules"
DOCS = REPO / "docs/room_demo"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def excerpt(data, start_frame, count):
    fps = data["fps"]
    if start_frame < 0 or count < 1 or start_frame + count > len(data["poses"]):
        raise ValueError("Excerpt is outside the exported video")
    result = {key: data[key] for key in ("trial_id", "fps", "bounds", "labels_not_applicable", "scene_assembly_note")}
    for key in ("poses", "recording_times", "pose_status"):
        result[key] = data[key][start_frame:start_frame + count]
    result.update(video_url="aria_excerpt.mp4", video_has_audio=False,
                  video_time_label="90-second excerpt / compressed review copy",
                  duration_s=count / fps, dashboard_offset_s=0, dashboard_duration_s=count / fps,
                  initial_video_time_s=min(20.7, (count - 1) / fps), initial_route_scope="nearby",
                  quality_page_url="METHODS.html")
    result["scenes"] = [{**scene, "ply": "room_model.ply", "details_url": "METHODS.html",
                         "video_start_s": 0, "video_end_s": count / fps} for scene in data["scenes"]]
    if len(result["scenes"]) != 1:
        raise ValueError("This demo expects one preserved joint room model")
    result["provenance"] = {
        "graph_uid": data["provenance"]["graph_uid"],
        "model_sha256": data["provenance"]["model_sha256"],
        "source_trial": "RoomScan3c", "source_video": "aria_room_scan.mp4",
        "source_frame_range_half_open": [start_frame, start_frame + count],
        "source_video_start_s": start_frame / fps,
        "source_video_end_s": (start_frame + count) / fps,
        "frame_count": count,
        "valid_pose_count": sum(p is not None for p in result["poses"]),
        "originals_modified": False,
        "timing_explanation": "Demo MP4 frame i maps to original exported MP4 frame i + "
            f"{start_frame}. Original VRS capture-relative times and calibrated MPS poses are "
            "retained exactly. Only the demo video clock resets to zero. Video is resized to "
            "704 x 704 and compressed, without dropping frames or changing the model. "
            "Unsupported poses remain missing. No action labels exist for this room scan."
    }
    return result


def copy_three(source, destination, visited):
    source = source.resolve()
    root = (VENDOR / "three").resolve()
    if not source.is_relative_to(root):
        raise ValueError("Vendor import escapes three package")
    if source in visited:
        return
    visited.add(source)
    target = destination / "vendor/three" / source.relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    text = source.read_text()
    for relative in re.findall(r"(?:from\s*|import\s*)['\"](\.[^'\"]+)['\"]", text):
        copy_three(source.parent / relative, destination, visited)


def build(output):
    if output.exists() or output.with_suffix(".zip").exists():
        raise FileExistsError("Existing packages are never overwritten; choose a fresh output")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe are needed only on the package-building computer")
    original = json.loads((ENV / "scene_playback_data.json").read_text())
    data = excerpt(original, 11700, 2700)
    model = QUALITY / "trained_30000/splat.ply"
    if digest(model) != original["provenance"]["model_sha256"]:
        raise ValueError("Source model hash differs from playback provenance")
    output.mkdir(parents=True)
    shutil.copy2(model, output / "room_model.ply")
    for name in ("scene_playback.js", "scene_camera_view.mjs", "scene_playback_core.mjs",
                 "scene_playback_quality.mjs", "scene_route_navigation.mjs", "scene_spatial_overview.mjs",
                 "spatial_plotly.min.js", "spatial_overview_data.json"):
        shutil.copy2(ENV / name, output / name)
    player = output / "scene_playback.js"
    player.write_text(player.read_text().replace("/Qwen2.5-VL-fork/vendor/gaussian-viewer/node_modules/", "./vendor/"))
    html = (ENV / "playback.html").read_text()
    html = html.replace("/Qwen2.5-VL-fork/vendor/gaussian-viewer/node_modules/", "./vendor/")
    nav = ('<nav class="card utility-bar" aria-label="PD Insighter demo navigation">'
           '<div class="nav-context"><span class="nav-mark">PD</span><div class="nav-location">'
           '<strong>RoomScan3c</strong><span>Portable room demo</span></div></div>'
           '<div class="nav-tabs"><span class="nav-tab active">Synchronized 3D</span>'
           '<a class="nav-tab" href="METHODS.html">Method &amp; included data</a>'
           '<a class="nav-tab" href="README.md">Quick start</a></div></nav>')
    html = re.sub(r'<nav class="card utility-bar".*?</nav>', lambda _: nav, html, count=1, flags=re.S)
    html = re.sub(r'<script id="shared-simulation-trial-switcher">.*?</script>', '', html, flags=re.S)
    html = html.replace('Synchronized 3D Playback</h1>', 'Room Reconstruction Demo</h1>')
    html = html.replace('<section class="workspace">', '<p class="notice">90-second excerpt: original video 06:30-08:00. '
        'The full room model is unchanged. Video is compressed; no action labels exist for this dedicated scan.</p><section class="workspace">')
    for link in ('quality_20261029/index.html', 'quality_20261029/methodology.md',
                 'quality_comparison_20261022/index.html', 'joint_first15_experiment/comparison.json'):
        html = html.replace(link, 'METHODS.html')
    (output / "index.html").write_text(html)
    save(output / "scene_playback_data.json", data)
    visited = set()
    for name in ("build/three.module.js", "examples/jsm/lines/LineSegments2.js",
                 "examples/jsm/lines/LineSegmentsGeometry.js", "examples/jsm/lines/LineMaterial.js"):
        copy_three(VENDOR / "three" / name, output, visited)
    splat_dir = output / "vendor/@mkkellogg/gaussian-splats-3d/build"
    splat_dir.mkdir(parents=True)
    shutil.copy2(VENDOR / "@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js", splat_dir)
    for name in ("three", "@mkkellogg/gaussian-splats-3d"):
        shutil.copy2(VENDOR / name / "LICENSE", output / "vendor" / name / "LICENSE")
    for name in ("README.md", "start_demo.py", "METHODS.html"):
        shutil.copy2(DOCS / name, output / name)
    shutil.copy2(QUALITY / "validation_comparison_2.jpg", output / "held_out_comparison.jpg")
    run = json.loads((QUALITY / "trained_30000/run.json").read_text())
    run.pop("command")
    save(output / "model_metadata.json", {"training": run, "camera": data["scenes"][0]["camera"],
        "graph_uid": data["provenance"]["graph_uid"], "model_sha256": digest(output / "room_model.ply"),
        "training_inputs_included": False, "static_model": True})
    command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-n", "-ss", "390",
        "-i", str(ENV / "aria_room_scan.mp4"), "-frames:v", "2700", "-map", "0:v:0", "-an",
        "-vf", "scale=704:704", "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p", "-g", "30", "-movflags", "+faststart", str(output / "aria_excerpt.mp4")]
    subprocess.run(command, check=True)
    probe = json.loads(subprocess.check_output([ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_streams", "-of", "json", str(output / "aria_excerpt.mp4")]))["streams"][0]
    if (int(probe["nb_frames"]), probe["avg_frame_rate"], int(probe["width"]), int(probe["height"])) != (2700, "30/1", 704, 704):
        raise ValueError("Video frame count, rate or dimensions do not match the excerpt")
    if abs(float(probe["duration"]) - data["duration_s"]) > .001:
        raise ValueError("Video duration does not match poses")
    for i in range(2700):
        assert data["poses"][i] == original["poses"][11700+i]
        assert data["recording_times"][i] == original["recording_times"][11700+i]
    files = [{"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(output.rglob("*")) if p.is_file()]
    save(output / "PACKAGE_MANIFEST.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Result-viewing demo, not full training reproduction", "trial": "RoomScan3c", "files": files,
        "bytes": sum(item["bytes"] for item in files), "video_frames": 2700, "duration_s": 90,
        "source_frame_range_half_open": [11700,14400], "pose_timestamp_alignment_verified": True,
        "source_model_sha256": data["provenance"]["model_sha256"]})
    target = output.with_suffix(".zip")
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output.parent),
                              compress_type=zipfile.ZIP_STORED if path.suffix in {".mp4", ".jpg"} else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
    report = {"zip": str(target), "bytes": target.stat().st_size, "sha256": digest(target),
              "files": len(files) + 1, "model_preserved": True, "frame_count_verified": True,
              "zip_crc_verified": True, "browser_test": "pending"}
    save(output.parent / (output.name + ".verification.json"), report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output.resolve())
