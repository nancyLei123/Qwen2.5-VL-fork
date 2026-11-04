#!/usr/bin/env python3
"""Connect the preserved RoomScan3c model to the existing route/video player."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from build_aria_scene_playback import publish_ui, three_pose
from reconstruct_dedicated_room import OUTPUT, Reader, inputs, save_json, sha256, verify_stats

ENV = OUTPUT.parent
FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"


def validate_video(path, count, fps):
    info = json.loads(subprocess.check_output([
        FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(path)
    ]))["streams"][0]
    numerator, denominator = map(int, info["avg_frame_rate"].split("/"))
    if int(info["nb_frames"]) != count or abs(numerator / denominator - fps) > 1e-9:
        raise ValueError("Encoded frame count or FPS differs from the exact VRS frame table")
    if abs(float(info["duration"]) - count / fps) > .001:
        raise ValueError("Encoded duration differs from the expected frame table")
    return info


def export():
    manifest = json.loads((OUTPUT / "dataset_manifest.json").read_text())
    verify_stats(manifest["source_stats"])
    video = ENV / "aria_room_scan.mp4"
    payload_path = ENV / "scene_playback_data.json"
    if video.exists() and payload_path.exists():
        data = json.loads(payload_path.read_text())
        validate_video(video, len(data["poses"]), data["fps"])
        if data["provenance"]["graph_uid"] != manifest["world_frame"]["graph_uid"]:
            raise ValueError("Existing playback belongs to a different world frame")
        return data
    if video.exists() or payload_path.exists():
        raise ValueError("Preserving incomplete prior export; inspect it before retrying")
    reader = Reader()
    fps, count = 30., len(reader.timestamps)
    duration = count / fps
    partial = ENV / f"aria_room_scan.partial_{time.time_ns()}.mp4"
    command = [FFMPEG, "-hide_banner", "-loglevel", "warning", "-n",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "1408x1408",
        "-framerate", "30", "-i", "pipe:0", "-an", "-c:v", "h264_videotoolbox",
        "-allow_sw", "0", "-b:v", "8M", "-pix_fmt", "yuv420p", "-g", "30", "-bf", "0",
        "-movflags", "+faststart", str(partial)]
    poses, status, gaps = [], [], []
    selected = {row["source_frame_index"]: row for row in manifest["frames"]["items"]}
    started = time.monotonic()
    with (ENV / "route_video_export.log").open("w") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=log)
        try:
            for index, stamp in enumerate(reader.timestamps):
                image, record = reader.provider.get_image_data_by_index(reader.sid, index)
                if not image.is_valid() or int(record.capture_timestamp_ns) != int(stamp):
                    raise ValueError(f"Frame {index}: missing image or timestamp mismatch")
                rgb = image.to_numpy_array()
                if rgb.shape != (1408, 1408, 3) or rgb.dtype != np.uint8:
                    raise ValueError(f"Unexpected RGB pixel layout at frame {index}: {rgb.shape}")
                process.stdin.write(np.ascontiguousarray(rgb).tobytes())
                try:
                    _, transform, _ = reader.pose(int(stamp))
                    upright = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
                    transform[:3, :3] = transform[:3, :3] @ upright
                    if index in selected and not np.allclose(transform, selected[index]["T_world_camera"], atol=1e-8):
                        raise RuntimeError("Playback pose differs from reconstruction pose")
                    poses.append(np.round(three_pose(transform), 9).tolist())
                    status.append("mps_supported")
                except ValueError as error:
                    poses.append(None)
                    status.append("unavailable")
                    gaps.append({"frame": index, "video_s": index / fps, "capture_ns": int(stamp), "reason": str(error)})
                if (index + 1) % 900 == 0:
                    progress = {"stage": "video_and_calibrated_route_export", "frames_done": index + 1,
                        "frames_total": count, "elapsed_s": time.monotonic() - started}
                    save_json(ENV / "route_export_progress.json", progress)
                    print(f"Export {index + 1}/{count} ({100 * (index + 1) / count:.1f}%)", flush=True)
            process.stdin.close()
            if process.wait():
                raise RuntimeError("Video encoding failed; see route_video_export.log")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise
    probe = validate_video(partial, count, fps)
    verify_stats(manifest["source_stats"])
    partial.rename(video)
    times = (reader.timestamps - reader.origin) / 1e9
    positions = np.asarray([pose[:3] for pose in poses if pose is not None])
    model = OUTPUT / "trained_30000/splat.ply"
    run = json.loads((OUTPUT / "trained_30000/run.json").read_text())
    if sha256(model) != run["model_sha256"]:
        raise ValueError("The preserved trained model changed")
    scene_note = "One static room model jointly trained across this scan (30,000 steps), not separate clips. The same model is reused along the recorded route; availability is not complete surface coverage. Blur, occlusion remnants and unobserved areas remain."
    data = {"trial_id": "RoomScan3c", "video_url": video.name, "video_has_audio": False,
        "video_time_label": "Complete Aria recording / review copy", "fps": fps, "duration_s": duration,
        "recording_times": times.round(9).tolist(), "poses": poses, "pose_status": status,
        "pose_recovery_audit": gaps, "dashboard_offset_s": 0, "dashboard_duration_s": duration,
        "labels_not_applicable": True, "initial_video_time_s": 410.7,
        "initial_route_scope": "nearby", "scene_assembly_note": scene_note,
        "quality_page_url": "quality_20261029/index.html",
        "bounds": [positions.min(axis=0).tolist(), positions.max(axis=0).tolist()],
        "scenes": [{"id": "joint_room_30000", "title": "Joint room / 30,000 steps",
            "start_s": 0., "end_s": float(times[-1]) + 1e-6,
            "video_start_s": 0., "video_end_s": duration,
            "ply": "quality_20261029/trained_30000/splat.ply", "iterations": 30000,
            "camera": manifest["camera"], "details_url": "quality_20261029/index.html",
            "training_start_s": min(r["recording_s"] for r in manifest["frames"]["items"]),
            "training_end_s": max(r["recording_s"] for r in manifest["frames"]["items"])}],
        "provenance": {"vrs": str(inputs()["vrs"]), "trajectory": str(inputs()["trajectory"]),
            "calibration": str(inputs()["calibration"]), "graph_uid": reader.graph,
            "frame_count": count, "valid_pose_count": len(positions),
            "pose_status_counts": dict(Counter(status)), "model_sha256": run["model_sha256"],
            "selected_training_export_poses_verified": len(selected),
            "video_minus_device_at_last_frame_s": float((count - 1) / fps - times[-1]),
            "timing_explanation": "Each MP4 frame maps by index to its exact VRS RGB capture timestamp, then to the calibrated MPS RGB pose with the online time offset. All 44,490 frames are retained. The review MP4 is 30 FPS and displayed upright by a 90-degree rotation. No pose gaps are extrapolated or bridged. This room scan has no action-classification or manual-label track; none is inferred.",
            "video_export": {"command": command, "ffprobe": probe, "elapsed_s": time.monotonic() - started},
            "originals_modified": False, "scene_availability": scene_note}}
    save_json(payload_path, data)
    save_json(ENV / "route_export_progress.json", {"stage": "video_and_route_ready", "state": "complete",
        "frames": count, "supported_poses": len(positions), "unsupported_poses": len(gaps)})
    print(json.dumps({"frames": count, "poses": len(positions), "gaps": len(gaps), "duration_s": duration}, indent=2))
    return data


def publish():
    page = publish_ui("RoomScan3c")
    # The room scan has no motion-analysis pages; do not publish dead navigation links.
    html = page.read_text()
    begin = html.index('<div class="tabs">')
    end = html.index('</div></nav>', begin)
    html = html[:begin] + '<div class="tabs"><a href="../../pd_insighter_trial_selector.html">All trials</a><a class="active" href="playback.html">Route + 3D playback</a><a href="quality_20261029/index.html">Quality comparison</a>' + html[end:]
    html = html.replace('camera direction and actions at the same time.', 'position and viewing direction at the same time.')
    html = html.replace('Scene names describe the original export groups. The action bars above use the current dashboard labels with the correct time offset. Unsupported poses and untrained intervals are explicitly shown as gaps.',
        'This is the dedicated room scan. A single static model is shown along the recorded camera route. No action labels have been generated for it. Unsupported camera poses are explicitly marked.')
    html = html.replace('../../../ARIA_ENVIRONMENT_RECONSTRUCTION.md', 'quality_20261029/methodology.md')
    page.write_text(html)
    (ENV / "index.html").write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=playback.html"><a href="playback.html">Room scan: route and synchronized 3D playback</a>')
    print(page)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("export", "publish"), required=True)
    args = parser.parse_args()
    {"export": export, "publish": publish}[args.stage]()
