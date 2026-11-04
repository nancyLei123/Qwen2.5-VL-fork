#!/usr/bin/env python3
"""Build a sanitized, static PD Insighter demo for representative trials."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from build_downward_dual_view_dashboards import (
    build_dashboard_html,
    build_stride_review_html,
    build_transition_review_html,
)
from build_full_session_manual_labeling import (
    build_html as build_manual_labeling_html,
    load_payload as load_manual_labeling_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "pd_insighter_demo"
TRIALS = ("Half_hour", "60minutes_1", "Ntrial3")
PAGE_NAMES = (
    "index.html",
    "pd_insighter_manual_labeling.html",
    "pd_insighter_stride_review.html",
    "pd_insighter_transition_review.html",
)
GITHUB_FILE_LIMIT_BYTES = 100 * 1024 * 1024


MEDIA_PROFILES: dict[tuple[str, str], dict[str, Any]] = {
    ("Half_hour", "aria"): {
        "filter": "fps=15,scale=640:640:force_original_aspect_ratio=decrease",
        "video_bitrate": "280k",
        "maxrate": "360k",
        "bufsize": "720k",
        "audio_bitrate": "32k",
    },
    ("Half_hour", "downward"): {
        "filter": "fps=15,scale=854:-2",
        "video_bitrate": "340k",
        "maxrate": "440k",
        "bufsize": "880k",
        "audio_bitrate": None,
    },
    ("Ntrial3", "aria"): {
        "filter": "fps=15,scale=640:640:force_original_aspect_ratio=decrease",
        "video_bitrate": "650k",
        "maxrate": "850k",
        "bufsize": "1700k",
        "audio_bitrate": "48k",
    },
    ("Ntrial3", "downward"): {
        "filter": "fps=15,scale=960:-2",
        "video_bitrate": "900k",
        "maxrate": "1200k",
        "bufsize": "2400k",
        "audio_bitrate": None,
    },
    ("60minutes_1", "aria"): {
        "filter": "fps=12,scale=576:576:force_original_aspect_ratio=decrease",
        "video_bitrate": "180k",
        "maxrate": "240k",
        "bufsize": "480k",
        "audio_bitrate": "24k",
    },
    ("60minutes_1", "downward"): {
        "filter": "fps=12,scale=768:-2",
        "video_bitrate": "210k",
        "maxrate": "280k",
        "bufsize": "560k",
        "audio_bitrate": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--transcode", action="store_true", help="Build lightweight web videos.")
    parser.add_argument("--ffmpeg", type=Path, help="Path to an ffmpeg executable.")
    return parser.parse_args()


def sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, str):
        home = str(Path.home())
        return value.replace(home, "[local source omitted]")
    return value


def trial_options(current_trial: str, page_name: str = "index.html") -> list[dict[str, str]]:
    items = [
        {
            "trial_id": "__selector__",
            "label": "Demo Home",
            "href": "../index.html",
        }
    ]
    labels = {
        "Half_hour": "Half_hour - long normal trial",
        "60minutes_1": "60minutes_1 - extended normal trial",
        "Ntrial3": "Ntrial3 - short indoor trial",
    }
    for trial_id in TRIALS:
        href = page_name if trial_id == current_trial else f"../{trial_id}/{page_name}"
        items.append({"trial_id": trial_id, "label": labels[trial_id], "href": href})
    return items


def prepare_dashboard_payload(trial_id: str) -> dict[str, Any]:
    source_path = RUNS_DIR / trial_id / "session_dashboard_manifest.json"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing dashboard manifest: {source_path}")
    payload = sanitize_value(json.loads(source_path.read_text()))
    payload.update(
        {
            "trial_selector_href": "../index.html",
            "full_session_href": "index.html",
            "manual_labeling_href": "pd_insighter_manual_labeling.html",
            "stride_review_href": "pd_insighter_stride_review.html",
            "transition_review_href": "pd_insighter_transition_review.html",
            "downward_session_video_uri": "media/downward.mp4",
            "aria_session_video_uri": "media/aria.mp4",
            "trial_jump_options": trial_options(trial_id),
        }
    )
    return payload


def prepare_manual_payload(trial_id: str) -> dict[str, Any]:
    payload = sanitize_value(load_manual_labeling_payload(trial_id))
    payload.update(
        {
            "trial_selector_href": "../index.html",
            "full_review_href": "index.html",
            "stride_review_href": "pd_insighter_stride_review.html",
            "transition_review_href": "pd_insighter_transition_review.html",
            "downward_session_video_uri": "media/downward.mp4",
            "aria_session_video_uri": "media/aria.mp4",
            "trial_jump_options": trial_options(trial_id, "pd_insighter_manual_labeling.html"),
        }
    )
    return payload


def write_trial_pages(trial_id: str, output_dir: Path) -> None:
    trial_dir = output_dir / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    payload = prepare_dashboard_payload(trial_id)
    pages = {
        "index.html": build_dashboard_html(payload),
        "pd_insighter_stride_review.html": build_stride_review_html(payload),
        "pd_insighter_transition_review.html": build_transition_review_html(payload),
        "pd_insighter_manual_labeling.html": build_manual_labeling_html(
            prepare_manual_payload(trial_id)
        ),
    }
    for name, content in pages.items():
        (trial_dir / name).write_text(content)
    print(f"{trial_id}: wrote {len(pages)} static dashboard pages", flush=True)


def discover_ffmpeg(explicit_path: Path | None) -> Path:
    if explicit_path:
        if not explicit_path.exists():
            raise FileNotFoundError(f"ffmpeg not found: {explicit_path}")
        return explicit_path
    env_path = os.environ.get("PD_INSIDER_FFMPEG")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    system_path = shutil.which("ffmpeg")
    if system_path:
        return Path(system_path)
    try:
        import imageio_ffmpeg

        return Path(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("Pass --ffmpeg or install ffmpeg before transcoding.") from exc


def source_video_path(trial_id: str, view: str) -> Path:
    return RUNS_DIR / trial_id / "downward_master_session" / f"{view}_session_video.mp4"


def transcode_video(ffmpeg: Path, trial_id: str, view: str, output_dir: Path) -> Path:
    source = source_video_path(trial_id, view)
    if not source.exists():
        raise FileNotFoundError(f"Missing source video: {source}")
    destination = output_dir / trial_id / "media" / f"{view}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    profile = copy.deepcopy(MEDIA_PROFILES[(trial_id, view)])
    build_info_path = destination.with_suffix(".build.json")
    build_info = {
        "source_size": source.stat().st_size,
        "source_mtime_ns": source.stat().st_mtime_ns,
        "profile": profile,
    }
    if destination.exists() and build_info_path.exists():
        try:
            if json.loads(build_info_path.read_text()) == build_info:
                print(f"{trial_id} {view}: reusing {destination.name}", flush=True)
                return destination
        except (OSError, json.JSONDecodeError):
            pass

    temporary = destination.with_name(f"{destination.stem}.tmp{destination.suffix}")
    command = [
        str(ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-stats_period",
        "10",
        "-i",
        str(source),
        "-map",
        "0:v:0",
    ]
    if profile["audio_bitrate"]:
        command.extend(["-map", "0:a:0?"])
    command.extend(
        [
            "-vf",
            profile["filter"],
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            profile["video_bitrate"],
            "-maxrate",
            profile["maxrate"],
            "-bufsize",
            profile["bufsize"],
            "-g",
            "30",
            "-keyint_min",
            "30",
            "-sc_threshold",
            "0",
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if profile["audio_bitrate"]:
        command.extend(
            [
                "-c:a",
                "aac",
                "-b:a",
                profile["audio_bitrate"],
                "-ac",
                "1",
                "-ar",
                "24000",
            ]
        )
    else:
        command.append("-an")
    command.extend(
        [
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
    )
    print(f"{trial_id} {view}: transcoding lightweight web video", flush=True)
    subprocess.run(command, check=True)
    if temporary.stat().st_size >= GITHUB_FILE_LIMIT_BYTES:
        raise RuntimeError(
            f"{temporary} is {temporary.stat().st_size / 1024 / 1024:.1f} MiB; "
            "it must remain below GitHub's 100 MiB file limit."
        )
    temporary.replace(destination)
    build_info_path.write_text(json.dumps(build_info, indent=2) + "\n")
    print(
        f"{trial_id} {view}: {destination.stat().st_size / 1024 / 1024:.1f} MiB",
        flush=True,
    )
    return destination


def build_manifest(output_dir: Path) -> None:
    media: list[dict[str, Any]] = []
    for trial_id in TRIALS:
        for view in ("aria", "downward"):
            path = output_dir / trial_id / "media" / f"{view}.mp4"
            media.append(
                {
                    "trial_id": trial_id,
                    "view": view,
                    "path": str(path.relative_to(output_dir)),
                    "size_mib": round(path.stat().st_size / 1024 / 1024, 2)
                    if path.exists()
                    else None,
                }
            )
    payload = {
        "format": "pd-insighter-static-demo-v1",
        "trials": list(TRIALS),
        "pages_per_trial": list(PAGE_NAMES),
        "media": media,
        "contains_raw_vrs": False,
        "contains_raw_mps_csv": False,
    }
    (output_dir / "demo_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ".nojekyll").touch()
    for trial_id in TRIALS:
        write_trial_pages(trial_id, output_dir)
    if args.transcode:
        ffmpeg = discover_ffmpeg(args.ffmpeg)
        for trial_id in TRIALS:
            for view in ("aria", "downward"):
                transcode_video(ffmpeg, trial_id, view, output_dir)
    build_manifest(output_dir)
    print(f"Static demo ready at: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
