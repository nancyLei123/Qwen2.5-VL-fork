#!/usr/bin/env python3
"""Serve PD-Insighter dashboards with a small annotation API."""

from __future__ import annotations

import argparse
import os
import json
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
from typing import Any
from urllib.parse import parse_qs, urlparse

from build_spoken_action_clip_excel import (
    ACTION_LABELS,
    DEFAULT_WORKBOOK_PATH,
    build_workbook_for_trials,
    discover_spoken_action_trials,
    load_manual_payload,
    process_trial,
    update_manual_annotation,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
QWEN_REPO = WORKSPACE_ROOT / "Qwen2.5-VL-fork"
RUNS_DIR = QWEN_REPO / "runs"
DEFAULT_MODEL = "mlx-community/whisper-tiny"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8766, help="Bind port.")
    return parser.parse_args()


def sanitize(value: Any) -> Any:
    try:
        import pandas as pd

        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except Exception:
        pass
    if isinstance(value, dict):
        return {key: sanitize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def get_trial_api_payload(trial_id: str) -> dict[str, Any]:
    sheets = process_trial(
        trial_id=trial_id,
        model=DEFAULT_MODEL,
        force_transcribe=False,
        min_stray_run_count=1,
        max_stray_run_duration_s=1.25,
    )
    meta_df = sheets[f"{trial_id}_meta"]
    clips_df = sheets[f"{trial_id}_clips"]
    meta_row = sanitize(meta_df.iloc[0].to_dict()) if not meta_df.empty else {}
    manual_payload = load_manual_payload(trial_id)
    return {
        "trial_id": trial_id,
        "action_options": list(ACTION_LABELS),
        "annotations": sanitize(manual_payload.get("annotations", {})),
        "summary": {
            "manual_override_count": meta_row.get("manual_override_count", 0),
            "manual_transition_override_count": meta_row.get("manual_transition_override_count", 0),
            "comparable_clip_count": meta_row.get("comparable_clip_count", 0),
            "stable_only_comparable_clip_count": meta_row.get("stable_only_comparable_clip_count", 0),
            "transition_clip_comparable_count": meta_row.get("transition_clip_comparable_count", 0),
            "strict_accuracy": meta_row.get("strict_accuracy"),
            "transition_aware_accuracy": meta_row.get("transition_aware_accuracy"),
            "component_transition_aware_accuracy": meta_row.get("component_transition_aware_accuracy"),
            "stable_only_accuracy": meta_row.get("stable_only_accuracy"),
            "transition_clip_accuracy": meta_row.get("transition_clip_accuracy"),
            "clip_count": meta_row.get("clip_count", len(clips_df)),
            "workbook_path": str(DEFAULT_WORKBOOK_PATH),
        },
    }


def load_session_manifest(trial_id: str) -> dict[str, Any]:
    manifest_path = RUNS_DIR / trial_id / "session_dashboard_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing session dashboard manifest for {trial_id}")
    return json.loads(manifest_path.read_text())


def parkinson_notes_path(trial_id: str) -> Path:
    return RUNS_DIR / trial_id / "parkinson_motion_review" / "manual_notes.json"


def parkinson_manual_segments_path(trial_id: str) -> Path:
    return RUNS_DIR / trial_id / "parkinson_motion_review" / "manual_segments.json"


def parkinson_freezing_segments_path(trial_id: str) -> Path:
    return RUNS_DIR / trial_id / "parkinson_motion_review" / "manual_freezing_segments.json"


def session_manual_segments_path(trial_id: str) -> Path:
    return RUNS_DIR / trial_id / "manual_labels" / "session_manual_segments.json"


def load_parkinson_notes(trial_id: str) -> dict[str, Any]:
    notes_path = parkinson_notes_path(trial_id)
    if not notes_path.exists():
        return {"trial_id": trial_id, "notes": {}}
    payload = json.loads(notes_path.read_text())
    if not isinstance(payload, dict):
        return {"trial_id": trial_id, "notes": {}}
    payload.setdefault("trial_id", trial_id)
    payload.setdefault("notes", {})
    return payload


def save_parkinson_note(body: dict[str, Any]) -> dict[str, Any]:
    trial_id = str(body.get("trial_id", "")).strip()
    note_key = str(body.get("note_key", "")).strip()
    if not trial_id:
        raise ValueError("Missing trial_id")
    if not note_key:
        raise ValueError("Missing note_key")

    payload = load_parkinson_notes(trial_id)
    notes = payload.setdefault("notes", {})
    notes[note_key] = {
        "note_key": note_key,
        "segment_id": str(body.get("segment_id", "")).strip(),
        "smoothing_key": str(body.get("smoothing_key", "")).strip(),
        "start_s": body.get("start_s"),
        "end_s": body.get("end_s"),
        "detected_motion": str(body.get("detected_motion", "")).strip(),
        "manual_label": str(body.get("manual_label", "")).strip(),
        "base_motion": str(body.get("base_motion", "")).strip(),
        "clinical_marker": str(body.get("clinical_marker", "")).strip(),
        "needs_review": bool(body.get("needs_review", False)),
        "note": str(body.get("note", "")),
    }
    notes_path = parkinson_notes_path(trial_id)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(json.dumps(payload, indent=2))
    return payload


def load_parkinson_manual_segments(trial_id: str) -> dict[str, Any]:
    segments_path = parkinson_manual_segments_path(trial_id)
    if not segments_path.exists():
        return {"trial_id": trial_id, "version": 1, "segments": []}
    payload = json.loads(segments_path.read_text())
    if not isinstance(payload, dict):
        return {"trial_id": trial_id, "version": 1, "segments": []}
    payload.setdefault("trial_id", trial_id)
    payload.setdefault("version", 1)
    payload.setdefault("segments", [])
    if not isinstance(payload["segments"], list):
        payload["segments"] = []
    return payload


def _coerce_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if not (parsed == parsed) or parsed in (float("inf"), float("-inf")):
        raise ValueError(f"{field_name} must be finite")
    return parsed


def save_parkinson_manual_segments(body: dict[str, Any]) -> dict[str, Any]:
    trial_id = str(body.get("trial_id", "")).strip()
    if not trial_id:
        raise ValueError("Missing trial_id")
    raw_segments = body.get("segments", [])
    if not isinstance(raw_segments, list):
        raise ValueError("segments must be a list")

    valid_labels = {"sitting", "standing", "walking", "upstairs", "downstairs"}
    duration_s = None
    try:
        duration_s = _coerce_float(load_session_manifest(trial_id).get("session_duration_s"), "duration_s")
    except Exception:
        duration_s = None

    cleaned_segments: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            raise ValueError(f"segment {index + 1} must be an object")
        segment_id = str(raw.get("id") or f"manual_{index + 1:04d}").strip()
        start_s = _coerce_float(raw.get("start_s"), "start_s")
        end_s = _coerce_float(raw.get("end_s"), "end_s")
        if start_s < 0:
            raise ValueError("start_s must be >= 0")
        if duration_s is not None and end_s > duration_s + 0.25:
            raise ValueError(f"end_s must be within the session duration ({duration_s:.1f}s)")
        if end_s <= start_s:
            raise ValueError("end_s must be greater than start_s")
        motion_label = str(raw.get("motion_label", "")).strip().lower()
        if motion_label not in valid_labels:
            raise ValueError(f"motion_label must be one of: {', '.join(sorted(valid_labels))}")
        confidence = str(raw.get("confidence", "medium")).strip().lower()
        if confidence not in {"high", "medium", "low", "uncertain"}:
            confidence = "medium"
        cleaned_segments.append(
            {
                "id": segment_id,
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "motion_label": motion_label,
                "confidence": confidence,
                "note": str(raw.get("note", "")),
                "source": "manual",
                "created_at": str(raw.get("created_at") or now),
                "updated_at": now,
            }
        )

    cleaned_segments.sort(key=lambda row: (row["start_s"], row["end_s"], row["id"]))
    non_overlapping_segments: list[dict[str, Any]] = []
    for segment in cleaned_segments:
        if non_overlapping_segments:
            previous_end = float(non_overlapping_segments[-1]["end_s"])
            if float(segment["start_s"]) < previous_end:
                segment["start_s"] = round(previous_end, 3)
        if float(segment["end_s"]) - float(segment["start_s"]) >= 0.1:
            non_overlapping_segments.append(segment)
    cleaned_segments = non_overlapping_segments
    payload = {
        "trial_id": trial_id,
        "version": 1,
        "updated_at": now,
        "segments": cleaned_segments,
    }
    segments_path = parkinson_manual_segments_path(trial_id)
    segments_path.parent.mkdir(parents=True, exist_ok=True)
    segments_path.write_text(json.dumps(payload, indent=2))
    return payload


def load_session_manual_segments(trial_id: str) -> dict[str, Any]:
    segments_path = session_manual_segments_path(trial_id)
    if not segments_path.exists():
        return {"trial_id": trial_id, "version": 1, "segments": []}
    payload = json.loads(segments_path.read_text())
    if not isinstance(payload, dict):
        return {"trial_id": trial_id, "version": 1, "segments": []}
    payload.setdefault("trial_id", trial_id)
    payload.setdefault("version", 1)
    payload.setdefault("segments", [])
    if not isinstance(payload["segments"], list):
        payload["segments"] = []
    return payload


def save_session_manual_segments(body: dict[str, Any]) -> dict[str, Any]:
    trial_id = str(body.get("trial_id", "")).strip()
    if not trial_id:
        raise ValueError("Missing trial_id")
    raw_segments = body.get("segments", [])
    if not isinstance(raw_segments, list):
        raise ValueError("segments must be a list")

    valid_labels = {"sitting", "standing", "walking", "upstairs", "downstairs"}
    duration_s = None
    try:
        duration_s = _coerce_float(load_session_manifest(trial_id).get("session_duration_s"), "duration_s")
    except Exception:
        duration_s = None

    now = datetime.now(timezone.utc).isoformat()
    cleaned_segments: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            raise ValueError(f"segment {index + 1} must be an object")
        start_s = _coerce_float(raw.get("start_s"), "start_s")
        end_s = _coerce_float(raw.get("end_s"), "end_s")
        if start_s < 0:
            raise ValueError("start_s must be >= 0")
        if duration_s is not None and end_s > duration_s + 0.25:
            raise ValueError(f"end_s must be within the session duration ({duration_s:.1f}s)")
        if end_s <= start_s:
            raise ValueError("end_s must be greater than start_s")
        motion_label = str(raw.get("motion_label", "")).strip().lower()
        if motion_label not in valid_labels:
            raise ValueError(f"motion_label must be one of: {', '.join(sorted(valid_labels))}")
        confidence = str(raw.get("confidence", "medium")).strip().lower()
        if confidence not in {"high", "medium", "low", "uncertain"}:
            confidence = "medium"
        cleaned_segments.append(
            {
                "id": str(raw.get("id") or f"session_manual_{index + 1:04d}").strip(),
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "motion_label": motion_label,
                "confidence": confidence,
                "note": str(raw.get("note", "")),
                "source": "manual_session",
                "created_at": str(raw.get("created_at") or now),
                "updated_at": now,
            }
        )

    cleaned_segments.sort(key=lambda row: (row["start_s"], row["end_s"], row["id"]))
    non_overlapping_segments: list[dict[str, Any]] = []
    for segment in cleaned_segments:
        if non_overlapping_segments:
            previous_end = float(non_overlapping_segments[-1]["end_s"])
            if float(segment["start_s"]) < previous_end:
                segment["start_s"] = round(previous_end, 3)
        if float(segment["end_s"]) - float(segment["start_s"]) >= 0.1:
            non_overlapping_segments.append(segment)

    payload = {
        "trial_id": trial_id,
        "version": 1,
        "updated_at": now,
        "segments": non_overlapping_segments,
    }
    segments_path = session_manual_segments_path(trial_id)
    segments_path.parent.mkdir(parents=True, exist_ok=True)
    segments_path.write_text(json.dumps(payload, indent=2))
    return payload


def load_parkinson_freezing_segments(trial_id: str) -> dict[str, Any]:
    segments_path = parkinson_freezing_segments_path(trial_id)
    if not segments_path.exists():
        return {"trial_id": trial_id, "version": 1, "segments": []}
    payload = json.loads(segments_path.read_text())
    if not isinstance(payload, dict):
        return {"trial_id": trial_id, "version": 1, "segments": []}
    payload.setdefault("trial_id", trial_id)
    payload.setdefault("version", 1)
    payload.setdefault("segments", [])
    if not isinstance(payload["segments"], list):
        payload["segments"] = []
    return payload


def save_parkinson_freezing_segments(body: dict[str, Any]) -> dict[str, Any]:
    trial_id = str(body.get("trial_id", "")).strip()
    if not trial_id:
        raise ValueError("Missing trial_id")
    raw_segments = body.get("segments", [])
    if not isinstance(raw_segments, list):
        raise ValueError("segments must be a list")

    valid_markers = {"freezing", "hesitation", "shuffling", "festinating", "cadence_irregularity"}
    duration_s = None
    try:
        duration_s = _coerce_float(load_session_manifest(trial_id).get("session_duration_s"), "duration_s")
    except Exception:
        duration_s = None

    now = datetime.now(timezone.utc).isoformat()
    cleaned_segments: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            raise ValueError(f"segment {index + 1} must be an object")
        start_s = _coerce_float(raw.get("start_s"), "start_s")
        end_s = _coerce_float(raw.get("end_s"), "end_s")
        if start_s < 0:
            raise ValueError("start_s must be >= 0")
        if duration_s is not None and end_s > duration_s + 0.25:
            raise ValueError(f"end_s must be within the session duration ({duration_s:.1f}s)")
        if end_s <= start_s:
            raise ValueError("end_s must be greater than start_s")
        marker_label = str(raw.get("marker_label", "freezing")).strip().lower()
        if marker_label not in valid_markers:
            raise ValueError(f"marker_label must be one of: {', '.join(sorted(valid_markers))}")
        confidence = str(raw.get("confidence", "medium")).strip().lower()
        if confidence not in {"high", "medium", "low", "uncertain"}:
            confidence = "medium"
        cleaned_segments.append(
            {
                "id": str(raw.get("id") or f"freezing_{index + 1:04d}").strip(),
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "marker_label": marker_label,
                "confidence": confidence,
                "note": str(raw.get("note", "")),
                "source": "manual",
                "created_at": str(raw.get("created_at") or now),
                "updated_at": now,
            }
        )

    cleaned_segments.sort(key=lambda row: (row["start_s"], row["end_s"], row["id"]))
    payload = {
        "trial_id": trial_id,
        "version": 1,
        "updated_at": now,
        "segments": cleaned_segments,
    }
    segments_path = parkinson_freezing_segments_path(trial_id)
    segments_path.parent.mkdir(parents=True, exist_ok=True)
    segments_path.write_text(json.dumps(payload, indent=2))
    return payload


def parse_frame_offsets_param(raw_value: str | None) -> list[int]:
    if not raw_value:
        return [-1, 0, 1]
    offsets: list[int] = []
    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        offsets.append(int(part))
    if not offsets:
        return [-1, 0, 1]
    return offsets


def ensure_frame_image(
    *,
    video_path: Path,
    cache_dir: Path,
    frame_idx: int,
) -> Path:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - environment defensive
        raise RuntimeError(f"OpenCV is required for frame extraction: {exc}") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"frame_{frame_idx:06d}.jpg"
    if output_path.exists() and output_path.stat().st_mtime >= video_path.stat().st_mtime:
        return output_path

    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to extract frame {frame_idx} from {video_path.name}")
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Failed to save frame image {output_path.name}")
    finally:
        capture.release()

    return output_path


def workspace_relative_url(path: Path) -> str:
    relative = path.relative_to(WORKSPACE_ROOT).as_posix()
    version = int(path.stat().st_mtime)
    return f"/{relative}?v={version}"


def build_frame_context_payload(
    *,
    trial_id: str,
    center_time_s: float,
    offsets: list[int],
) -> dict[str, Any]:
    manifest = load_session_manifest(trial_id)
    fps = float(manifest.get("session_fps") or 30.0)
    frame_duration_s = float(manifest.get("frame_duration_s") or (1.0 / fps if fps else 0.0))
    session_duration_s = float(manifest.get("session_duration_s") or 0.0)
    total_frames = int(manifest.get("session_frame_count") or round(session_duration_s * fps))
    if total_frames <= 0:
        raise ValueError(f"Invalid frame count for {trial_id}")

    center_time_s = max(0.0, min(float(center_time_s), session_duration_s))
    center_frame_idx = int(round(center_time_s * fps))
    center_frame_idx = max(0, min(center_frame_idx, total_frames - 1))

    run_dir = RUNS_DIR / trial_id
    downward_video = run_dir / "downward_master_session" / "downward_session_video.mp4"
    aria_video = run_dir / "downward_master_session" / "aria_session_video.mp4"
    cache_root = run_dir / "frame_context_cache"

    frames: list[dict[str, Any]] = []
    for offset_frames in offsets:
        frame_idx = max(0, min(center_frame_idx + int(offset_frames), total_frames - 1))
        frame_time_s = frame_idx / fps if fps else 0.0
        downward_frame = ensure_frame_image(
            video_path=downward_video,
            cache_dir=cache_root / "downward",
            frame_idx=frame_idx,
        ) if manifest.get("media_mode") != "aria_only" else None
        aria_frame = ensure_frame_image(
            video_path=aria_video,
            cache_dir=cache_root / "aria",
            frame_idx=frame_idx,
        )
        frames.append(
            {
                "offset_frames": int(offset_frames),
                "frame_idx": frame_idx,
                "time_s": frame_time_s,
                "downward_url": workspace_relative_url(downward_frame) if downward_frame else None,
                "aria_url": workspace_relative_url(aria_frame),
            }
        )

    return {
        "trial_id": trial_id,
        "session_fps": fps,
        "frame_duration_s": frame_duration_s,
        "center_time_s": center_time_s,
        "center_frame_idx": center_frame_idx,
        "frames": frames,
    }


class PDInsighterHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._range: tuple[int, int] | None = None
        super().__init__(*args, directory=str(WORKSPACE_ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(sanitize(payload)).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _parse_range_header(self, file_size: int) -> tuple[int, int] | None:
        range_header = self.headers.get("Range")
        if not range_header or not range_header.startswith("bytes="):
            return None
        try:
            range_spec = range_header.split("=", 1)[1].strip()
            start_text, end_text = range_spec.split("-", 1)
            if start_text == "":
                suffix_length = int(end_text)
                if suffix_length <= 0:
                    return None
                start = max(file_size - suffix_length, 0)
                end = file_size - 1
            else:
                start = int(start_text)
                end = file_size - 1 if end_text == "" else int(end_text)
            if start < 0 or end < start or start >= file_size:
                return None
            end = min(end, file_size - 1)
            return start, end
        except Exception:
            return None

    def send_head(self):  # type: ignore[override]
        parsed = urlparse(self.path)
        path = self.translate_path(parsed.path)
        if os.path.isdir(path):
            self._range = None
            return super().send_head()

        ctype = self.guess_type(path)
        if path.endswith("/"):
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        try:
            file_obj = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        try:
            fs = os.fstat(file_obj.fileno())
            file_size = fs.st_size
            self._range = self._parse_range_header(file_size)

            if self._range is not None:
                start, end = self._range
                content_length = end - start + 1
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(content_length))
            else:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Length", str(file_size))

            self.send_header("Content-type", ctype)
            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return file_obj
        except Exception:
            file_obj.close()
            raise

    def copyfile(self, source, outputfile) -> None:  # type: ignore[override]
        if self._range is None:
            shutil.copyfileobj(source, outputfile)
            return
        start, end = self._range
        remaining = end - start + 1
        source.seek(start)
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/manual-labels":
            query = parse_qs(parsed.query)
            trial_id = query.get("trial_id", [None])[0]
            if not trial_id:
                self._send_json({"error": "Missing trial_id"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                payload = get_trial_api_payload(trial_id)
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(payload)
            return
        if parsed.path == "/api/parkinson-notes":
            query = parse_qs(parsed.query)
            trial_id = query.get("trial_id", [None])[0]
            if not trial_id:
                self._send_json({"error": "Missing trial_id"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                payload = load_parkinson_notes(str(trial_id))
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(payload)
            return
        if parsed.path == "/api/parkinson-manual-segments":
            query = parse_qs(parsed.query)
            trial_id = query.get("trial_id", [None])[0]
            if not trial_id:
                self._send_json({"error": "Missing trial_id"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                payload = load_parkinson_manual_segments(str(trial_id))
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(payload)
            return
        if parsed.path == "/api/session-manual-segments":
            query = parse_qs(parsed.query)
            trial_id = query.get("trial_id", [None])[0]
            if not trial_id:
                self._send_json({"error": "Missing trial_id"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                payload = load_session_manual_segments(str(trial_id))
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(payload)
            return
        if parsed.path == "/api/parkinson-freezing-segments":
            query = parse_qs(parsed.query)
            trial_id = query.get("trial_id", [None])[0]
            if not trial_id:
                self._send_json({"error": "Missing trial_id"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                payload = load_parkinson_freezing_segments(str(trial_id))
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(payload)
            return
        if parsed.path == "/api/frame-context":
            query = parse_qs(parsed.query)
            trial_id = query.get("trial_id", [None])[0]
            center_time_raw = query.get("center_time_s", [None])[0]
            if not trial_id or center_time_raw is None:
                self._send_json({"error": "Missing trial_id or center_time_s"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                payload = build_frame_context_payload(
                    trial_id=str(trial_id),
                    center_time_s=float(center_time_raw),
                    offsets=parse_frame_offsets_param(query.get("offsets", [None])[0]),
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json(payload)
            return
        super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/manual-labels":
            self.send_error(HTTPStatus.METHOD_NOT_ALLOWED, "HEAD not supported for this endpoint")
            return
        super().do_HEAD()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/manual-labels":
            try:
                body = self._read_json_body()
                trial_id = str(body.get("trial_id", "")).strip()
                clip_idx = int(body.get("clip_idx"))
                mode = body.get("mode")
                actions = body.get("actions") or []
                if not trial_id:
                    raise ValueError("Missing trial_id")
                if not isinstance(actions, list):
                    raise ValueError("actions must be a list")

                update_manual_annotation(trial_id, clip_idx, mode, actions)
                workbook_path = build_workbook_for_trials(discover_spoken_action_trials(), DEFAULT_WORKBOOK_PATH)
                payload = get_trial_api_payload(trial_id)
                payload["saved"] = True
                payload["summary"]["workbook_path"] = str(workbook_path)
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return

            self._send_json(payload)
            return
        if parsed.path == "/api/parkinson-notes":
            try:
                payload = save_parkinson_note(self._read_json_body())
                payload["saved"] = True
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json(payload)
            return
        if parsed.path == "/api/parkinson-manual-segments":
            try:
                payload = save_parkinson_manual_segments(self._read_json_body())
                payload["saved"] = True
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json(payload)
            return
        if parsed.path == "/api/session-manual-segments":
            try:
                payload = save_session_manual_segments(self._read_json_body())
                payload["saved"] = True
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json(payload)
            return
        if parsed.path == "/api/parkinson-freezing-segments":
            try:
                payload = save_parkinson_freezing_segments(self._read_json_body())
                payload["saved"] = True
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json(payload)
            return
        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), PDInsighterHandler)
    print(f"Serving PD-Insighter with annotation API at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
