"""Storage and validation for manual symptom annotations.

Two kinds of files, both separate from the existing activity labels and Parkinson Review notes:

``runs/symptom_annotation_labels.json``
    The label catalogue shared by all trials. Labels have a stable ``id``; the visible ``name``
    and ``color`` can change, and labels are archived instead of deleted, so saved annotations
    never lose their label.

``runs/<trial>/symptom_annotations/annotations.json``
    One trial's annotations on the dashboard session clock (seconds from the start of the
    trial's session videos, the same clock as Full Session and Manual Labels).

Every save carries the ``revision`` it was based on. A save based on an older revision is
rejected (``ConflictError``) instead of overwriting newer work. Files are written atomically
(temporary file, fsync, rename) and the previous version is kept as ``*.bak``; if the main file
is unreadable, the backup is loaded and the response says so.

Automatic predictions live in ``runs/<trial>/symptom_annotations/detections/`` and review
decisions in ``detection_reviews.json`` (see ``symptom_detection_store.py``); they are never
written into ``annotations.json``. An annotation created by accepting a prediction is still a
manual annotation (``source: manual``); it only carries an ``origin`` that names the prediction.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LABELS_SCHEMA = "pd-insighter/symptom-labels@1"
ANNOTATIONS_SCHEMA = "pd-insighter/symptom-annotations@1"
LABELS_FILENAME = "symptom_annotation_labels.json"
ANNOTATIONS_DIRNAME = "symptom_annotations"
ANNOTATIONS_FILENAME = "annotations.json"

CONFIDENCE_VALUES = ("high", "medium", "low")
REVIEW_STATUS_VALUES = ("confirmed", "needs_review")
SOURCE_MANUAL = "manual"
DURATION_TOLERANCE_S = 1e-6  # float noise only; boundaries are never moved
MAX_NOTES_LENGTH = 2000
MAX_NAME_LENGTH = 60
MAX_DESCRIPTION_LENGTH = 300
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
COLOR_PATTERN = re.compile(r"^#[0-9a-fA-F]{6}$")
TRIAL_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,80}$")

# Provisional workflow labels until the research team agrees on the symptom categories.
DEFAULT_LABELS = (
    ("no_target_symptom", "No Target Symptom", "#5b8f6f",
     "Reviewed; the target symptom was not observed."),
    ("simulated_symptom", "Simulated Symptom", "#c4513d",
     "Reviewed; the defined simulated movement was observed."),
    ("uncertain", "Uncertain", "#d6a21e", "Reviewed; the evidence does not support a confident decision."),
    ("not_observable", "Not Observable", "#7b8794", "The relevant movement cannot be seen in the recording."),
)


class ValidationError(ValueError):
    """The submitted document is malformed; nothing was written."""


class ConflictError(RuntimeError):
    """The file changed since the client loaded it; nothing was written."""

    def __init__(self, message: str, current: dict[str, Any]):
        super().__init__(message)
        self.current = current


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def labels_path(runs_dir: Path) -> Path:
    return runs_dir / LABELS_FILENAME


def annotations_path(runs_dir: Path, trial_id: str) -> Path:
    return runs_dir / check_trial_id(trial_id) / ANNOTATIONS_DIRNAME / ANNOTATIONS_FILENAME


def check_trial_id(trial_id: Any) -> str:
    value = str(trial_id or "").strip()
    if not TRIAL_PATTERN.match(value):
        raise ValidationError("Invalid trial ID.")
    return value


# ---- file handling ------------------------------------------------------------------------------

def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write via a temporary file in the same folder; keep the previous file as *.bak."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            backup = path.with_name(path.name + ".bak")
            os.replace(path, backup)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def read_json_with_backup(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """(payload, warning). Falls back to *.bak when the main file is missing or unreadable."""
    backup = path.with_name(path.name + ".bak")
    for candidate, warning in ((path, None), (backup, "The latest file could not be read; loaded the backup.")):
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload, warning
    if path.exists():
        raise ValidationError(f"{path.name} is unreadable and has no usable backup.")
    return None, None


# ---- labels -------------------------------------------------------------------------------------

def default_labels_document() -> dict[str, Any]:
    stamp = now_iso()
    return {
        "schema": LABELS_SCHEMA, "revision": 0, "updated_at": None,
        "labels": [{"id": label_id, "name": name, "color": color, "description": description,
                    "archived": False, "created_at": stamp, "updated_at": stamp}
                   for label_id, name, color, description in DEFAULT_LABELS],
    }


def _text(value: Any, field: str, limit: int, required: bool = False) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be text.")
    value = " ".join(value.split()) if field == "name" else value.strip()
    if required and not value:
        raise ValidationError(f"{field} is required.")
    if len(value) > limit:
        raise ValidationError(f"{field} is longer than {limit} characters.")
    return value


def validate_labels(raw: Any, previous: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Clean a submitted label list. Labels can be added, renamed, recoloured or archived,
    but a label that was saved before cannot disappear (saved annotations may use it)."""
    if not isinstance(raw, list):
        raise ValidationError("labels must be a list.")
    stamp = now_iso()
    old = {item["id"]: item for item in (previous or {}).get("labels", [])}
    cleaned, ids, names = [], set(), {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValidationError(f"Label {index + 1} must be an object.")
        label_id = str(item.get("id", "")).strip()
        if not ID_PATTERN.match(label_id):
            raise ValidationError(f"Label {index + 1} has an invalid ID.")
        if label_id in ids:
            raise ValidationError(f"Duplicate label ID {label_id}.")
        name = _text(item.get("name"), "name", MAX_NAME_LENGTH, required=True)
        key = name.casefold()
        if key in names:
            raise ValidationError(f'Two labels are named "{name}". Use different names.')
        color = str(item.get("color", "")).strip()
        if not COLOR_PATTERN.match(color):
            raise ValidationError(f'Label "{name}" needs a color like #3a7d5c.')
        archived = item.get("archived", False)
        if not isinstance(archived, bool):
            raise ValidationError(f'Label "{name}": archived must be true or false.')
        description = _text(item.get("description"), "description", MAX_DESCRIPTION_LENGTH)
        before = old.get(label_id)
        changed = before is None or any(before.get(k) != v for k, v in
                                        (("name", name), ("color", color.lower()), ("description", description),
                                         ("archived", archived)))
        cleaned.append({
            "id": label_id, "name": name, "color": color.lower(), "description": description, "archived": archived,
            "created_at": (before or {}).get("created_at") or stamp,
            "updated_at": stamp if changed else (before or {}).get("updated_at", stamp),
        })
        ids.add(label_id)
        names[key] = label_id
    missing = sorted(set(old) - ids)
    if missing:
        raise ValidationError(f"Labels cannot be removed, only archived: {', '.join(missing)}.")
    return cleaned


def load_labels(runs_dir: Path) -> dict[str, Any]:
    payload, warning = read_json_with_backup(labels_path(runs_dir))
    if payload is None:
        return default_labels_document()
    if payload.get("schema") != LABELS_SCHEMA:
        raise ValidationError("Unsupported label catalogue format.")
    if warning:
        payload["warning"] = warning
    return payload


def save_labels(runs_dir: Path, body: dict[str, Any]) -> dict[str, Any]:
    current = load_labels(runs_dir)
    base = body.get("base_revision")
    if base != current["revision"]:
        raise ConflictError("Labels were changed elsewhere. Reload to see the latest list.", current)
    previous = current if current["revision"] > 0 else None
    document = {
        "schema": LABELS_SCHEMA, "revision": current["revision"] + 1, "updated_at": now_iso(),
        "labels": validate_labels(body.get("labels"), previous),
    }
    atomic_write_json(labels_path(runs_dir), document)
    return document


# ---- annotations --------------------------------------------------------------------------------

def _seconds(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a number of seconds.")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{field} must be a finite number.")
    return number


def _origin(value: Any, where: str) -> dict[str, Any] | None:
    """Optional link to the automatic prediction an annotation was accepted from."""
    if value in (None, ""):
        return None
    if not isinstance(value, dict) or value.get("type") != "detection":
        raise ValidationError(f"{where}: origin must describe a detection.")
    run_id, detection_id = str(value.get("run_id") or ""), str(value.get("detection_id") or "")
    if not 3 <= len(run_id) <= 100 or not 3 <= len(detection_id) <= 120:
        raise ValidationError(f"{where}: origin needs a run ID and a prediction ID.")
    return {"type": "detection", "run_id": run_id, "detection_id": detection_id, "edited": bool(value.get("edited"))}


def validate_annotations(raw: Any, *, trial_id: str, duration_s: float, label_ids: set[str],
                         previous: dict[str, Any] | None = None, annotator: str = "") -> list[dict[str, Any]]:
    """Clean submitted annotations. Times keep their precision (rounded to 1 microsecond);
    overlapping intervals are allowed and kept as they are."""
    if not isinstance(raw, list):
        raise ValidationError("annotations must be a list.")
    stamp = now_iso()
    old = {item["id"]: item for item in (previous or {}).get("annotations", [])}
    cleaned, ids = [], set()
    for index, item in enumerate(raw):
        where = f"Annotation {index + 1}"
        if not isinstance(item, dict):
            raise ValidationError(f"{where} must be an object.")
        annotation_id = str(item.get("id", "")).strip()
        if not ID_PATTERN.match(annotation_id):
            raise ValidationError(f"{where} has an invalid ID.")
        if annotation_id in ids:
            raise ValidationError(f"Duplicate annotation ID {annotation_id}.")
        start = _seconds(item.get("start_s"), f"{where} start time")
        end = _seconds(item.get("end_s"), f"{where} end time")
        if start < 0:
            raise ValidationError(f"{where}: start time cannot be negative.")
        if end <= start:
            raise ValidationError(f"{where}: end time must be after the start time.")
        if end > duration_s + DURATION_TOLERANCE_S:
            raise ValidationError(f"{where}: end time is after the end of the session ({duration_s:.2f} s).")
        label_id = str(item.get("label_id", "")).strip()
        if label_id not in label_ids:
            raise ValidationError(f"{where} uses an unknown label.")
        confidence = str(item.get("confidence", "medium")).strip().lower()
        if confidence not in CONFIDENCE_VALUES:
            raise ValidationError(f"{where}: confidence must be high, medium or low.")
        status = str(item.get("review_status", "confirmed")).strip().lower()
        if status not in REVIEW_STATUS_VALUES:
            raise ValidationError(f"{where}: review status must be confirmed or needs_review.")
        notes = _text(item.get("notes"), "notes", MAX_NOTES_LENGTH)
        before = old.get(annotation_id)
        entry = {
            "id": annotation_id, "label_id": label_id,
            "start_s": round(start, 6), "end_s": round(end, 6),
            "confidence": confidence, "review_status": status, "notes": notes,
            "annotator": _text(item.get("annotator") or (before or {}).get("annotator") or annotator,
                               "annotator", 60),
            "source": SOURCE_MANUAL,
        }
        origin = _origin(item.get("origin") if "origin" in item else (before or {}).get("origin"), where)
        if origin:
            entry["origin"] = origin
        changed = before is None or any(before.get(k) != entry[k] for k in
                                        ("label_id", "start_s", "end_s", "confidence", "review_status", "notes"))
        entry["created_at"] = (before or {}).get("created_at") or stamp
        entry["updated_at"] = stamp if changed else (before or {}).get("updated_at", stamp)
        cleaned.append(entry)
        ids.add(annotation_id)
    cleaned.sort(key=lambda a: (a["start_s"], a["end_s"], a["id"]))
    return cleaned


def empty_annotations_document(trial_id: str, duration_s: float) -> dict[str, Any]:
    return {"schema": ANNOTATIONS_SCHEMA, "trial_id": trial_id, "revision": 0, "updated_at": None,
            "session_clock": {"name": "dashboard session time", "units": "s", "duration_s": duration_s},
            "annotations": []}


def load_annotations(runs_dir: Path, trial_id: str, duration_s: float) -> dict[str, Any]:
    trial_id = check_trial_id(trial_id)
    payload, warning = read_json_with_backup(annotations_path(runs_dir, trial_id))
    if payload is None:
        return empty_annotations_document(trial_id, duration_s)
    if payload.get("schema") != ANNOTATIONS_SCHEMA:
        raise ValidationError("Unsupported annotation file format.")
    if payload.get("trial_id") != trial_id:
        raise ValidationError("The annotation file belongs to a different trial.")
    if warning:
        payload["warning"] = warning
    return payload


def save_annotations(runs_dir: Path, body: dict[str, Any], *, duration_s: float, labels: dict[str, Any]) -> dict[str, Any]:
    trial_id = check_trial_id(body.get("trial_id"))
    if not (isinstance(duration_s, (int, float)) and math.isfinite(duration_s) and duration_s > 0):
        raise ValidationError("The session duration is unknown, so annotations cannot be checked.")
    current = load_annotations(runs_dir, trial_id, duration_s)
    if body.get("base_revision") != current["revision"]:
        raise ConflictError("Annotations were changed elsewhere. Reload to see the latest version.", current)
    annotations = validate_annotations(
        body.get("annotations"), trial_id=trial_id, duration_s=float(duration_s),
        label_ids={label["id"] for label in labels.get("labels", [])},
        previous=current, annotator=str(body.get("annotator") or ""),
    )
    document = {
        "schema": ANNOTATIONS_SCHEMA, "trial_id": trial_id, "revision": current["revision"] + 1,
        "updated_at": now_iso(),
        "session_clock": {"name": "dashboard session time", "units": "s", "duration_s": float(duration_s)},
        "annotations": annotations,
    }
    atomic_write_json(annotations_path(runs_dir, trial_id), document)
    return document
