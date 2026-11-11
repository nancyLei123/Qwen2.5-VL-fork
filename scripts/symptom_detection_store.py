"""Storage for automatic symptom detections and for the review decisions made on them.

Kept apart from the manual annotations (``annotations.json``), which this module never writes:

``<store>/<trial>/symptom_annotations/detections/<run_id>.json``
    One detector run: what the detector proposed, with its version, configuration and evidence.
    A run is never edited by review. Running the same detector version and configuration again
    replaces the file only when the result differs, and keeps the previous file as ``*.bak``.
    A different version or configuration gets a different ``run_id`` and therefore its own file,
    so earlier predictions stay available for comparison.

``<store>/<trial>/symptom_annotations/detection_reviews.json``
    The reviewer's decision per prediction (accepted / rejected / needs_review), keyed by run and
    prediction ID, with a snapshot of what was decided on and the history of earlier decisions.
    Rejections are kept so false positives can be analysed later.

``<store>`` is the server's symptom store (``runs/`` by default), which git ignores.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import symptom_annotations as store

DETECTIONS_DIRNAME = "detections"
REVIEWS_FILENAME = "detection_reviews.json"
DETECTIONS_SCHEMA = "pd-insighter/symptom-detections@1"
REVIEWS_SCHEMA = "pd-insighter/symptom-detection-reviews@1"
DECISIONS = ("accepted", "rejected", "needs_review", "unreviewed")
RUN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{2,99}$")
DETECTION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{2,119}$")
MAX_NOTE_LENGTH = 1000
MAX_HISTORY = 20


def detections_dir(store_dir: Path, trial_id: str) -> Path:
    return store_dir / store.check_trial_id(trial_id) / store.ANNOTATIONS_DIRNAME / DETECTIONS_DIRNAME


def reviews_path(store_dir: Path, trial_id: str) -> Path:
    return store_dir / store.check_trial_id(trial_id) / store.ANNOTATIONS_DIRNAME / REVIEWS_FILENAME


def check_run_id(run_id: Any) -> str:
    text = str(run_id or "").strip()
    if not RUN_PATTERN.match(text):
        raise store.ValidationError("Invalid detection run ID.")
    return text


def _comparable(document: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key != "generated_at"}


def save_detections(store_dir: Path, document: dict[str, Any]) -> tuple[Path, str]:
    """Write one detector run. Returns the path and ``created`` / ``unchanged`` / ``replaced``."""
    if document.get("schema") != DETECTIONS_SCHEMA:
        raise store.ValidationError("Not a symptom-detections document.")
    path = detections_dir(store_dir, document.get("trial_id")) / f"{check_run_id(document.get('run_id'))}.json"
    payload = json.loads(json.dumps(document, allow_nan=False, default=float))   # plain JSON types only
    payload.setdefault("generated_at", store.now_iso())
    if path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = None
        if current is not None and _comparable(current) == _comparable(payload):
            return path, "unchanged"
        status = "replaced"
    else:
        status = "created"
    path.parent.mkdir(parents=True, exist_ok=True)
    store.atomic_write_json(path, payload)
    return path, status


def list_runs(store_dir: Path, trial_id: str) -> list[dict[str, Any]]:
    """Summaries of the saved runs of a trial, newest first."""
    folder = detections_dir(store_dir, trial_id)
    runs = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if document.get("schema") != DETECTIONS_SCHEMA or document.get("trial_id") != trial_id:
            continue
        detector = document.get("detector") or {}
        runs.append({"run_id": document.get("run_id") or path.stem, "generated_at": document.get("generated_at"),
                     "detector": {"name": detector.get("name"), "version": detector.get("version")},
                     "count": len(document.get("detections") or []),
                     "by_label": (document.get("summary") or {}).get("by_label") or {}})
    runs.sort(key=lambda run: (str(run["generated_at"] or ""), run["run_id"]), reverse=True)
    return runs


def load_detections(store_dir: Path, trial_id: str, run_id: str | None = None) -> dict[str, Any] | None:
    """One run (the newest when ``run_id`` is not given), or ``None`` when nothing is saved."""
    if run_id is None:
        runs = list_runs(store_dir, trial_id)
        if not runs:
            return None
        run_id = runs[0]["run_id"]
    path = detections_dir(store_dir, trial_id) / f"{check_run_id(run_id)}.json"
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    return document if document.get("schema") == DETECTIONS_SCHEMA else None


def empty_reviews_document(trial_id: str) -> dict[str, Any]:
    return {"schema": REVIEWS_SCHEMA, "trial_id": trial_id, "revision": 0, "updated_at": None, "reviews": {}}


def load_reviews(store_dir: Path, trial_id: str) -> dict[str, Any]:
    document, recovered = store.read_json_with_backup(reviews_path(store_dir, trial_id))
    if document is None or document.get("schema") != REVIEWS_SCHEMA:
        return empty_reviews_document(store.check_trial_id(trial_id))
    document.setdefault("reviews", {})
    if recovered:
        document["recovered_from"] = recovered
    return document


def save_review(store_dir: Path, body: dict[str, Any]) -> dict[str, Any]:
    """Record one decision. The detector output is only read, to check that the prediction exists."""
    trial_id = store.check_trial_id(body.get("trial_id"))
    run_id = check_run_id(body.get("run_id"))
    detection_id = str(body.get("detection_id") or "").strip()
    if not DETECTION_ID_PATTERN.match(detection_id):
        raise store.ValidationError("Invalid prediction ID.")
    decision = str(body.get("decision") or "").strip().lower()
    if decision not in DECISIONS:
        raise store.ValidationError("decision must be accepted, rejected, needs_review or unreviewed.")
    run = load_detections(store_dir, trial_id, run_id)
    if run is None:
        raise store.ValidationError("This detection run is not saved for the trial.")
    detection = next((item for item in run.get("detections", []) if item.get("id") == detection_id), None)
    if detection is None:
        raise store.ValidationError("This prediction is not part of the detection run.")

    current = load_reviews(store_dir, trial_id)
    if body.get("base_revision") != current["revision"]:
        raise store.ConflictError("The review decisions were changed in another window.", current)

    entry: dict[str, Any] = {
        "decision": decision, "decided_at": store.now_iso(),
        "reviewer": store._text(body.get("reviewer"), "reviewer", 60),
        "note": store._text(body.get("note"), "note", MAX_NOTE_LENGTH),
        "detection": {key: detection.get(key) for key in ("label_id", "start_s", "end_s", "score", "evidence_strength")},
    }
    if decision == "accepted":
        annotation_id = str(body.get("annotation_id") or "").strip()
        if not store.ID_PATTERN.match(annotation_id):
            raise store.ValidationError("An accepted prediction needs the ID of the annotation it created.")
        accepted = body.get("accepted") if isinstance(body.get("accepted"), dict) else {}
        start = store._seconds(accepted.get("start_s", detection.get("start_s")), "accepted start time")
        end = store._seconds(accepted.get("end_s", detection.get("end_s")), "accepted end time")
        if end <= start:
            raise store.ValidationError("The accepted end time must be after the start time.")
        label_id = str(accepted.get("label_id") or detection.get("label_id") or "").strip()
        entry.update(annotation_id=annotation_id, edited=bool(body.get("edited")),
                     accepted={"label_id": label_id, "start_s": round(start, 6), "end_s": round(end, 6)})

    reviews = current.get("reviews") or {}
    of_run = dict(reviews.get(run_id) or {})
    previous = of_run.get(detection_id)
    history = list((previous or {}).get("history") or [])
    if previous:
        history.append({key: value for key, value in previous.items() if key != "history"})
    entry["history"] = history[-MAX_HISTORY:]
    of_run[detection_id] = entry
    reviews[run_id] = of_run
    document = {"schema": REVIEWS_SCHEMA, "trial_id": trial_id, "revision": current["revision"] + 1,
                "updated_at": entry["decided_at"], "reviews": reviews}
    path = reviews_path(store_dir, trial_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    store.atomic_write_json(path, document)
    return document


def bundle(store_dir: Path, trial_id: str, run_id: str | None = None) -> dict[str, Any]:
    """Everything the page needs in one response: saved runs, one run, and the review decisions."""
    trial_id = store.check_trial_id(trial_id)
    runs = list_runs(store_dir, trial_id)
    run = load_detections(store_dir, trial_id, check_run_id(run_id) if run_id else None) if runs else None
    return {"trial_id": trial_id, "runs": runs, "run": run, "reviews": load_reviews(store_dir, trial_id)}
