#!/usr/bin/env python3
"""Build the Symptom Annotation page for PD-Insighter trials.

Writes ``runs/<trial>/pd_insighter_symptom_annotation.html`` and copies the page's CSS and JS
into ``runs/<trial>/symptom_annotation/``. Reads only the trial's saved manifests; it does not
touch analysis results, activity labels or saved annotations. Annotations are loaded by the page
from the server (``/api/symptom-annotations``), so rebuilding never loses them.

Clock: the page works in dashboard session time, the clock of the trial's session videos and of
Full Session / Manual Labels. Both session videos are already on that clock, so each video's
offset is 0. For downward footage attached to an Aria-only timeline (``downward_attachment``),
the padded video covers the whole session but only ``start_s``-``end_s`` is real footage; the
page shows that window and uses the recorded rotation.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pd_insighter_environment_navigation import inject_environment_tab  # noqa: E402
from symptom_annotation_navigation import SYMPTOM_PAGE, SYMPTOM_VIEW  # noqa: E402

QWEN_REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = QWEN_REPO / "runs"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates" / "symptom_annotation"
ASSET_DIR = "symptom_annotation"
ASSETS = ("symptom_annotation.css", "symptom_annotation_core.mjs", "symptom_annotation_app.mjs")
NAV_VIEWS = (
    ("Full Session", "index.html"),
    ("Manual Labels", "pd_insighter_manual_labeling.html"),
    ("Stride", "pd_insighter_stride_review.html"),
    ("Transitions", "pd_insighter_transition_review.html"),
)
# Older dual-view trials mount the downward camera upside down (Manual Labels also starts at 180 degrees).
LEGACY_DOWNWARD_TURNS = 2
ARIA_TURNS = 1


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def read_manifest(trial_id: str) -> dict[str, Any]:
    path = RUNS_DIR / trial_id / "session_dashboard_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{trial_id} has no session_dashboard_manifest.json")
    return read_json(path)


def video_file(trial_id: str, uri: str) -> Path | None:
    relative = unquote(urlsplit(uri).path)
    return RUNS_DIR / trial_id / relative if relative else None


def video_entries(trial_id: str, session: dict[str, Any], trial: dict[str, Any], duration: float) -> list[dict[str, Any]]:
    media_mode = session.get("media_mode") or trial.get("media_mode") or "dual_view"
    attachment = session.get("downward_attachment") or trial.get("downward_attachment") or None
    entries = []
    for key, title, uri_field, duration_field in (
            ("aria", "Aria glasses", "aria_session_video_uri", "aria_session_duration_s"),
            ("downward", "Downward camera", "downward_session_video_uri", "downward_session_duration_s")):
        uri = str(session.get(uri_field) or "")
        path = video_file(trial_id, uri)
        recorded = not (key == "downward" and (media_mode == "aria_only" or not uri))
        entry = {
            "key": key, "title": title, "uri": uri,
            "available": recorded and path is not None and path.is_file(),
            "missing": None if recorded else "not_recorded",
            "offset_s": 0.0, "duration_s": float(session.get(duration_field) or duration),
            "turns": ARIA_TURNS if key == "aria" else LEGACY_DOWNWARD_TURNS, "coverage_s": None,
        }
        if recorded and not entry["available"]:
            entry["missing"] = "missing_file"
        if key == "downward" and attachment:
            entry["turns"] = int(attachment.get("rotation_quarter_turns", 0)) % 4
            entry["coverage_s"] = [float(attachment["start_s"]), float(attachment["end_s"])]
        entries.append(entry)
    return entries


def sync_note(session: dict[str, Any]) -> str:
    attachment = session.get("downward_attachment")
    if attachment:
        verified = (attachment.get("verification") or {}).get("valid")
        return (f"Downward footage was attached to the unchanged Aria timeline (offset "
                f"{float(attachment.get('applied_offset_s', attachment.get('offset_s', 0))):.3f} s, "
                f"{'audio-verified' if verified else 'verification not recorded'}). Outside "
                f"{float(attachment['start_s']):.1f}-{float(attachment['end_s']):.1f} s the downward video is padding.")
    alignment = (session.get("time_information") or {}).get("alignment") or {}
    if not alignment:
        return "No alignment record for the two cameras; treat timing between them as unverified."
    parts = [f"Downward camera aligned to the Aria video by the research pipeline ({alignment.get('method', 'method not recorded')})."]
    if alignment.get("audio_offset_s") is not None:
        parts.append(f"Offset {float(alignment['audio_offset_s']):.3f} s.")
    if alignment.get("detail"):
        parts.append(str(alignment["detail"]))
    return " ".join(parts)


def trial_jump_options(current: str, building: frozenset[str] = frozenset()) -> list[dict[str, str]]:
    items = [{"trial_id": "__selector__", "label": "All Trials", "href": "../pd_insighter_trial_selector.html"}]
    for manifest in sorted(RUNS_DIR.glob("*/session_dashboard_manifest.json")):
        name = manifest.parent.name
        has_page = (RUNS_DIR / name / SYMPTOM_PAGE).is_file() or name == current or name in building
        href = SYMPTOM_PAGE if name == current else f"../{name}/{SYMPTOM_PAGE if has_page else 'index.html'}"
        items.append({"trial_id": name, "label": name if has_page else f"{name} (Full Session)", "href": href})
    return items


def problems(trial_id: str) -> list[str]:
    """Missing prerequisites for a usable page (empty when the trial is ready)."""
    try:
        session = read_manifest(trial_id)
    except FileNotFoundError as exc:
        return [str(exc)]
    issues = []
    if not float(session.get("session_duration_s") or 0) > 0:
        issues.append(f"{trial_id}: session_dashboard_manifest.json has no session duration")
    videos = video_entries(trial_id, session, read_json(RUNS_DIR / trial_id / "trial_manifest.json"),
                           float(session.get("session_duration_s") or 0))
    if not any(video["available"] for video in videos):
        issues.append(f"{trial_id}: no session video found (checked {', '.join(v['uri'] or v['key'] for v in videos)})")
    return issues


def load_payload(trial_id: str, building: frozenset[str] = frozenset()) -> dict[str, Any]:
    session = read_manifest(trial_id)
    trial = read_json(RUNS_DIR / trial_id / "trial_manifest.json")
    duration = float(session.get("session_duration_s") or 0)
    if duration <= 0:
        raise ValueError(f"{trial_id}: the manifest has no session duration")
    segments = session.get("continuous_model_segments") or session.get("model_segments") or []
    return {
        "trial_id": str(session.get("trial_id") or trial_id),
        "recording_stem": str(session.get("recording_stem") or trial_id),
        "session_duration_s": duration,
        "session_fps": float(session.get("session_fps") or 30),
        "media_mode": session.get("media_mode") or trial.get("media_mode") or "dual_view",
        "videos": video_entries(trial_id, session, trial, duration),
        # Lets the shared page-audio controller probe the same Aria file as the other views.
        "aria_session_video_uri": str(session.get("aria_session_video_uri") or ""),
        "activity_segments": [
            {"start_s": float(seg["start_s"]), "end_s": float(seg["end_s"]), "label": str(seg.get("label", ""))}
            for seg in segments if isinstance(seg, dict) and "start_s" in seg and "end_s" in seg
        ],
        "sync_note": sync_note(session),
        # Existing manual activity labels, read-only. Ntrial10 keeps them in the Parkinson Review
        # store (see action_label_path in pd_insighter_environment_navigation.py).
        "manual_activity_api": "/api/parkinson-manual-segments" if trial_id == "Ntrial10" else "/api/session-manual-segments",
        "trial_jump_options": trial_jump_options(trial_id, building),
    }


def asset_version() -> str:
    digest = hashlib.sha256()
    for name in ASSETS:
        digest.update((TEMPLATE_DIR / name).read_bytes())
    return digest.hexdigest()[:12]


def nav_tabs(trial_id: str) -> str:
    tabs = []
    for label, filename in NAV_VIEWS:
        if (RUNS_DIR / trial_id / filename).is_file():
            tabs.append(f'        <a class="nav-tab" href="{filename}">{html.escape(label)}</a>')
    return "\n".join(tabs)


def build_html(payload: dict[str, Any], trial_dir: Path) -> str:
    template = (TEMPLATE_DIR / "page.html").read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
    document = (template.replace("__PAYLOAD__", data)
                .replace("__NAV_TABS__", nav_tabs(payload["trial_id"]))
                .replace("__ASSET_VERSION__", asset_version())
                .replace("__TRIAL_ID__", html.escape(payload["trial_id"])))
    return inject_environment_tab(document, trial_dir, active_view=SYMPTOM_VIEW)


def build(trial_id: str, building: frozenset[str] = frozenset()) -> Path:
    trial_dir = RUNS_DIR / trial_id
    payload = load_payload(trial_id, building)
    assets = trial_dir / ASSET_DIR
    assets.mkdir(exist_ok=True)
    for name in ASSETS:
        shutil.copyfile(TEMPLATE_DIR / name, assets / name)
    page = trial_dir / SYMPTOM_PAGE
    page.write_text(build_html(payload, trial_dir), encoding="utf-8")
    # The trial switcher links a view only once its page exists; render again on the first build.
    final = build_html(payload, trial_dir)
    if page.read_text(encoding="utf-8") != final:
        page.write_text(final, encoding="utf-8")
    return page


def discover() -> list[str]:
    return [path.parent.name for path in sorted(RUNS_DIR.glob("*/session_dashboard_manifest.json"))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", nargs="*", help="Trial IDs. Defaults to every trial with a session manifest.")
    args = parser.parse_args()
    trials = args.trials or discover()
    for trial_id in trials:
        for issue in problems(trial_id):
            print(f"Warning: {issue}")
        print(f"Wrote {build(trial_id, frozenset(trials))}")


if __name__ == "__main__":
    main()
