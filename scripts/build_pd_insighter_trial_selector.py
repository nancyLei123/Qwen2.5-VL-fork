#!/usr/bin/env python3
"""Build the trial library without changing any recording or analysis outputs."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
import re
from urllib.parse import quote
import warnings

from pd_insighter_environment_navigation import environment_entry, SIMULATED_TRIALS

QWEN_REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = QWEN_REPO / "runs"
TEMPLATES = Path(__file__).parent / "templates"
FULL_SELECTOR_HTML = RUNS_DIR / "pd_insighter_trial_selector.html"
CLIP_SELECTOR_HTML = RUNS_DIR / "pd_insighter_trial_selector_clip.html"
LEGACY_SELECTOR_HTML = RUNS_DIR / "pd_insighter_trial_selector_legacy.html"
FULL_SESSION_DASHBOARD_NAME = "pd_insighter_session_review.html"
CANONICAL_TRIAL_ENTRY_NAME = "index.html"
LEGACY_DASHBOARD_NAME = "pd_insighter_rule_interactive.html"
ADDITIONAL_NORMAL_TRIALS = {"Half_hour", "Half_hour2", "60minutes_1", "60minutes_2"}
NORMAL_TRIALS = {f"Ntrial{i}" for i in range(1, 10)} | ADDITIONAL_NORMAL_TRIALS
GROUPS = {
    "normal": ("Normal motion", "Everyday movement recordings without instructed Parkinson simulation."),
    "parkinson": ("Parkinson simulation", "Instructed symptom simulations, not confirmed patient recordings."),
    "reconstruction": ("Dedicated 3D scans", "Recordings collected specifically to reconstruct the environment."),
    "archive": ("Earlier / unclassified", "Earlier trial1-trial8 recordings and records without a confirmed recording category."),
}
VIEW_FILES = {
    "manual_labeling_href": ("Manual Labels", "pd_insighter_manual_labeling.html"),
    "stride_review_href": ("Stride", "pd_insighter_stride_review.html"),
    "transition_review_href": ("Transitions", "pd_insighter_transition_review.html"),
    "parkinson_review_href": ("Parkinson Review", "pd_insighter_parkinson_review.html"),
}


def trial_sort_key(trial_id: str):
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", trial_id)]


def versioned_href(path: Path, rel_href: str) -> str:
    if not path.is_file():
        return ""
    return f"{rel_href}{'&' if '?' in rel_href else '?'}v={int(path.stat().st_mtime)}"


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (ValueError, OSError) as error:
        warnings.warn(f"Could not read {path}: {error}")
        return {}


def is_dashboard(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open() as stream:
        start = stream.read(4096)
    # A room scan's index redirect is not a full-session analysis dashboard.
    return not re.search(r'<meta\b[^>]*http-equiv=[\"\']refresh[\"\']', start, re.I)


def category_for(trial_id: str, manifest: dict, session: dict) -> str:
    declared = manifest.get("trial_type") or session.get("trial_type")
    if declared == "parkinson_simulated":
        return "parkinson"
    if declared == "normal":
        return "normal"
    if declared in {"environment_scan", "reconstruction"} or trial_id == "RoomScan3c":
        return "reconstruction"
    if trial_id in SIMULATED_TRIALS:
        return "parkinson"
    if trial_id in NORMAL_TRIALS:
        return "normal"
    # Never infer a diagnosis or recording purpose from the available algorithms.
    return "archive"


def collect_trials() -> list[dict]:
    directories = {p.parent for p in RUNS_DIR.glob("*/trial_manifest.json")}
    directories.update(p.parent for p in RUNS_DIR.glob("*/session_dashboard_manifest.json"))
    for filename in ("scene_playback_data.json", "reconstruction_summary.json"):
        directories.update(p.parent.parent for p in RUNS_DIR.glob(f"*/environment_reconstruction/{filename}"))
    items = []
    for run_dir in sorted(directories):
        if run_dir.name.startswith(("_", ".")):
            continue
        trial_id = run_dir.name
        manifest = read_json(run_dir / "trial_manifest.json")
        session = read_json(run_dir / "session_dashboard_manifest.json")
        playback = read_json(run_dir / "environment_reconstruction/scene_playback_data.json")
        category = category_for(trial_id, manifest, session)

        def href(path):
            return versioned_href(path, quote(path.relative_to(RUNS_DIR).as_posix(), safe="/")) if path else ""

        full = next((run_dir / name for name in (CANONICAL_TRIAL_ENTRY_NAME, FULL_SESSION_DASHBOARD_NAME)
                     if is_dashboard(run_dir / name)), None)
        legacy = run_dir / LEGACY_DASHBOARD_NAME
        legacy = legacy if not full and is_dashboard(legacy) else None
        environment = environment_entry(run_dir)
        source_vrs = playback.get("provenance", {}).get("vrs", "")
        stem = manifest.get("recording_stem") or session.get("recording_stem") or (Path(source_vrs).stem if source_vrs else trial_id)
        duration = session.get("session_duration_s") if full else playback.get("duration_s")
        duration = duration if isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0 else None
        primary = environment if category == "reconstruction" and environment else full or environment or legacy
        primary_label = "Open 3D environment" if primary and primary == environment else "Open Full Session" if full else "Open legacy review"
        item = {
            "trial_id": trial_id, "recording_stem": str(stem), "label": f"{trial_id} - {stem}",
            "category": category, "duration_s": duration,
            "duration_label": "Review duration" if full else "Scan duration",
            "media_mode": manifest.get("media_mode") or session.get("media_mode") or "",
            "full_dashboard_href": href(full), "legacy_dashboard_href": href(legacy),
            "environment_reconstruction_href": href(environment),
            "primary_href": href(primary), "primary_label": primary_label,
            "status": "ready" if primary else "registered",
            "manifest_href": href(run_dir / "trial_manifest.json"),
            "metrics_href": href(run_dir / "clip_metrics.csv"),
            "timeline_href": href(run_dir / "timeline_predictions.json"),
            "point_cloud_href": href(run_dir / "plots/point_cloud_3d.png"),
            "horizontal_href": href(run_dir / "plots/horizontal_displacement_by_clip.png"),
            "vertical_href": href(run_dir / "plots/vertical_displacement_by_clip.png"),
        }
        for key, (_, filename) in VIEW_FILES.items():
            path = run_dir / filename
            item[key] = href(path) if is_dashboard(path) else ""
        items.append(item)
    return sorted(items, key=lambda item: trial_sort_key(item["trial_id"]))


def collect_legacy_trials() -> list[dict]:
    return [item for item in collect_trials() if item["legacy_dashboard_href"]]


def duration_text(seconds):
    total = round(seconds)
    return f"{total // 60}:{total % 60:02d}"


def render_trial(item: dict) -> str:
    esc = html.escape
    trial_id = esc(item["trial_id"])
    category = item["category"]
    environment = bool(item["environment_reconstruction_href"])
    badges = []
    if environment:
        badges.append('<span class="badge environment">3D available</span>')
    if item["legacy_dashboard_href"]:
        badges.append('<span class="badge">Legacy interface</span>')
    if item["status"] == "registered":
        badges.append('<span class="badge pending">Registered only</span>')
    if item["media_mode"] == "aria_only":
        badges.append('<span class="badge">Aria only</span>')
    if category == "reconstruction":
        note = "Room scan for environment reconstruction. No action analysis is expected."
    elif item["status"] == "registered":
        note = "Recording registered; no review dashboard has been built yet."
    elif item["legacy_dashboard_href"]:
        note = "Earlier analysis, preserved in its original interface."
    elif category == "parkinson":
        note = "Simulated symptoms. Automatic estimates require manual review."
    else:
        note = "Review the recording and its available analyses."
    metadata = f'<span>{esc(item["duration_label"])} <strong>{duration_text(item["duration_s"])}</strong></span>' if item["duration_s"] else ""
    links = []
    for key, label in [(key, label) for key, (label, _) in VIEW_FILES.items()] + [("environment_reconstruction_href", "Environment 3D")]:
        if item[key] and item[key] != item["primary_href"]:
            links.append(f'<a href="{esc(item[key])}">{label}</a>')
    files = []
    for key, label in (("manifest_href", "Recording record"), ("metrics_href", "Clip metrics"),
                       ("timeline_href", "Classification data"), ("point_cloud_href", "Point cloud plot"),
                       ("horizontal_href", "Horizontal movement plot"), ("vertical_href", "Height plot")):
        if item[key]:
            files.append(f'<a href="{esc(item[key])}">{label}</a>')
    resources = f'<details class="files"><summary>Files &amp; plots</summary><div>{"".join(files)}</div></details>' if files else ""
    primary = (f'<a class="open-review" href="{esc(item["primary_href"])}" aria-label="{item["primary_label"]} for {trial_id}">{item["primary_label"]}<span aria-hidden="true"> &rarr;</span></a>'
               if item["primary_href"] else '<span class="unavailable">Dashboard not yet available</span>')
    return f'''<article class="trial-card" data-trial-id="{trial_id}" data-category="{category}" data-environment="{str(environment).lower()}" data-search="{esc(item['trial_id'] + ' ' + item['recording_stem'])}" aria-labelledby="title-{trial_id}">
      <div class="trial-top"><h3 id="title-{trial_id}">{trial_id}</h3><div class="badges">{''.join(badges)}</div></div>
      <p class="recording-name" title="{esc(item['recording_stem'])}">{esc(item['recording_stem'])}</p>
      <p class="trial-note">{note}</p><div class="metadata">{metadata}</div>
      <div class="trial-actions">{primary}<nav class="view-links" aria-label="{trial_id} analysis pages">{''.join(links)}</nav></div>
      {resources}
    </article>'''


def build_html(trials: list[dict], *, page_title="PD Insighter | Trial Library", **_legacy_options) -> str:
    """Retain the publisher API so reconstruction refreshes cannot restore the old UI."""
    counts = {key: sum(item["category"] == key for item in trials) for key in GROUPS}
    group_filters, sections = [], []
    for key, (title, description) in GROUPS.items():
        if key != "archive":
            group_filters.append(f'<button type="button" class="category-filter {key}" data-group="{key}" aria-pressed="false"><span>{title}</span><strong>{counts[key]:02d}</strong></button>')
        cards = "".join(render_trial(item) for item in trials if item["category"] == key)
        sections.append(f'''<section class="trial-section {key}" data-section="{key}" aria-labelledby="heading-{key}"{' hidden' if not counts[key] else ''}>
          <div class="section-heading"><div><h2 id="heading-{key}">{title} <span class="section-count">{counts[key]}</span></h2><p>{description}</p></div></div>
          <div class="trial-grid">{cards}</div></section>''')
    values = {
        "TITLE": html.escape(page_title), "TOTAL": str(len(trials)), "ARCHIVE_COUNT": str(counts["archive"]),
        "READY_COUNT": str(sum(item["status"] == "ready" for item in trials)),
        "ENV_COUNT": str(sum(bool(item["environment_reconstruction_href"]) for item in trials)),
        "CATEGORY_FILTERS": "".join(group_filters), "SECTIONS": "".join(sections),
        "SCRIPT": (TEMPLATES / "trial_library.mjs").read_text(),
    }
    template = (TEMPLATES / "pd_insighter_trial_library.html").read_text()
    return re.sub(r"__([A-Z_]+)__", lambda match: values[match[1]], template)


def build_redirect_html(target_href: str, message: str) -> str:
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={html.escape(target_href, quote=True)}"><title>PD Insighter | Trial Library</title></head><body><p>{html.escape(message)} <a href="{html.escape(target_href, quote=True)}">Open the trial library</a></p></body></html>'''


def main() -> None:
    trials = collect_trials()
    FULL_SELECTOR_HTML.write_text(build_html(trials))
    CLIP_SELECTOR_HTML.write_text(build_redirect_html("pd_insighter_trial_selector.html", "All recordings are now in the trial library."))
    LEGACY_SELECTOR_HTML.write_text(build_redirect_html("pd_insighter_trial_selector.html?group=archive", "Earlier recordings are preserved in the trial library."))
    print(f"Wrote {FULL_SELECTOR_HTML}")
    print(json.dumps({"total": len(trials), "categories": {key: sum(t['category'] == key for t in trials) for key in GROUPS}, "registered_only": [t['trial_id'] for t in trials if t['status'] == 'registered']}, indent=2))


if __name__ == "__main__":
    main()
