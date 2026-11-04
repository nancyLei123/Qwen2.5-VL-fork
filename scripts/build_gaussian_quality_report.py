#!/usr/bin/env python3
"""Build an offline appearance-comparison report without modifying experiment data.

Consumes compare_aria_gaussian_quality.publish_results(): scenes[].validation
and scenes[].runs[].evaluations[].views. Absolute runner paths and paths relative
to the experiment directory are supported. Only index.html and report.md are
written; the runner, source images, checkpoints, and playback stay untouched.

Optional findings.json is display-only, never a model-selection instruction:
{"decision": "Keep original models", "summary": "Reviewer-authored rationale",
 "next_step": "Review remaining views", "updated_at": "ISO timestamp",
 "scenes": [{"id": "upstairs_context", "decision": "Pending", "summary": "..."}]}
All fields are optional plain text; scenes may also be an object keyed by scene id.
No findings are inferred from partial metrics when this file is absent.

Optional deployed_reference/results.json follows check_aria_deployed_quality:
scenes[].id, scenes[].replay_validation.status, and scenes[].contact_sheet.
Only qualitative replay status and safe local image links are displayed; its
non-held-out metrics never enter the checkpoint tables, ranking, or selection.
"""
from __future__ import annotations

import argparse
from html import escape
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs/Ntrial3/environment_reconstruction/quality_comparison_20261022"
SCENES = ("upstairs_context", "sitting_context", "corridor_walking")
TITLES = {"upstairs_context": "Upstairs", "sitting_context": "Sitting area",
          "corridor_walking": "Corridor"}
CHECKPOINTS = (("baseline_7000", 7000), ("long_30000", 7000),
               ("long_30000", 15000), ("long_30000", 30000))
METRICS = (("psnr_db", "PSNR (dB), higher", 3),
           ("ssim", "SSIM, higher", 4), ("mae", "MAE, lower", 5))
METHOD = (
    "Five fixed, temporally distributed held-out RGB views per scene, with a 0.25 s "
    "temporal exclusion radius. Inspect all five views, not only the most favorable one. "
    "Every comparison uses the same view name, camera and original VRS recording time.",
    "Retrained 7k baseline (baseline_7000) is a NEW baseline trained with the identical "
    "previous recipe, but excludes the five validation images plus guard frames and "
    "uses a recolored initializer. It is NOT the original deployed model. The old "
    "single-view PSNR is not directly comparable. Any original-model image comparison "
    "must be labeled separately as qualitative / training-view, not this held-out comparison.",
    "Conditional evaluation with shared MPS poses and geometry from the whole recording. "
    "Held-out RGB is excluded from optimization and initializer recoloring; initializer "
    "colors use training images only. This prevents held-out training-color leakage, "
    "but does not make this an end-to-end held-out SLAM evaluation.",
    "baseline_7000 stops at 7,000 iterations and densifies until 3,500. long_30000 "
    "extends the learning-rate schedule and densifies until 15,000. Baseline versus "
    "long is a scheduling-confounded recipe comparison, not an isolated iteration "
    "ablation. The 7,000 / 15,000 / 30,000 long-run checkpoints share one schedule.",
    "This is a development validation set, not an independent test set. One run per "
    "recipe; repeat variability is not estimated. Camera order and the Torch RNG "
    "use seed 42 in the inspected OpenSplat source (including the Model constructor). "
    "There is no repeated-seed study or uncertainty estimate; deterministic GPU "
    "execution has not been established.",
    "Scores describe pixel appearance, not clinical accuracy, geometric accuracy, "
    "or whole-room completeness. No model is automatically promoted or replaced. "
    "Final candidate promotion requires measured multi-view gains and visual review, "
    "never the last iteration automatically.",
    "Image metrics use native OpenSplat SH3 renders. The current interactive viewer "
    "uses SH2; these are not equivalent pixel outputs. Browser DPR and hidden-scene "
    "SH1 downgrades are separate viewer issues, not evidence of native-render quality.",
)
SCORING = (
    "The runner scores full-frame RGB in encoded image space, normalized to [0, 1], "
    "without resizing or masking: PSNR (dB), SSIM (data_range=1), and MAE. "
    "Means below are arithmetic means over all five matched per-view scores; "
    "incomplete means and deltas are withheld. Delta is checkpoint minus baseline, "
    "so positive PSNR/SSIM and negative MAE favor the candidate. Display scaling "
    "and shared zoom do not change stored images or scores."
)
DEPLOYED_LIMITS = (
    "Qualitative / non-held-out deployed reference. The deployed originals trained "
    "on most of these five views. Any scores shown in the contact sheets are "
    "diagnostic context, not a fair held-out model ranking or promotion criterion. "
    "Replay checks the original held-out PNG only, not every camera pose or the "
    "interactive WebGL viewer. A mismatch requires investigation, not an assumed loader diagnosis."
)


def _objects(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _finite(value):
    return value if _number(value) else None


def _format(value, digits=3, signed=False):
    if not _number(value):
        return "Pending"
    return format(value, f"{ '+' if signed else ''}.{digits}f")


def _exact(value):
    # Keep the JSON recording_s precision; do not substitute requested_s or dashboard time.
    return str(value) if _number(value) else "Not published"


def _read_json(path, warnings=None):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        if warnings is not None:
            warnings.append(f"{path.name} is missing or unreadable ({type(error).__name__}). "
                            "This is a partial snapshot; rebuild after the runner publishes results.")
        return {}


def _local_path(raw, output, root):
    """Reject URLs and paths escaping the local experiment/source tree, including symlinks."""
    if not isinstance(raw, (str, Path)) or not str(raw):
        return None
    raw = str(raw)
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", raw) or raw.startswith("//") or "\\" in raw or "\0" in raw:
        return None
    try:
        path = Path(raw)
        path = (path if path.is_absolute() else output / path).resolve()
        if path.is_relative_to(root.resolve()) and path.is_file():
            return path
    except (OSError, ValueError, RuntimeError):
        pass
    return None


def _image_url(raw, output):
    path = _local_path(raw, output, output.parent)
    if path is None or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return None
    # Escape %, #, ?, quotes, Unicode, and spaces as URL bytes, then HTML-escape at use sites.
    return quote(Path(os.path.relpath(path, output)).as_posix(), safe="/")


def _checkpoint(run_id, step, run, evaluation, views, output):
    rows = _objects(evaluation.get("views"))
    matched = []
    for view in views:
        candidates = [row for row in rows if row.get("name") == view["name"]]
        row = candidates[0] if len(candidates) == 1 else {}
        time_matches = (_number(view["recording_s"]) and
                        row.get("recording_s") == view["recording_s"])
        if row and not time_matches:
            row = {}
            status = "VRS time mismatch; comparison withheld"
        elif len(candidates) > 1:
            status = "Duplicate view; comparison withheld"
        else:
            status = "Published" if row else "Pending"
        scores = {key: row.get(key) if _number(row.get(key)) else None for key, _, _ in METRICS}
        if row and any(value is None for value in scores.values()):
            status = "Partial metrics"
        matched.append({"name": view["name"], "recording_s": view["recording_s"],
                        "render": _image_url(row.get("render"), output),
                        "status": status, **scores})
    count = sum(all(row[key] is not None for key, _, _ in METRICS) for row in matched)
    complete = len(views) == 5 and len({v["name"] for v in views}) == 5 and count == 5
    means = {key: sum(row[key] for row in matched) / 5 if complete else None
             for key, _, _ in METRICS}
    label = "Retrained 7k baseline (baseline_7000)" if run_id == "baseline_7000" else run_id
    return {"recipe": run_id, "step": step, "label": f"{label} / {step:,}",
            "views": matched, "count": count, "complete": complete, "means": means,
            "gaussians": _finite(evaluation.get("gaussians")), "runtime_s": _finite(run.get("runtime_s")),
            "status": "5/5 scored" if complete else f"Pending / partial ({count}/5 scored)"}


def _scene(raw, scene_id, output):
    split_path = _local_path(raw.get("split"), output, output)
    split = _read_json(split_path) if split_path else {}
    validation = _objects(raw.get("validation", split.get("validation")))
    runs = _objects(raw.get("runs"))
    views = []
    for frame in validation:
        name = frame.get("name")
        if not isinstance(name, str) or not name:
            continue
        references = [row.get("reference") for run in runs
                      for evaluation in _objects(run.get("evaluations"))
                      for row in _objects(evaluation.get("views")) if row.get("name") == name]
        if isinstance(split.get("source"), str):
            references.append(str(Path(split["source"]) / "images" / name))
        if scene_id in SCENES and Path(name).name == name:
            references.append(str(output.parent / "gaussian_submaps" / scene_id / "images" / name))
        reference = next((url for raw_path in references if (url := _image_url(raw_path, output))), None)
        views.append({"name": name, "recording_s": _finite(frame.get("recording_s")),
                      "capture_timestamp_ns": _exact(frame.get("capture_timestamp_ns")),
                      "reference": reference})
    checkpoints = []
    for run_id, step in CHECKPOINTS:
        matches = [run for run in runs if run.get("id") == run_id]
        run = matches[0] if len(matches) == 1 else {}
        evaluations = [item for item in _objects(run.get("evaluations")) if item.get("step") == step]
        evaluation = evaluations[0] if len(evaluations) == 1 else {}
        checkpoints.append(_checkpoint(run_id, step, run, evaluation, views, output))
    baseline = checkpoints[0]
    for checkpoint in checkpoints:
        checkpoint["delta"] = {
            key: checkpoint["means"][key] - baseline["means"][key]
            if checkpoint["complete"] and baseline["complete"] else None
            for key, _, _ in METRICS
        }
    camera = raw.get("camera") if isinstance(raw.get("camera"), dict) else {}
    return {"id": scene_id, "title": TITLES.get(scene_id, scene_id), "views": views,
            "checkpoints": checkpoints, "training_count": _finite(raw.get("training_count")),
            "guard_count": _finite(raw.get("guard_count")), "width": _finite(camera.get("width")),
            "height": _finite(camera.get("height")), "validation_count": _finite(raw.get("validation_count"))}


def _findings(output, warnings):
    path = output / "findings.json"
    raw = _read_json(path, warnings) if path.exists() else {}

    def text_fields(record, fields):
        return {key: record[key].strip() for key in fields
                if isinstance(record.get(key), str) and record[key].strip()}

    findings = text_fields(raw, ("decision", "summary", "next_step", "updated_at"))
    scenes = raw.get("scenes")
    if isinstance(scenes, dict):
        scenes = [{**value, "id": key} if isinstance(value, dict) else {"id": key, "summary": value}
                  for key, value in scenes.items()]
    scene_records = [text_fields(scene, ("id", "decision", "summary")) for scene in _objects(scenes)]
    findings["scenes"] = [scene for scene in scene_records if scene.get("decision") or scene.get("summary")]
    if not any(findings.get(key) for key in ("decision", "summary", "next_step", "scenes")):
        return {}
    return findings


def _deployed_reference(output, warnings):
    path = output / "deployed_reference/results.json"
    if not path.exists():
        return None
    if _local_path(path, output, output) is None:
        warnings.append("deployed_reference/results.json is outside the experiment or unavailable; ignored.")
        return None
    read_warnings = []
    raw = _read_json(path, read_warnings)
    warnings.extend(f"deployed_reference/{warning}" for warning in read_warnings)
    statuses = {
        "exact": "Exact replay: identical decoded pixels.",
        "within_one_code_value": "Replay within one 8-bit code value per channel at every pixel.",
        "mismatch": "Replay mismatch: diagnostic only; investigate before interpretation.",
    }
    scenes = []
    for row in _objects(raw.get("scenes")):
        scene_id = row.get("id") if isinstance(row.get("id"), str) else "Unknown scene"
        replay = row.get("replay_validation")
        status = replay.get("status") if isinstance(replay, dict) else None
        status = status if isinstance(status, str) and status else None
        message = statuses.get(status, f"Unrecognized replay status: {status}" if status else "Replay status not yet published.")
        scenes.append({"id": scene_id, "title": TITLES.get(scene_id, scene_id),
                       "replay_status": message, "contact_sheet": _image_url(row.get("contact_sheet"), output)})
    return {"scenes": scenes}


def _snapshot(output):
    warnings = []
    results = _read_json(output / "results.json", warnings)
    raw_scenes = _objects(results.get("scenes"))
    ids = list(SCENES) + list(dict.fromkeys(
        row["id"] for row in raw_scenes if isinstance(row.get("id"), str) and row["id"] not in SCENES))
    scenes = []
    for scene_id in ids:
        matches = [row for row in raw_scenes if row.get("id") == scene_id]
        if len(matches) > 1:
            warnings.append(f"Duplicate scene {scene_id}; ambiguous results withheld.")
        scenes.append(_scene(matches[0] if len(matches) == 1 else {}, scene_id, output))
    progress_path = output / "progress.json"
    progress = _read_json(progress_path, warnings) if progress_path.exists() else {}
    progress_lines = [f"State: {progress.get('state', 'Not yet published')}"]
    if progress.get("stage"):
        progress_lines.append(f"Stage: {progress['stage']}")
    elif progress.get("state") != "complete":
        progress_lines.append("Stage: Not published")
    if _number(progress.get("step")):
        progress_lines.append(f"Stage steps: {_exact(progress['step'])} / {_exact(progress.get('total_steps'))}")
    if _number(progress.get("elapsed_s")):
        progress_lines.append(f"Stage elapsed: {_format(progress['elapsed_s'], 1)} s")
    if progress.get("started_at"):
        progress_lines.append(f"Stage started: {progress['started_at']}")
    findings = _findings(output, warnings)
    deployed = _deployed_reference(output, warnings)
    return {"trial": str(results.get("trial_id", "Ntrial3")),
            "updated_at": str(results.get("updated_at", "Not yet published")),
            "warnings": warnings, "scenes": scenes, "progress": " | ".join(progress_lines),
            "findings": findings, "deployed_reference": deployed}


def _metric_cells(values, signed=False):
    return [_format(values[key], digits, signed) for key, _, digits in METRICS]


SUMMARY_HEADERS = ("Recipe / checkpoint", "Coverage", "Mean PSNR (dB)", "Mean SSIM", "Mean MAE",
                   "Delta PSNR", "Delta SSIM", "Delta MAE", "Gaussians", "Recipe runtime (s)")
VIEW_HEADERS = ("Held-out view", "VRS recording_s", "Capture timestamp (ns)", "Recipe / checkpoint",
                "PSNR (dB)", "SSIM", "MAE", "Status")


def _summary_rows(scene):
    for checkpoint in scene["checkpoints"]:
        yield [checkpoint["label"], checkpoint["status"], *_metric_cells(checkpoint["means"]),
               *_metric_cells(checkpoint["delta"], signed=True),
               _format(checkpoint["gaussians"], 0), _format(checkpoint["runtime_s"], 1)]


def _view_rows(scene):
    for index, view in enumerate(scene["views"]):
        for checkpoint in scene["checkpoints"]:
            row = checkpoint["views"][index]
            yield [view["name"], _exact(view["recording_s"]), view["capture_timestamp_ns"],
                   checkpoint["label"], *_metric_cells(row), row["status"]]


def _html_table(headers, rows, caption):
    head = "".join(f'<th scope="col">{escape(cell)}</th>' for cell in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows)
    if not body:
        body = f'<tr><td colspan="{len(headers)}">The five fixed validation views are not yet published.</td></tr>'
    return (f'<div class="table-wrap" tabindex="0"><table><caption>{escape(caption)}</caption>'
            f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')


def _markdown_text(value):
    # Escape both raw HTML and Markdown syntax supplied through experiment metadata.
    text = escape(str(value), quote=False).replace("\n", " ").replace("\r", " ")
    return re.sub(r"([\\`*_{}\[\]()|])", r"\\\1", text)


def _markdown_table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
            *("| " + " | ".join(_markdown_text(cell) for cell in row) + " |" for row in rows)]


def _decision_paragraphs(data):
    findings = data["findings"]
    if not findings:
        return ["No decision has been published.",
                "Available scores are measurements, not a recommendation. Keep the original "
                "default if the benefit is unclear. Inspect all five views and separately compare "
                "the deployed original before approving a change."]
    paragraphs = [findings.get(key) for key in ("decision", "summary", "next_step") if findings.get(key)]
    for scene in findings["scenes"]:
        title = TITLES.get(scene.get("id"), scene.get("id", "Scene"))
        content = " ".join(scene.get(key, "") for key in ("decision", "summary")).strip()
        if content:
            paragraphs.append(f"{title}: {content}")
    return paragraphs


def _decision_source(data):
    if not data["findings"]:
        return "No findings.json summary supplied. No visual conclusions are inferred by this builder."
    timestamp = data["findings"].get("updated_at")
    return ("Supplied findings.json summary" + (f" ({timestamp})" if timestamp else "")
            + ". Display only: not independently verified, and does not change playback or promote a model.")


def _deployed_html(data):
    deployed = data["deployed_reference"]
    if deployed is None:
        return ""
    rows = []
    for scene in deployed["scenes"]:
        url = scene["contact_sheet"]
        link = (f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener">'
                f'Open {escape(scene["title"])} contact sheet</a>' if url else "Contact sheet not available.")
        rows.append(f'<li><strong>{escape(scene["title"])}</strong>: '
                    f'{escape(scene["replay_status"])} {link}</li>')
    content = "<ul>" + "".join(rows) + "</ul>" if rows else "<p>No scene checks published yet.</p>"
    return ('<section class="card" aria-labelledby="deployed-heading">'
            '<h2 id="deployed-heading">Deployed original / visual reference</h2>'
            f'<p class="hint">{escape(DEPLOYED_LIMITS)}</p>{content}</section>')


def _deployed_markdown(data):
    deployed = data["deployed_reference"]
    if deployed is None:
        return []
    lines = ["## Deployed original / visual reference", "", DEPLOYED_LIMITS, ""]
    for scene in deployed["scenes"]:
        url = scene["contact_sheet"]
        link = f"[Open contact sheet]({url})" if url else "Contact sheet not available."
        lines.append(f"- {_markdown_text(scene['title'])}: {_markdown_text(scene['replay_status'])} {link}")
    if not deployed["scenes"]:
        lines.append("No scene checks published yet.")
    return [*lines, ""]


def _markdown(data):
    lines = [f"# {_markdown_text(data['trial'])}: Gaussian quality comparison", "",
             f"Results snapshot: {_markdown_text(data['updated_at'])}.", "",
             "[Open the comparison](index.html) | [Original playback](../playback.html)", "",
             "## Decision summary", "",
             *[line for paragraph in _decision_paragraphs(data) for line in (_markdown_text(paragraph), "")],
             _markdown_text(_decision_source(data)), "",
             *_deployed_markdown(data),
             "## Runner progress (per-stage)", "", _markdown_text(data["progress"]), "",
             "Read directly from progress.json when this report was built. Stage steps are not "
             "overall progress; no overall percentage is inferred. Results and progress are separate snapshots.", "",
             "## Method and limits", "", *(f"- {note}" for note in METHOD), "", SCORING, "",
             "Runtime is total recipe training time, repeated for its checkpoints, not checkpoint timing. "
             "Pending is not zero. This builder reports a snapshot, not runner completion or source-integrity verification.", ""]
    lines.extend(f"- {_markdown_text(warning)}" for warning in data["warnings"])
    for scene in data["scenes"]:
        lines += ["", f"## {_markdown_text(scene['title'])}", "",
                  f"Training images: {_format(scene['training_count'], 0)}; temporal guards: "
                  f"{_format(scene['guard_count'], 0)}; fixed views published: {len(scene['views'])}/5.", "",
                  *_markdown_table(SUMMARY_HEADERS, _summary_rows(scene)), "",
                  "### All five views", "",
                  "Exact recording_s values are original VRS seconds, not requested sample times "
                  "or offset dashboard times. No human inspection is claimed by this report.", ""]
        if scene["views"]:
            lines += _markdown_table(VIEW_HEADERS, _view_rows(scene))
        else:
            lines.append("Validation views and per-view metrics are pending publication.")
    lines += ["", "## Inspect and rebuild", "",
              "Use the scene, held-out view, and candidate selectors in index.html. Compare all five "
              "views for each candidate at the same zoom; the checklist is session-only and manual. "
              "The initial candidate is the latest checkpoint with published scores, not a metric-ranked winner. "
              "Image links point to existing local originals; no images are copied and no external requests or CDNs are used.", "",
              "Rebuild with `python scripts/build_gaussian_quality_report.py --output <experiment-directory>` "
              "after new results are published. Only index.html and report.md are written. "
              "No models are promoted, trained, or modified.", ""]
    return "\n".join(lines)


CSS = """
:root{--ink:#182c25;--muted:#5f7267;--green:#17785c;--line:#d9e3da;--cream:#f6f6ef;--paper:#fffefa;--amber:#926020}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(ellipse at 0 0,#dce9d8,transparent 60%),var(--cream);font-family:"Avenir Next","Gill Sans","Trebuchet MS",sans-serif;line-height:1.6}
main{max-width:1720px;margin:auto;padding:24px 32px 48px}nav,.toolbar,.badges,.actions{display:flex;align-items:center;gap:12px;flex-wrap:wrap}nav{justify-content:space-between;border-bottom:1px solid var(--line);padding:10px 0 20px}a{color:var(--green);text-underline-offset:4px}header{padding:38px 0 26px;max-width:1100px}.eyebrow{color:var(--green);font-size:12px;letter-spacing:.16em;font-weight:800;text-transform:uppercase}h1{font-size:clamp(32px,4.5vw,60px);line-height:1.1;letter-spacing:-.045em;margin:12px 0 20px}h2{font-size:24px;line-height:1.3;margin:0 0 14px}h3{font-size:17px;margin:16px 0 8px}.intro{font-size:19px;color:var(--muted);max-width:850px}.badges span{font-size:12px;border:1px solid #b9d1c3;border-radius:30px;padding:5px 12px}.card{background:var(--paper);border:1px solid var(--line);border-radius:24px;padding:24px;margin:0 0 24px;box-shadow:0 10px 30px #223c2210}.notice{border-left:4px solid var(--amber);background:#faf2e1;padding:14px 18px;border-radius:8px;margin:18px 0}.muted,.hint{color:var(--muted)}.hint{font-size:13px}label{font-weight:700;font-size:13px;display:grid;gap:5px}select,button{font:inherit;border:1px solid #bdd1c2;border-radius:22px;background:white;color:var(--ink);padding:9px 16px;cursor:pointer;max-width:100%}select{min-width:180px}button:disabled{opacity:.5;cursor:default}button.primary{background:var(--green);color:white}button:focus-visible,a:focus-visible,select:focus-visible,input:focus-visible,.table-wrap:focus-visible{outline:3px solid #ba833b;outline-offset:3px}.toolbar{align-items:end}.toolbar>label{flex:1;min-width:180px}.readout{font-variant-numeric:tabular-nums;overflow-wrap:anywhere}.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-top:20px}figure{margin:0;min-width:0}figcaption{min-height:80px;overflow-wrap:anywhere}figcaption strong,figcaption span{display:block}.image-window{aspect-ratio:1;overflow:hidden;border-radius:14px;border:1px solid var(--line);background:#e9eee5;position:relative}.image-window img{display:block;width:100%;height:100%;object-fit:contain;transform:scale(var(--zoom,1));transform-origin:var(--origin,50% 50%)}.placeholder{position:absolute;inset:0;display:grid;place-content:center;text-align:center;padding:20px;color:var(--muted)}.image-link{display:inline-block;margin-top:8px;font-size:13px}.zoom{margin:20px 0;align-items:center}.zoom label{display:flex;align-items:center}.zoom input{accent-color:var(--green)}.review{display:flex;align-items:center;gap:9px}.review input{accent-color:var(--green);width:18px;height:18px}.checklist{display:flex;flex-wrap:wrap;gap:8px;padding:0;list-style:none}.checklist li{border:1px solid var(--line);border-radius:12px;padding:6px 10px;font-size:12px}.checklist li.checked{background:#e0eee2;border-color:#88b69d}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}caption{text-align:left;color:var(--muted);font-size:13px;padding:0 0 12px}th,td{text-align:left;padding:12px 10px;border-bottom:1px solid var(--line);white-space:nowrap}thead{background:#eef4eb}td:first-child{font-weight:650}details summary{cursor:pointer;font-weight:750;margin:16px 0}.method li{margin:12px 0;max-width:1200px}footer{font-size:12px;color:var(--muted)}[hidden]{display:none!important}.js-only{display:none}.js .js-only{display:flex}.js section.js-only{display:block}
@media(max-width:800px){main{padding:16px}.card{padding:16px;border-radius:18px}.images{grid-template-columns:1fr}figcaption{min-height:0;margin-bottom:8px}.toolbar>label{flex-basis:100%}header{padding-top:24px}.zoom{align-items:start;flex-direction:column}}
@media(prefers-reduced-motion:no-preference){header{animation:enter .45s ease-out}@keyframes enter{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}}
"""


JS = """
"use strict";
const data = JSON.parse(document.getElementById("report-data").textContent);
const $ = id => document.getElementById(id);
const seen = new Set();
const fmt = (value, digits) => Number.isFinite(value) ? value.toFixed(digits) : "Pending";
const exact = value => Number.isFinite(value) ? String(value) : "Not published";
let sceneIndex = 0, viewIndex = 0, candidateIndex = 3;
function options(select, entries) {
  select.replaceChildren();
  entries.forEach(([value, label]) => { const option = document.createElement("option");
    option.value = String(value); option.textContent = label; select.append(option); });
}
function reviewKey(index) { return JSON.stringify([sceneIndex, candidateIndex, index]); }
function renderReview(scene) {
  $("reviewed").checked = seen.has(reviewKey(viewIndex));
  $("reviewed").disabled = !scene.views.length;
  $("review-list").replaceChildren();
  scene.views.forEach((view, index) => {
    const item = document.createElement("li");
    const checked = seen.has(reviewKey(index));
    item.textContent = `View ${index + 1}: ${checked ? "inspected" : "not marked"}`;
    item.className = checked ? "checked" : "";
    $("review-list").append(item);
  });
  const count = scene.views.filter((_, index) => seen.has(reviewKey(index))).length;
  $("review-status").textContent = `${count}/5 manually marked for this candidate in this session. Inspect all five before drawing conclusions.`;
}
function panel(id, title, url, scores, view) {
  $(id + "-title").textContent = title;
  $(id + "-score").textContent = scores
    ? `PSNR ${fmt(scores.psnr_db, 3)} dB | SSIM ${fmt(scores.ssim, 4)} | MAE ${fmt(scores.mae, 5)}`
    : "Original held-out RGB reference";
  const img = $(id + "-image"), missing = $(id + "-missing"), link = $(id + "-link");
  img.hidden = true; missing.hidden = false; link.hidden = !url;
  missing.textContent = "Image not available in this snapshot. Rebuild after publication.";
  img.onload = () => { img.hidden = false; missing.hidden = true; };
  img.onerror = () => { img.hidden = true; missing.hidden = false; missing.textContent = "Image could not be loaded; its local file may have moved."; };
  img.alt = `${title}, ${view ? view.name : "view pending"}, VRS ${view ? exact(view.recording_s) : "pending"} s`;
  if (url) { img.src = url; link.href = url; }
  else { img.removeAttribute("src"); link.removeAttribute("href"); }
}
function renderView() {
  const scene = data.scenes[sceneIndex], view = scene.views[viewIndex];
  const baseline = scene.checkpoints[0], candidate = scene.checkpoints[candidateIndex];
  $("view").value = String(viewIndex);
  $("view-time").textContent = view
    ? `View ${viewIndex + 1}/${scene.views.length}: ${view.name} | VRS recording_s = ${exact(view.recording_s)} s | Capture timestamp = ${view.capture_timestamp_ns} ns`
    : "The five fixed held-out views have not yet been published.";
  panel("reference", "Ground truth", view?.reference, null, view);
  panel("baseline", "Retrained 7k baseline", baseline.views[viewIndex]?.render, baseline.views[viewIndex] || {}, view);
  panel("candidate", candidate.label, candidate.views[viewIndex]?.render, candidate.views[viewIndex] || {}, view);
  $("previous").disabled = !view || viewIndex === 0;
  $("next").disabled = !view || viewIndex === scene.views.length - 1;
  renderReview(scene);
}
function resetZoom() {
  $("zoom").value = "1"; $("zoom-value").textContent = "1.0x";
  $("images").style.setProperty("--zoom", "1");
  $("images").style.setProperty("--origin", "50% 50%");
}
function selectScene() {
  const scene = data.scenes[sceneIndex];
  viewIndex = 0;
  const available = scene.checkpoints.map((cp, i) => i > 0 && cp.count > 0 ? i : -1).filter(i => i > 0);
  candidateIndex = available.length ? available[available.length - 1] : 3;
  options($("view"), scene.views.map((view, i) => [i, `${i + 1}. ${view.name} / VRS ${exact(view.recording_s)} s`]));
  $("view").disabled = !scene.views.length;
  options($("candidate"), scene.checkpoints.slice(1).map((cp, i) => [i + 1, `${cp.label} (${cp.count}/5 scored)`]));
  $("candidate").value = String(candidateIndex);
  document.querySelectorAll(".scene-metrics").forEach((section, i) => { section.hidden = i !== sceneIndex; });
  $("scene-info").textContent = `${scene.title} | ${fmt(scene.training_count, 0)} training images | ${fmt(scene.guard_count, 0)} temporal guards | ${scene.views.length}/5 fixed views published | Source ${fmt(scene.width, 0)} x ${fmt(scene.height, 0)} px`;
  resetZoom(); renderView();
}
options($("scene"), data.scenes.map((scene, i) => [i, scene.title]));
$("scene").addEventListener("change", event => { sceneIndex = Number(event.target.value); selectScene(); });
$("view").addEventListener("change", event => { viewIndex = Number(event.target.value); renderView(); });
$("candidate").addEventListener("change", event => { candidateIndex = Number(event.target.value); renderView(); });
$("previous").addEventListener("click", () => { viewIndex--; renderView(); });
$("next").addEventListener("click", () => { viewIndex++; renderView(); });
$("reviewed").addEventListener("change", event => {
  if (event.target.checked) seen.add(reviewKey(viewIndex)); else seen.delete(reviewKey(viewIndex));
  renderReview(data.scenes[sceneIndex]);
});
$("zoom").addEventListener("input", event => {
  $("images").style.setProperty("--zoom", event.target.value);
  $("zoom-value").textContent = `${Number(event.target.value).toFixed(1)}x`;
});
$("reset-zoom").addEventListener("click", resetZoom);
document.querySelectorAll(".image-window").forEach(window => window.addEventListener("pointermove", event => {
  const rect = window.getBoundingClientRect();
  $("images").style.setProperty("--origin", `${100 * (event.clientX - rect.left) / rect.width}% ${100 * (event.clientY - rect.top) / rect.height}%`);
}));
selectScene();
document.documentElement.classList.add("js");
"""


def _html(data):
    panels = []
    for key, title in (("reference", "Ground truth"), ("baseline", "Retrained 7k baseline"), ("candidate", "Selected candidate")):
        panels.append(f'''<figure><figcaption><strong id="{key}-title">{title}</strong>
<span id="{key}-score" class="hint"></span></figcaption><div class="image-window">
<img id="{key}-image" alt="" hidden><div id="{key}-missing" class="placeholder">Image pending</div></div>
<a id="{key}-link" class="image-link" target="_blank" rel="noopener" hidden>Open original image</a></figure>''')
    sections = []
    for scene in data["scenes"]:
        sections.append('<section class="card scene-metrics">'
                        f'<h2>{escape(scene["title"])} / checkpoint comparison</h2>'
                        + _html_table(SUMMARY_HEADERS, _summary_rows(scene), "All five fixed views; no partial-view means or metric-ranked winner.")
                        + '<p class="hint">Delta = checkpoint minus baseline. Higher PSNR/SSIM and lower MAE are favorable. '
                        'Runtime is total recipe training time, repeated across checkpoints, not checkpoint timing.</p>'
                        '<details open><summary>All five views / exact VRS times and per-view metrics</summary>'
                        + _html_table(VIEW_HEADERS, _view_rows(scene), "Every planned checkpoint for every fixed view, including pending results.")
                        + '</details></section>')
    payload = json.dumps(data, ensure_ascii=True, allow_nan=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    warnings = "".join(f'<p class="notice">{escape(note)}</p>' for note in data["warnings"])
    complete = sum(cp["complete"] for scene in data["scenes"] for cp in scene["checkpoints"])
    total = sum(len(scene["checkpoints"]) for scene in data["scenes"])
    method = "".join(f"<li>{escape(note)}</li>" for note in METHOD)
    decision = "".join(f"<p>{escape(paragraph)}</p>" for paragraph in _decision_paragraphs(data))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'; form-action 'none'; object-src 'none'">
<title>{escape(data['trial'])} | Gaussian quality comparison</title><style>{CSS}</style></head><body><main>
<nav><strong>PD Insighter / Reconstruction research</strong><div class="actions"><a href="../playback.html">Original playback</a><a href="report.md">Method and numeric report</a></div></nav>
<header><div class="eyebrow">{escape(data['trial'])} / Development validation</div><h1>More training.<br>A closer reconstruction?</h1>
<p class="intro">Compare the original RGB image, the Retrained 7k baseline, and a long-run checkpoint at exactly the same held-out camera view.</p>
<div class="badges"><span>Local-only / no external requests</span><span>Five fixed views per scene</span><span>{complete}/{total} checkpoint summaries available</span></div>
<p class="hint">Results snapshot: {escape(data['updated_at'])}. Rebuild to include later results; pending is not zero.</p></header>
<section class="card" aria-labelledby="decision-heading"><div class="eyebrow">At a glance</div><h2 id="decision-heading">Decision summary</h2>{decision}<p class="hint">{escape(_decision_source(data))}</p></section>
{_deployed_html(data)}
<section class="card"><div class="eyebrow">Runner progress / per-stage snapshot</div><p class="readout">{escape(data['progress'])}</p><p class="hint">Read directly from progress.json at build time. Stage steps are not overall progress; no overall percentage is inferred. Results and progress are separate snapshots.</p></section>
{warnings}<div class="notice"><strong>Recipe comparison, not an iteration-only ablation.</strong> The long run changes both learning-rate and densification schedules. Development validation is not an independent test; appearance scores do not establish clinical or geometric accuracy.</div>
<noscript><p class="notice">Enable JavaScript for synchronized images and selectors. All checkpoint and per-view tables remain readable below; report.md contains the same numeric snapshot.</p></noscript>
<section class="card js-only"><h2>One view. Three equal-size images.</h2><div class="toolbar">
<label>Scene<select id="scene"></select></label><label>Shared held-out view<select id="view"></select></label><label>Candidate checkpoint<select id="candidate"></select></label></div>
<p class="hint">The initial candidate is the latest checkpoint with published scores, not a metric-ranked winner. Missing runs remain selectable.</p>
<p class="hint"><strong>Retrained baseline, not the deployed model.</strong> Same previous recipe, now excluding five validation images plus guard frames and using a recolored initializer. Old single-view PSNR is not directly comparable; original-model image comparisons must be separately labeled qualitative / training-view.</p>
<p class="hint"><strong>Native OpenSplat SH3 images and metrics.</strong> The current interactive viewer uses SH2; these are not equivalent pixel outputs. Promotion requires measured multi-view gains and visual review, never the last iteration automatically.</p>
<p id="scene-info" class="hint"></p><div class="toolbar"><button id="previous">Previous view</button><button id="next" class="primary">Next view</button></div>
<p id="view-time" class="readout" aria-live="polite"></p><div id="images" class="images">{''.join(panels)}</div>
<div class="toolbar zoom"><label>Shared zoom <input id="zoom" type="range" min="1" max="4" step="0.1" value="1"><output id="zoom-value">1.0x</output></label><button id="reset-zoom">Reset zoom</button><span class="hint">Move over any image to align the zoom center across all three.</span></div>
<label class="review"><input id="reviewed" type="checkbox">I inspected this view for the selected candidate</label><ul id="review-list" class="checklist"></ul><p id="review-status" class="hint" aria-live="polite"></p>
<p class="hint">The checklist is manual and session-only. Viewing an image is not automatically recorded as an inspection. VRS times are original recording seconds, not dashboard offsets.</p></section>
{''.join(sections)}<details class="card method"><summary>Method and limitations / what this experiment can tell us</summary><ul>{method}</ul><p>{escape(SCORING)}</p></details>
<footer>Independent, read-only reporting of experiment data. Original playback, images and models are unchanged. No automatic model promotion. This snapshot does not assert runner completion or independently verify source hashes.</footer>
</main><script id="report-data" type="application/json">{payload}</script><script>{JS}</script></body></html>
'''


def build(output: Path):
    """Write index.html and report.md in output, tolerating absent or partial results."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = _snapshot(output)
    html, markdown = _html(data), _markdown(data)
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / "report.md").write_text(markdown, encoding="utf-8")
    return output / "index.html", output / "report.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Experiment directory containing results.json (also the report destination)")
    for path in build(parser.parse_args().output):
        print(path)


if __name__ == "__main__":
    main()
