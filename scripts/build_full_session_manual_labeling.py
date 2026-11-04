#!/usr/bin/env python3
"""Build full-session manual labeling pages for normal PD-Insighter trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from pd_insighter_environment_navigation import environment_navigation, action_label_path


QWEN_REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = QWEN_REPO / "runs"
OUTPUT_NAME = "pd_insighter_manual_labeling.html"
STRIDE_REVIEW_NAME = "pd_insighter_stride_review.html"
TRANSITION_REVIEW_NAME = "pd_insighter_transition_review.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        nargs="*",
        help="Trial ids to build. Defaults to all runs with session_dashboard_manifest.json.",
    )
    return parser.parse_args()


def versioned_href(path: Path, rel_href: str) -> str:
    try:
        version = int(path.stat().st_mtime)
    except FileNotFoundError:
        return rel_href
    joiner = "&" if "?" in rel_href else "?"
    return f"{rel_href}{joiner}v={version}"


def collect_trial_jump_options(current_trial_id: str) -> list[dict[str, str]]:
    items = [
        {
            "trial_id": "__selector__",
            "label": "All Trials",
            "href": versioned_href(
                RUNS_DIR / "pd_insighter_trial_selector.html",
                "../pd_insighter_trial_selector.html",
            ),
        }
    ]
    manifests: list[tuple[str, str]] = []
    for manifest_path in sorted(RUNS_DIR.glob("*/session_dashboard_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            continue
        trial_id = str(manifest.get("trial_id") or manifest_path.parent.name)
        recording_stem = str(manifest.get("recording_stem") or trial_id)
        manifests.append((trial_id, recording_stem))

    def sort_key(item: tuple[str, str]) -> tuple[int, str]:
        trial_id = item[0]
        if trial_id.startswith("Ntrial"):
            try:
                return int(trial_id.removeprefix("Ntrial")), trial_id
            except ValueError:
                pass
        return 10**9, trial_id

    for trial_id, recording_stem in sorted(manifests, key=sort_key):
        target = RUNS_DIR / trial_id / OUTPUT_NAME
        relative_href = OUTPUT_NAME if trial_id == current_trial_id else f"../{trial_id}/{OUTPUT_NAME}"
        items.append(
            {
                "trial_id": trial_id,
                "label": f"{trial_id} - {recording_stem}",
                "href": versioned_href(target, relative_href),
            }
        )
    return items


def load_payload(trial_id: str) -> dict[str, Any]:
    run_dir = RUNS_DIR / trial_id
    manifest_path = run_dir / "session_dashboard_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing session dashboard manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    continuous_segments = manifest.get("continuous_model_segments") or []
    use_continuous_prediction = bool(continuous_segments)
    full_review_path = run_dir / "index.html"
    if not full_review_path.exists():
        full_review_path = run_dir / "pd_insighter_session_review.html"
    stride_review_path = run_dir / STRIDE_REVIEW_NAME
    transition_review_path = run_dir / TRANSITION_REVIEW_NAME
    transition_review_href = ""
    if manifest.get("transition_review_href") and transition_review_path.exists():
        transition_review_href = versioned_href(transition_review_path, TRANSITION_REVIEW_NAME)
    initial_manual_segments: list[dict[str, Any]] = []
    manual_segments_path = action_label_path(run_dir)
    if manual_segments_path.exists():
        try:
            manual_payload = json.loads(manual_segments_path.read_text())
            if isinstance(manual_payload, dict) and isinstance(manual_payload.get("segments"), list):
                initial_manual_segments = manual_payload["segments"]
        except (OSError, json.JSONDecodeError):
            initial_manual_segments = []
    return {
        "trial_id": str(manifest.get("trial_id", trial_id)),
        "recording_stem": str(manifest.get("recording_stem", trial_id)),
        "session_duration_s": float(manifest.get("session_duration_s") or 0),
        "downward_session_duration_s": float(
            manifest.get("downward_session_duration_s")
            or manifest.get("session_duration_s")
            or 0
        ),
        "session_fps": float(manifest.get("session_fps") or 30),
        "downward_session_video_uri": str(manifest.get("downward_session_video_uri") or ""),
        "aria_session_video_uri": str(manifest.get("aria_session_video_uri") or ""),
        "model_segments": continuous_segments if use_continuous_prediction else (manifest.get("model_segments") or []),
        "model_prediction_title": "Continuous Automatic Classification" if use_continuous_prediction else "Automatic Classification",
        "model_prediction_note": "variable periods; 0.5s boundary evidence" if use_continuous_prediction else "fixed 2s windows",
        "initial_manual_segments": initial_manual_segments,
        "time_spent_records": manifest.get("time_spent_records") or [],
        "trial_selector_href": versioned_href(RUNS_DIR / "pd_insighter_trial_selector.html", "../pd_insighter_trial_selector.html"),
        "full_review_href": versioned_href(full_review_path, full_review_path.name),
        "stride_review_href": versioned_href(stride_review_path, STRIDE_REVIEW_NAME),
        "transition_review_href": transition_review_href,
        "trial_jump_options": collect_trial_jump_options(trial_id),
    }


def discover_trials() -> list[str]:
    trials: list[str] = []
    for manifest_path in sorted(RUNS_DIR.glob("*/session_dashboard_manifest.json")):
        trials.append(manifest_path.parent.name)
    return trials


@environment_navigation
def build_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=True)
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PD-Insighter Manual Labeling</title>
  <style>
    :root {
      --bg: #f4f1e8;
      --card: rgba(255,255,255,.94);
      --ink: #17231c;
      --muted: #657066;
      --line: rgba(35,48,40,.13);
      --green: #167a55;
      --green-soft: #e7f4ec;
      --danger: #b94338;
      --shadow: 0 18px 45px rgba(31,43,35,.10);
      --sitting: #e6ab02;
      --standing: #7570b3;
      --walking: #d95f02;
      --upstairs: #1b9e77;
      --downstairs: #666666;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 8% 8%, rgba(22,122,85,.13), transparent 28%),
        linear-gradient(135deg, #f8f4e8 0%, #eef3ed 54%, #f5f1e8 100%);
      color: var(--ink);
      font-family: Avenir Next, Trebuchet MS, Verdana, sans-serif;
    }
    button, select, input, textarea {
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      color: var(--ink);
      padding: 10px 13px;
      min-height: 42px;
    }
    button {
      cursor: pointer;
      font-weight: 850;
      box-shadow: 0 6px 16px rgba(35,48,40,.06);
    }
    button.primary { background: var(--green); border-color: var(--green); color: #fff; }
    button.danger { color: var(--danger); border-color: rgba(185,67,56,.35); }
    button:disabled { opacity: .45; cursor: not-allowed; }
    button.icon-button {
      width: 28px;
      height: 28px;
      padding: 0;
      justify-content: center;
      border-radius: 999px;
      line-height: 1;
      font-size: 16px;
      background: rgba(255,255,255,.86);
    }
    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 900;
      letter-spacing: .06em;
      text-transform: uppercase;
    }
    textarea {
      border-radius: 18px;
      min-height: 86px;
      resize: vertical;
      text-transform: none;
      letter-spacing: 0;
      font-weight: 600;
      width: 100%;
    }
    .shell { max-width: 1720px; margin: 0 auto; padding: 22px; display: grid; gap: 18px; }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 28px; box-shadow: var(--shadow); padding: 18px; }
    .page-nav {
      position: sticky;
      top: 12px;
      z-index: 30;
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 12px 14px;
      border: 1px solid rgba(23,35,28,.10);
      border-radius: 20px;
      background: rgba(255,255,255,.94);
      box-shadow: 0 14px 36px rgba(31,43,35,.10);
      backdrop-filter: blur(14px);
    }
    .nav-context { display: flex; align-items: center; gap: 11px; flex: 0 0 auto; min-width: 190px; }
    .nav-mark {
      width: 38px;
      height: 38px;
      border-radius: 13px;
      display: grid;
      place-items: center;
      color: #fff;
      background: linear-gradient(145deg, var(--green), #10563e);
      box-shadow: 0 8px 18px rgba(16,86,62,.20);
      font-size: 12px;
      font-weight: 900;
      letter-spacing: .04em;
    }
    .nav-location { display: grid; gap: 2px; min-width: 0; line-height: 1.1; }
    .nav-location strong { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: .92rem; }
    .nav-location span { color: var(--muted); font-size: .76rem; font-weight: 800; }
    .nav-tabs {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 3px;
      flex: 1 1 auto;
      min-width: 0;
      padding: 4px;
      border: 1px solid rgba(23,35,28,.10);
      border-radius: 16px;
      background: rgba(23,35,28,.045);
    }
    .nav-tab {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 36px;
      padding: 0 12px;
      border-radius: 12px;
      color: var(--muted);
      text-decoration: none;
      white-space: nowrap;
      font-size: 13px;
      font-weight: 900;
    }
    .nav-tab:hover { color: var(--ink); background: rgba(255,255,255,.76); }
    .nav-tab.active { color: var(--green); background: #fff; box-shadow: 0 4px 12px rgba(31,43,35,.08); }
    .nav-trial-switcher { display: flex; align-items: center; gap: 9px; flex: 0 1 340px; min-width: 250px; }
    .nav-trial-switcher > span { color: var(--muted); font-size: .7rem; font-weight: 900; letter-spacing: .08em; text-transform: uppercase; }
    .nav-trial-select { width: 100%; min-width: 0; border-radius: 13px; }
    .hero { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(320px, .6fr); gap: 18px; align-items: stretch; }
    h1, h2, h3, p { margin-top: 0; }
    h1 { font-size: clamp(32px, 4vw, 58px); line-height: .95; letter-spacing: -.055em; margin-bottom: 12px; }
    h2 { font-size: 22px; letter-spacing: -.035em; margin-bottom: 6px; }
    p { color: var(--muted); line-height: 1.5; }
    .pills { display: flex; flex-wrap: wrap; gap: 8px; }
    .pill { display: inline-flex; gap: 6px; align-items: center; border: 1px solid var(--line); border-radius: 999px; padding: 8px 11px; background: #fff; color: var(--muted); font-weight: 800; font-size: 13px; }
    .summary-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .metric { background: #fff; border: 1px solid var(--line); border-radius: 20px; padding: 13px; }
    .metric b { display: block; font-size: 24px; letter-spacing: -.04em; }
    .metric span { color: var(--muted); font-size: 12px; font-weight: 850; text-transform: uppercase; letter-spacing: .06em; }
    .video-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 18px; }
    .video-card { min-width: 0; }
    .video-card h2 { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .video-frame { position: relative; min-width: 0; background: #07100b; border-radius: 22px; overflow: hidden; aspect-ratio: 16/9; display: grid; place-items: center; }
    .aria-frame { aspect-ratio: 1/1; }
    video { width: 100%; height: 100%; object-fit: contain; display: block; }
    #downwardVideo { transform: rotate(180deg); }
    #ariaVideo { transform: rotate(90deg); }
    .rotated-quarter {
      position: absolute;
      top: 50%;
      left: 50%;
      width: var(--rotated-video-width, auto);
      height: var(--rotated-video-height, 100%);
      max-width: none;
      max-height: none;
    }
    .controls { display: grid; gap: 12px; }
    .control-row { display: flex; flex-wrap: wrap; gap: 10px; align-items: end; }
    .time-readout { margin-left: auto; font-size: 24px; font-weight: 950; letter-spacing: -.04em; }
    input[type="range"] { width: 100%; padding: 0; accent-color: var(--green); }
    .timeline-stack { display: grid; gap: 14px; }
    .timeline-row { display: grid; grid-template-columns: 170px minmax(0, 1fr); gap: 12px; align-items: center; }
    .timeline-label { color: var(--muted); font-weight: 950; text-transform: uppercase; font-size: 12px; letter-spacing: .06em; }
    .timeline { position: relative; height: 68px; background: #fff; border: 1px solid var(--line); border-radius: 18px; overflow: hidden; touch-action: none; }
    .overview-timeline { height: 48px; cursor: pointer; }
    .model-seg, .manual-seg, .gap-seg, .drag-preview, .playhead, .overview-seg, .overview-window { position: absolute; top: 0; bottom: 0; }
    .model-seg { opacity: .72; border-right: 1px solid rgba(255,255,255,.42); cursor: pointer; }
    .manual-seg { top: 8px; bottom: 8px; border-radius: 15px; border: 2px solid rgba(23,35,28,.42); opacity: .9; cursor: pointer; overflow: hidden; }
    .manual-seg.active { border: 3px solid var(--ink); box-shadow: 0 0 0 5px rgba(22,122,85,.12); }
    .manual-seg span { display: block; padding: 8px 42px 8px 10px; font-size: 12px; font-weight: 950; white-space: nowrap; color: #142119; text-shadow: 0 1px 0 rgba(255,255,255,.45); }
    .manual-delete { position: absolute; right: 6px; top: 50%; transform: translateY(-50%); z-index: 3; }
    .gap-seg { top: 8px; bottom: 8px; border-radius: 15px; background: repeating-linear-gradient(135deg, rgba(185,67,56,.18), rgba(185,67,56,.18) 8px, rgba(185,67,56,.06) 8px, rgba(185,67,56,.06) 16px); border: 2px dashed rgba(185,67,56,.45); cursor: pointer; }
    .gap-seg:hover { background: rgba(185,67,56,.18); }
    .drag-preview { top: 8px; bottom: 8px; border-radius: 15px; border: 3px dashed var(--ink); background: rgba(22,122,85,.18); pointer-events: none; }
    .pending-seg { position: absolute; top: 8px; bottom: 8px; border-radius: 15px; border: 3px solid var(--ink); background: rgba(22,122,85,.18); pointer-events: none; z-index: 6; box-shadow: 0 0 0 4px rgba(22,122,85,.10); }
    .playhead { width: 3px; background: var(--ink); pointer-events: none; z-index: 20; }
    .overview-seg { border-right: 1px solid rgba(255,255,255,.35); cursor: pointer; }
    .overview-model-seg { top: 7px; bottom: 25px; opacity: .42; }
    .overview-manual-seg { top: 24px; bottom: 7px; opacity: .88; border-radius: 5px; }
    .overview-window { border: 3px solid var(--ink); border-radius: 12px; background: rgba(22,122,85,.08); pointer-events: none; z-index: 8; }
    .window-toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: end; margin: 0 0 12px; }
    .window-toolbar label.inline-check { flex-direction: row; align-items: center; gap: 8px; min-width: auto; }
    .window-toolbar label.inline-check input { width: auto; }
    .pending-panel {
      display: grid;
      gap: 12px;
      margin: 12px 0;
      padding: 14px;
      border: 2px solid rgba(22,122,85,.28);
      border-radius: 20px;
      background: linear-gradient(135deg, rgba(22,122,85,.10), rgba(255,255,255,.86));
    }
    .pending-panel.hidden { display: none; }
    .pending-actions { display: flex; flex-wrap: wrap; gap: 8px; }
    .pending-actions button { color: #142119; border-color: rgba(23,35,28,.28); }
    .relabel-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: -4px; }
    .relabel-actions button { color: #142119; border-color: rgba(23,35,28,.28); }
    .relabel-actions button.active { outline: 3px solid rgba(23,35,28,.32); box-shadow: 0 0 0 5px rgba(22,122,85,.10); }
    .pending-meta { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; color: var(--muted); font-size: 13px; font-weight: 800; }
    .manual-grid { display: grid; grid-template-columns: minmax(0, 1fr) 420px; gap: 18px; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: end; margin: 12px 0; }
    .editor { display: grid; gap: 12px; align-content: start; }
    .editor-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .status { color: var(--muted); font-weight: 800; font-size: 13px; }
    .sync-status {
      display: inline-flex;
      align-items: center;
      width: fit-content;
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 7px 10px;
      background: #fff;
      color: var(--muted);
      font-size: 12px;
      font-weight: 900;
    }
    .sync-status.good { color: var(--green); background: var(--green-soft); border-color: rgba(22,122,85,.25); }
    .sync-status.adjusting { color: #8a5b00; background: #fff5d8; border-color: rgba(138,91,0,.25); }
    .sync-status.ended { color: var(--muted); background: #f1f3f1; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px; }
    .legend span { display: inline-flex; gap: 6px; align-items: center; color: var(--muted); font-weight: 800; font-size: 13px; }
    .swatch { width: 14px; height: 14px; border-radius: 4px; border: 1px solid rgba(0,0,0,.1); }
    .small { font-size: 12px; color: var(--muted); line-height: 1.45; }
    .mono { font-family: Menlo, Consolas, monospace; }
    .segment-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; align-items: stretch; margin-bottom: 8px; }
    .segment-main { width: 100%; justify-content: flex-start; border-radius: 16px; text-align: left; min-width: 0; }
    .segment-main span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .segment-list-delete { border-radius: 16px; padding-inline: 12px; background: #fff; }
    @media (max-width: 1050px) {
      .hero, .video-grid, .manual-grid { grid-template-columns: 1fr; }
      .timeline-row { grid-template-columns: 1fr; }
      .time-readout { margin-left: 0; }
    }
    @media (max-width: 760px) {
      .shell { padding: 12px; }
      .page-nav { position: static; display: grid; align-items: stretch; }
      .nav-context { min-width: 0; }
      .nav-tabs { justify-content: flex-start; overflow-x: auto; }
      .nav-tab { flex: 0 0 auto; }
      .nav-trial-switcher { min-width: 0; max-width: none; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <nav class="page-nav" aria-label="PD Insighter navigation">
      <div class="nav-context">
        <span class="nav-mark" aria-hidden="true">PD</span>
        <div class="nav-location">
          <strong id="trialPill"></strong>
          <span>Manual Labels</span>
        </div>
      </div>
      <div class="nav-tabs" aria-label="Trial views">
        <a id="fullReviewLink" class="nav-tab" href="#">Full Session</a>
        <span class="nav-tab active" aria-current="page">Manual Labels</span>
        <a id="strideReviewLink" class="nav-tab" href="#">Stride</a>
        <a id="transitionReviewLink" class="nav-tab" href="#">Transitions</a>
      </div>
      <label class="nav-trial-switcher" for="trialJumpSelect">
        <span>Trial</span>
        <select id="trialJumpSelect" class="nav-trial-select" aria-label="Switch trial or return to all trials"></select>
      </label>
    </nav>

    <section class="hero">
      <div class="card">
        <h1>Full-Session Manual Labeling</h1>
        <p>Press Space to play/pause, drag directly on the manual timeline to label action ranges, and keep the whole session covered without overlapping labels. Action labels are saved separately from generated pages.</p>
        <div class="pills">
          <span class="pill">Space = play/pause</span>
          <span class="pill">Drag-to-label ranges</span>
          <span class="pill">Existing labels are protected</span>
          <span class="pill">New labels fill empty time only</span>
          <span class="pill">Gap finder</span>
          <span class="pill">Saved outside regenerated HTML</span>
        </div>
      </div>
      <aside class="card summary-grid">
        <div class="metric"><b id="coverageMetric">--</b><span>Manual Coverage</span></div>
        <div class="metric"><b id="gapMetric">--</b><span>Unlabeled Gaps</span></div>
        <div class="metric"><b id="transitionAwareMetric">--</b><span>Transition-Aware Agreement</span></div>
        <div class="metric"><b id="exactMetric">--</b><span>Exact Agreement</span></div>
      </aside>
    </section>

    <section class="video-grid">
      <div class="card video-card">
        <h2>Downward View <button id="rotateDownwardBtn" type="button">Rotate Downward 90°</button></h2>
        <div class="video-frame"><video id="downwardVideo" playsinline muted preload="metadata"></video></div>
      </div>
      <div class="card video-card">
        <h2>Aria View <button id="rotateAriaBtn" type="button">Rotate Aria 90°</button></h2>
        <div class="video-frame aria-frame"><video id="ariaVideo" playsinline preload="metadata"></video></div>
      </div>
    </section>

    <section class="card controls">
      <div class="control-row">
        <button id="playBtn" class="primary" type="button">Play</button>
        <button id="pauseBtn" type="button">Pause</button>
        <button id="backBtn" type="button">-2s</button>
        <button id="forwardBtn" type="button">+2s</button>
        <button id="startRangeBtn" type="button">Start Label Here</button>
        <button id="endRangeBtn" class="primary" type="button">End Here + Save</button>
        <button id="cancelRangeBtn" type="button">Cancel Start</button>
        <label>Speed
          <select id="speedSelect">
            <option value="0.5">0.5x</option>
            <option value="0.75">0.75x</option>
            <option value="1" selected>1.0x</option>
            <option value="1.25">1.25x</option>
            <option value="1.5">1.5x</option>
          </select>
        </label>
        <div class="time-readout"><span id="timeNow">0.0s</span> / <span id="timeTotal">--</span></div>
      </div>
      <input id="scrubber" type="range" min="0" max="1" step="0.01" value="0" />
      <div id="syncStatus" class="sync-status">Sync checking...</div>
      <div class="small">Tip: press Space to play/pause. Pause at the start point, keep “Start drag from playhead” checked, then drag to the end point and choose the motion label. Existing labels are protected; new labels only fill unlabeled time. Use the ×, Delete/Backspace, or the list Delete button to remove a label.</div>
    </section>

    <section class="card">
      <h2>Timeline</h2>
      <div class="window-toolbar">
        <label>Timeline window
          <select id="timelineWindowSelect">
            <option value="30">30 seconds</option>
            <option value="60">1 minute</option>
            <option value="120" selected>2 minutes</option>
            <option value="300">5 minutes</option>
            <option value="600">10 minutes</option>
            <option value="full">Full session</option>
          </select>
        </label>
        <button id="prevWindowBtn" type="button">← Previous Window</button>
        <button id="nextWindowBtn" type="button">Next Window →</button>
        <button id="centerWindowBtn" type="button">Center On Video</button>
        <button id="fullWindowBtn" type="button">Show Full Session</button>
        <label class="inline-check">
          <input id="followWindowCheckbox" type="checkbox" checked />
          Follow video
        </label>
        <label class="inline-check" title="When checked, dragging on the manual row starts from the current video/playhead time instead of the mouse-down point.">
          <input id="usePlayheadDragStartCheckbox" type="checkbox" checked />
          Start drag from playhead
        </label>
        <span id="timelineWindowStatus" class="status">Timeline window loading...</span>
      </div>
      <div class="timeline-stack">
        <div class="timeline-row">
          <div class="timeline-label">Session Overview</div>
          <div id="overviewTimeline" class="timeline overview-timeline"></div>
        </div>
        <div class="timeline-row">
          <div class="timeline-label">__MODEL_TIMELINE_TITLE__<br><span class="small">__MODEL_TIMELINE_NOTE__</span></div>
          <div id="modelTimeline" class="timeline"></div>
        </div>
        <div class="timeline-row">
          <div class="timeline-label">Manual Labels<br><span class="small">drag here</span></div>
          <div id="manualTimeline" class="timeline"></div>
        </div>
      </div>
      <div class="toolbar">
        <label>Default label for gap tools
          <select id="labelSelect">
            <option value="sitting">sitting</option>
            <option value="standing">standing</option>
            <option value="walking" selected>walking</option>
            <option value="upstairs">upstairs</option>
            <option value="downstairs">downstairs</option>
          </select>
        </label>
        <label>Confidence
          <select id="confidenceSelect">
            <option value="high">high</option>
            <option value="medium" selected>medium</option>
            <option value="low">low</option>
            <option value="uncertain">uncertain</option>
          </select>
        </label>
        <button id="startGapBtn" type="button">Jump To Next Gap</button>
        <button id="fillCurrentGapBtn" type="button">Fill Current Gap</button>
        <button id="fillAllGapsBtn" type="button">Fill All Gaps With Current Label</button>
        <button id="saveAllBtn" class="primary" type="button">Save Manual Labels</button>
        <span id="saveStatus" class="status">Loading manual labels...</span>
      </div>
      <div id="pendingPanel" class="pending-panel hidden">
        <div>
          <h3 style="margin-bottom:4px;">Choose Label For Selected Range</h3>
          <div class="pending-meta">
            <span id="pendingRangeText">No range selected.</span>
            <span id="pendingRangeHint"></span>
          </div>
        </div>
        <div id="pendingActionButtons" class="pending-actions">
          <button type="button" data-pending-label="sitting" style="background:#e6ab02;">sitting</button>
          <button type="button" data-pending-label="standing" style="background:#7570b3;color:#fff;">standing</button>
          <button type="button" data-pending-label="walking" style="background:#d95f02;color:#fff;">walking</button>
          <button type="button" data-pending-label="upstairs" style="background:#1b9e77;color:#fff;">upstairs</button>
          <button type="button" data-pending-label="downstairs" style="background:#666666;color:#fff;">downstairs</button>
          <button id="pendingCancelBtn" type="button">Cancel Selection</button>
        </div>
      </div>
      <div id="legend" class="legend"></div>
    </section>

    <section class="manual-grid">
      <div class="card">
        <h2>Manual Segments</h2>
        <p class="small">Select a segment below or click a segment on the manual timeline to edit it precisely.</p>
        <div id="segmentList"></div>
      </div>
      <aside class="card editor">
        <h2>Selected Segment</h2>
        <label>Segment
          <select id="segmentSelect"></select>
        </label>
        <div class="editor-grid">
          <label>Start (s)
            <input id="startInput" type="number" min="0" step="0.1" />
          </label>
          <label>End (s)
            <input id="endInput" type="number" min="0" step="0.1" />
          </label>
        </div>
        <label>Motion
          <select id="segmentLabelSelect">
            <option value="sitting">sitting</option>
            <option value="standing">standing</option>
            <option value="walking">walking</option>
            <option value="upstairs">upstairs</option>
            <option value="downstairs">downstairs</option>
          </select>
        </label>
        <div id="selectedActionButtons" class="relabel-actions" aria-label="Change selected segment action">
          <button type="button" data-relabel-action="sitting" style="background:#e6ab02;">Change to sitting</button>
          <button type="button" data-relabel-action="standing" style="background:#7570b3;color:#fff;">Change to standing</button>
          <button type="button" data-relabel-action="walking" style="background:#d95f02;color:#fff;">Change to walking</button>
          <button type="button" data-relabel-action="upstairs" style="background:#1b9e77;color:#fff;">Change to upstairs</button>
          <button type="button" data-relabel-action="downstairs" style="background:#666666;color:#fff;">Change to downstairs</button>
        </div>
        <label>Segment confidence
          <select id="segmentConfidenceSelect">
            <option value="high">high</option>
            <option value="medium">medium</option>
            <option value="low">low</option>
            <option value="uncertain">uncertain</option>
          </select>
        </label>
        <label>Note
          <textarea id="noteInput" placeholder="Optional note, e.g. waiting at traffic light, uncertain slope, crowd blocking view..."></textarea>
        </label>
        <div class="toolbar">
          <button id="setStartBtn" type="button">Set Start Here</button>
          <button id="setEndBtn" type="button">Set End Here</button>
          <button id="saveSegmentBtn" class="primary" type="button">Save Segment</button>
          <button id="deleteSegmentBtn" class="danger" type="button">Delete Segment</button>
        </div>
        <p id="editorStatus" class="status">No segment selected yet.</p>
      </aside>
    </section>
  </main>

  <script>
    const data = __DATA__;
    const colors = {
      sitting: "#e6ab02",
      standing: "#7570b3",
      walking: "#d95f02",
      upstairs: "#1b9e77",
      downstairs: "#666666",
      missing: "#b6bec6"
    };
    const actions = ["sitting", "standing", "walking", "upstairs", "downstairs"];
    const trialId = data.trial_id;
    const duration = Number(data.session_duration_s || 0);
    const downwardDuration = Number(data.downward_session_duration_s || duration);
    const modelSegments = (data.model_segments || []).map(seg => ({
      ...seg,
      start_s: Number(seg.start_s),
      end_s: Number(seg.end_s),
      label: String(seg.label || "")
    })).filter(seg => Number.isFinite(seg.start_s) && Number.isFinite(seg.end_s));

    const downwardVideo = document.getElementById("downwardVideo");
    const ariaVideo = document.getElementById("ariaVideo");
    const scrubber = document.getElementById("scrubber");
    const playBtn = document.getElementById("playBtn");
    const pauseBtn = document.getElementById("pauseBtn");
    const overviewTimeline = document.getElementById("overviewTimeline");
    const modelTimeline = document.getElementById("modelTimeline");
    const manualTimeline = document.getElementById("manualTimeline");
    const timelineWindowSelect = document.getElementById("timelineWindowSelect");
    const timelineWindowStatus = document.getElementById("timelineWindowStatus");
    const followWindowCheckbox = document.getElementById("followWindowCheckbox");
    const usePlayheadDragStartCheckbox = document.getElementById("usePlayheadDragStartCheckbox");
    const labelSelect = document.getElementById("labelSelect");
    const confidenceSelect = document.getElementById("confidenceSelect");
    const saveStatus = document.getElementById("saveStatus");
    const syncStatus = document.getElementById("syncStatus");
    const pendingPanel = document.getElementById("pendingPanel");
    const pendingRangeText = document.getElementById("pendingRangeText");
    const pendingRangeHint = document.getElementById("pendingRangeHint");
    const pendingActionButtons = document.getElementById("pendingActionButtons");
    const segmentSelect = document.getElementById("segmentSelect");
    const segmentLabelSelect = document.getElementById("segmentLabelSelect");
    const segmentConfidenceSelect = document.getElementById("segmentConfidenceSelect");
    const startInput = document.getElementById("startInput");
    const endInput = document.getElementById("endInput");
    const noteInput = document.getElementById("noteInput");
    const editorStatus = document.getElementById("editorStatus");
    const segmentList = document.getElementById("segmentList");

    let manualSegments = [];
    let selectedId = "";
    let quickStartS = null;
    let dragStartS = null;
    let dragStartClientX = 0;
    let dragPointerId = null;
    let dragStartedOnSegmentId = "";
    let dragHasMoved = false;
    let dragPreviewEls = [];
    let pendingSelection = null;
    let suppressTimelineClickUntil = 0;
    let saveTimer = null;
    let downwardTurns = 2;
    let ariaTurns = 1;
    let viewStartS = 0;
    let viewEndS = duration;
    let syncAnimationId = null;
    let lastSyncStatusUpdateMs = 0;
    const MIN_DRAG_LABEL_S = 0.1;
    const DRAG_PIXEL_THRESHOLD = 4;
    const SNAP_BOUNDARY_S = 0.25;
    const SOFT_SYNC_THRESHOLD_S = 0.025;
    const HARD_SYNC_THRESHOLD_S = 0.12;

    function clamp(value, min, max) {
      return Math.min(Math.max(Number(value) || 0, min), max);
    }
    function fmt(seconds) {
      const s = Math.max(0, Number(seconds) || 0);
      const m = Math.floor(s / 60);
      const rest = s - m * 60;
      return m > 0 ? `${m}:${rest.toFixed(1).padStart(4, "0")}` : `${rest.toFixed(1)}s`;
    }
    function viewDuration() {
      return Math.max(0.001, viewEndS - viewStartS);
    }
    function timelineWindowSize() {
      const value = timelineWindowSelect.value;
      return value === "full" ? duration : Math.min(duration, Number(value) || duration);
    }
    function pctFull(start, end) {
      const left = duration > 0 ? (start / duration) * 100 : 0;
      const width = duration > 0 ? ((end - start) / duration) * 100 : 0;
      return { left: `${left}%`, width: `${Math.max(0, width)}%` };
    }
    function pct(start, end) {
      const clippedStart = Math.max(start, viewStartS);
      const clippedEnd = Math.min(end, viewEndS);
      if (clippedEnd <= clippedStart) return null;
      const left = ((clippedStart - viewStartS) / viewDuration()) * 100;
      const width = ((clippedEnd - clippedStart) / viewDuration()) * 100;
      return { left: `${left}%`, width: `${Math.max(0, width)}%` };
    }
    function setTimelineWindow(centerS, sizeS = timelineWindowSize(), rerender = true) {
      const size = Math.min(duration, Math.max(1, Number(sizeS) || duration));
      if (size >= duration - 0.001) {
        viewStartS = 0;
        viewEndS = duration;
      } else {
        const center = clamp(centerS, 0, duration);
        viewStartS = clamp(center - size / 2, 0, Math.max(0, duration - size));
        viewEndS = Math.min(duration, viewStartS + size);
      }
      updateTimelineWindowStatus();
      if (rerender) renderAll();
    }
    function panTimelineWindow(direction) {
      const shift = viewDuration() * 0.8 * direction;
      setTimelineWindow((viewStartS + viewEndS) / 2 + shift, viewDuration(), true);
    }
    function ensureTimeInWindow(timeS, rerender = true) {
      if (timelineWindowSelect.value === "full") return;
      const margin = viewDuration() * 0.18;
      if (timeS < viewStartS + margin || timeS > viewEndS - margin) {
        setTimelineWindow(timeS, viewDuration(), rerender);
      }
    }
    function updateTimelineWindowStatus() {
      timelineWindowStatus.textContent = `Showing ${fmt(viewStartS)} - ${fmt(viewEndS)} (${fmt(viewDuration())})`;
    }
    function escapeHtml(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    }
    function normalizeAction(label) {
      const raw = String(label || "").trim().toLowerCase().replaceAll("_", "-");
      if (raw === "sit") return "sitting";
      if (raw === "stand") return "standing";
      if (raw === "walk") return "walking";
      if (raw === "go upstairs") return "upstairs";
      if (raw === "go downstairs") return "downstairs";
      return raw;
    }
    function modelActions(label) {
      const raw = String(label || "").toLowerCase().replaceAll("_", "-");
      const found = [];
      for (const action of actions) {
        const stem = action === "sitting" ? "sit" : action === "standing" ? "stand" : action === "walking" ? "walk" : action;
        if (raw.includes(action) || raw.includes(stem)) found.push(action);
      }
      const single = normalizeAction(raw);
      if (!found.length && actions.includes(single)) found.push(single);
      return [...new Set(found)];
    }
    function segmentAtTime(segments, timeS) {
      return segments.find(seg => timeS >= seg.start_s && timeS < seg.end_s) || null;
    }
    function seek(timeS, play = false) {
      const target = clamp(timeS, 0, duration);
      downwardVideo.currentTime = clamp(target, 0, Math.max(0, downwardDuration - 0.05));
      ariaVideo.currentTime = target;
      syncFollower(true);
      if (followWindowCheckbox.checked) ensureTimeInWindow(target, true);
      if (play) playBoth();
      renderPlayheads(target);
      renderMetrics();
    }
    async function playBoth() {
      syncFollower(true);
      try {
        const playPromises = [ariaVideo.play()];
        if ((ariaVideo.currentTime || 0) <= Math.max(0, downwardDuration - 0.05)) {
          playPromises.push(downwardVideo.play());
        }
        const results = await Promise.allSettled(playPromises);
        if (results[0]?.status === "rejected") throw results[0].reason;
        syncFollower(true);
        startSyncLoop();
        if (saveStatus.textContent.startsWith("Playback blocked")) {
          saveStatus.textContent = "Playback ready.";
        }
      } catch (error) {
        pauseBoth();
        saveStatus.textContent = "Playback blocked: click Play or the video once, then Space should work.";
      }
    }
    function pauseBoth() {
      ariaVideo.pause();
      downwardVideo.pause();
      stopSyncLoop();
    }
    function togglePlayback() {
      if (!ariaVideo.paused || !downwardVideo.paused) {
        pauseBoth();
      } else {
        playBoth();
      }
    }
    function syncFollower(force = false) {
      const target = clamp(ariaVideo.currentTime, 0, duration);
      const downwardMaxTime = Math.max(0, downwardDuration - 0.05);
      if (target > downwardMaxTime) {
        downwardVideo.pause();
        downwardVideo.playbackRate = ariaVideo.playbackRate || 1;
        updateSyncStatus(0, "ended");
        return;
      }
      const delta = Number(downwardVideo.currentTime || 0) - target;
      const absDelta = Math.abs(delta);
      const baseRate = Number(ariaVideo.playbackRate || 1);
      if (force || !Number.isFinite(delta) || absDelta > HARD_SYNC_THRESHOLD_S) {
        try {
          downwardVideo.currentTime = clamp(target, 0, downwardMaxTime);
        } catch (error) {}
        downwardVideo.playbackRate = baseRate;
        updateSyncStatus(
          delta,
          Number.isFinite(delta) && absDelta <= SOFT_SYNC_THRESHOLD_S ? "good" : "adjusting",
          true,
        );
        return;
      }
      if (!downwardVideo.seeking && absDelta > SOFT_SYNC_THRESHOLD_S) {
        const rateCorrection = clamp(-delta * 0.45, -0.05, 0.05);
        downwardVideo.playbackRate = baseRate * (1 + rateCorrection);
        updateSyncStatus(delta, "adjusting");
      } else {
        downwardVideo.playbackRate = baseRate;
        updateSyncStatus(delta, "good");
      }
    }
    function updateSyncStatus(deltaS, state, force = false) {
      const now = performance.now();
      if (!force && now - lastSyncStatusUpdateMs < 180) return;
      lastSyncStatusUpdateMs = now;
      syncStatus.className = `sync-status ${state}`;
      if (state === "ended") {
        syncStatus.textContent = "Downward recording ended; Aria continues";
        return;
      }
      const deltaMs = Math.round((Number(deltaS) || 0) * 1000);
      syncStatus.textContent = state === "good"
        ? `Synchronized · downward ${deltaMs >= 0 ? "+" : ""}${deltaMs} ms`
        : `Correcting sync · downward ${deltaMs >= 0 ? "+" : ""}${deltaMs} ms`;
    }
    function syncAnimationFrame() {
      syncAnimationId = null;
      if (!ariaVideo.paused && !ariaVideo.ended) {
        syncFollower(false);
        syncAnimationId = window.requestAnimationFrame(syncAnimationFrame);
      }
    }
    function startSyncLoop() {
      if (syncAnimationId == null) {
        syncAnimationId = window.requestAnimationFrame(syncAnimationFrame);
      }
    }
    function stopSyncLoop() {
      if (syncAnimationId != null) {
        window.cancelAnimationFrame(syncAnimationId);
        syncAnimationId = null;
      }
      downwardVideo.playbackRate = ariaVideo.playbackRate || 1;
    }
    function pointerTime(event, el, fullSession = false) {
      const rect = el.getBoundingClientRect();
      const ratio = (event.clientX - rect.left) / Math.max(1, rect.width);
      const start = fullSession ? 0 : viewStartS;
      const span = fullSession ? duration : viewDuration();
      return clamp(start + ratio * span, 0, duration);
    }
    function clearDragPreview() {
      for (const el of dragPreviewEls) el.remove();
      dragPreviewEls = [];
    }
    function renderDragPreview(startS, endS) {
      clearDragPreview();
      const ranges = availableRangesWithin(startS, endS);
      for (const range of ranges) {
        const el = document.createElement("div");
        const pos = pct(range.start_s, range.end_s);
        if (!pos) continue;
        el.className = "drag-preview";
        el.style.left = pos.left;
        el.style.width = pos.width;
        el.style.background = colors[labelSelect.value] || "rgba(22,122,85,.18)";
        el.style.opacity = ".66";
        manualTimeline.appendChild(el);
        dragPreviewEls.push(el);
      }
    }
    function normalizeSegment(raw, idx = 0) {
      const start = clamp(raw.start_s, 0, duration);
      const end = clamp(raw.end_s, 0, duration);
      const label = actions.includes(normalizeAction(raw.motion_label)) ? normalizeAction(raw.motion_label) : "walking";
      const confidence = ["high", "medium", "low", "uncertain"].includes(String(raw.confidence || "").toLowerCase()) ? String(raw.confidence).toLowerCase() : "medium";
      const now = new Date().toISOString();
      return {
        id: String(raw.id || `manual_${Date.now()}_${idx}_${Math.random().toString(16).slice(2)}`),
        start_s: Number(Math.min(start, end).toFixed(3)),
        end_s: Number(Math.max(start, end).toFixed(3)),
        motion_label: label,
        confidence,
        note: String(raw.note || ""),
        source: "manual_session",
        created_at: String(raw.created_at || now),
        updated_at: String(raw.updated_at || now)
      };
    }
    function sortSegments() {
      manualSegments = manualSegments
        .map(normalizeSegment)
        .filter(seg => seg.end_s - seg.start_s >= 0.1)
        .sort((a, b) => (a.start_s - b.start_s) || (a.end_s - b.end_s));
    }
    function segmentBoundaries() {
      const boundaries = [0, duration];
      for (const seg of manualSegments) {
        boundaries.push(seg.start_s, seg.end_s);
      }
      return boundaries
        .filter(value => Number.isFinite(value))
        .sort((a, b) => a - b);
    }
    function snapTimeToBoundary(timeS, boundaries = segmentBoundaries()) {
      let best = timeS;
      let bestDistance = SNAP_BOUNDARY_S;
      for (const boundary of boundaries) {
        const distance = Math.abs(timeS - boundary);
        if (distance <= bestDistance) {
          best = boundary;
          bestDistance = distance;
        }
      }
      return clamp(best, 0, duration);
    }
    function snapRangeToNearbyBoundaries(startS, endS) {
      const boundaries = segmentBoundaries();
      let start = snapTimeToBoundary(startS, boundaries);
      let end = snapTimeToBoundary(endS, boundaries);
      if (Math.abs(end - start) < MIN_DRAG_LABEL_S) {
        start = startS;
        end = endS;
      }
      return {
        start_s: clamp(Math.min(start, end), 0, duration),
        end_s: clamp(Math.max(start, end), 0, duration)
      };
    }
    function availableRangesWithin(startS, endS) {
      sortSegments();
      let start = clamp(Math.min(startS, endS), 0, duration);
      let end = clamp(Math.max(startS, endS), 0, duration);
      ({ start_s: start, end_s: end } = snapRangeToNearbyBoundaries(start, end));
      if (end - start < MIN_DRAG_LABEL_S) return [];
      const ranges = [];
      let cursor = start;
      for (const seg of manualSegments) {
        if (seg.end_s <= start || seg.start_s >= end) continue;
        if (seg.start_s > cursor + MIN_DRAG_LABEL_S) {
          ranges.push({
            start_s: Number(cursor.toFixed(3)),
            end_s: Number(Math.min(seg.start_s, end).toFixed(3))
          });
        }
        cursor = Math.max(cursor, seg.end_s);
      }
      if (cursor < end - MIN_DRAG_LABEL_S) {
        ranges.push({
          start_s: Number(cursor.toFixed(3)),
          end_s: Number(end.toFixed(3))
        });
      }
      return ranges.filter(range => range.end_s - range.start_s >= MIN_DRAG_LABEL_S);
    }
    function describeRanges(ranges) {
      if (!ranges.length) return "No unlabeled time selected.";
      if (ranges.length === 1) return `${fmt(ranges[0].start_s)} - ${fmt(ranges[0].end_s)}`;
      return `${ranges.length} unlabeled parts: ${ranges.map(range => `${fmt(range.start_s)}-${fmt(range.end_s)}`).join(", ")}`;
    }
    function clearPendingSelection(render = true) {
      pendingSelection = null;
      if (render) renderAll();
    }
    function setPendingSelection(startS, endS) {
      const ranges = availableRangesWithin(startS, endS);
      if (!ranges.length) {
        pendingSelection = null;
        saveStatus.textContent = "That drag range only covered existing labels. Delete or edit an existing label if you want to change it.";
        renderAll();
        return;
      }
      pendingSelection = {
        start_s: clamp(Math.min(startS, endS), 0, duration),
        end_s: clamp(Math.max(startS, endS), 0, duration),
        ranges
      };
      saveStatus.textContent = "Range selected. Choose a motion label to save it.";
      renderAll();
    }
    function applyPendingLabel(label) {
      if (!pendingSelection) return;
      const selection = pendingSelection;
      pendingSelection = null;
      const created = addManualSegment(selection.start_s, selection.end_s, label, confidenceSelect.value, "");
      if (created.length) {
        saveStatus.textContent = `Added ${label} label for ${describeRanges(created)}.`;
      }
    }
    function renderPendingPanel() {
      if (!pendingSelection) {
        pendingPanel.classList.add("hidden");
        pendingRangeText.textContent = "No range selected.";
        pendingRangeHint.textContent = "";
        return;
      }
      pendingPanel.classList.remove("hidden");
      pendingRangeText.textContent = `Selected: ${describeRanges(pendingSelection.ranges)}`;
      pendingRangeHint.textContent = pendingSelection.ranges.length > 1
        ? "Existing labels inside your drag were skipped."
        : "Choose a motion below to save this range.";
    }
    function snapNewSegmentsToNeighbors(newSegments) {
      const ids = new Set(newSegments.map(seg => seg.id));
      if (!ids.size) return;
      sortSegments();
      const now = new Date().toISOString();
      for (let i = 0; i < manualSegments.length; i += 1) {
        const seg = manualSegments[i];
        if (!ids.has(seg.id)) continue;
        let start = seg.start_s;
        let end = seg.end_s;
        const prev = manualSegments[i - 1];
        const next = manualSegments[i + 1];
        if (start > 0 && start <= SNAP_BOUNDARY_S) start = 0;
        if (prev) {
          const gapBefore = start - prev.end_s;
          if (gapBefore > 0 && gapBefore <= SNAP_BOUNDARY_S) start = prev.end_s;
        }
        if (duration - end > 0 && duration - end <= SNAP_BOUNDARY_S) end = duration;
        if (next) {
          const gapAfter = next.start_s - end;
          if (gapAfter > 0 && gapAfter <= SNAP_BOUNDARY_S) end = next.start_s;
        }
        manualSegments[i] = {
          ...seg,
          start_s: Number(start.toFixed(3)),
          end_s: Number(end.toFixed(3)),
          updated_at: now
        };
      }
      sortSegments();
    }
    function gaps() {
      sortSegments();
      const result = [];
      let cursor = 0;
      for (const seg of manualSegments) {
        if (seg.start_s > cursor + 0.1) result.push({ start_s: cursor, end_s: seg.start_s });
        cursor = Math.max(cursor, seg.end_s);
      }
      if (cursor < duration - 0.1) result.push({ start_s: cursor, end_s: duration });
      return result;
    }
    function addManualSegment(startS, endS, label, confidence = "medium", note = "") {
      const ranges = availableRangesWithin(startS, endS);
      if (!ranges.length) {
        saveStatus.textContent = "No unlabeled time in that drag range; existing labels were kept.";
        return [];
      }
      const now = new Date().toISOString();
      const newSegments = ranges.map((range, idx) => normalizeSegment({
        id: `manual_${Date.now()}_${idx}_${Math.random().toString(16).slice(2)}`,
        start_s: range.start_s,
        end_s: range.end_s,
        motion_label: label,
        confidence,
        note,
        created_at: now,
        updated_at: now
      }));
      manualSegments = manualSegments.concat(newSegments);
      snapNewSegmentsToNeighbors(newSegments);
      selectedId = newSegments[0]?.id || selectedId;
      sortSegments();
      renderAll();
      scheduleSave();
      return newSegments;
    }
    function selectedSegment() {
      return manualSegments.find(seg => seg.id === selectedId) || null;
    }
    function deleteManualSegment(id) {
      if (!id) return;
      const deleted = manualSegments.find(seg => seg.id === id);
      manualSegments = manualSegments.filter(seg => seg.id !== id);
      pendingSelection = null;
      if (selectedId === id) {
        selectedId = manualSegments[0]?.id || "";
      }
      renderAll();
      scheduleSave();
      saveStatus.textContent = deleted
        ? `Deleted ${deleted.motion_label} label (${fmt(deleted.start_s)} - ${fmt(deleted.end_s)}).`
        : "Deleted selected label.";
    }
    function relabelSelectedSegment(label) {
      const seg = selectedSegment();
      const normalized = normalizeAction(label);
      if (!seg || !actions.includes(normalized)) return;
      const now = new Date().toISOString();
      manualSegments = manualSegments.map(item => {
        if (item.id !== seg.id) return item;
        return {
          ...item,
          motion_label: normalized,
          updated_at: now
        };
      });
      segmentLabelSelect.value = normalized;
      pendingSelection = null;
      renderAll();
      scheduleSave();
      saveStatus.textContent = `Changed selected label to ${normalized} (${fmt(seg.start_s)} - ${fmt(seg.end_s)}).`;
    }
    function scheduleSave() {
      saveStatus.textContent = "Unsaved changes...";
      window.clearTimeout(saveTimer);
      saveTimer = window.setTimeout(() => saveManualSegments(true), 800);
    }
    function localKey() {
      return `pd-insighter-session-manual:${trialId}`;
    }
    async function loadManualSegments() {
      try {
        const response = await fetch(`/api/session-manual-segments?trial_id=${encodeURIComponent(trialId)}`, { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        manualSegments = (payload.segments || []).map(normalizeSegment);
        window.localStorage.setItem(localKey(), JSON.stringify(manualSegments));
        saveStatus.textContent = manualSegments.length ? "Manual labels loaded from file" : "No manual labels yet";
      } catch (error) {
        const locallySaved = JSON.parse(window.localStorage.getItem(localKey()) || "[]");
        const initialSegments = data.initial_manual_segments || [];
        manualSegments = (locallySaved.length ? locallySaved : initialSegments).map(normalizeSegment);
        saveStatus.textContent = locallySaved.length
          ? "Manual labels loaded locally"
          : (manualSegments.length ? "Saved manual labels loaded from this static demo" : `Manual labels unavailable: ${error}`);
      }
      sortSegments();
      selectedId = manualSegments[0]?.id || "";
      renderAll();
    }
    async function saveManualSegments(silent = false) {
      sortSegments();
      window.localStorage.setItem(localKey(), JSON.stringify(manualSegments));
      if (!silent) saveStatus.textContent = "Saving...";
      try {
        const response = await fetch("/api/session-manual-segments", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ trial_id: trialId, segments: manualSegments })
        });
        if (!response.ok) {
          const err = await response.json().catch(() => ({}));
          throw new Error(err.error || `HTTP ${response.status}`);
        }
        const payload = await response.json();
        manualSegments = (payload.segments || []).map(normalizeSegment);
        sortSegments();
        window.localStorage.setItem(localKey(), JSON.stringify(manualSegments));
        saveStatus.textContent = `Saved ${manualSegments.length} segment(s) to file`;
      } catch (error) {
        saveStatus.textContent = `Saved locally; file save failed: ${error}`;
      }
      renderAll();
    }
    function renderOverviewTimeline() {
      overviewTimeline.innerHTML = "";
      for (const seg of modelSegments) {
        const el = document.createElement("div");
        const pos = pctFull(seg.start_s, seg.end_s);
        el.className = "overview-seg overview-model-seg";
        el.style.left = pos.left;
        el.style.width = pos.width;
        const acts = modelActions(seg.label);
        el.style.background = acts.length === 1 ? colors[acts[0]] : String(seg.color || "#b6bec6");
        el.title = `Model: ${seg.label} | ${fmt(seg.start_s)} - ${fmt(seg.end_s)}`;
        overviewTimeline.appendChild(el);
      }
      for (const seg of manualSegments) {
        const el = document.createElement("div");
        const pos = pctFull(seg.start_s, seg.end_s);
        el.className = "overview-seg overview-manual-seg";
        el.style.left = pos.left;
        el.style.width = pos.width;
        el.style.background = colors[seg.motion_label] || colors.missing;
        el.title = `Manual: ${seg.motion_label} | ${fmt(seg.start_s)} - ${fmt(seg.end_s)}`;
        overviewTimeline.appendChild(el);
      }
      const windowEl = document.createElement("div");
      const pos = pctFull(viewStartS, viewEndS);
      windowEl.className = "overview-window";
      windowEl.style.left = pos.left;
      windowEl.style.width = pos.width;
      overviewTimeline.appendChild(windowEl);
    }
    function renderModelTimeline() {
      modelTimeline.innerHTML = "";
      for (const seg of modelSegments) {
        const el = document.createElement("div");
        const pos = pct(seg.start_s, seg.end_s);
        if (!pos) continue;
        el.className = "model-seg";
        el.style.left = pos.left;
        el.style.width = pos.width;
        const acts = modelActions(seg.label);
        el.style.background = acts.length === 1 ? colors[acts[0]] : String(seg.color || "#b6bec6");
        el.title = `${seg.label} | ${fmt(seg.start_s)} - ${fmt(seg.end_s)}`;
        el.addEventListener("pointerdown", event => {
          event.preventDefault();
          seek(pointerTime(event, modelTimeline), true);
        });
        modelTimeline.appendChild(el);
      }
    }
    function renderManualTimeline() {
      manualTimeline.innerHTML = "";
      for (const gap of gaps()) {
        const el = document.createElement("div");
        const pos = pct(gap.start_s, gap.end_s);
        if (!pos) continue;
        el.className = "gap-seg";
        el.style.left = pos.left;
        el.style.width = pos.width;
        el.title = `Unlabeled gap | ${fmt(gap.start_s)} - ${fmt(gap.end_s)}`;
        el.addEventListener("click", event => {
          if (Date.now() < suppressTimelineClickUntil) return;
          event.stopPropagation();
          selectedId = "";
          seek(gap.start_s, false);
          editorStatus.textContent = `Gap selected: ${fmt(gap.start_s)} - ${fmt(gap.end_s)}. Use Fill Current Gap.`;
        });
        manualTimeline.appendChild(el);
      }
      if (pendingSelection) {
        for (const range of pendingSelection.ranges) {
          const pos = pct(range.start_s, range.end_s);
          if (!pos) continue;
          const el = document.createElement("div");
          el.className = "pending-seg";
          el.style.left = pos.left;
          el.style.width = pos.width;
          el.title = `Selected unlabeled range | ${fmt(range.start_s)} - ${fmt(range.end_s)}`;
          manualTimeline.appendChild(el);
        }
      }
      for (const seg of manualSegments) {
        const el = document.createElement("div");
        const pos = pct(seg.start_s, seg.end_s);
        if (!pos) continue;
        el.className = `manual-seg${seg.id === selectedId ? " active" : ""}`;
        el.dataset.id = seg.id;
        el.style.left = pos.left;
        el.style.width = pos.width;
        el.style.background = colors[seg.motion_label] || colors.missing;
        el.innerHTML = `
          <span>${escapeHtml(seg.motion_label)} · ${fmt(seg.start_s)}-${fmt(seg.end_s)}</span>
          <button class="icon-button danger manual-delete" type="button" aria-label="Delete ${escapeHtml(seg.motion_label)} label" title="Delete this label">×</button>
        `;
        el.title = `${seg.motion_label} | ${fmt(seg.start_s)} - ${fmt(seg.end_s)} | ${seg.confidence}`;
        const deleteButton = el.querySelector(".manual-delete");
        deleteButton.addEventListener("pointerdown", event => {
          event.stopPropagation();
        });
        deleteButton.addEventListener("click", event => {
          event.stopPropagation();
          deleteManualSegment(seg.id);
        });
        el.addEventListener("click", event => {
          if (Date.now() < suppressTimelineClickUntil) return;
          event.stopPropagation();
          selectedId = seg.id;
          seek(seg.start_s, false);
          renderAll();
        });
        manualTimeline.appendChild(el);
      }
      renderPlayheads(Number(ariaVideo.currentTime || 0));
    }
    function renderPlayheads(timeS) {
      const oldOverviewPlayhead = overviewTimeline.querySelector(".playhead");
      if (oldOverviewPlayhead) oldOverviewPlayhead.remove();
      const overviewPlayhead = document.createElement("div");
      overviewPlayhead.className = "playhead";
      overviewPlayhead.style.left = `${duration > 0 ? (clamp(timeS, 0, duration) / duration) * 100 : 0}%`;
      overviewTimeline.appendChild(overviewPlayhead);
      for (const tl of [modelTimeline, manualTimeline]) {
        const old = tl.querySelector(".playhead");
        if (old) old.remove();
        if (timeS < viewStartS || timeS > viewEndS) continue;
        const playhead = document.createElement("div");
        playhead.className = "playhead";
        playhead.style.left = `${((clamp(timeS, viewStartS, viewEndS) - viewStartS) / viewDuration()) * 100}%`;
        tl.appendChild(playhead);
      }
      scrubber.value = String(clamp(timeS, 0, duration));
      document.getElementById("timeNow").textContent = fmt(timeS);
    }
    function renderSegmentEditor() {
      segmentSelect.innerHTML = "";
      for (const seg of manualSegments) {
        const option = document.createElement("option");
        option.value = seg.id;
        option.textContent = `${seg.motion_label} | ${fmt(seg.start_s)} - ${fmt(seg.end_s)}`;
        option.selected = seg.id === selectedId;
        segmentSelect.appendChild(option);
      }
      const seg = selectedSegment();
      if (!seg) {
        startInput.value = "";
        endInput.value = "";
        noteInput.value = "";
        editorStatus.textContent = "No segment selected yet.";
        document.querySelectorAll("[data-relabel-action]").forEach(btn => {
          btn.disabled = true;
          btn.classList.remove("active");
        });
        return;
      }
      startInput.value = seg.start_s.toFixed(2);
      endInput.value = seg.end_s.toFixed(2);
      segmentLabelSelect.value = seg.motion_label;
      segmentConfidenceSelect.value = seg.confidence;
      noteInput.value = seg.note || "";
      document.querySelectorAll("[data-relabel-action]").forEach(btn => {
        const label = btn.getAttribute("data-relabel-action") || "";
        btn.disabled = false;
        btn.classList.toggle("active", label === seg.motion_label);
      });
      editorStatus.textContent = `Editing ${seg.motion_label}, ${fmt(seg.start_s)} - ${fmt(seg.end_s)}.`;
    }
    function renderSegmentList() {
      if (!manualSegments.length) {
        segmentList.innerHTML = `<p class="small">No manual segments yet. Drag on the manual timeline or use Start Label Here / End Here + Save.</p>`;
        return;
      }
      segmentList.innerHTML = manualSegments.map(seg => `
        <div class="segment-row">
          <button class="segment-main" type="button" data-id="${escapeHtml(seg.id)}" style="background:${colors[seg.motion_label] || colors.missing};">
            <b>${escapeHtml(seg.motion_label)}</b> <span class="mono">${fmt(seg.start_s)} - ${fmt(seg.end_s)}</span> <span>${escapeHtml(seg.confidence)}</span>
          </button>
          <button class="danger segment-list-delete" type="button" data-delete-id="${escapeHtml(seg.id)}" title="Delete this label">Delete</button>
        </div>
      `).join("");
      segmentList.querySelectorAll("button[data-id]").forEach(btn => {
        btn.addEventListener("click", () => {
          selectedId = btn.getAttribute("data-id") || "";
          const seg = selectedSegment();
          if (seg) seek(seg.start_s, false);
          renderAll();
        });
      });
      segmentList.querySelectorAll("button[data-delete-id]").forEach(btn => {
        btn.addEventListener("click", () => {
          deleteManualSegment(btn.getAttribute("data-delete-id") || "");
        });
      });
    }
    function renderLegend() {
      document.getElementById("legend").innerHTML = actions.map(action => (
        `<span><i class="swatch" style="background:${colors[action]}"></i>${action}</span>`
      )).join("") + `<span><i class="swatch" style="background:repeating-linear-gradient(135deg, rgba(185,67,56,.18), rgba(185,67,56,.18) 8px, rgba(185,67,56,.06) 8px, rgba(185,67,56,.06) 16px);"></i>unlabeled gap</span>`;
    }
    function computeAgreement() {
      let covered = 0;
      let exact = 0;
      let transitionAware = 0;
      for (const manual of manualSegments) {
        covered += manual.end_s - manual.start_s;
        for (const model of modelSegments) {
          const start = Math.max(manual.start_s, model.start_s);
          const end = Math.min(manual.end_s, model.end_s);
          if (end <= start) continue;
          const overlap = end - start;
          const acts = modelActions(model.label);
          if (normalizeAction(model.label) === manual.motion_label) exact += overlap;
          if (acts.includes(manual.motion_label)) transitionAware += overlap;
        }
      }
      return {
        covered,
        coverage: duration > 0 ? covered / duration : 0,
        gapCount: gaps().length,
        exact: covered > 0 ? exact / covered : null,
        transitionAware: covered > 0 ? transitionAware / covered : null
      };
    }
    function percent(value) {
      return value == null ? "--" : `${(value * 100).toFixed(1)}%`;
    }
    function renderMetrics() {
      const m = computeAgreement();
      document.getElementById("coverageMetric").textContent = percent(m.coverage);
      document.getElementById("gapMetric").textContent = String(m.gapCount);
      document.getElementById("transitionAwareMetric").textContent = percent(m.transitionAware);
      document.getElementById("exactMetric").textContent = percent(m.exact);
    }
    function renderAll() {
      sortSegments();
      renderOverviewTimeline();
      renderModelTimeline();
      renderManualTimeline();
      renderSegmentEditor();
      renderSegmentList();
      renderLegend();
      renderMetrics();
      renderPendingPanel();
      updateTimelineWindowStatus();
    }
    function saveSelectedSegmentFromEditor() {
      const seg = selectedSegment();
      if (!seg) return;
      const start = clamp(Number(startInput.value), 0, duration);
      const end = clamp(Number(endInput.value), 0, duration);
      const newStart = Math.min(start, end);
      const newEnd = Math.max(start, end);
      const newLabel = normalizeAction(segmentLabelSelect.value);
      const confidence = segmentConfidenceSelect.value || "medium";
      const note = noteInput.value || "";
      const id = seg.id;
      const now = new Date().toISOString();
      if (Math.abs(newStart - seg.start_s) < 0.001 && Math.abs(newEnd - seg.end_s) < 0.001) {
        manualSegments = manualSegments.map(item => {
          if (item.id !== id) return item;
          return {
            ...item,
            motion_label: newLabel,
            confidence,
            note,
            updated_at: now
          };
        });
        pendingSelection = null;
        renderAll();
        scheduleSave();
        saveStatus.textContent = `Updated selected segment (${fmt(newStart)} - ${fmt(newEnd)}).`;
        return;
      }
      const previousSegments = manualSegments.slice();
      manualSegments = manualSegments.filter(item => item.id !== id);
      pendingSelection = null;
      const created = addManualSegment(newStart, newEnd, newLabel, confidence, note);
      if (!created.length) {
        manualSegments = previousSegments;
        selectedId = id;
        renderAll();
        saveStatus.textContent = "Could not move this segment because the new range is already labeled.";
        return;
      }
      selectedId = created[0].id;
      renderAll();
      scheduleSave();
    }
    function currentGap() {
      const time = Number(ariaVideo.currentTime || 0);
      return gaps().find(gap => time >= gap.start_s - 0.001 && time <= gap.end_s + 0.001) || gaps()[0] || null;
    }
    function bindDragLabeling() {
      manualTimeline.addEventListener("pointerdown", event => {
        event.preventDefault();
        event.stopPropagation();
        const pointerStart = pointerTime(event, manualTimeline);
        const playheadStart = clamp(ariaVideo.currentTime || 0, 0, duration);
        const canStartFromPlayhead = usePlayheadDragStartCheckbox.checked
          && playheadStart >= viewStartS - 0.001
          && playheadStart <= viewEndS + 0.001;
        dragStartS = canStartFromPlayhead ? playheadStart : pointerStart;
        if (usePlayheadDragStartCheckbox.checked && !canStartFromPlayhead) {
          saveStatus.textContent = "Playhead is outside this zoom window, so drag started from the mouse point.";
        }
        dragStartClientX = event.clientX;
        dragPointerId = event.pointerId;
        dragStartedOnSegmentId = event.target.closest(".manual-seg")?.dataset.id || "";
        dragHasMoved = false;
        renderDragPreview(dragStartS, dragStartS);
        try {
          manualTimeline.setPointerCapture(event.pointerId);
        } catch {
          // Pointer capture is a convenience; dragging still works without it.
        }
      });
      manualTimeline.addEventListener("pointermove", event => {
        if (dragPointerId != null && event.pointerId !== dragPointerId) return;
        if (dragStartS == null) return;
        const now = pointerTime(event, manualTimeline);
        if (Math.abs(event.clientX - dragStartClientX) >= DRAG_PIXEL_THRESHOLD || Math.abs(now - dragStartS) >= MIN_DRAG_LABEL_S) {
          dragHasMoved = true;
        }
        renderDragPreview(dragStartS, now);
      });
      manualTimeline.addEventListener("pointerup", event => {
        if (dragPointerId != null && event.pointerId !== dragPointerId) return;
        if (dragStartS == null) return;
        const end = pointerTime(event, manualTimeline);
        clearDragPreview();
        const start = dragStartS;
        const clickedTime = end;
        const clickedSegmentId = dragStartedOnSegmentId;
        const shouldCreateLabel = dragHasMoved && Math.abs(end - start) >= MIN_DRAG_LABEL_S;
        dragStartS = null;
        dragPointerId = null;
        dragStartedOnSegmentId = "";
        dragHasMoved = false;
        if (shouldCreateLabel) {
          suppressTimelineClickUntil = Date.now() + 350;
          setPendingSelection(start, end);
        } else if (clickedSegmentId) {
          selectedId = clickedSegmentId;
          const seg = selectedSegment();
          if (seg) seek(seg.start_s, false);
          renderAll();
        } else {
          seek(clickedTime, false);
        }
      });
      manualTimeline.addEventListener("pointercancel", () => {
        clearDragPreview();
        dragStartS = null;
        dragPointerId = null;
        dragStartedOnSegmentId = "";
        dragHasMoved = false;
      });
    }
    function applyRotation(video, turns) {
      const n = ((turns % 4) + 4) % 4;
      video.style.transform = n % 2 === 1
        ? `translate(-50%, -50%) rotate(${n * 90}deg)`
        : `rotate(${n * 90}deg)`;
      video.classList.toggle("rotated-quarter", n % 2 === 1);
      if (n % 2 === 1) {
        const frame = video.parentElement;
        const rect = frame ? frame.getBoundingClientRect() : null;
        if (rect && rect.width > 0 && rect.height > 0) {
          // Size from the natural video aspect before rotating. This keeps the
          // rotated video fully visible instead of stretching it into the frame.
          const naturalW = video.videoWidth || rect.height;
          const naturalH = video.videoHeight || rect.width;
          const scale = Math.min(rect.width / naturalH, rect.height / naturalW);
          video.style.setProperty("--rotated-video-width", `${naturalW * scale}px`);
          video.style.setProperty("--rotated-video-height", `${naturalH * scale}px`);
        }
      } else {
        video.style.removeProperty("--rotated-video-width");
        video.style.removeProperty("--rotated-video-height");
      }
    }
    function init() {
      document.getElementById("trialPill").textContent = trialId;
      document.getElementById("fullReviewLink").href = data.full_review_href || "index.html";
      document.getElementById("strideReviewLink").href = data.stride_review_href || "pd_insighter_stride_review.html";
      const transitionReviewLink = document.getElementById("transitionReviewLink");
      if (data.transition_review_href) {
        transitionReviewLink.href = data.transition_review_href;
      } else {
        transitionReviewLink.hidden = true;
      }
      const trialJumpSelect = document.getElementById("trialJumpSelect");
      (data.trial_jump_options || []).forEach(item => {
        const option = document.createElement("option");
        option.value = item.href;
        option.textContent = item.trial_id === "__selector__" ? "All Trials" : item.label;
        option.selected = item.trial_id === trialId;
        trialJumpSelect.appendChild(option);
      });
      trialJumpSelect.addEventListener("change", event => {
        if (event.target.value) window.location.href = event.target.value;
      });
      document.getElementById("timeTotal").textContent = fmt(duration);
      downwardVideo.src = data.downward_session_video_uri;
      ariaVideo.src = data.aria_session_video_uri;
      scrubber.max = String(duration);
      scrubber.step = "0.05";
      if (duration <= 120) {
        timelineWindowSelect.value = "full";
      }
      setTimelineWindow(0, timelineWindowSize(), false);
      applyRotation(downwardVideo, downwardTurns);
      applyRotation(ariaVideo, ariaTurns);
      bindDragLabeling();
      loadManualSegments();
    }
    playBtn.addEventListener("click", playBoth);
    pauseBtn.addEventListener("click", pauseBoth);
    pendingActionButtons.addEventListener("click", event => {
      const label = event.target.closest("button[data-pending-label]")?.dataset.pendingLabel;
      if (label) applyPendingLabel(label);
    });
    document.getElementById("pendingCancelBtn").addEventListener("click", () => {
      clearPendingSelection(true);
      saveStatus.textContent = "Canceled selected range.";
    });
    document.addEventListener("keydown", event => {
      const target = event.target;
      const tag = String(target?.tagName || "").toLowerCase();
      const inputType = String(target?.type || "").toLowerCase();
      const isTyping = target?.isContentEditable
        || tag === "textarea"
        || (tag === "input" && !["checkbox", "button", "range"].includes(inputType));
      const isSpace = event.code === "Space" || event.key === " " || event.key === "Spacebar";
      if (isSpace && !isTyping) {
        event.preventDefault();
        event.stopPropagation();
        if (!event.repeat) togglePlayback();
      } else if ((event.code === "Delete" || event.code === "Backspace") && !isTyping && selectedId) {
        event.preventDefault();
        event.stopPropagation();
        deleteManualSegment(selectedId);
      }
    }, true);
    document.addEventListener("keyup", event => {
      const target = event.target;
      const tag = String(target?.tagName || "").toLowerCase();
      const inputType = String(target?.type || "").toLowerCase();
      const isTyping = target?.isContentEditable
        || tag === "textarea"
        || (tag === "input" && !["checkbox", "button", "range"].includes(inputType));
      const isSpace = event.code === "Space" || event.key === " " || event.key === "Spacebar";
      if (isSpace && !isTyping) {
        event.preventDefault();
        event.stopPropagation();
      }
    }, true);
    document.getElementById("backBtn").addEventListener("click", () => seek(Number(ariaVideo.currentTime || 0) - 2, true));
    document.getElementById("forwardBtn").addEventListener("click", () => seek(Number(ariaVideo.currentTime || 0) + 2, true));
    timelineWindowSelect.addEventListener("change", () => {
      const center = Number(ariaVideo.currentTime || 0) || ((viewStartS + viewEndS) / 2);
      setTimelineWindow(center, timelineWindowSize(), true);
    });
    document.getElementById("prevWindowBtn").addEventListener("click", () => panTimelineWindow(-1));
    document.getElementById("nextWindowBtn").addEventListener("click", () => panTimelineWindow(1));
    document.getElementById("centerWindowBtn").addEventListener("click", () => setTimelineWindow(Number(ariaVideo.currentTime || 0), timelineWindowSize(), true));
    document.getElementById("fullWindowBtn").addEventListener("click", () => {
      timelineWindowSelect.value = "full";
      setTimelineWindow(Number(ariaVideo.currentTime || 0), duration, true);
    });
    overviewTimeline.addEventListener("pointerdown", event => {
      event.preventDefault();
      const target = pointerTime(event, overviewTimeline, true);
      seek(target, false);
      if (timelineWindowSelect.value !== "full") setTimelineWindow(target, timelineWindowSize(), true);
    });
    document.getElementById("speedSelect").addEventListener("change", event => {
      const rate = Number(event.target.value || 1);
      ariaVideo.playbackRate = rate;
      downwardVideo.playbackRate = rate;
    });
    scrubber.addEventListener("input", event => seek(Number(event.target.value || 0), false));
    scrubber.addEventListener("change", event => seek(Number(event.target.value || 0), true));
    ariaVideo.addEventListener("timeupdate", () => {
      syncFollower(false);
      if (followWindowCheckbox.checked) ensureTimeInWindow(ariaVideo.currentTime || 0, true);
      renderPlayheads(ariaVideo.currentTime || 0);
      renderMetrics();
    });
    ariaVideo.addEventListener("play", startSyncLoop);
    ariaVideo.addEventListener("pause", () => {
      downwardVideo.pause();
      stopSyncLoop();
    });
    ariaVideo.addEventListener("seeking", () => syncFollower(true));
    ariaVideo.addEventListener("seeked", () => syncFollower(true));
    ariaVideo.addEventListener("waiting", () => syncFollower(true));
    ariaVideo.addEventListener("ratechange", () => syncFollower(false));
    downwardVideo.addEventListener("play", () => {
      if (ariaVideo.paused) downwardVideo.pause();
    });
    downwardVideo.addEventListener("pause", () => {
      if (!ariaVideo.paused && Math.abs((downwardVideo.currentTime || 0) - (ariaVideo.currentTime || 0)) > 0.4) syncFollower(true);
    });
    modelTimeline.addEventListener("pointerdown", event => {
      event.preventDefault();
      seek(pointerTime(event, modelTimeline), true);
    });
    document.getElementById("startRangeBtn").addEventListener("click", () => {
      quickStartS = Number(ariaVideo.currentTime || 0);
      saveStatus.textContent = `Started ${labelSelect.value} at ${fmt(quickStartS)}.`;
    });
    document.getElementById("endRangeBtn").addEventListener("click", () => {
      if (quickStartS == null) {
        saveStatus.textContent = "No start time selected yet.";
        return;
      }
      addManualSegment(quickStartS, Number(ariaVideo.currentTime || 0), labelSelect.value, confidenceSelect.value, "");
      quickStartS = null;
    });
    document.getElementById("cancelRangeBtn").addEventListener("click", () => {
      quickStartS = null;
      saveStatus.textContent = "Canceled label start.";
    });
    document.getElementById("startGapBtn").addEventListener("click", () => {
      const gap = gaps()[0];
      if (!gap) {
        saveStatus.textContent = "No unlabeled gaps left.";
        return;
      }
      seek(gap.start_s, false);
      saveStatus.textContent = `Next gap: ${fmt(gap.start_s)} - ${fmt(gap.end_s)}.`;
    });
    document.getElementById("fillCurrentGapBtn").addEventListener("click", () => {
      const gap = currentGap();
      if (!gap) {
        saveStatus.textContent = "No gap at current time.";
        return;
      }
      addManualSegment(gap.start_s, gap.end_s, labelSelect.value, confidenceSelect.value, "");
    });
    document.getElementById("fillAllGapsBtn").addEventListener("click", () => {
      const allGaps = gaps();
      if (!allGaps.length) {
        saveStatus.textContent = "No gaps to fill.";
        return;
      }
      for (const gap of allGaps) {
        addManualSegment(gap.start_s, gap.end_s, labelSelect.value, confidenceSelect.value, "");
      }
      saveManualSegments(false);
    });
    document.getElementById("saveAllBtn").addEventListener("click", () => saveManualSegments(false));
    segmentSelect.addEventListener("change", event => {
      selectedId = event.target.value;
      const seg = selectedSegment();
      if (seg) seek(seg.start_s, false);
      renderAll();
    });
    document.getElementById("saveSegmentBtn").addEventListener("click", saveSelectedSegmentFromEditor);
    document.querySelectorAll("[data-relabel-action]").forEach(btn => {
      btn.addEventListener("click", () => {
        relabelSelectedSegment(btn.getAttribute("data-relabel-action") || "");
      });
    });
    document.getElementById("deleteSegmentBtn").addEventListener("click", () => {
      deleteManualSegment(selectedId);
    });
    document.getElementById("setStartBtn").addEventListener("click", () => {
      startInput.value = Number(ariaVideo.currentTime || 0).toFixed(2);
    });
    document.getElementById("setEndBtn").addEventListener("click", () => {
      endInput.value = Number(ariaVideo.currentTime || 0).toFixed(2);
    });
    document.getElementById("rotateDownwardBtn").addEventListener("click", () => {
      downwardTurns += 1;
      applyRotation(downwardVideo, downwardTurns);
    });
    document.getElementById("rotateAriaBtn").addEventListener("click", () => {
      ariaTurns += 1;
      applyRotation(ariaVideo, ariaTurns);
    });
    downwardVideo.addEventListener("loadedmetadata", () => applyRotation(downwardVideo, downwardTurns));
    ariaVideo.addEventListener("loadedmetadata", () => applyRotation(ariaVideo, ariaTurns));
    window.addEventListener("resize", () => {
      applyRotation(downwardVideo, downwardTurns);
      applyRotation(ariaVideo, ariaTurns);
    });
    init();
  </script>
</body>
</html>
"""
    return (
        html
        .replace("__DATA__", data_json)
        .replace("__MODEL_TIMELINE_TITLE__", str(payload.get("model_prediction_title") or "Automatic Classification"))
        .replace("__MODEL_TIMELINE_NOTE__", str(payload.get("model_prediction_note") or "zoomed"))
    )


def main() -> None:
    args = parse_args()
    trial_ids = args.trials or discover_trials()
    built: list[Path] = []
    for trial_id in trial_ids:
        payload = load_payload(trial_id)
        output_path = RUNS_DIR / trial_id / OUTPUT_NAME
        output_path.write_text(build_html(payload))
        built.append(output_path)
    for path in built:
        print(f"Wrote manual labeling page: {path}")


if __name__ == "__main__":
    main()
