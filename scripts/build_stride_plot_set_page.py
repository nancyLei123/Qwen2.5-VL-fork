#!/usr/bin/env python3
"""Build an interactive stride/step exploration plot-set page.

The page is intentionally framed as a research inspection tool. Project Aria
SLAM provides head/device motion, not direct foot contacts, so all step and
stride measurements are displayed as proxies that should be validated against
the synchronized video.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

QWEN_REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = QWEN_REPO / "runs"
DIAG_DIR = RUNS_DIR / "stride_step_diagnostics"
DATA_DIR = DIAG_DIR / "data"
OUT_PATH = DIAG_DIR / "stride_plot_set_gallery.html"


def clean_value(value: Any) -> Any:
    """Convert pandas/numpy values into compact JSON-safe values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 5)
    if hasattr(value, "item"):
        return clean_value(value.item())
    return value


def records_from_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    df = pd.read_csv(path)
    return records_from_dataframe(df)


def records_from_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(column): clean_value(row[column]) for column in df.columns}
        for _, row in df.iterrows()
    ]


def build_full_trial_series(trials: list[str], max_points_per_trial: int = 6000) -> list[dict[str, Any]]:
    """Load/downsample full SLAM trajectories for full-trial X/Y views."""
    rows: list[dict[str, Any]] = []
    for trial_id in trials:
        manifest_path = RUNS_DIR / trial_id / "trial_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        trajectory_path = Path(str(manifest["mps_outputs"]["closed_loop_trajectory_csv"]["path"]))
        if not trajectory_path.exists():
            continue

        use_columns = [
            "tracking_timestamp_us",
            "utc_timestamp_ns",
            "tx_world_device",
            "ty_world_device",
            "tz_world_device",
        ]
        available_columns = pd.read_csv(trajectory_path, nrows=0).columns
        trajectory = pd.read_csv(
            trajectory_path,
            usecols=lambda column: column in use_columns,
        ).sort_values("tracking_timestamp_us" if "tracking_timestamp_us" in available_columns else "utc_timestamp_ns")
        timestamp_column = "tracking_timestamp_us" if "tracking_timestamp_us" in trajectory.columns else "utc_timestamp_ns"
        timestamp_scale = 1_000_000.0 if timestamp_column == "tracking_timestamp_us" else 1_000_000_000.0
        trajectory["session_s"] = (trajectory[timestamp_column] - trajectory[timestamp_column].iloc[0]) / timestamp_scale

        dt = trajectory["session_s"].diff().replace(0, pd.NA)
        dx = trajectory["tx_world_device"].diff()
        dy = trajectory["ty_world_device"].diff()
        dz = trajectory["tz_world_device"].diff()
        trajectory["linear_speed_mps"] = ((dx * dx + dy * dy + dz * dz) ** 0.5 / dt).fillna(0.0)
        trajectory["linear_speed_mps"] = trajectory["linear_speed_mps"].rolling(20, min_periods=1, center=True).median()
        heading_dx = trajectory["tx_world_device"].diff(20)
        heading_dy = trajectory["ty_world_device"].diff(20)
        trajectory["local_heading_deg"] = np.degrees(np.arctan2(heading_dy, heading_dx))
        trajectory["local_heading_deg"] = trajectory["local_heading_deg"].ffill().bfill().fillna(0.0)
        trajectory["trial_id"] = trial_id

        stride = max(1, int(math.ceil(len(trajectory) / max_points_per_trial)))
        sampled = trajectory.iloc[::stride].copy()
        if sampled["session_s"].iloc[-1] != trajectory["session_s"].iloc[-1]:
            sampled = pd.concat([sampled, trajectory.iloc[[-1]]], ignore_index=True)
        rows.extend(
            records_from_dataframe(
                sampled[
                    [
                        "trial_id",
                        "session_s",
                        "tx_world_device",
                        "ty_world_device",
                        "tz_world_device",
                        "linear_speed_mps",
                        "local_heading_deg",
                    ]
                ]
            )
        )
    return rows


def build_payload() -> dict[str, Any]:
    bouts = records_from_csv(DATA_DIR / "step_stride_bout_summary.csv")
    events = records_from_csv(DATA_DIR / "step_stride_event_estimates.csv")
    series = records_from_csv(DATA_DIR / "step_stride_resampled_bout_timeseries.csv")
    label_summary = records_from_csv(DATA_DIR / "step_stride_label_summary.csv")

    trials = sorted({str(row["trial_id"]) for row in bouts if row.get("trial_id")})
    full_series = build_full_trial_series(trials)
    generated_from = {
        "bouts": str(DATA_DIR / "step_stride_bout_summary.csv"),
        "events": str(DATA_DIR / "step_stride_event_estimates.csv"),
        "timeseries": str(DATA_DIR / "step_stride_resampled_bout_timeseries.csv"),
        "full_series": "MPS closed_loop_trajectory.csv files from trial manifests",
    }
    return {
        "trials": trials,
        "bouts": bouts,
        "events": events,
        "series": series,
        "full_series": full_series,
        "label_summary": label_summary,
        "generated_from": generated_from,
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stride / Step Plot Set - SLAM Research Exploration</title>
  <style>
    :root {
      --ink: #18221f;
      --muted: #65726b;
      --line: #dfe7df;
      --paper: #fbfaf6;
      --panel: #ffffff;
      --green: #167a55;
      --green-soft: #dff3e9;
      --orange: #d36f2f;
      --blue: #3279b7;
      --red: #c94f4f;
      --purple: #7a5cb8;
      --sand: #f3eadb;
      --shadow: 0 18px 50px rgba(32, 45, 38, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 8% 6%, rgba(22, 122, 85, 0.10), transparent 30rem),
        linear-gradient(180deg, #f7f4ed 0%, #fbfaf6 40%, #f5f7f2 100%);
      font-family: "Avenir Next", "Trebuchet MS", sans-serif;
    }
    header {
      padding: 34px clamp(22px, 4vw, 56px) 18px;
    }
    .eyebrow {
      color: var(--green);
      font-size: 0.78rem;
      font-weight: 800;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }
    h1 {
      max-width: 1050px;
      margin: 8px 0 10px;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(2rem, 4vw, 4rem);
      line-height: 0.98;
      font-weight: 700;
    }
    .subhead {
      max-width: 980px;
      color: var(--muted);
      font-size: 1.05rem;
      line-height: 1.55;
    }
    main {
      display: grid;
      gap: 18px;
      padding: 0 clamp(16px, 3vw, 44px) 44px;
    }
    .card {
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid rgba(29, 50, 40, 0.10);
      border-radius: 24px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .card-inner { padding: 22px; }
    .section-title {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin: 0 0 12px;
    }
    h2, h3 { margin: 0; line-height: 1.15; }
    h2 { font-size: 1.45rem; }
    h3 { font-size: 1.05rem; }
    p { color: var(--muted); line-height: 1.55; margin: 0.45rem 0; }
    .logic-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }
    .logic-step {
      position: relative;
      min-height: 164px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: linear-gradient(180deg, #ffffff, #fbfcf8);
    }
    .logic-step .num {
      display: inline-grid;
      place-items: center;
      width: 30px;
      height: 30px;
      border-radius: 999px;
      background: var(--green-soft);
      color: var(--green);
      font-weight: 900;
      margin-bottom: 10px;
    }
    .logic-step strong { display: block; margin-bottom: 5px; }
    .logic-step span { color: var(--muted); line-height: 1.45; font-size: 0.92rem; }
    .controls {
      display: grid;
      grid-template-columns: 170px 170px minmax(280px, 1fr) repeat(3, max-content);
      gap: 12px;
      align-items: end;
      padding: 18px 22px;
      background: linear-gradient(90deg, rgba(22,122,85,0.10), rgba(211,111,47,0.06));
      border-bottom: 1px solid var(--line);
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 0.76rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    select, button {
      min-height: 40px;
      border-radius: 999px;
      border: 1px solid #ccd8cf;
      background: #ffffff;
      color: var(--ink);
      padding: 0 14px;
      font: inherit;
      font-weight: 700;
    }
    button {
      cursor: pointer;
      transition: 150ms ease;
    }
    button:hover { transform: translateY(-1px); box-shadow: 0 8px 22px rgba(22,122,85,0.14); }
    .check {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      height: 40px;
      padding: 0 12px;
      border-radius: 999px;
      border: 1px solid #ccd8cf;
      background: rgba(255,255,255,0.78);
      color: var(--ink);
      font-size: 0.9rem;
      font-weight: 800;
      text-transform: none;
      letter-spacing: 0;
    }
    .check input { accent-color: var(--green); }
    .plot-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(320px, 0.7fr);
      gap: 16px;
      padding: 18px;
    }
    .plot-card {
      border: 1px solid var(--line);
      border-radius: 20px;
      background: #fffefa;
      overflow: hidden;
    }
    .plot-card h3 {
      padding: 14px 16px 3px;
    }
    .plot-note {
      padding: 0 16px 10px;
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.45;
    }
    .plot-tools {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 0 16px 12px;
      align-items: center;
    }
    .tool-button {
      min-height: 32px;
      padding: 0 11px;
      font-size: 0.82rem;
      border-color: #cbd8cf;
      background: #ffffff;
    }
    .tool-hint {
      color: var(--muted);
      font-size: 0.82rem;
      font-weight: 700;
    }
    svg {
      width: 100%;
      display: block;
      background:
        linear-gradient(90deg, rgba(223,231,223,0.55) 1px, transparent 1px),
        linear-gradient(180deg, rgba(223,231,223,0.55) 1px, transparent 1px);
      background-size: 64px 64px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      padding: 12px;
      border-radius: 16px;
      background: #f7faf4;
      border: 1px solid var(--line);
    }
    .metric small {
      display: block;
      color: var(--muted);
      font-size: 0.72rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .metric b {
      display: block;
      margin-top: 3px;
      font-size: 1.05rem;
    }
    .legend-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fff;
      padding: 7px 10px;
      color: var(--muted);
      font-size: 0.84rem;
      font-weight: 700;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--muted);
    }
    .dot.left { background: var(--blue); }
    .dot.right { background: var(--red); }
    .dot.high { background: var(--green); }
    .dot.medium { background: var(--orange); }
    .dot.low { background: #909991; }
    .table-wrap {
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: #fff;
      max-height: 440px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }
    th {
      position: sticky;
      top: 0;
      z-index: 2;
      background: #f1f5ee;
      color: #415046;
      text-align: left;
      white-space: nowrap;
      font-size: 0.75rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    th, td { padding: 9px 10px; border-bottom: 1px solid #edf1ea; }
    td { color: #28352f; white-space: nowrap; }
    td.warning { white-space: normal; min-width: 240px; color: #7b5441; }
    .manual-select {
      min-height: 30px;
      border-radius: 10px;
      padding: 0 8px;
      font-weight: 700;
    }
    .callout {
      padding: 14px 16px;
      border-radius: 18px;
      background: var(--sand);
      border: 1px solid #e7d9c2;
      color: #54473a;
      line-height: 1.5;
    }
    .link-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 12px;
    }
    a.button-link {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 38px;
      padding: 0 14px;
      border-radius: 999px;
      background: var(--green);
      color: #fff;
      text-decoration: none;
      font-weight: 800;
    }
    .summary-table {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
    }
    .axis-label { fill: #4b5a51; font-size: 12px; font-weight: 800; }
    .tick { stroke: #cbd7cd; stroke-width: 1; }
    .tick-text { fill: #617068; font-size: 11px; }
    .path-line { fill: none; stroke: #1d2b28; stroke-width: 2.4; }
    .speed-line { fill: none; stroke: var(--orange); stroke-width: 2; }
    .height-line { fill: none; stroke: #26342f; stroke-width: 2; }
    .sway-line { fill: none; stroke: var(--purple); stroke-width: 2; }
    .step-line { fill: none; stroke: var(--green); stroke-width: 2.2; }
    .stride-line { fill: none; stroke: var(--purple); stroke-width: 2.2; }
    .event-label { fill: #17231f; font-size: 10px; font-weight: 900; paint-order: stroke; stroke: white; stroke-width: 3px; }
    .step-arrow {
      stroke-width: 2.8;
      opacity: 0.94;
      stroke-linecap: round;
      filter: drop-shadow(0 2px 2px rgba(255,255,255,0.95));
    }
    .step-arrow-halo {
      stroke: #ffffff;
      stroke-width: 7.0;
      opacity: 0.88;
      stroke-linecap: round;
    }
    .direction-label {
      fill: #9b4d18;
      font-size: 12px;
      font-weight: 900;
      paint-order: stroke;
      stroke: white;
      stroke-width: 4px;
    }
    .axis-zero { stroke: #89958d; stroke-width: 1.2; stroke-dasharray: 4 4; }
    @media (max-width: 1040px) {
      .logic-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .controls { grid-template-columns: 1fr 1fr; }
      .plot-layout { grid-template-columns: 1fr; }
    }
    @media (max-width: 680px) {
      header { padding-top: 24px; }
      .logic-grid { grid-template-columns: 1fr; }
      .controls { grid-template-columns: 1fr; }
      .card-inner, .plot-layout { padding: 14px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="eyebrow">SLAM step and stride exploration</div>
    <h1>Stride / Step Plot Set for Parkinson Motion Review</h1>
    <p class="subhead">
      This page combines the professor's X/Y path idea with our current SLAM stride diagnostics.
      It shows floor-plane movement, local direction of travel, step-like markers, estimated left/right
      side, raw height, lateral head sway, and stride-proxy values in one place.
    </p>
  </header>

  <main>
    <section class="card">
      <div class="card-inner">
        <div class="section-title">
          <h2>Research Logic</h2>
          <span class="pill"><span class="dot high"></span>proxy, not confirmed foot contact</span>
        </div>
        <p>
          The main question is whether raw SLAM head-position data can reveal step rhythm, stride length,
          and left/right balance. The logic below keeps the analysis interpretable and checkable against
          the synchronized downward video.
        </p>
        <div class="logic-grid">
          <div class="logic-step">
            <span class="num">1</span>
            <strong>Use X/Y as floor path</strong>
            <span>X and Y are treated as the horizontal floor plane. The plot shows the head/device path from above.</span>
          </div>
          <div class="logic-step">
            <span class="num">2</span>
            <strong>Estimate travel direction locally</strong>
            <span>Forward direction is calculated from the smoothed X/Y path, not assumed to be a fixed raw Y axis. This helps during turns.</span>
          </div>
          <div class="logic-step">
            <span class="num">3</span>
            <strong>Detect step-like rhythm</strong>
            <span>The current detector uses lateral head-sway extrema and places the event between neighboring extrema. Low-Z candidates are marked separately for validation.</span>
          </div>
          <div class="logic-step">
            <span class="num">4</span>
            <strong>Estimate step and stride proxies</strong>
            <span>Step proxy is horizontal path length between neighboring events. Stride proxy is horizontal path length across every other event.</span>
          </div>
        </div>
      </div>
    </section>

    <section class="card">
      <div class="controls">
        <label>Trial
          <select id="trialSelect"></select>
        </label>
        <label>View Scope
          <select id="scopeSelect">
            <option value="bout">Selected Bout</option>
            <option value="full">Full Trial</option>
          </select>
        </label>
        <label>Bout / Review Window
          <select id="boutSelect"></select>
        </label>
        <label class="check"><input id="showArrows" type="checkbox" checked> step arrows</label>
        <label class="check"><input id="showLabels" type="checkbox" checked> marker labels</label>
        <button id="exportManual">Export Manual Side CSV</button>
      </div>

      <div class="plot-layout">
        <div class="plot-card">
          <h3>X/Y Movement Path + Step Markers</h3>
          <div class="plot-note">
            This is the professor-style top-down view: path, estimated step-like events, step-level travel direction,
            and left/right proxy. Use View Scope to switch between the selected local bout and the full trial path.
            Each detected step has an arrow. Blue arrows mean estimated left; red arrows mean estimated right.
          </div>
          <div class="plot-tools">
            <button class="tool-button" id="xyZoomIn">Zoom In</button>
            <button class="tool-button" id="xyZoomOut">Zoom Out</button>
            <button class="tool-button" id="xyResetZoom">Reset View</button>
            <span class="tool-hint">Wheel = zoom, drag = pan</span>
          </div>
          <svg id="xyPlot" viewBox="0 0 900 590" role="img" aria-label="XY path plot"></svg>
        </div>

        <aside class="plot-card">
          <h3>Selected Bout Summary</h3>
          <div class="plot-note" id="boutNarrative"></div>
          <div class="card-inner">
            <div class="metric-grid" id="metricGrid"></div>
            <div class="legend-row">
              <span class="pill"><span class="dot left"></span>estimated left</span>
              <span class="pill"><span class="dot right"></span>estimated right</span>
              <span class="pill"><span class="dot high"></span>high confidence</span>
              <span class="pill"><span class="dot medium"></span>medium confidence</span>
              <span class="pill"><span class="dot low"></span>low/review</span>
            </div>
            <div class="callout" style="margin-top:14px;">
              Important caveat: left/right is an initial SLAM-only estimate from lateral sway and device orientation.
              It should be checked with the downward video before being treated as ground truth.
            </div>
            <div class="link-row" id="relatedLinks"></div>
          </div>
        </aside>

        <div class="plot-card">
          <h3>Height + Speed Over Time</h3>
          <div class="plot-note">
            Raw Z height is useful for posture and stair movement. Speed helps identify movement start/stop
            and possible shuffling or freezing changes. Gray triangles mark simple low-Z candidates.
          </div>
          <svg id="heightSpeedPlot" viewBox="0 0 900 330" role="img" aria-label="Height and speed plot"></svg>
        </div>

        <div class="plot-card">
          <h3>Lateral Head Sway + Step Events</h3>
          <div class="plot-note">
            This shows the signal used for the current step proxy. Events are placed between neighboring
            lateral sway extrema, because visual foot contacts often happen between head-sway peaks.
          </div>
          <svg id="swayPlot" viewBox="0 0 900 330" role="img" aria-label="Lateral sway plot"></svg>
        </div>

        <div class="plot-card">
          <h3>Step / Stride Proxy Lengths</h3>
          <div class="plot-note">
            Step proxy uses adjacent events. Stride proxy uses every-other event. Missing values are kept visible
            because they often mean a bout boundary, short window, or low-confidence event context.
          </div>
          <svg id="lengthPlot" viewBox="0 0 900 330" role="img" aria-label="Step and stride length plot"></svg>
        </div>

        <div class="plot-card">
          <h3>How To Read This Plot Set</h3>
          <div class="card-inner">
            <p><strong>For step length:</strong> check whether step markers line up with regular oscillations in lateral sway and plausible forward progress in X/Y.</p>
            <p><strong>For turns:</strong> expect lower confidence, because left/right and stride length are harder when the local direction changes quickly.</p>
            <p><strong>For small/shuffling steps:</strong> expect shorter step proxies and possibly lower confidence. These are clinically interesting, so they should not be hidden.</p>
            <p><strong>For manual correction:</strong> use the event table below to mark left/right/uncertain after checking the synchronized downward video. The corrections stay in this browser and can be exported as CSV.</p>
          </div>
        </div>
      </div>
    </section>

    <section class="card">
      <div class="card-inner">
        <div class="section-title">
          <h2>Event Table + Manual Side Check</h2>
          <button id="clearManual">Clear Manual Side Corrections</button>
        </div>
        <p>
          This table is meant for collaborative validation: compare each event time with the video,
          then mark whether the estimated left/right side looks correct.
        </p>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Time</th>
                <th>Auto Side</th>
                <th>Manual Side</th>
                <th>Foot Conf.</th>
                <th>Event Conf.</th>
                <th>Step Proxy</th>
                <th>Stride Proxy</th>
                <th>Lateral Sway</th>
                <th>Warning / Reason</th>
              </tr>
            </thead>
            <tbody id="eventTable"></tbody>
          </table>
        </div>
      </div>
    </section>

    <section class="card">
      <div class="card-inner">
        <h2>Across-Label Summary</h2>
        <p>
          These summaries only aggregate the current diagnostic output. They are useful for deciding whether
          the step proxy looks different across walking, upstairs, and downstairs before using it in a model.
        </p>
        <div class="summary-table">
          <table>
            <thead id="summaryHead"></thead>
            <tbody id="summaryBody"></tbody>
          </table>
        </div>
      </div>
    </section>
  </main>

  <script>
    const DATA = __DATA_JSON__;
    const W = 900;
    const H_XY = 590;
    const H_SMALL = 330;
    const M = { left: 58, right: 42, top: 30, bottom: 46 };
    const manualPrefix = "stride-plot-set-manual-side:";
    let xyZoomState = null;
    let xyDragState = null;

    const labelNames = {
      walking: "Walking",
      upstairs: "Upstairs",
      downstairs: "Downstairs",
      sitting: "Sitting",
      standing: "Standing",
      "sit-to-stand": "Sit-to-Stand",
      "stand-to-sit": "Stand-to-Sit",
      "stand-to-walk": "Stand-to-Walk",
      "walk-to-stand": "Walk-to-Stand"
    };

    function fmt(value, digits = 2, suffix = "") {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
      return `${Number(value).toFixed(digits)}${suffix}`;
    }

    function byId(id) { return document.getElementById(id); }

    function svgEl(name, attrs = {}) {
      const el = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
      return el;
    }

    function clear(svg) {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
    }

    function boutKey(row) {
      return `${row.trial_id}::${row.bout_id}`;
    }

    function currentTrial() {
      return byId("trialSelect").value || DATA.trials[0];
    }

    function currentBout() {
      const key = byId("boutSelect").value;
      return DATA.bouts.find(row => boutKey(row) === key) || DATA.bouts[0];
    }

    function currentScope() {
      return byId("scopeSelect").value || "bout";
    }

    function rowsForBout(source, bout) {
      return source.filter(row => row.trial_id === bout.trial_id && Number(row.bout_id) === Number(bout.bout_id));
    }

    function rowsForTrial(source, trial) {
      return source.filter(row => row.trial_id === trial);
    }

    function extent(values, pad = 0.08) {
      const nums = values.map(Number).filter(Number.isFinite);
      if (!nums.length) return [0, 1];
      let min = Math.min(...nums);
      let max = Math.max(...nums);
      if (Math.abs(max - min) < 1e-9) {
        min -= 0.5;
        max += 0.5;
      }
      const extra = (max - min) * pad;
      return [min - extra, max + extra];
    }

    function scale(domain, range) {
      const [d0, d1] = domain;
      const [r0, r1] = range;
      return value => r0 + (Number(value) - d0) * (r1 - r0) / (d1 - d0 || 1);
    }

    function drawAxes(svg, xDomain, yDomain, width, height, xLabel, yLabel) {
      const x = scale(xDomain, [M.left, width - M.right]);
      const y = scale(yDomain, [height - M.bottom, M.top]);
      for (let i = 0; i <= 4; i += 1) {
        const xv = xDomain[0] + (xDomain[1] - xDomain[0]) * i / 4;
        const xp = x(xv);
        svg.appendChild(svgEl("line", { x1: xp, y1: M.top, x2: xp, y2: height - M.bottom, class: "tick" }));
        svg.appendChild(svgEl("text", { x: xp, y: height - 18, "text-anchor": "middle", class: "tick-text" })).textContent = fmt(xv, 1);
      }
      for (let i = 0; i <= 4; i += 1) {
        const yv = yDomain[0] + (yDomain[1] - yDomain[0]) * i / 4;
        const yp = y(yv);
        svg.appendChild(svgEl("line", { x1: M.left, y1: yp, x2: width - M.right, y2: yp, class: "tick" }));
        svg.appendChild(svgEl("text", { x: 14, y: yp + 4, class: "tick-text" })).textContent = fmt(yv, 2);
      }
      svg.appendChild(svgEl("text", { x: width / 2, y: height - 5, "text-anchor": "middle", class: "axis-label" })).textContent = xLabel;
      const yl = svgEl("text", { x: 18, y: height / 2, class: "axis-label", transform: `rotate(-90 18 ${height / 2})`, "text-anchor": "middle" });
      yl.textContent = yLabel;
      svg.appendChild(yl);
      return { x, y };
    }

    function aspectAdjustedXYDomains(rows) {
      let xDomain = extent(rows.map(row => row.tx_world_device), 0.10);
      let yDomain = extent(rows.map(row => row.ty_world_device), 0.10);
      const plotW = W - M.left - M.right;
      const plotH = H_XY - M.top - M.bottom;
      const xRange = xDomain[1] - xDomain[0];
      const yRange = yDomain[1] - yDomain[0];
      const targetRatio = plotW / plotH;
      const actualRatio = xRange / yRange;
      if (actualRatio > targetRatio) {
        const newYRange = xRange / targetRatio;
        const mid = (yDomain[0] + yDomain[1]) / 2;
        yDomain = [mid - newYRange / 2, mid + newYRange / 2];
      } else {
        const newXRange = yRange * targetRatio;
        const mid = (xDomain[0] + xDomain[1]) / 2;
        xDomain = [mid - newXRange / 2, mid + newXRange / 2];
      }
      return { xDomain, yDomain };
    }

    function xyViewKey(context) {
      return `${context.trial_id}:${context.scope}:${context.bout_key || "full"}`;
    }

    function ensureXYZoomState(context, baseXDomain, baseYDomain) {
      const key = xyViewKey(context);
      if (!xyZoomState || xyZoomState.key !== key) {
        xyZoomState = {
          key,
          xDomain: baseXDomain.slice(),
          yDomain: baseYDomain.slice(),
          baseXDomain: baseXDomain.slice(),
          baseYDomain: baseYDomain.slice()
        };
      } else {
        xyZoomState.baseXDomain = baseXDomain.slice();
        xyZoomState.baseYDomain = baseYDomain.slice();
        xyZoomState.xDomain = clampDomain(xyZoomState.xDomain, xyZoomState.baseXDomain);
        xyZoomState.yDomain = clampDomain(xyZoomState.yDomain, xyZoomState.baseYDomain);
      }
    }

    function clampDomain(domain, baseDomain) {
      const width = domain[1] - domain[0];
      const baseWidth = baseDomain[1] - baseDomain[0];
      if (!Number.isFinite(width) || width >= baseWidth) return baseDomain.slice();
      let output = domain.slice();
      if (output[0] < baseDomain[0]) {
        output = [baseDomain[0], baseDomain[0] + width];
      }
      if (output[1] > baseDomain[1]) {
        output = [baseDomain[1] - width, baseDomain[1]];
      }
      return output;
    }

    function zoomXYDomain(factor, centerData = null) {
      if (!xyZoomState) return;
      const xDomain = xyZoomState.xDomain;
      const yDomain = xyZoomState.yDomain;
      const cx = centerData ? centerData.x : (xDomain[0] + xDomain[1]) / 2;
      const cy = centerData ? centerData.y : (yDomain[0] + yDomain[1]) / 2;
      const newXHalf = (xDomain[1] - xDomain[0]) * factor / 2;
      const newYHalf = (yDomain[1] - yDomain[0]) * factor / 2;
      xyZoomState.xDomain = clampDomain([cx - newXHalf, cx + newXHalf], xyZoomState.baseXDomain);
      xyZoomState.yDomain = clampDomain([cy - newYHalf, cy + newYHalf], xyZoomState.baseYDomain);
    }

    function panXYDomain(dxPx, dyPx) {
      if (!xyZoomState) return;
      const plotW = W - M.left - M.right;
      const plotH = H_XY - M.top - M.bottom;
      const xRange = xyZoomState.xDomain[1] - xyZoomState.xDomain[0];
      const yRange = xyZoomState.yDomain[1] - xyZoomState.yDomain[0];
      const dxData = dxPx * xRange / plotW;
      const dyData = dyPx * yRange / plotH;
      xyZoomState.xDomain = clampDomain(
        [xyZoomState.xDomain[0] - dxData, xyZoomState.xDomain[1] - dxData],
        xyZoomState.baseXDomain
      );
      xyZoomState.yDomain = clampDomain(
        [xyZoomState.yDomain[0] + dyData, xyZoomState.yDomain[1] + dyData],
        xyZoomState.baseYDomain
      );
    }

    function resetXYZoom() {
      if (!xyZoomState) return;
      xyZoomState.xDomain = xyZoomState.baseXDomain.slice();
      xyZoomState.yDomain = xyZoomState.baseYDomain.slice();
    }

    function svgPointFromEvent(svg, event) {
      const point = svg.createSVGPoint();
      point.x = event.clientX;
      point.y = event.clientY;
      return point.matrixTransform(svg.getScreenCTM().inverse());
    }

    function svgXYPointToData(point) {
      if (!xyZoomState) return null;
      const plotW = W - M.left - M.right;
      const plotH = H_XY - M.top - M.bottom;
      const px = Math.min(W - M.right, Math.max(M.left, point.x));
      const py = Math.min(H_XY - M.bottom, Math.max(M.top, point.y));
      const x = xyZoomState.xDomain[0] + (px - M.left) / plotW * (xyZoomState.xDomain[1] - xyZoomState.xDomain[0]);
      const y = xyZoomState.yDomain[0] + (H_XY - M.bottom - py) / plotH * (xyZoomState.yDomain[1] - xyZoomState.yDomain[0]);
      return { x, y };
    }

    function pathFromRows(rows, xScale, yScale, xField, yField) {
      return rows
        .filter(row => Number.isFinite(Number(row[xField])) && Number.isFinite(Number(row[yField])))
        .map((row, index) => `${index === 0 ? "M" : "L"} ${xScale(row[xField])} ${yScale(row[yField])}`)
        .join(" ");
    }

    function addArrowMarker(svg) {
      const defs = svgEl("defs");
      [
        ["xyArrowHeadLeft", "#3279b7"],
        ["xyArrowHeadRight", "#c94f4f"],
        ["xyArrowHeadUnknown", "#8a948c"]
      ].forEach(([id, fill]) => {
        const marker = svgEl("marker", {
          id,
          markerWidth: 8,
          markerHeight: 8,
          refX: 7,
          refY: 3,
          orient: "auto",
          markerUnits: "strokeWidth"
        });
        marker.appendChild(svgEl("path", { d: "M0,0 L0,6 L9,3 z", fill }));
        defs.appendChild(marker);
      });
      const clip = svgEl("clipPath", { id: "xyClip" });
      clip.appendChild(svgEl("rect", {
        x: M.left,
        y: M.top,
        width: W - M.left - M.right,
        height: H_XY - M.top - M.bottom,
      }));
      defs.appendChild(clip);
      svg.appendChild(defs);
    }

    function sideColor(row) {
      const side = String(row.estimated_foot_side || "");
      if (side === "left") return "var(--blue)";
      if (side === "right") return "var(--red)";
      return "#8a948c";
    }

    function sideMarker(row) {
      const side = String(row.estimated_foot_side || "");
      if (side === "left") return "url(#xyArrowHeadLeft)";
      if (side === "right") return "url(#xyArrowHeadRight)";
      return "url(#xyArrowHeadUnknown)";
    }

    function localStepAngle(event, rows) {
      let angle = Number(event.local_heading_deg) * Math.PI / 180;
      if (Number.isFinite(angle)) return angle;
      const eventTime = Number(event.session_s);
      let nearestIndex = 0;
      let nearestDistance = Infinity;
      rows.forEach((row, index) => {
        const distance = Math.abs(Number(row.session_s) - eventTime);
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearestIndex = index;
        }
      });
      const windowSize = Math.max(2, Math.floor(rows.length / 80));
      const prev = rows[Math.max(0, nearestIndex - windowSize)];
      const next = rows[Math.min(rows.length - 1, nearestIndex + windowSize)];
      return Math.atan2(
        Number(next.ty_world_device) - Number(prev.ty_world_device),
        Number(next.tx_world_device) - Number(prev.tx_world_device)
      );
    }

    function confidenceStroke(row) {
      const conf = String(row.event_confidence || row.foot_side_confidence || "");
      if (conf === "high") return "var(--green)";
      if (conf === "medium") return "var(--orange)";
      return "#7d877f";
    }

    function manualKey(event) {
      return `${manualPrefix}${event.trial_id}:${event.bout_id}:${event.event_index}`;
    }

    function manualValue(event) {
      return localStorage.getItem(manualKey(event)) || "";
    }

    function drawXY(context, rows, events) {
      const svg = byId("xyPlot");
      clear(svg);
      if (!rows.length) return;
      addArrowMarker(svg);

      const baseDomains = aspectAdjustedXYDomains(rows);
      ensureXYZoomState(context, baseDomains.xDomain, baseDomains.yDomain);
      const xDomain = xyZoomState.xDomain.slice();
      const yDomain = xyZoomState.yDomain.slice();

      const { x, y } = drawAxes(svg, xDomain, yDomain, W, H_XY, "tx_world_device (m)", "ty_world_device (m)");
      const plotGroup = svgEl("g", { "clip-path": "url(#xyClip)" });
      svg.appendChild(plotGroup);
      plotGroup.appendChild(svgEl("path", { d: pathFromRows(rows, x, y, "tx_world_device", "ty_world_device"), class: "path-line" }));
      svg.appendChild(svgEl("text", { x: W - M.right, y: 20, "text-anchor": "end", class: "direction-label" })).textContent =
        context.scope === "full" ? "Full trial path" : "Selected bout path";

      const start = rows[0];
      const end = rows[rows.length - 1];
      plotGroup.appendChild(svgEl("circle", { cx: x(start.tx_world_device), cy: y(start.ty_world_device), r: 7, fill: "var(--green)" }));
      plotGroup.appendChild(svgEl("text", { x: x(start.tx_world_device) + 10, y: y(start.ty_world_device) - 8, class: "event-label" })).textContent = "start";
      plotGroup.appendChild(svgEl("circle", { cx: x(end.tx_world_device), cy: y(end.ty_world_device), r: 7, fill: "var(--red)" }));
      plotGroup.appendChild(svgEl("text", { x: x(end.tx_world_device) + 10, y: y(end.ty_world_device) - 8, class: "event-label" })).textContent = "end";

      if (byId("showArrows").checked) {
        events.forEach(event => {
          const cx = x(event.tx_world_device_m);
          const cy = y(event.ty_world_device_m);
          const angle = localStepAngle(event, rows);
          if (!Number.isFinite(cx) || !Number.isFinite(cy) || !Number.isFinite(angle)) return;
          const forwardX = Math.cos(angle);
          const forwardY = -Math.sin(angle);
          const len = context.scope === "full" ? 34 : 46;
          const sx = cx;
          const sy = cy;
          const x2 = sx + forwardX * len;
          const y2 = sy + forwardY * len;
          plotGroup.appendChild(svgEl("line", { x1: sx, y1: sy, x2, y2, class: "step-arrow-halo" }));
          plotGroup.appendChild(svgEl("line", {
            x1: sx,
            y1: sy,
            x2,
            y2,
            class: "step-arrow",
            stroke: sideColor(event),
            "marker-end": sideMarker(event)
          }));
        });
      }

      events.forEach(event => {
        const cx = x(event.tx_world_device_m);
        const cy = y(event.ty_world_device_m);
        const conf = event.event_confidence || "low";
        const r = conf === "high" ? 8.5 : conf === "medium" ? 7.5 : 6.5;
        plotGroup.appendChild(svgEl("circle", {
          cx, cy, r,
          fill: sideColor(event),
          stroke: confidenceStroke(event),
          "stroke-width": conf === "high" ? 4 : 2.5,
          opacity: conf === "low" ? 0.72 : 0.92
        }));
        if (byId("showLabels").checked) {
          const side = String(event.estimated_foot_side || "?").slice(0, 1).toUpperCase();
          plotGroup.appendChild(svgEl("text", { x: cx + 9, y: cy - 9, class: "event-label" })).textContent = `${event.event_index}:${side}`;
        }
      });

      const overlay = svgEl("rect", {
        x: M.left,
        y: M.top,
        width: W - M.left - M.right,
        height: H_XY - M.top - M.bottom,
        fill: "transparent",
        "pointer-events": "all",
        cursor: "grab"
      });
      overlay.onwheel = event => {
        event.preventDefault();
        const point = svgPointFromEvent(svg, event);
        const center = svgXYPointToData(point);
        zoomXYDomain(event.deltaY < 0 ? 0.78 : 1.28, center);
        render();
      };
      overlay.onpointerdown = event => {
        event.preventDefault();
        xyDragState = { lastX: event.clientX, lastY: event.clientY };
        overlay.setAttribute("cursor", "grabbing");
      };
      svg.appendChild(overlay);
    }

    function drawHeightSpeed(context, rows, events) {
      const svg = byId("heightSpeedPlot");
      clear(svg);
      if (!rows.length) return;
      const xDomain = [Number(context.start_s), Number(context.end_s)];
      const yDomain = extent(rows.map(row => row.tz_world_device), 0.16);
      const speedDomain = extent(rows.map(row => row.linear_speed_mps), 0.12);
      const { x, y } = drawAxes(svg, xDomain, yDomain, W, H_SMALL, "session time (s)", "raw SLAM Z height (m)");
      const ySpeed = scale(speedDomain, [H_SMALL - M.bottom, M.top]);
      svg.appendChild(svgEl("path", { d: pathFromRows(rows, x, y, "session_s", "tz_world_device"), class: "height-line" }));
      svg.appendChild(svgEl("path", { d: pathFromRows(rows, x, ySpeed, "session_s", "linear_speed_mps"), class: "speed-line" }));
      svg.appendChild(svgEl("text", { x: W - M.right, y: 20, "text-anchor": "end", fill: "var(--orange)", class: "axis-label" })).textContent = "orange = speed";
      findLowZCandidates(rows).forEach(row => {
        const cx = x(row.session_s);
        const cy = y(row.tz_world_device);
        const tri = `${cx},${cy + 6} ${cx - 5},${cy - 4} ${cx + 5},${cy - 4}`;
        svg.appendChild(svgEl("polygon", { points: tri, fill: "#6f7972", opacity: "0.75" }));
      });
      svg.appendChild(svgEl("text", { x: W - M.right, y: 38, "text-anchor": "end", fill: "#6f7972", class: "axis-label" })).textContent = "gray triangles = low-Z candidates";
      events.forEach(event => {
        const cx = x(event.session_s);
        const cy = y(event.tz_world_device_m);
        svg.appendChild(svgEl("line", { x1: cx, y1: M.top, x2: cx, y2: H_SMALL - M.bottom, stroke: sideColor(event), "stroke-width": 1, opacity: 0.25 }));
        svg.appendChild(svgEl("circle", { cx, cy, r: 5, fill: sideColor(event), stroke: "#fff", "stroke-width": 1.6 }));
      });
    }

    function findLowZCandidates(rows) {
      if (rows.length < 12) return [];
      const times = rows.map(row => Number(row.session_s));
      const values = rows.map(row => Number(row.tz_world_device));
      const dt = Math.max(0.001, (times[times.length - 1] - times[0]) / Math.max(1, times.length - 1));
      const smoothRadius = Math.max(1, Math.round(0.06 / dt));
      const smooth = values.map((_, index) => {
        const a = Math.max(0, index - smoothRadius);
        const b = Math.min(values.length - 1, index + smoothRadius);
        let sum = 0;
        let count = 0;
        for (let j = a; j <= b; j += 1) {
          if (Number.isFinite(values[j])) {
            sum += values[j];
            count += 1;
          }
        }
        return count ? sum / count : values[index];
      });
      const sorted = smooth.filter(Number.isFinite).slice().sort((a, b) => a - b);
      if (!sorted.length) return [];
      const p05 = sorted[Math.floor(0.05 * (sorted.length - 1))];
      const p95 = sorted[Math.floor(0.95 * (sorted.length - 1))];
      const prominence = Math.max(0.006, 0.08 * (p95 - p05));
      const searchRadius = Math.max(3, Math.round(0.22 / dt));
      const minSpacing = Math.max(4, Math.round(0.32 / dt));
      const candidates = [];
      let lastIndex = -Infinity;
      for (let i = searchRadius; i < smooth.length - searchRadius; i += 1) {
        let isMin = true;
        let leftMax = -Infinity;
        let rightMax = -Infinity;
        for (let j = i - searchRadius; j <= i + searchRadius; j += 1) {
          if (smooth[j] < smooth[i]) isMin = false;
          if (j < i) leftMax = Math.max(leftMax, smooth[j]);
          if (j > i) rightMax = Math.max(rightMax, smooth[j]);
        }
        if (!isMin) continue;
        if (Math.min(leftMax - smooth[i], rightMax - smooth[i]) < prominence) continue;
        if (i - lastIndex < minSpacing) {
          const previous = candidates[candidates.length - 1];
          if (previous && Number(previous.tz_world_device) > Number(rows[i].tz_world_device)) {
            candidates[candidates.length - 1] = rows[i];
            lastIndex = i;
          }
          continue;
        }
        candidates.push(rows[i]);
        lastIndex = i;
      }
      return candidates;
    }

    function drawSway(bout, rows, events) {
      const svg = byId("swayPlot");
      clear(svg);
      if (!rows.length) return;
      const xDomain = [Number(bout.bout_start_s), Number(bout.bout_end_s)];
      const yDomain = extent(rows.map(row => row.lateral_sway_m), 0.18);
      const { x, y } = drawAxes(svg, xDomain, yDomain, W, H_SMALL, "session time (s)", "lateral head sway (m)");
      const zeroY = y(0);
      svg.appendChild(svgEl("line", { x1: M.left, y1: zeroY, x2: W - M.right, y2: zeroY, class: "axis-zero" }));
      svg.appendChild(svgEl("path", { d: pathFromRows(rows, x, y, "session_s", "lateral_sway_m"), class: "sway-line" }));
      events.forEach(event => {
        const cx = x(event.session_s);
        const cy = y(event.lateral_sway_m);
        svg.appendChild(svgEl("circle", { cx, cy, r: 5.8, fill: sideColor(event), stroke: confidenceStroke(event), "stroke-width": 2 }));
        if (byId("showLabels").checked) {
          svg.appendChild(svgEl("text", { x: cx + 7, y: cy - 7, class: "event-label" })).textContent = event.event_index;
        }
      });
    }

    function eventLinePath(events, field, xScale, yScale) {
      return events
        .filter(event => Number.isFinite(Number(event[field])))
        .map((event, index) => `${index === 0 ? "M" : "L"} ${xScale(event.session_s)} ${yScale(event[field])}`)
        .join(" ");
    }

    function drawLengths(context, events) {
      const svg = byId("lengthPlot");
      clear(svg);
      const xDomain = [Number(context.start_s), Number(context.end_s)];
      const values = events.flatMap(event => [event.valid_step_path_length_m, event.valid_stride_path_length_m]).filter(v => Number.isFinite(Number(v)));
      const yDomain = extent(values.length ? values : [0, 1], 0.20);
      yDomain[0] = Math.min(0, yDomain[0]);
      const { x, y } = drawAxes(svg, xDomain, yDomain, W, H_SMALL, "session time (s)", "proxy length (m)");
      svg.appendChild(svgEl("path", { d: eventLinePath(events, "valid_step_path_length_m", x, y), class: "step-line" }));
      svg.appendChild(svgEl("path", { d: eventLinePath(events, "valid_stride_path_length_m", x, y), class: "stride-line" }));
      events.forEach(event => {
        if (Number.isFinite(Number(event.valid_step_path_length_m))) {
          svg.appendChild(svgEl("circle", { cx: x(event.session_s), cy: y(event.valid_step_path_length_m), r: 5, fill: "var(--green)", stroke: "#fff", "stroke-width": 1.4 }));
        }
        if (Number.isFinite(Number(event.valid_stride_path_length_m))) {
          svg.appendChild(svgEl("rect", { x: x(event.session_s) - 5, y: y(event.valid_stride_path_length_m) - 5, width: 10, height: 10, fill: "var(--purple)", stroke: "#fff", "stroke-width": 1.4 }));
        }
      });
      svg.appendChild(svgEl("text", { x: W - M.right, y: 20, "text-anchor": "end", class: "axis-label", fill: "var(--green)" })).textContent = "green = step proxy";
      svg.appendChild(svgEl("text", { x: W - M.right, y: 38, "text-anchor": "end", class: "axis-label", fill: "var(--purple)" })).textContent = "purple = stride proxy";
    }

    function updateMetrics(context, bout, rows, events) {
      const metricGrid = byId("metricGrid");
      const metrics = context.scope === "full"
        ? [
            ["Trial", context.trial_id],
            ["View", "Full trial"],
            ["Duration", `${fmt(context.start_s, 1, "s")} - ${fmt(context.end_s, 1, "s")}`],
            ["Trajectory Points", String(rows.length)],
            ["Step Events", String(events.length)],
            ["Stride Bouts", String(rowsForTrial(DATA.bouts, context.trial_id).length)],
            ["X Range", fmt(Math.max(...rows.map(row => Number(row.tx_world_device))) - Math.min(...rows.map(row => Number(row.tx_world_device))), 2, " m")],
            ["Y Range", fmt(Math.max(...rows.map(row => Number(row.ty_world_device))) - Math.min(...rows.map(row => Number(row.ty_world_device))), 2, " m")],
            ["Height Range", fmt(Math.max(...rows.map(row => Number(row.tz_world_device))) - Math.min(...rows.map(row => Number(row.tz_world_device))), 2, " m")],
            ["Local Bout", `${labelNames[bout.label] || bout.label} ${fmt(bout.bout_start_s, 1)}-${fmt(bout.bout_end_s, 1)}s`]
          ]
        : [
            ["Label", labelNames[bout.label] || bout.label],
            ["Window", `${fmt(bout.bout_start_s, 1, "s")} - ${fmt(bout.bout_end_s, 1, "s")}`],
            ["Confidence", bout.confidence || "N/A"],
            ["Cadence", fmt(bout.cadence_steps_per_s, 2, " events/s")],
            ["Median Step Proxy", fmt(bout.median_step_path_length_m, 3, " m")],
            ["Median Stride Proxy", fmt(bout.median_stride_path_length_m, 3, " m")],
            ["Path Length", fmt(bout.horizontal_path_length_m, 2, " m")],
            ["Heading Change", fmt(bout.heading_change_deg, 1, " deg")],
            ["Step Events", String(events.length)],
            ["Straightness", fmt(bout.straightness_net_over_path, 2)]
          ];
      metricGrid.innerHTML = metrics.map(([name, value]) => `<div class="metric"><small>${name}</small><b>${value}</b></div>`).join("");

      byId("boutNarrative").textContent = context.scope === "full"
        ? `${context.trial_id} full-trial view. This shows the complete downsampled SLAM X/Y path and all currently detected stride events for that trial. The lateral-sway plot below remains tied to the selected bout because lateral sway is computed in a local direction frame.`
        : `${bout.trial_id}, ${labelNames[bout.label] || bout.label}, ${fmt(bout.bout_start_s, 1)}-${fmt(bout.bout_end_s, 1)}s. ` +
          `Use this window to check whether SLAM step markers match the visible downward-video rhythm. ` +
          (bout.warning ? `Current warning: ${bout.warning}` : "No major diagnostic warning for this window.");

      byId("relatedLinks").innerHTML = `
        <a class="button-link" href="../${bout.trial_id}/pd_insighter_stride_review.html">Open Trial Stride Review</a>
        <a class="button-link" href="../${bout.trial_id}/index.html">Open PD Insighter</a>
      `;
    }

    function renderEventTable(events) {
      const tbody = byId("eventTable");
      tbody.innerHTML = events.map(event => {
        const selected = manualValue(event);
        const options = ["", "left", "right", "uncertain"].map(value => {
          const label = value || "not checked";
          return `<option value="${value}" ${selected === value ? "selected" : ""}>${label}</option>`;
        }).join("");
        const warning = [event.event_warning, event.foot_side_warning].filter(Boolean).join("; ");
        return `
          <tr>
            <td>${event.event_index}</td>
            <td>${fmt(event.session_s, 2, "s")}</td>
            <td>${event.estimated_foot_side || "N/A"}</td>
            <td><select class="manual-select" data-key="${manualKey(event)}">${options}</select></td>
            <td>${event.foot_side_confidence || "N/A"}</td>
            <td>${event.event_confidence || "N/A"}</td>
            <td>${fmt(event.valid_step_path_length_m, 3, " m")}</td>
            <td>${fmt(event.valid_stride_path_length_m, 3, " m")}</td>
            <td>${fmt(event.lateral_sway_m, 3, " m")}</td>
            <td class="warning">${warning || ""}</td>
          </tr>
        `;
      }).join("");
      tbody.querySelectorAll("select.manual-select").forEach(select => {
        select.addEventListener("change", event => {
          const el = event.target;
          if (el.value) localStorage.setItem(el.dataset.key, el.value);
          else localStorage.removeItem(el.dataset.key);
        });
      });
    }

    function renderSummary() {
      const rows = DATA.label_summary || [];
      const head = byId("summaryHead");
      const body = byId("summaryBody");
      if (!rows.length) {
        head.innerHTML = "";
        body.innerHTML = "<tr><td>No summary data available.</td></tr>";
        return;
      }
      const columns = Object.keys(rows[0]);
      head.innerHTML = `<tr>${columns.map(column => `<th>${column}</th>`).join("")}</tr>`;
      body.innerHTML = rows.map(row => `<tr>${columns.map(column => `<td>${typeof row[column] === "number" ? fmt(row[column], 3) : (row[column] ?? "")}</td>`).join("")}</tr>`).join("");
    }

    function populateControls() {
      const trialSelect = byId("trialSelect");
      trialSelect.innerHTML = DATA.trials.map(trial => `<option value="${trial}">${trial}</option>`).join("");
      trialSelect.addEventListener("change", () => {
        populateBouts();
        render();
      });
      byId("scopeSelect").addEventListener("change", render);
      byId("boutSelect").addEventListener("change", render);
      byId("showArrows").addEventListener("change", render);
      byId("showLabels").addEventListener("change", render);
      byId("xyZoomIn").addEventListener("click", () => {
        zoomXYDomain(0.72);
        render();
      });
      byId("xyZoomOut").addEventListener("click", () => {
        zoomXYDomain(1.38);
        render();
      });
      byId("xyResetZoom").addEventListener("click", () => {
        resetXYZoom();
        render();
      });
      populateBouts();
    }

    function populateBouts() {
      const trial = currentTrial();
      const bouts = DATA.bouts.filter(row => row.trial_id === trial);
      const select = byId("boutSelect");
      select.innerHTML = bouts.map(row => {
        const label = labelNames[row.label] || row.label;
        const confidence = row.confidence ? `, ${row.confidence}` : "";
        return `<option value="${boutKey(row)}">${label}: ${fmt(row.bout_start_s, 1)}-${fmt(row.bout_end_s, 1)}s${confidence}</option>`;
      }).join("");
    }

    function render() {
      const bout = currentBout();
      const trial = currentTrial();
      const scope = currentScope();
      const boutRows = rowsForBout(DATA.series, bout);
      const boutEvents = rowsForBout(DATA.events, bout);
      const fullRows = rowsForTrial(DATA.full_series || [], trial);
      const fullEvents = rowsForTrial(DATA.events || [], trial);
      const rows = scope === "full" && fullRows.length ? fullRows : boutRows;
      const events = scope === "full" ? fullEvents : boutEvents;
      const start_s = rows.length ? Number(rows[0].session_s) : Number(bout.bout_start_s);
      const end_s = rows.length ? Number(rows[rows.length - 1].session_s) : Number(bout.bout_end_s);
      const context = { trial_id: trial, scope, start_s, end_s };
      drawXY(context, rows, events);
      drawHeightSpeed(context, rows, events);
      drawSway(bout, boutRows, boutEvents);
      drawLengths(context, events);
      updateMetrics(context, bout, rows, events);
      renderEventTable(events);
    }

    function exportManualCorrections() {
      const rows = [];
      (DATA.events || []).forEach(event => {
        const manual_side = manualValue(event);
        if (!manual_side) return;
        rows.push({
          trial_id: event.trial_id,
          bout_id: event.bout_id,
          event_index: event.event_index,
          session_s: event.session_s,
          auto_side: event.estimated_foot_side || "",
          manual_side,
          auto_foot_confidence: event.foot_side_confidence || "",
          event_confidence: event.event_confidence || ""
        });
      });
      const columns = ["trial_id", "bout_id", "event_index", "session_s", "auto_side", "manual_side", "auto_foot_confidence", "event_confidence"];
      const csv = [columns.join(",")].concat(rows.map(row => columns.map(column => {
        const value = row[column] ?? "";
        return `"${String(value).replaceAll('"', '""')}"`;
      }).join(","))).join("\\n");
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "stride_manual_side_corrections.csv";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }

    function clearManualCorrections() {
      if (!confirm("Clear manual side corrections stored in this browser for this plot page?")) return;
      Object.keys(localStorage).forEach(key => {
        if (key.startsWith(manualPrefix)) localStorage.removeItem(key);
      });
      render();
    }

    byId("exportManual").addEventListener("click", exportManualCorrections);
    byId("clearManual").addEventListener("click", clearManualCorrections);
    window.addEventListener("pointermove", event => {
      if (!xyDragState) return;
      const dx = event.clientX - xyDragState.lastX;
      const dy = event.clientY - xyDragState.lastY;
      xyDragState.lastX = event.clientX;
      xyDragState.lastY = event.clientY;
      panXYDomain(dx, dy);
      render();
    });
    window.addEventListener("pointerup", () => {
      xyDragState = null;
    });

    populateControls();
    renderSummary();
    render();
  </script>
</body>
</html>
"""


def build_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return HTML_TEMPLATE.replace("__DATA_JSON__", data_json)


def main() -> None:
    payload = build_payload()
    if not payload["bouts"]:
        raise SystemExit(
            "No stride diagnostics found. Run scripts/generate_step_stride_diagnostics.py first."
        )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(build_html(payload), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
