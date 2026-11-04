#!/usr/bin/env python3
"""Generate instructor-facing plots for SLAM step and left/right logic."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "runs" / "stride_step_diagnostics" / "data"
OUT_DIR = (
    REPO_ROOT
    / "runs"
    / "stride_step_diagnostics"
    / "left_right_step_logic_plots"
)
ALL_TRIALS_DIR = OUT_DIR / "all_trials"

EXAMPLE_CLEAN = ("Ntrial3", 0)
EXAMPLE_STRAIGHT = ("Ntrial2", 1)
EXAMPLE_TURN = ("Ntrial2", 0)

SIDE_COLORS = {
    "left": "#2f80c0",
    "right": "#d44c4c",
    "unknown": "#777777",
}

PATH_SIDE_COLORS = {
    "lateral_positive": "#6f4dbf",
    "lateral_negative": "#df7f2f",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(DATA_DIR / "step_stride_event_estimates.csv")
    timeseries = pd.read_csv(DATA_DIR / "step_stride_resampled_bout_timeseries.csv")
    bouts = pd.read_csv(DATA_DIR / "step_stride_bout_summary.csv")
    return events, timeseries, bouts


def bout_frame(df: pd.DataFrame, trial_id: str, bout_id: int) -> pd.DataFrame:
    return df[(df["trial_id"] == trial_id) & (df["bout_id"] == bout_id)].copy()


def nearest_value(ts: pd.DataFrame, time_s: float, column: str) -> float:
    if pd.isna(time_s) or ts.empty:
        return float("nan")
    idx = (ts["session_s"] - time_s).abs().idxmin()
    return float(ts.loc[idx, column])


def nearest_row(ts: pd.DataFrame, time_s: float) -> pd.Series:
    idx = (ts["session_s"] - time_s).abs().idxmin()
    return ts.loc[idx]


def forward_and_path_left(local_heading_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(local_heading_deg)
    forward = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    path_left = np.array([-forward[1], forward[0]], dtype=float)
    return forward, path_left


def side_color(side: str | float) -> str:
    if not isinstance(side, str):
        return SIDE_COLORS["unknown"]
    return SIDE_COLORS.get(side.lower(), SIDE_COLORS["unknown"])


def slug(value: object) -> str:
    text = str(value).strip().lower() or "unknown"
    out: list[str] = []
    for char in text:
        if char.isalnum():
            out.append(char)
        elif char in {" ", "-", "_", "/"}:
            out.append("_")
    return "_".join("".join(out).split("_"))


def side_short(side: object) -> str:
    if isinstance(side, str) and side:
        return side[0].upper()
    return "?"


def add_common_footer(fig: plt.Figure, text: str) -> None:
    fig.text(
        0.01,
        0.01,
        text,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#4f5d52",
    )


def plot_lateral_sway_step_timing(
    events: pd.DataFrame,
    timeseries: pd.DataFrame,
    bouts: pd.DataFrame,
    trial_id: str = EXAMPLE_CLEAN[0],
    bout_id: int = EXAMPLE_CLEAN[1],
    out_dir: Path = OUT_DIR,
    out_name: str = "01_clean_bout_lateral_sway_step_timing.png",
) -> Path:
    ev = bout_frame(events, trial_id, bout_id)
    ts = bout_frame(timeseries, trial_id, bout_id)
    bout = bout_frame(bouts, trial_id, bout_id).iloc[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(
        ts["session_s"],
        ts["lateral_sway_m"],
        color="#1f2d35",
        linewidth=2.2,
        label="lateral_sway_m",
    )
    ax.axhline(0, color="#66706a", linewidth=1.2)
    ax.fill_between(
        ts["session_s"],
        0,
        ts["lateral_sway_m"],
        where=ts["lateral_sway_m"] >= 0,
        color="#e8ddff",
        alpha=0.85,
        label="positive = head/device is path-left",
    )
    ax.fill_between(
        ts["session_s"],
        0,
        ts["lateral_sway_m"],
        where=ts["lateral_sway_m"] < 0,
        color="#ffe5cf",
        alpha=0.85,
        label="negative = head/device is path-right",
    )

    peak_seen: set[tuple[float, str]] = set()
    for _, row in ev.iterrows():
        for time_col, side_col in [
            ("previous_sway_peak_session_s", "previous_sway_peak_side_proxy"),
            ("sway_peak_session_s", "sway_peak_side_proxy"),
        ]:
            peak_time = row.get(time_col)
            side = row.get(side_col)
            if pd.isna(peak_time) or not isinstance(side, str):
                continue
            key = (round(float(peak_time), 3), side)
            if key in peak_seen:
                continue
            peak_seen.add(key)
            ax.scatter(
                peak_time,
                nearest_value(ts, float(peak_time), "lateral_sway_m"),
                s=95,
                marker="x",
                color=PATH_SIDE_COLORS.get(side, "#555555"),
                linewidth=2.4,
                zorder=5,
            )

    for _, row in ev.iterrows():
        color = side_color(row.get("estimated_foot_side", "unknown"))
        ax.axvline(row["session_s"], color=color, alpha=0.35, linewidth=1.5)
        ax.scatter(
            row["session_s"],
            row["lateral_sway_m"],
            color=color,
            edgecolor="white",
            linewidth=1.0,
            s=95,
            zorder=6,
        )
        ax.text(
            row["session_s"],
            row["lateral_sway_m"] + 0.004,
            f"{int(row['event_index'])}\n{side_short(row.get('estimated_foot_side'))}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
            weight="bold",
        )

    ax.set_title(
        f"{trial_id} bout {bout_id}: step timing from lateral head-sway extrema",
        fontsize=16,
        weight="bold",
    )
    ax.set_xlabel("session time (s)")
    ax.set_ylabel("lateral_sway_m\n(+ path-left, - path-right)")
    ax.legend(loc="upper left", ncols=2, fontsize=9)
    ax.grid(True, alpha=0.22)
    ax.text(
        0.99,
        0.04,
        "Colored vertical lines = estimated step events.\n"
        "The event time is the midpoint between adjacent lateral extrema,\n"
        "so it does not need to equal the lowest Z height point.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", alpha=0.9),
    )
    add_common_footer(
        fig,
        f"Bout window: {bout['bout_start_s']:.1f}s-{bout['bout_end_s']:.1f}s. "
        "Blue = estimated left, red = estimated right.",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = out_dir / out_name
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_left_right_alignment(
    events: pd.DataFrame,
    trial_id: str,
    bout_id: int,
    out_name: str,
    title_suffix: str,
    out_dir: Path = OUT_DIR,
) -> Path:
    ev = bout_frame(events, trial_id, bout_id).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(ev))
    colors = [side_color(v) for v in ev["estimated_foot_side"]]
    ax.bar(x, ev["foot_side_alignment"], color=colors, alpha=0.88)
    ax.axhline(0, color="#20292f", linewidth=1.3)
    ax.axhspan(0, 1.0, color="#dfefff", alpha=0.35)
    ax.axhspan(-1.0, 0, color="#ffe2e2", alpha=0.35)
    for i, row in ev.iterrows():
        side_short = "L" if row["estimated_foot_side"] == "left" else "R"
        y = row["foot_side_alignment"]
        ax.text(
            i,
            y + (0.045 if y >= 0 else -0.075),
            f"{side_short}\n{row['session_s']:.2f}s",
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=8,
            weight="bold",
            color=side_color(row["estimated_foot_side"]),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(ev["event_index"].astype(int).astype(str))
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("detected step event index")
    ax.set_ylabel("foot_side_alignment\n+ means aligns with device-left")
    ax.set_title(f"{trial_id} bout {bout_id}: left/right sign rule {title_suffix}", fontsize=15, weight="bold")
    ax.text(
        0.02,
        0.94,
        "Rule shown here:\n"
        "foot_side_alignment = dot(sway-side direction, device-left direction)\n"
        "positive alignment -> estimated LEFT\n"
        "negative alignment -> estimated RIGHT",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", alpha=0.92),
    )
    ax.grid(True, axis="y", alpha=0.24)
    fig.tight_layout()
    out = out_dir / out_name
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_xy_path_axes(
    events: pd.DataFrame,
    timeseries: pd.DataFrame,
    bouts: pd.DataFrame,
    trial_id: str = EXAMPLE_STRAIGHT[0],
    bout_id: int = EXAMPLE_STRAIGHT[1],
    out_dir: Path = OUT_DIR,
    out_name: str = "03_xy_path_direction_and_left_right_axes.png",
) -> Path:
    ev = bout_frame(events, trial_id, bout_id)
    ts = bout_frame(timeseries, trial_id, bout_id)
    bout = bout_frame(bouts, trial_id, bout_id).iloc[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.plot(
        ts["tx_world_device"],
        ts["ty_world_device"],
        color="#242b2f",
        linewidth=2.4,
        label="SLAM X/Y head path",
    )
    ax.scatter(
        ev["tx_world_device_m"],
        ev["ty_world_device_m"],
        c=[side_color(v) for v in ev["estimated_foot_side"]],
        s=90,
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
        label="estimated step events",
    )

    path_len = max(
        ts["tx_world_device"].max() - ts["tx_world_device"].min(),
        ts["ty_world_device"].max() - ts["ty_world_device"].min(),
    )
    arrow_len = max(path_len * 0.07, 0.16)

    for _, row in ev.iterrows():
        x = float(row["tx_world_device_m"])
        y = float(row["ty_world_device_m"])
        forward, path_left = forward_and_path_left(float(row["local_heading_deg"]))
        near = nearest_row(ts, float(row["session_s"]))
        device_left = np.array(
            [near["device_left_x_world"], near["device_left_y_world"]], dtype=float
        )
        if np.linalg.norm(device_left) > 0:
            device_left = device_left / np.linalg.norm(device_left)

        ax.arrow(
            x,
            y,
            forward[0] * arrow_len,
            forward[1] * arrow_len,
            width=0.012,
            head_width=0.065,
            length_includes_head=True,
            color="#16835b",
            alpha=0.75,
            zorder=4,
        )
        ax.arrow(
            x,
            y,
            path_left[0] * arrow_len * 0.82,
            path_left[1] * arrow_len * 0.82,
            width=0.009,
            head_width=0.052,
            length_includes_head=True,
            color="#7655c7",
            alpha=0.72,
            zorder=4,
        )
        ax.arrow(
            x,
            y,
            device_left[0] * arrow_len * 0.62,
            device_left[1] * arrow_len * 0.62,
            width=0.007,
            head_width=0.044,
            length_includes_head=True,
            color="#d88320",
            alpha=0.72,
            zorder=4,
        )
        ax.text(
            x,
            y,
            str(int(row["event_index"])),
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            weight="bold",
            zorder=6,
        )

    ax.plot([], [], color="#16835b", linewidth=3, label="local forward / travel direction")
    ax.plot([], [], color="#7655c7", linewidth=3, label="local path-left axis")
    ax.plot([], [], color="#d88320", linewidth=3, label="device-left axis from orientation")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("tx_world_device (m)")
    ax.set_ylabel("ty_world_device (m)")
    ax.set_title(
        f"{trial_id} bout {bout_id}: X/Y path, travel direction, and left/right axes",
        fontsize=15,
        weight="bold",
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.23)
    ax.text(
        0.02,
        0.02,
        "The arrows start at the actual SLAM X/Y step-event position.\n"
        "Forward/path-left come from the local X/Y path. Device-left comes from orientation.\n"
        "Left/right is decided by whether the sway-side axis aligns with device-left.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", alpha=0.92),
    )
    add_common_footer(
        fig,
        f"Bout window: {bout['bout_start_s']:.1f}s-{bout['bout_end_s']:.1f}s. "
        "Raw world X/Y is not assumed to be forward; forward is recalculated locally.",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = out_dir / out_name
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_turning_uncertainty(
    events: pd.DataFrame,
    timeseries: pd.DataFrame,
    bouts: pd.DataFrame,
    trial_id: str = EXAMPLE_TURN[0],
    bout_id: int = EXAMPLE_TURN[1],
    out_dir: Path = OUT_DIR,
    out_name: str = "04_turning_bout_why_left_right_confidence_drops.png",
) -> Path:
    ev = bout_frame(events, trial_id, bout_id).reset_index(drop=True)
    ts = bout_frame(timeseries, trial_id, bout_id)
    bout = bout_frame(bouts, trial_id, bout_id).iloc[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    axes[0].plot(ts["session_s"], ts["lateral_sway_m"], color="#1f2d35", linewidth=2)
    axes[0].axhline(0, color="#66706a", linewidth=1)
    axes[0].scatter(
        ev["session_s"],
        ev["lateral_sway_m"],
        c=[side_color(v) for v in ev["estimated_foot_side"]],
        s=80,
        edgecolor="white",
        zorder=5,
    )
    axes[0].set_ylabel("lateral_sway_m")
    axes[0].set_title(
        f"{trial_id} bout {bout_id}: same step logic, but turning lowers confidence",
        fontsize=15,
        weight="bold",
    )
    axes[0].grid(True, alpha=0.22)
    axes[0].text(
        0.02,
        0.92,
        f"Heading changes by {bout['heading_change_deg']:.1f} degrees in this bout.\n"
        "The rhythmic peaks are still present, but left/right mapping is less certain\n"
        "because local forward and head/device orientation keep changing.",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", alpha=0.92),
    )

    x = np.arange(len(ev))
    axes[1].bar(x, ev["foot_side_alignment"], color=[side_color(v) for v in ev["estimated_foot_side"]], alpha=0.88)
    axes[1].axhline(0, color="#20292f", linewidth=1.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{int(i)}\n{t:.1f}s" for i, t in zip(ev["event_index"], ev["session_s"])])
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_ylabel("foot_side_alignment")
    axes[1].set_xlabel("detected step event")
    axes[1].grid(True, axis="y", alpha=0.22)

    heading_ax = axes[1].twinx()
    heading_ax.plot(
        np.arange(len(ev)),
        ev["local_heading_deg"],
        color="#6b6f76",
        linestyle="--",
        marker="o",
        linewidth=1.5,
        label="local heading angle",
    )
    heading_ax.set_ylabel("local heading angle (deg)", color="#6b6f76")

    fig.tight_layout()
    out = out_dir / out_name
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_step_stride_lengths(
    events: pd.DataFrame,
    trial_id: str = EXAMPLE_CLEAN[0],
    bout_id: int = EXAMPLE_CLEAN[1],
    out_dir: Path = OUT_DIR,
    out_name: str = "05_step_and_stride_proxy_lengths.png",
) -> Path:
    ev = bout_frame(events, trial_id, bout_id).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(
        ev["event_index"],
        ev["valid_step_path_length_m"],
        marker="o",
        linewidth=2.2,
        color="#1f78b4",
        label="step proxy length: current event - previous event",
    )
    ax.plot(
        ev["event_index"],
        ev["valid_stride_path_length_m"],
        marker="s",
        linewidth=2.2,
        color="#7b4dbd",
        label="stride proxy length: current event - event two steps before",
    )
    ax.scatter(
        ev["event_index"],
        ev["valid_step_path_length_m"],
        c=[side_color(v) for v in ev["estimated_foot_side"]],
        s=95,
        edgecolor="white",
        zorder=5,
    )
    ax.set_xlabel("detected step event index")
    ax.set_ylabel("horizontal path distance (m)")
    ax.set_title(
        f"{trial_id} bout {bout_id}: how step and stride proxy lengths are measured",
        fontsize=15,
        weight="bold",
    )
    ax.grid(True, alpha=0.24)
    ax.legend(loc="upper left")
    ax.text(
        0.98,
        0.95,
        "Important: these are head/device path distances,\n"
        "not direct foot-to-foot distances.\n"
        "They should be validated with downward video.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", alpha=0.92),
    )
    fig.tight_layout()
    out = out_dir / out_name
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_gallery(paths: list[Path]) -> Path:
    gallery = OUT_DIR / "left_right_step_logic_gallery.html"
    cards = []
    for path in paths:
        cards.append(
            f"""
            <section class="card">
              <h2>{path.stem.replace('_', ' ')}</h2>
              <a href="{path.name}" target="_blank"><img src="{path.name}" alt="{path.stem}"></a>
            </section>
            """
        )
    gallery.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Left/Right Step Logic Plot Gallery</title>
  <style>
    :root {{
      --ink: #17211d;
      --muted: #647067;
      --paper: #f7f4ec;
      --card: #ffffff;
      --accent: #1d7a5d;
    }}
    body {{
      margin: 0;
      font-family: Avenir Next, Helvetica Neue, sans-serif;
      background: linear-gradient(135deg, #f8f5ec 0%, #edf4ef 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 1220px;
      margin: 0 auto;
      padding: 36px 22px 60px;
    }}
    .hero {{
      border: 1px solid #dbe4dc;
      border-radius: 28px;
      padding: 26px 30px;
      background: rgba(255, 255, 255, 0.86);
      box-shadow: 0 18px 44px rgba(31, 51, 42, 0.10);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(28px, 4vw, 44px);
      letter-spacing: -0.04em;
    }}
    p {{
      line-height: 1.55;
      color: var(--muted);
      font-size: 16px;
    }}
    .card {{
      margin-top: 26px;
      padding: 20px;
      border-radius: 24px;
      background: var(--card);
      border: 1px solid #dde7df;
      box-shadow: 0 14px 34px rgba(31, 51, 42, 0.08);
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 20px;
    }}
    img {{
      display: block;
      width: 100%;
      border-radius: 16px;
      border: 1px solid #e0e5df;
      background: white;
    }}
    code {{
      background: #eef4ef;
      border-radius: 7px;
      padding: 2px 6px;
      color: var(--accent);
    }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <h1>Left/Right Step Logic Plot Gallery</h1>
    <p>
      These plots explain how the current SLAM-only step proxy is detected and how
      path-side sway is mapped to estimated left/right steps. The key signs are:
      <code>lateral_sway_m &gt; 0</code> means path-left, 
      <code>lateral_sway_m &lt; 0</code> means path-right, and
      <code>foot_side_alignment &gt; 0</code> maps the event to estimated left.
    </p>
  </section>
  {''.join(cards)}
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return gallery


def write_readme(paths: list[Path], gallery: Path) -> Path:
    readme = OUT_DIR / "README.md"
    lines = [
        "# Left/Right Step Logic Plots",
        "",
        "These plots are intended to accompany `SLAM_COORDINATES_STEP_STRIDE_EXPLANATION.md`.",
        "",
        f"Gallery: `{gallery}`",
        "",
        "Complete all-trials gallery: `all_trials/index.html`",
        "",
        "## Main Idea",
        "",
        "- Raw `tx_world_device` / `ty_world_device` are horizontal MPS world coordinates; raw Y is not assumed to be walking-forward.",
        "- Local forward direction is calculated from the smoothed X/Y walking path.",
        "- Local path-left is the direction 90 degrees left of that local forward direction.",
        "- `lateral_sway_m > 0` means the head/device is on the local path-left side of the smoothed path.",
        "- `lateral_sway_m < 0` means the head/device is on the local path-right side.",
        "- Step events are placed at the midpoint between neighboring positive/negative lateral-sway extrema.",
        "- Anatomical left/right is estimated by comparing the sway-side direction with the device-left axis from Aria orientation.",
        "- Positive `foot_side_alignment` means estimated left; negative means estimated right.",
        "",
        "## Raw SLAM Columns We Look At",
        "",
        "Raw SLAM file: `VRS_Pre_Processing/data/<trial>/mps_*_vrs/slam/closed_loop_trajectory.csv`.",
        "",
        "- `tracking_timestamp_us`: raw device tracking time; converted into `session_s` for plotting and video alignment.",
        "- `tx_world_device`, `ty_world_device`, `tz_world_device`: raw Aria/device position in the MPS world/map frame. X/Y are horizontal position; Z is height/up.",
        "- `qx_world_device`, `qy_world_device`, `qz_world_device`, `qw_world_device`: raw device orientation quaternion. We use it to estimate device-forward and device-left axes.",
        "- `device_linear_velocity_x_device`, `device_linear_velocity_y_device`, `device_linear_velocity_z_device`: raw velocity in the device frame. We combine them into speed magnitude.",
        "- `quality_score`: raw SLAM pose quality score; useful for checking whether the pose should be trusted.",
        "",
        "## Processed Parameters We Created",
        "",
        "- `session_s`: processed time in seconds, used to match SLAM features to videos.",
        "- `local_heading_deg`: processed travel direction from the smoothed X/Y path. This avoids assuming raw X or Y is forward.",
        "- `lateral_sway_m`: processed side-to-side head movement relative to travel direction. Positive means path-left; negative means path-right.",
        "- `sway_peak_session_s`: processed time of a lateral head-sway peak/trough.",
        "- `session_s` in the event table: processed step-event proxy time, currently midpoint between two neighboring lateral extrema.",
        "- `horizontal_path_progress_m`: processed cumulative horizontal path distance inside the bout.",
        "- `valid_step_path_length_m`: processed distance between neighboring step-event proxies along the X/Y head path.",
        "- `valid_stride_path_length_m`: processed distance between an event and the event two steps before along the X/Y head path.",
        "- `estimated_foot_side`: processed initial left/right estimate from comparing sway-side direction with device-left direction.",
        "- `foot_side_alignment`: processed signed alignment score. Positive maps to estimated left; negative maps to estimated right.",
        "",
        "Important warning: these are SLAM head/device proxies. They are useful for gait reasoning, but they are not direct foot-contact measurements and should be checked with downward video.",
        "",
        "## How `lateral_sway_m` Is Calculated",
        "",
        "1. Resample the raw SLAM trajectory inside each locomotion bout to 50 Hz.",
        "2. Interpolate raw `tx_world_device`, `ty_world_device`, `tz_world_device`, velocity, and quaternion values onto that common time axis.",
        "3. Smooth raw X and Y using a centered 0.75 s rolling mean. This gives the local path centerline.",
        "4. Compute the gradient of the smoothed centerline and normalize it. This is the local travel/forward direction.",
        "5. Compute path-left as 90 degrees left of local forward: `path_left = [-forward_y, forward_x]`.",
        "6. Compute residual head movement relative to the centerline: `residual_xy = raw_xy - centerline_xy`.",
        "7. Project the residual onto path-left: `lateral_sway_m = dot(residual_xy, path_left)`.",
        "8. Smooth `lateral_sway_m` using a short centered 0.12 s rolling mean.",
        "",
        "Interpretation: positive `lateral_sway_m` means the head/device is on the path-left side of the smoothed walking path; negative means path-right.",
        "",
        "## How Step And Stride Proxy Events Are Calculated",
        "",
        "- We detect positive and negative extrema in `lateral_sway_m` using an adaptive prominence threshold: `max(0.006 m, 14% of the 95th-5th percentile lateral range)`.",
        "- Same-side peaks that are too close together are merged so small noisy bumps do not become extra steps.",
        "- The step proxy event time is placed at the midpoint between neighboring lateral extrema.",
        "- `valid_step_path_length_m` is the change in cumulative X/Y head path distance between neighboring step proxy events.",
        "- `valid_stride_path_length_m` is the change in cumulative X/Y head path distance between the current event and the event two steps before.",
        "",
        "This explains why the detected step marker may not sit at the lowest Z height point. We are detecting a lateral head-sway rhythm proxy, not foot contact directly.",
        "",
        "## How Left/Right Is Estimated",
        "",
        "1. Rotate the device-forward axis with the raw quaternion to estimate device-forward in world coordinates.",
        "2. Align device-forward with local path-forward so the direction is not flipped backward.",
        "3. Compute device-left as 90 degrees left of device-forward.",
        "4. Compare the sway-side direction with device-left using a dot product.",
        "5. If `foot_side_alignment > 0`, the event is estimated left; if `< 0`, estimated right.",
        "",
        "Confidence is lower when the person is turning, when device orientation disagrees with path direction, or when the lateral sway signal is weak.",
        "",
        "## Plot Files",
        "",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme


def html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --ink: #17211d;
      --muted: #647067;
      --card: #ffffff;
      --line: #dbe4dc;
      --accent: #1d7a5d;
      --paper: #f7f4ec;
    }}
    body {{
      margin: 0;
      font-family: Avenir Next, Helvetica Neue, sans-serif;
      background: linear-gradient(135deg, #f8f5ec 0%, #edf4ef 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 34px 20px 62px;
    }}
    .hero, .card {{
      background: rgba(255,255,255,.9);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: 0 14px 34px rgba(31, 51, 42, .08);
    }}
    .hero {{ padding: 24px 28px; margin-bottom: 22px; }}
    .card {{ padding: 18px; margin-top: 18px; }}
    h1 {{ margin: 0 0 10px; font-size: clamp(28px, 4vw, 44px); letter-spacing: -.04em; }}
    h2 {{ margin: 0 0 10px; font-size: 20px; letter-spacing: -.02em; }}
    p {{ color: var(--muted); line-height: 1.55; }}
    a {{ color: var(--accent); font-weight: 800; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 16px; overflow: hidden; }}
    th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #e5ece6; font-size: 14px; }}
    th {{ color: var(--muted); text-transform: uppercase; letter-spacing: .06em; font-size: 12px; }}
    img {{ display: block; width: 100%; border-radius: 16px; border: 1px solid #e0e5df; background: white; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
    .control-panel {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; align-items: end; }}
    label {{ display: block; color: var(--muted); text-transform: uppercase; font-size: 12px; font-weight: 800; letter-spacing: .06em; margin-bottom: 6px; }}
    select, button {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: white;
      color: var(--ink);
      font: inherit;
      font-weight: 700;
      padding: 11px 12px;
    }}
    button {{ background: var(--accent); color: white; cursor: pointer; }}
    button:hover {{ filter: brightness(.96); }}
    .parameter-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; margin-top: 14px; }}
    .param-card {{ border: 1px solid #e2ebe4; border-radius: 18px; padding: 14px; background: #fbfdfb; }}
    .param-card h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .param-card p {{ margin: 0; font-size: 13px; }}
    .steps {{ counter-reset: step; display: grid; gap: 10px; margin-top: 14px; }}
    .step {{ border-left: 4px solid #b8d8c7; padding: 10px 12px; background: #fbfdfb; border-radius: 12px; }}
    .step:before {{ counter-increment: step; content: counter(step); display: inline-flex; width: 22px; height: 22px; align-items: center; justify-content: center; border-radius: 999px; background: var(--accent); color: white; font-size: 12px; font-weight: 900; margin-right: 8px; }}
    .step b {{ font-size: 14px; }}
    .step p {{ margin: 6px 0 0 31px; font-size: 13px; }}
    .tag-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 2px; }}
    .pill {{ display: inline-flex; border: 1px solid var(--line); border-radius: 999px; padding: 6px 10px; background: white; font-size: 12px; font-weight: 800; color: var(--muted); }}
    .raw {{ color: #2c6f55; border-color: #b8d8c7; background: #eef8f2; }}
    .processed {{ color: #7655c7; border-color: #d5c8f2; background: #f3efff; }}
    .warning {{ color: #a45f15; border-color: #edd0a3; background: #fff7e7; }}
    .muted {{ color: var(--muted); }}
    code {{ background: #eef4ef; border-radius: 7px; padding: 2px 6px; color: var(--accent); }}
  </style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


def slam_parameter_panel(*, compact: bool = False) -> str:
    intro = (
        "This panel separates raw MPS SLAM columns from the processed parameters "
        "we created for stride and left/right reasoning. The raw file to inspect is "
        "<code>VRS_Pre_Processing/data/&lt;trial&gt;/mps_*_vrs/slam/closed_loop_trajectory.csv</code>."
    )
    raw_cards = [
        (
            "Position: tx/ty/tz_world_device",
            "Raw SLAM translation of the Aria device in the MPS world/map frame. "
            "tx and ty are horizontal coordinates; tz is height/up. For the X/Y path plot, "
            "we use tx and ty directly. For posture and stairs, tz is the main height signal.",
        ),
        (
            "Orientation: qx/qy/qz/qw_world_device",
            "Raw quaternion describing device orientation in the world frame. We rotate the "
            "device-forward axis from this quaternion, then derive device-left. This is what "
            "lets us make an initial left/right estimate, but confidence drops during turns "
            "or large head rotations.",
        ),
        (
            "Velocity: device_linear_velocity_*_device",
            "Raw linear velocity in the device frame. We convert it to a speed magnitude for "
            "movement intensity. Speed helps check whether a period is walking-like or mostly still.",
        ),
        (
            "Quality/time: tracking_timestamp_us, quality_score",
            "Raw timestamp and tracking quality. Timestamps are converted to session seconds. "
            "Quality score helps us know whether the SLAM pose is reliable enough to interpret.",
        ),
    ]
    processed_cards = [
        (
            "session_s",
            "Processed time axis in seconds. This makes all plots line up with video and manual labels.",
        ),
        (
            "local_heading_deg",
            "Processed direction of travel from the smoothed X/Y path. We do not assume raw X or raw Y is forward.",
        ),
        (
            "lateral_sway_m",
            "Processed side-to-side head movement relative to the local travel direction. Positive means path-left; negative means path-right. Peaks/troughs are used as a step proxy.",
        ),
        (
            "sway_peak_session_s and session_s",
            "The sway peak time is the actual head-sway extremum. The step event time is currently the midpoint between neighboring extrema, so it may not be the lowest height point.",
        ),
        (
            "horizontal_path_progress_m",
            "Processed cumulative X/Y path distance inside the bout. Step proxy length is the change in this value between neighboring detected events.",
        ),
        (
            "estimated_foot_side and foot_side_alignment",
            "Processed left/right estimate. Positive alignment means the sway-side direction agrees with device-left, so we label estimated left; negative means estimated right.",
        ),
    ]
    processing_steps = [
        (
            "Resample raw SLAM pose inside a locomotion bout",
            "We read raw `tx_world_device`, `ty_world_device`, `tz_world_device`, velocity, and quaternion columns from `closed_loop_trajectory.csv`. "
            "For this diagnostic analysis, the bout is resampled to 50 Hz so X/Y/Z and orientation are on a clean common time axis.",
        ),
        (
            "Build a smoothed X/Y path centerline",
            "The raw X/Y head path has small side-to-side oscillations from walking. We smooth X and Y with a centered 0.75 s rolling mean. "
            "This smoothed curve becomes the local path centerline.",
        ),
        (
            "Estimate local travel direction",
            "At each time point, we take the gradient of the smoothed centerline and normalize it. "
            "That vector is the local forward/travel direction. This is why we do not assume raw X or raw Y is forward.",
        ),
        (
            "Calculate path-left and lateral sway",
            "Path-left is 90 degrees left of the local forward vector: `path_left = [-forward_y, forward_x]`. "
            "Then `residual_xy = raw_xy - smoothed_center_xy`, and `lateral_sway_m = dot(residual_xy, path_left)`. "
            "After that, we smooth lateral sway with a short 0.12 s rolling mean.",
        ),
        (
            "Find step-like rhythm from lateral extrema",
            "We find positive peaks and negative troughs in `lateral_sway_m`. The peak prominence threshold is adaptive: "
            "`max(0.006 m, 14% of the 95th-5th percentile lateral range)`. Neighboring same-side peaks are merged when they are too close.",
        ),
        (
            "Place the step proxy event",
            "A lateral peak/trough is not itself treated as the foot contact. The current step proxy time is the midpoint between two neighboring lateral extrema. "
            "So this detected time may not be the lowest Z height point.",
        ),
        (
            "Compute step/stride proxy length",
            "`horizontal_path_progress_m` is cumulative X/Y head path distance. Step proxy length is the distance change from the previous event. "
            "Stride proxy length is the distance change from the event two steps before.",
        ),
        (
            "Estimate left vs right",
            "We rotate the device-forward axis using the raw quaternion to get device-forward in world coordinates, align it with path-forward, then compute device-left. "
            "If the sway-side direction aligns positively with device-left, we label estimated left; if negative, estimated right.",
        ),
    ]
    plot_rows = [
        (
            "01 lateral sway step timing",
            "Check whether side-to-side rhythm has clear alternating peaks/troughs and where the detected event times fall.",
        ),
        (
            "02 left/right alignment sign rule",
            "Shows the sign of foot_side_alignment. Positive bars are estimated left; negative bars are estimated right.",
        ),
        (
            "03 X/Y path direction and axes",
            "Shows raw X/Y path, local travel direction, path-left axis, and device-left axis at each detected event.",
        ),
        (
            "04 heading and confidence context",
            "Shows why turns or changing orientation lower confidence even when step rhythm is visible.",
        ),
        (
            "05 step and stride proxy lengths",
            "Shows processed step/stride distances from SLAM head path. These are proxies, not direct foot-contact distances.",
        ),
    ]

    def cards_html(cards: list[tuple[str, str]], tag: str, tag_class: str) -> str:
        return "".join(
            f"""
            <div class="param-card">
              <h3>{title}</h3>
              <p>{text}</p>
              <div class="tag-row"><span class="pill {tag_class}">{tag}</span></div>
            </div>
            """
            for title, text in cards
        )

    plot_table = "".join(
        f"<tr><td>{name}</td><td>{reason}</td></tr>"
        for name, reason in plot_rows
    )
    if compact:
        return f"""
          <section class="card">
            <h2>Valuable Parameters To Look At From SLAM</h2>
            <p>{intro}</p>
            <div class="tag-row">
              <span class="pill raw">raw SLAM columns</span>
              <span class="pill processed">processed analysis features</span>
              <span class="pill warning">validate with downward video</span>
            </div>
            <div class="steps">
              <div class="step"><b>How `lateral_sway_m` is made</b><p>Raw X/Y is smoothed into a local centerline; local forward comes from that centerline; path-left is 90 degrees left of forward; `lateral_sway_m = dot(raw_xy - centerline_xy, path_left)`.</p></div>
              <div class="step"><b>How step events are made</b><p>We detect positive/negative extrema in lateral sway and place the step proxy at the midpoint between neighboring extrema, not necessarily at the lowest Z point.</p></div>
              <div class="step"><b>How left/right is made</b><p>Raw orientation quaternion gives device-left. Positive alignment between sway-side direction and device-left means estimated left; negative means estimated right.</p></div>
            </div>
            <table>
              <tr><th>Plot</th><th>What it helps explain</th></tr>
              {plot_table}
            </table>
          </section>
        """

    return f"""
      <section class="card">
        <h2>Valuable Parameters To Look At From SLAM</h2>
        <p>{intro}</p>
        <div class="tag-row">
          <span class="pill raw">raw from MPS closed_loop_trajectory.csv</span>
          <span class="pill processed">processed by our stride pipeline</span>
          <span class="pill warning">head/device proxy, not direct foot contact</span>
        </div>
        <h2>Important Raw SLAM Columns</h2>
        <div class="parameter-grid">
          {cards_html(raw_cards, "raw SLAM", "raw")}
        </div>
        <h2 style="margin-top:18px;">Processed Parameters Used In This Page</h2>
        <div class="parameter-grid">
          {cards_html(processed_cards, "processed", "processed")}
        </div>
        <h2 style="margin-top:18px;">How Raw SLAM Becomes Lateral Sway, Steps, And Left/Right</h2>
        <div class="steps">
          {''.join(f'<div class="step"><b>{title}</b><p>{text}</p></div>' for title, text in processing_steps)}
        </div>
        <h2 style="margin-top:18px;">Which Plot To Use For Each Question</h2>
        <table>
          <tr><th>Plot</th><th>What it helps explain</th></tr>
          {plot_table}
        </table>
      </section>
    """


def rel_href(from_file: Path, target: Path) -> str:
    return target.relative_to(from_file.parent).as_posix() if target.parent == from_file.parent else Path(
        *target.relative_to(from_file.parent).parts
    ).as_posix()


def relative_link(from_file: Path, target: Path) -> str:
    return target.relative_to(from_file.parent).as_posix()


def write_bout_gallery(
    *,
    out_dir: Path,
    trial_id: str,
    bout: pd.Series,
    paths: list[Path],
) -> Path:
    gallery = out_dir / "gallery.html"
    cards = []
    for path in paths:
        cards.append(
            f"""
            <section class="card">
              <h2>{path.stem.replace('_', ' ')}</h2>
              <a href="{path.name}" target="_blank"><img src="{path.name}" alt="{path.stem}"></a>
            </section>
            """
        )
    title = f"{trial_id} bout {int(bout['bout_id'])}: {bout['label']}"
    body = f"""
      <section class="hero">
        <h1>{title}</h1>
        <p>
          Window <b>{float(bout['bout_start_s']):.1f}s-{float(bout['bout_end_s']):.1f}s</b>.
          Confidence: <b>{bout['confidence']}</b>.
          Detected step events: <b>{int(bout['detected_step_events'])}</b>.
          Heading change: <b>{float(bout['heading_change_deg']):.1f} deg</b>.
        </p>
        <p><a href="../index.html">Back to {trial_id}</a> · <a href="../../index.html">Back to all trials</a></p>
      </section>
      {slam_parameter_panel(compact=True)}
      {''.join(cards)}
    """
    gallery.write_text(html_page(title, body), encoding="utf-8")
    return gallery


def write_trial_index(trial_dir: Path, trial_id: str, rows: list[dict[str, object]]) -> Path:
    index_path = trial_dir / "index.html"
    bout_options = []
    table_rows = []
    for row in rows:
        bout_options.append(
            f"""
            <option value="{row['gallery']}">
              bout {row['bout_id']:03d} | {row['label']} | {row['start_s']:.1f}-{row['end_s']:.1f}s | {row['confidence']}
            </option>
            """
        )
        table_rows.append(
            f"""
            <tr>
              <td><a href="{row['gallery']}">bout {row['bout_id']:03d}</a></td>
              <td>{row['label']}</td>
              <td>{row['start_s']:.1f}-{row['end_s']:.1f}s</td>
              <td>{row['confidence']}</td>
              <td>{row['events']}</td>
              <td>{row['heading_change_deg']:.1f} deg</td>
              <td>{row['warning']}</td>
            </tr>
            """
        )
    body = f"""
      <section class="hero">
        <h1>{trial_id}: All Step / Left-Right Logic Plots</h1>
        <p>Each bout folder contains all five explanation plots: lateral sway timing, left/right alignment, X/Y direction axes, turning/heading confidence, and step/stride proxy length.</p>
        <p><a href="../index.html">Back to all trials</a></p>
      </section>
      {slam_parameter_panel(compact=True)}
      <section class="card">
        <h2>Choose A Bout In {trial_id}</h2>
        <p class="muted">Use this dropdown when the table is too long. Bouts stay grouped inside the current trial.</p>
        <div class="control-panel">
          <div>
            <label for="trialBoutSelect">Bout</label>
            <select id="trialBoutSelect">
              {''.join(bout_options)}
            </select>
          </div>
          <div>
            <button id="trialBoutOpenButton" type="button">Open Selected Bout</button>
          </div>
        </div>
      </section>
      <section class="card">
        <h2>All Bouts In {trial_id}</h2>
        <table>
          <tr><th>Bout</th><th>Label</th><th>Time</th><th>Confidence</th><th>Events</th><th>Heading</th><th>Warning</th></tr>
          {''.join(table_rows)}
        </table>
      </section>
      <script>
        const trialBoutSelect = document.getElementById("trialBoutSelect");
        const trialBoutOpenButton = document.getElementById("trialBoutOpenButton");
        trialBoutOpenButton.addEventListener("click", () => {{
          if (trialBoutSelect.value) window.location.href = trialBoutSelect.value;
        }});
      </script>
    """
    index_path.write_text(html_page(f"{trial_id} step logic plots", body), encoding="utf-8")
    return index_path


def write_all_trials_index(all_dir: Path, rows: list[dict[str, object]]) -> Path:
    index_path = all_dir / "index.html"
    trial_ids = sorted({str(row["trial_id"]) for row in rows})
    selector_rows = [
        {
            "trial_id": str(row["trial_id"]),
            "bout_id": int(row["bout_id"]),
            "label": str(row["label"]),
            "start_s": float(row["start_s"]),
            "end_s": float(row["end_s"]),
            "confidence": str(row["confidence"]),
            "events": int(row["events"]),
            "heading_change_deg": float(row["heading_change_deg"]),
            "gallery": str(row["gallery"]),
        }
        for row in rows
    ]
    trial_cards = []
    for trial_id in trial_ids:
        count = sum(1 for row in rows if row["trial_id"] == trial_id)
        trial_cards.append(
            f"""
            <section class="card">
              <h2>{trial_id}</h2>
              <p>{count} locomotion bouts with full five-plot explanation sets.</p>
              <p><a href="{trial_id}/index.html">Open {trial_id} plots</a></p>
            </section>
            """
        )
    table_rows = []
    for row in rows:
        table_rows.append(
            f"""
            <tr>
              <td>{row['trial_id']}</td>
              <td><a href="{row['gallery']}">bout {row['bout_id']:03d}</a></td>
              <td>{row['label']}</td>
              <td>{row['start_s']:.1f}-{row['end_s']:.1f}s</td>
              <td>{row['confidence']}</td>
              <td>{row['events']}</td>
              <td>{row['heading_change_deg']:.1f} deg</td>
            </tr>
            """
        )
    body = f"""
      <section class="hero">
        <h1>All Trials: Step Detection And Left/Right Logic Plots</h1>
        <p>
          This folder expands the curated example gallery into a complete plot set for every trial
          and every detected locomotion bout currently available in the stride diagnostics data.
          Generated trials: <b>{', '.join(trial_ids)}</b>.
        </p>
        <p>
          The five plots in each bout show how step timing, path-side sway, estimated left/right,
          direction-of-travel, turning confidence, and step/stride proxy length were decided.
        </p>
      </section>
      {slam_parameter_panel()}
      <section class="card">
        <h2>Choose Trial And Bout</h2>
        <p class="muted">
          Bouts are grouped by trial here, so you can jump directly to one review window without scrolling through the whole table.
        </p>
        <div class="control-panel">
          <div>
            <label for="trialSelect">Trial</label>
            <select id="trialSelect"></select>
          </div>
          <div>
            <label for="boutSelect">Bout In Selected Trial</label>
            <select id="boutSelect"></select>
          </div>
          <div>
            <button id="openBoutButton" type="button">Open Selected Bout</button>
          </div>
        </div>
      </section>
      <div class="grid">{''.join(trial_cards)}</div>
      <section class="card">
        <h2>All Bout Galleries</h2>
        <table>
          <tr><th>Trial</th><th>Bout</th><th>Label</th><th>Time</th><th>Confidence</th><th>Events</th><th>Heading</th></tr>
          {''.join(table_rows)}
        </table>
      </section>
      <script>
        const boutRows = {json.dumps(selector_rows)};
        const trialSelect = document.getElementById("trialSelect");
        const boutSelect = document.getElementById("boutSelect");
        const openBoutButton = document.getElementById("openBoutButton");

        function uniqueTrials() {{
          return Array.from(new Set(boutRows.map(row => row.trial_id))).sort();
        }}

        function formatBout(row) {{
          const start = Number(row.start_s).toFixed(1);
          const end = Number(row.end_s).toFixed(1);
          const heading = Number(row.heading_change_deg).toFixed(1);
          return `bout ${{String(row.bout_id).padStart(3, "0")}} | ${{row.label}} | ${{start}}-${{end}}s | ${{row.confidence}} | ${{row.events}} events | heading ${{heading}}°`;
        }}

        function fillTrials() {{
          trialSelect.innerHTML = "";
          uniqueTrials().forEach(trialId => {{
            const option = document.createElement("option");
            option.value = trialId;
            option.textContent = `${{trialId}} (${{boutRows.filter(row => row.trial_id === trialId).length}} bouts)`;
            trialSelect.appendChild(option);
          }});
        }}

        function fillBouts() {{
          const trialId = trialSelect.value;
          boutSelect.innerHTML = "";
          boutRows
            .filter(row => row.trial_id === trialId)
            .sort((a, b) => a.bout_id - b.bout_id)
            .forEach(row => {{
              const option = document.createElement("option");
              option.value = row.gallery;
              option.textContent = formatBout(row);
              boutSelect.appendChild(option);
            }});
        }}

        trialSelect.addEventListener("change", fillBouts);
        openBoutButton.addEventListener("click", () => {{
          if (boutSelect.value) window.location.href = boutSelect.value;
        }});
        fillTrials();
        fillBouts();
      </script>
    """
    index_path.write_text(html_page("All trials step logic plots", body), encoding="utf-8")
    return index_path


def generate_all_trial_plots(events: pd.DataFrame, timeseries: pd.DataFrame, bouts: pd.DataFrame) -> tuple[list[Path], Path]:
    ALL_TRIALS_DIR.mkdir(parents=True, exist_ok=True)
    all_paths: list[Path] = []
    index_rows: list[dict[str, object]] = []

    for trial_id in sorted(bouts["trial_id"].unique()):
        trial_dir = ALL_TRIALS_DIR / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_rows: list[dict[str, object]] = []
        trial_bouts = bouts[bouts["trial_id"] == trial_id].sort_values("bout_id")
        for _, bout in trial_bouts.iterrows():
            bout_id = int(bout["bout_id"])
            ts = bout_frame(timeseries, trial_id, bout_id)
            if ts.empty:
                continue
            bout_dir = trial_dir / f"bout_{bout_id:03d}_{slug(bout['label'])}"
            paths = [
                plot_lateral_sway_step_timing(
                    events,
                    timeseries,
                    bouts,
                    trial_id,
                    bout_id,
                    bout_dir,
                    "01_lateral_sway_step_timing.png",
                ),
                plot_left_right_alignment(
                    events,
                    trial_id,
                    bout_id,
                    "02_left_right_alignment_sign_rule.png",
                    "for this bout",
                    bout_dir,
                ),
                plot_xy_path_axes(
                    events,
                    timeseries,
                    bouts,
                    trial_id,
                    bout_id,
                    bout_dir,
                    "03_xy_path_direction_and_left_right_axes.png",
                ),
                plot_turning_uncertainty(
                    events,
                    timeseries,
                    bouts,
                    trial_id,
                    bout_id,
                    bout_dir,
                    "04_heading_and_confidence_context.png",
                ),
                plot_step_stride_lengths(
                    events,
                    trial_id,
                    bout_id,
                    bout_dir,
                    "05_step_and_stride_proxy_lengths.png",
                ),
            ]
            gallery = write_bout_gallery(out_dir=bout_dir, trial_id=trial_id, bout=bout, paths=paths)
            all_paths.extend(paths)
            row = {
                "trial_id": trial_id,
                "bout_id": bout_id,
                "label": str(bout["label"]),
                "start_s": float(bout["bout_start_s"]),
                "end_s": float(bout["bout_end_s"]),
                "confidence": str(bout["confidence"]),
                "events": int(bout["detected_step_events"]),
                "heading_change_deg": float(bout["heading_change_deg"]),
                "warning": "" if pd.isna(bout.get("warning")) else str(bout.get("warning")),
                "gallery": gallery.relative_to(ALL_TRIALS_DIR).as_posix(),
            }
            index_rows.append(row)
            trial_row = dict(row)
            trial_row["gallery"] = gallery.relative_to(trial_dir).as_posix()
            trial_rows.append(trial_row)
        write_trial_index(trial_dir, trial_id, trial_rows)

    all_index = write_all_trials_index(ALL_TRIALS_DIR, index_rows)
    return all_paths, all_index


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events, timeseries, bouts = load_data()

    paths = [
        plot_lateral_sway_step_timing(events, timeseries, bouts),
        plot_left_right_alignment(
            events,
            *EXAMPLE_CLEAN,
            out_name="02_clean_bout_left_right_alignment_sign_rule.png",
            title_suffix="in a high-confidence bout",
        ),
        plot_xy_path_axes(events, timeseries, bouts),
        plot_turning_uncertainty(events, timeseries, bouts),
        plot_step_stride_lengths(events),
    ]
    gallery = write_gallery(paths)
    readme = write_readme(paths, gallery)
    all_paths, all_index = generate_all_trial_plots(events, timeseries, bouts)
    print(f"Wrote {len(paths)} plots")
    print(f"Gallery: {gallery}")
    print(f"README: {readme}")
    print(f"Wrote {len(all_paths)} all-trials plots")
    print(f"All-trials index: {all_index}")


if __name__ == "__main__":
    main()
