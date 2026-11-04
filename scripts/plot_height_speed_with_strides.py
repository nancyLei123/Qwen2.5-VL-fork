#!/usr/bin/env python3
"""Overlay stride-proxy annotations on raw SLAM height/speed plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

QWEN_REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = QWEN_REPO / "runs"
STRIDE_DATA_DIR = RUNS_DIR / "stride_step_diagnostics" / "data"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_slam_parameters import (  # noqa: E402
    add_action_color_legend,
    load_effective_manual_intervals,
    make_session_trajectory,
    style_state_spans,
)


SIDE_COLORS = {
    "left": "#2f78b7",
    "right": "#c94c4c",
    "unknown": "#64748b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", default="Ntrial3")
    parser.add_argument("--bout-id", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        default=str(RUNS_DIR / "stride_step_diagnostics" / "height_speed_stride_overlay"),
    )
    return parser.parse_args()


def add_rolling_speed(traj: pd.DataFrame) -> pd.DataFrame:
    work = traj[["session_s", "tz_world_device", "linear_speed_mps"]].dropna().copy()
    if work.empty:
        return work
    dt = float(work["session_s"].diff().median()) if len(work) > 1 else 0.001
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.001
    rolling_0_5_count = max(1, int(round(0.5 / dt)))
    work["rolling_mean_speed_0_5s"] = (
        work["linear_speed_mps"].rolling(rolling_0_5_count, center=True, min_periods=1).mean()
    )
    return work


def load_stride_events(trial_id: str) -> pd.DataFrame:
    events_path = STRIDE_DATA_DIR / "step_stride_event_estimates.csv"
    if not events_path.exists():
        return pd.DataFrame()
    events = pd.read_csv(events_path)
    if events.empty or "trial_id" not in events.columns:
        return pd.DataFrame()
    events = events[events["trial_id"].astype(str).eq(trial_id)].copy()
    if events.empty:
        return events
    events = events.sort_values(["session_s", "bout_id", "event_index"]).reset_index(drop=True)
    return events


def load_bouts(trial_id: str) -> pd.DataFrame:
    bouts_path = STRIDE_DATA_DIR / "step_stride_bout_summary.csv"
    if not bouts_path.exists():
        return pd.DataFrame()
    bouts = pd.read_csv(bouts_path)
    if bouts.empty or "trial_id" not in bouts.columns:
        return pd.DataFrame()
    return bouts[bouts["trial_id"].astype(str).eq(trial_id)].copy().sort_values(["bout_start_s", "bout_id"])


def side_color(row: pd.Series) -> str:
    side = str(row.get("estimated_foot_side", "")).strip().lower()
    return SIDE_COLORS.get(side, SIDE_COLORS["unknown"])


def format_stride(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(number):
        return "N/A"
    return f"{number:.2f} m"


def setup_height_speed_axes(
    ax_height: Any,
    traj: pd.DataFrame,
    states: pd.DataFrame,
    *,
    title: str,
    xlim: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    used_labels = style_state_spans(ax_height, states)
    add_action_color_legend(ax_height, used_labels)

    if xlim is not None:
        plot_traj = traj[(traj["session_s"] >= xlim[0]) & (traj["session_s"] <= xlim[1])].copy()
    else:
        plot_traj = traj.copy()
    stride = max(1, len(plot_traj) // 18000)
    height_sample = plot_traj.iloc[::stride]

    height_line = ax_height.plot(
        height_sample["session_s"],
        height_sample["tz_world_device"],
        color="#1f2d3a",
        linewidth=1.05,
        alpha=0.92,
        label="raw SLAM height, tz_world_device",
    )
    ax_height.set_ylabel("raw SLAM height, tz_world_device (m)", color="#1f2d3a")
    ax_height.tick_params(axis="y", labelcolor="#1f2d3a")
    ax_height.grid(axis="y", color="#d7dde5", linewidth=0.8, alpha=0.65)

    ax_speed = ax_height.twinx()
    speed_line = ax_speed.plot(
        plot_traj["session_s"],
        plot_traj["rolling_mean_speed_0_5s"],
        color="#d36b42",
        linewidth=1.0,
        alpha=0.88,
        label="0.5s rolling mean speed",
    )
    ax_speed.set_ylabel("linear speed magnitude (m/s)", color="#a84f24")
    ax_speed.tick_params(axis="y", labelcolor="#a84f24")

    lines = height_line + speed_line
    labels = [line.get_label() for line in lines]
    ax_height.legend(lines, labels, loc="upper right", title="Plotted values", fontsize=8, title_fontsize=8)
    ax_height.set_title(title)
    if xlim is not None:
        ax_height.set_xlim(*xlim)
    return ax_height, ax_speed


def plot_full_session_overview(trial_id: str, traj: pd.DataFrame, states: pd.DataFrame, events: pd.DataFrame, out_dir: Path) -> Path:
    stride_events = events[pd.to_numeric(events.get("valid_stride_path_length_m"), errors="coerce").notna()].copy()
    fig, (ax_height, ax_stride) = plt.subplots(
        2,
        1,
        figsize=(17, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    setup_height_speed_axes(
        ax_height,
        traj,
        states,
        title=f"{trial_id}: height/speed over time with stride-proxy markers",
    )

    if not events.empty:
        y_min, y_max = ax_height.get_ylim()
        rug_y = y_min + 0.04 * (y_max - y_min)
        colors = [side_color(row) for _, row in events.iterrows()]
        ax_height.scatter(
            events["session_s"],
            np.full(len(events), rug_y),
            c=colors,
            marker="|",
            s=38,
            linewidths=1.2,
            alpha=0.62,
            label="detected step timing",
            zorder=5,
        )
        ax_height.text(
            0.01,
            0.03,
            "Small blue/red ticks = estimated step times from lateral head-sway; purple points below = stride proxy at the later step.",
            transform=ax_height.transAxes,
            fontsize=8,
            color="#334155",
            bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.88},
        )

    used_labels = style_state_spans(ax_stride, states)
    stride_values = pd.to_numeric(stride_events.get("valid_stride_path_length_m"), errors="coerce")
    if not stride_events.empty:
        ax_stride.plot(
            stride_events["session_s"],
            stride_values,
            color="#7a4fb3",
            linewidth=1.0,
            alpha=0.65,
            label="valid stride proxy length",
        )
        ax_stride.scatter(
            stride_events["session_s"],
            stride_values,
            c="#7a4fb3",
            s=16,
            alpha=0.82,
            zorder=4,
        )
        label_step = max(1, len(stride_events) // 28)
        for index, (_, row) in enumerate(stride_events.iterrows()):
            if index % label_step != 0:
                continue
            value = row.get("valid_stride_path_length_m")
            ax_stride.annotate(
                format_stride(value),
                (float(row["session_s"]), float(value)),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#4c1d95",
                rotation=35,
            )
    add_action_color_legend(ax_stride, used_labels)
    ax_stride.set_ylabel("stride proxy length (m)")
    ax_stride.set_xlabel("session time (s)")
    ax_stride.grid(axis="y", color="#d7dde5", linewidth=0.8, alpha=0.65)
    ax_stride.legend(loc="upper right", fontsize=8)

    fig.text(
        0.01,
        0.01,
        "Stride proxy = horizontal SLAM path distance from the current estimated step event to the event two steps earlier. It is not a direct foot-contact measurement.",
        fontsize=8,
        color="#3f4752",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out_path = out_dir / f"{trial_id}_height_speed_with_stride_overview.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_zoomed_bout(
    trial_id: str,
    bout_id: int,
    traj: pd.DataFrame,
    states: pd.DataFrame,
    events: pd.DataFrame,
    bouts: pd.DataFrame,
    out_dir: Path,
) -> Path | None:
    if bouts.empty:
        return None
    bout_hit = bouts[pd.to_numeric(bouts["bout_id"], errors="coerce").eq(bout_id)]
    if bout_hit.empty:
        return None
    bout = bout_hit.iloc[0]
    bout_events = events[pd.to_numeric(events.get("bout_id"), errors="coerce").eq(bout_id)].copy()
    start_s = float(bout["bout_start_s"])
    end_s = float(bout["bout_end_s"])
    xlim = (max(0.0, start_s - 0.8), end_s + 0.8)

    fig, ax_height = plt.subplots(figsize=(15, 6.2))
    setup_height_speed_axes(
        ax_height,
        traj,
        states,
        title=f"{trial_id} review window {bout_id}: height/speed with individual stride-proxy notes",
        xlim=xlim,
    )
    y_min, y_max = ax_height.get_ylim()
    y_range = max(1e-9, y_max - y_min)

    if not bout_events.empty:
        sorted_events = bout_events.sort_values("session_s").reset_index(drop=True)
        for _, row in sorted_events.iterrows():
            event_time = float(row["session_s"])
            color = side_color(row)
            ax_height.axvline(event_time, color=color, linewidth=0.9, alpha=0.42, zorder=1)
            ax_height.text(
                event_time,
                y_min + 0.02 * y_range,
                str(row.get("estimated_foot_side", ""))[:1].upper() or "?",
                ha="center",
                va="bottom",
                fontsize=7,
                color=color,
                fontweight="bold",
            )

        for index, row in sorted_events.iterrows():
            stride_value = row.get("valid_stride_path_length_m")
            if pd.isna(stride_value) or index < 2:
                continue
            start_time = float(sorted_events.iloc[index - 2]["session_s"])
            end_time = float(row["session_s"])
            level = index % 4
            y = y_max - (0.08 + level * 0.055) * y_range
            ax_height.plot([start_time, end_time], [y, y], color="#7a4fb3", linewidth=1.25, alpha=0.90)
            ax_height.plot([start_time, start_time], [y - 0.012 * y_range, y + 0.012 * y_range], color="#7a4fb3", linewidth=1.0)
            ax_height.plot([end_time, end_time], [y - 0.012 * y_range, y + 0.012 * y_range], color="#7a4fb3", linewidth=1.0)
            ax_height.text(
                (start_time + end_time) / 2.0,
                y + 0.014 * y_range,
                f"stride {format_stride(stride_value)}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#4c1d95",
                bbox={"facecolor": "white", "edgecolor": "#d8b4fe", "alpha": 0.88, "pad": 1.8},
            )

    ax_height.set_xlabel("session time (s)")
    fig.text(
        0.01,
        0.01,
        "Blue/red vertical lines = estimated step timings. Purple brackets span every-other step event, which is the current stride-proxy definition.",
        fontsize=8,
        color="#3f4752",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out_path = out_dir / f"{trial_id}_bout{bout_id:02d}_height_speed_with_stride_notes.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj, _meta = make_session_trajectory(args.trial)
    traj = add_rolling_speed(traj)
    states = load_effective_manual_intervals(args.trial)
    events = load_stride_events(args.trial)
    bouts = load_bouts(args.trial)

    outputs: list[Path] = []
    outputs.append(plot_full_session_overview(args.trial, traj, states, events, out_dir))
    zoomed = plot_zoomed_bout(args.trial, args.bout_id, traj, states, events, bouts, out_dir)
    if zoomed is not None:
        outputs.append(zoomed)

    print("Wrote height/speed stride overlay plot(s):")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
