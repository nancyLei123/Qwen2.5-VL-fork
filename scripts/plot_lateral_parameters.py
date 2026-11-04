#!/usr/bin/env python3
"""Plot lateral parameters used for SLAM-only step/left-right exploration.

These plots explain how the current step proxy and left/right estimate are
derived from side-to-side head motion, local travel direction, and device
orientation. They are intentionally diagnostic, not final clinical measures.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "runs" / "stride_step_diagnostics" / "data"
OUT_DIR = REPO / "runs" / "stride_step_diagnostics" / "lateral_parameter_plots"

LABEL_COLORS = {
    "walking": "#3a9d55",
    "upstairs": "#e58f2a",
    "downstairs": "#c94c4c",
}
SIDE_COLORS = {
    "left": "#2f78b7",
    "right": "#c94c4c",
    "uncertain": "#7a8793",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    series = pd.read_csv(DATA_DIR / "step_stride_resampled_bout_timeseries.csv")
    events = pd.read_csv(DATA_DIR / "step_stride_event_estimates.csv")
    bouts = pd.read_csv(DATA_DIR / "step_stride_bout_summary.csv")
    return series, events, bouts


def safe_name(value: str) -> str:
    return (
        str(value)
        .replace(" ", "_")
        .replace("/", "_")
        .replace(".", "p")
        .replace(":", "")
    )


def make_output_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_text_note(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=8, color="#5f6b64")


def plot_lateral_range_by_label(bouts: pd.DataFrame) -> Path:
    usable = bouts.copy()
    labels = [label for label in ["walking", "upstairs", "downstairs"] if label in set(usable["label"])]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    data = [usable.loc[usable["label"] == label, "lateral_range_m"].dropna().to_numpy() for label in labels]
    box = ax.boxplot(data, labels=[label.title() for label in labels], patch_artist=True)
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(LABEL_COLORS.get(label, "#94a3b8"))
        patch.set_alpha(0.38)
    for i, label in enumerate(labels, start=1):
        y = usable.loc[usable["label"] == label, "lateral_range_m"].dropna().to_numpy()
        x = np.full_like(y, i, dtype=float) + np.linspace(-0.08, 0.08, len(y)) if len(y) else []
        ax.scatter(x, y, s=32, color=LABEL_COLORS.get(label, "#94a3b8"), edgecolor="white", linewidth=0.6, alpha=0.85)
    ax.axhline(0.018, color="#6d3aa0", linestyle="--", linewidth=1.4, label="current weak-sway warning: 0.018 m")
    ax.set_title("Lateral head-sway range by locomotion label")
    ax.set_ylabel("lateral_range_m within local bout")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", frameon=True)
    add_text_note(fig, "Lateral range is the 95th-5th percentile spread of lateral_sway_m. Small values can mean small/shuffling steps or weak head-sway signal.")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = OUT_DIR / "01_combined_lateral_range_by_label.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_alignment_summary(events: pd.DataFrame) -> Path:
    usable = events.dropna(subset=["foot_side_alignment", "device_path_forward_alignment"]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=False)
    for ax, column, title, ylabel in [
        (
            axes[0],
            "foot_side_alignment",
            "Foot-side alignment",
            "alignment with device-left axis",
        ),
        (
            axes[1],
            "device_path_forward_alignment",
            "Device-forward vs path-forward alignment",
            "absolute alignment",
        ),
    ]:
        for label, part in usable.groupby("label"):
            color = LABEL_COLORS.get(label, "#94a3b8")
            ax.scatter(
                part["session_s"],
                part[column],
                s=22,
                alpha=0.70,
                color=color,
                label=label.title(),
                edgecolor="white",
                linewidth=0.3,
            )
        if column == "foot_side_alignment":
            ax.axhline(0.75, color="#167a55", linestyle="--", linewidth=1.0, alpha=0.8, label="strong +/-0.75")
            ax.axhline(-0.75, color="#167a55", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.axhline(0.45, color="#c9632e", linestyle=":", linewidth=1.2, alpha=0.9, label="moderate +/-0.45")
            ax.axhline(-0.45, color="#c9632e", linestyle=":", linewidth=1.2, alpha=0.9)
            ax.axhline(0, color="#334155", linewidth=0.8, alpha=0.4)
        else:
            ax.axhline(0.45, color="#c9632e", linestyle=":", linewidth=1.2, alpha=0.9, label="warning <0.45")
            ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.set_xlabel("session time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=5, frameon=True)
    add_text_note(
        fig,
        "foot_side_alignment maps local sway side to estimated device-left/right. device_path_forward_alignment warns when head/device forward differs from travel direction.",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.90])
    out = OUT_DIR / "02_combined_alignment_parameters_over_time.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_lateral_vs_step_length(events: pd.DataFrame) -> Path:
    usable = events.dropna(subset=["prominence_m", "valid_step_path_length_m"]).copy()
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for (label, side), part in usable.groupby(["label", "estimated_foot_side"]):
        color = SIDE_COLORS.get(str(side), "#7a8793")
        marker = {"walking": "o", "upstairs": "^", "downstairs": "s"}.get(str(label), "o")
        ax.scatter(
            part["prominence_m"],
            part["valid_step_path_length_m"],
            s=46,
            alpha=0.75,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.5,
            label=f"{label} / {side}",
        )
    ax.axvline(0.006, color="#6d3aa0", linestyle="--", linewidth=1.2, label="min prominence 0.006 m")
    ax.axhline(0.12, color="#c9632e", linestyle=":", linewidth=1.2, label="small step warning 0.12 m")
    ax.axhline(0.07, color="#b91c1c", linestyle=":", linewidth=1.2, label="low-confidence small step 0.07 m")
    ax.set_title("Lateral-sway prominence vs step-proxy length")
    ax.set_xlabel("lateral event prominence_m")
    ax.set_ylabel("valid_step_path_length_m")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=8)
    add_text_note(fig, "This plot explains why small/shuffling steps can be missed: weak lateral prominence and very short horizontal step proxy reduce confidence.")
    fig.tight_layout(rect=[0, 0.05, 0.78, 1])
    out = OUT_DIR / "03_lateral_prominence_vs_step_proxy_length.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_trial_lateral_timeline(trial_id: str, series: pd.DataFrame, events: pd.DataFrame) -> Path:
    trial_series = series[series["trial_id"] == trial_id].copy()
    trial_events = events[events["trial_id"] == trial_id].copy()
    labels = sorted(trial_series["label"].dropna().unique())
    if trial_series.empty:
        raise ValueError(f"No series rows for {trial_id}")

    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)

    for label, part in trial_series.groupby("label"):
        color = LABEL_COLORS.get(label, "#94a3b8")
        axes[0].plot(part["session_s"], part["lateral_sway_m"], color=color, linewidth=1.0, alpha=0.85, label=label.title())
    axes[0].axhline(0, color="#334155", linewidth=0.8, alpha=0.45)
    for side, part in trial_events.groupby("estimated_foot_side"):
        color = SIDE_COLORS.get(str(side), "#7a8793")
        axes[0].scatter(part["session_s"], part["lateral_sway_m"], s=32, color=color, edgecolor="white", linewidth=0.6, zorder=4, label=f"event {side}")
    axes[0].set_ylabel("lateral_sway_m")
    axes[0].set_title(f"{trial_id}: lateral parameters over time")
    axes[0].legend(loc="upper left", ncol=4, frameon=True, fontsize=8)

    for label, part in trial_series.groupby("label"):
        axes[1].plot(part["session_s"], part["local_heading_deg"], color=LABEL_COLORS.get(label, "#94a3b8"), linewidth=1.0, alpha=0.9)
    axes[1].set_ylabel("local_heading_deg")
    axes[1].grid(alpha=0.2)

    axes[2].plot(
        trial_series["session_s"],
        trial_series["device_path_forward_alignment"],
        color="#1f2d3a",
        linewidth=1.0,
        label="device_path_forward_alignment",
    )
    axes[2].axhline(0.45, color="#c9632e", linestyle=":", linewidth=1.2, label="warning <0.45")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("alignment")
    axes[2].legend(loc="upper left", frameon=True, fontsize=8)
    axes[2].grid(alpha=0.2)

    for side, part in trial_events.groupby("estimated_foot_side"):
        color = SIDE_COLORS.get(str(side), "#7a8793")
        axes[3].scatter(part["session_s"], part["foot_side_alignment"], s=42, color=color, edgecolor="white", linewidth=0.6, label=f"{side}")
    axes[3].axhline(0, color="#334155", linewidth=0.8, alpha=0.45)
    axes[3].axhline(0.75, color="#167a55", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[3].axhline(-0.75, color="#167a55", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[3].axhline(0.45, color="#c9632e", linestyle=":", linewidth=1.2, alpha=0.9)
    axes[3].axhline(-0.45, color="#c9632e", linestyle=":", linewidth=1.2, alpha=0.9)
    axes[3].set_ylabel("foot_side_alignment")
    axes[3].set_xlabel("session time (s)")
    axes[3].legend(loc="upper left", frameon=True, fontsize=8)
    axes[3].grid(alpha=0.2)

    for ax in axes:
        for _, row in trial_events.iterrows():
            conf = str(row.get("event_confidence", ""))
            alpha = 0.16 if conf == "high" else 0.10 if conf == "medium" else 0.06
            color = SIDE_COLORS.get(str(row.get("estimated_foot_side")), "#7a8793")
            ax.axvline(float(row["session_s"]), color=color, alpha=alpha, linewidth=1.0)

    add_text_note(
        fig,
        "Panel 1: lateral sway and detected left/right events. Panel 2: local travel heading. Panel 3: head/device forward alignment with path. Panel 4: event-level side alignment.",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = OUT_DIR / f"04_{safe_name(trial_id)}_lateral_parameter_timeline.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_readme(paths: list[Path], trials: list[str]) -> None:
    lines = [
        "# Lateral Parameter Plots",
        "",
        "These plots explain the lateral parameters used for current SLAM-only step and left/right exploration.",
        "",
        "Important caveat: these are head/device-motion proxies, not confirmed foot-contact or clinical gait measures.",
        "",
        "## What Each Parameter Means",
        "",
        "| Parameter | Meaning | Why It Matters |",
        "| --- | --- | --- |",
        "| `lateral_sway_m` | Side-to-side head/device residual relative to local path | Main signal for step-like rhythm |",
        "| `lateral_range_m` | 95th-5th percentile spread of lateral sway in a bout | Weak range can mean small/shuffling steps or low confidence |",
        "| `prominence_m` | Strength of a detected lateral-sway event | Small prominence can lead to missed/low-confidence steps |",
        "| `local_heading_deg` | Direction of local X/Y travel path | Shows turning and direction changes |",
        "| `device_path_forward_alignment` | Whether head/device forward direction matches travel direction | Low value means left/right mapping is less reliable |",
        "| `foot_side_alignment` | Event-level alignment between sway side and device-left axis | Sign/magnitude helps estimate left vs right side |",
        "| `valid_step_path_length_m` | Horizontal path distance between neighboring step proxies | Step-length proxy, not true foot placement |",
        "",
        "## Current Threshold References",
        "",
        "- Weak lateral-sway warning: about `lateral_range_m < 0.018 m`.",
        "- Minimum peak prominence: `max(0.006 m, 0.14 * lateral_range_m)`.",
        "- Strong foot-side alignment: `abs(foot_side_alignment) >= 0.75`.",
        "- Moderate foot-side alignment: `0.45 <= abs(foot_side_alignment) < 0.75`.",
        "- Low device/path alignment warning: `device_path_forward_alignment < 0.45`.",
        "- Small step-proxy warning: `valid_step_path_length_m < 0.12 m`.",
        "- Very small low-confidence step-proxy: `valid_step_path_length_m < 0.07 m`.",
        "",
        "## Generated Plots",
        "",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "",
            "## Included Trials",
            "",
            ", ".join(trials) if trials else "No trials found.",
            "",
            "## Simple Meeting Explanation",
            "",
            "We estimate steps from side-to-side head motion, not from direct foot contact. First, X/Y SLAM motion is converted into local forward and lateral directions. Then lateral sway peaks/troughs are detected and used as the rhythm signal. Left/right is estimated by comparing the local sway side with the Aria device-left direction from orientation. If the head/device direction differs from the walking path, or if the lateral sway is weak, confidence decreases.",
            "",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    make_output_dir()
    series, events, bouts = load_data()
    paths: list[Path] = []
    paths.append(plot_lateral_range_by_label(bouts))
    paths.append(plot_alignment_summary(events))
    paths.append(plot_lateral_vs_step_length(events))
    trials = sorted(series["trial_id"].dropna().unique())
    for trial_id in trials:
        paths.append(plot_trial_lateral_timeline(str(trial_id), series, events))
    write_readme(paths, [str(trial) for trial in trials])
    print(f"Wrote {len(paths)} plot(s) to {OUT_DIR}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
