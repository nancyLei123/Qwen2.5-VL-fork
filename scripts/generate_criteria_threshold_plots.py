#!/usr/bin/env python3
"""Generate step-by-step criteria-v2 plots with explicit threshold overlays."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
DEFAULT_OUTPUT_DIR = RUNS_DIR / "raw_parameter_diagnostics" / "criteria_step_by_step_thresholds"

ACTION_ORDER = ["sitting", "standing", "walking", "upstairs", "downstairs", "transition"]
ACTION_COLORS = {
    "sitting": "#2d6cdf",
    "standing": "#d17b00",
    "walking": "#198754",
    "upstairs": "#c2410c",
    "downstairs": "#6f42c1",
    "transition": "#6b7280",
}

FIXED_STAIR_HEIGHT_DELTA_MIN_M = 0.18
FIXED_NEAR_ZERO_HEIGHT_SLOPE_MPS = 0.08
FIXED_POSTURE_TRANSITION_PATH_MAX_M = 0.55
FIXED_POSTURE_TRANSITION_SPEED_MIN_MPS = 0.20
FIXED_POSTURE_TRANSITION_SLOPE_MIN_MPS = 0.08
FIXED_STAIR_RATIO_MIN = 1.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        nargs="+",
        default=None,
        help="Trials to include. Defaults to all trials in reconstructed_motion_feature_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder for criteria explanation plots.",
    )
    parser.add_argument(
        "--window-s",
        type=float,
        default=2.0,
        help="Window length to plot. The current dashboard clip comparison uses 2.0 s windows.",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def plot_label(row: pd.Series) -> str:
    count = int(float(row.get("effective_manual_action_count", 0) or 0))
    if count > 1:
        return "transition"
    label = str(row.get("effective_action_1") or row.get("effective_manual_label") or "").strip().lower()
    return label if label in ACTION_COLORS else "transition"


def load_summary() -> pd.DataFrame:
    summary_path = RUNS_DIR / "reconstructed_motion_feature_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
    return pd.read_csv(summary_path)


def load_trial_features(trial_id: str, window_s: float) -> pd.DataFrame:
    path = RUNS_DIR / trial_id / "raw_parameter_diagnostics" / "reconstructed_motion_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing reconstructed features for {trial_id}: {path}")
    df = pd.read_csv(path)
    df = df[np.isclose(pd.to_numeric(df["window_s"], errors="coerce"), window_s)].copy()
    df = df[pd.to_numeric(df.get("effective_manual_action_count", 0), errors="coerce").fillna(0) > 0].copy()
    numeric_cols = [
        "center_s",
        "window_median_height_m",
        "window_height_delta_m",
        "height_slope_mps",
        "horizontal_path_length_m",
        "net_horizontal_displacement_m",
        "linear_speed_mean_mps",
        "moving_fraction",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["plot_label"] = df.apply(plot_label, axis=1)
    return df.dropna(subset=["horizontal_path_length_m", "linear_speed_mean_mps", "height_slope_mps"])


def trial_thresholds(summary_row: pd.Series) -> dict[str, float | None]:
    return {
        "walking_path_min_m": safe_float(summary_row.get("criteria_v2_walking_path_min_m")),
        "walking_speed_min_mps": safe_float(summary_row.get("criteria_v2_walking_speed_min_mps")),
        "stair_path_min_m": safe_float(summary_row.get("criteria_v2_stair_path_min_m")),
        "stair_speed_min_mps": safe_float(summary_row.get("criteria_v2_stair_speed_min_mps")),
        "stair_height_slope_min_mps": safe_float(summary_row.get("criteria_v2_stair_height_slope_min_mps")),
        "sitting_baseline_z_m": safe_float(summary_row.get("sitting_baseline_z_m")),
        "standing_baseline_z_m": safe_float(summary_row.get("standing_baseline_z_m")),
        "posture_z_band_m": safe_float(summary_row.get("posture_z_band_m")),
    }


def panel_shape(count: int) -> tuple[int, int]:
    cols = 3 if count > 4 else 2
    rows = int(math.ceil(count / cols))
    return rows, cols


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def scatter_by_label(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str) -> None:
    for label in ACTION_ORDER:
        part = df[df["plot_label"] == label]
        if part.empty:
            continue
        ax.scatter(
            part[x_col],
            part[y_col],
            s=20,
            alpha=0.72,
            color=ACTION_COLORS[label],
            label=label,
            edgecolors="white",
            linewidths=0.25,
        )


def boxplot_by_label(ax: plt.Axes, df: pd.DataFrame, value_col: str) -> None:
    labels = [label for label in ACTION_ORDER if not df[df["plot_label"] == label].empty]
    values = [df.loc[df["plot_label"] == label, value_col].dropna().to_numpy() for label in labels]
    if not values:
        ax.text(0.5, 0.5, "No labeled windows", ha="center", va="center", transform=ax.transAxes)
        return
    box = ax.boxplot(values, patch_artist=True, tick_labels=labels, showfliers=False)
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(ACTION_COLORS[label])
        patch.set_alpha(0.55)
    ax.tick_params(axis="x", rotation=35)


def add_common_legend(fig: plt.Figure) -> None:
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ACTION_COLORS[label], label=label, markersize=7)
        for label in ACTION_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False)


def plot_speed_boxplots(trial_data: dict[str, pd.DataFrame], thresholds: dict[str, dict[str, float | None]], out_dir: Path) -> None:
    rows, cols = panel_shape(len(trial_data))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 3.8), squeeze=False)
    fig.suptitle("Step 0A. Speed threshold setup: mean linear speed by label", fontsize=16, fontweight="bold")
    for ax, (trial_id, df) in zip(axes.ravel(), trial_data.items()):
        boxplot_by_label(ax, df, "linear_speed_mean_mps")
        t = thresholds[trial_id]
        if t["walking_speed_min_mps"] is not None:
            ax.axhline(t["walking_speed_min_mps"], color="#111827", linestyle="--", linewidth=1.4, label="walking speed min")
        if t["stair_speed_min_mps"] is not None:
            ax.axhline(t["stair_speed_min_mps"], color="#9a3412", linestyle=":", linewidth=1.6, label="stair speed min")
        ax.set_title(trial_id)
        ax.set_ylabel("linear_speed_mean_mps")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(trial_data) :]:
        ax.axis("off")
    save_figure(fig, out_dir / "01A_speed_boxplot_with_thresholds.png")


def plot_path_boxplots(trial_data: dict[str, pd.DataFrame], thresholds: dict[str, dict[str, float | None]], out_dir: Path) -> None:
    rows, cols = panel_shape(len(trial_data))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 3.8), squeeze=False)
    fig.suptitle("Step 0B. Horizontal path threshold setup: path length by label", fontsize=16, fontweight="bold")
    for ax, (trial_id, df) in zip(axes.ravel(), trial_data.items()):
        boxplot_by_label(ax, df, "horizontal_path_length_m")
        t = thresholds[trial_id]
        if t["walking_path_min_m"] is not None:
            ax.axhline(t["walking_path_min_m"], color="#111827", linestyle="--", linewidth=1.4, label="walking path min")
        if t["stair_path_min_m"] is not None:
            ax.axhline(t["stair_path_min_m"], color="#9a3412", linestyle=":", linewidth=1.6, label="stair path min")
        ax.set_title(trial_id)
        ax.set_ylabel("horizontal_path_length_m")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(trial_data) :]:
        ax.axis("off")
    save_figure(fig, out_dir / "01B_horizontal_path_boxplot_with_thresholds.png")


def plot_slope_boxplots(trial_data: dict[str, pd.DataFrame], thresholds: dict[str, dict[str, float | None]], out_dir: Path) -> None:
    rows, cols = panel_shape(len(trial_data))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 3.8), squeeze=False)
    fig.suptitle("Step 0C. Vertical trend threshold setup: height slope by label", fontsize=16, fontweight="bold")
    for ax, (trial_id, df) in zip(axes.ravel(), trial_data.items()):
        boxplot_by_label(ax, df, "height_slope_mps")
        t = thresholds[trial_id]
        stair_slope = t["stair_height_slope_min_mps"]
        ax.axhspan(-FIXED_NEAR_ZERO_HEIGHT_SLOPE_MPS, FIXED_NEAR_ZERO_HEIGHT_SLOPE_MPS, color="#d1d5db", alpha=0.25, label="near-zero slope band")
        if stair_slope is not None:
            ax.axhline(stair_slope, color="#9a3412", linestyle=":", linewidth=1.6, label="+ stair slope min")
            ax.axhline(-stair_slope, color="#9a3412", linestyle=":", linewidth=1.6, label="- stair slope min")
        ax.set_title(trial_id)
        ax.set_ylabel("height_slope_mps")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(trial_data) :]:
        ax.axis("off")
    save_figure(fig, out_dir / "01C_height_slope_boxplot_with_thresholds.png")


def plot_transition_guard(trial_data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    rows, cols = panel_shape(len(trial_data))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 4.0), squeeze=False)
    fig.suptitle("Step 1. Posture-transition guard: large height change + small horizontal path is not automatically stairs", fontsize=15, fontweight="bold")
    for ax, (trial_id, df) in zip(axes.ravel(), trial_data.items()):
        scatter_by_label(ax, df, "horizontal_path_length_m", "window_height_delta_m")
        ax.axvspan(0, FIXED_POSTURE_TRANSITION_PATH_MAX_M, color="#fef3c7", alpha=0.25)
        ax.axvline(FIXED_POSTURE_TRANSITION_PATH_MAX_M, color="#92400e", linestyle="--", linewidth=1.4, label="path < 0.55 guard")
        ax.axhline(FIXED_STAIR_HEIGHT_DELTA_MIN_M, color="#991b1b", linestyle=":", linewidth=1.5, label="+/- 0.18 m delta")
        ax.axhline(-FIXED_STAIR_HEIGHT_DELTA_MIN_M, color="#991b1b", linestyle=":", linewidth=1.5)
        ax.set_title(trial_id)
        ax.set_xlabel("horizontal_path_length_m")
        ax.set_ylabel("window_height_delta_m")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(trial_data) :]:
        ax.axis("off")
    add_common_legend(fig)
    save_figure(fig, out_dir / "02_posture_transition_guard_height_delta_vs_path.png")


def plot_stair_slope_vs_path(
    trial_data: dict[str, pd.DataFrame],
    thresholds: dict[str, dict[str, float | None]],
    out_dir: Path,
) -> None:
    rows, cols = panel_shape(len(trial_data))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 4.0), squeeze=False)
    fig.suptitle("Step 2A. Stair detection: vertical trend direction vs horizontal path", fontsize=16, fontweight="bold")
    for ax, (trial_id, df) in zip(axes.ravel(), trial_data.items()):
        t = thresholds[trial_id]
        stair_path = t["stair_path_min_m"]
        stair_slope = t["stair_height_slope_min_mps"]
        scatter_by_label(ax, df, "horizontal_path_length_m", "height_slope_mps")
        ax.axhspan(-FIXED_NEAR_ZERO_HEIGHT_SLOPE_MPS, FIXED_NEAR_ZERO_HEIGHT_SLOPE_MPS, color="#d1d5db", alpha=0.2, label="walking near-zero slope")
        if stair_path is not None:
            ax.axvline(stair_path, color="#9a3412", linestyle="--", linewidth=1.4, label="stair path min")
        if stair_slope is not None:
            ax.axhline(stair_slope, color="#c2410c", linestyle=":", linewidth=1.6, label="+ stair slope min")
            ax.axhline(-stair_slope, color="#6f42c1", linestyle=":", linewidth=1.6, label="- stair slope min")
        ax.set_title(trial_id)
        ax.set_xlabel("horizontal_path_length_m")
        ax.set_ylabel("height_slope_mps")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(trial_data) :]:
        ax.axis("off")
    add_common_legend(fig)
    save_figure(fig, out_dir / "03_stair_detection_height_slope_vs_path.png")


def plot_stair_delta_vs_path(
    trial_data: dict[str, pd.DataFrame],
    thresholds: dict[str, dict[str, float | None]],
    out_dir: Path,
) -> None:
    rows, cols = panel_shape(len(trial_data))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 4.0), squeeze=False)
    fig.suptitle("Step 2B. Stair detection: vertical delta direction vs horizontal path", fontsize=16, fontweight="bold")
    for ax, (trial_id, df) in zip(axes.ravel(), trial_data.items()):
        t = thresholds[trial_id]
        scatter_by_label(ax, df, "horizontal_path_length_m", "window_height_delta_m")
        if t["stair_path_min_m"] is not None:
            ax.axvline(t["stair_path_min_m"], color="#9a3412", linestyle="--", linewidth=1.4, label="stair path min")
        ax.axhline(FIXED_STAIR_HEIGHT_DELTA_MIN_M, color="#c2410c", linestyle=":", linewidth=1.6, label="+ 0.18 m")
        ax.axhline(-FIXED_STAIR_HEIGHT_DELTA_MIN_M, color="#6f42c1", linestyle=":", linewidth=1.6, label="- 0.18 m")
        ax.set_title(trial_id)
        ax.set_xlabel("horizontal_path_length_m")
        ax.set_ylabel("window_height_delta_m")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(trial_data) :]:
        ax.axis("off")
    add_common_legend(fig)
    save_figure(fig, out_dir / "04_stair_detection_vertical_delta_vs_path.png")


def plot_walking_path_vs_speed(
    trial_data: dict[str, pd.DataFrame],
    thresholds: dict[str, dict[str, float | None]],
    out_dir: Path,
) -> None:
    rows, cols = panel_shape(len(trial_data))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 4.0), squeeze=False)
    fig.suptitle("Step 3. Walking detection: high horizontal path and speed, low vertical trend", fontsize=16, fontweight="bold")
    for ax, (trial_id, df) in zip(axes.ravel(), trial_data.items()):
        t = thresholds[trial_id]
        scatter_by_label(ax, df, "horizontal_path_length_m", "linear_speed_mean_mps")
        flat = df[df["height_slope_mps"].abs() <= FIXED_NEAR_ZERO_HEIGHT_SLOPE_MPS]
        if not flat.empty:
            ax.scatter(
                flat["horizontal_path_length_m"],
                flat["linear_speed_mean_mps"],
                s=44,
                facecolors="none",
                edgecolors="#111827",
                linewidths=0.6,
                label="abs(height_slope) <= 0.08",
            )
        if t["walking_path_min_m"] is not None:
            ax.axvline(t["walking_path_min_m"], color="#111827", linestyle="--", linewidth=1.4, label="walking path min")
        if t["walking_speed_min_mps"] is not None:
            ax.axhline(t["walking_speed_min_mps"], color="#111827", linestyle=":", linewidth=1.6, label="walking speed min")
        ax.set_title(trial_id)
        ax.set_xlabel("horizontal_path_length_m")
        ax.set_ylabel("linear_speed_mean_mps")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(trial_data) :]:
        ax.axis("off")
    add_common_legend(fig)
    save_figure(fig, out_dir / "05_walking_detection_path_vs_speed.png")


def plot_posture_height_over_time(
    trial_data: dict[str, pd.DataFrame],
    thresholds: dict[str, dict[str, float | None]],
    out_dir: Path,
) -> None:
    rows, cols = panel_shape(len(trial_data))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 4.0), squeeze=False)
    fig.suptitle("Step 4. Sitting vs standing: local median height compared with posture band", fontsize=16, fontweight="bold")
    for ax, (trial_id, df) in zip(axes.ravel(), trial_data.items()):
        t = thresholds[trial_id]
        for label in ACTION_ORDER:
            part = df[df["plot_label"] == label]
            if part.empty:
                continue
            ax.scatter(
                part["center_s"],
                part["window_median_height_m"],
                s=18,
                alpha=0.75,
                color=ACTION_COLORS[label],
                label=label,
            )
        if t["sitting_baseline_z_m"] is not None:
            ax.axhline(t["sitting_baseline_z_m"], color=ACTION_COLORS["sitting"], linestyle=":", linewidth=1.4, label="sitting baseline")
        if t["standing_baseline_z_m"] is not None:
            ax.axhline(t["standing_baseline_z_m"], color=ACTION_COLORS["standing"], linestyle=":", linewidth=1.4, label="standing baseline")
        if t["posture_z_band_m"] is not None:
            ax.axhline(t["posture_z_band_m"], color="#111827", linestyle="--", linewidth=1.6, label="posture band")
        ax.set_title(trial_id)
        ax.set_xlabel("session time (s)")
        ax.set_ylabel("window_median_height_m")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(trial_data) :]:
        ax.axis("off")
    save_figure(fig, out_dir / "06_posture_detection_height_over_time.png")


def write_threshold_table(summary: pd.DataFrame, trials: list[str], out_dir: Path) -> None:
    cols = [
        "trial_id",
        "sitting_baseline_z_m",
        "standing_baseline_z_m",
        "posture_z_band_m",
        "criteria_v2_walking_path_min_m",
        "criteria_v2_walking_speed_min_mps",
        "criteria_v2_stair_path_min_m",
        "criteria_v2_stair_speed_min_mps",
        "criteria_v2_stair_height_slope_min_mps",
        "criteria_v2_strict_accuracy",
        "criteria_v2_transition_aware_accuracy",
        "criteria_v2_stable_only_accuracy",
    ]
    table = summary[summary["trial_id"].isin(trials)][cols].copy()
    table["fixed_stair_height_delta_min_m"] = FIXED_STAIR_HEIGHT_DELTA_MIN_M
    table["fixed_near_zero_height_slope_mps"] = FIXED_NEAR_ZERO_HEIGHT_SLOPE_MPS
    table["fixed_posture_transition_path_max_m"] = FIXED_POSTURE_TRANSITION_PATH_MAX_M
    table["fixed_stair_path_to_vertical_ratio_min"] = FIXED_STAIR_RATIO_MIN
    table.to_csv(out_dir / "criteria_v2_threshold_table.csv", index=False)


def write_readme(out_dir: Path, trials: list[str]) -> None:
    readme = f"""# Criteria v2 Step-by-Step Threshold Plots

This folder explains the current `criteria_v2_prediction` logic using threshold-overlaid plots.

Trials included: {", ".join(trials)}

## How To Present The Folder

1. `01A_speed_boxplot_with_thresholds.png`
   - Shows how `linear_speed_mean_mps` differs by motion label.
   - Overlaid lines show `walking_speed_min` and `stair_speed_min`.
   - These thresholds are trial-relative: they come from the 90th percentile of 2-second SLAM window speed with guardrails.

2. `01B_horizontal_path_boxplot_with_thresholds.png`
   - Shows `horizontal_path_length_m` by label.
   - Overlaid lines show `walking_path_min` and `stair_path_min`.
   - This explains why path length is more useful than endpoint displacement for real movement.

3. `01C_height_slope_boxplot_with_thresholds.png`
   - Shows `height_slope_mps` by label.
   - The gray band is the near-zero vertical trend band: `abs(height_slope_mps) <= 0.08`.
   - Positive slope supports upstairs, negative slope supports downstairs, near-zero slope supports walking/flat movement.

4. `02_posture_transition_guard_height_delta_vs_path.png`
   - Explains why sit-to-stand or stand-to-sit should not automatically become upstairs/downstairs.
   - Guard idea: large height change plus small horizontal path can be a posture transition.
   - Key fixed values: `abs(height_delta) >= 0.18 m`, `horizontal_path < 0.55 m`, `abs(height_slope) >= 0.08 m/s`, `mean_speed >= 0.20 m/s`.

5. `03_stair_detection_height_slope_vs_path.png`
   - Main stair plot.
   - Upstairs should have enough horizontal path and positive height slope.
   - Downstairs should have enough horizontal path and negative height slope.

6. `04_stair_detection_vertical_delta_vs_path.png`
   - Complementary stair plot.
   - Shows the fixed vertical delta threshold: `+/- 0.18 m`.
   - Useful for explaining vertical direction in simpler terms than slope.

7. `05_walking_detection_path_vs_speed.png`
   - Main walking plot.
   - Walking needs high horizontal path and speed, while staying close to the near-zero vertical trend band.
   - Hollow circles mark windows with `abs(height_slope_mps) <= 0.08`.

8. `06_posture_detection_height_over_time.png`
   - Main sitting/standing plot.
   - Shows `window_median_height_m` over time.
   - Sitting/standing baselines and the posture band are drawn when available.

9. `criteria_v2_threshold_table.csv`
   - Exact threshold values used for each trial.

## Exact Threshold Formulas

For each trial, thresholds are estimated from 2-second SLAM windows:

```text
path_p90 = 90th percentile of horizontal_path_length_m
speed_p90 = 90th percentile of linear_speed_mean_mps
slope_abs_p90 = 90th percentile of abs(height_slope_mps)

walking_path_min = clamp(0.55 * path_p90, 0.60, 1.40)
walking_speed_min = clamp(0.55 * speed_p90, 0.35, 0.75)

stair_path_min = clamp(0.35 * path_p90, 0.30, 0.75)
stair_speed_min = clamp(0.35 * speed_p90, 0.18, 0.45)
stair_height_slope_min = clamp(0.50 * slope_abs_p90, 0.07, 0.16)

stair_height_delta_min = 0.18 m
near_zero_height_slope = 0.08 m/s
stair_path_to_vertical_ratio_min = 1.15
```

`clamp(value, low, high)` means the threshold is not allowed below `low` or above `high`.

## Important Interpretation

`window_height_delta_m` is not posture height. It is only `z_last - z_first` inside a window.

For sitting vs standing, the more important feature is `window_median_height_m`, compared against the posture band.
"""
    (out_dir / "README.md").write_text(readme)


def main() -> None:
    args = parse_args()
    summary = load_summary()
    trials = args.trials or summary["trial_id"].astype(str).tolist()
    trials = [trial for trial in trials if trial in set(summary["trial_id"].astype(str))]
    if not trials:
        raise ValueError("No matching trials found.")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_data = {trial: load_trial_features(trial, args.window_s) for trial in trials}
    summary_by_trial = {str(row["trial_id"]): row for _, row in summary.iterrows()}
    thresholds = {trial: trial_thresholds(summary_by_trial[trial]) for trial in trials}

    write_threshold_table(summary, trials, out_dir)
    plot_speed_boxplots(trial_data, thresholds, out_dir)
    plot_path_boxplots(trial_data, thresholds, out_dir)
    plot_slope_boxplots(trial_data, thresholds, out_dir)
    plot_transition_guard(trial_data, out_dir)
    plot_stair_slope_vs_path(trial_data, thresholds, out_dir)
    plot_stair_delta_vs_path(trial_data, thresholds, out_dir)
    plot_walking_path_vs_speed(trial_data, thresholds, out_dir)
    plot_posture_height_over_time(trial_data, thresholds, out_dir)
    write_readme(out_dir, trials)

    print(f"Wrote criteria threshold plots to: {out_dir}")


if __name__ == "__main__":
    main()
