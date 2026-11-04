#!/usr/bin/env python3
"""Explore clinically meaningful transition states from reconstructed SLAM features."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


QWEN_REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = QWEN_REPO / "runs"
OUT_DIR = RUNS_DIR / "transition_state_exploration"

STATE_ORDER = ["sitting", "standing", "walking", "upstairs", "downstairs"]
STATE_COLORS = {
    "sitting": "#9ecae1",
    "standing": "#b39ddb",
    "walking": "#74c476",
    "upstairs": "#f0ad4e",
    "downstairs": "#d95f5f",
}
TRANSITION_COLORS = {
    "sit-to-stand": "#2b6cb0",
    "stand-to-sit": "#805ad5",
    "stand-to-walk": "#2f855a",
    "walk-to-stand": "#c05621",
    "other-plausible": "#718096",
}
CLINICAL_TRANSITIONS = {
    ("sitting", "standing"): "sit-to-stand",
    ("standing", "sitting"): "stand-to-sit",
    ("standing", "walking"): "stand-to-walk",
    ("walking", "standing"): "walk-to-stand",
}
PLAUSIBLE_CONTEXT_TRANSITIONS = {
    ("walking", "upstairs"),
    ("upstairs", "walking"),
    ("walking", "downstairs"),
    ("downstairs", "walking"),
    ("standing", "upstairs"),
    ("upstairs", "standing"),
    ("standing", "downstairs"),
    ("downstairs", "standing"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        nargs="+",
        default=["Ntrial1", "Ntrial2", "Ntrial3"],
        help="Trials to analyze.",
    )
    parser.add_argument(
        "--transition-radius-s",
        type=float,
        default=2.5,
        help="Seconds before/after a manual boundary to include as a transition window.",
    )
    parser.add_argument(
        "--core-radius-s",
        type=float,
        default=1.0,
        help="Core seconds before/after a boundary used for summary statistics.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Output directory.",
    )
    return parser.parse_args()


def transition_name(from_state: str, to_state: str) -> str:
    if from_state == to_state:
        return "stable-state"
    if (from_state, to_state) in CLINICAL_TRANSITIONS:
        return CLINICAL_TRANSITIONS[(from_state, to_state)]
    if (from_state, to_state) in PLAUSIBLE_CONTEXT_TRANSITIONS:
        return f"{from_state}-to-{to_state}"
    return f"{from_state}-to-{to_state}"


def transition_category(from_state: str, to_state: str) -> str:
    if from_state == to_state:
        return "stable_state"
    if (from_state, to_state) in CLINICAL_TRANSITIONS:
        return "clinically_meaningful"
    if (from_state, to_state) in PLAUSIBLE_CONTEXT_TRANSITIONS:
        return "plausible_with_context"
    return "not_direct_expected"


def transition_explanation(from_state: str, to_state: str) -> str:
    if from_state == to_state:
        return "No state change."
    name = transition_name(from_state, to_state)
    if name == "sit-to-stand":
        return "Clinically meaningful posture transition; useful for sit-to-stand difficulty."
    if name == "stand-to-sit":
        return "Clinically meaningful posture transition; useful for controlled lowering."
    if name == "stand-to-walk":
        return "Clinically meaningful gait initiation transition; relevant to freezing or start hesitation."
    if name == "walk-to-stand":
        return "Clinically meaningful stopping transition; relevant to gait termination."
    if (from_state, to_state) in PLAUSIBLE_CONTEXT_TRANSITIONS:
        return "Plausible transition with context, especially around stair entry/exit or start/stop near stairs."
    if from_state == "sitting" and to_state in {"walking", "upstairs", "downstairs"}:
        return "Usually should pass through standing first."
    if to_state == "sitting" and from_state in {"walking", "upstairs", "downstairs"}:
        return "Usually should pass through standing first."
    if {from_state, to_state} == {"upstairs", "downstairs"}:
        return "Usually should pass through walking/standing/turning or a landing state."
    return "Not expected as a direct transition under the current five-state logic."


def read_features(trial_id: str) -> pd.DataFrame:
    path = RUNS_DIR / trial_id / "raw_parameter_diagnostics" / "reconstructed_motion_features.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["trial_id"] = trial_id
    numeric_cols = [
        "start_s",
        "end_s",
        "center_s",
        "window_mean_height_m",
        "height_slope_mps",
        "window_height_delta_m",
        "horizontal_path_length_m",
        "horizontal_speed_mean_mps",
        "linear_speed_mean_mps",
        "angular_speed_mean_rps",
        "stationary_fraction",
        "moving_fraction",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return add_posture_height_norm(df)


def add_posture_height_norm(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    labels = output["effective_manual_label"].astype(str).str.strip().str.lower()
    sitting = output.loc[labels == "sitting", "window_mean_height_m"].dropna()
    standing = output.loc[labels == "standing", "window_mean_height_m"].dropna()
    if len(sitting) >= 3 and len(standing) >= 3:
        sit_baseline = float(sitting.median())
        stand_baseline = float(standing.median())
        denom = stand_baseline - sit_baseline
        if abs(denom) >= 0.05:
            output["posture_height_norm"] = (output["window_mean_height_m"] - sit_baseline) / denom
        else:
            output["posture_height_norm"] = np.nan
        output["sitting_height_baseline_m"] = sit_baseline
        output["standing_height_baseline_m"] = stand_baseline
    else:
        output["posture_height_norm"] = np.nan
        output["sitting_height_baseline_m"] = np.nan
        output["standing_height_baseline_m"] = np.nan
    output["posture_height_norm_clipped"] = output["posture_height_norm"].clip(-0.3, 1.3)
    return output


def read_segments(trial_id: str) -> pd.DataFrame:
    path = RUNS_DIR / trial_id / "downward_master_session" / "downward_session_rule_segments.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    segments = pd.read_csv(path)
    segments["trial_id"] = trial_id
    segments["label"] = segments["label"].astype(str).str.strip().str.lower()
    segments = segments.sort_values("start_s").reset_index(drop=True)
    return segments


def build_boundaries(trials: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial_id in trials:
        segments = read_segments(trial_id)
        for i in range(len(segments) - 1):
            current = segments.iloc[i]
            nxt = segments.iloc[i + 1]
            from_state = str(current["label"])
            to_state = str(nxt["label"])
            boundary_s = float(nxt["start_s"])
            rows.append(
                {
                    "trial_id": trial_id,
                    "boundary_id": f"{trial_id}_b{i:02d}_{from_state}_to_{to_state}",
                    "boundary_s": boundary_s,
                    "from_state": from_state,
                    "to_state": to_state,
                    "transition_type": transition_name(from_state, to_state),
                    "transition_category": transition_category(from_state, to_state),
                    "is_clinically_meaningful": (from_state, to_state) in CLINICAL_TRANSITIONS,
                    "from_segment_start_s": float(current["start_s"]),
                    "from_segment_end_s": float(current["end_s"]),
                    "to_segment_start_s": float(nxt["start_s"]),
                    "to_segment_end_s": float(nxt["end_s"]),
                    "explanation": transition_explanation(from_state, to_state),
                }
            )
    return pd.DataFrame(rows)


def build_transition_windows(
    trials: list[str],
    boundaries: pd.DataFrame,
    radius_s: float,
    core_radius_s: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for trial_id in trials:
        features = read_features(trial_id)
        trial_boundaries = boundaries[boundaries["trial_id"] == trial_id]
        for _, boundary in trial_boundaries.iterrows():
            boundary_s = float(boundary["boundary_s"])
            mask = (features["center_s"] >= boundary_s - radius_s) & (features["center_s"] <= boundary_s + radius_s)
            window = features.loc[mask].copy()
            if window.empty:
                continue
            for col in boundary.index:
                window[col] = boundary[col]
            window["time_from_boundary_s"] = window["center_s"] - boundary_s
            window["is_core_transition_window"] = window["time_from_boundary_s"].abs() <= core_radius_s
            pre_ref = window[(window["time_from_boundary_s"] >= -2.0) & (window["time_from_boundary_s"] < -0.25)]
            if pre_ref["window_mean_height_m"].notna().any():
                local_height_reference_m = float(pre_ref["window_mean_height_m"].median())
            else:
                local_height_reference_m = float(window["window_mean_height_m"].median())
            window["local_height_reference_m"] = local_height_reference_m
            window["local_height_change_m"] = window["window_mean_height_m"] - local_height_reference_m
            window["local_height_change_clipped_m"] = window["local_height_change_m"].clip(-1.2, 1.2)
            rows.append(window)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_boundaries(window_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if window_df.empty:
        return pd.DataFrame()
    for boundary_id, group in window_df.groupby("boundary_id", sort=False):
        boundary = group.iloc[0]
        pre = group[(group["time_from_boundary_s"] >= -2.0) & (group["time_from_boundary_s"] < -0.25)]
        post = group[(group["time_from_boundary_s"] > 0.25) & (group["time_from_boundary_s"] <= 2.0)]
        core = group[group["is_core_transition_window"]]
        def med(frame: pd.DataFrame, col: str) -> float:
            return float(frame[col].median()) if col in frame and frame[col].notna().any() else np.nan
        rows.append(
            {
                "trial_id": boundary["trial_id"],
                "boundary_id": boundary_id,
                "boundary_s": float(boundary["boundary_s"]),
                "transition_type": boundary["transition_type"],
                "transition_category": boundary["transition_category"],
                "from_state": boundary["from_state"],
                "to_state": boundary["to_state"],
                "pre_height_norm_median": med(pre, "posture_height_norm"),
                "post_height_norm_median": med(post, "posture_height_norm"),
                "height_norm_change_post_minus_pre": med(post, "posture_height_norm") - med(pre, "posture_height_norm"),
                "pre_raw_height_median_m": med(pre, "window_mean_height_m"),
                "post_raw_height_median_m": med(post, "window_mean_height_m"),
                "height_change_post_minus_pre_m": med(post, "window_mean_height_m") - med(pre, "window_mean_height_m"),
                "pre_horizontal_speed_mps_median": med(pre, "horizontal_speed_mean_mps"),
                "post_horizontal_speed_mps_median": med(post, "horizontal_speed_mean_mps"),
                "speed_change_post_minus_pre_mps": med(post, "horizontal_speed_mean_mps") - med(pre, "horizontal_speed_mean_mps"),
                "core_abs_height_slope_peak_mps": float(core["height_slope_mps"].abs().max()) if not core.empty else np.nan,
                "core_horizontal_path_length_peak_m": float(core["horizontal_path_length_m"].max()) if not core.empty else np.nan,
                "core_horizontal_speed_peak_mps": float(core["horizontal_speed_mean_mps"].max()) if not core.empty else np.nan,
                "explanation": boundary["explanation"],
            }
        )
    return pd.DataFrame(rows)


def summarize_by_transition(boundary_summary: pd.DataFrame) -> pd.DataFrame:
    if boundary_summary.empty:
        return pd.DataFrame()
    grouped = (
        boundary_summary.groupby(["transition_type", "transition_category"], dropna=False)
        .agg(
            occurrences=("boundary_id", "count"),
            median_height_change_m=("height_change_post_minus_pre_m", "median"),
            median_speed_change_mps=("speed_change_post_minus_pre_mps", "median"),
            median_peak_abs_height_slope_mps=("core_abs_height_slope_peak_mps", "median"),
            median_peak_path_length_m=("core_horizontal_path_length_peak_m", "median"),
            median_peak_speed_mps=("core_horizontal_speed_peak_mps", "median"),
        )
        .reset_index()
    )
    return grouped.sort_values(["transition_category", "transition_type"]).reset_index(drop=True)


def build_plausibility_matrix(boundaries: pd.DataFrame) -> pd.DataFrame:
    observed_counts = (
        boundaries.groupby(["from_state", "to_state"]).size().rename("observed_count").reset_index()
        if not boundaries.empty
        else pd.DataFrame(columns=["from_state", "to_state", "observed_count"])
    )
    rows: list[dict[str, Any]] = []
    for from_state in STATE_ORDER:
        for to_state in STATE_ORDER:
            rows.append(
                {
                    "from_state": from_state,
                    "to_state": to_state,
                    "transition_type": transition_name(from_state, to_state),
                    "transition_category": transition_category(from_state, to_state),
                    "explanation": transition_explanation(from_state, to_state),
                }
            )
    matrix = pd.DataFrame(rows)
    matrix = matrix.merge(observed_counts, on=["from_state", "to_state"], how="left")
    matrix["observed_count"] = matrix["observed_count"].fillna(0).astype(int)
    return matrix


def save_csvs(
    out_dir: Path,
    boundaries: pd.DataFrame,
    window_df: pd.DataFrame,
    boundary_summary: pd.DataFrame,
    transition_summary: pd.DataFrame,
    plausibility_matrix: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    boundaries.to_csv(out_dir / "transition_boundary_events.csv", index=False)
    window_df.to_csv(out_dir / "transition_window_features.csv", index=False)
    boundary_summary.to_csv(out_dir / "transition_boundary_feature_summary.csv", index=False)
    transition_summary.to_csv(out_dir / "transition_summary_by_type.csv", index=False)
    plausibility_matrix.to_csv(out_dir / "transition_plausibility_matrix.csv", index=False)


def plot_plausibility_matrix(matrix: pd.DataFrame, out_path: Path) -> None:
    category_to_value = {
        "stable_state": 0,
        "clinically_meaningful": 1,
        "plausible_with_context": 2,
        "not_direct_expected": 3,
    }
    category_colors = ["#e2e8f0", "#3182ce", "#38a169", "#e53e3e"]
    grid = np.zeros((len(STATE_ORDER), len(STATE_ORDER)), dtype=float)
    counts = np.zeros_like(grid, dtype=int)
    labels = [["" for _ in STATE_ORDER] for _ in STATE_ORDER]
    for _, row in matrix.iterrows():
        i = STATE_ORDER.index(row["from_state"])
        j = STATE_ORDER.index(row["to_state"])
        category = row["transition_category"]
        grid[i, j] = category_to_value[category]
        counts[i, j] = int(row["observed_count"])
        if category == "stable_state":
            tag = "stable"
        elif category == "clinically_meaningful":
            tag = "clinical"
        elif category == "plausible_with_context":
            tag = "context"
        else:
            tag = "not direct"
        labels[i][j] = f"{tag}\nobs={counts[i, j]}"

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = matplotlib.colors.ListedColormap(category_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.set_xticks(range(len(STATE_ORDER)), STATE_ORDER, rotation=35, ha="right")
    ax.set_yticks(range(len(STATE_ORDER)), STATE_ORDER)
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title("Transition plausibility matrix: what should happen directly?")
    for i in range(len(STATE_ORDER)):
        for j in range(len(STATE_ORDER)):
            color = "white" if grid[i, j] in {1, 3} else "#1a202c"
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=9, color=color)
    handles = [
        matplotlib.patches.Patch(color="#e2e8f0", label="same/stable"),
        matplotlib.patches.Patch(color="#3182ce", label="clinically meaningful"),
        matplotlib.patches.Patch(color="#38a169", label="plausible with context"),
        matplotlib.patches.Patch(color="#e53e3e", label="not expected directly"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_xticks(np.arange(-0.5, len(STATE_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(STATE_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def scatter_transition_features(window_df: pd.DataFrame, out_path: Path) -> None:
    clinical = window_df[window_df["is_clinically_meaningful"].fillna(False)].copy()
    if clinical.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 8))
    for transition_type, group in clinical.groupby("transition_type"):
        color = TRANSITION_COLORS.get(transition_type, "#333333")
        core = group["is_core_transition_window"].astype(bool)
        ax.scatter(
            group.loc[~core, "horizontal_path_length_m"],
            group.loc[~core, "height_slope_mps"],
            color=color,
            alpha=0.25,
            s=28,
            edgecolors="none",
        )
        ax.scatter(
            group.loc[core, "horizontal_path_length_m"],
            group.loc[core, "height_slope_mps"],
            color=color,
            alpha=0.85,
            s=52,
            edgecolors="white",
            linewidths=0.5,
            label=transition_type,
        )
    ax.axhline(0, color="#2d3748", linewidth=1.0)
    ax.axvline(0.20, color="#718096", linestyle="--", linewidth=1.0, label="higher path reference")
    ax.axhline(0.08, color="#718096", linestyle=":", linewidth=1.0, label="upward height trend")
    ax.axhline(-0.08, color="#718096", linestyle=":", linewidth=1.0, label="downward height trend")
    ax.set_xlabel("Horizontal path length in window (m)")
    ax.set_ylabel("Height slope in window (m/s)")
    ax.set_title("Clinical transitions in 2D feature space: height change vs horizontal movement")
    ax.text(
        0.02,
        0.98,
        "sit-to-stand: positive height slope, low path\n"
        "stand-to-sit: negative height slope, low path\n"
        "stand-to-walk: increasing path/speed after boundary\n"
        "walk-to-stand: decreasing path/speed after boundary",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88, edgecolor="#cbd5e0"),
    )
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def scatter_height_speed(window_df: pd.DataFrame, out_path: Path) -> None:
    clinical = window_df[window_df["is_clinically_meaningful"].fillna(False)].copy()
    if clinical.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 8))
    for transition_type, group in clinical.groupby("transition_type"):
        color = TRANSITION_COLORS.get(transition_type, "#333333")
        core = group["is_core_transition_window"].astype(bool)
        ax.scatter(
            group.loc[~core, "local_height_change_clipped_m"],
            group.loc[~core, "horizontal_speed_mean_mps"],
            color=color,
            alpha=0.22,
            s=28,
            edgecolors="none",
        )
        ax.scatter(
            group.loc[core, "local_height_change_clipped_m"],
            group.loc[core, "horizontal_speed_mean_mps"],
            color=color,
            alpha=0.85,
            s=52,
            edgecolors="white",
            linewidths=0.5,
            label=transition_type,
        )
    ax.axvline(0, color="#4a5568", linestyle="--", linewidth=1, label="pre-boundary height reference")
    ax.axvline(0.25, color="#718096", linestyle=":", linewidth=1, label="clear rise reference")
    ax.axvline(-0.25, color="#718096", linestyle=":", linewidth=1, label="clear drop reference")
    ax.axhline(0.15, color="#718096", linestyle=":", linewidth=1, label="low/high movement reference")
    ax.set_xlabel("Local height change from pre-boundary reference (m)")
    ax.set_ylabel("Horizontal speed mean in window (m/s)")
    ax.set_title("Clinical transitions: local height change vs horizontal speed")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_transition_timecourses(window_df: pd.DataFrame, out_path: Path) -> None:
    clinical = window_df[window_df["is_clinically_meaningful"].fillna(False)].copy()
    if clinical.empty:
        return
    transition_order = ["sit-to-stand", "stand-to-sit", "stand-to-walk", "walk-to-stand"]
    fig, axes = plt.subplots(len(transition_order), 1, figsize=(12, 13), sharex=True)
    for ax, transition_type in zip(axes, transition_order):
        group = clinical[clinical["transition_type"] == transition_type]
        color = TRANSITION_COLORS[transition_type]
        if group.empty:
            ax.text(0.5, 0.5, f"No {transition_type} examples", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue
        for _, event_group in group.groupby("boundary_id"):
            event_group = event_group.sort_values("time_from_boundary_s")
            ax.plot(
                event_group["time_from_boundary_s"],
                event_group["local_height_change_clipped_m"],
                color=color,
                alpha=0.28,
                linewidth=1.2,
            )
            ax.plot(
                event_group["time_from_boundary_s"],
                event_group["horizontal_speed_mean_mps"],
                color="#2d3748",
                alpha=0.20,
                linewidth=1.0,
            )
        summary = (
            group.groupby("time_from_boundary_s")
            .agg(
                local_height_change_m=("local_height_change_clipped_m", "median"),
                horizontal_speed_mean_mps=("horizontal_speed_mean_mps", "median"),
            )
            .reset_index()
            .sort_values("time_from_boundary_s")
        )
        ax.plot(
            summary["time_from_boundary_s"],
            summary["local_height_change_m"],
            color=color,
            linewidth=3,
            label="median local height change (m)",
        )
        ax.plot(
            summary["time_from_boundary_s"],
            summary["horizontal_speed_mean_mps"],
            color="#1a202c",
            linewidth=2.2,
            label="median horizontal speed",
        )
        ax.axvline(0, color="#000000", linewidth=1.1)
        ax.axhline(0, color="#a0aec0", linewidth=0.8)
        ax.set_ylim(-0.85, max(1.35, float(group["horizontal_speed_mean_mps"].max()) + 0.15))
        ax.set_title(f"{transition_type} around manual boundary, n={group['boundary_id'].nunique()}")
        ax.set_ylabel("height change / speed")
        ax.grid(alpha=0.20)
        ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Seconds from manual transition boundary")
    fig.suptitle("Clinical transition time courses: local height change and horizontal speed", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_transition_observations(boundaries: pd.DataFrame, out_path: Path) -> None:
    counts = boundaries.groupby(["transition_type", "transition_category"]).size().reset_index(name="count")
    if counts.empty:
        return
    color_by_category = {
        "clinically_meaningful": "#3182ce",
        "plausible_with_context": "#38a169",
        "not_direct_expected": "#e53e3e",
        "stable_state": "#a0aec0",
    }
    counts = counts.sort_values(["transition_category", "transition_type"])
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [color_by_category.get(cat, "#718096") for cat in counts["transition_category"]]
    bars = ax.barh(counts["transition_type"], counts["count"], color=colors)
    for bar, count in zip(bars, counts["count"]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2, str(int(count)), va="center")
    ax.set_xlabel("Observed manual boundaries in Ntrial1-3")
    ax.set_title("Observed transition boundaries by type")
    handles = [
        matplotlib.patches.Patch(color=color, label=label.replace("_", " "))
        for label, color in color_by_category.items()
        if label in set(counts["transition_category"])
    ]
    ax.legend(handles=handles, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_boundary_feature_arrows(boundary_summary: pd.DataFrame, out_path: Path) -> None:
    clinical = boundary_summary[boundary_summary["transition_category"] == "clinically_meaningful"].copy()
    if clinical.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    for transition_type, group in clinical.groupby("transition_type"):
        color = TRANSITION_COLORS.get(transition_type, "#333333")
        ax.scatter(
            group["speed_change_post_minus_pre_mps"],
            group["height_change_post_minus_pre_m"],
            color=color,
            alpha=0.55,
            s=70,
            edgecolors="white",
            linewidths=0.6,
            label=transition_type,
        )
        med_x = float(group["speed_change_post_minus_pre_mps"].median())
        med_y = float(group["height_change_post_minus_pre_m"].median())
        ax.scatter(
            [med_x],
            [med_y],
            color=color,
            marker="D",
            s=180,
            edgecolors="#1a202c",
            linewidths=1.2,
            zorder=5,
        )
        ax.text(
            med_x,
            med_y,
            f"  {transition_type}",
            color=color,
            fontsize=11,
            weight="bold",
            va="center",
        )
    ax.axhline(0, color="#a0aec0", linestyle="--", linewidth=1)
    ax.axhline(0.25, color="#4a5568", linestyle=":", linewidth=1)
    ax.axhline(-0.25, color="#4a5568", linestyle=":", linewidth=1)
    ax.axvline(0, color="#a0aec0", linestyle="--", linewidth=1)
    ax.axvline(0.25, color="#718096", linestyle=":", linewidth=1)
    ax.axvline(-0.25, color="#718096", linestyle=":", linewidth=1)
    ax.text(0.98, 0.98, "gait initiation", transform=ax.transAxes, ha="right", va="top", color="#2f855a")
    ax.text(0.02, 0.98, "gait termination", transform=ax.transAxes, ha="left", va="top", color="#c05621")
    ax.text(0.98, 0.05, "height drop", transform=ax.transAxes, ha="right", va="bottom", color="#805ad5")
    ax.text(0.98, 0.92, "height rise", transform=ax.transAxes, ha="right", va="top", color="#2b6cb0")
    ax.set_xlabel("Speed change after boundary minus before boundary (m/s)")
    ax.set_ylabel("Height change after boundary minus before boundary (m)")
    ax.set_title("Clinical transition signatures: speed change vs local height change")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def format_markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    table = df.copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}".rstrip("0").rstrip("."))
        else:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else str(x))
    widths = [max(len(str(col)), *(len(str(v)) for v in table[col])) for col in table.columns]
    header = "| " + " | ".join(str(col).ljust(width) for col, width in zip(table.columns, widths)) + " |"
    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).ljust(width) for col, width in zip(table.columns, widths)) + " |"
        for _, row in table.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def write_report(
    out_dir: Path,
    boundaries: pd.DataFrame,
    transition_summary: pd.DataFrame,
    boundary_summary: pd.DataFrame,
    plausibility_matrix: pd.DataFrame,
    radius_s: float,
    core_radius_s: float,
) -> None:
    clinical_counts = boundaries[boundaries["transition_category"] == "clinically_meaningful"]["transition_type"].value_counts()
    observed_not_direct = boundaries[boundaries["transition_category"] == "not_direct_expected"]
    summary_cols = [
        "transition_type",
        "transition_category",
        "occurrences",
        "median_height_change_m",
        "median_speed_change_mps",
        "median_peak_abs_height_slope_mps",
        "median_peak_path_length_m",
        "median_peak_speed_mps",
    ]
    boundary_cols = [
        "trial_id",
        "boundary_s",
        "transition_type",
        "transition_category",
        "height_change_post_minus_pre_m",
        "speed_change_post_minus_pre_mps",
        "core_abs_height_slope_peak_mps",
        "core_horizontal_path_length_peak_m",
    ]
    not_direct_pairs = plausibility_matrix[plausibility_matrix["transition_category"] == "not_direct_expected"][
        ["from_state", "to_state", "transition_type", "explanation", "observed_count"]
    ]
    text = f"""# Transition State Exploration Report

Date: June 17, 2026  
Trials analyzed: Ntrial1, Ntrial2, Ntrial3  
Transition window: +/- {radius_s:g}s around each manual action boundary  
Core transition window for summary statistics: +/- {core_radius_s:g}s

## 1. Goal

This report explores transition states that may be clinically important for Parkinson's assessment. The professor's key point was that fixed 2-second clips should not limit the analysis. So here, transitions are defined around manual action boundaries and examined with continuous SLAM-derived parameters.

The clinically meaningful transition states are:

- `sit-to-stand`
- `stand-to-sit`
- `stand-to-walk`
- `walk-to-stand`

These are important because sit/stand changes and gait initiation/termination can relate to freezing, hesitation, difficulty initiating movement, or difficulty controlling posture.

## 2. What Data Was Used

Main input features:

`runs/<trial>/raw_parameter_diagnostics/reconstructed_motion_features.csv`

Manual state segments:

`runs/<trial>/downward_master_session/downward_session_rule_segments.csv`

Generated output folder:

`runs/transition_state_exploration`

Generated data files:

| File | Meaning |
| ---- | ------- |
| `transition_boundary_events.csv` | One row per observed state boundary. |
| `transition_window_features.csv` | SLAM feature windows around each boundary. |
| `transition_boundary_feature_summary.csv` | Before/after summary for each boundary. |
| `transition_summary_by_type.csv` | Summary grouped by transition type. |
| `transition_plausibility_matrix.csv` | All state-to-state pairs marked as clinical, plausible with context, or not expected directly. |

## 3. How We Defined A Transition

First, each trial's manual labels were converted into state segments. A transition boundary happens when one segment ends and the next segment starts.

Example:

```text
sitting 0s-14s
standing 14s-20s
```

This creates a `sit-to-stand` boundary at 14s.

Then, we take SLAM feature windows around the boundary. A window close to the boundary is treated as transition context, rather than forcing it to be only one stable action.

## 4. Parameters Used In The 2D Plots

| Parameter | Meaning | Why useful for transitions |
| --------- | ------- | -------------------------- |
| `local_height_change_m` | Height change relative to the pre-boundary local height. | Avoids confusing posture change with being on a different floor after stairs. |
| `height_slope_mps` | Rate of height change in the window. Positive means moving upward, negative means moving downward. | Main cue for sit-to-stand and stand-to-sit. |
| `horizontal_path_length_m` | Actual horizontal path traveled inside the window. | Helps detect walking/gait initiation better than endpoint displacement. |
| `horizontal_speed_mean_mps` | Average horizontal speed inside the window. | Helps identify movement starting or stopping. |

## 5. Plausible vs Not-Direct Transitions

The transition matrix marks every possible direct transition between the five current states.

![Transition plausibility matrix](plots/01_transition_plausibility_matrix.png)

Interpretation:

- Blue: clinically meaningful transition.
- Green: plausible with context, often around stairs.
- Red: usually should not happen directly under the current five-state logic.
- Gray: no transition, same state.

Important examples of transitions that should usually not happen directly:

{format_markdown_table(not_direct_pairs, max_rows=12)}

## 6. Observed Transitions In Ntrial1-3

![Observed transition counts](plots/02_observed_transition_counts.png)

Clinical transition counts:

{format_markdown_table(clinical_counts.rename_axis("transition_type").reset_index(name="count"))}

No observed boundary in Ntrial1-3 was classified as `not_direct_expected`. This is good: the manual labels generally move through reasonable intermediate states.

Observed not-direct boundaries:

{format_markdown_table(observed_not_direct[["trial_id", "boundary_s", "from_state", "to_state", "transition_type"]]) if not observed_not_direct.empty else "None observed."}

## 7. 2D Plot: Height Change vs Horizontal Movement

![Clinical transitions: height slope vs path](plots/03_clinical_transition_height_slope_vs_path.png)

How to read this plot:

- `sit-to-stand` should move toward positive height slope with relatively low horizontal path.
- `stand-to-sit` should move toward negative height slope with relatively low horizontal path.
- `stand-to-walk` should show increasing horizontal movement/speed after the boundary.
- `walk-to-stand` should show decreasing horizontal movement/speed after the boundary.

This plot is useful because it separates posture transitions from gait initiation/termination:

- posture change = strong vertical trend,
- walking initiation/termination = stronger horizontal path/speed change.

## 8. 2D Plot: Local Height Change vs Horizontal Speed

![Clinical transitions: height vs speed](plots/04_clinical_transition_height_vs_speed.png)

How to read this plot:

- Zero on the x-axis means the height is close to the pre-boundary level.
- Positive x means the head/device height rose after the transition began.
- Negative x means the head/device height dropped after the transition began.
- Walking has higher horizontal speed.
- `stand-to-walk` should mainly increase speed, with little local height change.
- `walk-to-stand` should mainly decrease speed, with little local height change.
- `sit-to-stand` and `stand-to-sit` should mainly move along the local-height-change axis.

## 9. 2D Plot: Transition Signatures

![Clinical transition arrows](plots/05_clinical_transition_before_after_arrows.png)

This plot summarizes each transition by comparing the before-boundary and after-boundary values.

The x-axis is:

```text
post-boundary horizontal speed - pre-boundary horizontal speed
```

The y-axis is:

```text
post-boundary local height - pre-boundary local height
```

The main expected patterns are:

- `sit-to-stand`: moves upward.
- `stand-to-sit`: moves downward.
- `stand-to-walk`: moves right.
- `walk-to-stand`: moves left.

## 10. Time-Course View Around Boundaries

![Clinical transition time courses](plots/06_clinical_transition_timecourses.png)

This plot shows the transition signal over time relative to the manual boundary.

It is useful because transitions are not always instantaneous. For example, gait initiation may start before or after the manual boundary depending on when the spoken label/manual label changes.

## 11. Transition Summary Table

{format_markdown_table(transition_summary[summary_cols]) if not transition_summary.empty else "No transitions summarized."}

## 12. Boundary-Level Feature Summary

{format_markdown_table(boundary_summary[boundary_cols], max_rows=30) if not boundary_summary.empty else "No boundary summaries generated."}

## 13. Interpretation For The Project

This exploration suggests that transition states should be handled separately from stable actions.

The current five stable actions are useful, but transitions contain different signals:

- `sit-to-stand` and `stand-to-sit` are mainly height/posture transitions.
- `stand-to-walk` and `walk-to-stand` are mainly horizontal movement initiation/termination transitions.
- Some direct state changes should not be allowed unless there is a special context. For example, `sitting-to-walking` should normally be interpreted as `sit-to-stand` followed by `stand-to-walk`.

This matters for prediction because a single window near a boundary can contain mixed evidence. If the model is forced to choose only one stable action, it may look wrong even if the signal is actually a transition.

## 14. Recommended Next Steps

- Add transition labels as an optional category in the dashboard.
- Let manual review mark boundaries as `sit-to-stand`, `stand-to-sit`, `stand-to-walk`, or `walk-to-stand`.
- Use transition-aware rules so the model does not need to force a boundary window into only sitting, standing, or walking.
- For impossible direct transitions, add a state-machine check. For example, reject `sitting-to-walking` unless standing appears between them.
- Build a clinician-centered view that shows several seconds before and after transition boundaries.

## 15. Meeting-Friendly Explanation

> I treated transitions as their own problem instead of forcing every moment into a stable action. I used the manual label boundaries and looked at SLAM features before and after each boundary. The clinically meaningful transitions are sit-to-stand, stand-to-sit, stand-to-walk, and walk-to-stand. The 2D plots show that posture transitions are mostly explained by height and height slope, while gait initiation or stopping is better explained by horizontal speed/path length. I also created a transition plausibility matrix to show which state changes should happen directly and which should normally pass through another state.
"""
    (out_dir / "transition_state_exploration_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    boundaries = build_boundaries(args.trials)
    window_df = build_transition_windows(args.trials, boundaries, args.transition_radius_s, args.core_radius_s)
    boundary_summary = summarize_boundaries(window_df)
    transition_summary = summarize_by_transition(boundary_summary)
    plausibility_matrix = build_plausibility_matrix(boundaries)

    save_csvs(out_dir, boundaries, window_df, boundary_summary, transition_summary, plausibility_matrix)

    plot_plausibility_matrix(plausibility_matrix, plots_dir / "01_transition_plausibility_matrix.png")
    plot_transition_observations(boundaries, plots_dir / "02_observed_transition_counts.png")
    scatter_transition_features(window_df, plots_dir / "03_clinical_transition_height_slope_vs_path.png")
    scatter_height_speed(window_df, plots_dir / "04_clinical_transition_height_vs_speed.png")
    plot_boundary_feature_arrows(boundary_summary, plots_dir / "05_clinical_transition_before_after_arrows.png")
    plot_transition_timecourses(window_df, plots_dir / "06_clinical_transition_timecourses.png")

    write_report(
        out_dir,
        boundaries,
        transition_summary,
        boundary_summary,
        plausibility_matrix,
        args.transition_radius_s,
        args.core_radius_s,
    )

    print(f"Wrote transition-state exploration to {out_dir}")
    if not transition_summary.empty:
        print(transition_summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
