#!/usr/bin/env python3
"""Explore raw SLAM trajectory parameters without assuming 2-second clips."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


QWEN_REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = QWEN_REPO / "runs"
WORKBOOK_PATH = RUNS_DIR / "spoken_action_clip_labels.xlsx"

ACTION_COLORS = {
    "sitting": "#2f78b7",
    "standing": "#7a4fb3",
    "walking": "#3a9d55",
    "upstairs": "#e58f2a",
    "downstairs": "#c94c4c",
    "go upstairs": "#e58f2a",
    "go downstairs": "#c94c4c",
}
ACTION_DISPLAY_LABELS = {
    "sitting": "Sitting",
    "standing": "Standing",
    "walking": "Walking",
    "upstairs": "Upstairs",
    "go upstairs": "Upstairs",
    "downstairs": "Downstairs",
    "go downstairs": "Downstairs",
}


def normalize_action_label(label: str) -> str:
    return str(label).strip().lower().replace("go ", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", nargs="+", default=["Ntrial1", "Ntrial2", "Ntrial3"])
    parser.add_argument("--sample-fps", type=float, default=30.0)
    parser.add_argument("--window-sizes", nargs="+", type=float, default=[0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--window-step-s", type=float, default=0.25)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_manifest(trial_id: str) -> dict[str, Any]:
    path = RUNS_DIR / trial_id / "trial_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing trial manifest: {path}")
    return load_json(path)


def estimate_alignment_offset_s(trajectory_df: pd.DataFrame, eye_gaze_csv: Path | None) -> float:
    if eye_gaze_csv is None or not eye_gaze_csv.exists() or trajectory_df.empty:
        return 0.0
    eye_df = pd.read_csv(eye_gaze_csv, usecols=["tracking_timestamp_us"])
    if eye_df.empty:
        return 0.0
    return max(
        0.0,
        (float(trajectory_df["tracking_timestamp_us"].iloc[0]) - float(eye_df["tracking_timestamp_us"].iloc[0]))
        / 1_000_000.0,
    )


def load_sync_window(trial_id: str) -> tuple[float, float]:
    sync_path = RUNS_DIR / trial_id / "downward_sync" / "sync_suggestion.json"
    if not sync_path.exists():
        return 0.0, 0.0
    sync = load_json(sync_path)
    return float(sync.get("aria_trim_start_s", 0.0)), float(sync.get("shared_duration_s", 0.0))


def load_speech_states(trial_id: str) -> pd.DataFrame:
    if not WORKBOOK_PATH.exists():
        return pd.DataFrame(columns=["label", "start_s", "end_s"])
    sheet_name = f"{trial_id}_speech_states"
    try:
        states = pd.read_excel(WORKBOOK_PATH, sheet_name=sheet_name)
    except ValueError:
        return pd.DataFrame(columns=["label", "start_s", "end_s"])
    required = {"label", "start_s", "end_s"}
    if not required.issubset(states.columns):
        return pd.DataFrame(columns=["label", "start_s", "end_s"])
    return states[list(required)].copy()


def load_effective_manual_intervals(trial_id: str) -> pd.DataFrame:
    """Return clip-level effective manual labels, splitting transition clips for plotting.

    The workbook's review sheet is the best current ground truth because it uses
    manual corrections when present and falls back to the spoken-action label.
    For transition labels such as sitting-standing, the interval is split evenly
    across the effective action columns so both action colors appear on plots.
    """
    from pd_insighter_environment_navigation import action_label_path, SIMULATED_TRIALS
    manual_path = action_label_path(RUNS_DIR / trial_id)
    if trial_id in SIMULATED_TRIALS and manual_path.exists():
        segments = load_json(manual_path).get("segments", [])
        rows = [{"label": row["motion_label"], "display_label": ACTION_DISPLAY_LABELS.get(row["motion_label"], row["motion_label"]),
                 "start_s": row["start_s"], "end_s": row["end_s"], "source_label": row["motion_label"], "source": "manual_action_segments"}
                for row in segments if row.get("motion_label") in ACTION_DISPLAY_LABELS and row["end_s"] > row["start_s"]]
        if rows:
            return pd.DataFrame(rows)
    if not WORKBOOK_PATH.exists():
        return load_speech_states(trial_id)

    sheet_name = f"{trial_id}_review"
    try:
        review = pd.read_excel(WORKBOOK_PATH, sheet_name=sheet_name)
    except ValueError:
        return load_speech_states(trial_id)

    required = {"clip_start_s", "clip_end_s", "effective_manual_label"}
    if not required.issubset(review.columns):
        return load_speech_states(trial_id)

    rows: list[dict[str, Any]] = []
    action_cols = [f"effective_action_{idx}" for idx in range(1, 6)]
    for _, clip in review.iterrows():
        start_s = float(clip["clip_start_s"])
        end_s = float(clip["clip_end_s"])
        if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
            continue

        actions = [
            normalize_action_label(str(clip[col]))
            for col in action_cols
            if col in review.columns and pd.notna(clip[col]) and str(clip[col]).strip()
        ]
        actions = [action for action in actions if action in ACTION_DISPLAY_LABELS]
        if not actions and pd.notna(clip["effective_manual_label"]):
            label = normalize_action_label(str(clip["effective_manual_label"]).split("-")[0])
            if label in ACTION_DISPLAY_LABELS:
                actions = [label]
        if not actions:
            continue

        segment_duration_s = (end_s - start_s) / len(actions)
        for action_index, action in enumerate(actions):
            rows.append(
                {
                    "label": action,
                    "display_label": ACTION_DISPLAY_LABELS[action],
                    "start_s": start_s + action_index * segment_duration_s,
                    "end_s": start_s + (action_index + 1) * segment_duration_s,
                    "source_label": str(clip["effective_manual_label"]),
                    "source": "effective_manual_label",
                }
            )

    if not rows:
        return load_speech_states(trial_id)
    return pd.DataFrame(rows)


def label_at_time(time_s: float, states: pd.DataFrame) -> str:
    if states.empty:
        return ""
    hit = states[(states["start_s"] <= time_s) & (states["end_s"] > time_s)]
    if hit.empty:
        return ""
    return str(hit.iloc[0]["label"])


def annotate_with_labels(df: pd.DataFrame, states: pd.DataFrame, time_col: str = "session_s") -> pd.DataFrame:
    output = df.copy()
    if states.empty:
        output["speech_label"] = ""
        return output
    labels = []
    state_rows = states.sort_values("start_s").to_dict("records")
    idx = 0
    for time_s in output[time_col].to_numpy():
        while idx + 1 < len(state_rows) and float(state_rows[idx]["end_s"]) <= time_s:
            idx += 1
        current = state_rows[idx] if state_rows else None
        if current and float(current["start_s"]) <= time_s < float(current["end_s"]):
            labels.append(str(current["label"]))
        else:
            labels.append("")
    output["speech_label"] = labels
    return output


def add_derived_columns(traj: pd.DataFrame) -> pd.DataFrame:
    output = traj.copy()
    output["linear_speed_mps"] = np.sqrt(
        output["device_linear_velocity_x_device"] ** 2
        + output["device_linear_velocity_y_device"] ** 2
        + output["device_linear_velocity_z_device"] ** 2
    )
    output["angular_speed_rps"] = np.sqrt(
        output["angular_velocity_x_device"] ** 2
        + output["angular_velocity_y_device"] ** 2
        + output["angular_velocity_z_device"] ** 2
    )
    output["xy_from_start_m"] = np.sqrt(
        (output["tx_world_device"] - output["tx_world_device"].iloc[0]) ** 2
        + (output["ty_world_device"] - output["ty_world_device"].iloc[0]) ** 2
    )
    output["z_from_start_m"] = output["tz_world_device"] - output["tz_world_device"].iloc[0]
    output["z_centered_m"] = output["tz_world_device"] - output["tz_world_device"].median()
    return output


def make_session_trajectory(trial_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = load_manifest(trial_id)
    trajectory_csv = Path(manifest["mps_outputs"]["closed_loop_trajectory_csv"]["path"])
    eye_gaze_csv = Path(manifest["mps_outputs"]["eye_gaze_csv"]["path"])
    traj = pd.read_csv(trajectory_csv)
    alignment_offset_s = estimate_alignment_offset_s(traj, eye_gaze_csv)
    if manifest.get("trial_type") == "parkinson_simulated" and manifest.get("slam_time_alignment") != "legacy_session_clock":
        metadata = load_json(Path(manifest["extracted_media"]["video_mp4"]["path"]).parent / "audio_metadata.json")
        alignment_offset_s = (float(traj["tracking_timestamp_us"].iloc[0]) - float(metadata["video_start_device_ns"]) / 1000.0) / 1e6
    aria_trim_start_s, shared_duration_s = load_sync_window(trial_id)
    first_ts = float(traj["tracking_timestamp_us"].iloc[0])
    traj["trajectory_rel_s"] = (traj["tracking_timestamp_us"] - first_ts) / 1_000_000.0
    traj["aria_video_s"] = traj["trajectory_rel_s"] + alignment_offset_s
    traj["session_s"] = traj["aria_video_s"] - aria_trim_start_s
    if shared_duration_s > 0:
        traj = traj[(traj["session_s"] >= 0.0) & (traj["session_s"] <= shared_duration_s)].copy()
    traj = add_derived_columns(traj)
    meta = {
        "trial_id": trial_id,
        "trajectory_csv": str(trajectory_csv),
        "eye_gaze_csv": str(eye_gaze_csv),
        "alignment_offset_s": alignment_offset_s,
        "aria_trim_start_s": aria_trim_start_s,
        "shared_duration_s": shared_duration_s,
        "sample_count": int(len(traj)),
        "duration_s": float(traj["session_s"].max() - traj["session_s"].min()) if len(traj) else 0.0,
        "median_sample_dt_s": float(traj["session_s"].diff().median()) if len(traj) > 1 else None,
    }
    return traj, meta


def downsample_session(traj: pd.DataFrame, fps: float, states: pd.DataFrame) -> pd.DataFrame:
    if traj.empty:
        return traj
    bin_size = 1.0 / fps
    work = traj.copy()
    work["time_bin"] = np.floor(work["session_s"] / bin_size).astype(int)
    grouped = work.groupby("time_bin", as_index=False).agg(
        session_s=("session_s", "mean"),
        tx_world_device=("tx_world_device", "mean"),
        ty_world_device=("ty_world_device", "mean"),
        tz_world_device=("tz_world_device", "mean"),
        z_from_start_m=("z_from_start_m", "mean"),
        z_centered_m=("z_centered_m", "mean"),
        xy_from_start_m=("xy_from_start_m", "mean"),
        linear_speed_mps=("linear_speed_mps", "mean"),
        angular_speed_rps=("angular_speed_rps", "mean"),
        quality_score=("quality_score", "mean"),
    )
    return annotate_with_labels(grouped, states)


def sliding_window_metrics(
    traj: pd.DataFrame,
    states: pd.DataFrame,
    window_sizes: list[float],
    step_s: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if traj.empty:
        return pd.DataFrame(rows)

    times = traj["session_s"].to_numpy()
    x = traj["tx_world_device"].to_numpy()
    y = traj["ty_world_device"].to_numpy()
    z = traj["tz_world_device"].to_numpy()
    speed = traj["linear_speed_mps"].to_numpy()
    angular = traj["angular_speed_rps"].to_numpy()
    duration = float(times[-1] - times[0])

    for window_s in window_sizes:
        if duration < window_s:
            continue
        starts = np.arange(0.0, duration - window_s + 1e-9, step_s)
        for start_s in starts:
            end_s = start_s + window_s
            left = int(np.searchsorted(times, start_s, side="left"))
            right = int(np.searchsorted(times, end_s, side="left"))
            if right - left < 2:
                continue
            dx = float(x[right - 1] - x[left])
            dy = float(y[right - 1] - y[left])
            dz = float(z[right - 1] - z[left])
            actual_dt_s = float(times[right - 1] - times[left])
            if not np.isfinite(actual_dt_s) or actual_dt_s <= 0:
                actual_dt_s = window_s
            horizontal_path_length_m = float(np.sum(np.hypot(np.diff(x[left:right]), np.diff(y[left:right]))))
            center_s = start_s + window_s / 2.0
            rows.append(
                {
                    "window_s": window_s,
                    "start_s": start_s,
                    "end_s": end_s,
                    "center_s": center_s,
                    "horizontal_delta_m": float(math.hypot(dx, dy)),
                    "horizontal_path_length_m": horizontal_path_length_m,
                    "vertical_delta_m": dz,
                    "height_slope_mps": dz / actual_dt_s,
                    "mean_z_m": float(np.mean(z[left:right])),
                    "median_z_m": float(np.median(z[left:right])),
                    "std_z_m": float(np.std(z[left:right])),
                    "min_z_m": float(np.min(z[left:right])),
                    "max_z_m": float(np.max(z[left:right])),
                    "z_range_m": float(np.max(z[left:right]) - np.min(z[left:right])),
                    "mean_linear_speed_mps": float(np.mean(speed[left:right])),
                    "mean_angular_speed_rps": float(np.mean(angular[left:right])),
                    "speech_label": label_at_time(center_s, states),
                }
            )
    return pd.DataFrame(rows)


def speech_state_height_summary(traj: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if states.empty or traj.empty:
        return pd.DataFrame(rows)
    for idx, state in states.iterrows():
        start_s = float(state["start_s"])
        end_s = float(state["end_s"])
        label = str(state["label"])
        section = traj[(traj["session_s"] >= start_s) & (traj["session_s"] < end_s)]
        if section.empty:
            continue
        rows.append(
            {
                "state_index": int(idx),
                "label": label,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
                "sample_count": int(len(section)),
                "mean_z_m": float(section["tz_world_device"].mean()),
                "median_z_m": float(section["tz_world_device"].median()),
                "std_z_m": float(section["tz_world_device"].std(ddof=0)),
                "min_z_m": float(section["tz_world_device"].min()),
                "max_z_m": float(section["tz_world_device"].max()),
                "mean_x_m": float(section["tx_world_device"].mean()),
                "mean_y_m": float(section["ty_world_device"].mean()),
                "xy_range_m": float(
                    math.hypot(
                        section["tx_world_device"].max() - section["tx_world_device"].min(),
                        section["ty_world_device"].max() - section["ty_world_device"].min(),
                    )
                ),
                "mean_linear_speed_mps": float(section["linear_speed_mps"].mean()),
                "mean_angular_speed_rps": float(section["angular_speed_rps"].mean()),
            }
        )
    return pd.DataFrame(rows)


def style_state_spans(ax: Any, states: pd.DataFrame) -> list[str]:
    used_labels: list[str] = []
    for _, state in states.iterrows():
        label = normalize_action_label(str(state["label"]))
        color = ACTION_COLORS.get(label, "#b8c0cc")
        ax.axvspan(float(state["start_s"]), float(state["end_s"]), color=color, alpha=0.14, linewidth=0)
        if label in ACTION_DISPLAY_LABELS and label not in used_labels:
            used_labels.append(label)
    return used_labels


def add_action_color_legend(ax: Any, used_labels: list[str]) -> None:
    if not used_labels:
        return
    from matplotlib.patches import Patch

    ordered_labels = [
        label
        for label in ["sitting", "standing", "walking", "upstairs", "downstairs"]
        if label in used_labels
    ]
    handles = [
        Patch(facecolor=ACTION_COLORS[label], alpha=0.28, label=ACTION_DISPLAY_LABELS[label])
        for label in ordered_labels
    ]
    legend = ax.legend(
        handles=handles,
        title="Manual action colors",
        loc="upper left",
        frameon=True,
        fontsize=8,
        title_fontsize=8,
    )
    ax.add_artist(legend)


def color_series_by_label(labels: pd.Series) -> list[str]:
    return [ACTION_COLORS.get(normalize_action_label(str(label)), "#d36b42") for label in labels]


def plot_mean_z_by_motion_label_boxplot(
    trial_id: str,
    out_dir: Path,
    window_df: pd.DataFrame,
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if window_df.empty or "mean_z_m" not in window_df.columns or "speech_label" not in window_df.columns:
        return None

    two_s = window_df[np.isclose(window_df["window_s"], 2.0)].copy()
    if two_s.empty:
        return None

    two_s["action_label"] = two_s["speech_label"].map(normalize_action_label)
    label_order = ["sitting", "standing", "walking", "upstairs", "downstairs"]
    labels = [
        label
        for label in label_order
        if label in set(two_s["action_label"]) and two_s.loc[two_s["action_label"] == label, "mean_z_m"].notna().any()
    ]
    if not labels:
        return None

    data = [two_s.loc[two_s["action_label"] == label, "mean_z_m"].dropna().to_numpy() for label in labels]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    box = ax.boxplot(
        data,
        tick_labels=[ACTION_DISPLAY_LABELS[label] for label in labels],
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#1f2d3a", "linewidth": 1.8},
        whiskerprops={"color": "#4c5661", "linewidth": 1.0},
        capprops={"color": "#4c5661", "linewidth": 1.0},
    )
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(ACTION_COLORS[label])
        patch.set_alpha(0.35)
        patch.set_edgecolor(ACTION_COLORS[label])

    rng = np.random.default_rng(7)
    for x_index, (values, label) in enumerate(zip(data, labels), start=1):
        jitter = rng.normal(0, 0.045, size=len(values))
        ax.scatter(
            np.full(len(values), x_index) + jitter,
            values,
            s=16,
            color=ACTION_COLORS[label],
            alpha=0.35,
            edgecolors="none",
        )
        median = float(np.median(values))
        ax.text(
            x_index,
            median,
            f" n={len(values)}",
            ha="left",
            va="center",
            fontsize=8,
            color="#1f2d3a",
            fontweight="bold",
        )

    ax.axhline(0, color="#6b7280", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_title(f"{trial_id}: 2s mean SLAM z by effective manual motion label")
    ax.set_xlabel("Effective manual motion label")
    ax.set_ylabel("2s window mean_z_m (raw MPS z, meters)")
    ax.grid(axis="y", color="#d7dde5", linewidth=0.8, alpha=0.7)
    fig.text(
        0.01,
        0.01,
        "Each point is one 2s synchronized SLAM window. Labels come from effective manual labels in the workbook.",
        fontsize=8,
        color="#3f4752",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_path = out_dir / "mean_z_by_motion_label_boxplot.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_speed_by_motion_label_boxplot(
    trial_id: str,
    out_dir: Path,
    window_df: pd.DataFrame,
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    required = {"window_s", "mean_linear_speed_mps", "speech_label"}
    if window_df.empty or not required.issubset(window_df.columns):
        return None

    two_s = window_df[np.isclose(window_df["window_s"], 2.0)].copy()
    if two_s.empty:
        return None

    two_s["action_label"] = two_s["speech_label"].map(normalize_action_label)
    label_order = ["sitting", "standing", "walking", "upstairs", "downstairs"]
    labels = [
        label
        for label in label_order
        if label in set(two_s["action_label"])
        and two_s.loc[two_s["action_label"] == label, "mean_linear_speed_mps"].notna().any()
    ]
    if not labels:
        return None

    data = [
        two_s.loc[two_s["action_label"] == label, "mean_linear_speed_mps"].dropna().to_numpy()
        for label in labels
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    box = ax.boxplot(
        data,
        tick_labels=[ACTION_DISPLAY_LABELS[label] for label in labels],
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#1f2d3a", "linewidth": 1.8},
        whiskerprops={"color": "#4c5661", "linewidth": 1.0},
        capprops={"color": "#4c5661", "linewidth": 1.0},
    )
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(ACTION_COLORS[label])
        patch.set_alpha(0.35)
        patch.set_edgecolor(ACTION_COLORS[label])

    rng = np.random.default_rng(21)
    for x_index, (values, label) in enumerate(zip(data, labels), start=1):
        jitter = rng.normal(0, 0.045, size=len(values))
        ax.scatter(
            np.full(len(values), x_index) + jitter,
            values,
            s=16,
            color=ACTION_COLORS[label],
            alpha=0.35,
            edgecolors="none",
        )
        median = float(np.median(values))
        ax.text(
            x_index,
            median,
            f" n={len(values)}",
            ha="left",
            va="center",
            fontsize=8,
            color="#1f2d3a",
            fontweight="bold",
        )

    ax.set_title(f"{trial_id}: 2s mean linear speed by effective manual motion label")
    ax.set_xlabel("Effective manual motion label")
    ax.set_ylabel("2s window mean_linear_speed_mps")
    ax.grid(axis="y", color="#d7dde5", linewidth=0.8, alpha=0.7)
    fig.text(
        0.01,
        0.01,
        "Each point is one 2s synchronized SLAM window. Speed is the mean of instantaneous velocity magnitude in that window.",
        fontsize=8,
        color="#3f4752",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_path = out_dir / "mean_linear_speed_by_motion_label_boxplot.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_window_feature_by_motion_label_boxplot(
    trial_id: str,
    out_dir: Path,
    window_df: pd.DataFrame,
    feature_col: str,
    y_label: str,
    title_feature: str,
    filename: str,
    caption: str,
    *,
    draw_zero_line: bool = False,
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    required = {"window_s", feature_col, "speech_label"}
    if window_df.empty or not required.issubset(window_df.columns):
        return None

    two_s = window_df[np.isclose(window_df["window_s"], 2.0)].copy()
    if two_s.empty:
        return None

    two_s["action_label"] = two_s["speech_label"].map(normalize_action_label)
    label_order = ["sitting", "standing", "walking", "upstairs", "downstairs"]
    labels = [
        label
        for label in label_order
        if label in set(two_s["action_label"])
        and two_s.loc[two_s["action_label"] == label, feature_col].notna().any()
    ]
    if not labels:
        return None

    data = [two_s.loc[two_s["action_label"] == label, feature_col].dropna().to_numpy() for label in labels]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    box = ax.boxplot(
        data,
        tick_labels=[ACTION_DISPLAY_LABELS[label] for label in labels],
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#1f2d3a", "linewidth": 1.8},
        whiskerprops={"color": "#4c5661", "linewidth": 1.0},
        capprops={"color": "#4c5661", "linewidth": 1.0},
    )
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(ACTION_COLORS[label])
        patch.set_alpha(0.35)
        patch.set_edgecolor(ACTION_COLORS[label])

    rng = np.random.default_rng(33)
    for x_index, (values, label) in enumerate(zip(data, labels), start=1):
        jitter = rng.normal(0, 0.045, size=len(values))
        ax.scatter(
            np.full(len(values), x_index) + jitter,
            values,
            s=16,
            color=ACTION_COLORS[label],
            alpha=0.35,
            edgecolors="none",
        )
        median = float(np.median(values))
        ax.text(
            x_index,
            median,
            f" n={len(values)}",
            ha="left",
            va="center",
            fontsize=8,
            color="#1f2d3a",
            fontweight="bold",
        )

    if draw_zero_line:
        ax.axhline(0, color="#6b7280", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_title(f"{trial_id}: 2s {title_feature} by effective manual motion label")
    ax.set_xlabel("Effective manual motion label")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", color="#d7dde5", linewidth=0.8, alpha=0.7)
    fig.text(0.01, 0.01, caption, fontsize=8, color="#3f4752")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_path = out_dir / filename
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_mean_z_vs_speed_scatter(
    trial_id: str,
    out_dir: Path,
    window_df: pd.DataFrame,
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    required = {"window_s", "mean_z_m", "mean_linear_speed_mps", "speech_label"}
    if window_df.empty or not required.issubset(window_df.columns):
        return None

    two_s = window_df[np.isclose(window_df["window_s"], 2.0)].copy()
    two_s["action_label"] = two_s["speech_label"].map(normalize_action_label)
    two_s = two_s[two_s["action_label"].isin(ACTION_DISPLAY_LABELS)]
    if two_s.empty:
        return None

    label_order = ["sitting", "standing", "walking", "upstairs", "downstairs"]
    labels = [label for label in label_order if label in set(two_s["action_label"])]

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    for label in labels:
        part = two_s[two_s["action_label"] == label]
        ax.scatter(
            part["mean_z_m"],
            part["mean_linear_speed_mps"],
            s=42,
            color=ACTION_COLORS[label],
            alpha=0.72,
            edgecolors="white",
            linewidths=0.45,
            label=ACTION_DISPLAY_LABELS[label],
        )

    ax.set_title(f"{trial_id}: 2s mean SLAM z vs mean linear speed")
    ax.set_xlabel("2s window mean_z_m (raw MPS z, meters)")
    ax.set_ylabel("2s window mean_linear_speed_mps")
    ax.grid(color="#d7dde5", linewidth=0.8, alpha=0.7)
    handles = [
        Patch(facecolor=ACTION_COLORS[label], alpha=0.72, label=ACTION_DISPLAY_LABELS[label])
        for label in labels
    ]
    ax.legend(handles=handles, title="Effective manual label", loc="best", frameon=True)
    fig.text(
        0.01,
        0.01,
        "Each point is one 2s synchronized SLAM window. This tests posture height vs movement intensity.",
        fontsize=8,
        color="#3f4752",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_path = out_dir / "mean_z_vs_mean_linear_speed_scatter.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_horizontal_displacement_vs_speed_scatter(
    trial_id: str,
    out_dir: Path,
    window_df: pd.DataFrame,
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    required = {"window_s", "horizontal_delta_m", "mean_linear_speed_mps", "speech_label"}
    if window_df.empty or not required.issubset(window_df.columns):
        return None

    two_s = window_df[np.isclose(window_df["window_s"], 2.0)].copy()
    two_s["action_label"] = two_s["speech_label"].map(normalize_action_label)
    two_s = two_s[two_s["action_label"].isin(ACTION_DISPLAY_LABELS)]
    if two_s.empty:
        return None

    label_order = ["sitting", "standing", "walking", "upstairs", "downstairs"]
    labels = [label for label in label_order if label in set(two_s["action_label"])]

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    for label in labels:
        part = two_s[two_s["action_label"] == label]
        ax.scatter(
            part["horizontal_delta_m"],
            part["mean_linear_speed_mps"],
            s=42,
            color=ACTION_COLORS[label],
            alpha=0.72,
            edgecolors="white",
            linewidths=0.45,
            label=ACTION_DISPLAY_LABELS[label],
        )

    ax.set_title(f"{trial_id}: 2s horizontal displacement vs mean linear speed")
    ax.set_xlabel("2s window horizontal_displacement_m = sqrt(dx^2 + dy^2)")
    ax.set_ylabel("2s window mean_linear_speed_mps")
    ax.grid(color="#d7dde5", linewidth=0.8, alpha=0.7)
    handles = [
        Patch(facecolor=ACTION_COLORS[label], alpha=0.72, label=ACTION_DISPLAY_LABELS[label])
        for label in labels
    ]
    ax.legend(handles=handles, title="Effective manual label", loc="best", frameon=True)
    fig.text(
        0.01,
        0.01,
        "Each point is one 2s synchronized SLAM window. Horizontal displacement is endpoint XY distance; speed is the mean of frame-level velocity magnitude.",
        fontsize=8,
        color="#3f4752",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_path = out_dir / "horizontal_displacement_vs_mean_linear_speed_scatter.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_horizontal_path_length_vs_speed_scatter(
    trial_id: str,
    out_dir: Path,
    window_df: pd.DataFrame,
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    required = {"window_s", "horizontal_path_length_m", "mean_linear_speed_mps", "speech_label"}
    if window_df.empty or not required.issubset(window_df.columns):
        return None

    two_s = window_df[np.isclose(window_df["window_s"], 2.0)].copy()
    two_s["action_label"] = two_s["speech_label"].map(normalize_action_label)
    two_s = two_s[two_s["action_label"].isin(ACTION_DISPLAY_LABELS)]
    if two_s.empty:
        return None

    label_order = ["sitting", "standing", "walking", "upstairs", "downstairs"]
    labels = [label for label in label_order if label in set(two_s["action_label"])]

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    for label in labels:
        part = two_s[two_s["action_label"] == label]
        ax.scatter(
            part["horizontal_path_length_m"],
            part["mean_linear_speed_mps"],
            s=42,
            color=ACTION_COLORS[label],
            alpha=0.72,
            edgecolors="white",
            linewidths=0.45,
            label=ACTION_DISPLAY_LABELS[label],
        )

    ax.set_title(f"{trial_id}: 2s horizontal path length vs mean linear speed")
    ax.set_xlabel("2s window horizontal_path_length_m = sum of frame-to-frame XY movement")
    ax.set_ylabel("2s window mean_linear_speed_mps")
    ax.grid(color="#d7dde5", linewidth=0.8, alpha=0.7)
    handles = [
        Patch(facecolor=ACTION_COLORS[label], alpha=0.72, label=ACTION_DISPLAY_LABELS[label])
        for label in labels
    ]
    ax.legend(handles=handles, title="Effective manual label", loc="best", frameon=True)
    fig.text(
        0.01,
        0.01,
        "Each point is one 2s synchronized SLAM window. Path length accumulates frame-to-frame XY movement, so it captures motion that endpoint displacement can miss.",
        fontsize=8,
        color="#3f4752",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_path = out_dir / "horizontal_path_length_vs_mean_linear_speed_scatter.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_vertical_displacement_vs_speed_scatter(
    trial_id: str,
    out_dir: Path,
    window_df: pd.DataFrame,
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    required = {"window_s", "vertical_delta_m", "mean_linear_speed_mps", "speech_label"}
    if window_df.empty or not required.issubset(window_df.columns):
        return None

    two_s = window_df[np.isclose(window_df["window_s"], 2.0)].copy()
    two_s["action_label"] = two_s["speech_label"].map(normalize_action_label)
    two_s = two_s[two_s["action_label"].isin(ACTION_DISPLAY_LABELS)]
    if two_s.empty:
        return None

    label_order = ["sitting", "standing", "walking", "upstairs", "downstairs"]
    labels = [label for label in label_order if label in set(two_s["action_label"])]

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    for label in labels:
        part = two_s[two_s["action_label"] == label]
        ax.scatter(
            part["vertical_delta_m"],
            part["mean_linear_speed_mps"],
            s=42,
            color=ACTION_COLORS[label],
            alpha=0.72,
            edgecolors="white",
            linewidths=0.45,
            label=ACTION_DISPLAY_LABELS[label],
        )

    ax.axvline(0.0, color="#4b5563", linewidth=1.0, alpha=0.65)
    ax.set_title(f"{trial_id}: 2s vertical displacement vs mean linear speed")
    ax.set_xlabel("2s window vertical_displacement_m = z_end - z_start")
    ax.set_ylabel("2s window mean_linear_speed_mps")
    ax.grid(color="#d7dde5", linewidth=0.8, alpha=0.7)
    handles = [
        Patch(facecolor=ACTION_COLORS[label], alpha=0.72, label=ACTION_DISPLAY_LABELS[label])
        for label in labels
    ]
    ax.legend(handles=handles, title="Effective manual label", loc="best", frameon=True)
    fig.text(
        0.01,
        0.01,
        "Each point is one 2s synchronized SLAM window. Positive vertical displacement means upward net z movement; negative means downward.",
        fontsize=8,
        color="#3f4752",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_path = out_dir / "vertical_displacement_vs_mean_linear_speed_scatter.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_vertical_change_vs_path_length_scatters(
    trial_id: str,
    out_dir: Path,
    window_df: pd.DataFrame,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    required = {
        "window_s",
        "horizontal_path_length_m",
        "vertical_delta_m",
        "height_slope_mps",
        "speech_label",
    }
    if window_df.empty or not required.issubset(window_df.columns):
        return []

    two_s = window_df[np.isclose(window_df["window_s"], 2.0)].copy()
    two_s["action_label"] = two_s["speech_label"].map(normalize_action_label)
    two_s = two_s[two_s["action_label"].isin(ACTION_DISPLAY_LABELS)]
    if two_s.empty:
        return []

    label_order = ["sitting", "standing", "walking", "upstairs", "downstairs"]
    labels = [label for label in label_order if label in set(two_s["action_label"])]
    handles = [
        Patch(facecolor=ACTION_COLORS[label], alpha=0.72, label=ACTION_DISPLAY_LABELS[label])
        for label in labels
    ]
    specs = [
        (
            "vertical_delta_m",
            "2s window vertical_delta_m (z_end - z_start)",
            f"{trial_id}: 2s vertical delta vs horizontal path length",
            "vertical_delta_vs_horizontal_path_length_scatter.png",
            "This tests net height change against actual XY path traveled inside the 2s window.",
        ),
        (
            "height_slope_mps",
            "2s window height_slope_mps = vertical_delta_m / window duration",
            f"{trial_id}: 2s height slope vs horizontal path length",
            "height_slope_vs_horizontal_path_length_scatter.png",
            "This normalizes net height change by time, helping compare upward/downward movement rate.",
        ),
    ]

    outputs: list[Path] = []
    for y_col, y_label, title, filename, caption in specs:
        fig, ax = plt.subplots(figsize=(8.8, 6.2))
        for label in labels:
            part = two_s[two_s["action_label"] == label]
            ax.scatter(
                part["horizontal_path_length_m"],
                part[y_col],
                s=42,
                color=ACTION_COLORS[label],
                alpha=0.72,
                edgecolors="white",
                linewidths=0.45,
                label=ACTION_DISPLAY_LABELS[label],
            )

        ax.axhline(0.0, color="#4b5563", linewidth=1.0, alpha=0.65)
        ax.set_title(title)
        ax.set_xlabel("2s window horizontal_path_length_m = sum of frame-to-frame XY movement")
        ax.set_ylabel(y_label)
        ax.grid(color="#d7dde5", linewidth=0.8, alpha=0.7)
        ax.legend(handles=handles, title="Effective manual label", loc="best", frameon=True)
        fig.text(
            0.01,
            0.01,
            f"Each point is one 2s synchronized SLAM window. {caption}",
            fontsize=8,
            color="#3f4752",
        )
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        output_path = out_dir / filename
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        outputs.append(output_path)

    return outputs


def plot_instantaneous_speed_over_time(
    trial_id: str,
    out_dir: Path,
    traj: pd.DataFrame,
    window_df: pd.DataFrame,
    states: pd.DataFrame,
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    required = {"session_s", "linear_speed_mps"}
    if traj.empty or not required.issubset(traj.columns):
        return None

    work = traj[["session_s", "linear_speed_mps"]].dropna().copy()
    if work.empty:
        return None

    dt = float(work["session_s"].diff().median()) if len(work) > 1 else 0.001
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.001
    rolling_0_5_count = max(1, int(round(0.5 / dt)))
    rolling_2_count = max(1, int(round(2.0 / dt)))
    work["rolling_mean_speed_0_5s"] = work["linear_speed_mps"].rolling(
        rolling_0_5_count, center=True, min_periods=1
    ).mean()
    work["rolling_mean_speed_2s"] = work["linear_speed_mps"].rolling(
        rolling_2_count, center=True, min_periods=1
    ).mean()

    # Draw a thinned raw trace so the plot remains readable while still showing spikes.
    raw_stride = max(1, len(work) // 18000)
    raw = work.iloc[::raw_stride]

    fig, ax = plt.subplots(figsize=(13, 5))
    used_labels = style_state_spans(ax, states)
    ax.plot(
        raw["session_s"],
        raw["linear_speed_mps"],
        color="#6b7280",
        linewidth=0.35,
        alpha=0.3,
        label="instantaneous speed, raw samples",
    )
    ax.plot(
        work["session_s"],
        work["rolling_mean_speed_0_5s"],
        color="#0f766e",
        linewidth=1.15,
        alpha=0.9,
        label="0.5s rolling mean speed",
    )
    ax.plot(
        work["session_s"],
        work["rolling_mean_speed_2s"],
        color="#1f2d3a",
        linewidth=1.25,
        alpha=0.9,
        label="2s rolling mean speed",
    )

    two_s = window_df[np.isclose(window_df["window_s"], 2.0)].copy() if not window_df.empty else pd.DataFrame()
    if not two_s.empty and "mean_linear_speed_mps" in two_s.columns:
        ax.scatter(
            two_s["center_s"],
            two_s["mean_linear_speed_mps"],
            s=20,
            color="#d36b42",
            alpha=0.75,
            label="old 2s window mean speed",
            zorder=4,
        )

    ax.set_title(f"{trial_id}: instantaneous linear speed vs rolling/window speed")
    ax.set_xlabel("session time (s)")
    ax.set_ylabel("linear speed magnitude (m/s)")
    ax.grid(axis="y", color="#d7dde5", linewidth=0.8, alpha=0.65)
    add_action_color_legend(ax, used_labels)
    ax.legend(loc="upper right", title="Speed values", fontsize=8, title_fontsize=8)
    fig.text(
        0.01,
        0.01,
        "Instantaneous speed = sqrt(vx^2 + vy^2 + vz^2) at each SLAM sample. Rolling/window means smooth this raw signal.",
        fontsize=8,
        color="#3f4752",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_path = out_dir / "instantaneous_linear_speed_over_time.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_height_and_speed_over_time(
    trial_id: str,
    out_dir: Path,
    traj: pd.DataFrame,
    states: pd.DataFrame,
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    required = {"session_s", "tz_world_device", "linear_speed_mps"}
    if traj.empty or not required.issubset(traj.columns):
        return None

    work = traj[["session_s", "tz_world_device", "linear_speed_mps"]].dropna().copy()
    if work.empty:
        return None

    dt = float(work["session_s"].diff().median()) if len(work) > 1 else 0.001
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.001
    rolling_0_5_count = max(1, int(round(0.5 / dt)))
    work["rolling_mean_speed_0_5s"] = work["linear_speed_mps"].rolling(
        rolling_0_5_count, center=True, min_periods=1
    ).mean()

    height_stride = max(1, len(work) // 18000)
    height_sample = work.iloc[::height_stride]

    fig, ax_height = plt.subplots(figsize=(13, 5.4))
    used_labels = style_state_spans(ax_height, states)
    height_line = ax_height.plot(
        height_sample["session_s"],
        height_sample["tz_world_device"],
        color="#1f2d3a",
        linewidth=1.05,
        alpha=0.9,
        label="raw SLAM height, tz_world_device",
    )
    ax_height.set_xlabel("session time (s)")
    ax_height.set_ylabel("raw SLAM height, tz_world_device (m)", color="#1f2d3a")
    ax_height.tick_params(axis="y", labelcolor="#1f2d3a")
    ax_height.grid(axis="y", color="#d7dde5", linewidth=0.8, alpha=0.65)
    add_action_color_legend(ax_height, used_labels)

    ax_speed = ax_height.twinx()
    speed_line = ax_speed.plot(
        work["session_s"],
        work["rolling_mean_speed_0_5s"],
        color="#d36b42",
        linewidth=1.0,
        alpha=0.85,
        label="0.5s rolling mean speed",
    )
    ax_speed.set_ylabel("linear speed magnitude (m/s)", color="#a84f24")
    ax_speed.tick_params(axis="y", labelcolor="#a84f24")

    lines = height_line + speed_line
    labels = [line.get_label() for line in lines]
    ax_height.legend(lines, labels, loc="upper right", title="Plotted values", fontsize=8, title_fontsize=8)
    ax_height.set_title(f"{trial_id}: height and speed over time")
    fig.text(
        0.01,
        0.01,
        "Height is raw MPS z. Speed is a centered 0.5s rolling mean of instantaneous SLAM velocity magnitude. Background = effective manual labels.",
        fontsize=8,
        color="#3f4752",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_path = out_dir / "height_and_speed_over_time.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_instantaneous_z_vs_speed_scatter(
    trial_id: str,
    out_dir: Path,
    traj: pd.DataFrame,
    states: pd.DataFrame,
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    required = {"session_s", "tz_world_device", "linear_speed_mps"}
    if traj.empty or not required.issubset(traj.columns):
        return None

    stride = max(1, len(traj) // 7000)
    sample = traj.iloc[::stride][["session_s", "tz_world_device", "linear_speed_mps"]].dropna().copy()
    if sample.empty:
        return None
    sample = annotate_with_labels(sample, states)
    sample["action_label"] = sample["speech_label"].map(normalize_action_label)
    sample = sample[sample["action_label"].isin(ACTION_DISPLAY_LABELS)]
    if sample.empty:
        return None

    label_order = ["sitting", "standing", "walking", "upstairs", "downstairs"]
    labels = [label for label in label_order if label in set(sample["action_label"])]

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    for label in labels:
        part = sample[sample["action_label"] == label]
        ax.scatter(
            part["tz_world_device"],
            part["linear_speed_mps"],
            s=12,
            color=ACTION_COLORS[label],
            alpha=0.48,
            edgecolors="none",
        )

    handles = [
        Patch(facecolor=ACTION_COLORS[label], alpha=0.72, label=ACTION_DISPLAY_LABELS[label])
        for label in labels
    ]
    ax.legend(handles=handles, title="Effective manual label", loc="best", frameon=True)
    ax.set_title(f"{trial_id}: raw SLAM z vs instantaneous linear speed")
    ax.set_xlabel("raw tz_world_device (meters)")
    ax.set_ylabel("instantaneous linear speed magnitude (m/s)")
    ax.grid(color="#d7dde5", linewidth=0.8, alpha=0.7)
    fig.text(
        0.01,
        0.01,
        "Each point is a sampled raw SLAM timestamp, not a 2s window. Color = effective manual label.",
        fontsize=8,
        color="#3f4752",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_path = out_dir / "instantaneous_z_vs_speed_scatter.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_combined_mean_z_vs_speed_scatter(trial_ids: list[str]) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows: list[pd.DataFrame] = []
    for trial_id in trial_ids:
        path = RUNS_DIR / trial_id / "raw_parameter_diagnostics" / "sliding_window_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        required = {"window_s", "mean_z_m", "mean_linear_speed_mps", "speech_label"}
        if df.empty or not required.issubset(df.columns):
            continue
        df = df[np.isclose(df["window_s"], 2.0)].copy()
        df["trial_id"] = trial_id
        df["action_label"] = df["speech_label"].map(normalize_action_label)
        df = df[df["action_label"].isin(ACTION_DISPLAY_LABELS)]
        rows.append(df)

    if not rows:
        return None

    combined = pd.concat(rows, ignore_index=True)
    label_order = ["sitting", "standing", "walking", "upstairs", "downstairs"]
    labels = [label for label in label_order if label in set(combined["action_label"])]
    if not labels:
        return None

    output_dir = RUNS_DIR / "raw_parameter_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    markers = {"Ntrial1": "o", "Ntrial2": "s", "Ntrial3": "^"}
    for label in labels:
        for trial_id in trial_ids:
            part = combined[(combined["action_label"] == label) & (combined["trial_id"] == trial_id)]
            if part.empty:
                continue
            ax.scatter(
                part["mean_z_m"],
                part["mean_linear_speed_mps"],
                s=38,
                marker=markers.get(trial_id, "o"),
                color=ACTION_COLORS[label],
                alpha=0.58,
                edgecolors="white",
                linewidths=0.4,
            )

    action_handles = [
        Patch(facecolor=ACTION_COLORS[label], alpha=0.72, label=ACTION_DISPLAY_LABELS[label])
        for label in labels
    ]
    action_legend = ax.legend(
        handles=action_handles,
        title="Effective manual label",
        loc="upper left",
        frameon=True,
    )
    ax.add_artist(action_legend)

    from matplotlib.lines import Line2D

    trial_handles = [
        Line2D(
            [0],
            [0],
            marker=markers.get(trial_id, "o"),
            color="none",
            markerfacecolor="#4b5563",
            markeredgecolor="white",
            markersize=8,
            label=trial_id,
        )
        for trial_id in trial_ids
    ]
    ax.legend(handles=trial_handles, title="Trial marker", loc="upper right", frameon=True)

    ax.set_title("Ntrial1-3 combined: 2s mean SLAM z vs mean linear speed")
    ax.set_xlabel("2s window mean_z_m (raw MPS z, meters)")
    ax.set_ylabel("2s window mean_linear_speed_mps")
    ax.grid(color="#d7dde5", linewidth=0.8, alpha=0.7)
    fig.text(
        0.01,
        0.01,
        "Raw MPS z is not re-originated, so cross-trial height offsets remain visible. Color = manual label; marker = trial.",
        fontsize=8,
        color="#3f4752",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_path = output_dir / "mean_z_vs_mean_linear_speed_scatter_Ntrial1_Ntrial2_Ntrial3.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_combined_horizontal_displacement_vs_speed_scatter(trial_ids: list[str]) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows: list[pd.DataFrame] = []
    for trial_id in trial_ids:
        path = RUNS_DIR / trial_id / "raw_parameter_diagnostics" / "sliding_window_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        required = {"window_s", "horizontal_delta_m", "mean_linear_speed_mps", "speech_label"}
        if df.empty or not required.issubset(df.columns):
            continue
        df = df[np.isclose(df["window_s"], 2.0)].copy()
        df["trial_id"] = trial_id
        df["action_label"] = df["speech_label"].map(normalize_action_label)
        df = df[df["action_label"].isin(ACTION_DISPLAY_LABELS)]
        rows.append(df)

    if not rows:
        return None

    combined = pd.concat(rows, ignore_index=True)
    label_order = ["sitting", "standing", "walking", "upstairs", "downstairs"]
    labels = [label for label in label_order if label in set(combined["action_label"])]
    if not labels:
        return None

    output_dir = RUNS_DIR / "raw_parameter_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    markers = {"Ntrial1": "o", "Ntrial2": "s", "Ntrial3": "^"}
    for label in labels:
        for trial_id in trial_ids:
            part = combined[(combined["action_label"] == label) & (combined["trial_id"] == trial_id)]
            if part.empty:
                continue
            ax.scatter(
                part["horizontal_delta_m"],
                part["mean_linear_speed_mps"],
                s=38,
                marker=markers.get(trial_id, "o"),
                color=ACTION_COLORS[label],
                alpha=0.58,
                edgecolors="white",
                linewidths=0.4,
            )

    action_handles = [
        Patch(facecolor=ACTION_COLORS[label], alpha=0.72, label=ACTION_DISPLAY_LABELS[label])
        for label in labels
    ]
    action_legend = ax.legend(
        handles=action_handles,
        title="Effective manual label",
        loc="upper left",
        frameon=True,
    )
    ax.add_artist(action_legend)

    from matplotlib.lines import Line2D

    trial_handles = [
        Line2D(
            [0],
            [0],
            marker=markers.get(trial_id, "o"),
            color="none",
            markerfacecolor="#4b5563",
            markeredgecolor="white",
            markersize=8,
            label=trial_id,
        )
        for trial_id in trial_ids
    ]
    ax.legend(handles=trial_handles, title="Trial marker", loc="upper right", frameon=True)

    ax.set_title("Ntrial1-3 combined: 2s horizontal displacement vs mean linear speed")
    ax.set_xlabel("2s window horizontal_displacement_m = sqrt(dx^2 + dy^2)")
    ax.set_ylabel("2s window mean_linear_speed_mps")
    ax.grid(color="#d7dde5", linewidth=0.8, alpha=0.7)
    fig.text(
        0.01,
        0.01,
        "Color = effective manual label; marker = trial. Endpoint displacement and mean speed are related but not identical.",
        fontsize=8,
        color="#3f4752",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_path = output_dir / "horizontal_displacement_vs_mean_linear_speed_scatter_Ntrial1_Ntrial2_Ntrial3.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_combined_horizontal_path_length_vs_speed_scatter(trial_ids: list[str]) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows: list[pd.DataFrame] = []
    for trial_id in trial_ids:
        path = RUNS_DIR / trial_id / "raw_parameter_diagnostics" / "sliding_window_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        required = {"window_s", "horizontal_path_length_m", "mean_linear_speed_mps", "speech_label"}
        if df.empty or not required.issubset(df.columns):
            continue
        df = df[np.isclose(df["window_s"], 2.0)].copy()
        df["trial_id"] = trial_id
        df["action_label"] = df["speech_label"].map(normalize_action_label)
        df = df[df["action_label"].isin(ACTION_DISPLAY_LABELS)]
        rows.append(df)

    if not rows:
        return None

    combined = pd.concat(rows, ignore_index=True)
    label_order = ["sitting", "standing", "walking", "upstairs", "downstairs"]
    labels = [label for label in label_order if label in set(combined["action_label"])]
    if not labels:
        return None

    output_dir = RUNS_DIR / "raw_parameter_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    markers = {"Ntrial1": "o", "Ntrial2": "s", "Ntrial3": "^"}
    for label in labels:
        for trial_id in trial_ids:
            part = combined[(combined["action_label"] == label) & (combined["trial_id"] == trial_id)]
            if part.empty:
                continue
            ax.scatter(
                part["horizontal_path_length_m"],
                part["mean_linear_speed_mps"],
                s=38,
                marker=markers.get(trial_id, "o"),
                color=ACTION_COLORS[label],
                alpha=0.58,
                edgecolors="white",
                linewidths=0.4,
            )

    action_handles = [
        Patch(facecolor=ACTION_COLORS[label], alpha=0.72, label=ACTION_DISPLAY_LABELS[label])
        for label in labels
    ]
    action_legend = ax.legend(
        handles=action_handles,
        title="Effective manual label",
        loc="upper left",
        frameon=True,
    )
    ax.add_artist(action_legend)

    from matplotlib.lines import Line2D

    trial_handles = [
        Line2D(
            [0],
            [0],
            marker=markers.get(trial_id, "o"),
            color="none",
            markerfacecolor="#4b5563",
            markeredgecolor="white",
            markersize=8,
            label=trial_id,
        )
        for trial_id in trial_ids
    ]
    ax.legend(handles=trial_handles, title="Trial marker", loc="upper right", frameon=True)

    ax.set_title("Ntrial1-3 combined: 2s horizontal path length vs mean linear speed")
    ax.set_xlabel("2s window horizontal_path_length_m = sum of frame-to-frame XY movement")
    ax.set_ylabel("2s window mean_linear_speed_mps")
    ax.grid(color="#d7dde5", linewidth=0.8, alpha=0.7)
    fig.text(
        0.01,
        0.01,
        "Color = effective manual label; marker = trial. Path length is usually more informative than endpoint displacement for total movement amount.",
        fontsize=8,
        color="#3f4752",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_path = output_dir / "horizontal_path_length_vs_mean_linear_speed_scatter_Ntrial1_Ntrial2_Ntrial3.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_combined_vertical_displacement_vs_speed_scatter(trial_ids: list[str]) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows: list[pd.DataFrame] = []
    for trial_id in trial_ids:
        path = RUNS_DIR / trial_id / "raw_parameter_diagnostics" / "sliding_window_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        required = {"window_s", "vertical_delta_m", "mean_linear_speed_mps", "speech_label"}
        if df.empty or not required.issubset(df.columns):
            continue
        df = df[np.isclose(df["window_s"], 2.0)].copy()
        df["trial_id"] = trial_id
        df["action_label"] = df["speech_label"].map(normalize_action_label)
        df = df[df["action_label"].isin(ACTION_DISPLAY_LABELS)]
        rows.append(df)

    if not rows:
        return None

    combined = pd.concat(rows, ignore_index=True)
    label_order = ["sitting", "standing", "walking", "upstairs", "downstairs"]
    labels = [label for label in label_order if label in set(combined["action_label"])]
    if not labels:
        return None

    output_dir = RUNS_DIR / "raw_parameter_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    markers = {"Ntrial1": "o", "Ntrial2": "s", "Ntrial3": "^"}
    for label in labels:
        for trial_id in trial_ids:
            part = combined[(combined["action_label"] == label) & (combined["trial_id"] == trial_id)]
            if part.empty:
                continue
            ax.scatter(
                part["vertical_delta_m"],
                part["mean_linear_speed_mps"],
                s=38,
                marker=markers.get(trial_id, "o"),
                color=ACTION_COLORS[label],
                alpha=0.58,
                edgecolors="white",
                linewidths=0.4,
            )

    action_handles = [
        Patch(facecolor=ACTION_COLORS[label], alpha=0.72, label=ACTION_DISPLAY_LABELS[label])
        for label in labels
    ]
    action_legend = ax.legend(
        handles=action_handles,
        title="Effective manual label",
        loc="upper left",
        frameon=True,
    )
    ax.add_artist(action_legend)

    from matplotlib.lines import Line2D

    trial_handles = [
        Line2D(
            [0],
            [0],
            marker=markers.get(trial_id, "o"),
            color="none",
            markerfacecolor="#4b5563",
            markeredgecolor="white",
            markersize=8,
            label=trial_id,
        )
        for trial_id in trial_ids
    ]
    ax.legend(handles=trial_handles, title="Trial marker", loc="upper right", frameon=True)

    ax.axvline(0.0, color="#4b5563", linewidth=1.0, alpha=0.65)
    ax.set_title("Ntrial1-3 combined: 2s vertical displacement vs mean linear speed")
    ax.set_xlabel("2s window vertical_displacement_m = z_end - z_start")
    ax.set_ylabel("2s window mean_linear_speed_mps")
    ax.grid(color="#d7dde5", linewidth=0.8, alpha=0.7)
    fig.text(
        0.01,
        0.01,
        "Color = effective manual label; marker = trial. Positive x values indicate upward net z movement; negative values indicate downward.",
        fontsize=8,
        color="#3f4752",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_path = output_dir / "vertical_displacement_vs_mean_linear_speed_scatter_Ntrial1_Ntrial2_Ntrial3.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_combined_vertical_change_vs_path_length_scatters(trial_ids: list[str]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows: list[pd.DataFrame] = []
    for trial_id in trial_ids:
        path = RUNS_DIR / trial_id / "raw_parameter_diagnostics" / "sliding_window_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        required = {
            "window_s",
            "horizontal_path_length_m",
            "vertical_delta_m",
            "height_slope_mps",
            "speech_label",
        }
        if df.empty or not required.issubset(df.columns):
            continue
        df = df[np.isclose(df["window_s"], 2.0)].copy()
        df["trial_id"] = trial_id
        df["action_label"] = df["speech_label"].map(normalize_action_label)
        df = df[df["action_label"].isin(ACTION_DISPLAY_LABELS)]
        rows.append(df)

    if not rows:
        return []

    combined = pd.concat(rows, ignore_index=True)
    label_order = ["sitting", "standing", "walking", "upstairs", "downstairs"]
    labels = [label for label in label_order if label in set(combined["action_label"])]
    if not labels:
        return []

    output_dir = RUNS_DIR / "raw_parameter_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    markers = {"Ntrial1": "o", "Ntrial2": "s", "Ntrial3": "^"}
    specs = [
        (
            "vertical_delta_m",
            "2s window vertical_delta_m (z_end - z_start)",
            "Ntrial1-3 combined: 2s vertical delta vs horizontal path length",
            "vertical_delta_vs_horizontal_path_length_scatter_Ntrial1_Ntrial2_Ntrial3.png",
            "Color = effective manual label; marker = trial. Positive values indicate upward net z movement; negative values indicate downward net z movement.",
        ),
        (
            "height_slope_mps",
            "2s window height_slope_mps = vertical_delta_m / window duration",
            "Ntrial1-3 combined: 2s height slope vs horizontal path length",
            "height_slope_vs_horizontal_path_length_scatter_Ntrial1_Ntrial2_Ntrial3.png",
            "Color = effective manual label; marker = trial. Height slope is the time-normalized version of vertical delta.",
        ),
    ]

    from matplotlib.lines import Line2D

    outputs: list[Path] = []
    for y_col, y_label, title, filename, caption in specs:
        fig, ax = plt.subplots(figsize=(8.8, 6.2))
        for label in labels:
            for trial_id in trial_ids:
                part = combined[(combined["action_label"] == label) & (combined["trial_id"] == trial_id)]
                if part.empty:
                    continue
                ax.scatter(
                    part["horizontal_path_length_m"],
                    part[y_col],
                    s=38,
                    marker=markers.get(trial_id, "o"),
                    color=ACTION_COLORS[label],
                    alpha=0.58,
                    edgecolors="white",
                    linewidths=0.4,
                )

        action_handles = [
            Patch(facecolor=ACTION_COLORS[label], alpha=0.72, label=ACTION_DISPLAY_LABELS[label])
            for label in labels
        ]
        action_legend = ax.legend(
            handles=action_handles,
            title="Effective manual label",
            loc="upper left",
            frameon=True,
        )
        ax.add_artist(action_legend)

        trial_handles = [
            Line2D(
                [0],
                [0],
                marker=markers.get(trial_id, "o"),
                color="none",
                markerfacecolor="#4b5563",
                markeredgecolor="white",
                markersize=8,
                label=trial_id,
            )
            for trial_id in trial_ids
        ]
        ax.legend(handles=trial_handles, title="Trial marker", loc="upper right", frameon=True)

        ax.axhline(0.0, color="#4b5563", linewidth=1.0, alpha=0.65)
        ax.set_title(title)
        ax.set_xlabel("2s window horizontal_path_length_m = sum of frame-to-frame XY movement")
        ax.set_ylabel(y_label)
        ax.grid(color="#d7dde5", linewidth=0.8, alpha=0.7)
        fig.text(0.01, 0.01, caption, fontsize=8, color="#3f4752")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        outputs.append(output_path)

    return outputs


def plot_combined_mean_z_by_motion_label_boxplot(trial_ids: list[str]) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[pd.DataFrame] = []
    for trial_id in trial_ids:
        path = RUNS_DIR / trial_id / "raw_parameter_diagnostics" / "sliding_window_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty or "mean_z_m" not in df.columns or "speech_label" not in df.columns:
            continue
        df = df[np.isclose(df["window_s"], 2.0)].copy()
        df["trial_id"] = trial_id
        df["action_label"] = df["speech_label"].map(normalize_action_label)
        rows.append(df)

    if not rows:
        return None

    combined = pd.concat(rows, ignore_index=True)
    label_order = ["sitting", "standing", "walking", "upstairs", "downstairs"]
    labels = [
        label
        for label in label_order
        if label in set(combined["action_label"])
        and combined.loc[combined["action_label"] == label, "mean_z_m"].notna().any()
    ]
    if not labels:
        return None

    data = [combined.loc[combined["action_label"] == label, "mean_z_m"].dropna().to_numpy() for label in labels]
    output_dir = RUNS_DIR / "raw_parameter_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    box = ax.boxplot(
        data,
        tick_labels=[ACTION_DISPLAY_LABELS[label] for label in labels],
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#1f2d3a", "linewidth": 1.8},
        whiskerprops={"color": "#4c5661", "linewidth": 1.0},
        capprops={"color": "#4c5661", "linewidth": 1.0},
    )
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(ACTION_COLORS[label])
        patch.set_alpha(0.35)
        patch.set_edgecolor(ACTION_COLORS[label])

    rng = np.random.default_rng(17)
    for x_index, (values, label) in enumerate(zip(data, labels), start=1):
        jitter = rng.normal(0, 0.045, size=len(values))
        ax.scatter(
            np.full(len(values), x_index) + jitter,
            values,
            s=14,
            color=ACTION_COLORS[label],
            alpha=0.3,
            edgecolors="none",
        )
        median = float(np.median(values))
        ax.text(x_index, median, f" n={len(values)}", ha="left", va="center", fontsize=8, color="#1f2d3a")

    ax.axhline(0, color="#6b7280", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_title("Ntrial1-3 combined: 2s mean SLAM z by effective manual motion label")
    ax.set_xlabel("Effective manual motion label")
    ax.set_ylabel("2s window mean_z_m (raw MPS z, meters)")
    ax.grid(axis="y", color="#d7dde5", linewidth=0.8, alpha=0.7)
    fig.text(
        0.01,
        0.01,
        "Combined raw MPS z values are not re-originated, so cross-trial origin differences can widen boxes.",
        fontsize=8,
        color="#3f4752",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_path = output_dir / "mean_z_by_motion_label_boxplot_Ntrial1_Ntrial2_Ntrial3.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_combined_speed_by_motion_label_boxplot(trial_ids: list[str]) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[pd.DataFrame] = []
    for trial_id in trial_ids:
        path = RUNS_DIR / trial_id / "raw_parameter_diagnostics" / "sliding_window_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        required = {"window_s", "mean_linear_speed_mps", "speech_label"}
        if df.empty or not required.issubset(df.columns):
            continue
        df = df[np.isclose(df["window_s"], 2.0)].copy()
        df["trial_id"] = trial_id
        df["action_label"] = df["speech_label"].map(normalize_action_label)
        rows.append(df)

    if not rows:
        return None

    combined = pd.concat(rows, ignore_index=True)
    label_order = ["sitting", "standing", "walking", "upstairs", "downstairs"]
    labels = [
        label
        for label in label_order
        if label in set(combined["action_label"])
        and combined.loc[combined["action_label"] == label, "mean_linear_speed_mps"].notna().any()
    ]
    if not labels:
        return None

    data = [
        combined.loc[combined["action_label"] == label, "mean_linear_speed_mps"].dropna().to_numpy()
        for label in labels
    ]
    output_dir = RUNS_DIR / "raw_parameter_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    box = ax.boxplot(
        data,
        tick_labels=[ACTION_DISPLAY_LABELS[label] for label in labels],
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#1f2d3a", "linewidth": 1.8},
        whiskerprops={"color": "#4c5661", "linewidth": 1.0},
        capprops={"color": "#4c5661", "linewidth": 1.0},
    )
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(ACTION_COLORS[label])
        patch.set_alpha(0.35)
        patch.set_edgecolor(ACTION_COLORS[label])

    rng = np.random.default_rng(23)
    for x_index, (values, label) in enumerate(zip(data, labels), start=1):
        jitter = rng.normal(0, 0.045, size=len(values))
        ax.scatter(
            np.full(len(values), x_index) + jitter,
            values,
            s=14,
            color=ACTION_COLORS[label],
            alpha=0.3,
            edgecolors="none",
        )
        median = float(np.median(values))
        ax.text(x_index, median, f" n={len(values)}", ha="left", va="center", fontsize=8, color="#1f2d3a")

    ax.set_title("Ntrial1-3 combined: 2s mean linear speed by effective manual motion label")
    ax.set_xlabel("Effective manual motion label")
    ax.set_ylabel("2s window mean_linear_speed_mps")
    ax.grid(axis="y", color="#d7dde5", linewidth=0.8, alpha=0.7)
    fig.text(
        0.01,
        0.01,
        "This plot tests whether speed separates moving actions from sitting/standing across Ntrial1-3.",
        fontsize=8,
        color="#3f4752",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_path = output_dir / "mean_linear_speed_by_motion_label_boxplot_Ntrial1_Ntrial2_Ntrial3.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_combined_window_feature_by_motion_label_boxplot(
    trial_ids: list[str],
    feature_col: str,
    y_label: str,
    title_feature: str,
    filename: str,
    caption: str,
    *,
    draw_zero_line: bool = False,
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[pd.DataFrame] = []
    for trial_id in trial_ids:
        path = RUNS_DIR / trial_id / "raw_parameter_diagnostics" / "sliding_window_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        required = {"window_s", feature_col, "speech_label"}
        if df.empty or not required.issubset(df.columns):
            continue
        df = df[np.isclose(df["window_s"], 2.0)].copy()
        df["trial_id"] = trial_id
        df["action_label"] = df["speech_label"].map(normalize_action_label)
        rows.append(df)

    if not rows:
        return None

    combined = pd.concat(rows, ignore_index=True)
    label_order = ["sitting", "standing", "walking", "upstairs", "downstairs"]
    labels = [
        label
        for label in label_order
        if label in set(combined["action_label"])
        and combined.loc[combined["action_label"] == label, feature_col].notna().any()
    ]
    if not labels:
        return None

    data = [combined.loc[combined["action_label"] == label, feature_col].dropna().to_numpy() for label in labels]
    output_dir = RUNS_DIR / "raw_parameter_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    box = ax.boxplot(
        data,
        tick_labels=[ACTION_DISPLAY_LABELS[label] for label in labels],
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#1f2d3a", "linewidth": 1.8},
        whiskerprops={"color": "#4c5661", "linewidth": 1.0},
        capprops={"color": "#4c5661", "linewidth": 1.0},
    )
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(ACTION_COLORS[label])
        patch.set_alpha(0.35)
        patch.set_edgecolor(ACTION_COLORS[label])

    rng = np.random.default_rng(57)
    for x_index, (values, label) in enumerate(zip(data, labels), start=1):
        jitter = rng.normal(0, 0.045, size=len(values))
        ax.scatter(
            np.full(len(values), x_index) + jitter,
            values,
            s=14,
            color=ACTION_COLORS[label],
            alpha=0.3,
            edgecolors="none",
        )
        median = float(np.median(values))
        ax.text(x_index, median, f" n={len(values)}", ha="left", va="center", fontsize=8, color="#1f2d3a")

    if draw_zero_line:
        ax.axhline(0, color="#6b7280", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_title(f"Ntrial1-3 combined: 2s {title_feature} by effective manual motion label")
    ax.set_xlabel("Effective manual motion label")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", color="#d7dde5", linewidth=0.8, alpha=0.7)
    fig.text(0.01, 0.01, caption, fontsize=8, color="#3f4752")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def make_plots(
    trial_id: str,
    out_dir: Path,
    traj: pd.DataFrame,
    sample_df: pd.DataFrame,
    window_df: pd.DataFrame,
    states: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if sample_df.empty:
        return

    fig, ax = plt.subplots(figsize=(13, 5))
    used_labels = style_state_spans(ax, states)
    ax.plot(sample_df["session_s"], sample_df["tz_world_device"], color="#1f2d3a", linewidth=1.2, label="raw z height")
    ax.plot(sample_df["session_s"], sample_df["z_centered_m"], color="#35877d", linewidth=0.8, alpha=0.7, label="z centered by trial median")
    ax.set_title(f"{trial_id}: Raw SLAM z over synchronized session with manual action labels")
    ax.set_xlabel("session time (s)")
    ax.set_ylabel("z / relative z (m)")
    add_action_color_legend(ax, used_labels)
    ax.legend(loc="upper right", title="Plotted values", fontsize=8, title_fontsize=8)
    fig.text(
        0.01,
        0.01,
        "Background colors show effective manual action labels from the workbook; transition clips are split across their component actions.",
        fontsize=8,
        color="#3f4752",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "height_over_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        sample_df["tx_world_device"],
        sample_df["ty_world_device"],
        c=sample_df["tz_world_device"],
        cmap="viridis",
        s=4,
        alpha=0.75,
    )
    ax.set_title(f"{trial_id}: XY path colored by z")
    ax.set_xlabel("x world device (m)")
    ax.set_ylabel("y world device (m)")
    ax.axis("equal")
    fig.colorbar(scatter, ax=ax, label="z world device (m)")
    fig.tight_layout()
    fig.savefig(out_dir / "xy_path_colored_by_height.png", dpi=180)
    plt.close(fig)

    if not window_df.empty:
        fig, ax = plt.subplots(figsize=(13, 5))
        for window_s, part in window_df.groupby("window_s"):
            ax.plot(part["center_s"], part["vertical_delta_m"], linewidth=0.9, label=f"{window_s:g}s window")
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(f"{trial_id}: vertical delta changes with window size")
        ax.set_xlabel("window center time (s)")
        ax.set_ylabel("z_last - z_first (m)")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "vertical_delta_by_window_size.png", dpi=180)
        plt.close(fig)

        two_s = window_df[np.isclose(window_df["window_s"], 2.0)].copy()
        if not two_s.empty:
            fig, ax1 = plt.subplots(figsize=(13, 5))
            used_labels = style_state_spans(ax1, states)
            ax1.plot(two_s["center_s"], two_s["mean_z_m"], color="#1f2d3a", linewidth=1.1, label="2s mean z")
            ax1.set_xlabel("window center time (s)")
            ax1.set_ylabel("mean z in 2s window (m)")
            ax2 = ax1.twinx()
            bar_colors = color_series_by_label(two_s["speech_label"])
            ax2.bar(
                two_s["center_s"],
                two_s["vertical_delta_m"],
                width=1.6,
                color=bar_colors,
                alpha=0.38,
            )
            ax2.set_ylabel("vertical delta in 2s window (m)")
            ax1.set_title(f"{trial_id}: posture height vs old vertical displacement with manual action labels")
            add_action_color_legend(ax1, used_labels)
            from matplotlib.patches import Patch

            handles1, labels1 = ax1.get_legend_handles_labels()
            value_handles = handles1 + [
                Patch(facecolor="#8a94a6", alpha=0.35, label="bar height = 2s vertical delta")
            ]
            value_labels = labels1 + ["bar height = 2s vertical delta"]
            ax1.legend(
                value_handles,
                value_labels,
                loc="upper right",
                title="Plotted values",
                fontsize=8,
                title_fontsize=8,
            )
            fig.text(
                0.01,
                0.01,
                "Line = 2s mean raw z. Bars = 2s vertical delta. Action colors come from effective manual labels.",
                fontsize=8,
                color="#3f4752",
            )
            fig.tight_layout()
            fig.savefig(out_dir / "mean_height_vs_vertical_delta_2s.png", dpi=180)
            plt.close(fig)

        plot_mean_z_by_motion_label_boxplot(trial_id, out_dir, window_df)
        plot_speed_by_motion_label_boxplot(trial_id, out_dir, window_df)
        plot_window_feature_by_motion_label_boxplot(
            trial_id,
            out_dir,
            window_df,
            "horizontal_delta_m",
            "2s window horizontal_displacement_m = sqrt(dx^2 + dy^2)",
            "horizontal displacement",
            "horizontal_displacement_by_motion_label_boxplot.png",
            "Each point is one 2s synchronized SLAM window. This is endpoint XY displacement, not total path length.",
        )
        plot_window_feature_by_motion_label_boxplot(
            trial_id,
            out_dir,
            window_df,
            "horizontal_path_length_m",
            "2s window horizontal_path_length_m",
            "horizontal path length",
            "horizontal_path_length_by_motion_label_boxplot.png",
            "Each point is one 2s synchronized SLAM window. Path length sums frame-to-frame XY movement inside the window.",
        )
        plot_window_feature_by_motion_label_boxplot(
            trial_id,
            out_dir,
            window_df,
            "vertical_delta_m",
            "2s window vertical_displacement_m = z_end - z_start",
            "vertical displacement",
            "vertical_displacement_by_motion_label_boxplot.png",
            "Each point is one 2s synchronized SLAM window. Positive values indicate upward net z movement; negative values indicate downward.",
            draw_zero_line=True,
        )
        plot_window_feature_by_motion_label_boxplot(
            trial_id,
            out_dir,
            window_df,
            "height_slope_mps",
            "2s window height_slope_mps",
            "height slope",
            "height_slope_by_motion_label_boxplot.png",
            "Each point is one 2s synchronized SLAM window. Height slope is vertical displacement divided by the actual window duration.",
            draw_zero_line=True,
        )
        plot_mean_z_vs_speed_scatter(trial_id, out_dir, window_df)
        plot_horizontal_displacement_vs_speed_scatter(trial_id, out_dir, window_df)
        plot_horizontal_path_length_vs_speed_scatter(trial_id, out_dir, window_df)
        plot_vertical_displacement_vs_speed_scatter(trial_id, out_dir, window_df)
        plot_vertical_change_vs_path_length_scatters(trial_id, out_dir, window_df)
        plot_instantaneous_speed_over_time(trial_id, out_dir, traj, window_df, states)
        plot_height_and_speed_over_time(trial_id, out_dir, traj, states)
        plot_instantaneous_z_vs_speed_scatter(trial_id, out_dir, traj, states)


def write_trial_diagnostics(trial_id: str, args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RUNS_DIR / trial_id / "raw_parameter_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{trial_id}: loading and aligning SLAM trajectory", flush=True)
    traj, meta = make_session_trajectory(trial_id)
    states = load_effective_manual_intervals(trial_id)
    print(f"{trial_id}: downsampling to {args.sample_fps:g} fps", flush=True)
    sample_df = downsample_session(traj, args.sample_fps, states)
    print(f"{trial_id}: computing sliding-window metrics", flush=True)
    window_df = sliding_window_metrics(traj, states, args.window_sizes, args.window_step_s)
    print(f"{trial_id}: summarizing speech-state height bands", flush=True)
    state_summary_df = speech_state_height_summary(traj, states)

    sample_df.to_csv(out_dir / f"sample_{int(args.sample_fps)}fps.csv", index=False)
    window_df.to_csv(out_dir / "sliding_window_metrics.csv", index=False)
    state_summary_df.to_csv(out_dir / "speech_state_height_summary.csv", index=False)
    print(f"{trial_id}: rendering diagnostic plots", flush=True)
    make_plots(trial_id, out_dir, traj, sample_df, window_df, states)

    if not state_summary_df.empty:
        by_label = (
            state_summary_df.groupby("label")
            .agg(
                state_count=("label", "size"),
                total_duration_s=("duration_s", "sum"),
                median_state_mean_z_m=("mean_z_m", "median"),
                min_state_mean_z_m=("mean_z_m", "min"),
                max_state_mean_z_m=("mean_z_m", "max"),
                median_xy_range_m=("xy_range_m", "median"),
            )
            .reset_index()
        )
        by_label.to_csv(out_dir / "height_summary_by_label.csv", index=False)

    summary = {
        **meta,
        "output_dir": str(out_dir),
        "sample_csv": str(out_dir / f"sample_{int(args.sample_fps)}fps.csv"),
        "sliding_window_csv": str(out_dir / "sliding_window_metrics.csv"),
        "state_height_summary_csv": str(out_dir / "speech_state_height_summary.csv"),
        "plots": [
            str(out_dir / "height_over_time.png"),
            str(out_dir / "xy_path_colored_by_height.png"),
            str(out_dir / "vertical_delta_by_window_size.png"),
            str(out_dir / "mean_height_vs_vertical_delta_2s.png"),
            str(out_dir / "mean_z_by_motion_label_boxplot.png"),
            str(out_dir / "mean_linear_speed_by_motion_label_boxplot.png"),
            str(out_dir / "horizontal_displacement_by_motion_label_boxplot.png"),
            str(out_dir / "horizontal_path_length_by_motion_label_boxplot.png"),
            str(out_dir / "vertical_displacement_by_motion_label_boxplot.png"),
            str(out_dir / "height_slope_by_motion_label_boxplot.png"),
            str(out_dir / "mean_z_vs_mean_linear_speed_scatter.png"),
            str(out_dir / "horizontal_displacement_vs_mean_linear_speed_scatter.png"),
            str(out_dir / "horizontal_path_length_vs_mean_linear_speed_scatter.png"),
            str(out_dir / "vertical_displacement_vs_mean_linear_speed_scatter.png"),
            str(out_dir / "vertical_delta_vs_horizontal_path_length_scatter.png"),
            str(out_dir / "height_slope_vs_horizontal_path_length_scatter.png"),
            str(out_dir / "instantaneous_linear_speed_over_time.png"),
            str(out_dir / "height_and_speed_over_time.png"),
            str(out_dir / "instantaneous_z_vs_speed_scatter.png"),
        ],
    }
    (out_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    summaries = []
    for trial_id in args.trials:
        print(f"=== {trial_id} ===", flush=True)
        summary = write_trial_diagnostics(trial_id, args)
        summaries.append(summary)
        print(f"Wrote diagnostics to {summary['output_dir']}", flush=True)
    combined_boxplot = plot_combined_mean_z_by_motion_label_boxplot(args.trials)
    if combined_boxplot is not None:
        print(f"Wrote combined boxplot to {combined_boxplot}", flush=True)
    combined_speed_boxplot = plot_combined_speed_by_motion_label_boxplot(args.trials)
    if combined_speed_boxplot is not None:
        print(f"Wrote combined speed boxplot to {combined_speed_boxplot}", flush=True)
    combined_feature_boxplot_specs = [
        (
            "horizontal_delta_m",
            "2s window horizontal_displacement_m = sqrt(dx^2 + dy^2)",
            "horizontal displacement",
            "horizontal_displacement_by_motion_label_boxplot_Ntrial1_Ntrial2_Ntrial3.png",
            "Endpoint XY displacement by label. This is useful but can understate curved or back-and-forth motion.",
            False,
        ),
        (
            "horizontal_path_length_m",
            "2s window horizontal_path_length_m",
            "horizontal path length",
            "horizontal_path_length_by_motion_label_boxplot_Ntrial1_Ntrial2_Ntrial3.png",
            "Path length sums frame-to-frame XY movement and is usually a stronger movement-amount feature than endpoint displacement.",
            False,
        ),
        (
            "vertical_delta_m",
            "2s window vertical_displacement_m = z_end - z_start",
            "vertical displacement",
            "vertical_displacement_by_motion_label_boxplot_Ntrial1_Ntrial2_Ntrial3.png",
            "Positive values indicate upward net z movement; negative values indicate downward net z movement.",
            True,
        ),
        (
            "height_slope_mps",
            "2s window height_slope_mps",
            "height slope",
            "height_slope_by_motion_label_boxplot_Ntrial1_Ntrial2_Ntrial3.png",
            "Height slope is vertical displacement divided by duration, making vertical trend comparable across future window sizes.",
            True,
        ),
    ]
    for feature_col, y_label, title_feature, filename, caption, draw_zero_line in combined_feature_boxplot_specs:
        combined_feature_boxplot = plot_combined_window_feature_by_motion_label_boxplot(
            args.trials,
            feature_col,
            y_label,
            title_feature,
            filename,
            caption,
            draw_zero_line=draw_zero_line,
        )
        if combined_feature_boxplot is not None:
            print(f"Wrote combined {title_feature} boxplot to {combined_feature_boxplot}", flush=True)
    combined_scatter = plot_combined_mean_z_vs_speed_scatter(args.trials)
    if combined_scatter is not None:
        print(f"Wrote combined scatter to {combined_scatter}", flush=True)
    combined_horizontal_scatter = plot_combined_horizontal_displacement_vs_speed_scatter(args.trials)
    if combined_horizontal_scatter is not None:
        print(f"Wrote combined horizontal displacement scatter to {combined_horizontal_scatter}", flush=True)
    combined_horizontal_path_scatter = plot_combined_horizontal_path_length_vs_speed_scatter(args.trials)
    if combined_horizontal_path_scatter is not None:
        print(f"Wrote combined horizontal path length scatter to {combined_horizontal_path_scatter}", flush=True)
    combined_vertical_scatter = plot_combined_vertical_displacement_vs_speed_scatter(args.trials)
    if combined_vertical_scatter is not None:
        print(f"Wrote combined vertical displacement scatter to {combined_vertical_scatter}", flush=True)
    combined_vertical_path_scatters = plot_combined_vertical_change_vs_path_length_scatters(args.trials)
    for combined_vertical_path_scatter in combined_vertical_path_scatters:
        print(f"Wrote combined vertical/path scatter to {combined_vertical_path_scatter}", flush=True)
    print(json.dumps({"trials": summaries}, indent=2))


if __name__ == "__main__":
    main()
