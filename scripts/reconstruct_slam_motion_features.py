#!/usr/bin/env python3
"""Reconstruct SLAM motion features and test interpretable motion rules.

This script is intentionally analysis-first. It does not modify the dashboard or
the spoken-label workbook; it writes reproducible feature and accuracy outputs
under each trial's raw_parameter_diagnostics folder.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from explore_slam_parameters import RUNS_DIR, WORKBOOK_PATH, make_session_trajectory


MOTION_LABEL_ALIASES = {
    "sit": "sitting",
    "stand": "standing",
    "walk": "walking",
    "go upstairs": "upstairs",
    "go downstairs": "downstairs",
    "upstair": "upstairs",
    "downstair": "downstairs",
}

CORE_ACTIONS = {"sitting", "standing", "walking", "upstairs", "downstairs"}
CLINICAL_TRANSITION_LABELS = {
    ("sitting", "standing"): "sit-to-stand",
    ("standing", "sitting"): "stand-to-sit",
    ("standing", "walking"): "stand-to-walk",
    ("walking", "standing"): "walk-to-stand",
}


@dataclass(frozen=True)
class ReferenceInterval:
    start_s: float
    end_s: float
    actions: tuple[str, ...]
    label: str
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", nargs="+", default=["Ntrial1", "Ntrial2", "Ntrial3"])
    parser.add_argument("--window-sizes", nargs="+", type=float, default=[0.5, 1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--window-step-s", type=float, default=0.5)
    parser.add_argument("--gap-threshold-s", type=float, default=0.02)
    return parser.parse_args()


def normalize_motion_label(label: Any) -> str:
    if label is None or pd.isna(label):
        return ""
    normalized = str(label).strip().lower().replace("_", "-")
    normalized = " ".join(normalized.split())
    normalized = MOTION_LABEL_ALIASES.get(normalized, normalized)
    return normalized if normalized in CORE_ACTIONS else normalized


def unique_ordered(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        if value and value not in output:
            output.append(value)
    return output


def format_actions(actions: list[str] | tuple[str, ...]) -> str:
    return "-".join(actions)


def transition_actions_from_label(label: Any) -> list[str]:
    if label is None or pd.isna(label):
        return []
    normalized = str(label).strip().lower().replace("_", "-")
    normalized = " ".join(normalized.split())
    normalized = normalized.replace(" to ", "-to-")
    normalized = normalized.replace(" -> ", "-to-")

    if "-to-" in normalized:
        pieces = normalized.split("-to-")
    else:
        pieces = normalized.replace("|", "-").replace("/", "-").replace(",", "-").split("-")
    actions = [normalize_motion_label(piece) for piece in pieces]
    actions = [action for action in actions if action in CORE_ACTIONS]
    return unique_ordered(actions)


def canonical_transition_label(actions: list[str] | tuple[str, ...]) -> str:
    normalized_actions = unique_ordered([normalize_motion_label(action) for action in actions])
    normalized_actions = [action for action in normalized_actions if action in CORE_ACTIONS]
    if not normalized_actions:
        return ""
    if len(normalized_actions) == 1:
        return normalized_actions[0]
    pair = (normalized_actions[0], normalized_actions[1])
    if pair in CLINICAL_TRANSITION_LABELS:
        return CLINICAL_TRANSITION_LABELS[pair]
    return "-to-".join(normalized_actions)


def canonical_prediction_label(label: Any) -> str:
    stable = normalize_motion_label(label)
    if stable in CORE_ACTIONS:
        return stable
    transition_actions = transition_actions_from_label(label)
    if len(transition_actions) > 1:
        return canonical_transition_label(transition_actions)
    return stable


def parse_label_display(label: Any) -> list[str]:
    transition_actions = transition_actions_from_label(label)
    if transition_actions:
        return transition_actions
    normalized = normalize_motion_label(label)
    if not normalized:
        return []
    pieces = [normalize_motion_label(piece) for piece in normalized.replace("|", "-").split("-")]
    return unique_ordered([piece for piece in pieces if piece in CORE_ACTIONS])


def actions_from_review_row(row: pd.Series) -> list[str]:
    actions: list[str] = []
    for idx in range(1, 6):
        action = normalize_motion_label(row.get(f"effective_action_{idx}", ""))
        if action:
            actions.append(action)
    if not actions:
        actions = parse_label_display(row.get("effective_manual_label", ""))
    return unique_ordered(actions)


def load_review_sheet(trial_id: str) -> pd.DataFrame:
    if not WORKBOOK_PATH.exists():
        return load_session_manual_review(trial_id)
    sheet_name = f"{trial_id}_review"
    try:
        review_df = pd.read_excel(WORKBOOK_PATH, sheet_name=sheet_name)
    except ValueError:
        return load_session_manual_review(trial_id)
    if review_df.empty:
        return load_session_manual_review(trial_id)
    return review_df


def load_session_manual_review(trial_id: str) -> pd.DataFrame:
    """Build clip-level review rows from full-session manual labeling JSON.

    Long normal trials are labeled with continuous manual segments rather than
    the earlier spoken-action workbook sheets. This adapter keeps those labels
    as the source of truth and evaluates the existing 2s clip predictions
    against the manual segment with the largest temporal overlap. When manual
    labels do not exist yet, it still returns unlabeled clip rows so the newest
    criteria can be generated for the whole normal trial.
    """
    manual_path = RUNS_DIR / trial_id / "manual_labels" / "session_manual_segments.json"

    metrics_path = RUNS_DIR / trial_id / "downward_master_session" / "downward_session_clip_metrics.csv"
    if not metrics_path.exists():
        metrics_path = RUNS_DIR / trial_id / "clip_metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame()

    segments: list[dict[str, Any]] = []
    if manual_path.exists():
        try:
            payload = json.loads(manual_path.read_text())
        except json.JSONDecodeError:
            return pd.DataFrame()
        segments = payload.get("segments", [])

    metrics = pd.read_csv(metrics_path)
    required = {"clip_idx", "start_s", "end_s"}
    if not required.issubset(metrics.columns):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, clip in metrics.iterrows():
        clip_start = float(clip["start_s"])
        clip_end = float(clip["end_s"])
        if not math.isfinite(clip_start) or not math.isfinite(clip_end) or clip_end <= clip_start:
            continue

        action_overlap: dict[str, float] = {}
        total_overlap = 0.0
        for segment in segments:
            label = normalize_motion_label(segment.get("motion_label", ""))
            if label not in CORE_ACTIONS:
                continue
            segment_start = float(segment.get("start_s", 0.0))
            segment_end = float(segment.get("end_s", 0.0))
            overlap = max(0.0, min(clip_end, segment_end) - max(clip_start, segment_start))
            if overlap <= 0:
                continue
            action_overlap[label] = action_overlap.get(label, 0.0) + overlap
            total_overlap += overlap
        # Keep a secondary action only when the clip really crosses a boundary.
        # Unlabeled clips are still kept so the improved model can be generated
        # for the full session; they are ignored by the accuracy summary.
        ordered_actions = (
            [
                label
                for label, overlap in sorted(action_overlap.items(), key=lambda item: (-item[1], item[0]))
                if overlap / max(total_overlap, 1e-9) >= 0.20
            ]
            if action_overlap
            else []
        )
        ordered_actions = unique_ordered(ordered_actions)

        predicted_label = (
            clip.get("criteria_v3_prediction")
            or clip.get("criteria_v2_prediction")
            or clip.get("hybrid_prediction")
            or clip.get("rule_prediction")
            or ""
        )
        row: dict[str, Any] = {
            "clip_idx": int(clip["clip_idx"]),
            "clip_start_s": clip_start,
            "clip_end_s": clip_end,
            "predicted_label": predicted_label,
            "effective_manual_label": format_actions(ordered_actions),
            "effective_manual_action_count": len(ordered_actions),
            "effective_label_source": "session_manual_segments",
            "effective_overlap_s": total_overlap,
            "mean_z_m": clip.get("mean_z_m", np.nan),
        }
        for idx in range(5):
            row[f"effective_action_{idx + 1}"] = ordered_actions[idx] if idx < len(ordered_actions) else ""
        rows.append(row)

    return pd.DataFrame(rows)


def build_reference_intervals(review_df: pd.DataFrame) -> list[ReferenceInterval]:
    intervals: list[ReferenceInterval] = []
    if review_df.empty:
        return intervals
    required = {"clip_start_s", "clip_end_s"}
    if not required.issubset(review_df.columns):
        return intervals
    for _, row in review_df.iterrows():
        actions = actions_from_review_row(row)
        if not actions:
            continue
        intervals.append(
            ReferenceInterval(
                start_s=float(row["clip_start_s"]),
                end_s=float(row["clip_end_s"]),
                actions=tuple(actions),
                label=format_actions(actions),
                source=str(row.get("effective_label_source", "workbook_review")),
            )
        )
    return intervals


def actions_for_interval(
    start_s: float,
    end_s: float,
    intervals: list[ReferenceInterval],
) -> tuple[list[str], float, str]:
    if not intervals:
        return [], 0.0, ""

    action_overlap: dict[str, float] = {}
    action_order: dict[str, int] = {}
    source_overlap: dict[str, float] = {}
    coverage_overlap = 0.0
    for interval in intervals:
        overlap = max(0.0, min(end_s, interval.end_s) - max(start_s, interval.start_s))
        if overlap <= 0.0:
            continue
        coverage_overlap += overlap
        for action in interval.actions:
            action_overlap[action] = action_overlap.get(action, 0.0) + overlap
            if action not in action_order:
                action_order[action] = len(action_order)
        source_overlap[interval.source] = source_overlap.get(interval.source, 0.0) + overlap

    if not action_overlap:
        return [], 0.0, ""
    actions = [item[0] for item in sorted(action_overlap.items(), key=lambda item: (-item[1], action_order[item[0]]))]
    sources = [item[0] for item in sorted(source_overlap.items(), key=lambda item: (-item[1], item[0]))]
    return actions, coverage_overlap, sources[0] if sources else ""


def add_motion_derived_columns(traj: pd.DataFrame) -> pd.DataFrame:
    output = traj.copy()
    output["x_rel_start_m"] = output["tx_world_device"] - output["tx_world_device"].iloc[0]
    output["y_rel_start_m"] = output["ty_world_device"] - output["ty_world_device"].iloc[0]
    output["z_rel_start_m"] = output["tz_world_device"] - output["tz_world_device"].iloc[0]
    output["z_rel_trial_median_m"] = output["tz_world_device"] - output["tz_world_device"].median()

    dt = output["session_s"].diff().fillna(0.0).to_numpy()
    dx = output["tx_world_device"].diff().fillna(0.0).to_numpy()
    dy = output["ty_world_device"].diff().fillna(0.0).to_numpy()
    dz = output["tz_world_device"].diff().fillna(0.0).to_numpy()
    horizontal_step = np.sqrt(dx**2 + dy**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        horizontal_speed = np.divide(horizontal_step, dt, out=np.zeros_like(horizontal_step), where=dt > 0)
        vertical_speed = np.divide(dz, dt, out=np.zeros_like(dz), where=dt > 0)

    output["sample_dt_s"] = dt
    output["horizontal_step_m"] = horizontal_step
    output["vertical_step_m"] = dz
    output["horizontal_speed_world_mps"] = horizontal_speed
    output["vertical_speed_world_mps"] = vertical_speed
    output["horizontal_path_cum_m"] = np.cumsum(horizontal_step)
    output["positive_z_cum_m"] = np.cumsum(np.maximum(dz, 0.0))
    output["negative_z_cum_m"] = np.cumsum(np.maximum(-dz, 0.0))

    roll, pitch, yaw = quaternion_to_euler(
        output["qx_world_device"].to_numpy(),
        output["qy_world_device"].to_numpy(),
        output["qz_world_device"].to_numpy(),
        output["qw_world_device"].to_numpy(),
    )
    output["roll_rad"] = roll
    output["pitch_rad"] = pitch
    output["yaw_rad"] = yaw
    return output


def quaternion_to_euler(
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    qw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def estimate_posture_baselines(review_df: pd.DataFrame, traj: pd.DataFrame) -> dict[str, Any]:
    baselines: dict[str, Any] = {
        "sitting_baseline_z_m": None,
        "standing_baseline_z_m": None,
        "standing_median_z_m": None,
        "posture_z_band_m": None,
        "posture_baseline_method": "none",
    }
    if not review_df.empty and {"effective_manual_action_count", "mean_z_m"}.issubset(review_df.columns):
        stable = review_df[review_df["effective_manual_action_count"].fillna(0).astype(int) == 1].copy()
        stable["effective_action_norm"] = stable.apply(
            lambda row: actions_from_review_row(row)[0] if actions_from_review_row(row) else "",
            axis=1,
        )
        sitting_z = pd.to_numeric(stable.loc[stable["effective_action_norm"] == "sitting", "mean_z_m"], errors="coerce").dropna()
        standing_z = pd.to_numeric(
            stable.loc[stable["effective_action_norm"] == "standing", "mean_z_m"], errors="coerce"
        ).dropna()

        if not sitting_z.empty:
            baselines["sitting_baseline_z_m"] = float(sitting_z.median())
        if not standing_z.empty:
            baselines["standing_median_z_m"] = float(standing_z.median())
            if baselines["sitting_baseline_z_m"] is not None:
                above_sitting = standing_z[standing_z > float(baselines["sitting_baseline_z_m"]) + 0.10]
                standing_reference = above_sitting.quantile(0.25) if not above_sitting.empty else standing_z.quantile(0.25)
            else:
                standing_reference = standing_z.median()
            baselines["standing_baseline_z_m"] = float(standing_reference)

        sitting = baselines["sitting_baseline_z_m"]
        standing = baselines["standing_baseline_z_m"]
        if sitting is not None and standing is not None and abs(float(standing) - float(sitting)) >= 0.15:
            baselines["posture_z_band_m"] = float((float(sitting) + float(standing)) / 2.0)
            baselines["posture_baseline_method"] = "workbook_stable_label_medians"
            return baselines
        if standing is not None:
            baselines["posture_baseline_method"] = "workbook_standing_only"
            return baselines

    clustered_band = estimate_posture_band_from_low_motion(traj)
    if clustered_band is not None:
        baselines.update(clustered_band)
    return baselines


def estimate_posture_band_from_low_motion(traj: pd.DataFrame) -> dict[str, Any] | None:
    low_motion = traj[
        (traj["linear_speed_mps"] < 0.15)
        & (traj["horizontal_speed_world_mps"] < 0.15)
        & traj["tz_world_device"].notna()
    ]["tz_world_device"].sort_values()
    if len(low_motion) < 200:
        return None
    sample = low_motion.iloc[np.linspace(0, len(low_motion) - 1, min(len(low_motion), 2000)).astype(int)].to_numpy()
    gaps = np.diff(sample)
    if len(gaps) == 0:
        return None
    gap_index = int(np.argmax(gaps))
    gap = float(gaps[gap_index])
    if gap < 0.12:
        return None
    lower = sample[: gap_index + 1]
    upper = sample[gap_index + 1 :]
    if len(lower) < 100 or len(upper) < 100:
        return None
    sitting = float(np.median(lower))
    standing = float(np.median(upper))
    return {
        "sitting_baseline_z_m": sitting,
        "standing_baseline_z_m": standing,
        "standing_median_z_m": standing,
        "posture_z_band_m": float((sample[gap_index] + sample[gap_index + 1]) / 2.0),
        "posture_baseline_method": "low_motion_z_cluster_gap",
    }


def estimate_motion_rule_thresholds(traj: pd.DataFrame, window_s: float = 2.0, step_s: float = 1.0) -> dict[str, float]:
    """Estimate trial-relative movement thresholds from raw SLAM windows.

    The thresholds still include conservative absolute guardrails so posture
    transitions in a no-walking trial do not become "walking" just because they
    are the fastest movements in that session.
    """
    if traj.empty:
        return {}

    times = traj["session_s"].to_numpy()
    x = traj["tx_world_device"].to_numpy()
    y = traj["ty_world_device"].to_numpy()
    z = traj["tz_world_device"].to_numpy()
    speed = traj["linear_speed_mps"].to_numpy()
    duration = float(times[-1] - times[0])
    if duration < window_s:
        return {}

    rows: list[dict[str, float]] = []
    starts = np.arange(0.0, duration - window_s + 1e-9, step_s)
    for start_s in starts:
        end_s = start_s + window_s
        left = int(np.searchsorted(times, start_s, side="left"))
        right = int(np.searchsorted(times, end_s, side="left"))
        if right - left < 2:
            continue
        section_t = times[left:right]
        section_z = z[left:right]
        horizontal_path = float(np.sum(np.hypot(np.diff(x[left:right]), np.diff(y[left:right]))))
        height_delta = float(section_z[-1] - section_z[0])
        t_centered = section_t - float(np.mean(section_t))
        denom = float(np.dot(t_centered, t_centered))
        height_slope = (
            float(np.dot(t_centered, section_z - float(np.mean(section_z))) / denom) if denom > 0 else 0.0
        )
        rows.append(
            {
                "horizontal_path_length_m": horizontal_path,
                "linear_speed_mean_mps": float(np.mean(speed[left:right])),
                "height_slope_abs_mps": abs(height_slope),
                "height_delta_abs_m": abs(height_delta),
            }
        )

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    path_p90 = float(df["horizontal_path_length_m"].quantile(0.90))
    speed_p90 = float(df["linear_speed_mean_mps"].quantile(0.90))
    slope_abs_p90 = float(df["height_slope_abs_mps"].quantile(0.90))

    return {
        "rule_window_s": float(window_s),
        "path_p90_m": path_p90,
        "speed_p90_mps": speed_p90,
        "slope_abs_p90_mps": slope_abs_p90,
        "walking_path_min_m": float(max(0.60, min(1.40, 0.55 * path_p90))),
        "walking_speed_min_mps": float(max(0.35, min(0.75, 0.55 * speed_p90))),
        "stair_path_min_m": float(max(0.30, min(0.75, 0.35 * path_p90))),
        "stair_speed_min_mps": float(max(0.18, min(0.45, 0.35 * speed_p90))),
        "stair_height_delta_min_m": 0.18,
        "stair_height_slope_min_mps": float(max(0.07, min(0.16, 0.50 * slope_abs_p90))),
        "stair_path_to_vertical_ratio_min": 1.15,
        "near_zero_height_slope_mps": 0.08,
    }


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def feature_for_interval(
    traj: pd.DataFrame,
    start_s: float,
    end_s: float,
    baselines: dict[str, Any],
    intervals: list[ReferenceInterval],
    gap_threshold_s: float,
) -> dict[str, Any] | None:
    times = traj["session_s"].to_numpy()
    left = int(np.searchsorted(times, start_s, side="left"))
    right = int(np.searchsorted(times, end_s, side="left"))
    if right - left < 2:
        return None

    section = traj.iloc[left:right]
    t = section["session_s"].to_numpy()
    x = section["tx_world_device"].to_numpy()
    y = section["ty_world_device"].to_numpy()
    z = section["tz_world_device"].to_numpy()
    speed = section["linear_speed_mps"].to_numpy()
    horizontal_speed = section["horizontal_speed_world_mps"].to_numpy()
    angular = section["angular_speed_rps"].to_numpy()
    quality = pd.to_numeric(section["quality_score"], errors="coerce").to_numpy()
    sample_dt = section["sample_dt_s"].to_numpy()

    horizontal_path = float(section["horizontal_path_cum_m"].iloc[-1] - section["horizontal_path_cum_m"].iloc[0])
    positive_z = float(section["positive_z_cum_m"].iloc[-1] - section["positive_z_cum_m"].iloc[0])
    negative_z = float(section["negative_z_cum_m"].iloc[-1] - section["negative_z_cum_m"].iloc[0])
    net_horizontal = float(math.hypot(x[-1] - x[0], y[-1] - y[0]))
    height_delta = float(z[-1] - z[0])
    t_centered = t - float(np.mean(t))
    denom = float(np.dot(t_centered, t_centered))
    height_slope = float(np.dot(t_centered, z - float(np.mean(z))) / denom) if denom > 0 else 0.0

    center_s = (start_s + end_s) / 2.0
    center_idx = int(np.searchsorted(times, center_s, side="left"))
    center_idx = min(max(center_idx, 0), len(traj) - 1)
    center_row = traj.iloc[center_idx]

    actions, overlap_s, label_source = actions_for_interval(start_s, end_s, intervals)
    standing_baseline = finite_float(baselines.get("standing_baseline_z_m"))
    sitting_baseline = finite_float(baselines.get("sitting_baseline_z_m"))
    median_height = float(np.median(z))

    gravity = section[["gravity_x_world", "gravity_y_world", "gravity_z_world"]].to_numpy(dtype=float)
    gravity_norm = np.sqrt(np.sum(gravity**2, axis=1))
    roll = np.unwrap(section["roll_rad"].to_numpy())
    pitch = np.unwrap(section["pitch_rad"].to_numpy())
    yaw = np.unwrap(section["yaw_rad"].to_numpy())

    row: dict[str, Any] = {
        "start_s": float(start_s),
        "end_s": float(end_s),
        "center_s": float(center_s),
        "duration_s": float(end_s - start_s),
        "sample_count": int(len(section)),
        "tx_world_device": float(center_row["tx_world_device"]),
        "ty_world_device": float(center_row["ty_world_device"]),
        "tz_world_device": float(center_row["tz_world_device"]),
        "x_rel_start_m": float(center_row["x_rel_start_m"]),
        "y_rel_start_m": float(center_row["y_rel_start_m"]),
        "z_rel_start_m": float(center_row["z_rel_start_m"]),
        "z_rel_trial_median_m": float(center_row["z_rel_trial_median_m"]),
        "height_from_standing_baseline_m": ""
        if standing_baseline is None
        else float(median_height - standing_baseline),
        "height_from_sitting_baseline_m": "" if sitting_baseline is None else float(median_height - sitting_baseline),
        "window_mean_height_m": float(np.mean(z)),
        "window_median_height_m": median_height,
        "window_min_height_m": float(np.min(z)),
        "window_max_height_m": float(np.max(z)),
        "window_height_range_m": float(np.max(z) - np.min(z)),
        "window_height_delta_m": height_delta,
        "height_slope_mps": height_slope,
        "positive_height_accumulation_m": positive_z,
        "negative_height_accumulation_m": negative_z,
        "net_horizontal_displacement_m": net_horizontal,
        "horizontal_path_length_m": horizontal_path,
        "horizontal_speed_mean_mps": float(np.mean(horizontal_speed)),
        "horizontal_speed_median_mps": float(np.median(horizontal_speed)),
        "horizontal_speed_p95_mps": float(np.percentile(horizontal_speed, 95)),
        "horizontal_speed_std_mps": float(np.std(horizontal_speed)),
        "linear_speed_mean_mps": float(np.mean(speed)),
        "linear_speed_median_mps": float(np.median(speed)),
        "linear_speed_p95_mps": float(np.percentile(speed, 95)),
        "stationary_fraction": float(np.mean(horizontal_speed < 0.10)),
        "moving_fraction": float(np.mean(horizontal_speed > 0.25)),
        "path_efficiency": float(net_horizontal / horizontal_path) if horizontal_path > 1e-9 else 0.0,
        "roll_rad": float(center_row["roll_rad"]),
        "pitch_rad": float(center_row["pitch_rad"]),
        "yaw_rad": float(center_row["yaw_rad"]),
        "roll_change_rad": float(roll[-1] - roll[0]),
        "pitch_change_rad": float(pitch[-1] - pitch[0]),
        "yaw_change_rad": float(yaw[-1] - yaw[0]),
        "angular_speed_mean_rps": float(np.mean(angular)),
        "angular_speed_p95_rps": float(np.percentile(angular, 95)),
        "angular_stationary_fraction": float(np.mean(angular < 0.10)),
        "orientation_jitter": float(np.mean([np.std(roll), np.std(pitch), np.std(yaw)])),
        "quality_score_mean": float(np.nanmean(quality)) if not np.all(np.isnan(quality)) else "",
        "quality_score_min": float(np.nanmin(quality)) if not np.all(np.isnan(quality)) else "",
        "tracking_gap_count": int(np.sum(sample_dt[1:] > gap_threshold_s)),
        "tracking_gap_max_s": float(np.max(sample_dt[1:])) if len(sample_dt) > 1 else 0.0,
        "gravity_norm_mean": float(np.mean(gravity_norm)),
        "gravity_consistency": float(np.std(gravity_norm)),
        "effective_manual_label": format_actions(actions),
        "effective_manual_action_count": len(actions),
        "effective_label_source": label_source,
        "effective_overlap_s": float(overlap_s),
        "effective_overlap_fraction": float(overlap_s / max(end_s - start_s, 1e-9)),
    }
    for idx in range(5):
        row[f"effective_action_{idx + 1}"] = actions[idx] if idx < len(actions) else ""
    return row


def build_window_feature_table(
    traj: pd.DataFrame,
    baselines: dict[str, Any],
    intervals: list[ReferenceInterval],
    window_sizes: list[float],
    step_s: float,
    gap_threshold_s: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if traj.empty:
        return pd.DataFrame(rows)
    duration = float(traj["session_s"].max() - traj["session_s"].min())
    for window_s in window_sizes:
        if duration < window_s:
            continue
        starts = np.arange(0.0, duration - window_s + 1e-9, step_s)
        for start_s in starts:
            row = feature_for_interval(
                traj=traj,
                start_s=float(start_s),
                end_s=float(start_s + window_s),
                baselines=baselines,
                intervals=intervals,
                gap_threshold_s=gap_threshold_s,
            )
            if row is None:
                continue
            row["window_s"] = float(window_s)
            row["improved_rule_prediction"] = classify_improved_rule(row, baselines)
            row["criteria_v2_prediction"] = classify_motion_criteria_v2(row, baselines)
            rows.append(row)
    if not rows:
        return pd.DataFrame(rows)
    columns = ["window_s"] + [column for column in rows[0] if column != "window_s"]
    df = pd.DataFrame(rows)[columns]
    df = apply_transition_inclusive_predictions(df, baselines)
    return df


def classify_improved_rule(features: dict[str, Any] | pd.Series, baselines: dict[str, Any]) -> str:
    net_horizontal = float(features.get("net_horizontal_displacement_m", 0.0) or 0.0)
    horizontal_path = float(features.get("horizontal_path_length_m", 0.0) or 0.0)
    height_delta = float(features.get("window_height_delta_m", 0.0) or 0.0)
    height_slope = float(features.get("height_slope_mps", 0.0) or 0.0)
    mean_speed = float(features.get("linear_speed_mean_mps", 0.0) or 0.0)
    moving_fraction = float(features.get("moving_fraction", 0.0) or 0.0)
    median_height = float(features.get("window_median_height_m", features.get("window_mean_height_m", 0.0)) or 0.0)

    stair_motion = (
        (horizontal_path >= 0.55 and mean_speed >= 0.25)
        or (net_horizontal >= 0.50 and moving_fraction >= 0.35)
        or (horizontal_path >= 0.75)
    )
    if stair_motion and (height_delta >= 0.18 or height_slope >= 0.08):
        return "upstairs"
    if stair_motion and (height_delta <= -0.18 or height_slope <= -0.08):
        return "downstairs"

    walking_motion = (
        horizontal_path >= 0.65
        or net_horizontal >= 0.65
        or (net_horizontal >= 0.45 and mean_speed >= 0.28)
        or mean_speed >= 0.65
        or moving_fraction >= 0.55
    )
    if walking_motion:
        return "walking"

    posture_band = finite_float(baselines.get("posture_z_band_m"))
    if posture_band is not None:
        return "sitting" if median_height <= posture_band else "standing"

    return "standing"


def classify_motion_criteria_v2(features: dict[str, Any] | pd.Series, baselines: dict[str, Any]) -> str:
    """Plot-derived interpretable criteria for the five current motion labels.

    The logic mirrors the raw-parameter plots:
    speed/path length describe movement intensity, height slope describes
    vertical direction, and local mean height separates sitting from standing.
    """
    net_horizontal = float(features.get("net_horizontal_displacement_m", 0.0) or 0.0)
    horizontal_path = float(features.get("horizontal_path_length_m", 0.0) or 0.0)
    height_delta = float(features.get("window_height_delta_m", 0.0) or 0.0)
    height_slope = float(features.get("height_slope_mps", 0.0) or 0.0)
    mean_speed = float(features.get("linear_speed_mean_mps", 0.0) or 0.0)
    moving_fraction = float(features.get("moving_fraction", 0.0) or 0.0)
    median_height = float(features.get("window_median_height_m", features.get("window_mean_height_m", 0.0)) or 0.0)

    thresholds = baselines.get("criteria_v2_thresholds", {}) or {}
    walking_path_min = float(thresholds.get("walking_path_min_m", 0.65))
    walking_speed_min = float(thresholds.get("walking_speed_min_mps", 0.45))
    stair_path_min = float(thresholds.get("stair_path_min_m", 0.35))
    stair_speed_min = float(thresholds.get("stair_speed_min_mps", 0.20))
    stair_delta_min = float(thresholds.get("stair_height_delta_min_m", 0.18))
    stair_slope_min = float(thresholds.get("stair_height_slope_min_mps", 0.08))
    stair_ratio_min = float(thresholds.get("stair_path_to_vertical_ratio_min", 1.15))
    near_zero_slope = float(thresholds.get("near_zero_height_slope_mps", 0.08))

    # Sit/stand transitions can have large vertical change, but they are not stairs.
    # If the height remains inside the local posture zone and horizontal travel is
    # limited, keep the label in the posture family before applying stair logic.
    sitting_baseline = finite_float(baselines.get("sitting_baseline_z_m"))
    standing_baseline = finite_float(baselines.get("standing_baseline_z_m"))
    if sitting_baseline is not None and standing_baseline is not None:
        posture_low = min(sitting_baseline, standing_baseline)
        posture_high = max(sitting_baseline, standing_baseline)
        posture_range = max(posture_high - posture_low, 1e-6)
        posture_margin = max(0.08, 0.25 * posture_range)
        within_posture_zone = (posture_low - posture_margin) <= median_height <= (posture_high + posture_margin)
        posture_transition_like = (
            abs(height_delta) >= stair_delta_min
            and abs(height_slope) >= 0.08
            and mean_speed >= 0.20
            and horizontal_path < 0.55
            and within_posture_zone
        )
        if posture_transition_like:
            posture_edge_fraction = 0.03
            if height_delta < 0 and median_height > sitting_baseline + posture_edge_fraction * posture_range:
                return "standing"
            if height_delta > 0 and median_height < standing_baseline - posture_edge_fraction * posture_range:
                return "sitting"

    vertical_ratio = horizontal_path / max(abs(height_delta), 1e-6)
    stair_like = (
        abs(height_delta) >= stair_delta_min
        and abs(height_slope) >= stair_slope_min
        and horizontal_path >= stair_path_min
        and mean_speed >= stair_speed_min
        and vertical_ratio >= stair_ratio_min
    )
    weak_path_stair_like = (
        abs(height_delta) >= stair_delta_min
        and abs(height_slope) >= 0.09
        and horizontal_path >= 0.35
        and moving_fraction >= 0.30
        and vertical_ratio >= stair_ratio_min
    )
    if stair_like or weak_path_stair_like:
        return "upstairs" if height_delta > 0 else "downstairs"

    walking_like = (
        (
            horizontal_path >= walking_path_min
            and mean_speed >= walking_speed_min
            and abs(height_slope) <= near_zero_slope
        )
        or (
            horizontal_path >= 0.85
            and mean_speed >= 0.45
            and abs(height_slope) <= near_zero_slope
        )
        or (
            net_horizontal >= 0.80
            and moving_fraction >= 0.45
            and abs(height_delta) < stair_delta_min
        )
        or (
            net_horizontal >= 0.45
            and mean_speed >= 0.28
            and moving_fraction >= 0.35
            and abs(height_delta) < stair_delta_min
        )
    )
    if walking_like:
        return "walking"

    posture_band = finite_float(baselines.get("posture_z_band_m"))
    if posture_band is not None:
        return "sitting" if median_height <= posture_band else "standing"

    if mean_speed < 0.20 and horizontal_path < 0.30:
        return "standing"
    return "walking" if mean_speed >= walking_speed_min else "standing"


def numeric_feature(row: dict[str, Any] | pd.Series, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def active_posture_transition_label(
    row: dict[str, Any] | pd.Series,
    baselines: dict[str, Any],
    prev_base: str,
    current_base: str,
    next_base: str,
) -> str:
    """Detect active sit/stand transitions from local height change.

    This is deliberately limited to the posture neighborhood so stair segments
    with large vertical movement do not become sit-to-stand/stand-to-sit.
    """
    posture_neighbors = {prev_base, current_base, next_base} & {"sitting", "standing"}
    if len(posture_neighbors) < 2:
        return ""

    sitting_baseline = finite_float(baselines.get("sitting_baseline_z_m"))
    standing_baseline = finite_float(baselines.get("standing_baseline_z_m"))
    posture_range = (
        abs(float(standing_baseline) - float(sitting_baseline))
        if sitting_baseline is not None and standing_baseline is not None
        else 0.40
    )
    height_delta_min = max(0.18, min(0.30, 0.40 * posture_range))

    height_delta = numeric_feature(row, "window_height_delta_m")
    height_slope = numeric_feature(row, "height_slope_mps")
    horizontal_path = numeric_feature(row, "horizontal_path_length_m")
    mean_speed = numeric_feature(row, "linear_speed_mean_mps")
    moving_fraction = numeric_feature(row, "moving_fraction")

    posture_motion_like = (
        abs(height_delta) >= height_delta_min
        and abs(height_slope) >= 0.08
        and horizontal_path <= 0.70
        and mean_speed <= 0.65
        and moving_fraction <= 0.70
    )
    if not posture_motion_like:
        return ""
    return "sit-to-stand" if height_delta > 0 else "stand-to-sit"


def low_confidence_boundary_transition_label(
    row: dict[str, Any] | pd.Series,
    current_base: str,
    next_base: str,
    baselines: dict[str, Any],
) -> str:
    if current_base == next_base or current_base not in CORE_ACTIONS or next_base not in CORE_ACTIONS:
        return ""

    thresholds = baselines.get("criteria_v2_thresholds", {}) or {}
    walking_path_min = float(thresholds.get("walking_path_min_m", 0.65))
    walking_speed_min = float(thresholds.get("walking_speed_min_mps", 0.45))
    stair_slope_min = float(thresholds.get("stair_height_slope_min_mps", 0.08))

    horizontal_path = numeric_feature(row, "horizontal_path_length_m")
    mean_speed = numeric_feature(row, "linear_speed_mean_mps")
    height_slope = numeric_feature(row, "height_slope_mps")

    if current_base == "walking" and next_base == "standing":
        if horizontal_path < max(0.65, 0.55 * walking_path_min) or mean_speed < max(0.35, 0.60 * walking_speed_min):
            return canonical_transition_label([current_base, next_base])
    if current_base in {"upstairs", "downstairs"} and next_base in {"walking", "standing"}:
        if horizontal_path < 1.65 or mean_speed < 0.85 or abs(height_slope) < max(0.12, stair_slope_min):
            return canonical_transition_label([current_base, next_base])
    if current_base == "walking" and next_base in {"upstairs", "downstairs"}:
        if abs(height_slope) >= 0.08:
            return canonical_transition_label([current_base, next_base])
    return ""


def apply_transition_inclusive_predictions(df: pd.DataFrame, baselines: dict[str, Any]) -> pd.DataFrame:
    if df.empty or "criteria_v2_prediction" not in df.columns:
        return df

    output = df.copy()
    sort_columns = [column for column in ["window_s", "clip_idx", "center_s", "clip_start_s", "start_s"] if column in output.columns]
    if sort_columns:
        output = output.sort_values(sort_columns).copy()

    output["criteria_v3_base_prediction"] = output["criteria_v2_prediction"].map(canonical_prediction_label)
    output["criteria_v3_prediction"] = output["criteria_v3_base_prediction"]
    output["criteria_v3_transition_reason"] = ""

    group_keys = ["window_s"] if "window_s" in output.columns else [None]
    grouped = output.groupby(group_keys, sort=False, dropna=False) if group_keys != [None] else [(None, output)]
    for _, group in grouped:
        indices = list(group.index)
        bases = [str(output.at[index, "criteria_v3_base_prediction"]) for index in indices]
        first_pass: dict[int, tuple[str, str]] = {}
        for pos, index in enumerate(indices):
            prev_base = bases[pos - 1] if pos > 0 else ""
            current_base = bases[pos]
            next_base = bases[pos + 1] if pos + 1 < len(bases) else ""
            active_label = active_posture_transition_label(output.loc[index], baselines, prev_base, current_base, next_base)
            if active_label:
                first_pass[index] = (active_label, "active_posture_height_change")

        assigned = dict(first_pass)
        for pos, index in enumerate(indices):
            prev_base = bases[pos - 1] if pos > 0 else ""
            current_base = bases[pos]
            if (
                index not in assigned
                and prev_base != current_base
                and prev_base in CORE_ACTIONS
                and current_base in CORE_ACTIONS
            ):
                label = canonical_transition_label([prev_base, current_base])
                prev_index = indices[pos - 1]
                prev_assigned_label = assigned.get(prev_index, ("", ""))[0]
                if label != prev_assigned_label:
                    assigned[index] = (label, "stable_state_boundary_entered")

        for pos, index in enumerate(indices):
            if index in assigned:
                continue
            current_base = bases[pos]
            next_base = bases[pos + 1] if pos + 1 < len(bases) else ""
            label = low_confidence_boundary_transition_label(output.loc[index], current_base, next_base, baselines)
            if label:
                next_index = indices[pos + 1] if pos + 1 < len(indices) else None
                next_assigned_label = assigned.get(next_index, ("", ""))[0] if next_index is not None else ""
                if label != next_assigned_label:
                    assigned[index] = (label, "decelerating_or_vertical_boundary_exit")

        for index, (label, reason) in assigned.items():
            output.at[index, "criteria_v3_prediction"] = label
            output.at[index, "criteria_v3_transition_reason"] = reason

    if sort_columns:
        output = output.sort_index()
    return output


def apply_contextual_posture_predictions(df: pd.DataFrame, baselines: dict[str, Any]) -> pd.DataFrame:
    """Refine stable posture labels with before/after height context.

    A single world-Z posture band is unreliable when a long recording crosses
    floors or uneven ground. This pass keeps the criteria_v2 locomotion labels
    unchanged and tracks sitting/standing state through local vertical
    transitions at 2, 5, and 10 second scales.

    The correction is intentionally conservative:
    - contextual standing may replace a global-band sitting label;
    - contextual sitting replaces standing only when the learned posture range
      is implausibly large, indicating that the global band mixed floor levels.
    """
    if df.empty or "criteria_v2_prediction" not in df.columns:
        return df

    output = df.copy()
    output["criteria_v4_base_prediction"] = output["criteria_v2_prediction"].map(canonical_prediction_label)
    output["criteria_v4_context_state"] = ""
    output["criteria_v4_prediction"] = output["criteria_v4_base_prediction"]
    output["criteria_v4_reason"] = "criteria_v2_retained"
    output["criteria_v4_transition_evidence"] = ""

    sitting_baseline = finite_float(baselines.get("sitting_baseline_z_m"))
    standing_baseline = finite_float(baselines.get("standing_baseline_z_m"))
    posture_range = (
        abs(float(standing_baseline) - float(sitting_baseline))
        if sitting_baseline is not None and standing_baseline is not None
        else 0.40
    )
    delta_2_min = max(0.18, min(0.30, 0.40 * posture_range))
    delta_5_min = max(0.16, min(0.30, 0.40 * posture_range))
    delta_10_min = max(0.25, min(0.45, 0.65 * posture_range))
    unreliable_global_posture_band = posture_range >= 1.20

    sort_candidates = [
        column
        for column in ["window_s", "clip_idx", "center_s", "clip_start_s", "start_s"]
        if column in output.columns
    ]
    if sort_candidates:
        working = output.sort_values(sort_candidates).copy()
    else:
        working = output.copy()

    group_keys = ["window_s"] if "window_s" in working.columns else [None]
    grouped = working.groupby(group_keys, sort=False, dropna=False) if group_keys != [None] else [(None, working)]

    for _, group in grouped:
        indices = list(group.index)
        bases = [canonical_prediction_label(working.at[index, "criteria_v4_base_prediction"]) for index in indices]
        raw_targets: list[str] = [""] * len(indices)
        evidence_by_pos: list[str] = [""] * len(indices)

        for pos, index in enumerate(indices):
            if bases[pos] not in {"sitting", "standing"}:
                continue
            row = working.loc[index]
            horizontal_path = numeric_feature(row, "horizontal_path_length_m")
            if horizontal_path > 0.80:
                continue

            delta_2 = numeric_feature(row, "window_height_delta_m")
            slope_2 = numeric_feature(row, "height_slope_mps")
            delta_5 = numeric_feature(row, "context5_height_delta_m")
            slope_5 = numeric_feature(row, "context5_height_slope_mps")
            delta_10 = numeric_feature(row, "context10_height_delta_m")
            slope_10 = numeric_feature(row, "context10_height_slope_mps")

            direction = 0
            evidence = ""
            if abs(delta_2) >= delta_2_min and abs(slope_2) >= 0.06:
                direction = 1 if delta_2 > 0 else -1
                evidence = "2s_height_change"
            elif abs(delta_5) >= delta_5_min and abs(slope_5) >= 0.035:
                direction = 1 if delta_5 > 0 else -1
                evidence = "5s_height_context"
            elif abs(delta_10) >= delta_10_min and abs(slope_10) >= 0.035:
                direction = 1 if delta_10 > 0 else -1
                evidence = "10s_height_context"

            if direction:
                raw_targets[pos] = "standing" if direction > 0 else "sitting"
                evidence_by_pos[pos] = evidence

        # Collapse adjacent detections of the same physical transition into one
        # state update. One empty clip is tolerated because slow transitions can
        # cross a fixed clip boundary.
        events: dict[int, tuple[str, str]] = {}
        pos = 0
        while pos < len(indices):
            if not raw_targets[pos]:
                pos += 1
                continue
            target = raw_targets[pos]
            candidate_positions = [pos]
            cursor = pos + 1
            tolerated_gap = 0
            while cursor < len(indices):
                if raw_targets[cursor] == target:
                    candidate_positions.append(cursor)
                    tolerated_gap = 0
                    cursor += 1
                elif not raw_targets[cursor] and tolerated_gap < 1:
                    tolerated_gap += 1
                    cursor += 1
                else:
                    break

            peak_pos = max(
                candidate_positions,
                key=lambda candidate: max(
                    abs(numeric_feature(working.loc[indices[candidate]], "height_slope_mps")),
                    abs(numeric_feature(working.loc[indices[candidate]], "context5_height_slope_mps")),
                ),
            )
            events[peak_pos] = (target, evidence_by_pos[peak_pos] or "multi_scale_height_context")
            pos = max(cursor, pos + 1)

        posture_state = bases[0] if bases and bases[0] in {"sitting", "standing"} else "standing"
        for pos, index in enumerate(indices):
            base = bases[pos]
            if base in {"walking", "upstairs", "downstairs"}:
                working.at[index, "criteria_v4_context_state"] = "standing"
                working.at[index, "criteria_v4_prediction"] = base
                working.at[index, "criteria_v4_reason"] = "criteria_v2_locomotion_retained"
                posture_state = "standing"
                continue

            event = events.get(pos)
            if event is not None:
                target, evidence = event
                working.at[index, "criteria_v4_context_state"] = posture_state
                working.at[index, "criteria_v4_transition_evidence"] = evidence
                working.at[index, "criteria_v4_reason"] = "posture_transition_boundary"
                posture_state = target
                continue

            working.at[index, "criteria_v4_context_state"] = posture_state
            if base == "sitting" and posture_state == "standing":
                working.at[index, "criteria_v4_prediction"] = "standing"
                working.at[index, "criteria_v4_reason"] = "context_overrides_global_low_height"
            elif (
                base == "standing"
                and posture_state == "sitting"
                and unreliable_global_posture_band
            ):
                working.at[index, "criteria_v4_prediction"] = "sitting"
                working.at[index, "criteria_v4_reason"] = "context_overrides_unreliable_global_band"

    for column in [
        "criteria_v4_base_prediction",
        "criteria_v4_context_state",
        "criteria_v4_prediction",
        "criteria_v4_reason",
        "criteria_v4_transition_evidence",
    ]:
        output.loc[working.index, column] = working[column]
    return output


def prediction_matches(prediction: str, actions: list[str]) -> tuple[bool | None, bool | None, bool | None]:
    normalized = canonical_prediction_label(prediction)
    if not actions:
        return None, None, None
    target = canonical_transition_label(actions)
    exact = normalized == target
    component_match = normalized in actions
    transition_inclusive = exact
    return exact, transition_inclusive, component_match


def summarize_accuracy(rows: pd.DataFrame, prediction_col: str) -> dict[str, Any]:
    comparable = rows[rows["effective_manual_action_count"].fillna(0).astype(int) > 0].copy()
    if comparable.empty:
        return {
            "comparable_clip_count": 0,
            "stable_only_comparable_clip_count": 0,
            "transition_clip_comparable_count": 0,
            "strict_accuracy": None,
            "transition_aware_accuracy": None,
            "component_transition_aware_accuracy": None,
            "stable_only_accuracy": None,
            "transition_clip_accuracy": None,
        }
    strict_values = []
    transition_values = []
    component_values = []
    stable_values = []
    transition_clip_values = []
    for _, row in comparable.iterrows():
        actions = [normalize_motion_label(row.get(f"effective_action_{idx}", "")) for idx in range(1, 6)]
        actions = [action for action in actions if action]
        strict, transition_aware, component_match = prediction_matches(str(row.get(prediction_col, "")), actions)
        if strict is not None:
            strict_values.append(bool(strict))
            transition_values.append(bool(transition_aware))
            component_values.append(bool(component_match))
        if len(actions) == 1 and strict is not None:
            stable_values.append(bool(strict))
        if len(actions) > 1 and strict is not None:
            transition_clip_values.append(bool(strict))
    return {
        "comparable_clip_count": len(strict_values),
        "stable_only_comparable_clip_count": len(stable_values),
        "transition_clip_comparable_count": len(transition_clip_values),
        "strict_accuracy": float(np.mean(strict_values)) if strict_values else None,
        "transition_aware_accuracy": float(np.mean(transition_values)) if transition_values else None,
        "component_transition_aware_accuracy": float(np.mean(component_values)) if component_values else None,
        "stable_only_accuracy": float(np.mean(stable_values)) if stable_values else None,
        "transition_clip_accuracy": float(np.mean(transition_clip_values)) if transition_clip_values else None,
    }


def build_clip_comparison(
    trial_id: str,
    traj: pd.DataFrame,
    review_df: pd.DataFrame,
    baselines: dict[str, Any],
    intervals: list[ReferenceInterval],
    gap_threshold_s: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if review_df.empty:
        return pd.DataFrame(rows)
    duration = float(traj["session_s"].max())
    for _, review_row in review_df.iterrows():
        start_s = float(review_row["clip_start_s"])
        end_s = float(review_row["clip_end_s"])
        feature_row = feature_for_interval(traj, start_s, end_s, baselines, intervals, gap_threshold_s)
        if feature_row is None:
            continue
        center_s = (start_s + end_s) / 2.0
        context_5 = feature_for_centered_context(traj, center_s, 5.0, duration, baselines, intervals, gap_threshold_s)
        context_10 = feature_for_centered_context(traj, center_s, 10.0, duration, baselines, intervals, gap_threshold_s)
        actions = actions_from_review_row(review_row)
        old_prediction = normalize_motion_label(review_row.get("predicted_label", ""))
        improved_prediction = classify_improved_rule(feature_row, baselines)
        criteria_v2_prediction = classify_motion_criteria_v2(feature_row, baselines)
        old_strict, old_transition, old_component = prediction_matches(old_prediction, actions)
        new_strict, new_transition, new_component = prediction_matches(improved_prediction, actions)
        criteria_v2_strict, criteria_v2_transition, criteria_v2_component = prediction_matches(
            criteria_v2_prediction,
            actions,
        )

        output = {
            "trial_id": trial_id,
            "clip_idx": int(review_row["clip_idx"]),
            "clip_start_s": start_s,
            "clip_end_s": end_s,
            "old_prediction": old_prediction,
            "improved_rule_prediction": improved_prediction,
            "criteria_v2_prediction": criteria_v2_prediction,
            "criteria_target_label": canonical_transition_label(actions),
            "effective_manual_label": format_actions(actions),
            "effective_manual_action_count": len(actions),
            "old_strict_match": "" if old_strict is None else ("yes" if old_strict else "no"),
            "old_transition_aware_match": "" if old_transition is None else ("yes" if old_transition else "no"),
            "old_component_transition_aware_match": ""
            if old_component is None
            else ("yes" if old_component else "no"),
            "improved_strict_match": "" if new_strict is None else ("yes" if new_strict else "no"),
            "improved_transition_aware_match": "" if new_transition is None else ("yes" if new_transition else "no"),
            "improved_component_transition_aware_match": ""
            if new_component is None
            else ("yes" if new_component else "no"),
            "criteria_v2_strict_match": ""
            if criteria_v2_strict is None
            else ("yes" if criteria_v2_strict else "no"),
            "criteria_v2_transition_aware_match": ""
            if criteria_v2_transition is None
            else ("yes" if criteria_v2_transition else "no"),
            "criteria_v2_component_transition_aware_match": ""
            if criteria_v2_component is None
            else ("yes" if criteria_v2_component else "no"),
        }
        for idx in range(5):
            output[f"effective_action_{idx + 1}"] = actions[idx] if idx < len(actions) else ""
        output.update(feature_row)
        output["context5_height_slope_mps"] = "" if context_5 is None else context_5["height_slope_mps"]
        output["context5_height_delta_m"] = "" if context_5 is None else context_5["window_height_delta_m"]
        output["context5_horizontal_path_length_m"] = "" if context_5 is None else context_5["horizontal_path_length_m"]
        output["context10_height_slope_mps"] = "" if context_10 is None else context_10["height_slope_mps"]
        output["context10_height_delta_m"] = "" if context_10 is None else context_10["window_height_delta_m"]
        output["context10_horizontal_path_length_m"] = "" if context_10 is None else context_10["horizontal_path_length_m"]
        output["error_interpretation"] = interpret_error(output, baselines)
        rows.append(output)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = apply_transition_inclusive_predictions(result, baselines)
    result = apply_contextual_posture_predictions(result, baselines)
    v3_strict_values: list[str] = []
    v3_transition_values: list[str] = []
    v3_component_values: list[str] = []
    v4_strict_values: list[str] = []
    v4_transition_values: list[str] = []
    v4_component_values: list[str] = []
    for _, row in result.iterrows():
        actions = [normalize_motion_label(row.get(f"effective_action_{idx}", "")) for idx in range(1, 6)]
        actions = [action for action in actions if action]
        strict, transition_aware, component = prediction_matches(row.get("criteria_v3_prediction", ""), actions)
        v3_strict_values.append("" if strict is None else ("yes" if strict else "no"))
        v3_transition_values.append("" if transition_aware is None else ("yes" if transition_aware else "no"))
        v3_component_values.append("" if component is None else ("yes" if component else "no"))
        strict, transition_aware, component = prediction_matches(row.get("criteria_v4_prediction", ""), actions)
        v4_strict_values.append("" if strict is None else ("yes" if strict else "no"))
        v4_transition_values.append("" if transition_aware is None else ("yes" if transition_aware else "no"))
        v4_component_values.append("" if component is None else ("yes" if component else "no"))
    result["criteria_v3_strict_match"] = v3_strict_values
    result["criteria_v3_transition_aware_match"] = v3_transition_values
    result["criteria_v3_component_transition_aware_match"] = v3_component_values
    result["criteria_v4_strict_match"] = v4_strict_values
    result["criteria_v4_transition_aware_match"] = v4_transition_values
    result["criteria_v4_component_transition_aware_match"] = v4_component_values
    return result


def feature_for_centered_context(
    traj: pd.DataFrame,
    center_s: float,
    window_s: float,
    duration_s: float,
    baselines: dict[str, Any],
    intervals: list[ReferenceInterval],
    gap_threshold_s: float,
) -> dict[str, Any] | None:
    start_s = max(0.0, center_s - window_s / 2.0)
    end_s = min(duration_s, center_s + window_s / 2.0)
    if end_s - start_s < 0.25:
        return None
    return feature_for_interval(traj, start_s, end_s, baselines, intervals, gap_threshold_s)


def interpret_error(row: dict[str, Any], baselines: dict[str, Any]) -> str:
    actions = [row.get(f"effective_action_{idx}", "") for idx in range(1, 6)]
    actions = [normalize_motion_label(action) for action in actions if normalize_motion_label(action)]
    old_prediction = normalize_motion_label(row.get("old_prediction", ""))
    new_prediction = normalize_motion_label(row.get("improved_rule_prediction", ""))
    old_transition_match = row.get("old_transition_aware_match") == "yes"
    old_strict_match = row.get("old_strict_match") == "yes"

    if old_strict_match:
        return "old_rule_exact_match"
    if len(actions) > 1 and old_transition_match:
        return "transition_window_single_label_output"

    posture_band = finite_float(baselines.get("posture_z_band_m"))
    median_height = finite_float(row.get("window_median_height_m"))
    if posture_band is not None and median_height is not None:
        height_posture = "sitting" if median_height <= posture_band else "standing"
        if set(actions).issubset({"sitting", "standing"}) and height_posture not in actions:
            return "reference_label_conflicts_with_slam_height_band"
        if old_prediction in {"sitting", "standing"} and old_prediction != height_posture:
            return "posture_height_feature_definition"

    height_delta = float(row.get("window_height_delta_m", 0.0) or 0.0)
    horizontal_path = float(row.get("horizontal_path_length_m", 0.0) or 0.0)
    if any(action in {"upstairs", "downstairs"} for action in actions) and abs(height_delta) >= 0.18:
        return "stair_threshold_or_context_window"
    if "walking" in actions and horizontal_path >= 0.45:
        return "walking_motion_threshold"
    if new_prediction in actions and old_prediction not in actions:
        return "improved_rule_recovers_old_mismatch"
    return "needs_video_or_manual_review"


def write_trial_outputs(trial_id: str, args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RUNS_DIR / trial_id / "raw_parameter_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{trial_id}: loading synchronized SLAM and workbook labels", flush=True)
    traj, meta = make_session_trajectory(trial_id)
    traj = add_motion_derived_columns(traj)
    review_df = load_review_sheet(trial_id)
    intervals = build_reference_intervals(review_df)
    baselines = estimate_posture_baselines(review_df, traj)
    baselines["criteria_v2_thresholds"] = estimate_motion_rule_thresholds(traj)

    print(f"{trial_id}: writing reconstructed window features", flush=True)
    features_df = build_window_feature_table(
        traj=traj,
        baselines=baselines,
        intervals=intervals,
        window_sizes=args.window_sizes,
        step_s=args.window_step_s,
        gap_threshold_s=args.gap_threshold_s,
    )
    features_csv = out_dir / "reconstructed_motion_features.csv"
    features_df.to_csv(features_csv, index=False)

    print(f"{trial_id}: comparing old and improved rules on workbook clips", flush=True)
    clip_comparison_df = build_clip_comparison(
        trial_id=trial_id,
        traj=traj,
        review_df=review_df,
        baselines=baselines,
        intervals=intervals,
        gap_threshold_s=args.gap_threshold_s,
    )
    clip_comparison_csv = out_dir / "clip_feature_rule_comparison.csv"
    clip_comparison_df.to_csv(clip_comparison_csv, index=False)

    old_summary = summarize_accuracy(clip_comparison_df, "old_prediction") if not clip_comparison_df.empty else {}
    improved_summary = summarize_accuracy(clip_comparison_df, "improved_rule_prediction") if not clip_comparison_df.empty else {}
    criteria_v2_summary = summarize_accuracy(clip_comparison_df, "criteria_v2_prediction") if not clip_comparison_df.empty else {}
    criteria_v3_summary = summarize_accuracy(clip_comparison_df, "criteria_v3_prediction") if not clip_comparison_df.empty else {}
    criteria_v4_summary = summarize_accuracy(clip_comparison_df, "criteria_v4_prediction") if not clip_comparison_df.empty else {}
    confusion_csv = out_dir / "improved_rule_confusion.csv"
    criteria_v2_confusion_csv = out_dir / "criteria_v2_confusion.csv"
    criteria_v3_confusion_csv = out_dir / "criteria_v3_confusion.csv"
    criteria_v4_confusion_csv = out_dir / "criteria_v4_confusion.csv"
    if not clip_comparison_df.empty:
        confusion = (
            clip_comparison_df.groupby(["effective_manual_label", "improved_rule_prediction"], dropna=False)
            .size()
            .reset_index(name="clip_count")
        )
        confusion.to_csv(confusion_csv, index=False)
        criteria_v2_confusion = (
            clip_comparison_df.groupby(["effective_manual_label", "criteria_v2_prediction"], dropna=False)
            .size()
            .reset_index(name="clip_count")
        )
        criteria_v2_confusion.to_csv(criteria_v2_confusion_csv, index=False)
        criteria_v3_confusion = (
            clip_comparison_df.groupby(["effective_manual_label", "criteria_v3_prediction"], dropna=False)
            .size()
            .reset_index(name="clip_count")
        )
        criteria_v3_confusion.to_csv(criteria_v3_confusion_csv, index=False)
        criteria_v4_confusion = (
            clip_comparison_df.groupby(["effective_manual_label", "criteria_v4_prediction"], dropna=False)
            .size()
            .reset_index(name="clip_count")
        )
        criteria_v4_confusion.to_csv(criteria_v4_confusion_csv, index=False)
    else:
        pd.DataFrame(columns=["effective_manual_label", "improved_rule_prediction", "clip_count"]).to_csv(
            confusion_csv,
            index=False,
        )
        pd.DataFrame(columns=["effective_manual_label", "criteria_v2_prediction", "clip_count"]).to_csv(
            criteria_v2_confusion_csv,
            index=False,
        )
        pd.DataFrame(columns=["effective_manual_label", "criteria_v3_prediction", "clip_count"]).to_csv(
            criteria_v3_confusion_csv,
            index=False,
        )
        pd.DataFrame(columns=["effective_manual_label", "criteria_v4_prediction", "clip_count"]).to_csv(
            criteria_v4_confusion_csv,
            index=False,
        )

    ntrial_error_csv = ""
    if trial_id == "Ntrial1" and not clip_comparison_df.empty:
        error_df = clip_comparison_df[clip_comparison_df["old_strict_match"] != "yes"].copy()
        keep_cols = [
            "clip_idx",
            "clip_start_s",
            "clip_end_s",
            "old_prediction",
            "improved_rule_prediction",
            "criteria_v2_prediction",
            "criteria_v3_prediction",
            "criteria_v4_prediction",
            "criteria_v4_context_state",
            "criteria_v4_reason",
            "criteria_target_label",
            "effective_manual_label",
            "old_strict_match",
            "old_transition_aware_match",
            "improved_transition_aware_match",
            "criteria_v2_transition_aware_match",
            "criteria_v3_transition_aware_match",
            "window_mean_height_m",
            "window_median_height_m",
            "height_from_standing_baseline_m",
            "linear_speed_mean_mps",
            "horizontal_path_length_m",
            "net_horizontal_displacement_m",
            "window_height_delta_m",
            "height_slope_mps",
            "context5_height_slope_mps",
            "context10_height_slope_mps",
            "error_interpretation",
        ]
        ntrial_error_csv = str(out_dir / "ntrial1_prediction_error_analysis.csv")
        error_df[keep_cols].to_csv(ntrial_error_csv, index=False)

    summary = {
        "trial_id": trial_id,
        "sample_count": int(len(traj)),
        "duration_s": meta.get("duration_s"),
        "baselines": baselines,
        "feature_csv": str(features_csv),
        "clip_comparison_csv": str(clip_comparison_csv),
        "confusion_csv": str(confusion_csv),
        "criteria_v2_confusion_csv": str(criteria_v2_confusion_csv),
        "criteria_v3_confusion_csv": str(criteria_v3_confusion_csv),
        "criteria_v4_confusion_csv": str(criteria_v4_confusion_csv),
        "ntrial1_error_csv": ntrial_error_csv,
        "old_rule": old_summary,
        "improved_rule": improved_summary,
        "criteria_v2_rule": criteria_v2_summary,
        "criteria_v3_rule": criteria_v3_summary,
        "criteria_v4_rule": criteria_v4_summary,
    }
    summary_json = out_dir / "reconstructed_motion_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    summaries = [write_trial_outputs(trial_id, args) for trial_id in args.trials]
    combined_rows: list[dict[str, Any]] = []
    for summary in summaries:
        old = summary["old_rule"]
        improved = summary["improved_rule"]
        criteria_v2 = summary["criteria_v2_rule"]
        criteria_v3 = summary["criteria_v3_rule"]
        criteria_v4 = summary["criteria_v4_rule"]
        thresholds = summary["baselines"].get("criteria_v2_thresholds", {}) or {}
        row = {
            "trial_id": summary["trial_id"],
            "posture_baseline_method": summary["baselines"].get("posture_baseline_method"),
            "sitting_baseline_z_m": summary["baselines"].get("sitting_baseline_z_m"),
            "standing_baseline_z_m": summary["baselines"].get("standing_baseline_z_m"),
            "posture_z_band_m": summary["baselines"].get("posture_z_band_m"),
            "criteria_v2_walking_path_min_m": thresholds.get("walking_path_min_m"),
            "criteria_v2_walking_speed_min_mps": thresholds.get("walking_speed_min_mps"),
            "criteria_v2_stair_path_min_m": thresholds.get("stair_path_min_m"),
            "criteria_v2_stair_speed_min_mps": thresholds.get("stair_speed_min_mps"),
            "criteria_v2_stair_height_slope_min_mps": thresholds.get("stair_height_slope_min_mps"),
            "old_strict_accuracy": old.get("strict_accuracy"),
            "old_transition_aware_accuracy": old.get("transition_aware_accuracy"),
            "old_stable_only_accuracy": old.get("stable_only_accuracy"),
            "improved_strict_accuracy": improved.get("strict_accuracy"),
            "improved_transition_aware_accuracy": improved.get("transition_aware_accuracy"),
            "improved_stable_only_accuracy": improved.get("stable_only_accuracy"),
            "criteria_v2_strict_accuracy": criteria_v2.get("strict_accuracy"),
            "criteria_v2_transition_aware_accuracy": criteria_v2.get("transition_aware_accuracy"),
            "criteria_v2_component_transition_aware_accuracy": criteria_v2.get("component_transition_aware_accuracy"),
            "criteria_v2_stable_only_accuracy": criteria_v2.get("stable_only_accuracy"),
            "criteria_v2_transition_clip_accuracy": criteria_v2.get("transition_clip_accuracy"),
            "criteria_v3_strict_accuracy": criteria_v3.get("strict_accuracy"),
            "criteria_v3_transition_aware_accuracy": criteria_v3.get("transition_aware_accuracy"),
            "criteria_v3_component_transition_aware_accuracy": criteria_v3.get("component_transition_aware_accuracy"),
            "criteria_v3_stable_only_accuracy": criteria_v3.get("stable_only_accuracy"),
            "criteria_v3_transition_clip_accuracy": criteria_v3.get("transition_clip_accuracy"),
            "criteria_v4_strict_accuracy": criteria_v4.get("strict_accuracy"),
            "criteria_v4_transition_aware_accuracy": criteria_v4.get("transition_aware_accuracy"),
            "criteria_v4_component_transition_aware_accuracy": criteria_v4.get("component_transition_aware_accuracy"),
            "criteria_v4_stable_only_accuracy": criteria_v4.get("stable_only_accuracy"),
            "criteria_v4_transition_clip_accuracy": criteria_v4.get("transition_clip_accuracy"),
            "comparable_clip_count": improved.get("comparable_clip_count"),
            "stable_only_comparable_clip_count": improved.get("stable_only_comparable_clip_count"),
            "transition_clip_comparable_count": criteria_v3.get("transition_clip_comparable_count"),
            "feature_csv": summary["feature_csv"],
            "clip_comparison_csv": summary["clip_comparison_csv"],
            "criteria_v2_confusion_csv": summary["criteria_v2_confusion_csv"],
            "criteria_v3_confusion_csv": summary["criteria_v3_confusion_csv"],
            "criteria_v4_confusion_csv": summary["criteria_v4_confusion_csv"],
        }
        combined_rows.append(row)
    combined_df = pd.DataFrame(combined_rows)
    combined_csv = RUNS_DIR / "reconstructed_motion_feature_summary.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(json.dumps({"summary_csv": str(combined_csv), "trials": summaries}, indent=2))


if __name__ == "__main__":
    main()
