#!/usr/bin/env python3
"""Explore step/stride proxy features from raw SLAM head motion.

This script is intentionally conservative: Project Aria SLAM gives head/device
motion, not foot contacts. The outputs below should be read as step/stride
proxies and then checked against the synchronized downward video.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

matplotlib.use("Agg")
import matplotlib.pyplot as plt

QWEN_REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = QWEN_REPO / "runs"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_slam_parameters import (  # noqa: E402
    ACTION_DISPLAY_LABELS,
    load_effective_manual_intervals,
    make_session_trajectory,
)

LOCOMOTION_LABELS = {"walking", "upstairs", "downstairs"}
LABEL_COLORS = {
    "walking": "#3a9d55",
    "upstairs": "#e58f2a",
    "downstairs": "#c94c4c",
}
DEVICE_FORWARD_AXIS = np.array([0.0, 0.0, 1.0])


@dataclass
class BoutResult:
    bout: dict[str, Any]
    series: pd.DataFrame
    events: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", nargs="+", default=["Ntrial1", "Ntrial2", "Ntrial3"])
    parser.add_argument("--out-dir", default=str(RUNS_DIR / "stride_step_diagnostics"))
    parser.add_argument("--resample-fps", type=float, default=50.0)
    parser.add_argument("--bout-s", type=float, default=6.0)
    parser.add_argument("--bout-step-s", type=float, default=6.0)
    parser.add_argument("--min-bout-s", type=float, default=4.0)
    parser.add_argument("--make-contact-sheets", action="store_true", default=True)
    parser.add_argument("--no-contact-sheets", dest="make_contact_sheets", action="store_false")
    return parser.parse_args()


def rolling_mean(values: np.ndarray, window_samples: int) -> np.ndarray:
    window_samples = max(1, int(window_samples))
    if window_samples % 2 == 0:
        window_samples += 1
    return (
        pd.Series(values)
        .rolling(window_samples, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def merge_locomotion_intervals(states: pd.DataFrame, max_gap_s: float = 0.25) -> pd.DataFrame:
    """Merge adjacent manually labeled walking/stair intervals."""
    rows: list[dict[str, Any]] = []
    if states.empty:
        return pd.DataFrame(columns=["label", "start_s", "end_s"])

    for label in sorted(LOCOMOTION_LABELS):
        part = states[states["label"] == label].sort_values("start_s")
        current: dict[str, Any] | None = None
        for _, row in part.iterrows():
            start_s = float(row["start_s"])
            end_s = float(row["end_s"])
            if current is None:
                current = {"label": label, "start_s": start_s, "end_s": end_s}
                continue
            if start_s <= float(current["end_s"]) + max_gap_s:
                current["end_s"] = max(float(current["end_s"]), end_s)
            else:
                rows.append(current)
                current = {"label": label, "start_s": start_s, "end_s": end_s}
        if current is not None:
            rows.append(current)

    if not rows:
        return pd.DataFrame(columns=["label", "start_s", "end_s"])
    return pd.DataFrame(rows).sort_values(["start_s", "label"]).reset_index(drop=True)


def load_model_locomotion_states(trial_id: str) -> pd.DataFrame:
    """Fallback locomotion intervals from dashboard model predictions.

    Manual labels are preferable. For long new trials, manual labels may not
    exist yet, but the full-session dashboard already has clip-level motion
    predictions. Those predicted walking/stair regions are useful enough to
    create stride-review windows that can then be checked against video.
    """
    continuous_path = RUNS_DIR / trial_id / "raw_parameter_diagnostics/continuous_motion_segments.json"
    if continuous_path.exists():
        segments = json.loads(continuous_path.read_text()).get("segments", [])
        rows = [{"label": row["label"], "start_s": row["start_s"], "end_s": row["end_s"],
                 "source": row.get("source", "continuous_classification")}
                for row in segments if row.get("label") in LOCOMOTION_LABELS and row["end_s"] > row["start_s"]]
        if rows:
            return pd.DataFrame(rows)
    candidates = [
        RUNS_DIR / trial_id / "downward_master_session" / "downward_session_clip_metrics.csv",
        RUNS_DIR / trial_id / "clip_metrics.csv",
    ]
    metrics_path = next((path for path in candidates if path.exists()), None)
    if metrics_path is None:
        return pd.DataFrame(columns=["label", "start_s", "end_s"])

    metrics = pd.read_csv(metrics_path)
    label_column = next(
        (
            column
            for column in ["hybrid_prediction", "criteria_v4_prediction", "criteria_v3_prediction", "rule_prediction"]
            if column in metrics.columns
        ),
        None,
    )
    if label_column is None or not {"start_s", "end_s"}.issubset(metrics.columns):
        return pd.DataFrame(columns=["label", "start_s", "end_s"])

    rows: list[dict[str, Any]] = []
    for _, row in metrics.iterrows():
        label = str(row.get(label_column) or "").strip().lower()
        label = label.replace("go upstairs", "upstairs").replace("go downstairs", "downstairs")
        if label not in LOCOMOTION_LABELS:
            continue
        start_s = float(row.get("start_s") or 0.0)
        end_s = float(row.get("end_s") or start_s)
        if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
            continue
        rows.append(
            {
                "label": label,
                "start_s": start_s,
                "end_s": end_s,
                "source": f"model_{label_column}",
            }
        )
    if not rows:
        return pd.DataFrame(columns=["label", "start_s", "end_s"])
    return pd.DataFrame(rows)


def split_into_local_bouts(interval: pd.Series, bout_s: float, bout_step_s: float, min_bout_s: float) -> list[dict[str, Any]]:
    """Split longer locomotion intervals so a local forward direction is meaningful."""
    label = str(interval["label"])
    start_s = float(interval["start_s"])
    end_s = float(interval["end_s"])
    duration_s = end_s - start_s
    if duration_s < min_bout_s:
        if duration_s >= 1.0:
            return [
                {
                    "label": label,
                    "start_s": start_s,
                    "end_s": end_s,
                    "parent_start_s": start_s,
                    "parent_end_s": end_s,
                    "review_only": True,
                }
            ]
        return []
    if duration_s <= bout_s + 2.0:
        return [
            {
                "label": label,
                "start_s": start_s,
                "end_s": end_s,
                "parent_start_s": start_s,
                "parent_end_s": end_s,
                "review_only": False,
            }
        ]

    bouts: list[dict[str, Any]] = []
    cursor = start_s
    while cursor + min_bout_s <= end_s:
        bout_end_s = min(cursor + bout_s, end_s)
        if bout_end_s - cursor >= min_bout_s:
            bouts.append(
                {
                    "label": label,
                    "start_s": cursor,
                    "end_s": bout_end_s,
                    "parent_start_s": start_s,
                    "parent_end_s": end_s,
                    "review_only": False,
                }
            )
        cursor += bout_step_s
    return bouts


def add_session_horizontal_path(traj: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative session-level horizontal path for cross-window stride metrics."""
    output = traj.sort_values("session_s").copy()
    dx = output["tx_world_device"].diff().fillna(0.0)
    dy = output["ty_world_device"].diff().fillna(0.0)
    output["session_horizontal_path_m"] = np.hypot(dx, dy).cumsum()
    return output


def pca_forward_lateral_axes(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = xy - xy.mean(axis=0)
    cov = np.cov(centered.T)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    forward_axis = eig_vecs[:, int(np.argmax(eig_vals))]
    displacement = xy[-1] - xy[0]
    if np.dot(forward_axis, displacement) < 0:
        forward_axis = -forward_axis
    lateral_axis = np.array([-forward_axis[1], forward_axis[0]])
    return forward_axis, lateral_axis


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    return values / np.maximum(norms[:, None], 1e-9)


def rotate_vectors_by_quaternion(
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    qw: np.ndarray,
    vectors: np.ndarray,
) -> np.ndarray:
    """Rotate device-frame vectors into world-frame vectors.

    The MPS columns are named q*_world_device, so this treats the quaternion as
    the device-to-world rotation. We only use this as an initial left/right
    estimate and keep a confidence flag because head orientation can differ
    from torso/foot orientation.
    """
    q = np.column_stack([qx, qy, qz])
    qw_col = qw[:, None]
    q_norm = np.sqrt(np.sum(q * q, axis=1) + qw * qw)
    q = q / np.maximum(q_norm[:, None], 1e-9)
    qw_col = qw_col / np.maximum(q_norm[:, None], 1e-9)
    uv = np.cross(q, vectors)
    uuv = np.cross(q, uv)
    return vectors + 2.0 * (qw_col * uv + uuv)


def interpolate_device_forward_axes(
    t: np.ndarray,
    section: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    qx = np.interp(t, section["session_s"], section["qx_world_device"])
    qy = np.interp(t, section["session_s"], section["qy_world_device"])
    qz = np.interp(t, section["session_s"], section["qz_world_device"])
    qw = np.interp(t, section["session_s"], section["qw_world_device"])
    repeated_forward = np.repeat(DEVICE_FORWARD_AXIS[None, :], len(t), axis=0)
    device_forward_world = rotate_vectors_by_quaternion(qx, qy, qz, qw, repeated_forward)
    device_forward_xy = device_forward_world[:, :2]
    horizontal_norm = np.linalg.norm(device_forward_xy, axis=1)
    device_forward_xy_unit = device_forward_xy / np.maximum(horizontal_norm[:, None], 1e-9)
    return device_forward_xy_unit, horizontal_norm


def adaptive_path_lateral_sway(
    x: np.ndarray,
    y: np.ndarray,
    fps: float,
    fallback_forward_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Estimate lateral head sway relative to the local path direction.

    A fixed PCA direction works well for straight walking, but it can turn
    true forward motion into apparent lateral motion when the participant turns.
    This local path frame follows the smoothed trajectory direction over time.
    """
    center_window = max(3, round(0.75 * fps))
    residual_smooth_window = max(3, round(0.12 * fps))
    center_x = rolling_mean(x, center_window)
    center_y = rolling_mean(y, center_window)
    center_xy = np.column_stack([center_x, center_y])

    dx = np.gradient(center_x)
    dy = np.gradient(center_y)
    tangents = np.column_stack([dx, dy])
    norms = np.linalg.norm(tangents, axis=1)
    fallback = fallback_forward_axis / max(float(np.linalg.norm(fallback_forward_axis)), 1e-9)
    low_motion = norms < max(1e-5, np.nanpercentile(norms, 10) * 0.25)
    tangents[low_motion] = fallback
    norms = np.linalg.norm(tangents, axis=1)
    tangents = tangents / np.maximum(norms[:, None], 1e-9)

    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    residual_xy = np.column_stack([x, y]) - center_xy
    lateral_sway = np.sum(residual_xy * normals, axis=1)
    lateral_sway = rolling_mean(lateral_sway, residual_smooth_window)

    heading_rad = np.unwrap(np.arctan2(tangents[:, 1], tangents[:, 0]))
    heading_change_deg = float(abs(np.degrees(heading_rad[-1] - heading_rad[0]))) if len(heading_rad) > 1 else 0.0
    return lateral_sway, tangents, heading_rad, heading_change_deg


def classify_bout_confidence(
    event_count: int,
    valid_step_count: int,
    cadence_steps_per_s: float,
    straightness: float,
    lateral_range_m: float,
    alternation_rate: float,
    step_cv: float | None,
    heading_change_deg: float,
) -> tuple[str, str]:
    reasons: list[str] = []
    if event_count < 5:
        reasons.append("few lateral-sway events")
    if not 0.6 <= cadence_steps_per_s <= 2.4:
        reasons.append("cadence outside broad human walking range")
    if valid_step_count < 3:
        reasons.append("few valid step-length estimates")
    if straightness < 0.45:
        reasons.append("strong turning/curved route; side direction is less certain")
    if heading_change_deg > 65:
        reasons.append("large heading change in bout")
    if lateral_range_m < 0.018:
        reasons.append("weak lateral head-sway signal")
    if alternation_rate < 0.65 and event_count >= 4:
        reasons.append("lateral extrema do not alternate cleanly")
    if step_cv is not None and step_cv > 0.60:
        reasons.append("step-proxy lengths vary strongly inside the bout")

    high_quality_steps = step_cv is None or step_cv <= 0.55
    high_quality_direction = straightness >= 0.55 and heading_change_deg <= 65
    if (
        event_count >= 8
        and valid_step_count >= 6
        and 0.8 <= cadence_steps_per_s <= 2.2
        and lateral_range_m >= 0.018
        and alternation_rate >= 0.70
        and high_quality_steps
        and high_quality_direction
    ):
        return "high", ""
    if event_count >= 5 and valid_step_count >= 3 and 0.6 <= cadence_steps_per_s <= 2.4:
        return "medium", "; ".join(reasons)
    return "low", "; ".join(reasons)


def classify_event_confidence(
    event: pd.Series,
    *,
    prominence_threshold_m: float,
    lateral_range_m: float,
    straightness: float,
    heading_change_deg: float,
) -> tuple[str, str]:
    """Score one detected step-like event so uncertain small/turning steps are visible."""
    score = 0
    reasons: list[str] = []
    ratio = float(event.get("prominence_m", 0.0)) / max(prominence_threshold_m, 1e-9)
    if ratio >= 1.8:
        score += 2
    elif ratio >= 1.15:
        score += 1
    else:
        reasons.append("weak peak prominence")

    interval = event.get("inter_event_interval_s")
    if pd.notna(interval):
        interval = float(interval)
        if 0.28 <= interval <= 1.35:
            score += 1
        else:
            reasons.append("unusual time gap from previous step-like event")
    else:
        reasons.append("first event in bout has no previous-step spacing")

    alternates = event.get("alternates_from_previous")
    if pd.notna(alternates):
        if bool(alternates):
            score += 1
        else:
            reasons.append("same-side repeat instead of clean alternation")

    step_length = event.get("valid_step_path_length_m")
    if pd.notna(step_length):
        step_length = float(step_length)
        score += 1
        if step_length < 0.12:
            reasons.append("small step-proxy length; validate carefully")
    else:
        reasons.append("step length outside accepted range")

    if lateral_range_m < 0.018:
        reasons.append("weak lateral head-sway range")
        score -= 1
    if straightness < 0.45 or heading_change_deg > 65:
        reasons.append("turning context; side direction is less certain")
        score -= 1

    if pd.notna(step_length) and float(step_length) < 0.07:
        return "low", "; ".join(reasons)
    if score >= 4 and not (straightness < 0.45 or heading_change_deg > 85):
        return "high", "; ".join(reasons)
    if score >= 2:
        return "medium", "; ".join(reasons)
    return "low", "; ".join(reasons)


def estimate_foot_side_from_sway(
    side_proxy: str,
    event_confidence: str,
    *,
    path_left_axis_xy: np.ndarray,
    device_left_axis_xy: np.ndarray,
    device_forward_horizontal_norm: float,
    device_path_forward_alignment: float,
    heading_change_deg: float,
) -> tuple[str, str, float, str]:
    """Estimate anatomical foot side from SLAM-only head sway.

    This assumes lateral head sway points roughly toward the stance/loading
    side. It is intentionally confidence-scored because head orientation is not
    always torso/foot orientation, especially while turning.
    """
    side_sign = 1.0 if "positive" in str(side_proxy) else -1.0
    sway_axis_xy = side_sign * path_left_axis_xy
    alignment = float(np.dot(sway_axis_xy, device_left_axis_xy))
    estimated_side = "left" if alignment >= 0 else "right"
    side_alignment_abs = abs(alignment)

    score = 0
    warnings: list[str] = []
    if side_alignment_abs >= 0.75:
        score += 2
    elif side_alignment_abs >= 0.45:
        score += 1
        warnings.append("head/device left axis only moderately matches path-side direction")
    else:
        warnings.append("head/device orientation does not clearly map path-side to left/right")

    if event_confidence == "high":
        score += 2
    elif event_confidence == "medium":
        score += 1
    else:
        warnings.append("step event itself is low confidence")

    if device_forward_horizontal_norm < 0.40:
        score -= 1
        warnings.append("device forward axis is not strongly horizontal")
    if device_path_forward_alignment < 0.45:
        score -= 1
        warnings.append("head/device forward direction differs from local walking direction")
    if heading_change_deg > 65:
        score -= 1
        warnings.append("turning/head-orientation context makes left/right less certain")

    if score >= 3:
        confidence = "high"
    elif score >= 1:
        confidence = "medium"
    else:
        confidence = "low"
    return estimated_side, confidence, alignment, "; ".join(warnings)


def analyze_bout(
    trial_id: str,
    bout_id: int,
    traj: pd.DataFrame,
    bout: dict[str, Any],
    fps: float,
) -> BoutResult | None:
    start_s = float(bout["start_s"])
    end_s = float(bout["end_s"])
    duration_s = end_s - start_s
    section = traj[(traj["session_s"] >= start_s) & (traj["session_s"] <= end_s)].copy()
    if section.empty or len(section) < 50:
        return None

    t = np.arange(start_s, end_s, 1.0 / fps)
    if len(t) < max(10, int(0.5 * fps)):
        return None

    x = np.interp(t, section["session_s"], section["tx_world_device"])
    y = np.interp(t, section["session_s"], section["ty_world_device"])
    z = np.interp(t, section["session_s"], section["tz_world_device"])
    speed = np.interp(t, section["session_s"], section["linear_speed_mps"])
    session_horizontal_path = np.interp(t, section["session_s"], section["session_horizontal_path_m"])
    device_forward_xy_unit, device_forward_horizontal_norm = interpolate_device_forward_axes(t, section)
    xy = np.column_stack([x, y])
    step_distances = np.hypot(np.diff(x), np.diff(y))
    horizontal_path_progress = np.concatenate([[0.0], np.cumsum(step_distances)])
    horizontal_path_length_m = float(horizontal_path_progress[-1])
    net_horizontal_displacement_m = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    straightness = net_horizontal_displacement_m / horizontal_path_length_m if horizontal_path_length_m > 1e-9 else 0.0
    review_reasons: list[str] = []
    if bool(bout.get("review_only")):
        review_reasons.append("short labeled interval kept for manual review")
    if horizontal_path_length_m < 1.0:
        review_reasons.append("horizontal path below automatic stride-bout threshold")
    if net_horizontal_displacement_m < 0.35:
        review_reasons.append("net horizontal displacement below automatic stride-bout threshold")

    forward_axis, _lateral_axis = pca_forward_lateral_axes(xy)
    fixed_axis_forward_progress = xy @ forward_axis
    fixed_axis_forward_progress = fixed_axis_forward_progress - fixed_axis_forward_progress[0]
    forward_progress = horizontal_path_progress.copy()
    lateral_sway_m, local_tangents, local_heading_rad, heading_change_deg = adaptive_path_lateral_sway(
        x,
        y,
        fps,
        forward_axis,
    )
    path_left_axes = np.column_stack([-local_tangents[:, 1], local_tangents[:, 0]])
    device_path_forward_alignment_signed = np.sum(device_forward_xy_unit * local_tangents, axis=1)
    device_forward_xy_path_aligned = device_forward_xy_unit.copy()
    device_forward_xy_path_aligned[device_path_forward_alignment_signed < 0] *= -1.0
    device_path_forward_alignment = np.abs(device_path_forward_alignment_signed)
    device_left_xy_unit = np.column_stack(
        [-device_forward_xy_path_aligned[:, 1], device_forward_xy_path_aligned[:, 0]]
    )
    lateral_range_m = float(np.percentile(lateral_sway_m, 95) - np.percentile(lateral_sway_m, 5))
    prominence_m = max(0.006, 0.14 * lateral_range_m)
    min_event_distance_samples = max(1, round(0.24 * fps))

    positive_peaks, positive_props = find_peaks(
        lateral_sway_m,
        prominence=prominence_m,
        distance=min_event_distance_samples,
    )
    negative_peaks, negative_props = find_peaks(
        -lateral_sway_m,
        prominence=prominence_m,
        distance=min_event_distance_samples,
    )

    raw_events: list[dict[str, Any]] = []
    for peak_index, peak in enumerate(positive_peaks):
        raw_events.append(
            {
                "sample_index": int(peak),
                "side_proxy": "lateral_positive",
                "prominence_m": float(positive_props["prominences"][peak_index]),
            }
        )
    for peak_index, peak in enumerate(negative_peaks):
        raw_events.append(
            {
                "sample_index": int(peak),
                "side_proxy": "lateral_negative",
                "prominence_m": float(negative_props["prominences"][peak_index]),
            }
        )
    raw_events = sorted(raw_events, key=lambda event: event["sample_index"])

    # Keep alternating left/right-like lateral extrema; if two same-side peaks are
    # too close together, keep the stronger one.
    filtered_events: list[dict[str, Any]] = []
    same_side_merge_samples = round(0.70 * fps)
    min_spacing_samples = round(0.18 * fps)
    for event in raw_events:
        if not filtered_events:
            filtered_events.append(event)
            continue
        previous = filtered_events[-1]
        spacing = int(event["sample_index"]) - int(previous["sample_index"])
        if event["side_proxy"] == previous["side_proxy"] and spacing < same_side_merge_samples:
            if float(event["prominence_m"]) > float(previous["prominence_m"]):
                filtered_events[-1] = event
        elif spacing >= min_spacing_samples:
            filtered_events.append(event)

    # The lateral extrema are head-sway peaks/troughs. A visually observed foot
    # contact usually falls between two neighboring extrema, so the primary
    # event time below is the midpoint between adjacent sway extrema.
    event_rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(filtered_events[1:]):
        previous_event = filtered_events[event_index]
        peak_sample_index = int(event["sample_index"])
        previous_peak_sample_index = int(previous_event["sample_index"])
        sample_index = int(round((previous_peak_sample_index + peak_sample_index) / 2.0))
        sample_index = max(0, min(sample_index, len(t) - 1))
        session_s = float(t[sample_index])
        row = {
            "trial_id": trial_id,
            "bout_id": bout_id,
            "label": str(bout["label"]),
            "event_index": event_index,
            "session_s": session_s,
            "timing_method": "midpoint_between_lateral_extrema",
            "previous_sway_peak_session_s": float(t[previous_peak_sample_index]),
            "sway_peak_session_s": float(t[peak_sample_index]),
            "previous_sway_peak_side_proxy": previous_event["side_proxy"],
            "sway_peak_side_proxy": event["side_proxy"],
            "previous_sway_peak_prominence_m": float(previous_event["prominence_m"]),
            "sway_peak_prominence_m": float(event["prominence_m"]),
            "side_proxy": event["side_proxy"],
            "prominence_m": float(min(previous_event["prominence_m"], event["prominence_m"])),
            "forward_progress_m": float(forward_progress[sample_index]),
            "horizontal_path_progress_m": float(horizontal_path_progress[sample_index]),
            "lateral_sway_m": float(lateral_sway_m[sample_index]),
            "local_heading_deg": float(np.degrees(local_heading_rad[sample_index])),
            "heading_change_in_bout_deg": float(heading_change_deg),
            "device_path_forward_alignment": float(device_path_forward_alignment[sample_index]),
            "device_forward_horizontal_norm": float(device_forward_horizontal_norm[sample_index]),
            "tx_world_device_m": float(x[sample_index]),
            "ty_world_device_m": float(y[sample_index]),
            "tz_world_device_m": float(z[sample_index]),
            "session_horizontal_path_m": float(session_horizontal_path[sample_index]),
            "linear_speed_mps": float(speed[sample_index]),
        }
        event_rows.append(row)

    events = pd.DataFrame(event_rows)
    if events.empty:
        cadence_steps_per_s = 0.0
        valid_step_count = 0
    else:
        events["step_forward_length_m"] = events["forward_progress_m"].diff()
        events["step_path_length_m"] = events["horizontal_path_progress_m"].diff()
        events["stride_forward_length_m"] = events["forward_progress_m"].diff(2)
        events["stride_path_length_m"] = events["horizontal_path_progress_m"].diff(2)
        events["inter_event_interval_s"] = events["session_s"].diff()
        events["alternates_from_previous"] = events["side_proxy"].ne(events["side_proxy"].shift()).astype(object)
        events.loc[events.index[0], "alternates_from_previous"] = None
        valid_step = events["step_path_length_m"].where(events["step_path_length_m"].between(0.03, 1.60))
        valid_stride = events["stride_path_length_m"].where(events["stride_path_length_m"].between(0.08, 3.20))
        events["valid_step_path_length_m"] = valid_step
        events["valid_stride_path_length_m"] = valid_stride
        events["rolling_last4_step_path_length_m"] = valid_step.rolling(4, min_periods=1).mean()
        events["rolling_last4_step_forward_length_m"] = (
            events["step_forward_length_m"].where(events["step_forward_length_m"].between(0.03, 1.60)).rolling(4, min_periods=1).mean()
        )
        cadence_steps_per_s = len(events) / duration_s if duration_s > 0 else 0.0
        valid_step_count = int(valid_step.notna().sum())

    valid_steps = events["valid_step_path_length_m"].dropna() if "valid_step_path_length_m" in events else pd.Series(dtype=float)
    valid_strides = events["valid_stride_path_length_m"].dropna() if "valid_stride_path_length_m" in events else pd.Series(dtype=float)
    if len(events) > 1 and "alternates_from_previous" in events:
        alternation_rate = float(events["alternates_from_previous"].dropna().astype(bool).mean())
    else:
        alternation_rate = 0.0
    step_cv = (
        float(valid_steps.std() / valid_steps.mean())
        if len(valid_steps) >= 3 and float(valid_steps.mean()) > 1e-9
        else None
    )

    confidence, warning = classify_bout_confidence(
        event_count=int(len(events)),
        valid_step_count=valid_step_count,
        cadence_steps_per_s=float(cadence_steps_per_s),
        straightness=float(straightness),
        lateral_range_m=lateral_range_m,
        alternation_rate=alternation_rate,
        step_cv=step_cv,
        heading_change_deg=heading_change_deg,
    )
    if review_reasons or events.empty:
        confidence = "review"
        warning_parts = review_reasons.copy()
        if events.empty:
            warning_parts.append("no step-like events detected; kept for video review")
        if warning:
            warning_parts.append(warning)
        warning = "; ".join(part for part in warning_parts if part)

    if not events.empty:
        event_confidences = events.apply(
            lambda row: classify_event_confidence(
                row,
                prominence_threshold_m=prominence_m,
                lateral_range_m=lateral_range_m,
                straightness=float(straightness),
                heading_change_deg=heading_change_deg,
            ),
            axis=1,
        )
        events["event_confidence"] = [value[0] for value in event_confidences]
        events["event_warning"] = [value[1] for value in event_confidences]
        foot_side_estimates: list[tuple[str, str, float, str]] = []
        for _, row in events.iterrows():
            sample_index = int(round((float(row["session_s"]) - start_s) * fps))
            sample_index = max(0, min(sample_index, len(t) - 1))
            foot_side_estimates.append(
                estimate_foot_side_from_sway(
                    str(row["side_proxy"]),
                    str(row["event_confidence"]),
                    path_left_axis_xy=path_left_axes[sample_index],
                    device_left_axis_xy=device_left_xy_unit[sample_index],
                    device_forward_horizontal_norm=float(device_forward_horizontal_norm[sample_index]),
                    device_path_forward_alignment=float(device_path_forward_alignment[sample_index]),
                    heading_change_deg=heading_change_deg,
                )
            )
        events["estimated_foot_side"] = [value[0] for value in foot_side_estimates]
        events["foot_side_confidence"] = [value[1] for value in foot_side_estimates]
        events["foot_side_alignment"] = [value[2] for value in foot_side_estimates]
        events["foot_side_warning"] = [value[3] for value in foot_side_estimates]
        events["bout_confidence"] = confidence
        events["bout_warning"] = warning
        events["bout_start_s"] = start_s
        events["bout_end_s"] = end_s

    bout_summary = {
        "trial_id": trial_id,
        "bout_id": bout_id,
        "label": str(bout["label"]),
        "bout_start_s": start_s,
        "bout_end_s": end_s,
        "duration_s": duration_s,
        "parent_interval_start_s": float(bout["parent_start_s"]),
        "parent_interval_end_s": float(bout["parent_end_s"]),
        "horizontal_path_length_m": horizontal_path_length_m,
        "net_horizontal_displacement_m": net_horizontal_displacement_m,
        "straightness_net_over_path": straightness,
        "lateral_range_m": lateral_range_m,
        "peak_prominence_threshold_m": prominence_m,
        "heading_change_deg": float(heading_change_deg),
        "alternation_rate": alternation_rate,
        "step_length_cv": step_cv,
        "detected_lateral_extrema": int(len(filtered_events)),
        "detected_step_events": int(len(events)),
        "detected_lateral_events": int(len(events)),
        "valid_step_count": int(valid_step_count),
        "review_window": bool(confidence == "review" or valid_step_count == 0),
        "window_kind": "review_window" if confidence == "review" or valid_step_count == 0 else "stride_bout",
        "cadence_steps_per_s": float(cadence_steps_per_s),
        "median_step_path_length_m": float(valid_steps.median()) if not valid_steps.empty else math.nan,
        "p10_step_path_length_m": float(valid_steps.quantile(0.10)) if len(valid_steps) >= 3 else math.nan,
        "p90_step_path_length_m": float(valid_steps.quantile(0.90)) if len(valid_steps) >= 3 else math.nan,
        "mean_step_path_length_m": float(valid_steps.mean()) if not valid_steps.empty else math.nan,
        "median_stride_path_length_m": float(valid_strides.median()) if not valid_strides.empty else math.nan,
        "rolling_last4_step_path_length_end_m": (
            float(events["rolling_last4_step_path_length_m"].dropna().iloc[-1])
            if "rolling_last4_step_path_length_m" in events and events["rolling_last4_step_path_length_m"].notna().any()
            else math.nan
        ),
        "confidence": confidence,
        "warning": warning,
    }

    series = pd.DataFrame(
        {
            "trial_id": trial_id,
            "bout_id": bout_id,
            "label": str(bout["label"]),
            "session_s": t,
            "tx_world_device": x,
            "ty_world_device": y,
            "tz_world_device": z,
            "linear_speed_mps": speed,
            "forward_progress_m": forward_progress,
            "fixed_axis_forward_progress_m": fixed_axis_forward_progress,
            "horizontal_path_progress_m": horizontal_path_progress,
            "lateral_sway_m": lateral_sway_m,
            "local_heading_deg": np.degrees(local_heading_rad),
            "device_forward_x_world": device_forward_xy_path_aligned[:, 0],
            "device_forward_y_world": device_forward_xy_path_aligned[:, 1],
            "device_left_x_world": device_left_xy_unit[:, 0],
            "device_left_y_world": device_left_xy_unit[:, 1],
            "device_forward_horizontal_norm": device_forward_horizontal_norm,
            "device_path_forward_alignment": device_path_forward_alignment,
        }
    )
    return BoutResult(bout=bout_summary, series=series, events=events)


def plot_bout_diagnostics(result: BoutResult, out_dir: Path) -> None:
    bout = result.bout
    series = result.series
    events = result.events
    label = str(bout["label"])
    display = ACTION_DISPLAY_LABELS.get(label, label.title())
    prefix = f"{bout['trial_id']}_bout{int(bout['bout_id']):02d}_{label}_{bout['bout_start_s']:.1f}-{bout['bout_end_s']:.1f}s"
    safe_prefix = prefix.replace(".", "p")

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(series["session_s"], series["lateral_sway_m"], color="#1f2d3a", linewidth=1.2)
    if not events.empty:
        colors = events["side_proxy"].map({"lateral_positive": "#2f78b7", "lateral_negative": "#c94c4c"}).fillna("#333333")
        axes[0].scatter(events["session_s"], events["lateral_sway_m"], c=colors, s=38, zorder=4, label="estimated step timing")
        if "sway_peak_session_s" in events:
            peak_y = np.interp(events["sway_peak_session_s"], series["session_s"], series["lateral_sway_m"])
            axes[0].scatter(events["sway_peak_session_s"], peak_y, c=colors, s=18, marker="x", zorder=4, label="later head-sway extremum")
    axes[0].axhline(0.0, color="#888888", linewidth=0.8, alpha=0.5)
    axes[0].set_ylabel("lateral head sway (m)")
    axes[0].set_title(
        f"{bout['trial_id']} {display} {bout['bout_start_s']:.1f}-{bout['bout_end_s']:.1f}s: midpoint step proxy from lateral sway"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper left", frameon=True)

    axes[1].plot(series["session_s"], series["horizontal_path_progress_m"], color="#3a9d55", linewidth=1.3, label="horizontal path progress")
    axes[1].plot(series["session_s"], series["forward_progress_m"], color="#7a4fb3", linewidth=1.1, alpha=0.85, label="local forward progress")
    if not events.empty:
        axes[1].scatter(events["session_s"], events["horizontal_path_progress_m"], c="#111111", s=24, zorder=4)
    axes[1].set_ylabel("progress (m)")
    axes[1].legend(loc="upper left", frameon=True)

    if "rolling_last4_step_path_length_m" in events and events["rolling_last4_step_path_length_m"].notna().any():
        axes[2].plot(
            events["session_s"],
            events["rolling_last4_step_path_length_m"],
            marker="o",
            color="#e58f2a",
            linewidth=1.4,
            label="rolling average of last 4 step-proxy lengths",
        )
        axes[2].scatter(events["session_s"], events["valid_step_path_length_m"], color="#3a9d55", s=28, alpha=0.75, label="single step-proxy length")
    axes[2].set_ylabel("length (m)")
    axes[2].set_xlabel("session time (s)")
    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        axes[2].legend(handles, labels, loc="upper left", frameon=True)
    axes[2].text(
        0.01,
        0.04,
        f"Confidence: {bout['confidence']} | cadence {bout['cadence_steps_per_s']:.2f} events/s | "
        f"median step-proxy {bout['median_step_path_length_m']:.3f} m",
        transform=axes[2].transAxes,
        fontsize=9,
        color="#334155",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{safe_prefix}_lateral_sway_and_step_lengths.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(series["tx_world_device"], series["ty_world_device"], color="#64748b", linewidth=1.2)
    ax.scatter(series["tx_world_device"].iloc[0], series["ty_world_device"].iloc[0], c="#3a9d55", s=60, label="start", zorder=5)
    ax.scatter(series["tx_world_device"].iloc[-1], series["ty_world_device"].iloc[-1], c="#c94c4c", s=60, label="end", zorder=5)
    if not events.empty:
        event_xy = pd.merge(
            events[["session_s", "side_proxy"]],
            series[["session_s", "tx_world_device", "ty_world_device"]],
            on="session_s",
            how="left",
        )
        colors = event_xy["side_proxy"].map({"lateral_positive": "#2f78b7", "lateral_negative": "#c94c4c"}).fillna("#333333")
        ax.scatter(event_xy["tx_world_device"], event_xy["ty_world_device"], c=colors, s=35, alpha=0.9, label="estimated step timing")
    ax.set_title(f"{bout['trial_id']} {display}: XY path with detected step-proxy events")
    ax.set_xlabel("tx_world_device (m)")
    ax.set_ylabel("ty_world_device (m)")
    ax.axis("equal")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{safe_prefix}_xy_path_with_step_events.png", dpi=180)
    plt.close(fig)


def plot_combined_summaries(events: pd.DataFrame, bouts: pd.DataFrame, out_dir: Path) -> None:
    usable_events = events[
        events["bout_confidence"].isin(["high", "medium"])
        & events["valid_step_path_length_m"].notna()
    ].copy()
    if not usable_events.empty:
        labels = [label for label in ["walking", "upstairs", "downstairs"] if label in set(usable_events["label"])]
        data = [usable_events.loc[usable_events["label"] == label, "valid_step_path_length_m"].to_numpy() for label in labels]
        fig, ax = plt.subplots(figsize=(9, 6))
        box = ax.boxplot(data, patch_artist=True)
        ax.set_xticks(range(1, len(labels) + 1), [ACTION_DISPLAY_LABELS.get(label, label.title()) for label in labels])
        for patch, label in zip(box["boxes"], labels):
            patch.set_facecolor(LABEL_COLORS.get(label, "#94a3b8"))
            patch.set_alpha(0.45)
        ax.set_title("Step-proxy path length by locomotion label")
        ax.set_ylabel("step-proxy path length between lateral head-sway extrema (m)")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "combined_step_proxy_length_by_label_boxplot.png", dpi=180)
        plt.close(fig)

    usable_bouts = bouts[bouts["confidence"].isin(["high", "medium"])].copy()
    if not usable_bouts.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, part in usable_bouts.groupby("label"):
            ax.scatter(
                part["median_step_path_length_m"],
                part["cadence_steps_per_s"],
                s=70,
                alpha=0.85,
                color=LABEL_COLORS.get(label, "#94a3b8"),
                label=ACTION_DISPLAY_LABELS.get(label, label.title()),
                edgecolor="white",
                linewidth=0.8,
            )
            for _, row in part.iterrows():
                ax.annotate(
                    f"{row['trial_id']} {row['bout_start_s']:.0f}s",
                    (row["median_step_path_length_m"], row["cadence_steps_per_s"]),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=7,
                    alpha=0.75,
                )
        ax.set_title("Cadence vs median step-proxy length by local bout")
        ax.set_xlabel("median step-proxy path length (m)")
        ax.set_ylabel("detected lateral events per second")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        fig.savefig(out_dir / "combined_cadence_vs_step_proxy_length.png", dpi=180)
        plt.close(fig)


def apply_continuous_step_stride_metrics(
    events: pd.DataFrame,
    *,
    max_bridge_gap_s: float = 2.5,
) -> pd.DataFrame:
    """Compute display-facing step/stride proxies across adjacent review windows.

    Bout-local windows are useful for zooming and local heading estimation, but
    they artificially reset step/stride history. This pass bridges only nearby
    events with the same locomotion label so long continuous walks do not show
    N/A at every review-window boundary.
    """
    if events.empty or "session_horizontal_path_m" not in events.columns:
        return events

    output = events.sort_values(["trial_id", "session_s", "bout_id", "event_index"]).copy()
    output["bout_local_step_path_length_m"] = output.get("step_path_length_m", np.nan)
    output["bout_local_stride_path_length_m"] = output.get("stride_path_length_m", np.nan)
    output["bout_local_valid_step_path_length_m"] = output.get("valid_step_path_length_m", np.nan)
    output["bout_local_valid_stride_path_length_m"] = output.get("valid_stride_path_length_m", np.nan)

    continuous_group_ids: list[int] = []
    group_id = -1
    previous_trial: str | None = None
    previous_label: str | None = None
    previous_time: float | None = None
    for row in output.to_dict(orient="records"):
        trial_id = str(row.get("trial_id", ""))
        label = str(row.get("label", ""))
        time_s = float(row.get("session_s", math.nan))
        starts_new_group = (
            trial_id != previous_trial
            or label != previous_label
            or previous_time is None
            or not math.isfinite(time_s)
            or not math.isfinite(previous_time)
            or time_s - previous_time > max_bridge_gap_s
        )
        if starts_new_group:
            group_id += 1
        continuous_group_ids.append(group_id)
        previous_trial = trial_id
        previous_label = label
        previous_time = time_s

    output["continuous_locomotion_group_id"] = continuous_group_ids
    grouped = output.groupby("continuous_locomotion_group_id", sort=False)
    output["session_inter_event_interval_s"] = grouped["session_s"].diff()
    output["session_step_path_length_m"] = grouped["session_horizontal_path_m"].diff()
    output["session_stride_path_length_m"] = grouped["session_horizontal_path_m"].diff(2)
    output["session_valid_step_path_length_m"] = output["session_step_path_length_m"].where(
        output["session_step_path_length_m"].between(0.03, 1.60)
    )
    output["session_valid_stride_path_length_m"] = output["session_stride_path_length_m"].where(
        output["session_stride_path_length_m"].between(0.08, 3.20)
    )

    output["step_path_length_m"] = output["session_step_path_length_m"]
    output["stride_path_length_m"] = output["session_stride_path_length_m"]
    output["valid_step_path_length_m"] = output["session_valid_step_path_length_m"]
    output["valid_stride_path_length_m"] = output["session_valid_stride_path_length_m"]
    output["rolling_last4_step_path_length_m"] = (
        grouped["valid_step_path_length_m"]
        .rolling(4, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    output["step_stride_metric_scope"] = "continuous_label_run"

    recovered_mask = (
        output["bout_local_valid_stride_path_length_m"].isna()
        & output["valid_stride_path_length_m"].notna()
    )
    if recovered_mask.any() and "event_warning" in output.columns:
        output.loc[recovered_mask, "event_warning"] = output.loc[recovered_mask, "event_warning"].apply(
            lambda value: "; ".join(
                part
                for part in str(value or "").split("; ")
                if part
                and part != "first event in bout has no previous-step spacing"
                and part != "step length outside accepted range"
            )
            or "stride recovered from previous nearby event in same continuous label run"
        )

    return output.sort_index()


def make_downward_contact_sheet(result: BoutResult, out_dir: Path, max_frames: int = 8) -> Path | None:
    trial_id = str(result.bout["trial_id"])
    video_path = RUNS_DIR / trial_id / "downward_master_session" / "downward_session_video.mp4"
    if not video_path.exists() or result.events.empty:
        return None

    events = result.events.dropna(subset=["session_s"])
    if events.empty:
        return None
    if len(events) > max_frames:
        center_index = len(events) // 2
        start_index = max(0, center_index - max_frames // 2)
        events = events.iloc[start_index : start_index + max_frames]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    for _, event in events.iterrows():
        frame_index = max(0, int(round(float(event["session_s"]) * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        height, width = frame.shape[:2]
        target_width = 300
        target_height = max(1, int(height * target_width / width))
        frame = cv2.resize(frame, (target_width, target_height))
        foot_text = str(event.get("estimated_foot_side", "")).strip() or str(event["side_proxy"]).replace("lateral_", "")
        confidence_text = str(event.get("foot_side_confidence", "")).strip()
        label_text = f"t={float(event['session_s']):.2f}s est {foot_text} {confidence_text}".strip()
        cv2.rectangle(frame, (0, 0), (target_width, 28), (255, 255, 255), -1)
        cv2.putText(frame, label_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 30, 40), 1, cv2.LINE_AA)
        frames.append(frame)
    cap.release()
    if not frames:
        return None

    cols = min(4, len(frames))
    rows = math.ceil(len(frames) / cols)
    frame_h, frame_w = frames[0].shape[:2]
    sheet = np.full((rows * frame_h, cols * frame_w, 3), 245, dtype=np.uint8)
    for index, frame in enumerate(frames):
        r = index // cols
        c = index % cols
        sheet[r * frame_h : (r + 1) * frame_h, c * frame_w : (c + 1) * frame_w] = frame

    out_stem = (
        f"{trial_id}_bout{int(result.bout['bout_id']):02d}_{result.bout['label']}_"
        f"{float(result.bout['bout_start_s']):.1f}-{float(result.bout['bout_end_s']):.1f}s_downward_contact_sheet.jpg"
    )
    out_name = f"{Path(out_stem).stem.replace('.', 'p')}.jpg"
    out_path = out_dir / out_name
    cv2.imwrite(str(out_path), sheet)
    return out_path


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(str(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_readme(out_dir: Path, bouts: pd.DataFrame, events: pd.DataFrame) -> None:
    high_medium = (
        bouts[bouts["confidence"].isin(["high", "medium"])].copy()
        if "confidence" in bouts.columns
        else pd.DataFrame()
    )
    by_label = (
        high_medium.groupby("label")
        .agg(
            usable_bouts=("bout_id", "count"),
            median_step_proxy_m=("median_step_path_length_m", "median"),
            median_stride_proxy_m=("median_stride_path_length_m", "median"),
            median_cadence_events_per_s=("cadence_steps_per_s", "median"),
        )
        .reset_index()
        if not high_medium.empty
        else pd.DataFrame()
    )
    by_label_text = dataframe_to_markdown_table(by_label) if not by_label.empty else "No high/medium confidence bouts found."
    total_events = len(events)

    readme = f"""# Step / Stride Length Exploration

This folder is a first-pass SLAM-only exploration of step-like patterns. It uses Aria/MPS head-device motion, so the results should be treated as **step/stride proxies**, not confirmed foot-contact measurements.

## What The Script Does

1. Uses effective manual labels to find locomotion intervals when available, otherwise continuous automatic classification: walking, upstairs, and downstairs. Automatically selected bouts are not ground truth.
2. Splits longer locomotion intervals into shorter local bouts, usually about 6 seconds. This avoids using one global direction across turns.
3. Projects raw SLAM x/y motion into:
   - local forward progress
   - lateral head sway
4. Detects alternating lateral head-sway extrema, then places each estimated step event at the midpoint between adjacent extrema. This better matches visually observed footstep timing than marking only the later head-sway peak.
5. Uses the MPS device orientation quaternion as a first-pass way to map path-side sway onto estimated left/right foot side.
6. Estimates:
   - step-proxy length = horizontal path length between consecutive estimated step events
   - stride-proxy length = horizontal path length across every other estimated step event
   - rolling last-4-step average = average of the most recent four step-proxy lengths
7. Saves event CSVs, bout summaries, plots, and a few downward-video contact sheets for visual validation.

## Why This Is Only A Proxy

SLAM tracks the Aria/head position, not the feet. Lateral head sway often follows left/right stepping rhythm, but it can be affected by turns, head motion, stairs, and recording noise. The `session_s` column is the estimated step time, currently defined as the midpoint between adjacent lateral extrema. The `sway_peak_session_s` column keeps the later head-sway peak/trough time for checking. The `side_proxy` column means lateral-positive or lateral-negative head sway. The `estimated_foot_side` column is an initial SLAM-only estimate based on the device orientation quaternion and should still be checked against the downward video.

## Current High/Medium Confidence Summary

{by_label_text}

Total estimated step events across all confidence levels: {total_events}

## Main Files

- `data/step_stride_bout_summary.csv`: one row per local walking/stair bout.
- `data/step_stride_event_estimates.csv`: one row per estimated step event, with the source lateral-sway peak times included.
- `data/step_stride_label_summary.csv`: aggregated high/medium confidence results by action label.
- `plots/combined_step_proxy_length_by_label_boxplot.png`: easiest first summary plot.
- `plots/combined_cadence_vs_step_proxy_length.png`: checks whether cadence and estimated step length look plausible.
- `plots/*_lateral_sway_and_step_lengths.png`: per-bout lateral-sway peaks plus rolling last-4-step estimates.
- `plots/*_xy_path_with_step_events.png`: local x/y trajectory with detected step-like events.
- `validation_frames/*_downward_contact_sheet.jpg`: sampled downward-video frames at detected step-like event times.

## How To Explain This In The Meeting

Simple version:

> I started with SLAM only. For each walking or stair segment, I projected the x/y trajectory into local forward movement and side-to-side head movement. Then I detected repeated side-to-side head-sway peaks/troughs and placed the estimated step time between adjacent extrema, because visual foot contacts often happen between head-sway peaks. From those estimated step events, I estimated step length using horizontal path distance between neighboring events, and I also computed a rolling average over the last four steps. I also added an initial left/right foot estimate using the Aria orientation quaternion, but it is still marked with confidence because head orientation can differ from the true foot/body direction.

Important caveat:

> This is not true foot-contact detection yet. It is a head-motion proxy. The next validation step is to open the synchronized downward video around the detected event times and check whether the peaks really correspond to left/right stepping rhythm.

## Suggested Next Steps

1. Validate several high-confidence walking bouts against the downward view.
2. Decide whether lateral-positive/lateral-negative peaks correspond consistently to left/right stepping.
3. Add orientation-aware projection if we need anatomical left/right rather than local side-to-side sway.
4. Use rolling last-4-step length as a candidate gait feature for walking stability and asymmetry.
5. Treat stairs separately from flat walking, because stair step-proxy lengths are naturally smaller in the current data.
"""
    (out_dir / "README.md").write_text(readme)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    data_dir = out_dir / "data"
    plot_dir = out_dir / "plots"
    validation_dir = out_dir / "validation_frames"
    for directory in [data_dir, plot_dir, validation_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    all_bouts: list[dict[str, Any]] = []
    all_events: list[pd.DataFrame] = []
    all_series: list[pd.DataFrame] = []
    contact_rows: list[dict[str, Any]] = []

    for trial_id in args.trials:
        traj, _meta = make_session_trajectory(trial_id)
        traj = add_session_horizontal_path(traj)
        states = load_effective_manual_intervals(trial_id)
        intervals = merge_locomotion_intervals(states)
        if intervals.empty:
            states = load_model_locomotion_states(trial_id)
            intervals = merge_locomotion_intervals(states)
        (data_dir / f"{trial_id}_locomotion_intervals.csv").write_text(intervals.to_csv(index=False))
        bout_specs: list[dict[str, Any]] = []
        for _, interval in intervals.iterrows():
            bout_specs.extend(split_into_local_bouts(interval, args.bout_s, args.bout_step_s, args.min_bout_s))

        for local_index, bout_spec in enumerate(bout_specs):
            result = analyze_bout(trial_id, local_index, traj, bout_spec, args.resample_fps)
            if result is None:
                continue
            all_bouts.append(result.bout)
            all_series.append(result.series)
            if not result.events.empty:
                all_events.append(result.events)
            plot_bout_diagnostics(result, plot_dir)
            if args.make_contact_sheets and result.bout["confidence"] in {"high", "medium"}:
                contact_path = make_downward_contact_sheet(result, validation_dir)
                if contact_path is not None:
                    contact_rows.append(
                        {
                            "trial_id": trial_id,
                            "bout_id": int(result.bout["bout_id"]),
                            "label": result.bout["label"],
                            "bout_start_s": result.bout["bout_start_s"],
                            "bout_end_s": result.bout["bout_end_s"],
                            "contact_sheet": str(contact_path),
                        }
                    )

    bouts = pd.DataFrame(all_bouts)
    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    events = apply_continuous_step_stride_metrics(events) if not events.empty else events
    series = pd.concat(all_series, ignore_index=True) if all_series else pd.DataFrame()

    bouts.to_csv(data_dir / "step_stride_bout_summary.csv", index=False)
    events.to_csv(data_dir / "step_stride_event_estimates.csv", index=False)
    series.to_csv(data_dir / "step_stride_resampled_bout_timeseries.csv", index=False)
    pd.DataFrame(contact_rows).to_csv(data_dir / "downward_validation_contact_sheets.csv", index=False)

    if not bouts.empty:
        usable = bouts[bouts["confidence"].isin(["high", "medium"])].copy()
        label_summary = (
            usable.groupby("label")
            .agg(
                usable_bouts=("bout_id", "count"),
                median_step_proxy_m=("median_step_path_length_m", "median"),
                p10_step_proxy_m=("p10_step_path_length_m", "median"),
                p90_step_proxy_m=("p90_step_path_length_m", "median"),
                median_stride_proxy_m=("median_stride_path_length_m", "median"),
                median_cadence_events_per_s=("cadence_steps_per_s", "median"),
                median_lateral_range_m=("lateral_range_m", "median"),
            )
            .reset_index()
            if not usable.empty
            else pd.DataFrame()
        )
        label_summary.to_csv(data_dir / "step_stride_label_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(data_dir / "step_stride_label_summary.csv", index=False)

    if not events.empty and not bouts.empty:
        plot_combined_summaries(events, bouts, plot_dir)
    write_readme(out_dir, bouts, events)

    print(f"Wrote step/stride diagnostics to {out_dir}")
    if not bouts.empty:
        print(bouts[["trial_id", "label", "bout_start_s", "bout_end_s", "confidence", "median_step_path_length_m", "cadence_steps_per_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
