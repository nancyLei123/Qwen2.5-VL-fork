#!/usr/bin/env python3
"""Build a Parkinson-focused continuous-motion review page.

This page is intentionally separate from the normal PD-Insighter classifier.
It is designed for Ntrial10-style simulated Parkinson motion review where fixed
2-second labels are less useful than longer motion periods plus context.
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
from pd_insighter_environment_navigation import environment_navigation, parkinson_navigation


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"


@dataclass(frozen=True)
class SmoothConfig:
    key: str
    label: str
    smooth_s: float
    analysis_s: float
    context_s: float


SMOOTH_CONFIGS = [
    SmoothConfig("low", "Low smoothing: keeps quick gait changes", 0.5, 4.0, 8.0),
    SmoothConfig("medium", "Medium smoothing: default clinical review", 1.5, 6.0, 10.0),
    SmoothConfig("high", "High smoothing: emphasizes long motion periods", 3.0, 10.0, 14.0),
]


MOTION_COLORS = {
    "sitting": "#e6ab02",
    "standing": "#7570b3",
    "walking": "#d95f02",
    "upstairs": "#1b9e77",
    "downstairs": "#666666",
    "unknown": "#c9c9c9",
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def quantile(values: pd.Series | np.ndarray, q: float, default: float = 0.0) -> float:
    arr = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if arr.size == 0:
        return default
    return float(np.quantile(arr, q))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def get_nested_path(data: dict[str, Any], *keys: str) -> Path | None:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, dict):
        cur = cur.get("path")
    if not cur:
        return None
    return Path(str(cur))


def rolling_mean(series: pd.Series, window_samples: int) -> pd.Series:
    window_samples = max(1, int(window_samples))
    return series.rolling(window_samples, center=True, min_periods=1).mean()


def rolling_sum(series: pd.Series, window_samples: int) -> pd.Series:
    window_samples = max(1, int(window_samples))
    return series.rolling(window_samples, center=True, min_periods=1).sum()


def rolling_std(series: pd.Series, window_samples: int) -> pd.Series:
    window_samples = max(1, int(window_samples))
    return series.rolling(window_samples, center=True, min_periods=1).std().fillna(0.0)


def rolling_range(series: pd.Series, window_samples: int) -> pd.Series:
    window_samples = max(1, int(window_samples))
    roll = series.rolling(window_samples, center=True, min_periods=1)
    return roll.max() - roll.min()


def resample_trajectory(trajectory_csv: Path, recording_start_epoch_s: float, duration_s: float, bin_s: float, session_start_tracking_us: float | None = None) -> pd.DataFrame:
    columns = [
        "utc_timestamp_ns",
        "tx_world_device",
        "ty_world_device",
        "tz_world_device",
        "device_linear_velocity_x_device",
        "device_linear_velocity_y_device",
        "device_linear_velocity_z_device",
        "angular_velocity_x_device",
        "angular_velocity_y_device",
        "angular_velocity_z_device",
    ]
    if session_start_tracking_us is not None:
        columns.append("tracking_timestamp_us")
    raw = pd.read_csv(trajectory_csv, usecols=columns)
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna(subset=["utc_timestamp_ns", "tx_world_device", "ty_world_device", "tz_world_device"])
    raw["session_time_s"] = raw["utc_timestamp_ns"].astype(float) / 1e9 - float(recording_start_epoch_s)
    if session_start_tracking_us is not None:
        raw["session_time_s"] = (raw["tracking_timestamp_us"].astype(float) - session_start_tracking_us) / 1e6
    raw = raw[(raw["session_time_s"] >= -5.0) & (raw["session_time_s"] <= duration_s + 5.0)].copy()
    raw = raw.sort_values("session_time_s").drop_duplicates("session_time_s")
    if raw.empty:
        raise ValueError(f"No SLAM samples overlap the session time range in {trajectory_csv}")

    target_t = np.arange(0.0, duration_s + bin_s / 2.0, bin_s)
    source_t = raw["session_time_s"].to_numpy()

    out = pd.DataFrame({"time_s": target_t})
    for src, dst in [
        ("tx_world_device", "x_m"),
        ("ty_world_device", "y_m"),
        ("tz_world_device", "z_m"),
        ("device_linear_velocity_x_device", "vx_device_mps"),
        ("device_linear_velocity_y_device", "vy_device_mps"),
        ("device_linear_velocity_z_device", "vz_device_mps"),
        ("angular_velocity_x_device", "wx_rps"),
        ("angular_velocity_y_device", "wy_rps"),
        ("angular_velocity_z_device", "wz_rps"),
    ]:
        out[dst] = np.interp(target_t, source_t, raw[src].astype(float).to_numpy())

    out["device_velocity_mag_mps"] = np.sqrt(
        out["vx_device_mps"] ** 2 + out["vy_device_mps"] ** 2 + out["vz_device_mps"] ** 2
    )
    out["angular_velocity_mag_rps"] = np.sqrt(out["wx_rps"] ** 2 + out["wy_rps"] ** 2 + out["wz_rps"] ** 2)
    return out


def posture_split_from_height(features: pd.DataFrame) -> float | None:
    low_motion = features[
        (features["mean_speed_mps"] <= quantile(features["mean_speed_mps"], 0.35))
        & (features["path_length_m"] <= quantile(features["path_length_m"], 0.35))
    ]["z_smooth_m"].dropna()
    if len(low_motion) < 40:
        return None
    values = np.sort(low_motion.to_numpy())
    gaps = np.diff(values)
    if gaps.size == 0:
        return None
    best_gap_index = int(np.argmax(gaps))
    low_count = best_gap_index + 1
    high_count = len(values) - low_count
    gap = float(gaps[best_gap_index])
    if gap < 0.18 or low_count < max(12, len(values) * 0.12) or high_count < max(12, len(values) * 0.12):
        return None
    return float((values[best_gap_index] + values[best_gap_index + 1]) / 2.0)


def sample_period_s(df: pd.DataFrame) -> float:
    diffs = df["time_s"].diff().dropna()
    if diffs.empty:
        return 0.5
    return max(0.001, safe_float(diffs.median(), 0.5))


def rolling_any(mask: pd.Series, window_samples: int) -> pd.Series:
    window_samples = max(1, int(window_samples))
    return mask.astype(int).rolling(window_samples, center=True, min_periods=1).max().fillna(0).astype(bool)


def fill_short_false_gaps(mask: np.ndarray, max_gap_samples: int) -> np.ndarray:
    """Fill short gaps between true regions for state-like motion periods."""
    if mask.size == 0:
        return mask
    filled = mask.astype(bool).copy()
    true_indices = np.flatnonzero(filled)
    if true_indices.size < 2:
        return filled
    start = true_indices[0]
    previous = true_indices[0]
    for idx in true_indices[1:]:
        gap = idx - previous - 1
        if 0 < gap <= max_gap_samples:
            filled[previous + 1 : idx] = True
        previous = idx
    if start > 0:
        filled[:start] = False
    return filled


def build_stair_direction(
    df: pd.DataFrame,
    *,
    vertical_stair: float,
    speed_move: float,
    path_move: float,
    cfg: SmoothConfig,
) -> pd.Series:
    """Detect upstairs/downstairs as continuous states with hysteresis.

    Parkinson-like stair walking can include plateaus, pauses, or two steps per
    stair. A strict per-frame threshold misses those middle pieces, so we use a
    strong vertical cue to start the stair state and weaker same-direction cues
    to keep it alive.
    """
    bin_s = sample_period_s(df)
    context_n = max(3, round(cfg.context_s / bin_s))
    gap_n = max(2, round(5.0 / bin_s))
    moving = (df["mean_speed_mps"] >= speed_move * 0.75) | (df["path_length_m"] >= path_move * 0.65)
    weak_vertical = max(vertical_stair * 0.28, 0.055)
    weak_slope = max(weak_vertical / max(cfg.analysis_s, bin_s), 0.006)

    strong_up = moving & (df["vertical_delta_m"] >= vertical_stair * 0.62)
    strong_down = moving & (df["vertical_delta_m"] <= -vertical_stair * 0.62)
    weak_up = moving & ((df["vertical_delta_m"] >= weak_vertical) | (df["height_slope_mps"] >= weak_slope))
    weak_down = moving & ((df["vertical_delta_m"] <= -weak_vertical) | (df["height_slope_mps"] <= -weak_slope))

    near_up = rolling_any(strong_up, context_n)
    near_down = rolling_any(strong_down, context_n)
    up_mask = (near_up & weak_up & ~strong_down).to_numpy()
    down_mask = (near_down & weak_down & ~strong_up).to_numpy()
    up_mask = fill_short_false_gaps(up_mask, gap_n)
    down_mask = fill_short_false_gaps(down_mask, gap_n)

    direction = np.zeros(len(df), dtype=int)
    for idx in range(len(df)):
        if up_mask[idx] and not down_mask[idx]:
            direction[idx] = 1
        elif down_mask[idx] and not up_mask[idx]:
            direction[idx] = -1
        elif up_mask[idx] and down_mask[idx]:
            direction[idx] = 1 if safe_float(df.loc[idx, "vertical_delta_m"]) >= 0 else -1
    return pd.Series(direction, index=df.index)


def build_posture_context(
    df: pd.DataFrame,
    *,
    posture_split: float | None,
    posture_delta: float,
    stair_direction: pd.Series,
    cfg: SmoothConfig,
    speed_strong: float,
    path_strong: float,
    progress_low: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Infer sitting/standing from nearby sit-down and stand-up transitions."""
    bin_s = sample_period_s(df)
    # Do not let the tail of an upstairs/downstairs period become a false
    # sit-down/stand-up transition. This is especially important when the
    # wearer is standing on a different floor level after stair travel.
    near_stair = rolling_any(stair_direction != 0, max(3, round(max(cfg.analysis_s * 2.0, 16.0) / bin_s)))
    non_stair = (stair_direction == 0) & ~near_stair
    local_posture_motion = (
        (df["mean_speed_mps"] <= speed_strong * 1.15)
        & (df["path_length_m"] <= path_strong * 0.85)
        & (df["net_displacement_m"] <= max(progress_low * 0.8, 0.52))
        & non_stair
    )
    stand_up_event = local_posture_motion & (df["vertical_delta_m"] >= posture_delta)
    sit_down_event = local_posture_motion & (df["vertical_delta_m"] <= -posture_delta)

    event_type = np.zeros(len(df), dtype=int)
    event_type[np.flatnonzero(stand_up_event.to_numpy())] = 1
    event_type[np.flatnonzero(sit_down_event.to_numpy())] = -1

    first_events = np.flatnonzero(event_type)
    if first_events.size:
        # If the first clear event is standing up, the earlier state was sitting.
        state = "sitting" if event_type[first_events[0]] == 1 else "standing"
    else:
        state = "standing"

    states: list[str] = []
    reasons: list[str] = []
    event_names: list[str] = []
    for idx, row in df.iterrows():
        evt = event_type[int(idx)] if int(idx) < len(event_type) else 0
        event_name = ""
        if evt == 1:
            state = "standing"
            event_name = "stand-up transition nearby"
        elif evt == -1:
            state = "sitting"
            event_name = "sit-down transition nearby"

        z = safe_float(row["z_smooth_m"])
        if event_name:
            reason = event_name
        elif posture_split is not None:
            low_height = z < posture_split
            if state == "sitting":
                reason = "sitting state carried from nearby sit-down/stand-up transition"
            elif low_height:
                reason = "low/bent height or lower floor, but no local sit-down context, so kept as standing"
            else:
                reason = "standing state carried from nearby transition or posture context"
        else:
            reason = "posture split not reliable; carried nearby posture state"

        states.append(state)
        reasons.append(reason)
        event_names.append(event_name)

    return pd.Series(states, index=df.index), pd.Series(reasons, index=df.index), pd.Series(event_names, index=df.index)


def build_features(base: pd.DataFrame, cfg: SmoothConfig, bin_s: float) -> pd.DataFrame:
    smooth_n = max(1, round(cfg.smooth_s / bin_s))
    analysis_n = max(3, round(cfg.analysis_s / bin_s))
    half_n = max(1, analysis_n // 2)
    df = base.copy()

    df["x_smooth_m"] = rolling_mean(df["x_m"], smooth_n)
    df["y_smooth_m"] = rolling_mean(df["y_m"], smooth_n)
    df["z_smooth_m"] = rolling_mean(df["z_m"], smooth_n)
    df["dx_m"] = df["x_smooth_m"].diff().fillna(0.0)
    df["dy_m"] = df["y_smooth_m"].diff().fillna(0.0)
    df["dz_m"] = df["z_smooth_m"].diff().fillna(0.0)
    df["horizontal_step_m"] = np.sqrt(df["dx_m"] ** 2 + df["dy_m"] ** 2)
    df["position_speed_mps"] = df["horizontal_step_m"] / bin_s
    df["mean_speed_mps"] = rolling_mean(df["position_speed_mps"], analysis_n)
    df["path_length_m"] = rolling_sum(df["horizontal_step_m"], analysis_n)
    df["net_displacement_m"] = np.sqrt(
        (df["x_smooth_m"].shift(-half_n) - df["x_smooth_m"].shift(half_n)) ** 2
        + (df["y_smooth_m"].shift(-half_n) - df["y_smooth_m"].shift(half_n)) ** 2
    )
    df["net_displacement_m"] = df["net_displacement_m"].fillna(0.0)
    df["path_efficiency"] = (df["net_displacement_m"] / df["path_length_m"].replace(0.0, np.nan)).fillna(0.0).clip(0.0, 1.0)
    df["vertical_delta_m"] = (df["z_smooth_m"].shift(-half_n) - df["z_smooth_m"].shift(half_n)).fillna(0.0)
    df["height_slope_mps"] = df["vertical_delta_m"] / max(cfg.analysis_s, bin_s)
    df["z_range_m"] = rolling_range(df["z_smooth_m"], analysis_n)
    df["angular_mean_rps"] = rolling_mean(df["angular_velocity_mag_rps"], analysis_n)

    long_smooth_n = max(smooth_n, round(3.0 / bin_s))
    df["z_residual_m"] = df["z_m"] - rolling_mean(df["z_m"], long_smooth_n)
    df["z_oscillation_m"] = rolling_range(df["z_residual_m"], analysis_n)

    residual = df["z_residual_m"].to_numpy()
    event = np.zeros(len(df), dtype=float)
    if len(residual) >= 3:
        noise = max(float(np.nanquantile(np.abs(residual), 0.65)), 0.006)
        local_low = (residual[1:-1] < residual[:-2]) & (residual[1:-1] <= residual[2:]) & (np.abs(residual[1:-1]) >= noise)
        event[1:-1] = local_low.astype(float)
    # This is a proxy, not a final clinical cadence measure.
    df["step_event_proxy"] = event
    df["cadence_proxy_spm"] = rolling_sum(df["step_event_proxy"], analysis_n) / max(cfg.analysis_s, bin_s) * 60.0
    df["step_length_proxy_m"] = (df["net_displacement_m"] / rolling_sum(df["step_event_proxy"], analysis_n).clip(lower=1.0)).fillna(0.0)
    return df


def classify_features(features: pd.DataFrame, cfg: SmoothConfig) -> tuple[pd.DataFrame, dict[str, float | str | None]]:
    df = features.copy()
    bin_s = sample_period_s(df)
    context_n = max(3, round(cfg.context_s / bin_s))

    speed_move = max(quantile(df["mean_speed_mps"], 0.55), 0.035)
    speed_strong = max(quantile(df["mean_speed_mps"], 0.75), speed_move * 1.15, 0.055)
    path_move = max(quantile(df["path_length_m"], 0.55), 0.18)
    path_strong = max(quantile(df["path_length_m"], 0.75), path_move * 1.15, 0.28)
    vertical_stair = max(quantile(np.abs(df["vertical_delta_m"]), 0.82), 0.18)
    progress_low = max(quantile(df["net_displacement_m"], 0.25), 0.08)
    oscillation_high = max(quantile(df["z_oscillation_m"], 0.72), 0.025)
    cadence_high = max(quantile(df["cadence_proxy_spm"], 0.70), 10.0)
    step_length_short = max(0.16, quantile(df["step_length_proxy_m"], 0.35))
    posture_split = posture_split_from_height(df)

    stair_direction = build_stair_direction(
        df,
        vertical_stair=vertical_stair,
        speed_move=speed_move,
        path_move=path_move,
        cfg=cfg,
    )
    posture_candidates = df[
        (df["path_length_m"] <= path_strong * 1.25)
        & (df["net_displacement_m"] <= max(progress_low * 2.2, 0.65))
        & (stair_direction == 0)
    ]["vertical_delta_m"].abs()
    posture_delta = max(quantile(posture_candidates, 0.78), 0.18)
    posture_delta = min(posture_delta, max(0.42, vertical_stair * 0.95))
    posture_state, posture_reason, posture_event = build_posture_context(
        df,
        posture_split=posture_split,
        posture_delta=posture_delta,
        stair_direction=stair_direction,
        cfg=cfg,
        speed_strong=speed_strong,
        path_strong=path_strong,
        progress_low=progress_low,
    )

    labels: list[str] = []
    base_labels: list[str] = []
    markers: list[str] = []
    confidence: list[str] = []
    reasons: list[str] = []

    moving_mask = (df["mean_speed_mps"] >= speed_move) | (df["path_length_m"] >= path_move)
    walking_context_mask = moving_mask & (stair_direction == 0)
    near_walking = rolling_any(walking_context_mask, context_n)

    for i, row in df.iterrows():
        speed = safe_float(row["mean_speed_mps"])
        path = safe_float(row["path_length_m"])
        net = safe_float(row["net_displacement_m"])
        vertical = safe_float(row["vertical_delta_m"])
        efficiency = safe_float(row["path_efficiency"])
        cadence = safe_float(row["cadence_proxy_spm"])
        oscillation = safe_float(row["z_oscillation_m"])

        is_moving = speed >= speed_move or path >= path_move
        is_strong_moving = speed >= speed_strong or path >= path_strong
        slow_travel_candidate = (
            speed >= speed_move * 0.65
            and path >= path_move * 0.62
            and net >= progress_low * 0.9
            and efficiency >= 0.55
            and cadence >= cadence_high * 0.6
        )
        stair_dir = int(stair_direction.iloc[i])

        marker = ""
        reason_parts = []
        conf = "medium"

        if stair_dir == 1:
            base = "upstairs"
            reason_parts.append("upstairs state: strong positive height trend, extended through weaker same-direction stair steps")
            conf = "high" if vertical >= vertical_stair * 0.8 else "medium"
        elif stair_dir == -1:
            base = "downstairs"
            reason_parts.append("downstairs state: strong negative height trend, extended through weaker same-direction stair steps")
            conf = "high" if abs(vertical) >= vertical_stair * 0.8 else "medium"
        elif is_moving:
            base = "walking"
            reason_parts.append("horizontal path/speed above adaptive moving threshold")
            conf = "high" if is_strong_moving and efficiency >= 0.45 else "medium"
        else:
            if bool(near_walking.iloc[i]) and oscillation >= oscillation_high and cadence >= cadence_high * 0.55:
                base = "walking"
                reason_parts.append("near walking context with low forward progress but repeated step-like motion")
                conf = "low"
            else:
                base = str(posture_state.iloc[i])
                reason_parts.append(str(posture_reason.iloc[i]))
                conf = "high" if str(posture_event.iloc[i]) else "medium"

        if base == "walking":
            if path >= path_move and net <= progress_low and cadence >= cadence_high * 0.7:
                reason_parts.append("step-like signal without much forward progress; keep as walking but review video")
                conf = "low"
            elif cadence >= cadence_high and efficiency < 0.45:
                reason_parts.append("many step-like oscillations with inefficient path; keep as walking but review video")
                conf = "medium"
            elif cadence >= cadence_high and safe_float(row["step_length_proxy_m"]) <= step_length_short:
                reason_parts.append("short step-length proxy with repeated gait rhythm; keep as walking")
                conf = "medium"
        elif base == "standing" and slow_travel_candidate:
            reason_parts.append("standing label has sustained X/Y travel and step-like cadence, so review as possible standing-to-walking")
            conf = "low"

        label = base
        labels.append(label)
        base_labels.append(base)
        markers.append(marker)
        confidence.append(conf)
        reasons.append("; ".join(reason_parts))

    df["stair_direction"] = stair_direction
    df["posture_state"] = posture_state
    df["posture_reason"] = posture_reason
    df["posture_event"] = posture_event
    df["base_motion"] = base_labels
    df["clinical_marker"] = markers
    df["motion_label"] = labels
    df["confidence"] = confidence
    df["reason"] = reasons

    thresholds: dict[str, float | str | None] = {
        "smoothing_key": cfg.key,
        "smooth_s": cfg.smooth_s,
        "analysis_window_s": cfg.analysis_s,
        "context_s": cfg.context_s,
        "speed_move_mps": speed_move,
        "speed_strong_mps": speed_strong,
        "path_move_m": path_move,
        "path_strong_m": path_strong,
        "vertical_stair_delta_m": vertical_stair,
        "progress_low_m": progress_low,
        "oscillation_high_m": oscillation_high,
        "cadence_high_spm": cadence_high,
        "step_length_short_m": step_length_short,
        "posture_transition_delta_m": posture_delta,
        "posture_split_z_m": posture_split,
    }
    return df, thresholds


def segment_features(classified: pd.DataFrame, cfg: SmoothConfig, duration_s: float) -> list[dict[str, Any]]:
    df = classified.reset_index(drop=True)
    if df.empty:
        return []

    raw_segments: list[dict[str, Any]] = []
    start_idx = 0
    current = str(df.loc[0, "motion_label"])
    for idx in range(1, len(df)):
        label = str(df.loc[idx, "motion_label"])
        if label != current:
            raw_segments.append({"start_idx": start_idx, "end_idx": idx - 1, "label": current})
            start_idx = idx
            current = label
    raw_segments.append({"start_idx": start_idx, "end_idx": len(df) - 1, "label": current})

    # Merge tiny jitter segments into neighbors so the review remains period-based.
    merged: list[dict[str, Any]] = []
    min_regular_s = 3.0
    for seg in raw_segments:
        start_t = safe_float(df.loc[seg["start_idx"], "time_s"])
        end_t = safe_float(df.loc[seg["end_idx"], "time_s"])
        dur = end_t - start_t
        label = str(seg["label"])
        if merged and dur < min_regular_s:
            merged[-1]["end_idx"] = seg["end_idx"]
            merged[-1]["label"] = merged[-1]["label"]
        else:
            merged.append(dict(seg))

    segments: list[dict[str, Any]] = []
    for seg_idx, seg in enumerate(merged, start=1):
        start_i = int(seg["start_idx"])
        end_i = int(seg["end_idx"])
        start_s = max(0.0, safe_float(df.loc[start_i, "time_s"]))
        end_s = min(duration_s, safe_float(df.loc[end_i, "time_s"]) + 0.5)
        if end_s <= start_s:
            end_s = min(duration_s, start_s + 0.5)

        during = df[(df["time_s"] >= start_s) & (df["time_s"] <= end_s)]
        context_start = max(0.0, start_s - cfg.context_s)
        context_end = min(duration_s, end_s + cfg.context_s)
        before = df[(df["time_s"] >= context_start) & (df["time_s"] < start_s)]
        after = df[(df["time_s"] > end_s) & (df["time_s"] <= context_end)]

        base = str(during["base_motion"].mode().iloc[0]) if not during.empty else "unknown"
        marker_values = [m for m in during["clinical_marker"].dropna().astype(str).tolist() if m]
        marker = str(pd.Series(marker_values).mode().iloc[0]) if marker_values else ""
        confidence = str(during["confidence"].mode().iloc[0]) if not during.empty else "low"
        reason_values = [r for r in during["reason"].dropna().astype(str).tolist() if r]
        reason = reason_values[0] if reason_values else ""

        def summarize(piece: pd.DataFrame) -> dict[str, float]:
            if piece.empty:
                return {
                    "mean_speed_mps": 0.0,
                    "path_length_m": 0.0,
                    "net_displacement_m": 0.0,
                    "vertical_delta_m": 0.0,
                    "height_slope_mps": 0.0,
                    "cadence_proxy_spm": 0.0,
                    "step_length_proxy_m": 0.0,
                    "path_efficiency": 0.0,
                }
            return {
                "mean_speed_mps": safe_float(piece["mean_speed_mps"].mean()),
                "path_length_m": safe_float(piece["horizontal_step_m"].sum()),
                "net_displacement_m": safe_float(
                    math.hypot(piece["x_smooth_m"].iloc[-1] - piece["x_smooth_m"].iloc[0], piece["y_smooth_m"].iloc[-1] - piece["y_smooth_m"].iloc[0])
                ),
                "vertical_delta_m": safe_float(piece["z_smooth_m"].iloc[-1] - piece["z_smooth_m"].iloc[0]),
                "height_slope_mps": safe_float((piece["z_smooth_m"].iloc[-1] - piece["z_smooth_m"].iloc[0]) / max(piece["time_s"].iloc[-1] - piece["time_s"].iloc[0], 0.5)),
                "cadence_proxy_spm": safe_float(piece["cadence_proxy_spm"].mean()),
                "step_length_proxy_m": safe_float(piece["step_length_proxy_m"].median()),
                "path_efficiency": safe_float(piece["path_efficiency"].mean()),
            }

        display_name = base
        segments.append(
            {
                "id": f"{cfg.key}_segment_{seg_idx:03d}",
                "index": seg_idx,
                "label": display_name,
                "base_motion": base,
                "clinical_marker": marker,
                "confidence": confidence,
                "reason": reason,
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "duration_s": round(end_s - start_s, 3),
                "context_start_s": round(context_start, 3),
                "context_end_s": round(context_end, 3),
                "summary_before": summarize(before),
                "summary_during": summarize(during),
                "summary_after": summarize(after),
            }
        )
    return segments


def to_records_for_browser(df: pd.DataFrame) -> list[dict[str, float | str]]:
    columns = [
        "time_s",
        "x_smooth_m",
        "y_smooth_m",
        "z_smooth_m",
        "mean_speed_mps",
        "path_length_m",
        "net_displacement_m",
        "vertical_delta_m",
        "height_slope_mps",
        "cadence_proxy_spm",
        "step_length_proxy_m",
        "path_efficiency",
        "stair_direction",
        "posture_state",
        "posture_event",
        "base_motion",
        "clinical_marker",
        "motion_label",
        "confidence",
    ]
    records: list[dict[str, float | str]] = []
    for record in df[columns].to_dict(orient="records"):
        clean: dict[str, float | str] = {}
        for key, value in record.items():
            if isinstance(value, str):
                clean[key] = value
            else:
                clean[key] = round(safe_float(value), 5)
        records.append(clean)
    return records


@environment_navigation
def build_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>{payload['trial_id']} - Parkinson Review</title>
<style>
:root {{
  --ink: #1e2520;
  --muted: #5f6b61;
  --line: #dce5dc;
  --panel: rgba(255,255,255,0.92);
  --green: #167852;
  --green-soft: #dff3eb;
  --gold: #b97921;
  --rose: #b64a4a;
  --bg: #f4f1e8;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: Avenir Next, Gill Sans, Trebuchet MS, sans-serif;
  color: var(--ink);
  background:
    radial-gradient(circle at top left, rgba(63,132,93,.18), transparent 32rem),
    linear-gradient(135deg, #f7f3e9, #eef4ed 55%, #f9efe5);
}}
.shell {{ width: min(1800px, calc(100vw - 40px)); margin: 0 auto; padding: 24px 0 48px; }}
.hero {{
  display: grid;
  grid-template-columns: 1.4fr .8fr;
  gap: 18px;
  align-items: stretch;
  margin-bottom: 18px;
}}
.card {{
  background: var(--panel);
  border: 1px solid rgba(75,94,77,.18);
  border-radius: 24px;
  box-shadow: 0 20px 55px rgba(51,65,54,.10);
}}
.intro {{ padding: 24px; }}
h1 {{ margin: 0 0 10px; font-size: clamp(28px, 4vw, 48px); line-height: 1.02; letter-spacing: -.04em; }}
h2 {{ margin: 0; font-size: 22px; letter-spacing: -.03em; }}
p {{ color: var(--muted); line-height: 1.48; margin: 8px 0 0; }}
.pill-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }}
.pill {{ border: 1px solid var(--line); border-radius: 999px; padding: 8px 12px; background: #fff; font-weight: 700; font-size: 13px; }}
.summary-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 18px; }}
.metric {{ background: #fff; border: 1px solid var(--line); border-radius: 18px; padding: 14px; }}
.metric b {{ display: block; font-size: 24px; letter-spacing: -.03em; }}
.metric span {{ color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }}
.video-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 18px; }}
.video-card {{ padding: 14px; overflow: hidden; }}
.video-title {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; gap: 10px; }}
.video-wrap {{ aspect-ratio: 1; min-height: 0; background: #111; border-radius: 20px; overflow: hidden; display: grid; place-items: center; }}
video {{ width: 100%; height: 100%; object-fit: contain; background: #080808; cursor: pointer; }}
#downwardVideo {{ transform: rotate(180deg); }}
#ariaVideo {{ transform: rotate(90deg); }}
.controls {{ padding: 18px; display: flex; flex-direction: column; gap: 16px; margin-bottom: 18px; }}
.control-row {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
.playback-row {{ padding-bottom: 12px; border-bottom: 1px solid rgba(75,94,77,.12); }}
.selection-row {{ align-items: end; }}
button, select, input {{
  border: 1px solid var(--line);
  background: #fff;
  color: var(--ink);
  border-radius: 999px;
  padding: 10px 14px;
  font-weight: 800;
  font: inherit;
  min-height: 44px;
  white-space: nowrap;
}}
button.primary {{ background: var(--green); color: #fff; border-color: var(--green); }}
button.danger {{ color: #9d2f2f; border-color: rgba(157,47,47,.25); }}
button:hover, select:hover, input:hover {{ transform: translateY(-1px); box-shadow: 0 8px 18px rgba(36,54,42,.10); }}
label {{ display: grid; gap: 6px; color: var(--muted); font-size: 12px; font-weight: 900; text-transform: uppercase; letter-spacing: .06em; }}
select {{ min-width: 250px; max-width: min(520px, 100%); text-transform: none; font-weight: 750; letter-spacing: 0; color: var(--ink); }}
input {{ min-width: 110px; max-width: 160px; font-weight: 800; }}
.check-label {{ display: inline-flex; grid-template-columns: none; align-items: center; gap: 8px; text-transform: none; letter-spacing: 0; font-size: 13px; color: var(--ink); background: #fff; border: 1px solid var(--line); border-radius: 999px; padding: 10px 13px; }}
.check-label input {{ accent-color: var(--green); }}
.time-readout {{ margin-left: auto; font-weight: 900; font-size: 24px; letter-spacing: -.03em; text-align: right; min-width: 190px; }}
.timeline-card {{ padding: 18px; margin-bottom: 18px; }}
.timeline {{ position: relative; height: 58px; border-radius: 16px; border: 1px solid var(--line); background: #fff; overflow: hidden; margin-top: 14px; }}
.timeline-seg {{ position: absolute; top: 0; bottom: 0; border-right: 1px solid rgba(0,0,0,.08); cursor: pointer; opacity: .72; }}
.timeline-seg:hover {{ opacity: 1; filter: saturate(1.2); }}
.timeline-context {{ position: absolute; top: 0; bottom: 0; background: rgba(28,125,88,.13); border-left: 2px dashed rgba(22,120,82,.5); border-right: 2px dashed rgba(22,120,82,.5); pointer-events: none; }}
.timeline-selected {{ position: absolute; top: 6px; bottom: 6px; border: 3px solid #17231c; border-radius: 14px; pointer-events: none; }}
.playhead {{ position: absolute; top: 0; bottom: 0; width: 3px; background: #17231c; pointer-events: none; }}
.manual-review-card {{ padding: 18px; margin-bottom: 18px; }}
.manual-toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: end; margin-top: 14px; }}
.quick-label-toolbar {{ background: #f6faf6; border: 1px solid var(--line); border-radius: 20px; padding: 12px; }}
.manual-editor-grid {{ display: grid; grid-template-columns: repeat(4, minmax(120px, auto)) minmax(240px, 1fr); gap: 10px; align-items: end; margin-top: 14px; }}
.manual-editor-grid textarea {{ min-height: 54px; }}
.scrub-timeline {{ height: 42px; cursor: pointer; touch-action: none; background: linear-gradient(90deg, #f7faf5, #fff); }}
.scrub-track {{ position: absolute; left: 12px; right: 12px; top: 18px; height: 6px; border-radius: 999px; background: #dce5dc; pointer-events: none; }}
.scrub-progress {{ position: absolute; left: 12px; top: 18px; height: 6px; border-radius: 999px; background: var(--green); pointer-events: none; }}
.scrub-knob {{ position: absolute; top: 9px; width: 24px; height: 24px; margin-left: -12px; border-radius: 50%; background: #17231c; box-shadow: 0 8px 18px rgba(23,35,28,.25); pointer-events: none; }}
.scrub-action-row {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 8px 0 4px; }}
.scrub-action-row .manual-row-label {{ margin: 0; }}
.manual-row-label {{ margin: 12px 0 6px; color: var(--muted); font-size: 12px; font-weight: 900; text-transform: uppercase; letter-spacing: .06em; }}
.manual-timeline {{ height: 72px; cursor: crosshair; touch-action: none; }}
.manual-timeline-seg {{ position: absolute; top: 8px; bottom: 8px; border-radius: 14px; border: 2px solid rgba(23,35,28,.42); cursor: pointer; opacity: .84; overflow: hidden; }}
.manual-timeline-seg:hover {{ opacity: 1; filter: saturate(1.18); }}
.manual-timeline-seg.active {{ border: 3px solid #17231c; box-shadow: 0 0 0 5px rgba(22,120,82,.12); opacity: 1; }}
.manual-timeline-seg span {{ display: block; padding: 8px 10px; color: #17231c; font-size: 12px; font-weight: 900; text-shadow: 0 1px 0 rgba(255,255,255,.5); white-space: nowrap; }}
.manual-gap {{ position: absolute; top: 8px; bottom: 8px; border-radius: 14px; background: repeating-linear-gradient(135deg, rgba(188,61,55,.18), rgba(188,61,55,.18) 8px, rgba(188,61,55,.07) 8px, rgba(188,61,55,.07) 16px); border: 2px dashed rgba(188,61,55,.5); cursor: pointer; }}
.manual-gap:hover {{ background: rgba(188,61,55,.2); }}
.manual-drag-preview {{ position: absolute; top: 8px; bottom: 8px; border-radius: 14px; border: 3px dashed #17231c; background: rgba(22,120,82,.18); pointer-events: none; }}
.manual-start-marker {{ position: absolute; top: 0; bottom: 0; width: 4px; background: #d66b2e; pointer-events: none; box-shadow: 0 0 0 5px rgba(214,107,46,.12); }}
.clinical-timeline {{ height: 52px; cursor: crosshair; touch-action: none; background: linear-gradient(90deg, #fbf8f4, #fff); }}
.clinical-timeline-seg {{ position: absolute; top: 7px; bottom: 7px; border-radius: 13px; border: 2px solid rgba(111,57,34,.38); cursor: pointer; opacity: .86; overflow: hidden; }}
.clinical-timeline-seg:hover {{ opacity: 1; filter: saturate(1.18); }}
.clinical-timeline-seg.active {{ border: 3px solid #17231c; box-shadow: 0 0 0 5px rgba(214,107,46,.12); opacity: 1; }}
.clinical-timeline-seg span {{ display: block; padding: 7px 10px; color: #17231c; font-size: 12px; font-weight: 900; white-space: nowrap; }}
.quick-label-state {{ color: var(--muted); font-size: 13px; font-weight: 850; min-height: 44px; display: inline-flex; align-items: center; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }}
.legend span {{ display: inline-flex; gap: 6px; align-items: center; color: var(--muted); font-weight: 800; font-size: 13px; }}
.swatch {{ width: 14px; height: 14px; border-radius: 4px; border: 1px solid rgba(0,0,0,.08); }}
.main-grid {{ display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(360px, .9fr); gap: 18px; }}
.plot-card, .details-card {{ padding: 18px; }}
.plot-header {{ display: flex; justify-content: space-between; gap: 12px; align-items: start; margin-bottom: 12px; }}
.plot-tools {{ display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; min-width: 250px; }}
.plot-tools button {{ padding: 8px 12px; min-height: 38px; font-size: 13px; }}
.plot-wrap {{ background: #fff; border: 1px solid var(--line); border-radius: 18px; overflow: hidden; }}
svg {{ display: block; width: 100%; }}
.details-card {{ display: grid; gap: 14px; align-content: start; }}
.selected-title {{ font-size: 24px; font-weight: 900; letter-spacing: -.03em; }}
.confidence {{ display: inline-flex; width: fit-content; border-radius: 999px; padding: 7px 11px; background: var(--green-soft); color: #07563a; font-weight: 900; }}
.evidence {{ background: #fff; border: 1px solid var(--line); border-radius: 18px; padding: 14px; }}
.manual-field {{ margin-top: 12px; }}
.manual-field select {{ width: 100%; border-radius: 16px; }}
textarea {{ width: 100%; min-height: 112px; resize: vertical; border: 1px solid var(--line); border-radius: 16px; padding: 12px; font: inherit; color: var(--ink); background: #fff; }}
.note-actions {{ display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-top: 10px; }}
.save-status {{ color: var(--muted); font-size: 12px; font-weight: 800; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th, td {{ text-align: left; border-bottom: 1px solid var(--line); padding: 9px 6px; }}
th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
.small-note {{ color: var(--muted); font-size: 12px; line-height: 1.45; }}
.xy-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 18px; }}
@media (max-width: 980px) {{
  .hero, .video-grid, .main-grid, .xy-grid {{ grid-template-columns: 1fr; }}
  .time-readout {{ margin-left: 0; text-align: left; }}
  .plot-header {{ flex-direction: column; }}
  .plot-tools {{ justify-content: flex-start; }}
}}
</style>
</head>
<body>
<main class=\"shell\">
  {parkinson_navigation(payload['trial_id'])}
  <section class=\"hero\">
    <div class=\"card intro\">
      <h1>Parkinson Review</h1>
      <p>{payload['trial_id']}: simulated Parkinson motion, not patient data. This exploratory classifier is separate from the normal-trial classifier. Review continuous motion periods and possible hesitation/shuffling with the synchronized videos; these flags are not clinical diagnoses.</p>
      <p><strong>Same saved classification as Full Session: high smoothing.</strong> Low/medium smoothing below are review previews only. Action labels are shared with Manual Labels; freezing and gait markers are a separate layer and may overlap actions.</p>
      <div class=\"pill-row\">
        <span class=\"pill\">Continuous SLAM, resampled at 0.5s</span>
        <span class=\"pill\">Before / during / after context</span>
        <span class=\"pill\">Smoothing can be changed</span>
        <span class=\"pill\">Normal Ntrials unchanged</span>
      </div>
    </div>
    <div class=\"card summary-grid\">
      <div class=\"metric\"><b id=\"metricSegments\">--</b><span>Detected Periods</span></div>
      <div class=\"metric\"><b id=\"metricDuration\">--</b><span>Session Duration</span></div>
      <div class=\"metric\"><b id=\"metricSmooth\">--</b><span>Smoothing</span></div>
      <div class=\"metric\"><b id=\"metricContext\">--</b><span>Context Window</span></div>
      <div class=\"metric\"><b id=\"metricManualAccuracy\">--</b><span>Manual Accuracy</span></div>
      <div class=\"metric\"><b id=\"metricManualCoverage\">--</b><span>Manual Coverage</span></div>
    </div>
  </section>

  <section class=\"video-grid\">
    <div class=\"card video-card\">
      <div class=\"video-title\"><h2>Downward View</h2><span class=\"pill\">default 180°</span></div>
      <div class=\"video-wrap\"><video id=\"downwardVideo\" src=\"{payload['media']['downward']}\" playsinline preload=\"metadata\"></video></div>
    </div>
    <div class=\"card video-card\">
      <div class=\"video-title\"><h2>Aria View</h2><span class=\"pill\">default 90°</span></div>
      <div class=\"video-wrap\"><video id=\"ariaVideo\" src=\"{payload['media']['aria']}\" playsinline preload=\"metadata\"></video></div>
    </div>
  </section>

  <section class=\"card controls\">
    <div class=\"control-row playback-row\">
      <button class=\"primary\" id=\"playPauseBtn\">Play</button>
      <button id=\"playContextBtn\">Play With Context</button>
      <button id=\"playRangeBtn\">Play Detected Range</button>
      <button id=\"prevBtn\">Previous Period</button>
      <button id=\"nextBtn\">Next Period</button>
      <button id=\"back2Btn\">-2s</button>
      <button id=\"forward2Btn\">+2s</button>
      <button id="audioToggleBtn" type="button" aria-pressed="false">Audio Off</button>
      <label class=\"check-label\"><input type=\"checkbox\" id=\"autoFollowToggle\" checked> Follow video period</label>
    </div>
    <div class=\"control-row selection-row\">
      <label>Smoothing
        <select id=\"smoothingSelect\"></select>
      </label>
      <label>Detected Period
        <select id=\"segmentSelect\"></select>
      </label>
      <label>Plot Window
        <select id=\"plotWindowSelect\">
          <option value=\"context\">Selected context</option>
          <option value=\"wide\">Wide around period</option>
          <option value=\"full\">Full session</option>
        </select>
      </label>
      <div class=\"time-readout\"><span id=\"timeNow\">0.0</span>s / <span id=\"timeTotal\">--</span>s</div>
    </div>
  </section>

  <section class=\"card timeline-card\">
    <h2>Full-Session Motion Timeline</h2>
    <p>Click any period to seek both videos. The dashed green band is the before/after context used to review the selected detection.</p>
    <div class=\"timeline\" id=\"timeline\"></div>
    <div class=\"legend\" id=\"legend\"></div>
  </section>

  <section class=\"card manual-review-card\">
    <h2>Manual Ground Truth Timeline</h2>
    <p>Fast workflow: choose the action, then drag directly on the blank manual bar to create a labeled time range. Or press Start Label Here while watching the videos, then End Here + Save when the action changes.</p>
    <div class=\"manual-toolbar quick-label-toolbar\">
      <label>Label to apply
        <select id=\"quickManualLabelSelect\">
          <option value=\"sitting\">sitting</option>
          <option value=\"standing\">standing</option>
          <option value=\"walking\">walking</option>
          <option value=\"upstairs\">upstairs</option>
          <option value=\"downstairs\">downstairs</option>
        </select>
      </label>
      <button id=\"quickStartLabelBtn\">Start Label Here</button>
      <button class=\"primary\" id=\"quickEndLabelBtn\">End Here + Save</button>
      <button id=\"cancelQuickLabelBtn\">Cancel Start</button>
      <button id=\"startNextGapBtn\">Start Next Unlabeled Gap</button>
      <label class=\"check-label\"><input type=\"checkbox\" id=\"continuousLabelToggle\" checked> Continuous labels / snap gaps</label>
      <span class=\"quick-label-state\" id=\"quickLabelStatus\">Tip: drag across the manual bar to label a range.</span>
    </div>
    <div class=\"scrub-action-row\">
      <div class=\"manual-row-label\">Video seek bar: click or drag to start the synchronized videos anywhere</div>
      <button id=\"manualScrubPlayPauseBtn\">Play Video</button>
    </div>
    <div class=\"timeline scrub-timeline\" id=\"manualScrubTimeline\"></div>
    <div class=\"manual-row-label\">Freezing / gait markers: drag here to mark periods that need closer clinician review. These can overlap action labels.</div>
    <div class=\"manual-toolbar quick-label-toolbar\">
      <label>Marker to apply
        <select id=\"clinicalMarkerSelect\">
          <option value=\"freezing\">freezing</option>
          <option value=\"hesitation\">hesitation</option>
          <option value=\"shuffling\">shuffling</option>
          <option value=\"festinating\">festinating</option>
          <option value=\"cadence_irregularity\">cadence irregularity</option>
        </select>
      </label>
      <button id=\"clinicalStartBtn\">Start Marker Here</button>
      <button class=\"primary\" id=\"clinicalEndBtn\">End Marker + Save</button>
      <button id=\"clinicalCancelBtn\">Cancel Marker Start</button>
      <button class=\"danger\" id=\"clinicalDeleteBtn\">Delete Selected Marker</button>
      <span class=\"quick-label-state\" id=\"clinicalStatus\">No freezing/gait markers loaded yet.</span>
    </div>
    <div class=\"timeline clinical-timeline\" id=\"clinicalTimeline\"></div>
    <div class=\"manual-row-label\">Manual labels: drag here to create corrected action ranges</div>
    <div class=\"timeline manual-timeline\" id=\"manualTimeline\"></div>
    <div class=\"manual-toolbar\">
      <button id=\"addManualFromPeriodBtn\">Add From Detected Period</button>
      <button id=\"addManualAtTimeBtn\">Add Around Current Time</button>
      <label>Manual Segment
        <select id=\"manualSegmentSelect\"></select>
      </label>
      <span class=\"save-status\" id=\"manualCoverageSummary\">No manual labels yet</span>
    </div>
    <div class=\"manual-editor-grid\">
      <label>Start (s)
        <input id=\"manualStartInput\" type=\"number\" min=\"0\" step=\"0.1\">
      </label>
      <label>End (s)
        <input id=\"manualEndInput\" type=\"number\" min=\"0\" step=\"0.1\">
      </label>
      <label>Motion
        <select id=\"manualGroundTruthLabelSelect\">
          <option value=\"sitting\">sitting</option>
          <option value=\"standing\">standing</option>
          <option value=\"walking\">walking</option>
          <option value=\"upstairs\">upstairs</option>
          <option value=\"downstairs\">downstairs</option>
        </select>
      </label>
      <label>Confidence
        <select id=\"manualConfidenceSelect\">
          <option value=\"high\">high</option>
          <option value=\"medium\">medium</option>
          <option value=\"low\">low</option>
          <option value=\"uncertain\">uncertain</option>
        </select>
      </label>
      <label>Manual note
        <textarea id=\"manualSegmentNote\" placeholder=\"Optional note about this manually labeled range...\"></textarea>
      </label>
    </div>
    <div class=\"note-actions\">
      <button id=\"setManualStartBtn\">Set Start Here</button>
      <button id=\"setManualEndBtn\">Set End Here</button>
      <button class=\"primary\" id=\"saveManualSegmentBtn\">Save Manual Labels</button>
      <button class=\"danger\" id=\"deleteManualSegmentBtn\">Delete Selected Manual Segment</button>
      <span class=\"save-status\" id=\"manualSegmentStatus\">Manual labels not loaded yet</span>
    </div>
  </section>

  <section class=\"main-grid\">
    <div class=\"card plot-card\">
      <div class=\"plot-header\">
        <div>
          <h2>Context Plot: Height, Speed, And Vertical Trend</h2>
          <p>The selected period is centered between context before and after it. Click the plot to seek the videos; when follow mode is on, this plot updates as playback enters a new detected period.</p>
        </div>
        <div class=\"plot-tools\" aria-label=\"Context plot zoom controls\">
          <button id=\"zoomInBtn\">Zoom In</button>
          <button id=\"zoomOutBtn\">Zoom Out</button>
          <button id=\"resetZoomBtn\">Reset Plot</button>
        </div>
      </div>
      <div class=\"plot-wrap\"><svg id=\"signalPlot\" viewBox=\"0 0 900 420\" role=\"img\"></svg></div>
    </div>
    <aside class=\"card details-card\">
      <div>
        <div class=\"selected-title\" id=\"selectedTitle\">Select a period</div>
        <p id=\"selectedTime\">--</p>
      </div>
      <span class=\"confidence\" id=\"selectedConfidence\">-- confidence</span>
      <div class=\"evidence\">
        <h2>Why It Was Detected</h2>
        <p id=\"selectedReason\">--</p>
      </div>
      <div class=\"evidence\">
        <h2>Before / During / After</h2>
        <table id=\"contextTable\"></table>
      </div>
      <div class=\"evidence\">
        <h2>Manual Classification</h2>
        <p class=\"small-note\">If the detected motion is wrong for this period, choose the corrected five-motion label here. Notes are optional.</p>
        <label class=\"manual-field\">Corrected motion
          <select id=\"manualLabelSelect\">
            <option value=\"\">Keep detected label</option>
            <option value=\"sitting\">sitting</option>
            <option value=\"standing\">standing</option>
            <option value=\"walking\">walking</option>
            <option value=\"upstairs\">upstairs</option>
            <option value=\"downstairs\">downstairs</option>
          </select>
        </label>
        <p class=\"small-note\" id=\"manualCorrectionStatus\">No manual correction for this period.</p>
        <textarea id=\"manualNote\" placeholder=\"Write a note for this detected period...\"></textarea>
        <div class=\"note-actions\">
          <label class=\"check-label\"><input type=\"checkbox\" id=\"needsReviewToggle\"> Mark as needs closer look</label>
          <button id=\"saveNoteBtn\">Save Manual Review</button>
          <span class=\"save-status\" id=\"noteStatus\">Not saved yet</span>
        </div>
      </div>
      <p class=\"small-note\">This is an exploratory review tool, not a medical diagnosis. The useful part is whether the detected range plus surrounding context matches what we see in the synchronized videos.</p>
    </aside>
  </section>

  <section class=\"xy-grid\">
    <div class=\"card plot-card\">
      <h2>X/Y Path In Selected Context</h2>
      <p>Direction is based on the local SLAM path. This helps separate real forward travel from turning or small oscillations.</p>
      <div class=\"plot-wrap\"><svg id=\"xyPlot\" viewBox=\"0 0 640 460\" role=\"img\"></svg></div>
    </div>
    <div class=\"card plot-card\">
      <h2>Criteria Notes</h2>
      <p><b>Posture</b> uses nearby sit-down/stand-up transitions before trusting sitting. <b>Stairs</b> use a state rule: strong up/down height trend starts the stair period, then weaker same-direction steps can keep it active. The current page only classifies the five base motions.</p>
      <div class=\"evidence\"><table id=\"thresholdTable\"></table></div>
    </div>
  </section>
</main>
<script>
const DASHBOARD_DATA = {data_json};
const MOTION_COLORS = {json.dumps(MOTION_COLORS, ensure_ascii=True)};
const downwardVideo = document.getElementById('downwardVideo');
const ariaVideo = document.getElementById('ariaVideo');
downwardVideo.muted = true;
ariaVideo.muted = true;
document.getElementById('audioToggleBtn').addEventListener('click', event => {{
  ariaVideo.muted = !ariaVideo.muted;
  event.currentTarget.textContent = ariaVideo.muted ? 'Audio Off' : 'Audio On';
  event.currentTarget.setAttribute('aria-pressed', String(!ariaVideo.muted));
}});
const playPauseBtn = document.getElementById('playPauseBtn');
const smoothingSelect = document.getElementById('smoothingSelect');
const segmentSelect = document.getElementById('segmentSelect');
const plotWindowSelect = document.getElementById('plotWindowSelect');
const timeline = document.getElementById('timeline');
const signalPlot = document.getElementById('signalPlot');
const xyPlot = document.getElementById('xyPlot');
const autoFollowToggle = document.getElementById('autoFollowToggle');
const manualLabelSelect = document.getElementById('manualLabelSelect');
const manualNote = document.getElementById('manualNote');
const needsReviewToggle = document.getElementById('needsReviewToggle');
const saveNoteBtn = document.getElementById('saveNoteBtn');
const noteStatus = document.getElementById('noteStatus');
const manualCorrectionStatus = document.getElementById('manualCorrectionStatus');
const manualTimeline = document.getElementById('manualTimeline');
const manualSegmentSelect = document.getElementById('manualSegmentSelect');
const manualStartInput = document.getElementById('manualStartInput');
const manualEndInput = document.getElementById('manualEndInput');
const manualGroundTruthLabelSelect = document.getElementById('manualGroundTruthLabelSelect');
const manualConfidenceSelect = document.getElementById('manualConfidenceSelect');
const manualSegmentNote = document.getElementById('manualSegmentNote');
const manualCoverageSummary = document.getElementById('manualCoverageSummary');
const manualSegmentStatus = document.getElementById('manualSegmentStatus');
const manualScrubTimeline = document.getElementById('manualScrubTimeline');
const manualScrubPlayPauseBtn = document.getElementById('manualScrubPlayPauseBtn');
const quickManualLabelSelect = document.getElementById('quickManualLabelSelect');
const quickLabelStatus = document.getElementById('quickLabelStatus');
const continuousLabelToggle = document.getElementById('continuousLabelToggle');
const clinicalMarkerSelect = document.getElementById('clinicalMarkerSelect');
const clinicalTimeline = document.getElementById('clinicalTimeline');
const clinicalStatus = document.getElementById('clinicalStatus');
let smoothingKey = DASHBOARD_DATA.defaultSmoothing;
let selectedIndex = 0;
let selectedManualIndex = -1;
let selectedClinicalIndex = -1;
let quickLabelStartS = null;
let clinicalStartS = null;
let manualDragRange = null;
let clinicalDragRange = null;
let scrubDragging = false;
let stopAt = null;
let syncing = false;
let autoFollow = true;
let plotWindowMode = 'context';
let plotZoomScale = 1;
let manualNotes = {{}};
let manualSegments = [];
let freezingSegments = [];
let manualSaveTimer = null;
let freezingSaveTimer = null;

function fmt(value, digits=2) {{
  if (!Number.isFinite(value)) return '--';
  return Number(value).toFixed(digits);
}}

function currentBundle() {{
  return DASHBOARD_DATA.smoothing[smoothingKey];
}}

function currentSegments() {{
  return currentBundle().segments || [];
}}

function currentRecords() {{
  return currentBundle().records || [];
}}

function colorFor(base) {{
  return MOTION_COLORS[base] || MOTION_COLORS.unknown;
}}

function segmentIndexAtTime(timeS) {{
  const segments = currentSegments();
  for (let i = 0; i < segments.length; i += 1) {{
    if (timeS >= segments[i].start_s && timeS < segments[i].end_s) return i;
  }}
  return -1;
}}

function noteKeyForSegment(seg) {{
  if (!seg) return '';
  return `${{smoothingKey}}:${{seg.start_s.toFixed(1)}}-${{seg.end_s.toFixed(1)}}:${{seg.base_motion}}`;
}}

function localNoteStorageKey() {{
  return `parkinson-notes:${{DASHBOARD_DATA.trial_id}}`;
}}

async function loadManualNotes() {{
  try {{
    const response = await fetch(`/api/parkinson-notes?trial_id=${{encodeURIComponent(DASHBOARD_DATA.trial_id)}}`);
    if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
    const payload = await response.json();
    manualNotes = payload.notes || {{}};
    window.localStorage.setItem(localNoteStorageKey(), JSON.stringify(manualNotes));
    noteStatus.textContent = 'Notes loaded';
  }} catch (err) {{
    try {{
      manualNotes = JSON.parse(window.localStorage.getItem(localNoteStorageKey()) || '{{}}');
      noteStatus.textContent = 'Notes loaded locally';
    }} catch (_err) {{
      manualNotes = {{}};
      noteStatus.textContent = 'Notes unavailable';
    }}
  }}
  renderNoteForSelected();
}}

function localManualSegmentsStorageKey() {{
  return `parkinson-manual-segments:${{DASHBOARD_DATA.trial_id}}`;
}}

function localFreezingSegmentsStorageKey() {{
  return `parkinson-freezing-segments:${{DASHBOARD_DATA.trial_id}}`;
}}

function normalizeManualSegment(raw, fallbackIndex=0) {{
  const start = Math.max(0, Number(raw.start_s ?? raw.start ?? 0));
  const end = Math.min(DASHBOARD_DATA.duration_s, Math.max(start + 0.1, Number(raw.end_s ?? raw.end ?? start + 4)));
  const label = ['sitting', 'standing', 'walking', 'upstairs', 'downstairs'].includes(String(raw.motion_label || '').toLowerCase())
    ? String(raw.motion_label).toLowerCase()
    : 'standing';
  const now = new Date().toISOString();
  return {{
    id: String(raw.id || `manual_${{Date.now()}}_${{fallbackIndex}}_${{Math.random().toString(16).slice(2)}}`),
    start_s: Number(start.toFixed(3)),
    end_s: Number(end.toFixed(3)),
    motion_label: label,
    confidence: ['high', 'medium', 'low', 'uncertain'].includes(String(raw.confidence || '').toLowerCase()) ? String(raw.confidence).toLowerCase() : 'medium',
    note: String(raw.note || ''),
    source: 'manual',
    created_at: String(raw.created_at || now),
    updated_at: String(raw.updated_at || now),
  }};
}}

function sortManualSegments() {{
  const selectedId = manualSegments[selectedManualIndex]?.id || null;
  manualSegments.sort((a, b) => (a.start_s - b.start_s) || (a.end_s - b.end_s));
  const cleaned = [];
  for (const rawSeg of manualSegments) {{
    const seg = {{...rawSeg}};
    const previous = cleaned[cleaned.length - 1];
    if (previous && seg.start_s < previous.end_s) {{
      seg.start_s = Number(previous.end_s.toFixed(3));
    }}
    if (seg.end_s - seg.start_s >= 0.1) cleaned.push(seg);
  }}
  manualSegments = cleaned;
  if (selectedId) selectedManualIndex = manualSegments.findIndex(seg => seg.id === selectedId);
  if (selectedManualIndex < 0 && manualSegments.length) selectedManualIndex = 0;
}}

async function loadManualSegments() {{
  try {{
    const response = await fetch(`/api/parkinson-manual-segments?trial_id=${{encodeURIComponent(DASHBOARD_DATA.trial_id)}}`);
    if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
    const payload = await response.json();
    manualSegments = (payload.segments || []).map(normalizeManualSegment);
    sortManualSegments();
    window.localStorage.setItem(localManualSegmentsStorageKey(), JSON.stringify(manualSegments));
    manualSegmentStatus.textContent = 'Manual labels loaded from file';
  }} catch (err) {{
    try {{
      manualSegments = JSON.parse(window.localStorage.getItem(localManualSegmentsStorageKey()) || '[]').map(normalizeManualSegment);
      sortManualSegments();
      manualSegmentStatus.textContent = 'Manual labels loaded locally';
    }} catch (_err) {{
      manualSegments = [];
      selectedManualIndex = -1;
      manualSegmentStatus.textContent = 'No manual labels loaded';
    }}
  }}
  renderManualLayer();
}}

async function saveManualSegments(silent=false) {{
  manualSegments = manualSegments.map(normalizeManualSegment);
  sortManualSegments();
  window.localStorage.setItem(localManualSegmentsStorageKey(), JSON.stringify(manualSegments));
  if (!silent) manualSegmentStatus.textContent = 'Saving manual labels...';
  try {{
    const response = await fetch('/api/parkinson-manual-segments', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{trial_id: DASHBOARD_DATA.trial_id, segments: manualSegments}}),
    }});
    if (!response.ok) {{
      const errorPayload = await response.json().catch(() => ({{}}));
      throw new Error(errorPayload.error || `HTTP ${{response.status}}`);
    }}
    const saved = await response.json();
    manualSegments = (saved.segments || []).map(normalizeManualSegment);
    sortManualSegments();
    window.localStorage.setItem(localManualSegmentsStorageKey(), JSON.stringify(manualSegments));
    manualSegmentStatus.textContent = 'Manual labels saved to file';
  }} catch (err) {{
    manualSegmentStatus.textContent = `Saved locally; file save failed: ${{err.message || err}}`;
  }}
  renderManualLayer();
}}

function normalizeFreezingSegment(raw, fallbackIndex=0) {{
  const start = Math.max(0, Number(raw.start_s ?? raw.start ?? 0));
  const end = Math.min(DASHBOARD_DATA.duration_s, Math.max(start + 0.1, Number(raw.end_s ?? raw.end ?? start + 4)));
  const valid = ['freezing', 'hesitation', 'shuffling', 'festinating', 'cadence_irregularity'];
  const marker = valid.includes(String(raw.marker_label || '').toLowerCase()) ? String(raw.marker_label).toLowerCase() : 'freezing';
  const now = new Date().toISOString();
  return {{
    id: String(raw.id || `freezing_${{Date.now()}}_${{fallbackIndex}}_${{Math.random().toString(16).slice(2)}}`),
    start_s: Number(start.toFixed(3)),
    end_s: Number(end.toFixed(3)),
    marker_label: marker,
    confidence: ['high', 'medium', 'low', 'uncertain'].includes(String(raw.confidence || '').toLowerCase()) ? String(raw.confidence).toLowerCase() : 'medium',
    note: String(raw.note || ''),
    source: 'manual',
    created_at: String(raw.created_at || now),
    updated_at: String(raw.updated_at || now),
  }};
}}

function sortFreezingSegments() {{
  const selectedId = freezingSegments[selectedClinicalIndex]?.id || null;
  freezingSegments.sort((a, b) => (a.start_s - b.start_s) || (a.end_s - b.end_s));
  if (selectedId) selectedClinicalIndex = freezingSegments.findIndex(seg => seg.id === selectedId);
  if (selectedClinicalIndex < 0 && freezingSegments.length) selectedClinicalIndex = 0;
}}

async function loadFreezingSegments() {{
  try {{
    const response = await fetch(`/api/parkinson-freezing-segments?trial_id=${{encodeURIComponent(DASHBOARD_DATA.trial_id)}}`);
    if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
    const payload = await response.json();
    freezingSegments = (payload.segments || []).map(normalizeFreezingSegment);
    sortFreezingSegments();
    window.localStorage.setItem(localFreezingSegmentsStorageKey(), JSON.stringify(freezingSegments));
    clinicalStatus.textContent = freezingSegments.length ? `${{freezingSegments.length}} gait marker(s) loaded` : 'No freezing/gait markers yet';
  }} catch (err) {{
    try {{
      freezingSegments = JSON.parse(window.localStorage.getItem(localFreezingSegmentsStorageKey()) || '[]').map(normalizeFreezingSegment);
      sortFreezingSegments();
      clinicalStatus.textContent = 'Freezing/gait markers loaded locally';
    }} catch (_err) {{
      freezingSegments = [];
      selectedClinicalIndex = -1;
      clinicalStatus.textContent = 'No freezing/gait markers loaded';
    }}
  }}
  renderClinicalTimeline();
  renderSignalPlot();
}}

async function saveFreezingSegments(silent=false) {{
  freezingSegments = freezingSegments.map(normalizeFreezingSegment);
  sortFreezingSegments();
  window.localStorage.setItem(localFreezingSegmentsStorageKey(), JSON.stringify(freezingSegments));
  if (!silent) clinicalStatus.textContent = 'Saving freezing/gait markers...';
  try {{
    const response = await fetch('/api/parkinson-freezing-segments', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{trial_id: DASHBOARD_DATA.trial_id, segments: freezingSegments}}),
    }});
    if (!response.ok) {{
      const errorPayload = await response.json().catch(() => ({{}}));
      throw new Error(errorPayload.error || `HTTP ${{response.status}}`);
    }}
    const saved = await response.json();
    freezingSegments = (saved.segments || []).map(normalizeFreezingSegment);
    sortFreezingSegments();
    window.localStorage.setItem(localFreezingSegmentsStorageKey(), JSON.stringify(freezingSegments));
    clinicalStatus.textContent = `Saved ${{freezingSegments.length}} freezing/gait marker(s)`;
  }} catch (err) {{
    clinicalStatus.textContent = `Saved locally; marker file save failed: ${{err.message || err}}`;
  }}
  renderClinicalTimeline();
  renderSignalPlot();
}}

function scheduleFreezingSave() {{
  clinicalStatus.textContent = 'Unsaved freezing/gait marker changes';
  window.clearTimeout(freezingSaveTimer);
  freezingSaveTimer = window.setTimeout(() => saveFreezingSegments(true), 700);
}}

function freezingColor(label) {{
  return {{
    freezing: '#c9533b',
    hesitation: '#d8892f',
    shuffling: '#8d6ac8',
    festinating: '#4f83c2',
    cadence_irregularity: '#83712c',
  }}[label] || '#c9533b';
}}

function markerDisplay(label) {{
  return String(label || '').replaceAll('_', ' ');
}}

function createFreezingSegment(startS, endS, markerLabel, note='', shouldSeek=false) {{
  const start = Math.max(0, Math.min(startS, endS));
  const end = Math.min(DASHBOARD_DATA.duration_s, Math.max(startS, endS));
  if (end - start < 0.2) {{
    clinicalStatus.textContent = 'Marker range is too short.';
    return null;
  }}
  const now = new Date().toISOString();
  const segment = normalizeFreezingSegment({{
    id: `freezing_${{Date.now()}}_${{Math.random().toString(16).slice(2)}}`,
    start_s: start,
    end_s: end,
    marker_label: markerLabel,
    confidence: 'medium',
    note,
    created_at: now,
    updated_at: now,
  }});
  freezingSegments.push(segment);
  selectedClinicalIndex = freezingSegments.length - 1;
  sortFreezingSegments();
  renderClinicalTimeline();
  saveFreezingSegments(true);
  if (shouldSeek) seekBoth(segment.start_s, false);
  return segment;
}}

function renderClinicalTimeline() {{
  clinicalTimeline.innerHTML = '';
  freezingSegments.forEach((seg, idx) => {{
    const div = document.createElement('div');
    div.className = `clinical-timeline-seg${{idx === selectedClinicalIndex ? ' active' : ''}}`;
    div.style.left = `${{seg.start_s / DASHBOARD_DATA.duration_s * 100}}%`;
    div.style.width = `${{Math.max(0.22, (seg.end_s - seg.start_s) / DASHBOARD_DATA.duration_s * 100)}}%`;
    div.style.background = freezingColor(seg.marker_label);
    div.title = `${{markerDisplay(seg.marker_label)}} ${{fmt(seg.start_s,1)}}-${{fmt(seg.end_s,1)}}s`;
    div.innerHTML = `<span>${{markerDisplay(seg.marker_label)}}</span>`;
    div.addEventListener('click', (evt) => {{
      evt.stopPropagation();
      selectedClinicalIndex = idx;
      clinicalStatus.textContent = `Selected ${{markerDisplay(seg.marker_label)}} ${{fmt(seg.start_s,1)}}-${{fmt(seg.end_s,1)}}s`;
      renderClinicalTimeline();
      seekBoth(seg.start_s, true);
    }});
    clinicalTimeline.appendChild(div);
  }});
  if (clinicalStartS !== null) {{
    const marker = document.createElement('div');
    marker.className = 'manual-start-marker';
    marker.style.left = `${{clinicalStartS / DASHBOARD_DATA.duration_s * 100}}%`;
    marker.title = `clinical marker start: ${{fmt(clinicalStartS,1)}}s`;
    clinicalTimeline.appendChild(marker);
  }}
  if (clinicalDragRange) {{
    const start = Math.min(clinicalDragRange.start_s, clinicalDragRange.end_s);
    const end = Math.max(clinicalDragRange.start_s, clinicalDragRange.end_s);
    const preview = document.createElement('div');
    preview.className = 'manual-drag-preview';
    preview.style.left = `${{start / DASHBOARD_DATA.duration_s * 100}}%`;
    preview.style.width = `${{Math.max(0.18, (end - start) / DASHBOARD_DATA.duration_s * 100)}}%`;
    clinicalTimeline.appendChild(preview);
  }}
  const playhead = document.createElement('div');
  playhead.className = 'playhead';
  playhead.id = 'clinicalPlayhead';
  clinicalTimeline.appendChild(playhead);
  updatePlayhead();
}}

function scheduleManualSave() {{
  manualSegmentStatus.textContent = 'Unsaved manual label changes';
  window.clearTimeout(manualSaveTimer);
  manualSaveTimer = window.setTimeout(() => saveManualSegments(true), 700);
}}

function selectedManualSegment() {{
  if (selectedManualIndex < 0 || selectedManualIndex >= manualSegments.length) return null;
  return manualSegments[selectedManualIndex];
}}

function mergedManualCoverage() {{
  const sorted = manualSegments
    .map(seg => ({{start_s: Math.max(0, seg.start_s), end_s: Math.min(DASHBOARD_DATA.duration_s, seg.end_s)}}))
    .filter(seg => seg.end_s > seg.start_s)
    .sort((a, b) => a.start_s - b.start_s);
  const merged = [];
  for (const seg of sorted) {{
    const last = merged[merged.length - 1];
    if (last && seg.start_s <= last.end_s + 0.15) {{
      last.end_s = Math.max(last.end_s, seg.end_s);
    }} else {{
      merged.push({{...seg}});
    }}
  }}
  return merged;
}}

function unlabeledGaps(minGapS=0.2) {{
  const gaps = [];
  let cursor = 0;
  for (const seg of mergedManualCoverage()) {{
    if (seg.start_s - cursor >= minGapS) gaps.push({{start_s: cursor, end_s: seg.start_s}});
    cursor = Math.max(cursor, seg.end_s);
  }}
  if (DASHBOARD_DATA.duration_s - cursor >= minGapS) gaps.push({{start_s: cursor, end_s: DASHBOARD_DATA.duration_s}});
  return gaps;
}}

function lastManualEndBefore(timeS) {{
  let best = 0;
  for (const seg of manualSegments) {{
    if (seg.end_s <= timeS + 0.2) best = Math.max(best, seg.end_s);
  }}
  return best;
}}

function snapStartForContinuousLabel(proposedStart, proposedEnd) {{
  if (!continuousLabelToggle.checked) return proposedStart;
  for (const gap of unlabeledGaps(0.35)) {{
    if (proposedStart >= gap.start_s - 0.2 && proposedStart <= gap.end_s + 0.2 && proposedEnd > gap.start_s) {{
      return gap.start_s;
    }}
  }}
  const previousEnd = lastManualEndBefore(proposedEnd);
  if (Math.abs(proposedStart - previousEnd) <= 3.0 || proposedStart > previousEnd) return previousEnd;
  return proposedStart;
}}

function firstUnlabeledGapAtOrAfter(timeS) {{
  const gaps = unlabeledGaps(0.35);
  return gaps.find(gap => gap.end_s >= timeS) || gaps[0] || null;
}}

function constrainNewManualRange(startS, endS) {{
  const start = Math.max(0, Math.min(startS, endS));
  const end = Math.min(DASHBOARD_DATA.duration_s, Math.max(startS, endS));
  const gaps = unlabeledGaps(0.05);
  const gap = gaps.find(item => start >= item.start_s - 0.2 && start <= item.end_s + 0.2)
    || gaps.find(item => end > item.start_s && start < item.end_s);
  if (!gap) return null;
  const clippedStart = Math.max(start, gap.start_s);
  const clippedEnd = Math.min(end, gap.end_s);
  if (clippedEnd - clippedStart < 0.1) return null;
  return {{start_s: Number(clippedStart.toFixed(3)), end_s: Number(clippedEnd.toFixed(3)), gap}};
}}

function manualEditBounds(segmentId) {{
  const others = manualSegments
    .filter(seg => seg.id !== segmentId)
    .sort((a, b) => a.start_s - b.start_s);
  const current = manualSegments.find(seg => seg.id === segmentId);
  const currentStart = current ? current.start_s : 0;
  let minStart = 0;
  let maxEnd = DASHBOARD_DATA.duration_s;
  for (const seg of others) {{
    if (seg.end_s <= currentStart + 0.001) minStart = Math.max(minStart, seg.end_s);
    if (seg.start_s >= currentStart - 0.001) maxEnd = Math.min(maxEnd, seg.start_s);
  }}
  return {{minStart, maxEnd}};
}}

function createManualSegment(startS, endS, label, note='', shouldSeek=false) {{
  const constrained = constrainNewManualRange(startS, endS);
  if (!constrained) {{
    quickLabelStatus.textContent = 'This area is already labeled. New labels cannot overlap existing labels.';
    manualSegmentStatus.textContent = 'No label created because it would overlap another label';
    renderManualTimeline();
    return null;
  }}
  const now = new Date().toISOString();
  const segment = normalizeManualSegment({{
    id: `manual_${{Date.now()}}_${{Math.random().toString(16).slice(2)}}`,
    start_s: constrained.start_s,
    end_s: constrained.end_s,
    motion_label: label,
    confidence: 'medium',
    note,
    created_at: now,
    updated_at: now,
  }});
  manualSegments.push(segment);
  selectedManualIndex = manualSegments.length - 1;
  sortManualSegments();
  renderManualLayer();
  saveManualSegments(true);
  if (shouldSeek) seekBoth(segment.start_s, false);
  return segment;
}}

function renderManualScrubTimeline() {{
  manualScrubTimeline.innerHTML = '';
  currentSegments().forEach(seg => {{
    const div = document.createElement('div');
    div.className = 'timeline-seg';
    div.style.top = '0';
    div.style.bottom = '0';
    div.style.left = `${{seg.start_s / DASHBOARD_DATA.duration_s * 100}}%`;
    div.style.width = `${{Math.max(0.12, (seg.end_s - seg.start_s) / DASHBOARD_DATA.duration_s * 100)}}%`;
    div.style.background = colorFor(seg.base_motion);
    div.style.opacity = '.22';
    div.style.pointerEvents = 'none';
    manualScrubTimeline.appendChild(div);
  }});
  const track = document.createElement('div');
  track.className = 'scrub-track';
  manualScrubTimeline.appendChild(track);
  const currentTime = downwardVideo.currentTime || 0;
  const progress = document.createElement('div');
  progress.className = 'scrub-progress';
  progress.id = 'manualScrubProgress';
  progress.style.width = `${{Math.max(0, currentTime / DASHBOARD_DATA.duration_s * 100)}}%`;
  manualScrubTimeline.appendChild(progress);
  const knob = document.createElement('div');
  knob.className = 'scrub-knob';
  knob.id = 'manualScrubKnob';
  knob.style.left = `${{currentTime / DASHBOARD_DATA.duration_s * 100}}%`;
  knob.title = `${{fmt(currentTime,1)}}s`;
  manualScrubTimeline.appendChild(knob);
}}

function renderManualLayer() {{
  renderManualSegmentOptions();
  renderManualScrubTimeline();
  renderManualTimeline();
  renderManualEditor();
  renderAccuracySummary();
  renderSignalPlot();
}}

function renderManualSegmentOptions() {{
  manualSegmentSelect.innerHTML = '';
  if (!manualSegments.length) {{
    const opt = document.createElement('option');
    opt.value = '-1';
    opt.textContent = 'No manual segment yet';
    manualSegmentSelect.appendChild(opt);
    manualSegmentSelect.value = '-1';
    return;
  }}
  manualSegments.forEach((seg, idx) => {{
    const opt = document.createElement('option');
    opt.value = String(idx);
    opt.textContent = `${{String(idx + 1).padStart(2, '0')}}. ${{seg.motion_label}} | ${{fmt(seg.start_s,1)}}-${{fmt(seg.end_s,1)}}s`;
    manualSegmentSelect.appendChild(opt);
  }});
  if (selectedManualIndex < 0 || selectedManualIndex >= manualSegments.length) selectedManualIndex = 0;
  manualSegmentSelect.value = String(selectedManualIndex);
}}

function timeFromTimelineEvent(evt, element) {{
  const rect = element.getBoundingClientRect();
  const pct = Math.max(0, Math.min(1, (evt.clientX - rect.left) / rect.width));
  return pct * DASHBOARD_DATA.duration_s;
}}

function timeFromManualTimelineEvent(evt) {{
  return timeFromTimelineEvent(evt, manualTimeline);
}}

function formatQuickLabelStart() {{
  if (quickLabelStartS === null) return '';
  return `Started ${{quickManualLabelSelect.value}} at ${{fmt(quickLabelStartS,1)}}s. Click End Here + Save when the action changes.`;
}}

function renderManualTimeline() {{
  manualTimeline.innerHTML = '';
  const total = manualSegments.reduce((sum, seg) => sum + Math.max(0, seg.end_s - seg.start_s), 0);
  const gaps = unlabeledGaps(0.35);
  const gapTotal = gaps.reduce((sum, gap) => sum + Math.max(0, gap.end_s - gap.start_s), 0);
  manualCoverageSummary.textContent = manualSegments.length
    ? `${{manualSegments.length}} manual segments, ${{fmt(total,1)}}s labeled, ${{fmt(gapTotal,1)}}s unlabeled gaps`
    : 'No manual labels yet';
  gaps.forEach((gap) => {{
    const div = document.createElement('div');
    div.className = 'manual-gap';
    div.style.left = `${{gap.start_s / DASHBOARD_DATA.duration_s * 100}}%`;
    div.style.width = `${{Math.max(0.2, (gap.end_s - gap.start_s) / DASHBOARD_DATA.duration_s * 100)}}%`;
    div.title = `unlabeled gap ${{fmt(gap.start_s,1)}}-${{fmt(gap.end_s,1)}}s; click to fill with ${{quickManualLabelSelect.value}}`;
    div.addEventListener('click', (evt) => {{
      evt.stopPropagation();
      createManualSegment(gap.start_s, gap.end_s, quickManualLabelSelect.value, 'Filled unlabeled gap from manual timeline.', false);
      quickLabelStatus.textContent = `Filled gap ${{fmt(gap.start_s,1)}}-${{fmt(gap.end_s,1)}}s as ${{quickManualLabelSelect.value}}.`;
    }});
    manualTimeline.appendChild(div);
  }});
  manualSegments.forEach((seg, idx) => {{
    const div = document.createElement('div');
    div.className = `manual-timeline-seg${{idx === selectedManualIndex ? ' active' : ''}}`;
    div.style.left = `${{seg.start_s / DASHBOARD_DATA.duration_s * 100}}%`;
    div.style.width = `${{Math.max(0.28, (seg.end_s - seg.start_s) / DASHBOARD_DATA.duration_s * 100)}}%`;
    div.style.background = colorFor(seg.motion_label);
    div.title = `${{seg.motion_label}} ${{fmt(seg.start_s,1)}}-${{fmt(seg.end_s,1)}}s`;
    div.innerHTML = `<span>${{seg.motion_label}}</span>`;
    div.addEventListener('click', () => {{
      selectedManualIndex = idx;
      renderManualLayer();
      seekBoth(seg.start_s, true);
    }});
    manualTimeline.appendChild(div);
  }});
  if (quickLabelStartS !== null) {{
    const marker = document.createElement('div');
    marker.className = 'manual-start-marker';
    marker.style.left = `${{quickLabelStartS / DASHBOARD_DATA.duration_s * 100}}%`;
    marker.title = `manual label start: ${{fmt(quickLabelStartS,1)}}s`;
    manualTimeline.appendChild(marker);
    quickLabelStatus.textContent = formatQuickLabelStart();
  }}
  if (manualDragRange) {{
    const start = Math.min(manualDragRange.start_s, manualDragRange.end_s);
    const end = Math.max(manualDragRange.start_s, manualDragRange.end_s);
    const preview = document.createElement('div');
    preview.className = 'manual-drag-preview';
    preview.style.left = `${{start / DASHBOARD_DATA.duration_s * 100}}%`;
    preview.style.width = `${{Math.max(0.18, (end - start) / DASHBOARD_DATA.duration_s * 100)}}%`;
    manualTimeline.appendChild(preview);
  }}
  const playhead = document.createElement('div');
  playhead.className = 'playhead';
  playhead.id = 'manualPlayhead';
  manualTimeline.appendChild(playhead);
  updatePlayhead();
}}

function renderManualEditor() {{
  const seg = selectedManualSegment();
  const disabled = !seg;
  [manualStartInput, manualEndInput, manualGroundTruthLabelSelect, manualConfidenceSelect, manualSegmentNote].forEach(el => {{
    el.disabled = disabled;
  }});
  document.getElementById('setManualStartBtn').disabled = disabled;
  document.getElementById('setManualEndBtn').disabled = disabled;
  document.getElementById('saveManualSegmentBtn').disabled = disabled;
  document.getElementById('deleteManualSegmentBtn').disabled = disabled;
  if (!seg) {{
    manualStartInput.value = '';
    manualEndInput.value = '';
    manualGroundTruthLabelSelect.value = 'standing';
    manualConfidenceSelect.value = 'medium';
    manualSegmentNote.value = '';
    return;
  }}
  manualStartInput.value = fmt(seg.start_s, 1);
  manualEndInput.value = fmt(seg.end_s, 1);
  manualGroundTruthLabelSelect.value = seg.motion_label;
  manualConfidenceSelect.value = seg.confidence;
  manualSegmentNote.value = seg.note || '';
}}

function normalizeMotionLabel(label) {{
  return String(label || '').toLowerCase().replace(/^go\\s+/, '').trim();
}}

function intervalOverlap(aStart, aEnd, bStart, bEnd) {{
  return Math.max(0, Math.min(aEnd, bEnd) - Math.max(aStart, bStart));
}}

function calculateManualAccuracy() {{
  const modelSegments = currentSegments();
  if (!manualSegments.length || !modelSegments.length) return null;
  let total = 0;
  let correct = 0;
  const byLabel = {{}};
  for (const manual of manualSegments) {{
    const manualLabel = normalizeMotionLabel(manual.motion_label);
    if (!byLabel[manualLabel]) byLabel[manualLabel] = {{correct: 0, total: 0}};
    for (const model of modelSegments) {{
      const overlap = intervalOverlap(manual.start_s, manual.end_s, model.start_s, model.end_s);
      if (overlap <= 0) continue;
      const predicted = normalizeMotionLabel(model.base_motion || model.label);
      total += overlap;
      byLabel[manualLabel].total += overlap;
      if (predicted === manualLabel) {{
        correct += overlap;
        byLabel[manualLabel].correct += overlap;
      }}
    }}
  }}
  const labeledDuration = manualSegments.reduce((sum, seg) => sum + Math.max(0, seg.end_s - seg.start_s), 0);
  return {{
    total,
    correct,
    accuracy: total > 0 ? correct / total : 0,
    labeledDuration,
    coverage: DASHBOARD_DATA.duration_s > 0 ? labeledDuration / DASHBOARD_DATA.duration_s : 0,
    byLabel,
  }};
}}

function renderAccuracySummary() {{
  const summary = calculateManualAccuracy();
  const accuracyEl = document.getElementById('metricManualAccuracy');
  const coverageEl = document.getElementById('metricManualCoverage');
  if (!summary) {{
    accuracyEl.textContent = '--';
    coverageEl.textContent = '--';
    accuracyEl.title = 'Add manual action labels to calculate accuracy.';
    coverageEl.title = 'Add manual action labels to calculate coverage.';
    return;
  }}
  accuracyEl.textContent = `${{(summary.accuracy * 100).toFixed(1)}}%`;
  coverageEl.textContent = `${{(summary.coverage * 100).toFixed(1)}}%`;
  accuracyEl.title = `Correct ${{fmt(summary.correct,1)}}s / compared ${{fmt(summary.total,1)}}s using ${{smoothingKey}} smoothing.`;
  coverageEl.title = `Manual action labels cover ${{fmt(summary.labeledDuration,1)}}s / ${{fmt(DASHBOARD_DATA.duration_s,1)}}s.`;
}}

function updateSelectedManualFromEditor() {{
  const seg = selectedManualSegment();
  if (!seg) return;
  const selectedId = seg.id;
  const bounds = manualEditBounds(selectedId);
  const rawStart = Math.max(0, Math.min(DASHBOARD_DATA.duration_s - 0.1, Number(manualStartInput.value)));
  const rawEnd = Math.max(rawStart + 0.1, Math.min(DASHBOARD_DATA.duration_s, Number(manualEndInput.value)));
  const start = Math.max(bounds.minStart, Math.min(bounds.maxEnd - 0.1, rawStart));
  const end = Math.max(start + 0.1, Math.min(bounds.maxEnd, rawEnd));
  if (end - start < 0.1) {{
    manualSegmentStatus.textContent = 'Cannot resize this label into another label. Delete or edit the neighboring label first.';
    renderManualEditor();
    return;
  }}
  if (Math.abs(start - rawStart) > 0.001 || Math.abs(end - rawEnd) > 0.001) {{
    manualSegmentStatus.textContent = 'Adjusted to avoid overlapping another manual label';
  }}
  seg.start_s = Number(start.toFixed(3));
  seg.end_s = Number(end.toFixed(3));
  seg.motion_label = manualGroundTruthLabelSelect.value;
  seg.confidence = manualConfidenceSelect.value;
  seg.note = manualSegmentNote.value;
  seg.updated_at = new Date().toISOString();
  sortManualSegments();
  selectedManualIndex = manualSegments.findIndex(item => item.id === selectedId);
  renderManualLayer();
  scheduleManualSave();
}}

function renderNoteForSelected() {{
  const seg = currentSegments()[selectedIndex];
  if (!seg) return;
  const key = noteKeyForSegment(seg);
  const saved = manualNotes[key] || {{}};
  manualLabelSelect.value = saved.manual_label || '';
  manualNote.value = saved.note || '';
  needsReviewToggle.checked = Boolean(saved.needs_review);
  if (saved.manual_label) {{
    manualCorrectionStatus.textContent = `Saved correction: detected ${{seg.base_motion}} -> manual ${{saved.manual_label}}.`;
  }} else {{
    manualCorrectionStatus.textContent = `No manual correction. Effective label remains detected ${{seg ? seg.base_motion : '--'}}.`;
  }}
  if (saved.manual_label || saved.note || saved.needs_review) {{
    noteStatus.textContent = saved.manual_label ? 'Saved manual correction' : (saved.needs_review ? 'Saved: needs closer look' : 'Saved note');
  }} else {{
    noteStatus.textContent = 'No manual review for this period';
  }}
}}

async function saveCurrentNote() {{
  const seg = currentSegments()[selectedIndex];
  if (!seg) return;
  const key = noteKeyForSegment(seg);
  const payload = {{
    trial_id: DASHBOARD_DATA.trial_id,
    note_key: key,
    segment_id: seg.id,
    smoothing_key: smoothingKey,
    start_s: seg.start_s,
    end_s: seg.end_s,
    detected_motion: seg.base_motion,
    manual_label: manualLabelSelect.value,
    base_motion: seg.base_motion,
    clinical_marker: seg.clinical_marker || '',
    needs_review: needsReviewToggle.checked,
    note: manualNote.value,
  }};
  manualNotes[key] = payload;
  window.localStorage.setItem(localNoteStorageKey(), JSON.stringify(manualNotes));
  noteStatus.textContent = 'Saving...';
  try {{
    const response = await fetch('/api/parkinson-notes', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(payload),
    }});
    if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
    const saved = await response.json();
    manualNotes = saved.notes || manualNotes;
    window.localStorage.setItem(localNoteStorageKey(), JSON.stringify(manualNotes));
    noteStatus.textContent = 'Saved to file';
  }} catch (err) {{
    noteStatus.textContent = 'Saved locally; restart server to save to file';
  }}
}}

function syncVideos(source) {{
  if (syncing) return;
  const other = source === downwardVideo ? ariaVideo : downwardVideo;
  if (Math.abs(other.currentTime - source.currentTime) > 0.18) {{
    syncing = true;
    other.currentTime = source.currentTime;
    setTimeout(() => syncing = false, 80);
  }}
}}

function seekBoth(t, autoplay=false) {{
  const clamped = Math.max(0, Math.min(DASHBOARD_DATA.duration_s, t));
  downwardVideo.currentTime = clamped;
  ariaVideo.currentTime = clamped;
  updatePlayhead();
  if (autoplay) playBoth();
}}

function updatePlaybackButtons(isPlaying) {{
  playPauseBtn.textContent = isPlaying ? 'Pause' : 'Play';
  manualScrubPlayPauseBtn.textContent = isPlaying ? 'Pause Video' : 'Play Video';
}}

function playBoth() {{
  Promise.allSettled([downwardVideo.play(), ariaVideo.play()]);
  updatePlaybackButtons(true);
}}

function pauseBoth() {{
  downwardVideo.pause();
  ariaVideo.pause();
  updatePlaybackButtons(false);
}}

function renderSelectors() {{
  smoothingSelect.innerHTML = '';
  Object.entries(DASHBOARD_DATA.smoothing).forEach(([key, bundle]) => {{
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = bundle.label;
    smoothingSelect.appendChild(opt);
  }});
  smoothingSelect.value = smoothingKey;
  renderSegmentOptions();
}}

function renderSegmentOptions() {{
  const segments = currentSegments();
  segmentSelect.innerHTML = '';
  segments.forEach((seg, idx) => {{
    const opt = document.createElement('option');
    opt.value = String(idx);
    opt.textContent = `${{String(idx + 1).padStart(2, '0')}}. ${{seg.base_motion}} | ${{fmt(seg.start_s,1)}}-${{fmt(seg.end_s,1)}}s`;
    segmentSelect.appendChild(opt);
  }});
  if (selectedIndex >= segments.length) selectedIndex = Math.max(0, segments.length - 1);
  segmentSelect.value = String(selectedIndex);
}}

function renderMetrics() {{
  const bundle = currentBundle();
  document.getElementById('metricSegments').textContent = String(bundle.segments.length);
  document.getElementById('metricDuration').textContent = `${{fmt(DASHBOARD_DATA.duration_s,0)}}s`;
  document.getElementById('metricSmooth').textContent = `${{bundle.thresholds.smooth_s}}s`;
  document.getElementById('metricContext').textContent = `±${{bundle.thresholds.context_s}}s`;
  document.getElementById('timeTotal').textContent = fmt(DASHBOARD_DATA.duration_s,1);
  renderAccuracySummary();
}}

function renderLegend() {{
  const legend = document.getElementById('legend');
  legend.innerHTML = '';
  Object.entries(MOTION_COLORS).forEach(([label, color]) => {{
    if (label === 'unknown') return;
    const item = document.createElement('span');
    item.innerHTML = `<i class=\"swatch\" style=\"background:${{color}}\"></i>${{label}}`;
    legend.appendChild(item);
  }});
}}

function renderTimeline() {{
  const segments = currentSegments();
  const selected = segments[selectedIndex];
  timeline.innerHTML = '';
  segments.forEach((seg, idx) => {{
    const div = document.createElement('div');
    div.className = 'timeline-seg';
    div.style.left = `${{seg.start_s / DASHBOARD_DATA.duration_s * 100}}%`;
    div.style.width = `${{Math.max(0.18, (seg.end_s - seg.start_s) / DASHBOARD_DATA.duration_s * 100)}}%`;
    div.style.background = colorFor(seg.base_motion);
    div.title = `${{seg.label}} ${{fmt(seg.start_s,1)}}-${{fmt(seg.end_s,1)}}s`;
    div.addEventListener('click', () => selectSegment(idx, true));
    timeline.appendChild(div);
  }});
  if (selected) {{
    const ctx = document.createElement('div');
    ctx.className = 'timeline-context';
    ctx.style.left = `${{selected.context_start_s / DASHBOARD_DATA.duration_s * 100}}%`;
    ctx.style.width = `${{(selected.context_end_s - selected.context_start_s) / DASHBOARD_DATA.duration_s * 100}}%`;
    timeline.appendChild(ctx);
    const sel = document.createElement('div');
    sel.className = 'timeline-selected';
    sel.style.left = `${{selected.start_s / DASHBOARD_DATA.duration_s * 100}}%`;
    sel.style.width = `${{Math.max(0.25, (selected.end_s - selected.start_s) / DASHBOARD_DATA.duration_s * 100)}}%`;
    timeline.appendChild(sel);
  }}
  const playhead = document.createElement('div');
  playhead.className = 'playhead';
  playhead.id = 'playhead';
  timeline.appendChild(playhead);
  updatePlayhead();
}}

function scale(value, min, max, outMin, outMax) {{
  if (max <= min) return (outMin + outMax) / 2;
  return outMin + (value - min) / (max - min) * (outMax - outMin);
}}

function pointsFor(records, key, min, max, xMin, xMax, yMin, yMax) {{
  return records.map(r => `${{scale(r.time_s, xMin, xMax, 54, 842).toFixed(1)}},${{scale(r[key], min, max, yMax, yMin).toFixed(1)}}`).join(' ');
}}

function basePlotBoundsForSegment(seg) {{
  if (!seg) return [0, DASHBOARD_DATA.duration_s];
  if (plotWindowMode === 'full') return [0, DASHBOARD_DATA.duration_s];
  if (plotWindowMode === 'wide') {{
    const center = (seg.start_s + seg.end_s) / 2;
    const halfWindow = Math.max(60, (seg.end_s - seg.start_s) * 2.5);
    return [Math.max(0, center - halfWindow), Math.min(DASHBOARD_DATA.duration_s, center + halfWindow)];
  }}
  return [seg.context_start_s, seg.context_end_s];
}}

function plotBoundsForSegment(seg) {{
  if (!seg) return [0, DASHBOARD_DATA.duration_s];
  const [baseMin, baseMax] = basePlotBoundsForSegment(seg);
  const center = (seg.start_s + seg.end_s) / 2;
  const baseSpan = Math.max(3.0, baseMax - baseMin);
  const targetSpan = Math.min(DASHBOARD_DATA.duration_s, Math.max(3.0, baseSpan * plotZoomScale));
  let start = center - targetSpan / 2;
  let end = center + targetSpan / 2;
  if (start < 0) {{
    end = Math.min(DASHBOARD_DATA.duration_s, end - start);
    start = 0;
  }}
  if (end > DASHBOARD_DATA.duration_s) {{
    start = Math.max(0, start - (end - DASHBOARD_DATA.duration_s));
    end = DASHBOARD_DATA.duration_s;
  }}
  return [start, end];
}}

function setPlotZoom(nextScale) {{
  plotZoomScale = Math.max(0.12, Math.min(12, nextScale));
  renderSignalPlot();
  renderXYPlot();
}}

function renderSignalPlot() {{
  const segments = currentSegments();
  const selected = segments[selectedIndex];
  const records = currentRecords();
  if (!selected || !records.length) return;
  const [xMin, xMax] = plotBoundsForSegment(selected);
  const rows = records.filter(r => r.time_s >= xMin && r.time_s <= xMax);
  const zVals = rows.map(r => r.z_smooth_m);
  const sVals = rows.map(r => r.mean_speed_mps);
  const vVals = rows.map(r => r.vertical_delta_m);
  const zMin = Math.min(...zVals), zMax = Math.max(...zVals);
  const speedMin = 0, speedMax = Math.max(...sVals, 0.1);
  const vertAbs = Math.max(...vVals.map(v => Math.abs(v)), 0.08);
  const zPts = pointsFor(rows, 'z_smooth_m', zMin, zMax, xMin, xMax, 48, 188);
  const speedPts = pointsFor(rows, 'mean_speed_mps', speedMin, speedMax, xMin, xMax, 214, 344);
  const vertPts = pointsFor(rows, 'vertical_delta_m', -vertAbs, vertAbs, xMin, xMax, 214, 344);
  const sx = scale(selected.start_s, xMin, xMax, 54, 842);
  const ex = scale(selected.end_s, xMin, xMax, 54, 842);
  const nowX = scale(downwardVideo.currentTime || 0, xMin, xMax, 54, 842);
  const csx = scale(selected.context_start_s, xMin, xMax, 54, 842);
  const cex = scale(selected.context_end_s, xMin, xMax, 54, 842);
  const manualRects = manualSegments
    .filter(seg => seg.end_s >= xMin && seg.start_s <= xMax)
    .map(seg => {{
      const msx = scale(Math.max(seg.start_s, xMin), xMin, xMax, 54, 842);
      const mex = scale(Math.min(seg.end_s, xMax), xMin, xMax, 54, 842);
      const width = Math.max(2, mex - msx);
      return `<rect x=\"${{msx}}\" y=\"58\" width=\"${{width}}\" height=\"18\" rx=\"7\" fill=\"${{colorFor(seg.motion_label)}}\" opacity=\".55\" stroke=\"#17231c\" stroke-width=\"1\"/>
        <text x=\"${{msx + 4}}\" y=\"72\" font-size=\"10\" font-weight=\"900\" fill=\"#17231c\">${{seg.motion_label}}</text>`;
    }})
    .join('');
  const freezingRects = freezingSegments
    .filter(seg => seg.end_s >= xMin && seg.start_s <= xMax)
    .map(seg => {{
      const fsx = scale(Math.max(seg.start_s, xMin), xMin, xMax, 54, 842);
      const fex = scale(Math.min(seg.end_s, xMax), xMin, xMax, 54, 842);
      const width = Math.max(2, fex - fsx);
      return `<rect x=\"${{fsx}}\" y=\"80\" width=\"${{width}}\" height=\"16\" rx=\"7\" fill=\"${{freezingColor(seg.marker_label)}}\" opacity=\".66\" stroke=\"#17231c\" stroke-width=\"1\"/>
        <text x=\"${{fsx + 4}}\" y=\"92\" font-size=\"10\" font-weight=\"900\" fill=\"#17231c\">${{markerDisplay(seg.marker_label)}}</text>`;
    }})
    .join('');
  signalPlot.innerHTML = `
    <rect x=\"0\" y=\"0\" width=\"900\" height=\"420\" fill=\"#fff\"/>
    <rect x=\"54\" y=\"34\" width=\"788\" height=\"332\" rx=\"14\" fill=\"#fbfcfb\" stroke=\"#dce5dc\"/>
    <rect x=\"${{csx}}\" y=\"34\" width=\"${{Math.max(1, cex - csx)}}\" height=\"332\" fill=\"rgba(28,125,88,.07)\"/>
    <rect x=\"${{sx}}\" y=\"34\" width=\"${{Math.max(2, ex - sx)}}\" height=\"332\" fill=\"rgba(22,120,82,.16)\"/>
    ${{manualRects}}
    ${{freezingRects}}
    <line x1=\"${{sx}}\" y1=\"34\" x2=\"${{sx}}\" y2=\"366\" stroke=\"#167852\" stroke-width=\"2\" stroke-dasharray=\"6 6\"/>
    <line x1=\"${{ex}}\" y1=\"34\" x2=\"${{ex}}\" y2=\"366\" stroke=\"#167852\" stroke-width=\"2\" stroke-dasharray=\"6 6\"/>
    <line x1=\"54\" y1=\"188\" x2=\"842\" y2=\"188\" stroke=\"#dce5dc\"/>
    <line x1=\"54\" y1=\"344\" x2=\"842\" y2=\"344\" stroke=\"#dce5dc\"/>
    <polyline points=\"${{zPts}}\" fill=\"none\" stroke=\"#26313a\" stroke-width=\"3\"/>
    <polyline points=\"${{speedPts}}\" fill=\"none\" stroke=\"#d66b2e\" stroke-width=\"3\"/>
    <polyline points=\"${{vertPts}}\" fill=\"none\" stroke=\"#2d6aa0\" stroke-width=\"2\" stroke-dasharray=\"5 5\"/>
    <line x1=\"${{nowX}}\" y1=\"34\" x2=\"${{nowX}}\" y2=\"366\" stroke=\"#111\" stroke-width=\"2\" opacity=\".75\"/>
    <text x=\"54\" y=\"24\" font-size=\"14\" font-weight=\"800\" fill=\"#26313a\">height z (top panel)</text>
    <text x=\"54\" y=\"205\" font-size=\"14\" font-weight=\"800\" fill=\"#d66b2e\">speed</text>
    <text x=\"150\" y=\"205\" font-size=\"14\" font-weight=\"800\" fill=\"#2d6aa0\">vertical trend</text>
    <text x=\"678\" y=\"24\" font-size=\"13\" fill=\"#5f6b61\">view: ${{plotWindowMode}}, zoom x${{fmt(1 / plotZoomScale,2)}}</text>
    <text x=\"54\" y=\"390\" font-size=\"13\" fill=\"#5f6b61\">${{fmt(xMin,1)}}s</text>
    <text x=\"802\" y=\"390\" font-size=\"13\" fill=\"#5f6b61\">${{fmt(xMax,1)}}s</text>
    <text x=\"${{(sx+ex)/2}}\" y=\"54\" text-anchor=\"middle\" font-size=\"13\" font-weight=\"900\" fill=\"#07563a\">detected period</text>
  `;
}}

function renderXYPlot() {{
  const segments = currentSegments();
  const selected = segments[selectedIndex];
  const records = currentRecords();
  if (!selected || !records.length) return;
  const [timeMin, timeMax] = plotBoundsForSegment(selected);
  const rows = records.filter(r => r.time_s >= timeMin && r.time_s <= timeMax);
  const xs = rows.map(r => r.x_smooth_m);
  const ys = rows.map(r => r.y_smooth_m);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const padX = Math.max((xMax - xMin) * .12, .2);
  const padY = Math.max((yMax - yMin) * .12, .2);
  const minX = xMin - padX, maxX = xMax + padX, minY = yMin - padY, maxY = yMax + padY;
  const pts = rows.map(r => `${{scale(r.x_smooth_m, minX, maxX, 46, 594).toFixed(1)}},${{scale(r.y_smooth_m, minY, maxY, 398, 44).toFixed(1)}}`).join(' ');
  const selectedRows = rows.filter(r => r.time_s >= selected.start_s && r.time_s <= selected.end_s);
  const selectedPts = selectedRows.map(r => `${{scale(r.x_smooth_m, minX, maxX, 46, 594).toFixed(1)}},${{scale(r.y_smooth_m, minY, maxY, 398, 44).toFixed(1)}}`).join(' ');
  let arrows = '';
  const stride = Math.max(1, Math.floor(rows.length / 10));
  for (let i = stride; i < rows.length; i += stride) {{
    const a = rows[i - stride], b = rows[i];
    const x1 = scale(a.x_smooth_m, minX, maxX, 46, 594), y1 = scale(a.y_smooth_m, minY, maxY, 398, 44);
    const x2 = scale(b.x_smooth_m, minX, maxX, 46, 594), y2 = scale(b.y_smooth_m, minY, maxY, 398, 44);
    arrows += `<line x1=\"${{x1}}\" y1=\"${{y1}}\" x2=\"${{x2}}\" y2=\"${{y2}}\" stroke=\"#167852\" stroke-width=\"2\" marker-end=\"url(#arrow)\" opacity=\".8\"/>`;
  }}
  xyPlot.innerHTML = `
    <defs><marker id=\"arrow\" markerWidth=\"7\" markerHeight=\"7\" refX=\"6\" refY=\"3.5\" orient=\"auto\"><polygon points=\"0 0, 7 3.5, 0 7\" fill=\"#167852\"/></marker></defs>
    <rect x=\"0\" y=\"0\" width=\"640\" height=\"460\" fill=\"#fff\"/>
    <rect x=\"46\" y=\"34\" width=\"548\" height=\"364\" rx=\"14\" fill=\"#fbfcfb\" stroke=\"#dce5dc\"/>
    <polyline points=\"${{pts}}\" fill=\"none\" stroke=\"#c8d0ca\" stroke-width=\"7\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>
    <polyline points=\"${{selectedPts}}\" fill=\"none\" stroke=\"#17231c\" stroke-width=\"4\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>
    ${{arrows}}
    <text x=\"46\" y=\"24\" font-size=\"14\" font-weight=\"900\" fill=\"#26313a\">XY path in context; black = detected period</text>
    <text x=\"48\" y=\"430\" font-size=\"13\" fill=\"#5f6b61\">x/y are MPS world coordinates; view ${{fmt(timeMin,1)}}-${{fmt(timeMax,1)}}s; black = detected period.</text>
  `;
}}

function renderDetails() {{
  const seg = currentSegments()[selectedIndex];
  if (!seg) return;
  document.getElementById('selectedTitle').textContent = seg.label;
  document.getElementById('selectedTime').textContent = `detected ${{fmt(seg.start_s,1)}}s-${{fmt(seg.end_s,1)}}s, reviewed with context ${{fmt(seg.context_start_s,1)}}s-${{fmt(seg.context_end_s,1)}}s`;
  document.getElementById('selectedConfidence').textContent = `${{seg.confidence}} confidence`;
  document.getElementById('selectedReason').textContent = seg.reason || 'No reason recorded.';
  const rows = [
    ['Before', seg.summary_before],
    ['During', seg.summary_during],
    ['After', seg.summary_after],
  ];
  document.getElementById('contextTable').innerHTML = `
    <tr><th>Period</th><th>Speed</th><th>Path</th><th>Vertical</th><th>Cadence proxy</th></tr>
    ${{rows.map(([label, s]) => `<tr><td>${{label}}</td><td>${{fmt(s.mean_speed_mps,3)}} m/s</td><td>${{fmt(s.path_length_m,2)}} m</td><td>${{fmt(s.vertical_delta_m,2)}} m</td><td>${{fmt(s.cadence_proxy_spm,1)}} /min</td></tr>`).join('')}}
  `;
}}

function renderThresholds() {{
  const t = currentBundle().thresholds;
  const rows = [
    ['Moving speed threshold', `${{fmt(t.speed_move_mps,3)}} m/s`, 'Above this suggests active movement.'],
    ['Strong movement threshold', `${{fmt(t.speed_strong_mps,3)}} m/s`, 'Higher-confidence walking/travel.'],
    ['Horizontal path threshold', `${{fmt(t.path_move_m,2)}} m`, 'Cumulative x/y path within the analysis window.'],
    ['Stair vertical threshold', `${{fmt(t.vertical_stair_delta_m,2)}} m`, 'Positive means upstairs; negative means downstairs.'],
    ['Posture transition threshold', `${{fmt(t.posture_transition_delta_m,2)}} m`, 'Height change used to confirm sit-down or stand-up context.'],
    ['Posture split reference', `${{fmt(t.posture_split_z_m,2)}} m`, 'Used only as support; sitting now also needs transition context.'],
    ['Low progress threshold', `${{fmt(t.progress_low_m,2)}} m`, 'Separates real travel from nearly-stationary periods.'],
    ['High cadence proxy', `${{fmt(t.cadence_high_spm,1)}} /min`, 'Repeated height minima proxy, not final step cadence.'],
    ['Short step proxy', `${{fmt(t.step_length_short_m,2)}} m`, 'Supports review of short walking steps.'],
  ];
  document.getElementById('thresholdTable').innerHTML = `<tr><th>Rule value</th><th>Current threshold</th><th>Meaning</th></tr>${{rows.map(r => `<tr><td>${{r[0]}}</td><td>${{r[1]}}</td><td>${{r[2]}}</td></tr>`).join('')}}`;
}}

function updatePlayhead() {{
  const currentTime = downwardVideo.currentTime || 0;
  document.getElementById('timeNow').textContent = fmt(currentTime, 1);
  const ph = document.getElementById('playhead');
  if (ph) ph.style.left = `${{currentTime / DASHBOARD_DATA.duration_s * 100}}%`;
  const manualPh = document.getElementById('manualPlayhead');
  if (manualPh) manualPh.style.left = `${{currentTime / DASHBOARD_DATA.duration_s * 100}}%`;
  const clinicalPh = document.getElementById('clinicalPlayhead');
  if (clinicalPh) clinicalPh.style.left = `${{currentTime / DASHBOARD_DATA.duration_s * 100}}%`;
  const scrubProgress = document.getElementById('manualScrubProgress');
  if (scrubProgress) scrubProgress.style.width = `${{currentTime / DASHBOARD_DATA.duration_s * 100}}%`;
  const scrubKnob = document.getElementById('manualScrubKnob');
  if (scrubKnob) {{
    scrubKnob.style.left = `${{currentTime / DASHBOARD_DATA.duration_s * 100}}%`;
    scrubKnob.title = `${{fmt(currentTime,1)}}s`;
  }}
  const seg = currentSegments()[selectedIndex];
  const liveIndex = segmentIndexAtTime(currentTime);
  if (autoFollow && liveIndex >= 0 && liveIndex !== selectedIndex) {{
    selectSegment(liveIndex, false, false);
    return;
  }}
  if (stopAt && currentTime >= stopAt - 0.04) {{
    stopAt = null;
    pauseBoth();
  }}
  if (seg && currentTime >= seg.context_start_s && currentTime <= seg.context_end_s) {{
    renderSignalPlot();
  }}
}}

function selectSegment(idx, autoplay=false, shouldSeek=true) {{
  selectedIndex = Math.max(0, Math.min(currentSegments().length - 1, idx));
  segmentSelect.value = String(selectedIndex);
  const seg = currentSegments()[selectedIndex];
  renderTimeline();
  renderDetails();
  renderSignalPlot();
  renderXYPlot();
  renderNoteForSelected();
  if (seg && shouldSeek) seekBoth(seg.start_s, autoplay);
}}

function renderAll() {{
  renderMetrics();
  renderSegmentOptions();
  renderLegend();
  renderTimeline();
  renderDetails();
  renderSignalPlot();
  renderXYPlot();
  renderThresholds();
  renderNoteForSelected();
  renderManualScrubTimeline();
}}

playPauseBtn.addEventListener('click', () => downwardVideo.paused ? playBoth() : pauseBoth());
manualScrubPlayPauseBtn.addEventListener('click', () => downwardVideo.paused ? playBoth() : pauseBoth());
downwardVideo.addEventListener('click', () => downwardVideo.paused ? playBoth() : pauseBoth());
ariaVideo.addEventListener('click', () => ariaVideo.paused ? playBoth() : pauseBoth());
downwardVideo.addEventListener('timeupdate', () => {{ syncVideos(downwardVideo); updatePlayhead(); }});
ariaVideo.addEventListener('timeupdate', () => syncVideos(ariaVideo));
downwardVideo.addEventListener('pause', () => {{ if (ariaVideo.paused === false) ariaVideo.pause(); updatePlaybackButtons(false); }});
ariaVideo.addEventListener('pause', () => {{ if (downwardVideo.paused === false) downwardVideo.pause(); updatePlaybackButtons(false); }});
downwardVideo.addEventListener('play', () => updatePlaybackButtons(true));
smoothingSelect.addEventListener('change', () => {{
  smoothingKey = smoothingSelect.value;
  selectedIndex = 0;
  stopAt = null;
  renderAll();
  const first = currentSegments()[0];
  if (first) seekBoth(first.start_s, false);
}});
autoFollowToggle.addEventListener('change', () => {{
  autoFollow = autoFollowToggle.checked;
}});
plotWindowSelect.addEventListener('change', () => {{
  plotWindowMode = plotWindowSelect.value;
  plotZoomScale = 1;
  renderSignalPlot();
  renderXYPlot();
}});
document.getElementById('zoomInBtn').addEventListener('click', () => setPlotZoom(plotZoomScale * 0.62));
document.getElementById('zoomOutBtn').addEventListener('click', () => setPlotZoom(plotZoomScale * 1.62));
document.getElementById('resetZoomBtn').addEventListener('click', () => setPlotZoom(1));
saveNoteBtn.addEventListener('click', () => saveCurrentNote());
manualLabelSelect.addEventListener('change', () => {{
  const seg = currentSegments()[selectedIndex];
  if (manualLabelSelect.value) {{
    manualCorrectionStatus.textContent = `Unsaved correction: detected ${{seg.base_motion}} -> manual ${{manualLabelSelect.value}}.`;
  }} else {{
    manualCorrectionStatus.textContent = `Unsaved: keep detected label ${{seg.base_motion}}.`;
  }}
  noteStatus.textContent = 'Unsaved changes';
}});
manualNote.addEventListener('change', () => {{
  noteStatus.textContent = 'Unsaved changes';
}});
needsReviewToggle.addEventListener('change', () => {{
  noteStatus.textContent = 'Unsaved changes';
}});
manualSegmentSelect.addEventListener('change', () => {{
  selectedManualIndex = Number(manualSegmentSelect.value);
  renderManualLayer();
  const seg = selectedManualSegment();
  if (seg) seekBoth(seg.start_s, true);
}});
document.getElementById('addManualFromPeriodBtn').addEventListener('click', () => {{
  const seg = currentSegments()[selectedIndex];
  if (!seg) return;
  quickManualLabelSelect.value = seg.base_motion;
  createManualSegment(seg.start_s, seg.end_s, seg.base_motion, 'Initialized from detected period.', true);
}});
document.getElementById('addManualAtTimeBtn').addEventListener('click', () => {{
  const now = downwardVideo.currentTime || 0;
  const liveIndex = segmentIndexAtTime(now);
  const liveSeg = liveIndex >= 0 ? currentSegments()[liveIndex] : currentSegments()[selectedIndex];
  const label = liveSeg?.base_motion || 'standing';
  quickManualLabelSelect.value = label;
  createManualSegment(Math.max(0, now - 4), Math.min(DASHBOARD_DATA.duration_s, now + 4), label, 'Initialized around current video time.', true);
}});
function startQuickLabelAtCurrentTime() {{
  quickLabelStartS = Math.max(0, Math.min(DASHBOARD_DATA.duration_s, downwardVideo.currentTime || 0));
  if (continuousLabelToggle.checked) {{
    const previousEnd = lastManualEndBefore(quickLabelStartS + 0.01);
    if (Math.abs(quickLabelStartS - previousEnd) <= 3.0 || quickLabelStartS >= previousEnd) quickLabelStartS = previousEnd;
  }}
  quickLabelStatus.textContent = formatQuickLabelStart();
  renderManualTimeline();
}}

function finishQuickLabelAtCurrentTime() {{
  if (quickLabelStartS === null) {{
    quickLabelStatus.textContent = 'Click Start Label Here first, or drag directly on the manual bar.';
    return;
  }}
  const now = Math.max(0, Math.min(DASHBOARD_DATA.duration_s, downwardVideo.currentTime || 0));
  const start = Math.min(quickLabelStartS, now);
  const end = Math.max(quickLabelStartS, now);
  if (end - start < 0.35) {{
    quickLabelStatus.textContent = 'The label range is too short. Move the video a little farther, then end again.';
    return;
  }}
  const label = quickManualLabelSelect.value;
  quickLabelStartS = null;
  const snappedStart = snapStartForContinuousLabel(start, end);
  const created = createManualSegment(snappedStart, end, label, 'Created with Start Label Here / End Here workflow.', false);
  if (created) quickLabelStatus.textContent = `Saved ${{label}} from ${{fmt(created.start_s,1)}}s to ${{fmt(created.end_s,1)}}s.`;
}}

document.getElementById('quickStartLabelBtn').addEventListener('click', startQuickLabelAtCurrentTime);
document.getElementById('quickEndLabelBtn').addEventListener('click', finishQuickLabelAtCurrentTime);
document.getElementById('cancelQuickLabelBtn').addEventListener('click', () => {{
  quickLabelStartS = null;
  quickLabelStatus.textContent = 'Canceled. You can drag the manual bar or start a new label.';
  renderManualTimeline();
}});
document.getElementById('clinicalStartBtn').addEventListener('click', () => {{
  clinicalStartS = Math.max(0, Math.min(DASHBOARD_DATA.duration_s, downwardVideo.currentTime || 0));
  clinicalStatus.textContent = `Started ${{markerDisplay(clinicalMarkerSelect.value)}} marker at ${{fmt(clinicalStartS,1)}}s.`;
  renderClinicalTimeline();
}});
document.getElementById('clinicalEndBtn').addEventListener('click', () => {{
  if (clinicalStartS === null) {{
    clinicalStatus.textContent = 'Click Start Marker Here first, or drag directly on the freezing/gait marker row.';
    return;
  }}
  const now = Math.max(0, Math.min(DASHBOARD_DATA.duration_s, downwardVideo.currentTime || 0));
  const start = Math.min(clinicalStartS, now);
  const end = Math.max(clinicalStartS, now);
  const created = createFreezingSegment(start, end, clinicalMarkerSelect.value, 'Created with Start Marker Here / End Marker workflow.', false);
  clinicalStartS = null;
  if (created) clinicalStatus.textContent = `Saved ${{markerDisplay(created.marker_label)}} marker ${{fmt(created.start_s,1)}}-${{fmt(created.end_s,1)}}s.`;
}});
document.getElementById('clinicalCancelBtn').addEventListener('click', () => {{
  clinicalStartS = null;
  clinicalStatus.textContent = 'Canceled marker start.';
  renderClinicalTimeline();
}});
document.getElementById('clinicalDeleteBtn').addEventListener('click', () => {{
  if (selectedClinicalIndex < 0) {{
    clinicalStatus.textContent = 'Select a freezing/gait marker first.';
    return;
  }}
  freezingSegments.splice(selectedClinicalIndex, 1);
  selectedClinicalIndex = Math.min(selectedClinicalIndex, freezingSegments.length - 1);
  renderClinicalTimeline();
  saveFreezingSegments(true);
}});
document.getElementById('startNextGapBtn').addEventListener('click', () => {{
  const gap = firstUnlabeledGapAtOrAfter(downwardVideo.currentTime || 0);
  if (!gap) {{
    quickLabelStatus.textContent = 'No unlabeled gap found. The session appears fully covered.';
    return;
  }}
  quickLabelStartS = gap.start_s;
  seekBoth(gap.start_s, false);
  quickLabelStatus.textContent = `Started ${{quickManualLabelSelect.value}} at next gap ${{fmt(gap.start_s,1)}}s. End before ${{fmt(gap.end_s,1)}}s or continue if this action lasts longer.`;
  renderManualTimeline();
}});
document.getElementById('setManualStartBtn').addEventListener('click', () => {{
  const seg = selectedManualSegment();
  if (!seg) return;
  manualStartInput.value = fmt(Math.min(downwardVideo.currentTime || 0, seg.end_s - 0.1), 1);
  updateSelectedManualFromEditor();
}});
document.getElementById('setManualEndBtn').addEventListener('click', () => {{
  const seg = selectedManualSegment();
  if (!seg) return;
  manualEndInput.value = fmt(Math.max(downwardVideo.currentTime || 0, seg.start_s + 0.1), 1);
  updateSelectedManualFromEditor();
}});
document.getElementById('saveManualSegmentBtn').addEventListener('click', () => {{
  updateSelectedManualFromEditor();
  saveManualSegments(false);
}});
document.getElementById('deleteManualSegmentBtn').addEventListener('click', () => {{
  if (selectedManualIndex < 0) return;
  manualSegments.splice(selectedManualIndex, 1);
  selectedManualIndex = Math.min(selectedManualIndex, manualSegments.length - 1);
  renderManualLayer();
  saveManualSegments(true);
}});
[manualStartInput, manualEndInput, manualGroundTruthLabelSelect, manualConfidenceSelect, manualSegmentNote].forEach(el => {{
  el.addEventListener('change', updateSelectedManualFromEditor);
}});
quickManualLabelSelect.addEventListener('change', () => {{
  if (quickLabelStartS !== null) quickLabelStatus.textContent = formatQuickLabelStart();
}});
function scrubToEvent(evt, autoplay=true) {{
  const t = timeFromTimelineEvent(evt, manualScrubTimeline);
  seekBoth(t, autoplay);
  quickLabelStatus.textContent = `Video moved to ${{fmt(t,1)}}s. Drag the lower manual row to label this area.`;
}}

manualScrubTimeline.addEventListener('pointerdown', (evt) => {{
  evt.preventDefault();
  scrubDragging = true;
  manualScrubTimeline.setPointerCapture?.(evt.pointerId);
  scrubToEvent(evt, true);
}});
clinicalTimeline.addEventListener('pointerdown', (evt) => {{
  if (evt.target.closest('.clinical-timeline-seg')) return;
  evt.preventDefault();
  const t = timeFromTimelineEvent(evt, clinicalTimeline);
  clinicalDragRange = {{start_s: t, end_s: t}};
  clinicalTimeline.setPointerCapture?.(evt.pointerId);
  renderClinicalTimeline();
}});
manualTimeline.addEventListener('pointerdown', (evt) => {{
  if (evt.target.closest('.manual-timeline-seg')) return;
  evt.preventDefault();
  const t = timeFromManualTimelineEvent(evt);
  manualDragRange = {{start_s: t, end_s: t}};
  manualTimeline.setPointerCapture?.(evt.pointerId);
  renderManualTimeline();
}});
window.addEventListener('pointermove', (evt) => {{
  if (scrubDragging) {{
    evt.preventDefault();
    scrubToEvent(evt, false);
    return;
  }}
  if (clinicalDragRange) {{
    evt.preventDefault();
    clinicalDragRange.end_s = timeFromTimelineEvent(evt, clinicalTimeline);
    renderClinicalTimeline();
    return;
  }}
  if (!manualDragRange) return;
  evt.preventDefault();
  manualDragRange.end_s = timeFromManualTimelineEvent(evt);
  renderManualTimeline();
}});
window.addEventListener('pointerup', (evt) => {{
  if (scrubDragging) {{
    evt.preventDefault();
    scrubDragging = false;
    scrubToEvent(evt, true);
    return;
  }}
  if (clinicalDragRange) {{
    evt.preventDefault();
    clinicalDragRange.end_s = timeFromTimelineEvent(evt, clinicalTimeline);
    const start = Math.min(clinicalDragRange.start_s, clinicalDragRange.end_s);
    const end = Math.max(clinicalDragRange.start_s, clinicalDragRange.end_s);
    clinicalDragRange = null;
    if (end - start < 0.25) {{
      renderClinicalTimeline();
      seekBoth(start, true);
      clinicalStatus.textContent = `Jumped to ${{fmt(start,1)}}s. Drag longer to create a freezing/gait marker.`;
      return;
    }}
    const created = createFreezingSegment(start, end, clinicalMarkerSelect.value, 'Created by dragging on the freezing/gait marker row.', false);
    if (created) clinicalStatus.textContent = `Saved ${{markerDisplay(created.marker_label)}} marker ${{fmt(created.start_s,1)}}-${{fmt(created.end_s,1)}}s.`;
    return;
  }}
  if (!manualDragRange) return;
  evt.preventDefault();
  manualDragRange.end_s = timeFromManualTimelineEvent(evt);
  const start = Math.min(manualDragRange.start_s, manualDragRange.end_s);
  const end = Math.max(manualDragRange.start_s, manualDragRange.end_s);
  manualDragRange = null;
  if (end - start < 0.35) {{
    renderManualTimeline();
    seekBoth(start, true);
    quickLabelStatus.textContent = `Jumped to ${{fmt(start,1)}}s. Drag longer to create a manual label.`;
    return;
  }}
  const label = quickManualLabelSelect.value;
  const snappedStart = snapStartForContinuousLabel(start, end);
  const created = createManualSegment(snappedStart, end, label, 'Created by dragging on the manual timeline.', false);
  if (created) quickLabelStatus.textContent = `Saved ${{label}} from ${{fmt(created.start_s,1)}}s to ${{fmt(created.end_s,1)}}s.`;
}});
segmentSelect.addEventListener('change', () => selectSegment(Number(segmentSelect.value), true));
document.getElementById('prevBtn').addEventListener('click', () => selectSegment(selectedIndex - 1, true));
document.getElementById('nextBtn').addEventListener('click', () => selectSegment(selectedIndex + 1, true));
document.getElementById('back2Btn').addEventListener('click', () => seekBoth((downwardVideo.currentTime || 0) - 2, true));
document.getElementById('forward2Btn').addEventListener('click', () => seekBoth((downwardVideo.currentTime || 0) + 2, true));
document.getElementById('playContextBtn').addEventListener('click', () => {{
  const seg = currentSegments()[selectedIndex];
  if (!seg) return;
  stopAt = seg.context_end_s;
  seekBoth(seg.context_start_s, true);
}});
document.getElementById('playRangeBtn').addEventListener('click', () => {{
  const seg = currentSegments()[selectedIndex];
  if (!seg) return;
  stopAt = seg.end_s;
  seekBoth(seg.start_s, true);
}});
timeline.addEventListener('click', (evt) => {{
  if (evt.target.classList.contains('timeline-seg')) return;
  const rect = timeline.getBoundingClientRect();
  const pct = (evt.clientX - rect.left) / rect.width;
  seekBoth(pct * DASHBOARD_DATA.duration_s, true);
}});
signalPlot.addEventListener('click', (evt) => {{
  const seg = currentSegments()[selectedIndex];
  if (!seg) return;
  const [timeMin, timeMax] = plotBoundsForSegment(seg);
  const rect = signalPlot.getBoundingClientRect();
  const xView = (evt.clientX - rect.left) / rect.width * 900;
  const frac = Math.max(0, Math.min(1, (xView - 54) / (842 - 54)));
  seekBoth(timeMin + frac * (timeMax - timeMin), true);
}});
signalPlot.addEventListener('wheel', (evt) => {{
  evt.preventDefault();
  setPlotZoom(plotZoomScale * (evt.deltaY < 0 ? 0.82 : 1.22));
}}, {{ passive: false }});
xyPlot.addEventListener('click', (evt) => {{
  const seg = currentSegments()[selectedIndex];
  if (!seg) return;
  const [timeMin, timeMax] = plotBoundsForSegment(seg);
  const rows = currentRecords().filter(r => r.time_s >= timeMin && r.time_s <= timeMax);
  if (!rows.length) return;
  const xs = rows.map(r => r.x_smooth_m);
  const ys = rows.map(r => r.y_smooth_m);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const padX = Math.max((xMax - xMin) * .12, .2);
  const padY = Math.max((yMax - yMin) * .12, .2);
  const minX = xMin - padX, maxX = xMax + padX, minY = yMin - padY, maxY = yMax + padY;
  const rect = xyPlot.getBoundingClientRect();
  const xView = (evt.clientX - rect.left) / rect.width * 640;
  const yView = (evt.clientY - rect.top) / rect.height * 460;
  let best = rows[0];
  let bestDist = Infinity;
  for (const row of rows) {{
    const px = scale(row.x_smooth_m, minX, maxX, 46, 594);
    const py = scale(row.y_smooth_m, minY, maxY, 398, 44);
    const dist = (px - xView) * (px - xView) + (py - yView) * (py - yView);
    if (dist < bestDist) {{
      bestDist = dist;
      best = row;
    }}
  }}
  seekBoth(best.time_s, true);
}});
document.addEventListener('keydown', (evt) => {{
  const tag = String(evt.target?.tagName || '').toUpperCase();
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag)) return;
  if (evt.code === 'Space') {{
    evt.preventDefault();
    downwardVideo.paused ? playBoth() : pauseBoth();
    return;
  }}
  if (evt.key === '[') {{
    evt.preventDefault();
    startQuickLabelAtCurrentTime();
    return;
  }}
  if (evt.key === ']') {{
    evt.preventDefault();
    finishQuickLabelAtCurrentTime();
    return;
  }}
  const labelByKey = {{
    '1': 'sitting',
    '2': 'standing',
    '3': 'walking',
    '4': 'upstairs',
    '5': 'downstairs',
  }};
  if (labelByKey[evt.key]) {{
    evt.preventDefault();
    quickManualLabelSelect.value = labelByKey[evt.key];
    if (quickLabelStartS !== null) {{
      quickLabelStatus.textContent = formatQuickLabelStart();
    }} else {{
      quickLabelStatus.textContent = `Next manual label will be ${{labelByKey[evt.key]}}.`;
    }}
  }}
}});
renderSelectors();
renderAll();
selectSegment(0, false);
loadManualNotes();
loadManualSegments();
loadFreezingSegments();
</script>
</body>
</html>
"""


def build_parkinson_review(trial_id: str, *, session_manifest: dict[str, Any] | None = None) -> Path:
    run_dir = RUNS_DIR / trial_id
    session_manifest_path = run_dir / "session_dashboard_manifest.json"
    trial_manifest_path = run_dir / "trial_manifest.json"
    if session_manifest is None and not session_manifest_path.exists():
        raise FileNotFoundError(f"Missing session dashboard manifest: {session_manifest_path}")
    if not trial_manifest_path.exists():
        raise FileNotFoundError(f"Missing trial manifest: {trial_manifest_path}")

    session_manifest = session_manifest if session_manifest is not None else load_json(session_manifest_path)
    trial_manifest = load_json(trial_manifest_path)
    trajectory_csv = get_nested_path(trial_manifest, "mps_outputs", "closed_loop_trajectory_csv")
    if trajectory_csv is None or not trajectory_csv.exists():
        raise FileNotFoundError("Could not find MPS closed_loop_trajectory.csv from trial_manifest.json")

    duration_s = safe_float(session_manifest.get("session_duration_s"))
    recording_start_epoch_s = safe_float(session_manifest.get("recording_start_epoch_s"))
    if duration_s <= 0 or recording_start_epoch_s <= 0:
        raise ValueError("Session manifest must include session_duration_s and recording_start_epoch_s")

    bin_s = 0.5
    base = resample_trajectory(trajectory_csv, recording_start_epoch_s, duration_s, bin_s,
                               session_manifest.get("session_start_tracking_us"))
    out_dir = run_dir / "parkinson_motion_review"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "trial_id": trial_id,
        "duration_s": round(duration_s, 3),
        "bin_s": bin_s,
        "defaultSmoothing": "high",
        "media": {
            "downward": str(session_manifest.get("downward_session_video_uri", "downward_master_session/downward_session_video.mp4")),
            "aria": str(session_manifest.get("aria_session_video_uri", "downward_master_session/aria_session_video.mp4")),
        },
        "smoothing": {},
    }

    for cfg in SMOOTH_CONFIGS:
        features = build_features(base, cfg, bin_s)
        classified, thresholds = classify_features(features, cfg)
        segments = segment_features(classified, cfg, duration_s)
        classified.to_csv(out_dir / f"{trial_id}_parkinson_motion_{cfg.key}.csv", index=False)
        (out_dir / f"{trial_id}_parkinson_segments_{cfg.key}.json").write_text(json.dumps(segments, indent=2))
        payload["smoothing"][cfg.key] = {
            "label": cfg.label,
            "thresholds": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in thresholds.items()},
            "segments": segments,
            "records": to_records_for_browser(classified),
        }

    (out_dir / f"{trial_id}_parkinson_review_payload.json").write_text(json.dumps(payload, indent=2))
    return publish_parkinson_review(payload)


def publish_parkinson_review(payload: dict[str, Any]) -> Path:
    """Render saved estimates without recomputing classifications or modifying labels."""
    trial_id = payload["trial_id"]
    run_dir = RUNS_DIR / trial_id
    html_path = run_dir / "pd_insighter_parkinson_review.html"
    document = build_html(payload)
    if trial_id != "Ntrial10":
        # Both labeling pages use one durable action-label store for new trials.
        document = document.replace("/api/parkinson-manual-segments", "/api/session-manual-segments")
    segments = continuous_segments_from_payload(payload)
    diagnostics = run_dir / "raw_parameter_diagnostics"
    diagnostics.mkdir(exist_ok=True)
    (diagnostics / "continuous_motion_segments.json").write_text(json.dumps({
        "version": "parkinson_high_v1", "method": "Parkinson-specific adaptive SLAM criteria, high smoothing",
        "segments": segments, "accuracy": {},
        "limitations": ["Simulated motions; not clinically validated.", "Accuracy requires independent manual labels."]
    }, indent=2))
    html_path.write_text(document)
    return html_path


def continuous_segments_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    segments = []
    for row in payload["smoothing"]["high"]["segments"]:
        if row["end_s"] <= row["start_s"]:
            continue
        label = row["base_motion"]
        if segments and segments[-1]["label"] == label and abs(segments[-1]["end_s"] - row["start_s"]) < 1e-6:
            segments[-1]["end_s"] = row["end_s"]
        else:
            segments.append(dict(label=label, start_s=row["start_s"], end_s=row["end_s"],
                                 source="parkinson_high_v1", start_boundary_source="0.5s SLAM evidence",
                                 start_boundary_confidence="unvalidated"))
    for row in segments:
        row["duration_s"] = row["end_s"] - row["start_s"]
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Ntrial10 Parkinson motion review page.")
    parser.add_argument("--trial", default="Ntrial10", help="Trial id to build. Intended for Ntrial10.")
    args = parser.parse_args()

    html_path = build_parkinson_review(args.trial)
    print(f"Built Parkinson motion review: {html_path}")
    print(f"Open: http://127.0.0.1:8782/Qwen2.5-VL-fork/runs/{args.trial}/pd_insighter_parkinson_review.html")


if __name__ == "__main__":
    main()
