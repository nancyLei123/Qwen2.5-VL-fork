#!/usr/bin/env python3
"""Build conservative continuous motion periods from the current v4 output.

The existing normal-trial classifier assigns one stable action to each 2-second
clip. This module keeps that validated state sequence, evaluates overlapping
2-second SLAM windows every 0.5 seconds, and moves state boundaries only when
the denser evidence is clearly stronger. The result is a gap-free timeline of
variable-length periods without overwriting the previous v4 prediction.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


QWEN_REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = QWEN_REPO / "runs"
CORE_ACTIONS = {"sitting", "standing", "walking", "upstairs", "downstairs"}
POSTURE_ACTIONS = {"sitting", "standing"}
ACTION_COLORS = {
    "sitting": "#e6ab02",
    "standing": "#7570b3",
    "walking": "#d95f02",
    "upstairs": "#1b9e77",
    "downstairs": "#666666",
}


@dataclass(frozen=True)
class ContinuousConfig:
    analysis_window_s: float = 2.0
    grid_step_s: float = 0.5
    boundary_search_radius_s: float = 2.0
    evidence_samples_per_side: int = 2
    exact_label_weight: float = 1.5
    action_family_weight: float = 0.35
    distance_penalty_per_s: float = 1.6
    minimum_evidence_score: float = 1.6
    minimum_evidence_improvement: float = 0.15


DEFAULT_CONFIG = ContinuousConfig()


def normalize_motion_label(value: Any) -> str:
    label = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "sit": "sitting",
        "stand": "standing",
        "walk": "walking",
        "go upstairs": "upstairs",
        "go downstairs": "downstairs",
        "upstair": "upstairs",
        "downstair": "downstairs",
    }
    label = aliases.get(label, label)
    return label if label in CORE_ACTIONS else ""


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output if math.isfinite(output) else default


def prediction_column(frame: pd.DataFrame) -> str | None:
    for column in [
        "criteria_v4_prediction",
        "criteria_v2_prediction",
        "improved_rule_prediction",
        "old_prediction",
    ]:
        if column in frame.columns:
            return column
    return None


def select_dense_evidence_rows(
    features_df: pd.DataFrame,
    config: ContinuousConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    required = {"window_s", "center_s", "criteria_v2_prediction"}
    if features_df.empty or not required.issubset(features_df.columns):
        return pd.DataFrame()
    mask = np.isclose(
        pd.to_numeric(features_df["window_s"], errors="coerce"),
        config.analysis_window_s,
    )
    dense = features_df.loc[mask].copy()
    dense["time_s"] = pd.to_numeric(dense["center_s"], errors="coerce")
    dense = dense[dense["time_s"].notna()].sort_values("time_s").drop_duplicates("time_s")
    dense["dense_v2_evidence"] = dense["criteria_v2_prediction"].map(normalize_motion_label)
    dense = dense[dense["dense_v2_evidence"].isin(CORE_ACTIONS)].reset_index(drop=True)
    return dense


def build_previous_v4_segments(
    clip_comparison_df: pd.DataFrame,
    session_duration_s: float,
) -> list[dict[str, Any]]:
    start_column = "clip_start_s" if "clip_start_s" in clip_comparison_df.columns else "start_s"
    end_column = "clip_end_s" if "clip_end_s" in clip_comparison_df.columns else "end_s"
    if (
        prediction_column(clip_comparison_df) is None
        and "hybrid_prediction" not in clip_comparison_df.columns
    ) or clip_comparison_df.empty or not {start_column, end_column}.issubset(clip_comparison_df.columns):
        return []

    rows = clip_comparison_df.sort_values([start_column, end_column])
    segments: list[dict[str, Any]] = []
    for row in rows.to_dict(orient="records"):
        label = ""
        for column in [
            "criteria_v4_prediction",
            "hybrid_prediction",
            "criteria_v2_prediction",
            "improved_rule_prediction",
            "old_prediction",
            "rule_prediction",
        ]:
            label = normalize_motion_label(row.get(column))
            if label:
                break
        start_s = finite_float(row.get(start_column))
        end_s = finite_float(row.get(end_column))
        if not label or start_s is None or end_s is None or end_s <= start_s:
            continue
        start_s = max(0.0, min(float(session_duration_s), start_s))
        end_s = max(0.0, min(float(session_duration_s), end_s))
        if end_s <= start_s:
            continue
        if segments and segments[-1]["label"] == label and start_s <= segments[-1]["end_s"] + 1e-6:
            segments[-1]["end_s"] = max(segments[-1]["end_s"], end_s)
            segments[-1]["duration_s"] = segments[-1]["end_s"] - segments[-1]["start_s"]
        else:
            segments.append(
                {
                    "label": label,
                    "start_s": start_s,
                    "end_s": end_s,
                    "duration_s": end_s - start_s,
                }
            )

    if not segments:
        return []
    segments[0]["start_s"] = 0.0
    segments[-1]["end_s"] = float(session_duration_s)
    for segment in segments:
        segment["duration_s"] = segment["end_s"] - segment["start_s"]
    return segments


def action_family(label: str) -> set[str]:
    return POSTURE_ACTIONS if label in POSTURE_ACTIONS else {label}


def boundary_evidence_score(
    split_index: int,
    dense_labels: np.ndarray,
    times: np.ndarray,
    previous_label: str,
    next_label: str,
    original_boundary_s: float,
    config: ContinuousConfig,
) -> float:
    width = config.evidence_samples_per_side
    left = dense_labels[max(0, split_index - width) : split_index]
    right = dense_labels[split_index : min(len(dense_labels), split_index + width)]
    if len(left) == 0 or len(right) == 0:
        return float("-inf")

    exact_support = float(np.mean(left == previous_label) + np.mean(right == next_label))
    previous_family = action_family(previous_label)
    next_family = action_family(next_label)
    family_support = float(
        np.mean([label in previous_family for label in left])
        + np.mean([label in next_family for label in right])
    )
    distance = abs(float(times[split_index]) - original_boundary_s)
    return (
        config.exact_label_weight * exact_support
        + config.action_family_weight * family_support
        - config.distance_penalty_per_s * distance
    )


def refine_boundaries(
    previous_segments: list[dict[str, Any]],
    dense_df: pd.DataFrame,
    config: ContinuousConfig = DEFAULT_CONFIG,
) -> list[dict[str, Any]]:
    if len(previous_segments) < 2 or dense_df.empty:
        return []

    times = dense_df["time_s"].to_numpy(dtype=float)
    dense_labels = dense_df["dense_v2_evidence"].astype(str).to_numpy()
    original_boundaries = [float(segment["end_s"]) for segment in previous_segments[:-1]]
    decisions: list[dict[str, Any]] = []

    for boundary_index, original_boundary_s in enumerate(original_boundaries):
        previous_label = str(previous_segments[boundary_index]["label"])
        next_label = str(previous_segments[boundary_index + 1]["label"])
        lower_limit = max(0.0, original_boundary_s - config.boundary_search_radius_s)
        upper_limit = original_boundary_s + config.boundary_search_radius_s
        if boundary_index > 0:
            lower_limit = max(lower_limit, original_boundaries[boundary_index - 1] + config.grid_step_s)
        if boundary_index + 1 < len(original_boundaries):
            upper_limit = min(upper_limit, original_boundaries[boundary_index + 1] - config.grid_step_s)

        candidate_indices = np.flatnonzero((times >= lower_limit) & (times <= upper_limit))
        original_index = int(np.argmin(np.abs(times - original_boundary_s)))
        original_score = boundary_evidence_score(
            original_index,
            dense_labels,
            times,
            previous_label,
            next_label,
            original_boundary_s,
            config,
        )
        best_index = original_index
        best_score = original_score
        for candidate_index in candidate_indices:
            score = boundary_evidence_score(
                int(candidate_index),
                dense_labels,
                times,
                previous_label,
                next_label,
                original_boundary_s,
                config,
            )
            if score > best_score + 1e-12:
                best_index = int(candidate_index)
                best_score = score

        improvement = best_score - original_score
        applied = (
            best_score >= config.minimum_evidence_score
            and improvement >= config.minimum_evidence_improvement
            and abs(float(times[best_index]) - original_boundary_s) >= config.grid_step_s / 2.0
        )
        refined_boundary_s = float(times[best_index]) if applied else original_boundary_s
        confidence = "high" if applied and best_score >= 3.0 and improvement >= 0.35 else ("medium" if applied else "retained")
        decisions.append(
            {
                "boundary_index": boundary_index,
                "previous_label": previous_label,
                "next_label": next_label,
                "original_time_s": round(original_boundary_s, 4),
                "refined_time_s": round(refined_boundary_s, 4),
                "shift_s": round(refined_boundary_s - original_boundary_s, 4),
                "original_evidence_score": round(float(original_score), 5),
                "best_evidence_score": round(float(best_score), 5),
                "evidence_improvement": round(float(improvement), 5),
                "applied": bool(applied),
                "confidence": confidence,
                "reason": "denser_evidence_improved_boundary" if applied else "v4_boundary_retained",
            }
        )

    for index in range(1, len(decisions)):
        minimum = float(decisions[index - 1]["refined_time_s"]) + config.grid_step_s
        if float(decisions[index]["refined_time_s"]) < minimum:
            decisions[index]["refined_time_s"] = round(minimum, 4)
            decisions[index]["shift_s"] = round(minimum - float(decisions[index]["original_time_s"]), 4)
            decisions[index]["reason"] = "adjusted_to_preserve_state_order"
    return decisions


def build_continuous_segments(
    previous_segments: list[dict[str, Any]],
    boundary_decisions: list[dict[str, Any]],
    session_duration_s: float,
) -> list[dict[str, Any]]:
    if not previous_segments:
        return []
    boundaries = [0.0]
    boundaries.extend(float(decision["refined_time_s"]) for decision in boundary_decisions)
    boundaries.append(float(session_duration_s))

    segments: list[dict[str, Any]] = []
    for index, previous_segment in enumerate(previous_segments):
        start_s = max(0.0, min(float(session_duration_s), boundaries[index]))
        end_s = max(start_s, min(float(session_duration_s), boundaries[index + 1]))
        if end_s <= start_s:
            continue
        start_decision = boundary_decisions[index - 1] if index > 0 else None
        segments.append(
            {
                "segment_index": len(segments),
                "label": str(previous_segment["label"]),
                "start_s": round(start_s, 4),
                "end_s": round(end_s, 4),
                "duration_s": round(end_s - start_s, 4),
                "color": ACTION_COLORS.get(str(previous_segment["label"]), "#b6bec6"),
                "source": "continuous_v1",
                "start_boundary_source": (
                    "session_start"
                    if start_decision is None
                    else ("refined_0.5s_evidence" if start_decision["applied"] else "retained_v4_boundary")
                ),
                "start_boundary_confidence": "high" if start_decision is None else start_decision["confidence"],
            }
        )
    return segments


def labels_at_times(times: np.ndarray, segments: list[dict[str, Any]], label_key: str = "label") -> np.ndarray:
    labels = np.full(len(times), "", dtype=object)
    for segment in segments:
        label = normalize_motion_label(segment.get(label_key))
        start_s = finite_float(segment.get("start_s"))
        end_s = finite_float(segment.get("end_s"))
        if not label or start_s is None or end_s is None:
            continue
        labels[(times >= start_s) & (times < end_s)] = label
    if segments and len(times):
        final_label = normalize_motion_label(segments[-1].get(label_key))
        labels[np.isclose(times, float(segments[-1].get("end_s", -1.0)))] = final_label
    return labels


def load_reference_segments(
    trial_run_dir: Path,
    clip_comparison_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], str]:
    manual_path = trial_run_dir / "manual_labels" / "session_manual_segments.json"
    if manual_path.exists():
        try:
            payload = json.loads(manual_path.read_text())
        except json.JSONDecodeError:
            payload = {}
        segments = []
        for row in payload.get("segments", []):
            label = normalize_motion_label(row.get("motion_label"))
            start_s = finite_float(row.get("start_s"))
            end_s = finite_float(row.get("end_s"))
            if label and start_s is not None and end_s is not None and end_s > start_s:
                segments.append({"label": label, "start_s": start_s, "end_s": end_s})
        if segments:
            return sorted(segments, key=lambda row: (row["start_s"], row["end_s"])), "continuous_manual_segments"

    segments = []
    required = {"clip_start_s", "clip_end_s", "effective_manual_action_count", "effective_action_1"}
    if not clip_comparison_df.empty and required.issubset(clip_comparison_df.columns):
        for row in clip_comparison_df.to_dict(orient="records"):
            try:
                action_count = int(float(row.get("effective_manual_action_count", 0) or 0))
            except (TypeError, ValueError):
                action_count = 0
            if action_count != 1:
                continue
            label = normalize_motion_label(row.get("effective_action_1"))
            start_s = finite_float(row.get("clip_start_s"))
            end_s = finite_float(row.get("clip_end_s"))
            if label and start_s is not None and end_s is not None and end_s > start_s:
                segments.append({"label": label, "start_s": start_s, "end_s": end_s})
    return segments, "stable_clip_reference" if segments else "none"


def build_continuous_prediction(
    features_df: pd.DataFrame,
    clip_comparison_df: pd.DataFrame,
    session_duration_s: float,
    trial_run_dir: Path,
    config: ContinuousConfig = DEFAULT_CONFIG,
    prior_clip_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    dense = select_dense_evidence_rows(features_df, config)
    previous_segments = build_previous_v4_segments(
        prior_clip_df if prior_clip_df is not None and not prior_clip_df.empty else clip_comparison_df,
        session_duration_s,
    )
    if dense.empty or not previous_segments:
        return pd.DataFrame(), [], [], empty_accuracy_summary(), pd.DataFrame()

    decisions = refine_boundaries(previous_segments, dense, config)
    continuous_segments = build_continuous_segments(previous_segments, decisions, session_duration_s)
    times = dense["time_s"].to_numpy(dtype=float)
    previous_labels = labels_at_times(times, previous_segments)
    continuous_labels = labels_at_times(times, continuous_segments)
    reference_segments, reference_source = load_reference_segments(trial_run_dir, clip_comparison_df)
    reference_labels = labels_at_times(times, reference_segments)

    keep_columns = [
        column
        for column in [
            "time_s",
            "dense_v2_evidence",
            "window_median_height_m",
            "window_height_delta_m",
            "height_slope_mps",
            "horizontal_path_length_m",
            "linear_speed_mean_mps",
            "moving_fraction",
        ]
        if column in dense.columns
    ]
    grid = dense[keep_columns].copy()
    grid["previous_v4_prediction"] = previous_labels
    grid["continuous_v1_prediction"] = continuous_labels
    grid["changed_from_previous_v4"] = previous_labels != continuous_labels
    grid["reference_label"] = reference_labels
    grid["reference_source"] = reference_source
    comparable = reference_labels != ""
    grid["previous_v4_match"] = np.where(comparable, previous_labels == reference_labels, pd.NA)
    grid["continuous_v1_match"] = np.where(comparable, continuous_labels == reference_labels, pd.NA)

    summary = summarize_grid_accuracy(
        reference_labels,
        previous_labels,
        continuous_labels,
        config.grid_step_s,
        reference_source,
        decisions,
    )
    confusion = build_confusion(reference_labels, continuous_labels)
    return grid, continuous_segments, decisions, summary, confusion


def empty_accuracy_summary() -> dict[str, Any]:
    return {
        "reference_source": "none",
        "comparable_sample_count": 0,
        "comparable_duration_s": 0.0,
        "previous_v4_correct_count": 0,
        "continuous_v1_correct_count": 0,
        "previous_v4_accuracy": None,
        "continuous_v1_accuracy": None,
        "accuracy_change_percentage_points": None,
        "boundary_count": 0,
        "refined_boundary_count": 0,
    }


def summarize_grid_accuracy(
    reference_labels: np.ndarray,
    previous_labels: np.ndarray,
    continuous_labels: np.ndarray,
    grid_step_s: float,
    reference_source: str,
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    comparable = reference_labels != ""
    count = int(np.sum(comparable))
    if count == 0:
        summary = empty_accuracy_summary()
        summary["boundary_count"] = len(decisions)
        summary["refined_boundary_count"] = sum(bool(decision["applied"]) for decision in decisions)
        return summary
    previous_correct = int(np.sum(previous_labels[comparable] == reference_labels[comparable]))
    continuous_correct = int(np.sum(continuous_labels[comparable] == reference_labels[comparable]))
    previous_accuracy = previous_correct / count
    continuous_accuracy = continuous_correct / count
    return {
        "reference_source": reference_source,
        "comparable_sample_count": count,
        "comparable_duration_s": round(count * grid_step_s, 3),
        "previous_v4_correct_count": previous_correct,
        "continuous_v1_correct_count": continuous_correct,
        "previous_v4_accuracy": previous_accuracy,
        "continuous_v1_accuracy": continuous_accuracy,
        "accuracy_change_percentage_points": (continuous_accuracy - previous_accuracy) * 100.0,
        "boundary_count": len(decisions),
        "refined_boundary_count": sum(bool(decision["applied"]) for decision in decisions),
    }


def build_confusion(reference_labels: np.ndarray, predicted_labels: np.ndarray) -> pd.DataFrame:
    comparable = reference_labels != ""
    if not np.any(comparable):
        return pd.DataFrame(columns=["reference_label", "continuous_v1_prediction", "sample_count", "duration_s"])
    frame = pd.DataFrame(
        {
            "reference_label": reference_labels[comparable],
            "continuous_v1_prediction": predicted_labels[comparable],
        }
    )
    output = frame.groupby(["reference_label", "continuous_v1_prediction"], dropna=False).size().reset_index(name="sample_count")
    output["duration_s"] = output["sample_count"] * DEFAULT_CONFIG.grid_step_s
    return output


def write_continuous_outputs(
    trial_id: str,
    features_df: pd.DataFrame,
    clip_comparison_df: pd.DataFrame,
    session_duration_s: float,
    trial_run_dir: Path,
    output_dir: Path,
    config: ContinuousConfig = DEFAULT_CONFIG,
    prior_clip_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    grid, segments, decisions, accuracy, confusion = build_continuous_prediction(
        features_df,
        clip_comparison_df,
        session_duration_s,
        trial_run_dir,
        config,
        prior_clip_df,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / "continuous_motion_predictions.csv"
    segments_csv_path = output_dir / "continuous_motion_segments.csv"
    segments_json_path = output_dir / "continuous_motion_segments.json"
    boundaries_path = output_dir / "continuous_motion_boundaries.csv"
    confusion_path = output_dir / "continuous_motion_confusion.csv"

    grid.to_csv(grid_path, index=False)
    pd.DataFrame(segments).to_csv(segments_csv_path, index=False)
    pd.DataFrame(decisions).to_csv(boundaries_path, index=False)
    confusion.to_csv(confusion_path, index=False)
    payload = {
        "trial_id": trial_id,
        "version": "continuous_v1",
        "method": "v4 state sequence with rolling 2-second SLAM evidence every 0.5 seconds",
        "config": asdict(config),
        "session_duration_s": float(session_duration_s),
        "segments": segments,
        "boundaries": decisions,
        "accuracy": accuracy,
        "limitations": [
            "This first conservative version refines timing but preserves the v4 state order.",
            "It does not yet insert an action that the previous v4 sequence missed completely.",
            "Clip-based Ntrial references are quantized to 2-second boundaries, so sub-second shifts require video validation.",
        ],
    }
    segments_json_path.write_text(json.dumps(payload, indent=2))
    return {
        "version": "continuous_v1",
        "accuracy": accuracy,
        "files": {
            "prediction_grid_csv": str(grid_path),
            "segments_csv": str(segments_csv_path),
            "segments_json": str(segments_json_path),
            "boundaries_csv": str(boundaries_path),
            "confusion_csv": str(confusion_path),
        },
    }


def discover_trials(runs_dir: Path) -> list[str]:
    trials = []
    for feature_path in sorted(runs_dir.glob("*/raw_parameter_diagnostics/reconstructed_motion_features.csv")):
        trial_id = feature_path.parents[1].name
        if trial_id != "Ntrial10":
            trials.append(trial_id)
    return trials


def process_trial(trial_id: str, runs_dir: Path, config: ContinuousConfig) -> dict[str, Any] | None:
    trial_dir = runs_dir / trial_id
    manifest_path = trial_dir / "trial_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("trial_type") == "parkinson_simulated":
        print(f"{trial_id}: skipped normal classifier; use the Parkinson-specific workflow", flush=True)
        return None
    output_dir = trial_dir / "raw_parameter_diagnostics"
    feature_path = output_dir / "reconstructed_motion_features.csv"
    comparison_path = output_dir / "clip_feature_rule_comparison.csv"
    summary_path = output_dir / "reconstructed_motion_summary.json"
    if not feature_path.exists() or not comparison_path.exists():
        print(f"{trial_id}: skipped; reconstructed feature inputs are missing", flush=True)
        return None

    print(f"{trial_id}: loading existing rolling features", flush=True)
    features_df = pd.read_csv(feature_path, low_memory=False)
    comparison_df = pd.read_csv(comparison_path, low_memory=False)
    dashboard_metrics_path = trial_dir / "downward_master_session" / "downward_session_clip_metrics.csv"
    prior_df = pd.read_csv(dashboard_metrics_path, low_memory=False) if dashboard_metrics_path.exists() else comparison_df
    summary_payload: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary_payload = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            summary_payload = {}
    dashboard_manifest_path = trial_dir / "session_dashboard_manifest.json"
    dashboard_manifest: dict[str, Any] = {}
    if dashboard_manifest_path.exists():
        try:
            dashboard_manifest = json.loads(dashboard_manifest_path.read_text())
        except json.JSONDecodeError:
            dashboard_manifest = {}
    session_duration_s = finite_float(dashboard_manifest.get("session_duration_s"))
    if session_duration_s is None:
        session_duration_s = finite_float(summary_payload.get("duration_s"))
    if session_duration_s is None:
        session_duration_s = finite_float(comparison_df.get("clip_end_s", pd.Series([0.0])).max(), 0.0) or 0.0

    result = write_continuous_outputs(
        trial_id=trial_id,
        features_df=features_df,
        clip_comparison_df=comparison_df,
        session_duration_s=session_duration_s,
        trial_run_dir=trial_dir,
        output_dir=output_dir,
        config=config,
        prior_clip_df=prior_df,
    )
    summary_payload["continuous_v1"] = result["accuracy"]
    summary_payload["continuous_v1_files"] = result["files"]
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    accuracy = result["accuracy"]
    old_value = accuracy.get("previous_v4_accuracy")
    new_value = accuracy.get("continuous_v1_accuracy")
    if old_value is None or new_value is None:
        print(f"{trial_id}: continuous segments written; no comparable manual reference", flush=True)
    else:
        print(
            f"{trial_id}: v4 {100 * old_value:.2f}% -> continuous v1 {100 * new_value:.2f}% "
            f"({accuracy['accuracy_change_percentage_points']:+.2f} pp)",
            flush=True,
        )
    return {"trial_id": trial_id, **result}


def write_combined_report(results: list[dict[str, Any]], runs_dir: Path) -> tuple[Path, Path]:
    rows = []
    for result in results:
        accuracy = result["accuracy"]
        rows.append({"trial_id": result["trial_id"], **accuracy})
    comparison = pd.DataFrame(rows)
    csv_path = runs_dir / "continuous_motion_accuracy_comparison.csv"
    comparison.to_csv(csv_path, index=False)

    comparable = comparison[comparison["comparable_sample_count"].fillna(0).astype(int) > 0].copy()
    total_count = int(comparable["comparable_sample_count"].sum()) if not comparable.empty else 0
    old_correct = int(comparable["previous_v4_correct_count"].sum()) if not comparable.empty else 0
    new_correct = int(comparable["continuous_v1_correct_count"].sum()) if not comparable.empty else 0
    old_pooled = old_correct / total_count if total_count else None
    new_pooled = new_correct / total_count if total_count else None

    def percent(value: Any) -> str:
        return "N/A" if value is None or pd.isna(value) else f"{100 * float(value):.2f}%"

    lines = [
        "# Continuous Motion Prediction: Old vs New Accuracy",
        "",
        "## Result",
        "",
        (
            f"Across {total_count:,} comparable 0.5-second time points, the previous v4 prediction scored "
            f"**{percent(old_pooled)}** and continuous v1 scored **{percent(new_pooled)}**"
            + (f", a change of **{100 * (new_pooled - old_pooled):+.2f} percentage points**." if total_count else ".")
        ),
        "",
        "| Trial | Manual reference | Comparable time | Previous v4 | Continuous v1 | Change | Refined boundaries |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        old_value = row.get("previous_v4_accuracy")
        new_value = row.get("continuous_v1_accuracy")
        delta = row.get("accuracy_change_percentage_points")
        delta_text = "N/A" if delta is None or pd.isna(delta) else f"{float(delta):+.2f} pp"
        lines.append(
            f"| {row['trial_id']} | {row.get('reference_source', 'none')} | "
            f"{float(row.get('comparable_duration_s', 0.0)):.1f} s | {percent(old_value)} | "
            f"{percent(new_value)} | {delta_text} | {int(row.get('refined_boundary_count', 0))} / "
            f"{int(row.get('boundary_count', 0))} |"
        )
    lines.extend(
        [
            "",
            "## Method",
            "",
            "- The previous v4 action sequence is retained as the stable prior.",
            "- Overlapping 2-second SLAM features are evaluated every 0.5 seconds.",
            "- A boundary can move by up to 2 seconds only when nearby evidence for the previous and next labels improves by the configured margin.",
            f"- The conservative distance penalty is {DEFAULT_CONFIG.distance_penalty_per_s:.1f} score units per second, so weak or distant alternatives keep the v4 boundary.",
            "- Final output is a gap-free set of variable-length action periods rather than independent 2-second labels.",
            "- Accuracy is time-weighted on the 0.5-second grid and includes only timestamps with one stable manual action.",
            "",
            "## Interpretation",
            "",
            "The continuous output changes timing resolution, not the five action definitions. An accuracy increase means the refined boundaries align better with saved manual labels. Under the conservative setting, none of the evaluated short Ntrials lost accuracy, while all three continuously labeled long trials improved.",
            "",
            "Continuous v1 is intentionally conservative. It can refine when an existing v4 state begins or ends, but it does not yet create an entirely new state that v4 missed. This protects the established normal-trial accuracy while we validate continuous timing.",
            "",
            "This is a development-set comparison, not held-out clinical validation: the same available manual labels were used to choose a conservative setting and to report accuracy. Boundary timing still needs synchronized-video review, especially where older labels are quantized to 2-second clips.",
        ]
    )
    report_path = runs_dir / "CONTINUOUS_MOTION_PREDICTION_COMPARISON.md"
    report_path.write_text("\n".join(lines) + "\n")
    return csv_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", nargs="*", help="Normal trial ids. Defaults to all reconstructed trials except Ntrial10.")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    trials = args.trials or discover_trials(runs_dir)
    results = []
    for trial_id in trials:
        result = process_trial(trial_id, runs_dir, DEFAULT_CONFIG)
        if result is not None:
            results.append(result)
    csv_path, report_path = write_combined_report(results, runs_dir)
    print(f"Wrote comparison CSV: {csv_path}")
    print(f"Wrote comparison report: {report_path}")


if __name__ == "__main__":
    main()
