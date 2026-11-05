#!/usr/bin/env python3
"""Build full-session clinician-review dashboards for current normal trials."""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from process_aria_trial import (
    ClipWindow,
    apply_hybrid_predictions,
    build_clip_metrics,
    create_subclip_media_files_ffmpeg,
    discover_audio_companion,
    discover_ffmpeg_executable,
    estimate_alignment_offset_s,
    load_start_time,
    load_video_info,
    linear_speed,
    relative_media_href,
    write_interactive_dashboard,
)
from explore_slam_parameters import load_effective_manual_intervals
from video_time_information import load_or_build_time_information
from pd_insighter_environment_navigation import environment_navigation


QWEN_REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = QWEN_REPO / "runs"
FULL_SESSION_DASHBOARD_NAME = "pd_insighter_session_review.html"
LEGACY_DASHBOARD_NAME = "pd_insighter_dual_view.html"
LEGACY_RULE_DASHBOARD_NAME = "pd_insighter_rule_interactive.html"
CANONICAL_TRIAL_ENTRY_NAME = "index.html"
CLIP_REVIEW_DASHBOARD_NAME = "pd_insighter_clip_review.html"
STRIDE_REVIEW_DASHBOARD_NAME = "pd_insighter_stride_review.html"
TRANSITION_REVIEW_DASHBOARD_NAME = "pd_insighter_transition_review.html"
MANUAL_LABELING_DASHBOARD_NAME = "pd_insighter_manual_labeling.html"
TRANSITION_REVIEW_TRIAL_EXCLUDE: set[str] = set()
ADDITIONAL_NORMAL_TRIALS = {"Half_hour", "Half_hour2", "60minutes_1", "60minutes_2"}
ARIA_TAIL_EXTENDED_TRIALS = {"Half_hour2"}
CLINICAL_TRANSITION_PAIRS = {
    ("sitting", "standing"),
    ("standing", "sitting"),
    ("standing", "walking"),
    ("walking", "standing"),
}

TRANSITION_COLORS = {
    "sit-to-stand": "#d97706",
    "stand-to-sit": "#2563eb",
    "stand-to-walk": "#059669",
    "walk-to-stand": "#dc2626",
    "stair-context": "#0f8f83",
    "needs-review": "#c026d3",
    "other-transition": "#7c3aed",
}

TRANSITION_CLASSIFICATION_COLORS = {
    "clinical": "#16815d",
    "possible": "#d48716",
    "less_possible": "#c026d3",
}

SIGNAL_DISPLAY_TARGET_HZ = 15.0
SIGNAL_DISPLAY_MAX_POINTS = 60_000

# These profiles are intentionally explicit so each transition type can be tuned
# independently as more clinician-reviewed examples become available.
TRANSITION_DETECTION_PROFILES: dict[str, dict[str, Any]] = {
    "sit-to-stand": {
        "signal": "SLAM height",
        "search_padding_s": 2.0,
        "initial_half_window_s": 0.75,
        "start_height_progress": 0.08,
        "end_height_progress": 0.92,
        "minimum_height_change_m": 0.05,
        "safety_margin_s": 0.20,
    },
    "stand-to-sit": {
        "signal": "SLAM height",
        "search_padding_s": 2.0,
        "initial_half_window_s": 0.75,
        "start_height_progress": 0.08,
        "end_height_progress": 0.92,
        "minimum_height_change_m": 0.05,
        "safety_margin_s": 0.20,
    },
    "stand-to-walk": {
        "signal": "0.5 s rolling speed",
        "search_padding_s": 2.0,
        "initial_half_window_s": 0.75,
        "start_speed_fraction": 0.28,
        "end_speed_fraction": 0.72,
        "minimum_start_speed_change_mps": 0.05,
        "minimum_end_speed_change_mps": 0.12,
        "safety_margin_s": 0.20,
    },
    "walk-to-stand": {
        "signal": "0.5 s rolling speed",
        "search_padding_s": 2.0,
        "initial_half_window_s": 0.75,
        "start_speed_fraction": 0.28,
        "end_speed_fraction": 0.28,
        "minimum_start_speed_change_mps": 0.08,
        "minimum_end_speed_change_mps": 0.06,
        "safety_margin_s": 0.20,
    },
    "stair-context": {
        "signal": "SLAM height trend + X/Y path",
        "search_padding_s": 2.5,
        "initial_half_window_s": 1.0,
        "safety_margin_s": 0.30,
    },
    "default": {
        "signal": "SLAM height + speed + X/Y path",
        "search_padding_s": 2.0,
        "initial_half_window_s": 0.75,
        "safety_margin_s": 0.25,
    },
}

COLOR_MAP = {
    "sitting": "#e6ab02",
    "walking": "#d95f02",
    "standing": "#7570b3",
    "go downstairs": "#666666",
    "downstairs": "#666666",
    "go upstairs": "#1b9e77",
    "upstairs": "#1b9e77",
    "missing": "#b6bec6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        nargs="+",
        required=True,
        help="Trial ids to build, e.g. Ntrial1 Ntrial2 Ntrial3.",
    )
    return parser.parse_args()


def infer_clip_seconds(run_dir: Path) -> float:
    clip_metrics_path = run_dir / "clip_metrics.csv"
    if clip_metrics_path.exists():
        clip_df = pd.read_csv(clip_metrics_path)
        if not clip_df.empty:
            return float(clip_df["end_s"].iloc[0] - clip_df["start_s"].iloc[0])
    return 2.0


def normalize_action_label(label: str | None) -> str:
    value = str(label or "").strip().lower()
    if value == "sit":
        return "sitting"
    if value == "stand":
        return "standing"
    if value == "walk":
        return "walking"
    if value == "go upstairs":
        return "upstairs"
    if value == "go downstairs":
        return "downstairs"
    return value


def actions_from_label_display(label: str | None) -> list[str]:
    value = str(label or "").strip().lower().replace("_", "-")
    value = value.replace(" to ", "-to-").replace(" -> ", "-to-")
    if "-to-" in value:
        pieces = value.split("-to-")
    else:
        pieces = (
            value.replace("|", "-")
            .replace("/", "-")
            .replace(",", "-")
            .split("-")
        )
    actions: list[str] = []
    for piece in pieces:
        action = normalize_action_label(piece)
        if action in COLOR_MAP and action not in actions:
            actions.append(action)
    return actions


def color_for_label(label: str | None) -> str:
    actions = actions_from_label_display(label)
    if not actions:
        normalized = normalize_action_label(label)
        return COLOR_MAP.get(normalized, COLOR_MAP["missing"])
    if len(actions) == 1:
        return COLOR_MAP.get(actions[0], COLOR_MAP["missing"])
    pct = 100 / len(actions)
    stops = []
    for index, action in enumerate(actions):
        start = index * pct
        end = (index + 1) * pct
        stops.append(f"{COLOR_MAP.get(action, COLOR_MAP['missing'])} {start:.3f}% {end:.3f}%")
    return f"linear-gradient(90deg, {', '.join(stops)})"


def attach_spoken_action_columns(trial_id: str, clip_df: pd.DataFrame) -> pd.DataFrame:
    spoken_path = RUNS_DIR / trial_id / "spoken_action_labels" / "spoken_action_analysis.json"
    if not spoken_path.exists():
        return clip_df

    spoken_payload = json.loads(spoken_path.read_text())
    spoken_rows = spoken_payload.get("clip_rows", [])
    if not spoken_rows:
        return clip_df

    spoken_df = pd.DataFrame(spoken_rows)
    if "clip_idx" not in spoken_df.columns:
        return clip_df

    join_columns = [
        "clip_idx",
        "spoken_words_detected_in_clip",
        "spoken_word_event_count",
        "spoken_word_times_s",
        "assumed_actions_in_clip",
        "primary_assumed_action",
        "is_transition_clip",
        "state_change_times_in_clip_s",
    ]
    available_columns = [column for column in join_columns if column in spoken_df.columns]
    if "clip_idx" not in available_columns:
        return clip_df

    merged = clip_df.merge(spoken_df[available_columns], on="clip_idx", how="left")
    merged["automatic_spoken_action"] = merged["assumed_actions_in_clip"].fillna("")
    empty_auto_mask = merged["automatic_spoken_action"].astype(str).str.strip().eq("")
    merged.loc[empty_auto_mask, "automatic_spoken_action"] = merged.loc[empty_auto_mask, "primary_assumed_action"].fillna("")
    return merged


def attach_latest_criteria_predictions(trial_id: str, clip_df: pd.DataFrame) -> pd.DataFrame:
    """Use the newest reconstructed five-motion criteria as the primary label.

    criteria_v4 adds conservative multi-scale posture context for long sessions.
    criteria_v3 is still carried through as transition context, while the main
    prediction remains one of the five core actions for normal-trial accuracy.
    """
    comparison_path = RUNS_DIR / trial_id / "raw_parameter_diagnostics" / "clip_feature_rule_comparison.csv"
    output = clip_df.copy()
    output["old_rule_prediction"] = output.get("rule_prediction", "").astype(str)
    output["old_hybrid_prediction"] = output.get("hybrid_prediction", "").astype(str)
    output["model_prediction_source"] = "original_rule"
    if not comparison_path.exists():
        return output

    comparison_df = pd.read_csv(comparison_path)
    join_columns = [
        "clip_idx",
        "improved_rule_prediction",
        "criteria_v2_prediction",
        "criteria_v2_strict_match",
        "criteria_v2_transition_aware_match",
        "criteria_v2_component_transition_aware_match",
        "criteria_v3_base_prediction",
        "criteria_v3_prediction",
        "criteria_v3_transition_reason",
        "criteria_v3_strict_match",
        "criteria_v3_transition_aware_match",
        "criteria_v3_component_transition_aware_match",
        "criteria_v4_base_prediction",
        "criteria_v4_context_state",
        "criteria_v4_prediction",
        "criteria_v4_reason",
        "criteria_v4_transition_evidence",
        "criteria_v4_strict_match",
        "criteria_v4_transition_aware_match",
        "criteria_v4_component_transition_aware_match",
        "criteria_target_label",
        "effective_manual_label",
        "effective_manual_action_count",
        "height_slope_mps",
        "horizontal_path_length_m",
        "window_median_height_m",
        "linear_speed_mean_mps",
    ]
    available_columns = [column for column in join_columns if column in comparison_df.columns]
    if "criteria_v4_prediction" in available_columns:
        prediction_column = "criteria_v4_prediction"
    elif "criteria_v2_prediction" in available_columns:
        prediction_column = "criteria_v2_prediction"
    else:
        prediction_column = "criteria_v3_prediction"
    if "clip_idx" not in available_columns or prediction_column not in available_columns:
        return output

    output = output.merge(comparison_df[available_columns], on="clip_idx", how="left")
    criteria = output[prediction_column].fillna("").astype(str).str.strip()
    has_criteria = criteria.ne("")
    output.loc[has_criteria, "hybrid_prediction"] = criteria[has_criteria]
    source_by_column = {
        "criteria_v4_prediction": "criteria_v4_contextual_posture",
        "criteria_v2_prediction": "criteria_v2",
        "criteria_v3_prediction": "criteria_v3",
    }
    output.loc[has_criteria, "model_prediction_source"] = source_by_column[prediction_column]
    return output


def build_clip_windows(output_dir: Path, clip_seconds: float, effective_duration_s: float) -> list[ClipWindow]:
    clip_count = int(math.floor(effective_duration_s / clip_seconds))
    placeholder_dir = output_dir / "session_clip_index"
    windows: list[ClipWindow] = []
    for clip_idx in range(clip_count):
        start_s = clip_idx * clip_seconds
        end_s = (clip_idx + 1) * clip_seconds
        windows.append(
            ClipWindow(
                clip_idx=clip_idx,
                start_s=start_s,
                end_s=end_s,
                start_epoch_s=0.0,
                end_epoch_s=0.0,
                clip_path=placeholder_dir / f"clip_{clip_idx:04d}.mp4",
            )
        )
    return windows


def build_model_segments(clip_df: pd.DataFrame) -> list[dict[str, Any]]:
    valid_df = clip_df.copy().sort_values("clip_idx").reset_index(drop=True)
    if valid_df.empty:
        return []

    segments: list[dict[str, Any]] = []
    current_label = str(valid_df["hybrid_prediction"].iloc[0])
    current_start_idx = 0
    for idx in range(1, len(valid_df) + 1):
        next_label = str(valid_df["hybrid_prediction"].iloc[idx]) if idx < len(valid_df) else None
        if idx == len(valid_df) or next_label != current_label:
            start_row = valid_df.iloc[current_start_idx]
            end_row = valid_df.iloc[idx - 1]
            segments.append(
                {
                    "label": current_label,
                    "start_s": float(start_row["start_s"]),
                    "end_s": float(end_row["end_s"]),
                    "duration_s": float(end_row["end_s"] - start_row["start_s"]),
                    "clip_start_idx": int(start_row["clip_idx"]),
                    "clip_end_idx": int(end_row["clip_idx"]),
                    "color": color_for_label(current_label),
                }
            )
            if idx < len(valid_df):
                current_label = next_label or ""
                current_start_idx = idx
    return segments


def load_continuous_motion_data(
    trial_id: str,
    session_duration_s: float,
) -> dict[str, Any]:
    """Load a validated, gap-free continuous timeline without replacing v4."""
    source_path = (
        RUNS_DIR
        / trial_id
        / "raw_parameter_diagnostics"
        / "continuous_motion_segments.json"
    )
    fallback = {
        "available": False,
        "version": "",
        "method": "",
        "segments": [],
        "accuracy": {},
        "source_path": str(source_path),
    }
    if not source_path.exists():
        return fallback
    try:
        payload = json.loads(source_path.read_text())
    except (json.JSONDecodeError, OSError):
        return fallback

    valid_segments: list[dict[str, Any]] = []
    previous_end_s = 0.0
    for raw_segment in payload.get("segments", []):
        label = normalize_action_label(str(raw_segment.get("label") or ""))
        start_s = finite_float(raw_segment.get("start_s"))
        end_s = finite_float(raw_segment.get("end_s"))
        if label not in {"sitting", "standing", "walking", "upstairs", "downstairs"}:
            return fallback
        if start_s is None or end_s is None or end_s <= start_s:
            return fallback
        start_s = clamp_float(start_s, 0.0, float(session_duration_s))
        end_s = clamp_float(end_s, 0.0, float(session_duration_s))
        if abs(start_s - previous_end_s) > 0.051:
            return fallback
        if end_s <= start_s:
            continue
        valid_segments.append(
            {
                "segment_index": len(valid_segments),
                "label": label,
                "start_s": round(start_s, 4),
                "end_s": round(end_s, 4),
                "duration_s": round(end_s - start_s, 4),
                "color": COLOR_MAP.get(label, COLOR_MAP["missing"]),
                "source": str(raw_segment.get("source") or "continuous_v1"),
                "start_boundary_source": str(
                    raw_segment.get("start_boundary_source") or "retained_v4_boundary"
                ),
                "start_boundary_confidence": str(
                    raw_segment.get("start_boundary_confidence") or "retained"
                ),
            }
        )
        previous_end_s = end_s

    if not valid_segments or abs(previous_end_s - float(session_duration_s)) > 0.051:
        return fallback
    return {
        "available": True,
        "version": str(payload.get("version") or "continuous_v1"),
        "method": str(payload.get("method") or ""),
        "segments": valid_segments,
        "accuracy": payload.get("accuracy") if isinstance(payload.get("accuracy"), dict) else {},
        "limitations": payload.get("limitations") if isinstance(payload.get("limitations"), list) else [],
        "source_path": str(source_path),
    }


def build_time_spent_records(clip_df: pd.DataFrame, clip_seconds: float) -> list[dict[str, Any]]:
    duration_by_label = clip_df.groupby("hybrid_prediction").size().mul(clip_seconds).sort_values(ascending=False)
    records: list[dict[str, Any]] = []
    for label, seconds in duration_by_label.items():
        normalized = normalize_action_label(str(label))
        records.append(
            {
                "label": str(label),
                "normalized_label": normalized,
                "seconds": float(seconds),
                "minutes": float(seconds / 60.0),
                "color": color_for_label(str(label)),
            }
        )
    return records


def build_time_spent_records_from_segments(
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    duration_by_label: dict[str, float] = {}
    for segment in segments:
        label = normalize_action_label(str(segment.get("label") or "missing"))
        duration_s = finite_float(segment.get("duration_s"))
        if duration_s is None:
            start_s = finite_float(segment.get("start_s"))
            end_s = finite_float(segment.get("end_s"))
            duration_s = (end_s - start_s) if start_s is not None and end_s is not None else 0.0
        duration_by_label[label] = duration_by_label.get(label, 0.0) + max(0.0, duration_s)
    return [
        {
            "label": label,
            "normalized_label": label,
            "seconds": seconds,
            "minutes": seconds / 60.0,
            "color": color_for_label(label),
        }
        for label, seconds in sorted(duration_by_label.items(), key=lambda item: item[1], reverse=True)
    ]


def build_transition_candidates(
    clip_df: pd.DataFrame,
    model_segments: list[dict[str, Any]],
    *,
    session_duration_s: float,
) -> list[dict[str, Any]]:
    raw_candidates: list[dict[str, Any]] = []

    for index in range(1, len(model_segments)):
        prev_segment = model_segments[index - 1]
        next_segment = model_segments[index]
        time_s = float(next_segment["start_s"])
        raw_candidates.append(
            {
                "time_s": time_s,
                "source": "model",
                "summary_label": f"{prev_segment['label']} -> {next_segment['label']}",
                "prev_label": str(prev_segment["label"]),
                "next_label": str(next_segment["label"]),
            }
        )

    for row in clip_df.to_dict(orient="records"):
        if not str(row.get("is_transition_clip") or "").strip():
            continue
        assumed_actions = [
            part.strip()
            for part in str(row.get("assumed_actions_in_clip") or "").split("|")
            if part.strip()
        ]
        change_points = [
            part.strip()
            for part in str(row.get("state_change_times_in_clip_s") or "").split(",")
            if part.strip()
        ]
        for idx, raw_time in enumerate(change_points):
            try:
                time_s = float(raw_time)
            except ValueError:
                continue
            prev_label = assumed_actions[min(idx, max(len(assumed_actions) - 1, 0))] if assumed_actions else ""
            next_label = assumed_actions[min(idx + 1, max(len(assumed_actions) - 1, 0))] if len(assumed_actions) > 1 else ""
            summary = f"{prev_label} -> {next_label}".strip(" ->") or "spoken transition"
            raw_candidates.append(
                {
                    "time_s": time_s,
                    "source": "spoken",
                    "summary_label": summary,
                    "prev_label": prev_label,
                    "next_label": next_label,
                }
            )

    if not raw_candidates:
        return []

    raw_candidates.sort(key=lambda item: (float(item["time_s"]), str(item["source"])))
    merged: list[dict[str, Any]] = []
    merge_tolerance_s = 0.35

    for candidate in raw_candidates:
        time_s = float(candidate["time_s"])
        if merged and abs(time_s - float(merged[-1]["time_s"])) <= merge_tolerance_s:
            existing = merged[-1]
            sources = set(existing.get("sources", []))
            sources.add(str(candidate["source"]))
            existing["sources"] = sorted(sources)
            labels = existing.setdefault("labels", [])
            if candidate["summary_label"] not in labels:
                labels.append(str(candidate["summary_label"]))
            if str(candidate["source"]) == "spoken":
                existing["summary_label"] = str(candidate["summary_label"])
            continue

        merged.append(
            {
                "candidate_idx": len(merged),
                "time_s": clamp_float(time_s, 0.0, session_duration_s),
                "sources": [str(candidate["source"])],
                "summary_label": str(candidate["summary_label"]),
                "labels": [str(candidate["summary_label"])],
            }
        )

    return merged


def transition_display_label(from_action: str, to_action: str) -> str:
    pair = (normalize_action_label(from_action), normalize_action_label(to_action))
    label_map = {
        ("sitting", "standing"): "sit-to-stand",
        ("standing", "sitting"): "stand-to-sit",
        ("standing", "walking"): "stand-to-walk",
        ("walking", "standing"): "walk-to-stand",
    }
    return label_map.get(pair, f"{pair[0]}-to-{pair[1]}")


def transition_category(from_action: str, to_action: str) -> str:
    pair = (normalize_action_label(from_action), normalize_action_label(to_action))
    if pair in CLINICAL_TRANSITION_PAIRS:
        return "clinically_meaningful"
    if "upstairs" in pair or "downstairs" in pair:
        return "stair_context"
    return "other_state_change"


def transition_detection_profile(from_action: str, to_action: str) -> dict[str, Any]:
    transition_type = transition_display_label(from_action, to_action)
    if transition_type in TRANSITION_DETECTION_PROFILES:
        return TRANSITION_DETECTION_PROFILES[transition_type].copy()
    pair = {normalize_action_label(from_action), normalize_action_label(to_action)}
    if "upstairs" in pair or "downstairs" in pair:
        return TRANSITION_DETECTION_PROFILES["stair-context"].copy()
    return TRANSITION_DETECTION_PROFILES["default"].copy()


def transition_review_metadata(from_action: str, to_action: str) -> dict[str, Any]:
    pair = (normalize_action_label(from_action), normalize_action_label(to_action))
    transition_type = transition_display_label(*pair)
    if pair in CLINICAL_TRANSITION_PAIRS:
        likelihood = "common"
        review_priority = "standard"
        clinical_interest = "clinically meaningful transition"
        color = TRANSITION_COLORS[transition_type]
        classification = "clinical"
        classification_label = "Clinical transition"
    elif "upstairs" in pair or "downstairs" in pair:
        likelihood = "less_likely"
        review_priority = "context_review"
        clinical_interest = "context-dependent stair transition"
        color = TRANSITION_COLORS["stair-context"]
        classification = "possible"
        classification_label = "Possible / context transition"
    else:
        likelihood = "needs_review"
        review_priority = "high"
        clinical_interest = "unusual direct transition"
        color = TRANSITION_COLORS["needs-review"]
        classification = "less_possible"
        classification_label = "Less possible / review"
    return {
        "state_kind": "transition",
        "transition_type": transition_type,
        "transition_category": transition_category(*pair),
        "likelihood": likelihood,
        "likelihood_label": likelihood.replace("_", " ").title(),
        "review_priority": review_priority,
        "clinical_interest": clinical_interest,
        "color": color,
        "classification": classification,
        "classification_label": classification_label,
        "classification_color": TRANSITION_CLASSIFICATION_COLORS[classification],
        "detection_parameters": transition_detection_profile(*pair),
    }


def load_transition_state_segments(
    trial_id: str,
    session_duration_s: float,
) -> tuple[list[dict[str, Any]], str]:
    """Load persistent manual states first, then fall back to effective labels."""
    run_dir = RUNS_DIR / trial_id
    candidate_paths = [
        run_dir / "manual_labels" / "session_manual_segments.json",
        run_dir / "parkinson_motion_review" / "manual_segments.json",
    ]
    rows: list[dict[str, Any]] = []
    source = ""
    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            continue
        try:
            payload = load_json(candidate_path)
            rows = list(payload.get("segments") or [])
        except (OSError, ValueError, TypeError):
            rows = []
        if rows:
            source = "saved_manual_session_labels"
            break

    if not rows:
        try:
            effective = load_effective_manual_intervals(trial_id)
        except Exception:
            effective = pd.DataFrame()
        if not effective.empty:
            rows = effective.to_dict(orient="records")
            source = "effective_manual_or_spoken_labels"

    normalized_rows: list[dict[str, Any]] = []
    valid_actions = {"sitting", "standing", "walking", "upstairs", "downstairs"}
    for row in rows:
        start_s = finite_float(row.get("start_s"))
        end_s = finite_float(row.get("end_s"))
        label = normalize_action_label(row.get("motion_label") or row.get("label"))
        if start_s is None or end_s is None or label not in valid_actions:
            continue
        start_s = clamp_float(start_s, 0.0, session_duration_s)
        end_s = clamp_float(end_s, 0.0, session_duration_s)
        if end_s <= start_s:
            continue
        normalized_rows.append(
            {
                "label": label,
                "display_label": label.title(),
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
                "source": source,
                "color": COLOR_MAP.get(label, COLOR_MAP["missing"]),
            }
        )

    normalized_rows.sort(key=lambda item: (float(item["start_s"]), float(item["end_s"])))
    collapsed: list[dict[str, Any]] = []
    for row in normalized_rows:
        if (
            collapsed
            and collapsed[-1]["label"] == row["label"]
            and float(row["start_s"]) <= float(collapsed[-1]["end_s"]) + 0.25
        ):
            collapsed[-1]["end_s"] = max(float(collapsed[-1]["end_s"]), float(row["end_s"]))
            collapsed[-1]["duration_s"] = float(collapsed[-1]["end_s"]) - float(collapsed[-1]["start_s"])
            continue
        collapsed.append(row.copy())
    return collapsed, source


def transition_time_is_manually_covered(time_s: float, segments: list[dict[str, Any]]) -> bool:
    return any(
        float(segment["start_s"]) - 0.25 <= time_s <= float(segment["end_s"]) + 0.25
        for segment in segments
    )


def build_manual_transition_candidates(
    trial_id: str,
    segments: list[dict[str, Any]],
    session_duration_s: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for previous, following in zip(segments, segments[1:]):
        from_action = str(previous["label"])
        to_action = str(following["label"])
        if from_action == to_action:
            continue
        gap_s = float(following["start_s"]) - float(previous["end_s"])
        if gap_s > 1.0:
            continue
        boundary_s = (float(previous["end_s"]) + float(following["start_s"])) / 2.0
        metadata = transition_review_metadata(from_action, to_action)
        profile = metadata["detection_parameters"]
        half_window_s = float(profile.get("initial_half_window_s", 0.75))
        candidates.append(
            {
                "transition_id": f"{trial_id}_transition_{len(candidates):03d}",
                "source": str(previous.get("source") or "manual_state_boundary"),
                "from_action": from_action,
                "to_action": to_action,
                **metadata,
                "initial_start_s": clamp_float(boundary_s - half_window_s, 0.0, session_duration_s),
                "initial_end_s": clamp_float(boundary_s + half_window_s, 0.0, session_duration_s),
                "from_segment_start_s": float(previous["start_s"]),
                "from_segment_end_s": float(previous["end_s"]),
                "to_segment_start_s": float(following["start_s"]),
                "to_segment_end_s": float(following["end_s"]),
                "boundary_s": round(boundary_s, 4),
                "clip_start_idx": int(max(0.0, boundary_s) // 2),
                "clip_end_idx": int(max(0.0, boundary_s) // 2),
            }
        )
    return candidates


def clip_label_actions(label: Any) -> list[str]:
    return actions_from_label_display(str(label or ""))


def clip_state_from_label(label: Any) -> dict[str, Any]:
    raw_label = str(label or "").strip()
    actions = clip_label_actions(raw_label)
    if len(actions) >= 2:
        return {
            "kind": "transition",
            "label": transition_display_label(actions[0], actions[-1]),
            "from_action": actions[0],
            "to_action": actions[-1],
            "action": "",
        }
    if len(actions) == 1:
        return {
            "kind": "stable",
            "label": actions[0],
            "from_action": "",
            "to_action": "",
            "action": actions[0],
        }
    normalized = normalize_action_label(raw_label)
    if normalized in COLOR_MAP and normalized != "missing":
        return {
            "kind": "stable",
            "label": normalized,
            "from_action": "",
            "to_action": "",
            "action": normalized,
        }
    return {
        "kind": "unknown",
        "label": raw_label or "unknown",
        "from_action": "",
        "to_action": "",
        "action": "",
    }


def build_clip_state_segments(clip_df: pd.DataFrame) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    if clip_df.empty:
        return segments

    ordered = clip_df.sort_values("start_s").reset_index(drop=True)
    current: dict[str, Any] | None = None
    for row in ordered.to_dict(orient="records"):
        label = row.get("hybrid_prediction") or row.get("criteria_v3_prediction") or row.get("rule_prediction") or ""
        state = clip_state_from_label(label)
        key = (state["kind"], state["label"], state["from_action"], state["to_action"], state["action"])
        start_s = float(row.get("start_s") or 0.0)
        end_s = float(row.get("end_s") or start_s)
        clip_idx = int(float(row.get("clip_idx") or 0))
        if current and current["key"] == key and abs(float(current["end_s"]) - start_s) <= 1e-6:
            current["end_s"] = end_s
            current["clip_end_idx"] = clip_idx
            current["source_labels"].append(str(label))
            continue
        current = {
            "key": key,
            "kind": state["kind"],
            "label": state["label"],
            "from_action": state["from_action"],
            "to_action": state["to_action"],
            "action": state["action"],
            "start_s": start_s,
            "end_s": end_s,
            "clip_start_idx": clip_idx,
            "clip_end_idx": clip_idx,
            "source_labels": [str(label)],
        }
        segments.append(current)
    return segments


def last_stable_segment(segments: list[dict[str, Any]], before_index: int) -> dict[str, Any] | None:
    for index in range(before_index - 1, -1, -1):
        if segments[index].get("kind") == "stable":
            return segments[index]
    return None


def next_stable_segment(segments: list[dict[str, Any]], after_index: int) -> dict[str, Any] | None:
    for index in range(after_index + 1, len(segments)):
        if segments[index].get("kind") == "stable":
            return segments[index]
    return None


def build_transition_signal_frame(
    trajectory_df: pd.DataFrame,
    session_alignment_offset_s: float,
    effective_duration_s: float,
) -> pd.DataFrame:
    required_columns = {
        "tracking_timestamp_us",
        "tx_world_device",
        "ty_world_device",
        "tz_world_device",
        "device_linear_velocity_x_device",
        "device_linear_velocity_y_device",
        "device_linear_velocity_z_device",
    }
    if trajectory_df.empty or not required_columns.issubset(trajectory_df.columns):
        return pd.DataFrame()

    signal_df = trajectory_df[list(required_columns)].copy()
    for column in required_columns:
        signal_df[column] = pd.to_numeric(signal_df[column], errors="coerce")
    signal_df = (
        signal_df
        .dropna(subset=["tracking_timestamp_us", "tx_world_device", "ty_world_device", "tz_world_device"])
        .sort_values("tracking_timestamp_us")
        .reset_index(drop=True)
    )
    if signal_df.empty:
        return signal_df

    first_ts = float(signal_df["tracking_timestamp_us"].iloc[0])
    signal_df["session_s"] = (
        (signal_df["tracking_timestamp_us"] - first_ts) / 1_000_000.0
        + float(session_alignment_offset_s)
    )
    signal_df = signal_df[(signal_df["session_s"] >= 0.0) & (signal_df["session_s"] <= float(effective_duration_s))].copy()
    if signal_df.empty:
        return signal_df

    signal_df["instantaneous_speed_mps"] = np.sqrt(
        signal_df["device_linear_velocity_x_device"].fillna(0.0) ** 2
        + signal_df["device_linear_velocity_y_device"].fillna(0.0) ** 2
        + signal_df["device_linear_velocity_z_device"].fillna(0.0) ** 2
    )
    session_values = signal_df["session_s"].to_numpy()
    finite_diffs = np.diff(session_values)
    finite_diffs = finite_diffs[np.isfinite(finite_diffs) & (finite_diffs > 0)]
    median_dt = float(np.median(finite_diffs)) if len(finite_diffs) else 0.01
    rolling_points = max(1, int(round(0.5 / max(median_dt, 1e-4))))
    signal_df["rolling_speed_mps"] = (
        signal_df["instantaneous_speed_mps"]
        .rolling(window=rolling_points, center=True, min_periods=1)
        .mean()
    )
    dx = signal_df["tx_world_device"].diff().fillna(0.0)
    dy = signal_df["ty_world_device"].diff().fillna(0.0)
    signal_df["horizontal_step_m"] = np.sqrt(dx * dx + dy * dy)
    signal_df["horizontal_path_m"] = signal_df["horizontal_step_m"].cumsum()
    return signal_df


def sample_signal_for_interactive_display(
    signal_df: pd.DataFrame,
    *,
    base_point_budget: int,
    target_hz: float = SIGNAL_DISPLAY_TARGET_HZ,
    maximum_points: int = SIGNAL_DISPLAY_MAX_POINTS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep enough temporal detail for gait waves while bounding dashboard size."""
    if signal_df.empty or "session_s" not in signal_df.columns:
        return signal_df.copy(), {
            "method": "unavailable",
            "source_sample_count": 0,
            "display_sample_count": 0,
            "source_sample_rate_hz": None,
            "display_sample_rate_hz": None,
            "target_display_rate_hz": float(target_hz),
        }

    source_count = len(signal_df)
    start_s = float(signal_df["session_s"].iloc[0])
    end_s = float(signal_df["session_s"].iloc[-1])
    duration_s = max(0.0, end_s - start_s)
    target_points = int(math.ceil(duration_s * max(float(target_hz), 0.1))) + 1
    point_budget = min(
        max(int(maximum_points), 2),
        max(int(base_point_budget), target_points, 2),
    )

    sample_every = max(1, int(math.ceil(source_count / point_budget)))
    if sample_every == 1:
        sample_df = signal_df.copy()
        method = "full-resolution"
    else:
        # A block mean is more stable than selecting every Nth row and avoids
        # phase-dependent aliasing of the small vertical gait oscillation.
        numeric_columns = [
            column
            for column in signal_df.columns
            if pd.api.types.is_numeric_dtype(signal_df[column])
        ]
        block_ids = np.arange(source_count, dtype=np.int64) // sample_every
        sample_df = (
            signal_df[numeric_columns]
            .groupby(block_ids, sort=False)
            .mean(numeric_only=True)
            .reset_index(drop=True)
        )
        endpoints = signal_df.iloc[[0, -1]][numeric_columns]
        sample_df = (
            pd.concat([endpoints.iloc[[0]], sample_df, endpoints.iloc[[1]]], ignore_index=True)
            .sort_values("session_s")
            .drop_duplicates(subset=["session_s"], keep="first")
            .reset_index(drop=True)
        )
        method = "block-mean-time-preserving"

    source_rate_hz = (source_count - 1) / duration_s if source_count > 1 and duration_s > 0 else None
    display_duration_s = (
        float(sample_df["session_s"].iloc[-1]) - float(sample_df["session_s"].iloc[0])
        if len(sample_df) > 1
        else 0.0
    )
    display_rate_hz = (
        (len(sample_df) - 1) / display_duration_s
        if len(sample_df) > 1 and display_duration_s > 0
        else None
    )
    metadata = {
        "method": method,
        "source_sample_count": int(source_count),
        "display_sample_count": int(len(sample_df)),
        "source_sample_rate_hz": round(float(source_rate_hz), 3) if source_rate_hz is not None else None,
        "display_sample_rate_hz": round(float(display_rate_hz), 3) if display_rate_hz is not None else None,
        "target_display_rate_hz": float(target_hz),
        "point_budget": int(point_budget),
    }
    return sample_df, metadata


def robust_window_median(signal_df: pd.DataFrame, start_s: float, end_s: float, column: str) -> float | None:
    if signal_df.empty or column not in signal_df.columns:
        return None
    if end_s <= start_s:
        return None
    values = signal_df.loc[
        (signal_df["session_s"] >= start_s) & (signal_df["session_s"] <= end_s),
        column,
    ].dropna()
    if values.empty:
        return None
    return float(values.median())


def first_time_where(part: pd.DataFrame, mask: pd.Series) -> float | None:
    if part.empty or mask.empty:
        return None
    hits = part.loc[mask]
    if hits.empty:
        return None
    return float(hits["session_s"].iloc[0])


def refine_transition_window(candidate: dict[str, Any], signal_df: pd.DataFrame, session_duration_s: float) -> dict[str, Any]:
    initial_start = float(candidate["initial_start_s"])
    initial_end = float(candidate["initial_end_s"])
    center_s = (initial_start + initial_end) / 2.0
    profile = dict(candidate.get("detection_parameters") or {})
    search_padding_s = float(profile.get("search_padding_s", 1.5))
    safety_margin_s = float(profile.get("safety_margin_s", 0.20))
    search_start = clamp_float(initial_start - search_padding_s, 0.0, session_duration_s)
    search_end = clamp_float(initial_end + search_padding_s, 0.0, session_duration_s)
    refined_start = initial_start
    refined_end = initial_end
    reason = "label-window-expanded"
    confidence = "medium" if candidate.get("source") == "transition_label" else "review"

    if not signal_df.empty:
        part = signal_df[(signal_df["session_s"] >= search_start) & (signal_df["session_s"] <= search_end)].copy()
        if not part.empty:
            before_start = max(float(candidate.get("from_segment_start_s", initial_start - 2.0)), initial_start - 2.0, 0.0)
            before_end = min(float(candidate.get("from_segment_end_s", initial_start)), initial_start)
            after_start = max(float(candidate.get("to_segment_start_s", initial_end)), initial_end)
            after_end = min(float(candidate.get("to_segment_end_s", initial_end + 2.0)), initial_end + 2.0, session_duration_s)
            before_z = robust_window_median(signal_df, before_start, before_end, "tz_world_device")
            after_z = robust_window_median(signal_df, after_start, after_end, "tz_world_device")
            before_speed = robust_window_median(signal_df, before_start, before_end, "rolling_speed_mps") or 0.0
            after_speed = robust_window_median(signal_df, after_start, after_end, "rolling_speed_mps") or 0.0
            from_action = normalize_action_label(str(candidate.get("from_action") or ""))
            to_action = normalize_action_label(str(candidate.get("to_action") or ""))
            pair = (from_action, to_action)

            if before_z is not None and after_z is not None and pair in {
                ("sitting", "standing"),
                ("standing", "sitting"),
            }:
                dz = after_z - before_z
                minimum_height_change = float(profile.get("minimum_height_change_m", 0.05))
                start_progress_threshold = float(profile.get("start_height_progress", 0.08))
                end_progress_threshold = float(profile.get("end_height_progress", 0.92))
                if abs(dz) >= minimum_height_change:
                    progress = (part["tz_world_device"] - before_z) / dz
                    start_time = first_time_where(part, progress >= start_progress_threshold)
                    post_start = part[part["session_s"] >= (start_time if start_time is not None else center_s)]
                    end_progress = (post_start["tz_world_device"] - before_z) / dz if not post_start.empty else pd.Series(dtype=float)
                    end_time = first_time_where(post_start, end_progress >= end_progress_threshold) if not post_start.empty else None
                    if start_time is not None:
                        refined_start = start_time
                    if end_time is not None:
                        refined_end = end_time
                    reason = f"height-progress-{start_progress_threshold:.0%}-to-{end_progress_threshold:.0%}"
                    confidence = "high"

            elif pair in {("standing", "walking"), ("walking", "standing")}:
                speed_delta = after_speed - before_speed
                if pair == ("standing", "walking"):
                    start_fraction = float(profile.get("start_speed_fraction", 0.28))
                    end_fraction = float(profile.get("end_speed_fraction", 0.72))
                    minimum_start = float(profile.get("minimum_start_speed_change_mps", 0.05))
                    minimum_end = float(profile.get("minimum_end_speed_change_mps", 0.12))
                    start_threshold = before_speed + max(minimum_start, start_fraction * max(speed_delta, 0.0))
                    end_threshold = before_speed + max(minimum_end, end_fraction * max(speed_delta, 0.0))
                    start_time = first_time_where(part, part["rolling_speed_mps"] >= start_threshold)
                    post_start = part[part["session_s"] >= (start_time if start_time is not None else center_s)]
                    end_time = first_time_where(post_start, post_start["rolling_speed_mps"] >= end_threshold) if not post_start.empty else None
                else:
                    speed_drop = max(before_speed - after_speed, 0.0)
                    start_fraction = float(profile.get("start_speed_fraction", 0.28))
                    end_fraction = float(profile.get("end_speed_fraction", 0.28))
                    minimum_start = float(profile.get("minimum_start_speed_change_mps", 0.08))
                    minimum_end = float(profile.get("minimum_end_speed_change_mps", 0.06))
                    start_threshold = before_speed - max(minimum_start, start_fraction * speed_drop)
                    end_threshold = after_speed + max(minimum_end, end_fraction * speed_drop)
                    start_time = first_time_where(part, part["rolling_speed_mps"] <= start_threshold)
                    post_start = part[part["session_s"] >= (start_time if start_time is not None else center_s)]
                    end_time = first_time_where(post_start, post_start["rolling_speed_mps"] <= end_threshold) if not post_start.empty else None
                if start_time is not None:
                    refined_start = start_time
                if end_time is not None:
                    refined_end = end_time
                reason = "speed-change-threshold"
                confidence = "high" if start_time is not None and end_time is not None else "medium"

    refined_start = clamp_float(refined_start - safety_margin_s, 0.0, session_duration_s)
    refined_end = clamp_float(refined_end + safety_margin_s, 0.0, session_duration_s)
    boundary_s = finite_float(candidate.get("boundary_s"))
    if boundary_s is not None and str(candidate.get("source") or "").startswith(("saved_manual", "effective_manual")):
        refined_start = min(refined_start, clamp_float(boundary_s - safety_margin_s, 0.0, session_duration_s))
        refined_end = max(refined_end, clamp_float(boundary_s + safety_margin_s, 0.0, session_duration_s))
    if refined_end <= refined_start:
        refined_start = clamp_float(center_s - 0.4, 0.0, session_duration_s)
        refined_end = clamp_float(center_s + 0.4, 0.0, session_duration_s)

    metrics_part = signal_df[(signal_df["session_s"] >= refined_start) & (signal_df["session_s"] <= refined_end)] if not signal_df.empty else pd.DataFrame()
    height_change = None
    path_change = None
    peak_speed = None
    if not metrics_part.empty:
        height_change = float(metrics_part["tz_world_device"].iloc[-1] - metrics_part["tz_world_device"].iloc[0])
        path_change = float(metrics_part["horizontal_path_m"].iloc[-1] - metrics_part["horizontal_path_m"].iloc[0])
        peak_speed = float(metrics_part["rolling_speed_mps"].max())

    output = candidate.copy()
    output.update(
        {
            "start_s": round(refined_start, 4),
            "end_s": round(refined_end, 4),
            "center_s": round((refined_start + refined_end) / 2.0, 4),
            "duration_s": round(refined_end - refined_start, 4),
            "refinement_reason": reason,
            "confidence": confidence,
            "height_change_m": round(height_change, 5) if height_change is not None and math.isfinite(height_change) else None,
            "xy_path_change_m": round(path_change, 5) if path_change is not None and math.isfinite(path_change) else None,
            "peak_speed_mps": round(peak_speed, 5) if peak_speed is not None and math.isfinite(peak_speed) else None,
            "detection_parameters": profile,
            "parameter_signal": str(profile.get("signal") or "SLAM context"),
        }
    )
    return output


def build_transition_review_data(
    trial_id: str,
    clip_df: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    session_alignment_offset_s: float,
    effective_duration_s: float,
    *,
    max_points: int = 5200,
    automatic_segments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False,
        "windows": [],
        "samples": [],
        "action_spans": [],
        "stable_states": [],
        "unlikely_examples": [],
        "parameter_profiles": TRANSITION_DETECTION_PROFILES,
        "sampling": {},
        "source_note": (
            "Transition records are initialized from persistent manual state boundaries when available, "
            "with model boundaries used only outside manual coverage. Type-specific SLAM parameters then refine "
            "the start and end, including a small safety margin for video review."
        ),
    }
    if trial_id in TRANSITION_REVIEW_TRIAL_EXCLUDE:
        result["source_note"] = "Transition review is intentionally skipped for this simulated Parkinson recording for now."
        return result

    signal_df = build_transition_signal_frame(trajectory_df, session_alignment_offset_s, effective_duration_s)
    if not signal_df.empty:
        sample_df, sampling_metadata = sample_signal_for_interactive_display(
            signal_df,
            base_point_budget=max_points,
        )
        result["sampling"] = sampling_metadata
        display_rate_hz = sampling_metadata.get("display_sample_rate_hz")
        if display_rate_hz is not None:
            result["source_note"] += (
                f" The interactive height trace retains approximately {display_rate_hz:.1f} samples/s "
                "using short block means so gait-scale vertical oscillations remain visible when zoomed."
            )
        result["samples"] = [
            {
                "t": round(float(row["session_s"]), 4),
                "x": round(float(row["tx_world_device"]), 5),
                "y": round(float(row["ty_world_device"]), 5),
                "z": round(float(row["tz_world_device"]), 5),
                "speed": round(float(row["rolling_speed_mps"]), 5),
                "path": round(float(row["horizontal_path_m"]), 5),
            }
            for row in sample_df.to_dict(orient="records")
            if math.isfinite(float(row["session_s"]))
        ]

    manual_segments, manual_source = load_transition_state_segments(trial_id, float(effective_duration_s))
    result["manual_state_source"] = manual_source or "none"
    result["stable_states"] = [
        {
            **segment,
            "start_s": round(float(segment["start_s"]), 4),
            "end_s": round(float(segment["end_s"]), 4),
            "duration_s": round(float(segment["duration_s"]), 4),
        }
        for segment in manual_segments
    ]
    result["action_spans"] = [
        {
            "label": segment["label"],
            "display_label": segment["display_label"],
            "start_s": round(float(segment["start_s"]), 4),
            "end_s": round(float(segment["end_s"]), 4),
            "color": segment["color"],
        }
        for segment in manual_segments
    ]

    raw_candidates = build_manual_transition_candidates(
        trial_id,
        manual_segments,
        float(effective_duration_s),
    )
    model_state_segments = build_clip_state_segments(clip_df)
    if automatic_segments is not None:
        model_state_segments = [
            {**segment, "kind": "stable", "action": segment["label"],
             "clip_start_idx": int(segment["start_s"] // 2),
             "clip_end_idx": int(max(0, segment["end_s"] - 1e-6) // 2)}
            for segment in automatic_segments
        ]
    for index, segment in enumerate(model_state_segments):
        if segment.get("kind") == "transition":
            previous_stable = last_stable_segment(model_state_segments, index)
            next_stable = next_stable_segment(model_state_segments, index)
            from_action = str(previous_stable.get("action") if previous_stable else segment.get("from_action") or "")
            to_action = str(next_stable.get("action") if next_stable else segment.get("to_action") or "")
            if not from_action or not to_action or from_action == to_action:
                continue
            boundary_s = (float(segment["start_s"]) + float(segment["end_s"])) / 2.0
            if transition_time_is_manually_covered(boundary_s, manual_segments):
                continue
            metadata = transition_review_metadata(from_action, to_action)
            raw_candidates.append(
                {
                    "transition_id": f"{trial_id}_transition_{len(raw_candidates):03d}",
                    "source": "transition_label",
                    "from_action": from_action,
                    "to_action": to_action,
                    **metadata,
                    "initial_start_s": float(segment["start_s"]),
                    "initial_end_s": float(segment["end_s"]),
                    "from_segment_start_s": float(previous_stable["start_s"]) if previous_stable else float(segment["start_s"]),
                    "from_segment_end_s": float(previous_stable["end_s"]) if previous_stable else float(segment["start_s"]),
                    "to_segment_start_s": float(next_stable["start_s"]) if next_stable else float(segment["end_s"]),
                    "to_segment_end_s": float(next_stable["end_s"]) if next_stable else float(segment["end_s"]),
                    "clip_start_idx": int(segment["clip_start_idx"]),
                    "clip_end_idx": int(segment["clip_end_idx"]),
                }
            )
        elif segment.get("kind") == "stable":
            next_segment = model_state_segments[index + 1] if index + 1 < len(model_state_segments) else None
            if not next_segment or next_segment.get("kind") != "stable":
                continue
            from_action = str(segment.get("action") or "")
            to_action = str(next_segment.get("action") or "")
            if not from_action or not to_action or from_action == to_action:
                continue
            boundary_s = float(next_segment["start_s"])
            if transition_time_is_manually_covered(boundary_s, manual_segments):
                continue
            metadata = transition_review_metadata(from_action, to_action)
            half_window_s = float(metadata["detection_parameters"].get("initial_half_window_s", 0.75))
            raw_candidates.append(
                {
                    "transition_id": f"{trial_id}_transition_{len(raw_candidates):03d}",
                    "source": "stable_boundary",
                    "from_action": from_action,
                    "to_action": to_action,
                    **metadata,
                    "initial_start_s": clamp_float(boundary_s - half_window_s, 0.0, float(effective_duration_s)),
                    "initial_end_s": clamp_float(boundary_s + half_window_s, 0.0, float(effective_duration_s)),
                    "from_segment_start_s": float(segment["start_s"]),
                    "from_segment_end_s": float(segment["end_s"]),
                    "to_segment_start_s": float(next_segment["start_s"]),
                    "to_segment_end_s": float(next_segment["end_s"]),
                    "clip_start_idx": int(segment["clip_end_idx"]),
                    "clip_end_idx": int(next_segment["clip_start_idx"]),
                }
            )

    raw_candidates.sort(key=lambda item: float(item["initial_start_s"]))
    for candidate_index, candidate in enumerate(raw_candidates):
        candidate["transition_id"] = f"{trial_id}_transition_{candidate_index:03d}"

    refined: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, int]] = set()
    for candidate in raw_candidates:
        key = (
            str(candidate["from_action"]),
            str(candidate["to_action"]),
            int(round(float(candidate["initial_start_s"]) * 10)),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        refined_candidate = refine_transition_window(candidate, signal_df, float(effective_duration_s))
        refined_candidate["transition_idx"] = len(refined)
        refined_candidate["display_label"] = (
            f"{refined_candidate['from_action']} -> {refined_candidate['to_action']}"
        )
        refined.append(refined_candidate)

    result["windows"] = refined
    result["unlikely_examples"] = [
        candidate for candidate in refined if candidate.get("likelihood") == "needs_review"
    ]
    result["summary_counts"] = {
        "total": len(refined),
        "common": sum(candidate.get("likelihood") == "common" for candidate in refined),
        "less_likely": sum(candidate.get("likelihood") == "less_likely" for candidate in refined),
        "needs_review": sum(candidate.get("likelihood") == "needs_review" for candidate in refined),
        "clinical": sum(candidate.get("classification") == "clinical" for candidate in refined),
        "possible": sum(candidate.get("classification") == "possible" for candidate in refined),
        "less_possible": sum(candidate.get("classification") == "less_possible" for candidate in refined),
        "clinically_interesting": sum(
            candidate.get("transition_category") == "clinically_meaningful" for candidate in refined
        ),
    }
    result["available"] = bool(refined)
    return result


def clamp_float(value: float, min_value: float, max_value: float) -> float:
    return min(max(value, min_value), max_value)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def can_reuse_media(
    output_path: Path,
    metadata_path: Path,
    *,
    inputs: list[Path],
    config: dict[str, Any],
) -> bool:
    if not output_path.exists() or not metadata_path.exists():
        return False
    try:
        existing = load_json(metadata_path)
    except Exception:
        return False
    if existing != config:
        return False
    newest_input = max(path.stat().st_mtime for path in inputs if path.exists())
    return output_path.stat().st_mtime >= newest_input


def run_ffmpeg(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def resolve_ffmpeg_executable() -> str | None:
    ffmpeg_exe = discover_ffmpeg_executable()
    if ffmpeg_exe:
        return ffmpeg_exe
    for candidate in [
        shutil.which("ffmpeg"),
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def relative_versioned_href(base_path: Path, target_path: Path) -> str:
    href = relative_media_href(base_path, target_path)
    try:
        version = int(target_path.stat().st_mtime)
    except FileNotFoundError:
        return href
    joiner = "&" if "?" in href else "?"
    return f"{href}{joiner}v={version}"


def quantize_time_to_frame(time_s: float, fps: float) -> float:
    if not fps or fps <= 0:
        return time_s
    return round(time_s * fps) / fps


def video_duration_s(video_path: Path) -> float:
    fps, frame_count, _, _ = load_video_info(video_path)
    if fps <= 0:
        return 0.0
    return frame_count / fps


def floor_to_clip_duration(duration_s: float, clip_seconds: float) -> float:
    return math.floor(max(0.0, duration_s) / clip_seconds) * clip_seconds


def ensure_downward_session_video(
    ffmpeg_exe: str,
    source_video_path: Path,
    output_path: Path,
    metadata_path: Path,
    *,
    trim_start_s: float,
    duration_s: float,
) -> Path:
    config = {
        "type": "downward_session_video",
        "source_video_path": str(source_video_path),
        "trim_start_s": round(trim_start_s, 6),
        "duration_s": round(duration_s, 6),
        "force_seek_friendly_gop_frames": 30,
        "include_silent_audio_track": True,
    }
    if can_reuse_media(output_path, metadata_path, inputs=[source_video_path], config=config):
        return output_path

    temp_output = output_path.with_suffix(".tmp.mp4")
    command = [
        ffmpeg_exe,
        "-y",
        "-ss",
        f"{trim_start_s:.6f}",
        "-i",
        str(source_video_path),
        "-f",
        "lavfi",
        "-t",
        f"{duration_s:.6f}",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-t",
        f"{duration_s:.6f}",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "30",
        "-keyint_min",
        "30",
        "-sc_threshold",
        "0",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(temp_output),
    ]
    run_ffmpeg(command)
    temp_output.replace(output_path)
    save_json(metadata_path, config)
    return output_path


def ensure_aria_session_video(
    ffmpeg_exe: str,
    source_video_path: Path,
    source_audio_path: Path,
    output_path: Path,
    metadata_path: Path,
    *,
    video_trim_start_s: float,
    audio_trim_start_s: float,
    duration_s: float,
) -> Path:
    config = {
        "type": "aria_session_video",
        "source_video_path": str(source_video_path),
        "source_audio_path": str(source_audio_path),
        "video_trim_start_s": round(video_trim_start_s, 6),
        "audio_trim_start_s": round(audio_trim_start_s, 6),
        "duration_s": round(duration_s, 6),
        "audio_pad_to_duration": True,
    }
    if can_reuse_media(output_path, metadata_path, inputs=[source_video_path, source_audio_path], config=config):
        return output_path

    temp_output = output_path.with_suffix(".tmp.mp4")
    command = [
        ffmpeg_exe,
        "-y",
        "-ss",
        f"{video_trim_start_s:.6f}",
        "-i",
        str(source_video_path),
        "-ss",
        f"{audio_trim_start_s:.6f}",
        "-i",
        str(source_audio_path),
        "-t",
        f"{duration_s:.6f}",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-af",
        "apad",
        "-movflags",
        "+faststart",
        str(temp_output),
    ]
    run_ffmpeg(command)
    temp_output.replace(output_path)
    save_json(metadata_path, config)
    return output_path


def collect_ntrial_dashboard_options(current_trial_id: str, current_dashboard_path: Path) -> list[dict[str, str]]:
    items = [
        {
            "trial_id": "__selector__",
            "label": "Back to Trial Selector",
            "href": relative_versioned_href(current_dashboard_path, RUNS_DIR / "pd_insighter_trial_selector.html"),
        }
    ]
    manifests: list[tuple[str, str]] = []
    for manifest_path in sorted(RUNS_DIR.glob("*/trial_manifest.json")):
        try:
            manifest = load_json(manifest_path)
        except Exception:
            continue
        trial_id = str(manifest.get("trial_id", manifest_path.parent.name))
        if not trial_id.startswith("Ntrial") and trial_id not in ADDITIONAL_NORMAL_TRIALS and manifest.get("trial_type") != "parkinson_simulated":
            continue
        recording_stem = str(manifest.get("recording_stem", manifest_path.parent.name))
        manifests.append((trial_id, recording_stem))

    def sort_key(item: tuple[str, str]) -> tuple[int, str]:
        trial_id = item[0]
        if trial_id.startswith("Ntrial"):
            try:
                return int(trial_id.removeprefix("Ntrial")), trial_id
            except ValueError:
                pass
        return 10**9, trial_id

    manifests.sort(key=sort_key)
    for trial_id, recording_stem in manifests:
        target_dashboard = RUNS_DIR / trial_id / CANONICAL_TRIAL_ENTRY_NAME
        items.append(
            {
                "trial_id": trial_id,
                "label": f"{trial_id} - {recording_stem}",
                "href": relative_versioned_href(current_dashboard_path, target_dashboard),
                "selected": str(trial_id == current_trial_id).lower(),
            }
        )
    return items


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def finite_int(value: Any, default: int = 0) -> int:
    result = finite_float(value)
    return int(result) if result is not None else default


def clean_cell_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def load_stride_review_data(trial_id: str, dashboard_path: Path) -> dict[str, Any]:
    """Load first-pass SLAM step/stride proxy data for the full-session dashboard."""
    diagnostics_dir = RUNS_DIR / "stride_step_diagnostics"
    local_diagnostics = RUNS_DIR / trial_id / "stride_step_diagnostics"
    if (local_diagnostics / "data/step_stride_event_estimates.csv").exists():
        diagnostics_dir = local_diagnostics
    data_dir = diagnostics_dir / "data"
    events_path = data_dir / "step_stride_event_estimates.csv"
    bouts_path = data_dir / "step_stride_bout_summary.csv"
    label_summary_path = data_dir / "step_stride_label_summary.csv"
    plots_dir = diagnostics_dir / "plots"

    result: dict[str, Any] = {
        "available": False,
        "events": [],
        "bouts": [],
        "label_summary": [],
        "source_note": "SLAM head-motion step/stride proxy; validate with downward video before treating as foot-contact stride length.",
        "combined_boxplot_href": (
            relative_versioned_href(dashboard_path, plots_dir / "combined_step_proxy_length_by_label_boxplot.png")
            if (plots_dir / "combined_step_proxy_length_by_label_boxplot.png").exists()
            else None
        ),
    }
    event_columns = [
        "trial_id",
        "bout_id",
        "label",
        "event_index",
        "session_s",
        "timing_method",
        "previous_sway_peak_session_s",
        "sway_peak_session_s",
        "previous_sway_peak_side_proxy",
        "sway_peak_side_proxy",
        "previous_sway_peak_prominence_m",
        "sway_peak_prominence_m",
        "side_proxy",
        "prominence_m",
        "forward_progress_m",
        "horizontal_path_progress_m",
        "lateral_sway_m",
        "tz_world_device_m",
        "linear_speed_mps",
        "local_heading_deg",
        "heading_change_in_bout_deg",
        "device_path_forward_alignment",
        "device_forward_horizontal_norm",
        "inter_event_interval_s",
        "alternates_from_previous",
        "step_path_length_m",
        "stride_path_length_m",
        "valid_step_path_length_m",
        "valid_stride_path_length_m",
        "rolling_last4_step_path_length_m",
        "event_confidence",
        "event_warning",
        "estimated_foot_side",
        "foot_side_confidence",
        "foot_side_alignment",
        "foot_side_warning",
        "bout_confidence",
        "bout_warning",
        "bout_start_s",
        "bout_end_s",
    ]
    if events_path.exists():
        try:
            events_df = pd.read_csv(events_path)
        except pd.errors.EmptyDataError:
            events_df = pd.DataFrame(columns=event_columns)
        events_df = events_df[events_df.get("trial_id", "").astype(str) == trial_id].copy()
        available_event_columns = [column for column in event_columns if column in events_df.columns]
        for row in events_df.sort_values(["session_s", "bout_id", "event_index"])[available_event_columns].to_dict(orient="records"):
            result["events"].append(
                {
                    "trial_id": clean_cell_text(row.get("trial_id")) or trial_id,
                    "bout_id": int(finite_float(row.get("bout_id")) or 0),
                    "label": clean_cell_text(row.get("label")),
                    "event_index": int(finite_float(row.get("event_index")) or 0),
                    "session_s": finite_float(row.get("session_s")),
                    "timing_method": clean_cell_text(row.get("timing_method")),
                    "previous_sway_peak_session_s": finite_float(row.get("previous_sway_peak_session_s")),
                    "sway_peak_session_s": finite_float(row.get("sway_peak_session_s")),
                    "previous_sway_peak_side_proxy": clean_cell_text(row.get("previous_sway_peak_side_proxy")),
                    "sway_peak_side_proxy": clean_cell_text(row.get("sway_peak_side_proxy")),
                    "previous_sway_peak_prominence_m": finite_float(row.get("previous_sway_peak_prominence_m")),
                    "sway_peak_prominence_m": finite_float(row.get("sway_peak_prominence_m")),
                    "side_proxy": clean_cell_text(row.get("side_proxy")),
                    "prominence_m": finite_float(row.get("prominence_m")),
                    "forward_progress_m": finite_float(row.get("forward_progress_m")),
                    "horizontal_path_progress_m": finite_float(row.get("horizontal_path_progress_m")),
                    "lateral_sway_m": finite_float(row.get("lateral_sway_m")),
                    "tz_world_device_m": finite_float(row.get("tz_world_device_m")),
                    "linear_speed_mps": finite_float(row.get("linear_speed_mps")),
                    "local_heading_deg": finite_float(row.get("local_heading_deg")),
                    "heading_change_in_bout_deg": finite_float(row.get("heading_change_in_bout_deg")),
                    "device_path_forward_alignment": finite_float(row.get("device_path_forward_alignment")),
                    "device_forward_horizontal_norm": finite_float(row.get("device_forward_horizontal_norm")),
                    "inter_event_interval_s": finite_float(row.get("inter_event_interval_s")),
                    "alternates_from_previous": clean_cell_text(row.get("alternates_from_previous")),
                    "step_path_length_m": finite_float(row.get("step_path_length_m")),
                    "stride_path_length_m": finite_float(row.get("stride_path_length_m")),
                    "valid_step_path_length_m": finite_float(row.get("valid_step_path_length_m")),
                    "valid_stride_path_length_m": finite_float(row.get("valid_stride_path_length_m")),
                    "rolling_last4_step_path_length_m": finite_float(row.get("rolling_last4_step_path_length_m")),
                    "event_confidence": clean_cell_text(row.get("event_confidence")),
                    "event_warning": clean_cell_text(row.get("event_warning")),
                    "estimated_foot_side": clean_cell_text(row.get("estimated_foot_side")),
                    "foot_side_confidence": clean_cell_text(row.get("foot_side_confidence")),
                    "foot_side_alignment": finite_float(row.get("foot_side_alignment")),
                    "foot_side_warning": clean_cell_text(row.get("foot_side_warning")),
                    "bout_confidence": clean_cell_text(row.get("bout_confidence")),
                    "bout_warning": clean_cell_text(row.get("bout_warning")),
                    "bout_start_s": finite_float(row.get("bout_start_s")),
                    "bout_end_s": finite_float(row.get("bout_end_s")),
                }
            )

    if bouts_path.exists():
        bouts_df = pd.read_csv(bouts_path)
        bouts_df = bouts_df[bouts_df.get("trial_id", "").astype(str) == trial_id].copy()
        bout_columns = [
            "trial_id",
            "bout_id",
            "label",
            "bout_start_s",
            "bout_end_s",
            "duration_s",
            "horizontal_path_length_m",
            "net_horizontal_displacement_m",
            "straightness_net_over_path",
            "lateral_range_m",
            "peak_prominence_threshold_m",
            "heading_change_deg",
            "alternation_rate",
            "step_length_cv",
            "detected_lateral_events",
            "detected_step_events",
            "valid_step_count",
            "review_window",
            "window_kind",
            "cadence_steps_per_s",
            "median_step_path_length_m",
            "median_stride_path_length_m",
            "rolling_last4_step_path_length_end_m",
            "confidence",
            "warning",
        ]
        available_bout_columns = [column for column in bout_columns if column in bouts_df.columns]
        for row in bouts_df.sort_values(["bout_start_s", "bout_id"])[available_bout_columns].to_dict(orient="records"):
            result["bouts"].append(
                {
                    "trial_id": clean_cell_text(row.get("trial_id")) or trial_id,
                    "bout_id": int(finite_float(row.get("bout_id")) or 0),
                    "label": clean_cell_text(row.get("label")),
                    "bout_start_s": finite_float(row.get("bout_start_s")),
                    "bout_end_s": finite_float(row.get("bout_end_s")),
                    "duration_s": finite_float(row.get("duration_s")),
                    "horizontal_path_length_m": finite_float(row.get("horizontal_path_length_m")),
                    "net_horizontal_displacement_m": finite_float(row.get("net_horizontal_displacement_m")),
                    "straightness_net_over_path": finite_float(row.get("straightness_net_over_path")),
                    "lateral_range_m": finite_float(row.get("lateral_range_m")),
                    "peak_prominence_threshold_m": finite_float(row.get("peak_prominence_threshold_m")),
                    "heading_change_deg": finite_float(row.get("heading_change_deg")),
                    "alternation_rate": finite_float(row.get("alternation_rate")),
                    "step_length_cv": finite_float(row.get("step_length_cv")),
                    "detected_lateral_events": int(finite_float(row.get("detected_lateral_events")) or 0),
                    "detected_step_events": int(finite_float(row.get("detected_step_events")) or 0),
                    "valid_step_count": int(finite_float(row.get("valid_step_count")) or 0),
                    "review_window": bool(str(row.get("review_window")).lower() in {"true", "1", "yes"}),
                    "window_kind": clean_cell_text(row.get("window_kind")),
                    "cadence_steps_per_s": finite_float(row.get("cadence_steps_per_s")),
                    "median_step_path_length_m": finite_float(row.get("median_step_path_length_m")),
                    "median_stride_path_length_m": finite_float(row.get("median_stride_path_length_m")),
                    "rolling_last4_step_path_length_end_m": finite_float(row.get("rolling_last4_step_path_length_end_m")),
                    "confidence": clean_cell_text(row.get("confidence")),
                    "warning": clean_cell_text(row.get("warning")),
                }
            )

    if label_summary_path.exists():
        label_df = pd.read_csv(label_summary_path)
        for row in label_df.to_dict(orient="records"):
            result["label_summary"].append(
                {
                    "label": clean_cell_text(row.get("label")),
                    "usable_bouts": int(finite_float(row.get("usable_bouts")) or 0),
                    "median_step_proxy_m": finite_float(row.get("median_step_proxy_m")),
                    "median_stride_proxy_m": finite_float(row.get("median_stride_proxy_m")),
                    "median_cadence_events_per_s": finite_float(row.get("median_cadence_events_per_s")),
                }
            )

    result["available"] = bool(result["events"] or result["bouts"])
    return result


def build_stride_signal_data(
    trial_id: str,
    trajectory_df: pd.DataFrame,
    session_alignment_offset_s: float,
    effective_duration_s: float,
    *,
    max_points: int = 2800,
) -> dict[str, Any]:
    """Build compact raw-height/speed traces for the interactive stride page."""
    result: dict[str, Any] = {
        "samples": [],
        "action_spans": [],
        "sampling": {},
        "source_note": (
            "Height is raw MPS tz_world_device aligned to session time. "
            "Speed is a centered 0.5s rolling mean of instantaneous SLAM velocity magnitude. "
            "Purple stride markers come from the SLAM head-motion step proxy."
        ),
    }
    required_columns = {
        "tracking_timestamp_us",
        "tz_world_device",
        "device_linear_velocity_x_device",
        "device_linear_velocity_y_device",
        "device_linear_velocity_z_device",
    }
    if trajectory_df.empty or not required_columns.issubset(trajectory_df.columns):
        return result

    signal_df = trajectory_df[list(required_columns)].copy()
    for column in required_columns:
        signal_df[column] = pd.to_numeric(signal_df[column], errors="coerce")
    signal_df = signal_df.dropna(subset=["tracking_timestamp_us", "tz_world_device"]).sort_values("tracking_timestamp_us")
    if signal_df.empty:
        return result

    first_ts = float(signal_df["tracking_timestamp_us"].iloc[0])
    signal_df["session_s"] = (
        (signal_df["tracking_timestamp_us"] - first_ts) / 1_000_000.0
        + float(session_alignment_offset_s)
    )
    signal_df = signal_df[(signal_df["session_s"] >= 0.0) & (signal_df["session_s"] <= float(effective_duration_s))].copy()
    if signal_df.empty:
        return result

    signal_df["instantaneous_speed_mps"] = np.sqrt(
        signal_df["device_linear_velocity_x_device"].fillna(0.0) ** 2
        + signal_df["device_linear_velocity_y_device"].fillna(0.0) ** 2
        + signal_df["device_linear_velocity_z_device"].fillna(0.0) ** 2
    )
    session_values = signal_df["session_s"].to_numpy()
    diffs = np.diff(session_values)
    finite_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    median_dt = float(np.median(finite_diffs)) if len(finite_diffs) else 0.01
    rolling_points = max(1, int(round(0.5 / max(median_dt, 1e-4))))
    signal_df["rolling_speed_mps"] = (
        signal_df["instantaneous_speed_mps"]
        .rolling(window=rolling_points, center=True, min_periods=1)
        .mean()
    )

    sample_df, sampling_metadata = sample_signal_for_interactive_display(
        signal_df,
        base_point_budget=max_points,
    )
    result["sampling"] = sampling_metadata
    display_rate_hz = sampling_metadata.get("display_sample_rate_hz")
    if display_rate_hz is not None:
        result["source_note"] += (
            f" The interactive trace retains approximately {display_rate_hz:.1f} samples/s "
            "using short block means so long recordings keep gait-scale height detail."
        )

    result["samples"] = [
        {
            "t": round(float(row["session_s"]), 4),
            "z": round(float(row["tz_world_device"]), 5),
            "speed": round(float(row["rolling_speed_mps"]), 5),
        }
        for row in sample_df.to_dict(orient="records")
        if math.isfinite(float(row["session_s"]))
    ]

    try:
        action_intervals = load_effective_manual_intervals(trial_id)
    except Exception:
        action_intervals = pd.DataFrame()
    if not action_intervals.empty:
        for row in action_intervals.to_dict(orient="records"):
            start_s = finite_float(row.get("start_s"))
            end_s = finite_float(row.get("end_s"))
            if start_s is None or end_s is None:
                continue
            start_s = clamp_float(start_s, 0.0, float(effective_duration_s))
            end_s = clamp_float(end_s, 0.0, float(effective_duration_s))
            if end_s <= start_s:
                continue
            label = normalize_action_label(str(row.get("label") or ""))
            result["action_spans"].append(
                {
                    "label": label,
                    "display_label": clean_cell_text(row.get("display_label")) or label.title(),
                    "start_s": round(start_s, 4),
                    "end_s": round(end_s, 4),
                    "color": COLOR_MAP.get(label, COLOR_MAP["missing"]),
                }
            )

    return result


def build_session_payload(
    *,
    trial_id: str,
    manifest: dict[str, Any],
    clip_df: pd.DataFrame,
    model_segments: list[dict[str, Any]],
    time_spent_records: list[dict[str, Any]],
    output_dir: Path,
    downward_session_video: Path | None,
    aria_session_video: Path,
    clip_seconds: float,
    effective_duration_s: float,
    session_fps: float,
    session_frame_count: int,
    transition_candidates: list[dict[str, Any]],
    stride_signal: dict[str, Any] | None = None,
    transition_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recording_stem = str(manifest.get("recording_stem", trial_id))
    dashboard_path = output_dir / FULL_SESSION_DASHBOARD_NAME
    continuous_motion = load_continuous_motion_data(trial_id, effective_duration_s)
    manual_timeline_segments, manual_timeline_source = load_transition_state_segments(
        trial_id,
        effective_duration_s,
    )
    clip_records: list[dict[str, Any]] = []
    for row in clip_df.to_dict(orient="records"):
        label = row.get("hybrid_prediction") or row.get("rule_prediction") or "missing"
        normalized = normalize_action_label(str(label))
        clip_records.append(
            {
                "clip_idx": int(row["clip_idx"]),
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
                "duration_s": float(row["end_s"] - row["start_s"]),
                "label": str(label),
                "normalized_label": normalized,
                "rule_prediction": str(row.get("rule_prediction") or ""),
                "old_rule_prediction": str(row.get("old_rule_prediction") or row.get("rule_prediction") or ""),
                "old_hybrid_prediction": str(row.get("old_hybrid_prediction") or ""),
                "improved_rule_prediction": str(row.get("improved_rule_prediction") or ""),
                "criteria_v2_prediction": str(row.get("criteria_v2_prediction") or ""),
                "criteria_v2_strict_match": str(row.get("criteria_v2_strict_match") or ""),
                "criteria_v2_transition_aware_match": str(row.get("criteria_v2_transition_aware_match") or ""),
                "criteria_v2_component_transition_aware_match": str(row.get("criteria_v2_component_transition_aware_match") or ""),
                "criteria_v3_base_prediction": str(row.get("criteria_v3_base_prediction") or ""),
                "criteria_v3_prediction": str(row.get("criteria_v3_prediction") or ""),
                "criteria_v3_transition_reason": str(row.get("criteria_v3_transition_reason") or ""),
                "criteria_v3_strict_match": str(row.get("criteria_v3_strict_match") or ""),
                "criteria_v3_transition_aware_match": str(row.get("criteria_v3_transition_aware_match") or ""),
                "criteria_v3_component_transition_aware_match": str(row.get("criteria_v3_component_transition_aware_match") or ""),
                "criteria_v4_base_prediction": str(row.get("criteria_v4_base_prediction") or ""),
                "criteria_v4_context_state": str(row.get("criteria_v4_context_state") or ""),
                "criteria_v4_prediction": str(row.get("criteria_v4_prediction") or ""),
                "criteria_v4_reason": str(row.get("criteria_v4_reason") or ""),
                "criteria_v4_transition_evidence": str(row.get("criteria_v4_transition_evidence") or ""),
                "criteria_v4_strict_match": str(row.get("criteria_v4_strict_match") or ""),
                "criteria_v4_transition_aware_match": str(row.get("criteria_v4_transition_aware_match") or ""),
                "criteria_v4_component_transition_aware_match": str(row.get("criteria_v4_component_transition_aware_match") or ""),
                "criteria_target_label": str(row.get("criteria_target_label") or ""),
                "model_prediction_source": str(row.get("model_prediction_source") or "original_rule"),
                "criteria_v2_effective_manual_label": str(row.get("effective_manual_label") or ""),
                "criteria_v2_effective_manual_action_count": finite_int(row.get("effective_manual_action_count")),
                "automatic_spoken_action": str(row.get("automatic_spoken_action") or ""),
                "spoken_words_detected_in_clip": str(row.get("spoken_words_detected_in_clip") or ""),
                "spoken_word_event_count": finite_int(row.get("spoken_word_event_count")),
                "spoken_word_times_s": str(row.get("spoken_word_times_s") or ""),
                "assumed_actions_in_clip": str(row.get("assumed_actions_in_clip") or ""),
                "primary_assumed_action": str(row.get("primary_assumed_action") or ""),
                "is_transition_clip": str(row.get("is_transition_clip") or ""),
                "state_change_times_in_clip_s": str(row.get("state_change_times_in_clip_s") or ""),
                "horizontal_displacement_m": float(row["horizontal_displacement_m"]),
                "vertical_displacement_m": float(row["vertical_displacement_m"]),
                "mean_linear_speed_mps": float(row["mean_linear_speed_mps"]),
                "speed_variance": float(row.get("speed_variance") or 0.0),
                "mean_angular_speed_rps": float(row["mean_angular_speed_rps"]),
                "angular_speed_variance": float(row.get("angular_speed_variance") or 0.0),
                "gravity_variance": float(row.get("gravity_variance") or 0.0),
                "mean_z_m": float(row.get("mean_z_m") or 0.0),
                "height_slope_mps": float(row.get("height_slope_mps") or 0.0),
                "horizontal_path_length_m": float(row.get("horizontal_path_length_m") or 0.0),
                "window_median_height_m": float(row.get("window_median_height_m") or row.get("mean_z_m") or 0.0),
                "linear_speed_mean_mps": float(row.get("linear_speed_mean_mps") or row.get("mean_linear_speed_mps") or 0.0),
                "color": COLOR_MAP.get(normalized, COLOR_MAP["missing"]),
            }
        )

    return {
        "trial_id": trial_id,
        "recording_stem": recording_stem,
        "session_duration_s": effective_duration_s,
        "clip_seconds": clip_seconds,
        "clip_count": len(clip_records),
        "session_fps": session_fps,
        "frame_duration_s": (1.0 / session_fps) if session_fps > 0 else 0.0,
        "session_frame_count": session_frame_count,
        "recording_start_epoch_s": float(clip_df["start_epoch_s"].min()) if not clip_df.empty else 0.0,
        "primary_view_label": "Downward View",
        "secondary_view_label": "Aria View",
        "audio_master_label": "Aria View",
        "trial_selector_href": relative_versioned_href(dashboard_path, RUNS_DIR / "pd_insighter_trial_selector.html"),
        "downward_session_video_uri": relative_versioned_href(dashboard_path, downward_session_video) if downward_session_video else "",
        "aria_session_video_uri": relative_versioned_href(dashboard_path, aria_session_video),
        "frame_context_api_url": f"/api/frame-context?trial_id={trial_id}",
        "time_spent_records": time_spent_records,
        "continuous_time_spent_records": build_time_spent_records_from_segments(
            continuous_motion["segments"]
        ) if continuous_motion["available"] else [],
        "model_segments": model_segments,
        "continuous_model_segments": continuous_motion["segments"],
        "continuous_motion": {
            key: value
            for key, value in continuous_motion.items()
            if key != "segments"
        },
        "manual_timeline_segments": manual_timeline_segments,
        "manual_timeline_source": manual_timeline_source,
        "clip_records": clip_records,
        "stride_review": load_stride_review_data(trial_id, dashboard_path),
        "stride_signal": stride_signal or {"samples": [], "action_spans": []},
        "transition_candidates": transition_candidates,
        "transition_review": transition_review or {"available": False, "windows": [], "samples": [], "action_spans": []},
        "trial_jump_options": collect_ntrial_dashboard_options(trial_id, dashboard_path),
    }


def build_legacy_redirect_html(target_name: str) -> str:
    escaped_target = html.escape(target_name, quote=True)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta http-equiv="refresh" content="0; url={escaped_target}" />
  <title>Redirecting…</title>
  <script>
    window.location.replace("{escaped_target}");
  </script>
</head>
<body>
  <p>Redirecting to the current PD-Insighter session dashboard…</p>
</body>
</html>
"""


def build_optional_clip_review(
    *,
    trial_id: str,
    run_dir: Path,
    recording_stem: str,
    session_output_dir: Path,
    clip_df: pd.DataFrame,
    model_segments: list[dict[str, Any]],
    windows: list[ClipWindow],
    downward_session_video: Path,
    aria_session_video: Path,
) -> Path | None:
    output_html = run_dir / CLIP_REVIEW_DASHBOARD_NAME
    gif_dir = run_dir / "clip_gifs"
    downward_clip_media_dir = session_output_dir / "downward_clip_media"
    aria_clip_media_dir = session_output_dir / "aria_clip_media"
    clip_media_map = create_subclip_media_files_ffmpeg(
        downward_session_video,
        downward_clip_media_dir,
        windows,
    )
    if not clip_media_map:
        clip_media_map = {
            window.clip_idx: downward_clip_media_dir / f"clip_{window.clip_idx:04d}.mp4"
            for window in windows
            if (downward_clip_media_dir / f"clip_{window.clip_idx:04d}.mp4").exists()
        }
    secondary_clip_media_map = create_subclip_media_files_ffmpeg(
        aria_session_video,
        aria_clip_media_dir,
        windows,
    )
    if not secondary_clip_media_map:
        secondary_clip_media_map = {
            window.clip_idx: aria_clip_media_dir / f"clip_{window.clip_idx:04d}.mp4"
            for window in windows
            if (aria_clip_media_dir / f"clip_{window.clip_idx:04d}.mp4").exists()
        }
    if not clip_media_map or not secondary_clip_media_map:
        return None

    segments_df = pd.DataFrame(model_segments)
    trial_selector_href = relative_versioned_href(output_html, RUNS_DIR / "pd_insighter_trial_selector.html")
    full_session_href = relative_versioned_href(output_html, run_dir / CANONICAL_TRIAL_ENTRY_NAME)
    write_interactive_dashboard(
        clip_df,
        segments_df,
        output_html,
        gif_dir,
        trial_id=trial_id,
        recording_stem=recording_stem,
        clip_media_map=clip_media_map,
        secondary_clip_media_map=secondary_clip_media_map,
        primary_view_label="Downward View",
        secondary_view_label="Aria View",
        primary_default_rotation_deg=180,
        secondary_default_rotation_deg=90,
        trial_selector_href=trial_selector_href,
        secondary_mode_href=full_session_href,
        secondary_mode_label="Open Full Session Review",
    )
    return output_html


@environment_navigation
def build_dashboard_html(payload: dict[str, Any]) -> str:
    title = html.escape(f"{payload['trial_id']} - {payload['recording_stem']}")
    page_payload = payload_with_signal_samples(
        payload,
        include_stride_samples=False,
        include_transition_samples=False,
    )
    data_json = json.dumps(page_payload)
    trial_selector_href = html.escape(str(payload.get("trial_selector_href") or "../pd_insighter_trial_selector.html"), quote=True)
    stride_review_href = payload.get("stride_review_href")
    stride_review_link_html = (
        f'<a class="nav-tab" href="{html.escape(str(stride_review_href), quote=True)}">Stride</a>'
        if stride_review_href
        else ""
    )
    manual_labeling_href = payload.get("manual_labeling_href")
    manual_labeling_link_html = (
        f'<a class="nav-tab" href="{html.escape(str(manual_labeling_href), quote=True)}">Manual Labels</a>'
        if manual_labeling_href
        else ""
    )
    manual_labeling_callout_html = (
        f'<a class="utility-link" href="{html.escape(str(manual_labeling_href), quote=True)}">Edit Continuous Manual Labels</a>'
        if manual_labeling_href
        else '<span class="muted">Manual labeling page is not available for this trial.</span>'
    )
    transition_review_href = payload.get("transition_review_href")
    transition_review_link_html = (
        f'<a class="nav-tab" href="{html.escape(str(transition_review_href), quote=True)}">Transitions</a>'
        if transition_review_href
        else ""
    )
    template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #eef3f1;
      --card: rgba(255,255,255,0.92);
      --border: rgba(21, 33, 28, 0.08);
      --text: #16211d;
      --muted: #66756f;
      --accent: #1c8c6a;
      --accent-strong: #13624a;
      --shadow: 0 22px 48px rgba(17, 28, 24, 0.08);
      --soft: rgba(28,140,106,0.08);
      --timeline-bg: rgba(21,33,28,0.07);
      --timeline-stroke: rgba(21,33,28,0.1);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(28,140,106,0.14), transparent 26%),
        linear-gradient(180deg, #f5faf8 0%, var(--bg) 100%);
    }
    .page {
      max-width: 1760px;
      margin: 0 auto;
      padding: 22px;
      display: grid;
      gap: 18px;
    }
    .page > *, .top-grid > *, .detail-grid > *, .media-grid > *, .frame-review-grid > *, .stride-review-grid > * {
      min-width: 0;
      max-width: 100%;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
      padding: 22px;
    }
    .utility-bar {
      display: flex;
      align-items: center;
      gap: 16px;
      position: sticky;
      top: 12px;
      z-index: 15;
      padding: 12px 14px;
    }
    .nav-context {
      display: flex;
      align-items: center;
      gap: 11px;
      flex: 0 0 auto;
      min-width: 190px;
    }
    .nav-mark {
      width: 38px;
      height: 38px;
      border-radius: 13px;
      display: grid;
      place-items: center;
      color: white;
      background: linear-gradient(145deg, var(--accent), var(--accent-strong));
      box-shadow: 0 8px 18px rgba(19,98,74,0.20);
      font-size: 12px;
      font-weight: 900;
      letter-spacing: 0.04em;
    }
    .nav-location {
      display: grid;
      gap: 2px;
      line-height: 1.1;
    }
    .nav-location strong { font-size: 14px; }
    .nav-location span { color: var(--muted); font-size: 12px; font-weight: 700; }
    .nav-tabs {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 3px;
      flex: 1 1 auto;
      min-width: 0;
      padding: 4px;
      border: 1px solid var(--border);
      border-radius: 16px;
      background: rgba(21,33,28,0.045);
    }
    .nav-tab {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 36px;
      padding: 0 12px;
      border-radius: 12px;
      color: var(--muted);
      text-decoration: none;
      white-space: nowrap;
      font-size: 13px;
      font-weight: 800;
    }
    .nav-tab:hover { color: var(--text); background: rgba(255,255,255,0.76); }
    .nav-tab.active {
      color: var(--accent-strong);
      background: white;
      box-shadow: 0 4px 12px rgba(17,28,24,0.08);
    }
    .trial-switcher {
      display: flex;
      align-items: center;
      gap: 9px;
      flex: 0 1 340px;
      min-width: 250px;
    }
    .trial-switcher > span {
      color: var(--muted);
      font-size: 11px;
      font-weight: 900;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .utility-link, .mini-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.96);
      color: var(--text);
      text-decoration: none;
      font-size: 13px;
      font-weight: 700;
      padding: 9px 14px;
      cursor: pointer;
    }
    .audio-toggle {
      gap: 8px;
      min-width: 106px;
      transition: border-color 0.16s ease, background 0.16s ease, color 0.16s ease;
    }
    .audio-toggle::before {
      content: "";
      width: 8px;
      height: 8px;
      flex: 0 0 8px;
      border-radius: 50%;
      background: #98a39f;
    }
    .audio-toggle[aria-pressed="true"] {
      border-color: rgba(28,140,106,0.30);
      background: rgba(28,140,106,0.10);
      color: var(--accent-strong);
    }
    .audio-toggle[aria-pressed="true"]::before {
      background: var(--accent);
      box-shadow: 0 0 0 4px rgba(28,140,106,0.10);
    }
    .utility-select, .speed-select, .annotation-field select, .annotation-field input[type="number"] {
      border-radius: 14px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.97);
      color: var(--text);
      font-size: 14px;
      padding: 10px 12px;
      min-width: 280px;
      font-weight: 600;
    }
    .trial-switcher .utility-select {
      width: 100%;
      min-width: 0;
      padding-block: 9px;
    }
    .hero {
      display: grid;
      gap: 14px;
    }
    .eyebrow {
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 12px;
      font-weight: 800;
    }
    .hero h1 {
      margin: 0;
      font-size: clamp(32px, 3.8vw, 52px);
      line-height: 0.96;
      max-width: 14ch;
    }
    .hero-copy {
      color: var(--muted);
      max-width: 80ch;
      font-size: 15px;
      line-height: 1.55;
    }
    .hero-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.82);
      padding: 9px 13px;
      font-size: 13px;
      font-weight: 700;
    }
    .top-grid, .detail-grid {
      display: grid;
      grid-template-columns: 1.8fr 1.1fr;
      gap: 18px;
    }
    .section-heading {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 12px;
      margin-bottom: 14px;
      flex-wrap: wrap;
    }
    .section-heading h2 {
      margin: 0;
      font-size: 24px;
    }
    .section-heading p, .small-note, .muted {
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }
    .time-spent-bars {
      display: flex;
      align-items: end;
      justify-content: space-around;
      gap: 14px;
      min-height: 260px;
      margin-top: 8px;
    }
    .bar-col {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 10px;
      min-width: 72px;
    }
    .bar-col strong {
      font-size: 13px;
      text-align: center;
    }
    .bar {
      width: min(100%, 94px);
      border-radius: 14px 14px 0 0;
      min-height: 10px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.28);
    }
    .mono {
      font-variant-numeric: tabular-nums;
    }
    .summary-grid {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }
    .summary-card, .metric-card {
      border-radius: 18px;
      padding: 14px 15px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(21,33,28,0.05);
    }
    .summary-card b, .metric-card b, .annotation-field span {
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-size: 12px;
      font-weight: 800;
    }
    .summary-card span, .metric-card span {
      display: block;
      font-size: 17px;
      line-height: 1.3;
    }
    .time-information-card {
      display: grid;
      gap: 16px;
      overflow: hidden;
    }
    .time-information-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }
    .time-information-item {
      min-height: 116px;
      padding: 16px;
      border-radius: 18px;
      border: 1px solid rgba(21,33,28,0.07);
      background: linear-gradient(145deg, rgba(255,255,255,0.92), rgba(236,246,242,0.72));
    }
    .time-information-item b {
      display: block;
      margin-bottom: 8px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-size: 11px;
      font-weight: 850;
    }
    .time-information-value {
      display: block;
      font-size: 19px;
      line-height: 1.25;
      font-weight: 800;
    }
    .time-information-note {
      display: block;
      margin-top: 7px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
    .clock-status {
      display: inline-flex;
      align-items: center;
      width: fit-content;
      padding: 7px 11px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 850;
      background: rgba(28,140,106,0.12);
      color: var(--accent-strong);
    }
    .clock-status.review { background: rgba(217,95,2,0.12); color: #94420a; }
    .clock-status.not-independently-verifiable { background: rgba(117,112,179,0.13); color: #514c91; }
    .time-information-details {
      border-top: 1px solid var(--border);
      padding-top: 14px;
    }
    .time-information-details summary {
      cursor: pointer;
      font-weight: 800;
      color: var(--accent-strong);
    }
    .verification-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 10px 22px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    .session-card {
      display: grid;
      gap: 18px;
    }
    .media-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 18px;
    }
    .view-card {
      display: grid;
      gap: 10px;
      padding: 16px;
      border-radius: 20px;
      background: rgba(248, 251, 250, 0.88);
      border: 1px solid rgba(21,33,28,0.05);
    }
    .view-card h3 {
      margin: 0;
      font-size: 18px;
    }
    .view-card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    .view-frame {
      width: 100%;
      min-height: 420px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(180deg, #0d1014, #10161c);
      border-radius: 18px;
      overflow: hidden;
      padding: 12px;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
    }
    .view-frame video {
      width: 100%;
      height: auto;
      max-height: 640px;
      max-width: 100%;
      object-fit: contain;
      border-radius: 12px;
      background: #000;
      cursor: pointer;
      transition: transform 0.16s ease;
      transform-origin: center center;
    }
    .view-frame video.rotated-quarter-turn {
      width: auto;
      height: 100%;
      max-width: 100%;
      max-height: 100%;
    }
    .transport {
      display: grid;
      gap: 12px;
      padding: 16px;
      border-radius: 20px;
      background: rgba(247,251,249,0.92);
      border: 1px solid rgba(21,33,28,0.05);
    }
    .transport-top {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    .transport-readouts {
      display: grid;
      justify-items: end;
      gap: 6px;
    }
    .sync-status {
      display: inline-flex;
      align-items: center;
      width: fit-content;
      border-radius: 999px;
      border: 1px solid var(--border);
      padding: 6px 9px;
      background: rgba(255,255,255,0.84);
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
    }
    .sync-status.good { color: var(--accent-strong); background: rgba(28,140,106,0.10); }
    .sync-status.adjusting { color: #8a5b00; background: rgba(230,171,2,0.13); }
    .sync-status.ended { color: var(--muted); background: rgba(21,33,28,0.06); }
    .transport-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .play-button {
      border: none;
      background: linear-gradient(180deg, var(--accent), var(--accent-strong));
      color: white;
      font-size: 14px;
      font-weight: 800;
      padding: 10px 16px;
      border-radius: 999px;
      cursor: pointer;
      box-shadow: 0 10px 20px rgba(28,140,106,0.18);
    }
    .global-scrubber {
      display: grid;
      gap: 7px;
      min-width: 0;
    }
    .scrubber-label-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
    }
    .session-scrubber-shell {
      position: relative;
      height: 34px;
      display: flex;
      align-items: center;
    }
    .session-scrubber-rail {
      position: absolute;
      left: 0;
      right: 0;
      top: 13px;
      height: 8px;
      border-radius: 999px;
      background: rgba(21,33,28,0.12);
      box-shadow: inset 0 1px 2px rgba(21,33,28,0.12);
      overflow: hidden;
      pointer-events: none;
    }
    .session-selection-highlight {
      position: absolute;
      top: 0;
      bottom: 0;
      min-width: 2px;
      border-radius: 999px;
      background: rgba(28,140,106,0.45);
      box-shadow: inset 0 0 0 1px rgba(19,98,74,0.50);
    }
    .session-axis-playhead {
      top: 4px;
      bottom: 4px;
      z-index: 2;
    }
    .session-scrubber-shell input[type="range"] {
      position: relative;
      z-index: 3;
      width: 100%;
      height: 34px;
      margin: 0;
      appearance: none;
      -webkit-appearance: none;
      background: transparent;
      cursor: pointer;
    }
    .session-scrubber-shell input[type="range"]::-webkit-slider-runnable-track {
      height: 8px;
      border-radius: 999px;
      background: transparent;
    }
    .session-scrubber-shell input[type="range"]::-webkit-slider-thumb {
      width: 20px;
      height: 20px;
      margin-top: -6px;
      appearance: none;
      -webkit-appearance: none;
      border-radius: 50%;
      border: 3px solid white;
      background: var(--accent-strong);
      box-shadow: 0 3px 10px rgba(17,28,24,0.28);
    }
    .session-scrubber-shell input[type="range"]::-moz-range-track {
      height: 8px;
      border-radius: 999px;
      background: transparent;
    }
    .session-scrubber-shell input[type="range"]::-moz-range-thumb {
      width: 16px;
      height: 16px;
      border-radius: 50%;
      border: 3px solid white;
      background: var(--accent-strong);
      box-shadow: 0 3px 10px rgba(17,28,24,0.28);
    }
    .timeline-board {
      display: grid;
      gap: 12px;
      padding: 16px;
      border-radius: 20px;
      border: 1px solid rgba(21,33,28,0.07);
      background: rgba(248,251,250,0.78);
    }
    .timeline-overview-header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 12px;
      padding-bottom: 3px;
    }
    .timeline-overview-header h3 {
      margin: 0 0 3px;
      font-size: 18px;
    }
    .timeline-overview-header p {
      margin: 0;
      color: var(--muted);
      font-size: 12px;
    }
    .legend-slot {
      display: flex;
      justify-content: flex-end;
      min-width: 0;
    }
    .motion-color-legend {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      flex-wrap: wrap;
      gap: 7px 12px;
      max-width: 660px;
      padding: 9px 11px;
      border-radius: 14px;
      border: 1px solid rgba(21,33,28,0.07);
      background: rgba(255,255,255,0.80);
    }
    .motion-color-legend-title {
      color: var(--muted);
      font-size: 10px;
      font-weight: 900;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .motion-color-legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--text);
      font-size: 11px;
      font-weight: 800;
      text-transform: capitalize;
      white-space: nowrap;
    }
    .motion-color-swatch {
      width: 11px;
      height: 11px;
      border-radius: 4px;
      box-shadow: inset 0 0 0 1px rgba(15,22,19,0.14);
    }
    .timeline-row {
      display: grid;
      grid-template-columns: 150px 1fr;
      gap: 12px;
      align-items: center;
    }
    .timeline-label {
      font-size: 13px;
      font-weight: 800;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .timeline-label span {
      display: block;
      margin-top: 3px;
      font-size: 10px;
      line-height: 1.25;
      letter-spacing: 0.02em;
      text-transform: none;
      font-weight: 700;
    }
    .continuous-timeline-row .timeline-label {
      color: var(--accent-strong);
    }
    .continuous-track {
      height: 44px;
      border: 2px solid rgba(28,140,106,0.34);
      box-shadow: 0 8px 22px rgba(28,140,106,0.10);
    }
    .manual-track {
      height: 36px;
      border-color: rgba(21,33,28,0.18);
    }
    .continuous-method-note {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 16px;
      align-items: center;
      padding: 11px 14px;
      border-radius: 15px;
      background: rgba(28,140,106,0.08);
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .continuous-method-note strong { color: var(--accent-strong); }
    .timeline-track {
      position: relative;
      height: 28px;
      border-radius: 999px;
      background: var(--timeline-bg);
      border: 1px solid var(--timeline-stroke);
      overflow: hidden;
      cursor: pointer;
    }
    .timeline-segment {
      position: absolute;
      top: 0;
      bottom: 0;
      opacity: 0.96;
      pointer-events: none;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      border-right: 1px solid rgba(255,255,255,0.45);
    }
    .timeline-segment-label {
      max-width: 100%;
      padding: 0 6px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: rgba(255,255,255,0.96);
      font-size: 10px;
      font-weight: 850;
      text-shadow: 0 1px 2px rgba(0,0,0,0.38);
    }
    .timeline-selection {
      position: absolute;
      top: 2px;
      bottom: 2px;
      border-radius: 999px;
      border: 2px solid rgba(15,22,19,0.88);
      pointer-events: none;
      z-index: 4;
      background: rgba(255,255,255,0.06);
      box-shadow: 0 0 0 1px rgba(255,255,255,0.78), 0 5px 14px rgba(15,22,19,0.18);
    }
    .continuous-track .timeline-selection::before,
    .continuous-track .timeline-selection::after {
      content: "";
      position: absolute;
      top: 50%;
      width: 7px;
      height: 22px;
      transform: translateY(-50%);
      border-radius: 999px;
      background: #0f1714;
      box-shadow: 0 0 0 2px rgba(255,255,255,0.90);
    }
    .continuous-track .timeline-selection::before { left: -4px; }
    .continuous-track .timeline-selection::after { right: -4px; }
    .timeline-playhead {
      position: absolute;
      top: -3px;
      bottom: -3px;
      width: 3px;
      transform: translateX(-1px);
      border-radius: 999px;
      background: #0f1714;
      box-shadow: 0 0 0 2px rgba(255,255,255,0.72);
      pointer-events: none;
      z-index: 6;
    }
    .timeline-marker {
      position: absolute;
      top: 0;
      bottom: 0;
      width: 2px;
      background: rgba(21,33,28,0.12);
      pointer-events: none;
    }
    .period-player {
      display: grid;
      gap: 16px;
      scroll-margin-top: 92px;
      padding: 18px;
      border-radius: 22px;
      border: 1px solid rgba(28,140,106,0.20);
      background:
        radial-gradient(circle at top right, rgba(28,140,106,0.13), transparent 38%),
        linear-gradient(145deg, rgba(255,255,255,0.96), rgba(238,247,243,0.92));
      box-shadow: 0 16px 32px rgba(17,28,24,0.07);
      animation: selected-range-reveal 0.24s ease both;
    }
    .period-player[hidden] {
      display: none;
    }
    @keyframes selected-range-reveal {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .period-player-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 14px;
      flex-wrap: wrap;
    }
    .period-player-title {
      display: grid;
      gap: 5px;
    }
    .period-player-title h3 {
      margin: 0;
      font-size: 20px;
    }
    .period-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 108px;
      padding: 9px 14px;
      border-radius: 999px;
      color: white;
      font-size: 13px;
      font-weight: 850;
      text-transform: capitalize;
      box-shadow: 0 8px 18px rgba(17,28,24,0.15);
    }
    .period-controls {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .period-controls .play-button {
      min-width: 124px;
    }
    .period-prediction-panel {
      display: grid;
      gap: 9px;
      padding: 14px;
      border-radius: 18px;
      border: 1px solid rgba(28,140,106,0.16);
      background: rgba(255,255,255,0.78);
    }
    .period-prediction-heading {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    .period-prediction-heading strong {
      font-size: 15px;
    }
    .period-prediction-heading span {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .period-prediction-copy {
      display: grid;
      gap: 3px;
    }
    .period-prediction-track {
      height: 70px;
      border-radius: 16px;
      border: 2px solid rgba(28,140,106,0.30);
      background: rgba(21,33,28,0.06);
      box-shadow: inset 0 1px 3px rgba(21,33,28,0.08);
      cursor: pointer;
    }
    .period-prediction-track .timeline-segment-label {
      padding: 0 10px;
      font-size: 13px;
      text-transform: capitalize;
    }
    .period-prediction-track .timeline-playhead {
      top: -5px;
      bottom: -5px;
      width: 4px;
      background: #0b1712;
      box-shadow: 0 0 0 2px white, 0 5px 13px rgba(15,22,19,0.24);
    }
    .period-prediction-axis {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 6px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 750;
    }
    .period-prediction-axis span:nth-child(2),
    .period-prediction-axis span:nth-child(3),
    .period-prediction-axis span:nth-child(4) { text-align: center; }
    .period-prediction-axis span:last-child { text-align: right; }
    .range-boundary-controls {
      display: grid;
      grid-template-columns: repeat(2, minmax(180px, 260px));
      gap: 10px;
      align-items: end;
    }
    .range-boundary-field {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 850;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .range-boundary-field input {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: rgba(255,255,255,0.92);
      color: var(--ink);
      font: 750 14px/1.2 "Avenir Next", "Trebuchet MS", sans-serif;
      padding: 10px 11px;
    }
    .period-scrubber-wrap {
      display: grid;
      grid-template-columns: minmax(64px, auto) minmax(180px, 1fr) minmax(64px, auto);
      gap: 12px;
      align-items: center;
    }
    .period-scrubber-wrap input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
      cursor: pointer;
    }
    .period-detail-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
    }
    .period-detail {
      padding: 11px 12px;
      border-radius: 14px;
      background: rgba(255,255,255,0.78);
      border: 1px solid rgba(21,33,28,0.06);
    }
    .period-detail b {
      display: block;
      margin-bottom: 5px;
      color: var(--muted);
      font-size: 10px;
      font-weight: 850;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    .period-detail span {
      font-size: 14px;
      font-weight: 750;
    }
    .manual-review-callout {
      display: grid;
      gap: 12px;
      padding: 16px;
      border-radius: 18px;
      background: rgba(28,140,106,0.08);
      border: 1px solid rgba(28,140,106,0.14);
    }
    .stride-card {
      display: grid;
      gap: 16px;
      border-color: rgba(28,140,106,0.16);
      background:
        radial-gradient(circle at top right, rgba(28,140,106,0.12), transparent 32%),
        var(--card);
    }
    .stride-toolbar {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 12px;
      flex-wrap: wrap;
    }
    .stride-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .stride-track {
      height: 38px;
      background: rgba(11, 24, 19, 0.08);
    }
    .stride-bout-segment {
      position: absolute;
      top: 5px;
      bottom: 5px;
      border-radius: 999px;
      opacity: 0.28;
      pointer-events: none;
    }
    .stride-event-marker {
      position: absolute;
      top: 4px;
      width: 11px;
      height: 30px;
      border: 2px solid rgba(255,255,255,0.9);
      border-radius: 999px;
      transform: translateX(-50%);
      cursor: pointer;
      z-index: 5;
      box-shadow: 0 6px 14px rgba(17,28,24,0.18);
    }
    .stride-event-marker.active {
      width: 18px;
      background: #111827 !important;
      box-shadow: 0 0 0 5px rgba(28,140,106,0.18), 0 8px 18px rgba(17,28,24,0.22);
    }
    .stride-event-marker.conf-medium { opacity: 0.78; }
    .stride-event-marker.conf-low {
      opacity: 0.46;
      border-style: dashed;
      filter: saturate(0.72);
    }
    .stride-current-line {
      position: absolute;
      top: 0;
      bottom: 0;
      width: 2px;
      background: rgba(17,24,39,0.72);
      pointer-events: none;
      z-index: 6;
    }
    .stride-review-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(280px, 0.85fr);
      gap: 14px;
      align-items: start;
    }
    .stride-event-list {
      display: grid;
      gap: 8px;
      max-height: 300px;
      overflow: auto;
      padding-right: 4px;
    }
    .stride-event-button {
      width: 100%;
      display: grid;
      grid-template-columns: 14px minmax(70px, 0.7fr) minmax(0, 1fr);
      gap: 10px;
      align-items: center;
      text-align: left;
      border: 1px solid rgba(21,33,28,0.08);
      border-radius: 16px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.82);
      color: var(--text);
      cursor: pointer;
    }
    .stride-event-button.active {
      border-color: rgba(28,140,106,0.4);
      background: rgba(235,249,244,0.94);
      box-shadow: 0 8px 18px rgba(28,140,106,0.10);
    }
    .stride-event-button.conf-medium {
      background: rgba(255,249,238,0.88);
    }
    .stride-event-button.conf-low {
      background: rgba(255,255,255,0.55);
      border-style: dashed;
    }
    .stride-side-dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: #64748b;
    }
    .stride-side-dot.positive {
      background: #2f78b7;
    }
    .stride-side-dot.negative {
      background: #c94c4c;
    }
    .stride-event-main {
      display: grid;
      gap: 2px;
    }
    .stride-event-main strong {
      font-size: 14px;
    }
    .stride-event-main span, .stride-event-metrics {
      font-size: 12px;
      color: var(--muted);
    }
    .stride-event-metrics {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 7px;
    }
    .stride-explainer {
      display: grid;
      gap: 9px;
      padding: 14px;
      border-radius: 18px;
      background: rgba(255,255,255,0.72);
      border: 1px dashed rgba(28,140,106,0.22);
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }
    .stride-explainer b {
      color: var(--text);
    }
    .transition-card {
      display: grid;
      gap: 16px;
    }
    .transition-toolbar {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 12px;
      flex-wrap: wrap;
    }
    .transition-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .transition-selects {
      display: flex;
      align-items: end;
      gap: 10px;
      flex-wrap: wrap;
    }
    .compact-field {
      min-width: 180px;
    }
    .transition-summary-grid {
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }
    .frame-review-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }
    .frame-strip-card {
      display: grid;
      gap: 10px;
      padding: 16px;
      border-radius: 20px;
      background: rgba(248, 251, 250, 0.88);
      border: 1px solid rgba(21,33,28,0.05);
    }
    .frame-strip-header {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 10px;
      flex-wrap: wrap;
    }
    .frame-strip-header h3 {
      margin: 0;
      font-size: 18px;
    }
    .frame-strip {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 10px;
    }
    .frame-tile {
      display: grid;
      gap: 8px;
      padding: 10px;
      border-radius: 16px;
      background: rgba(255,255,255,0.84);
      border: 1px solid rgba(21,33,28,0.06);
    }
    .frame-thumb-button {
      border: none;
      padding: 0;
      margin: 0;
      background: transparent;
      cursor: pointer;
      display: block;
      width: 100%;
      aspect-ratio: 4 / 3;
      border-radius: 12px;
      overflow: hidden;
      background: #0d1014;
    }
    .frame-thumb {
      width: 100%;
      height: 100%;
      object-fit: cover;
      border-radius: 12px;
      background: #0d1014;
      display: block;
      transform-origin: center center;
      transition: transform 0.16s ease;
    }
    .frame-thumb.rotated-quarter-turn {
      object-fit: contain;
    }
    .frame-caption {
      display: grid;
      gap: 3px;
      font-size: 12px;
      color: var(--muted);
    }
    .frame-caption strong {
      color: var(--text);
      font-size: 13px;
    }
    .detail-grid {
      align-items: start;
    }
    .metric-grid, .annotation-status-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
    }
    .annotation-panel {
      display: grid;
      gap: 12px;
    }
    .annotation-form {
      display: grid;
      gap: 10px;
    }
    .annotation-form-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      align-items: end;
    }
    .annotation-field {
      display: grid;
      gap: 6px;
    }
    .annotation-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .annotation-actions button {
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.96);
      padding: 9px 14px;
      cursor: pointer;
      font-weight: 700;
      color: var(--text);
    }
    .annotation-actions button.primary {
      background: linear-gradient(180deg, var(--accent), var(--accent-strong));
      color: white;
      border-color: transparent;
    }
    @media (max-width: 1200px) {
      .top-grid, .detail-grid, .media-grid, .frame-review-grid, .stride-review-grid {
        grid-template-columns: 1fr;
      }
    }
    @media (max-width: 760px) {
      body { padding: 0; }
      .page { padding: 14px; }
      .card { padding: 18px; border-radius: 20px; }
      .utility-bar {
        position: static;
        display: grid;
        align-items: stretch;
      }
      .nav-context { min-width: 0; }
      .nav-tabs {
        justify-content: flex-start;
        overflow-x: auto;
      }
      .nav-tab { flex: 0 0 auto; }
      .trial-switcher { min-width: 0; max-width: none; }
      .utility-select, .speed-select, .annotation-field select, .annotation-field input[type="number"] {
        width: 100%;
        min-width: 0;
        max-width: 100%;
      }
      .hero h1 {
        font-size: clamp(29px, 11vw, 40px);
        overflow-wrap: anywhere;
      }
      .transport-readouts { justify-items: start; }
      .transport-actions { align-items: stretch; }
      .transport-actions .speed-select { flex-basis: 100%; }
      .time-spent-bars {
        justify-content: flex-start;
        overflow-x: auto;
        padding-bottom: 8px;
      }
      .bar-col { flex: 0 0 72px; }
      .timeline-row {
        grid-template-columns: 1fr;
        gap: 6px;
      }
      .timeline-overview-header {
        align-items: flex-start;
        flex-direction: column;
      }
      .legend-slot, .motion-color-legend {
        width: 100%;
        justify-content: flex-start;
      }
      .period-scrubber-wrap {
        grid-template-columns: 1fr;
        gap: 6px;
      }
      .range-boundary-controls {
        grid-template-columns: 1fr;
      }
      #periodCurrentText, #periodDurationText {
        justify-self: start;
      }
      .view-frame {
        min-height: 280px;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <nav class="card utility-bar" aria-label="PD Insighter navigation">
      <div class="nav-context">
        <span class="nav-mark" aria-hidden="true">PD</span>
        <div class="nav-location">
          <strong>__TRIAL_ID__</strong>
          <span>Full Session Review</span>
        </div>
      </div>
      <div class="nav-tabs" aria-label="Trial views">
        <span class="nav-tab active" aria-current="page">Full Session</span>
        __MANUAL_LABELING_LINK__
        __STRIDE_REVIEW_LINK__
        __TRANSITION_REVIEW_LINK__
      </div>
      <label class="trial-switcher" for="trialJumpSelect">
        <span>Trial</span>
        <select id="trialJumpSelect" class="utility-select" aria-label="Switch trial or return to all trials"></select>
      </label>
    </nav>

    <section class="card hero">
      <div class="eyebrow">PD-Insighter Clinician Review</div>
      <h1 id="heroTitle"></h1>
      <p id="simulationNotice" hidden style="padding:12px 16px;background:#fff4dc;border-left:4px solid #c78b23;border-radius:10px;">Simulated Parkinson trial. Automatic classification and step/stride estimates are exploratory, not confirmed measurements or a clinical diagnosis. Please review posture and stair labels against the videos before interpreting totals.</p>
      <div id="simulationTools" hidden style="margin:14px 0;line-height:1.8;">
        <strong id="simulationClassificationStatus"></strong><br>
        <a class="utility-link" href="#continuousTimelineRow">View automatic classification</a>
        <a class="utility-link" href="pd_insighter_manual_labeling.html">Label actions</a>
        <a class="utility-link" href="pd_insighter_parkinson_review.html">Freezing / gait markers &amp; smoothing</a>
      </div>
      <div class="hero-copy">
        Review the synchronized videos against one gap-free automatic-classification timeline. Drag across the classification row to choose any start and end time, including a range that crosses several motions.
      </div>
      <div class="hero-pills">
        <span class="pill"><b>Trial</b> <span id="trialPill"></span></span>
        <span class="pill"><b>Recording</b> <span id="recordingPill"></span></span>
        <span class="pill"><b>Audio Source</b> <span id="audioPill"></span></span>
        <span class="pill"><b>Automatic Classification</b> Continuous motion periods</span>
      </div>
    </section>

    <div class="top-grid">
      <section class="card">
        <div class="section-heading">
          <div>
            <h2>Time Spent in Data Capture (min)</h2>
            <p>Motion totals from the continuous automatic classification across the synchronized session.</p>
          </div>
        </div>
        <div id="timeSpentBars" class="time-spent-bars"></div>
      </section>
      <aside class="card">
        <div class="section-heading">
          <div>
            <h2>Session Summary</h2>
            <p>Accuracy and review state refresh after manual corrections are saved.</p>
          </div>
        </div>
        <div id="summaryGrid" class="summary-grid"></div>
      </aside>
    </div>

    <section id="timeInformationSection" class="card time-information-card">
      <div class="section-heading">
        <div>
          <h2>Recording Time &amp; Clock Verification</h2>
          <p>Date, start time, cross-device start difference, and long-session drift checked from video metadata and synchronized audio.</p>
        </div>
        <span id="clockStatus" class="clock-status"></span>
      </div>
      <div id="timeInformationGrid" class="time-information-grid"></div>
      <details class="time-information-details">
        <summary>How this was verified</summary>
        <div id="timeInformationDetails" class="verification-grid"></div>
      </details>
    </section>

    <section class="card session-card">
      <div class="section-heading">
        <div>
          <h2>Full Session Review</h2>
          <p>One shared transport controls both views. Video position, automatic classification, and manual labels use the same horizontal session-time axis.</p>
        </div>
      </div>
      <div class="media-grid">
        <div class="view-card">
          <div class="view-card-header">
            <h3 id="primaryViewTitle"></h3>
            <button id="rotateDownwardButton" class="mini-button" type="button">Rotate Downward 90°</button>
          </div>
          <div class="small-note">Visual master view for gait, floor contact, and environmental context.</div>
          <div class="view-frame">
            <video id="downwardVideo" playsinline muted preload="metadata"></video>
          </div>
        </div>
        <div class="view-card">
          <div class="view-card-header">
            <h3 id="secondaryViewTitle"></h3>
            <button id="rotateAriaButton" class="mini-button" type="button">Rotate Aria 90°</button>
          </div>
          <div class="small-note">Synchronized Aria egocentric view. This video carries the shared audio.</div>
          <div class="view-frame">
            <video id="ariaVideo" playsinline preload="metadata"></video>
          </div>
        </div>
      </div>
      <div class="transport">
        <div class="transport-top">
          <div class="transport-actions">
            <button id="playPauseButton" class="play-button" type="button">Play Session</button>
            <button id="restartButton" class="mini-button" type="button">Restart</button>
            <button id="backButton" class="mini-button" type="button">-2s</button>
            <button id="forwardButton" class="mini-button" type="button">+2s</button>
            <button id="audioToggleButton" class="mini-button audio-toggle" type="button" aria-pressed="true">Audio On</button>
            <select id="speedSelect" class="speed-select">
              <option value="0.5">0.5x</option>
              <option value="0.75">0.75x</option>
              <option value="1" selected>1.0x</option>
              <option value="1.25">1.25x</option>
              <option value="1.5">1.5x</option>
            </select>
          </div>
          <div class="transport-readouts">
            <div id="timeReadout" class="mono muted"></div>
            <div id="syncStatus" class="sync-status">Sync checking...</div>
          </div>
        </div>
      </div>
      <div id="periodPlayer" class="period-player" hidden>
        <div class="period-player-header">
          <div class="period-player-title">
            <div class="eyebrow">Focused Selection</div>
            <h3 id="periodTitle">Custom time range</h3>
            <div id="periodRangeText" class="mono muted"></div>
          </div>
          <span id="periodLabelBadge" class="period-badge">Custom selection</span>
        </div>
        <div class="period-prediction-panel">
          <div class="period-prediction-heading">
            <div class="period-prediction-copy">
              <strong>Enlarged Automatic Classification</strong>
              <span id="periodPredictionReadout">Classification periods within this selection</span>
            </div>
            <div class="legend-slot">
              <div id="focusedMotionColorLegend" class="motion-color-legend" aria-label="Focused motion color key"></div>
            </div>
          </div>
          <div id="periodPredictionTrack" class="timeline-track period-prediction-track" aria-label="Enlarged automatic classification for selected range"></div>
          <div id="periodPredictionAxis" class="period-prediction-axis"></div>
        </div>
        <div class="period-scrubber-wrap">
          <span id="periodCurrentText" class="mono">0.0s</span>
          <input id="periodScrubber" aria-label="Position within selected range" type="range" min="0" max="1" step="0.01" value="0" />
          <span id="periodDurationText" class="mono">0.0s</span>
        </div>
        <div class="period-controls">
          <button id="playPeriodButton" class="play-button" type="button">Play Selection</button>
          <button id="periodStartButton" class="mini-button" type="button">Go to Selection Start</button>
          <button id="setSelectionStartButton" class="mini-button" type="button">Set Start = Current</button>
          <button id="setSelectionEndButton" class="mini-button" type="button">Set End = Current</button>
          <button id="clearSelectionButton" class="mini-button" type="button">Close Selection</button>
        </div>
        <div class="range-boundary-controls">
          <label class="range-boundary-field">
            <span>Selection Start (session s)</span>
            <input id="selectionStartInput" type="number" min="0" step="0.01" value="0" />
          </label>
          <label class="range-boundary-field">
            <span>Selection End (session s)</span>
            <input id="selectionEndInput" type="number" min="0" step="0.01" value="1" />
          </label>
        </div>
        <div id="periodDetailGrid" class="period-detail-grid"></div>
      </div>
      <div id="timelineBoard" class="timeline-board">
        <div class="timeline-overview-header">
          <div>
            <h3>Full Session Overview</h3>
            <p>Use the complete timeline to choose any review range; the focused view opens above after you finish dragging.</p>
          </div>
          <div class="legend-slot">
            <div id="overviewMotionColorLegend" class="motion-color-legend" aria-label="Overview motion color key"></div>
          </div>
        </div>
        <div class="timeline-row video-position-row">
          <div class="timeline-label">Video Position<span>drag to seek; shaded area is your selected range</span></div>
          <div class="global-scrubber">
            <div class="scrubber-label-row">
              <span>Full synchronized session</span>
              <span id="fullSessionRangeReadout" class="mono"></span>
            </div>
            <div class="session-scrubber-shell">
              <div class="session-scrubber-rail">
                <div id="sessionSelectionHighlight" class="session-selection-highlight"></div>
              </div>
              <div id="sessionAxisPlayhead" class="timeline-playhead session-axis-playhead"></div>
              <input id="sessionScrubber" aria-label="Full session video position" type="range" min="0" max="1" step="0.01" value="0" />
            </div>
          </div>
        </div>
        <div id="continuousTimelineRow" class="timeline-row continuous-timeline-row">
          <div class="timeline-label">Continuous Automatic Classification<span>drag to choose a range and play it immediately</span></div>
          <div id="continuousTrack" class="timeline-track continuous-track"></div>
        </div>
        <div class="timeline-row">
          <div class="timeline-label">Manual Labels<span>saved/reference ranges aligned to Aria time</span></div>
          <div id="manualTrack" class="timeline-track manual-track"></div>
        </div>
        <div id="continuousPredictionNote" class="continuous-method-note"></div>
      </div>
    </section>

    <section class="card stride-card">
      <div class="section-heading">
        <div>
          <h2>Stride / Step Review</h2>
          <p>First-pass SLAM head-motion step proxy. Use the downward video to check whether each detected side-to-side peak matches a real step.</p>
        </div>
      </div>
      <div class="stride-toolbar">
        <div class="stride-actions">
          <button id="prevStrideButton" class="mini-button" type="button">Previous Detected Step</button>
          <button id="nextStrideButton" class="mini-button" type="button">Next Detected Step</button>
          <button id="nearestStrideButton" class="mini-button" type="button">Jump To Nearest Step</button>
        </div>
        <label class="annotation-field compact-field">
          <span>Event List Window</span>
          <select id="strideWindowSelect">
            <option value="1">±1s</option>
            <option value="2" selected>±2s</option>
            <option value="4">±4s</option>
            <option value="8">±8s</option>
          </select>
        </label>
        <label class="annotation-field compact-field">
          <span>Average Last N Steps</span>
          <input id="strideAverageStepsInput" type="number" min="1" max="20" step="1" value="4" />
        </label>
      </div>
      <div id="strideSummaryGrid" class="summary-grid"></div>
      <div id="strideTrack" class="timeline-track stride-track"></div>
      <div class="stride-review-grid">
        <div>
          <div class="small-note" id="strideEventListNote"></div>
          <div id="strideEventList" class="stride-event-list"></div>
        </div>
        <div class="stride-explainer">
          <div><b>How to read this:</b> each step marker is estimated between neighboring lateral head-sway extrema. The length shown is horizontal path traveled between estimated step events, and the rolling average uses your chosen last-N previous detected steps across the trial.</div>
          <div><b>Important:</b> this is a head/Aria SLAM proxy, not confirmed foot-contact stride length. Please validate with the downward view.</div>
          <div id="strideSourceNote"></div>
        </div>
      </div>
    </section>

    <section class="card transition-card">
      <div class="section-heading">
        <div>
          <h2>Transition Review</h2>
          <p>Inspect the selected moment with variable local context and synchronized frame neighborhoods, without constraining review to automatic-classification boundaries.</p>
        </div>
      </div>
      <div class="transition-toolbar">
        <div class="transition-actions">
          <button id="prevTransitionButton" class="mini-button" type="button">Previous Transition</button>
          <button id="prevFrameButton" class="mini-button" type="button">-1 Frame</button>
          <button id="nextFrameButton" class="mini-button" type="button">+1 Frame</button>
          <button id="nextTransitionButton" class="mini-button" type="button">Next Transition</button>
        </div>
        <div class="transition-selects">
          <label class="annotation-field compact-field">
            <span>Context Window</span>
            <select id="contextWindowSelect">
              <option value="0.5">±0.5s</option>
              <option value="1.0" selected>±1.0s</option>
              <option value="2.0">±2.0s</option>
              <option value="3.0">±3.0s</option>
            </select>
          </label>
          <label class="annotation-field compact-field">
            <span>Frame Neighborhood</span>
            <select id="framePresetSelect">
              <option value="adjacent" selected>Prev / Current / Next</option>
              <option value="micro">±3 Frames</option>
              <option value="half_second">±15 Frames</option>
            </select>
          </label>
        </div>
      </div>
      <div id="transitionSummaryGrid" class="summary-grid transition-summary-grid"></div>
      <div class="frame-review-grid">
        <div class="frame-strip-card">
          <div class="frame-strip-header">
            <h3>Downward Frame Context</h3>
            <div id="downwardFrameStripNote" class="small-note"></div>
          </div>
          <div id="downwardFrameStrip" class="frame-strip"></div>
        </div>
        <div class="frame-strip-card">
          <div class="frame-strip-header">
            <h3>Aria Frame Context</h3>
            <div id="ariaFrameStripNote" class="small-note"></div>
          </div>
          <div id="ariaFrameStrip" class="frame-strip"></div>
        </div>
      </div>
    </section>

    <div class="detail-grid">
      <section class="card">
        <div class="section-heading">
          <div>
            <h2>Current Motion Context</h2>
            <p>Automatic-classification boundary, manual-label agreement, and the current position within the selected continuous period.</p>
          </div>
        </div>
        <div id="clipPanel" class="metric-grid"></div>
      </section>
      <section class="card">
        <div class="section-heading">
          <div>
            <h2>Manual Label Alignment</h2>
            <p>Saved manual ranges use the same Aria-based session clock as the videos and automatic classification.</p>
          </div>
        </div>
        <div class="annotation-panel">
          <div id="annotationStatus" class="annotation-status-grid"></div>
          <div class="manual-review-callout">
            <div>Use the full-session labeling workspace to add or adjust exact boundaries. Regenerating this review page does not overwrite those saved labels.</div>
            __MANUAL_LABELING_CALLOUT__
            <div id="annotationServerNote" class="muted">Loading saved continuous manual labels...</div>
          </div>
          <div class="annotation-form" hidden>
            <div class="annotation-form-row">
              <label class="annotation-field">
                <span>Label Mode</span>
                <select id="manualModeSelect">
                  <option value="">Keep auto spoken action</option>
                  <option value="single">Single motion</option>
                  <option value="transition">Transition</option>
                </select>
              </label>
              <label id="singleActionWrap" class="annotation-field" style="display:none;">
                <span>Motion</span>
                <select id="singleActionSelect"></select>
              </label>
              <label id="transitionCountWrap" class="annotation-field" style="display:none;">
                <span>Transition Count</span>
                <select id="transitionCountSelect">
                  <option value="2">2 motions</option>
                  <option value="3">3 motions</option>
                  <option value="4">4 motions</option>
                  <option value="5">5 motions</option>
                </select>
              </label>
            </div>
            <div id="transitionActionsWrap" class="annotation-form-row" style="display:none;"></div>
            <div class="annotation-actions">
              <button id="saveManualLabelButton" class="primary" type="button">Save Manual Label</button>
              <button id="resetManualLabelButton" type="button">Reset to Auto</button>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>
  <script>
    const dashboardData = __DATA__;
    const baseColors = {
      sitting: "#e6ab02",
      walking: "#d95f02",
      standing: "#7570b3",
      upstairs: "#1b9e77",
      downstairs: "#666666",
      missing: "#b6bec6"
    };
    const actionOptions = ["walking", "sitting", "standing", "upstairs", "downstairs"];
    const trialId = dashboardData.trial_id;
    const clipSeconds = Number(dashboardData.clip_seconds);
    const sessionDuration = Number(dashboardData.session_duration_s);
    const downwardSessionDuration = Number(dashboardData.downward_session_duration_s || sessionDuration);
    const sessionFps = Number(dashboardData.session_fps || 30);
    const frameDuration = Number(dashboardData.frame_duration_s || (sessionFps > 0 ? (1 / sessionFps) : 0));
    const clips = dashboardData.clip_records.slice().sort((a, b) => a.clip_idx - b.clip_idx);
    const modelSegments = dashboardData.model_segments || [];
    const continuousSegments = (dashboardData.continuous_model_segments || [])
      .map(segment => ({
        ...segment,
        start_s: Number(segment.start_s),
        end_s: Number(segment.end_s),
      }))
      .filter(segment => Number.isFinite(segment.start_s) && Number.isFinite(segment.end_s));
    const embeddedManualSegments = (dashboardData.manual_timeline_segments || [])
      .map(segment => ({
        ...segment,
        motion_label: normalizeActionLabel(segment.motion_label || segment.label),
        start_s: Number(segment.start_s),
        end_s: Number(segment.end_s),
      }))
      .filter(segment => (
        actionOptions.includes(segment.motion_label)
        && Number.isFinite(segment.start_s)
        && Number.isFinite(segment.end_s)
        && segment.end_s > segment.start_s
      ));
    const continuousMotion = dashboardData.continuous_motion || {};
    const continuousAccuracy = continuousMotion.accuracy || {};
    const transitionCandidates = (dashboardData.transition_candidates || []).slice().sort((a, b) => Number(a.time_s) - Number(b.time_s));
    const timeSpentRecords = (dashboardData.continuous_time_spent_records || []).length
      ? dashboardData.continuous_time_spent_records
      : (dashboardData.time_spent_records || []);
    const timeInformation = dashboardData.time_information || {};
    const strideReview = dashboardData.stride_review || {};
    const strideEvents = (strideReview.events || [])
      .map(event => ({ ...event, session_s: Number(event.session_s) }))
      .filter(event => Number.isFinite(event.session_s))
      .sort((a, b) => a.session_s - b.session_s);
    const strideBouts = (strideReview.bouts || [])
      .map(bout => ({ ...bout, bout_start_s: Number(bout.bout_start_s), bout_end_s: Number(bout.bout_end_s) }))
      .filter(bout => Number.isFinite(bout.bout_start_s) && Number.isFinite(bout.bout_end_s))
      .sort((a, b) => a.bout_start_s - b.bout_start_s);
    const jumpOptions = dashboardData.trial_jump_options || [];
    const annotationApiUrl = `/api/manual-labels?trial_id=${encodeURIComponent(trialId)}`;
    const annotationSaveUrl = "/api/manual-labels";
    const sessionManualApiUrl = `/api/session-manual-segments?trial_id=${encodeURIComponent(trialId)}`;
    const frameContextApiUrl = String(dashboardData.frame_context_api_url || `/api/frame-context?trial_id=${encodeURIComponent(trialId)}`);
    const frameContextPresets = {
      adjacent: { label: "Prev / Current / Next", offsets: [-1, 0, 1] },
      micro: { label: "±3 Frames", offsets: [-3, -2, -1, 0, 1, 2, 3] },
      half_second: { label: "±15 Frames", offsets: [-15, -10, -5, 0, 5, 10, 15] },
    };

    const heroTitle = document.getElementById("heroTitle");
    const trialPill = document.getElementById("trialPill");
    const recordingPill = document.getElementById("recordingPill");
    const audioPill = document.getElementById("audioPill");
    const timeSpentBars = document.getElementById("timeSpentBars");
    const summaryGrid = document.getElementById("summaryGrid");
    const timeInformationSection = document.getElementById("timeInformationSection");
    const timeInformationGrid = document.getElementById("timeInformationGrid");
    const timeInformationDetails = document.getElementById("timeInformationDetails");
    const clockStatus = document.getElementById("clockStatus");
    const trialJumpSelect = document.getElementById("trialJumpSelect");
    const downwardVideo = document.getElementById("downwardVideo");
    const ariaVideo = document.getElementById("ariaVideo");
    const rotateDownwardButton = document.getElementById("rotateDownwardButton");
    const rotateAriaButton = document.getElementById("rotateAriaButton");
    const primaryViewTitle = document.getElementById("primaryViewTitle");
    const secondaryViewTitle = document.getElementById("secondaryViewTitle");
    const playPauseButton = document.getElementById("playPauseButton");
    const audioToggleButton = document.getElementById("audioToggleButton");
    const restartButton = document.getElementById("restartButton");
    const backButton = document.getElementById("backButton");
    const forwardButton = document.getElementById("forwardButton");
    const speedSelect = document.getElementById("speedSelect");
    const timeReadout = document.getElementById("timeReadout");
    const syncStatus = document.getElementById("syncStatus");
    const sessionScrubber = document.getElementById("sessionScrubber");
    const sessionSelectionHighlight = document.getElementById("sessionSelectionHighlight");
    const sessionAxisPlayhead = document.getElementById("sessionAxisPlayhead");
    const fullSessionRangeReadout = document.getElementById("fullSessionRangeReadout");
    const continuousTimelineRow = document.getElementById("continuousTimelineRow");
    const continuousTrack = document.getElementById("continuousTrack");
    const continuousPredictionNote = document.getElementById("continuousPredictionNote");
    const manualTrack = document.getElementById("manualTrack");
    const overviewMotionColorLegend = document.getElementById("overviewMotionColorLegend");
    const focusedMotionColorLegend = document.getElementById("focusedMotionColorLegend");
    const periodPlayer = document.getElementById("periodPlayer");
    const periodTitle = document.getElementById("periodTitle");
    const periodRangeText = document.getElementById("periodRangeText");
    const periodLabelBadge = document.getElementById("periodLabelBadge");
    const periodPredictionTrack = document.getElementById("periodPredictionTrack");
    const periodPredictionAxis = document.getElementById("periodPredictionAxis");
    const periodPredictionReadout = document.getElementById("periodPredictionReadout");
    const playPeriodButton = document.getElementById("playPeriodButton");
    const periodStartButton = document.getElementById("periodStartButton");
    const setSelectionStartButton = document.getElementById("setSelectionStartButton");
    const setSelectionEndButton = document.getElementById("setSelectionEndButton");
    const clearSelectionButton = document.getElementById("clearSelectionButton");
    const selectionStartInput = document.getElementById("selectionStartInput");
    const selectionEndInput = document.getElementById("selectionEndInput");
    const periodScrubber = document.getElementById("periodScrubber");
    const periodCurrentText = document.getElementById("periodCurrentText");
    const periodDurationText = document.getElementById("periodDurationText");
    const periodDetailGrid = document.getElementById("periodDetailGrid");
    const strideSummaryGrid = document.getElementById("strideSummaryGrid");
    const strideTrack = document.getElementById("strideTrack");
    const strideEventList = document.getElementById("strideEventList");
    const strideEventListNote = document.getElementById("strideEventListNote");
    const strideSourceNote = document.getElementById("strideSourceNote");
    const prevStrideButton = document.getElementById("prevStrideButton");
    const nextStrideButton = document.getElementById("nextStrideButton");
    const nearestStrideButton = document.getElementById("nearestStrideButton");
    const strideWindowSelect = document.getElementById("strideWindowSelect");
    const strideAverageStepsInput = document.getElementById("strideAverageStepsInput");
    const transitionSummaryGrid = document.getElementById("transitionSummaryGrid");
    const prevTransitionButton = document.getElementById("prevTransitionButton");
    const nextTransitionButton = document.getElementById("nextTransitionButton");
    const prevFrameButton = document.getElementById("prevFrameButton");
    const nextFrameButton = document.getElementById("nextFrameButton");
    const contextWindowSelect = document.getElementById("contextWindowSelect");
    const framePresetSelect = document.getElementById("framePresetSelect");
    const downwardFrameStrip = document.getElementById("downwardFrameStrip");
    const ariaFrameStrip = document.getElementById("ariaFrameStrip");
    const downwardFrameStripNote = document.getElementById("downwardFrameStripNote");
    const ariaFrameStripNote = document.getElementById("ariaFrameStripNote");
    const clipPanel = document.getElementById("clipPanel");
    const annotationStatus = document.getElementById("annotationStatus");
    const manualModeSelect = document.getElementById("manualModeSelect");
    const singleActionWrap = document.getElementById("singleActionWrap");
    const singleActionSelect = document.getElementById("singleActionSelect");
    const transitionCountWrap = document.getElementById("transitionCountWrap");
    const transitionCountSelect = document.getElementById("transitionCountSelect");
    const transitionActionsWrap = document.getElementById("transitionActionsWrap");
    const saveManualLabelButton = document.getElementById("saveManualLabelButton");
    const resetManualLabelButton = document.getElementById("resetManualLabelButton");
    const annotationServerNote = document.getElementById("annotationServerNote");

    const annotationState = {
      apiAvailable: false,
      annotations: {},
      sessionSegments: embeddedManualSegments.slice(),
      summary: null,
    };
    const annotationEditorState = {
      clipIdx: null,
      dirty: false,
      autosaveTimer: null,
      saveInFlight: false,
    };

    let currentTime = 0;
    let currentClipIdx = 0;
    let transitionContextRadiusS = 1.0;
    let strideEventWindowS = 2.0;
    let strideAverageStepCount = 4;
    let framePresetKey = "adjacent";
    let isScrubbing = false;
    let sessionScrubberResumePlayback = false;
    let periodScrubberResumePlayback = false;
    let isProgrammaticSeeking = false;
    let seekRequestId = 0;
    let downwardRotationQuarterTurns = 0;
    let ariaRotationQuarterTurns = 0;
    let lastTimelineInteractionAt = 0;
    let lastTimelineInteractionTarget = -1;
    let frameContextRequestId = 0;
    let lastFrameContextCenterKey = "";
    let syncAnimationId = null;
    let lastSyncStatusUpdateMs = 0;
    let selectedRangeStartS = 0;
    let selectedRangeEndS = Math.min(sessionDuration, 10);
    let hasSelectedReviewRange = false;
    let isSelectingReviewRange = false;
    let selectionPlaybackEndS = null;
    const MIN_SELECTION_DURATION_S = 0.05;
    const SOFT_SYNC_THRESHOLD_S = 0.025;
    const HARD_SYNC_THRESHOLD_S = 0.12;
    const AUDIO_STORAGE_KEY = "pd-insighter-audio-enabled-v1";

    function fmtTime(seconds) {
      const s = Math.max(0, Number(seconds) || 0);
      const totalTenths = Math.round(s * 10);
      const mins = Math.floor(totalTenths / 600);
      const secs = Math.floor((totalTenths % 600) / 10);
      const tenths = totalTenths % 10;
      return mins > 0 ? `${mins}:${String(secs).padStart(2, "0")}.${tenths}` : `${secs}.${tenths}s`;
    }

    function formatBoundarySource(value) {
      const source = String(value || "").trim().toLowerCase();
      const labels = {
        session_start: "session start",
        retained_v4_boundary: "retained classifier boundary",
        refined_0_5s_evidence: "refined nearby evidence",
      };
      return labels[source] || source.replaceAll("_", " ") || "retained classifier boundary";
    }

    function clamp(value, min, max) {
      return Math.min(Math.max(value, min), max);
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function normalizeActionLabel(label) {
      const raw = String(label || "").trim().toLowerCase();
      if (raw === "sit") return "sitting";
      if (raw === "stand") return "standing";
      if (raw === "walk") return "walking";
      if (raw === "go upstairs") return "upstairs";
      if (raw === "go downstairs") return "downstairs";
      return raw;
    }

    function parseActionDisplay(value) {
      const raw = String(value || "").trim().toLowerCase();
      if (!raw) return [];
      const transitionActions = transitionLabelToActions(raw);
      if (transitionActions.length) return transitionActions;
      return raw
        .split(/\\s*(?:\\||-|,|\\/)\\s*/)
        .map(item => normalizeActionLabel(item))
        .filter(item => actionOptions.includes(item));
    }

    function formatActionDisplay(actions) {
      const ordered = [];
      for (const action of actions || []) {
        const normalized = normalizeActionLabel(action);
        if (normalized && !ordered.includes(normalized)) ordered.push(normalized);
      }
      return ordered.join("-");
    }

    function canonicalTransitionLabel(actions) {
      const ordered = [];
      for (const action of actions || []) {
        const normalized = normalizeActionLabel(action);
        if (normalized && actionOptions.includes(normalized) && !ordered.includes(normalized)) {
          ordered.push(normalized);
        }
      }
      if (!ordered.length) return "";
      if (ordered.length === 1) return ordered[0];
      const key = `${ordered[0]}-${ordered[1]}`;
      const clinicalTransitions = {
        "sitting-standing": "sit-to-stand",
        "standing-sitting": "stand-to-sit",
        "standing-walking": "stand-to-walk",
        "walking-standing": "walk-to-stand",
      };
      return clinicalTransitions[key] || ordered.join("-to-");
    }

    function transitionLabelToActions(label) {
      const raw = String(label || "").trim().toLowerCase()
        .replace(/_/g, "-")
        .replace(/\\s+to\\s+/g, "-to-")
        .replace(/\\s*->\\s*/g, "-to-");
      if (!raw) return [];
      const parts = raw.includes("-to-")
        ? raw.split("-to-")
        : raw.split(/\\s*(?:\\||-|,|\\/)\\s*/);
      const actions = [];
      for (const part of parts) {
        const action = normalizeActionLabel(part);
        if (actionOptions.includes(action) && !actions.includes(action)) actions.push(action);
      }
      return actions.length > 1 ? actions : [];
    }

    function buildActionOptionsHtml(selectedValue = "") {
      return actionOptions
        .map(action => `<option value="${action}"${selectedValue === action ? " selected" : ""}>${action}</option>`)
        .join("");
    }

    function labelBackground(labelDisplay) {
      const actions = parseActionDisplay(labelDisplay);
      if (!actions.length) return baseColors.missing;
      if (actions.length === 1) return baseColors[actions[0]] || baseColors.missing;
      const pct = 100 / actions.length;
      const stops = [];
      actions.forEach((action, index) => {
        const start = (index * pct).toFixed(3);
        const end = ((index + 1) * pct).toFixed(3);
        const color = baseColors[action] || baseColors.missing;
        stops.push(`${color} ${start}% ${end}%`);
      });
      return `linear-gradient(90deg, ${stops.join(", ")})`;
    }

    function normalizeModelLabel(label) {
      const transitionActions = transitionLabelToActions(label);
      if (transitionActions.length) return canonicalTransitionLabel(transitionActions);
      return normalizeActionLabel(label);
    }

    function getModelPredictionLabel(clip) {
      return clip.label || clip.criteria_v4_prediction || clip.criteria_v2_prediction || clip.criteria_v3_prediction || clip.rule_prediction || "";
    }

    function getClipAtTime(timeSeconds) {
      const safeTime = clamp(Number(timeSeconds) || 0, 0, sessionDuration);
      return clips.find(clip => safeTime >= clip.start_s && safeTime < clip.end_s) || clips[clips.length - 1];
    }

    function getContinuousSegmentAtTime(timeSeconds) {
      if (!continuousSegments.length) return null;
      const safeTime = clamp(Number(timeSeconds) || 0, 0, sessionDuration);
      return continuousSegments.find(segment => safeTime >= segment.start_s && safeTime < segment.end_s)
        || continuousSegments[continuousSegments.length - 1];
    }

    function getManualActionsAtTime(timeSeconds, fallbackClip) {
      const safeTime = clamp(Number(timeSeconds) || 0, 0, sessionDuration);
      const actions = (annotationState.sessionSegments || [])
        .filter(segment => safeTime >= Number(segment.start_s) && safeTime < Number(segment.end_s))
        .map(segment => normalizeActionLabel(segment.motion_label || segment.label))
        .filter(action => actionOptions.includes(action));
      return actions.length ? [...new Set(actions)] : [];
    }

    function getManualAnnotationForClip(clipIdx) {
      return annotationState.annotations[String(clipIdx)] || null;
    }

    function annotationPayloadKey(mode, actions) {
      return JSON.stringify({
        mode: mode || "",
        actions: Array.isArray(actions) ? actions.filter(Boolean) : [],
      });
    }

    function getAutoActionParts(clip) {
      return parseActionDisplay(clip.automatic_spoken_action || clip.assumed_actions_in_clip || clip.primary_assumed_action || "");
    }

    function getSessionManualActions(clip) {
      const overlaps = (annotationState.sessionSegments || [])
        .filter(segment => Number(segment.end_s) > Number(clip.start_s) && Number(segment.start_s) < Number(clip.end_s))
        .sort((a, b) => Number(a.start_s) - Number(b.start_s));
      const actions = [];
      for (const segment of overlaps) {
        const action = normalizeActionLabel(segment.motion_label || segment.label);
        if (actionOptions.includes(action) && !actions.includes(action)) actions.push(action);
      }
      return actions;
    }

    function getEffectiveActions(clip) {
      const annotation = getManualAnnotationForClip(clip.clip_idx);
      if (annotation && Array.isArray(annotation.actions) && annotation.actions.length) {
        return annotation.actions.slice();
      }
      const liveSessionActions = getSessionManualActions(clip);
      if (liveSessionActions.length) {
        return liveSessionActions;
      }
      const sessionManualActions = parseActionDisplay(clip.criteria_v2_effective_manual_label || "");
      if (sessionManualActions.length) {
        return sessionManualActions;
      }
      return getAutoActionParts(clip);
    }

    function getManualCorrectionDisplay(clip) {
      const annotation = getManualAnnotationForClip(clip.clip_idx);
      if (!annotation || !Array.isArray(annotation.actions) || !annotation.actions.length) return "";
      return formatActionDisplay(annotation.actions);
    }

    function getEffectiveLabelDisplay(clip) {
      return formatActionDisplay(getEffectiveActions(clip));
    }

    function getModelMatchFlags(clip) {
      const effectiveActions = getEffectiveActions(clip);
      const prediction = normalizeModelLabel(getModelPredictionLabel(clip));
      if (!effectiveActions.length) {
        return { strict: null, transitionAware: null, componentAware: null };
      }
      const normalizedActions = effectiveActions.map(action => normalizeActionLabel(action)).filter(action => actionOptions.includes(action));
      const target = canonicalTransitionLabel(normalizedActions);
      const componentAware = normalizedActions.includes(prediction);
      return {
        strict: prediction === target,
        transitionAware: prediction === target,
        componentAware,
      };
    }

    function calculateDashboardAccuracySummary() {
      let comparable = 0;
      let strictMatches = 0;
      let stableComparable = 0;
      let stableMatches = 0;
      for (const clip of clips) {
        const actions = getEffectiveActions(clip)
          .map(action => normalizeActionLabel(action))
          .filter(action => actionOptions.includes(action));
        if (!actions.length) continue;
        const flags = getModelMatchFlags(clip);
        comparable += 1;
        if (flags.strict) strictMatches += 1;
        if (actions.length === 1) {
          stableComparable += 1;
          if (flags.strict) stableMatches += 1;
        }
      }
      return {
        strict_accuracy: comparable ? strictMatches / comparable : null,
        transition_aware_accuracy: comparable ? strictMatches / comparable : null,
        stable_only_accuracy: stableComparable ? stableMatches / stableComparable : null,
        manual_override_count: Object.keys(annotationState.annotations || {}).length,
      };
    }

    function fmtAccuracy(value) {
      if (value == null || Number.isNaN(Number(value))) return "N/A";
      return `${(Number(value) * 100).toFixed(1)}%`;
    }

    function fmtPercentagePointChange(value) {
      if (value == null || Number.isNaN(Number(value))) return "N/A";
      const number = Number(value);
      return `${number >= 0 ? "+" : ""}${number.toFixed(2)} pp`;
    }

    function accuracyReferenceLabel(value) {
      if (value === "continuous_manual_segments") return "saved continuous manual labels";
      if (value === "stable_clip_reference") return "stable 2s manual reference";
      return "no comparable manual reference";
    }

    function fmtMetric(value, digits = 3, suffix = "") {
      if (value == null || value === "") return "N/A";
      const number = Number(value);
      if (!Number.isFinite(number)) return "N/A";
      return `${number.toFixed(digits)}${suffix}`;
    }

    function finiteNumberOrNull(value) {
      if (value == null || value === "") return null;
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    }

    function getStrideEventKey(event) {
      if (!event) return "";
      return `${event.bout_id}:${event.event_index}:${Number(event.session_s).toFixed(3)}`;
    }

    function strideSideLabel(sideProxy) {
      const value = String(sideProxy || "");
      if (value.includes("positive")) return "Lateral +";
      if (value.includes("negative")) return "Lateral -";
      return value || "Side peak";
    }

    function strideSideClass(sideProxy) {
      const value = String(sideProxy || "");
      if (value.includes("positive")) return "positive";
      if (value.includes("negative")) return "negative";
      return "";
    }

    function strideConfidenceClass(confidence) {
      const value = String(confidence || "").toLowerCase();
      if (value === "review") return "conf-low";
      if (value === "low") return "conf-low";
      if (value === "medium") return "conf-medium";
      return "conf-high";
    }

    function strideConfidenceLabel(confidence) {
      const value = String(confidence || "").toLowerCase();
      if (value === "review") return "review only";
      if (value === "low") return "low";
      if (value === "medium") return "medium";
      if (value === "high") return "high";
      return "unknown";
    }

    function strideFootSideLabel(side) {
      const value = String(side || "").toLowerCase();
      if (value === "left") return "Est. left foot";
      if (value === "right") return "Est. right foot";
      return "Foot side unknown";
    }

    function getNearestStrideEvent(targetTime = currentTime) {
      if (!strideEvents.length) return null;
      let best = strideEvents[0];
      let bestDistance = Math.abs(Number(best.session_s) - targetTime);
      for (const event of strideEvents.slice(1)) {
        const distance = Math.abs(Number(event.session_s) - targetTime);
        if (distance < bestDistance) {
          best = event;
          bestDistance = distance;
        }
      }
      return best;
    }

    function getAdjacentStrideEvent(direction) {
      if (!strideEvents.length) return null;
      if (direction < 0) {
        for (let index = strideEvents.length - 1; index >= 0; index -= 1) {
          if (Number(strideEvents[index].session_s) < currentTime - 1e-6) return strideEvents[index];
        }
      } else {
        for (const event of strideEvents) {
          if (Number(event.session_s) > currentTime + 1e-6) return event;
        }
      }
      return null;
    }

    function getActiveStrideBout(targetTime = currentTime) {
      return strideBouts.find(bout => targetTime >= bout.bout_start_s && targetTime <= bout.bout_end_s) || null;
    }

    function getStrideBoutForEvent(event) {
      if (!event) return null;
      return strideBouts.find(bout => Number(bout.bout_id) === Number(event.bout_id) && String(bout.label) === String(event.label)) || null;
    }

    function getRollingStepAverage(event, events = strideEvents, count = strideAverageStepCount) {
      if (!event) return null;
      const limit = clamp(Math.round(Number(count) || 4), 1, 20);
      const eventTime = Number(event.session_s);
      const previousEvents = events
        .filter(candidate => Number(candidate.session_s) <= eventTime + 1e-6)
        .sort((a, b) => Number(a.session_s) - Number(b.session_s));
      const values = [];
      for (let index = previousEvents.length - 1; index >= 0 && values.length < limit; index -= 1) {
        const rawValue = previousEvents[index].valid_step_path_length_m;
        if (rawValue == null || rawValue === "") continue;
        const value = Number(rawValue);
        if (Number.isFinite(value)) values.push(value);
      }
      if (!values.length) return null;
      return values.reduce((sum, value) => sum + value, 0) / values.length;
    }

    function rollingStepLabel() {
      return `Avg Last ${strideAverageStepCount} ${strideAverageStepCount === 1 ? "Step" : "Steps"}`;
    }

    function getStrideEventsAround(targetTime = currentTime, radiusS = strideEventWindowS) {
      return strideEvents.filter(event => Math.abs(Number(event.session_s) - targetTime) <= radiusS);
    }

    function renderStrideTrack(nearestEvent) {
      strideTrack.innerHTML = "";
      if (!strideBouts.length) {
        strideTrack.classList.add("muted");
        return;
      }
      strideTrack.classList.remove("muted");
      strideBouts.forEach(bout => {
        const segment = document.createElement("div");
        segment.className = "stride-bout-segment";
        segment.style.left = `${(bout.bout_start_s / sessionDuration) * 100}%`;
        segment.style.width = `${Math.max(0.15, ((bout.bout_end_s - bout.bout_start_s) / sessionDuration) * 100)}%`;
        segment.style.background = labelBackground(bout.label);
        const segmentKind = Number(bout.detected_step_events || 0) > 0 ? "stride bout" : "review window";
        segment.title = `${bout.label} ${segmentKind} • ${fmtTime(bout.bout_start_s)} - ${fmtTime(bout.bout_end_s)} • ${bout.confidence || "unknown"} confidence`;
        strideTrack.appendChild(segment);
      });
      if (!strideEvents.length) {
        const currentLine = document.createElement("div");
        currentLine.className = "stride-current-line";
        currentLine.style.left = `${(currentTime / sessionDuration) * 100}%`;
        strideTrack.appendChild(currentLine);
        return;
      }
      const activeKey = getStrideEventKey(nearestEvent);
      let markerEvents = getStrideEventsAround(currentTime, Math.max(strideEventWindowS, 2.0));
      if (!markerEvents.length && nearestEvent) markerEvents = [nearestEvent];
      markerEvents.forEach(event => {
        const marker = document.createElement("button");
        marker.type = "button";
        const eventConfidence = event.event_confidence || event.bout_confidence || "unknown";
        const sourcePeakNote = Number.isFinite(Number(event.sway_peak_session_s)) ? ` • source peak ${fmtTime(event.sway_peak_session_s)}` : "";
        const rollingAverage = getRollingStepAverage(event);
        marker.className = `stride-event-marker ${strideConfidenceClass(eventConfidence)} ${getStrideEventKey(event) === activeKey ? "active" : ""}`;
        marker.style.left = `${(Number(event.session_s) / sessionDuration) * 100}%`;
        marker.style.background = strideSideClass(event.side_proxy) === "positive" ? "#2f78b7" : "#c94c4c";
        marker.title = `${strideFootSideLabel(event.estimated_foot_side)} @ ${fmtTime(event.session_s)}${sourcePeakNote} • ${strideSideLabel(event.side_proxy)} • foot ${strideConfidenceLabel(event.foot_side_confidence)} • event ${strideConfidenceLabel(eventConfidence)} • step ${fmtMetric(event.valid_step_path_length_m, 3, " m")} • last${strideAverageStepCount} ${fmtMetric(rollingAverage, 3, " m")}${event.foot_side_warning ? ` • ${event.foot_side_warning}` : ""}${event.event_warning ? ` • ${event.event_warning}` : ""}`;
        marker.addEventListener("click", clickEvent => {
          clickEvent.preventDefault();
          clickEvent.stopPropagation();
          seekSession(Number(event.session_s), { preservePlayback: false });
        });
        strideTrack.appendChild(marker);
      });
      const currentLine = document.createElement("div");
      currentLine.className = "stride-current-line";
      currentLine.style.left = `${(currentTime / sessionDuration) * 100}%`;
      strideTrack.appendChild(currentLine);
      strideTrack.onpointerdown = event => {
        if (event.target !== strideTrack) return;
        const rect = strideTrack.getBoundingClientRect();
        const ratio = clamp((Number(event.clientX) - rect.left) / Math.max(rect.width, 1), 0, 1);
        seekAndPlayFromUserGesture(ratio * sessionDuration);
      };
    }

    function renderStrideEventList(nearestEvent) {
      strideEventList.innerHTML = "";
      if (!strideEvents.length) {
        strideEventListNote.textContent = "No stride/step proxy events are available for this trial yet.";
        return;
      }
      let visibleEvents = getStrideEventsAround();
      if (!visibleEvents.length && nearestEvent) {
        visibleEvents = [nearestEvent];
        strideEventListNote.textContent = `No event inside ±${strideEventWindowS.toFixed(0)}s. Showing nearest detected event.`;
      } else {
        strideEventListNote.textContent = `Detected events inside ±${strideEventWindowS.toFixed(0)}s of current playback time. Click one to pause both videos there.`;
      }
      const activeKey = getStrideEventKey(nearestEvent);
      visibleEvents.slice(0, 16).forEach(event => {
        const button = document.createElement("button");
        button.type = "button";
        const eventConfidence = event.event_confidence || event.bout_confidence || "unknown";
        const sourcePeakText = Number.isFinite(Number(event.sway_peak_session_s)) ? `source peak ${fmtTime(event.sway_peak_session_s)}` : "midpoint step timing";
        const rollingAverage = getRollingStepAverage(event);
        button.className = `stride-event-button ${strideConfidenceClass(eventConfidence)} ${getStrideEventKey(event) === activeKey ? "active" : ""}`;
        button.title = event.event_warning || "";
        button.innerHTML = `
          <span class="stride-side-dot ${strideSideClass(event.side_proxy)}"></span>
          <span class="stride-event-main">
            <strong>${escapeHtml(strideFootSideLabel(event.estimated_foot_side))} @ <span class="mono">${fmtTime(event.session_s)}</span></strong>
            <span>${escapeHtml(strideSideLabel(event.side_proxy))} • event ${escapeHtml(strideConfidenceLabel(eventConfidence))} • foot ${escapeHtml(strideConfidenceLabel(event.foot_side_confidence))} • ${escapeHtml(sourcePeakText)}</span>
          </span>
          <span class="stride-event-metrics">
            <span>step ${fmtMetric(event.valid_step_path_length_m, 3, "m")}</span>
            <span>stride ${fmtMetric(event.valid_stride_path_length_m, 3, "m")}</span>
            <span>last${strideAverageStepCount} ${fmtMetric(rollingAverage, 3, "m")}</span>
          </span>
        `;
        button.addEventListener("click", () => seekSession(Number(event.session_s), { preservePlayback: false }));
        strideEventList.appendChild(button);
      });
    }

    function renderStrideReview() {
      const nearestEvent = getNearestStrideEvent();
      const activeBout = getActiveStrideBout();
      const nearestBout = getStrideBoutForEvent(nearestEvent);
      const prevEvent = getAdjacentStrideEvent(-1);
      const nextEvent = getAdjacentStrideEvent(1);
      prevStrideButton.disabled = !prevEvent;
      nextStrideButton.disabled = !nextEvent;
      nearestStrideButton.disabled = !nearestEvent;
      strideSourceNote.textContent = strideReview.source_note || "SLAM head-motion proxy; validate with video.";

      if (!strideEvents.length && !strideBouts.length) {
        strideSummaryGrid.innerHTML = `
          <div class="summary-card"><b>Status</b><span>No stride events for this trial</span></div>
          <div class="summary-card"><b>Reason</b><span>No usable walking/stair bout was found, or diagnostics have not been generated.</span></div>
        `;
        renderStrideTrack(null);
        renderStrideEventList(null);
        return;
      }

      const distanceToNearest = nearestEvent ? Math.abs(Number(nearestEvent.session_s) - currentTime) : null;
      const nearestText = nearestEvent
        ? `${strideFootSideLabel(nearestEvent.estimated_foot_side)} @ ${fmtTime(nearestEvent.session_s)}`
        : "N/A";
      const boutText = activeBout
        ? `${activeBout.label} • ${fmtTime(activeBout.bout_start_s)} - ${fmtTime(activeBout.bout_end_s)}`
        : "Outside detected locomotion bout";
      const confidenceText = nearestEvent?.event_confidence || activeBout?.confidence || nearestEvent?.bout_confidence || nearestBout?.confidence || "N/A";
      const cadenceValue = activeBout?.cadence_steps_per_s ?? nearestBout?.cadence_steps_per_s;
      const nearestRollingAverage = getRollingStepAverage(nearestEvent);
      strideSummaryGrid.innerHTML = `
        <div class="summary-card"><b>Nearest Step Proxy</b><span>${escapeHtml(nearestText)}</span></div>
        <div class="summary-card"><b>Time Difference</b><span class="mono">${distanceToNearest == null ? "N/A" : `${distanceToNearest.toFixed(2)}s`}</span></div>
        <div class="summary-card"><b>Step Proxy Length</b><span>${fmtMetric(nearestEvent?.valid_step_path_length_m, 3, " m")}</span></div>
        <div class="summary-card"><b>Stride Proxy Length</b><span>${fmtMetric(nearestEvent?.valid_stride_path_length_m, 3, " m")}</span></div>
        <div class="summary-card"><b>${escapeHtml(rollingStepLabel())}</b><span>${fmtMetric(nearestRollingAverage, 3, " m")}</span></div>
        <div class="summary-card"><b>Current Review Window</b><span>${escapeHtml(boutText)}</span></div>
        <div class="summary-card"><b>Nearest Event Confidence</b><span>${escapeHtml(confidenceText)}</span></div>
        <div class="summary-card"><b>Nearest Bout Cadence</b><span>${fmtMetric(cadenceValue, 2, " events/s")}</span></div>
      `;
      renderStrideTrack(nearestEvent);
      renderStrideEventList(nearestEvent);
    }

    function getContextBounds(centerTime = currentTime) {
      return {
        start: clamp(centerTime - transitionContextRadiusS, 0, sessionDuration),
        end: clamp(centerTime + transitionContextRadiusS, 0, sessionDuration),
      };
    }

    function getNearestTransitionCandidate(targetTime = currentTime) {
      if (!transitionCandidates.length) return null;
      let best = transitionCandidates[0];
      let bestDistance = Math.abs(Number(best.time_s) - targetTime);
      for (const candidate of transitionCandidates.slice(1)) {
        const distance = Math.abs(Number(candidate.time_s) - targetTime);
        if (distance < bestDistance) {
          best = candidate;
          bestDistance = distance;
        }
      }
      return best;
    }

    function getAdjacentTransitionCandidate(direction) {
      if (!transitionCandidates.length) return null;
      if (direction < 0) {
        for (let index = transitionCandidates.length - 1; index >= 0; index -= 1) {
          if (Number(transitionCandidates[index].time_s) < currentTime - 1e-6) return transitionCandidates[index];
        }
      } else {
        for (const candidate of transitionCandidates) {
          if (Number(candidate.time_s) > currentTime + 1e-6) return candidate;
        }
      }
      return null;
    }

    function renderTransitionSummary() {
      const bounds = getContextBounds();
      const centerFrame = Math.round(currentTime * sessionFps);
      const nearestTransition = getNearestTransitionCandidate();
      const prevTransition = getAdjacentTransitionCandidate(-1);
      const nextTransition = getAdjacentTransitionCandidate(1);
      const activeClip = getClipAtTime(currentTime);
      const nearestSummary = nearestTransition
        ? `${nearestTransition.summary_label} @ ${fmtTime(Number(nearestTransition.time_s))}`
        : "No transition candidates";
      const sourceSummary = nearestTransition
        ? String((nearestTransition.sources || []).join(" + ") || "model")
        : "—";
      transitionSummaryGrid.innerHTML = `
        <div class="summary-card"><b>Center Time</b><span class="mono">${fmtTime(currentTime)}</span></div>
        <div class="summary-card"><b>Center Frame</b><span class="mono">#${centerFrame}</span></div>
        <div class="summary-card"><b>Context Window</b><span class="mono">${fmtTime(bounds.start)} - ${fmtTime(bounds.end)}</span></div>
        <div class="summary-card"><b>Frame Rate</b><span>${sessionFps.toFixed(1)} fps</span></div>
        <div class="summary-card"><b>Nearest Transition</b><span>${escapeHtml(nearestSummary)}</span></div>
        <div class="summary-card"><b>Transition Source</b><span>${escapeHtml(sourceSummary)}</span></div>
        <div class="summary-card"><b>Current Model</b><span>${escapeHtml(activeClip.label)}</span></div>
        <div class="summary-card"><b>Current Spoken</b><span>${escapeHtml(activeClip.automatic_spoken_action || "Not detected")}</span></div>
      `;
      prevTransitionButton.disabled = !prevTransition;
      nextTransitionButton.disabled = !nextTransition;
    }

    function renderFrameStrip(containerEl, frames, viewKey) {
      containerEl.innerHTML = "";
      frames.forEach(frame => {
        const tile = document.createElement("div");
        tile.className = "frame-tile";
        const button = document.createElement("button");
        button.className = "frame-thumb-button";
        button.type = "button";
        button.title = `Jump to ${fmtTime(Number(frame.time_s))}`;
        button.innerHTML = `<img class="frame-thumb" src="${viewKey === "downward" ? frame.downward_url : frame.aria_url}" alt="${viewKey} frame ${frame.frame_idx}" />`;
        button.addEventListener("click", () => {
          seekSession(Number(frame.time_s), { preservePlayback: false });
        });
        const caption = document.createElement("div");
        caption.className = "frame-caption";
        caption.innerHTML = `
          <strong>${frame.offset_frames === 0 ? "Current" : `${frame.offset_frames > 0 ? "+" : ""}${frame.offset_frames} frame`}</strong>
          <span class="mono">${fmtTime(Number(frame.time_s))}</span>
          <span class="mono">#${frame.frame_idx}</span>
        `;
        tile.appendChild(button);
        tile.appendChild(caption);
        containerEl.appendChild(tile);
      });
      if (viewKey === "downward") {
        applyDownwardFrameStripRotation();
      }
      if (viewKey === "aria") {
        applyAriaFrameStripRotation();
      }
    }

    function applyDownwardFrameStripRotation() {
      const normalizedTurns = ((downwardRotationQuarterTurns % 4) + 4) % 4;
      const degrees = normalizedTurns * 90;
      const quarterTurn = normalizedTurns % 2 === 1;
      downwardFrameStrip.querySelectorAll(".frame-thumb").forEach((imgEl) => {
        imgEl.style.transform = `rotate(${degrees}deg)`;
        imgEl.classList.toggle("rotated-quarter-turn", quarterTurn);
      });
    }

    function applyAriaFrameStripRotation() {
      const normalizedTurns = ((ariaRotationQuarterTurns % 4) + 4) % 4;
      const degrees = normalizedTurns * 90;
      const quarterTurn = normalizedTurns % 2 === 1;
      ariaFrameStrip.querySelectorAll(".frame-thumb").forEach((imgEl) => {
        imgEl.style.transform = `rotate(${degrees}deg)`;
        imgEl.classList.toggle("rotated-quarter-turn", quarterTurn);
      });
    }

    async function requestFrameContext(force = false) {
      if (!ariaVideo.paused && !force) return;
      const preset = frameContextPresets[framePresetKey] || frameContextPresets.adjacent;
      const roundedTime = (Math.round(currentTime * sessionFps) / sessionFps).toFixed(3);
      const requestKey = `${roundedTime}|${framePresetKey}`;
      if (!force && requestKey === lastFrameContextCenterKey) return;
      lastFrameContextCenterKey = requestKey;
      const requestId = ++frameContextRequestId;
      const url = `${frameContextApiUrl}&center_time_s=${encodeURIComponent(roundedTime)}&offsets=${encodeURIComponent(preset.offsets.join(","))}`;
      try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        if (requestId !== frameContextRequestId) return;
        const frames = payload.frames || [];
        downwardFrameStripNote.textContent = `${preset.label} around ${fmtTime(Number(payload.center_time_s || currentTime))}`;
        ariaFrameStripNote.textContent = `${frames.length} synchronized frame${frames.length === 1 ? "" : "s"}`;
        renderFrameStrip(downwardFrameStrip, frames, "downward");
        renderFrameStrip(ariaFrameStrip, frames, "aria");
      } catch (error) {
        if (requestId !== frameContextRequestId) return;
        downwardFrameStripNote.textContent = `Frame context unavailable: ${error}`;
        ariaFrameStripNote.textContent = "";
        downwardFrameStrip.innerHTML = "";
        ariaFrameStrip.innerHTML = "";
      }
    }

    function refreshTransitionReview(forceFrames = false) {
      renderTransitionSummary();
      requestFrameContext(forceFrames);
    }

    function buildDisplaySegments(sourceClips, labelGetter) {
      const segments = [];
      let currentLabel = labelGetter(sourceClips[0]);
      let startClip = sourceClips[0];
      for (let index = 1; index <= sourceClips.length; index += 1) {
        const nextLabel = index < sourceClips.length ? labelGetter(sourceClips[index]) : null;
        if (index === sourceClips.length || nextLabel !== currentLabel) {
          const endClip = sourceClips[index - 1];
          segments.push({
            label: currentLabel,
            start_s: startClip.start_s,
            end_s: endClip.end_s,
            clip_start_idx: startClip.clip_idx,
            clip_end_idx: endClip.clip_idx,
          });
          if (index < sourceClips.length) {
            currentLabel = nextLabel;
            startClip = sourceClips[index];
          }
        }
      }
      return segments;
    }

    function getSelectedReviewRange() {
      return {
        start_s: clamp(Number(selectedRangeStartS) || 0, 0, sessionDuration),
        end_s: clamp(Number(selectedRangeEndS) || 0, 0, sessionDuration),
      };
    }

    function setSelectedReviewRange(firstTime, secondTime) {
      let start = clamp(Number(firstTime) || 0, 0, sessionDuration);
      let end = clamp(Number(secondTime) || 0, 0, sessionDuration);
      if (end < start) [start, end] = [end, start];
      if (end - start < MIN_SELECTION_DURATION_S && sessionDuration > 0) {
        if (start + MIN_SELECTION_DURATION_S <= sessionDuration) {
          end = start + MIN_SELECTION_DURATION_S;
        } else {
          start = Math.max(0, end - MIN_SELECTION_DURATION_S);
        }
      }
      selectedRangeStartS = start;
      selectedRangeEndS = end;
      selectionPlaybackEndS = null;
      return getSelectedReviewRange();
    }

    function setSelectedReviewStart(value) {
      const start = clamp(Number(value) || 0, 0, sessionDuration);
      let end = getSelectedReviewRange().end_s;
      if (end - start < MIN_SELECTION_DURATION_S) {
        end = Math.min(sessionDuration, start + MIN_SELECTION_DURATION_S);
      }
      return setSelectedReviewRange(start, end);
    }

    function setSelectedReviewEnd(value) {
      const end = clamp(Number(value) || 0, 0, sessionDuration);
      let start = getSelectedReviewRange().start_s;
      if (end - start < MIN_SELECTION_DURATION_S) {
        start = Math.max(0, end - MIN_SELECTION_DURATION_S);
      }
      return setSelectedReviewRange(start, end);
    }

    function getContinuousSegmentsInRange(range = getSelectedReviewRange()) {
      return continuousSegments.filter(segment => (
        Number(segment.end_s) > Number(range.start_s) + 0.0001
        && Number(segment.start_s) < Number(range.end_s) - 0.0001
      ));
    }

    function getDistinctPredictionSequence(segments) {
      const labels = [];
      segments.forEach(segment => {
        const label = normalizeActionLabel(segment.label) || "missing";
        if (labels[labels.length - 1] !== label) labels.push(label);
      });
      return labels;
    }

    function getManualTimelineDisplaySegments() {
      const ranges = (annotationState.sessionSegments || [])
        .map(segment => ({
          ...segment,
          label: normalizeActionLabel(segment.motion_label || segment.label),
          start_s: Number(segment.start_s),
          end_s: Number(segment.end_s),
        }))
        .filter(segment => (
          actionOptions.includes(segment.label)
          && Number.isFinite(segment.start_s)
          && Number.isFinite(segment.end_s)
          && segment.end_s > segment.start_s
        ));
      if (ranges.length) return ranges;
      return [{ label: "unlabeled", start_s: 0, end_s: sessionDuration }];
    }

    function buildTimelineTrack(trackEl, segments, activeRange, options = {}) {
      const signature = segments
        .map(segment => `${segment.label}:${Number(segment.start_s).toFixed(3)}:${Number(segment.end_s).toFixed(3)}`)
        .join("|");
      trackEl._timelineOptions = options;

      if (trackEl.dataset.segmentSignature !== signature) {
        trackEl.dataset.segmentSignature = signature;
        trackEl.innerHTML = "";
        segments.forEach(segment => {
          const el = document.createElement("div");
          const widthPercent = ((Number(segment.end_s) - Number(segment.start_s)) / sessionDuration) * 100;
          el.className = "timeline-segment";
          el.style.left = `${(Number(segment.start_s) / sessionDuration) * 100}%`;
          el.style.width = `${Math.max(0, widthPercent)}%`;
          el.style.background = labelBackground(segment.label);
          const boundaryText = segment.start_boundary_source
            ? ` • start: ${formatBoundarySource(segment.start_boundary_source)}`
            : "";
          el.title = `${segment.label || "unlabeled"} • ${fmtTime(segment.start_s)} - ${fmtTime(segment.end_s)}${boundaryText}`;
          trackEl.appendChild(el);
        });

        const selection = document.createElement("div");
        selection.className = "timeline-selection";
        trackEl.appendChild(selection);
        const playhead = document.createElement("div");
        playhead.className = "timeline-playhead";
        trackEl.appendChild(playhead);

        const resolveTimelineTarget = event => {
          const rect = trackEl.getBoundingClientRect();
          const clientX = Number(event?.clientX);
          if (!Number.isFinite(clientX) || rect.width <= 0) return currentTime;
          return clamp(((clientX - rect.left) / rect.width) * sessionDuration, 0, sessionDuration);
        };
        const previewTarget = event => {
          const target = resolveTimelineTarget(event);
          selectionPlaybackEndS = null;
          setSessionPositionImmediate(target);
          return target;
        };
        trackEl.onpointerdown = event => {
          event.preventDefault();
          trackEl._timelineDragging = true;
          trackEl._timelineResumePlayback = !ariaVideo.paused;
          trackEl._timelinePointerStartX = Number(event.clientX);
          trackEl._timelineSelectionAnchor = resolveTimelineTarget(event);
          trackEl._timelineDidRangeDrag = false;
          trackEl._timelinePreviousRange = getSelectedReviewRange();
          try { trackEl.setPointerCapture(event.pointerId); } catch (error) {}
          pauseMediaSilently();
          isScrubbing = true;
          previewTarget(event);
        };
        trackEl.onpointermove = event => {
          if (!trackEl._timelineDragging) return;
          event.preventDefault();
          const target = resolveTimelineTarget(event);
          const movedPixels = Math.abs(Number(event.clientX) - Number(trackEl._timelinePointerStartX));
          if (trackEl._timelineOptions?.rangeSelectable && movedPixels >= 3) {
            trackEl._timelineDidRangeDrag = true;
            isSelectingReviewRange = true;
            setSelectedReviewRange(trackEl._timelineSelectionAnchor, target);
          }
          previewTarget(event);
        };
        trackEl.onpointerup = event => {
          if (!trackEl._timelineDragging) return;
          event.preventDefault();
          const target = resolveTimelineTarget(event);
          const selectedByDrag = Boolean(
            trackEl._timelineOptions?.rangeSelectable && trackEl._timelineDidRangeDrag
          );
          if (selectedByDrag) setSelectedReviewRange(trackEl._timelineSelectionAnchor, target);
          const shouldResume = Boolean(trackEl._timelineResumePlayback);
          trackEl._timelineDragging = false;
          isScrubbing = false;
          try { trackEl.releasePointerCapture(event.pointerId); } catch (error) {}
          if (selectedByDrag) {
            const range = getSelectedReviewRange();
            const shouldRevealFocusedSelection = !hasSelectedReviewRange;
            isSelectingReviewRange = false;
            hasSelectedReviewRange = true;
            selectionPlaybackEndS = range.end_s;
            seekAndPlayFromUserGesture(range.start_s);
            if (shouldRevealFocusedSelection) {
              window.requestAnimationFrame(() => {
                periodPlayer.scrollIntoView({ behavior: "smooth", block: "start" });
              });
            }
          } else {
            isSelectingReviewRange = false;
            seekSession(target, { preservePlayback: shouldResume });
          }
        };
        trackEl.onpointercancel = () => {
          if (trackEl._timelineDidRangeDrag && trackEl._timelinePreviousRange) {
            setSelectedReviewRange(
              trackEl._timelinePreviousRange.start_s,
              trackEl._timelinePreviousRange.end_s,
            );
            refreshDynamicPanels();
          }
          trackEl._timelineDragging = false;
          isSelectingReviewRange = false;
          isScrubbing = false;
          refreshDynamicPanels();
        };
      }

      const selection = trackEl.querySelector(".timeline-selection");
      if (selection && activeRange) {
        selection.style.display = "block";
        selection.style.left = `${(Number(activeRange.start_s) / sessionDuration) * 100}%`;
        selection.style.width = `${((Number(activeRange.end_s) - Number(activeRange.start_s)) / sessionDuration) * 100}%`;
      } else if (selection) {
        selection.style.display = "none";
      }
      const playhead = trackEl.querySelector(".timeline-playhead");
      if (playhead) playhead.style.left = `${(currentTime / sessionDuration) * 100}%`;
    }

    function renderAllTimelines(activeClip) {
      const selectedRange = hasSelectedReviewRange || isSelectingReviewRange
        ? getSelectedReviewRange()
        : null;
      const hasContinuousPrediction = continuousSegments.length > 0;
      continuousTimelineRow.style.display = hasContinuousPrediction ? "grid" : "none";
      continuousPredictionNote.style.display = hasContinuousPrediction ? "flex" : "none";
      if (hasContinuousPrediction) {
        buildTimelineTrack(continuousTrack, continuousSegments, selectedRange, { rangeSelectable: true });
        continuousPredictionNote.innerHTML = `
          <strong>Continuous automatic classification</strong>
          <span>Drag freely across the classification row to set and immediately play a review range. The selection may include part of one motion or cross several motions.</span>
          <span>${continuousSegments.length} variable-length motion periods in this session.</span>
        `;
      }
      buildTimelineTrack(manualTrack, getManualTimelineDisplaySegments(), selectedRange);
    }

    function renderMotionColorLegend(target) {
      const displayOrder = ["sitting", "standing", "walking", "upstairs", "downstairs"];
      target.innerHTML = `
        <span class="motion-color-legend-title">Motion color key</span>
        ${displayOrder.map(action => `
          <span class="motion-color-legend-item">
            <span class="motion-color-swatch" style="background:${baseColors[action]}"></span>
            ${escapeHtml(action)}
          </span>
        `).join("")}
      `;
    }

    function renderTimeSpentBars() {
      timeSpentBars.innerHTML = "";
      const maxMinutes = Math.max(...timeSpentRecords.map(record => record.minutes), 0.01);
      timeSpentRecords.forEach(record => {
        const col = document.createElement("div");
        col.className = "bar-col";
        col.innerHTML = `
          <div class="muted mono">${record.minutes.toFixed(2)} min</div>
          <div class="bar" style="background:${record.color};height:${Math.max(14, 220 * (record.minutes / maxMinutes))}px;"></div>
          <strong>${escapeHtml(record.label)}</strong>
        `;
        timeSpentBars.appendChild(col);
      });
    }

    function renderTimeInformation() {
      if (!timeInformation.available) {
        timeInformationSection.style.display = "none";
        return;
      }
      const aria = timeInformation.aria || {};
      const downward = timeInformation.downward || {};
      const alignment = timeInformation.alignment || {};
      const drift = timeInformation.drift || {};
      const independentDownwardClock = Boolean(downward.is_independent_clock_value);
      const agreementError = Number(alignment.clock_agreement_error_s);
      const hasAgreementError = alignment.clock_agreement_error_s != null && Number.isFinite(agreementError);
      const agreementValue = hasAgreementError
        ? `${Math.abs(agreementError).toFixed(2)} s difference`
        : "Not independently testable";
      let driftValue = "Not measured";
      let driftNote = drift.summary || "No drift result is available.";
      if (drift.status === "measured") {
        const signedDrift = Number(drift.drift_s || 0);
        driftValue = `${signedDrift >= 0 ? "+" : ""}${signedDrift.toFixed(2)} s`;
        driftNote = `${Number(drift.shared_duration_s || 0) / 60 >= 1 ? (Number(drift.shared_duration_s || 0) / 60).toFixed(1) + " min" : "short overlap"} · ${drift.confidence || "review"} confidence`;
      }
      clockStatus.textContent = alignment.label || "Timing available";
      clockStatus.className = `clock-status ${String(alignment.status || "").replaceAll("_", "-")}`;
      timeInformationGrid.innerHTML = `
        <div class="time-information-item">
          <b>Recording Date</b>
          <span class="time-information-value">${escapeHtml(timeInformation.recording_date || "Unknown")}</span>
          <span class="time-information-note">Timezone: ${escapeHtml(timeInformation.timezone || "local")}</span>
        </div>
        <div class="time-information-item">
          <b>Aria Start</b>
          <span class="time-information-value mono">${escapeHtml(aria.start_minute_display || aria.start_display || "Unknown")}</span>
          <span class="time-information-note">Exact: ${escapeHtml(aria.start_display || "Unknown")} · VRS metadata</span>
        </div>
        <div class="time-information-item">
          <b>Downward Start</b>
          <span class="time-information-value mono">${escapeHtml(downward.start_minute_display || downward.start_display || "Unknown")}</span>
          <span class="time-information-note">Exact: ${escapeHtml(downward.start_display || "Unknown")} · ${independentDownwardClock ? "recorder clock" : "derived from audio"}</span>
        </div>
        <div class="time-information-item">
          <b>Recording Start Difference</b>
          <span class="time-information-value">${Math.abs(Number(alignment.audio_offset_s || 0)).toFixed(2)} s</span>
          <span class="time-information-note">${escapeHtml(alignment.start_order || "Audio-aligned start order unavailable")}</span>
        </div>
        <div class="time-information-item">
          <b>Clock vs Audio Agreement</b>
          <span class="time-information-value">${escapeHtml(agreementValue)}</span>
          <span class="time-information-note">${escapeHtml(alignment.detail || "")}</span>
        </div>
        <div class="time-information-item">
          <b>Long-Session Clock Drift</b>
          <span class="time-information-value">${escapeHtml(driftValue)}</span>
          <span class="time-information-note">${escapeHtml(driftNote)}</span>
        </div>
      `;
      const driftMethod = drift.status === "measured"
        ? `${drift.method}; ${drift.sample_count} windows of ${drift.window_duration_s}s.`
        : drift.summary || "Not measured.";
      timeInformationDetails.innerHTML = `
        <div><strong>Aria source:</strong> ${escapeHtml(aria.source || "VRS metadata")}. NTP enabled: ${aria.ntp_enabled ? "yes" : "no"}; external clock verified: ${aria.external_clock_verified ? "yes" : "no"}.</div>
        <div><strong>Downward source:</strong> ${escapeHtml(String(downward.start_source || "unknown").replaceAll("_", " "))}. ${escapeHtml(downward.timezone_note || "")}</div>
        <div><strong>Start alignment:</strong> ${escapeHtml(alignment.method || "audio matching")}; peak agreement ${alignment.peak_f1_score == null ? "N/A" : (100 * Number(alignment.peak_f1_score)).toFixed(1) + "%"}.</div>
        <div><strong>Drift method:</strong> ${escapeHtml(driftMethod)}</div>
        <div style="grid-column:1/-1"><strong>Interpretation limit:</strong> ${escapeHtml(timeInformation.verification_note || "")}</div>
      `;
    }

    function getManualCoveragePercent() {
      if (!sessionDuration) return 0;
      const coveredSeconds = (annotationState.sessionSegments || []).reduce((total, segment) => {
        const start = clamp(Number(segment.start_s) || 0, 0, sessionDuration);
        const end = clamp(Number(segment.end_s) || 0, 0, sessionDuration);
        return total + Math.max(0, end - start);
      }, 0);
      return clamp((coveredSeconds / sessionDuration) * 100, 0, 100);
    }

    function calculateContinuousManualAccuracy(manualSegments, modelSegments, duration) {
      let comparableSeconds = 0;
      let correctSeconds = 0;
      for (const manual of manualSegments) {
        const manualLabel = normalizeActionLabel(manual.motion_label || manual.label);
        if (!actionOptions.includes(manualLabel)) continue;
        for (const model of modelSegments) {
          const overlap = Math.max(0,
            Math.min(Number(manual.end_s), Number(model.end_s), duration)
            - Math.max(Number(manual.start_s), Number(model.start_s), 0));
          if (!Number.isFinite(overlap) || overlap <= 0) continue;
          comparableSeconds += overlap;
          if (manualLabel === normalizeActionLabel(model.label)) correctSeconds += overlap;
        }
      }
      return comparableSeconds ? correctSeconds / comparableSeconds : null;
    }

    function getCurrentContinuousReview(activeClip) {
      const segment = getContinuousSegmentAtTime(currentTime);
      const manualActions = getManualActionsAtTime(currentTime, activeClip);
      const manualLabel = manualActions.length ? formatActionDisplay(manualActions) : "Unlabeled";
      const agreement = segment && manualActions.length
        ? manualActions.includes(normalizeActionLabel(segment.label))
        : null;
      return {
        segment,
        manualLabel,
        agreement,
        agreementText: agreement == null ? "Needs manual label" : (agreement ? "Match" : "Mismatch"),
      };
    }

    function renderSelectedPredictionTrack(range, rangeSegments) {
      const start = Number(range.start_s);
      const end = Number(range.end_s);
      const duration = Math.max(MIN_SELECTION_DURATION_S, end - start);
      const signature = [start.toFixed(3), end.toFixed(3)]
        .concat(rangeSegments.map(segment => (
          `${segment.label}:${Number(segment.start_s).toFixed(3)}:${Number(segment.end_s).toFixed(3)}`
        )))
        .join("|");

      if (periodPredictionTrack.dataset.segmentSignature !== signature) {
        periodPredictionTrack.dataset.segmentSignature = signature;
        periodPredictionTrack.innerHTML = "";
        rangeSegments.forEach(segment => {
          const clippedStart = Math.max(start, Number(segment.start_s));
          const clippedEnd = Math.min(end, Number(segment.end_s));
          if (clippedEnd <= clippedStart) return;
          const widthPercent = ((clippedEnd - clippedStart) / duration) * 100;
          const el = document.createElement("div");
          el.className = "timeline-segment";
          el.style.left = `${((clippedStart - start) / duration) * 100}%`;
          el.style.width = `${widthPercent}%`;
          el.style.background = labelBackground(segment.label);
          el.title = `${segment.label} · ${fmtTime(clippedStart)} - ${fmtTime(clippedEnd)}`;
          periodPredictionTrack.appendChild(el);
        });
        const playhead = document.createElement("div");
        playhead.className = "timeline-playhead";
        periodPredictionTrack.appendChild(playhead);
        const tickRatios = [0, 0.25, 0.5, 0.75, 1];
        periodPredictionAxis.innerHTML = tickRatios
          .map(ratio => `<span>${fmtTime(start + duration * ratio)}</span>`)
          .join("");
      }

      const playhead = periodPredictionTrack.querySelector(".timeline-playhead");
      if (playhead) {
        const isInside = currentTime >= start && currentTime <= end;
        playhead.style.display = isInside ? "block" : "none";
        playhead.style.left = `${clamp(((currentTime - start) / duration) * 100, 0, 100)}%`;
      }
      periodPredictionReadout.textContent = rangeSegments.length
        ? `${rangeSegments.length} variable-length classification periods · hover a color block for its exact action and time`
        : "No automatic classification in this selection";
    }

    function renderPeriodPlayer(activeClip) {
      const review = getCurrentContinuousReview(activeClip);
      const range = getSelectedReviewRange();
      const start = Number(range.start_s);
      const end = Number(range.end_s);
      const duration = Math.max(0, end - start);
      if (!hasSelectedReviewRange || duration <= 0) {
        periodPlayer.hidden = true;
        sessionSelectionHighlight.style.display = "none";
        fullSessionRangeReadout.textContent = `${fmtTime(currentTime)} / ${fmtTime(sessionDuration)}`;
        sessionAxisPlayhead.style.left = `${(currentTime / sessionDuration) * 100}%`;
        return;
      }
      periodPlayer.hidden = false;
      const offset = clamp(currentTime - start, 0, duration);
      const isInsideRange = currentTime >= start && currentTime <= end;
      const rangeSegments = getContinuousSegmentsInRange(range);
      const predictionSequence = getDistinctPredictionSequence(rangeSegments);
      const sequenceText = predictionSequence.length
        ? predictionSequence.join(" → ")
        : "No automatic classification in selected range";
      const badgeText = predictionSequence.length === 0
        ? "No automatic classification"
        : (predictionSequence.length === 1 ? predictionSequence[0] : `${predictionSequence.length} motions`);

      periodTitle.textContent = "Custom time range";
      periodRangeText.textContent = `${fmtTime(start)} - ${fmtTime(end)} · ${fmtTime(duration)} duration`;
      periodLabelBadge.textContent = badgeText;
      periodLabelBadge.style.background = predictionSequence.length
        ? labelBackground(predictionSequence.join("|"))
        : baseColors.missing;
      renderSelectedPredictionTrack(range, rangeSegments);
      periodScrubber.max = String(Math.max(0.01, duration));
      periodScrubber.step = "0.01";
      periodScrubber.value = String(offset);
      periodCurrentText.textContent = isInsideRange ? fmtTime(offset) : "outside range";
      periodDurationText.textContent = fmtTime(duration);
      if (document.activeElement !== selectionStartInput) selectionStartInput.value = start.toFixed(2);
      if (document.activeElement !== selectionEndInput) selectionEndInput.value = end.toFixed(2);
      periodDetailGrid.innerHTML = `
        <div class="period-detail"><b>Classifications In Selection</b><span>${escapeHtml(sequenceText)}</span></div>
        <div class="period-detail"><b>Classification Periods Crossed</b><span>${rangeSegments.length}</span></div>
        <div class="period-detail"><b>Current Automatic Classification</b><span>${escapeHtml(review.segment?.label || "not available")}</span></div>
        <div class="period-detail"><b>Manual At Current Time</b><span>${escapeHtml(review.manualLabel)}</span></div>
        <div class="period-detail"><b>Automatic Classification / Manual</b><span>${escapeHtml(review.agreementText)}</span></div>
      `;
      playPeriodButton.textContent = !ariaVideo.paused && selectionPlaybackEndS != null
        ? "Pause Selection"
        : "Play Selection";
      sessionSelectionHighlight.style.display = "block";
      sessionSelectionHighlight.style.left = `${(start / sessionDuration) * 100}%`;
      sessionSelectionHighlight.style.width = `${(duration / sessionDuration) * 100}%`;
      fullSessionRangeReadout.textContent = `${fmtTime(currentTime)} / ${fmtTime(sessionDuration)} · selected ${fmtTime(start)}-${fmtTime(end)}`;
      sessionAxisPlayhead.style.left = `${(currentTime / sessionDuration) * 100}%`;
    }

    function renderSummary(activeClip) {
      const review = getCurrentContinuousReview(activeClip);
      const timeAccuracy = dashboardData.trial_type === "parkinson_simulated"
        ? calculateContinuousManualAccuracy(annotationState.sessionSegments || [], continuousSegments, sessionDuration)
        : continuousAccuracy.continuous_v1_accuracy;
      summaryGrid.innerHTML = `
        <div class="summary-card"><b>Total Session</b><span class="mono">${fmtTime(sessionDuration)}</span></div>
        <div class="summary-card"><b>Continuous Periods</b><span>${continuousSegments.length}</span></div>
        <div class="summary-card" title="Exact action agreement, weighted by duration where both manual labels and automatic classification exist."><b>Continuous Time Accuracy</b><span>${fmtAccuracy(timeAccuracy)}</span></div>
        <div class="summary-card"><b>Manual Coverage</b><span>${getManualCoveragePercent().toFixed(1)}%</span></div>
        <div class="summary-card"><b>Current Automatic Classification</b><span>${escapeHtml(review.segment?.label || "not available")}</span></div>
        <div class="summary-card"><b>Current Manual Label</b><span>${escapeHtml(review.manualLabel)}</span></div>
        <div class="summary-card"><b>Current Alignment</b><span>${escapeHtml(review.agreementText)}</span></div>
        <div class="summary-card"><b>Selected Review Range</b><span class="mono">${fmtTime(getSelectedReviewRange().start_s)} - ${fmtTime(getSelectedReviewRange().end_s)}</span></div>
      `;
    }

    function renderClipPanel(activeClip) {
      const review = getCurrentContinuousReview(activeClip);
      const segment = review.segment;
      const selection = getSelectedReviewRange();
      const selectionDuration = Math.max(0, selection.end_s - selection.start_s);
      const segmentStart = Number(segment?.start_s || 0);
      const segmentEnd = Number(segment?.end_s || 0);
      const boundaryText = segment
        ? `${formatBoundarySource(segment.start_boundary_source)} · ${String(segment.start_boundary_confidence || "not rated").replaceAll("_", " ")}`
        : "N/A";
      clipPanel.innerHTML = `
        <div class="metric-card"><b>Current Automatic Classification</b><span>${escapeHtml(segment?.label || "not available")}</span></div>
        <div class="metric-card"><b>Current Classification Period</b><span class="mono">${segment ? `${fmtTime(segmentStart)} - ${fmtTime(segmentEnd)}` : "N/A"}</span></div>
        <div class="metric-card"><b>Selected Review Range</b><span class="mono">${fmtTime(selection.start_s)} - ${fmtTime(selection.end_s)}</span></div>
        <div class="metric-card"><b>Selection Duration</b><span class="mono">${fmtTime(selectionDuration)}</span></div>
        <div class="metric-card"><b>Current Boundary Evidence</b><span>${escapeHtml(boundaryText)}</span></div>
        <div class="metric-card"><b>Manual Label Now</b><span>${escapeHtml(review.manualLabel)}</span></div>
        <div class="metric-card"><b>Automatic Classification / Manual</b><span>${escapeHtml(review.agreementText)}</span></div>
      `;
    }

    function renderAnnotationStatus(activeClip) {
      const review = getCurrentContinuousReview(activeClip);
      const source = dashboardData.manual_timeline_source === "saved_manual_session_labels"
        ? "Saved continuous manual ranges"
        : "Available reference labels";
      annotationStatus.innerHTML = `
        <div class="metric-card"><b>Label Source</b><span>${escapeHtml(source)}</span></div>
        <div class="metric-card"><b>Saved Ranges</b><span>${annotationState.sessionSegments.length}</span></div>
        <div class="metric-card"><b>Coverage</b><span>${getManualCoveragePercent().toFixed(1)}%</span></div>
        <div class="metric-card"><b>Manual Label Now</b><span>${escapeHtml(review.manualLabel)}</span></div>
        <div class="metric-card"><b>Continuous Agreement</b><span>${escapeHtml(review.agreementText)}</span></div>
      `;
    }

    function renderTransitionSelectors(count, selectedActions = []) {
      transitionActionsWrap.innerHTML = "";
      for (let index = 0; index < count; index += 1) {
        const label = document.createElement("label");
        label.className = "annotation-field";
        label.innerHTML = `<span>Motion ${index + 1}</span><select data-transition-index="${index}">${buildActionOptionsHtml(selectedActions[index] || "")}</select>`;
        transitionActionsWrap.appendChild(label);
      }
    }

    function updateAnnotationVisibility() {
      const mode = manualModeSelect.value;
      singleActionWrap.style.display = mode === "single" ? "grid" : "none";
      transitionCountWrap.style.display = mode === "transition" ? "grid" : "none";
      transitionActionsWrap.style.display = mode === "transition" ? "grid" : "none";
      if (mode === "transition") {
        const count = Number(transitionCountSelect.value || 2);
        const currentSelections = Array.from(transitionActionsWrap.querySelectorAll("select")).map(select => select.value);
        renderTransitionSelectors(count, currentSelections);
      }
    }

    function syncAnnotationInputsToClip(activeClip, force = false) {
      if (!activeClip) return;
      if (!force && annotationEditorState.dirty) {
        return;
      }
      const annotation = getManualAnnotationForClip(activeClip.clip_idx);
      manualModeSelect.value = annotation ? annotation.mode : "";
      if (annotation && annotation.mode === "single") {
        singleActionSelect.value = (annotation.actions && annotation.actions[0]) || actionOptions[0];
      } else {
        singleActionSelect.value = actionOptions[0];
      }
      const transitionCount = annotation && annotation.mode === "transition"
        ? Math.max(2, Math.min(5, (annotation.actions || []).length))
        : 2;
      transitionCountSelect.value = String(transitionCount);
      renderTransitionSelectors(transitionCount, annotation && annotation.mode === "transition" ? annotation.actions : []);
      updateAnnotationVisibility();
      annotationEditorState.clipIdx = activeClip.clip_idx;
      annotationEditorState.dirty = false;
    }

    function collectManualAnnotationInput() {
      const mode = manualModeSelect.value;
      if (!mode) return { mode: "", actions: [] };
      if (mode === "single") {
        return { mode: "single", actions: [singleActionSelect.value].filter(Boolean) };
      }
      const actions = Array.from(transitionActionsWrap.querySelectorAll("select"))
        .map(select => select.value)
        .filter(Boolean);
      return { mode: "transition", actions };
    }

    function clearAnnotationAutosaveTimer() {
      if (annotationEditorState.autosaveTimer) {
        clearTimeout(annotationEditorState.autosaveTimer);
        annotationEditorState.autosaveTimer = null;
      }
    }

    function bindAnnotationEditorToCurrentClip() {
      const activeClip = getClipAtTime(currentTime);
      if (!activeClip) return null;
      annotationEditorState.clipIdx = activeClip.clip_idx;
      return activeClip;
    }

    function markAnnotationEditorDirty() {
      const activeClip = bindAnnotationEditorToCurrentClip();
      if (!activeClip) return null;
      annotationEditorState.dirty = true;
      return activeClip;
    }

    function queueAnnotationAutosave() {
      if (!annotationState.apiAvailable) return;
      clearAnnotationAutosaveTimer();
      annotationEditorState.autosaveTimer = window.setTimeout(() => {
        saveManualAnnotationForCurrentClip(false, { autosave: true });
      }, 500);
    }

    function handleAnnotationInputChange() {
      pauseSession();
      const activeClip = markAnnotationEditorDirty();
      if (!activeClip) return;
      annotationServerNote.textContent = `Pending manual label for clip #${activeClip.clip_idx}...`;
      queueAnnotationAutosave();
    }

    async function loadAnnotationState() {
      const fetchJson = async url => {
        const response = await fetch(url, { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      };
      const [clipResult, sessionResult] = await Promise.allSettled([
        fetchJson(annotationApiUrl),
        fetchJson(sessionManualApiUrl),
      ]);

      if (clipResult.status === "fulfilled") {
        const payload = clipResult.value || {};
        annotationState.apiAvailable = true;
        annotationState.annotations = payload.annotations || {};
        annotationState.summary = payload.summary || null;
      } else {
        annotationState.apiAvailable = false;
        annotationState.annotations = {};
        annotationState.summary = null;
      }

      if (sessionResult.status === "fulfilled") {
        const payload = sessionResult.value || {};
        annotationState.sessionSegments = Array.isArray(payload.segments) && payload.segments.length
          ? payload.segments
          : embeddedManualSegments.slice();
      } else {
        annotationState.sessionSegments = embeddedManualSegments.slice();
      }

      if (annotationState.sessionSegments.length) {
        const rangeKind = dashboardData.manual_timeline_source === "saved_manual_session_labels"
          ? "saved continuous manual ranges"
          : "available reference ranges";
        annotationServerNote.textContent = `${annotationState.sessionSegments.length} ${rangeKind} are aligned to the Aria session clock.`;
      } else {
        annotationServerNote.textContent = "No continuous manual ranges are saved yet. Open Manual Labeling to create them.";
      }
    }

    async function saveManualAnnotationForCurrentClip(resetToAuto = false, options = {}) {
      const activeClip = annotationEditorState.clipIdx != null
        ? clips.find(clip => clip.clip_idx === annotationEditorState.clipIdx)
        : getClipAtTime(currentTime);
      if (!annotationState.apiAvailable || !activeClip) {
        annotationServerNote.textContent = "Manual labeling save is unavailable on this page.";
        return;
      }
      clearAnnotationAutosaveTimer();
      const payload = resetToAuto
        ? { trial_id: trialId, clip_idx: activeClip.clip_idx, mode: "", actions: [] }
        : (() => {
            const collected = collectManualAnnotationInput();
            return { trial_id: trialId, clip_idx: activeClip.clip_idx, mode: collected.mode, actions: collected.actions };
          })();
      try {
        annotationEditorState.saveInFlight = true;
        saveManualLabelButton.disabled = true;
        resetManualLabelButton.disabled = true;
        annotationServerNote.textContent = resetToAuto
          ? `Resetting clip #${activeClip.clip_idx} to auto...`
          : `Saving manual label for clip #${activeClip.clip_idx}...`;
        const response = await fetch(annotationSaveUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const updated = await response.json();
        annotationState.apiAvailable = true;
        annotationState.annotations = updated.annotations || {};
        annotationState.summary = updated.summary || null;
        annotationEditorState.clipIdx = activeClip.clip_idx;
        annotationEditorState.dirty = false;
        annotationServerNote.textContent = updated.summary && updated.summary.workbook_path
          ? `Saved to Excel: ${updated.summary.workbook_path}`
          : "Manual label saved.";
        refreshDynamicPanels();
      } catch (error) {
        annotationServerNote.textContent = `Save failed: ${error}`;
      } finally {
        annotationEditorState.saveInFlight = false;
        saveManualLabelButton.disabled = false;
        resetManualLabelButton.disabled = false;
      }
    }

    function updatePlayButton() {
      playPauseButton.textContent = ariaVideo.paused ? "Play Session" : "Pause Session";
      if (playPeriodButton) {
        playPeriodButton.textContent = !ariaVideo.paused && selectionPlaybackEndS != null
          ? "Pause Selection"
          : "Play Selection";
      }
    }

    function updateAudioButton() {
      const enabled = !ariaVideo.muted && Number(ariaVideo.volume) > 0;
      audioToggleButton.textContent = enabled ? "Audio On" : "Audio Off";
      audioToggleButton.setAttribute("aria-pressed", enabled ? "true" : "false");
      audioToggleButton.setAttribute("aria-label", enabled ? "Turn Aria audio off" : "Turn Aria audio on");
      audioPill.textContent = `${dashboardData.audio_master_label} · ${enabled ? "on" : "off"}`;
    }

    function setAudioEnabled(enabled, persist = true) {
      downwardVideo.defaultMuted = true;
      downwardVideo.muted = true;
      if (enabled && Number(ariaVideo.volume) <= 0) ariaVideo.volume = 1;
      ariaVideo.muted = !enabled;
      if (persist) {
        try { window.localStorage.setItem(AUDIO_STORAGE_KEY, enabled ? "true" : "false"); } catch (error) {}
      }
      updateAudioButton();
    }

    function restoreAudioPreference() {
      let enabled = true;
      try {
        const saved = window.localStorage.getItem(AUDIO_STORAGE_KEY);
        if (saved !== null) enabled = saved !== "false";
      } catch (error) {}
      setAudioEnabled(enabled, false);
    }

    function debugLog(label, payload = {}) {
      try {
        console.log("[pd-session]", label, JSON.stringify(payload));
      } catch (error) {}
    }

    function pauseMediaSilently() {
      try {
        ariaVideo.pause();
      } catch (error) {}
      try {
        downwardVideo.pause();
      } catch (error) {}
    }

    function syncFollower(force = false) {
      if (!Number.isFinite(ariaVideo.currentTime)) return;
      const target = clamp(ariaVideo.currentTime, 0, sessionDuration);
      const downwardMaxTime = Math.max(0, downwardSessionDuration - 0.05);
      if (target > downwardMaxTime) {
        downwardVideo.pause();
        downwardVideo.playbackRate = ariaVideo.playbackRate || 1;
        updateSyncStatus(0, "ended");
        return;
      }
      const delta = Number(downwardVideo.currentTime || 0) - target;
      const absDelta = Math.abs(delta);
      const baseRate = Number(ariaVideo.playbackRate || 1);
      if (force || !Number.isFinite(delta) || absDelta > HARD_SYNC_THRESHOLD_S) {
        try {
          downwardVideo.currentTime = clamp(target, 0, downwardMaxTime);
        } catch (error) {}
        downwardVideo.playbackRate = baseRate;
        updateSyncStatus(
          delta,
          Number.isFinite(delta) && absDelta <= SOFT_SYNC_THRESHOLD_S ? "good" : "adjusting",
          true,
        );
        return;
      }
      if (!downwardVideo.seeking && absDelta > SOFT_SYNC_THRESHOLD_S) {
        const rateCorrection = clamp(-delta * 0.45, -0.05, 0.05);
        downwardVideo.playbackRate = baseRate * (1 + rateCorrection);
        updateSyncStatus(delta, "adjusting");
      } else {
        downwardVideo.playbackRate = baseRate;
        updateSyncStatus(delta, "good");
      }
    }

    function updateSyncStatus(deltaS, state, force = false) {
      const now = performance.now();
      if (!force && now - lastSyncStatusUpdateMs < 180) return;
      lastSyncStatusUpdateMs = now;
      syncStatus.className = `sync-status ${state}`;
      if (state === "ended") {
        syncStatus.textContent = "Downward recording ended; Aria continues";
        return;
      }
      const deltaMs = Math.round((Number(deltaS) || 0) * 1000);
      syncStatus.textContent = state === "good"
        ? `Synchronized · downward ${deltaMs >= 0 ? "+" : ""}${deltaMs} ms`
        : `Correcting sync · downward ${deltaMs >= 0 ? "+" : ""}${deltaMs} ms`;
    }

    function syncAnimationFrame() {
      syncAnimationId = null;
      if (!ariaVideo.paused && !ariaVideo.ended) {
        syncFollower(false);
        syncAnimationId = window.requestAnimationFrame(syncAnimationFrame);
      }
    }

    function startSyncLoop() {
      if (syncAnimationId == null) {
        syncAnimationId = window.requestAnimationFrame(syncAnimationFrame);
      }
    }

    function stopSyncLoop() {
      if (syncAnimationId != null) {
        window.cancelAnimationFrame(syncAnimationId);
        syncAnimationId = null;
      }
      downwardVideo.playbackRate = ariaVideo.playbackRate || 1;
    }

    function waitForEventOnce(target, eventName, timeoutMs = 5000) {
      return new Promise((resolve, reject) => {
        let settled = false;
        const cleanup = () => {
          target.removeEventListener(eventName, onEvent);
          if (timer) clearTimeout(timer);
        };
        const onEvent = () => {
          if (settled) return;
          settled = true;
          cleanup();
          resolve();
        };
        const timer = timeoutMs > 0 ? setTimeout(() => {
          if (settled) return;
          settled = true;
          cleanup();
          reject(new Error(`Timed out waiting for ${eventName}`));
        }, timeoutMs) : null;
        target.addEventListener(eventName, onEvent, { once: true });
      });
    }

    async function ensureVideoReadyForSeek(videoEl) {
      if (videoEl.readyState >= 1) return;
      try {
        await waitForEventOnce(videoEl, "loadedmetadata", 8000);
      } catch (error) {}
    }

    async function seekVideoPrecisely(videoEl, targetSeconds) {
      const targetMax = videoEl === downwardVideo ? Math.max(0, downwardSessionDuration - 0.05) : sessionDuration;
      const target = clamp(Number(targetSeconds) || 0, 0, targetMax);
      debugLog("seekVideoPrecisely:start", { id: videoEl.id, target });
      await ensureVideoReadyForSeek(videoEl);
      if (Number.isFinite(videoEl.currentTime) && Math.abs((videoEl.currentTime || 0) - target) <= 0.12 && !videoEl.seeking) {
        debugLog("seekVideoPrecisely:skip", { id: videoEl.id, currentTime: videoEl.currentTime, target });
        return;
      }
      try {
        videoEl.currentTime = target;
      } catch (error) {
        debugLog("seekVideoPrecisely:set-failed", { id: videoEl.id, target, error: String(error) });
        return;
      }
      if (videoEl.seeking) {
        try {
          await waitForEventOnce(videoEl, "seeked", 5000);
        } catch (error) {}
      }
      if (Math.abs((videoEl.currentTime || 0) - target) > 0.12) {
        try {
          videoEl.currentTime = target;
        } catch (error) {}
        if (videoEl.seeking) {
          try {
            await waitForEventOnce(videoEl, "seeked", 5000);
          } catch (error) {}
        }
      }
      debugLog("seekVideoPrecisely:done", { id: videoEl.id, currentTime: videoEl.currentTime, target, seeking: videoEl.seeking });
    }

    function setPlaybackRate(rate) {
      try {
        ariaVideo.playbackRate = rate;
        downwardVideo.playbackRate = rate;
      } catch (error) {}
    }

    function applyDownwardRotation() {
      const normalizedTurns = ((downwardRotationQuarterTurns % 4) + 4) % 4;
      const degrees = normalizedTurns * 90;
      const quarterTurn = normalizedTurns % 2 === 1;
      downwardVideo.style.transform = `rotate(${degrees}deg)`;
      downwardVideo.classList.toggle("rotated-quarter-turn", quarterTurn);
      applyDownwardFrameStripRotation();
      rotateDownwardButton.textContent = degrees === 0
        ? "Rotate Downward 90°"
        : `Downward ${degrees}° → +90°`;
      try {
        window.localStorage.setItem(`pd-insighter-downward-rotation-v2:${trialId}`, String(normalizedTurns));
      } catch (error) {}
    }

    function applyAriaRotation() {
      const normalizedTurns = ((ariaRotationQuarterTurns % 4) + 4) % 4;
      const degrees = normalizedTurns * 90;
      const quarterTurn = normalizedTurns % 2 === 1;
      ariaVideo.style.transform = `rotate(${degrees}deg)`;
      ariaVideo.classList.toggle("rotated-quarter-turn", quarterTurn);
      applyAriaFrameStripRotation();
      rotateAriaButton.textContent = degrees === 0
        ? "Rotate Aria 90°"
        : `Aria ${degrees}° → +90°`;
      try {
        window.localStorage.setItem(`pd-insighter-aria-rotation-v2:${trialId}`, String(normalizedTurns));
      } catch (error) {}
    }

    function setVideoTimeImmediate(videoEl, targetSeconds) {
      const targetMax = videoEl === downwardVideo ? Math.max(0, downwardSessionDuration - 0.05) : sessionDuration;
      const target = clamp(Number(targetSeconds) || 0, 0, targetMax);
      try {
        if (typeof videoEl.fastSeek === "function") {
          videoEl.fastSeek(target);
        } else {
          videoEl.currentTime = target;
        }
      } catch (error) {
        try {
          videoEl.currentTime = target;
        } catch (innerError) {}
      }
    }

    function setSessionPositionImmediate(targetSeconds) {
      const nextTime = clamp(Number(targetSeconds) || 0, 0, sessionDuration);
      setVideoTimeImmediate(downwardVideo, nextTime);
      setVideoTimeImmediate(ariaVideo, nextTime);
      currentTime = nextTime;
      refreshDynamicPanels();
    }

    async function seekSession(targetSeconds, options = {}) {
      const nextTime = clamp(Number(targetSeconds) || 0, 0, sessionDuration);
      const config = typeof options === "boolean" ? { preservePlayback: options } : (options || {});
      const shouldResume = Boolean(config.autoplay) || (Boolean(config.preservePlayback) && !ariaVideo.paused);
      const requestId = ++seekRequestId;
      debugLog("seekSession:start", { targetSeconds, nextTime, shouldResume, requestId, currentTime });
      pauseMediaSilently();
      isProgrammaticSeeking = true;
      currentTime = nextTime;
      refreshDynamicPanels();
      await Promise.all([
        seekVideoPrecisely(ariaVideo, nextTime),
        seekVideoPrecisely(downwardVideo, nextTime),
      ]);
      if (requestId !== seekRequestId) {
        debugLog("seekSession:stale", { requestId, latest: seekRequestId });
        return;
      }
      isProgrammaticSeeking = false;
      const settledAriaTime = Number(ariaVideo.currentTime);
      const ariaReachedTarget = Number.isFinite(settledAriaTime)
        && Math.abs(settledAriaTime - nextTime) <= 0.5;
      currentTime = clamp(ariaReachedTarget ? settledAriaTime : nextTime, 0, sessionDuration);
      debugLog("seekSession:after-seek", { requestId, currentTime, ariaCurrent: ariaVideo.currentTime, downwardCurrent: downwardVideo.currentTime });
      if (ariaReachedTarget) {
        syncFollower(true);
      } else {
        syncStatus.className = "sync-status adjusting";
        syncStatus.textContent = "Video position queued while media loads";
      }
      refreshDynamicPanels();
      requestFrameContext(true);
      if (shouldResume) {
        await playSession();
      } else {
        updatePlayButton();
      }
    }

    function seekAndPlayFromUserGesture(targetSeconds) {
      const nextTime = clamp(Number(targetSeconds) || 0, 0, sessionDuration);
      const requestId = ++seekRequestId;
      debugLog("seekAndPlayFromUserGesture:start", { targetSeconds, nextTime, requestId });
      pauseMediaSilently();
      isProgrammaticSeeking = true;
      setSessionPositionImmediate(nextTime);
      syncFollower(true);
      const playPromises = [ariaVideo.play()];
      if (nextTime <= Math.max(0, downwardSessionDuration - 0.10)) {
        playPromises.push(downwardVideo.play());
      }
      Promise.allSettled(playPromises).then(() => {
        updatePlayButton();
      });
      Promise.allSettled([
        seekVideoPrecisely(ariaVideo, nextTime),
        seekVideoPrecisely(downwardVideo, nextTime),
      ]).then(() => {
        if (requestId !== seekRequestId) return;
        isProgrammaticSeeking = false;
        currentTime = clamp(
          Number.isFinite(ariaVideo.currentTime) ? ariaVideo.currentTime : nextTime,
          0,
          sessionDuration,
        );
        syncFollower(true);
        refreshDynamicPanels();
        requestFrameContext(true);
        updatePlayButton();
        debugLog("seekAndPlayFromUserGesture:settled", { requestId, currentTime, ariaCurrent: ariaVideo.currentTime, downwardCurrent: downwardVideo.currentTime });
      });
    }

    async function playSession() {
      debugLog("playSession:start", { ariaCurrent: ariaVideo.currentTime, downwardCurrent: downwardVideo.currentTime });
      syncFollower(true);
      try {
        const playPromises = [ariaVideo.play()];
        if ((ariaVideo.currentTime || currentTime || 0) <= Math.max(0, downwardSessionDuration - 0.10)) {
          playPromises.push(downwardVideo.play());
        } else {
          downwardVideo.pause();
        }
        await Promise.all(playPromises);
        syncFollower(true);
        startSyncLoop();
      } catch (error) {}
      debugLog("playSession:done", { ariaCurrent: ariaVideo.currentTime, downwardCurrent: downwardVideo.currentTime, ariaPaused: ariaVideo.paused, downwardPaused: downwardVideo.paused });
      updatePlayButton();
    }

    function pauseSession() {
      ariaVideo.pause();
      downwardVideo.pause();
      stopSyncLoop();
      updatePlayButton();
    }

    function refreshDynamicPanels() {
      const activeClip = getClipAtTime(currentTime);
      currentClipIdx = activeClip.clip_idx;
      timeReadout.textContent = `${fmtTime(currentTime)} / ${fmtTime(sessionDuration)}`;
      sessionScrubber.value = String(currentTime);
      renderPeriodPlayer(activeClip);
      renderSummary(activeClip);
      renderStrideReview();
      renderTransitionSummary();
      renderClipPanel(activeClip);
      renderAnnotationStatus(activeClip);
      syncAnnotationInputsToClip(activeClip);
      renderAllTimelines(activeClip);
    }

    function buildTrialJump() {
      jumpOptions.forEach(item => {
        const option = document.createElement("option");
        option.value = item.href;
        option.textContent = item.trial_id === "__selector__" ? "All Trials" : item.label;
        if (item.trial_id === trialId) option.selected = true;
        trialJumpSelect.appendChild(option);
      });
      trialJumpSelect.addEventListener("change", event => {
        const href = event.target.value;
        if (href) {
          window.location.href = href;
        }
      });
    }

    function wireStaticContent() {
      heroTitle.textContent = dashboardData.recording_stem;
      document.getElementById("simulationNotice").hidden = dashboardData.trial_type !== "parkinson_simulated";
      document.getElementById("simulationTools").hidden = dashboardData.trial_type !== "parkinson_simulated";
      document.getElementById("simulationClassificationStatus").textContent =
        `Automatic classification loaded: ${(dashboardData.continuous_model_segments || []).length} continuous periods. Saved method: Parkinson criteria, high smoothing.`;
      trialPill.textContent = dashboardData.trial_id;
      recordingPill.textContent = dashboardData.recording_stem;
      audioPill.textContent = dashboardData.audio_master_label;
      primaryViewTitle.textContent = dashboardData.primary_view_label;
      secondaryViewTitle.textContent = dashboardData.secondary_view_label;
      singleActionSelect.innerHTML = buildActionOptionsHtml(actionOptions[0]);
      renderMotionColorLegend(overviewMotionColorLegend);
      renderMotionColorLegend(focusedMotionColorLegend);
      downwardVideo.src = dashboardData.downward_session_video_uri;
      downwardVideo.defaultMuted = true;
      downwardVideo.muted = true;
      ariaVideo.src = dashboardData.aria_session_video_uri;
      restoreAudioPreference();
      try {
        downwardRotationQuarterTurns = Number(window.localStorage.getItem(`pd-insighter-downward-rotation-v2:${trialId}`) || 2) || 2;
      } catch (error) {
        downwardRotationQuarterTurns = 2;
      }
      try {
        ariaRotationQuarterTurns = Number(window.localStorage.getItem(`pd-insighter-aria-rotation-v2:${trialId}`) || 1) || 1;
      } catch (error) {
        ariaRotationQuarterTurns = 1;
      }
      applyDownwardRotation();
      applyAriaRotation();
      try {
        strideAverageStepCount = clamp(Math.round(Number(window.localStorage.getItem(`pd-insighter-stride-average-steps:${trialId}`) || 4)), 1, 20);
      } catch (error) {
        strideAverageStepCount = 4;
      }
      strideAverageStepsInput.value = String(strideAverageStepCount);
      sessionScrubber.max = String(sessionDuration);
      sessionScrubber.step = "0.01";
      selectionStartInput.max = String(sessionDuration);
      selectionEndInput.max = String(sessionDuration);
      fullSessionRangeReadout.textContent = `${fmtTime(0)} / ${fmtTime(sessionDuration)}`;
      buildTrialJump();
      renderTimeSpentBars();
      renderTimeInformation();
    }

    playPauseButton.addEventListener("click", () => {
      selectionPlaybackEndS = null;
      if (ariaVideo.paused) {
        playSession();
      } else {
        pauseSession();
      }
    });
    audioToggleButton.addEventListener("click", () => {
      setAudioEnabled(ariaVideo.muted || Number(ariaVideo.volume) <= 0);
    });
    restartButton.addEventListener("click", () => {
      selectionPlaybackEndS = null;
      seekSession(0, { preservePlayback: true });
    });
    backButton.addEventListener("click", () => {
      selectionPlaybackEndS = null;
      seekSession(currentTime - 2, { preservePlayback: true });
    });
    forwardButton.addEventListener("click", () => {
      selectionPlaybackEndS = null;
      seekSession(currentTime + 2, { preservePlayback: true });
    });
    periodStartButton.addEventListener("click", () => {
      selectionPlaybackEndS = null;
      seekSession(getSelectedReviewRange().start_s, { preservePlayback: false });
    });
    playPeriodButton.addEventListener("click", () => {
      const range = getSelectedReviewRange();
      if (range.end_s <= range.start_s) return;
      if (!ariaVideo.paused && selectionPlaybackEndS != null) {
        selectionPlaybackEndS = null;
        pauseSession();
        refreshDynamicPanels();
        return;
      }
      const start = Number(range.start_s);
      const end = Number(range.end_s);
      const target = currentTime < start || currentTime >= end - 0.04 ? start : currentTime;
      selectionPlaybackEndS = end;
      seekAndPlayFromUserGesture(target);
      refreshDynamicPanels();
    });
    clearSelectionButton.addEventListener("click", () => {
      selectionPlaybackEndS = null;
      hasSelectedReviewRange = false;
      isSelectingReviewRange = false;
      refreshDynamicPanels();
    });
    periodPredictionTrack.addEventListener("pointerdown", event => {
      if (!hasSelectedReviewRange) return;
      event.preventDefault();
      const range = getSelectedReviewRange();
      const rect = periodPredictionTrack.getBoundingClientRect();
      const ratio = clamp((Number(event.clientX) - rect.left) / Math.max(rect.width, 1), 0, 1);
      const target = Number(range.start_s) + ratio * (Number(range.end_s) - Number(range.start_s));
      const shouldResume = !ariaVideo.paused;
      const keepSelectionStop = selectionPlaybackEndS != null;
      selectionPlaybackEndS = keepSelectionStop ? Number(range.end_s) : null;
      seekSession(target, { preservePlayback: shouldResume });
    });
    setSelectionStartButton.addEventListener("click", () => {
      const range = setSelectedReviewStart(currentTime);
      seekSession(range.start_s, { preservePlayback: false });
    });
    setSelectionEndButton.addEventListener("click", () => {
      const range = setSelectedReviewEnd(currentTime);
      seekSession(range.end_s, { preservePlayback: false });
    });
    selectionStartInput.addEventListener("change", event => {
      const range = setSelectedReviewStart(event.target.value);
      seekSession(range.start_s, { preservePlayback: false });
    });
    selectionEndInput.addEventListener("change", event => {
      const range = setSelectedReviewEnd(event.target.value);
      seekSession(range.end_s, { preservePlayback: false });
    });
    speedSelect.addEventListener("change", event => {
      setPlaybackRate(Number(event.target.value || 1));
    });
    strideWindowSelect.addEventListener("change", event => {
      strideEventWindowS = Number(event.target.value || 2.0) || 2.0;
      renderStrideReview();
    });
    strideAverageStepsInput.addEventListener("input", event => {
      strideAverageStepCount = clamp(Math.round(Number(event.target.value || 4)), 1, 20);
      event.target.value = String(strideAverageStepCount);
      try { window.localStorage.setItem(`pd-insighter-stride-average-steps:${trialId}`, String(strideAverageStepCount)); } catch (error) {}
      renderStrideReview();
    });
    prevStrideButton.addEventListener("click", () => {
      const target = getAdjacentStrideEvent(-1);
      if (target) seekSession(Number(target.session_s), { preservePlayback: false });
    });
    nextStrideButton.addEventListener("click", () => {
      const target = getAdjacentStrideEvent(1);
      if (target) seekSession(Number(target.session_s), { preservePlayback: false });
    });
    nearestStrideButton.addEventListener("click", () => {
      const target = getNearestStrideEvent();
      if (target) seekSession(Number(target.session_s), { preservePlayback: false });
    });
    contextWindowSelect.addEventListener("change", event => {
      transitionContextRadiusS = Number(event.target.value || 1.0) || 1.0;
      renderTransitionSummary();
    });
    framePresetSelect.addEventListener("change", event => {
      framePresetKey = String(event.target.value || "adjacent");
      requestFrameContext(true);
    });
    prevTransitionButton.addEventListener("click", () => {
      const target = getAdjacentTransitionCandidate(-1);
      if (target) seekSession(Number(target.time_s), { preservePlayback: false });
    });
    nextTransitionButton.addEventListener("click", () => {
      const target = getAdjacentTransitionCandidate(1);
      if (target) seekSession(Number(target.time_s), { preservePlayback: false });
    });
    prevFrameButton.addEventListener("click", () => {
      seekSession(currentTime - frameDuration, { preservePlayback: false });
    });
    nextFrameButton.addEventListener("click", () => {
      seekSession(currentTime + frameDuration, { preservePlayback: false });
    });
    rotateDownwardButton.addEventListener("click", () => {
      downwardRotationQuarterTurns = (downwardRotationQuarterTurns + 1) % 4;
      applyDownwardRotation();
    });
    rotateAriaButton.addEventListener("click", () => {
      ariaRotationQuarterTurns = (ariaRotationQuarterTurns + 1) % 4;
      applyAriaRotation();
    });
    sessionScrubber.addEventListener("pointerdown", () => {
      sessionScrubberResumePlayback = !ariaVideo.paused;
      pauseMediaSilently();
      isScrubbing = true;
    });
    sessionScrubber.addEventListener("input", event => {
      if (!isScrubbing) {
        sessionScrubberResumePlayback = !ariaVideo.paused;
        pauseMediaSilently();
      }
      isScrubbing = true;
      selectionPlaybackEndS = null;
      setSessionPositionImmediate(clamp(Number(event.target.value || 0), 0, sessionDuration));
    });
    sessionScrubber.addEventListener("change", event => {
      isScrubbing = false;
      seekSession(Number(event.target.value || 0), { preservePlayback: sessionScrubberResumePlayback });
    });
    periodScrubber.addEventListener("pointerdown", () => {
      periodScrubberResumePlayback = !ariaVideo.paused;
      pauseMediaSilently();
      isScrubbing = true;
    });
    periodScrubber.addEventListener("input", event => {
      const range = getSelectedReviewRange();
      if (!isScrubbing) {
        periodScrubberResumePlayback = !ariaVideo.paused;
        pauseMediaSilently();
      }
      isScrubbing = true;
      selectionPlaybackEndS = null;
      const target = Number(range.start_s) + Number(event.target.value || 0);
      setSessionPositionImmediate(clamp(target, Number(range.start_s), Number(range.end_s)));
    });
    periodScrubber.addEventListener("change", event => {
      const range = getSelectedReviewRange();
      isScrubbing = false;
      const target = Number(range.start_s) + Number(event.target.value || 0);
      seekSession(
        clamp(target, Number(range.start_s), Number(range.end_s)),
        { preservePlayback: periodScrubberResumePlayback },
      );
    });
    manualModeSelect.addEventListener("change", () => {
      updateAnnotationVisibility();
      handleAnnotationInputChange();
    });
    transitionCountSelect.addEventListener("change", () => {
      updateAnnotationVisibility();
      handleAnnotationInputChange();
    });
    singleActionSelect.addEventListener("change", handleAnnotationInputChange);
    transitionActionsWrap.addEventListener("change", handleAnnotationInputChange);
    saveManualLabelButton.addEventListener("click", () => saveManualAnnotationForCurrentClip(false));
    resetManualLabelButton.addEventListener("click", () => saveManualAnnotationForCurrentClip(true));
    downwardVideo.addEventListener("click", () => {
      if (ariaVideo.paused) playSession();
      else pauseSession();
    });
    ariaVideo.addEventListener("click", () => {
      if (ariaVideo.paused) playSession();
      else pauseSession();
    });
    ariaVideo.addEventListener("play", () => {
      updatePlayButton();
      startSyncLoop();
    });
    ariaVideo.addEventListener("pause", () => {
      downwardVideo.pause();
      stopSyncLoop();
      updatePlayButton();
      requestFrameContext(true);
    });
    ariaVideo.addEventListener("timeupdate", () => {
      if (isScrubbing || isProgrammaticSeeking) return;
      currentTime = clamp(ariaVideo.currentTime || 0, 0, sessionDuration);
      if (selectionPlaybackEndS != null && currentTime >= Number(selectionPlaybackEndS) - 0.02) {
        const stopAt = Number(selectionPlaybackEndS);
        selectionPlaybackEndS = null;
        pauseSession();
        setSessionPositionImmediate(stopAt);
        requestFrameContext(true);
        return;
      }
      syncFollower(false);
      refreshDynamicPanels();
    });
    ariaVideo.addEventListener("seeking", () => {
      if (isScrubbing || isProgrammaticSeeking) return;
      currentTime = clamp(ariaVideo.currentTime || 0, 0, sessionDuration);
      syncFollower(true);
      refreshDynamicPanels();
    });
    ariaVideo.addEventListener("seeked", () => syncFollower(true));
    ariaVideo.addEventListener("waiting", () => syncFollower(true));
    ariaVideo.addEventListener("ratechange", () => {
      try {
        downwardVideo.playbackRate = ariaVideo.playbackRate || 1;
      } catch (error) {}
    });
    ariaVideo.addEventListener("volumechange", updateAudioButton);
    ariaVideo.addEventListener("ended", () => {
      pauseSession();
      seekSession(sessionDuration, { preservePlayback: false });
    });

    (async () => {
      wireStaticContent();
      await loadAnnotationState();
      setPlaybackRate(1);
      refreshDynamicPanels();
      requestFrameContext(true);
    })();
  </script>
</body>
</html>
"""
    return (
        template
        .replace("__TITLE__", title)
        .replace("__DATA__", data_json)
        .replace("__TRIAL_ID__", html.escape(str(payload["trial_id"])))
        .replace("__TRIAL_SELECTOR_HREF__", trial_selector_href)
        .replace("__MANUAL_LABELING_LINK__", manual_labeling_link_html)
        .replace("__MANUAL_LABELING_CALLOUT__", manual_labeling_callout_html)
        .replace("__STRIDE_REVIEW_LINK__", stride_review_link_html)
        .replace("__TRANSITION_REVIEW_LINK__", transition_review_link_html)
    )


def payload_with_signal_samples(
    payload: dict[str, Any],
    *,
    include_stride_samples: bool,
    include_transition_samples: bool,
) -> dict[str, Any]:
    """Avoid embedding large signal arrays in pages that never render them."""
    page_payload = dict(payload)
    stride_signal = dict(payload.get("stride_signal") or {})
    transition_review = dict(payload.get("transition_review") or {})
    if not include_stride_samples:
        stride_signal["samples"] = []
    if not include_transition_samples:
        transition_review["samples"] = []
    page_payload["stride_signal"] = stride_signal
    page_payload["transition_review"] = transition_review
    return page_payload


@environment_navigation
def build_stride_review_html(payload: dict[str, Any]) -> str:
    title = html.escape(f"{payload['trial_id']} - Stride Review")
    page_payload = payload_with_signal_samples(
        payload,
        include_stride_samples=True,
        include_transition_samples=False,
    )
    data_json = json.dumps(page_payload)
    full_session_href = html.escape(str(payload.get("full_session_href") or CANONICAL_TRIAL_ENTRY_NAME), quote=True)
    manual_labeling_href = html.escape(
        str(payload.get("manual_labeling_href") or MANUAL_LABELING_DASHBOARD_NAME),
        quote=True,
    )
    transition_review_href = payload.get("transition_review_href")
    transition_review_link_html = (
        f'<a class="nav-tab" href="{html.escape(str(transition_review_href), quote=True)}">Transitions</a>'
        if transition_review_href
        else ""
    )
    template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #f3f0e8;
      --card: rgba(255,255,255,0.92);
      --text: #1d211b;
      --muted: #687166;
      --border: rgba(29,33,27,0.10);
      --accent: #1f7a58;
      --accent-strong: #14523c;
      --walk: #d95f02;
      --upstairs: #1b9e77;
      --downstairs: #666666;
      --blue: #2f78b7;
      --red: #c94c4c;
      --shadow: 0 24px 60px rgba(29,33,27,0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(31,122,88,0.13), transparent 28%),
        radial-gradient(circle at bottom right, rgba(217,95,2,0.12), transparent 30%),
        linear-gradient(180deg, #fbfaf5 0%, var(--bg) 100%);
    }
    .page {
      max-width: 1580px;
      margin: 0 auto;
      padding: 22px;
      display: grid;
      gap: 18px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 26px;
      padding: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }
    .page-nav {
      position: sticky;
      top: 12px;
      z-index: 30;
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 12px 14px;
      border: 1px solid rgba(29,33,27,0.10);
      border-radius: 20px;
      background: rgba(255,255,255,0.94);
      box-shadow: 0 14px 36px rgba(29,33,27,0.10);
      backdrop-filter: blur(14px);
    }
    .nav-context {
      display: flex;
      align-items: center;
      gap: 11px;
      flex: 0 0 auto;
      min-width: 190px;
    }
    .nav-mark {
      width: 38px;
      height: 38px;
      border-radius: 13px;
      display: grid;
      place-items: center;
      color: #fff;
      background: linear-gradient(145deg, var(--accent), var(--accent-strong));
      box-shadow: 0 8px 18px rgba(20,82,60,0.20);
      font-size: 12px;
      font-weight: 900;
      letter-spacing: 0.04em;
    }
    .nav-location { display: grid; gap: 2px; min-width: 0; line-height: 1.1; }
    .nav-location strong { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 0.92rem; }
    .nav-location span { color: var(--muted); font-size: 0.76rem; font-weight: 800; }
    .nav-tabs {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 3px;
      flex: 1 1 auto;
      min-width: 0;
      padding: 4px;
      border: 1px solid rgba(29,33,27,0.10);
      border-radius: 16px;
      background: rgba(29,33,27,0.045);
    }
    .nav-tab {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 36px;
      padding: 0 12px;
      border-radius: 12px;
      color: var(--muted);
      text-decoration: none;
      white-space: nowrap;
      font-size: 13px;
      font-weight: 900;
    }
    .nav-tab:hover { color: var(--text); background: rgba(255,255,255,0.76); }
    .nav-tab.active {
      color: var(--accent-strong);
      background: #fff;
      box-shadow: 0 4px 12px rgba(29,33,27,0.08);
    }
    .nav-trial-switcher {
      display: flex;
      align-items: center;
      gap: 9px;
      flex: 0 1 340px;
      min-width: 250px;
    }
    .nav-trial-switcher > span {
      color: var(--muted);
      font-size: 0.7rem;
      font-weight: 900;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .nav-trial-select { width: 100%; min-width: 0; border-radius: 13px; }
    .topbar {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }
    .link-row {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .utility-link, .mini-button, select {
      border: 1px solid var(--border);
      border-radius: 999px;
      background: rgba(255,255,255,0.88);
      color: var(--text);
      padding: 9px 13px;
      font-weight: 800;
      text-decoration: none;
      cursor: pointer;
    }
    .mini-button.primary {
      border-color: transparent;
      color: white;
      background: linear-gradient(180deg, var(--accent), var(--accent-strong));
      box-shadow: 0 12px 24px rgba(31,122,88,0.18);
    }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: end;
    }
    .eyebrow {
      color: var(--accent-strong);
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      font-weight: 900;
    }
    h1, h2, h3, p { margin-top: 0; }
    h1 {
      margin-bottom: 8px;
      font-size: clamp(34px, 4vw, 64px);
      letter-spacing: -0.05em;
      line-height: 0.96;
    }
    h2 {
      margin-bottom: 6px;
      font-size: 22px;
      letter-spacing: -0.02em;
    }
    h3 {
      margin-bottom: 8px;
      font-size: 17px;
    }
    .muted {
      color: var(--muted);
      line-height: 1.45;
    }
    .mono {
      font-variant-numeric: tabular-nums;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
    }
    .summary-card {
      border-radius: 18px;
      background: rgba(255,255,255,0.74);
      border: 1px solid rgba(29,33,27,0.06);
      padding: 14px;
    }
    .summary-card b, label span {
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-size: 12px;
      font-weight: 900;
    }
    .summary-card span {
      display: block;
      font-size: 18px;
      line-height: 1.25;
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 18px;
      align-items: start;
    }
    .media-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 14px;
    }
    .view-card {
      display: grid;
      gap: 10px;
      padding: 14px;
      border-radius: 22px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(29,33,27,0.06);
    }
    .view-header {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .view-frame {
      min-height: 420px;
      border-radius: 18px;
      background: linear-gradient(180deg, #0d1014, #10161c);
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 10px;
    }
    .view-frame video {
      width: 100%;
      max-height: 680px;
      object-fit: contain;
      border-radius: 12px;
      background: #000;
      transform-origin: center center;
    }
    .view-frame video.rotated-quarter-turn {
      width: auto;
      height: 100%;
      max-width: 100%;
      max-height: 100%;
    }
    .control-panel {
      display: grid;
      gap: 12px;
      margin-top: 14px;
      padding: 16px;
      border-radius: 22px;
      background: rgba(255,255,255,0.74);
      border: 1px solid rgba(29,33,27,0.06);
    }
    .control-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    .actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .mini-button.active {
      border-color: rgba(31,122,88,0.34);
      background: rgba(236,250,243,0.96);
      color: var(--accent-strong);
      box-shadow: inset 0 0 0 1px rgba(31,122,88,0.08);
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    .signal-panel {
      display: grid;
      gap: 12px;
      margin-top: 16px;
      padding: 16px;
      border-radius: 24px;
      background: rgba(255,255,255,0.76);
      border: 1px solid rgba(29,33,27,0.07);
    }
    .signal-header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
      flex-wrap: wrap;
    }
    .signal-chart {
      min-height: 360px;
      border-radius: 22px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.96), rgba(255,255,255,0.72)),
        linear-gradient(90deg, rgba(31,122,88,0.04), rgba(217,95,2,0.04));
      border: 1px solid rgba(29,33,27,0.08);
      overflow: hidden;
      position: relative;
    }
    .signal-chart svg {
      display: block;
      width: 100%;
      height: 360px;
      cursor: crosshair;
    }
    .signal-legend {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      padding: 10px 12px;
      border-radius: 18px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(29,33,27,0.07);
    }
    .signal-legend-title {
      color: var(--muted);
      font-size: 12px;
      font-weight: 900;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      margin-right: 3px;
    }
    .signal-legend-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 5px 9px;
      background: rgba(255,255,255,0.82);
      border: 1px solid rgba(29,33,27,0.08);
      color: var(--text);
      font-size: 12px;
      font-weight: 800;
    }
    .signal-legend-swatch {
      width: 22px;
      height: 12px;
      border-radius: 999px;
      border: 1px solid rgba(29,33,27,0.14);
      box-shadow: inset 0 0 0 999px rgba(255,255,255,0.18);
    }
    .signal-note {
      font-size: 12px;
    }
    .bout-panel {
      display: grid;
      gap: 12px;
      position: sticky;
      top: 14px;
    }
    label {
      display: grid;
      gap: 6px;
    }
    select {
      width: 100%;
      border-radius: 16px;
      font-weight: 700;
    }
    input[type="number"] {
      width: 100%;
      box-sizing: border-box;
      border-radius: 16px;
      border: 1px solid rgba(29,33,27,0.12);
      background: rgba(255,255,255,0.94);
      color: var(--text);
      font-size: 15px;
      font-weight: 800;
      padding: 10px 12px;
    }
    .local-timeline {
      position: relative;
      height: 92px;
      border-radius: 24px;
      border: 1px solid rgba(29,33,27,0.08);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.78), rgba(255,255,255,0.50)),
        repeating-linear-gradient(90deg, rgba(29,33,27,0.06) 0 1px, transparent 1px 8.333%);
      overflow: visible;
    }
    .timeline-fill {
      position: absolute;
      top: 22px;
      left: 18px;
      right: 18px;
      height: 8px;
      border-radius: 999px;
      background: rgba(29,33,27,0.12);
    }
    .event-marker {
      position: absolute;
      top: 8px;
      width: 34px;
      height: 44px;
      transform: translateX(-50%);
      border-radius: 14px;
      border: 2px solid rgba(255,255,255,0.96);
      color: white;
      font-size: 11px;
      font-weight: 900;
      cursor: pointer;
      box-shadow: 0 8px 18px rgba(29,33,27,0.18);
    }
    .event-marker.positive { background: var(--blue); }
    .event-marker.negative { background: var(--red); }
    .event-marker.conf-medium { opacity: 0.78; }
    .event-marker.conf-low {
      opacity: 0.46;
      border-style: dashed;
      filter: saturate(0.72);
    }
    .event-marker.active {
      width: 46px;
      height: 54px;
      top: 3px;
      background: #111827;
      z-index: 8;
      box-shadow: 0 0 0 7px rgba(31,122,88,0.18), 0 14px 26px rgba(29,33,27,0.22);
    }
    .current-line {
      position: absolute;
      top: 0;
      bottom: 0;
      width: 2px;
      background: rgba(17,24,39,0.70);
      z-index: 7;
      pointer-events: none;
    }
    .timeline-labels {
      position: absolute;
      left: 18px;
      right: 18px;
      bottom: 12px;
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 12px;
    }
    .event-table {
      display: grid;
      gap: 8px;
      max-height: 440px;
      overflow: auto;
      padding-right: 4px;
    }
    .event-row {
      width: 100%;
      display: grid;
      grid-template-columns: 54px 1fr 1.25fr;
      gap: 10px;
      align-items: center;
      border: 1px solid rgba(29,33,27,0.08);
      border-radius: 18px;
      background: rgba(255,255,255,0.78);
      padding: 11px 12px;
      cursor: pointer;
      text-align: left;
      color: var(--text);
    }
    .event-row.active {
      border-color: rgba(31,122,88,0.42);
      background: rgba(236,250,243,0.96);
      box-shadow: 0 10px 22px rgba(31,122,88,0.11);
    }
    .event-row.conf-low {
      background: rgba(255,255,255,0.54);
      border-style: dashed;
    }
    .event-row.conf-medium {
      background: rgba(255,250,241,0.82);
    }
    .event-badge {
      display: inline-grid;
      place-items: center;
      width: 38px;
      height: 38px;
      border-radius: 14px;
      color: white;
      font-weight: 900;
    }
    .event-badge.positive { background: var(--blue); }
    .event-badge.negative { background: var(--red); }
    .event-main {
      display: grid;
      gap: 4px;
    }
    .event-main strong { font-size: 14px; }
    .event-main span, .event-metrics {
      color: var(--muted);
      font-size: 12px;
    }
    .confidence-pill {
      display: inline-flex;
      width: fit-content;
      align-items: center;
      border-radius: 999px;
      padding: 3px 7px;
      font-size: 11px;
      font-weight: 900;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      background: rgba(31,122,88,0.10);
      color: var(--accent-strong);
    }
    .confidence-pill.conf-medium {
      background: rgba(229,143,42,0.13);
      color: #8a4b05;
    }
    .confidence-pill.conf-low {
      background: rgba(201,76,76,0.12);
      color: #923232;
    }
    .event-metrics {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 6px;
      text-align: right;
    }
    .empty-state {
      padding: 24px;
      border-radius: 22px;
      background: rgba(255,255,255,0.72);
      border: 1px dashed rgba(29,33,27,0.16);
    }
    @media (max-width: 1180px) {
      .workspace, .media-grid, .hero {
        grid-template-columns: 1fr;
      }
      .bout-panel {
        position: static;
      }
    }
    @media (max-width: 760px) {
      .page { padding: 14px; }
      .card { padding: 16px; border-radius: 20px; }
      .page-nav { position: static; display: grid; align-items: stretch; }
      .nav-context { min-width: 0; }
      .nav-tabs { justify-content: flex-start; overflow-x: auto; }
      .nav-tab { flex: 0 0 auto; }
      .nav-trial-switcher { min-width: 0; max-width: none; }
      .view-frame { min-height: 280px; }
      .event-row { grid-template-columns: 44px 1fr; }
      .event-metrics { grid-column: 1 / -1; text-align: left; }
    }
  </style>
</head>
<body>
  <div class="page">
    <nav class="page-nav" aria-label="PD Insighter navigation">
      <div class="nav-context">
        <span class="nav-mark" aria-hidden="true">PD</span>
        <div class="nav-location">
          <strong>__TRIAL_ID__</strong>
          <span>Stride Review</span>
        </div>
      </div>
      <div class="nav-tabs" aria-label="Trial views">
        <a class="nav-tab" href="__FULL_SESSION_HREF__">Full Session</a>
        <a class="nav-tab" href="__MANUAL_LABELING_HREF__">Manual Labels</a>
        <span class="nav-tab active" aria-current="page">Stride</span>
        __TRANSITION_REVIEW_LINK__
      </div>
      <label class="nav-trial-switcher" for="trialJumpSelect">
        <span>Trial</span>
        <select id="trialJumpSelect" class="nav-trial-select" aria-label="Switch trial or return to all trials"></select>
      </label>
    </nav>

    <section class="card hero">
      <div>
        <div class="eyebrow">PD-Insighter Stride Review</div>
        <h1 id="heroTitle"></h1>
        <p class="muted">This page zooms into one locomotion bout at a time, so the detected step/stride markers are not compressed across the full session timeline.</p>
      </div>
      <div id="heroStats" class="summary-grid"></div>
    </section>

    <section id="emptyState" class="card empty-state" style="display:none;"></section>

    <section id="strideWorkspace" class="workspace">
      <main class="card">
        <div class="media-grid">
          <div class="view-card">
            <div class="view-header">
              <h3>Downward View</h3>
              <button id="rotateDownwardButton" class="mini-button" type="button">Rotate Downward 90°</button>
            </div>
            <div class="view-frame">
              <video id="downwardVideo" playsinline muted preload="metadata"></video>
            </div>
          </div>
          <div class="view-card">
            <div class="view-header">
              <h3>Aria View</h3>
              <button id="rotateAriaButton" class="mini-button" type="button">Rotate Aria 90°</button>
            </div>
            <div class="view-frame">
              <video id="ariaVideo" playsinline preload="metadata"></video>
            </div>
          </div>
        </div>

        <div class="control-panel">
          <div class="control-row">
            <div class="actions">
              <button id="playPauseButton" class="mini-button primary" type="button">Play</button>
              <button id="boutStartButton" class="mini-button" type="button">Bout Start</button>
              <button id="prevEventButton" class="mini-button" type="button">Previous Step</button>
              <button id="nextEventButton" class="mini-button" type="button">Next Step</button>
              <button id="backSmallButton" class="mini-button" type="button">-0.25s</button>
              <button id="forwardSmallButton" class="mini-button" type="button">+0.25s</button>
            </div>
            <div class="actions">
              <select id="speedSelect">
                <option value="0.25">0.25x</option>
                <option value="0.5" selected>0.5x</option>
                <option value="0.75">0.75x</option>
                <option value="1">1.0x</option>
              </select>
              <span id="timeReadout" class="mono muted"></span>
            </div>
          </div>
          <input id="sessionScrubber" type="range" min="0" max="1" step="0.01" value="0" />
        </div>

        <section class="signal-panel">
          <div class="signal-header">
            <div>
              <h2>Height / Speed + Stride Signal</h2>
              <p class="muted">Use this plot to compare raw SLAM height, smoothed speed, and detected stride-proxy points against the videos.</p>
            </div>
            <div class="actions">
              <button id="signalZoomWindowButton" class="mini-button active" type="button">Selected Window</button>
              <button id="signalZoomContextButton" class="mini-button" type="button">Nearby Context</button>
              <button id="signalZoomFullButton" class="mini-button" type="button">Full Session</button>
            </div>
          </div>
          <div id="signalChart" class="signal-chart"></div>
          <div id="signalChartLegend" class="signal-legend"></div>
          <div id="signalChartNote" class="muted signal-note"></div>
        </section>

        <section style="display:grid;gap:14px;margin-top:16px;">
          <div>
            <h2>Selected Bout Timeline</h2>
            <p class="muted">Markers are spaced only across the selected bout duration, not the entire session.</p>
          </div>
          <div id="localTimeline" class="local-timeline"></div>
          <div id="eventTable" class="event-table"></div>
        </section>
      </main>

      <aside class="card bout-panel">
        <label>
          <span>Locomotion Bout</span>
          <select id="boutSelect"></select>
        </label>
        <label>
          <span>Average Last N Steps</span>
          <input id="rollingStepCountInput" type="number" min="1" max="20" step="1" value="4" />
        </label>
        <div id="boutSummaryGrid" class="summary-grid"></div>
        <div class="summary-card">
          <b>Validation Reminder</b>
          <span class="muted">These are SLAM/head-motion step proxies. Use the downward video to confirm whether each marker matches a visible foot step. Last-N averages can include previous detected steps from earlier review windows.</span>
        </div>
        <div id="labelSummaryGrid" class="summary-grid"></div>
      </aside>
    </section>
  </div>

  <script>
    const dashboardData = __DATA__;
    const strideReview = dashboardData.stride_review || {};
    const strideEvents = (strideReview.events || [])
      .map(event => ({ ...event, session_s: Number(event.session_s), bout_id: Number(event.bout_id), event_index: Number(event.event_index) }))
      .filter(event => Number.isFinite(event.session_s))
      .sort((a, b) => a.session_s - b.session_s);
    const strideSignal = dashboardData.stride_signal || {};
    const signalSamples = (strideSignal.samples || [])
      .map(sample => ({ t: Number(sample.t), z: Number(sample.z), speed: Number(sample.speed) }))
      .filter(sample => Number.isFinite(sample.t) && Number.isFinite(sample.z) && Number.isFinite(sample.speed))
      .sort((a, b) => a.t - b.t);
    const signalActionSpans = (strideSignal.action_spans || [])
      .map(span => ({ ...span, start_s: Number(span.start_s), end_s: Number(span.end_s) }))
      .filter(span => Number.isFinite(span.start_s) && Number.isFinite(span.end_s) && span.end_s > span.start_s);
    const strideBouts = (strideReview.bouts || [])
      .map(bout => ({ ...bout, bout_id: Number(bout.bout_id), bout_start_s: Number(bout.bout_start_s), bout_end_s: Number(bout.bout_end_s) }))
      .filter(bout => Number.isFinite(bout.bout_start_s) && Number.isFinite(bout.bout_end_s))
      .sort((a, b) => a.bout_start_s - b.bout_start_s);
    const labelSummary = strideReview.label_summary || [];
    const sessionDuration = Number(dashboardData.session_duration_s || 0);
    const trialId = dashboardData.trial_id;
    const trialJumpOptions = dashboardData.trial_jump_options || [];

    const trialJumpSelect = document.getElementById("trialJumpSelect");
    const heroTitle = document.getElementById("heroTitle");
    const heroStats = document.getElementById("heroStats");
    const emptyState = document.getElementById("emptyState");
    const strideWorkspace = document.getElementById("strideWorkspace");
    const downwardVideo = document.getElementById("downwardVideo");
    const ariaVideo = document.getElementById("ariaVideo");
    const rotateDownwardButton = document.getElementById("rotateDownwardButton");
    const rotateAriaButton = document.getElementById("rotateAriaButton");
    const playPauseButton = document.getElementById("playPauseButton");
    const boutStartButton = document.getElementById("boutStartButton");
    const prevEventButton = document.getElementById("prevEventButton");
    const nextEventButton = document.getElementById("nextEventButton");
    const backSmallButton = document.getElementById("backSmallButton");
    const forwardSmallButton = document.getElementById("forwardSmallButton");
    const speedSelect = document.getElementById("speedSelect");
    const timeReadout = document.getElementById("timeReadout");
    const sessionScrubber = document.getElementById("sessionScrubber");
    const boutSelect = document.getElementById("boutSelect");
    const rollingStepCountInput = document.getElementById("rollingStepCountInput");
    const boutSummaryGrid = document.getElementById("boutSummaryGrid");
    const labelSummaryGrid = document.getElementById("labelSummaryGrid");
    const localTimeline = document.getElementById("localTimeline");
    const eventTable = document.getElementById("eventTable");
    const signalChart = document.getElementById("signalChart");
    const signalChartLegend = document.getElementById("signalChartLegend");
    const signalChartNote = document.getElementById("signalChartNote");
    const signalZoomWindowButton = document.getElementById("signalZoomWindowButton");
    const signalZoomContextButton = document.getElementById("signalZoomContextButton");
    const signalZoomFullButton = document.getElementById("signalZoomFullButton");

    let currentTime = 0;
    let activeBoutId = null;
    let activeEventKey = "";
    let isScrubbing = false;
    let isProgrammaticSeeking = false;
    let downwardRotationQuarterTurns = 2;
    let ariaRotationQuarterTurns = 1;
    let rollingStepCount = 4;
    let signalZoomMode = "window";

    function clamp(value, min, max) {
      return Math.min(Math.max(Number(value) || 0, min), max);
    }

    function fmtTime(seconds) {
      const s = Math.max(0, Number(seconds) || 0);
      const mins = Math.floor(s / 60);
      const secs = Math.floor(s % 60);
      const tenths = Math.floor((s - Math.floor(s)) * 10);
      return mins > 0 ? `${mins}:${String(secs).padStart(2, "0")}.${tenths}` : `${secs}.${tenths}s`;
    }

    function fmtMetric(value, digits = 3, suffix = "") {
      if (value == null || value === "") return "N/A";
      const number = Number(value);
      if (!Number.isFinite(number)) return "N/A";
      return `${number.toFixed(digits)}${suffix}`;
    }

    function finiteNumberOrNull(value) {
      if (value == null || value === "") return null;
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function eventKey(event) {
      return event ? `${event.bout_id}:${event.event_index}:${Number(event.session_s).toFixed(3)}` : "";
    }

    function sideClass(sideProxy) {
      return String(sideProxy || "").includes("positive") ? "positive" : "negative";
    }

    function sideLabel(sideProxy) {
      return String(sideProxy || "").includes("positive") ? "Lateral +" : "Lateral -";
    }

    function confidenceClass(confidence) {
      const text = String(confidence || "").toLowerCase();
      if (text === "review") return "conf-low";
      if (text === "low") return "conf-low";
      if (text === "medium") return "conf-medium";
      return "conf-high";
    }

    function confidenceLabel(confidence) {
      const text = String(confidence || "").toLowerCase();
      if (text === "review") return "Review only";
      if (text === "low") return "Low confidence";
      if (text === "medium") return "Medium confidence";
      if (text === "high") return "High confidence";
      return "Unknown confidence";
    }

    function footSideLabel(side) {
      const text = String(side || "").toLowerCase();
      if (text === "left") return "Est. Left Foot";
      if (text === "right") return "Est. Right Foot";
      return "Foot side unknown";
    }

    function labelColor(label) {
      if (label === "walking") return "var(--walk)";
      if (label === "upstairs") return "var(--upstairs)";
      if (label === "downstairs") return "var(--downstairs)";
      return "#94a3b8";
    }

    function reviewWindowLabel(bout) {
      if (!bout) return "Review Window";
      return Number(bout.detected_step_events || 0) > 0 && String(bout.confidence || "") !== "review"
        ? "Stride Bout"
        : "Review Window";
    }

    function getBout() {
      return strideBouts.find(bout => Number(bout.bout_id) === Number(activeBoutId)) || strideBouts[0] || null;
    }

    function getBoutEvents(bout = getBout()) {
      if (!bout) return [];
      return strideEvents.filter(event => Number(event.bout_id) === Number(bout.bout_id));
    }

    function getRollingStepAverage(event, events = strideEvents, count = rollingStepCount) {
      if (!event) return null;
      const limit = clamp(Math.round(Number(count) || 4), 1, 20);
      const eventTime = Number(event.session_s);
      const previousEvents = events
        .filter(candidate => Number(candidate.session_s) <= eventTime + 1e-6)
        .sort((a, b) => Number(a.session_s) - Number(b.session_s));
      const values = [];
      for (let index = previousEvents.length - 1; index >= 0 && values.length < limit; index -= 1) {
        const rawValue = previousEvents[index].valid_step_path_length_m;
        if (rawValue == null || rawValue === "") continue;
        const value = Number(rawValue);
        if (Number.isFinite(value)) values.push(value);
      }
      if (!values.length) return null;
      return values.reduce((sum, value) => sum + value, 0) / values.length;
    }

    function rollingStepLabel() {
      return `Avg Last ${rollingStepCount} ${rollingStepCount === 1 ? "Step" : "Steps"}`;
    }

    function scaleValue(value, domainMin, domainMax, rangeMin, rangeMax) {
      const span = domainMax - domainMin;
      if (!Number.isFinite(span) || Math.abs(span) < 1e-9) return (rangeMin + rangeMax) / 2;
      return rangeMin + ((value - domainMin) / span) * (rangeMax - rangeMin);
    }

    function paddedDomain(values, paddingFraction = 0.08, forceZeroMin = false) {
      const finiteValues = values.map(Number).filter(Number.isFinite);
      if (!finiteValues.length) return forceZeroMin ? [0, 1] : [-1, 1];
      let minValue = Math.min(...finiteValues);
      let maxValue = Math.max(...finiteValues);
      if (forceZeroMin) minValue = Math.min(0, minValue);
      if (Math.abs(maxValue - minValue) < 1e-9) {
        const pad = Math.max(0.05, Math.abs(maxValue) * 0.1);
        return [minValue - pad, maxValue + pad];
      }
      const pad = (maxValue - minValue) * paddingFraction;
      return [minValue - pad, maxValue + pad];
    }

    function linePath(points) {
      if (!points.length) return "";
      return points.map((point, index) => `${index === 0 ? "M" : "L"}${point[0].toFixed(2)},${point[1].toFixed(2)}`).join(" ");
    }

    function thinSignalSamples(rows, maxPoints = 6000) {
      if (rows.length <= maxPoints) return rows;
      const bucketCount = Math.max(1, Math.floor((maxPoints - 2) / 3));
      const bucketSize = Math.ceil((rows.length - 2) / bucketCount);
      const result = [rows[0]];
      for (let start = 1; start < rows.length - 1; start += bucketSize) {
        const bucket = rows.slice(start, Math.min(rows.length - 1, start + bucketSize));
        if (!bucket.length) continue;
        const candidates = [
          bucket.reduce((best, row) => Number(row.z) < Number(best.z) ? row : best),
          bucket.reduce((best, row) => Number(row.z) > Number(best.z) ? row : best),
          bucket.reduce((best, row) => Number(row.speed) > Number(best.speed) ? row : best),
        ];
        [...new Set(candidates)]
          .sort((a, b) => Number(a.t) - Number(b.t))
          .forEach(row => result.push(row));
      }
      result.push(rows.at(-1));
      return result;
    }

    function getSignalRange(bout = getBout()) {
      if (!bout || signalZoomMode === "full") {
        return { start: 0, end: sessionDuration || (signalSamples.at(-1)?.t || 1) };
      }
      const start = Number(bout.bout_start_s);
      const end = Number(bout.bout_end_s);
      const duration = Math.max(0.5, end - start);
      if (signalZoomMode === "context") {
        const pad = Math.max(8, duration * 1.5);
        return { start: clamp(start - pad, 0, sessionDuration), end: clamp(end + pad, 0, sessionDuration) };
      }
      const pad = Math.max(0.6, duration * 0.12);
      return { start: clamp(start - pad, 0, sessionDuration), end: clamp(end + pad, 0, sessionDuration) };
    }

    function renderSignalZoomButtons() {
      signalZoomWindowButton.classList.toggle("active", signalZoomMode === "window");
      signalZoomContextButton.classList.toggle("active", signalZoomMode === "context");
      signalZoomFullButton.classList.toggle("active", signalZoomMode === "full");
    }

    function setSignalZoomMode(mode) {
      signalZoomMode = mode;
      renderSignalZoomButtons();
      renderSignalChart();
    }

    function renderSignalLegend(visibleSpans = []) {
      const labelOrder = ["sitting", "standing", "walking", "upstairs", "downstairs"];
      const sourceSpans = visibleSpans.length ? visibleSpans : signalActionSpans;
      const labelMap = new Map();
      sourceSpans.forEach(span => {
        const key = String(span.label || "").trim().toLowerCase();
        if (!key || labelMap.has(key)) return;
        labelMap.set(key, {
          label: key,
          displayLabel: span.display_label || key.replace(/\\b\\w/g, char => char.toUpperCase()),
          color: span.color || "#94a3b8",
        });
      });
      const legendItems = [...labelMap.values()].sort((a, b) => {
        const aIndex = labelOrder.indexOf(a.label);
        const bIndex = labelOrder.indexOf(b.label);
        return (aIndex === -1 ? 99 : aIndex) - (bIndex === -1 ? 99 : bIndex)
          || a.displayLabel.localeCompare(b.displayLabel);
      });
      if (!legendItems.length) {
        signalChartLegend.innerHTML = `<span class="signal-legend-title">Shaded areas</span><span class="muted">No manual motion labels visible in this range.</span>`;
        return;
      }
      signalChartLegend.innerHTML = `
        <span class="signal-legend-title">Shaded areas = manual motion label</span>
        ${legendItems.map(item => `
          <span class="signal-legend-chip" title="Background area color for ${escapeHtml(item.displayLabel)}">
            <span class="signal-legend-swatch" style="background:${escapeHtml(item.color)};"></span>
            ${escapeHtml(item.displayLabel)}
          </span>
        `).join("")}
      `;
    }

    function renderSignalChart() {
      renderSignalZoomButtons();
      if (!signalSamples.length) {
        signalChart.innerHTML = `<div class="empty-state"><strong>Height/speed signal unavailable.</strong><p class="muted">The raw SLAM trajectory could not be loaded into this generated dashboard.</p></div>`;
        renderSignalLegend(signalActionSpans);
        signalChartNote.textContent = strideSignal.source_note || "";
        return;
      }

      const bout = getBout();
      const range = getSignalRange(bout);
      if (range.end <= range.start) range.end = range.start + 1;
      const visibleSamples = signalSamples.filter(sample => sample.t >= range.start && sample.t <= range.end);
      const plotSamples = thinSignalSamples(visibleSamples);
      const visibleEvents = strideEvents.filter(event => event.session_s >= range.start && event.session_s <= range.end);
      const visibleSpans = signalActionSpans.filter(span => span.end_s >= range.start && span.start_s <= range.end);
      renderSignalLegend(visibleSpans);
      if (visibleSamples.length < 2) {
        signalChart.innerHTML = `<div class="empty-state"><strong>Not enough samples in this range.</strong><p class="muted">Try Nearby Context or Full Session.</p></div>`;
        signalChartNote.textContent = `Showing ${fmtTime(range.start)}-${fmtTime(range.end)}.`;
        return;
      }

      const width = 1000;
      const height = 360;
      const margin = { left: 70, right: 74, top: 26, bottom: 46 };
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = 220;
      const strideTop = margin.top + plotHeight + 42;
      const strideHeight = 44;
      const xForTime = time => margin.left + scaleValue(time, range.start, range.end, 0, plotWidth);
      const [heightMin, heightMax] = paddedDomain(visibleSamples.map(sample => sample.z), 0.08);
      const [speedMin, speedMax] = paddedDomain(visibleSamples.map(sample => sample.speed), 0.10, true);
      const strideValues = visibleEvents
        .map(event => finiteNumberOrNull(event.valid_stride_path_length_m))
        .filter(value => value != null);
      const [strideMinRaw, strideMax] = paddedDomain(strideValues, 0.14, true);
      const strideMin = Math.min(0, strideMinRaw);
      const yHeight = z => margin.top + scaleValue(z, heightMax, heightMin, 0, plotHeight);
      const ySpeed = speed => margin.top + scaleValue(speed, speedMax, speedMin, 0, plotHeight);
      const yStride = value => strideTop + scaleValue(value, strideMax, strideMin, 0, strideHeight);
      const heightPath = linePath(plotSamples.map(sample => [xForTime(sample.t), yHeight(sample.z)]));
      const speedPath = linePath(plotSamples.map(sample => [xForTime(sample.t), ySpeed(sample.speed)]));

      const gridLines = Array.from({ length: 5 }, (_, index) => {
        const ratio = index / 4;
        const y = margin.top + ratio * plotHeight;
        const zLabel = (heightMax - ratio * (heightMax - heightMin)).toFixed(2);
        const speedLabel = (speedMax - ratio * (speedMax - speedMin)).toFixed(2);
        return `
          <line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="rgba(29,33,27,0.10)" />
          <text x="${margin.left - 10}" y="${y + 4}" text-anchor="end" font-size="12" fill="#4b5563">${zLabel}</text>
          <text x="${width - margin.right + 10}" y="${y + 4}" text-anchor="start" font-size="12" fill="#b45309">${speedLabel}</text>
        `;
      }).join("");

      const timeTicks = Array.from({ length: 5 }, (_, index) => {
        const time = range.start + (index / 4) * (range.end - range.start);
        const x = xForTime(time);
        return `
          <line x1="${x}" y1="${margin.top}" x2="${x}" y2="${strideTop + strideHeight}" stroke="rgba(29,33,27,0.055)" />
          <text x="${x}" y="${height - 18}" text-anchor="middle" font-size="12" fill="#687166">${fmtTime(time)}</text>
        `;
      }).join("");

      const actionRects = visibleSpans.map(span => {
        const x1 = xForTime(Math.max(span.start_s, range.start));
        const x2 = xForTime(Math.min(span.end_s, range.end));
        const widthPx = Math.max(0, x2 - x1);
        const label = widthPx > 56 ? `<text x="${x1 + 6}" y="${margin.top + 17}" font-size="11" font-weight="800" fill="rgba(29,33,27,0.55)">${escapeHtml(span.display_label || span.label)}</text>` : "";
        return `
          <rect x="${x1}" y="${margin.top}" width="${widthPx}" height="${strideTop + strideHeight - margin.top}" fill="${escapeHtml(span.color || "#94a3b8")}" opacity="0.13" />
          ${label}
        `;
      }).join("");

      const selectedBoutOutline = bout ? (() => {
        const x1 = xForTime(Math.max(Number(bout.bout_start_s), range.start));
        const x2 = xForTime(Math.min(Number(bout.bout_end_s), range.end));
        if (x2 <= x1) return "";
        return `<rect x="${x1}" y="${margin.top + 1}" width="${x2 - x1}" height="${strideTop + strideHeight - margin.top - 2}" fill="none" stroke="rgba(31,122,88,0.48)" stroke-width="2" stroke-dasharray="7 5" rx="8" />`;
      })() : "";

      const eventLines = visibleEvents.map(event => {
        const x = xForTime(event.session_s);
        const color = sideClass(event.side_proxy) === "positive" ? "#2f78b7" : "#c94c4c";
        const dash = confidenceClass(event.event_confidence || event.bout_confidence) === "conf-low" ? ' stroke-dasharray="4 4"' : "";
        return `
          <line x1="${x}" y1="${margin.top}" x2="${x}" y2="${strideTop + strideHeight}" stroke="${color}" stroke-width="1.5" opacity="0.34"${dash} />
          <circle cx="${x}" cy="${margin.top + 8}" r="3.3" fill="${color}" opacity="0.88">
            <title>${escapeHtml(`${footSideLabel(event.estimated_foot_side)} · ${fmtTime(event.session_s)} · ${confidenceLabel(event.event_confidence || event.bout_confidence)}`)}</title>
          </circle>
        `;
      }).join("");

      const stridePoints = visibleEvents.map(event => {
        const value = finiteNumberOrNull(event.valid_stride_path_length_m);
        if (value == null) return "";
        const x = xForTime(event.session_s);
        const y = yStride(value);
        const showLabel = signalZoomMode !== "full" || visibleEvents.length <= 18;
        const label = showLabel ? `<text x="${x}" y="${y - 9}" text-anchor="middle" font-size="11" font-weight="900" fill="#6d3aa0">${value.toFixed(2)}m</text>` : "";
        return `
          <circle cx="${x}" cy="${y}" r="5.4" fill="#7a4fb3" stroke="white" stroke-width="2">
            <title>${escapeHtml(`stride proxy ${value.toFixed(3)} m · ${fmtTime(event.session_s)} · step ${event.event_index}`)}</title>
          </circle>
          ${label}
        `;
      }).join("");

      const currentLine = currentTime >= range.start && currentTime <= range.end ? (() => {
        const x = xForTime(currentTime);
        return `
          <line x1="${x}" y1="${margin.top - 10}" x2="${x}" y2="${strideTop + strideHeight + 10}" stroke="#111827" stroke-width="2.2" opacity="0.75" />
          <text x="${x + 6}" y="${margin.top - 12}" font-size="12" font-weight="900" fill="#111827">now ${fmtTime(currentTime)}</text>
        `;
      })() : "";

      signalChart.innerHTML = `
        <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Interactive SLAM height, speed, and stride signal plot">
          <rect x="0" y="0" width="${width}" height="${height}" fill="rgba(255,255,255,0.54)" />
          ${actionRects}
          ${gridLines}
          ${timeTicks}
          ${selectedBoutOutline}
          <path d="${heightPath}" fill="none" stroke="#27313a" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />
          <path d="${speedPath}" fill="none" stroke="#d95f02" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" opacity="0.9" />
          ${eventLines}
          <line x1="${margin.left}" y1="${strideTop + strideHeight}" x2="${width - margin.right}" y2="${strideTop + strideHeight}" stroke="rgba(29,33,27,0.18)" />
          <text x="${margin.left}" y="${strideTop - 10}" font-size="12" font-weight="900" fill="#6d3aa0">purple points = valid stride proxy length at later step event</text>
          <text x="${margin.left - 10}" y="${margin.top - 8}" text-anchor="end" font-size="12" font-weight="900" fill="#27313a">height (m)</text>
          <text x="${width - margin.right + 10}" y="${margin.top - 8}" text-anchor="start" font-size="12" font-weight="900" fill="#d95f02">speed (m/s)</text>
          <text x="${margin.left - 10}" y="${strideTop + strideHeight + 4}" text-anchor="end" font-size="11" fill="#6d3aa0">0m</text>
          <text x="${margin.left - 10}" y="${strideTop + 4}" text-anchor="end" font-size="11" fill="#6d3aa0">${strideMax.toFixed(2)}m</text>
          ${stridePoints}
          ${currentLine}
        </svg>
      `;

      const svg = signalChart.querySelector("svg");
      if (svg) {
        svg.addEventListener("click", event => {
          const rect = svg.getBoundingClientRect();
          const viewX = ((event.clientX - rect.left) / Math.max(1, rect.width)) * width;
          if (viewX < margin.left || viewX > width - margin.right) return;
          const ratio = (viewX - margin.left) / plotWidth;
          seekTo(range.start + ratio * (range.end - range.start), { pause: true });
        });
      }

      const validStrideCount = strideValues.length;
      const zoomLabel = signalZoomMode === "full" ? "full session" : signalZoomMode === "context" ? "nearby context" : "selected window";
      signalChartNote.textContent = `${zoomLabel}: ${fmtTime(range.start)}-${fmtTime(range.end)} · ${visibleEvents.length} step markers · ${validStrideCount} valid stride-proxy values. Click the plot to seek the videos. ${strideSignal.source_note || ""}`;
    }

    function getNearestEvent(targetTime = currentTime, events = strideEvents) {
      if (!events.length) return null;
      let best = events[0];
      let bestDistance = Math.abs(best.session_s - targetTime);
      for (const event of events.slice(1)) {
        const distance = Math.abs(event.session_s - targetTime);
        if (distance < bestDistance) {
          best = event;
          bestDistance = distance;
        }
      }
      return best;
    }

    function getAdjacentEvent(direction) {
      const events = getBoutEvents();
      if (!events.length) return null;
      if (direction < 0) {
        for (let index = events.length - 1; index >= 0; index -= 1) {
          if (events[index].session_s < currentTime - 1e-6) return events[index];
        }
      } else {
        for (const event of events) {
          if (event.session_s > currentTime + 1e-6) return event;
        }
      }
      return null;
    }

    function maybeSwitchBoutForTime() {
      const currentBout = getBout();
      if (currentBout && currentTime >= currentBout.bout_start_s && currentTime <= currentBout.bout_end_s) return;
      const containingBout = strideBouts.find(bout => currentTime >= bout.bout_start_s && currentTime <= bout.bout_end_s);
      if (containingBout && Number(containingBout.bout_id) !== Number(activeBoutId)) {
        activeBoutId = Number(containingBout.bout_id);
        boutSelect.value = String(activeBoutId);
      }
    }

    function applyRotation(videoEl, turns) {
      const normalized = ((turns % 4) + 4) % 4;
      const degrees = normalized * 90;
      videoEl.style.transform = `rotate(${degrees}deg)`;
      videoEl.classList.toggle("rotated-quarter-turn", normalized % 2 === 1);
      return degrees;
    }

    function applyDownwardRotation() {
      const degrees = applyRotation(downwardVideo, downwardRotationQuarterTurns);
      rotateDownwardButton.textContent = degrees === 0 ? "Rotate Downward 90°" : `Downward ${degrees}° → +90°`;
      try { window.localStorage.setItem(`pd-insighter-downward-rotation-v2:${trialId}`, String(downwardRotationQuarterTurns)); } catch (error) {}
    }

    function applyAriaRotation() {
      const degrees = applyRotation(ariaVideo, ariaRotationQuarterTurns);
      rotateAriaButton.textContent = degrees === 0 ? "Rotate Aria 90°" : `Aria ${degrees}° → +90°`;
      try { window.localStorage.setItem(`pd-insighter-aria-rotation-v2:${trialId}`, String(ariaRotationQuarterTurns)); } catch (error) {}
    }

    function syncFollower(force = false) {
      const target = clamp(ariaVideo.currentTime || currentTime, 0, sessionDuration);
      if (force || Math.abs((downwardVideo.currentTime || 0) - target) > 0.10) {
        downwardVideo.currentTime = target;
      }
    }

    function setPlaybackRate(rate) {
      ariaVideo.playbackRate = rate;
      downwardVideo.playbackRate = rate;
    }

    function pauseBoth() {
      ariaVideo.pause();
      downwardVideo.pause();
      playPauseButton.textContent = "Play";
    }

    async function playBoth() {
      syncFollower(true);
      try {
        await Promise.all([ariaVideo.play(), downwardVideo.play()]);
      } catch (error) {}
      playPauseButton.textContent = ariaVideo.paused ? "Play" : "Pause";
    }

    function seekTo(seconds, { pause = true } = {}) {
      const target = clamp(seconds, 0, sessionDuration);
      isProgrammaticSeeking = true;
      if (pause) pauseBoth();
      currentTime = target;
      ariaVideo.currentTime = target;
      downwardVideo.currentTime = target;
      maybeSwitchBoutForTime();
      render();
      window.setTimeout(() => { isProgrammaticSeeking = false; syncFollower(true); render(); }, 120);
    }

    function renderHero() {
      heroTitle.textContent = `${dashboardData.trial_id} Stride Review`;
      const usableBouts = strideBouts.filter(bout => ["high", "medium"].includes(String(bout.confidence)));
      heroStats.innerHTML = `
        <div class="summary-card"><b>Detected Events</b><span>${strideEvents.length}</span></div>
        <div class="summary-card"><b>Review Windows</b><span>${strideBouts.length}</span></div>
        <div class="summary-card"><b>High/Medium Bouts</b><span>${usableBouts.length}</span></div>
      `;
    }

    function renderBoutSelect() {
      boutSelect.innerHTML = "";
      strideBouts.forEach(bout => {
        const option = document.createElement("option");
        option.value = String(bout.bout_id);
        option.textContent = `${reviewWindowLabel(bout)} ${bout.bout_id}: ${bout.label} ${fmtTime(bout.bout_start_s)}-${fmtTime(bout.bout_end_s)} (${confidenceLabel(bout.confidence || "unknown")})`;
        boutSelect.appendChild(option);
      });
      if (activeBoutId != null) boutSelect.value = String(activeBoutId);
    }

    function renderSummary() {
      const bout = getBout();
      const events = getBoutEvents(bout);
      const nearest = getNearestEvent(currentTime, events);
      activeEventKey = eventKey(nearest);
      if (!bout) return;
      const nearestRollingAverage = getRollingStepAverage(nearest);
      boutSummaryGrid.innerHTML = `
        <div class="summary-card"><b>Selected ${escapeHtml(reviewWindowLabel(bout))}</b><span>${escapeHtml(bout.label)} · ${fmtTime(bout.bout_start_s)}-${fmtTime(bout.bout_end_s)}</span></div>
        <div class="summary-card"><b>Confidence</b><span>${escapeHtml(confidenceLabel(bout.confidence || "N/A"))}</span></div>
        <div class="summary-card"><b>Median Step Proxy</b><span>${fmtMetric(bout.median_step_path_length_m, 3, " m")}</span></div>
        <div class="summary-card"><b>Median Stride Proxy</b><span>${fmtMetric(bout.median_stride_path_length_m, 3, " m")}</span></div>
        <div class="summary-card"><b>Cadence</b><span>${fmtMetric(bout.cadence_steps_per_s, 2, " events/s")}</span></div>
        <div class="summary-card"><b>Detected Events In Window</b><span>${events.length}</span></div>
        <div class="summary-card"><b>Current Event</b><span>${nearest ? `${sideLabel(nearest.side_proxy)} @ ${fmtTime(nearest.session_s)}` : "N/A"}</span></div>
        <div class="summary-card"><b>${escapeHtml(rollingStepLabel())}</b><span>${fmtMetric(nearestRollingAverage, 3, " m")}</span></div>
      `;
      timeReadout.textContent = `${fmtTime(currentTime)} / ${fmtTime(sessionDuration)}`;
      sessionScrubber.value = String(currentTime);
      prevEventButton.disabled = !getAdjacentEvent(-1);
      nextEventButton.disabled = !getAdjacentEvent(1);
    }

    function renderLabelSummary() {
      labelSummaryGrid.innerHTML = "";
      labelSummary.forEach(row => {
        const card = document.createElement("div");
        card.className = "summary-card";
        card.innerHTML = `
          <b>${escapeHtml(row.label || "label")}</b>
          <span>${fmtMetric(row.median_step_proxy_m, 3, " m")} step</span>
          <span class="muted">${fmtMetric(row.median_stride_proxy_m, 3, " m")} stride · ${row.usable_bouts || 0} bouts</span>
        `;
        labelSummaryGrid.appendChild(card);
      });
    }

    function renderTimeline() {
      const bout = getBout();
      const events = getBoutEvents(bout);
      localTimeline.innerHTML = "";
      if (!bout) return;
      const fill = document.createElement("div");
      fill.className = "timeline-fill";
      localTimeline.appendChild(fill);
      const duration = Math.max(0.001, bout.bout_end_s - bout.bout_start_s);
      events.forEach(event => {
        const marker = document.createElement("button");
        marker.type = "button";
        const sourcePeakNote = Number.isFinite(Number(event.sway_peak_session_s)) ? ` · source peak ${fmtTime(event.sway_peak_session_s)}` : "";
        const rollingAverage = getRollingStepAverage(event);
        marker.className = `event-marker ${sideClass(event.side_proxy)} ${confidenceClass(event.event_confidence || event.bout_confidence)} ${eventKey(event) === activeEventKey ? "active" : ""}`;
        marker.style.left = `${clamp(((event.session_s - bout.bout_start_s) / duration) * 100, 0, 100)}%`;
        marker.textContent = String(event.event_index);
        marker.title = `${footSideLabel(event.estimated_foot_side)} @ ${fmtTime(event.session_s)}${sourcePeakNote} · ${sideLabel(event.side_proxy)} · foot ${confidenceLabel(event.foot_side_confidence)} · event ${confidenceLabel(event.event_confidence || event.bout_confidence)} · step ${fmtMetric(event.valid_step_path_length_m, 3, " m")} · last${rollingStepCount} ${fmtMetric(rollingAverage, 3, " m")}${event.foot_side_warning ? ` · ${event.foot_side_warning}` : ""}${event.event_warning ? ` · ${event.event_warning}` : ""}`;
        marker.addEventListener("click", () => seekTo(event.session_s, { pause: true }));
        localTimeline.appendChild(marker);
      });
      const currentLine = document.createElement("div");
      currentLine.className = "current-line";
      currentLine.style.left = `${clamp(((currentTime - bout.bout_start_s) / duration) * 100, 0, 100)}%`;
      localTimeline.appendChild(currentLine);
      const labels = document.createElement("div");
      labels.className = "timeline-labels";
      labels.innerHTML = `<span>${fmtTime(bout.bout_start_s)}</span><span>${fmtTime(bout.bout_end_s)}</span>`;
      localTimeline.appendChild(labels);
    }

    function renderEventTable() {
      const events = getBoutEvents();
      eventTable.innerHTML = "";
      if (!events.length) {
        const row = document.createElement("div");
        row.className = "empty-state";
        row.innerHTML = `<strong>No step markers detected in this review window.</strong><p class="muted">Use the synchronized videos and scrubber to manually check whether a step was missed.</p>`;
        eventTable.appendChild(row);
        return;
      }
      events.forEach(event => {
        const row = document.createElement("button");
        row.type = "button";
        const eventConfidence = event.event_confidence || event.bout_confidence || "unknown";
        const sourcePeakText = Number.isFinite(Number(event.sway_peak_session_s)) ? `source peak ${fmtTime(event.sway_peak_session_s)}` : "midpoint step timing";
        const rollingAverage = getRollingStepAverage(event);
        row.className = `event-row ${confidenceClass(eventConfidence)} ${eventKey(event) === activeEventKey ? "active" : ""}`;
        row.title = event.event_warning || "";
        row.innerHTML = `
          <span class="event-badge ${sideClass(event.side_proxy)}">${event.event_index}</span>
          <span class="event-main">
            <strong>${escapeHtml(footSideLabel(event.estimated_foot_side))} · <span class="mono">${fmtTime(event.session_s)}</span></strong>
            <span>${escapeHtml(sideLabel(event.side_proxy))} · ${escapeHtml(event.label || "locomotion")} · bout ${escapeHtml(event.bout_confidence || "unknown")} · ${escapeHtml(sourcePeakText)}</span>
            <span class="confidence-pill ${confidenceClass(eventConfidence)}">${escapeHtml(confidenceLabel(eventConfidence))}</span>
            <span class="muted">foot side ${escapeHtml(confidenceLabel(event.foot_side_confidence || "unknown").toLowerCase())}</span>
          </span>
          <span class="event-metrics">
            <span>step ${fmtMetric(event.valid_step_path_length_m, 3, "m")}</span>
            <span>stride ${fmtMetric(event.valid_stride_path_length_m, 3, "m")}</span>
            <span>last${rollingStepCount} ${fmtMetric(rollingAverage, 3, "m")}</span>
          </span>
        `;
        row.addEventListener("click", () => seekTo(event.session_s, { pause: true }));
        eventTable.appendChild(row);
      });
    }

    function render() {
      if (!strideBouts.length) return;
      maybeSwitchBoutForTime();
      renderSummary();
      renderTimeline();
      renderEventTable();
      renderSignalChart();
    }

    function strideReviewHref(item) {
      if (item.trial_id === "__selector__") return item.href;
      const target = new URL(item.href, window.location.href);
      target.pathname = target.pathname.replace(/[/]index[.]html$/, "/pd_insighter_stride_review.html");
      target.search = "";
      return target.href;
    }

    function buildTrialJump() {
      trialJumpSelect.innerHTML = "";
      trialJumpOptions.forEach(item => {
        const option = document.createElement("option");
        option.value = strideReviewHref(item);
        option.textContent = item.trial_id === "__selector__" ? "All Trials" : item.label;
        option.selected = item.trial_id === trialId;
        trialJumpSelect.appendChild(option);
      });
      trialJumpSelect.addEventListener("change", event => {
        if (event.target.value) window.location.href = event.target.value;
      });
    }

    function initialize() {
      buildTrialJump();
      renderHero();
      downwardVideo.src = dashboardData.downward_session_video_uri;
      downwardVideo.defaultMuted = true;
      downwardVideo.muted = true;
      ariaVideo.src = dashboardData.aria_session_video_uri;
      ariaVideo.muted = false;
      sessionScrubber.max = String(sessionDuration);
      sessionScrubber.step = "0.01";
      try { downwardRotationQuarterTurns = Number(window.localStorage.getItem(`pd-insighter-downward-rotation-v2:${trialId}`) || 2) || 2; } catch (error) {}
      try { ariaRotationQuarterTurns = Number(window.localStorage.getItem(`pd-insighter-aria-rotation-v2:${trialId}`) || 1) || 1; } catch (error) {}
      applyDownwardRotation();
      applyAriaRotation();
      try {
        rollingStepCount = clamp(Math.round(Number(window.localStorage.getItem(`pd-insighter-stride-average-steps:${trialId}`) || 4)), 1, 20);
      } catch (error) {
        rollingStepCount = 4;
      }
      rollingStepCountInput.value = String(rollingStepCount);
      setPlaybackRate(Number(speedSelect.value || 0.5));

      if (!strideBouts.length) {
        strideWorkspace.style.display = "none";
        emptyState.style.display = "block";
        emptyState.innerHTML = `<h2>No stride review windows available for this trial</h2><p class="muted">This trial may not contain walking/stair intervals, or the stride diagnostic script has not been run for it yet.</p>`;
        return;
      }
      const preferredBout = strideBouts.find(bout => String(bout.confidence) === "high")
        || strideBouts.find(bout => String(bout.confidence) === "medium")
        || strideBouts[0];
      activeBoutId = Number(preferredBout.bout_id);
      renderBoutSelect();
      renderLabelSummary();
      const firstEvent = getBoutEvents()[0] || strideEvents[0];
      currentTime = firstEvent ? firstEvent.session_s : getBout().bout_start_s;
      seekTo(currentTime, { pause: true });
    }

    boutSelect.addEventListener("change", event => {
      activeBoutId = Number(event.target.value);
      signalZoomMode = "window";
      const firstEvent = getBoutEvents()[0];
      const bout = getBout();
      seekTo(firstEvent ? firstEvent.session_s : bout.bout_start_s, { pause: true });
    });
    signalZoomWindowButton.addEventListener("click", () => setSignalZoomMode("window"));
    signalZoomContextButton.addEventListener("click", () => setSignalZoomMode("context"));
    signalZoomFullButton.addEventListener("click", () => setSignalZoomMode("full"));
    rollingStepCountInput.addEventListener("input", event => {
      rollingStepCount = clamp(Math.round(Number(event.target.value || 4)), 1, 20);
      event.target.value = String(rollingStepCount);
      try { window.localStorage.setItem(`pd-insighter-stride-average-steps:${trialId}`, String(rollingStepCount)); } catch (error) {}
      render();
    });
    playPauseButton.addEventListener("click", () => {
      if (ariaVideo.paused) playBoth();
      else pauseBoth();
    });
    boutStartButton.addEventListener("click", () => {
      const bout = getBout();
      if (bout) seekTo(bout.bout_start_s, { pause: true });
    });
    prevEventButton.addEventListener("click", () => {
      const event = getAdjacentEvent(-1);
      if (event) seekTo(event.session_s, { pause: true });
    });
    nextEventButton.addEventListener("click", () => {
      const event = getAdjacentEvent(1);
      if (event) seekTo(event.session_s, { pause: true });
    });
    backSmallButton.addEventListener("click", () => seekTo(currentTime - 0.25, { pause: true }));
    forwardSmallButton.addEventListener("click", () => seekTo(currentTime + 0.25, { pause: true }));
    speedSelect.addEventListener("change", event => setPlaybackRate(Number(event.target.value || 0.5)));
    rotateDownwardButton.addEventListener("click", () => {
      downwardRotationQuarterTurns = (downwardRotationQuarterTurns + 1) % 4;
      applyDownwardRotation();
    });
    rotateAriaButton.addEventListener("click", () => {
      ariaRotationQuarterTurns = (ariaRotationQuarterTurns + 1) % 4;
      applyAriaRotation();
    });
    sessionScrubber.addEventListener("input", event => {
      isScrubbing = true;
      currentTime = clamp(Number(event.target.value || 0), 0, sessionDuration);
      render();
    });
    sessionScrubber.addEventListener("change", event => {
      isScrubbing = false;
      seekTo(Number(event.target.value || 0), { pause: true });
    });
    ariaVideo.addEventListener("timeupdate", () => {
      if (isScrubbing || isProgrammaticSeeking) return;
      currentTime = clamp(ariaVideo.currentTime || 0, 0, sessionDuration);
      syncFollower(false);
      render();
    });
    ariaVideo.addEventListener("play", () => { playPauseButton.textContent = "Pause"; });
    ariaVideo.addEventListener("pause", () => { playPauseButton.textContent = "Play"; });
    ariaVideo.addEventListener("ratechange", () => { downwardVideo.playbackRate = ariaVideo.playbackRate || 0.5; });
    downwardVideo.addEventListener("click", () => ariaVideo.paused ? playBoth() : pauseBoth());
    ariaVideo.addEventListener("click", () => ariaVideo.paused ? playBoth() : pauseBoth());

    initialize();
  </script>
</body>
</html>
"""
    return (
        template
        .replace("__TITLE__", title)
        .replace("__DATA__", data_json)
        .replace("__TRIAL_ID__", html.escape(str(payload["trial_id"])))
        .replace("__FULL_SESSION_HREF__", full_session_href)
        .replace("__MANUAL_LABELING_HREF__", manual_labeling_href)
        .replace("__TRANSITION_REVIEW_LINK__", transition_review_link_html)
    )


@environment_navigation
def build_transition_review_html(payload: dict[str, Any]) -> str:
    title = html.escape(f"{payload['trial_id']} - Transition Review")
    page_payload = payload_with_signal_samples(
        payload,
        include_stride_samples=False,
        include_transition_samples=True,
    )
    data_json = json.dumps(page_payload)
    full_session_href = html.escape(str(payload.get("full_session_href") or CANONICAL_TRIAL_ENTRY_NAME), quote=True)
    manual_labeling_href = html.escape(
        str(payload.get("manual_labeling_href") or MANUAL_LABELING_DASHBOARD_NAME),
        quote=True,
    )
    stride_review_href = html.escape(
        str(payload.get("stride_review_href") or STRIDE_REVIEW_DASHBOARD_NAME),
        quote=True,
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --ink: #17211d;
      --muted: #66736b;
      --line: #dfe8e0;
      --paper: #fbfaf4;
      --panel: #ffffff;
      --green: #167a55;
      --green-soft: #e3f4ec;
      --pink: #c026d3;
      --pink-soft: rgba(192, 38, 211, 0.16);
      --orange: #c9632e;
      --blue: #2f78b7;
      --purple: #7570b3;
      --shadow: 0 18px 48px rgba(28, 42, 34, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 0%, rgba(231, 93, 137, 0.16), transparent 28rem),
        radial-gradient(circle at 92% 6%, rgba(22, 122, 85, 0.14), transparent 30rem),
        linear-gradient(180deg, #f6f2ea 0%, #fbfaf4 42%, #f4f7f1 100%);
      font-family: "Avenir Next", "Trebuchet MS", sans-serif;
    }}
    .page {{ width: min(1800px, calc(100vw - 30px)); margin: 0 auto; padding: 24px 0 42px; }}
    .page-nav {{
      position: sticky;
      top: 12px;
      z-index: 20;
      display: flex;
      align-items: center;
      gap: 16px;
      margin-bottom: 18px;
      padding: 12px 14px;
      border: 1px solid rgba(23,33,29,0.10);
      border-radius: 20px;
      background: rgba(255,255,255,0.94);
      box-shadow: 0 14px 36px rgba(28,42,34,0.10);
      backdrop-filter: blur(14px);
    }}
    .nav-context {{
      display: flex;
      align-items: center;
      gap: 11px;
      flex: 0 0 auto;
      min-width: 190px;
    }}
    .nav-mark {{
      width: 38px;
      height: 38px;
      border-radius: 13px;
      display: grid;
      place-items: center;
      color: white;
      background: linear-gradient(145deg, var(--green), #10563e);
      box-shadow: 0 8px 18px rgba(16,86,62,0.20);
      font-size: 12px;
      font-weight: 900;
      letter-spacing: 0.04em;
    }}
    .nav-location {{ display: grid; gap: 2px; min-width: 0; line-height: 1.1; }}
    .nav-location strong {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 0.92rem; }}
    .nav-location span {{ color: var(--muted); font-size: 0.76rem; font-weight: 800; }}
    .nav-tabs {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 3px;
      flex: 1 1 auto;
      min-width: 0;
      padding: 4px;
      border: 1px solid rgba(23,33,29,0.10);
      border-radius: 16px;
      background: rgba(23,33,29,0.045);
    }}
    .nav-tab {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 36px;
      padding: 0 12px;
      border-radius: 12px;
      color: var(--muted);
      text-decoration: none;
      white-space: nowrap;
      font-size: 13px;
      font-weight: 900;
    }}
    .nav-tab:hover {{ color: var(--ink); background: rgba(255,255,255,0.76); }}
    .nav-tab.active {{
      color: var(--green);
      background: #fff;
      box-shadow: 0 4px 12px rgba(28,42,34,0.08);
    }}
    .nav-trial-switcher {{
      display: flex;
      align-items: center;
      gap: 9px;
      flex: 0 1 340px;
      min-width: 250px;
    }}
    .nav-trial-switcher > span {{
      color: var(--muted);
      font-size: 0.7rem;
      font-weight: 900;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .nav-trial-select {{ width: 100%; min-width: 0; border-radius: 13px; }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 18px;
      align-items: end;
      padding: 26px;
      border: 1px solid rgba(23,33,29,0.10);
      border-radius: 28px;
      background: rgba(255,255,255,0.86);
      box-shadow: var(--shadow);
    }}
    .eyebrow {{ color: var(--pink); font-weight: 900; letter-spacing: 0.12em; text-transform: uppercase; font-size: 0.78rem; }}
    h1 {{ margin: 8px 0 8px; font-family: Georgia, "Times New Roman", serif; font-size: clamp(2rem, 4vw, 4.4rem); line-height: 0.98; }}
    h2, h3 {{ margin: 0; line-height: 1.15; }}
    p {{ color: var(--muted); line-height: 1.5; margin: 0.35rem 0; }}
    .utility-links {{ display: flex; flex-wrap: wrap; gap: 10px; justify-content: flex-end; }}
    .button, button, select {{
      min-height: 38px;
      border-radius: 999px;
      border: 1px solid #cfd9d1;
      background: #fff;
      color: var(--ink);
      padding: 0 14px;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }}
    .button {{ display: inline-flex; align-items: center; text-decoration: none; }}
    .button.primary {{ background: var(--green); color: #fff; border-color: var(--green); }}
    .card {{
      margin-top: 18px;
      border: 1px solid rgba(23,33,29,0.10);
      border-radius: 24px;
      background: rgba(255,255,255,0.92);
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .card-inner {{ padding: 20px; }}
    .media-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 16px; }}
    .view-card {{ display: grid; gap: 8px; padding: 14px; border: 1px solid var(--line); border-radius: 20px; background: #fffdf8; }}
    .view-frame {{ min-height: 310px; border-radius: 18px; overflow: hidden; background: #101817; display: grid; place-items: center; }}
    video {{ width: 100%; height: 100%; max-height: 440px; object-fit: contain; background: #101817; }}
    video.rotated-quarter-turn {{ width: auto; height: min(74vw, 440px); max-width: none; }}
    .control-row {{ display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-top: 14px; }}
    button.primary {{ background: var(--green); color: #fff; border-color: var(--green); }}
    .select-label {{ color: var(--muted); font-weight: 900; }}
    .transition-select {{ width: 100%; min-width: 0; max-width: none; }}
    .review-toolbar {{ display: grid; gap: 14px; }}
    .transition-picker {{ display: grid; grid-template-columns: auto minmax(0, 1fr); align-items: center; gap: 12px; }}
    .global-locator {{
      padding: 15px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: linear-gradient(180deg, #fbfdf9, #f3f8f4);
    }}
    .global-locator-head {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 14px; }}
    .global-locator-title {{ display: grid; gap: 3px; }}
    .global-locator-title span {{ color: var(--muted); font-size: 0.76rem; font-weight: 900; letter-spacing: 0.08em; text-transform: uppercase; }}
    .global-locator-title strong {{ font-size: 1rem; }}
    .global-track {{
      position: relative;
      height: 34px;
      margin-top: 26px;
      overflow: visible;
      border: 1px solid rgba(23,33,29,0.12);
      border-radius: 10px;
      background: #e8eeea;
    }}
    .global-action-track, .global-selected-range, .global-playhead {{ position: absolute; inset-block: 0; pointer-events: none; }}
    .global-action-track {{ inset-inline: 0; overflow: hidden; border-radius: 9px; }}
    .global-action-segment {{ position: absolute; inset-block: 0; opacity: 0.30; }}
    .global-selected-range {{
      min-width: 8px;
      border: 2px solid var(--pink);
      border-radius: 7px;
      background: rgba(192,38,211,0.30);
      box-shadow: 0 0 0 3px rgba(192,38,211,0.10);
    }}
    .global-selected-marker {{
      position: absolute;
      top: -21px;
      bottom: -5px;
      width: 3px;
      border-radius: 999px;
      background: var(--pink);
      transform: translateX(-50%);
      pointer-events: none;
    }}
    .global-selected-marker::before {{
      content: "selected";
      position: absolute;
      left: 50%;
      top: -4px;
      transform: translate(-50%, -100%);
      padding: 3px 7px;
      border-radius: 999px;
      color: #fff;
      background: var(--pink);
      font-size: 0.65rem;
      font-weight: 900;
      white-space: nowrap;
    }}
    .global-playhead {{ top: -3px; bottom: -3px; width: 2px; background: #111827; transform: translateX(-50%); }}
    .global-playhead::after {{ content: ""; position: absolute; top: -4px; left: 50%; width: 8px; height: 8px; border-radius: 50%; background: #111827; transform: translateX(-50%); }}
    .global-track input[type="range"] {{ position: absolute; inset: 0; width: 100%; min-width: 0; height: 100%; margin: 0; opacity: 0; cursor: pointer; }}
    .global-scale {{ display: flex; justify-content: space-between; margin-top: 7px; color: var(--muted); font-size: 0.76rem; font-weight: 800; }}
    .playback-row {{ display: flex; flex-wrap: wrap; align-items: center; gap: 9px; margin-top: 15px; }}
    .audio-toggle {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-width: 118px;
      transition: border-color 0.16s ease, background 0.16s ease, color 0.16s ease;
    }}
    .audio-toggle::before {{
      content: "";
      width: 8px;
      height: 8px;
      flex: 0 0 8px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 0 4px rgba(22,122,85,0.10);
    }}
    .audio-toggle[data-muted="true"] {{
      color: var(--muted);
      background: #f4f7f4;
    }}
    .audio-toggle[data-muted="true"]::before {{
      background: #98a39f;
      box-shadow: none;
    }}
    .classification-panel {{ margin-top: 16px; padding: 14px; border: 1px solid var(--line); border-radius: 18px; background: #f9fbf6; }}
    .classification-panel h3 {{ margin-bottom: 4px; }}
    .classification-bar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 10px; }}
    .classification-filter {{ display: inline-flex; align-items: center; gap: 7px; border-width: 2px; background: #fff; }}
    .classification-filter.active {{ color: #fff; background: var(--classification-color); border-color: var(--classification-color); box-shadow: 0 5px 14px color-mix(in srgb, var(--classification-color) 22%, transparent); }}
    .classification-dot {{ width: 10px; height: 10px; border-radius: 50%; background: var(--classification-color); }}
    .classification-filter.active .classification-dot {{ background: #fff; }}
    .type-filter {{ flex: 1 1 260px; min-width: min(100%, 260px); max-width: 420px; margin-left: 0; }}
    input[type="range"] {{ flex: 1; min-width: 260px; accent-color: var(--green); }}
    .readout {{ color: var(--muted); font-weight: 800; }}
    .workspace {{ display: grid; grid-template-columns: minmax(0, 2.35fr) minmax(300px, 0.58fr); gap: 16px; }}
    .plot-card {{ border: 1px solid var(--line); border-radius: 20px; background: #fffdf8; overflow: hidden; }}
    .plot-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: start; padding: 16px 16px 10px; }}
    .plot-head-copy {{ min-width: 0; }}
    .plot-head-actions {{ display: flex; flex: 0 0 auto; flex-wrap: wrap; justify-content: flex-end; align-items: center; gap: 7px; max-width: 58%; }}
    .plot-head-actions button {{ min-width: 36px; min-height: 34px; padding: 0 10px; font-size: 0.82rem; }}
    .plot-head-actions .zoom-symbol {{ width: 36px; padding: 0; font-size: 1.25rem; line-height: 1; }}
    .plot-range-readout {{ color: var(--green); background: var(--green-soft); }}
    .plot-scale-readout {{ color: #536159; background: #f5f7f4; }}
    .plot-scale-readout.clipped {{ color: #9a4d12; background: #fff1e5; border-color: #efb47f; }}
    .plot-toolbar {{ display: flex; align-items: center; padding: 7px 16px 8px; border-top: 1px solid rgba(23,33,29,0.06); }}
    .plot-interaction-hint {{ width: 100%; color: var(--muted); font-size: 0.78rem; font-weight: 800; }}
    .plot-note {{ color: var(--muted); font-size: 0.9rem; padding: 0 16px 12px; }}
    .selected-detail {{ margin: 12px 16px 0; padding: 12px; border-radius: 16px; background: #f9fbf6; border: 1px solid var(--line); }}
    .selected-detail h3 {{ margin: 0 0 8px; }}
    .selected-metrics {{ display: flex; flex-wrap: wrap; gap: 8px; color: var(--muted); font-size: 0.88rem; }}
    .selected-metrics span {{ display: inline-flex; align-items: center; border-radius: 999px; border: 1px solid var(--line); background: #fff; padding: 5px 8px; font-weight: 800; }}
    .parameter-box {{ margin-top: 10px; padding: 10px 12px; border-radius: 13px; background: #f1f7f3; color: #425149; font-size: 0.86rem; line-height: 1.45; }}
    svg {{ width: 100%; display: block; background: linear-gradient(90deg, rgba(223,232,224,0.6) 1px, transparent 1px), linear-gradient(180deg, rgba(223,232,224,0.6) 1px, transparent 1px); background-size: 66px 66px; }}
    .interactive-plot {{ cursor: crosshair; touch-action: none; user-select: none; }}
    .interactive-plot.dragging {{ cursor: grabbing; }}
    aside h2 {{ font-size: 1.35rem; }}
    aside > p {{ font-size: 0.84rem; }}
    .transition-list {{ display: grid; gap: 7px; max-height: 700px; overflow: auto; padding-right: 4px; }}
    .transition-item {{
      width: 100%;
      display: grid;
      gap: 5px;
      text-align: left;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: #fff;
      padding: 9px 10px;
    }}
    .transition-item.active {{ box-shadow: 0 0 0 3px color-mix(in srgb, var(--transition-color, var(--pink)) 22%, transparent); }}
    .transition-item strong {{ font-size: 0.94rem; }}
    .transition-meta {{ display: flex; flex-wrap: wrap; gap: 6px; color: var(--muted); font-size: 0.76rem; }}
    .pill {{ display: inline-flex; align-items: center; gap: 6px; padding: 5px 8px; border-radius: 999px; background: #f4f7f1; border: 1px solid var(--line); font-weight: 800; }}
    .pill.pink {{ background: rgba(231, 93, 137, 0.12); color: #9c254e; border-color: rgba(231, 93, 137, 0.26); }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }}
    .summary-card {{ padding: 9px 10px; border-radius: 14px; border: 1px solid var(--line); background: #f9fbf6; }}
    .summary-card b {{ display: block; font-size: 0.73rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
    .summary-card span {{ display: block; margin-top: 4px; font-weight: 900; }}
    .review-panel {{ margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--line); }}
    .review-panel h3 {{ color: #8b1a98; }}
    .review-list {{ display: grid; gap: 8px; margin-top: 9px; }}
    .review-item {{ width: 100%; text-align: left; border-radius: 14px; border: 1px solid rgba(192,38,211,0.28); background: rgba(192,38,211,0.06); padding: 10px; }}
    .likelihood-common {{ color: #087453; background: rgba(5,150,105,0.10); }}
    .likelihood-less_likely {{ color: #9a5a06; background: rgba(217,119,6,0.10); }}
    .likelihood-needs_review {{ color: #8b1a98; background: rgba(192,38,211,0.10); }}
    .empty {{ padding: 28px; text-align: center; color: var(--muted); }}
    @media (max-width: 1050px) {{
      .hero, .workspace, .media-grid {{ grid-template-columns: 1fr; }}
      .utility-links {{ justify-content: flex-start; }}
      .summary-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .plot-head {{ flex-direction: column; }}
      .plot-head-actions {{ justify-content: flex-start; max-width: 100%; }}
    }}
    @media (max-width: 640px) {{
      .summary-grid, .transition-picker {{ grid-template-columns: 1fr; }}
      .page {{ width: min(100vw - 18px, 1800px); }}
      .page-nav {{ position: static; display: grid; align-items: stretch; }}
      .nav-context {{ min-width: 0; }}
      .nav-tabs {{ justify-content: flex-start; overflow-x: auto; }}
      .nav-tab {{ flex: 0 0 auto; }}
      .nav-trial-switcher {{ min-width: 0; max-width: none; }}
      .global-locator-head {{ display: grid; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <nav class="page-nav" aria-label="PD Insighter navigation">
      <div class="nav-context">
        <span class="nav-mark" aria-hidden="true">PD</span>
        <div class="nav-location">
          <strong>{html.escape(str(payload['trial_id']))}</strong>
          <span>Transition Review</span>
        </div>
      </div>
      <div class="nav-tabs" aria-label="Trial views">
        <a class="nav-tab" href="{full_session_href}">Full Session</a>
        <a class="nav-tab" href="{manual_labeling_href}">Manual Labels</a>
        <a class="nav-tab" href="{stride_review_href}">Stride</a>
        <span class="nav-tab active" aria-current="page">Transitions</span>
      </div>
      <label class="nav-trial-switcher" for="trialJumpSelect">
        <span>Trial</span>
        <select id="trialJumpSelect" class="nav-trial-select" aria-label="Switch trial or return to all trials"></select>
      </label>
    </nav>

    <section class="hero">
      <div>
        <div class="eyebrow">Transition-state review</div>
        <h1>{html.escape(str(payload['trial_id']))}: transition timing from raw SLAM + video</h1>
        <p>
          This page focuses on when motion changes actually begin and end. It uses high-confidence states before and after,
          then stores the middle change as a first-class transition state with its own type, timing, likelihood, and
          detection parameters. Each colored transition region is slightly wider for safe synchronized-video review.
        </p>
      </div>
    </section>

    <section class="card">
      <div class="card-inner">
        <div class="review-toolbar">
          <div class="transition-picker">
            <label class="select-label" for="transitionSelect">Selected transition</label>
            <select id="transitionSelect" class="transition-select" title="Select detected transition"></select>
          </div>
          <div class="classification-panel" style="margin-top:0">
            <h3>Filter Transitions</h3>
            <p>Show all transitions or narrow the list by evidence category and transition type.</p>
            <div class="classification-bar" id="classificationBar">
              <button class="classification-filter active" type="button" data-classification="all" style="--classification-color:#33423b">All Transitions <span id="allClassificationCount"></span></button>
              <button class="classification-filter" type="button" data-classification="clinical" style="--classification-color:#16815d"><i class="classification-dot"></i>Clinical <span id="clinicalClassificationCount"></span></button>
              <button class="classification-filter" type="button" data-classification="possible" style="--classification-color:#d48716"><i class="classification-dot"></i>Possible / Context <span id="possibleClassificationCount"></span></button>
              <button class="classification-filter" type="button" data-classification="less_possible" style="--classification-color:#c026d3"><i class="classification-dot"></i>Less Possible <span id="lessPossibleClassificationCount"></span></button>
              <select id="transitionTypeFilter" class="type-filter" title="Filter by transition type"></select>
            </div>
          </div>
          <div class="global-locator" aria-label="Selected transition location in the full session">
            <div class="global-locator-head">
              <div class="global-locator-title">
                <span>Location in full session</span>
                <strong id="globalLocationTitle">Select a transition</strong>
              </div>
              <span id="globalLocationReadout" class="pill">No transition selected</span>
            </div>
            <div class="global-track">
              <div id="globalActionTrack" class="global-action-track"></div>
              <div id="globalSelectedRange" class="global-selected-range"></div>
              <div id="globalSelectedMarker" class="global-selected-marker"></div>
              <div id="globalPlayhead" class="global-playhead"></div>
              <input id="sessionScrubber" type="range" min="0" max="{float(payload.get('session_duration_s') or 0):.3f}" step="0.033" value="0" aria-label="Full session position">
            </div>
            <div class="global-scale"><span>Session start · 0.0s</span><span id="globalSessionEnd"></span></div>
          </div>
        </div>

        <div class="media-grid">
          <div class="view-card">
            <h3>Downward View</h3>
            <div class="view-frame"><video id="downwardVideo" playsinline muted preload="metadata"></video></div>
          </div>
          <div class="view-card">
            <h3>Aria View + Audio</h3>
            <div class="view-frame"><video id="ariaVideo" playsinline preload="metadata"></video></div>
          </div>
        </div>
        <div class="playback-row">
          <button id="playTransitionButton" class="primary" type="button">Play Transition Only</button>
          <button id="playPauseButton" type="button">Play With Context</button>
          <button id="audioToggleButton" class="audio-toggle" type="button" aria-pressed="false" aria-label="Mute Aria audio" data-muted="false">Mute Audio</button>
          <button id="prevButton" type="button">Previous Transition</button>
          <button id="nextButton" type="button">Next Transition</button>
          <button id="rotateDownwardButton" type="button">Rotate Downward</button>
          <button id="rotateAriaButton" type="button">Rotate Aria</button>
          <span class="pill">Context playback adds 0.5s before and after</span>
          <span id="timeReadout" class="readout">0.0s</span>
        </div>
      </div>
    </section>

    <section class="card">
      <div class="card-inner workspace" id="workspace">
        <div>
          <div class="plot-card">
            <div class="plot-head">
              <div class="plot-head-copy">
                <h2>Full-Session Transition Map</h2>
                <p>Use the fixed controls to inspect a shorter time range without changing the height scale.</p>
              </div>
              <div class="plot-head-actions" aria-label="Full-session plot controls">
                <span id="fullPlotRangeReadout" class="pill plot-range-readout">Full session</span>
                <span id="fullPlotYScaleReadout" class="pill plot-scale-readout" title="The vertical scale uses the full-session minimum and maximum">Y session-fixed</span>
                <button id="fullPlotZoomOutButton" class="zoom-symbol" type="button" title="Show more time" aria-label="Zoom out in time">−</button>
                <button id="fullPlotZoomInButton" class="zoom-symbol" type="button" title="Show less time" aria-label="Zoom in in time">+</button>
                <button id="fullPlotFifteenMinuteButton" type="button" title="Show a 15-minute observation window without changing the full-session vertical scale">15 min</button>
                <button id="fullPlotResetButton" type="button" title="Show the complete session">Full</button>
              </div>
            </div>
            <div class="plot-toolbar">
              <span class="plot-interaction-hint">gray line = current video time · click = seek video · + / − = zoom around gray line · drag = pan while keeping gray line visible</span>
            </div>
            <svg id="fullPlot" class="interactive-plot" viewBox="0 0 1280 760" role="img" aria-label="Interactive full-session transition plot"></svg>
            <div class="plot-note" id="fullPlotNote"></div>
          </div>
          <div class="plot-card" style="margin-top:16px;">
            <div class="plot-head">
              <div class="plot-head-copy">
                <h2>Local Height Fluctuation</h2>
                <p>A wider, locally scaled view for seeing vertical motion patterns around the selected transition.</p>
              </div>
              <div class="plot-head-actions" aria-label="Local fluctuation plot controls">
                <span id="detailPlotYScaleReadout" class="pill plot-scale-readout" title="This vertical scale adapts to the selected local time window">Local Y auto</span>
                <select id="contextSelect" title="Choose the local time window" aria-label="Local fluctuation time window">
                  <option value="15">30 s view</option>
                  <option value="30" selected>1 min view</option>
                  <option value="60">2 min view</option>
                  <option value="120">4 min view</option>
                </select>
                <span class="pill" id="selectedTransitionPill">No transition selected</span>
              </div>
            </div>
            <div class="selected-detail">
              <h3 id="selectedTitle">No transition selected</h3>
              <div class="selected-metrics" id="selectedMetrics"></div>
              <div class="parameter-box" id="selectedParameterDetails"></div>
            </div>
            <svg id="detailPlot" viewBox="0 0 960 420" role="img" aria-label="Locally scaled height fluctuation plot around the selected transition"></svg>
            <div class="plot-note" id="detailPlotNote"></div>
          </div>
        </div>
        <aside>
          <h2>Detected Transitions</h2>
          <p>Click a row to jump both videos to that first-class transition state.</p>
          <div class="summary-grid" id="summaryGrid" style="grid-template-columns:1fr 1fr;"></div>
          <div class="transition-list" id="transitionList"></div>
          <div class="review-panel">
            <h3>Needs Review / Unusual Examples</h3>
            <p>These are not called impossible. They are saved here because the direct state change is unusual or needs more context.</p>
            <div class="review-list" id="reviewList"></div>
          </div>
        </aside>
      </div>
      <div id="emptyState" class="empty" style="display:none;"></div>
    </section>
  </div>

  <script>
    const dashboardData = {data_json};
    const trialId = String(dashboardData.trial_id || "");
    const trialJumpOptions = dashboardData.trial_jump_options || [];
    const trialJumpSelect = document.getElementById("trialJumpSelect");
    const transitionReview = dashboardData.transition_review || {{}};
    const transitionSampling = transitionReview.sampling || {{}};
    const transitions = (transitionReview.windows || []).slice().sort((a, b) => Number(a.start_s) - Number(b.start_s));
    const unusualTransitions = transitionReview.unlikely_examples || transitions.filter(item => item.likelihood === "needs_review");
    const samples = (transitionReview.samples || []).slice().sort((a, b) => Number(a.t) - Number(b.t));
    const actionSpans = transitionReview.action_spans || [];
    const sessionDuration = Number(dashboardData.session_duration_s || 0);
    const downwardVideo = document.getElementById("downwardVideo");
    const ariaVideo = document.getElementById("ariaVideo");
    const playTransitionButton = document.getElementById("playTransitionButton");
    const playPauseButton = document.getElementById("playPauseButton");
    const audioToggleButton = document.getElementById("audioToggleButton");
    const prevButton = document.getElementById("prevButton");
    const nextButton = document.getElementById("nextButton");
    const rotateDownwardButton = document.getElementById("rotateDownwardButton");
    const rotateAriaButton = document.getElementById("rotateAriaButton");
    const contextSelect = document.getElementById("contextSelect");
    const transitionSelect = document.getElementById("transitionSelect");
    const transitionTypeFilter = document.getElementById("transitionTypeFilter");
    const classificationButtons = Array.from(document.querySelectorAll(".classification-filter"));
    const timeReadout = document.getElementById("timeReadout");
    const scrubber = document.getElementById("sessionScrubber");
    const summaryGrid = document.getElementById("summaryGrid");
    const transitionList = document.getElementById("transitionList");
    const reviewList = document.getElementById("reviewList");
    const workspace = document.getElementById("workspace");
    const emptyState = document.getElementById("emptyState");
    const fullPlot = document.getElementById("fullPlot");
    const fullPlotZoomInButton = document.getElementById("fullPlotZoomInButton");
    const fullPlotZoomOutButton = document.getElementById("fullPlotZoomOutButton");
    const fullPlotFifteenMinuteButton = document.getElementById("fullPlotFifteenMinuteButton");
    const fullPlotResetButton = document.getElementById("fullPlotResetButton");
    const fullPlotRangeReadout = document.getElementById("fullPlotRangeReadout");
    const fullPlotYScaleReadout = document.getElementById("fullPlotYScaleReadout");
    const detailPlot = document.getElementById("detailPlot");
    const detailPlotYScaleReadout = document.getElementById("detailPlotYScaleReadout");
    const fullPlotNote = document.getElementById("fullPlotNote");
    const detailPlotNote = document.getElementById("detailPlotNote");
    const selectedTransitionPill = document.getElementById("selectedTransitionPill");
    const selectedTitle = document.getElementById("selectedTitle");
    const selectedMetrics = document.getElementById("selectedMetrics");
    const selectedParameterDetails = document.getElementById("selectedParameterDetails");
    const globalActionTrack = document.getElementById("globalActionTrack");
    const globalSelectedRange = document.getElementById("globalSelectedRange");
    const globalSelectedMarker = document.getElementById("globalSelectedMarker");
    const globalPlayhead = document.getElementById("globalPlayhead");
    const globalLocationTitle = document.getElementById("globalLocationTitle");
    const globalLocationReadout = document.getElementById("globalLocationReadout");
    const globalSessionEnd = document.getElementById("globalSessionEnd");
    let currentTime = 0;
    let activeTransitionIndex = transitions.length ? 0 : -1;
    let activeClassification = "all";
    let activeTransitionType = "all";
    let downwardTurns = 2;
    let ariaTurns = 1;
    let syncTimer = null;
    let activeReviewRange = null;
    let activePlaybackMode = "context";
    const fullPlotExtentEnd = Math.max(sessionDuration, Number(samples[samples.length - 1]?.t || 0), 1);
    const FULL_PLOT_MIN_WINDOW_S = Math.min(6, fullPlotExtentEnd);
    const FULL_PLOT_WIDTH = 1280;
    const FULL_PLOT_HEIGHT = 760;
    const FULL_PLOT_MARGIN = {{ left: 78, right: 58, top: 38, bottom: 52 }};
    let fullPlotViewStart = 0;
    let fullPlotViewEnd = fullPlotExtentEnd;
    let fullPlotLockedDomains = null;
    let fullPlotDragState = null;
    const TRANSITION_VIDEO_PADDING_S = 0.5;
    const AUDIO_STORAGE_KEY = "pd-insighter-audio-enabled-v1";
    const labelColors = {{
      sitting: "#e6ab02",
      standing: "#7570b3",
      walking: "#d95f02",
      upstairs: "#1b9e77",
      downstairs: "#666666",
      missing: "#b6bec6"
    }};

    function transitionReviewHref(item) {{
      if (item.trial_id === "__selector__") return item.href;
      const target = new URL(item.href, window.location.href);
      target.pathname = target.pathname.replace(/[/]index[.]html$/, "/pd_insighter_transition_review.html");
      return target.href;
    }}
    function buildTrialJump() {{
      trialJumpSelect.innerHTML = "";
      trialJumpOptions.forEach(item => {{
        const option = document.createElement("option");
        option.value = transitionReviewHref(item);
        option.textContent = item.trial_id === "__selector__" ? "All Trials" : item.label;
        option.selected = item.trial_id === trialId;
        trialJumpSelect.appendChild(option);
      }});
      trialJumpSelect.addEventListener("change", event => {{
        if (event.target.value) window.location.href = event.target.value;
      }});
    }}

    function clamp(value, min, max) {{ return Math.min(max, Math.max(min, value)); }}
    function fmtTime(value) {{ return `${{Number(value || 0).toFixed(1)}}s`; }}
    function fmtClock(value) {{
      const total = Math.max(0, Number(value) || 0);
      const hours = Math.floor(total / 3600);
      const minutes = Math.floor((total % 3600) / 60);
      const seconds = (total % 60).toFixed(1).padStart(4, "0");
      return hours > 0
        ? `${{hours}}:${{String(minutes).padStart(2, "0")}}:${{seconds}}`
        : `${{minutes}}:${{seconds}}`;
    }}
    function fmtMetric(value, digits = 2, suffix = "") {{
      const number = Number(value);
      return Number.isFinite(number) ? `${{number.toFixed(digits)}}${{suffix}}` : "N/A";
    }}
    function transitionArrowLabel(transition) {{
      const fromAction = String(transition.from_action || "").trim();
      const toAction = String(transition.to_action || "").trim();
      if (fromAction && toAction) return `${{fromAction}} \u2192 ${{toAction}}`;
      return String(transition.display_label || transition.transition_type || "transition")
        .replace(/\\s*->\\s*/g, " \u2192 ")
        .replace(/\\s+-to-\\s+/g, " \u2192 ");
    }}
    function transitionOptionLabel(transition, index) {{
      return `${{String(index + 1).padStart(2, "0")}}. ${{transitionArrowLabel(transition)}} · ${{fmtTime(transition.start_s)}}-${{fmtTime(transition.end_s)}} · ${{String(transition.classification_label || transition.likelihood_label || "Review")}}`;
    }}
    function buildGlobalActionTrack() {{
      globalActionTrack.innerHTML = "";
      if (!(sessionDuration > 0)) return;
      actionSpans.forEach(span => {{
        const start = clamp(Number(span.start_s || 0), 0, sessionDuration);
        const end = clamp(Number(span.end_s || start), start, sessionDuration);
        const segment = document.createElement("span");
        segment.className = "global-action-segment";
        segment.style.left = `${{start / sessionDuration * 100}}%`;
        segment.style.width = `${{Math.max(0, end - start) / sessionDuration * 100}}%`;
        segment.style.background = span.color || labelColors.missing;
        segment.title = `${{String(span.label || "motion")}} · ${{fmtClock(start)}}-${{fmtClock(end)}}`;
        globalActionTrack.appendChild(segment);
      }});
      globalSessionEnd.textContent = `Session end · ${{fmtClock(sessionDuration)}}`;
    }}
    function renderGlobalLocator() {{
      const safeDuration = Math.max(sessionDuration, 0.001);
      globalPlayhead.style.left = `${{clamp(currentTime / safeDuration * 100, 0, 100)}}%`;
      if (activeTransitionIndex < 0) {{
        globalSelectedRange.style.display = "none";
        globalSelectedMarker.style.display = "none";
        globalLocationTitle.textContent = "No transition selected";
        globalLocationReadout.textContent = "Choose a transition above";
        return;
      }}
      const transition = transitions[activeTransitionIndex];
      const start = clamp(Number(transition.start_s || 0), 0, sessionDuration);
      const end = clamp(Number(transition.end_s || start), start, sessionDuration);
      const center = clamp((start + end) / 2, 0, sessionDuration);
      const startPct = start / safeDuration * 100;
      const widthPct = Math.max(0, end - start) / safeDuration * 100;
      const centerPct = center / safeDuration * 100;
      globalSelectedRange.style.display = "block";
      globalSelectedMarker.style.display = "block";
      globalSelectedRange.style.left = `${{startPct}}%`;
      globalSelectedRange.style.width = `${{widthPct}}%`;
      globalSelectedMarker.style.left = `${{centerPct}}%`;
      globalSelectedRange.style.borderColor = transitionColor(transition);
      globalSelectedRange.title = `Exact transition range: ${{fmtClock(start)}}-${{fmtClock(end)}}`;
      globalLocationTitle.textContent = `#${{activeTransitionIndex + 1}} ${{transitionArrowLabel(transition)}}`;
      globalLocationReadout.textContent = `${{fmtClock(start)}}-${{fmtClock(end)}} · ${{centerPct.toFixed(1)}}% through session`;
    }}
    function escapeHtml(value) {{
      return String(value ?? "").replace(/[&<>"']/g, ch => ({{ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }}[ch]));
    }}
    function svgEl(name, attrs = {{}}) {{
      const el = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)));
      return el;
    }}
    function installPlotClip(svg, id, width, height, margin) {{
      const defs = svgEl("defs");
      const clip = svgEl("clipPath", {{ id }});
      clip.appendChild(svgEl("rect", {{
        x: margin.left,
        y: margin.top,
        width: Math.max(0, width - margin.left - margin.right),
        height: Math.max(0, height - margin.top - margin.bottom),
      }}));
      defs.appendChild(clip);
      svg.appendChild(defs);
      return `url(#${{id}})`;
    }}
    function clear(el) {{ while (el.firstChild) el.removeChild(el.firstChild); }}
    function valuesFor(field, data = samples) {{
      return data.map(row => Number(row[field])).filter(Number.isFinite);
    }}
    function paddedDomain(values, pad = 0.08) {{
      if (!values.length) return [0, 1];
      let min = Math.min(...values);
      let max = Math.max(...values);
      if (Math.abs(max - min) < 1e-9) {{ min -= 1; max += 1; }}
      const extra = (max - min) * pad;
      return [min - extra, max + extra];
    }}
    function scale(domain, range) {{
      const [d0, d1] = domain;
      const [r0, r1] = range;
      const span = Math.max(1e-9, d1 - d0);
      return value => r0 + ((Number(value) - d0) / span) * (r1 - r0);
    }}
    function normalizedFullPlotWindow(start, end) {{
      const total = fullPlotExtentEnd;
      const minimum = Math.min(FULL_PLOT_MIN_WINDOW_S, total);
      let width = clamp(Number(end) - Number(start), minimum, total);
      let safeStart = Number(start);
      if (!Number.isFinite(safeStart)) safeStart = 0;
      let safeEnd = safeStart + width;
      if (safeStart < 0) {{
        safeEnd -= safeStart;
        safeStart = 0;
      }}
      if (safeEnd > total) {{
        safeStart -= safeEnd - total;
        safeEnd = total;
      }}
      safeStart = clamp(safeStart, 0, Math.max(0, total - width));
      safeEnd = clamp(safeStart + width, safeStart, total);
      return {{ start: safeStart, end: safeEnd }};
    }}
    function lockFullPlotDomains() {{
      const zDomain = paddedDomain(valuesFor("z", samples), 0.12);
      const speedValues = valuesFor("speed", samples);
      const speedMinimum = Math.min(0, ...speedValues);
      const speedMaximum = Math.max(...speedValues, 0.1);
      const speedPadding = Math.max((speedMaximum - speedMinimum) * 0.08, 0.01);
      const speedDomain = [speedMinimum, speedMaximum + speedPadding];
      fullPlotLockedDomains = {{
        z: zDomain,
        speed: speedDomain,
        referenceStart: 0,
        referenceEnd: fullPlotExtentEnd,
      }};
    }}
    function setFullPlotWindow(start, end, options = {{}}) {{
      let next = normalizedFullPlotWindow(start, end);
      if (options.keepAnchorVisible && Number.isFinite(Number(currentTime))) {{
        const anchor = clamp(Number(currentTime), 0, fullPlotExtentEnd);
        const width = next.end - next.start;
        if (anchor < next.start) next = normalizedFullPlotWindow(anchor, anchor + width);
        if (anchor > next.end) next = normalizedFullPlotWindow(anchor - width, anchor);
      }}
      fullPlotViewStart = next.start;
      fullPlotViewEnd = next.end;
      if (!fullPlotLockedDomains) lockFullPlotDomains();
      drawFullPlot();
    }}
    function fullPlotZoomAnchor() {{
      const playbackTime = Number(currentTime);
      if (Number.isFinite(playbackTime)) return clamp(playbackTime, 0, fullPlotExtentEnd);
      return (fullPlotViewStart + fullPlotViewEnd) / 2;
    }}
    function keepPlaybackVisibleInFullPlot() {{
      const width = Math.max(FULL_PLOT_MIN_WINDOW_S, fullPlotViewEnd - fullPlotViewStart);
      const edgeGuard = Math.min(width * 0.08, 5);
      if (currentTime >= fullPlotViewStart + edgeGuard && currentTime <= fullPlotViewEnd - edgeGuard) return;
      const next = normalizedFullPlotWindow(currentTime - width / 2, currentTime + width / 2);
      fullPlotViewStart = next.start;
      fullPlotViewEnd = next.end;
    }}
    function zoomFullPlot(scaleFactor, anchorTime = fullPlotZoomAnchor()) {{
      const oldWidth = Math.max(FULL_PLOT_MIN_WINDOW_S, fullPlotViewEnd - fullPlotViewStart);
      const anchor = clamp(anchorTime, 0, fullPlotExtentEnd);
      const newWidth = clamp(oldWidth * scaleFactor, FULL_PLOT_MIN_WINDOW_S, fullPlotExtentEnd);
      setFullPlotWindow(anchor - newWidth / 2, anchor + newWidth / 2);
    }}
    function resetFullPlotWindow() {{
      setFullPlotWindow(0, fullPlotExtentEnd);
    }}
    function showFifteenMinuteWindow() {{
      const width = Math.min(15 * 60, fullPlotExtentEnd);
      const center = fullPlotZoomAnchor();
      setFullPlotWindow(center - width / 2, center + width / 2);
    }}
    function fullPlotWindowForTransition(transition) {{
      const start = Number(transition.start_s || 0);
      const end = Number(transition.end_s || start);
      const center = (start + end) / 2;
      const width = clamp(
        Math.max(16, (end - start) * 6),
        FULL_PLOT_MIN_WINDOW_S,
        Math.min(60, fullPlotExtentEnd),
      );
      return normalizedFullPlotWindow(center - width / 2, center + width / 2);
    }}
    function fullPlotEventTime(event) {{
      const rect = fullPlot.getBoundingClientRect();
      const svgX = (event.clientX - rect.left) / Math.max(rect.width, 1) * FULL_PLOT_WIDTH;
      const plotWidth = FULL_PLOT_WIDTH - FULL_PLOT_MARGIN.left - FULL_PLOT_MARGIN.right;
      const ratio = clamp((svgX - FULL_PLOT_MARGIN.left) / Math.max(plotWidth, 1), 0, 1);
      return fullPlotViewStart + ratio * (fullPlotViewEnd - fullPlotViewStart);
    }}
    function linePath(data, x, y, field) {{
      return data
        .filter(row => Number.isFinite(Number(row.t)) && Number.isFinite(Number(row[field])))
        .map((row, index) => `${{index === 0 ? "M" : "L"}} ${{x(row.t).toFixed(2)}} ${{y(row[field]).toFixed(2)}}`)
        .join(" ");
    }}
    function thinSignalSamples(rows, maxPoints = 7000) {{
      if (rows.length <= maxPoints) return rows;
      const bucketCount = Math.max(1, Math.floor((maxPoints - 2) / 3));
      const bucketSize = Math.ceil((rows.length - 2) / bucketCount);
      const result = [rows[0]];
      for (let start = 1; start < rows.length - 1; start += bucketSize) {{
        const bucket = rows.slice(start, Math.min(rows.length - 1, start + bucketSize));
        if (!bucket.length) continue;
        const candidates = [
          bucket.reduce((best, row) => Number(row.z) < Number(best.z) ? row : best),
          bucket.reduce((best, row) => Number(row.z) > Number(best.z) ? row : best),
          bucket.reduce((best, row) => Number(row.speed) > Number(best.speed) ? row : best),
        ];
        [...new Set(candidates)]
          .sort((a, b) => Number(a.t) - Number(b.t))
          .forEach(row => result.push(row));
      }}
      result.push(rows.at(-1));
      return result;
    }}
    function setRotation(videoEl, turns) {{
      const normalized = ((turns % 4) + 4) % 4;
      videoEl.style.transform = `rotate(${{normalized * 90}}deg)`;
      videoEl.classList.toggle("rotated-quarter-turn", normalized % 2 === 1);
    }}
    function applyRotations() {{
      setRotation(downwardVideo, downwardTurns);
      setRotation(ariaVideo, ariaTurns);
    }}
    function transitionPlaybackBounds(transition) {{
      return {{
        start: clamp(Number(transition.start_s || 0) - TRANSITION_VIDEO_PADDING_S, 0, sessionDuration),
        end: clamp(Number(transition.end_s || 0) + TRANSITION_VIDEO_PADDING_S, 0, sessionDuration),
      }};
    }}
    function transitionExactBounds(transition) {{
      const start = clamp(Number(transition.start_s || 0), 0, sessionDuration);
      return {{
        start,
        end: clamp(Number(transition.end_s || start), start, sessionDuration),
      }};
    }}
    function syncVideoTime(target, options = {{}}) {{
      if (!options.keepTransitionRange) activeReviewRange = null;
      const safe = clamp(Number(target) || 0, 0, sessionDuration);
      currentTime = safe;
      [downwardVideo, ariaVideo].forEach(video => {{
        try {{ video.currentTime = safe; }} catch (error) {{}}
      }});
      renderDynamic();
    }}
    function updatePlaybackButtons() {{
      const playing = !ariaVideo.paused || !downwardVideo.paused;
      playTransitionButton.textContent = playing && activePlaybackMode === "transition"
        ? "Pause Transition"
        : "Play Transition Only";
      playPauseButton.textContent = playing && activePlaybackMode === "context"
        ? "Pause Context"
        : "Play With Context";
    }}
    function updateAudioButton() {{
      const enabled = !ariaVideo.muted && Number(ariaVideo.volume) > 0;
      audioToggleButton.textContent = enabled ? "Mute Audio" : "Unmute Audio";
      audioToggleButton.dataset.muted = enabled ? "false" : "true";
      audioToggleButton.setAttribute("aria-pressed", enabled ? "false" : "true");
      audioToggleButton.setAttribute("aria-label", enabled ? "Mute Aria audio" : "Unmute Aria audio");
      audioToggleButton.title = enabled
        ? "Mute the Aria video's audio"
        : "Turn on audio from the Aria video";
    }}
    function setAudioEnabled(enabled, persist = true) {{
      downwardVideo.defaultMuted = true;
      downwardVideo.muted = true;
      if (enabled && Number(ariaVideo.volume) <= 0) ariaVideo.volume = 1;
      ariaVideo.muted = !enabled;
      if (persist) {{
        try {{ window.localStorage.setItem(AUDIO_STORAGE_KEY, enabled ? "true" : "false"); }} catch (error) {{}}
      }}
      updateAudioButton();
    }}
    function restoreAudioPreference() {{
      let enabled = true;
      try {{
        const saved = window.localStorage.getItem(AUDIO_STORAGE_KEY);
        if (saved !== null) enabled = saved !== "false";
      }} catch (error) {{}}
      setAudioEnabled(enabled, false);
    }}
    function pauseBoth() {{
      ariaVideo.pause();
      downwardVideo.pause();
      updatePlaybackButtons();
    }}
    async function playBoth() {{
      startSyncTimer();
      let target = clamp(currentTime || ariaVideo.currentTime || downwardVideo.currentTime || 0, 0, sessionDuration);
      if (
        activeReviewRange
        && (target < activeReviewRange.start - 0.05 || target >= activeReviewRange.end - 0.02)
      ) {{
        target = activeReviewRange.start;
        syncVideoTime(target, {{ keepTransitionRange: true }});
      }}
      try {{ ariaVideo.currentTime = target; }} catch (error) {{}}
      try {{ downwardVideo.currentTime = target; }} catch (error) {{}}
      let ariaStarted = false;
      let fallbackMuted = false;
      try {{
        await ariaVideo.play();
        ariaStarted = true;
      }} catch (error) {{
        ariaVideo.muted = true;
        try {{
          await ariaVideo.play();
          ariaStarted = true;
          fallbackMuted = true;
        }} catch (fallbackError) {{}}
      }}
      if (!ariaStarted) {{
        pauseBoth();
        return;
      }}
      try {{
        downwardVideo.currentTime = ariaVideo.currentTime || target;
        await downwardVideo.play();
      }} catch (error) {{}}
      updatePlaybackButtons();
      if (fallbackMuted) {{
        updateAudioButton();
        playPauseButton.title = "Aria audio was muted because the browser blocked autoplay.";
      }}
      window.setTimeout(() => {{
        if (ariaVideo.paused && downwardVideo.paused) updatePlaybackButtons();
      }}, 150);
    }}
    function playSelectedRange(mode) {{
      if (activeTransitionIndex < 0) return;
      const isPlaying = !ariaVideo.paused || !downwardVideo.paused;
      if (isPlaying && activePlaybackMode === mode) {{
        pauseBoth();
        return;
      }}
      pauseBoth();
      activePlaybackMode = mode;
      const transition = transitions[activeTransitionIndex];
      activeReviewRange = mode === "transition"
        ? transitionExactBounds(transition)
        : transitionPlaybackBounds(transition);
      syncVideoTime(activeReviewRange.start, {{ keepTransitionRange: true }});
      playBoth();
    }}
    function togglePlayback() {{
      if (!ariaVideo.paused || !downwardVideo.paused) pauseBoth();
      else playBoth();
    }}
    function handleVideoClick(event) {{
      event.preventDefault();
      event.stopPropagation();
      togglePlayback();
    }}
    function startSyncTimer() {{
      if (syncTimer) return;
      syncTimer = window.setInterval(() => {{
        if (!ariaVideo.paused && Math.abs((downwardVideo.currentTime || 0) - (ariaVideo.currentTime || 0)) > 0.16) {{
          try {{ downwardVideo.currentTime = ariaVideo.currentTime; }} catch (error) {{}}
        }}
      }}, 350);
    }}
    function selectTransition(index, seek = true) {{
      if (!transitions.length) return;
      pauseBoth();
      activeTransitionIndex = clamp(index, 0, transitions.length - 1);
      const transition = transitions[activeTransitionIndex];
      const nextPlotWindow = fullPlotWindowForTransition(transition);
      fullPlotViewStart = nextPlotWindow.start;
      fullPlotViewEnd = nextPlotWindow.end;
      if (!fullPlotLockedDomains) lockFullPlotDomains();
      activePlaybackMode = "context";
      activeReviewRange = transitionPlaybackBounds(transition);
      if (seek) syncVideoTime(activeReviewRange.start, {{ keepTransitionRange: true }});
      else renderDynamic();
    }}
    function nearestTransitionIndex(targetTime) {{
      const visible = filteredTransitions();
      if (!visible.length) return -1;
      let bestIndex = transitions.indexOf(visible[0]);
      let bestDistance = Math.abs(Number(visible[0].center_s) - targetTime);
      visible.forEach(transition => {{
        const index = transitions.indexOf(transition);
        const distance = Math.abs(Number(transition.center_s) - targetTime);
        if (distance < bestDistance) {{ bestDistance = distance; bestIndex = index; }}
      }});
      return bestIndex;
    }}
    function transitionClassification(transition) {{
      if (transition.classification) return String(transition.classification);
      if (transition.transition_category === "clinically_meaningful") return "clinical";
      if (transition.transition_category === "stair_context") return "possible";
      return "less_possible";
    }}
    function classificationColor(transition) {{
      return String(transition.classification_color || ({{
        clinical: "#16815d",
        possible: "#d48716",
        less_possible: "#c026d3",
      }})[transitionClassification(transition)] || "#7c3aed");
    }}
    function filteredTransitions() {{
      return transitions.filter(transition => {{
        const classificationMatches = activeClassification === "all" || transitionClassification(transition) === activeClassification;
        const typeMatches = activeTransitionType === "all" || String(transition.transition_type || "") === activeTransitionType;
        return classificationMatches && typeMatches;
      }});
    }}
    function selectAdjacentTransition(direction) {{
      const visible = filteredTransitions();
      if (!visible.length) return;
      const current = transitions[activeTransitionIndex];
      let position = visible.indexOf(current);
      if (position < 0) position = direction > 0 ? -1 : 0;
      const nextPosition = clamp(position + direction, 0, visible.length - 1);
      selectTransition(transitions.indexOf(visible[nextPosition]), true);
    }}
    function transitionColor(transition) {{
      return String(transition.color || "#7c3aed");
    }}
    function likelihoodClass(transition) {{
      return `likelihood-${{String(transition.likelihood || "needs_review")}}`;
    }}
    function formatParameterKey(key) {{
      return String(key).replaceAll("_", " ").replace(/\\b\\w/g, ch => ch.toUpperCase());
    }}
    function formatDetectionParameters(transition) {{
      const entries = Object.entries(transition.detection_parameters || {{}});
      if (!entries.length) return "No type-specific parameter profile is available.";
      return entries.map(([key, value]) => `${{formatParameterKey(key)}}: <b>${{escapeHtml(value)}}</b>`).join(" · ");
    }}
    function drawAxes(svg, xDomain, yDomain, width, height, margin, yLabel) {{
      const x = scale(xDomain, [margin.left, width - margin.right]);
      const y = scale(yDomain, [height - margin.bottom, margin.top]);
      svg.appendChild(svgEl("line", {{ x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, stroke: "rgba(23,33,29,0.40)", "stroke-width": 1.2 }}));
      svg.appendChild(svgEl("line", {{ x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, stroke: "rgba(23,33,29,0.40)", "stroke-width": 1.2 }}));
      for (let i = 0; i <= 5; i += 1) {{
        const tx = margin.left + ((width - margin.left - margin.right) * i / 5);
        const time = xDomain[0] + ((xDomain[1] - xDomain[0]) * i / 5);
        svg.appendChild(svgEl("line", {{ x1: tx, y1: margin.top, x2: tx, y2: height - margin.bottom, stroke: "rgba(23,33,29,0.10)" }}));
        const label = svgEl("text", {{ x: tx, y: height - 14, "text-anchor": "middle", fill: "#66736b", "font-size": 11, "font-weight": 800 }});
        label.textContent = fmtClock(time);
        svg.appendChild(label);
      }}
      const ySpan = Math.abs(yDomain[1] - yDomain[0]);
      const yDigits = ySpan < 0.2 ? 3 : (ySpan < 2 ? 2 : 1);
      for (let i = 0; i <= 4; i += 1) {{
        const ty = margin.top + ((height - margin.top - margin.bottom) * i / 4);
        const value = yDomain[1] - ((yDomain[1] - yDomain[0]) * i / 4);
        svg.appendChild(svgEl("line", {{ x1: margin.left, y1: ty, x2: width - margin.right, y2: ty, stroke: "rgba(23,33,29,0.10)" }}));
        const tick = svgEl("text", {{ x: margin.left - 9, y: ty + 4, "text-anchor": "end", fill: "#4d5d55", "font-size": 11, "font-weight": 800 }});
        tick.textContent = Number(value).toFixed(yDigits);
        svg.appendChild(tick);
      }}
      const yText = svgEl("text", {{ x: 12, y: 18, fill: "#33423b", "font-size": 12, "font-weight": 900 }});
      yText.textContent = yLabel;
      svg.appendChild(yText);
      return {{ x, y }};
    }}
    function drawTransitionBands(svg, x, yTop, yBottom, transitionsToDraw, options = {{}}) {{
      transitionsToDraw.forEach((transition, index) => {{
        const domainStart = Number(options.xDomain?.[0] ?? -Infinity);
        const domainEnd = Number(options.xDomain?.[1] ?? Infinity);
        const rawStart = Number(transition.start_s);
        const rawEnd = Number(transition.end_s);
        if (rawEnd < domainStart || rawStart > domainEnd) return;
        const clippedStart = Math.max(rawStart, domainStart);
        const clippedEnd = Math.min(rawEnd, domainEnd);
        const start = x(clippedStart);
        const end = x(clippedEnd);
        const width = Math.max(2, end - start);
        const typeColor = transitionColor(transition);
        const categoryColor = classificationColor(transition);
        const rect = svgEl("rect", {{
          x: start,
          y: yTop,
          width,
          height: yBottom - yTop,
          fill: categoryColor,
          opacity: transitions.indexOf(transition) === activeTransitionIndex || options.forceStrong ? 0.25 : 0.15,
          rx: 6,
          "pointer-events": "none",
        }});
        const title = svgEl("title");
        title.textContent = `${{transitionArrowLabel(transition)}} · ${{fmtTime(transition.start_s)}}-${{fmtTime(transition.end_s)}} · ${{transition.classification_label || transition.likelihood_label || "Review"}}`;
        rect.appendChild(title);
        svg.appendChild(rect);
        if (options.label) {{
          const label = svgEl("text", {{ x: (start + end) / 2, y: yTop + 18, "text-anchor": "middle", fill: typeColor, "font-size": 12, "font-weight": 900, "paint-order": "stroke", stroke: "white", "stroke-width": 4 }});
          label.textContent = String(transition.transition_type || "transition").toUpperCase();
          svg.appendChild(label);
        }}
      }});
    }}
    function drawFullPlot() {{
      clear(fullPlot);
      if (!samples.length) return;
      const normalized = normalizedFullPlotWindow(fullPlotViewStart, fullPlotViewEnd);
      fullPlotViewStart = normalized.start;
      fullPlotViewEnd = normalized.end;
      const W = FULL_PLOT_WIDTH, H = FULL_PLOT_HEIGHT;
      const margin = FULL_PLOT_MARGIN;
      const xDomain = [fullPlotViewStart, fullPlotViewEnd];
      let part = samples.filter(row => Number(row.t) >= xDomain[0] && Number(row.t) <= xDomain[1]);
      if (!part.length) part = samples;
      const plotPart = thinSignalSamples(part);
      if (!fullPlotLockedDomains) lockFullPlotDomains();
      const zDomain = fullPlotLockedDomains.z;
      const speedDomain = fullPlotLockedDomains.speed;
      const {{ x, y }} = drawAxes(fullPlot, xDomain, zDomain, W, H, margin, "height z (m)");
      const ySpeed = scale(speedDomain, [H - margin.bottom, margin.top]);
      const plotClip = installPlotClip(fullPlot, "fullPlotSignalClip", W, H, margin);
      const zValues = valuesFor("z", part);
      const speedValues = valuesFor("speed", part);
      const hasClippedHeight = zValues.some(value => value < zDomain[0] || value > zDomain[1]);
      const hasClippedSpeed = speedValues.some(value => value < speedDomain[0] || value > speedDomain[1]);
      const hasClippedValues = hasClippedHeight || hasClippedSpeed;
      actionSpans.filter(span => Number(span.end_s) >= xDomain[0] && Number(span.start_s) <= xDomain[1]).forEach(span => {{
        const visibleStart = Math.max(Number(span.start_s), xDomain[0]);
        const visibleEnd = Math.min(Number(span.end_s), xDomain[1]);
        const x1 = x(visibleStart), x2 = x(visibleEnd);
        fullPlot.appendChild(svgEl("rect", {{ x: x1, y: margin.top, width: Math.max(1, x2 - x1), height: H - margin.top - margin.bottom, fill: span.color || labelColors.missing, opacity: 0.10, "pointer-events": "none" }}));
      }});
      const filtered = filteredTransitions();
      const visibleTransitions = filtered.filter(transition => Number(transition.end_s) >= xDomain[0] && Number(transition.start_s) <= xDomain[1]);
      drawTransitionBands(fullPlot, x, margin.top, H - margin.bottom, visibleTransitions, {{ xDomain }});
      fullPlot.appendChild(svgEl("path", {{ d: linePath(plotPart, x, y, "z"), fill: "none", stroke: "#1f2c28", "stroke-width": 2.2, "clip-path": plotClip, "pointer-events": "none" }}));
      fullPlot.appendChild(svgEl("path", {{ d: linePath(plotPart, x, ySpeed, "speed"), fill: "none", stroke: "var(--orange)", "stroke-width": 1.8, "clip-path": plotClip, "pointer-events": "none" }}));
      if (activeTransitionIndex >= 0) {{
        const selected = transitions[activeTransitionIndex];
        const selectedCenter = (Number(selected.start_s || 0) + Number(selected.end_s || 0)) / 2;
        const selectedVisible = Number(selected.end_s) >= xDomain[0] && Number(selected.start_s) <= xDomain[1];
        if (selectedVisible) {{
          const selectedX = x(clamp(selectedCenter, xDomain[0], xDomain[1]));
          const selectedColor = transitionColor(selected);
          fullPlot.appendChild(svgEl("line", {{
            x1: selectedX,
            y1: margin.top,
            x2: selectedX,
            y2: H - margin.bottom,
            stroke: selectedColor,
            "stroke-width": 4,
            opacity: 0.95,
            "pointer-events": "none",
          }}));
          fullPlot.appendChild(svgEl("circle", {{ cx: selectedX, cy: margin.top + 7, r: 6, fill: selectedColor, stroke: "white", "stroke-width": 2, "pointer-events": "none" }}));
          const selectedLabel = svgEl("text", {{
            x: clamp(selectedX, margin.left + 62, W - margin.right - 62),
            y: margin.top + 23,
            "text-anchor": "middle",
            fill: selectedColor,
            "font-size": 11,
            "font-weight": 900,
            "paint-order": "stroke",
            stroke: "white",
            "stroke-width": 4,
            "pointer-events": "none",
          }});
          selectedLabel.textContent = `SELECTED #${{activeTransitionIndex + 1}}`;
          fullPlot.appendChild(selectedLabel);
        }}
      }}
      const zoomAnchorTime = fullPlotZoomAnchor();
      if (zoomAnchorTime >= xDomain[0] && zoomAnchorTime <= xDomain[1]) {{
        const currentX = x(zoomAnchorTime);
        const zoomCenterLine = svgEl("line", {{
          x1: currentX,
          y1: margin.top,
          x2: currentX,
          y2: H - margin.bottom,
          stroke: "#7b8580",
          "stroke-width": 2.4,
          opacity: 0.95,
          "pointer-events": "none",
        }});
        const zoomCenterTitle = svgEl("title");
        zoomCenterTitle.textContent = `Video time / zoom center · ${{fmtClock(zoomAnchorTime)}}`;
        zoomCenterLine.appendChild(zoomCenterTitle);
        fullPlot.appendChild(zoomCenterLine);
      }}
      const speedLabel = svgEl("text", {{ x: W - margin.right, y: 18, "text-anchor": "end", fill: "var(--orange)", "font-size": 12, "font-weight": 900 }});
      speedLabel.textContent = "orange = speed";
      fullPlot.appendChild(speedLabel);
      const selected = activeTransitionIndex >= 0 ? transitions[activeTransitionIndex] : null;
      const selectedInside = selected && Number(selected.end_s) >= xDomain[0] && Number(selected.start_s) <= xDomain[1];
      const selectedLocation = selected
        ? `Selected #${{activeTransitionIndex + 1}} is ${{selectedInside ? "inside" : "outside"}} this view at ${{fmtClock(selected.start_s)}}-${{fmtClock(selected.end_s)}}. `
        : "";
      const windowDuration = xDomain[1] - xDomain[0];
      fullPlotRangeReadout.textContent = `${{fmtClock(xDomain[0])}}-${{fmtClock(xDomain[1])}} · ${{fmtClock(windowDuration)}} visible`;
      fullPlotZoomInButton.disabled = windowDuration <= FULL_PLOT_MIN_WINDOW_S + 0.01;
      fullPlotZoomOutButton.disabled = windowDuration >= fullPlotExtentEnd - 0.01;
      fullPlotResetButton.disabled = windowDuration >= fullPlotExtentEnd - 0.01;
      const ySpan = Math.abs(zDomain[1] - zDomain[0]);
      const yDigits = ySpan < 0.2 ? 3 : (ySpan < 2 ? 2 : 1);
      fullPlotYScaleReadout.classList.toggle("clipped", hasClippedValues);
      fullPlotYScaleReadout.textContent = `Y session ${{zDomain[0].toFixed(yDigits)}}-${{zDomain[1].toFixed(yDigits)}}m${{hasClippedValues ? " · clipped" : ""}}`;
      fullPlotYScaleReadout.title = hasClippedValues
        ? "Some values fall outside the full-session scale and are clipped for display safety."
        : "The vertical scale uses the full-session minimum and maximum and does not change with time zoom or pan.";
      const samplingRate = Number(transitionSampling.display_sample_rate_hz);
      const samplingText = Number.isFinite(samplingRate)
        ? ` Interactive trace: ${{samplingRate.toFixed(1)}} samples/s.`
        : "";
      const clippingText = hasClippedValues
        ? " Some values fall outside the full-session scale and are clipped for display safety."
        : "";
      fullPlotNote.textContent = `${{selectedLocation}}Showing ${{visibleTransitions.length}} transition states in this time window (${{filtered.length}} after filters, ${{transitions.length}} total). Height and speed scales use full-session ranges and stay fixed while using +, −, or drag.${{samplingText}}${{clippingText}}`;
    }}
    function drawDetailPlot() {{
      clear(detailPlot);
      if (!samples.length || activeTransitionIndex < 0) return;
      const transition = transitions[activeTransitionIndex];
      const playbackBounds = transitionPlaybackBounds(transition);
      const radius = Number(contextSelect.value || 30);
      const center = fullPlotZoomAnchor();
      const localWindow = normalizedFullPlotWindow(center - radius, center + radius);
      const start = localWindow.start;
      const end = localWindow.end;
      const part = samples.filter(row => row.t >= start && row.t <= end);
      if (!part.length) return;
      const W = 960, H = 420;
      const margin = {{ left: 72, right: 58, top: 32, bottom: 46 }};
      const xDomain = [start, end];
      const zDomain = paddedDomain(valuesFor("z", part), 0.15);
      const speedDomain = [0, Math.max(...valuesFor("speed", part), 0.1) * 1.2];
      const pathStart = Number(part[0].path || 0);
      const detailRows = part.map(row => ({{ ...row, path_delta: Number(row.path || 0) - pathStart }}));
      const pathDomain = [0, Math.max(...valuesFor("path_delta", detailRows), 0.08) * 1.2];
      const {{ x, y }} = drawAxes(detailPlot, xDomain, zDomain, W, H, margin, "local height z (m)");
      const ySpeed = scale(speedDomain, [H - margin.bottom, margin.top]);
      const yPath = scale(pathDomain, [H - margin.bottom, margin.top]);
      const plotClip = installPlotClip(detailPlot, "detailPlotSignalClip", W, H, margin);
      actionSpans.filter(span => span.end_s >= start && span.start_s <= end).forEach(span => {{
        const x1 = x(Math.max(span.start_s, start)), x2 = x(Math.min(span.end_s, end));
        detailPlot.appendChild(svgEl("rect", {{ x: x1, y: margin.top, width: Math.max(1, x2 - x1), height: H - margin.top - margin.bottom, fill: span.color || labelColors.missing, opacity: 0.10 }}));
      }});
      const localTransitions = filteredTransitions().filter(item => Number(item.end_s) >= start && Number(item.start_s) <= end);
      drawTransitionBands(detailPlot, x, margin.top, H - margin.bottom, localTransitions, {{ xDomain }});
      detailPlot.appendChild(svgEl("path", {{ d: linePath(detailRows, x, y, "z"), fill: "none", stroke: "#1f2c28", "stroke-width": 2.8, "clip-path": plotClip }}));
      detailPlot.appendChild(svgEl("path", {{ d: linePath(detailRows, x, ySpeed, "speed"), fill: "none", stroke: "var(--orange)", "stroke-width": 1.4, opacity: 0.38, "clip-path": plotClip }}));
      detailPlot.appendChild(svgEl("path", {{ d: linePath(detailRows, x, yPath, "path_delta"), fill: "none", stroke: "var(--green)", "stroke-width": 1.4, opacity: 0.42, "clip-path": plotClip }}));
      const centerX = x(center);
      const detailCenterLine = svgEl("line", {{
        x1: centerX,
        y1: margin.top,
        x2: centerX,
        y2: H - margin.bottom,
        stroke: "#7b8580",
        "stroke-width": 2.4,
        opacity: 0.95,
        "pointer-events": "none",
      }});
      const detailCenterTitle = svgEl("title");
      detailCenterTitle.textContent = `Shared video time · ${{fmtClock(center)}}`;
      detailCenterLine.appendChild(detailCenterTitle);
      detailPlot.appendChild(detailCenterLine);
      const legend = svgEl("text", {{ x: W - margin.right, y: 18, "text-anchor": "end", fill: "#33423b", "font-size": 12, "font-weight": 900 }});
      legend.textContent = "black=local height, faint orange=speed, faint green=X/Y path";
      detailPlot.appendChild(legend);
      detailPlot.onclick = event => {{
        const rect = detailPlot.getBoundingClientRect();
        const ratio = clamp((event.clientX - rect.left) / Math.max(rect.width, 1), 0, 1);
        syncVideoTime(start + ratio * (end - start));
      }};
      const localYSpan = Math.abs(zDomain[1] - zDomain[0]);
      const localYDigits = localYSpan < 0.2 ? 3 : (localYSpan < 2 ? 2 : 1);
      detailPlotYScaleReadout.textContent = `Local Y ${{zDomain[0].toFixed(localYDigits)}}-${{zDomain[1].toFixed(localYDigits)}}m`;
      detailPlotYScaleReadout.title = `Automatically scaled from ${{fmtClock(start)}} to ${{fmtClock(end)}}; raw SLAM values are unchanged.`;
      selectedTransitionPill.textContent = `${{transitionArrowLabel(transition)}} · ${{fmtTime(transition.start_s)}}-${{fmtTime(transition.end_s)}}`;
      selectedTitle.textContent = `Video time ${{fmtClock(center)}} · local context ${{fmtClock(start)}}-${{fmtClock(end)}}`;
      selectedMetrics.innerHTML = `
        <span>time ${{fmtTime(transition.start_s)}}-${{fmtTime(transition.end_s)}}</span>
        <span>video review ${{fmtTime(playbackBounds.start)}}-${{fmtTime(playbackBounds.end)}}</span>
        <span>duration ${{fmtMetric(transition.duration_s, 2, "s")}}</span>
        <span style="color:${{classificationColor(transition)}};border-color:${{classificationColor(transition)}}">${{escapeHtml(transition.classification_label || transition.likelihood_label || "Needs Review")}}</span>
        <span>height change ${{fmtMetric(transition.height_change_m, 3, "m")}}</span>
        <span>X/Y path ${{fmtMetric(transition.xy_path_change_m, 3, "m")}}</span>
        <span>peak speed ${{fmtMetric(transition.peak_speed_mps, 2, "m/s")}}</span>
        <span>${{escapeHtml(transition.confidence || "review")}} confidence</span>
        <span>source: ${{escapeHtml(String(transition.source || "unknown").replaceAll("_", " "))}}</span>
      `;
      selectedParameterDetails.innerHTML = `<b>${{escapeHtml(transition.parameter_signal || "SLAM context")}} profile:</b> ${{formatDetectionParameters(transition)}}`;
      detailPlotNote.textContent = `This window follows the gray video playhead at ${{fmtClock(center)}}. The left Y-axis automatically fits the surrounding ${{fmtClock(end - start)}} so smaller height fluctuations remain visible. Raw SLAM values are unchanged; transition ranges use background color only.`;
    }}
    function buildTransitionSelect() {{
      transitionSelect.innerHTML = "";
      const visible = filteredTransitions();
      visible.forEach(transition => {{
        const index = transitions.indexOf(transition);
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = transitionOptionLabel(transition, index);
        transitionSelect.appendChild(option);
      }});
      transitionSelect.disabled = visible.length === 0;
      if (activeTransitionIndex >= 0) transitionSelect.value = String(activeTransitionIndex);
    }}
    function buildTypeFilter() {{
      const types = Array.from(new Set(transitions.map(item => String(item.transition_type || "")).filter(Boolean))).sort();
      transitionTypeFilter.innerHTML = `<option value="all">All transition types</option>`;
      types.forEach(type => {{
        const option = document.createElement("option");
        option.value = type;
        option.textContent = type.replaceAll("-to-", " → ").replaceAll("-", " ");
        transitionTypeFilter.appendChild(option);
      }});
      transitionTypeFilter.value = activeTransitionType;
    }}
    function renderClassificationBar() {{
      const counts = {{
        all: transitions.length,
        clinical: transitions.filter(item => transitionClassification(item) === "clinical").length,
        possible: transitions.filter(item => transitionClassification(item) === "possible").length,
        less_possible: transitions.filter(item => transitionClassification(item) === "less_possible").length,
      }};
      document.getElementById("allClassificationCount").textContent = `(${{counts.all}})`;
      document.getElementById("clinicalClassificationCount").textContent = `(${{counts.clinical}})`;
      document.getElementById("possibleClassificationCount").textContent = `(${{counts.possible}})`;
      document.getElementById("lessPossibleClassificationCount").textContent = `(${{counts.less_possible}})`;
      classificationButtons.forEach(button => button.classList.toggle("active", button.dataset.classification === activeClassification));
    }}
    function applyTransitionFilters() {{
      const visible = filteredTransitions();
      let activeChanged = false;
      if (visible.length && !visible.includes(transitions[activeTransitionIndex])) {{
        activeTransitionIndex = transitions.indexOf(visible[0]);
        currentTime = transitionPlaybackBounds(visible[0]).start;
        activeChanged = true;
      }} else if (!visible.length) {{
        activeTransitionIndex = -1;
        activeChanged = true;
      }}
      buildTransitionSelect();
      if (activeChanged && activeTransitionIndex >= 0) selectTransition(activeTransitionIndex, true);
      else renderDynamic();
    }}
    function renderSummary() {{
      const clinical = transitions.filter(t => transitionClassification(t) === "clinical").length;
      const possible = transitions.filter(t => transitionClassification(t) === "possible").length;
      const lessPossible = transitions.filter(t => transitionClassification(t) === "less_possible").length;
      summaryGrid.innerHTML = `
        <div class="summary-card"><b>Total</b><span>${{transitions.length}}</span></div>
        <div class="summary-card"><b>Clinical</b><span style="color:#16815d">${{clinical}}</span></div>
        <div class="summary-card"><b>Possible / Context</b><span style="color:#d48716">${{possible}}</span></div>
        <div class="summary-card"><b>Less Possible</b><span style="color:#c026d3">${{lessPossible}}</span></div>
      `;
    }}
    function renderList() {{
      transitionList.innerHTML = "";
      const visible = filteredTransitions();
      if (!visible.length) {{
        transitionList.innerHTML = `<p>No transitions match both selected filters.</p>`;
        return;
      }}
      visible.forEach(transition => {{
        const index = transitions.indexOf(transition);
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.transitionIndex = String(index);
        button.className = `transition-item ${{index === activeTransitionIndex ? "active" : ""}}`;
        button.style.setProperty("--transition-color", transitionColor(transition));
        button.style.borderLeft = `7px solid ${{classificationColor(transition)}}`;
        button.innerHTML = `
          <strong>${{escapeHtml(transitionArrowLabel(transition))}}</strong>
          <span class="transition-meta">
            <span class="pill" style="border-color:${{transitionColor(transition)}}">${{fmtTime(transition.start_s)}}-${{fmtTime(transition.end_s)}} · ${{fmtMetric(transition.duration_s, 2, "s")}}</span>
            <span class="pill" style="color:${{classificationColor(transition)}};border-color:${{classificationColor(transition)}}">${{escapeHtml(transition.classification_label || transition.likelihood_label || "Needs Review")}}</span>
          </span>
        `;
        button.addEventListener("click", () => selectTransition(index, true));
        transitionList.appendChild(button);
      }});
    }}
    function renderReviewList() {{
      reviewList.innerHTML = "";
      const visibleReviews = filteredTransitions().filter(item => transitionClassification(item) === "less_possible");
      if (!visibleReviews.length) {{
        reviewList.innerHTML = `<p>No unusual direct transitions are currently recorded.</p>`;
        return;
      }}
      visibleReviews.forEach(transition => {{
        const index = transitions.findIndex(item => item.transition_id === transition.transition_id);
        const button = document.createElement("button");
        button.type = "button";
        button.className = "review-item";
        button.innerHTML = `<strong>${{escapeHtml(transitionArrowLabel(transition))}}</strong><br><span>${{fmtTime(transition.start_s)}}-${{fmtTime(transition.end_s)}} · ${{escapeHtml(transition.clinical_interest || "needs context review")}}</span>`;
        button.addEventListener("click", () => selectTransition(Math.max(0, index), true));
        reviewList.appendChild(button);
      }});
    }}
    function renderDynamic() {{
      scrubber.value = String(clamp(currentTime, 0, sessionDuration));
      if (transitionSelect && String(activeTransitionIndex) !== transitionSelect.value) {{
        transitionSelect.value = String(activeTransitionIndex);
      }}
      timeReadout.textContent = `${{fmtTime(currentTime)}} / ${{fmtTime(sessionDuration)}}`;
      renderGlobalLocator();
      drawFullPlot();
      drawDetailPlot();
      if (activeTransitionIndex < 0) {{
        selectedTransitionPill.textContent = "No matching transition";
        selectedTitle.textContent = "Adjust the classification or type filter";
        selectedMetrics.innerHTML = "";
        selectedParameterDetails.textContent = "No transition matches both selected filters.";
        detailPlotYScaleReadout.textContent = "Local Y auto";
        detailPlotNote.textContent = "";
      }}
      renderSummary();
      renderClassificationBar();
      renderList();
      renderReviewList();
    }}
    function init() {{
      buildTrialJump();
      downwardVideo.src = dashboardData.downward_session_video_uri;
      downwardVideo.defaultMuted = true;
      downwardVideo.muted = true;
      ariaVideo.src = dashboardData.aria_session_video_uri;
      restoreAudioPreference();
      applyRotations();
      buildGlobalActionTrack();
      if (!transitionReview.available || !transitions.length) {{
        workspace.style.display = "none";
        emptyState.style.display = "block";
        emptyState.innerHTML = `<h2>No transition review windows for this trial yet</h2><p>${{escapeHtml(transitionReview.source_note || "No transitions detected.")}}</p>`;
        return;
      }}
      buildTypeFilter();
      buildTransitionSelect();
      selectTransition(0, true);
      startSyncTimer();
    }}
    fullPlotZoomInButton.addEventListener("click", () => zoomFullPlot(0.5));
    fullPlotZoomOutButton.addEventListener("click", () => zoomFullPlot(2));
    fullPlotFifteenMinuteButton.addEventListener("click", showFifteenMinuteWindow);
    fullPlotResetButton.addEventListener("click", resetFullPlotWindow);
    fullPlot.addEventListener("pointerdown", event => {{
      if (event.button !== 0) return;
      fullPlotDragState = {{
        pointerId: event.pointerId,
        startClientX: event.clientX,
        viewStart: fullPlotViewStart,
        viewEnd: fullPlotViewEnd,
        moved: false,
      }};
      try {{ fullPlot.setPointerCapture(event.pointerId); }} catch (error) {{}}
    }});
    fullPlot.addEventListener("pointermove", event => {{
      if (!fullPlotDragState || event.pointerId !== fullPlotDragState.pointerId) return;
      const dx = event.clientX - fullPlotDragState.startClientX;
      if (Math.abs(dx) > 3) fullPlotDragState.moved = true;
      if (!fullPlotDragState.moved) return;
      event.preventDefault();
      fullPlot.classList.add("dragging");
      const rect = fullPlot.getBoundingClientRect();
      const plotPixelWidth = Math.max(1, rect.width * (FULL_PLOT_WIDTH - FULL_PLOT_MARGIN.left - FULL_PLOT_MARGIN.right) / FULL_PLOT_WIDTH);
      const viewWidth = fullPlotDragState.viewEnd - fullPlotDragState.viewStart;
      const shift = -dx / plotPixelWidth * viewWidth;
      setFullPlotWindow(
        fullPlotDragState.viewStart + shift,
        fullPlotDragState.viewEnd + shift,
        {{ keepAnchorVisible: true }},
      );
    }});
    function finishFullPlotPointer(event, cancelled = false) {{
      if (!fullPlotDragState || event.pointerId !== fullPlotDragState.pointerId) return;
      const wasMoved = fullPlotDragState.moved;
      try {{ fullPlot.releasePointerCapture(event.pointerId); }} catch (error) {{}}
      fullPlotDragState = null;
      fullPlot.classList.remove("dragging");
      if (!cancelled && !wasMoved) {{
        const targetTime = fullPlotEventTime(event);
        syncVideoTime(targetTime);
      }}
    }}
    fullPlot.addEventListener("pointerup", event => finishFullPlotPointer(event));
    fullPlot.addEventListener("pointercancel", event => finishFullPlotPointer(event, true));
    playTransitionButton.addEventListener("click", () => playSelectedRange("transition"));
    playPauseButton.addEventListener("click", () => playSelectedRange("context"));
    audioToggleButton.addEventListener("click", () => {{
      setAudioEnabled(ariaVideo.muted || Number(ariaVideo.volume) <= 0);
    }});
    prevButton.addEventListener("click", () => selectAdjacentTransition(-1));
    nextButton.addEventListener("click", () => selectAdjacentTransition(1));
    transitionSelect.addEventListener("change", event => selectTransition(Number(event.target.value || 0), true));
    classificationButtons.forEach(button => button.addEventListener("click", () => {{
      activeClassification = String(button.dataset.classification || "all");
      applyTransitionFilters();
    }}));
    transitionTypeFilter.addEventListener("change", event => {{
      activeTransitionType = String(event.target.value || "all");
      applyTransitionFilters();
    }});
    rotateDownwardButton.addEventListener("click", () => {{ downwardTurns += 1; applyRotations(); }});
    rotateAriaButton.addEventListener("click", () => {{ ariaTurns += 1; applyRotations(); }});
    contextSelect.addEventListener("change", () => renderDynamic());
    scrubber.addEventListener("input", event => syncVideoTime(Number(event.target.value || 0)));
    ariaVideo.addEventListener("timeupdate", () => {{
      if (!ariaVideo.paused) {{
        currentTime = clamp(ariaVideo.currentTime || 0, 0, sessionDuration);
        keepPlaybackVisibleInFullPlot();
        if (activeReviewRange && currentTime >= activeReviewRange.end - 0.02) {{
          currentTime = activeReviewRange.end;
          pauseBoth();
          [downwardVideo, ariaVideo].forEach(video => {{
            try {{ video.currentTime = currentTime; }} catch (error) {{}}
          }});
          renderDynamic();
          return;
        }}
        if (!activeReviewRange) {{
          const nearest = nearestTransitionIndex(currentTime);
          if (nearest >= 0 && Math.abs(Number(transitions[nearest].center_s) - currentTime) < 1.0) {{
            activeTransitionIndex = nearest;
          }}
        }}
        renderDynamic();
      }}
    }});
    ariaVideo.addEventListener("play", () => {{ updatePlaybackButtons(); startSyncTimer(); }});
    ariaVideo.addEventListener("pause", () => {{ downwardVideo.pause(); updatePlaybackButtons(); }});
    downwardVideo.addEventListener("click", handleVideoClick);
    ariaVideo.addEventListener("click", handleVideoClick);
    init();
  </script>
</body>
</html>
"""


def build_trial(trial_id: str) -> bool:
    run_dir = RUNS_DIR / trial_id
    manifest_path = run_dir / "trial_manifest.json"
    sync_path = run_dir / "downward_sync" / "sync_suggestion.json"
    if not manifest_path.exists():
        return False

    manifest = load_json(manifest_path)
    if manifest.get("downward_attachment"):
        # These media were attached to an established Aria clock. The older
        # overlap-trimming workflow would invalidate labels, strides and 3D poses.
        from attach_downward_to_existing_trial import refresh_attached_views
        refresh_attached_views(trial_id)
        return True
    aria_only = not sync_path.exists()
    if aria_only and manifest.get("media_mode") != "aria_only":
        return False
    sync_data = load_json(sync_path) if not aria_only else {
        "aria_audio": str(Path(manifest["extracted_media"]["video_mp4"]["path"]).parent / "audio_track.wav"),
        "shared_duration_s": video_duration_s(Path(manifest["extracted_media"]["video_mp4"]["path"])),
    }
    simulated = manifest.get("trial_type") == "parkinson_simulated"
    if not trial_id.startswith("Ntrial") and trial_id not in ADDITIONAL_NORMAL_TRIALS and not simulated:
        return False

    extracted_video_path = Path(str(manifest["extracted_media"]["video_mp4"]["path"]))
    trajectory_csv = Path(str(manifest["mps_outputs"]["closed_loop_trajectory_csv"]["path"]))
    eye_gaze_csv = Path(str(manifest["mps_outputs"]["eye_gaze_csv"]["path"]))
    vrs_json_path = Path(str(manifest["raw_data"]["vrs_json"]["path"]))
    downward_video_path = Path(str(sync_data["downward_video"])) if not aria_only else None
    aria_audio_path = Path(str(sync_data["aria_audio"]))

    required = [extracted_video_path, trajectory_csv, vrs_json_path, aria_audio_path]
    if downward_video_path:
        required.append(downward_video_path)
    if not all(path.exists() for path in required):
        return False

    ffmpeg_exe = resolve_ffmpeg_executable()

    clip_seconds = infer_clip_seconds(run_dir)
    extracted_video_fps, _, _, _ = load_video_info(extracted_video_path)
    downward_video_fps = load_video_info(downward_video_path)[0] if downward_video_path else extracted_video_fps
    aria_trim_start_s = quantize_time_to_frame(float(sync_data.get("aria_trim_start_s", 0.0)), extracted_video_fps)
    downward_trim_start_s = quantize_time_to_frame(float(sync_data.get("downward_trim_start_s", 0.0)), downward_video_fps)
    shared_duration_s = float(sync_data.get("shared_duration_s", 0.0))
    aria_available_duration_s = max(0.0, video_duration_s(extracted_video_path) - aria_trim_start_s)
    downward_available_duration_s = max(0.0, video_duration_s(downward_video_path) - downward_trim_start_s) if downward_video_path else 0.0
    if aria_only:
        effective_duration_s = aria_available_duration_s
    elif trial_id in ARIA_TAIL_EXTENDED_TRIALS:
        # Half_hour2 has a shorter downward recording. Keep the aligned start,
        # but let the clinician continue reviewing the remaining Aria session.
        effective_duration_s = floor_to_clip_duration(aria_available_duration_s, clip_seconds)
    else:
        effective_duration_s = floor_to_clip_duration(shared_duration_s, clip_seconds)
    if effective_duration_s < clip_seconds:
        return False

    output_dir = run_dir / "downward_master_session"
    output_dir.mkdir(parents=True, exist_ok=True)
    recording_stem = str(manifest.get("recording_stem", run_dir.name))

    audio_path_from_video, audio_offset_s = discover_audio_companion(extracted_video_path)
    if audio_path_from_video and audio_path_from_video.exists():
        aria_audio_path = audio_path_from_video
    aria_audio_trim_start_s = max(0.0, aria_trim_start_s + audio_offset_s)

    downward_session_video_path = output_dir / "downward_session_video.mp4"
    aria_session_video_path = output_dir / "aria_session_video.mp4"
    if ffmpeg_exe:
        downward_session_video = ensure_downward_session_video(
            ffmpeg_exe,
            downward_video_path,
            downward_session_video_path,
            output_dir / "downward_session_video.build.json",
            trim_start_s=downward_trim_start_s,
            duration_s=min(effective_duration_s, downward_available_duration_s),
        ) if downward_video_path else None
        aria_session_video = ensure_aria_session_video(
            ffmpeg_exe,
            extracted_video_path,
            aria_audio_path,
            aria_session_video_path,
            output_dir / "aria_session_video.build.json",
            video_trim_start_s=aria_trim_start_s,
            audio_trim_start_s=aria_audio_trim_start_s,
            duration_s=effective_duration_s,
        )
    elif downward_session_video_path.exists() and aria_session_video_path.exists():
        downward_session_video = downward_session_video_path
        aria_session_video = aria_session_video_path
    else:
        raise RuntimeError("Unable to locate ffmpeg and no reusable session videos exist.")

    windows = build_clip_windows(output_dir, clip_seconds, effective_duration_s)
    trajectory_df = pd.read_csv(trajectory_csv)
    alignment_offset_s = estimate_alignment_offset_s(trajectory_df, eye_gaze_csv if eye_gaze_csv.exists() else None)
    session_start_tracking_us = None
    if simulated and manifest.get("slam_time_alignment") != "legacy_session_clock":
        audio_metadata = load_json(extracted_video_path.parent / "audio_metadata.json")
        video_start_us = float(audio_metadata["video_start_device_ns"]) / 1000.0
        alignment_offset_s = (float(trajectory_df["tracking_timestamp_us"].iloc[0]) - video_start_us) / 1e6
        session_start_tracking_us = video_start_us + aria_trim_start_s * 1e6
    stride_signal = build_stride_signal_data(
        trial_id,
        trajectory_df,
        alignment_offset_s - aria_trim_start_s,
        effective_duration_s,
    )
    session_start_epoch_s = load_start_time(vrs_json_path) + aria_trim_start_s
    if session_start_tracking_us is not None:
        first_row = trajectory_df.iloc[0]
        session_start_epoch_s = float(first_row["utc_timestamp_ns"]) / 1e9 + (session_start_tracking_us - float(first_row["tracking_timestamp_us"])) / 1e6
    clip_df = build_clip_metrics(
        windows,
        session_start_epoch_s,
        trajectory_df,
        alignment_offset_s - aria_trim_start_s,
    )
    clip_df = apply_hybrid_predictions(clip_df, None, ["sitting"])
    clip_df = attach_spoken_action_columns(trial_id, clip_df)
    clip_df = attach_latest_criteria_predictions(trial_id, clip_df)
    if simulated:
        from build_parkinson_motion_review import build_parkinson_review
        build_parkinson_review(trial_id, session_manifest={
            "session_duration_s": effective_duration_s,
            "recording_start_epoch_s": session_start_epoch_s,
            "session_start_tracking_us": session_start_tracking_us,
            "downward_session_video_uri": "downward_master_session/downward_session_video.mp4" if not aria_only else "",
            "aria_session_video_uri": "downward_master_session/aria_session_video.mp4",
        })
        continuous = load_continuous_motion_data(trial_id, effective_duration_s)
        if not continuous["available"]:
            raise ValueError(f"{trial_id}: Parkinson continuous timeline is incomplete")
        for index, clip in clip_df.iterrows():
            overlaps = [(max(0.0, min(float(clip.end_s), s["end_s"]) - max(float(clip.start_s), s["start_s"])), s["label"]) for s in continuous["segments"]]
            label = max(overlaps, key=lambda pair: pair[0])[1]
            clip_df.loc[index, ["rule_prediction", "hybrid_prediction"]] = label
        clip_df["model_prediction_source"] = "parkinson_high_v1"
    clip_df.to_csv(output_dir / "downward_session_clip_metrics.csv", index=False)

    model_segments = build_model_segments(clip_df)
    if simulated:
        model_segments = [{**segment, "color": color_for_label(segment["label"]),
                           "clip_start_idx": int(segment["start_s"] // clip_seconds),
                           "clip_end_idx": min(len(clip_df) - 1, int(max(0, segment["end_s"] - 1e-6) // clip_seconds))}
                          for segment in continuous["segments"]]
    pd.DataFrame(model_segments).to_csv(output_dir / "downward_session_rule_segments.csv", index=False)
    save_json(output_dir / "downward_session_rule_segments.json", {"segments": model_segments})

    retired_clip_review_path = run_dir / CLIP_REVIEW_DASHBOARD_NAME
    retired_clip_review_path.write_text(build_legacy_redirect_html(CANONICAL_TRIAL_ENTRY_NAME))

    time_spent_records = build_time_spent_records(clip_df, clip_seconds)
    downward_session_duration_s = video_duration_s(downward_session_video) if downward_session_video else 0.0
    aria_session_duration_s = video_duration_s(aria_session_video)
    session_fps, session_frame_count, _, _ = load_video_info(aria_session_video)
    transition_candidates = build_transition_candidates(
        clip_df,
        model_segments,
        session_duration_s=effective_duration_s,
    )
    transition_review = build_transition_review_data(
        trial_id,
        clip_df,
        trajectory_df,
        alignment_offset_s - aria_trim_start_s,
        effective_duration_s,
        automatic_segments=continuous["segments"] if simulated else None,
    )
    session_payload = build_session_payload(
        trial_id=trial_id,
        manifest=manifest,
        clip_df=clip_df,
        model_segments=model_segments,
        time_spent_records=time_spent_records,
        output_dir=run_dir,
        downward_session_video=downward_session_video,
        aria_session_video=aria_session_video,
        clip_seconds=clip_seconds,
        effective_duration_s=effective_duration_s,
        session_fps=float(session_fps),
        session_frame_count=int(session_frame_count),
        transition_candidates=transition_candidates,
        stride_signal=stride_signal,
        transition_review=transition_review,
    )
    session_payload["downward_session_duration_s"] = downward_session_duration_s
    session_payload["aria_session_duration_s"] = aria_session_duration_s
    session_payload["shared_overlap_duration_s"] = floor_to_clip_duration(shared_duration_s, clip_seconds) if not aria_only else 0.0
    session_payload["media_mode"] = "aria_only" if aria_only else "dual_view"
    session_payload["time_information"] = load_or_build_time_information(trial_id, manifest, sync_data) if not aria_only else {"available": False}
    session_payload["full_session_href"] = CANONICAL_TRIAL_ENTRY_NAME
    session_payload["trial_type"] = manifest.get("trial_type", "unspecified")
    if session_start_tracking_us is not None:
        session_payload["session_start_tracking_us"] = session_start_tracking_us
    session_payload["manual_labeling_href"] = MANUAL_LABELING_DASHBOARD_NAME
    session_payload["stride_review_href"] = STRIDE_REVIEW_DASHBOARD_NAME
    if transition_review.get("available") and trial_id not in TRANSITION_REVIEW_TRIAL_EXCLUDE:
        session_payload["transition_review_href"] = TRANSITION_REVIEW_DASHBOARD_NAME
    save_json(run_dir / "session_dashboard_manifest.json", session_payload)

    stride_review_path = run_dir / STRIDE_REVIEW_DASHBOARD_NAME
    stride_review_path.write_text(build_stride_review_html(session_payload))
    transition_review_path = run_dir / TRANSITION_REVIEW_DASHBOARD_NAME
    if transition_review.get("available") and trial_id not in TRANSITION_REVIEW_TRIAL_EXCLUDE:
        transition_review_path.write_text(build_transition_review_html(session_payload))
    else:
        transition_review_path.write_text(build_legacy_redirect_html(CANONICAL_TRIAL_ENTRY_NAME))

    dashboard_html = build_dashboard_html(session_payload)
    canonical_path = run_dir / CANONICAL_TRIAL_ENTRY_NAME
    canonical_path.write_text(dashboard_html)
    for legacy_name in [FULL_SESSION_DASHBOARD_NAME, LEGACY_DASHBOARD_NAME, LEGACY_RULE_DASHBOARD_NAME]:
        legacy_path = run_dir / legacy_name
        legacy_path.write_text(build_legacy_redirect_html(CANONICAL_TRIAL_ENTRY_NAME))
    print(f"Built full-session clinician dashboard: {trial_id}")
    return True


def main() -> None:
    args = parse_args()
    built = 0
    built_trial_ids: list[str] = []
    for trial_id in args.trials:
        if build_trial(trial_id):
            built += 1
            built_trial_ids.append(trial_id)
    print(f"Built {built} full-session dashboard(s).")
    if built_trial_ids:
        manual_labeling_cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "build_full_session_manual_labeling.py"),
            "--trials",
            *built_trial_ids,
        ]
        subprocess.run(manual_labeling_cmd, check=True)
    selector_cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "build_pd_insighter_trial_selector.py"),
    ]
    subprocess.run(selector_cmd, check=True)


if __name__ == "__main__":
    main()
