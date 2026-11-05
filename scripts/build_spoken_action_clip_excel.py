#!/usr/bin/env python3
"""Extract spoken action words from synced audio and map them onto clip windows.

This script is designed for the downward-synced Ntrial dashboards, but it also
works on regular analyzed trials as a fallback.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = WORKSPACE_ROOT / "Qwen2.5-VL-fork" / "runs"
DEFAULT_WORKBOOK_PATH = RUNS_DIR / "spoken_action_clip_labels.xlsx"
MANUAL_LABELS_DIRNAME = "manual_labels"

ACTION_LABELS = ("walking", "sitting", "standing", "upstairs", "downstairs")
RARE_ACTION_LABELS = {"upstairs", "downstairs"}
CLINICAL_TRANSITION_LABELS = {
    ("sitting", "standing"): "sit-to-stand",
    ("standing", "sitting"): "stand-to-sit",
    ("standing", "walking"): "stand-to-walk",
    ("walking", "standing"): "walk-to-stand",
}
ANALYSIS_VERSION = 5
MIN_EVENT_DURATION_S = 0.05
SINGLE_TOKEN_ALIASES = {
    "walking": "walking",
    "walkin": "walking",
    "walk": "walking",
    "working": "walking",
    "waking": "walking",
    "sitting": "sitting",
    "sittin": "sitting",
    "sit": "sitting",
    "seating": "sitting",
    "setting": "sitting",
    "siting": "sitting",
    "standing": "standing",
    "standin": "standing",
    "stand": "standing",
    "stating": "standing",
    "staring": "standing",
    "upstairs": "upstairs",
    "upstair": "upstairs",
    "upstares": "upstairs",
    "downstairs": "downstairs",
    "downstair": "downstairs",
    "downstares": "downstairs",
}

FUZZY_PHRASE_ALIASES = {
    "up stairs": "upstairs",
    "upstair": "upstairs",
    "up stare": "upstairs",
    "down stairs": "downstairs",
    "downstair": "downstairs",
    "down stare": "downstairs",
    "walk in": "walking",
}

BLOCKED_FUZZY_CLEANED_TEXT = {
    "stay",
    "stayed",
    "stayin",
    "stayedin",
    "staytuned",
    "tuned",
}

MANUAL_EXPORT_COLUMNS = [
    "clip_idx",
    "manual_correction_mode",
    "manual_correction_label",
    "manual_correction_action_count",
    "manual_last_updated",
    "manual_action_1",
    "manual_action_2",
    "manual_action_3",
    "manual_action_4",
    "manual_action_5",
]

OVERVIEW_COLUMNS = [
    "trial_id",
    "clip_count",
    "clip_coverage_s",
    "synced_audio_window_s",
    "detected_target_word_count",
    "detected_state_count",
    "manual_override_count",
    "manual_transition_override_count",
    "comparable_clip_count",
    "stable_only_comparable_clip_count",
    "transition_clip_comparable_count",
    "strict_accuracy",
    "transition_aware_accuracy",
    "component_transition_aware_accuracy",
    "stable_only_accuracy",
    "transition_clip_accuracy",
    "audio_trim_start_s",
    "audio_source",
    "model",
]

REVIEW_EXPORT_COLUMNS = [
    "clip_idx",
    "clip_name",
    "clip_start_s",
    "clip_end_s",
    "predicted_label",
    "model_prediction_normalized",
    "spoken_words_detected_in_clip",
    "spoken_word_event_count",
    "spoken_word_times_s",
    "automatic_spoken_action",
    "assumed_actions_in_clip",
    "primary_assumed_action",
    "is_transition_clip",
    "state_change_times_in_clip_s",
    "manual_correction_mode",
    "manual_correction_label",
    "manual_correction_action_count",
    "manual_is_transition",
    "manual_last_updated",
    "effective_label_source",
    "effective_manual_label",
    "effective_manual_action_count",
    "effective_is_transition",
    "effective_transition_target_label",
    "agreement_status",
    "model_exact_match_effective_manual",
    "model_matches_effective_manual",
    "model_matches_any_effective_action",
    "horizontal_displacement_m",
    "vertical_displacement_m",
    "mean_linear_speed_mps",
    "mean_angular_speed_rps",
    "mean_z_m",
    "manual_action_1",
    "manual_action_2",
    "manual_action_3",
    "manual_action_4",
    "manual_action_5",
    "effective_action_1",
    "effective_action_2",
    "effective_action_3",
    "effective_action_4",
    "effective_action_5",
]


@dataclass
class WordEvent:
    label: str
    start_s: float
    end_s: float
    probability: float
    raw_text: str


@dataclass
class LabelRun:
    label: str
    start_s: float
    end_s: float
    support_count: int
    support_words: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        nargs="+",
        required=True,
        help="Trial ids to process, e.g. Ntrial1 Ntrial2 Ntrial3",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RUNS_DIR / "spoken_action_clip_labels.xlsx",
        help="Workbook output path.",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/whisper-tiny",
        help="MLX Whisper model repo or local path.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Candidate count for the enhanced fallback transcription pass.",
    )
    parser.add_argument(
        "--patience",
        type=float,
        default=1.2,
        help="Reserved for future decoder tuning; kept in the saved transcription config.",
    )
    parser.add_argument(
        "--segment-fuzzy-threshold",
        type=float,
        default=0.74,
        help="Similarity threshold for recovering target action words from segment text.",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        help="Keep Whisper conditioning on previous text. By default this is disabled for repeated action words.",
    )
    parser.add_argument(
        "--force-transcribe",
        action="store_true",
        help="Ignore cached trial transcription outputs and rerun transcription.",
    )
    parser.add_argument(
        "--min-stray-run-count",
        type=int,
        default=1,
        help="Runs at or below this count can be merged away if they look like isolated stray labels.",
    )
    parser.add_argument(
        "--max-stray-run-duration-s",
        type=float,
        default=1.25,
        help="Maximum run duration for merging an isolated stray label between two matching neighbors.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_manifest(trial_id: str) -> dict[str, Any]:
    manifest_path = RUNS_DIR / trial_id / "trial_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing trial manifest: {manifest_path}")
    return load_json(manifest_path)


def discover_trial_paths(trial_id: str) -> dict[str, Path]:
    run_dir = RUNS_DIR / trial_id
    downward_dir = run_dir / "downward_master_session"
    sync_json = run_dir / "downward_sync" / "sync_suggestion.json"

    if downward_dir.exists():
        clip_csv = downward_dir / "downward_session_clip_metrics.csv"
    else:
        clip_csv = run_dir / "clip_metrics.csv"
    if not clip_csv.exists():
        raise FileNotFoundError(f"Missing clip metrics CSV for {trial_id}: {clip_csv}")

    spoken_dir = run_dir / "spoken_action_labels"
    spoken_dir.mkdir(parents=True, exist_ok=True)
    manual_dir = run_dir / MANUAL_LABELS_DIRNAME
    manual_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "clip_csv": clip_csv,
        "sync_json": sync_json,
        "spoken_dir": spoken_dir,
        "manual_labels_path": manual_dir / "manual_clip_labels.json",
    }


def discover_audio_window(trial_id: str, manifest: dict[str, Any], sync_json: Path, clip_rows: list[dict[str, str]]) -> tuple[Path, float, float]:
    if manifest.get("downward_attachment"):
        audio_path = Path(str(manifest["extracted_media"]["video_mp4"]["path"])).parent / "audio_track.wav"
        trim_start_s = 0.0
        duration_s = float(manifest["downward_attachment"]["session_duration_s"])
    elif sync_json.exists():
        sync = load_json(sync_json)
        audio_path = Path(sync["aria_audio"])
        trim_start_s = float(sync.get("aria_trim_start_s", 0.0))
        duration_s = float(sync.get("shared_duration_s", 0.0))
    else:
        audio_path = Path(str(manifest["extracted_media"]["video_mp4"]["path"])).parent / "audio_track.wav"
        trim_start_s = 0.0
        duration_s = max(float(row["end_s"]) for row in clip_rows) if clip_rows else 0.0

    if not audio_path.exists():
        raise FileNotFoundError(f"Missing extracted Aria audio for {trial_id}: {audio_path}")
    return audio_path, trim_start_s, duration_s


def load_clip_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def apply_criteria_prediction_overrides(trial_id: str, clip_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    comparison_path = RUNS_DIR / trial_id / "raw_parameter_diagnostics" / "clip_feature_rule_comparison.csv"
    if not comparison_path.exists():
        return clip_rows

    with comparison_path.open(newline="") as handle:
        comparison_rows = list(csv.DictReader(handle))
    prediction_by_clip: dict[int, dict[str, str]] = {}
    for row in comparison_rows:
        try:
            clip_idx = int(row.get("clip_idx", ""))
        except ValueError:
            continue
        prediction = (
            row.get("criteria_v4_prediction")
            or row.get("criteria_v3_prediction")
            or row.get("criteria_v2_prediction")
            or ""
        ).strip()
        if not prediction:
            continue
        prediction_by_clip[clip_idx] = row

    if not prediction_by_clip:
        return clip_rows

    output: list[dict[str, str]] = []
    for row in clip_rows:
        merged = dict(row)
        try:
            clip_idx = int(merged.get("clip_idx", ""))
        except ValueError:
            output.append(merged)
            continue
        criteria_row = prediction_by_clip.get(clip_idx)
        if criteria_row:
            prediction = (
                criteria_row.get("criteria_v4_prediction")
                or criteria_row.get("criteria_v3_prediction")
                or criteria_row.get("criteria_v2_prediction")
                or ""
            ).strip()
            merged["hybrid_prediction"] = prediction
            merged["criteria_v4_prediction"] = criteria_row.get("criteria_v4_prediction", "")
            merged["criteria_v4_reason"] = criteria_row.get("criteria_v4_reason", "")
            merged["criteria_v3_prediction"] = criteria_row.get("criteria_v3_prediction", "")
            merged["criteria_v3_transition_reason"] = criteria_row.get("criteria_v3_transition_reason", "")
            merged["criteria_v2_prediction"] = criteria_row.get("criteria_v2_prediction", "")
            merged["criteria_target_label"] = criteria_row.get("criteria_target_label", "")
        output.append(merged)
    return output


def load_session_audio(audio_path: Path, trim_start_s: float, duration_s: float, target_sr: int = 16000) -> np.ndarray:
    sr, samples = wavfile.read(audio_path)
    samples = samples.astype(np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    start_idx = max(0, int(round(trim_start_s * sr)))
    if duration_s > 0:
        end_idx = min(len(samples), int(round((trim_start_s + duration_s) * sr)))
    else:
        end_idx = len(samples)
    samples = samples[start_idx:end_idx]
    if sr != target_sr:
        samples = resample_poly(samples, target_sr, sr)

    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    if peak > 0:
        samples = samples / peak
    return samples.astype(np.float32)


def moving_average(values: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1 or len(values) == 0:
        return values
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    return np.convolve(values, kernel, mode="same")


def detect_speech_regions(
    samples_16k: np.ndarray,
    *,
    target_sr: int = 16000,
    frame_ms: int = 30,
    hop_ms: int = 10,
    pad_ms: int = 80,
    merge_gap_ms: int = 160,
    min_region_ms: int = 180,
) -> list[tuple[float, float]]:
    if len(samples_16k) == 0:
        return []

    frame = max(1, int(round(target_sr * frame_ms / 1000.0)))
    hop = max(1, int(round(target_sr * hop_ms / 1000.0)))
    pad = int(round(target_sr * pad_ms / 1000.0))
    merge_gap = int(round(target_sr * merge_gap_ms / 1000.0))
    min_region = int(round(target_sr * min_region_ms / 1000.0))

    frame_starts = list(range(0, max(1, len(samples_16k) - frame + 1), hop))
    if not frame_starts:
        frame_starts = [0]

    energies = []
    for start in frame_starts:
        window = samples_16k[start : start + frame]
        if len(window) < frame:
            padded = np.zeros(frame, dtype=np.float32)
            padded[: len(window)] = window
            window = padded
        energies.append(float(np.sqrt(np.mean(np.square(window)) + 1e-10)))

    energy = np.asarray(energies, dtype=np.float32)
    smooth_frames = max(1, int(round(90.0 / hop_ms)))
    smooth_energy = moving_average(energy, smooth_frames)

    noise_floor = float(np.percentile(smooth_energy, 35))
    speech_peak = float(np.percentile(smooth_energy, 92))
    dynamic_threshold = noise_floor + 0.28 * max(0.0, speech_peak - noise_floor)
    threshold = max(dynamic_threshold, noise_floor * 1.8, 0.012)
    active = smooth_energy >= threshold

    regions: list[tuple[int, int]] = []
    region_start = None
    for idx, is_active in enumerate(active):
        if is_active and region_start is None:
            region_start = frame_starts[idx]
        elif not is_active and region_start is not None:
            region_end = frame_starts[idx] + frame
            regions.append((region_start, region_end))
            region_start = None
    if region_start is not None:
        regions.append((region_start, len(samples_16k)))

    merged_regions: list[tuple[int, int]] = []
    for start, end in regions:
        start = max(0, start - pad)
        end = min(len(samples_16k), end + pad)
        if not merged_regions:
            merged_regions.append((start, end))
            continue
        prev_start, prev_end = merged_regions[-1]
        if start - prev_end <= merge_gap:
            merged_regions[-1] = (prev_start, max(prev_end, end))
        else:
            merged_regions.append((start, end))

    filtered_regions = [
        (start / target_sr, end / target_sr)
        for start, end in merged_regions
        if (end - start) >= min_region
    ]
    return filtered_regions


def build_utterance_windows(
    speech_regions: list[tuple[float, float]],
    *,
    max_window_s: float = 1.2,
    hop_s: float = 0.45,
    min_window_s: float = 0.22,
) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    for region_start, region_end in speech_regions:
        duration = max(0.0, region_end - region_start)
        if duration <= 0:
            continue
        if duration <= max_window_s * 1.15:
            if duration >= min_window_s:
                windows.append((region_start, region_end))
            continue

        start = region_start
        while start < region_end:
            end = min(region_end, start + max_window_s)
            if end - start >= min_window_s:
                windows.append((start, end))
            if end >= region_end:
                break
            start += hop_s

    deduped: list[tuple[float, float]] = []
    for start, end in windows:
        if deduped and abs(start - deduped[-1][0]) < 1e-3 and abs(end - deduped[-1][1]) < 1e-3:
            continue
        deduped.append((start, end))
    return deduped


def build_transcription_config(
    model: str,
    beam_size: int,
    patience: float,
    segment_fuzzy_threshold: float,
    condition_on_previous_text: bool,
) -> dict[str, Any]:
    return {
        "analysis_version": ANALYSIS_VERSION,
        "strategy": "local utterance scan plus stable transcription recovery, with duplicate timestamp cleanup and nearby-conflict guards",
        "model": model,
        "beam_size": beam_size,
        "patience": patience,
        "segment_fuzzy_threshold": segment_fuzzy_threshold,
        "condition_on_previous_text": condition_on_previous_text,
        "stable_profile": {
            "temperature": 0.0,
            "best_of": 1,
            "condition_on_previous_text": condition_on_previous_text,
            "prompt_style": "short action-vocabulary prompt",
        },
        "enhanced_profile": {
            "temperature": [0.0, 0.2, 0.4],
            "best_of": max(1, int(beam_size)),
            "condition_on_previous_text": condition_on_previous_text,
            "logprob_threshold": -0.8,
            "no_speech_threshold": 0.45,
            "hallucination_silence_threshold": 0.8,
            "prompt_style": "strong constrained action-vocabulary prompt",
        },
        "utterance_profile": {
            "temperature": 0.0,
            "best_of": 1,
            "condition_on_previous_text": False,
            "prompt_style": "short local utterance prompt",
            "scan_window_s": 1.2,
            "scan_hop_s": 0.45,
            "energy_frame_ms": 30,
            "energy_hop_ms": 10,
            "energy_pad_ms": 80,
            "energy_merge_gap_ms": 160,
            "min_region_ms": 180,
        },
        "rare_action_labels": sorted(RARE_ACTION_LABELS),
    }


def transcribe_audio(
    samples_16k: np.ndarray,
    *,
    model: str,
    beam_size: int,
    patience: float,
    condition_on_previous_text: bool,
    profile: str,
) -> dict[str, Any]:
    import mlx_whisper

    decode_kwargs: dict[str, Any] = {}
    shared_kwargs = {
        "verbose": False,
        "path_or_hf_repo": model,
        "language": "en",
        "word_timestamps": True,
    }

    if profile == "stable":
        return mlx_whisper.transcribe(
            samples_16k,
            temperature=0.0,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=(
                "The speaker repeats action words from this vocabulary: "
                "walking, sitting, standing, upstairs, downstairs."
            ),
            **shared_kwargs,
        )

    if profile == "utterance":
        return mlx_whisper.transcribe(
            samples_16k,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=(
                "The speaker says only these action words, sometimes repeated: "
                "walking, sitting, standing, upstairs, downstairs. "
                "Prefer the exact matching action word."
            ),
            **shared_kwargs,
        )

    if beam_size and beam_size > 1:
        decode_kwargs["best_of"] = int(beam_size)

    return mlx_whisper.transcribe(
        samples_16k,
        temperature=(0.0, 0.2, 0.4),
        condition_on_previous_text=condition_on_previous_text,
        logprob_threshold=-0.8,
        no_speech_threshold=0.45,
        hallucination_silence_threshold=0.8,
        initial_prompt=(
            "The speaker repeatedly says only these exact action words: "
            "walking, sitting, standing, upstairs, downstairs. "
            "Prefer these words over similar sounding alternatives."
        ),
        **shared_kwargs,
        **decode_kwargs,
    )


def clean_token(text: str) -> str:
    return re.sub(r"[^a-z]+", "", text.lower())


def normalize_single_token(token: str) -> str | None:
    if token in SINGLE_TOKEN_ALIASES:
        return SINGLE_TOKEN_ALIASES[token]
    if token.startswith("walk"):
        return "walking"
    if token.startswith("sitt"):
        return "sitting"
    if token.startswith("stand"):
        return "standing"
    if token.startswith("upstair"):
        return "upstairs"
    if token.startswith("downstair"):
        return "downstairs"
    return None


def similarity_score(candidate: str, target: str) -> float:
    return difflib.SequenceMatcher(None, candidate, target).ratio()


def normalize_motion_label(label: str | None) -> str | None:
    if not label:
        return None
    cleaned = clean_token(label)
    alias_map = {
        "goupstairs": "upstairs",
        "godownstairs": "downstairs",
        "upstairs": "upstairs",
        "upstair": "upstairs",
        "downstairs": "downstairs",
        "downstair": "downstairs",
        "walking": "walking",
        "walkingg": "walking",
        "walk": "walking",
        "standing": "standing",
        "stand": "standing",
        "sitting": "sitting",
        "sit": "sitting",
    }
    if cleaned in alias_map:
        return alias_map[cleaned]
    return normalize_single_token(cleaned)


def transition_actions_from_label(label: str | None) -> list[str]:
    if not label:
        return []
    value = str(label).strip().lower().replace("_", "-")
    value = re.sub(r"\s+to\s+", "-to-", value)
    value = re.sub(r"\s*->\s*", "-to-", value)
    if "-to-" in value:
        parts = value.split("-to-")
    else:
        parts = re.split(r"\s*(?:\||-|,|/)\s*", value)
    actions = [normalize_motion_label(part) for part in parts if part.strip()]
    return unique_ordered([action for action in actions if action in ACTION_LABELS])


def canonical_transition_label(actions: list[str]) -> str:
    ordered = unique_ordered([normalize_motion_label(action) or "" for action in actions])
    ordered = [action for action in ordered if action in ACTION_LABELS]
    if not ordered:
        return ""
    if len(ordered) == 1:
        return ordered[0]
    pair = (ordered[0], ordered[1])
    return CLINICAL_TRANSITION_LABELS.get(pair, "-to-".join(ordered))


def canonical_prediction_label(label: str | None) -> str:
    transition_actions = transition_actions_from_label(label)
    if len(transition_actions) > 1:
        return canonical_transition_label(transition_actions)
    stable = normalize_motion_label(label or "")
    if stable in ACTION_LABELS:
        return stable
    return (label or "").strip().lower()


def fuzzy_match_motion_label(text: str, threshold: float) -> str | None:
    if not text:
        return None
    lowered = text.strip().lower()
    if lowered in FUZZY_PHRASE_ALIASES:
        return FUZZY_PHRASE_ALIASES[lowered]
    normalized = normalize_motion_label(lowered)
    if normalized:
        return normalized

    cleaned = clean_token(lowered)
    if not cleaned:
        return None
    if cleaned in BLOCKED_FUZZY_CLEANED_TEXT:
        return None

    best_label = None
    best_score = 0.0
    for label in ACTION_LABELS:
        score = similarity_score(cleaned, label)
        if score > best_score:
            best_label = label
            best_score = score
    if best_label and best_score >= threshold:
        return best_label
    return None


def parse_action_display(value: str | None) -> list[str]:
    if not value:
        return []
    transition_actions = transition_actions_from_label(value)
    if transition_actions:
        return transition_actions
    parts = re.split(r"\s*(?:\||-|,|/)\s*", value.strip())
    normalized = [normalize_motion_label(part) for part in parts if part.strip()]
    return unique_ordered([item for item in normalized if item])


def format_action_display(actions: list[str]) -> str:
    ordered = unique_ordered([normalize_motion_label(action) or "" for action in actions])
    ordered = [item for item in ordered if item]
    return "-".join(ordered)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_manual_payload(trial_id: str) -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        "updated_at": now_iso(),
        "annotations": {},
    }


def load_manual_payload(trial_id: str, manual_labels_path: Path | None = None) -> dict[str, Any]:
    path = manual_labels_path or discover_trial_paths(trial_id)["manual_labels_path"]
    if not path.exists():
        return default_manual_payload(trial_id)
    payload = load_json(path)
    if "annotations" not in payload or not isinstance(payload["annotations"], dict):
        payload["annotations"] = {}
    payload.setdefault("trial_id", trial_id)
    payload.setdefault("updated_at", "")
    return payload


def save_manual_payload(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = now_iso()
    save_json(path, payload)


def normalize_annotation(mode: str | None, actions: list[str] | None) -> dict[str, Any] | None:
    normalized_actions = unique_ordered([normalize_motion_label(action) or "" for action in (actions or [])])
    normalized_actions = [item for item in normalized_actions if item]
    if not normalized_actions:
        return None

    normalized_mode = "transition" if (mode == "transition" or len(normalized_actions) > 1) else "single"
    if normalized_mode == "single":
        normalized_actions = normalized_actions[:1]
    return {
        "mode": normalized_mode,
        "actions": normalized_actions,
        "label_display": format_action_display(normalized_actions),
        "action_count": len(normalized_actions),
        "updated_at": now_iso(),
    }


def update_manual_annotation(
    trial_id: str,
    clip_idx: int,
    mode: str | None,
    actions: list[str] | None,
) -> dict[str, Any]:
    paths = discover_trial_paths(trial_id)
    payload = load_manual_payload(trial_id, paths["manual_labels_path"])
    key = str(int(clip_idx))
    annotation = normalize_annotation(mode, actions)
    auto_actions = automatic_actions_for_clip(trial_id, clip_idx)
    if annotation is None or (
        auto_actions
        and unique_ordered(annotation.get("actions", [])) == unique_ordered(auto_actions)
    ):
        payload["annotations"].pop(key, None)
    else:
        payload["annotations"][key] = annotation
    save_manual_payload(paths["manual_labels_path"], payload)
    return payload


def discover_spoken_action_trials() -> list[str]:
    trials: list[str] = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        if not run_dir.name.startswith("Ntrial"):
            continue
        if (run_dir / "spoken_action_labels" / "spoken_action_analysis.json").exists():
            trials.append(run_dir.name)
    return sorted(trials)


def automatic_actions_for_clip(trial_id: str, clip_idx: int) -> list[str]:
    cache_path = discover_trial_paths(trial_id)["spoken_dir"] / "spoken_action_analysis.json"
    if not cache_path.exists():
        return []
    cached = load_json(cache_path)
    for row in cached.get("clip_rows", []):
        if int(row.get("clip_idx", -1)) == int(clip_idx):
            return parse_action_display(row.get("assumed_actions_in_clip", ""))
    return []


def extract_target_word_events(transcription: dict[str, Any]) -> list[WordEvent]:
    events: list[WordEvent] = []
    segments = transcription.get("segments", [])
    for segment in segments:
        words = segment.get("words", []) or []
        i = 0
        while i < len(words):
            word = words[i]
            token = clean_token(str(word.get("word", "")))
            label = normalize_single_token(token)
            start_s = float(word.get("start", segment.get("start", 0.0)))
            end_s = float(word.get("end", start_s))
            probability = float(word.get("probability", 0.0))
            raw_text = str(word.get("word", ""))

            if token in {"up", "down"} and i + 1 < len(words):
                next_word = words[i + 1]
                next_token = clean_token(str(next_word.get("word", "")))
                if next_token.startswith("stair"):
                    label = "upstairs" if token == "up" else "downstairs"
                    end_s = float(next_word.get("end", next_word.get("start", end_s)))
                    probability = float((probability + float(next_word.get("probability", 0.0))) / 2.0)
                    raw_text = f"{raw_text} {next_word.get('word', '')}".strip()
                    i += 1

            if label in ACTION_LABELS:
                events.append(
                    WordEvent(
                        label=label,
                        start_s=max(0.0, start_s),
                        end_s=max(start_s, end_s),
                        probability=probability,
                        raw_text=raw_text,
                    )
                )
            i += 1

    events.sort(key=lambda item: (item.start_s, item.end_s, item.label))
    return dedupe_word_events(events)


def shift_word_events(events: list[WordEvent], offset_s: float) -> list[WordEvent]:
    shifted: list[WordEvent] = []
    for event in events:
        shifted.append(
            WordEvent(
                label=event.label,
                start_s=max(0.0, event.start_s + offset_s),
                end_s=max(0.0, event.end_s + offset_s),
                probability=event.probability,
                raw_text=event.raw_text,
            )
        )
    return shifted


def extract_utterance_word_events(
    samples_16k: np.ndarray,
    *,
    model: str,
    beam_size: int,
    patience: float,
    segment_fuzzy_threshold: float,
) -> tuple[list[WordEvent], list[tuple[float, float]], list[dict[str, Any]]]:
    speech_regions = detect_speech_regions(samples_16k)
    windows = build_utterance_windows(speech_regions)
    utterance_events: list[WordEvent] = []
    utterance_debug: list[dict[str, Any]] = []

    for window_start_s, window_end_s in windows:
        start_idx = max(0, int(round(window_start_s * 16000)))
        end_idx = min(len(samples_16k), int(round(window_end_s * 16000)))
        if end_idx - start_idx < int(round(0.18 * 16000)):
            continue
        utterance_samples = samples_16k[start_idx:end_idx]
        transcription = transcribe_audio(
            utterance_samples,
            model=model,
            beam_size=beam_size,
            patience=patience,
            condition_on_previous_text=False,
            profile="utterance",
        )

        window_events = extract_target_word_events(transcription)
        if not window_events:
            window_events = extract_segment_run_events(
                transcription,
                threshold=max(0.68, segment_fuzzy_threshold - 0.03),
            )
        if not window_events:
            fallback_label = fuzzy_match_motion_label(str(transcription.get("text", "")).strip(), max(0.68, segment_fuzzy_threshold - 0.03))
            if fallback_label:
                window_events = [
                    WordEvent(
                        label=fallback_label,
                        start_s=0.0,
                        end_s=max(0.05, window_end_s - window_start_s),
                        probability=0.18,
                        raw_text=f"[utterance_text] {str(transcription.get('text', '')).strip()}",
                    )
                ]

        shifted = shift_word_events(window_events, window_start_s)
        utterance_events.extend(shifted)
        utterance_debug.append(
            {
                "start_s": window_start_s,
                "end_s": window_end_s,
                "text": str(transcription.get("text", "")).strip(),
                "events": [asdict(event) for event in shifted],
            }
        )

    utterance_events = dedupe_word_events(utterance_events)
    return utterance_events, speech_regions, utterance_debug


def extract_recovered_segment_events(
    transcription: dict[str, Any],
    *,
    threshold: float,
    allowed_labels: set[str] | None = None,
) -> list[WordEvent]:
    recovered: list[WordEvent] = []
    for segment in transcription.get("segments", []) or []:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        tokens = re.findall(r"[A-Za-z']+", text)
        if not tokens:
            continue
        segment_start = float(segment.get("start", 0.0))
        segment_end = float(segment.get("end", segment_start))
        duration = max(0.05, segment_end - segment_start)
        token_count = max(1, len(tokens))

        for idx, token in enumerate(tokens):
            candidate = token.lower()
            label = fuzzy_match_motion_label(candidate, threshold)
            token_span_start = segment_start + duration * (idx / token_count)
            token_span_end = segment_start + duration * ((idx + 1) / token_count)
            if label and (allowed_labels is None or label in allowed_labels):
                recovered.append(
                    WordEvent(
                        label=label,
                        start_s=token_span_start,
                        end_s=token_span_end,
                        probability=0.25,
                        raw_text=f"[segment_fuzzy] {token}",
                    )
                )

            if idx + 1 < len(tokens):
                phrase = f"{tokens[idx].lower()} {tokens[idx + 1].lower()}"
                label = fuzzy_match_motion_label(phrase, threshold)
                phrase_span_end = segment_start + duration * ((idx + 2) / token_count)
                if label and (allowed_labels is None or label in allowed_labels):
                    recovered.append(
                        WordEvent(
                            label=label,
                            start_s=token_span_start,
                            end_s=phrase_span_end,
                            probability=0.2,
                            raw_text=f"[segment_fuzzy] {phrase}",
                        )
                    )
    recovered.sort(key=lambda item: (item.start_s, item.end_s, item.label))
    return dedupe_word_events(recovered)


def extract_segment_run_events(
    transcription: dict[str, Any],
    *,
    threshold: float,
    allowed_labels: set[str] | None = None,
) -> list[WordEvent]:
    recovered: list[WordEvent] = []
    for segment in transcription.get("segments", []) or []:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        tokens = re.findall(r"[A-Za-z']+", text)
        if not tokens:
            continue
        segment_start = float(segment.get("start", 0.0))
        segment_end = float(segment.get("end", segment_start))
        duration = max(0.05, segment_end - segment_start)
        token_count = max(1, len(tokens))

        candidates: list[tuple[str, float, float, str]] = []
        idx = 0
        while idx < len(tokens):
            label = None
            consume = 1
            token_span_start = segment_start + duration * (idx / token_count)

            if idx + 1 < len(tokens):
                phrase = f"{tokens[idx].lower()} {tokens[idx + 1].lower()}"
                phrase_label = fuzzy_match_motion_label(phrase, threshold)
                if phrase_label and (allowed_labels is None or phrase_label in allowed_labels):
                    label = phrase_label
                    consume = 2

            if label is None:
                single_label = fuzzy_match_motion_label(tokens[idx].lower(), threshold)
                if single_label and (allowed_labels is None or single_label in allowed_labels):
                    label = single_label

            if label:
                token_span_end = segment_start + duration * ((idx + consume) / token_count)
                candidates.append(
                    (
                        label,
                        token_span_start,
                        token_span_end,
                        " ".join(tokens[idx : idx + consume]),
                    )
                )
            idx += consume

        if not candidates:
            continue

        run_label, run_start, run_end, run_raw = candidates[0]
        for label, cand_start, cand_end, raw_text in candidates[1:]:
            if label == run_label:
                run_end = cand_end
                run_raw = f"{run_raw} {raw_text}".strip()
                continue
            recovered.append(
                WordEvent(
                    label=run_label,
                    start_s=run_start,
                    end_s=run_end,
                    probability=0.22,
                    raw_text=f"[segment_run] {run_raw}",
                )
            )
            run_label, run_start, run_end, run_raw = label, cand_start, cand_end, raw_text

        recovered.append(
            WordEvent(
                label=run_label,
                start_s=run_start,
                end_s=run_end,
                probability=0.22,
                raw_text=f"[segment_run] {run_raw}",
            )
        )

    recovered.sort(key=lambda item: (item.start_s, item.end_s, item.label))
    return dedupe_word_events(recovered)


def merge_recovered_events(
    primary_events: list[WordEvent],
    recovered_events: list[WordEvent],
    *,
    overlap_tolerance_s: float = 0.75,
) -> list[WordEvent]:
    merged = list(primary_events)
    for event in recovered_events:
        already_covered = False
        for existing in merged:
            if existing.label != event.label:
                continue
            if abs(existing.start_s - event.start_s) <= overlap_tolerance_s or intersects(
                existing.start_s,
                max(existing.end_s, existing.start_s + 0.05),
                event.start_s,
                max(event.end_s, event.start_s + 0.05),
            ):
                already_covered = True
                break
        if not already_covered:
            merged.append(event)
    merged.sort(key=lambda item: (item.start_s, item.end_s, item.label))
    return dedupe_word_events(merged)


def event_end_s(event: WordEvent) -> float:
    return max(event.end_s, event.start_s + MIN_EVENT_DURATION_S)


def event_gap_s(first: WordEvent, second: WordEvent) -> float:
    first_end = event_end_s(first)
    second_end = event_end_s(second)
    if intersects(first.start_s, first_end, second.start_s, second_end):
        return 0.0
    if first_end < second.start_s:
        return second.start_s - first_end
    return first.start_s - second_end


def choose_preferred_word_event(current: WordEvent, candidate: WordEvent) -> WordEvent:
    current_duration = event_end_s(current) - current.start_s
    candidate_duration = event_end_s(candidate) - candidate.start_s
    if candidate.probability > current.probability + 1e-9:
        return candidate
    if math.isclose(candidate.probability, current.probability, abs_tol=1e-9) and candidate_duration > current_duration:
        return candidate
    return current


def dedupe_word_events(events: list[WordEvent], *, same_label_tolerance_s: float = 0.22) -> list[WordEvent]:
    """Collapse timestamp duplicates from overlapping Whisper windows.

    The speaker often repeats the same action word, so we only dedupe events
    that are effectively the same timestamp or overlap tightly. Repeated words
    spaced farther apart remain as independent speech cues.
    """
    if not events:
        return []

    deduped: list[WordEvent] = []
    for event in sorted(events, key=lambda item: (item.start_s, item.end_s, item.label, -item.probability)):
        duplicate_index = None
        for idx in range(len(deduped) - 1, -1, -1):
            existing = deduped[idx]
            if existing.start_s < event.start_s - same_label_tolerance_s - MIN_EVENT_DURATION_S:
                break
            if existing.label != event.label:
                continue
            close_start = abs(existing.start_s - event.start_s) <= same_label_tolerance_s
            close_gap = event_gap_s(existing, event) <= same_label_tolerance_s
            if close_start or close_gap:
                duplicate_index = idx
                break
        if duplicate_index is None:
            deduped.append(event)
            continue
        deduped[duplicate_index] = choose_preferred_word_event(deduped[duplicate_index], event)

    deduped.sort(key=lambda item: (item.start_s, item.end_s, item.label))
    return deduped


def merge_gap_fill_events(
    primary_events: list[WordEvent],
    filler_events: list[WordEvent],
    *,
    same_label_tolerance_s: float = 0.35,
    different_label_tolerance_s: float = 0.45,
) -> list[WordEvent]:
    merged = dedupe_word_events(primary_events)
    for event in filler_events:
        conflicting_overlap = False
        redundant_same_label = False
        event_end = event_end_s(event)
        for existing in merged:
            existing_end = event_end_s(existing)
            overlaps = intersects(existing.start_s, existing_end, event.start_s, event_end)
            if existing.label == event.label:
                if overlaps or abs(existing.start_s - event.start_s) <= same_label_tolerance_s:
                    redundant_same_label = True
                    break
                continue
            if overlaps or event_gap_s(existing, event) <= different_label_tolerance_s:
                conflicting_overlap = True
                break
        if conflicting_overlap or redundant_same_label:
            continue
        merged.append(event)
    return dedupe_word_events(merged)


def collapse_to_runs(events: list[WordEvent]) -> list[LabelRun]:
    if not events:
        return []

    runs: list[LabelRun] = []
    current = LabelRun(
        label=events[0].label,
        start_s=events[0].start_s,
        end_s=events[0].end_s,
        support_count=1,
        support_words=[events[0].raw_text],
    )
    for event in events[1:]:
        if event.label == current.label:
            current.end_s = max(current.end_s, event.end_s)
            current.support_count += 1
            current.support_words.append(event.raw_text)
            continue
        runs.append(current)
        current = LabelRun(
            label=event.label,
            start_s=event.start_s,
            end_s=event.end_s,
            support_count=1,
            support_words=[event.raw_text],
        )
    runs.append(current)
    return runs


def merge_isolated_stray_runs(runs: list[LabelRun], min_stray_run_count: int, max_stray_run_duration_s: float) -> list[LabelRun]:
    if len(runs) < 3:
        return runs

    merged: list[LabelRun] = []
    idx = 0
    while idx < len(runs):
        current = runs[idx]
        if 0 < idx < len(runs) - 1:
            prev_run = runs[idx - 1]
            next_run = runs[idx + 1]
            run_duration = max(0.0, current.end_s - current.start_s)
            if (
                prev_run.label == next_run.label
                and current.label != prev_run.label
                and current.support_count <= min_stray_run_count
                and run_duration <= max_stray_run_duration_s
            ):
                merged_prev = merged.pop() if merged else LabelRun(**asdict(prev_run))
                merged_prev.end_s = next_run.end_s
                merged_prev.support_count += current.support_count + next_run.support_count
                merged_prev.support_words.extend(current.support_words + next_run.support_words)
                merged.append(merged_prev)
                idx += 2
                continue
        merged.append(LabelRun(**asdict(current)))
        idx += 1
    return merged


def build_state_intervals(runs: list[LabelRun], session_duration_s: float) -> list[LabelRun]:
    if not runs:
        return []
    intervals: list[LabelRun] = []
    for idx, run in enumerate(runs):
        start_s = 0.0 if idx == 0 else run.start_s
        end_s = session_duration_s if idx == len(runs) - 1 else runs[idx + 1].start_s
        intervals.append(
            LabelRun(
                label=run.label,
                start_s=max(0.0, start_s),
                end_s=max(start_s, min(session_duration_s, end_s)),
                support_count=run.support_count,
                support_words=list(run.support_words),
            )
        )
    return intervals


def intersects(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return max(start_a, start_b) < min(end_a, end_b)


def word_intersects_clip(event: WordEvent, clip_start_s: float, clip_end_s: float) -> bool:
    start_s = event.start_s
    end_s = event.end_s if event.end_s > event.start_s else event.start_s + 0.05
    return intersects(start_s, end_s, clip_start_s, clip_end_s) or (
        clip_start_s <= event.start_s < clip_end_s
    )


def unique_ordered(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def normalize_model_prediction(label: str | None) -> str:
    return canonical_prediction_label(label)


def effective_actions_for_clip(auto_actions: list[str], manual_annotation: dict[str, Any] | None) -> list[str]:
    if manual_annotation and manual_annotation.get("actions"):
        return unique_ordered([normalize_motion_label(action) or "" for action in manual_annotation["actions"] if action])
    return unique_ordered(auto_actions)


def match_prediction_to_actions(prediction_label: str | None, actions: list[str]) -> tuple[bool | None, bool | None, bool | None]:
    if not actions:
        return None, None, None
    normalized_prediction = normalize_model_prediction(prediction_label)
    target_label = canonical_transition_label(actions)
    strict_exact = normalized_prediction == target_label
    transition_aware = strict_exact
    component_match = normalized_prediction in actions
    return strict_exact, transition_aware, component_match


def agreement_status_for_match(strict_exact: bool | None, transition_aware: bool | None) -> str:
    if transition_aware is None:
        return "not_comparable"
    if strict_exact:
        return "exact_match"
    if transition_aware:
        return "transition_match"
    return "mismatch"


def merge_manual_annotations_into_clip_rows(
    trial_id: str,
    clip_output: list[dict[str, Any]],
    manual_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    annotations = manual_payload.get("annotations", {})
    merged_rows: list[dict[str, Any]] = []
    manual_override_count = 0
    manual_transition_count = 0
    comparable_clip_count = 0
    strict_correct_count = 0
    transition_aware_correct_count = 0
    component_transition_aware_correct_count = 0
    stable_only_comparable_clip_count = 0
    stable_only_correct_count = 0
    transition_clip_comparable_count = 0
    transition_clip_correct_count = 0

    for row in clip_output:
        clip_idx = int(row["clip_idx"])
        auto_actions = parse_action_display(row.get("assumed_actions_in_clip", ""))
        auto_display = format_action_display(auto_actions)
        annotation = annotations.get(str(clip_idx))
        manual_actions = unique_ordered(annotation.get("actions", [])) if annotation else []
        manual_display = format_action_display(manual_actions)
        effective_actions = effective_actions_for_clip(auto_actions, annotation)
        effective_display = format_action_display(effective_actions)
        strict_exact, transition_aware, component_match = match_prediction_to_actions(
            row.get("predicted_label"),
            effective_actions,
        )
        if transition_aware is not None:
            comparable_clip_count += 1
            if strict_exact:
                strict_correct_count += 1
            if transition_aware:
                transition_aware_correct_count += 1
            if component_match:
                component_transition_aware_correct_count += 1
        if len(effective_actions) == 1 and strict_exact is not None:
            stable_only_comparable_clip_count += 1
            if strict_exact:
                stable_only_correct_count += 1
        if len(effective_actions) > 1 and strict_exact is not None:
            transition_clip_comparable_count += 1
            if strict_exact:
                transition_clip_correct_count += 1

        if annotation:
            manual_override_count += 1
            if annotation.get("mode") == "transition" or len(manual_actions) > 1:
                manual_transition_count += 1

        merged = dict(row)
        merged.update(
            {
                "automatic_spoken_action": auto_display,
                "automatic_spoken_words": row.get("spoken_words_detected_in_clip", ""),
                "manual_correction_mode": annotation.get("mode", "") if annotation else "",
                "manual_correction_label": manual_display,
                "manual_correction_action_count": len(manual_actions),
                "manual_is_transition": "yes" if len(manual_actions) > 1 else "",
                "manual_last_updated": annotation.get("updated_at", "") if annotation else "",
                "effective_label_source": "manual_override" if annotation else "automatic_spoken",
                "effective_manual_label": effective_display,
                "effective_manual_action_count": len(effective_actions),
                "effective_is_transition": "yes" if len(effective_actions) > 1 else "",
                "effective_transition_target_label": canonical_transition_label(effective_actions),
                "agreement_status": agreement_status_for_match(strict_exact, transition_aware),
                "model_prediction_normalized": normalize_model_prediction(row.get("predicted_label")),
                "model_exact_match_effective_manual": "" if strict_exact is None else ("yes" if strict_exact else "no"),
                "model_matches_effective_manual": "" if transition_aware is None else ("yes" if transition_aware else "no"),
                "model_matches_any_effective_action": ""
                if component_match is None
                else ("yes" if component_match else "no"),
            }
        )
        for index in range(5):
            merged[f"manual_action_{index + 1}"] = manual_actions[index] if index < len(manual_actions) else ""
            merged[f"effective_action_{index + 1}"] = effective_actions[index] if index < len(effective_actions) else ""
        merged_rows.append(merged)

    summary = {
        "trial_id": trial_id,
        "manual_override_count": manual_override_count,
        "manual_transition_override_count": manual_transition_count,
        "comparable_clip_count": comparable_clip_count,
        "stable_only_comparable_clip_count": stable_only_comparable_clip_count,
        "transition_clip_comparable_count": transition_clip_comparable_count,
        "strict_correct_count": strict_correct_count,
        "transition_aware_correct_count": transition_aware_correct_count,
        "component_transition_aware_correct_count": component_transition_aware_correct_count,
        "stable_only_correct_count": stable_only_correct_count,
        "transition_clip_correct_count": transition_clip_correct_count,
        "strict_accuracy": (strict_correct_count / comparable_clip_count) if comparable_clip_count else None,
        "transition_aware_accuracy": (transition_aware_correct_count / comparable_clip_count) if comparable_clip_count else None,
        "component_transition_aware_accuracy": (component_transition_aware_correct_count / comparable_clip_count)
        if comparable_clip_count
        else None,
        "stable_only_accuracy": (stable_only_correct_count / stable_only_comparable_clip_count)
        if stable_only_comparable_clip_count
        else None,
        "transition_clip_accuracy": (transition_clip_correct_count / transition_clip_comparable_count)
        if transition_clip_comparable_count
        else None,
    }
    return merged_rows, summary


def build_clip_rows(
    clip_rows: list[dict[str, str]],
    word_events: list[WordEvent],
    state_intervals: list[LabelRun],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in clip_rows:
        clip_idx = int(row["clip_idx"])
        clip_start_s = float(row["start_s"])
        clip_end_s = float(row["end_s"])

        clip_word_events = [event for event in word_events if word_intersects_clip(event, clip_start_s, clip_end_s)]
        detected_words = unique_ordered([event.label for event in clip_word_events])
        detected_word_times = [round(event.start_s, 3) for event in clip_word_events]

        overlapping_states = [
            state for state in state_intervals if intersects(state.start_s, state.end_s, clip_start_s, clip_end_s)
        ]
        assumed_actions = unique_ordered([state.label for state in overlapping_states])
        state_changes_inside = [
            round(state.start_s, 3)
            for state in overlapping_states[1:]
            if clip_start_s <= state.start_s < clip_end_s
        ]

        output.append(
            {
                "clip_idx": clip_idx,
                "clip_name": f"clip_{clip_idx:04d}",
                "clip_start_s": clip_start_s,
                "clip_end_s": clip_end_s,
                "predicted_label": row.get("criteria_v4_prediction")
                or row.get("criteria_v3_prediction")
                or row.get("hybrid_prediction")
                or row.get("criteria_v2_prediction")
                or row.get("rule_prediction")
                or "",
                "spoken_words_detected_in_clip": " | ".join(detected_words),
                "spoken_word_event_count": len(clip_word_events),
                "spoken_word_times_s": ", ".join(f"{value:.3f}" for value in detected_word_times),
                "assumed_actions_in_clip": " | ".join(assumed_actions),
                "primary_assumed_action": assumed_actions[0] if assumed_actions else "",
                "is_transition_clip": "yes" if len(assumed_actions) > 1 else "",
                "state_change_times_in_clip_s": ", ".join(f"{value:.3f}" for value in state_changes_inside),
                "horizontal_displacement_m": float(row.get("horizontal_displacement_m", 0.0) or 0.0),
                "vertical_displacement_m": float(row.get("vertical_displacement_m", 0.0) or 0.0),
                "mean_linear_speed_mps": float(row.get("mean_linear_speed_mps", 0.0) or 0.0),
                "mean_angular_speed_rps": float(row.get("mean_angular_speed_rps", 0.0) or 0.0),
                "mean_z_m": float(row.get("mean_z_m", 0.0) or 0.0),
            }
        )
    return output


def save_json(path: Path, payload: Any) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: convert(val) for key, val in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    path.write_text(json.dumps(convert(payload), indent=2))


def process_trial(
    trial_id: str,
    model: str,
    force_transcribe: bool,
    min_stray_run_count: int,
    max_stray_run_duration_s: float,
    beam_size: int = 5,
    patience: float = 1.2,
    segment_fuzzy_threshold: float = 0.74,
    condition_on_previous_text: bool = False,
) -> dict[str, pd.DataFrame]:
    manifest = load_manifest(trial_id)
    paths = discover_trial_paths(trial_id)
    clip_rows = load_clip_rows(paths["clip_csv"])
    clip_rows = apply_criteria_prediction_overrides(trial_id, clip_rows)
    audio_path, trim_start_s, duration_s = discover_audio_window(trial_id, manifest, paths["sync_json"], clip_rows)
    session_duration_s = max(float(row["end_s"]) for row in clip_rows) if clip_rows else duration_s
    transcription_config = build_transcription_config(
        model=model,
        beam_size=beam_size,
        patience=patience,
        segment_fuzzy_threshold=segment_fuzzy_threshold,
        condition_on_previous_text=condition_on_previous_text,
    )

    cache_path = paths["spoken_dir"] / "spoken_action_analysis.json"
    cached = load_json(cache_path) if cache_path.exists() else None
    cache_is_current = (
        cached is not None
        and cached.get("transcription_config", {}).get("analysis_version") == ANALYSIS_VERSION
    )

    if cache_is_current and not force_transcribe:
        cached = load_json(cache_path)
        word_events = [WordEvent(**item) for item in cached["word_events"]]
        runs = [LabelRun(**item) for item in cached["state_intervals"]]
        clip_output = build_clip_rows(clip_rows, word_events, runs)
    else:
        samples_16k = load_session_audio(audio_path, trim_start_s, duration_s or session_duration_s)
        stable_transcription = transcribe_audio(
            samples_16k,
            model=model,
            beam_size=beam_size,
            patience=patience,
            condition_on_previous_text=condition_on_previous_text,
            profile="stable",
        )
        enhanced_transcription = transcribe_audio(
            samples_16k,
            model=model,
            beam_size=beam_size,
            patience=patience,
            condition_on_previous_text=condition_on_previous_text,
            profile="enhanced",
        )
        utterance_word_events, speech_regions, utterance_debug = extract_utterance_word_events(
            samples_16k,
            model=model,
            beam_size=beam_size,
            patience=patience,
            segment_fuzzy_threshold=segment_fuzzy_threshold,
        )
        stable_primary_word_events = extract_target_word_events(stable_transcription)
        stable_segment_run_events = extract_segment_run_events(
            stable_transcription,
            threshold=max(segment_fuzzy_threshold, 0.8),
        )
        stable_word_events = merge_recovered_events(
            stable_primary_word_events,
            stable_segment_run_events,
            overlap_tolerance_s=0.35,
        )
        combined_primary_events = merge_gap_fill_events(
            utterance_word_events,
            stable_word_events,
            same_label_tolerance_s=0.35,
        )

        present_rare_labels = {event.label for event in combined_primary_events if event.label in RARE_ACTION_LABELS}
        fallback_rare_labels = RARE_ACTION_LABELS - present_rare_labels
        enhanced_primary_word_events = [
            event
            for event in extract_target_word_events(enhanced_transcription)
            if event.label in fallback_rare_labels
        ]
        recovered_segment_events = extract_segment_run_events(
            enhanced_transcription,
            threshold=max(segment_fuzzy_threshold, 0.82),
            allowed_labels=fallback_rare_labels,
        )
        rare_word_events = merge_recovered_events(
            enhanced_primary_word_events,
            recovered_segment_events,
            overlap_tolerance_s=0.35,
        )
        word_events = merge_gap_fill_events(
            combined_primary_events,
            rare_word_events,
            same_label_tolerance_s=0.35,
        )
        raw_runs = collapse_to_runs(word_events)
        smoothed_runs = merge_isolated_stray_runs(
            raw_runs,
            min_stray_run_count=min_stray_run_count,
            max_stray_run_duration_s=max_stray_run_duration_s,
        )
        runs = build_state_intervals(smoothed_runs, session_duration_s=session_duration_s)
        clip_output = build_clip_rows(clip_rows, word_events, runs)

        cache_payload = {
            "trial_id": trial_id,
            "audio_source": str(audio_path),
            "audio_trim_start_s": trim_start_s,
            "session_duration_s": session_duration_s,
            "model": model,
            "transcription_config": transcription_config,
            "word_events": [asdict(item) for item in word_events],
            "utterance_word_events": [asdict(item) for item in utterance_word_events],
            "primary_word_events": [asdict(item) for item in stable_primary_word_events],
            "stable_primary_word_events": [asdict(item) for item in stable_primary_word_events],
            "stable_segment_run_events": [asdict(item) for item in stable_segment_run_events],
            "stable_word_events": [asdict(item) for item in stable_word_events],
            "combined_primary_events": [asdict(item) for item in combined_primary_events],
            "enhanced_primary_word_events": [asdict(item) for item in enhanced_primary_word_events],
            "recovered_segment_events": [asdict(item) for item in recovered_segment_events],
            "rare_word_events": [asdict(item) for item in rare_word_events],
            "speech_regions": [{"start_s": start_s, "end_s": end_s} for start_s, end_s in speech_regions],
            "utterance_debug": utterance_debug,
            "state_intervals": [asdict(item) for item in runs],
            "clip_rows": clip_output,
            "transcription_text": stable_transcription.get("text", ""),
            "transcription_segments": stable_transcription.get("segments", []),
            "stable_transcription_text": stable_transcription.get("text", ""),
            "stable_transcription_segments": stable_transcription.get("segments", []),
            "enhanced_transcription_text": enhanced_transcription.get("text", ""),
            "enhanced_transcription_segments": enhanced_transcription.get("segments", []),
        }
        save_json(cache_path, cache_payload)

    manual_payload = load_manual_payload(trial_id, paths["manual_labels_path"])
    clip_output, accuracy_summary = merge_manual_annotations_into_clip_rows(
        trial_id,
        clip_output,
        manual_payload,
    )

    words_df = pd.DataFrame(
        [
            {
                "event_index": idx,
                "label": event.label,
                "start_s": event.start_s,
                "end_s": event.end_s,
                "probability": event.probability,
                "raw_text": event.raw_text,
            }
            for idx, event in enumerate(word_events)
        ]
    )
    states_df = pd.DataFrame(
        [
            {
                "state_index": idx,
                "label": state.label,
                "start_s": state.start_s,
                "end_s": state.end_s,
                "duration_s": max(0.0, state.end_s - state.start_s),
                "support_count": state.support_count,
                "support_words": " | ".join(unique_ordered(state.support_words)),
            }
            for idx, state in enumerate(runs)
        ]
    )
    clips_df = pd.DataFrame(clip_output)
    manual_rows = [
        {
            "clip_idx": int(clip_idx),
            "manual_correction_mode": annotation.get("mode", ""),
            "manual_correction_label": annotation.get("label_display", ""),
            "manual_correction_action_count": int(annotation.get("action_count", len(annotation.get("actions", [])))),
            "manual_last_updated": annotation.get("updated_at", ""),
            **{
                f"manual_action_{index + 1}": annotation.get("actions", [])[index]
                if index < len(annotation.get("actions", []))
                else ""
                for index in range(5)
            },
        }
        for clip_idx, annotation in sorted(manual_payload.get("annotations", {}).items(), key=lambda item: int(item[0]))
    ]
    manual_df = pd.DataFrame(manual_rows, columns=MANUAL_EXPORT_COLUMNS) if manual_rows else empty_manual_df()

    metadata_df = pd.DataFrame(
        [
            {
                "trial_id": trial_id,
                "audio_source": str(audio_path),
                "audio_trim_start_s": trim_start_s,
                "session_duration_s": session_duration_s,
                "clip_coverage_s": session_duration_s,
                "synced_audio_window_s": duration_s or session_duration_s,
                "clip_count": len(clips_df),
                "detected_target_word_count": len(words_df),
                "detected_state_count": len(states_df),
                "manual_override_count": accuracy_summary["manual_override_count"],
                "manual_transition_override_count": accuracy_summary["manual_transition_override_count"],
                "comparable_clip_count": accuracy_summary["comparable_clip_count"],
                "stable_only_comparable_clip_count": accuracy_summary["stable_only_comparable_clip_count"],
                "transition_clip_comparable_count": accuracy_summary["transition_clip_comparable_count"],
                "strict_accuracy": accuracy_summary["strict_accuracy"],
                "transition_aware_accuracy": accuracy_summary["transition_aware_accuracy"],
                "component_transition_aware_accuracy": accuracy_summary["component_transition_aware_accuracy"],
                "stable_only_accuracy": accuracy_summary["stable_only_accuracy"],
                "transition_clip_accuracy": accuracy_summary["transition_clip_accuracy"],
                "model": model,
                "beam_size": beam_size,
                "patience": patience,
                "segment_fuzzy_threshold": segment_fuzzy_threshold,
                "condition_on_previous_text": "yes" if condition_on_previous_text else "no",
                "assumption_1": "Aria synced audio is used as the spoken-label source.",
                "assumption_2": "The first detected spoken action is backfilled to the start of the synced session.",
                "assumption_3": "A spoken action remains active until a different action word is detected.",
                "assumption_4": "If multiple actions overlap one clip, both are shown.",
                "assumption_5": "If no manual correction is stored, the automatic spoken action is treated as accepted.",
                "assumption_6": "Transition-inclusive accuracy now requires the model to match the full effective label, including transition labels such as sit-to-stand.",
                "assumption_7": "Speech detection uses local utterance windows first, then stable word/segment recovery; duplicate timestamp events are collapsed and nearby conflicting filler events are ignored.",
                "assumption_8": "Stable-only accuracy excludes transition-state clips and scores only single-state clips against the effective manual label.",
                "assumption_9": "Component-aware accuracy is retained only as an older comparison: it counts a transition clip as matched if the model predicts any action inside the transition.",
            }
        ]
    )
    validate_trial_outputs(trial_id, clips_df, manual_df, metadata_df)
    return {
        f"{trial_id}_clips": clips_df,
        f"{trial_id}_words": words_df,
        f"{trial_id}_states": states_df,
        f"{trial_id}_manual": manual_df,
        f"{trial_id}_meta": metadata_df,
    }


def autosize_worksheet(worksheet: Any) -> None:
    for column_cells in worksheet.columns:
        max_len = 0
        column_letter = column_cells[0].column_letter
        for cell in column_cells:
            value = "" if cell.value is None else str(cell.value)
            max_len = max(max_len, len(value))
        worksheet.column_dimensions[column_letter].width = min(max(max_len + 2, 10), 40)


def empty_manual_df() -> pd.DataFrame:
    return pd.DataFrame(columns=MANUAL_EXPORT_COLUMNS)


def build_review_export_df(clips_df: pd.DataFrame) -> pd.DataFrame:
    export_df = clips_df.copy()
    missing = [column for column in REVIEW_EXPORT_COLUMNS if column not in export_df.columns]
    for column in missing:
        export_df[column] = ""
    return export_df[REVIEW_EXPORT_COLUMNS]


def build_session_export_df(meta_df: pd.DataFrame) -> pd.DataFrame:
    export_df = meta_df.copy()
    missing = [column for column in OVERVIEW_COLUMNS if column not in export_df.columns]
    for column in missing:
        export_df[column] = ""
    return export_df[OVERVIEW_COLUMNS]


def build_sheet_guide_df(trials: list[str]) -> pd.DataFrame:
    rows = [
        {
            "sheet_name": "overview",
            "purpose": "One-row summary per trial, including clip count, sync duration, and current accuracy snapshots.",
        },
        {
            "sheet_name": "sheet_guide",
            "purpose": "Explains what each worksheet is for.",
        },
        {
            "sheet_name": "field_guide",
            "purpose": "Defines the main labels and status fields used in the review sheets.",
        },
        {
            "sheet_name": "logic_notes",
            "purpose": "Records the core assumptions behind spoken-label carry-forward and accuracy scoring.",
        },
    ]
    for trial_id in trials:
        rows.extend(
            [
                {
                    "sheet_name": f"{trial_id}_review",
                    "purpose": "Main clip-by-clip sheet for review. Combines model output, spoken labels, manual correction, agreement status, and motion metrics.",
                },
                {
                    "sheet_name": f"{trial_id}_manual_edits",
                    "purpose": "Only the clips that have stored manual overrides from the dashboard.",
                },
                {
                    "sheet_name": f"{trial_id}_speech_words",
                    "purpose": "Raw spoken action words detected in synchronized audio with timestamps.",
                },
                {
                    "sheet_name": f"{trial_id}_speech_states",
                    "purpose": "Smoothed spoken-action intervals after carry-forward logic.",
                },
                {
                    "sheet_name": f"{trial_id}_session",
                    "purpose": "Trial-level metadata and accuracy summary for this specific trial.",
                },
            ]
        )
    return pd.DataFrame(rows)


def build_field_guide_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "field": "predicted_label",
                "group": "model",
                "meaning": "The current motion label predicted by the motion pipeline for that clip.",
                "example_or_values": "walking, sitting, standing, go upstairs, go downstairs",
            },
            {
                "field": "spoken_words_detected_in_clip",
                "group": "raw speech",
                "meaning": "Action words directly recognized inside the clip window from synchronized Aria audio.",
                "example_or_values": "walking | standing",
            },
            {
                "field": "automatic_spoken_action",
                "group": "auto speech label",
                "meaning": "Automatic spoken-action label after smoothing and carry-forward logic.",
                "example_or_values": "standing-walking",
            },
            {
                "field": "assumed_actions_in_clip",
                "group": "auto speech label",
                "meaning": "Actions inferred to be active during the clip after carry-forward logic; may contain one or more actions.",
                "example_or_values": "standing | walking",
            },
            {
                "field": "manual_correction_mode",
                "group": "manual labeling",
                "meaning": "How a manual correction was stored from the dashboard.",
                "example_or_values": "single, transition, blank",
            },
            {
                "field": "manual_correction_label",
                "group": "manual labeling",
                "meaning": "Your saved correction. Blank means the automatic spoken action is currently accepted.",
                "example_or_values": "sitting-standing",
            },
            {
                "field": "effective_manual_label",
                "group": "accuracy reference",
                "meaning": "The label actually used as the reference for accuracy: manual correction if present, otherwise the automatic spoken action.",
                "example_or_values": "walking",
            },
            {
                "field": "effective_label_source",
                "group": "accuracy reference",
                "meaning": "Whether the current accuracy reference comes from the automatic spoken label or from a manual override.",
                "example_or_values": "automatic_spoken, manual_override",
            },
            {
                "field": "agreement_status",
                "group": "accuracy result",
                "meaning": "Compact interpretation of model agreement against the effective reference label.",
                "example_or_values": "exact_match, transition_match, mismatch, not_comparable",
            },
            {
                "field": "model_exact_match_effective_manual",
                "group": "accuracy result",
                "meaning": "Strict agreement. Only counts yes when the effective label is a single motion and the model matches it exactly.",
                "example_or_values": "yes, no, blank",
            },
            {
                "field": "model_matches_effective_manual",
                "group": "accuracy result",
                "meaning": "Transition-inclusive agreement. Counts yes when the model matches the full effective label, including transition labels.",
                "example_or_values": "yes, no, blank",
            },
            {
                "field": "model_matches_any_effective_action",
                "group": "accuracy result",
                "meaning": "Older component-aware comparison. For transition clips, counts yes if the model matches any one action inside the effective label.",
                "example_or_values": "yes, no, blank",
            },
            {
                "field": "is_transition_clip",
                "group": "clip interpretation",
                "meaning": "Marked yes when more than one automatic spoken action overlaps the clip.",
                "example_or_values": "yes, blank",
            },
            {
                "field": "strict_accuracy",
                "group": "trial summary",
                "meaning": "Trial-level exact agreement rate against the effective reference labels.",
                "example_or_values": "0.704",
            },
            {
                "field": "transition_aware_accuracy",
                "group": "trial summary",
                "meaning": "Trial-level transition-inclusive agreement rate against the effective reference labels.",
                "example_or_values": "0.889",
            },
            {
                "field": "transition_clip_accuracy",
                "group": "trial summary",
                "meaning": "Accuracy on clips whose effective manual label contains more than one action.",
                "example_or_values": "0.500",
            },
            {
                "field": "stable_only_accuracy",
                "group": "trial summary",
                "meaning": "Trial-level agreement rate after excluding transition-state clips and scoring only single-state clips.",
                "example_or_values": "0.864",
            },
            {
                "field": "clip_coverage_s",
                "group": "trial summary",
                "meaning": "Total duration covered by the exported 2-second clips.",
                "example_or_values": "64.0",
            },
            {
                "field": "synced_audio_window_s",
                "group": "trial summary",
                "meaning": "Length of the synchronized audio window before clip rounding.",
                "example_or_values": "65.916",
            },
        ]
    )


def build_logic_notes_df() -> pd.DataFrame:
    notes = [
        "Aria synchronized audio is used as the spoken-label source.",
        "The first detected spoken action is backfilled to the start of the synchronized session.",
        "A spoken action remains active until a different action word is detected.",
        "If multiple actions overlap one clip, all active actions are retained in the automatic spoken label.",
        "If no manual correction is stored, the automatic spoken action is treated as accepted.",
        "Transition-inclusive accuracy requires the model prediction to match the full effective reference label, including transition labels.",
        "Component-aware accuracy is kept as an older comparison and only checks whether the model prediction is one action inside the transition.",
        "If a saved manual label is identical to the automatic spoken label, it is treated as keep-auto rather than a true override.",
    ]
    return pd.DataFrame({"logic_note": notes})


def validate_trial_outputs(trial_id: str, clips_df: pd.DataFrame, manual_df: pd.DataFrame, meta_df: pd.DataFrame) -> None:
    meta = meta_df.iloc[0]
    manual_override_count = int((clips_df["manual_correction_label"].fillna("") != "").sum())
    manual_transition_count = int((clips_df["manual_is_transition"].fillna("") == "yes").sum())
    comparable_count = int((clips_df["effective_manual_label"].fillna("") != "").sum())
    strict_yes = int((clips_df["model_exact_match_effective_manual"].fillna("") == "yes").sum())
    transition_yes = int((clips_df["model_matches_effective_manual"].fillna("") == "yes").sum())
    stable_only_mask = clips_df["effective_manual_action_count"].fillna(0).astype(int) == 1
    stable_only_count = int(stable_only_mask.sum())
    stable_only_yes = int(((clips_df["model_exact_match_effective_manual"].fillna("") == "yes") & stable_only_mask).sum())
    transition_clip_mask = clips_df["effective_manual_action_count"].fillna(0).astype(int) > 1
    transition_clip_count = int(transition_clip_mask.sum())
    transition_clip_yes = int(
        ((clips_df["model_exact_match_effective_manual"].fillna("") == "yes") & transition_clip_mask).sum()
    )

    assert manual_override_count == int(meta["manual_override_count"]), f"{trial_id}: manual override count mismatch"
    assert manual_transition_count == int(meta["manual_transition_override_count"]), f"{trial_id}: manual transition count mismatch"
    assert comparable_count == int(meta["comparable_clip_count"]), f"{trial_id}: comparable clip count mismatch"
    assert stable_only_count == int(meta["stable_only_comparable_clip_count"]), f"{trial_id}: stable-only comparable clip count mismatch"
    assert transition_clip_count == int(meta["transition_clip_comparable_count"]), f"{trial_id}: transition clip count mismatch"
    assert manual_override_count == len(manual_df.index), f"{trial_id}: manual sheet row mismatch"

    expected_strict = (strict_yes / comparable_count) if comparable_count else None
    expected_transition = (transition_yes / comparable_count) if comparable_count else None
    expected_stable_only = (stable_only_yes / stable_only_count) if stable_only_count else None
    expected_transition_clip = (transition_clip_yes / transition_clip_count) if transition_clip_count else None
    actual_strict = meta["strict_accuracy"]
    actual_transition = meta["transition_aware_accuracy"]
    actual_stable_only = meta["stable_only_accuracy"]
    actual_transition_clip = meta["transition_clip_accuracy"]
    if expected_strict is None:
        assert pd.isna(actual_strict), f"{trial_id}: strict accuracy should be blank"
    else:
        assert math.isclose(float(actual_strict), float(expected_strict), rel_tol=1e-9), f"{trial_id}: strict accuracy mismatch"
    if expected_transition is None:
        assert pd.isna(actual_transition), f"{trial_id}: transition-aware accuracy should be blank"
    else:
        assert math.isclose(float(actual_transition), float(expected_transition), rel_tol=1e-9), f"{trial_id}: transition-aware accuracy mismatch"
    if expected_stable_only is None:
        assert pd.isna(actual_stable_only), f"{trial_id}: stable-only accuracy should be blank"
    else:
        assert math.isclose(float(actual_stable_only), float(expected_stable_only), rel_tol=1e-9), f"{trial_id}: stable-only accuracy mismatch"
    if expected_transition_clip is None:
        assert pd.isna(actual_transition_clip), f"{trial_id}: transition clip accuracy should be blank"
    else:
        assert math.isclose(float(actual_transition_clip), float(expected_transition_clip), rel_tol=1e-9), f"{trial_id}: transition clip accuracy mismatch"


def style_workbook(writer: Any, overview_row_count: int) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill

    fills = {
        "overview": PatternFill("solid", fgColor="DCE6F8"),
        "guide": PatternFill("solid", fgColor="E9F5E6"),
        "review": PatternFill("solid", fgColor="F4E7D3"),
        "manual": PatternFill("solid", fgColor="F7D9E6"),
        "speech": PatternFill("solid", fgColor="E8F0FE"),
        "session": PatternFill("solid", fgColor="E8EEF8"),
    }

    for worksheet in writer.book.worksheets:
        autosize_worksheet(worksheet)
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        role = "review"
        if worksheet.title in {"overview"}:
            role = "overview"
        elif worksheet.title in {"sheet_guide", "field_guide", "logic_notes"}:
            role = "guide"
        elif worksheet.title.endswith("_manual_edits"):
            role = "manual"
        elif worksheet.title.endswith("_speech_words") or worksheet.title.endswith("_speech_states"):
            role = "speech"
        elif worksheet.title.endswith("_session"):
            role = "session"

        header_fill = fills[role]
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = Font(bold=True)
            cell.alignment = Alignment(vertical="center", wrap_text=True)

        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                if isinstance(cell.value, float) and "accuracy" in str(worksheet[1][cell.column - 1].value):
                    cell.number_format = "0.0%"

    if "overview" in writer.book.sheetnames:
        ws = writer.book["overview"]
        if overview_row_count >= 1:
            ws.freeze_panes = "A2"


def build_workbook_for_trials(
    trials: list[str],
    output_path: Path = DEFAULT_WORKBOOK_PATH,
    *,
    model: str = "mlx-community/whisper-tiny",
    force_transcribe: bool = False,
    min_stray_run_count: int = 1,
    max_stray_run_duration_s: float = 1.25,
    beam_size: int = 5,
    patience: float = 1.2,
    segment_fuzzy_threshold: float = 0.74,
    condition_on_previous_text: bool = False,
) -> Path:
    workbook_sheets: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []
    for trial_id in trials:
        trial_sheets = process_trial(
            trial_id=trial_id,
            model=model,
            force_transcribe=force_transcribe,
            min_stray_run_count=min_stray_run_count,
            max_stray_run_duration_s=max_stray_run_duration_s,
            beam_size=beam_size,
            patience=patience,
            segment_fuzzy_threshold=segment_fuzzy_threshold,
            condition_on_previous_text=condition_on_previous_text,
        )
        meta_sheet = trial_sheets[f"{trial_id}_meta"]
        clips_sheet = trial_sheets[f"{trial_id}_clips"]
        words_sheet = trial_sheets[f"{trial_id}_words"]
        states_sheet = trial_sheets[f"{trial_id}_states"]
        manual_sheet = trial_sheets[f"{trial_id}_manual"]

        workbook_sheets[f"{trial_id}_review"] = build_review_export_df(clips_sheet)
        workbook_sheets[f"{trial_id}_manual_edits"] = manual_sheet
        workbook_sheets[f"{trial_id}_speech_words"] = words_sheet
        workbook_sheets[f"{trial_id}_speech_states"] = states_sheet
        workbook_sheets[f"{trial_id}_session"] = build_session_export_df(meta_sheet)
        summary_rows.append(build_session_export_df(meta_sheet).iloc[0].to_dict())

    summary_df = pd.DataFrame(summary_rows, columns=OVERVIEW_COLUMNS)
    sheet_guide_df = build_sheet_guide_df(trials)
    field_guide_df = build_field_guide_df()
    logic_notes_df = build_logic_notes_df()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="overview")
        sheet_guide_df.to_excel(writer, index=False, sheet_name="sheet_guide")
        field_guide_df.to_excel(writer, index=False, sheet_name="field_guide")
        logic_notes_df.to_excel(writer, index=False, sheet_name="logic_notes")
        for sheet_name, dataframe in workbook_sheets.items():
            dataframe.to_excel(writer, index=False, sheet_name=sheet_name[:31])
        style_workbook(writer, overview_row_count=len(summary_df))

    return output_path


def main() -> None:
    args = parse_args()
    output_path = build_workbook_for_trials(
        args.trials,
        output_path=args.output,
        model=args.model,
        force_transcribe=args.force_transcribe,
        min_stray_run_count=args.min_stray_run_count,
        max_stray_run_duration_s=args.max_stray_run_duration_s,
        beam_size=args.beam_size,
        patience=args.patience,
        segment_fuzzy_threshold=args.segment_fuzzy_threshold,
        condition_on_previous_text=args.condition_on_previous_text,
    )
    print(f"Wrote workbook: {output_path}")


if __name__ == "__main__":
    main()
