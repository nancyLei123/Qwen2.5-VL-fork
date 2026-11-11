#!/usr/bin/env python3
"""Baseline automatic detector for Parkinson's-like movement events in a PD-Insighter trial.

This is a transparent signal-processing baseline, not a trained or validated model. It reads
the signals prepared by ``symptom_detection_signals.py`` and proposes candidate intervals for
manual review on the dashboard session clock. Nothing is diagnosed and no interval without a
candidate is declared symptom-free.

Event types (the taxonomy and every threshold live in ``DEFAULT_CONFIG`` and can be overridden):

``freezing_like``
    Walking was under way, forward progress stops for a few seconds, and the body keeps moving
    in place (head sway / step-band head bob, and leg movement in the downward video when it
    is available). A stop with a quiet body is treated as an ordinary stop and not reported.
``gait_hesitation``
    Either a short interruption of progress inside walking, or movement in place that goes on
    for a while right before walking starts (delayed start).
``shuffling_like``
    Sustained level walking with a short step-length estimate while the stepping rhythm is kept.
``slow_gait``
    Sustained level walking far below ordinary walking speed.

Tremor-like movement is deliberately not produced: in the recordings this was developed on,
neither the head track nor the 15 fps downward video shows a usable 3.5-7 Hz component.
Audio is loaded for reference but carries no weight, because footstep sounds could not be
separated from speech and room noise.

Score: every candidate gets a 0-100 *uncalibrated detection score*. It is the sum of evidence
points (each item has a fixed maximum and earns a share of it from one measurement through a
linear ramp between two thresholds) minus penalty points for ordinary explanations or poor
data. It is not a probability. See ``docs/symptom_annotation/DETECTION.md``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np

from symptom_detection_signals import ACTIVITY_CODES, GRID_HZ, TrialSignals, rolling_mean

DETECTOR_NAME = "pd-insighter-motion-baseline"
DETECTOR_VERSION = "0.1.0"
SCHEMA = "pd-insighter/symptom-detections@1"
SCORE_KIND = "uncalibrated detection score (0-100), not a probability"
SOURCE_LABELS = {"head_track": "Head track", "downward_video": "Downward video", "audio": "Audio",
                 "context": "Activity labels"}

DEFAULT_CONFIG: dict[str, Any] = {
    "labels": {
        "freezing_like": {"name": "Freezing-like episode", "color": "#b5475f", "enabled": True,
                          "description": "Progress stops during walking while the body keeps moving in place."},
        "gait_hesitation": {"name": "Gait hesitation", "color": "#c77f2a", "enabled": True,
                            "description": "Short interruption of walking, or a delayed start with movement in place."},
        "shuffling_like": {"name": "Shuffling-like gait", "color": "#7a5ea8", "enabled": True,
                           "description": "Level walking with short steps while the stepping rhythm is kept."},
        "slow_gait": {"name": "Slowed gait", "color": "#3f7f9c", "enabled": True,
                      "description": "Sustained level walking far below ordinary walking speed."},
    },
    "strength": {"high_score": 70, "medium_score": 45, "min_report_score": 30,
                 "source_share_for_independent": 0.4},
    "motion": {
        "move_speed_mps": 0.07,          # net travel above this counts as locomotion
        "halt_speed_mps": 0.045,         # net travel below this counts as no progress
        "min_locomotion_s": 1.5,
        "min_travel_extent_m": 0.45,     # ground (or height) covered within 8 s for movement to count as walking
        "stair_vertical_speed_mps": 0.04,
        "turn_yaw_rate_dps": 45.0,
    },
    "freezing_like": {
        "min_s": 2.0, "max_s": 30.0, "context_s": 4.0, "min_walking_before_fraction": 0.3,
        "sway_quiet_floor_m": 0.010, "sway_strong_m": 0.025,
        "flow_quiet_factor": 1.4, "posture_change_exclude_m": 0.25, "posture_change_penalty_m": 0.10,
        "turn_in_place_deg": 70.0,
        "min_in_place_share": 0.4,       # below this share of the in-place evidence a stop is an ordinary stop
        "points": {"progress_stopped": 15, "walking_before": 10, "walking_resumes": 8,
                   "in_place_head": 22, "in_place_video": 40, "duration": 5},
        "penalties": {"turning": 18, "posture": 20, "tracking": 10, "stairs": 6, "sitting": 20, "sources_disagree": 15},
    },
    "gait_hesitation": {
        "interruption_min_dip_s": 0.3, "interruption_min_s": 0.8, "interruption_max_s": 3.0, "flank_s": 2.5,
        "interruption_min_speed_before_mps": 0.15,
        "start_min_s": 1.2, "start_max_s": 6.0, "start_still_before_s": 3.0,
        "points": {"progress_dip": 25, "walking_both_sides": 20, "abrupt": 15, "in_place_head": 15,
                   "in_place_video": 20, "delay": 25, "still_before": 15, "walking_follows": 20},
        "penalties": {"turning": 22, "posture": 25, "tracking": 10, "stairs": 6, "sitting": 20, "sources_disagree": 15},
    },
    "gait_quality": {
        "window_s": 4.0, "min_bout_s": 5.0, "merge_gap_s": 1.5,
        "slow_speed_mps": 0.40, "very_slow_speed_mps": 0.20,
        "short_step_m": 0.30, "very_short_step_m": 0.15,
        "shuffle_min_step_rate_hz": 1.1, "shuffle_full_step_rate_hz": 1.6,
        "max_turning_fraction": 0.5, "min_bout_travel_m": 1.5, "max_height_range_m": 0.25,
        "points_slow": {"speed": 40, "sustained": 15, "rhythm": 10, "straight": 10, "video_moving": 25},
        "points_shuffle": {"step_length": 35, "rhythm_kept": 20, "sustained": 10, "straight": 10, "video_moving": 25},
        "penalties": {"turning": 15, "tracking": 10},
    },
}


# ----------------------------------------------------------------------------------------------
# helpers


def merge_config(overrides: dict[str, Any] | None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)

    def merge(target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value

    merge(config, overrides or {})
    return config


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def run_id(config: dict[str, Any]) -> str:
    return f"motion-baseline-{DETECTOR_VERSION}-{config_hash(config)[:8]}"


def ramp(value: float, low: float, high: float) -> float:
    """0 at ``low``, 1 at ``high``, linear in between (``low`` may be above ``high``)."""
    if value is None or not np.isfinite(value):
        return 0.0
    if high == low:
        return float(value >= high)
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open index runs where ``mask`` is true."""
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def close_gaps(mask: np.ndarray, max_gap_s: float) -> np.ndarray:
    out = mask.copy()
    limit = int(round(max_gap_s * GRID_HZ))
    for a, b in runs_of(~mask):
        if 0 < a and b < len(mask) and b - a <= limit:
            out[a:b] = True
    return out


def drop_short(mask: np.ndarray, min_s: float) -> np.ndarray:
    out = mask.copy()
    limit = int(round(min_s * GRID_HZ))
    for a, b in runs_of(mask):
        if b - a < limit:
            out[a:b] = False
    return out


def med(values: np.ndarray) -> float:
    good = values[np.isfinite(values)]
    return float(np.median(good)) if len(good) else float("nan")


def frac(mask: np.ndarray) -> float:
    return float(mask.mean()) if len(mask) else 0.0


class Evidence:
    """Collects the evidence items of one candidate and turns them into a score."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.limitations: list[str] = []

    def support(self, key: str, source: str, max_points: float, share: float, text: str,
                value: float | None = None, unit: str = "") -> float:
        points = round(max_points * float(np.clip(share, 0.0, 1.0)), 1)
        self.items.append({"id": key, "kind": "support", "source": source, "points": points,
                           "max_points": max_points, "text": text,
                           "value": None if value is None or not np.isfinite(value) else round(float(value), 4),
                           "unit": unit})
        return points

    def missing(self, key: str, source: str, max_points: float, text: str) -> None:
        self.items.append({"id": key, "kind": "missing", "source": source, "points": 0.0,
                           "max_points": max_points, "text": text, "value": None, "unit": ""})
        self.limitations.append(text)

    def penalty(self, key: str, source: str, max_points: float, share: float, text: str,
                value: float | None = None, unit: str = "") -> None:
        points = round(max_points * float(np.clip(share, 0.0, 1.0)), 1)
        if points <= 0:
            return
        self.items.append({"id": key, "kind": "penalty", "source": source, "points": -points,
                           "max_points": max_points, "text": text,
                           "value": None if value is None or not np.isfinite(value) else round(float(value), 4),
                           "unit": unit})

    def score(self) -> int:
        possible = sum(i["max_points"] for i in self.items if i["kind"] in ("support", "missing"))
        earned = sum(i["points"] for i in self.items)
        return int(round(100.0 * np.clip(earned / possible, 0.0, 1.0))) if possible else 0

    def independent_sources(self, share: float) -> list[str]:
        """Measured sources that each earned a real share of their own maximum."""
        out = []
        for source in ("head_track", "downward_video"):
            mine = [i for i in self.items if i["source"] == source and i["kind"] == "support"]
            possible = sum(i["max_points"] for i in mine)
            if possible and sum(i["points"] for i in mine) / possible >= share:
                out.append(source)
        return out


class Context:
    """Per-trial masks and reference levels shared by the detectors."""

    def __init__(self, signals: TrialSignals, config: dict[str, Any]):
        self.signals, self.config = signals, config
        f, t, motion = signals.features, signals.t, config["motion"]
        self.t, self.n = t, len(t)
        nan = np.full(self.n, np.nan)
        self.head_ok = f.get("head_ok", np.zeros(self.n, dtype=bool))
        self.video_ok = f.get("video_ok", np.zeros(self.n, dtype=bool))
        get = lambda name: f.get(name, nan)  # noqa: E731
        self.progress, self.speed = get("progress_speed"), get("speed")
        self.sway, self.bob_amp, self.step_rate = get("sway_amp"), get("bob_amp"), get("step_rate_hz")
        self.yaw, self.path_turn = get("yaw_rate"), get("path_turn_rate")
        self.height, self.vertical_speed = get("height"), get("vertical_speed")
        self.flow, self.flow_step = get("flow_smooth"), get("flow_step_amp")
        self.has_yaw = bool(np.isfinite(self.yaw).any())
        self.progress_fast = get("progress_fast")   # 1 s net travel, for short interruptions
        with np.errstate(invalid="ignore"):
            self.travelling = (get("travel_extent") >= motion["min_travel_extent_m"]) | (get("vertical_extent") >= motion["min_travel_extent_m"])
            loco = self.head_ok & (self.progress >= motion["move_speed_mps"]) & self.travelling
            self.loco = drop_short(close_gaps(loco, 0.6), motion["min_locomotion_s"])
            self.halt = self.head_ok & (self.progress < motion["halt_speed_mps"])
            self.stairs = np.abs(self.vertical_speed) >= motion["stair_vertical_speed_mps"]
            turn = self.yaw if self.has_yaw else self.path_turn
            self.turning = turn >= motion["turn_yaw_rate_dps"]
            quiet = self.halt & (self.speed < 0.04)
        activity = signals.activity if signals.activity is not None else np.zeros(self.n, dtype=np.int8)
        self.activity = activity
        self.stairs |= np.isin(activity, (ACTIVITY_CODES["upstairs"], ACTIVITY_CODES["downstairs"])) & ~np.isfinite(self.vertical_speed)
        # Reference levels while the body is quiet and while it is travelling, measured in this trial.
        self.sway_quiet = med(self.sway[quiet]) if quiet.any() else float("nan")
        self.flow_quiet = med(self.flow[quiet & self.video_ok]) if (quiet & self.video_ok).any() else float("nan")
        self.flow_walk = med(self.flow[self.loco & self.video_ok]) if (self.loco & self.video_ok).any() else float("nan")
        self.baselines = {"sway_quiet_m": self.sway_quiet, "flow_quiet": self.flow_quiet, "flow_walking": self.flow_walk,
                          "quiet_seconds": float(quiet.sum() / GRID_HZ), "locomotion_seconds": float(self.loco.sum() / GRID_HZ)}

    def sl(self, start_s: float, end_s: float) -> slice:
        a = int(np.clip(round(start_s * GRID_HZ), 0, self.n))
        b = int(np.clip(round(end_s * GRID_HZ), a, self.n))
        return slice(a, max(b, min(a + 1, self.n)))

    def interior(self, a: int, b: int, trim_s: float = 0.75) -> tuple[int, int]:
        """The inside of an interval. Features are smoothed over 1-2 s, so its edges still carry
        the movement just before and after; in-place movement is judged without them."""
        trim = min(int(trim_s * GRID_HZ), max(0, (b - a - int(0.5 * GRID_HZ)) // 2))
        return a + trim, b - trim

    def video_share(self, a: int, b: int) -> float:
        return frac(self.video_ok[a:b])

    def yaw_change_deg(self, a: int, b: int) -> float:
        values = self.yaw[a:b] if self.has_yaw else self.path_turn[a:b]
        return float(np.nansum(values) / GRID_HZ) if np.isfinite(values).any() else float("nan")

    def height_change_m(self, a: int, b: int, pad_s: float = 1.0) -> float:
        pad = int(pad_s * GRID_HZ)
        values = self.height[max(0, a - pad):min(self.n, b + pad)]
        values = values[np.isfinite(values)]
        return float(values.max() - values.min()) if len(values) else float("nan")

    def stairs_share(self, a: int, b: int) -> float:
        return frac(self.stairs[a:b])

    def in_place_video(self, evidence: Evidence, key: str, max_points: float, a: int, b: int, what: str) -> None:
        """Leg/body movement in the downward view relative to this trial's quiet and walking levels."""
        if self.video_share(a, b) < 0.6 or not np.isfinite(self.flow_quiet):
            reason = ("The downward camera has no footage for this interval, so leg movement could not be checked."
                      if self.signals.downward_coverage_s else "This trial has no downward camera, so leg movement could not be checked.")
            evidence.missing(key, "downward_video", max_points, reason)
            return
        a, b = self.interior(a, b)
        level = med(self.flow[a:b])
        low = self.flow_quiet * self.config["freezing_like"]["flow_quiet_factor"]
        high = max(self.flow_walk if np.isfinite(self.flow_walk) else low * 2, low * 1.5)
        share = ramp(level, low, high)
        text = (f"Downward view: image motion {what} is {level:.2f} px/frame "
                f"(quiet standing {self.flow_quiet:.2f}, walking {self.flow_walk:.2f} in this trial).")
        evidence.support(key, "downward_video", max_points, share, text, level, "px/frame")

    def in_place_head(self, evidence: Evidence, key: str, max_points: float, a: int, b: int) -> None:
        settings = self.config["freezing_like"]
        a, b = self.interior(a, b)
        sway = med(self.sway[a:b])
        quiet = self.sway_quiet if np.isfinite(self.sway_quiet) else 0.0
        low = max(settings["sway_quiet_floor_m"], quiet * 1.5)
        share = ramp(sway, low, max(settings["sway_strong_m"], low * 1.8))
        rate = med(self.step_rate[a:b])
        text = (f"Head keeps moving in place: sway {sway * 1000:.0f} mm (quiet {quiet * 1000:.0f} mm), "
                f"about {rate:.1f} step-like head bobs per second.")
        evidence.support(key, "head_track", max_points, share, text, sway, "m")

    def disagreement(self, evidence: Evidence, max_points: float) -> None:
        """Head says the body moves in place, the downward view shows it quiet: weaken the candidate."""
        by_id = {i["id"]: i for i in evidence.items}
        head, video = by_id.get("in_place_head"), by_id.get("in_place_video")
        if not head or not video or video["kind"] != "support":
            return
        head_share, video_share = head["points"] / head["max_points"], video["points"] / video["max_points"]
        if head_share >= 0.4 and video_share <= 0.15:
            evidence.penalty("sources_disagree", "downward_video", max_points, ramp(head_share - video_share, 0.3, 0.8),
                             "The downward view shows little movement while the head moves, so this is "
                             "probably upper-body movement rather than stepping in place.")

    def common_penalties(self, evidence: Evidence, penalties: dict[str, float], a: int, b: int,
                         posture_low: float = 0.10, posture_high: float = 0.25) -> None:
        bad = 1.0 - frac(self.head_ok[a:b])
        evidence.penalty("tracking", "head_track", penalties.get("tracking", 0), ramp(bad, 0.05, 0.4),
                         f"Head tracking is missing for {bad * 100:.0f}% of the interval.", bad, "fraction")
        if "posture" in penalties:
            change = self.height_change_m(a, b)
            evidence.penalty("posture", "head_track", penalties["posture"], ramp(change, posture_low, posture_high),
                             f"Head height changes by {change * 100:.0f} cm around the interval "
                             "(sitting down, standing up or bending is an ordinary explanation).", change, "m")
        if "sitting" in penalties:
            share = frac(self.activity[a:b] == ACTIVITY_CODES["sitting"])
            evidence.penalty("sitting", "context", penalties["sitting"], ramp(share, 0.3, 0.8),
                             "The existing activity classification says sitting here.", share, "fraction")
        if "stairs" in penalties:
            share = self.stairs_share(a, b)
            evidence.penalty("stairs", "head_track", penalties["stairs"], ramp(share, 0.2, 0.8),
                             "The interval is on stairs, where pausing on a step is common.", share, "fraction")


def finish(context: Context, label_id: str, a: int, b: int, evidence: Evidence, summary: str,
           measurements: dict[str, Any], kind: str = "") -> dict[str, Any] | None:
    config = context.config
    strength_cfg = config["strength"]
    score = evidence.score()
    if score < strength_cfg["min_report_score"]:
        return None
    independent = evidence.independent_sources(strength_cfg["source_share_for_independent"])
    if score >= strength_cfg["high_score"] and len(independent) >= 2:
        strength = "high"
    elif score >= strength_cfg["medium_score"]:
        strength = "medium"
    else:
        strength = "low"
    limitations = list(dict.fromkeys(evidence.limitations))
    if score >= strength_cfg["high_score"] and strength != "high":
        limitations.append("Only one measured source supports this candidate, so the evidence strength is capped at medium.")
    limitations.append("Detection baseline, not validated against reviewed labels. Confirm with the video.")
    start_s, end_s = float(context.t[a]), float(min(context.t[min(b, context.n - 1)], context.signals.duration_s))
    label = config["labels"][label_id]
    return {
        "id": f"{label_id}-{int(round(start_s * 1000)):08d}",
        "label_id": label_id, "label_name": label["name"], "kind": kind,
        "start_s": round(start_s, 2), "end_s": round(end_s, 2), "duration_s": round(end_s - start_s, 2),
        "score": score, "score_kind": SCORE_KIND, "evidence_strength": strength,
        "summary": summary,
        "evidence": evidence.items,
        "sources": sorted({i["source"] for i in evidence.items if i["kind"] == "support" and i["points"] > 0}),
        "independent_sources": independent,
        "measurements": {k: (None if isinstance(v, float) and not np.isfinite(v) else (round(v, 4) if isinstance(v, float) else v))
                         for k, v in measurements.items()},
        "limitations": limitations,
    }


# ----------------------------------------------------------------------------------------------
# detectors


def detect_freezing(context: Context, stats: dict[str, Any]) -> list[dict[str, Any]]:
    cfg, c = context.config["freezing_like"], context
    points, penalties = cfg["points"], cfg["penalties"]
    out = []
    window = int(cfg["context_s"] * GRID_HZ)
    for a, b in runs_of(c.halt):
        duration = (b - a) / GRID_HZ
        if duration < cfg["min_s"]:
            continue
        before = frac(c.loco[max(0, a - window):a])
        after = frac(c.loco[b:min(c.n, b + window)])
        if before < cfg["min_walking_before_fraction"]:
            continue                                   # nothing was under way: not an interruption of walking
        if duration > cfg["max_s"]:
            stats["long_stops_not_reported"] += 1      # walking ended; treated as ordinary standing
            continue
        height_change = c.height_change_m(a, b)
        if np.isfinite(height_change) and height_change >= cfg["posture_change_exclude_m"]:
            stats["posture_changes_not_reported"] += 1
            continue
        evidence = Evidence()
        progress = med(c.progress[a:b])
        evidence.support("progress_stopped", "head_track", points["progress_stopped"],
                         ramp(progress, context.config["motion"]["halt_speed_mps"], 0.015),
                         f"Net forward progress is {progress:.2f} m/s for {duration:.1f} s.", progress, "m/s")
        evidence.support("walking_before", "head_track", points["walking_before"], ramp(before, 0.3, 0.9),
                         f"Walking was under way in {before * 100:.0f}% of the {cfg['context_s']:.0f} s before.", before, "fraction")
        evidence.support("walking_resumes", "head_track", points["walking_resumes"], ramp(after, 0.2, 0.7),
                         f"Walking resumes in {after * 100:.0f}% of the {cfg['context_s']:.0f} s after.", after, "fraction")
        c.in_place_head(evidence, "in_place_head", points["in_place_head"], a, b)
        c.in_place_video(evidence, "in_place_video", points["in_place_video"], a, b, "during the stop")
        evidence.support("duration", "head_track", points["duration"], 1.0 - ramp(duration, 10.0, cfg["max_s"]),
                         f"Duration {duration:.1f} s is within the range of short freezing episodes.", duration, "s")
        in_place = [i for i in evidence.items if i["id"] in ("in_place_head", "in_place_video") and i["kind"] == "support"]
        if sum(i["points"] for i in in_place) < cfg["min_in_place_share"] * sum(i["max_points"] for i in in_place):
            stats["quiet_stops_not_reported"] += 1     # the body is quiet: an ordinary stop
            continue
        turn = c.yaw_change_deg(a, b)
        evidence.penalty("turning", "head_track", penalties["turning"], ramp(turn, cfg["turn_in_place_deg"], cfg["turn_in_place_deg"] * 2.5),
                         f"The head turns about {turn:.0f} degrees during the stop (turning in place is an ordinary explanation).", turn, "deg")
        c.disagreement(evidence, penalties["sources_disagree"])
        c.common_penalties(evidence, penalties, a, b, cfg["posture_change_penalty_m"], cfg["posture_change_exclude_m"])
        summary = (f"Walking was under way, then progress stopped for {duration:.1f} s while the body kept moving in place"
                   + (" and walking resumed." if after >= 0.2 else "."))
        item = finish(c, "freezing_like", a, b, evidence, summary,
                      {"duration_s": duration, "median_progress_mps": progress, "walking_before_fraction": before,
                       "walking_after_fraction": after, "median_sway_m": med(c.sway[a:b]),
                       "median_flow_px": med(c.flow[a:b]), "yaw_change_deg": turn, "height_change_m": height_change})
        if item:
            out.append(item)
        else:
            stats["below_report_score"] += 1
    return out


def detect_hesitation(context: Context, stats: dict[str, Any], taken: np.ndarray) -> list[dict[str, Any]]:
    cfg, c = context.config["gait_hesitation"], context
    points, penalties, motion = cfg["points"], cfg["penalties"], context.config["motion"]
    out = []
    flank = int(cfg["flank_s"] * GRID_HZ)

    # (a) short interruption of progress inside walking
    with np.errstate(invalid="ignore"):
        dip = c.head_ok & (c.progress_fast < motion["halt_speed_mps"] * 1.3)
    for a, b in runs_of(dip):
        if (b - a) / GRID_HZ < cfg["interruption_min_dip_s"] or taken[a:b].any():
            continue
        before, after = frac(c.loco[max(0, a - flank):a]), frac(c.loco[b:min(c.n, b + flank)])
        if before < 0.5 or after < 0.5:
            continue
        speed_before = med(c.progress_fast[max(0, a - flank):a])
        if not speed_before >= cfg["interruption_min_speed_before_mps"]:
            stats["slow_stepping_pauses_not_reported"] += 1   # the pause between two slow steps, not an interruption
            continue
        # The 1 s travel window shortens a stop by about half a second at each end; take the
        # interval out to where travel is back to half of what it was before.
        reach = int(0.6 * GRID_HZ)
        while a > 0 and reach and c.progress_fast[a - 1] < 0.5 * speed_before:
            a, reach = a - 1, reach - 1
        reach = int(0.6 * GRID_HZ)
        while b < c.n and reach and c.progress_fast[b] < 0.5 * speed_before:
            b, reach = b + 1, reach - 1
        duration = (b - a) / GRID_HZ
        if not cfg["interruption_min_s"] <= duration <= cfg["interruption_max_s"]:
            continue
        evidence = Evidence()
        level = med(c.progress_fast[a:b])
        evidence.support("progress_dip", "head_track", points["progress_dip"], ramp(level, motion["halt_speed_mps"] * 1.3, 0.02),
                         f"Movement drops to {level:.2f} m/s for {duration:.1f} s.", level, "m/s")
        evidence.support("walking_both_sides", "head_track", points["walking_both_sides"], ramp(min(before, after), 0.5, 0.9),
                         f"Walking is under way before ({before * 100:.0f}%) and after ({after * 100:.0f}%).", min(before, after), "fraction")
        evidence.support("abrupt", "head_track", points["abrupt"], ramp(speed_before, 0.10, 0.30),
                         f"Speed just before was {speed_before:.2f} m/s, so the drop is "
                         + ("abrupt." if speed_before >= 0.2 else "small."), speed_before, "m/s")
        c.in_place_head(evidence, "in_place_head", points["in_place_head"], a, b)
        c.in_place_video(evidence, "in_place_video", points["in_place_video"], a, b, "during the interruption")
        turn = c.yaw_change_deg(max(0, a - 10), min(c.n, b + 10))
        evidence.penalty("turning", "head_track", penalties["turning"], ramp(turn, 40.0, 110.0),
                         f"The head turns about {turn:.0f} degrees here (slowing down to turn is an ordinary explanation).", turn, "deg")
        c.disagreement(evidence, penalties["sources_disagree"])
        c.common_penalties(evidence, penalties, a, b)
        item = finish(c, "gait_hesitation", a, b, evidence,
                      f"Walking is interrupted for {duration:.1f} s and then continues.",
                      {"duration_s": duration, "median_speed_mps": level, "speed_before_mps": speed_before,
                       "yaw_change_deg": turn, "median_flow_px": med(c.flow[a:b])}, kind="interruption")
        if item:
            out.append(item)
        else:
            stats["below_report_score"] += 1

    # (b) delayed start: movement in place right before walking begins after standing still
    fl = context.config["freezing_like"]
    sway_low = max(fl["sway_quiet_floor_m"], (c.sway_quiet if np.isfinite(c.sway_quiet) else 0.0) * 1.5)
    with np.errstate(invalid="ignore"):
        active = c.head_ok & (c.sway >= sway_low) & ~c.loco
    still_window = int(cfg["start_still_before_s"] * GRID_HZ)
    for a_loco, b_loco in runs_of(c.loco):
        if a_loco < still_window or frac(c.loco[a_loco - still_window:a_loco]) > 0.1:
            continue                                   # walking was already under way shortly before
        a = a_loco
        limit = max(0, a_loco - int(cfg["start_max_s"] * GRID_HZ))
        gap = 0
        while a > limit and gap <= 6:                  # walk back over movement in place, bridging 0.3 s
            a -= 1
            gap = 0 if active[a] else gap + 1
        a += gap
        duration = (a_loco - a) / GRID_HZ
        if duration < cfg["start_min_s"] or taken[a:a_loco].any():
            continue
        if duration >= cfg["start_max_s"] - 0.3:
            stats["ongoing_activity_not_reported"] += 1   # busy in place for a long time, not a start attempt
            continue
        evidence = Evidence()
        evidence.support("delay", "head_track", points["delay"], ramp(duration, cfg["start_min_s"], 4.0),
                         f"The body moves in place for {duration:.1f} s before walking starts.", duration, "s")
        still = 1.0 - frac(c.loco[max(0, a - still_window):a])
        evidence.support("still_before", "head_track", points["still_before"], ramp(still, 0.5, 1.0),
                         "No walking in the seconds before.", still, "fraction")
        follows = min(1.0, (b_loco - a_loco) / GRID_HZ / 4.0)
        evidence.support("walking_follows", "head_track", points["walking_follows"], follows,
                         f"Walking follows for {(b_loco - a_loco) / GRID_HZ:.1f} s.", (b_loco - a_loco) / GRID_HZ, "s")
        c.in_place_head(evidence, "in_place_head", points["in_place_head"], a, a_loco)
        c.in_place_video(evidence, "in_place_video", points["in_place_video"], a, a_loco, "before walking starts")
        turn = c.yaw_change_deg(a, a_loco)
        evidence.penalty("turning", "head_track", penalties["turning"], ramp(turn, 50.0, 140.0),
                         f"The head turns about {turn:.0f} degrees before walking starts (turning towards the path is an ordinary explanation).", turn, "deg")
        c.disagreement(evidence, penalties["sources_disagree"])
        c.common_penalties(evidence, penalties, a, a_loco)
        item = finish(c, "gait_hesitation", a, a_loco, evidence,
                      f"Movement in place for {duration:.1f} s before walking starts.",
                      {"duration_s": duration, "yaw_change_deg": turn, "height_change_m": c.height_change_m(a, a_loco),
                       "median_sway_m": med(c.sway[a:a_loco]), "median_flow_px": med(c.flow[a:a_loco])}, kind="delayed_start")
        if item:
            out.append(item)
        else:
            stats["below_report_score"] += 1
    return out


def detect_gait_quality(context: Context, stats: dict[str, Any]) -> list[dict[str, Any]]:
    """Shuffling-like and slowed gait from sustained level walking."""
    cfg, c = context.config["gait_quality"], context
    labels = context.config["labels"]
    window = cfg["window_s"]
    with np.errstate(invalid="ignore", divide="ignore"):
        level_walk = c.loco & ~c.stairs & c.head_ok
        speed = rolling_mean(np.where(level_walk, c.progress, np.nan), window)
        rate = rolling_mean(np.where(level_walk, c.step_rate, np.nan), window)
        step_length = np.where(rate > 0.3, speed / rate, np.nan)
        shuffle = level_walk & (step_length < cfg["short_step_m"]) & (rate >= cfg["shuffle_min_step_rate_hz"])
        slow = level_walk & (speed < cfg["slow_speed_mps"]) & ~shuffle
    out = []
    for label_id, mask in (("shuffling_like", shuffle), ("slow_gait", slow)):
        if not labels[label_id]["enabled"]:
            continue
        mask = drop_short(close_gaps(mask, cfg["merge_gap_s"]) & c.head_ok & ~c.stairs, cfg["min_bout_s"])
        for a, b in runs_of(mask):
            duration = (b - a) / GRID_HZ
            turning = frac(c.turning[a:b])
            if turning > cfg["max_turning_fraction"]:
                stats["turning_bouts_not_reported"] += 1
                continue
            travelled = float(np.nansum(c.progress[a:b]) / GRID_HZ)
            if travelled < cfg["min_bout_travel_m"] or c.height_change_m(a, b, 0.0) > cfg["max_height_range_m"]:
                stats["short_or_uneven_bouts_not_reported"] += 1   # moving about a chair or table, or changing posture
                continue
            evidence = Evidence()
            v, r, length = med(speed[a:b]), med(rate[a:b]), med(step_length[a:b])
            if label_id == "slow_gait":
                points = cfg["points_slow"]
                evidence.support("speed", "head_track", points["speed"], ramp(v, cfg["slow_speed_mps"], cfg["very_slow_speed_mps"]),
                                 f"Walking speed is {v:.2f} m/s (reference for slow household walking: below {cfg['slow_speed_mps']:.2f} m/s).", v, "m/s")
                evidence.support("rhythm", "head_track", points["rhythm"], ramp(r, 0.4, 0.8),
                                 f"A stepping rhythm is present ({r:.1f} step-like head bobs per second), so this is walking rather than drifting.", r, "Hz")
                summary = f"Level walking at {v:.2f} m/s for {duration:.0f} s."
            else:
                points = cfg["points_shuffle"]
                evidence.support("step_length", "head_track", points["step_length"], ramp(length, cfg["short_step_m"], cfg["very_short_step_m"]),
                                 f"Estimated step length is {length:.2f} m (speed {v:.2f} m/s at {r:.1f} steps/s; head-motion estimate, not foot contacts).", length, "m")
                evidence.support("rhythm_kept", "head_track", points["rhythm_kept"],
                                 ramp(r, cfg["shuffle_min_step_rate_hz"], cfg["shuffle_full_step_rate_hz"]),
                                 f"The stepping rhythm is kept at {r:.1f} steps per second while each step is short.", r, "Hz")
                summary = f"Level walking with short steps (about {length:.2f} m at {r:.1f} steps/s) for {duration:.0f} s."
            evidence.support("sustained", "head_track", points["sustained"], ramp(duration, cfg["min_bout_s"], cfg["min_bout_s"] * 3),
                             f"The pattern is sustained for {duration:.0f} s.", duration, "s")
            evidence.support("straight", "head_track", points["straight"], 1.0 - ramp(turning, 0.1, cfg["max_turning_fraction"]),
                             f"Turning takes up {turning * 100:.0f}% of the interval.", turning, "fraction")
            if c.video_share(a, b) >= 0.6 and np.isfinite(c.flow_quiet):
                flow = med(c.flow[a:b])
                low = c.flow_quiet * context.config["freezing_like"]["flow_quiet_factor"]
                evidence.support("video_moving", "downward_video", points["video_moving"],
                                 ramp(flow, low, max(low * 1.5, c.flow_walk if np.isfinite(c.flow_walk) else low * 2)),
                                 f"Downward view confirms continuous movement (image motion {flow:.2f} px/frame, quiet standing {c.flow_quiet:.2f}).",
                                 flow, "px/frame")
            else:
                evidence.missing("video_moving", "downward_video", points["video_moving"],
                                 "The downward camera has no footage for this interval, so the steps could not be seen."
                                 if c.signals.downward_coverage_s else "This trial has no downward camera, so the steps could not be seen.")
            evidence.penalty("turning", "head_track", cfg["penalties"]["turning"], ramp(turning, 0.25, cfg["max_turning_fraction"]),
                             "Much of the interval is turning, where short slow steps are ordinary.", turning, "fraction")
            c.common_penalties(evidence, {"tracking": cfg["penalties"]["tracking"]}, a, b)
            evidence.limitations.append("Step rate and step length are estimated from head motion; feet are not tracked.")
            if label_id == "slow_gait":
                evidence.limitations.append("Slow walking in a small room or near furniture can be ordinary.")
            item = finish(c, label_id, a, b, evidence, summary,
                          {"duration_s": duration, "distance_m": travelled, "median_speed_mps": v, "median_step_rate_hz": r,
                           "median_step_length_m": length, "turning_fraction": turning,
                           "median_bob_amplitude_m": med(c.bob_amp[a:b]), "median_flow_px": med(c.flow[a:b])})
            if item:
                out.append(item)
            else:
                stats["below_report_score"] += 1
    return out


# ----------------------------------------------------------------------------------------------


def detect(signals: TrialSignals, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run every enabled detector and return the detection document (without ``generated_at``)."""
    config = merge_config(config)
    context = Context(signals, config)
    stats = {"quiet_stops_not_reported": 0, "long_stops_not_reported": 0, "posture_changes_not_reported": 0,
             "turning_bouts_not_reported": 0, "short_or_uneven_bouts_not_reported": 0, "ongoing_activity_not_reported": 0, "slow_stepping_pauses_not_reported": 0, "below_report_score": 0}
    detections: list[dict[str, Any]] = []
    labels = config["labels"]
    head = bool(signals.sources.get("head_track", {}).get("available"))
    if head:
        taken = np.zeros(context.n, dtype=bool)
        if labels["freezing_like"]["enabled"]:
            freezing = detect_freezing(context, stats)
            detections += freezing
            for item in freezing:
                taken[context.sl(item["start_s"], item["end_s"])] = True
        if labels["gait_hesitation"]["enabled"]:
            detections += detect_hesitation(context, stats, taken)
        detections += detect_gait_quality(context, stats)
    detections.sort(key=lambda d: (d["start_s"], d["label_id"]))
    seen: dict[str, int] = {}
    for item in detections:                      # IDs stay unique even if two candidates share a start
        seen[item["id"]] = seen.get(item["id"], 0) + 1
        if seen[item["id"]] > 1:
            item["id"] = f"{item['id']}-{seen[item['id']]}"
    counts: dict[str, dict[str, int]] = {}
    for item in detections:
        bucket = counts.setdefault(item["label_id"], {"total": 0, "high": 0, "medium": 0, "low": 0})
        bucket["total"] += 1
        bucket[item["evidence_strength"]] += 1
    notes = [
        "Candidates for manual review. Time without a candidate is not evidence that no symptom occurred.",
        "Scores are uncalibrated: they rank evidence, they are not probabilities.",
        "Tremor-like movement is not detected: the head track and the 15 fps downward video show no usable 3.5-7 Hz component.",
        "Audio is not used for scoring: footstep sounds could not be separated from speech and room noise.",
    ]
    if not head:
        notes.insert(0, "No head trajectory is saved for this trial, so nothing could be detected.")
    return {
        "schema": SCHEMA, "trial_id": signals.trial_id, "run_id": run_id(config),
        "detector": {"name": DETECTOR_NAME, "version": DETECTOR_VERSION, "config_sha256": config_hash(config), "config": config},
        "session_clock": {"name": "dashboard session time", "units": "s", "duration_s": signals.duration_s},
        "downward_coverage_s": list(signals.downward_coverage_s) if signals.downward_coverage_s else None,
        "sources": signals.sources,
        "source_labels": SOURCE_LABELS,
        "baselines": {k: (None if not np.isfinite(v) else round(float(v), 4)) for k, v in context.baselines.items()},
        "labels": [{"id": key, "name": value["name"], "color": value["color"], "description": value["description"]}
                   for key, value in labels.items() if value["enabled"]],
        "detections": detections,
        "summary": {"count": len(detections), "by_label": counts, "not_reported": stats, "notes": notes},
    }
