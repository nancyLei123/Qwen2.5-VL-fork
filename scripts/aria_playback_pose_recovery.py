"""Conservative playback-only interpolation across short low-quality SLAM gaps.

This never changes MPS files or training poses. Recovered poses are estimates,
not measured replacements, and must retain their status in the player.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from export_aria_gaussian_dataset import interpolate_pose

MAX_BRIDGE_NS = 50_000_000
MIN_QUALITY = 0.9


def support_reason(trajectory, query_ns):
    times = trajectory["times_ns"]
    if query_ns < times[0]:
        return "before_trajectory_start"
    if query_ns > times[-1]:
        return "after_trajectory_end"
    right = int(np.searchsorted(times, query_ns))
    left = max(0, right - 1)
    quality = trajectory["frame"].quality_score.to_numpy()[left:right + 1]
    if not np.isfinite(quality).all() or np.min(quality) < MIN_QUALITY:
        return "low_tracking_quality"
    if times[right] - times[left] > 20_000_000:
        return "trajectory_timestamp_gap"
    return "supported"


def _bridge(trajectory, left, right, query_ns):
    indices = [left, right]
    return interpolate_pose(
        trajectory["times_ns"][indices], trajectory["translations"][indices],
        trajectory["quaternions"][indices], query_ns, max_gap_ns=MAX_BRIDGE_NS,
    )[0]


def local_interpolation_check(trajectory, query_ns, span_ns):
    """Mask nearby good poses for a consistency test, not external accuracy."""
    times = trajectory["times_ns"]
    quality = trajectory["frame"].quality_score.to_numpy()
    position_errors, rotation_errors = [], []
    for offset in np.linspace(-2e9, 2e9, 81):
        stamp = query_ns + round(offset)
        left = int(np.searchsorted(times, stamp - span_ns // 2)) - 1
        right = int(np.searchsorted(times, stamp + span_ns // 2))
        if left < 0 or right >= len(times) or times[right] - times[left] > MAX_BRIDGE_NS:
            continue
        if not np.all(np.isfinite(quality[left:right + 1]) & (quality[left:right + 1] >= MIN_QUALITY)):
            continue
        try:
            truth = interpolate_pose(times, trajectory["translations"], trajectory["quaternions"], stamp)[0]
            estimate = _bridge(trajectory, left, right, stamp)
        except ValueError:
            continue
        position_errors.append(float(np.linalg.norm(truth[:3, 3] - estimate[:3, 3])))
        rotation_errors.append(float(np.degrees(Rotation.from_matrix(truth[:3, :3].T @ estimate[:3, :3]).magnitude())))
    return {
        "samples": len(position_errors),
        "max_position_error_m": max(position_errors, default=None),
        "max_rotation_error_deg": max(rotation_errors, default=None),
        "interpretation": "Nearby held-out MPS consistency only; not independent ground truth.",
    }


def recover_short_gap(trajectory, query_ns):
    """Return an estimated device transform and provenance, or a rejection."""
    reason = support_reason(trajectory, query_ns)
    audit = {"reason": reason, "query_ns": int(query_ns)}
    if reason != "low_tracking_quality":
        return None, audit
    times = trajectory["times_ns"]
    quality = trajectory["frame"].quality_score.to_numpy()
    good = np.flatnonzero(np.isfinite(quality) & (quality >= MIN_QUALITY))
    split = int(np.searchsorted(times[good], query_ns))
    if split == 0 or split == len(good):
        return None, {**audit, "rejection": "no_two_sided_reliable_anchors"}
    left, right = int(good[split - 1]), int(good[split])
    span = int(times[right] - times[left])
    audit.update(anchor_indices=[left, right], anchor_times_ns=[int(times[left]), int(times[right])], span_ms=span / 1e6)
    if span > MAX_BRIDGE_NS:
        return None, {**audit, "rejection": "reliable_anchors_more_than_50ms_apart"}
    distance = float(np.linalg.norm(trajectory["translations"][right] - trajectory["translations"][left]))
    rotation = Rotation.from_quat(trajectory["quaternions"][[left, right]])
    angle = float(np.degrees((rotation[0].inv() * rotation[1]).magnitude()))
    audit.update(anchor_translation_m=distance, anchor_rotation_deg=angle)
    if not np.isfinite([distance, angle]).all() or distance > 0.10 or angle > 10:
        return None, {**audit, "rejection": "anchor_motion_too_large"}
    check = local_interpolation_check(trajectory, query_ns, span)
    audit["local_validation"] = check
    if check["samples"] < 12 or check["max_position_error_m"] > 0.01 or check["max_rotation_error_deg"] > 1:
        return None, {**audit, "rejection": "local_interpolation_check_failed"}
    return _bridge(trajectory, left, right, query_ns), {
        **audit, "method": "linear translation + quaternion SLERP between reliable MPS anchors",
        "status": "interpolated_short_gap", "not_used_for_training": True,
    }
