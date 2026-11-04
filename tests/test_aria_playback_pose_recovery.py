import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from aria_playback_pose_recovery import recover_short_gap, support_reason


def trajectory():
    seconds = np.arange(5001) / 1000
    return {
        "times_ns": np.arange(5001, dtype=np.int64) * 1_000_000,
        "translations": np.column_stack([seconds, seconds * 0, seconds * 0]),
        "quaternions": Rotation.from_euler("z", (seconds * 0.1)[:, None]).as_quat(),
        "frame": pd.DataFrame({"quality_score": np.ones(5001)}),
    }


def test_short_interior_gap_is_estimated_and_audited_without_mutating_sources():
    tr = trajectory()
    tr["frame"].loc[2484:2517, "quality_score"] = 0.5
    original = tr["translations"].copy()
    pose, audit = recover_short_gap(tr, 2_500_000_000)
    assert pose is not None
    np.testing.assert_allclose(pose[:3, 3], [2.5, 0, 0])
    assert audit["span_ms"] == 35
    assert audit["status"] == "interpolated_short_gap"
    assert audit["not_used_for_training"]
    assert audit["local_validation"]["samples"] >= 12
    np.testing.assert_array_equal(original, tr["translations"])
    assert tr["frame"].quality_score.iloc[2500] == 0.5


def test_no_endpoint_extrapolation_or_replacement_of_supported_pose():
    tr = trajectory()
    for time in [-1, 5_000_000_001, 2_500_000_000]:
        assert recover_short_gap(tr, time)[0] is None
    assert support_reason(tr, -1) == "before_trajectory_start"
    assert support_reason(tr, 5_000_000_001) == "after_trajectory_end"


def test_long_gap_and_large_displacement_remain_unavailable():
    tr = trajectory()
    tr["frame"].loc[2450:2550, "quality_score"] = 0.5
    assert recover_short_gap(tr, 2_500_000_000)[0] is None
    tr = trajectory()
    tr["frame"].loc[2484:2517, "quality_score"] = 0.5
    tr["translations"][2518, 0] += 1
    assert recover_short_gap(tr, 2_500_000_000)[1]["rejection"] == "anchor_motion_too_large"


def test_missing_quality_is_not_treated_as_a_valid_measurement():
    tr = trajectory()
    tr["frame"].loc[2500, "quality_score"] = np.nan
    assert support_reason(tr, 2_500_000_000) == "low_tracking_quality"
    tr["frame"].loc[:, "quality_score"] = 0.5
    assert recover_short_gap(tr, 2_500_000_000)[0] is None


def test_insufficient_local_validation_cannot_publish_an_estimate():
    tr = trajectory()
    tr["frame"].loc[:, "quality_score"] = 0.5
    tr["frame"].loc[[2483, 2518], "quality_score"] = 1
    pose, audit = recover_short_gap(tr, 2_500_000_000)
    assert pose is None
    assert audit["rejection"] == "local_interpolation_check_failed"
