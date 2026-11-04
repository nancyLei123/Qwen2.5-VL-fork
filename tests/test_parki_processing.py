import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_parkinson_motion_review import resample_trajectory
from continuous_motion_predictions import ContinuousConfig, process_trial
from pd_insighter_environment_navigation import inject_environment_tab
from verify_trial_audio_sync import local_offset
from video_time_information import _creation_time_in_local_zone


def test_audio_offset_sign_and_precision():
    rng = np.random.default_rng(4)
    rate = 1000
    aria = rng.normal(size=60 * rate)
    downward = aria[7321:]
    result = local_offset(aria, downward, rate, 5.0, 20.0, 7.0)
    assert abs(result["audio_offset_s"] - 7.321) < 1 / rate
    assert result["absolute_waveform_correlation"] > 0.99


def test_verified_utc_metadata_preserves_legacy_clock_behavior():
    raw = "2026-10-28T01:21:00Z"
    verified = {"downward_media": {"creation_time_timezone": "UTC"}}
    assert str(_creation_time_in_local_zone(raw, verified)) == "2026-10-27 21:21:00"
    assert str(_creation_time_in_local_zone(raw, {})) == "2026-10-28 01:21:00"


def test_device_timestamps_align_slam_independently_of_wall_clock(tmp_path):
    t = np.arange(11, dtype=float)
    columns = {"tracking_timestamp_us": 1e8 + t * 1e6, "utc_timestamp_ns": 9e17 + t * 1e9,
               "tx_world_device": t, "ty_world_device": t * 0, "tz_world_device": t * 0 + 1.5}
    for prefix in ("device_linear_velocity", "angular_velocity"):
        for axis in "xyz":
            columns[f"{prefix}_{axis}_device"] = t * 0
    source = tmp_path / "trajectory.csv"
    pd.DataFrame(columns).to_csv(source, index=False)
    result = resample_trajectory(source, 1234, 3, 0.5, session_start_tracking_us=1.02e8)
    assert result["x_m"].iloc[0] == 2.0
    assert result["x_m"].iloc[-1] == 5.0


def test_normal_classifier_cannot_overwrite_simulated_trial(tmp_path):
    trial = tmp_path / "Parki"
    trial.mkdir()
    (trial / "trial_manifest.json").write_text(json.dumps({"trial_type": "parkinson_simulated"}))
    assert process_trial("Parki", tmp_path, ContinuousConfig()) is None


def test_clinical_navigation_is_not_duplicated(tmp_path):
    (tmp_path / "pd_insighter_parkinson_review.html").write_text("review")
    page = '<div class="nav-tabs"><a href="index.html">Full Session</a></div>'
    first = inject_environment_tab(page, tmp_path)
    second = inject_environment_tab(first, tmp_path)
    assert first == second
    assert first.count('data-view="parkinson"') == 1
