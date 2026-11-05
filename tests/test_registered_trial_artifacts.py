"""Local artifact checks; skipped in code-only checkouts without research data."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

RUNS = Path(__file__).resolve().parents[1] / "runs"
TRIALS = ["Fuchs", "Fuchs22", "Fuchs333"]


def read(path):
    if not path.exists():
        pytest.skip(f"Local artifact not yet available: {path.name}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("trial", TRIALS)
def test_analysis_and_navigation(trial):
    run = RUNS / trial
    summary = read(run / "summary.json")
    session = read(run / "session_dashboard_manifest.json")
    assert summary["accuracy"] is None
    assert summary["classification_validated"] is False
    assert session["trial_type"] == "parkinson_simulated"
    assert session["media_mode"] == "aria_only"
    assert not session["downward_session_video_uri"]
    assert (run / session["aria_session_video_uri"].split("?")[0]).is_file()
    assert session["stride_review"]["events"]
    timeline = read(run / "raw_parameter_diagnostics/continuous_motion_segments.json")["segments"]
    assert timeline[0]["start_s"] == 0
    assert timeline[-1]["end_s"] == pytest.approx(session["session_duration_s"], abs=.001)
    assert all(a["end_s"] == b["start_s"] for a, b in zip(timeline, timeline[1:]))
    pages = ["index.html", "pd_insighter_manual_labeling.html", "pd_insighter_stride_review.html",
             "pd_insighter_transition_review.html", "pd_insighter_parkinson_review.html"]
    for filename in pages:
        page = (run / filename).read_text()
        assert "Aria-only recording" in page
        assert "environment_reconstruction/playback.html" in page
        assert 'id="downwardVideo" src=""' not in page
        assert "PD Insighter navigation" in page


@pytest.mark.parametrize("trial", TRIALS)
def test_camera_geometry_and_honest_coverage(trial):
    root = RUNS / trial / "environment_reconstruction"
    data = read(root / "scene_playback_data.json")
    times = np.asarray(data["recording_times"])
    assert len(times) == len(data["poses"]) == len(data["pose_status"])
    assert np.all(np.diff(times) > 0)
    assert data["duration_s"] == pytest.approx(len(times) / data["fps"])
    assert data["provenance"]["estimated_pose_count"] == 0
    for pose, status in zip(data["poses"], data["pose_status"]):
        assert (pose is None) == (status == "unavailable")
        if pose is not None:
            assert np.isfinite(pose).all()
    coverage = np.zeros(len(times), dtype=bool)
    checked = 0
    for scene in data["scenes"]:
        assert (root / scene["ply"]).is_file()
        dataset = read(root / Path(scene["ply"]).parent.parent / "dataset_manifest.json")
        assert dataset["trial_id"] == trial
        assert dataset["world_frame"]["graph_uid"] == data["provenance"]["graph_uid"]
        frame_times = [frame["recording_s"] for frame in dataset["frames"]["items"]]
        assert scene["start_s"] == min(frame_times)
        assert scene["end_s"] <= max(frame_times) + 1.1e-6
        coverage |= (times >= scene["start_s"]) & (times < scene["end_s"])
        for frame in dataset["frames"]["items"]:
            index = int(np.argmin(np.abs(times - frame["recording_s"])))
            assert abs(times[index] - frame["recording_s"]) < 1e-6
            pose = data["poses"][index]
            assert pose is not None
            transform = np.eye(4)
            transform[:3, 3] = pose[:3]
            transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix() @ np.diag([1, -1, -1])
            np.testing.assert_allclose(transform, frame["T_world_camera"], atol=1e-7, rtol=0)
            checked += 1
    assert checked > 0
    assert data["coverage_summary"]["model_seconds"] == pytest.approx(coverage.sum() / data["fps"])
