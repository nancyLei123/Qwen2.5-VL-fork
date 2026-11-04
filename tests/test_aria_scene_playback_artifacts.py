"""Check published playback against independently exported training camera transforms."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1] / "runs/Ntrial3/environment_reconstruction"


@pytest.mark.skipif(not (ROOT / "scene_playback_data.json").is_file(), reason="Ntrial3 artifacts absent")
def test_playback_matches_every_exported_camera():
    data = json.loads((ROOT / "scene_playback_data.json").read_text())
    times = np.asarray(data["recording_times"])
    checked = 0
    for scene in data["scenes"]:
        source = ROOT / Path(scene["ply"]).parent.parent / "dataset_manifest.json"
        manifest = json.loads(source.read_text())
        assert manifest["world_frame"]["graph_uid"] == data["provenance"]["graph_uid"]
        for item in manifest["frames"]["items"]:
            index = int(np.argmin(np.abs(times - item["recording_s"])))
            assert abs(times[index] - item["recording_s"]) < 1e-6
            pose = data["poses"][index]
            assert pose is not None
            actual = np.eye(4)
            actual[:3, 3] = pose[:3]
            actual[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix() @ np.diag([1, -1, -1])
            np.testing.assert_allclose(actual, item["T_world_camera"], atol=1e-7, rtol=0)
            checked += 1
    assert checked > 900


@pytest.mark.skipif(not (ROOT / "scene_playback_data.json").is_file(), reason="Ntrial3 artifacts absent")
def test_playback_estimates_keep_provenance_and_unsupported_endpoints():
    data = json.loads((ROOT / "scene_playback_data.json").read_text())
    if "pose_status" not in data:
        pytest.skip("Playback has not been rebuilt with recovery provenance")
    assert len(data["pose_status"]) == len(data["poses"])
    assert data["poses"][0] is None and data["poses"][-1] is None
    times = np.asarray(data["recording_times"])
    for scene in data["scenes"]:
        covered = np.flatnonzero((times >= scene["start_s"]) & (times < scene["end_s"]))
        assert scene["video_start_s"] == pytest.approx(covered[0] / data["fps"])
        assert scene["video_end_s"] == pytest.approx((covered[-1] + 1) / data["fps"])
    audited = {item["frame"]: item for item in data["pose_recovery_audit"]}
    for index, status in enumerate(data["pose_status"]):
        assert (data["poses"][index] is None) == (status == "unavailable")
        if status == "interpolated_short_gap":
            audit = audited[index]
            assert audit["not_used_for_training"]
            assert audit["span_ms"] <= 50
            assert audit["anchor_times_ns"][0] < audit["query_ns"] < audit["anchor_times_ns"][1]
            assert audit["local_validation"]["samples"] >= 12
            assert audit["local_validation"]["max_position_error_m"] <= 0.01
            assert audit["local_validation"]["max_rotation_error_deg"] <= 1


@pytest.mark.skipif(not (ROOT / "coverage_recovery_audit/before/scene_playback_data.json").is_file(), reason="Audit baseline absent")
def test_recovery_preserves_every_previously_supported_pose_and_label_clock():
    before = json.loads((ROOT / "coverage_recovery_audit/before/scene_playback_data.json").read_text())
    after = json.loads((ROOT / "scene_playback_data.json").read_text())
    for index, pose in enumerate(before["poses"]):
        if pose is not None:
            assert after["poses"][index] == pose
    for key in ["fps", "duration_s", "recording_times", "dashboard_offset_s", "dashboard_duration_s", "labels_url", "video_url"]:
        assert after[key] == before[key]
