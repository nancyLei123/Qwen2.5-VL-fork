import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("scene_playback", SCRIPTS / "build_aria_scene_playback.py")
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)
sys.path.pop(0)


def test_video_dashboard_labels_preserve_existing_trim_offset():
    segments = [{"start_s": 0, "end_s": 4, "label": "standing"}]
    mapped = MODULE.convert_segments(segments, -15.133333, 236.8)
    assert mapped[0]["start_s"] == pytest.approx(15.133333)
    assert mapped[0]["end_s"] == pytest.approx(19.133333)
    assert segments[0]["start_s"] == 0


def test_three_camera_basis_preserves_optical_forward_and_upright():
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [12, 28, 78], degrees=True).as_matrix()
    transform[:3, 3] = [2, -3, 1]
    pose = MODULE.three_pose(transform)
    rotation = Rotation.from_quat(pose[3:]).as_matrix()
    np.testing.assert_allclose(rotation @ [0, 0, -1], transform[:3, :3] @ [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(rotation @ [0, 1, 0], transform[:3, :3] @ [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(pose[:3], [2, -3, 1])


def test_frame_count_mismatch_fails_instead_of_stretching_time():
    timestamps = np.array([100, 200, 300])
    metadata = {"written_frames": 2, "fps": 30}
    with pytest.raises(ValueError, match="frame counts"):
        MODULE.verify_frame_mapping(timestamps, metadata, {})


@pytest.mark.parametrize("trial", ["Ntrial3", "60minutes_1", "RoomScan3c", "Fuchs", "Fuchs22", "Fuchs333"])
def test_ui_only_publish_preserves_data_labels_and_models(tmp_path, monkeypatch, trial):
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    output = tmp_path / "runs" / trial / "environment_reconstruction"
    output.mkdir(parents=True)
    preserved = {
        output / "scene_playback_data.json": json.dumps({"trial_id": trial, "poses": [None]}),
        output / "splat_7000.ply": "unchanged model",
        output.parent / "session_dashboard_manifest.json": "unchanged labels",
    }
    for path, content in preserved.items():
        path.write_text(content)
    page = MODULE.publish_ui(trial)
    assert f"{trial} | Synchronized 3D Playback" in page.read_text()
    assert "Click any timeline to play from that time" in page.read_text()
    assert 'id="viewAimStatus"' in page.read_text()
    for label in ("Around camera", "Nearby route", "Aria viewpoint", "Reset view", "High / Retina", "Balanced"):
        assert label in page.read_text()
    for filename in MODULE.PLAYER_MODULES:
        assert (output / filename).read_bytes() == (MODULE.TEMPLATES / filename).read_bytes()
    assert (output / "spatial_plotly.min.js").stat().st_size > 1_000_000
    assert 'id="spatialPlot"' in page.read_text()
    for path, content in preserved.items():
        assert path.read_text() == content


class Camera:
    def get_time_offset_sec_device_camera(self):
        return -0.0025

    def get_transform_device_camera(self):
        return self

    def to_matrix(self):
        transform = np.eye(4)
        transform[1, 3] = 0.06
        return transform


def test_camera_pose_applies_time_offset_extrinsic_and_upright_rotation():
    trajectory = {
        "times_ns": np.array([0, 10_000_000]),
        "translations": np.array([[0, 0, 0], [1, 0, 0]]),
        "quaternions": np.array([[0, 0, 0, 1], [0, 0, 0, 1]]),
        "frame": pd.DataFrame({"quality_score": [1.0, 1.0]}),
    }
    pose = MODULE.camera_pose(trajectory, 2_500_000, Camera(), 0)
    np.testing.assert_allclose(pose[:3, 3], [0.5, 0.06, 0])
    np.testing.assert_allclose(pose[:3, :3], Rotation.from_euler("z", -90, degrees=True).as_matrix())
    assert MODULE.camera_pose(trajectory, 2_500_000, Camera(), 61_000_000) is None
    assert MODULE.camera_pose(trajectory, 10_000_000, Camera(), 0) is None
    trajectory["frame"].loc[0, "quality_score"] = 0.2
    assert MODULE.camera_pose(trajectory, 2_500_000, Camera(), 0) is None
    trajectory["frame"].loc[0, "quality_score"] = np.nan
    assert MODULE.camera_pose(trajectory, 2_500_000, Camera(), 0) is None
