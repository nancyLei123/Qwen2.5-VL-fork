import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reconstruct_aria_environment.py"
SPEC = importlib.util.spec_from_file_location("reconstruct_aria_environment", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_sample_evenly_keeps_endpoints():
    values = np.arange(30).reshape(10, 3)
    sampled = MODULE._sample_evenly(values, 4)
    assert sampled.shape == (4, 3)
    np.testing.assert_array_equal(sampled[0], values[0])
    np.testing.assert_array_equal(sampled[-1], values[-1])


def test_density_mesh_recovers_synthetic_surface():
    grid = np.linspace(-1.0, 1.0, 35)
    x, y = np.meshgrid(grid, grid)
    floor = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    wall = np.column_stack([x.ravel(), np.zeros(x.size), y.ravel() + 1.0])
    points = np.vstack([floor, wall]).astype(np.float32)
    mesh, summary = MODULE.build_density_mesh(
        points,
        voxel_size_m=0.12,
        density_sigma=1.0,
        density_level=0.10,
        marching_step=1,
        min_component_faces=20,
    )
    assert len(mesh.vertices) > 100
    assert len(mesh.faces) > 100
    assert summary["mesh_vertices"] == len(mesh.vertices)
    assert summary["mesh_faces"] == len(mesh.faces)


def test_video_timing_uses_device_offset_not_duration(tmp_path):
    video = tmp_path / "full.mp4"
    (tmp_path / "audio_metadata.json").write_text(json.dumps({"video_start_device_ns": 1_000_000_000}))
    trajectory = pd.DataFrame({"tracking_timestamp_us": [1_966_140, 11_966_140]})
    timing = MODULE.resolve_video_timing(video, trajectory, tmp_path / "trial")
    assert timing["mode"] == "device_timestamp_offset"
    assert timing["first_pose_video_s"] == pytest.approx(0.96614)
    assert timing["last_pose_video_s"] == pytest.approx(10.96614)


def test_trimmed_video_uses_original_start_metadata(tmp_path):
    extraction = tmp_path / "extraction"
    extraction.mkdir()
    (extraction / "audio_metadata.json").write_text(json.dumps({"video_start_device_ns": 1_000_000_000}))
    video = tmp_path / "trimmed.mp4"
    video.with_suffix(".build.json").write_text(json.dumps({
        "source_video_path": str(extraction / "full.mp4"), "video_trim_start_s": 3.0,
    }))
    timing = MODULE.resolve_video_timing(
        video, pd.DataFrame({"tracking_timestamp_us": [2_000_000, 8_000_000]}), tmp_path / "trial"
    )
    assert timing["first_pose_video_s"] == pytest.approx(-2.0)
    assert timing["last_pose_video_s"] == pytest.approx(4.0)


def test_unknown_timestamp_origin_is_not_assumed_aligned(tmp_path):
    timing = MODULE.resolve_video_timing(
        tmp_path / "video.mp4", pd.DataFrame({"tracking_timestamp_us": [1, 2]}), tmp_path
    )
    assert timing["mode"] == "unverified"
    assert timing["first_pose_video_s"] is None


def test_display_path_keeps_last_pose_and_timestamp_offset():
    trajectory = pd.DataFrame({"tracking_timestamp_us": np.arange(12_001) * 1_000 + 5_000_000})
    display = MODULE.display_trajectory(trajectory, 0.96614)
    assert len(display) == 5_000
    assert display["video_s"].iloc[0] == pytest.approx(0.96614)
    assert display["video_s"].iloc[-1] == pytest.approx(12.96614)


def test_point_graph_mismatch_is_rejected(tmp_path):
    points = tmp_path / "points.csv"
    pd.DataFrame({
        "graph_uid": ["wrong"], "px_world": [0], "py_world": [0], "pz_world": [0],
        "inv_dist_std": [0], "dist_std": [0],
    }).to_csv(points, index=False)
    with pytest.raises(ValueError, match="graph_uid"):
        MODULE.load_and_filter_points(points, np.zeros((1, 3)), .005, .01, 8, expected_graph_uid="expected")
