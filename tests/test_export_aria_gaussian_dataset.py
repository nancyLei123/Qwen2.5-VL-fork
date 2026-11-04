import importlib.util
from pathlib import Path
import struct

import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_aria_gaussian_dataset.py"
SPEC = importlib.util.spec_from_file_location("aria_gaussian_export", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_aria_gaussian_scene.py"
TRAIN_SPEC = importlib.util.spec_from_file_location("aria_gaussian_train", TRAIN_SCRIPT)
TRAIN_MODULE = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(TRAIN_MODULE)

SUBMAP_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/prepare_aria_gaussian_submaps.py"
)
SUBMAP_SPEC = importlib.util.spec_from_file_location("aria_gaussian_submaps", SUBMAP_SCRIPT)
SUBMAP_MODULE = importlib.util.module_from_spec(SUBMAP_SPEC)
SUBMAP_SPEC.loader.exec_module(SUBMAP_MODULE)


def test_camera_to_colmap_inverts_camera_to_world_transform():
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    transform[:3, 3] = [1.0, 2.0, 3.0]
    qvec, tvec = MODULE.camera_to_colmap(transform)
    rotation_camera_world = Rotation.from_quat(qvec[[1, 2, 3, 0]]).as_matrix()
    np.testing.assert_allclose(rotation_camera_world, transform[:3, :3].T, atol=1e-10)
    np.testing.assert_allclose(tvec, -rotation_camera_world @ transform[:3, 3])
    np.testing.assert_allclose(np.linalg.norm(qvec), 1.0)


def test_pinhole_projection_uses_camera_forward_z():
    points = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0, 0, -1]])
    uv, depth = MODULE.pinhole_project(points, np.eye(4), (100, 100, 50, 50))
    np.testing.assert_allclose(uv[:2], [[50, 50], [100, 50]])
    assert np.isnan(uv[2]).all()
    np.testing.assert_allclose(depth, [2, 2, -1])


def test_frame_selection_keeps_motion_and_maximum_time_gap():
    poses = []
    for x in [0.0, 0.01, 0.02, 0.10, 0.11]:
        pose = np.eye(4)
        pose[0, 3] = x
        poses.append(pose)
    timestamps = [0, 250_000_000, 500_000_000, 750_000_000, 1_500_000_000]
    selected = MODULE.select_frame_indices(poses, timestamps, 0.06, 4.0, 0.75)
    assert selected == [0, 3, 4]


def test_frame_selection_keeps_rotation_without_translation():
    first = np.eye(4)
    second = np.eye(4)
    second[:3, :3] = Rotation.from_euler("y", 5, degrees=True).as_matrix()
    selected = MODULE.select_frame_indices(
        [first, second], [0, 100_000_000], 0.06, 4.0, 0.75
    )
    assert selected == [0, 1]


def test_depth_visibility_rejects_far_point_in_same_tile():
    uv = np.array([[10.0, 10.0], [11.0, 11.0], [80.0, 80.0], [np.nan, 0]])
    depth = np.array([1.0, 3.0, 2.0, 1.0])
    visible = MODULE.depth_visible(uv, depth, 100, 100)
    np.testing.assert_array_equal(visible, [True, False, True, False])


def test_upright_rotation_changes_pixels_and_camera_basis_together():
    image = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    upright, transform = MODULE.rotate_upright(image, np.eye(4))
    np.testing.assert_array_equal(upright, [[3, 1], [4, 2]])
    expected = Rotation.from_euler("z", -90, degrees=True).as_matrix()
    np.testing.assert_allclose(transform[:3, :3], expected, atol=1e-10)
    original_point = np.array([[1.0, 2.0, 5.0]])
    uv, _ = MODULE.pinhole_project(original_point, transform, (10, 10, 5, 5))
    np.testing.assert_allclose(uv[0], [1, 7], atol=1e-10)


def test_colmap_writer_preserves_two_line_image_records(tmp_path):
    frame = {
        "transform_world_camera": np.eye(4),
        "name": "frame_00001.jpg",
    }
    MODULE.write_colmap_model(
        tmp_path,
        100,
        80,
        (50.0, 50.0, 49.5, 39.5),
        [frame],
        np.array([[1.0, 2.0, 3.0]]),
        np.array([[10, 20, 30]], dtype=np.uint8),
    )
    camera = (tmp_path / "sparse" / "0" / "cameras.txt").read_text()
    images = (tmp_path / "sparse" / "0" / "images.txt").read_text()
    points = (tmp_path / "sparse" / "0" / "points3D.txt").read_text()
    assert "1 PINHOLE 100 80 50 50 49.5 39.5" in camera
    assert "frame_00001.jpg\n\n" in images
    assert "1 1 2 3 10 20 30 0" in points

    with (tmp_path / "sparse" / "0" / "cameras.bin").open("rb") as stream:
        assert struct.unpack("<Q", stream.read(8))[0] == 1
        camera_id, model_id, width, height = struct.unpack("<IiQQ", stream.read(24))
        assert (camera_id, model_id, width, height) == (1, 1, 100, 80)
        assert struct.unpack("<dddd", stream.read(32)) == (50, 50, 49.5, 39.5)
    with (tmp_path / "sparse" / "0" / "images.bin").open("rb") as stream:
        assert struct.unpack("<Q", stream.read(8))[0] == 1
        assert struct.unpack("<I", stream.read(4))[0] == 1
    with (tmp_path / "sparse" / "0" / "points3D.bin").open("rb") as stream:
        assert struct.unpack("<Q", stream.read(8))[0] == 1
        assert struct.unpack("<Q", stream.read(8))[0] == 1
        assert struct.unpack("<ddd", stream.read(24)) == (1, 2, 3)
        assert struct.unpack("<BBB", stream.read(3)) == (10, 20, 30)


def test_open_splat_validation_metrics_are_parsed_from_last_report():
    log = "validation loss: 0.2\nvalidation PSNR: 10\nvalidation loss: 0.08\nvalidation PSNR: 23.7\n"
    assert TRAIN_MODULE.parse_validation_metrics(log) == (0.08, 23.7)


def test_validation_image_auto_selects_middle_frame():
    manifest = {
        "frames": {
            "items": [
                {"name": "frame_00001.jpg"},
                {"name": "frame_00002.jpg"},
                {"name": "frame_00003.jpg"},
            ]
        }
    }
    assert TRAIN_MODULE.resolve_validation_image(manifest, "auto") == "frame_00002.jpg"
    assert (
        TRAIN_MODULE.resolve_validation_image(manifest, "frame_00003.jpg")
        == "frame_00003.jpg"
    )


def test_gpu_script_uses_overrideable_binary_and_valid_continuations(tmp_path):
    command = SUBMAP_MODULE.training_command(
        tmp_path / "dataset", tmp_path / "build-mps/opensplat"
    )
    script = SUBMAP_MODULE.write_gpu_script(tmp_path, [command]).read_text()
    assert 'OPEN_SPLAT="${OPEN_SPLAT:-' in script
    assert '"$OPEN_SPLAT"' in script
    assert "--device \\\n  auto" in script
    assert "\n+" not in script


def test_submap_profiles_record_motion_context_around_clinical_action():
    sitting = next(item for item in SUBMAP_MODULE.SUBMAPS if item["name"] == "sitting_context")
    assert sitting["motion_sequence"] == [
        [0.0, 4.0, "standing"],
        [4.0, 13.0, "walking"],
        [13.0, 16.0, "standing"],
        [16.0, 22.0, "sitting"],
        [22.0, 24.0, "standing"],
    ]
    action_profiles = [item for item in SUBMAP_MODULE.SUBMAPS if item.get("selection_source") != "untrained_coverage"]
    assert len(action_profiles) == 4
    assert all(item["motion_sequence"] for item in action_profiles)


def test_gap_submaps_cover_missing_time_with_overlap_without_inventing_action_labels():
    profiles = {item["name"]: item for item in SUBMAP_MODULE.SUBMAPS if item.get("selection_source") == "untrained_coverage"}
    assert profiles["connecting_walk_gap"]["start_s"] < 60
    assert profiles["connecting_walk_gap"]["end_s"] > 78
    assert profiles["final_walk_gap"]["start_s"] < 216
    assert profiles["final_walk_gap"]["end_s"] > 236.5
    assert all(not item["motion_sequence"] and not item["states"] for item in profiles.values())


def test_ply_vertex_count_reads_open_splat_header(tmp_path):
    source = tmp_path / "source.ply"
    source.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 17\nend_header\n"
    )
    assert TRAIN_MODULE.ply_vertex_count(source) == 17
