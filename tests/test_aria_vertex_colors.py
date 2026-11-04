import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_aria_vertex_colors.py"
SPEC = importlib.util.spec_from_file_location("aria_vertex_colors", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pose_interpolation_uses_slerp_and_translation():
    times = np.array([0, 10_000_000])
    xyz = np.array([[0., 0, 0], [2., 0, 0]])
    q = Rotation.from_euler("z", [[0], [90]], degrees=True).as_quat()
    pose, gap = MODULE.interpolate_pose(times, xyz, q, 5_000_000)
    np.testing.assert_allclose(pose[:3, 3], [1, 0, 0])
    np.testing.assert_allclose(pose[:3, :3], Rotation.from_euler("z", 45, degrees=True).as_matrix(), atol=1e-10)
    assert gap == 10_000_000


def test_pose_outside_support_and_large_gap_are_rejected():
    times = np.array([0, 100_000_000])
    xyz = np.zeros((2, 3))
    q = np.array([[0., 0, 0, 1], [0., 0, 0, 1]])
    with pytest.raises(ValueError, match="outside"):
        MODULE.interpolate_pose(times, xyz, q, -1)
    with pytest.raises(ValueError, match="gap"):
        MODULE.interpolate_pose(times, xyz, q, 50_000_000)


def test_world_camera_direction_not_reversed():
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    pose[:3, 3] = [1, 2, 3]
    np.testing.assert_allclose(MODULE.world_to_camera(np.array([[1., 3, 4]]), pose), [[1, 0, 1]], atol=1e-10)


def test_project_rejects_behind_camera():
    class Camera:
        def project(self, point):
            return point[:2] / point[2]
    uv, _ = MODULE.project_points(np.array([[1., 2, 2], [1, 2, -2]]), np.eye(4), Camera())
    np.testing.assert_allclose(uv[0], [.5, 1])
    assert np.isnan(uv[1]).all()


def test_depth_visibility_rejects_far_point_and_invalid_pixel():
    pixels = np.array([[100., 100], [101, 101], [200, 200], [np.nan, 1]])
    mask = MODULE.point_depth_visibility(pixels, np.array([1., 3, 2, 1]), [640, 480])
    np.testing.assert_array_equal(mask, [True, False, True, False])


def test_foreground_mask_is_not_an_unrestricted_texture_pass():
    uv = np.array([[500., 500], [900, 500], [-1, 500], [500, 1500], [np.nan, 0]])
    np.testing.assert_array_equal(MODULE.static_pixel_mask(uv, [1408, 1408]), [True, False, False, False, False])


def test_bilinear_sampling_pixel_coordinates():
    image = np.array([[[0., 0, 0], [100, 100, 100]], [[100, 100, 100], [200, 200, 200]]])
    np.testing.assert_allclose(MODULE.bilinear_rgb(image, np.array([[.5, .5]])), [[100, 100, 100]])


def test_color_fusion_rejects_insufficient_and_inconsistent_views():
    samples = np.array([
        [[10., 10, 10], [10, 10, 10], [0, 0, 0]],
        [[12, 12, 12], [np.nan, np.nan, np.nan], [100, 100, 100]],
        [[11, 11, 11], [np.nan, np.nan, np.nan], [200, 200, 200]],
    ])
    rgb, count, mad, keep = MODULE.fuse_colors(samples)
    np.testing.assert_array_equal(keep, [True, False, False])
    np.testing.assert_array_equal(rgb[0], [11, 11, 11])
    assert count[0] == 3 and mad[0] == 1


def test_triangulation_does_not_bridge_large_depth_jump():
    points = np.array([[0., 0, 1], [.02, 0, 1], [0, .02, 1], [.02, .02, 5]])
    pixels = np.array([[0., 0], [10, 0], [0, 10], [10, 10]])
    faces, proposed = MODULE.supported_triangles(points, pixels)
    assert proposed == 2
    assert not (faces == 3).any()
