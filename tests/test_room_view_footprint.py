import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_room_view_footprint import camera_rays, surface_patches, rotated_video_uv


def test_viewer_rays_match_rectified_intrinsics():
    camera = dict(width=1408, height=1408, fx=704, fy=704, cx=703.5, cy=703.5)
    rays, count = camera_rays(camera)
    assert count == 81 and rays.shape == (113, 3)
    np.testing.assert_allclose(rays[40], [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1)
    np.testing.assert_allclose(rays[81] / -rays[81, 2], [-1, 1, -1])
    np.testing.assert_allclose(rays[89] / -rays[89, 2], [1, 1, -1])


def test_finite_patches_fit_planes_but_not_missing_surfaces():
    xy = np.array([(x, y, -2.) for x in np.arange(-.25, .26, .02) for y in np.arange(-.25, .26, .02)])
    centers = np.array([[0., 0., -2.], [.1, .1, -2.], [3., 3., -2.]])
    patches, audit = surface_patches(centers, xy)
    assert len(patches) == 2
    np.testing.assert_allclose(np.abs(patches[:, 5]), 1)
    assert np.all(patches[:, 6] <= .055)
    assert audit['plane_rms_median_m'] < 1e-10


def test_video_projection_undoes_both_camera_basis_changes():
    class Pinhole:
        def project(self, p):
            return np.array([p[0]/p[2]*704+703.5, p[1]/p[2]*704+703.5])
        def get_image_size(self):
            return 1408, 1408
    rays, _ = camera_rays(dict(width=1408, height=1408, fx=704, fy=704, cx=703.5, cy=703.5))
    uv = rotated_video_uv(Pinhole(), rays)
    np.testing.assert_allclose(uv[40], [.5, .5])
    np.testing.assert_allclose(uv[81], [0, 0])
    np.testing.assert_allclose(uv[89], [1, 0])
    np.testing.assert_allclose(uv[97], [1, 1])


def test_export_keeps_unavailable_frames_unknown_and_is_honest_about_geometry():
    env = Path(__file__).resolve().parents[1] / 'runs/RoomScan3c/environment_reconstruction'
    if not (env/'view_footprint/footprint_data.json').is_file():
        pytest.skip('Private RoomScan3c artifacts absent')
    data = json.loads((env/'scene_playback_data.json').read_text())
    payload = json.loads((env/'view_footprint/footprint_data.json').read_text())
    assert len(payload['frame_lut']) == len(data['poses'])
    assert all((p is None) == (i == -1) for p, i in zip(data['poses'], payload['frame_lut']))
    audit = payload['audit']
    assert audit['max_pose_matrix_difference'] < 1e-6
    assert audit['video_lut_max_error_pixels'] <= .21
    assert 'true occlusion or visibility' in audit['not_validated']
    assert 'physical surface accuracy' in audit['not_validated']
