import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reconstruct_dedicated_room import complete_ply, image_quality, nearest_index, select_views, split_views
from export_aria_gaussian_dataset import interpolate_pose, pinhole_project, rotate_upright


def frame(i):
    m = np.eye(4)
    m[0, 3] = i / 100
    return {"name": f"{i}.jpg", "recording_s": float(i), "T_world_camera": m.tolist(),
            "laplacian_variance": 80 + i % 12, "dark_fraction": .05, "bright_fraction": .01}


class RoomTests(unittest.TestCase):
    def test_nearest_edges(self):
        t = np.array([10, 20, 30])
        self.assertEqual(nearest_index(t, -2), 0)
        self.assertEqual(nearest_index(t, 100), 2)
        self.assertEqual(nearest_index(t, 25), 1)

    def test_selection_quality_and_full_time(self):
        frames = [frame(i) for i in range(300)]
        frames[12]["laplacian_variance"] = 1
        selected = select_views(frames, 60)
        self.assertEqual(len(selected), 60)
        self.assertNotIn("12.jpg", [r["name"] for r in selected])
        self.assertEqual(len({int(r["recording_s"] // 30) for r in selected}), 10)
        self.assertTrue(all(a["recording_s"] < b["recording_s"] for a, b in zip(selected, selected[1:])))

    def test_no_validation_leak(self):
        rows = [frame(i) for i in range(300)]
        train, val, guard = split_views(rows)
        self.assertEqual(len(val), 12)
        self.assertEqual(len(train) + len(val) + len(guard), len(rows))
        self.assertTrue(all(abs(a["recording_s"] - b["recording_s"]) > 1 for a in train for b in val))

    def test_quality_is_finite(self):
        q = image_quality(np.zeros((32, 32, 3), np.uint8))
        self.assertEqual(q["dark_fraction"], 1)
        self.assertEqual(q["laplacian_variance"], 0)

    def test_ply_is_not_published_mid_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ply"
            header = b"ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
            self.assertFalse(complete_ply(path))
            path.write_bytes(header + bytes(20))
            self.assertFalse(complete_ply(path))
            path.write_bytes(header + bytes(24))
            self.assertTrue(complete_ply(path))

    def test_upright_pixel_and_pose_rotation_agree(self):
        size = 9
        image = np.arange(size * size).reshape(size, size)
        upright, transform = rotate_upright(image, np.eye(4))
        points = np.array([[.25, -.5, 2.], [-.5, .25, 2.], [0., 0., 2.]])
        intrinsics = (4., 4., 4., 4.)
        before, _ = pinhole_project(points, np.eye(4), intrinsics)
        after, _ = pinhole_project(points, transform, intrinsics)
        np.testing.assert_allclose(after, np.column_stack([size - 1 - before[:, 1], before[:, 0]]))
        self.assertEqual(upright[2, 7], image[1, 2])

    def test_pose_interpolates_but_never_extrapolates(self):
        t = np.array([0, 10_000_000], dtype=np.int64)
        p = np.array([[0., 0., 0.], [2., 0., 0.]])
        q = np.array([[0., 0., 0., 1.], [0., 0., 0., 1.]])
        pose, gap = interpolate_pose(t, p, q, 5_000_000)
        np.testing.assert_allclose(pose[:3, 3], [1., 0., 0.])
        self.assertEqual(gap, 10_000_000)
        with self.assertRaises(ValueError):
            interpolate_pose(t, p, q, -1)
        with self.assertRaises(ValueError):
            interpolate_pose(t * 10, p, q, 50_000_000)


if __name__ == "__main__":
    unittest.main()
