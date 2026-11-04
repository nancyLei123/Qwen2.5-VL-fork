import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_room_floater_cleanup import candidate_mask, subset_ply
from prepare_gaussian_viewer_compat import parse_ply


def test_cleanup_requires_both_route_proximity_and_no_close_geometry_support():
    result = candidate_mask(np.array([.01, .01, .5, np.nan, .12]), np.array([.1, .01, .1, .1, .1]))
    assert result.tolist() == [True, False, False, False, False]


def test_subset_preserves_all_remaining_vertex_bytes():
    header = b'ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nend_header\n'
    rows = np.array([[1, 2], [3, 4], [5, 6]], dtype='<f4')
    raw = header + rows.tobytes()
    result = parse_ply(subset_ply(raw, np.array([True, False, True])))
    assert result['vertex_count'] == 2
    assert bytes(result['body']) == rows[[0, 2]].tobytes()
    assert raw == header + rows.tobytes()
    with pytest.raises(ValueError):
        subset_ply(raw, np.zeros(3, dtype=bool))
