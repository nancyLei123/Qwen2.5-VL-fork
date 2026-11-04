import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("metashape_comparison", Path(__file__).resolve().parents[1] / "scripts/run_metashape_comparison.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_withheld_image_is_not_training_input():
    frames = [{"name": "a.jpg"}, {"name": "b.jpg"}, {"name": "c.jpg"}, {"name": "held.jpg"}]
    train, held = MODULE.split_frames({"frames": {"items": frames}}, "held.jpg")
    assert len(train) == 3
    assert held["name"] == "held.jpg"
    assert held not in train


def test_ambiguous_or_missing_holdout_rejected():
    for frames in [[{"name": "a"}], [{"name": "held"}, {"name": "held"}]]:
        with pytest.raises(ValueError):
            MODULE.split_frames({"frames": {"items": frames}}, "held")


def test_source_directory_cannot_be_output(tmp_path):
    dataset = tmp_path / "sources" / "room"
    for output in [dataset, dataset / "results", dataset.parent]:
        with pytest.raises(ValueError):
            MODULE.validate_output(output, [dataset])
    assert MODULE.validate_output(tmp_path / "comparison", [dataset]) == tmp_path / "comparison"


def test_transform_validation_rejects_reflection():
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    MODULE.check_transform(identity)
    identity[0][0] = -1
    with pytest.raises(ValueError, match="right handed"):
        MODULE.check_transform(identity)


def test_protected_sources_detect_changes(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("original")
    plan = {"protected_sha256": {str(source): MODULE.digest(source)}}
    MODULE.verify_sources(plan)
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="Source changed"):
        MODULE.verify_sources(plan)


def test_bounds_ignore_colmap_comments(tmp_path):
    source = tmp_path / "points.txt"
    source.write_text("# header\n1 -2 1 0 0 0 0 0\n2 3 4 2 0 0 0 0\n")
    bounds = MODULE.point_bounds(source)
    assert bounds["min"] == [-2, 1, 0]
    assert bounds["max"] == [3, 4, 2]


def test_calibration_uses_principal_point_relative_to_image_center():
    class Calibration:
        pass

    class Sensor:
        class Type:
            Frame = "frame"

    class MS:
        pass

    MS.Calibration, MS.Sensor = Calibration, Sensor
    c = MODULE.make_calibration(MS, {"width": 1024, "height": 1024, "fx": 512, "fy": 512, "cx": 511.5, "cy": 511.5})
    assert c.cx == c.cy == -0.5
    assert c.f == 512
    assert c.b1 == 0
