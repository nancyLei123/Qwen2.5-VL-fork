import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def report(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("comparison_report", scripts / "build_metashape_comparison.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_mesh_pixels_count_in_full_frame_scores(report):
    reference = np.full((2, 2, 3), 128, dtype=np.uint8)
    gaussian = reference.copy()
    mesh = np.concatenate((reference, np.full((2, 2, 1), 255, dtype=np.uint8)), axis=2)
    mesh[0, 0, 3] = 0
    metrics = report.image_metrics(reference, gaussian, mesh)
    assert metrics["opaque_mesh_coverage_pct"] == 75
    assert metrics["missing_or_edge_pct"] == 25
    assert metrics["metashape_full_black_psnr_db"] < 20
    assert metrics["metashape_common_mask_psnr_db"] == 100


def test_exact_same_mask_is_used_for_both_methods(report):
    reference = np.full((2, 2, 3), 128, dtype=np.uint8)
    gaussian = reference.copy()
    gaussian[0, 0] = 0
    mesh = np.concatenate((reference, np.full((2, 2, 1), 255, dtype=np.uint8)), axis=2)
    mesh[0, 0, 3] = 0
    metrics = report.image_metrics(reference, gaussian, mesh)
    assert metrics["gaussian_common_mask_psnr_db"] == metrics["metashape_common_mask_psnr_db"] == 100
    assert metrics["gaussian_full_psnr_db"] < 20


def test_empty_coverage_is_not_reported_as_perfect(report):
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    mesh = np.zeros((2, 2, 4), dtype=np.uint8)
    metrics = report.image_metrics(image, image, mesh)
    assert metrics["opaque_mesh_coverage_pct"] == 0
    assert metrics["metashape_common_mask_psnr_db"] is None
    assert metrics["gaussian_common_mask_psnr_db"] is None


def test_dimensions_must_match_without_resampling(report):
    with pytest.raises(ValueError):
        report.psnr(np.zeros((2, 2, 3)), np.zeros((3, 3, 3)))


def test_incomplete_reconstruction_cannot_be_published(report, tmp_path):
    plan = {"protected_sha256": {}, "method": "test", "limitations": [],
            "scenes": {"room": {}}}
    (tmp_path / "comparison_plan.json").write_text(json.dumps(plan))
    (tmp_path / "room").mkdir()
    (tmp_path / "room/metashape_status.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(RuntimeError, match="has not finished"):
        report.build(tmp_path)
    assert not (tmp_path / "index.html").exists()
