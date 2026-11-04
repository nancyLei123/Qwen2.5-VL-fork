import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_aria_gaussian_quality import split_frames, metrics
from publish_aria_quality_choices import eligible, parse_reviews


def test_fixed_validation_and_temporal_guard_are_disjoint():
    frames = [{"name": f"f{i}.jpg", "recording_s": i / 6} for i in range(90)]
    train, validation, guard = split_frames(frames)
    assert len(validation) == 5
    assert len(train) + len(validation) + len(guard) == len(frames)
    assert len({f["name"] for f in train + validation + guard}) == len(frames)
    for frame in train:
        assert min(abs(frame["recording_s"] - v["recording_s"]) for v in validation) > 0.25
    assert split_frames(frames) == (train, validation, guard)


def test_small_dataset_is_not_silently_accepted():
    with pytest.raises(ValueError):
        split_frames([{"name": "a", "recording_s": 0}])


def test_metrics_no_resizing_or_masked_out_errors():
    a = Image.fromarray(np.full((16, 16, 3), 100, dtype=np.uint8))
    b = Image.fromarray(np.full((16, 16, 3), 90, dtype=np.uint8))
    score = metrics(a, b)
    assert score["psnr_db"] == pytest.approx(20 * np.log10(255 / 10), abs=1e-4)
    assert score["mae"] == pytest.approx(10 / 255)
    assert metrics(a, a)["ssim"] == pytest.approx(1)
    with pytest.raises(ValueError):
        metrics(a, b.resize((8, 8)))


def test_promotion_gate_rejects_cherry_picked_or_regressing_views():
    baseline = {"views": [{"name": str(i), "recording_s":i, "psnr_db":25, "ssim":0.8} for i in range(5)],
                "mean_psnr_db":25, "mean_ssim":0.8}
    candidate = {"views": [{"name": str(i), "recording_s":i, "psnr_db":26, "ssim":0.82} for i in range(5)],
                 "mean_psnr_db":26, "mean_ssim":0.82}
    assert eligible(baseline, candidate)
    candidate["views"][0]["psnr_db"] = 23
    assert not eligible(baseline, candidate)
    candidate["views"] = candidate["views"][1:]
    assert not eligible(baseline, candidate)


def test_promotion_requires_matching_times_and_consistent_finite_metrics():
    baseline = {"views": [{"name": str(i), "recording_s":i, "psnr_db":25, "ssim":0.8} for i in range(5)],
                "mean_psnr_db":25, "mean_ssim":0.8}
    candidate = {"views": [{"name": str(i), "recording_s":i, "psnr_db":26, "ssim":0.82} for i in range(5)],
                 "mean_psnr_db":26, "mean_ssim":0.82}
    assert eligible(baseline, candidate, baseline["views"])
    candidate["views"][0]["recording_s"] = 999
    assert not eligible(baseline, candidate)
    candidate["views"][0]["recording_s"] = 0
    candidate["mean_ssim"] = 0.99
    assert not eligible(baseline, candidate)
    candidate["mean_ssim"] = 0.82
    candidate["views"][0]["psnr_db"] = float("nan")
    assert not eligible(baseline, candidate)


def test_review_approval_is_bound_to_one_exact_hash_per_scene():
    digest = "a" * 64
    assert parse_reviews([f"stairs={digest}"]) == {"stairs":digest}
    for value in (["stairs"], ["stairs=short"], [f"stairs={digest}"] * 2):
        with pytest.raises(ValueError):
            parse_reviews(value)
