import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from publish_joint_long_comparison import validated_comparison, ENV, OUT
from compare_aria_gaussian_quality import sha256
from publish_long_trial_reconstruction import covered_interval
sys.path.pop(0)


def fixture():
    views = [{"name": f"view{i}", "reference": f"image{i}", "recording_s": i, "psnr_db": 20, "ssim": .8, "mae": .1} for i in range(30)]
    result = {"metrics": {"views": views}}
    return result, copy.deepcopy(result), {"validation": views}


def test_long_comparison_uses_identical_references_and_times():
    old, new, split = fixture()
    assert len(validated_comparison(old, new, split)) == 30


@pytest.mark.parametrize("failure", ["name", "reference", "time", "metric", "count"])
def test_mismatched_comparison_fails_closed(failure):
    old, new, split = fixture()
    view = new["metrics"]["views"][0]
    if failure == "name": view["name"] = "new"
    if failure == "reference": view["reference"] = "different"
    if failure == "time": view["recording_s"] = 999
    if failure == "metric": view["psnr_db"] = float("nan")
    if failure == "count": new["metrics"]["views"].pop()
    with pytest.raises(ValueError): validated_comparison(old, new, split)


def test_first30_deployment_preserves_all_previous_models_and_clocks():
    path = ENV / "reconstruction_summary.json"
    if not path.exists():
        pytest.skip("Local research artifacts are not available")
    summary = json.loads(path.read_text())
    if summary["status"] != "first30_extension":
        pytest.skip("First30 training has not been deployed yet")
    before = json.loads((ENV / "before_first30_extension/scene_playback_data.json").read_text())
    after = json.loads((ENV / "scene_playback_data.json").read_text())
    for key in ("poses", "recording_times", "video_url", "labels_url", "fps", "duration_s", "dashboard_offset_s"):
        assert before[key] == after[key]
    scenes = {s["id"]: s for s in after["scenes"]}
    assert len(scenes) == 15
    for old in before["scenes"]:
        assert scenes[old["id"]] == old
    measured = covered_interval(after["scenes"], 0, 1800)
    assert 1790 <= measured <= 1800
    assert after["coverage_summary"]["model_seconds"] == pytest.approx(measured)
    assert summary["first30_model_coverage_s"] == pytest.approx(measured)
    assert sum(p is None for p in after["poses"][:54000]) == 18
    reviews = json.loads((ENV / "first30_visual_review.json").read_text())["scenes"]
    for scene in after["scenes"]:
        if scene["id"].startswith("first30_"):
            assert reviews[scene["id"]]["reviewed_views"]
            assert reviews[scene["id"]]["finding"]


def test_optional_longer_model_has_a_review_and_preserves_the_7k_file():
    path = OUT / "viewer_30000.json"
    if not path.exists():
        pytest.skip("Long experiment has not been visually reviewed yet")
    original = json.loads((OUT / "viewer.json").read_text())
    longer = json.loads(path.read_text())
    assert original["scene"]["iterations"] == 7000
    assert longer["scene"]["iterations"] == 30000
    assert longer["scene"]["video_start_s"] == 0 and longer["scene"]["video_end_s"] == 900
    for model in (original, longer):
        assert sha256(ENV / model["scene"]["ply"]) == model["model_sha256"]
        assert model["status"] == "reviewed_experimental"
        assert model["visual_review"]["reviewed_views"]
    old = json.loads((OUT / "comparison.json").read_text())
    new = json.loads((OUT / "comparison_30000.json").read_text())
    split = json.loads((OUT / "split.json").read_text())
    validated_comparison(old, new, split)
