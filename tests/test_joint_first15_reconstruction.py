import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from joint_first15_reconstruction import validate_splits, run_paths, OUT, ENV
from compare_aria_gaussian_quality import sha256
sys.path.pop(0)


def split(train, validation, graph="metric-a", camera="camera-a"):
    return {"world_frame": graph, "camera": camera,
            "training": [{"recording_s": t} for t in train],
            "validation": [{"recording_s": t} for t in validation]}


def test_joint_scene_requires_a_shared_metric_frame_and_camera():
    validate_splits([split([1], [2]), split([3], [4])])
    with pytest.raises(ValueError, match="metric world"):
        validate_splits([split([1], [2]), split([3], [4], graph="other")])
    with pytest.raises(ValueError, match="camera model"):
        validate_splits([split([1], [2]), split([3], [4], camera="other")])


def test_joint_split_checks_leakage_across_submap_boundaries():
    with pytest.raises(ValueError, match="leakage"):
        validate_splits([split([1], [2]), split([2.1], [4])])


@pytest.fixture
def deployed_joint():
    if not (OUT / "viewer.json").exists():
        pytest.skip("Optional local joint experiment not published")
    return json.loads((OUT / "viewer.json").read_text())


def test_joint_model_is_additive_and_stays_inside_first15(deployed_joint):
    player = json.loads((ENV / "scene_playback_data.json").read_text())
    assert len(player["scenes"]) >= 7
    assert not any(s["id"] == deployed_joint["scene"]["id"] for s in player["scenes"])
    assert deployed_joint["graph_uid"] == player["provenance"]["graph_uid"]
    assert deployed_joint["scene"]["video_start_s"] == 0
    assert deployed_joint["scene"]["video_end_s"] == 900
    assert deployed_joint["default"] == "submaps"
    assert sha256(ENV / deployed_joint["scene"]["ply"]) == deployed_joint["model_sha256"]


def test_joint_evaluation_keeps_all_thirty_original_held_out_views(deployed_joint):
    joint = json.loads((OUT / "split.json").read_text())
    comparison = json.loads((OUT / "comparison.json").read_text())
    assert len(joint["validation"]) == len(comparison["metrics"]["views"]) == 30
    assert {v["name"] for v in joint["validation"]} == {v["name"] for v in joint["baseline"]}
    by_submap = {v["submap"] for v in joint["training"]}
    assert len(by_submap) == 6
    originals = [json.loads((ENV / "gaussian_submaps" / name / "validation_run/split.json").read_text()) for name in by_submap]
    validate_splits(originals)
    for old in originals:
        prefix = Path(old["source"]).name + "_"
        for frame in old["training"]:
            new = next(f for f in joint["training"] if f["name"] == prefix + frame["name"])
            assert new["T_world_camera"] == frame["T_world_camera"]
            assert new["capture_timestamp_ns"] == frame["capture_timestamp_ns"]


def test_longer_recipe_cannot_overwrite_baseline_outputs():
    old = run_paths(7000)
    new = run_paths(30000)
    assert set(old).isdisjoint(new)
    assert old == (OUT / "trained_7000", OUT / "comparison.json", OUT / "progress.json")
    with pytest.raises(ValueError):
        run_paths(15000)
