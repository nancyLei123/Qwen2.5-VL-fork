"""Regression checks for the additive first-15-minute deployment, when present."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_aria_gaussian_quality import sha256
from publish_long_trial_reconstruction import covered_interval
from run_long_trial_reconstruction import check_protected
sys.path.pop(0)

ENV = ROOT / "runs/60minutes_1/environment_reconstruction"


@pytest.fixture
def deployed():
    path = ENV / "reconstruction_summary.json"
    if not path.exists() or json.loads(path.read_text())["status"] not in ("first15_extension", "first30_extension"):
        pytest.skip("First-15-minute deployment is not yet present")
    return json.loads((ENV / "scene_playback_data.json").read_text())


def test_all_first15_frames_have_a_model_but_pose_gaps_stay_explicit(deployed):
    assert covered_interval(deployed["scenes"], 0, 900) == pytest.approx(900)
    assert sum(p is None for p in deployed["poses"][:27000]) == 9
    assert deployed["route_default_end_s"] in (900, 1800)
    assert len(deployed["scenes"]) >= 7


def test_pilot_models_timing_and_labels_are_unchanged(deployed):
    before = json.loads((ENV / "before_first15_route/scene_playback_data.json").read_text())
    for key in ("poses", "recording_times", "video_url", "fps", "duration_s", "dashboard_offset_s", "labels_url"):
        assert deployed[key] == before[key]
    scenes = {s["id"]: s for s in deployed["scenes"]}
    for previous in before["scenes"]:
        assert scenes[previous["id"]] == previous
    check_protected(ENV)


def test_every_published_model_matches_its_evaluated_file(deployed):
    for scene in deployed["scenes"]:
        model = ENV / scene["ply"]
        summary = json.loads((model.parent / "training_summary.json").read_text())
        assert sha256(model) == summary["metrics"]["model_sha256"]
        assert summary["iterations"] == 7000
        assert len(summary["metrics"]["views"]) == 5


def test_deployed_route_interface_matches_source(deployed):
    for name in ("scene_playback.js", "scene_route_navigation.mjs", "scene_playback_core.mjs", "scene_camera_view.mjs", "scene_spatial_overview.mjs"):
        assert (ENV / name).read_bytes() == (ROOT / "scripts/templates" / name).read_bytes()
    template = (ROOT / "scripts/templates/pd_insighter_scene_playback.html").read_text()
    assert (ENV / "playback.html").read_text() == template.replace("__TRIAL_ID__", "60minutes_1")
