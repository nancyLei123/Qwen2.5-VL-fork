import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from attach_downward_to_existing_trial import coverage_plan
from pd_insighter_environment_navigation import adapt_partial_downward


def test_shorter_video_preserves_full_aria_duration():
    plan = coverage_plan(9.504995596, 820.6, 1152.8)
    assert plan["start_s"] == 9.5
    assert plan["end_s"] == pytest.approx(830.1)
    assert plan["session_duration_s"] == 1152.8
    assert plan["padding_after_s"] == pytest.approx(322.7)
    assert abs(plan["quantization_error_s"]) <= 1 / 60


def test_negative_offset_trims_only_downward_prefix():
    plan = coverage_plan(-5.25, 120, 100)
    assert plan["source_trim_s"] == 5.25
    assert plan["start_s"] == 0
    assert plan["end_s"] == 100
    assert plan["padding_after_s"] == 0


def test_future_stride_refresh_uses_unchanged_aria_clock(tmp_path, monkeypatch):
    import explore_slam_parameters as module
    monkeypatch.setattr(module, "RUNS_DIR", tmp_path)
    trial = tmp_path / "Fuchs"
    (trial / "downward_sync").mkdir(parents=True)
    (trial / "session_dashboard_manifest.json").write_text(json.dumps({
        "session_duration_s": 1152.8, "downward_attachment": {"start_s": 9.5}}))
    (trial / "downward_sync/sync_suggestion.json").write_text(json.dumps({
        "aria_trim_start_s": 9.5, "shared_duration_s": 820.6}))
    assert module.load_sync_window("Fuchs") == (0, 1152.8)


@pytest.mark.parametrize("values", [(200, 10, 100), (-200, 10, 100), (float('nan'), 10, 100), (0, 0, 100)])
def test_invalid_or_nonoverlapping_media_is_rejected(values):
    with pytest.raises(ValueError):
        coverage_plan(*values)


def test_gap_overlay_and_rotation_are_scoped_to_attached_trials(tmp_path):
    page = '''<head></head><body><nav>Trial tabs</nav><video id="downwardVideo"></video>
    let downwardTurns = 2;
    let downwardRotationQuarterTurns = 2;
    downwardRotationQuarterTurns = Number(window.localStorage.getItem(`pd-insighter-downward-rotation-v2:${trialId}`) || 2) || 2;
    </body>'''
    assert adapt_partial_downward(page, tmp_path) == page
    (tmp_path / "trial_manifest.json").write_text(json.dumps({"downward_attachment": {
        "start_s": 10, "end_s": 70, "session_duration_s": 100, "rotation_quarter_turns": 0}}))
    result = adapt_partial_downward(page, tmp_path)
    assert 'let downwardTurns = 0;' in result
    assert 'let downwardRotationQuarterTurns = 0;' in result
    assert '?? 0) ?? 0' in result
    assert 'pd-insighter-downward-rotation-v3:' in result
    assert 'No downward recording at this time' in result
    assert 'Striped sections have no downward recording' in result
    assert 't >= coverage.start && t < coverage.end' in result
    assert 'left:10.000000%;width:60.000000%' in result
