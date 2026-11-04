import json
import sys
import shutil
import subprocess
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pd_insighter_environment_navigation import (
    action_label_path, inject_environment_tab, parkinson_navigation, VIEW_TARGETS,
)
from build_parkinson_motion_review import continuous_segments_from_payload


def test_parkinson_tab_is_current_and_round_trip_links_exist(tmp_path):
    trial = tmp_path / "Ntrial10"
    trial.mkdir()
    page = "<html><head></head><body>" + parkinson_navigation("Ntrial10") + "</body></html>"
    result = inject_environment_tab(page, trial, active_view="parkinson")
    assert 'data-view="parkinson" aria-current="page"' in result
    assert result.count('aria-current="page"') == 1
    for key, (_, filename) in VIEW_TARGETS.items():
        if key != "parkinson":
            assert f'href="{filename}"' in result
    assert inject_environment_tab(result, trial, active_view="parkinson") == result


def test_trial_switch_preserves_view_or_labels_fallback(tmp_path):
    for name in ("Parki", "Parki2", "Ntrial1"):
        root = tmp_path / name
        root.mkdir()
        (root / "session_dashboard_manifest.json").write_text("{}")
        (root / "index.html").write_text("page")
        if name != "Ntrial1":
            (root / "pd_insighter_parkinson_review.html").write_text("page")
    page = '<html><head></head><body><div class="nav-tabs"></div></body></html>'
    result = inject_environment_tab(page, tmp_path / "Parki", active_view="parkinson")
    assert '../Parki2/pd_insighter_parkinson_review.html' in result
    assert 'Ntrial1 (Full Session)' in result
    assert '../Ntrial1/index.html' in result


def test_existing_ntrial10_store_is_shared_without_migration(tmp_path):
    root = tmp_path / "Ntrial10"
    legacy = root / "parkinson_motion_review/manual_segments.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"segments":[{"start_s":0,"end_s":1,"motion_label":"sitting"}]}')
    before = legacy.read_bytes()
    assert action_label_path(root) == legacy
    page = '<head></head><body><div class="nav-tabs"></div><script>fetch("/api/session-manual-segments")</script></body>'
    result = inject_environment_tab(page, root)
    assert '/api/parkinson-manual-segments' in result
    assert '/api/session-manual-segments' not in result
    assert legacy.read_bytes() == before
    assert action_label_path(tmp_path / "Parki").name == "session_manual_segments.json"


def test_review_header_variants_receive_shared_layout(tmp_path):
    page = '<head></head><body><nav class="page-nav"><div class="nav-tabs"><span class="nav-tab active">Manual Labels</span><a id="fullReviewLink" class="nav-tab" href="#">Full Session</a></div><label class="nav-trial-switcher"></label></nav></body>'
    result = inject_environment_tab(page, tmp_path / "Parki", active_view="manual")
    assert 'class="page-nav utility-bar"' in result
    assert 'class="nav-trial-switcher trial-switcher"' in result
    assert 'href="index.html"' in result
    assert result.count('aria-current="page"') == 1
    assert inject_environment_tab(result, tmp_path / "Parki", active_view="manual") == result


def test_classification_export_preserves_boundaries_and_gaps():
    rows = [{"base_motion": "walking", "start_s": 0, "end_s": 2.5},
            {"base_motion": "walking", "start_s": 2.5, "end_s": 5},
            {"base_motion": "standing", "start_s": 5, "end_s": 6},
            {"base_motion": "standing", "start_s": 7, "end_s": 8}]
    payload = {"smoothing": {"high": {"segments": rows}}}
    before = json.dumps(payload)
    result = continuous_segments_from_payload(payload)
    assert [(r["start_s"], r["end_s"]) for r in result] == [(0, 5), (5, 6), (7, 8)]
    assert json.dumps(payload) == before


def test_full_session_accuracy_uses_exact_labeled_duration():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to exercise the browser accuracy helper")
    source = (Path(__file__).resolve().parents[1] / "scripts/build_downward_dual_view_dashboards.py").read_text()
    helper = source.split("    function calculateContinuousManualAccuracy(", 1)[1].split("    function getCurrentContinuousReview(", 1)[0]
    script = "function calculateContinuousManualAccuracy(" + helper + """
const assert = require('node:assert/strict');
const normalizeActionLabel = label => (label || '').toLowerCase();
const actionOptions = ['sitting', 'standing', 'walking'];
const model = [{start_s: 0, end_s: 4, label: 'sitting'}, {start_s: 4, end_s: 10, label: 'walking'}];
const manual = [{start_s: 0, end_s: 2.5, motion_label: 'sitting'}, {start_s: 2.5, end_s: 10, motion_label: 'walking'}];
assert.equal(calculateContinuousManualAccuracy(manual, model, 10), .85);
assert.equal(calculateContinuousManualAccuracy([], model, 10), null);
assert.equal(calculateContinuousManualAccuracy([{start_s: 10, end_s: 12, label: 'walking'}], model, 10), null);
assert.equal(calculateContinuousManualAccuracy([{start_s: 4, end_s: 12, label: 'walking'}], model, 10), 1);
assert.equal(calculateContinuousManualAccuracy([{start_s: 0, end_s: 2, label: 'unknown'}], model, 10), null);
"""
    subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)
