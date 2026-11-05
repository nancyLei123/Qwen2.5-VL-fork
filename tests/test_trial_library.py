import importlib.util
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("trial_library", ROOT / "scripts/build_pd_insighter_trial_selector.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_trial(root, name, *, kind=None, full=False, legacy=False, environment=False, manifest=True):
    trial = root / name
    trial.mkdir(parents=True)
    if manifest:
        (trial / "trial_manifest.json").write_text(json.dumps({"trial_id": name, "recording_stem": name, "trial_type": kind}))
    if full:
        (trial / "index.html").write_text('<html><h1>Session</h1></html>')
        (trial / "session_dashboard_manifest.json").write_text('{"session_duration_s":123}')
        (trial / "pd_insighter_stride_review.html").write_text('<html>Stride</html>')
    if legacy:
        (trial / MODULE.LEGACY_DASHBOARD_NAME).write_text('<html>Legacy review</html>')
    if environment:
        env = trial / "environment_reconstruction"
        env.mkdir()
        (env / "playback.html").write_text('<html>Environment</html>')
        (env / "scene_playback_data.json").write_text(json.dumps({"trial_id": name, "duration_s":321, "provenance":{"vrs":"/recordings/room-scan.vrs"}}))
    return trial


def test_discovers_environment_only_legacy_and_unbuilt_records(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "RUNS_DIR", tmp_path)
    make_trial(tmp_path, "Ntrial3", full=True, environment=True)
    room = make_trial(tmp_path, "RoomScan3c", manifest=False, environment=True)
    (room / "index.html").write_text('<meta http-equiv="refresh" content="0;url=environment_reconstruction/index.html">')
    make_trial(tmp_path, "trial1", legacy=True)
    make_trial(tmp_path, "trial4")
    make_trial(tmp_path, "Bus1")
    records = {t["trial_id"]:t for t in MODULE.collect_trials()}
    assert set(records) == {"Ntrial3", "RoomScan3c", "trial1", "trial4", "Bus1"}
    assert records["RoomScan3c"]["category"] == "reconstruction"
    assert records["RoomScan3c"]["primary_href"].startswith("RoomScan3c/environment_reconstruction/playback.html")
    assert not records["RoomScan3c"]["full_dashboard_href"]
    assert records["RoomScan3c"]["duration_s"] == 321
    assert records["Ntrial3"]["category"] == "normal"
    assert records["Ntrial3"]["duration_s"] == 123
    assert records["trial1"]["legacy_dashboard_href"]
    assert records["trial4"]["status"] == "registered"
    assert not records["Bus1"]["primary_href"]


def test_purpose_not_inferred_from_models_or_diagnostic_review(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "RUNS_DIR", tmp_path)
    make_trial(tmp_path, "NewSimulation", kind="parkinson_simulated", full=True, environment=True)
    make_trial(tmp_path, "NewNormal", kind="normal", full=True, environment=True)
    make_trial(tmp_path, "NewScan", kind="environment_scan", environment=True)
    unknown = make_trial(tmp_path, "Unknown", full=True, environment=True)
    (unknown / "pd_insighter_parkinson_review.html").write_text('<html>Review</html>')
    categories = {t["trial_id"]:t["category"] for t in MODULE.collect_trials()}
    assert categories == {"NewSimulation":"parkinson", "NewNormal":"normal", "NewScan":"reconstruction", "Unknown":"archive"}


def test_nested_backups_and_hidden_records_are_not_new_trials(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "RUNS_DIR", tmp_path)
    make_trial(tmp_path, "Ntrial1", full=True)
    make_trial(tmp_path, "_backup/Ntrial1", full=True)
    make_trial(tmp_path, "_temporary", full=True)
    assert [t["trial_id"] for t in MODULE.collect_trials()] == ["Ntrial1"]


def test_cards_have_no_invented_links_and_escape_recording_names(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "RUNS_DIR", tmp_path)
    trial = make_trial(tmp_path, "Parki", kind="parkinson_simulated", full=True)
    manifest = trial / "trial_manifest.json"
    manifest.write_text(json.dumps({"recording_stem":'<script>alert("x")</script> __SCRIPT__'}))
    records = MODULE.collect_trials()
    card = MODULE.render_trial(records[0])
    assert "&lt;script&gt;" in card and '<script>alert' not in card
    assert "Manual Labels" not in card and "Environment 3D" not in card
    assert "Stride" in card
    page = MODULE.build_html(records)
    assert "__SCRIPT__" in page  # Data is not reprocessed as template syntax.
    assert '<script>alert' not in page
    for key, href in records[0].items():
        if key.endswith("_href") and href:
            assert (tmp_path / unquote(urlsplit(href).path)).is_file()


def test_natural_sort_and_unclassified_default():
    assert sorted(["Ntrial10", "Ntrial2", "Ntrial1"], key=MODULE.trial_sort_key) == ["Ntrial1", "Ntrial2", "Ntrial10"]
    assert MODULE.category_for("Ntrial10", {}, {}) == "parkinson"
    assert MODULE.category_for("Unexpected", {}, {}) == "archive"


def test_publish_preserves_research_and_legacy_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "RUNS_DIR", tmp_path)
    for key, name in (("FULL_SELECTOR_HTML","pd_insighter_trial_selector.html"),
                      ("CLIP_SELECTOR_HTML","pd_insighter_trial_selector_clip.html"),
                      ("LEGACY_SELECTOR_HTML","pd_insighter_trial_selector_legacy.html")):
        monkeypatch.setattr(MODULE, key, tmp_path / name)
    make_trial(tmp_path, "Ntrial3", full=True, environment=True)
    make_trial(tmp_path, "trial4")
    before = {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    MODULE.main()
    assert all(p.read_bytes() == value for p, value in before.items())
    assert "?group=archive" in MODULE.LEGACY_SELECTOR_HTML.read_text()
    assert 'data-trial-id="trial4"' in MODULE.FULL_SELECTOR_HTML.read_text()
    assert "Dashboard not yet available" in MODULE.FULL_SELECTOR_HTML.read_text()
    assert 'name="viewport"' in MODULE.FULL_SELECTOR_HTML.read_text()


def test_local_inventory_is_complete_and_all_links_exist():
    if not (MODULE.RUNS_DIR / "Ntrial3/trial_manifest.json").exists():
        pytest.skip("Local research data is not included in a code-only checkout.")
    records = MODULE.collect_trials()
    ids = {t["trial_id"] for t in records}
    expected = {p.parent.name for p in MODULE.RUNS_DIR.glob("*/trial_manifest.json") if not p.parent.name.startswith(("_", "."))}
    assert expected <= ids
    assert "RoomScan3c" in ids
    assert {t["trial_id"] for t in records if t["category"] == "parkinson"} >= {"Ntrial10", "Parki", "Parki2", "Fuchs", "Fuchs22", "Fuchs333"}
    for item in records:
        for key, href in item.items():
            if key.endswith("_href") and href:
                path = MODULE.RUNS_DIR / unquote(urlsplit(href).path)
                assert path.is_file(), (item['trial_id'], key, href)
