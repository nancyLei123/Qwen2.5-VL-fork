import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pd_insighter_environment_navigation.py"
SPEC = importlib.util.spec_from_file_location("environment_navigation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PAGE = '<div class="nav-tabs" aria-label="Trial views"><a href="index.html">Full Session</a></div>'


def test_available_fifth_tab_is_relative_and_idempotent(tmp_path):
    target = tmp_path / "environment_reconstruction"
    target.mkdir()
    (target / "index.html").write_text("page")
    (target / "reconstruction_summary.json").write_text("{}")
    result = MODULE.inject_environment_tab(PAGE, tmp_path)
    assert 'href="environment_reconstruction/index.html"' in result
    assert result.count('data-view="environment"') == 1
    assert MODULE.inject_environment_tab(result, tmp_path) == result


def test_unbuilt_trial_is_disabled_not_broken(tmp_path):
    result = MODULE.inject_environment_tab(PAGE, tmp_path)
    assert 'aria-disabled="true"' in result
    assert 'href="environment_reconstruction/index.html"' not in result


def test_refresh_changes_navigation_not_manual_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "RUNS_ROOT", tmp_path)
    trial = tmp_path / "Ntrial3"
    trial.mkdir()
    (trial / "index.html").write_text(PAGE)
    labels = trial / "manual_labels"
    labels.mkdir()
    saved = labels / "session_manual_segments.json"
    saved.write_text('{"segments":[{"start_s":1,"end_s":2}]}')
    original = saved.read_bytes()
    assert MODULE.refresh_navigation(["Ntrial3"]) == [trial / "index.html"]
    assert saved.read_bytes() == original
