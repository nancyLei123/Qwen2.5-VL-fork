import importlib.util
import json
from pathlib import Path
import re


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


def make_trial(root, name, *, environment=True, analysis=True):
    trial = root / name
    trial.mkdir()
    if analysis:
        (trial / "session_dashboard_manifest.json").write_text("{}")
        for _, filename in MODULE.VIEW_TARGETS.values():
            (trial / filename).write_text('<html><head></head><body>' + PAGE + '</body></html>')
    if environment:
        env = trial / "environment_reconstruction"
        env.mkdir()
        (env / "scene_playback_data.json").write_text('{"poses":[null]}')
        (env / "reconstruction_summary.json").write_text("{}")
        (env / "playback.html").write_text('<html><head></head><body><main><nav class="card nav">Old standalone navigation</nav><header>Scene</header><video id="ariaVideo"></video><script type="module" src="scene_playback.js"></script></main></body></html>')
        (env / "index.html").write_text('<meta http-equiv="refresh" content="0;url=playback.html">')
        (env / "gaussian_submaps").mkdir()
        (env / "gaussian_submaps/index.html").write_text("Submaps")
    return trial


def switcher_items(document):
    return json.loads(re.search(r'(\[\{"label":.*?\])\.forEach', document)[1])


def test_environment_uses_shared_navigation_and_nested_paths(tmp_path):
    trial = make_trial(tmp_path, "Fuchs")
    path = trial / "environment_reconstruction/playback.html"
    result = MODULE.inject_environment_navigation(path.read_text(), trial, path)
    nav = re.search(r'<nav\b.*?</nav>', result, re.S)[0]
    assert 'class="card utility-bar"' in nav
    assert 'class="nav-location"><strong>Fuchs</strong><span>Environment</span>' in nav
    assert 'href="../index.html"' in nav
    assert 'href="../pd_insighter_parkinson_review.html"' in nav
    assert nav.count('aria-current="page"') == 1
    assert 'data-view="environment" aria-current="page">Environment' in nav
    assert 'id="trialJumpSelect"' in nav
    assert "Submaps" not in nav
    assert 'href="gaussian_submaps/index.html">Submaps' in result
    assert 'href="index.html">Methods' not in result  # This index is only a redirect.
    assert '<video id="ariaVideo"></video>' in result
    assert MODULE.inject_environment_navigation(result, trial, path) == result


def test_playback_is_direct_destination_and_trial_switch_preserves_view(tmp_path):
    current = make_trial(tmp_path, "Fuchs")
    make_trial(tmp_path, "Ntrial3")
    make_trial(tmp_path, "Parki", environment=False)
    make_trial(tmp_path, "RoomScan3c", analysis=False)
    result = MODULE.inject_environment_tab(PAGE, current)
    assert 'href="environment_reconstruction/playback.html"' in result
    path = current / "environment_reconstruction/playback.html"
    result = MODULE.inject_environment_navigation(path.read_text(), current, path)
    items = {item["label"]: item for item in switcher_items(result)}
    assert items["All Trials"]["value"] == "../../pd_insighter_trial_selector.html"
    assert items["Fuchs"]["selected"]
    assert items["Ntrial3"]["value"] == "../../Ntrial3/environment_reconstruction/playback.html"
    assert items["Parki (Full Session)"]["value"] == "../../Parki/index.html"
    assert items["RoomScan3c"]["value"] == "../../RoomScan3c/environment_reconstruction/playback.html"


def test_room_scan_has_no_dead_analysis_links(tmp_path):
    trial = make_trial(tmp_path, "RoomScan3c", analysis=False)
    (trial / "index.html").write_text('<meta http-equiv="refresh" content="0;url=environment_reconstruction/playback.html">')
    path = trial / "environment_reconstruction/playback.html"
    result = MODULE.inject_environment_navigation(path.read_text(), trial, path)
    nav = re.search(r'<nav\b.*?</nav>', result, re.S)[0]
    assert nav.count('aria-disabled="true"') == 4
    assert 'href="../index.html"' not in nav
    assert "Parkinson Review" not in nav
    assert nav.count('aria-current="page"') == 1


def test_refresh_updates_environment_without_touching_data_or_redirects(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "RUNS_ROOT", tmp_path)
    trial = make_trial(tmp_path, "Ntrial3")
    env = trial / "environment_reconstruction"
    protected = {p: p.read_bytes() for p in (env / "scene_playback_data.json", env / "index.html")}
    changed = MODULE.refresh_navigation(["Ntrial3"])
    assert env / "playback.html" in changed
    assert all(p.read_bytes() == original for p, original in protected.items())
    assert MODULE.refresh_navigation(["Ntrial3"]) == []


def test_template_marker_and_methods_page_have_same_active_tab(tmp_path):
    trial = make_trial(tmp_path, "Ntrial3")
    env = trial / "environment_reconstruction"
    methods = '<html><head></head><body><nav>Old</nav><header>Methods</header></body></html>'
    (env / "index.html").write_text(methods)
    template = '<html><head></head><body><!-- PD_INSIGHTER_NAVIGATION --><header>Playback</header></body></html>'
    result = MODULE.inject_environment_navigation(template, trial, env / "playback.html")
    assert '<!-- PD_INSIGHTER_NAVIGATION -->' not in result
    assert 'href="index.html">Methods &amp; results' in result
    methods = MODULE.inject_environment_navigation(methods, trial, env / "index.html")
    assert 'href="playback.html">Synchronized playback' in methods
    assert 'data-view="environment" aria-current="page"' in methods
    assert MODULE.inject_environment_navigation(methods, trial, env / "index.html") == methods
