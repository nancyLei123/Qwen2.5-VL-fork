"""Symptom annotation storage, API, navigation and page generation (synthetic data only)."""

import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STORE = load("symptom_annotations", "symptom_annotations.py")
NAV = load("nav_for_symptoms", "pd_insighter_environment_navigation.py")
SYMNAV = load("symptom_annotation_navigation", "symptom_annotation_navigation.py")
DURATION = 120.0


def labels_doc(runs):
    return STORE.load_labels(runs)


def save(runs, annotations, revision=0, trial="TrialA", duration=DURATION):
    return STORE.save_annotations(runs, {"trial_id": trial, "base_revision": revision, "annotations": annotations},
                                  duration_s=duration, labels=labels_doc(runs))


def ann(id_, start, end, label="simulated_symptom", **extra):
    return {"id": id_, "label_id": label, "start_s": start, "end_s": end, "confidence": "medium", **extra}


# ---- annotations ------------------------------------------------------------------------------

def test_create_read_update_delete_and_reload(tmp_path):
    doc = save(tmp_path, [ann("ann_one", 1.234567, 4.5)])
    assert doc["revision"] == 1
    assert doc["annotations"][0]["start_s"] == 1.234567  # sub-second precision kept
    again = STORE.load_annotations(tmp_path, "TrialA", DURATION)
    assert again["annotations"] == doc["annotations"]
    created = again["annotations"][0]["created_at"]
    edited = save(tmp_path, [ann("ann_one", 2.0, 4.5, label="uncertain", notes="moved")], revision=1)
    assert edited["annotations"][0]["label_id"] == "uncertain"
    assert edited["annotations"][0]["created_at"] == created
    deleted = save(tmp_path, [], revision=2)
    assert deleted["annotations"] == [] and deleted["revision"] == 3
    assert STORE.load_annotations(tmp_path, "TrialA", DURATION)["annotations"] == []


def test_overlapping_intervals_are_kept_as_entered(tmp_path):
    doc = save(tmp_path, [ann("ann_aaa", 10, 30), ann("ann_bbb", 20, 40, label="uncertain")])
    assert [(a["start_s"], a["end_s"]) for a in doc["annotations"]] == [(10, 30), (20, 40)]


@pytest.mark.parametrize("bad, message", [
    (ann("ann_bad", 5, 4), "after the start"),
    (ann("ann_bad", -1, 4), "negative"),
    (ann("ann_bad", 5, DURATION + 0.01), "end of the session"),
    (ann("ann_bad", float("nan"), 4), "finite"),
    (ann("ann_bad", "5", 6), "number"),
    (ann("ann_bad", True, 6), "number"),
    (ann("ann_bad", 1, 2, label="missing_label"), "unknown label"),
    (ann("ann_bad", 1, 2, confidence="sure"), "confidence"),
    (ann("ann_bad", 1, 2, review_status="done"), "review status"),
    (ann("x", 1, 2), "invalid ID"),
    ("not an object", "must be an object"),
])
def test_invalid_annotations_are_rejected_and_nothing_is_written(tmp_path, bad, message):
    with pytest.raises(STORE.ValidationError, match=message):
        save(tmp_path, [bad])
    assert not STORE.annotations_path(tmp_path, "TrialA").exists()


def test_end_exactly_at_session_end_is_fine_and_not_moved(tmp_path):
    doc = save(tmp_path, [ann("ann_end", 100, DURATION)])
    assert doc["annotations"][0]["end_s"] == DURATION


def test_duplicate_ids_and_bad_trials_are_rejected(tmp_path):
    with pytest.raises(STORE.ValidationError, match="Duplicate"):
        save(tmp_path, [ann("ann_dup", 1, 2), ann("ann_dup", 3, 4)])
    for trial in ("", "../x", "a b"):
        with pytest.raises(STORE.ValidationError, match="trial"):
            save(tmp_path, [], trial=trial)


def test_stale_saves_are_refused_instead_of_overwriting(tmp_path):
    save(tmp_path, [ann("ann_one", 1, 2)])
    with pytest.raises(STORE.ConflictError) as raised:
        save(tmp_path, [ann("ann_two", 3, 4)], revision=0)
    assert raised.value.current["revision"] == 1
    assert [a["id"] for a in STORE.load_annotations(tmp_path, "TrialA", DURATION)["annotations"]] == ["ann_one"]


def test_trials_are_stored_separately(tmp_path):
    save(tmp_path, [ann("ann_one", 1, 2)], trial="TrialA")
    save(tmp_path, [ann("ann_two", 3, 4)], trial="TrialB")
    assert [a["id"] for a in STORE.load_annotations(tmp_path, "TrialA", DURATION)["annotations"]] == ["ann_one"]
    assert [a["id"] for a in STORE.load_annotations(tmp_path, "TrialB", DURATION)["annotations"]] == ["ann_two"]
    path = STORE.annotations_path(tmp_path, "TrialA")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["trial_id"] = "TrialB"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(STORE.ValidationError, match="different trial"):
        STORE.load_annotations(tmp_path, "TrialA", DURATION)


def test_writes_are_atomic_and_a_damaged_file_falls_back_to_the_backup(tmp_path):
    save(tmp_path, [ann("ann_one", 1, 2)])
    save(tmp_path, [ann("ann_one", 1, 2), ann("ann_two", 3, 4)], revision=1)
    path = STORE.annotations_path(tmp_path, "TrialA")
    assert path.with_name(path.name + ".bak").is_file()
    assert not list(path.parent.glob("*.tmp"))
    path.write_text('{"truncated": ', encoding="utf-8")  # an interrupted write elsewhere
    recovered = STORE.load_annotations(tmp_path, "TrialA", DURATION)
    assert recovered["revision"] == 1 and "backup" in recovered["warning"]


def test_existing_activity_labels_and_parkinson_files_are_never_touched(tmp_path):
    trial = tmp_path / "TrialA"
    for relative in ("manual_labels/session_manual_segments.json", "parkinson_motion_review/manual_segments.json",
                     "parkinson_motion_review/manual_freezing_segments.json"):
        (trial / relative).parent.mkdir(parents=True, exist_ok=True)
        (trial / relative).write_text('{"segments": [1]}', encoding="utf-8")
    save(tmp_path, [ann("ann_one", 1, 2)])
    for relative in ("manual_labels/session_manual_segments.json", "parkinson_motion_review/manual_segments.json",
                     "parkinson_motion_review/manual_freezing_segments.json"):
        assert (trial / relative).read_text(encoding="utf-8") == '{"segments": [1]}'
    assert STORE.annotations_path(tmp_path, "TrialA") == trial / "symptom_annotations" / "annotations.json"


# ---- labels -----------------------------------------------------------------------------------

def test_default_labels_are_provisional_and_include_reviewed_states(tmp_path):
    doc = STORE.load_labels(tmp_path)
    assert doc["revision"] == 0
    assert [label["id"] for label in doc["labels"]] == ["no_target_symptom", "simulated_symptom", "uncertain", "not_observable"]
    assert not STORE.labels_path(tmp_path).exists()  # nothing written until the first save


def test_custom_labels_rename_and_archive_keep_annotations_linked(tmp_path):
    labels = STORE.load_labels(tmp_path)["labels"]
    labels.append({"id": "lbl_freeze0001", "name": "Freezing", "color": "#2F6DB5", "description": "Provisional"})
    doc = STORE.save_labels(tmp_path, {"base_revision": 0, "labels": labels})
    assert doc["labels"][-1]["color"] == "#2f6db5"
    save(tmp_path, [ann("ann_one", 1, 2, label="lbl_freeze0001")])
    renamed = [dict(label) for label in doc["labels"]]
    renamed[-1].update(name="Freezing episode", archived=True)
    doc = STORE.save_labels(tmp_path, {"base_revision": 1, "labels": renamed})
    assert doc["labels"][-1]["name"] == "Freezing episode" and doc["labels"][-1]["archived"]
    stored = STORE.load_annotations(tmp_path, "TrialA", DURATION)["annotations"][0]
    assert stored["label_id"] == "lbl_freeze0001"  # rename and archive do not break the link
    # Archived labels can still be used by saved annotations (e.g. when one is edited).
    save(tmp_path, [ann("ann_one", 1, 3, label="lbl_freeze0001")], revision=1)


def test_labels_cannot_be_removed_duplicated_or_malformed(tmp_path):
    doc = STORE.save_labels(tmp_path, {"base_revision": 0, "labels": STORE.load_labels(tmp_path)["labels"]})
    with pytest.raises(STORE.ValidationError, match="only archived"):
        STORE.save_labels(tmp_path, {"base_revision": 1, "labels": doc["labels"][1:]})
    twin = doc["labels"] + [{"id": "lbl_twin00001", "name": "  uncertain ", "color": "#123456"}]
    with pytest.raises(STORE.ValidationError, match="Two labels"):
        STORE.save_labels(tmp_path, {"base_revision": 1, "labels": twin})
    bad_color = doc["labels"] + [{"id": "lbl_color0001", "name": "New", "color": "red"}]
    with pytest.raises(STORE.ValidationError, match="color"):
        STORE.save_labels(tmp_path, {"base_revision": 1, "labels": bad_color})
    with pytest.raises(STORE.ConflictError):
        STORE.save_labels(tmp_path, {"base_revision": 0, "labels": doc["labels"]})


# ---- HTTP API ---------------------------------------------------------------------------------

@pytest.fixture
def api(tmp_path, monkeypatch):
    """The real request handler on a free port, with a synthetic runs/ folder."""
    server_module = load("pd_insighter_server_under_test", "pd_insighter_server.py")
    runs = tmp_path / "ws" / "Qwen2.5-VL-fork" / "runs"
    (runs / "TrialA").mkdir(parents=True)
    (runs / "TrialA" / "session_dashboard_manifest.json").write_text(json.dumps({"trial_id": "TrialA", "session_duration_s": DURATION}))
    monkeypatch.setattr(server_module, "RUNS_DIR", runs)
    monkeypatch.setattr(server_module, "SYMPTOM_STORE", runs)
    monkeypatch.setattr(server_module, "WORKSPACE_ROOT", tmp_path / "ws")
    server = ThreadingHTTPServer(("127.0.0.1", 0), server_module.PDInsighterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(base + path, data=data, method="POST" if body is not None else "GET",
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    call.runs = runs
    yield call
    server.shutdown()


def test_api_round_trip(api):
    status, doc = api("/api/symptom-annotations?trial_id=TrialA")
    assert status == 200 and doc["revision"] == 0 and doc["session_clock"]["duration_s"] == DURATION
    status, saved = api("/api/symptom-annotations", {"trial_id": "TrialA", "base_revision": 0, "annotator": "AB",
                                                     "annotations": [ann("ann_one", 1.5, 2.25)]})
    assert status == 200 and saved["revision"] == 1 and saved["annotations"][0]["annotator"] == "AB"
    status, loaded = api("/api/symptom-annotations?trial_id=TrialA")
    assert loaded["annotations"] == saved["annotations"]
    assert (api.runs / "TrialA" / "symptom_annotations" / "annotations.json").is_file()


def test_api_errors_are_clear(api):
    assert api("/api/symptom-annotations?trial_id=Missing")[0] == 404
    assert api("/api/symptom-annotations?trial_id=../etc")[0] == 400
    status, body = api("/api/symptom-annotations", {"trial_id": "TrialA", "base_revision": 0, "annotations": [ann("ann_bad", 9, 3)]})
    assert status == 400 and "after the start" in body["error"]
    api("/api/symptom-annotations", {"trial_id": "TrialA", "base_revision": 0, "annotations": []})
    status, body = api("/api/symptom-annotations", {"trial_id": "TrialA", "base_revision": 0, "annotations": []})
    assert status == 409 and body["conflict"] and body["current"]["revision"] == 1
    status, body = api("/api/symptom-labels", {"base_revision": 0, "labels": []})
    assert status == 200  # first save of the catalogue may start empty
    status, body = api("/api/symptom-labels", {"base_revision": 0, "labels": []})
    assert status == 409


def test_existing_annotation_endpoints_are_unchanged(api):
    # The activity-label store keeps its own file and API.
    status, body = api("/api/session-manual-segments?trial_id=TrialA")
    assert status == 200 and body["segments"] == []
    assert not (api.runs / "TrialA" / "symptom_annotations").exists()


# ---- navigation -------------------------------------------------------------------------------

PAGE = ('<html><head></head><body><div class="nav-tabs" aria-label="Trial views">'
        '<a class="nav-tab" href="index.html">Full Session</a></div><video id="ariaVideo"></video></body></html>')


def test_symptom_tab_appears_only_with_its_page_and_is_idempotent(tmp_path):
    assert "Symptom Annotation" not in NAV.inject_environment_tab(PAGE, tmp_path, active_view="stride")
    (tmp_path / SYMNAV.SYMPTOM_PAGE).write_text("x")
    page = NAV.inject_environment_tab(PAGE, tmp_path, active_view="stride")
    assert page.count('data-view="symptoms"') == 1
    assert f'href="{SYMNAV.SYMPTOM_PAGE}"' in page
    assert NAV.inject_environment_tab(page, tmp_path, active_view="stride") == page
    own = NAV.inject_environment_tab(PAGE, tmp_path, active_view="symptoms")
    assert '<span class="nav-tab active" data-view="symptoms" aria-current="page">Symptom Annotation</span>' in own


def test_symptom_tab_sits_beside_the_other_tabs_and_keeps_page_audio(tmp_path):
    (tmp_path / SYMNAV.SYMPTOM_PAGE).write_text("x")
    page = NAV.inject_environment_tab(PAGE, tmp_path, active_view="manual")
    tabs = page.split('class="nav-tabs"', 1)[1].split("</div>", 1)[0]
    assert tabs.index('data-view="symptoms"') < tabs.index('data-view="environment"')
    assert 'id="pd-page-audio"' in page  # Nancy's shared audio controller is still injected


def test_symptom_tab_links_use_forward_slashes_from_nested_pages(tmp_path):
    (tmp_path / SYMNAV.SYMPTOM_PAGE).write_text("x")
    nested = tmp_path / "environment_reconstruction"
    nested.mkdir()
    tab = SYMNAV.symptom_tab(tmp_path, nested, "environment")
    assert f'href="../{SYMNAV.SYMPTOM_PAGE}"' in tab and "\\" not in tab


def test_trial_switcher_handles_the_symptom_view(tmp_path):
    runs = tmp_path / "runs"
    for name in ("Parki", "Other"):
        (runs / name).mkdir(parents=True)
        (runs / name / "session_dashboard_manifest.json").write_text("{}")
        (runs / name / "index.html").write_text(PAGE)
    (runs / "Parki" / SYMNAV.SYMPTOM_PAGE).write_text(PAGE)
    page = NAV.inject_environment_tab(PAGE, runs / "Parki", active_view="symptoms")  # no KeyError
    switcher = page.split('id="shared-simulation-trial-switcher"', 1)[1]
    assert SYMNAV.SYMPTOM_PAGE in switcher and "Other (Full Session)" in switcher


def test_full_session_keeps_the_moment_when_switching(tmp_path):
    (tmp_path / SYMNAV.SYMPTOM_PAGE).write_text("x")
    page = NAV.inject_environment_tab(PAGE, tmp_path, active_view="full")
    assert page.count('id="pd-insighter-session-time"') == 1
    assert NAV.inject_environment_tab(page, tmp_path, active_view="full") == page
    assert "pd-insighter-session-time" not in NAV.inject_environment_tab(PAGE, tmp_path, active_view="stride")


def test_refresh_covers_the_symptom_page(tmp_path, monkeypatch):
    trial = tmp_path / "TrialA"
    trial.mkdir()
    (trial / SYMNAV.SYMPTOM_PAGE).write_text(PAGE.replace('href="index.html">Full Session</a>', 'href="index.html">Full Session</a>'))
    (trial / "index.html").write_text(PAGE)
    monkeypatch.setattr(NAV, "RUNS_ROOT", tmp_path)
    changed = NAV.refresh_navigation(["TrialA"])
    assert trial / SYMNAV.SYMPTOM_PAGE in changed
    assert 'aria-current="page">Symptom Annotation' in (trial / SYMNAV.SYMPTOM_PAGE).read_text()
    assert NAV.refresh_navigation(["TrialA"]) == []


# ---- page generation --------------------------------------------------------------------------

def write_trial(runs: Path, trial_id: str, session: dict, trial_manifest: dict | None = None, videos=("aria",)) -> Path:
    trial = runs / trial_id
    (trial / "downward_master_session").mkdir(parents=True, exist_ok=True)
    for key in videos:
        (trial / "downward_master_session" / f"{key}_session_video.mp4").write_bytes(b"\0")
    base = {"trial_id": trial_id, "recording_stem": f"{trial_id}_", "session_duration_s": 90.5, "session_fps": 30,
            "aria_session_video_uri": "downward_master_session/aria_session_video.mp4?v=1",
            "downward_session_video_uri": "downward_master_session/downward_session_video.mp4?v=1",
            "continuous_model_segments": [{"start_s": 0, "end_s": 40, "label": "walking"}]}
    (trial / "session_dashboard_manifest.json").write_text(json.dumps({**base, **session}))
    if trial_manifest is not None:
        (trial / "trial_manifest.json").write_text(json.dumps(trial_manifest))
    (trial / "index.html").write_text(PAGE)
    return trial


def page_data(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8").split('type="application/json">', 1)[1].split("</script>", 1)[0])


@pytest.fixture
def generator(tmp_path, monkeypatch):
    module = load("build_symptom_annotation_page_under_test", "build_symptom_annotation_page.py")
    runs = tmp_path / "Qwen2.5-VL-fork" / "runs"
    write_trial(runs, "TrialA", {})
    monkeypatch.setattr(module, "RUNS_DIR", runs)
    monkeypatch.setattr(module, "QWEN_REPO", runs.parent)
    return module, runs


def test_generated_page_marks_missing_media_and_is_reproducible(generator):
    module, runs = generator
    page = module.build("TrialA")
    first = page.read_text(encoding="utf-8")
    data = page_data(page)
    videos = {video["key"]: video for video in data["videos"]}
    assert videos["aria"]["available"] and not videos["downward"]["available"]
    assert videos["downward"]["missing"] == "missing_file"
    assert all(video["offset_s"] == 0.0 for video in data["videos"])
    assert videos["downward"]["turns"] == 2 and videos["downward"]["coverage_s"] is None  # legacy mount
    assert data["session_duration_s"] == 90.5 and data["activity_segments"][0]["label"] == "walking"
    assert "No alignment record" in data["sync_note"]
    # Manual activity labels are read through the existing endpoint, never copied into the page.
    assert data["manual_activity_api"] == "/api/session-manual-segments"
    assert data["aria_session_video_uri"].startswith("downward_master_session/aria_session_video.mp4")
    assert (runs / "TrialA" / "symptom_annotation" / "symptom_annotation_app.mjs").is_file()
    assert first.count('data-view="symptoms"') == 1
    assert 'id="pd-page-audio"' in first
    assert module.build("TrialA").read_text(encoding="utf-8") == first
    assert not (runs / "TrialA" / "symptom_annotations").exists()  # no annotation file is created


def test_attached_downward_footage_uses_its_recorded_window_and_rotation(generator):
    module, runs = generator
    attachment = {"start_s": 9.5, "end_s": 80.0, "session_duration_s": 90.5, "rotation_quarter_turns": 0,
                  "applied_offset_s": 9.5, "verification": {"valid": True}}
    write_trial(runs, "Attached", {"media_mode": "dual_view", "downward_attachment": attachment},
                {"media_mode": "dual_view", "downward_attachment": attachment}, videos=("aria", "downward"))
    data = page_data(module.build("Attached"))
    downward = {video["key"]: video for video in data["videos"]}["downward"]
    assert downward["available"] and downward["turns"] == 0 and downward["coverage_s"] == [9.5, 80.0]
    assert "padding" in data["sync_note"] and "audio-verified" in data["sync_note"]
    # Rotation in the trial manifest alone is also honoured.
    write_trial(runs, "TrialOnly", {}, {"downward_attachment": {**attachment, "rotation_quarter_turns": 3}},
                videos=("aria", "downward"))
    downward = {v["key"]: v for v in page_data(module.build("TrialOnly"))["videos"]}["downward"]
    assert downward["turns"] == 3 and downward["coverage_s"] == [9.5, 80.0]


def test_aria_only_trials_build_without_a_downward_camera(generator):
    module, runs = generator
    write_trial(runs, "AriaOnly", {"media_mode": "aria_only", "downward_session_video_uri": ""}, {"media_mode": "aria_only"})
    assert module.problems("AriaOnly") == []
    data = page_data(module.build("AriaOnly"))
    downward = {v["key"]: v for v in data["videos"]}["downward"]
    assert data["media_mode"] == "aria_only" and not downward["available"] and downward["missing"] == "not_recorded"


def test_missing_prerequisites_are_reported(generator):
    module, runs = generator
    (runs / "NoManifest").mkdir()
    assert "no session_dashboard_manifest.json" in module.problems("NoManifest")[0]
    with pytest.raises(FileNotFoundError):
        module.build("NoManifest")
    write_trial(runs, "NoVideo", {}, videos=())
    assert any("no session video" in issue for issue in module.problems("NoVideo"))
    write_trial(runs, "NoDuration", {"session_duration_s": 0})
    assert any("no session duration" in issue for issue in module.problems("NoDuration"))


def test_trial_library_links_the_page(generator, monkeypatch):
    module, runs = generator
    library = load("build_pd_insighter_trial_selector_under_test", "build_pd_insighter_trial_selector.py")
    monkeypatch.setattr(library, "RUNS_DIR", runs)
    item = next(t for t in library.collect_trials() if t["trial_id"] == "TrialA")
    assert item["symptom_annotation_href"] == ""
    module.build("TrialA")
    item = next(t for t in library.collect_trials() if t["trial_id"] == "TrialA")
    assert item["symptom_annotation_href"].startswith(f"TrialA/{SYMNAV.SYMPTOM_PAGE}?v=")
    assert ">Symptom Annotation</a>" in library.render_trial(item)


def test_update_script_is_a_dry_run_until_applied_and_then_idempotent(generator, monkeypatch, capsys):
    module, runs = generator
    update = load("update_symptom_annotation_pages_under_test", "update_symptom_annotation_pages.py")
    monkeypatch.setattr(update, "builder", module)
    monkeypatch.setattr(update.navigation, "RUNS_ROOT", runs)
    (runs / "Broken").mkdir()
    (runs / "Broken" / "session_dashboard_manifest.json").write_text(json.dumps({"session_duration_s": 0}))
    index = runs / "TrialA" / "index.html"
    before = index.read_text(encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["update", "--skip-library"])
    update.main()
    out = capsys.readouterr().out
    assert "Dry run" in out and "skip" in out and "Broken" in out
    assert not (runs / "TrialA" / SYMNAV.SYMPTOM_PAGE).exists()
    assert index.read_text(encoding="utf-8") == before
    monkeypatch.setattr(sys, "argv", ["update", "--skip-library", "--apply"])
    update.main()
    assert (runs / "TrialA" / SYMNAV.SYMPTOM_PAGE).is_file()
    assert not (runs / "Broken" / SYMNAV.SYMPTOM_PAGE).exists()
    assert 'data-view="symptoms"' in index.read_text(encoding="utf-8")
    after = index.read_text(encoding="utf-8")
    update.main()
    assert "0 page(s) written, 0 view(s)" in capsys.readouterr().out.splitlines()[-1]
    assert index.read_text(encoding="utf-8") == after
