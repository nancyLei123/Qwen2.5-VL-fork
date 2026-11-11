"""Automatic symptom detection: signals, detector, storage, review decisions, API and evaluation.

Everything here is synthetic: head tracks are generated, no recording or research data is read.
"""

import hashlib
import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import evaluate_symptom_detection as EVAL  # noqa: E402
import run_symptom_detection as RUN  # noqa: E402
import symptom_annotations as STORE  # noqa: E402
import symptom_detection as DET  # noqa: E402
import symptom_detection_signals as SIG  # noqa: E402
import symptom_detection_store as DSTORE  # noqa: E402

FPS = 30.0


# ---- synthetic movement ---------------------------------------------------------------------------

def track(segments, fps=FPS, seed=0):
    """Head positions for a list of (seconds, kind, speed).

    Kinds: walk (1.7 steps/s), slowwalk (0.8 steps/s), still, sway and tremble (movement in place), sit.
    """
    rng = np.random.default_rng(seed)
    x = 0.0
    rows, t = [], 0.0
    for seconds, kind, speed in segments:
        for _ in range(int(seconds * fps)):
            y, z = 0.0, 1.6
            if kind in ("walk", "slowwalk"):
                cadence = 1.7 if kind == "walk" else 0.8
                x += speed / fps
                z += 0.012 * np.sin(2 * np.pi * cadence * t)       # head bob, one cycle per step
                y += 0.02 * np.sin(2 * np.pi * cadence / 2 * t)    # side-to-side sway of walking
            elif kind == "sway":                                   # no progress, body keeps stepping in place
                y += 0.05 * np.sin(2 * np.pi * 1.5 * t)
                z += 0.008 * np.sin(2 * np.pi * 3.0 * t)
            elif kind == "tremble":                                # no progress, quick small movement in place
                y += 0.03 * np.sin(2 * np.pi * 2.0 * t)
            elif kind == "sit":                                    # leaning about in a chair: small excursions
                y += 0.10 * np.sin(2 * np.pi * 0.25 * t)
                z = 1.2
            rows.append([x, y, z])
            t += 1 / fps
    xyz = np.array(rows) + rng.normal(0, 0.0005, (len(rows), 3))
    return {"t": np.arange(len(rows)) / fps, "xyz": xyz}


def signals(segments, video=None, coverage=None, seed=0):
    """TrialSignals from a synthetic track. ``video`` maps kind -> flow level to add a downward-video source."""
    head = track(segments, seed=seed)
    duration = len(head["t"]) / FPS
    t = SIG.grid(duration)
    out = SIG.TrialSignals(trial_id="Synthetic", duration_s=duration, t=t)
    out.features.update(SIG.head_features(head, t))
    out.sources["head_track"] = {"available": True}
    out.activity = np.zeros(len(t), dtype=np.int8)
    if video is not None:
        flow = np.zeros(len(t))
        cursor = 0.0
        for seconds, kind, _ in segments:
            flow[(t >= cursor) & (t < cursor + seconds)] = video[kind]
            cursor += seconds
        ok = np.ones(len(t), dtype=bool)
        if coverage:
            ok = (t >= coverage[0]) & (t <= coverage[1])
        flow = np.where(ok, flow, np.nan)
        out.features.update(flow_smooth=flow, flow=flow, flow_step_amp=flow / 2, video_ok=ok)
        out.downward_coverage_s = coverage or (0.0, duration)
        out.sources["downward_video"] = {"available": True}
    else:
        out.features["video_ok"] = np.zeros(len(t), dtype=bool)
        out.sources["downward_video"] = {"available": False}
    return out


FLOW = {"walk": 2.4, "slowwalk": 2.0, "sway": 2.0, "tremble": 2.0, "still": 0.4, "sit": 0.5}
FREEZE = [(6, "still", 0), (14, "walk", 0.8), (5, "sway", 0), (12, "walk", 0.8), (6, "still", 0)]


def only(document, label):
    return [item for item in document["detections"] if item["label_id"] == label]


# ---- detector -------------------------------------------------------------------------------------

def test_freezing_like_episode_is_found_at_the_right_time_with_both_sources():
    document = DET.detect(signals(FREEZE, video=FLOW))
    found = only(document, "freezing_like")
    assert len(found) == 1
    item = found[0]
    assert abs(item["start_s"] - 20.0) < 1.0 and abs(item["end_s"] - 25.0) < 1.0      # session clock, seconds
    assert item["evidence_strength"] == "high" and item["score"] >= 70
    assert set(item["independent_sources"]) == {"head_track", "downward_video"}
    assert item["score_kind"].startswith("uncalibrated")
    assert all(entry["text"] for entry in item["evidence"])


def test_a_quiet_stop_is_an_ordinary_stop_and_not_reported():
    quiet = [(6, "still", 0), (14, "walk", 0.8), (5, "still", 0), (12, "walk", 0.8), (6, "still", 0)]
    document = DET.detect(signals(quiet, video=FLOW))
    assert only(document, "freezing_like") == []
    assert document["summary"]["not_reported"]["quiet_stops_not_reported"] >= 1


def test_without_downward_footage_the_candidate_is_weaker_and_says_why():
    with_video = only(DET.detect(signals(FREEZE, video=FLOW)), "freezing_like")[0]
    without = only(DET.detect(signals(FREEZE)), "freezing_like")[0]
    assert without["score"] < with_video["score"]
    assert without["evidence_strength"] != "high"                    # one source cannot make it high
    assert any("downward camera" in text for text in without["limitations"])
    assert any(entry["kind"] == "missing" for entry in without["evidence"])


def test_padding_outside_the_recorded_window_is_not_used_as_evidence():
    document = DET.detect(signals(FREEZE, video=FLOW, coverage=(26.0, 43.0)))   # footage starts after the episode
    item = only(document, "freezing_like")[0]
    video = next(entry for entry in item["evidence"] if entry["id"] == "in_place_video")
    assert video["kind"] == "missing" and "no footage for this interval" in video["text"]


def test_shifting_in_a_chair_is_not_walking():
    seated = [(40, "sit", 0), (5, "still", 0), (30, "sit", 0)]
    document = DET.detect(signals(seated, video=FLOW))
    assert document["detections"] == []


def test_slow_walking_is_reported_and_ordinary_walking_is_not():
    slow = DET.detect(signals([(4, "still", 0), (20, "slowwalk", 0.2), (4, "still", 0)], video=FLOW))
    found = only(slow, "slow_gait")
    assert len(found) == 1 and found[0]["measurements"]["median_speed_mps"] == pytest.approx(0.2, abs=0.05)
    assert abs(found[0]["start_s"] - 4.0) < 2.5 and abs(found[0]["end_s"] - 24.0) < 2.5
    normal = DET.detect(signals([(4, "still", 0), (20, "walk", 0.9), (4, "still", 0)], video=FLOW))
    assert only(normal, "slow_gait") == [] and only(normal, "shuffling_like") == []


def test_short_steps_at_a_kept_rhythm_are_shuffling_like():
    document = DET.detect(signals([(4, "still", 0), (20, "walk", 0.3), (4, "still", 0)], video=FLOW))
    found = only(document, "shuffling_like")                          # 0.3 m/s at 1.7 steps/s: about 0.18 m per step
    assert len(found) == 1 and found[0]["measurements"]["median_step_length_m"] < 0.3
    assert any("head motion" in text for text in found[0]["limitations"])


def test_a_short_interruption_inside_walking_is_a_hesitation():
    document = DET.detect(signals([(4, "still", 0), (12, "walk", 0.8), (1.4, "tremble", 0), (12, "walk", 0.8), (4, "still", 0)], video=FLOW))
    found = [item for item in only(document, "gait_hesitation") if item["kind"] == "interruption"]
    assert len(found) == 1 and abs(found[0]["start_s"] - 16.0) < 1.0


def test_scores_are_bounded_explained_and_ordered_by_evidence():
    evidence = DET.Evidence()
    evidence.support("a", "head_track", 60, 0.5, "half")
    evidence.support("b", "downward_video", 40, 1.0, "full")
    assert evidence.score() == 70                                     # (30 + 40) of 100 possible points
    evidence.penalty("p", "head_track", 30, 1.0, "ordinary explanation")
    assert evidence.score() == 40
    evidence.penalty("q", "head_track", 100, 1.0, "more")
    assert evidence.score() == 0                                      # never below 0
    evidence = DET.Evidence()
    evidence.support("a", "head_track", 50, 1.0, "full")
    evidence.missing("b", "downward_video", 50, "not visible")
    assert evidence.score() == 50                                     # missing evidence still counts in the total
    assert evidence.independent_sources(0.4) == ["head_track"]
    assert DET.ramp(0.5, 0, 1) == 0.5 and DET.ramp(2, 0, 1) == 1 and DET.ramp(0.02, 0.045, 0.015) > 0.8
    assert DET.ramp(float("nan"), 0, 1) == 0


def test_detection_is_reproducible_and_versioned_by_its_configuration():
    first, second = DET.detect(signals(FREEZE, video=FLOW)), DET.detect(signals(FREEZE, video=FLOW))
    assert first == second
    assert first["schema"] == DSTORE.DETECTIONS_SCHEMA and first["run_id"].startswith("motion-baseline-")
    changed = DET.detect(signals(FREEZE, video=FLOW), {"freezing_like": {"min_s": 9.0}})
    assert changed["run_id"] != first["run_id"] and only(changed, "freezing_like") == []
    renamed = DET.detect(signals(FREEZE, video=FLOW), {"labels": {"freezing_like": {"name": "FOG candidate"}}})
    assert only(renamed, "freezing_like")[0]["label_name"] == "FOG candidate"       # taxonomy is configurable
    disabled = DET.detect(signals(FREEZE, video=FLOW), {"labels": {"freezing_like": {"enabled": False}}})
    assert all(item["label_id"] != "freezing_like" for item in disabled["detections"])
    assert any("not evidence that no symptom occurred" in note for note in first["summary"]["notes"])


def test_no_head_track_means_no_invented_detections():
    empty = SIG.TrialSignals(trial_id="Empty", duration_s=30.0, t=SIG.grid(30.0))
    empty.sources["head_track"] = {"available": False}
    document = DET.detect(empty)
    assert document["detections"] == [] and "nothing could be detected" in document["summary"]["notes"][0]


# ---- signals and time mapping ---------------------------------------------------------------------

def write_trial(runs, trial="TrialA", *, playback=True, attachment=None, media_mode=None, segments=FREEZE):
    head = track(segments)
    duration = len(head["t"]) / FPS
    folder = runs / trial
    folder.mkdir(parents=True)
    samples = [{"t": float(t), "x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "speed": 0, "path": 0}
               for t, p in zip(head["t"][::2], head["xyz"][::2])]
    session = {"trial_id": trial, "session_duration_s": duration, "session_fps": FPS,
               "continuous_model_segments": [{"label": "standing", "start_s": 0.0, "end_s": 6.0},
                                             {"label": "walking", "start_s": 6.0, "end_s": duration}],
               "transition_review": {"samples": samples},
               "stride_review": {"events": [{"session_s": 7.5}, {"session_s": 8.1}]}}
    if attachment:
        session["downward_attachment"] = attachment
        session["downward_session_video_uri"] = "downward_master_session/downward_session_video.mp4?v=1"
        (folder / "downward_master_session").mkdir()
        (folder / "downward_master_session" / "downward_session_video.mp4").write_bytes(b"not a real video")
    if media_mode:
        session["media_mode"] = media_mode
    (folder / "session_dashboard_manifest.json").write_text(json.dumps(session))
    if playback:
        poses = [None, None] + [[*map(float, p), 0.0, 0.0, 0.0, 1.0] for p in head["xyz"][2:]]
        status = ["unavailable", "unavailable"] + ["mps_supported"] * (len(poses) - 2)
        (folder / "environment_reconstruction").mkdir()
        (folder / "environment_reconstruction" / "scene_playback_data.json").write_text(json.dumps(
            {"fps": FPS, "dashboard_offset_s": 0, "poses": poses, "pose_status": status}))
    return duration


def test_pose_frames_map_to_video_time_and_untracked_frames_stay_empty(tmp_path):
    write_trial(tmp_path)
    session = SIG.read_json(tmp_path / "TrialA" / "session_dashboard_manifest.json")
    head, report = SIG.load_head_track(tmp_path / "TrialA", session)
    assert report["has_orientation"] and report["rate_hz"] == FPS
    assert head["t"][90] == pytest.approx(3.0)                         # frame 90 of a 30 fps video is 3.0 s
    assert np.isnan(head["xyz"][0]).all() and np.isfinite(head["xyz"][2]).all()
    loaded = SIG.load_trial(tmp_path, "TrialA", use_video=False, use_audio=False)
    assert not loaded.features["head_ok"][0] and loaded.features["head_ok"][20]
    assert loaded.t[1] - loaded.t[0] == pytest.approx(1 / SIG.GRID_HZ)
    assert len(loaded.step_event_times) == 2 and loaded.activity[int(10 * SIG.GRID_HZ)] == SIG.ACTIVITY_CODES["walking"]


def test_manifest_samples_are_the_fallback_head_track(tmp_path):
    write_trial(tmp_path, playback=False)
    loaded = SIG.load_trial(tmp_path, "TrialA", use_video=False, use_audio=False)
    report = loaded.sources["head_track"]
    assert report["available"] and not report["has_orientation"] and "turning in place" in report["limitations"][0]
    assert only(DET.detect(loaded), "freezing_like")                  # the same episode is found from 15 Hz samples


def test_long_tracking_gaps_are_not_interpolated():
    t = np.array([0.0, 0.1, 0.2, 3.0, 3.1])
    values = np.array([0.0, 1.0, 2.0, 30.0, 31.0])
    out = SIG.nan_interp(np.array([0.15, 1.5, 3.05]), t, values, max_gap_s=0.5)
    assert out[0] == pytest.approx(1.5) and np.isnan(out[1]) and out[2] == pytest.approx(30.5)


def test_downward_coverage_comes_from_the_attachment_and_aria_only_has_none(tmp_path):
    duration = write_trial(tmp_path, attachment={"start_s": 9.5, "end_s": 30.0, "rotation_quarter_turns": 0})
    session = SIG.read_json(tmp_path / "TrialA" / "session_dashboard_manifest.json")
    coverage, path, note = SIG.downward_coverage(tmp_path / "TrialA", session, duration)
    assert coverage == (9.5, 30.0) and path.name == "downward_session_video.mp4" and "padding" in note
    session["media_mode"] = "aria_only"
    assert SIG.downward_coverage(tmp_path / "TrialA", session, duration)[0] is None


def test_video_features_mask_padding_and_dark_frames():
    t = SIG.grid(20.0)
    frames = np.arange(0, 20, 1 / 15)
    motion = {"t": frames, "flow": np.full(len(frames), 2.0), "flow_hi": np.full(len(frames), 3.0),
              "flow_moving": np.full(len(frames), 0.5), "luma": np.where(frames < 14, 90.0, 3.0)}   # lens covered after 14 s
    out = SIG.video_features(motion, t, coverage=(5.0, 18.0))
    ok = out["video_ok"]
    assert not ok[t < 5.0].any() and ok[(t > 6) & (t < 13)].all() and not ok[t > 14.5].any()
    assert np.isnan(out["flow"][t < 5.0]).all() and np.isnan(out["flow_smooth"][t > 16]).all()


def test_optical_flow_reads_only_the_recorded_window(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "downward.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (160, 90))
    if not writer.isOpened():
        pytest.skip("No MJPG writer in this OpenCV build.")
    rng = np.random.default_rng(1)
    texture = (rng.random((90, 480)) * 255).astype(np.uint8)
    for index in range(120):                                           # 4 s; the picture moves only in the second half
        shift = 0 if index < 60 else (index - 60) * 2
        frame = np.zeros((90, 160, 3), dtype=np.uint8) if index < 15 else cv2.cvtColor(texture[:, shift:shift + 160], cv2.COLOR_GRAY2BGR)
        writer.write(frame)
    writer.release()
    motion = SIG.video_motion(path, (1.0, 3.5), 4.0, cache_dir=tmp_path / "cache")
    assert motion["t"].min() >= 1.0 and motion["t"].max() <= 3.5
    assert np.median(np.diff(motion["t"])) == pytest.approx(1 / 15, abs=1e-6)
    still, moving = motion["flow"][motion["t"] < 1.9], motion["flow"][motion["t"] > 2.3]
    assert still.max() < 0.3 and moving.mean() > 1.0
    again = SIG.video_motion(path, (1.0, 3.5), 4.0, cache_dir=tmp_path / "cache")   # second call comes from the cache
    assert np.array_equal(again["flow"], motion["flow"]) and list((tmp_path / "cache").glob("*.npz"))


def test_audio_envelope_marks_onsets_without_reading_content(tmp_path):
    import wave
    rate = 16000
    samples = np.zeros(rate * 6, dtype=np.float32)
    for second in (1.0, 2.0, 3.0, 4.0):
        start = int(second * rate)
        samples[start:start + 400] += np.hanning(400) * np.sin(2 * np.pi * 300 * np.arange(400) / rate)
    with wave.open(str(tmp_path / "a.wav"), "wb") as wav:
        wav.setnchannels(1), wav.setsampwidth(2), wav.setframerate(rate)
        wav.writeframes((samples * 20000).astype("<i2").tobytes())
    out = SIG.audio_features(SIG.audio_envelope(tmp_path / "a.wav"), SIG.grid(6.0))
    assert len(out["audio_onset_times"]) == 4
    assert np.allclose(out["audio_onset_times"], [1.0, 2.0, 3.0, 4.0], atol=0.06)


# ---- storage ------------------------------------------------------------------------------------

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runs_are_saved_per_version_and_reruns_keep_the_previous_file(tmp_path):
    document = DET.detect(signals(FREEZE, video=FLOW))
    path, status = DSTORE.save_detections(tmp_path, document)
    assert status == "created" and path == tmp_path / "Synthetic" / "symptom_annotations" / "detections" / f"{document['run_id']}.json"
    assert DSTORE.save_detections(tmp_path, document)[1] == "unchanged"
    altered = json.loads(json.dumps(document))
    altered["detections"] = []
    assert DSTORE.save_detections(tmp_path, altered)[1] == "replaced"
    assert path.with_suffix(".json.bak").is_file()                     # the earlier predictions are still on disk
    other = DET.detect(signals(FREEZE, video=FLOW), {"freezing_like": {"min_s": 3.0}})
    DSTORE.save_detections(tmp_path, other)
    runs = DSTORE.list_runs(tmp_path, "Synthetic")
    assert {run["run_id"] for run in runs} == {document["run_id"], other["run_id"]}
    assert DSTORE.load_detections(tmp_path, "Synthetic")["run_id"] == runs[0]["run_id"]
    assert DSTORE.load_detections(tmp_path, "Synthetic", document["run_id"])["run_id"] == document["run_id"]
    assert DSTORE.load_detections(tmp_path, "Nobody") is None
    with pytest.raises(STORE.ValidationError):
        DSTORE.save_detections(tmp_path, {"schema": "something-else"})


def review(run, detection, decision, revision, **extra):
    return {"trial_id": "Synthetic", "run_id": run, "detection_id": detection, "decision": decision, "base_revision": revision, **extra}


def test_review_decisions_live_apart_from_predictions_and_annotations(tmp_path):
    document = DET.detect(signals(FREEZE, video=FLOW))
    path, _ = DSTORE.save_detections(tmp_path, document)
    before = digest(path)
    run, first = document["run_id"], document["detections"][0]["id"]
    saved = DSTORE.save_review(tmp_path, review(run, first, "rejected", 0, note="walking stopped to open a door", reviewer="AB"))
    entry = saved["reviews"][run][first]
    assert saved["revision"] == 1 and entry["decision"] == "rejected" and entry["detection"]["score"] == document["detections"][0]["score"]
    assert digest(path) == before                                      # the detector output is untouched
    assert not STORE.annotations_path(tmp_path, "Synthetic").exists()   # and so are the manual annotations
    changed = DSTORE.save_review(tmp_path, review(run, first, "accepted", 1, annotation_id="ann_abc12345",
                                                  accepted={"label_id": "freezing_like", "start_s": 19.5, "end_s": 25.0}, edited=True))
    entry = changed["reviews"][run][first]
    assert entry["annotation_id"] == "ann_abc12345" and entry["edited"] and entry["accepted"]["start_s"] == 19.5
    assert entry["history"][0]["decision"] == "rejected" and entry["history"][0]["note"]   # the rejection is preserved
    cleared = DSTORE.save_review(tmp_path, review(run, first, "unreviewed", 2))
    assert [h["decision"] for h in cleared["reviews"][run][first]["history"]] == ["rejected", "accepted"]
    assert DSTORE.load_reviews(tmp_path, "Synthetic")["revision"] == 3


@pytest.mark.parametrize("change, message", [
    ({"decision": "maybe"}, "decision must be"),
    ({"detection_id": "not-in-this-run"}, "not part of the detection run"),
    ({"run_id": "motion-baseline-9.9.9-ffffffff"}, "not saved for the trial"),
    ({"decision": "accepted"}, "needs the ID of the annotation"),
    ({"decision": "accepted", "annotation_id": "ann_abc12345", "accepted": {"start_s": 9, "end_s": 3}}, "after the start"),
])
def test_invalid_review_decisions_are_refused(tmp_path, change, message):
    document = DET.detect(signals(FREEZE, video=FLOW))
    DSTORE.save_detections(tmp_path, document)
    body = review(document["run_id"], document["detections"][0]["id"], "rejected", 0)
    body.update(change)
    with pytest.raises(STORE.ValidationError, match=message):
        DSTORE.save_review(tmp_path, body)
    assert not DSTORE.reviews_path(tmp_path, "Synthetic").exists()


def test_stale_review_saves_are_refused(tmp_path):
    document = DET.detect(signals(FREEZE, video=FLOW))
    DSTORE.save_detections(tmp_path, document)
    run, first = document["run_id"], document["detections"][0]["id"]
    DSTORE.save_review(tmp_path, review(run, first, "needs_review", 0))
    with pytest.raises(STORE.ConflictError) as error:
        DSTORE.save_review(tmp_path, review(run, first, "rejected", 0))
    assert error.value.current["revision"] == 1


def test_an_accepted_prediction_is_a_manual_annotation_that_remembers_its_origin(tmp_path):
    labels = STORE.load_labels(tmp_path)
    body = {"trial_id": "Synthetic", "base_revision": 0, "annotations": [
        {"id": "ann_abc12345", "label_id": "uncertain", "start_s": 20.0, "end_s": 25.0, "confidence": "high",
         "origin": {"type": "detection", "run_id": "motion-baseline-0.1.0-aaaaaaaa", "detection_id": "freezing_like-00020000", "edited": True}}]}
    saved = STORE.save_annotations(tmp_path, body, duration_s=60.0, labels=labels)["annotations"][0]
    assert saved["source"] == "manual" and saved["origin"]["detection_id"] == "freezing_like-00020000" and saved["origin"]["edited"]
    body = {"trial_id": "Synthetic", "base_revision": 1, "annotations": [{k: v for k, v in saved.items() if k != "origin"} | {"notes": "kept"}]}
    again = STORE.save_annotations(tmp_path, body, duration_s=60.0, labels=labels)["annotations"][0]
    assert again["origin"] == saved["origin"]                           # later edits do not lose the link
    bad = {"trial_id": "Synthetic", "base_revision": 2, "annotations": [dict(again, origin={"type": "guess"})]}
    with pytest.raises(STORE.ValidationError, match="origin"):
        STORE.save_annotations(tmp_path, bad, duration_s=60.0, labels=labels)


# ---- API ----------------------------------------------------------------------------------------

@pytest.fixture
def api(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("pd_insighter_server_for_detection", SCRIPTS / "pd_insighter_server.py")
    server_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server_module)
    runs = tmp_path / "ws" / "Qwen2.5-VL-fork" / "runs"
    store_dir = tmp_path / "store"
    (runs / "Synthetic").mkdir(parents=True)
    (runs / "Synthetic" / "session_dashboard_manifest.json").write_text(json.dumps({"trial_id": "Synthetic", "session_duration_s": 43.0}))
    monkeypatch.setattr(server_module, "RUNS_DIR", runs)
    monkeypatch.setattr(server_module, "SYMPTOM_STORE", store_dir)
    monkeypatch.setattr(server_module, "WORKSPACE_ROOT", tmp_path / "ws")
    server = ThreadingHTTPServer(("127.0.0.1", 0), server_module.PDInsighterHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
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

    call.runs, call.store = runs, store_dir
    yield call
    server.shutdown()


def test_api_serves_predictions_and_records_decisions(api):
    status, empty = api("/api/symptom-detections?trial_id=Synthetic")
    assert status == 200 and empty["runs"] == [] and empty["run"] is None and empty["reviews"]["revision"] == 0
    document = DET.detect(signals(FREEZE, video=FLOW))
    DSTORE.save_detections(api.store, document)
    status, bundle = api("/api/symptom-detections?trial_id=Synthetic")
    assert status == 200 and bundle["run"]["run_id"] == document["run_id"] and bundle["runs"][0]["count"] == len(document["detections"])
    first = document["detections"][0]["id"]
    status, reviews = api("/api/symptom-detection-reviews", review(document["run_id"], first, "rejected", 0))
    assert status == 200 and reviews["reviews"][document["run_id"]][first]["decision"] == "rejected"
    status, body = api("/api/symptom-detection-reviews", review(document["run_id"], first, "accepted", 0, annotation_id="ann_abc12345"))
    assert status == 409 and body["conflict"] and body["current"]["revision"] == 1
    assert api("/api/symptom-detection-reviews", review(document["run_id"], "nope-000", "rejected", 1))[0] == 400
    assert api("/api/symptom-detections?trial_id=Missing")[0] == 404
    assert api(f"/api/symptom-detections?trial_id=Synthetic&run_id={document['run_id']}")[1]["run"]["trial_id"] == "Synthetic"
    assert not (api.runs / "Synthetic" / "symptom_annotations").exists()     # nothing lands in the trial data
    assert api("/api/symptom-annotations?trial_id=Synthetic")[1]["annotations"] == []   # manual annotations stay empty


# ---- command line -------------------------------------------------------------------------------

def test_detector_command_is_a_dry_run_until_applied(tmp_path, monkeypatch, capsys):
    runs = tmp_path / "runs"
    write_trial(runs, playback=True)
    store_dir = tmp_path / "store"
    monkeypatch.setattr(RUN, "RUNS_DIR", runs)
    monkeypatch.setattr(RUN, "QWEN_REPO", tmp_path)
    before = {p: digest(p) for p in runs.rglob("*") if p.is_file()}
    assert RUN.main(["--no-video", "--no-audio", "--symptom-store", str(store_dir)]) == 0
    assert "Dry run" in capsys.readouterr().out and not store_dir.exists()
    report = tmp_path / "report.md"
    assert RUN.main(["--no-video", "--no-audio", "--symptom-store", str(store_dir), "--apply", "--report", str(report)]) == 0
    out = capsys.readouterr().out
    saved = list((store_dir / "TrialA" / "symptom_annotations" / "detections").glob("*.json"))
    assert "created" in out and len(saved) == 1
    assert json.loads(saved[0].read_text())["detections"]
    assert "Freezing-like episode" in report.read_text() and "uncalibrated" in report.read_text()
    assert RUN.main(["--no-video", "--no-audio", "--symptom-store", str(store_dir), "--apply"]) == 0
    assert "unchanged" in capsys.readouterr().out
    assert {p: digest(p) for p in runs.rglob("*") if p.is_file()} == before          # the trial data is only read


# ---- evaluation (synthetic reference annotations) ------------------------------------------------

def pred(id_, start, end, score=60, label="freezing_like"):
    return {"id": id_, "label_id": label, "start_s": start, "end_s": end, "score": score}


def ref(id_, start, end, label="freezing_like", origin=None):
    item = {"id": id_, "label_id": label, "start_s": start, "end_s": end}
    if origin:
        item["origin"] = {"type": "detection", "run_id": "r1", "detection_id": origin}
    return item


def document_for(predictions):
    return {"trial_id": "T", "run_id": "r1", "session_clock": {"duration_s": 200.0},
            "labels": [{"id": "freezing_like", "name": "Freezing-like episode"}], "detections": predictions}


LABELS = [{"id": "freezing_like", "name": "Freezing-like episode"}, {"id": "no_target_symptom", "name": "No Target Symptom"}]


def test_no_reference_means_no_metrics():
    result = EVAL.evaluate_trial(document_for([pred("p1", 10, 14)]), {"annotations": []}, None, LABELS)
    assert result["status"] == "no_reference" and "by_label" not in result
    assert EVAL.combine([result])["status"] == "no_reference"


def test_event_level_counts_precision_recall_and_timing():
    predictions = [pred("p1", 10, 14), pred("p2", 50, 53), pred("p3", 150, 154)]
    annotations = {"annotations": [ref("a1", 11, 15), ref("a2", 80, 84),                 # p1 matches a1; a2 is missed
                                   ref("a3", 40, 60, label="no_target_symptom")]}         # p2 falls in reviewed clear time
    result = EVAL.evaluate_trial(document_for(predictions), annotations, None, LABELS)
    event = result["by_label"]["freezing_like"]["event_level"]
    assert (event["true_positives"], event["false_positives"], event["false_negatives"]) == (1, 1, 1)
    assert event["unscored_predictions"] == 1 and event["unscored_ids"] == ["p3"]       # unreviewed time is not a false positive
    assert event["precision"] == 0.5 and event["recall"] == 0.5 and event["f1"] == 0.5
    assert event["mean_iou"] == pytest.approx(3 / 5)
    assert event["start_error_s"]["mean_signed"] == pytest.approx(-1.0) and event["end_error_s"]["mean_absolute"] == pytest.approx(1.0)
    assert event["missed_reference_ids"] == ["a2"]


def test_a_rejected_prediction_is_a_false_positive_even_in_unreviewed_time():
    annotations = {"annotations": [ref("a1", 11, 15)]}
    reviews = {"reviews": {"r1": {"p3": {"decision": "rejected"}}}}
    result = EVAL.evaluate_trial(document_for([pred("p1", 10, 14), pred("p3", 150, 154)]), annotations, reviews, LABELS)
    event = result["by_label"]["freezing_like"]["event_level"]
    assert event["false_positives"] == 1 and event["unscored_predictions"] == 0 and result["rejected_predictions"] == 1


def test_accepted_predictions_are_not_independent_evidence():
    annotations = {"annotations": [ref("a1", 10, 14, origin="p1"), ref("a2", 80, 84)]}
    result = EVAL.evaluate_trial(document_for([pred("p1", 10, 14), pred("p2", 81, 85)]), annotations, None, LABELS)
    entry = result["by_label"]["freezing_like"]
    assert result["references_from_accepted_predictions"] == 1 and entry["independent_references"] == 1
    assert entry["event_level"]["true_positives"] == 2
    independent = entry["event_level_independent_only"]
    assert independent["true_positives"] == 1 and independent["false_positives"] == 0     # p1 and its own annotation are left out


def test_time_level_agreement_is_separate_from_event_level():
    annotations = {"annotations": [ref("a1", 10, 20), ref("clear", 20, 40, label="no_target_symptom")]}
    result = EVAL.evaluate_trial(document_for([pred("p1", 15, 25)]), annotations, None, LABELS)
    time = result["by_label"]["freezing_like"]["time_level"]
    assert (time["true_positive_s"], time["false_positive_s"], time["false_negative_s"]) == (5.0, 5.0, 5.0)
    assert time["iou"] == pytest.approx(1 / 3) and time["reviewed_s"] == 30.0
    assert result["by_label"]["freezing_like"]["event_level"]["true_positives"] == 1    # one event, although only half the time agrees


def test_matching_is_one_to_one_and_respects_the_overlap_threshold():
    predictions = [pred("p1", 10, 20), pred("p2", 12, 18)]
    references = [ref("a1", 11, 19)]
    pairs = EVAL.match_events(predictions, references, 0.1)
    assert len(pairs) == 1 and pairs[0][0] == 0                                        # the better overlap wins; no double counting
    assert EVAL.match_events([pred("p", 0, 10)], [ref("a", 9.5, 30)], 0.1) == []        # touching is not matching
    assert EVAL.iou((0, 10), (5, 15)) == pytest.approx(1 / 3) and EVAL.iou((0, 1), (2, 3)) == 0


def test_score_thresholds_and_reliability_bands():
    predictions = [pred("p1", 10, 14, score=80), pred("p2", 50, 53, score=35), pred("p3", 70, 73, score=62)]
    annotations = {"annotations": [ref("a1", 10, 14), ref("a3", 70, 73), ref("clear", 40, 60, label="no_target_symptom")]}
    result = EVAL.evaluate_trial(document_for(predictions), annotations, None, LABELS)
    sweep = {row["min_score"]: row for row in result["by_label"]["freezing_like"]["by_min_score"]}
    assert sweep[0]["precision"] == pytest.approx(2 / 3) and sweep[40]["precision"] == 1.0 and sweep[70]["recall"] == 0.5
    summary = EVAL.combine([result])
    assert summary["event_level_overall"]["true_positives"] == 2
    bands = {row["score_from"]: row for row in summary["score_reliability"]}
    assert bands[30]["confirmed_fraction"] == 0.0 and bands[60]["confirmed_fraction"] == 1.0 and bands[80]["predictions"] == 1
    assert bands[0]["confirmed_fraction"] is None                                       # an empty band reports nothing


def test_uncertain_and_not_observable_annotations_are_left_out_and_labels_can_be_mapped():
    annotations = {"annotations": [ref("u", 10, 14, label="uncertain"), ref("s", 50, 54, label="simulated_symptom")]}
    plain = EVAL.evaluate_trial(document_for([pred("p1", 10, 14), pred("p2", 50, 54)]), annotations, None, LABELS)
    assert plain["by_label"]["freezing_like"]["references"] == 0
    assert plain["by_label"]["freezing_like"]["event_level"]["unscored_predictions"] == 1     # p1 lies in time that does not count
    mapped = EVAL.evaluate_trial(document_for([pred("p1", 10, 14), pred("p2", 50, 54)]), annotations, None, LABELS,
                                 mapping={"freezing_like": {"simulated_symptom"}})
    assert mapped["by_label"]["freezing_like"]["event_level"]["true_positives"] == 1


def test_evaluation_command_reports_nothing_without_reviewed_annotations(tmp_path, capsys):
    document = DET.detect(signals(FREEZE, video=FLOW))
    DSTORE.save_detections(tmp_path, document)
    assert EVAL.main(["--symptom-store", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "no accuracy" in out.lower() and "precision" not in out
