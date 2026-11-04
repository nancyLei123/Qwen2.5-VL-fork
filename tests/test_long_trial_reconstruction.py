import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_long_trial_reconstruction import nearest_indices, verify_alignment, check_protected, SPECS, FIRST15_SPECS, FIRST30_SPECS, batch_files
from publish_long_trial_reconstruction import scene_record, covered_interval, merged_specs
import build_aria_scene_playback as playback
from compare_aria_gaussian_quality import sha256
sys.path.pop(0)


def mapping():
    timestamps = np.array([100, 200, 299, 397, 496, 596], dtype=np.int64)
    metadata = {"written_frames": 6, "fps": 30}
    video = {"frames": 3, "fps": 30}
    audit = {"source_trim_frames": 2, "fps": 30,
             "frame_checks": [{"best_delta_frames": 0, "mean_rgb8_difference_by_delta": {"0": 1}} for _ in range(3)]}
    return timestamps, metadata, video, audit


def test_mapping_keeps_each_vrs_timestamp_not_a_constant_offset():
    values = mapping()
    np.testing.assert_array_equal(verify_alignment(*values), [299, 397, 496])
    assert len(np.unique(np.diff(verify_alignment(*values)))) > 1


@pytest.mark.parametrize("failure", ["count", "order", "fps", "trim", "end", "alignment", "pixels"])
def test_unverified_clock_fails_closed(failure):
    timestamps, metadata, video, audit = mapping()
    if failure == "count": metadata["written_frames"] = 7
    if failure == "order": timestamps[3] = 299
    if failure == "fps": video["fps"] = 29.97
    if failure == "trim": audit["source_trim_frames"] = -1
    if failure == "end": video["frames"] = 7
    if failure == "alignment": audit["frame_checks"][0]["best_delta_frames"] = 1
    if failure == "pixels": audit["frame_checks"][0]["mean_rgb8_difference_by_delta"]["0"] = 9
    with pytest.raises(ValueError): verify_alignment(timestamps, metadata, video, audit)


def test_nearest_calibration_handles_edges_and_ties():
    np.testing.assert_array_equal(nearest_indices(np.array([10, 30, 50]), np.array([0, 20, 21, 55])), [0, 0, 1, 2])


def scene_inputs(tmp_path):
    ply = tmp_path / "model.ply"
    ply.write_text("test model")
    spec = {"id": "room", "title": "Room", "start_s": 40, "end_s": 44}
    manifest = {"trial_id": "60minutes_1", "world_frame": {"graph_uid": "graph"},
                "frames": {"items": [{"recording_s": 41}, {"recording_s": 43}]}, "camera": {"width": 1024}}
    summary = {"trial_id": "60minutes_1", "iterations": 7000, "metrics": {
        "views": [{"psnr_db": 25, "ssim": .8, "mae": .05} for _ in range(5)],
        "model_sha256": sha256(ply), "mean_psnr_db": 25, "mean_ssim": .8, "gaussians": 100}}
    base = {"trial_id": "60minutes_1", "fps": 2, "recording_times": [40, 40.5, 41, 41.5, 42, 42.5, 43, 43.5],
            "provenance": {"graph_uid": "graph"}}
    return spec, manifest, summary, base, ply, tmp_path / "review.html"


def test_published_interval_never_claims_unsampled_endpoints(tmp_path):
    scene = scene_record(*scene_inputs(tmp_path))
    assert scene["start_s"] == 41
    assert scene["end_s"] == pytest.approx(43.000001)
    assert scene["video_start_s"] == 1
    assert scene["video_end_s"] == 3.5


@pytest.mark.parametrize("failure", ["trial", "graph", "nan", "views", "hash", "interval"])
def test_publisher_rejects_unverified_model(tmp_path, failure):
    spec, manifest, summary, base, ply, details = scene_inputs(tmp_path)
    if failure == "trial": manifest["trial_id"] = "Ntrial3"
    if failure == "graph": manifest["world_frame"]["graph_uid"] = "another"
    if failure == "nan": summary["metrics"]["views"][0]["psnr_db"] = float("nan")
    if failure == "views": summary["metrics"]["views"].pop()
    if failure == "hash": ply.write_text("changed model")
    if failure == "interval": spec["start_s"] = 100
    with pytest.raises(ValueError): scene_record(spec, manifest, summary, base, ply, details)


def test_label_change_is_never_overwritten(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text("original")
    (tmp_path / "video_alignment_audit.json").write_text(json.dumps({"protected_sha256": {str(labels): sha256(labels)}}))
    check_protected(tmp_path)
    labels.write_text("user's new labels")
    with pytest.raises(ValueError, match="Protected"): check_protected(tmp_path)
    assert labels.read_text() == "user's new labels"


def test_new_trial_ui_publish_does_not_touch_old_trial_or_data(tmp_path, monkeypatch):
    monkeypatch.setattr(playback, "ROOT", tmp_path)
    output = tmp_path / "runs/60minutes_1/environment_reconstruction"
    output.mkdir(parents=True)
    payload = output / "scene_playback_data.json"
    payload.write_text('{"trial_id":"60minutes_1"}')
    old = tmp_path / "runs/Ntrial3/environment_reconstruction/playback.html"
    old.parent.mkdir(parents=True)
    old.write_text("original Ntrial3")
    page = playback.publish_ui("60minutes_1")
    assert "60minutes_1 | Synchronized 3D Playback" in page.read_text()
    assert old.read_text() == "original Ntrial3"
    assert payload.read_text() == '{"trial_id":"60minutes_1"}'
    with pytest.raises(ValueError, match="Invalid"): playback.publish_ui("../Ntrial3")


def test_additive_plans_keep_pilot_and_new_scenes(tmp_path):
    pilot = {"inputs": {}, "source_stats": {}, "scenes": [{"id":"old", "start_s":460}]}
    extension = {**pilot, "scenes": [{"id":"new", "start_s":0}]}
    (tmp_path / "reconstruction_plan.json").write_text(json.dumps(pilot))
    (tmp_path / "first15_reconstruction_plan.json").write_text(json.dumps(extension))
    assert [s["id"] for s in merged_specs(tmp_path, True)] == ["new", "old"]
    extension["scenes"][0]["id"] = "old"
    (tmp_path / "first15_reconstruction_plan.json").write_text(json.dumps(extension))
    with pytest.raises(ValueError, match="Duplicate"): merged_specs(tmp_path, True)


def test_coverage_union_never_double_counts_or_includes_later_scene():
    intervals = [(0,460),(460,490),(490,530),(530,570),(570,605),(605,900),(1365,1402)]
    scenes = [{"video_start_s":a,"video_end_s":b} for a,b in intervals]
    assert covered_interval(scenes,0,900) == 900
    assert covered_interval(scenes+scenes,0,3400) == 937
    scenes[0]["video_start_s"] = .1
    assert covered_interval(scenes,0,900) == pytest.approx(899.9)


def test_route_controls_exist_and_do_not_replace_annotation_controls():
    text = (Path(__file__).resolve().parents[1] / "scripts/templates/pd_insighter_scene_playback.html").read_text()
    for control in ("routeScope","routeSeek","routeWindow","routeClock","routeBack","routeNext","routeMap","seek","automatic","reference"):
        assert text.count(f'id="{control}"') == 1
    assert text.index('id="routeMap"') < text.index('id="seek"')


def test_first30_plan_fills_gaps_without_replacing_existing_intervals():
    specs = sorted([*SPECS, *FIRST15_SPECS, *FIRST30_SPECS], key=lambda s: s["start"])
    assert len({s["id"] for s in specs}) == len(specs) == 15
    assert specs[0]["start"] == 0 and specs[-1]["end"] == 1800
    for a, b in zip(specs, specs[1:]):
        assert a["end"] == b["start"]
    assert batch_files("first30") == ("first30_reconstruction_plan.json", "first30_progress.json")
    with pytest.raises(ValueError): batch_files("all")


def test_first30_publisher_includes_all_previous_plans(tmp_path):
    for name, ident, start in [("reconstruction_plan.json", "pilot", 10), ("first15_reconstruction_plan.json", "before", 0), ("first30_reconstruction_plan.json", "later", 900)]:
        (tmp_path / name).write_text(json.dumps({"inputs": {}, "source_stats": {}, "scenes": [{"id": ident, "start_s": start}]}))
    assert [s["id"] for s in merged_specs(tmp_path, include_first30=True)] == ["before", "pilot", "later"]
