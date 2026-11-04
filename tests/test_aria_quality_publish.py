"""Exercise real publication against temporary data; no training or live assets."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def write_ply(path, label):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = ("ply\nformat ascii 1.0\n"
               f"comment {label}\nelement vertex 1\n"
               "property float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n")
    path.write_text(content, encoding="ascii")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files_under(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("quality_publish_integration", scripts / "publish_aria_quality_choices.py")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)

    env = tmp_path / "environment_reconstruction"
    output = env / "quality_comparison"
    output.mkdir(parents=True)
    monkeypatch.setattr(publisher, "ENV", env)
    scene_id = "upstairs_context"
    original_ply = f"gaussian_submaps/{scene_id}/trained_7000_fullres/splat_7000.ply"
    write_ply(env / original_ply, "original deployed model")
    camera = {"model": "PINHOLE", "width": 1024, "height": 1024,
              "fx": 512.0, "fy": 512.0, "cx": 511.5, "cy": 511.5}
    original_path = env / "scene_playback_data.json"
    original = {"trial_id": "Ntrial3", "provenance": {"graph_uid": "original-world"},
                "scenes": [{"id": scene_id, "ply": original_ply, "camera": camera,
                            "iterations": 7000, "start_s": 78.0, "end_s": 120.0}]}
    write_json(original_path, original)
    validation = [{"name": f"frame_{i:05d}.jpg", "recording_s": 80.123456789 + i * 5}
                  for i in range(5)]
    split_path = output / scene_id / "split.json"
    split = {"scene": scene_id, "world_frame": {"graph_uid": "original-world"},
             "camera": camera, "validation": validation}
    write_json(split_path, split)

    def evaluation(recipe, step, psnr, ssim):
        model = output / scene_id / recipe / f"splat_{step}.ply"
        digest = write_ply(model, f"{recipe} checkpoint {step}")
        views = [{**view, "psnr_db": psnr + i * 0.1, "ssim": ssim, "mae": 0.05}
                 for i, view in enumerate(validation)]
        return {"step": step, "model": str(model), "model_sha256": digest, "views": views,
                "mean_psnr_db": sum(row["psnr_db"] for row in views) / 5,
                "mean_ssim": sum(row["ssim"] for row in views) / 5}

    baseline = evaluation("baseline_7000", 7000, 25.0, 0.8)
    approved = evaluation("long_30000", 15000, 26.0, 0.82)
    unreviewed = evaluation("long_30000", 30000, 29.0, 0.9)
    results = {"trial_id": "Ntrial3", "scenes": [
        {"id": scene_id, "split": str(split_path), "validation": validation,
         "runs": [{"id": "baseline_7000", "evaluations": [baseline]},
                  {"id": "long_30000", "evaluations": [approved, unreviewed]}]}]}
    results_path = output / "results.json"
    write_json(results_path, results)
    reviews = {scene_id: approved["model_sha256"]}
    # Ensure provenance-rejection tests reach those checks, not a metric-gate failure.
    assert publisher.eligible(baseline, approved, validation)
    assert publisher.eligible(baseline, unreviewed, validation)
    return SimpleNamespace(publisher=publisher, env=env, output=output, scene_id=scene_id,
                           original=original, original_path=original_path, results=results,
                           results_path=results_path, split=split, split_path=split_path,
                           baseline=baseline, approved=approved, unreviewed=unreviewed,
                           reviews=reviews, choices_path=output / "playback_choices.json")


def assert_success_is_nondestructive(experiment, before, payload):
    after = files_under(experiment.env)
    relative_choices = experiment.choices_path.relative_to(experiment.env)
    assert set(after) - set(before) == {relative_choices}
    assert all(after[path] == content for path, content in before.items())
    assert json.loads(experiment.choices_path.read_text()) == payload
    assert payload["original_playback_sha256"] == hashlib.sha256(
        experiment.original_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("activate,expected", [(None, "original"), (False, "original"), (True, "compared")])
def test_defaults_to_original_until_activation_is_explicit(experiment, activate, expected):
    before = files_under(experiment.env)
    kwargs = {} if activate is None else {"activate": activate}
    payload = experiment.publisher.publish(experiment.output, experiment.reviews, **kwargs)
    assert payload["default_choice"] == expected
    assert set(payload["scenes"]) == {experiment.scene_id}
    assert_success_is_nondestructive(experiment, before, payload)


@pytest.mark.parametrize("activate", [False, True])
def test_approved_hash_wins_over_higher_scoring_unreviewed_checkpoint(experiment, activate):
    before = files_under(experiment.env)
    payload = experiment.publisher.publish(experiment.output, experiment.reviews, activate=activate)
    selected = payload["scenes"][experiment.scene_id]
    assert experiment.unreviewed["mean_psnr_db"] > experiment.approved["mean_psnr_db"]
    assert experiment.unreviewed["mean_ssim"] > experiment.approved["mean_ssim"]
    assert selected["model_sha256"] == selected["reviewed_model_sha256"] == experiment.approved["model_sha256"]
    assert selected["model_sha256"] != experiment.unreviewed["model_sha256"]
    assert selected["iterations"] == 15000
    assert (experiment.env / selected["ply"]).resolve() == Path(experiment.approved["model"])
    assert not Path(selected["ply"]).is_absolute()
    assert selected["mean_psnr_db"] == experiment.approved["mean_psnr_db"]
    assert selected["validation_views"] == 5
    assert selected["visually_reviewed"] is True
    assert selected["original_ply"] == experiment.original["scenes"][0]["ply"]
    assert selected["camera"] == experiment.original["scenes"][0]["camera"]
    assert (experiment.env / selected["details_url"]).resolve() == experiment.output / "index.html"
    assert_success_is_nondestructive(experiment, before, payload)


@pytest.mark.parametrize("mismatch,message", [
    ("world", "World-frame mismatch"),
    ("camera", "Camera intrinsics mismatch"),
    ("model_bytes", "Model provenance mismatch"),
    ("approved_hash", "Reviewed hash is not an eligible evaluated model"),
    ("outside_experiment", "Model provenance mismatch"),
])
def test_provenance_mismatch_refuses_without_overwriting_existing_choices(experiment, mismatch, message):
    if mismatch == "world":
        experiment.original["provenance"]["graph_uid"] = "different-original-world"
        write_json(experiment.original_path, experiment.original)
    elif mismatch == "camera":
        experiment.original["scenes"][0]["camera"]["fx"] = 600.0
        write_json(experiment.original_path, experiment.original)
    elif mismatch == "model_bytes":
        Path(experiment.approved["model"]).write_bytes(b"changed since evaluation")
    elif mismatch == "approved_hash":
        experiment.reviews[experiment.scene_id] = "0" * 64
    elif mismatch == "outside_experiment":
        outside = experiment.env / "not-an-experiment-model.ply"
        outside.write_bytes(Path(experiment.approved["model"]).read_bytes())
        experiment.approved["model"] = str(outside)
        write_json(experiment.results_path, experiment.results)
    write_json(experiment.choices_path, {"default_choice": "original", "sentinel": "keep this file"})
    before = files_under(experiment.env)
    with pytest.raises(ValueError, match=message):
        experiment.publisher.publish(experiment.output, experiment.reviews, activate=True)
    assert files_under(experiment.env) == before


@pytest.mark.parametrize("has_results", [True, False])
def test_no_approvals_publishes_empty_original_selection(experiment, has_results):
    if not has_results:
        experiment.results["scenes"] = []
        write_json(experiment.results_path, experiment.results)
    before = files_under(experiment.env)
    payload = experiment.publisher.publish(experiment.output, {})
    assert payload["default_choice"] == "original"
    assert payload["scenes"] == {}
    assert_success_is_nondestructive(experiment, before, payload)


def test_activation_without_any_approval_refuses_and_preserves_choices(experiment):
    write_json(experiment.choices_path, {"default_choice": "original", "sentinel": "keep this file"})
    before = files_under(experiment.env)
    with pytest.raises(ValueError, match="activation needs a candidate"):
        experiment.publisher.publish(experiment.output, {}, activate=True)
    assert files_under(experiment.env) == before
