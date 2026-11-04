"""The report uses only stdlib and runner-shaped fixtures, never the ML runner."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_gaussian_quality_report.py"


@pytest.fixture
def report():
    spec = importlib.util.spec_from_file_location("gaussian_quality_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_payload(output):
    html = (output / "index.html").read_text()
    match = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(match[1])


def write_results(output, scenes, **metadata):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "results.json"
    path.write_text(json.dumps({"trial_id": "Ntrial3", "updated_at": "2026-10-22T12:00:00Z",
                                "scenes": scenes, **metadata}))
    return path


def make_scene(output, name="upstairs_context", frame_names=None):
    frame_names = frame_names or [f"frame_{i:05d}.jpg" for i in range(5)]
    source = output.parent / "gaussian_submaps" / name
    (source / "images").mkdir(parents=True)
    frames = []
    for i, frame_name in enumerate(frame_names):
        (source / "images" / frame_name).write_bytes(b"original image fixture")
        frames.append({"name": frame_name, "recording_s": 77.983393163 + i,
                       "requested_s": -1, "capture_timestamp_ns": 1416312978900 + i})
    split_path = output / name / "split.json"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(json.dumps({"source": str(source), "validation": frames}))
    return {"id": name, "split": str(split_path), "camera": {"width": 1024, "height": 1024},
            "training_count": 100, "guard_count": 10, "validation_count": 5,
            "validation": frames, "runs": []}


def add_run(scene, output, run_id="baseline_7000", step=7000, count=5, psnr=20.0):
    folder = output / scene["id"] / run_id / f"eval_{step}"
    folder.mkdir(parents=True, exist_ok=True)
    views = []
    source = json.loads(Path(scene["split"]).read_text())["source"]
    for i, frame in enumerate(scene["validation"][:count]):
        render = folder / (Path(frame["name"]).stem + ".png")
        render.write_bytes(b"render fixture")
        views.append({"name": frame["name"], "recording_s": frame["recording_s"],
                      "reference": str(Path(source) / "images" / frame["name"]),
                      "render": str(render), "psnr_db": psnr + i, "ssim": 0.8, "mae": 0.05})
    evaluation = {"step": step, "views": views, "gaussians": 12345,
                  "mean_psnr_db": psnr + 2, "mean_ssim": 0.8}
    run = next((r for r in scene["runs"] if r["id"] == run_id), None)
    if run is None:
        run = {"id": run_id, "runtime_s": 321.25, "evaluations": []}
        scene["runs"].append(run)
    run["evaluations"].append(evaluation)
    return evaluation


@pytest.mark.parametrize("content", [None, "{unfinished", "[]", "null"])
def test_missing_or_unreadable_results_still_builds_pending_report(report, tmp_path, content):
    output = tmp_path / "experiment"
    output.mkdir()
    if content is not None:
        (output / "results.json").write_text(content)
    paths = report.build(output)
    assert paths == (output / "index.html", output / "report.md")
    data = read_payload(output)
    assert len(data["scenes"]) == 3
    assert data["warnings"]
    for scene in data["scenes"]:
        assert len(scene["checkpoints"]) == 4
        assert all(cp["count"] == 0 and not cp["complete"] for cp in scene["checkpoints"])
        assert all(cp["means"]["psnr_db"] is None for cp in scene["checkpoints"])
    assert "Pending" in paths[1].read_text()
    if content is not None:
        assert (output / "results.json").read_text() == content


def test_prepared_scene_references_original_images_before_runs_exist(report, tmp_path):
    output = tmp_path / "experiment"
    scene = make_scene(output)
    write_results(output, [scene])
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report.build(output)
    data = read_payload(output)["scenes"][0]
    assert len(data["views"]) == 5
    for view in data["views"]:
        assert view["reference"].startswith("../gaussian_submaps/upstairs_context/images/")
        assert (output / unquote(view["reference"])).is_file()
    assert len(data["checkpoints"]) == 4
    assert all(p.read_bytes() == content for p, content in before.items())
    assert set(p for p in tmp_path.rglob("*") if p.is_file()) - set(before) == {
        output / "index.html", output / "report.md"}


def test_all_checkpoints_match_by_name_and_keep_exact_vrs_times(report, tmp_path):
    output = tmp_path / "experiment"
    scene = make_scene(output)
    add_run(scene, output)
    candidate = add_run(scene, output, "long_30000", 7000, psnr=21.0)
    candidate["views"].reverse()
    add_run(scene, output, "long_30000", 15000, psnr=22.0)
    add_run(scene, output, "long_30000", 30000, psnr=23.0)
    write_results(output, [scene])
    report.build(output)
    built = read_payload(output)["scenes"][0]
    assert [cp["means"]["psnr_db"] for cp in built["checkpoints"]] == [22, 23, 24, 25]
    assert [cp["delta"]["psnr_db"] for cp in built["checkpoints"]] == [0, 1, 2, 3]
    assert all(cp["complete"] and cp["count"] == 5 for cp in built["checkpoints"])
    assert built["checkpoints"][1]["views"][0]["psnr_db"] == 21
    assert built["views"][0]["capture_timestamp_ns"] == "1416312978900"
    for path in (output / "index.html", output / "report.md"):
        text = path.read_text()
        assert "77.983393163" in text
        assert "1416312978900" in text
        assert "25.000" in text
        assert "+3.000" in text or "+3\\.000" in text
        assert "no repeated-seed study" in text
        assert "One run per recipe; repeat variability is not estimated" in text
        assert "seed 42" in text
        assert "deterministic GPU execution has not been established" in text
        assert "one seed per recipe" not in text
        assert "not an independent test" in text
        assert "shared MPS" in text
        assert "training-color leakage" in text
        assert "scheduling-confounded" in text
        assert "clinical accuracy" in text
        assert "native OpenSplat SH3" in text
        assert "SH2" in text
        assert "not equivalent pixel outputs" in text
        assert "never the last iteration automatically" in text
        assert "Retrained 7k baseline" in text
        assert "NOT the original deployed model" in text
        assert "single-view PSNR is not directly comparable" in text
        assert "qualitative / training-view" in text


@pytest.mark.parametrize("issue", ["partial", "time", "duplicate", "nonfinite", "no_baseline"])
def test_no_partial_or_mismatched_view_comparison(report, tmp_path, issue):
    output = tmp_path / "experiment"
    scene = make_scene(output)
    if issue != "no_baseline":
        add_run(scene, output)
    candidate = add_run(scene, output, "long_30000", 15000, count=3 if issue == "partial" else 5)
    if issue == "time":
        candidate["views"][0]["recording_s"] += 0.001
    if issue == "duplicate":
        candidate["views"].append(candidate["views"][0].copy())
    if issue == "nonfinite":
        candidate["views"][0]["psnr_db"] = float("nan")
    write_results(output, [scene])
    report.build(output)
    cp = read_payload(output)["scenes"][0]["checkpoints"][2]
    assert all(value is None for value in cp["delta"].values())
    if issue != "no_baseline":
        assert not cp["complete"]
        assert cp["means"]["psnr_db"] is None
    if issue in {"time", "duplicate"}:
        assert cp["views"][0]["render"] is None
        assert "withheld" in cp["views"][0]["status"]


def test_missing_images_do_not_erase_valid_metrics(report, tmp_path):
    output = tmp_path / "experiment"
    scene = make_scene(output)
    score = add_run(scene, output)
    Path(score["views"][0]["render"]).unlink()
    write_results(output, [scene])
    report.build(output)
    checkpoint = read_payload(output)["scenes"][0]["checkpoints"][0]
    assert checkpoint["complete"]
    assert checkpoint["views"][0]["render"] is None
    assert checkpoint["views"][1]["render"] is not None


def test_relative_urls_and_html_escaping(report, tmp_path):
    output = tmp_path / "experiment"
    names = ['view #1 & "quoted" % ? <tag> \u96ea.jpg'] + [f"view_{i}.jpg" for i in range(1, 5)]
    scene = make_scene(output, frame_names=names)
    evaluation = add_run(scene, output)
    evaluation["views"][0]["render"] = str(Path(evaluation["views"][0]["render"]).relative_to(output))
    injection = '</script><script>alert("bad")</script><img src="https://invalid.example/x">'
    write_results(output, [scene], trial_id=injection, updated_at=injection)
    report.build(output)
    html = (output / "index.html").read_text()
    data = read_payload(output)
    assert data["trial"] == injection
    assert injection not in html
    assert "&lt;/script&gt;" in html
    assert "\\u003c/script\\u003e" in html
    assert names[0] not in html
    assert "&lt;tag&gt;" in html
    assert "<script>alert" not in (output / "report.md").read_text()
    for url in (data["scenes"][0]["views"][0]["reference"],
                data["scenes"][0]["checkpoints"][0]["views"][0]["render"]):
        parsed = urlsplit(url)
        assert not parsed.scheme and not parsed.netloc and not parsed.fragment and not parsed.query
        assert "%23" in url and "%22" in url and "%25" in url and "%3F" in url
        assert "%E9%9B%AA" in url
        assert (output / unquote(url)).is_file()
    assert "connect-src 'none'" in html
    assert '<script src=' not in html
    assert "fetch(" not in html
    assert "innerHTML" not in html
    assert 'href="../playback.html"' in html


@pytest.mark.parametrize("raw", ["https://example.org/a.png", "//example.org/a.png",
                                 "javascript:alert(1)", "data:image/png;base64,AA", "file:///tmp/a.png",
                                 "../outside/../../secret.png", "bad\x00.png", "..\\secret.png"])
def test_rejects_nonlocal_or_escaping_image_paths(report, tmp_path, raw):
    output = tmp_path / "environment" / "experiment"
    output.mkdir(parents=True)
    assert report._image_url(raw, output) is None


def test_symlink_escape_and_non_image_are_not_exposed(report, tmp_path):
    output = tmp_path / "environment" / "experiment"
    output.mkdir(parents=True)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"private")
    (output / "link.png").symlink_to(outside)
    (output / "page.html").write_text("<script>bad()</script>")
    assert report._image_url("link.png", output) is None
    assert report._image_url("page.html", output) is None
    assert report._local_path(outside, output, output) is None


def test_cli_accepts_output_and_needs_no_runner_import(tmp_path):
    result = subprocess.run([sys.executable, "-I", str(SCRIPT), "--output", str(tmp_path)],
                            capture_output=True, text=True, check=True)
    assert str(tmp_path / "index.html") in result.stdout
    assert (tmp_path / "report.md").is_file()
    assert "from compare_aria_gaussian_quality" not in SCRIPT.read_text()


def test_rebuild_refreshes_snapshot_without_changing_results(report, tmp_path):
    report.build(tmp_path)
    path = write_results(tmp_path, [], updated_at="later snapshot")
    original = path.read_bytes()
    report.build(tmp_path)
    assert read_payload(tmp_path)["updated_at"] == "later snapshot"
    assert path.read_bytes() == original


def test_progress_is_actual_stage_snapshot_not_overall_percentage(report, tmp_path):
    progress = {"stage": "upstairs_context/long_30000 <stage>", "state": "running",
                "step": 8000, "total_steps": 30000, "elapsed_s": 173.1917679309845}
    path = tmp_path / "progress.json"
    path.write_text(json.dumps(progress))
    before = path.read_bytes()
    report.build(tmp_path)
    data = read_payload(tmp_path)
    assert "State: running" in data["progress"]
    assert "Stage steps: 8000 / 30000" in data["progress"]
    assert "Stage elapsed: 173.2 s" in data["progress"]
    assert path.read_bytes() == before
    for filename in ("index.html", "report.md"):
        text = (tmp_path / filename).read_text()
        assert "8000 / 30000" in text
        assert "&lt;stage&gt;" in text
        assert "no overall percentage is inferred" in text
        assert "26.7%" not in text


def test_completed_progress_does_not_imply_an_unpublished_stage(report, tmp_path):
    (tmp_path / "progress.json").write_text(json.dumps({"state": "complete"}))
    report.build(tmp_path)
    progress = read_payload(tmp_path)["progress"]
    assert "State: complete" in progress
    assert "Stage: Not published" not in progress


def test_absent_findings_makes_no_automatic_decision(report, tmp_path):
    report.build(tmp_path)
    assert read_payload(tmp_path)["findings"] == {}
    for filename in ("index.html", "report.md"):
        text = (tmp_path / filename).read_text()
        assert "No decision has been published" in text
        assert "Keep the original default if the benefit is unclear" in text
        assert "No visual conclusions are inferred" in text
        assert text.index("Decision summary") < text.index("Runner progress")
    html = (tmp_path / "index.html").read_text()
    assert '<details class="card method">' in html
    assert '<details class="card method" open>' not in html


@pytest.mark.parametrize("scene_format", ["list", "object"])
def test_optional_findings_render_safely_and_never_promote(report, tmp_path, scene_format):
    scene = {"id": "upstairs_context", "decision": "Keep original", "summary": "Visual review pending."}
    scenes = [scene] if scene_format == "list" else {"upstairs_context": scene}
    findings = {"decision": "Do not change the deployed model", "summary": '<script>bad()</script> & "quoted"',
                "next_step": "Inspect all five views [not a link](https://invalid.example)",
                "updated_at": "2026-10-22T12:00:00Z", "scenes": scenes,
                "default_choice": "compared", "model": "unreviewed.ply"}
    path = tmp_path / "findings.json"
    path.write_text(json.dumps(findings))
    before = path.read_bytes()
    report.build(tmp_path)
    assert path.read_bytes() == before
    assert set(p.name for p in tmp_path.iterdir()) == {"findings.json", "index.html", "report.md"}
    payload = read_payload(tmp_path)
    assert "default_choice" not in payload["findings"]
    assert not any(cp["complete"] for s in payload["scenes"] for cp in s["checkpoints"])
    for filename in ("index.html", "report.md"):
        text = (tmp_path / filename).read_text()
        assert "Do not change the deployed model" in text
        assert "Upstairs: Keep original Visual review pending." in text
        assert "not independently verified" in text
        assert "does not change playback or promote a model" in text
        assert "<script>bad()" not in text
        assert "&lt;script&gt;" in text
        assert text.index("Do not change the deployed model") < text.index("Runner progress")
    assert "\\[not a link\\]" in (tmp_path / "report.md").read_text()


@pytest.mark.parametrize("content", ["{unfinished", "[]", '{"summary": null, "scenes": [null, 42]}',
                                     '{"scenes": [{"id": "stairs", "summary": " ", "decision": ""}]}'])
def test_invalid_findings_do_not_break_report_or_invent_conclusions(report, tmp_path, content):
    (tmp_path / "findings.json").write_text(content)
    report.build(tmp_path)
    assert read_payload(tmp_path)["findings"] == {}
    assert "No decision has been published" in (tmp_path / "index.html").read_text()


def write_deployed_results(output, scenes):
    path = output / "deployed_reference/results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"label": "Qualitative / non-held-out deployed reference",
                                "eligible_for_promotion": False, "scenes": scenes}))
    return path


def test_absent_deployed_results_is_optional(report, tmp_path):
    report.build(tmp_path)
    assert read_payload(tmp_path)["deployed_reference"] is None
    assert 'id="deployed-heading"' not in (tmp_path / "index.html").read_text()
    assert not (tmp_path / "deployed_reference").exists()


@pytest.mark.parametrize("relative", [False, True])
def test_deployed_contact_sheets_and_replay_status_are_separate_from_ranking(report, tmp_path, relative):
    sheet = tmp_path / 'deployed_reference/upstairs_context/comparison #1 "quoted" % ? \u96ea.png'
    sheet.parent.mkdir(parents=True)
    sheet.write_bytes(b"contact sheet fixture")
    contact_sheet = str(sheet.relative_to(tmp_path) if relative else sheet)
    rows = [
        {"id": "upstairs_context", "contact_sheet": contact_sheet,
         "replay_validation": {"status": "exact", "exact_pixels_equal": True},
         "deployed_reference": {"mean_psnr_db": 1234567.89, "mean_ssim": 0.99999}},
        {"id": "sitting_context", "replay_validation": {"status": "within_one_code_value"}},
        {"id": "corridor_walking", "replay_validation": {"status": "mismatch"}},
    ]
    path = write_deployed_results(tmp_path, rows)
    original_results, original_sheet = path.read_bytes(), sheet.read_bytes()
    report.build(tmp_path)
    assert path.read_bytes() == original_results and sheet.read_bytes() == original_sheet
    data = read_payload(tmp_path)
    deployed = data["deployed_reference"]["scenes"]
    url = deployed[0]["contact_sheet"]
    assert (tmp_path / unquote(url)).resolve() == sheet
    assert "%23" in url and "%22" in url and "%25" in url and "%3F" in url
    assert "%E9%9B%AA" in url
    assert not urlsplit(url).scheme and not urlsplit(url).query and not urlsplit(url).fragment
    assert all(not cp["complete"] for scene in data["scenes"] for cp in scene["checkpoints"])
    for filename in ("index.html", "report.md"):
        text = (tmp_path / filename).read_text()
        assert text.index("Decision summary") < text.index("Deployed original / visual reference") < text.index("Runner progress")
        assert "Qualitative / non-held-out" in text
        assert "trained on most of these five views" in text
        assert "not a fair held-out model ranking or promotion criterion" in text
        assert "Exact replay: identical decoded pixels" in text
        assert "Replay within one 8-bit code value" in text
        assert "Replay mismatch: diagnostic only" in text
        assert "Contact sheet not available" in text
        assert "1234567.89" not in text
        assert url in text


@pytest.mark.parametrize("unsafe", ["https://example.org/contact.png", "//example.org/contact.png",
                                    "javascript:alert(1)", "data:image/png;base64,AA", "missing.png",
                                    "page.html", "escape.png"])
def test_deployed_contact_sheet_uses_existing_local_path_allowlist(report, tmp_path, unsafe):
    output = tmp_path / "environment" / "experiment"
    output.mkdir(parents=True)
    (output / "page.html").write_text("<script>unsafe()</script>")
    outside = tmp_path / "private.png"
    outside.write_bytes(b"outside allowed source tree")
    (output / "escape.png").symlink_to(outside)
    write_deployed_results(output, [{"id": "upstairs_context", "contact_sheet": unsafe}])
    report.build(output)
    scene = read_payload(output)["deployed_reference"]["scenes"][0]
    assert scene["contact_sheet"] is None
    assert scene["replay_status"] == "Replay status not yet published."
    for filename in ("index.html", "report.md"):
        assert "Contact sheet not available" in (output / filename).read_text()


def test_deployed_metadata_is_html_escaped_and_unknown_status_is_not_success(report, tmp_path):
    injection = '</script><script>alert("bad")</script>'
    write_deployed_results(tmp_path, [{"id": injection, "replay_validation": {"status": injection}}])
    report.build(tmp_path)
    for filename in ("index.html", "report.md"):
        text = (tmp_path / filename).read_text()
        assert injection not in text
        assert "&lt;/script&gt;" in text
        assert "Unrecognized replay status" in text
        assert "Exact replay: identical decoded pixels" not in text


@pytest.mark.parametrize("content", ['{"scenes": []}', "{unfinished", '{"scenes": [null, 42]}'])
def test_partial_deployed_results_do_not_claim_replay_success(report, tmp_path, content):
    path = tmp_path / "deployed_reference/results.json"
    path.parent.mkdir()
    path.write_text(content)
    report.build(tmp_path)
    assert read_payload(tmp_path)["deployed_reference"]["scenes"] == []
    assert "No scene checks published yet" in (tmp_path / "index.html").read_text()
    assert path.read_text() == content
