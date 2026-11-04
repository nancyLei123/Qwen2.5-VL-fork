#!/usr/bin/env python3
"""Additive, local-only joint scene experiment. Never replace the submaps."""
import argparse
import html
import json
import os
from pathlib import Path

import numpy as np

from compare_aria_gaussian_quality import (
    DEFAULT_BINARY, base_command, evaluate, execute, make_dataset, save_json,
    sha256, verify_sources,
)
from run_long_trial_reconstruction import check_protected

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "runs/60minutes_1/environment_reconstruction"
OUT = ENV / "joint_first15_experiment"


def validate_splits(splits):
    world, camera = splits[0]["world_frame"], splits[0]["camera"]
    validation_times = [f["recording_s"] for s in splits for f in s["validation"]]
    for split in splits:
        if split["world_frame"] != world or split["camera"] != camera:
            raise ValueError("Joint reconstruction requires the same metric world and camera model")
        for frame in split["training"]:
            if min(abs(frame["recording_s"] - t) for t in validation_times) <= .25:
                raise ValueError("Cross-submap held-out/guard leakage; rebuild training-only colors first")


def prepare():
    check_protected(ENV)
    if (OUT / "split.json").exists():
        result = json.loads((OUT / "split.json").read_text())
        verify_sources(result)
        return result
    data = json.loads((ENV / "scene_playback_data.json").read_text())
    scenes = [s for s in data["scenes"] if 0 <= s["video_start_s"] < s["video_end_s"] <= 900]
    if len(scenes) != 6:
        raise ValueError("Expected the six audited first-15-minute submaps")
    sources = [ENV / "gaussian_submaps" / s["id"] for s in scenes]
    splits = [json.loads((s / "validation_run/split.json").read_text()) for s in sources]
    validate_splits(splits)
    images = OUT / "images"
    images.mkdir(parents=True, exist_ok=True)
    groups = {key: [] for key in ("training", "validation", "guard")}
    provenance, baseline, initializer = {}, [], []
    for source, split in zip(sources, splits):
        verify_sources(split)
        provenance.update(split["source_sha256"])
        protected = source / "validation_run/split.json"
        provenance[str(protected)] = sha256(protected)
        point_file = source / "validation_run/train/sparse/0/points3D.txt"
        provenance[str(point_file)] = sha256(point_file)
        initializer.append(np.loadtxt(point_file, usecols=(1, 2, 3, 4, 5, 6)))
        scores = json.loads((source / "validation_run/trained_7000/evaluation/metrics.json").read_text())
        provenance[scores["model"]] = scores["model_sha256"]
        for key in groups:
            for frame in split[key]:
                name = source.name + "_" + frame["name"]
                target = images / name
                original = source / "images" / frame["name"]
                if not target.exists():
                    target.symlink_to(original)
                groups[key].append({**frame, "name": name, "submap": source.name})
        baseline.extend({**v, "name": source.name + "_" + v["name"], "submap": source.name} for v in scores["views"])
    points = np.concatenate(initializer)
    # Deduplicate identical MPS initializer coordinates, not trained Gaussians.
    # No scale fitting, ICP, pose adjustment or geometry completion is applied.
    _, indices = np.unique(points[:, :3], axis=0, return_index=True)
    points = points[np.sort(indices)]
    manifest = {"camera": splits[0]["camera"]}
    make_dataset(OUT / "train", manifest, groups["training"], OUT, points[:, :3], points[:, 3:6].astype(np.uint8))
    make_dataset(OUT / "eval", manifest, groups["training"] + groups["validation"] + groups["guard"], OUT,
                 points[:, :3], points[:, 3:6].astype(np.uint8))
    result = {"scene": "joint_first15", "source": str(OUT), **groups,
              "camera": splits[0]["camera"], "world_frame": splits[0]["world_frame"],
              "source_sha256": provenance, "initializer_points": len(points), "baseline": baseline,
              "limits": "One static Gaussian scene jointly optimized against all six groups of selected RGB keyframes. Original MPS coordinates/poses fixed. Same 30 held-out RGB views and 0.25s guards as the submaps; initializer colors from their training-only inputs. Shared MPS geometry/all-view support selection: conditional appearance comparison, not independent geometric or clinical validation. No moving-object masks or unseen-surface completion."}
    save_json(OUT / "split.json", result)
    return result


def run_paths(iterations):
    if iterations not in (7000, 30000):
        raise ValueError("Only the preserved 7k baseline and additive 30k recipe are supported")
    suffix = "" if iterations == 7000 else f"_{iterations}"
    return OUT / f"trained_{iterations}", OUT / f"comparison{suffix}.json", OUT / f"progress{suffix}.json"


def run(iterations=7000):
    split = prepare()
    target, comparison_path, progress_path = run_paths(iterations)
    target.mkdir(exist_ok=True)
    model = target / f"splat_{iterations}.ply"
    record_path = target / "run.json"
    if not record_path.exists():
        if model.exists() or (target / "training.log").exists():
            raise ValueError("Incomplete prior run preserved; inspect before retrying")
        command = base_command(DEFAULT_BINARY, OUT / "train", iterations) + [
            "--refine-every", "500", "--densify-from", "500", "--densify-until", str(iterations // 2), "-o", str(model)]
        runtime, _ = execute(command, target / "training.log", progress_path, f"joint_first15_{iterations}", iterations)
        save_json(record_path, {"command": command, "runtime_s": runtime,
                               "binary_sha256": sha256(DEFAULT_BINARY), "model_sha256": sha256(model)})
    record = json.loads(record_path.read_text())
    if sha256(model) != record["model_sha256"] or sha256(DEFAULT_BINARY) != record["binary_sha256"]:
        raise ValueError("Model/trainer changed")
    score = evaluate(DEFAULT_BINARY, OUT, split, model, iterations, target / "evaluation", progress_path)
    comparison = []
    for submap in dict.fromkeys(v["submap"] for v in split["baseline"]):
        old = [v for v in split["baseline"] if v["submap"] == submap]
        new = [v for v in score["views"] if v["name"] in {r["name"] for r in old}]
        comparison.append({"submap": submap, "views": len(new),
                           "original_psnr_db": float(np.mean([v["psnr_db"] for v in old])),
                           "joint_psnr_db": float(np.mean([v["psnr_db"] for v in new])),
                           "original_ssim": float(np.mean([v["ssim"] for v in old])),
                           "joint_ssim": float(np.mean([v["ssim"] for v in new]))})
    verify_sources(split)
    check_protected(ENV)
    save_json(comparison_path, {"status": "evaluated_pending_visual_review", "iterations": iterations, "metrics": score,
              "per_submap": comparison, "training_views": len(split["training"]),
              "limits": split["limits"], "originals_preserved": True})
    save_json(progress_path, {"state": "complete", "stage": "joint training and 30-view appearance comparison"})
    print(json.dumps(comparison, indent=2), flush=True)


def publish_reviewed():
    """Publish an optional experiment, not a replacement or accuracy claim."""
    result = json.loads((OUT / "comparison.json").read_text())
    split = json.loads((OUT / "split.json").read_text())
    verify_sources(split)
    check_protected(ENV)
    score = result["metrics"]
    if len(score["views"]) != 30 or sha256(score["model"]) != score["model_sha256"]:
        raise ValueError("Joint model/evaluation integrity check failed")
    data = json.loads((ENV / "scene_playback_data.json").read_text())
    sources = [s for s in data["scenes"] if s["video_end_s"] <= 900]
    review = {
        "status": "reviewed_experimental", "default": "submaps",
        "graph_uid": split["world_frame"]["graph_uid"],
        "model_sha256": score["model_sha256"],
        "visual_review": {
            "reviewed_views": ["first15_workroom_before_frame_00067", "corridor_walk_frame_00100",
                               "first15_sink_visit_frame_00083", "first15_return_walk_frame_00059",
                               "first15_workroom_after_frame_00063"],
            "finding": "Main corridor/door layout remains recognizable but soft. Close walls lack detail. Desktop after returning shows worse monitor/laptop/phone ghosting than its original submap. Not promoted as sharper or geometrically complete.",
        },
        "scene": {"id": "joint_first15_7000", "title": "Joint first 15 minutes / experimental",
                  "ply": os.path.relpath(score["model"], ENV), "camera": split["camera"],
                  "start_s": min(s["start_s"] for s in sources), "end_s": max(s["end_s"] for s in sources),
                  "video_start_s": 0, "video_end_s": 900, "iterations": 7000,
                  "details_url": "joint_first15_experiment/index.html"}}
    save_json(OUT / "viewer.json", review)
    rows = "".join(f'<tr><td>{html.escape(r["submap"])}</td><td>{r["original_psnr_db"]:.2f}</td><td>{r["joint_psnr_db"]:.2f}</td><td>{r["original_ssim"]:.3f}</td><td>{r["joint_ssim"]:.3f}</td></tr>' for r in result["per_submap"])
    baseline = {v["name"]: v for v in split["baseline"]}
    cards = []
    for view in score["views"]:
        old = baseline[view["name"]]
        figures = "".join(f'<figure><img loading="lazy" src="{html.escape(os.path.relpath(path, OUT), quote=True)}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'
                          for path, caption in [(view["reference"], "Held-out RGB reference"), (old["render"], "Original separate submap"), (view["render"], "Joint first-15-minute model")])
        cards.append(f'<section><h2>{html.escape(view["name"])}</h2><p>VRS capture {view["recording_s"]:.3f} s. PSNR {old["psnr_db"]:.2f} to {view["psnr_db"]:.2f} dB.</p><div class="images">{figures}</div></section>')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>First 15 minutes | Joint reconstruction comparison</title><style>
body{{font-family:"Avenir Next",sans-serif;background:#f3f4ec;color:#182921;margin:0;padding:24px}}main{{max-width:1250px;margin:auto}}section,header{{background:white;border:1px solid #d6e1d9;border-radius:18px;padding:20px;margin-bottom:18px}}h1{{letter-spacing:-.04em}}h2{{font-size:16px;overflow-wrap:anywhere}}p{{line-height:1.6}}a{{color:#187859}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:10px;border-bottom:1px solid #d6e1d9;text-align:left}}.images{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}figure{{margin:0}}img{{width:100%;display:block}}figcaption{{padding-top:6px;font-size:12px}}.notice{{background:#fff4df;padding:14px;border-radius:10px}}.table{{overflow:auto}}@media(max-width:700px){{.images{{grid-template-columns:1fr}}body{{padding:10px}}}}
</style><main><header><a href="../playback.html?assembly=joint">Open synchronized joint-scene experiment</a><h1>One shared scene, not automatically a sharper scene</h1>
<p>The previous viewer switched between six separately trained submaps. This local experiment optimizes <b>one Gaussian model</b> against {len(split["training"])} calibrated RGB keyframes across all six first-15-minute groups. MPS positions, orientations and metric world coordinates are unchanged. This is not an overlay of six finished models or a reconstruction of every video frame.</p>
<p class="notice">Keep both versions. The joint scene connects time intervals without model switches, but desktop detail after returning is worse. Corridor/return appearances have mixed improvements. Moving screens, hands and phones remain a limitation; unseen surfaces are not recovered. A nearby orbit is a novel view, not a guaranteed accurate photograph.</p>
<p>7,000 full-resolution iterations on the local Apple GPU; {score["gaussians"]:,} Gaussians. The same 30 withheld RGB views are compared below. Geometry and MPS poses are shared, so these are conditional appearance checks, not independent geometric or clinical validation. The joint model has one 7,000-step budget versus six separate 7,000-step models; this is an exploratory recipe comparison, not a controlled ablation of fusion.</p>
<div class="table"><table><tr><th>Section</th><th>Original PSNR</th><th>Joint PSNR</th><th>Original SSIM</th><th>Joint SSIM</th></tr>{rows}</table></div><p>Higher is better, but these scores do not establish correct geometry. Original models, source videos and manual labels are preserved. No research data was uploaded.</p></header>{''.join(cards)}</main></html>'''
    (OUT / "index.html").write_text(page)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--publish-reviewed", action="store_true")
    parser.add_argument("--iterations", type=int, choices=(7000, 30000), default=7000)
    args = parser.parse_args()
    if args.publish_reviewed and args.iterations != 7000:
        parser.error("The 30k experiment needs its own visual review before publication")
    publish_reviewed() if args.publish_reviewed else prepare() if args.prepare_only else run(args.iterations)
