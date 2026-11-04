#!/usr/bin/env python3
"""Publish the additive 60minutes_1 pilot after training and appearance checks."""
import argparse
import html
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from build_aria_scene_playback import publish_ui
from build_pd_insighter_trial_selector import FULL_SELECTOR_HTML, build_html, collect_trials
from compare_aria_gaussian_quality import save_json, sha256, verify_sources
from pd_insighter_environment_navigation import VIEW_FILES, refresh_navigation
from run_long_trial_reconstruction import ROOT, check_protected


def scene_record(spec, manifest, summary, base, ply, details):
    if manifest["trial_id"] != base["trial_id"] or summary["trial_id"] != base["trial_id"]:
        raise ValueError("Mixed trial assets")
    if manifest["world_frame"]["graph_uid"] != base["provenance"]["graph_uid"]:
        raise ValueError("Model and playback coordinate frames differ")
    score = summary["metrics"]
    if len(score["views"]) != 5 or not all(math.isfinite(v[k]) for v in score["views"] for k in ("psnr_db", "ssim", "mae")):
        raise ValueError("Five finite held-out appearance checks are required")
    if sha256(ply) != score["model_sha256"]:
        raise ValueError("Published model differs from evaluated model")
    samples = [f["recording_s"] for f in manifest["frames"]["items"]]
    start = max(spec["start_s"], min(samples))
    end = min(spec["end_s"], max(samples) + 1e-6)
    times = np.asarray(base["recording_times"])
    frames = np.flatnonzero((times >= start) & (times < end))
    if not len(frames):
        raise ValueError("No supported model interval on the video clock")
    return {"id": spec["id"], "title": spec["title"], "start_s": start, "end_s": end,
            "video_start_s": int(frames[0]) / base["fps"], "video_end_s": (int(frames[-1]) + 1) / base["fps"],
            "ply": str(ply), "details_url": str(details), "iterations": summary["iterations"],
            "camera": manifest["camera"], "validation_psnr_db": score["mean_psnr_db"],
            "validation_ssim": score["mean_ssim"], "training_start_s": spec["start_s"],
            "training_end_s": spec["end_s"], "gaussians": score["gaussians"]}


def clock(seconds):
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def comparison_image(views, path):
    sheet = Image.new("RGB", (1024, 5 * 542), "#f5f4eb")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(views):
        y = i * 542
        draw.text((10, y + 8), f"VRS {row['recording_s']:.3f}s | Reference", fill="#172b24")
        draw.text((522, y + 8), f"Held-out render | {row['psnr_db']:.2f} dB / SSIM {row['ssim']:.3f}", fill="#172b24")
        for j, key in enumerate(("reference", "render")):
            with Image.open(row[key]) as image:
                sheet.paste(image.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS), (j * 512, y + 30))
    sheet.save(path, quality=92)


def page(title, body, prefix=""):
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;color:#18332b;background:linear-gradient(120deg,#e1e9dd,#f7f5ed);font:17px/1.65 Georgia,serif}}main{{max-width:1280px;margin:auto;padding:32px}}nav,.card{{background:#ffffffed;border:1px solid #d6e1d8;border-radius:24px;padding:26px;margin-bottom:22px}}nav{{display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap}}h1,h2,h3,nav,.button{{font-family:'Trebuchet MS',sans-serif}}h1{{font-size:clamp(32px,4vw,54px);line-height:1.12}}h2{{margin:0 0 12px}}.eyebrow{{letter-spacing:.12em;text-transform:uppercase;font-size:13px}}a{{color:#116648}}.button{{display:inline-block;border-radius:30px;background:#116b4f;color:white;padding:12px 22px;text-decoration:none;font-weight:bold}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}}.grid .card{{padding:20px}}img{{display:block;width:100%;height:auto;border-radius:12px}}.note{{border-left:4px solid #c48b26;padding:8px 20px;background:#fff9e9}}small{{color:#52675d}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:8px;border-bottom:1px solid #d6e1d8}}@media(max-width:800px){{main{{padding:16px}}.grid{{grid-template-columns:1fr}}}}</style></head><body><main>
<nav><strong>PD Insighter / 60minutes_1</strong><div><a href="{prefix}../index.html">Full Session</a> &nbsp; <a href="{prefix}index.html">Environment</a> &nbsp; <a href="{prefix}playback.html">Synchronized 3D</a></div></nav>{body}</main></body></html>'''


def merged_specs(output, include_first15=False, include_first30=False):
    names = ["reconstruction_plan.json"]
    if include_first15 or include_first30:
        names.append("first15_reconstruction_plan.json")
    if include_first30:
        names.append("first30_reconstruction_plan.json")
    scenes = []
    for name in names:
        plan = json.loads((output / name).read_text())
        for key, value in plan["inputs"].items():
            stat = Path(value).stat()
            if {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != plan["source_stats"][key]:
                raise ValueError(f"Reconstruction source changed: {key}")
        scenes.extend(plan["scenes"])
    if len({s["id"] for s in scenes}) != len(scenes):
        raise ValueError("Duplicate scene IDs would replace an existing reconstruction")
    return sorted(scenes, key=lambda s: s["start_s"])


def covered_interval(scenes, start, end):
    intervals = sorted((max(start,s["video_start_s"]),min(end,s["video_end_s"])) for s in scenes)
    total, cursor = 0.0, start
    for a,b in intervals:
        total += max(0,b-max(a,cursor))
        cursor = max(cursor,b)
    return total


def publish(output, include_first15=False, include_first30=False):
    include_first15 = include_first15 or include_first30
    audit = check_protected(output)
    base = json.loads((output / "scene_playback_base.json").read_text())
    specs = merged_specs(output, include_first15, include_first30)
    visual_reviews = {}
    if include_first30:
        visual_reviews = json.loads((output / "first30_visual_review.json").read_text())["scenes"]
    if include_first15:
        backup = output / ("before_first30_extension" if include_first30 else "before_first15_route")
        backup.mkdir(exist_ok=True)
        for name in ("index.html", "playback.html", "scene_playback.js", "scene_playback_data.json", "reconstruction_summary.json", "gaussian_submaps/index.html"):
            original, target = output / name, backup / name
            if original.exists() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
    scenes, cards = [], []
    for spec in specs:
        source = output / "gaussian_submaps" / spec["id"]
        trained = source / "validation_run/trained_7000"
        manifest = json.loads((source / "dataset_manifest.json").read_text())
        summary = json.loads((trained / "training_summary.json").read_text())
        split = json.loads((source / "validation_run/split.json").read_text())
        if include_first30 and spec["id"].startswith("first30_"):
            review = visual_reviews.get(spec["id"], {})
            names = {v["name"] for v in summary["metrics"]["views"]}
            if not review.get("finding") or not review.get("reviewed_views") or not set(review["reviewed_views"]) <= names:
                raise ValueError(f"Visual review required before publishing {spec['id']}")
        verify_sources(split)
        ply = trained / "splat_7000.ply"
        details = source / "review.html"
        scene = scene_record(spec, manifest, summary, base, ply, details)
        if spec["id"] in visual_reviews:
            scene["review_note"] = visual_reviews[spec["id"]]["finding"]
        scene["ply"], scene["details_url"] = os.path.relpath(ply, output), os.path.relpath(details, output)
        scenes.append(scene)
        comparison_image(summary["metrics"]["views"], source / "held_out_comparison.jpg")
        score = summary["metrics"]
        target = scene["start_s"] + 1
        body = f'''<section class="card"><div class="eyebrow">Local reconstruction pilot / 7,000 iterations</div><h1>{html.escape(spec['title'])}</h1>
<p>Dashboard video {clock(scene['video_start_s'])} to {clock(scene['video_end_s'])}. {len(split['training'])} training images, five held-out images and {len(split['guard'])} temporal-guard images. Full 1024 x 1024 calibrated views; Apple GPU.</p>
<p><a class="button" href="../../playback.html?t={target:.9f}">Explore synchronized scene</a></p>
<p>Mean held-out PSNR {score['mean_psnr_db']:.2f} dB; SSIM {score['mean_ssim']:.3f}. {score['gaussians']:,} Gaussians.</p>
<p class="note">Appearance check only. MPS geometry and poses are shared, and initializer support was selected using the exported views before the split. Held-out RGB is excluded from optimization and initializer recoloring. These scores do not establish geometric or clinical accuracy. Browser SH2 colors differ from these offline SH3 renders.</p>
<p>Hand/phone occlusions, moving people and changing screens were not masked in this baseline. Residual ghosts and gaps remain possible. Preserve the original video as the primary evidence.</p><p>{html.escape(scene.get('review_note', ''))}</p></section>
<section class="card"><h2>Reference and held-out render</h2><p>Left: calibrated original image. Right: reconstruction from the matching camera pose, without optimizing on that image.</p><img src="held_out_comparison.jpg" alt="Five matched original and reconstructed views"></section>'''
        details.write_text(page(spec["title"], body, "../../"))
        preview = summary["metrics"]["views"][0 if spec["id"] == "workroom_transition" else 2]["render"]
        preview_url = os.path.relpath(preview, output)
        cards.append(f'''<article class="card"><img src="{preview_url}" alt="Held-out reconstructed {html.escape(spec['title'])}"><h2>{html.escape(spec['title'])}</h2><p>{clock(scene['video_start_s'])} - {clock(scene['video_end_s'])} on the dashboard</p><a class="button" href="playback.html?t={target:.9f}">Open 3D scene</a><p><a href="{scene['details_url']}">Inspect five-view comparison</a></p><small>7,000 iterations / PSNR {score['mean_psnr_db']:.2f} dB / SSIM {score['mean_ssim']:.3f}</small></article>''')
    scenes.sort(key=lambda scene: scene["start_s"])
    if any(a["end_s"] > b["start_s"] for a, b in zip(scenes, scenes[1:])):
        raise ValueError("Overlapping scene intervals require an explicit playback policy")
    base["scenes"] = scenes
    base["video_time_label"] = "Existing dashboard video"
    base["initial_video_time_s"] = scenes[0]["video_start_s"] + 2
    if include_first15:
        base["route_default_end_s"] = 1800 if include_first30 else 900
    base["provenance"]["timing_explanation"] = (
        "Video time is the existing 60minutes_1 dashboard clock. Dashboard frame i maps to source RGB frame i+1265, "
        "then its exact VRS timestamp, online camera time offset and calibrated MPS pose. Twelve image pairs verified the trim. "
        "The fixed-rate video clock differs from the device clock, so no constant-offset approximation is used. "
        "Existing manual and automatic labels are unchanged. Missing poses are not extrapolated.")
    if include_first30:
        original = json.loads((output / "before_first30_extension/scene_playback_data.json").read_text())
        by_id = {s["id"]: s for s in scenes}
        if any(by_id.get(s["id"]) != s for s in original["scenes"]):
            raise ValueError("An existing published scene changed; refusing the additive update")
        if base["poses"] != original["poses"] or base["recording_times"] != original["recording_times"]:
            raise ValueError("Playback poses/timing changed; refusing the additive update")
        base["coverage_summary"] = {"scope_s": 1800, "model_seconds": covered_interval(scenes, 0, 1800),
                                    "definition": "Time with an available trained model, not complete surfaces in every direction."}
    long_review_path = output / "joint_first15_experiment/viewer_30000.json"
    if long_review_path.exists():
        reviewed = json.loads(long_review_path.read_text())
        if (reviewed.get("status") == "reviewed_experimental" and reviewed["graph_uid"] == base["provenance"]["graph_uid"]
                and sha256(output / reviewed["scene"]["ply"]) == reviewed["model_sha256"]):
            base["joint_long_review_url"] = "joint_first15_experiment/viewer_30000.json"
    save_json(output / "scene_playback_data.json", base)
    publish_ui("60minutes_1")
    covered = covered_interval(scenes, 0, base["duration_s"])
    first15_covered = covered_interval(scenes, 0, 900)
    missing_poses = sum(pose is None for pose in base["poses"])
    summary = {"trial_id": "60minutes_1", "status": "first30_extension" if include_first30 else "first15_extension" if include_first15 else "three_submap_pilot", "scenes": scenes,
               "model_coverage_s": covered, "session_duration_s": base["duration_s"],
               "first15_model_coverage_s": first15_covered,
               "first30_model_coverage_s": covered_interval(scenes, 0, 1800),
               "first30_missing_pose_frames": sum(p is None for p in base["poses"][:round(1800*base["fps"])]),
               "first15_missing_pose_frames": sum(p is None for p in base["poses"][:round(900*base["fps"])]),
               "supported_pose_frames": len(base["poses"]) - missing_poses, "missing_pose_frames": missing_poses,
               "provenance": base["provenance"], "protected_sha256": audit["protected_sha256"],
               "limitations": ["Not a whole-session reconstruction", "No moving-object masks in this baseline",
                               "No body reconstruction", "Not clinically or metrically validated geometry"]}
    save_json(output / "reconstruction_summary.json", summary)
    first15_note = f"The first 15 dashboard minutes now have {first15_covered:.2f} / 900 seconds of trained-model coverage. Existing pilot models, including the later stair descent, are preserved. " if include_first15 else ""
    if include_first30:
        first15_note = f"The first 30 dashboard minutes have {summary['first30_model_coverage_s']:.2f} / 1,800 seconds of trained-model coverage. All seven earlier submaps are preserved. "
    body = f'''<section class="card"><div class="eyebrow">Environment / local reconstruction</div><h1>60minutes_1, in context</h1><p>Explore {len(scenes)} motion-centered scenes while following the same Aria video, camera position, facing direction and action labels used in the existing dashboard.</p><p><a class="button" href="playback.html">Open synchronized 3D playback</a></p><p class="note">{first15_note}The local scenes cover {covered:.2f} seconds ({100*covered/base['duration_s']:.1f}%) of the {clock(base['duration_s'])} review session. Model coverage means an available trained submap, not complete surfaces or views in every direction. This is not a complete building or whole-hour reconstruction. Untrained intervals remain explicitly marked on the timeline.</p></section>
<div class="grid">{''.join(cards)}</div>
<section class="card"><h2>How to review</h2><p>Around camera orbits near the recorded camera point and gold viewing rectangle. Nearby route shows the local path: past is teal, future is amber. Aria viewpoint uses the recorded camera pose directly and usually has the best observed coverage. Adjust Distance to move closer; dragging or clicking the route selects time in both views. Use Play or Space to resume.</p><p>Separate submaps share the MPS coordinate system but are not a fused building; the optional joint model connects the first 15 minutes only. These choices do not invent unseen surfaces. Repeated visits can overlap spatially; the route slider distinguishes visits. The recorded Aria video does not rotate when you orbit the reconstruction.</p><h2>Data and safeguards</h2><p>Project Aria RGB frames, online camera calibration, closed-loop MPS poses and semi-dense points initialize OpenSplat. Each separate scene was trained locally on the Apple GPU for 7,000 iterations with five held-out appearance checks. GaussianSplats3D displays the models; the original video remains the observation, not a synthetic replacement.</p><p>All {len(base['poses']):,} dashboard frames have exact source timestamps; {missing_poses} frames lack strict pose support ({missing_poses/base['fps']:.2f} seconds in total). Those gaps are marked, not invented. Manual labels, classification files and source recordings are preserved. Nothing was uploaded.</p><p class="note">The VRS reader found an incomplete final RGB record and rebuilt its read index in memory. The extracted video matches all readable RGB frames, and this dashboard ends before that incomplete tail. No raw recording was repaired or rewritten.</p><h2>Current limits</h2><p>Thin railings, blur, lighting changes, hands, phones and moving people can still produce incomplete surfaces or ghosts. This baseline has no moving-object masks. Areas not seen from enough directions cannot be recovered reliably by training alone. Keep the original video visible when interpreting the reconstruction.</p></section>'''
    (output / "index.html").write_text(page("60minutes_1 | Environment", body))
    (output / "gaussian_submaps/index.html").write_text(page("60minutes_1 | Submaps", f'<section class="card"><h1>{len(scenes)} reconstructed submaps</h1><p><a class="button" href="../index.html">Choose a scene and inspect its results</a></p></section>', "../"))
    backup = output / "ui_before_environment"
    backup.mkdir(exist_ok=True)
    for filename in VIEW_FILES:
        original = output.parent / filename
        if original.exists() and not (backup / filename).exists():
            shutil.copy2(original, backup / filename)
    refreshed = refresh_navigation(["60minutes_1"])
    if FULL_SELECTOR_HTML.exists() and not (backup / FULL_SELECTOR_HTML.name).exists():
        shutil.copy2(FULL_SELECTOR_HTML, backup / FULL_SELECTOR_HTML.name)
    FULL_SELECTOR_HTML.write_text(build_html(
        collect_trials(), page_title="PD-Insighter Trial Selector", hero_title="PD-Insighter Trial Selector",
        hero_copy="Select a trial and it will open the canonical full-session clinician review. Newly analyzed trials will appear here automatically when the selector is rebuilt.",
        selector_heading="Full-Session Review", selector_copy="Choose a processed trial to open its full-session review directly.",
        target_key="full_dashboard_href"))
    check_protected(output)
    print(json.dumps({"page": str(output / "playback.html"), "model_coverage_s": covered,
                      "missing_pose_frames": missing_poses, "navigation_updated": [str(p) for p in refreshed]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-first15", action="store_true")
    parser.add_argument("--include-first30", action="store_true")
    args = parser.parse_args()
    publish(ROOT / "runs/60minutes_1/environment_reconstruction", args.include_first15, args.include_first30)
