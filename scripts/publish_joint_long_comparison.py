#!/usr/bin/env python3
"""Compare the preserved 7k scene with the additive 30k recipe; publish after review."""
import argparse
import html
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from compare_aria_gaussian_quality import save_json, sha256, verify_sources
from joint_first15_reconstruction import ENV, OUT
from run_long_trial_reconstruction import check_protected


def validated_comparison(old, new, split):
    names = {v["name"] for v in split["validation"]}
    if len(names) != 30:
        raise ValueError("Expected the same 30 held-out views")
    for result in (old, new):
        views = result["metrics"]["views"]
        if len(views) != 30 or {v["name"] for v in views} != names:
            raise ValueError("Held-out comparison views changed")
        if not all(np.isfinite(v[k]) for v in views for k in ("psnr_db", "ssim", "mae")):
            raise ValueError("Nonfinite appearance metric")
    baseline = {v["name"]: v for v in old["metrics"]["views"]}
    for view in new["metrics"]["views"]:
        previous = baseline[view["name"]]
        if view["reference"] != previous["reference"] or view["recording_s"] != previous["recording_s"]:
            raise ValueError("Reference image or evaluation timestamp changed")
    return baseline


def build(review_file=None):
    check_protected(ENV)
    split = json.loads((OUT / "split.json").read_text())
    verify_sources(split)
    old = json.loads((OUT / "comparison.json").read_text())
    new = json.loads((OUT / "comparison_30000.json").read_text())
    baseline = validated_comparison(old, new, split)
    for result in (old, new):
        score = result["metrics"]
        if sha256(score["model"]) != score["model_sha256"]:
            raise ValueError("Evaluated model changed")
    report = OUT / "longer_training_comparison"
    report.mkdir(exist_ok=True)
    summary, rows, cards = [], [], []
    originals = {v["name"]: v for v in split["baseline"]}
    for name in dict.fromkeys(v["submap"] for v in split["baseline"]):
        views = [v for v in new["metrics"]["views"] if originals[v["name"]]["submap"] == name]
        item = {"submap": name, "views": len(views)}
        for label, values in (("original", [originals[v["name"]] for v in views]),
                              ("joint_7000", [baseline[v["name"]] for v in views]),
                              ("joint_30000", views)):
            for metric in ("psnr_db", "ssim"):
                item[f"{label}_{metric}"] = float(np.mean([v[metric] for v in values]))
        summary.append(item)
        rows.append(f'<tr><td>{html.escape(name)}</td>' + ''.join(f'<td>{item[f"{label}_psnr_db"]:.2f} / {item[f"{label}_ssim"]:.3f}</td>' for label in ("original", "joint_7000", "joint_30000")) + '</tr>')
    for view in new["metrics"]["views"]:
        images = [(view["reference"], "Held-out RGB reference"), (baseline[view["name"]]["render"], "Joint 7,000 steps"), (view["render"], "Joint 30,000 steps")]
        figures = ''.join(f'<figure><img loading="lazy" src="{html.escape(os.path.relpath(p, report), quote=True)}" alt="{caption}"><figcaption>{caption}</figcaption></figure>' for p, caption in images)
        cards.append(f'<section><h2>{html.escape(view["name"])}</h2><p>PSNR {baseline[view["name"]]["psnr_db"]:.2f} to {view["psnr_db"]:.2f} dB; SSIM {baseline[view["name"]]["ssim"]:.3f} to {view["ssim"]:.3f}.</p><div class="images">{figures}</div></section>')
    # One mid-range withheld view per group for a compact visual audit, not cherry-picked best views.
    selected = [new["metrics"]["views"][i] for i in (2, 7, 12, 17, 22, 27)]
    sheet = Image.new("RGB", (1200, len(selected) * 430), "#f5f4eb")
    draw = ImageDraw.Draw(sheet)
    for i, view in enumerate(selected):
        draw.text((8, i * 430 + 8), view["name"], fill="#17352a")
        for j, path in enumerate((view["reference"], baseline[view["name"]]["render"], view["render"])):
            with Image.open(path) as image:
                sheet.paste(image.convert("RGB").resize((400, 400), Image.Resampling.LANCZOS), (j * 400, i * 430 + 30))
    sheet.save(report / "six_section_comparison.jpg", quality=94)
    review = json.loads(Path(review_file).read_text()) if review_file else None
    if review and (not review.get("finding") or not review.get("reviewed_views") or not set(review["reviewed_views"]) <= {v["name"] for v in new["metrics"]["views"]}):
        raise ValueError("Review must identify actual held-out views and findings")
    finding = html.escape(review["finding"]) if review else "Awaiting visual review. The existing model choices are unchanged."
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>60minutes_1 | Longer training comparison</title><style>
body{{font:16px/1.55 'Avenir Next',sans-serif;background:#f3f4ec;color:#182921;margin:0;padding:24px}}main{{max-width:1280px;margin:auto}}section,header{{background:white;border:1px solid #d6e1d9;border-radius:18px;padding:20px;margin-bottom:18px}}h1{{letter-spacing:-.04em}}h2{{font-size:15px;overflow-wrap:anywhere}}a{{color:#187859}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #d6e1d9;text-align:left}}.images{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}figure{{margin:0}}img{{width:100%;display:block}}figcaption{{font-size:12px}}.notice{{background:#fff4df;padding:14px;border-radius:10px}}.table{{overflow:auto}}@media(max-width:700px){{.images{{grid-template-columns:1fr}}body{{padding:10px}}}}</style>
<main><header><a href="../../playback.html?assembly=joint30">Open synchronized comparison</a><h1>Does longer training help the connected scene?</h1>
<p>Same {len(split['training'])} training images, 1024 x 1024 resolution, fixed calibrated MPS poses and 30 withheld RGB views. The additive recipe runs 30,000 steps and permits densification through step 15,000; the preserved 7,000-step recipe densified through step 3,500. This compares two optimization recipes, not only iteration count. One run per recipe; no repeat-run uncertainty estimate.</p>
<p class="notice">{finding}</p><div class="table"><table><tr><th>Section</th><th>Separate original PSNR / SSIM</th><th>Joint 7k</th><th>Joint 30k</th></tr>{''.join(rows)}</table></div>
<p>Scores measure held-out appearance, not surface completeness or clinical accuracy. MPS geometry is shared. No dynamic-object masks are used: screens, hands, phones and other changing content can remain as ghosts. The 15-30 minute extension is trained separately; neither joint model covers that interval. All previous models, raw data and annotations are retained. Local Apple GPU only; no upload.</p></header>{''.join(cards)}</main></html>'''
    (report / "index.html").write_text(page)
    save_json(report / "summary.json", {"per_submap": summary, "old": old["metrics"], "new": new["metrics"], "visual_review": review})
    if review:
        old_manifest = json.loads((OUT / "viewer.json").read_text())
        model = new["metrics"]
        scene = {**old_manifest["scene"], "id": "joint_first15_30000", "title": "Joint first 15 minutes / 30,000 steps", "iterations": 30000,
                 "ply": os.path.relpath(model["model"], ENV), "details_url": "joint_first15_experiment/longer_training_comparison/index.html"}
        save_json(OUT / "viewer_30000.json", {"status": "reviewed_experimental", "default": "submaps", "graph_uid": old_manifest["graph_uid"],
                                             "model_sha256": model["model_sha256"], "scene": scene, "visual_review": review})
        source = ENV / "scene_playback_data.json"
        backup = ENV / "scene_playback_data.before_joint30.json"
        if not backup.exists():
            shutil.copy2(source, backup)
        payload = json.loads(source.read_text())
        payload["joint_long_review_url"] = "joint_first15_experiment/viewer_30000.json"
        save_json(source, payload)
    check_protected(ENV)
    print(json.dumps({"report": str(report / "index.html"), "per_submap": summary, "published": bool(review)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-file", type=Path)
    build(parser.parse_args().review_file)
