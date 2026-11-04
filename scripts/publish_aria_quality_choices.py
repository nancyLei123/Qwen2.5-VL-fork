#!/usr/bin/env python3
"""Publish visually reviewed reconstruction candidates without replacing originals."""
import argparse
import json
import math
import os
from pathlib import Path
import re

from compare_aria_gaussian_quality import DEFAULT_OUTPUT, ENV, save_json, sha256


def checked_views(evaluation, expected=None):
    rows = evaluation.get("views", [])
    if len(rows) != 5 or len({v.get("name") for v in rows}) != 5:
        return None
    for row in rows:
        for key in ("recording_s", "psnr_db", "ssim"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                return None
        if not -1 <= row["ssim"] <= 1:
            return None
    if expected is not None:
        expected_times = {v["name"]: v["recording_s"] for v in expected}
        if len(expected) != 5 or {v["name"]: v["recording_s"] for v in rows} != expected_times:
            return None
    for field, key in (("mean_psnr_db", "psnr_db"), ("mean_ssim", "ssim")):
        value = evaluation.get(field)
        actual = sum(v[key] for v in rows) / 5
        if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value - actual) > 1e-6:
            return None
    return {row["name"]: row for row in rows}


def eligible(baseline, candidate, expected=None):
    old, new = checked_views(baseline, expected), checked_views(candidate, expected)
    if not old or not new or old.keys() != new.keys():
        return False
    if any(old[name]["recording_s"] != new[name]["recording_s"] for name in old):
        return False
    deltas = [new[name]["psnr_db"] - old[name]["psnr_db"] for name in old]
    return (sum(deltas) / 5 >= 0.3
            and sum(new[name]["ssim"] - old[name]["ssim"] for name in old) / 5 >= 0.005
            and sum(delta > 0 for delta in deltas) >= 4
            and min(deltas) >= -0.5)


def publish(output, reviewed, activate=False):
    results = json.loads((output / "results.json").read_text())
    original = json.loads((ENV / "scene_playback_data.json").read_text())
    original_by_id = {s["id"]: s for s in original["scenes"]}
    choices = {}
    for scene in results["scenes"]:
        if scene["id"] not in reviewed:
            continue
        runs = scene["runs"]
        baseline = next((r["evaluations"][0] for r in runs if r["id"] == "baseline_7000" and r["evaluations"]), None)
        if not baseline:
            continue
        split = json.loads(Path(scene["split"]).read_text())
        candidates = [e for r in runs if r["id"] != "baseline_7000" for e in r["evaluations"]
                      if e["model_sha256"] == reviewed[scene["id"]]
                      and eligible(baseline, e, split["validation"])]
        if not candidates:
            raise ValueError(f"Reviewed hash is not an eligible evaluated model: {scene['id']}")
        best = max(candidates, key=lambda e: (e["mean_psnr_db"], e["mean_ssim"]))
        path = Path(best["model"]).resolve()
        if not path.is_relative_to(output.resolve()) or sha256(path) != best["model_sha256"]:
            raise ValueError("Model provenance mismatch")
        if split["world_frame"]["graph_uid"] != original["provenance"]["graph_uid"]:
            raise ValueError("World-frame mismatch")
        if split["camera"] != original_by_id[scene["id"]]["camera"]:
            raise ValueError("Camera intrinsics mismatch")
        choices[scene["id"]] = {
            "ply": os.path.relpath(path, ENV), "iterations": best["step"],
            "details_url": os.path.relpath(output / "index.html", ENV),
            "model_sha256": best["model_sha256"], "mean_psnr_db": best["mean_psnr_db"],
            "mean_ssim": best["mean_ssim"], "baseline_mean_psnr_db": baseline["mean_psnr_db"],
            "baseline_mean_ssim": baseline["mean_ssim"], "validation_views": len(best["views"]),
            "visually_reviewed": True,
            "reviewed_model_sha256": reviewed[scene["id"]],
            "original_ply": original_by_id[scene["id"]]["ply"],
            "camera": original_by_id[scene["id"]]["camera"],
        }
    if set(reviewed) != set(choices) or (activate and not choices):
        raise ValueError("Every approval must resolve to a verified candidate; activation needs a candidate")
    payload = {
        "trial_id": "Ntrial3", "default_choice": "compared" if choices and activate else "original", "scenes": choices,
        "criteria": "At least +0.3 dB mean PSNR, +0.005 mean SSIM, improvements in >=4/5 views, no >0.5 dB per-view regression, and explicit visual review. Gates are exploratory engineering choices, not statistical significance or clinical validation.",
        "limitation": "Selected using the development validation set. Compared to a retrained same-split 7k baseline, not an independent score of the original deployed model. Browser SH2 and offline SH3 rendering differ.",
        "original_playback_sha256": sha256(ENV / "scene_playback_data.json"),
    }
    save_json(output / "playback_choices.json", payload)
    return payload


def parse_reviews(values):
    reviews = {}
    for value in values:
        scene, separator, digest = value.partition("=")
        if not scene or not separator or not re.fullmatch(r"[a-f0-9]{64}", digest) or scene in reviews:
            raise ValueError("Each review must be a unique scene_id=exact_model_sha256")
        reviews[scene] = digest
    return reviews


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reviewed-model", action="append", default=[],
                        help="scene_id=SHA256, only after inspecting all five renders of that exact model. Repeat per scene; omit to keep originals only.")
    parser.add_argument("--activate", action="store_true",
                        help="Use the reviewed selection by default; otherwise keep originals as default.")
    args = parser.parse_args()
    print(json.dumps(publish(args.output, parse_reviews(args.reviewed_model), args.activate), indent=2))
