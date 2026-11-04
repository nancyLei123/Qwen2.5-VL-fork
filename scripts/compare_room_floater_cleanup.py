#!/usr/bin/env python3
"""Conservative local floater experiment; never replace the original model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from prepare_gaussian_viewer_compat import parse_ply
from reconstruct_dedicated_room import OUTPUT, render_checkpoint, save_json, sha256, verify_stats

ENV = OUTPUT.parent
OUT = ENV / "floater_cleanup_20261029"
BASE = OUTPUT / "trained_30000/splat.ply"


def subset_ply(raw, keep):
    parsed = parse_ply(raw)
    if keep.shape != (parsed["vertex_count"],) or keep.dtype != np.bool_ or not keep.any():
        raise ValueError("Invalid vertex selection")
    lines = [f"element vertex {int(keep.sum())}\n".encode() if line.startswith(b"element vertex ")
             else line for line in parsed["lines"]]
    rows = np.frombuffer(parsed["body"], dtype=np.uint8).reshape(parsed["vertex_count"], -1)
    return b"".join(lines) + rows[keep].tobytes()


def candidate_mask(route_distance, geometry_distance, radius=.12, protect=.05):
    # Proximity is only a heuristic. Preserve points close to supported static geometry.
    return (np.isfinite(route_distance) & np.isfinite(geometry_distance) &
            (route_distance < radius) & (geometry_distance > protect))


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT / "audit.json").exists():
        audit = json.loads((OUT / "audit.json").read_text())
        assert sha256(BASE) == audit["baseline_sha256"]
        assert sha256(OUT / "candidate.ply") == audit["candidate_sha256"]
        return audit
    split = json.loads((OUTPUT / "split.json").read_text())
    verify_stats(split["source_stats"])
    original = BASE.read_bytes()
    parsed = parse_ply(original)
    a = np.frombuffer(parsed["body"], dtype="<f4").reshape(parsed["vertex_count"], -1)
    names = parsed["properties"]
    xyz = a[:, [names.index(n) for n in ("x", "y", "z")]]
    if not np.isfinite(a).all():
        raise ValueError("Non-finite original Gaussian properties")
    playback = json.loads((ENV / "scene_playback_data.json").read_text())
    if playback["provenance"]["graph_uid"] != split["world_frame"]["graph_uid"]:
        raise ValueError("Route and reconstruction coordinate frames differ")
    route = np.array([p[:3] for p in playback["poses"] if p is not None])
    points = np.loadtxt(OUTPUT / "train/sparse/0/points3D.txt", usecols=(1, 2, 3))
    route_distance = cKDTree(route).query(xyz)[0]
    geometry_distance = cKDTree(points).query(xyz)[0]
    remove = candidate_mask(route_distance, geometry_distance)
    if remove.mean() > .01 or not remove.any():
        raise ValueError("Candidate outside conservative one-percent removal budget")
    candidate = subset_ply(original, ~remove)
    with (OUT / "candidate.ply").open("xb") as stream:
        stream.write(candidate)
    restored = parse_ply(candidate)
    kept = np.frombuffer(restored["body"], dtype="<f4").reshape(restored["vertex_count"], -1)
    assert np.array_equal(a[~remove], kept)
    np.savez_compressed(OUT / "removed_vertices.npz", original_indices=np.flatnonzero(remove),
                        positions=xyz[remove], route_distance_m=route_distance[remove],
                        initializer_distance_m=geometry_distance[remove])
    dc = a[:, [names.index(f"f_dc_{i}") for i in range(3)]]
    dark = np.clip(.5 + .28209479177387814 * dc, 0, 1).mean(axis=1) < .25
    audit = {"trial_id": "RoomScan3c", "baseline_sha256": sha256(BASE),
        "candidate_sha256": sha256(OUT / "candidate.ply"), "original_vertices": len(a),
        "removed_vertices": int(remove.sum()), "removed_percent": float(remove.mean()*100),
        "retained_vertices": len(kept), "removed_with_dark_dc_color": int((remove & dark).sum()),
        "route_radius_m": .12, "protect_initializer_within_m": .05,
        "retained_vertex_bytes_unchanged": True, "color_used_for_selection": False,
        "method": "Remove only Gaussian centers within 12 cm of a supported recorded RGB-camera center and more than 5 cm from a training initializer landmark. Actual RGB timestamps and route are unchanged.",
        "limitations": "Experimental spatial heuristic, not proof a point is false. MPS/initializer support is incomplete and may itself contain artifacts. No unseen surface is invented. This cannot remove all off-route floaters or recover occluded objects. Validation set was previously used for development; not a new independent test.",
        "default_model_changed": False}
    save_json(OUT / "audit.json", audit)
    return audit


def evaluate():
    audit = prepare()
    split = json.loads((OUTPUT / "split.json").read_text())
    for row in split["validation"]:
        if not (OUTPUT / "eval_views" / Path(row["name"]).stem / "sparse/0/points3D.bin").exists():
            raise ValueError("Expected preserved evaluation dataset is missing")
    for name in ("images", "masks", "train", "eval_views"):
        target = OUT / name
        if not target.exists():
            target.symlink_to(OUTPUT / name, target_is_directory=True)
    result = render_checkpoint(OUT, split, OUT / "candidate.ply", 30000)
    baseline = json.loads((OUTPUT / "evaluation_30000/metrics.json").read_text())
    old = {r["name"]: r for r in baseline["views"]}
    differences = [{"name": r["name"], "recording_s": r["recording_s"],
        "background_psnr_delta_db": r["background_psnr_db"] - old[r["name"]]["background_psnr_db"],
        "background_ssim_delta": r["background_ssim"] - old[r["name"]]["background_ssim"]}
        for r in result["views"]]
    report = {"audit": audit, "baseline": baseline, "candidate": result, "per_view_delta": differences,
        "mean_background_psnr_delta_db": float(np.mean([d["background_psnr_delta_db"] for d in differences])),
        "mean_background_ssim_delta": float(np.mean([d["background_ssim_delta"] for d in differences])),
        "visual_review": "pending", "default_model_changed": False}
    save_json(OUT / "comparison.json", report)
    verify_stats(split["source_stats"])
    assert sha256(BASE) == audit["baseline_sha256"]
    for batch in range(3):
        sheet = Image.new("RGB", (1200, 4 * 426), "#f4f5ef")
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(result["views"][batch*4:(batch+1)*4]):
            paths = [row["reference"], old[row["name"]]["render"], row["render"]]
            for col, (label, path) in enumerate(zip(("Actual Aria", "Original 30k", "Conservative cleanup"), paths)):
                image = Image.open(path).convert("RGB").resize((400, 400))
                sheet.paste(image, (400*col, 426*j+26))
                draw.text((400*col+8, 426*j+6), f"{row['recording_s']:.1f}s | {label}", fill="#193126")
        sheet.save(OUT / f"validation_comparison_{batch+1}.jpg", quality=94)
    print(json.dumps({k: report[k] for k in ("mean_background_psnr_delta_db", "mean_background_ssim_delta")}, indent=2))


def control():
    """Confirm that renderer/environment changes are not mistaken for cleanup effects."""
    audit = prepare()
    split = json.loads((OUTPUT / "split.json").read_text())
    folder = OUT / "baseline_control"
    folder.mkdir(exist_ok=True)
    for name in ("images", "masks", "train", "eval_views"):
        target = folder / name
        if not target.exists():
            target.symlink_to(OUTPUT / name, target_is_directory=True)
    result = render_checkpoint(folder, split, BASE, 30000)
    original = json.loads((OUTPUT / "evaluation_30000/metrics.json").read_text())
    old = {r["name"]: r for r in original["views"]}
    comparisons = []
    for row in result["views"]:
        a = np.asarray(Image.open(row["render"])).astype(np.int16)
        b = np.asarray(Image.open(old[row["name"]]["render"])).astype(np.int16)
        comparisons.append({"name": row["name"], "max_pixel_difference": int(np.abs(a-b).max()),
                            "mean_pixel_difference": float(np.abs(a-b).mean())})
    save_json(OUT / "baseline_control.json", {"views": comparisons,
        "baseline_sha256": audit["baseline_sha256"], "unchanged": sha256(BASE) == audit["baseline_sha256"]})
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "evaluate", "control"), required=True)
    args = parser.parse_args()
    {"prepare": prepare, "evaluate": evaluate, "control": control}[args.stage]()
