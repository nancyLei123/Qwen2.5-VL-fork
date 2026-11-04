"""Create a local-only comparison gallery from completed Metashape experiments."""

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from run_metashape_comparison import DEFAULT_OUTPUT, ROOT, verify_sources, write_json


def psnr(reference, estimate, mask=None):
    if reference.shape != estimate.shape:
        raise ValueError("Image dimensions must match; no resizing in scoring")
    difference = (reference.astype(np.float64) - estimate.astype(np.float64)) / 255.0
    if mask is not None:
        if mask.shape != reference.shape[:2]:
            raise ValueError("Invalid comparison mask")
        difference = difference[mask]
    if difference.size == 0:
        return None
    mse = float(np.mean(difference ** 2))
    return 100.0 if mse == 0 else round(-10 * math.log10(mse), 3)


def image_metrics(reference, gaussian, mesh_rgba):
    if mesh_rgba.shape != (*reference.shape[:2], 4):
        raise ValueError("Mesh render must be matching-size RGBA")
    alpha = mesh_rgba[:, :, 3:4].astype(np.float64) / 255
    mask = alpha[:, :, 0] >= 0.99
    mesh_on_black = mesh_rgba[:, :, :3].astype(np.float64) * alpha
    return {
        "opaque_mesh_coverage_pct": round(100 * float(mask.mean()), 2),
        "missing_or_edge_pct": round(100 * float((~mask).mean()), 2),
        "gaussian_full_psnr_db": psnr(reference, gaussian),
        "metashape_full_black_psnr_db": psnr(reference, mesh_on_black),
        "gaussian_common_mask_psnr_db": psnr(reference, gaussian, mask),
        "metashape_common_mask_psnr_db": psnr(reference, mesh_rgba[:, :, :3], mask),
    }


def build(output):
    output = Path(output)
    plan = json.loads((output / "comparison_plan.json").read_text())
    verify_sources(plan)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "trial": "Ntrial3", "status": "complete",
        "source_integrity": "SHA-256 verified; protected source images and Gaussian results unchanged",
        "method": plan["method"], "limitations": plan["limitations"], "scenes": {},
        "metric_notes": [
            "PSNR is computed from the saved 1024x1024 RGB images, in encoded sRGB; higher means closer pixel appearance, not better measured geometry.",
            "Full-frame Metashape PSNR composites alpha over black and includes missing pixels. Gaussian full-frame PSNR uses the existing saved render.",
            "Common-mask scores evaluate BOTH methods on exactly the same pixels where the Metashape render alpha is at least 0.99; always read with coverage.",
            "Coverage is only projected opaque mesh support in this ONE view, not room completeness or validated surface accuracy.",
            "Texture and mesh hole filling are disabled. Black or checkerboard gaps are not repaired by generative filling.",
        ],
    }
    review_path = output / "visual_review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    report["visual_summary"] = review.get("summary", "")
    for label, scene in plan["scenes"].items():
        directory = output / label
        status = json.loads((directory / "metashape_status.json").read_text())
        if status["status"] != "complete" or status.get("pipeline_revision", 0) < 2:
            raise RuntimeError(f"{label} has not finished the verified pipeline")
        reference = np.array(Image.open(directory / "reference.jpg").convert("RGB"))
        gaussian = np.array(Image.open(directory / "gaussian.png").convert("RGB"))
        with Image.open(directory / "metashape_rgba.png") as render:
            if render.mode != "RGBA":
                raise ValueError("Expected a real alpha channel, not inferred black pixels")
            mesh = np.array(render)
        metrics = image_metrics(reference, gaussian, mesh)
        # This display composite marks absent geometry; evaluation uses the original RGBA.
        y, x = np.indices(mesh.shape[:2])
        checker = np.where(((x // 24 + y // 24) % 2)[..., None] == 0,
                           np.array([213, 221, 213]), np.array([239, 243, 237]))
        alpha = mesh[:, :, 3:4] / 255.0
        composite = np.rint(mesh[:, :, :3] * alpha + checker * (1 - alpha)).astype(np.uint8)
        Image.fromarray(composite).save(directory / "metashape_checkered.jpg", quality=95)
        comparison = Image.new("RGB", (3072, 1024))
        for i, arr in enumerate((reference, gaussian, composite)):
            comparison.paste(Image.fromarray(arr), (i * 1024, 0))
        comparison.save(directory / "comparison.jpg", quality=95)
        report["scenes"][label] = {
            "title": {"room": "Room", "corridor": "Corridor", "stairs": "Stairs"}[label],
            "submap": scene["submap"], "segment": scene["segment"],
            "validation_frame": scene["validation_frame"], "camera": scene["camera"],
            "training_frames": scene["training_frame_count"],
            "metrics": metrics, "metashape": status,
            "gaussian_iterations": scene["gaussian_iterations"],
            "gaussian_reported_psnr_db": scene["gaussian_reported_psnr_db"],
            "visual_observation": review.get("scenes", {}).get(label, ""),
        }
    write_json(output / "comparison_results.json", report)
    for name in ("index.html", "gallery.js", "mesh.html", "mesh.js", "style.css"):
        shutil.copy2(ROOT / "scripts/templates/metashape_comparison" / name, output / name)
    lines = ["# NTrial3: Metashape / Gaussian Splatting pilot", "",
             "Local experiment, 19 October 2026. Existing PD Insighter results are unchanged.", "",
             "## Method", "",
             "Metashape Professional 2.3.2 runs locally on the Apple M5 Pro. This is an MPS-assisted baseline: fixed calibrated Aria camera poses and intrinsics, image matching and triangulation, full-resolution depth maps, a triangle mesh, and a 4096-pixel texture atlas. Mesh and texture hole filling are disabled. The reconstruction region is each existing MPS initializer point bounding box plus a 0.5 m margin. Missing regions can reflect visibility, poor matching, filtering or this crop, not necessarily absent source video. The exported PSX project is reopened to verify mesh persistence.", "",
             "The baseline is the existing 7,000-iteration Gaussian Splatting reconstruction. Both use the same source-frame set and omit the same one image from depth/texturing or Gaussian optimization. All times are original recording/VRS seconds, not offset dashboard time.", "",
             "## Results", "",
             "| Scene | Training images | Mesh support in test view | GS full PSNR | MS full PSNR (black gaps) | GS common-mask PSNR | MS common-mask PSNR |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label, scene in report["scenes"].items():
        m = scene["metrics"]
        lines.append(f"| {scene['title']} | {scene['training_frames']} | {m['opaque_mesh_coverage_pct']:.2f}% | {m['gaussian_full_psnr_db']:.2f} | {m['metashape_full_black_psnr_db']:.2f} | {m['gaussian_common_mask_psnr_db']} | {m['metashape_common_mask_psnr_db']} |")
    lines += ["", "PSNR values are dB, recomputed from the saved images without resizing. A single test view does not establish a winner for a whole scene.", "", "## Visual review", "", report["visual_summary"], ""]
    lines += [f"- {scene['title']}: {scene['visual_observation']}" for scene in report["scenes"].values() if scene["visual_observation"]]
    lines += ["", "## How to inspect", "",
              "Open index.html through the local PD Insighter server. Switch between Room, Corridor and Stairs; compare the three equal-size images and use shared image zoom. The Metashape 3D viewer is loaded only when requested. It starts at the withheld camera view and permits orbit/pan/zoom. Existing Gaussian viewers are linked, not replaced.", "",
              "Each scene folder contains reconstruction.psx plus reconstruction.files (keep together), model.obj/model.mtl/model.jpg (keep together), and the original RGBA test-view render. Source photos are linked in their existing Gaussian dataset folders, not duplicated; retain those paths if reprocessing the projects. The exported meshes and this local web viewer do not require an active Metashape license to view. Research data was not uploaded by this workflow. Installer retrieval and trial licensing were the only authorized external services.", "", "## Interpretation limits", ""]
    lines += ["- " + note for note in report["limitations"] + report["metric_notes"]]
    lines += ["", "## Reproduce", "", "```sh",
              "cd /Users/leijianan/gen1_glasses/Qwen2.5-VL-fork",
              "/Users/leijianan/gen1_glasses/tools/MetashapePro.app/Contents/MacOS/MetashapePro -r scripts/run_metashape_comparison.py --run --scene all",
              "/opt/anaconda3/bin/python scripts/build_metashape_comparison.py", "```", "",
              "An active trial/license is required to rerun reconstruction. Completed verified scenes are skipped. Source integrity is checked before and after processing.", "",
              "References: [Metashape official installer/trial](https://www.agisoft.com/downloads/installer/), [Metashape 2.3 Python API](https://www.agisoft.com/pdf/metashape_python_api_2_3_0.pdf).", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({k: v["metrics"] for k, v in report["scenes"].items()}, indent=2))
    print(output / "index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    build(parser.parse_args().output_dir)
