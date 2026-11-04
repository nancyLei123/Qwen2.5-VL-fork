#!/usr/bin/env python3
"""Local, non-destructive Ntrial3 reconstruction quality experiment.

Withholds five temporally distributed views plus a 0.25 s exclusion radius.
Recolors MPS geometry using training images only. This is a conditional novel
view evaluation, not an independent test of MPS or clinical geometry accuracy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image

from export_aria_gaussian_dataset import color_points, write_colmap_model
from train_aria_gaussian_scene import ply_vertex_count

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "runs/Ntrial3/environment_reconstruction"
DEFAULT_OUTPUT = ENV / "quality_comparison_20261022"
DEFAULT_BINARY = ROOT.parent / "tools/OpenSplat-1.2.2/build-mps-torch213/opensplat"
TORCH_LIB = Path("/Users/leijianan/Downloads/创业项目/无人机/outputs/uav_material_analysis/.venv_uav_bridge/lib/python3.11/site-packages/torch/lib")
SCENES = ("upstairs_context", "sitting_context", "corridor_walking")
RECIPES = {
    "baseline_7000": {"iterations": 7000, "densify_until": 3500, "refine_every": 500, "checkpoints": [7000]},
    "long_30000": {"iterations": 30000, "densify_until": 15000, "refine_every": 500, "checkpoints": [7000, 15000, 30000]},
}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_frames(frames, guard_s=0.25):
    if len(frames) < 30:
        raise ValueError("Need at least 30 frames for this five-view protocol")
    indices = np.round(np.linspace(0.1, 0.9, 5) * (len(frames) - 1)).astype(int)
    validation = [frames[i] for i in indices]
    names = {item["name"] for item in validation}
    times = [item["recording_s"] for item in validation]
    training, guard = [], []
    for item in frames:
        if item["name"] in names:
            continue
        if min(abs(item["recording_s"] - t) for t in times) <= guard_s:
            guard.append(item)
        else:
            training.append(item)
    if len(training) < 12:
        raise ValueError("Too few training frames after guard exclusion")
    return training, validation, guard


def metrics(reference, reconstruction):
    from skimage.metrics import structural_similarity

    a = np.asarray(reference.convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(reconstruction.convert("RGB"), dtype=np.float32) / 255.0
    if a.shape != b.shape:
        raise ValueError("Metrics require identical resolution; no resizing allowed")
    mse = float(np.mean((a - b) ** 2))
    return {
        "psnr_db": float(-10 * np.log10(max(mse, 1e-12))),
        "ssim": float(structural_similarity(a, b, channel_axis=2, data_range=1.0)),
        "mae": float(np.mean(np.abs(a - b))),
    }


def frame_records(frames, source):
    return [{**item, "path": source / "images" / item["name"],
             "transform_world_camera": np.asarray(item["T_world_camera"])} for item in frames]


def make_dataset(folder, manifest, frames, source, points, colors):
    (folder / "images").mkdir(parents=True, exist_ok=True)
    for item in frames:
        destination = folder / "images" / item["name"]
        if not destination.exists():
            destination.symlink_to(source / "images" / item["name"])
    c = manifest["camera"]
    write_colmap_model(folder, c["width"], c["height"],
                       tuple(c[k] for k in ("fx", "fy", "cx", "cy")),
                       frame_records(frames, source), points, colors)


def protected_sources(source):
    paths = list((source / "images").glob("*.jpg"))
    paths += [source / "dataset_manifest.json", source / "sparse/0/points3D.txt"]
    paths += list((source / "trained_7000_fullres").glob("*.ply"))
    paths += list((source / "trained_7000_fullres").glob("*.json"))
    paths += [ENV / "scene_playback_data.json", ENV.parent / "session_dashboard_manifest.json"]
    return {str(path): sha256(path) for path in paths if path.is_file()}


def prepare_scene(scene, output):
    source = ENV / "gaussian_submaps" / scene
    dest = output / scene
    manifest = json.loads((source / "dataset_manifest.json").read_text())
    split_path = dest / "split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text())
        verify_sources(split)
        return split
    dest.mkdir(parents=True, exist_ok=True)
    train, validation, guard = split_frames(manifest["frames"]["items"])
    provenance = protected_sources(source)
    points = np.loadtxt(source / "sparse/0/points3D.txt", usecols=(1, 2, 3), dtype=np.float32)
    c = manifest["camera"]
    colors, counts = color_points(points, frame_records(train, source),
                                  tuple(c[k] for k in ("fx", "fy", "cx", "cy")),
                                  c["width"], c["height"])
    keep = counts >= 2
    points, colors = points[keep], colors[keep]
    make_dataset(dest / "train", manifest, train, source, points, colors)
    # Evaluation contains all poses but executes zero training steps. The points
    # and their colors are exactly those of the training-only initializer.
    make_dataset(dest / "eval", manifest, manifest["frames"]["items"], source, points, colors)
    split = {
        "scene": scene, "source": str(source), "camera": c,
        "world_frame": manifest["world_frame"], "segment": manifest["segment"],
        "training": train, "validation": validation, "guard": guard,
        "guard_s": 0.25, "initializer_points": len(points),
        "source_sha256": provenance,
        "limits": "MPS poses and geometric points use the whole recording. Held-out RGB is excluded from optimization and initializer recoloring; this is not end-to-end held-out SLAM validation.",
    }
    save_json(split_path, split)
    return split


def verify_sources(split):
    mismatches = [name for name, digest in split["source_sha256"].items() if sha256(name) != digest]
    if mismatches:
        raise RuntimeError(f"Source changed; stopping rather than mixing experiments: {mismatches}")


def backend_env():
    env = dict(os.environ)
    if not TORCH_LIB.is_dir():
        raise FileNotFoundError(TORCH_LIB)
    env["DYLD_LIBRARY_PATH"] = str(TORCH_LIB) + (":" + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else "")
    return env


def execute(command, log, status_path, stage, total=None):
    started = time.time()
    save_json(status_path, {"stage": stage, "state": "running", "started_at": datetime.now(timezone.utc).isoformat(), "command": command})
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        process = subprocess.Popen(command, env=backend_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                if line.startswith("Using CPU"):
                    raise RuntimeError("Apple GPU is unavailable in this process; stopped CPU fallback immediately")
                match = re.match(r"Step (\d+):", line)
                if match and int(match[1]) % 1000 == 0:
                    step = int(match[1])
                    save_json(status_path, {"stage": stage, "state": "running", "step": step,
                                           "total_steps": total, "elapsed_s": time.time() - started})
                    print(f"{stage}: {line.strip()}", flush=True)
                elif line.startswith(("Using", "Densify", "Final prune")):
                    print(f"{stage}: {line.strip()}", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f"Command failed ({code}); see {log}")
    text = log.read_text()
    if "Using MPS" not in text:
        raise RuntimeError("Expected the local Apple GPU, refusing a silent CPU fallback")
    return time.time() - started, text


def base_command(binary, dataset, n):
    return [str(binary), str(dataset), "-n", str(n), "-d", "1", "--sh-degree", "3",
            "--sh-degree-interval", "1000", "--ssim-weight", "0.2", "--max-gaussians", "1000000"]


def evaluate(binary, scene_dir, split, ply, step, destination, status):
    score_path = destination / "metrics.json"
    if score_path.exists():
        score = json.loads(score_path.read_text())
        if score["model_sha256"] != sha256(ply):
            raise RuntimeError("Model changed after evaluation")
        return score
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    digest = sha256(ply)
    for item in split["validation"]:
        name = item["name"]
        render = destination / (Path(name).stem + ".png")
        log = destination / (Path(name).stem + ".log")
        # The loaded checkpoint step equals --num-iters, so the optimizer loop
        # must be skipped. Assert this from the log instead of assuming it.
        with tempfile.TemporaryDirectory(prefix="aria-quality-eval-") as temporary:
            temp = Path(temporary)
            command = base_command(binary, scene_dir / "eval", step) + [
                "--resume", str(ply), "--val", "--val-image", name,
                "--val-render", str(temp / "renders"), "-o", str(temp / "unused.ply")]
            _, text = execute(command, log, status, f"{split['scene']} evaluate {ply.name} {name}")
            if re.search(r"^Step \d+:", text, re.MULTILINE) or "Loaded " not in text:
                raise RuntimeError("Evaluation did not perform a verified render-only checkpoint load")
            shutil.copyfile(temp / "renders" / f"{step}.png", render)
        reference = Path(split["source"]) / "images" / name
        with Image.open(reference) as a, Image.open(render) as b:
            values = metrics(a, b)
        rows.append({"name": name, "recording_s": item["recording_s"], "reference": str(reference),
                     "render": str(render), **values})
    if sha256(ply) != digest:
        raise RuntimeError("Evaluation modified the checkpoint")
    score = {"step": step, "model": str(ply), "model_sha256": digest,
             "gaussians": ply_vertex_count(ply), "views": rows,
             "mean_psnr_db": float(np.mean([r["psnr_db"] for r in rows])),
             "mean_ssim": float(np.mean([r["ssim"] for r in rows]))}
    save_json(score_path, score)
    return score


def train_recipe(binary, output, scene, split, recipe_name):
    recipe = RECIPES[recipe_name]
    scene_dir = output / scene
    destination = scene_dir / recipe_name
    destination.mkdir(parents=True, exist_ok=True)
    record_path = destination / "run.json"
    status = output / "progress.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record["binary_sha256"] != sha256(binary) or record["recipe"] != recipe:
            raise RuntimeError("Recipe or binary changed: use a new experiment directory")
    else:
        final = destination / "splat.ply"
        if final.exists() or (destination / "training.log").exists():
            raise RuntimeError(f"Incomplete output exists; inspect before rerunning: {destination}")
        command = base_command(binary, scene_dir / "train", recipe["iterations"]) + [
            "--refine-every", str(recipe["refine_every"]), "--densify-from", "500",
            "--densify-until", str(recipe["densify_until"]), "-o", str(final)]
        if len(recipe["checkpoints"]) > 1:
            command += ["--save-every", "1000"]
        runtime, text = execute(command, destination / "training.log", status,
                                f"{scene}/{recipe_name}", recipe["iterations"])
        record = {"id": recipe_name, "scene": scene, "recipe": recipe,
                  "runtime_s": runtime, "command": command, "binary_sha256": sha256(binary),
                  "backend": "Apple MPS", "evaluations": [],
                  "interpretation": "Baseline uses the earlier 7k recipe. Long run extends the learning-rate and densification schedule, so baseline-vs-long is a recipe comparison, not an isolated iteration ablation. Checkpoints within the long run share one schedule. One run per recipe; stochastic variability is not estimated."}
        save_json(record_path, record)
    for step in recipe["checkpoints"]:
        ply = destination / ("splat.ply" if step == recipe["iterations"] else f"splat_{step}.ply")
        score = evaluate(binary, scene_dir, split, ply, step, destination / f"eval_{step}", status)
        record["evaluations"] = [e for e in record["evaluations"] if e["step"] != step] + [score]
        save_json(record_path, record)
        publish_results(output)
        print(f"RESULT {scene}/{recipe_name}/{step}: {score['mean_psnr_db']:.3f} dB, SSIM {score['mean_ssim']:.4f}", flush=True)
    return record


def publish_results(output):
    scenes = []
    for split_path in sorted(output.glob("*/split.json")):
        split = json.loads(split_path.read_text())
        runs = [json.loads(p.read_text()) for p in sorted(split_path.parent.glob("*/run.json"))]
        scenes.append({"id": split["scene"], "split": str(split_path), "camera": split["camera"],
                       "training_count": len(split["training"]), "validation_count": len(split["validation"]),
                       "guard_count": len(split["guard"]), "validation": split["validation"], "runs": runs})
    result = {"trial_id": "Ntrial3", "updated_at": datetime.now(timezone.utc).isoformat(), "scenes": scenes,
              "policy": "Local-only. Original models/video/labels unchanged. Five fixed held-out RGB views with 0.25 s temporal guards, training-only initializer colors; MPS poses/geometry are shared. Validation scores support appearance comparison, not clinical or geometric accuracy. This is a development validation set, not an untouched final test set."}
    save_json(output / "results.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--recipes", nargs="+", choices=tuple(RECIPES), default=list(RECIPES))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ENV or not output.is_relative_to(ENV):
        raise ValueError("Experiment must be in its own directory under environment_reconstruction")
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 15 * 1024**3:
        raise RuntimeError("Less than 15 GiB free; stopping")
    try:
        for scene in args.scenes:
            split = prepare_scene(scene, output)
            publish_results(output)
            print(f"PREPARED {scene}: {len(split['training'])} train, {len(split['validation'])} validation, {len(split['guard'])} guarded", flush=True)
            if args.prepare_only:
                continue
            for recipe in args.recipes:
                train_recipe(args.binary.resolve(), output, scene, split, recipe)
            verify_sources(split)
        save_json(output / "progress.json", {"state": "prepared" if args.prepare_only else "complete", "scenes": args.scenes, "recipes": args.recipes})
    except BaseException as error:
        save_json(output / "failure.json", {"error": str(error), "at": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    main()
