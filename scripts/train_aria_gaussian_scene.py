#!/usr/bin/env python3
"""Train and publish an auditable OpenSplat scene from an exported Aria dataset."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT
        / "runs/Ntrial3/environment_reconstruction/gaussian_splatting_trial",
    )
    parser.add_argument(
        "--opensplat",
        type=Path,
        default=ROOT.parent / "tools/OpenSplat-1.2.2/build-cpu/opensplat",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--downscale", type=float, default=4.0)
    parser.add_argument(
        "--validation-image",
        default="frame_00060.jpg",
        help="Image filename to withhold, or 'auto' for the middle selected frame.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "auto"),
        default="cpu",
        help="Use 'auto' with a Metal/CUDA-enabled OpenSplat binary.",
    )
    parser.add_argument("--max-gaussians", type=int, default=200_000)
    parser.add_argument("--require-gpu", action="store_true", help="Stop instead of silently falling back to CPU.")
    parser.add_argument("--sh-degree", type=int, default=1)
    parser.add_argument("--sh-degree-interval", type=int, default=100)
    parser.add_argument("--refine-every", type=int, default=500)
    parser.add_argument("--densify-from", type=int, default=500)
    parser.add_argument("--densify-until", type=int, default=-1)
    parser.add_argument(
        "--publish-only",
        action="store_true",
        help="Regenerate the HTML viewer from existing training outputs without retraining.",
    )
    return parser.parse_args()


def parse_validation_metrics(log: str) -> tuple[float, float]:
    loss_matches = re.findall(r"validation loss:\s*([0-9.eE+-]+)", log)
    psnr_matches = re.findall(r"validation PSNR:\s*([0-9.eE+-]+)", log)
    if not loss_matches or not psnr_matches:
        raise RuntimeError("OpenSplat output did not contain validation loss and PSNR.")
    return float(loss_matches[-1]), float(psnr_matches[-1])


def resolve_validation_image(manifest: dict, requested: str) -> str:
    items = manifest["frames"]["items"]
    if not items:
        raise ValueError("Dataset manifest has no selected frames.")
    available = {item["name"] for item in items}
    if requested == "auto":
        return items[len(items) // 2]["name"]
    if requested not in available:
        raise ValueError(f"Validation image is not in the dataset: {requested}")
    return requested


def scene_view_from_manifest(manifest: dict, validation_image: str) -> dict:
    positions = [
        [item["T_world_camera"][axis][3] for axis in range(3)]
        for item in manifest["frames"]["items"]
    ]
    mins = [min(point[axis] for point in positions) for axis in range(3)]
    maxs = [max(point[axis] for point in positions) for axis in range(3)]
    center = [(low + high) / 2 for low, high in zip(mins, maxs)]
    span = max(maxs[axis] - mins[axis] for axis in range(3))
    validation_frame = next(
        (item for item in manifest["frames"]["items"] if item["name"] == validation_image),
        manifest["frames"]["items"][len(manifest["frames"]["items"]) // 2],
    )
    transform = validation_frame["T_world_camera"]
    camera_position = [transform[axis][3] for axis in range(3)]
    camera_forward = [transform[axis][2] for axis in range(3)]
    camera_up = [-transform[axis][1] for axis in range(3)]
    held_out_look_at = [
        camera_position[axis] + 4.0 * camera_forward[axis]
        for axis in range(3)
    ]
    viewer_camera_position = [
        camera_position[axis] + camera_forward[axis]
        for axis in range(3)
    ]
    return {
        "mode": "validated_egocentric_view",
        "validation_image": validation_frame["name"],
        "recording_s": validation_frame["recording_s"],
        "camera_up": camera_up,
        "look_at": held_out_look_at,
        "camera_position": viewer_camera_position,
        "held_out_camera": {
            "position": camera_position,
            "up": camera_up,
            "look_at": held_out_look_at,
        },
        "route_center": center,
        "route_span_m": span,
        "camera_path_bounds_min_m": mins,
        "camera_path_bounds_max_m": maxs,
    }


def ply_vertex_count(source: Path) -> int:
    with source.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"Missing PLY header terminator in {source}")
            if line.startswith(b"element vertex "):
                vertex_count = int(line.split()[-1])
            if line == b"end_header\n":
                break
    try:
        return vertex_count
    except UnboundLocalError as error:
        raise ValueError(f"Missing vertex count in {source}") from error


def write_viewer_page(output: Path, summary: dict) -> None:
    view = summary["scene_view"]
    initial_view_label = view["mode"].replace("_", " ")
    segment = summary["dataset"]["segment"]
    trial_id = summary["trial_id"]
    viewer_path = "/Qwen2.5-VL-fork/vendor/gaussian-viewer/node_modules/@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js"
    three_path = "/Qwen2.5-VL-fork/vendor/gaussian-viewer/node_modules/three/build/three.module.js"
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Interactive Gaussian scene</title>
<style>
html,body,#splatViewer{{width:100%;height:100%;margin:0;overflow:hidden;background:#0b1410;font-family:"Avenir Next","Gill Sans",sans-serif}}
#splatViewer{{position:absolute;inset:0}}.hud{{position:absolute;z-index:4;left:16px;right:16px;top:14px;display:flex;justify-content:space-between;gap:12px;pointer-events:none}}
.pill{{max-width:min(620px,80vw);padding:9px 13px;border-radius:999px;background:#f8fbf6e8;color:#17231c;font-size:13px;font-weight:800;box-shadow:0 8px 24px #0004}}
.status.error{{background:#fff0ed;color:#8b2c1f}}.foot{{position:absolute;z-index:4;left:16px;bottom:14px;padding:8px 12px;border-radius:12px;background:#101914d9;color:#dbe8df;font-size:12px;pointer-events:none}}
</style><script type="importmap">{{"imports":{{"three":"{three_path}"}}}}</script></head><body>
<div id="splatViewer" aria-label="Interactive Gaussian reconstruction"></div><div class="hud"><div class="pill" id="status">Loading {summary['trained_splat_count']:,} trained Gaussians...</div><div class="pill">Drag: rotate · wheel/pinch: zoom · right-drag: pan</div></div>
<div class="foot">{html.escape(trial_id)} · recording {segment['start_s']:g}-{segment['end_s']:g} s · initial view = {initial_view_label} · metric MPS world frame</div>
<script type="module">
import * as GaussianSplats3D from "{viewer_path}";
const root=document.getElementById('splatViewer'); const status=document.getElementById('status');
try {{
  const viewer=new GaussianSplats3D.Viewer({{
    rootElement:root,
    cameraUp:{json.dumps(view['camera_up'])},
    initialCameraPosition:{json.dumps(view['camera_position'])},
    initialCameraLookAt:{json.dumps(view['look_at'])},
    sharedMemoryForWorkers:false,
    gpuAcceleratedSort:false,
    ignoreDevicePixelRatio:true,
    sphericalHarmonicsDegree:{summary['training_parameters']['spherical_harmonics_degree']},
    renderMode:GaussianSplats3D.RenderMode.Always,
    sceneRevealMode:GaussianSplats3D.SceneRevealMode.Gradual,
    logLevel:GaussianSplats3D.LogLevel.None
  }});
  await viewer.addSplatScene("{summary['viewer_splat_filename']}",{{splatAlphaRemovalThreshold:5,showLoadingUI:true,progressiveLoad:true}});
  viewer.start(); status.textContent="Interactive Gaussian scene loaded";
}} catch(error) {{
  console.error(error); status.textContent=`Viewer failed: ${{error?.message||error}}`; status.classList.add('error');
}}
</script></body></html>'''
    (output / "viewer.html").write_text(page)


def write_page(output: Path, summary: dict) -> None:
    render = f"renders/{summary['iterations']}.png"
    dataset = Path(summary["dataset_dir"])
    original = html.escape(
        os.path.relpath(dataset / "images" / summary["validation_image"], output),
        quote=True,
    )
    method_href = html.escape(os.path.relpath(dataset / "README.md", output), quote=True)
    environment_href = html.escape(
        os.path.relpath(
            ROOT / "runs" / summary["trial_id"] / "environment_reconstruction" / "index.html",
            output,
        ),
        quote=True,
    )
    segment = summary["dataset"]["segment"]
    submap_name = summary.get("submap_name", dataset.name).replace("_", " ")
    playback = ROOT / "runs" / summary["trial_id"] / "environment_reconstruction" / "playback.html"
    playback_link = ""
    if playback.is_file():
        href = html.escape(os.path.relpath(playback, output), quote=True)
        playback_link = f'<a class="button" href="{href}?t={segment["start_s"] + 1:g}">Synchronized video + 3D playback</a>'
    training = summary.get("training_parameters", {})
    densify_until = training.get("densify_until")
    if densify_until is not None and densify_until >= 0:
        densification_note = f"with densification through iteration {densify_until}"
    elif densify_until is not None:
        densification_note = "with automatic densification through half of training"
    else:
        densification_note = "with a conservative training configuration"
    formal_run = summary["status"] == "motion_centered_full_resolution_reconstruction"
    quality_heading = "What this reconstruction establishes" if formal_run else "What this proves"
    if formal_run:
        quality_notice = (
            f"<strong>Full-resolution motion-centered result.</strong> This submap uses "
            f"{summary['iterations']:,} iterations at native image resolution, {densification_note}. "
            "It is suitable for visual context review, but it is not evidence of exact obstacle "
            "dimensions, complete hidden surfaces or reliable moving-object reconstruction."
        )
        next_gate = (
            "Add dynamic-object masks, evaluate several held-out viewpoints, and improve capture "
            "overlap for the softer submaps before clinical interpretation."
        )
    else:
        quality_notice = (
            f"<strong>Not final clinical quality.</strong> This proof uses {summary['iterations']} "
            f"iterations at {summary['downscale_factor']:g}x lower image resolution, "
            f"{densification_note}. Residual blur is expected. It is not evidence of fine obstacle "
            "dimensions, complete hidden surfaces or reliable moving-object reconstruction."
        )
        next_gate = (
            "Train local motion-centered submaps at native resolution for several thousand "
            "iterations, add dynamic-object masks, and compare multiple held-out frames before "
            "placing the viewer in the clinician workflow."
        )
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(summary['trial_id'])} Gaussian scene | PD Insighter</title><style>
:root{{--ink:#17231c;--muted:#66736c;--green:#187859;--amber:#c47a16;--line:#dbe4da;--paper:#f3f1e8}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);font-family:"Avenir Next","Gill Sans",sans-serif;background:radial-gradient(circle at 10% 0,#e7f2e9,transparent 35%),var(--paper)}}
main{{max-width:1480px;margin:auto;padding:24px;display:grid;gap:20px}}.card{{background:#fffffff2;border:1px solid var(--line);border-radius:25px;padding:24px;box-shadow:0 16px 50px #21352212}}
h1{{font-size:clamp(32px,5vw,58px);letter-spacing:-.055em;margin:10px 0}}h2{{margin-top:0}}p{{color:var(--muted);line-height:1.55}}.eyebrow{{font-size:13px;text-transform:uppercase;letter-spacing:.12em;color:var(--green);font-weight:900}}
.button{{display:inline-block;padding:10px 15px;border-radius:99px;border:1px solid var(--line);background:white;color:var(--ink);text-decoration:none;font-weight:800;margin:4px}}
.proof{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}figure{{margin:0;min-width:0}}figure img{{width:100%;border-radius:16px;display:block}}figcaption{{padding-top:9px;font-weight:800}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.metric{{padding:15px;border-radius:16px;background:#f6f8f4}}.metric strong{{display:block;font-size:26px}}.metric span{{font-size:12px;color:var(--muted);font-weight:800}}.notice{{border-left:5px solid var(--amber);background:#fff7e5;padding:15px;border-radius:12px}}code{{font-family:"SFMono-Regular",monospace;font-size:.9em}}@media(max-width:760px){{main{{padding:10px}}.proof,.metrics{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header class="card"><span class="eyebrow">{html.escape(summary['trial_id'])} · {segment['start_s']:g}-{segment['end_s']:g} s metric submap</span><h1>{html.escape(submap_name.title())}: Gaussian Scene</h1>
<p>The held-out image below was not used for optimization. OpenSplat reconstructed its view from the remaining calibrated Aria frames and MPS geometry.</p>
<a class="button" href="{environment_href}">Back to Environment</a>{playback_link}<a class="button" href="{method_href}">Dataset method</a><a class="button" href="splat_{summary['iterations']}.ply">Download splat PLY</a></header>
<section class="card"><h2>Interactive trained scene</h2><p>Drag to rotate, scroll to zoom, and right-drag to pan through the actual OpenSplat output.</p><iframe src="viewer.html" title="Interactive trained Gaussian scene" style="width:100%;height:650px;border:0;border-radius:18px;background:#0b1410"></iframe></section>
<section class="card proof"><figure><img src="{render}" alt="Held-out view synthesized by OpenSplat"><figcaption>Automatic reconstruction, held-out view</figcaption></figure>
<figure><img src="{original}" alt="Original held-out Aria RGB frame"><figcaption>Original VRS RGB frame</figcaption></figure></section>
<section class="card"><div class="metrics"><div class="metric"><strong>{summary['validation_psnr_db']:.2f} dB</strong><span>HELD-OUT PSNR</span></div><div class="metric"><strong>{summary['validation_loss']:.3f}</strong><span>VALIDATION LOSS</span></div><div class="metric"><strong>{summary['image_count']}</strong><span>RECTIFIED VIEWS</span></div><div class="metric"><strong>{summary['point_count']:,}</strong><span>MPS INITIALIZER POINTS</span></div></div>
<h2>{quality_heading}</h2><p>Exact VRS RGB frames, online camera calibration, closed-loop MPS poses, metric points, COLMAP export and OpenSplat optimization work as one pipeline. Major surfaces and room layout align in the held-out view.</p>
<p class="notice">{quality_notice}</p>
<h2>Next quality gate</h2><p>{next_gate}</p>
<div><a class="button" href="training_summary.json">Training provenance JSON</a><a class="button" href="training.log">Training log</a><a class="button" href="cameras_{summary['iterations']}.json">Metric cameras JSON</a><a class="button" href="../contact_sheet.jpg">Frame contact sheet</a></div></section>
</main></body></html>'''
    (output / "index.html").write_text(page)


def main() -> None:
    args = parse_args()
    dataset = args.dataset_dir.resolve()
    opensplat = args.opensplat.resolve()
    output = (args.output_dir or dataset / "trained_proof").resolve()
    renders = output / "renders"
    output.mkdir(parents=True, exist_ok=True)
    renders.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset / "dataset_manifest.json"
    if not opensplat.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("OpenSplat and dataset manifest must exist.")
    dataset_manifest = json.loads(manifest_path.read_text())
    summary_path = output / "training_summary.json"
    if args.publish_only:
        if not summary_path.is_file():
            raise FileNotFoundError("Publish-only mode requires an existing training_summary.json.")
        summary = json.loads(summary_path.read_text())
        summary["scene_view"] = scene_view_from_manifest(
            dataset_manifest, summary["validation_image"]
        )
        source_splat = Path(summary["splat_ply"])
        summary["trained_splat_count"] = ply_vertex_count(source_splat)
        summary["viewer_splat_filename"] = source_splat.name
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        write_viewer_page(output, summary)
        write_page(output, summary)
        print(f"Republished Gaussian viewer: {output / 'viewer.html'}")
        return
    validation_image = resolve_validation_image(dataset_manifest, args.validation_image)
    image_path = dataset / "images" / validation_image
    if not image_path.is_file():
        raise FileNotFoundError(f"Validation image does not exist: {image_path}")
    splat = output / f"splat_{args.iterations}.ply"
    cameras = output / f"cameras_{args.iterations}.json"
    command = [
        str(opensplat),
        str(dataset),
        "-n",
        str(args.iterations),
        "-d",
        str(args.downscale),
        "--sh-degree",
        str(args.sh_degree),
        "--sh-degree-interval",
        str(args.sh_degree_interval),
        "--ssim-weight",
        "0.2",
        "--max-gaussians",
        str(args.max_gaussians),
        "--refine-every",
        str(args.refine_every),
        "--densify-from",
        str(args.densify_from),
        "--densify-until",
        str(args.densify_until),
        "--val",
        "--val-image",
        validation_image,
        "--val-render",
        str(renders),
        "-o",
        str(splat),
        "--output-cameras",
        str(cameras),
    ]
    if args.device == "cpu":
        command.insert(2, "--cpu")
    started = time.time()
    # Persist progress while training, including diagnostics if a long run is interrupted.
    with (output / "training.log").open("w") as log:
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:
            log.write(line)
            log.flush()
            if args.require_gpu and line.startswith("Using CPU"):
                process.terminate()
                process.wait()
                raise RuntimeError("GPU unavailable; stopped CPU fallback. Check GPU access before retrying.")
            step = re.match(r"Step (\d+):", line)
            if not step or int(step.group(1)) % 500 == 0:
                print(line, end="", flush=True)
        process.wait()
    runtime = time.time() - started
    training_log = (output / "training.log").read_text()
    if process.returncode:
        raise RuntimeError(f"OpenSplat exited with status {process.returncode}.")
    loss, psnr = parse_validation_metrics(training_log)
    render = renders / f"{args.iterations}.png"
    if not splat.is_file() or not cameras.is_file() or not render.is_file():
        raise RuntimeError("OpenSplat did not write every expected result.")
    trained_splat_count = ply_vertex_count(splat)
    backend_match = re.search(r"^Using\s+(.+)$", training_log, flags=re.MULTILINE)
    backend = backend_match.group(1).strip() if backend_match else args.device
    formal_run = args.iterations >= 7000 and args.downscale <= 1.0
    summary = {
        "status": (
            "motion_centered_full_resolution_reconstruction"
            if formal_run
            else "proof_of_pipeline_not_final_clinical_scene"
        ),
        "trial_id": dataset_manifest["trial_id"],
        "dataset_dir": str(dataset),
        "dataset": {
            "segment": dataset_manifest["segment"],
            "frames": {
                "selected": dataset_manifest["frames"]["selected"],
                "candidate_fps": dataset_manifest["frames"]["candidate_fps"],
            },
            "points": {
                "written": dataset_manifest["points"]["written"],
                "minimum_color_views": dataset_manifest["points"]["minimum_color_views"],
            },
        },
        "trainer": f"OpenSplat 1.2.2 {backend}",
        "backend": backend,
        "submap_name": dataset.name,
        "iterations": args.iterations,
        "downscale_factor": args.downscale,
        "validation_image": validation_image,
        "validation_loss": loss,
        "validation_psnr_db": psnr,
        "image_count": dataset_manifest["frames"]["selected"],
        "point_count": dataset_manifest["points"]["written"],
        "trained_splat_count": trained_splat_count,
        "viewer_splat_filename": splat.name,
        "scene_view": scene_view_from_manifest(dataset_manifest, validation_image),
        "training_parameters": {
            "spherical_harmonics_degree": args.sh_degree,
            "spherical_harmonics_degree_interval": args.sh_degree_interval,
            "maximum_gaussians": args.max_gaussians,
            "refine_every": args.refine_every,
            "densify_from": args.densify_from,
            "densify_until": args.densify_until,
        },
        "runtime_s": runtime,
        "splat_ply": str(splat),
        "camera_json": str(cameras),
        "held_out_render": str(render),
        "command": command,
        "interpretation": (
            "The held-out result validates a full-resolution motion-centered reconstruction. "
            "It supports visual context review but is not a complete or clinically validated "
            "geometric model."
            if formal_run
            else "The held-out result validates the calibrated data path and coarse scene "
            "alignment. It is intentionally under-trained and not a final clinical model."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    write_viewer_page(output, summary)
    write_page(output, summary)
    print(f"Published Gaussian proof: {output / 'index.html'}")


if __name__ == "__main__":
    main()
