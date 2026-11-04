"""Isolated, MPS-assisted Metashape baseline for existing Ntrial3 splat datasets.

Preparation requires only Python's standard library. Reconstruction requires an
activated Metashape Professional installation; this script never activates it.
"""

import argparse
import hashlib
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "runs/Ntrial3/environment_reconstruction"
DEFAULT_OUTPUT = ENV / "metashape_comparison_20261019"
SCENES = {"room": "sitting_context", "corridor": "corridor_walking", "stairs": "upstairs_context"}
PIPELINE_REVISION = 2


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def split_frames(manifest, validation_name):
    frames = manifest["frames"]["items"]
    if len({f["name"] for f in frames}) != len(frames):
        raise ValueError("Duplicate frame names")
    validation = [f for f in frames if f["name"] == validation_name]
    if len(validation) != 1:
        raise ValueError("Exactly one existing Gaussian validation frame is required")
    training = [f for f in frames if f["name"] != validation_name]
    if len(training) < 3:
        raise ValueError("Too few training frames")
    return training, validation[0]


def check_transform(matrix):
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("Expected a 4x4 transform")
    if not all(math.isfinite(v) for row in matrix for v in row):
        raise ValueError("Nonfinite transform")
    if any(abs(a - b) > 1e-8 for a, b in zip(matrix[3], (0, 0, 0, 1))):
        raise ValueError("Invalid homogeneous transform")
    for i in range(3):
        for j in range(3):
            if abs(sum(matrix[k][i] * matrix[k][j] for k in range(3)) - (i == j)) > 1e-5:
                raise ValueError("Rotation is not orthonormal")
    a, b, c = [row[:3] for row in matrix[:3]]
    det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])
    if abs(det - 1) > 1e-5:
        raise ValueError("Rotation must be right handed")


def point_bounds(path):
    low, high = [math.inf] * 3, [-math.inf] * 3
    count = 0
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith("#") or not line.strip():
                continue
            xyz = [float(v) for v in line.split()[1:4]]
            if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
                raise ValueError("Invalid MPS point")
            low = [min(a, b) for a, b in zip(low, xyz)]
            high = [max(a, b) for a, b in zip(high, xyz)]
            count += 1
    if not count:
        raise ValueError("Empty initializer point set")
    return {"min": low, "max": high, "point_count": count, "padding_m": 0.5}


def validate_output(output, datasets):
    output = Path(output).resolve()
    for dataset in datasets:
        dataset = Path(dataset).resolve()
        if output == dataset or dataset in output.parents or output in dataset.parents:
            raise ValueError("Comparison output must be separate from the source datasets")
    return output


def verify_sources(plan):
    changed = [p for p, expected in plan["protected_sha256"].items() if not Path(p).is_file() or digest(p) != expected]
    if changed:
        raise RuntimeError("Source changed since preparation: " + ", ".join(changed[:5]))


def prepare(output):
    datasets = [ENV / "gaussian_submaps" / v for v in SCENES.values()]
    output = validate_output(output, datasets)
    plan_path = output / "comparison_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        verify_sources(plan)
        return plan
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trial": "Ntrial3", "status": "prepared_not_reconstructed",
        "method": "Metashape depth maps and textured mesh with fixed MPS poses and rectified intrinsics",
        "not_tested": "Metashape independent image alignment / bundle adjustment",
        "output_dir": str(output), "scenes": {}, "protected_sha256": {},
        "limitations": [
            "One existing withheld image per scene: exploratory, not a robust benchmark.",
            "MPS poses/geometry use the recording; Gaussian initializer colors may include the withheld image. This is not a fully held-out end-to-end reconstruction test.",
            "The withheld image is excluded from Metashape depth estimation and texturing and was excluded from Gaussian optimization.",
            "Mesh coverage and image fidelity must be reported separately; missing mesh pixels must not silently disappear from scoring.",
            "No ground-truth surface measurements; appearance scores do not establish geometric or clinical validity.",
        ],
    }
    for label, name in SCENES.items():
        dataset = ENV / "gaussian_submaps" / name
        manifest_path = dataset / "dataset_manifest.json"
        baseline = dataset / "trained_7000_fullres"
        summary_path = baseline / "training_summary.json"
        manifest = json.loads(manifest_path.read_text())
        summary = json.loads(summary_path.read_text())
        training, validation = split_frames(manifest, summary["validation_image"])
        for frame in manifest["frames"]["items"]:
            check_transform(frame["T_world_camera"])
        destination = output / label
        destination.mkdir(exist_ok=True)
        source_image = dataset / "images" / validation["name"]
        gaussian_image = Path(summary["held_out_render"])
        shutil.copy2(source_image, destination / "reference.jpg")
        shutil.copy2(gaussian_image, destination / "gaussian.png")
        region = point_bounds(dataset / "sparse/0/points3D.txt")
        plan["scenes"][label] = {
            "dataset_dir": str(dataset), "submap": name, "segment": manifest["segment"],
            "camera": manifest["camera"], "training_frame_count": len(training),
            "validation_frame": validation, "training_frames": training, "region": region,
            "gaussian_summary": str(summary_path), "gaussian_iterations": summary["iterations"],
            "gaussian_reported_psnr_db": summary["validation_psnr_db"],
        }
        protected = [manifest_path, summary_path, gaussian_image, Path(summary["splat_ply"]), dataset / "sparse/0/points3D.txt"]
        protected += [dataset / "images" / f["name"] for f in manifest["frames"]["items"]]
        plan["protected_sha256"].update({str(p): digest(p) for p in protected})
    write_json(plan_path, plan)
    return plan


def make_calibration(ms, intrinsics):
    calibration = ms.Calibration()
    calibration.type = ms.Sensor.Type.Frame
    calibration.width, calibration.height = intrinsics["width"], intrinsics["height"]
    calibration.f = intrinsics["fy"]
    calibration.b1 = intrinsics["fx"] - intrinsics["fy"]
    calibration.cx = intrinsics["cx"] - intrinsics["width"] / 2
    calibration.cy = intrinsics["cy"] - intrinsics["height"] / 2
    return calibration


def validate_projection(ms, camera, frame, intrinsics):
    transform = ms.Matrix(frame["T_world_camera"])
    for x, y, z in [(0, 0, 2), (0.2, -0.1, 2), (-0.5, 0.4, 3)]:
        observed = camera.project(transform.mulp(ms.Vector((x, y, z))))
        expected = (intrinsics["fx"] * x / z + intrinsics["cx"], intrinsics["fy"] * y / z + intrinsics["cy"])
        if observed is None or max(abs(observed[i] - expected[i]) for i in range(2)) > 1e-4:
            raise RuntimeError("Metashape projection convention does not match Aria; aborting")


def reconstruct(ms, plan, label, downscale):
    output = Path(plan["output_dir"]) / label
    state_file = output / "metashape_status.json"
    if state_file.exists():
        saved = json.loads(state_file.read_text())
        if saved.get("status") == "complete" and saved.get("pipeline_revision") == PIPELINE_REVISION:
            if saved.get("depth_downscale") != downscale:
                raise ValueError("Use a separate --output-dir when changing reconstruction settings")
            print(label + ": already complete", flush=True)
            return
    scene = plan["scenes"][label]
    ms.app.settings.log_path = str(output / "metashape.log")
    ms.app.settings.log_enable = True
    started = time.time()
    state = {"scene": label, "status": "running", "version": ms.app.version,
             "pipeline_revision": PIPELINE_REVISION,
             "depth_downscale": downscale, "texture_size": 4096,
             "texture_downscale": 1, "texture_hole_filling": False,
             "interpolation": "disabled", "poses": "fixed MPS", "stage": "setup"}

    def stage(name):
        state.update(stage=name, elapsed_s=round(time.time() - started, 2))
        write_json(state_file, state)
        print(f"[{label}] {name}", flush=True)

    doc = ms.Document()
    project = output / "reconstruction.psx"
    calibration = make_calibration(ms, scene["camera"])
    # Save As reloads PSX internally; create chunk handles only after that call.
    doc.save(str(project))
    chunk = doc.addChunk()
    chunk.label = label + " / fixed MPS calibration"
    chunk.transform.matrix = ms.Matrix.Diag([1, 1, 1, 1])
    frames = scene["training_frames"]
    chunk.addPhotos([str(Path(scene["dataset_dir"]) / "images" / f["name"]) for f in frames])
    by_name = {f["name"]: f for f in frames}
    for sensor in chunk.sensors:
        sensor.width, sensor.height = calibration.width, calibration.height
        sensor.type = ms.Sensor.Type.Frame
        sensor.user_calib = calibration
        sensor.calibration = calibration
        sensor.fixed = True
    for camera in chunk.cameras:
        frame = by_name[Path(camera.photo.path).name]
        camera.transform = ms.Matrix(frame["T_world_camera"])
        validate_projection(ms, camera, frame, scene["camera"])
    region = ms.Region()
    bounds = scene["region"]
    region.center = ms.Vector([(a + b) / 2 for a, b in zip(bounds["min"], bounds["max"])])
    region.size = ms.Vector([b - a + 2 * bounds["padding_m"] for a, b in zip(bounds["min"], bounds["max"])])
    region.rot = ms.Matrix.Diag([1, 1, 1])
    chunk.region = region
    try:
        stage("save calibrated input")
        doc.save()
        stage("image matches and tie points with fixed MPS cameras")
        chunk.matchPhotos(downscale=1, generic_preselection=True, reference_preselection=False,
                          keypoint_limit=40000, tiepoint_limit=4000, keep_keypoints=True)
        chunk.triangulateTiePoints()
        for camera in chunk.cameras:
            frame = by_name[Path(camera.photo.path).name]
            if max(abs(camera.transform[i, j] - frame["T_world_camera"][i][j])
                   for i in range(4) for j in range(4)) > 1e-6:
                raise RuntimeError("MPS camera transform changed during triangulation")
            validate_projection(ms, camera, frame, scene["camera"])
        state["tie_points"] = len(chunk.tie_points.points)
        doc.save()
        stage("depth maps")
        chunk.buildDepthMaps(downscale=downscale, filter_mode=ms.MildFiltering, max_neighbors=16)
        doc.save()
        stage("mesh without hole filling")
        chunk.buildModel(source_data=ms.DepthMapsData, surface_type=ms.Arbitrary,
                         interpolation=ms.DisabledInterpolation, face_count=ms.HighFaceCount)
        doc.save()
        stage("UV and texture")
        chunk.buildUV(mapping_mode=ms.GenericMapping)
        finish_reconstruction(ms, doc, chunk, scene, calibration, output, state, stage, started)
    except Exception as exc:
        state.update(status="failed", error=str(exc), elapsed_s=round(time.time() - started, 2))
        write_json(state_file, state)
        raise


def finish_reconstruction(ms, doc, chunk, scene, calibration, output, state, stage, started):
    stage("full-resolution texture without hole filling")
    chunk.buildTexture(blending_mode=ms.MosaicBlending, texture_size=4096,
                       downscale=1, fill_holes=False)
    doc.save()
    stage("withheld-view render")
    validation_transform = ms.Matrix(scene["validation_frame"]["T_world_camera"])
    chunk.model.renderImage(validation_transform, calibration, add_alpha=True).save(str(output / "metashape_rgba.png"))
    stage("export textured model")
    chunk.exportModel(str(output / "model.obj"), binary=False, texture_format=ms.ImageFormatJPEG, save_texture=True)
    check = ms.Document()
    check.open(str(output / "reconstruction.psx"), read_only=True)
    if check.chunk.model is None or len(check.chunk.model.faces) != len(chunk.model.faces):
        raise RuntimeError("Saved project did not preserve the reconstructed mesh")
    state.update(status="complete", faces=len(chunk.model.faces), vertices=len(chunk.model.vertices),
                 elapsed_s=round(time.time() - started, 2))
    write_json(output / "metashape_status.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run", action="store_true", help="Requires an already activated Metashape license")
    parser.add_argument("--scene", choices=["all", *SCENES], default="all")
    parser.add_argument("--depth-downscale", type=int, choices=[1, 2, 4], default=1)
    args = parser.parse_args()
    plan = prepare(args.output_dir)
    print("Prepared: " + str(args.output_dir), flush=True)
    for label, scene in plan["scenes"].items():
        print(f"{label}: {scene['training_frame_count']} training images + {scene['validation_frame']['name']} withheld", flush=True)
    if not args.run:
        return
    import Metashape as ms
    if not ms.app.activated:
        raise RuntimeError("Metashape is not activated. Activate an authorized trial/license in the official application first.")
    devices = ms.app.enumGPUDevices()
    ms.app.gpu_mask = sum(1 << i for i in range(len(devices)))
    ms.app.cpu_enable = not bool(devices)
    write_json(Path(plan["output_dir"]) / "runtime.json", {"version": ms.app.version, "gpu_devices": devices, "gpu_mask": ms.app.gpu_mask})
    try:
        for label in (SCENES if args.scene == "all" else [args.scene]):
            reconstruct(ms, plan, label, args.depth_downscale)
    finally:
        verify_sources(plan)
        print("Verified original Gaussian results and input images are unchanged.", flush=True)


if __name__ == "__main__":
    main()
