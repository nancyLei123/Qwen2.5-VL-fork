#!/usr/bin/env python3
"""Resumable, local-only calibrated submap export and sequential GPU training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from build_registered_scene_playback import publish, save
from export_aria_gaussian_dataset import export_dataset

ROOT = Path(__file__).resolve().parents[1]


def plan(trial, duration):
    jobs = []
    start = 2.0
    while start < duration - 2:
        end = min(start + 178, duration - 1.5)
        jobs.append({"id": f"home_{len(jobs) + 1:03d}", "start_s": start,
                     "end_s": end, "status": "pending"})
        if end >= duration - 1.5:
            break
        start = end - 20
    return {"trial_id": trial, "iterations": 7000, "image_size": [1024, 1024],
            "policy": "Independent overlapping submaps in one trial's MPS world frame. No cross-trial registration, no fabricated poses or gap coverage.",
            "jobs": jobs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", nargs="+", required=True)
    parser.add_argument("--opensplat", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    plans = {}
    for trial in args.trials:
        directory = ROOT / "runs" / trial / "environment_reconstruction"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "reconstruction_jobs.json"
        if path.exists():
            plans[trial] = json.loads(path.read_text())
        else:
            manifest = json.loads((directory.parent / "trial_manifest.json").read_text())
            metadata = json.loads(Path(manifest["extracted_media"]["metadata_json"]["path"]).read_text())
            plans[trial] = plan(trial, metadata["written_frames"] / metadata["fps"])
            save(path, plans[trial])
    # Round-robin gives every trial an initial scene before extending long sessions.
    for index in range(max(len(p["jobs"]) for p in plans.values())):
        for trial, progress in plans.items():
            if index >= len(progress["jobs"]):
                continue
            job = progress["jobs"][index]
            output = ROOT / "runs" / trial / "environment_reconstruction"
            dataset = output / "gaussian_submaps" / job["id"]
            trained = dataset / "trained_7000_fullres"
            def update(state, error=None):
                job["status"] = state
                job["updated_at"] = datetime.now(timezone.utc).isoformat()
                job["error"] = error
                save(output / "reconstruction_jobs.json", progress)
                print(f"{trial} {job['id']} ({job['start_s']:.1f}-{job['end_s']:.1f}s): {state}", flush=True)
            try:
                if (trained / "training_summary.json").exists():
                    update("complete")
                    if (output / "scene_playback_data.json").exists():
                        publish(trial)
                    continue
                if not (dataset / "dataset_manifest.json").exists():
                    update("exporting_calibrated_views")
                    options = argparse.Namespace(trial_id=trial, start_s=job["start_s"], end_s=job["end_s"],
                        candidate_fps=2.0, width=1024, height=1024, focal_px=512,
                        min_translation_m=.12, min_rotation_deg=8, max_frame_gap_s=2,
                        max_point_path_distance_m=8, max_points=150000, min_color_views=2,
                        inv_dist_std_max=.005, dist_std_max=.01, output_dir=dataset)
                    export_dataset(options)
                update("prepared")
                if args.prepare_only:
                    continue
                update("training")
                subprocess.run([sys.executable, "-u", str(ROOT / "scripts/train_aria_gaussian_scene.py"),
                    "--dataset-dir", str(dataset), "--output-dir", str(trained),
                    "--opensplat", str(args.opensplat.resolve()), "--iterations", "7000", "--downscale", "1",
                    "--device", "auto", "--require-gpu", "--validation-image", "auto",
                    "--max-gaussians", "350000", "--sh-degree", "2", "--sh-degree-interval", "1000",
                    "--refine-every", "100", "--densify-until", "5000"], check=True)
                update("complete")
                if (output / "scene_playback_data.json").exists():
                    publish(trial)
            except Exception as error:
                update("failed", str(error))
                raise


if __name__ == "__main__":
    main()
