#!/usr/bin/env python3
"""Export a display-only MPS point cloud in the existing playback world frame."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def filter_chunk(frame, graph_uid, path_tree, inv_std=.005, dist_std=.01, radius=8.0):
    if frame.graph_uid.isna().any() or not frame.graph_uid.eq(graph_uid).all():
        raise ValueError("Point cloud and playback must share the same MPS graph_uid.")
    xyz = frame[["px_world", "py_world", "pz_world"]].to_numpy(dtype=np.float64)
    quality = (np.isfinite(xyz).all(axis=1) & np.isfinite(frame.inv_dist_std) &
               np.isfinite(frame.dist_std) & (frame.inv_dist_std >= 0) &
               (frame.dist_std >= 0) & (frame.inv_dist_std <= inv_std) & (frame.dist_std <= dist_std))
    points = xyz[quality]
    count = len(points)
    if count:
        points = points[path_tree.query(points, k=1, workers=2)[0] <= radius]
    return points, count


def display_sample(points, voxel=.06, limit=120_000):
    if not len(points):
        raise ValueError("No supported points remain after filtering.")
    # Keep actual estimated landmarks, not newly synthesized voxel centers.
    keys = np.floor(points / voxel).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    unique = points[np.sort(indices)]
    count = len(unique)
    if count > limit:
        unique = unique[np.sort(np.random.default_rng(42).choice(count, limit, replace=False))]
    return unique, count


def export(trial_id, points_path, limit=120_000):
    output = ROOT / "runs" / trial_id / "environment_reconstruction"
    playback_path = output / "scene_playback_data.json"
    original = playback_path.read_bytes()
    data = json.loads(original)
    if data["trial_id"] != trial_id:
        raise ValueError("Trial mismatch")
    # Sample the calibrated playback route at approximately 5 Hz.
    path = np.asarray([p[:3] for p in data["poses"][::6] if p is not None])
    if not len(path) or not np.isfinite(path).all():
        raise ValueError("No valid playback route")
    tree = cKDTree(path)
    columns = ["graph_uid", "px_world", "py_world", "pz_world", "inv_dist_std", "dist_std"]
    retained, raw_count, quality_count, nearby_count = [], 0, 0, 0
    for frame in pd.read_csv(points_path, usecols=columns, chunksize=250_000):
        points, quality = filter_chunk(frame, data["provenance"]["graph_uid"], tree)
        raw_count += len(frame); quality_count += quality; nearby_count += len(points)
        if len(points):
            sampled, _ = display_sample(points, limit=len(points))
            retained.append(sampled)
        print(f"Read {raw_count:,} landmarks; {nearby_count:,} pass quality + route filters", flush=True)
    if not retained:
        raise ValueError("No points available for this trial")
    points, voxel_count = display_sample(np.concatenate(retained), limit=limit)
    digest = hashlib.sha256()
    with points_path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    payload = {
        "schema_version": 1, "trial_id": trial_id, "created_at": datetime.now(timezone.utc).isoformat(),
        "graph_uid": data["provenance"]["graph_uid"], "units": "meters", "up_axis": "Z",
        "source": str(points_path), "source_sha256": digest.hexdigest(),
        "playback_sha256": hashlib.sha256(original).hexdigest(),
        "scope": "Session-wide static MPS landmark map; not time-resolved surface coverage or a textured mesh.",
        "filtering": {"raw_count": raw_count, "quality_count": quality_count,
                      "near_route_count": nearby_count, "voxel_count": voxel_count,
                      "display_count": len(points), "inv_dist_std_max": .005,
                      "dist_std_max": .01, "max_distance_to_sampled_rgb_route_m": 8,
                      "route_sample_step_frames": 6, "voxel_size_m": .06, "random_seed": 42},
        "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "points": np.round(points, 4).tolist(),
    }
    destination = output / "spatial_overview_data.json"
    destination.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    if playback_path.read_bytes() != original:
        raise RuntimeError("Playback changed during the export; review before publishing.")
    print(f"Wrote {len(points):,} display landmarks to {destination}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=120_000)
    args = parser.parse_args()
    if not args.trial_id.replace("_", "").isalnum() or args.max_points < 1:
        parser.error("Invalid trial ID or maximum point count")
    export(args.trial_id, args.points, args.max_points)
