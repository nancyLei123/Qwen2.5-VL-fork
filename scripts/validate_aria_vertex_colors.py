#!/usr/bin/env python3
"""Small, auditable Gen1 VRS-to-vertex-color experiment; never a room texture atlas."""

from __future__ import annotations

import argparse
import html
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import minimum_filter
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation, Slerp
import trimesh

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "VRS_Pre_Processing" / "data"


def interpolate_pose(times_ns, translations, quaternions_xyzw, query_ns, max_gap_ns=20_000_000):
    """Interpolate only supported timestamps; do not extrapolate across missing poses."""
    if query_ns < times_ns[0] or query_ns > times_ns[-1]:
        raise ValueError("Image timestamp is outside the supported trajectory.")
    b = int(np.searchsorted(times_ns, query_ns, side="left"))
    a = max(0, b - 1)
    span = int(times_ns[b] - times_ns[a])
    if span > max_gap_ns:
        raise ValueError("Image timestamp crosses a large pose gap.")
    fraction = (query_ns - int(times_ns[a])) / span if span else 0.0
    transform = np.eye(4)
    transform[:3, 3] = translations[a] * (1 - fraction) + translations[b] * fraction
    if a == b:
        transform[:3, :3] = Rotation.from_quat(quaternions_xyzw[a]).as_matrix()
    else:
        transform[:3, :3] = Slerp(
            [0.0, 1.0], Rotation.from_quat(quaternions_xyzw[[a, b]])
        )([fraction]).as_matrix()[0]
    return transform, span


def world_to_camera(points, transform_world_camera):
    return (points - transform_world_camera[:3, 3]) @ transform_world_camera[:3, :3]


def project_points(points_world, transform_world_camera, camera):
    points_camera = world_to_camera(points_world, transform_world_camera)
    uv = np.full((len(points_world), 2), np.nan)
    for i in np.flatnonzero(points_camera[:, 2] > 0):
        pixel = camera.project(points_camera[i])
        if pixel is not None:
            uv[i] = pixel
    return uv, np.linalg.norm(points_camera, axis=1)


def static_pixel_mask(uv, size):
    """Conservative hand/phone rectangle in this manually inspected short view."""
    width, height = size
    return (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 8) & (uv[:, 0] < width - 8)
        & (uv[:, 1] >= 8) & (uv[:, 1] < height - 8)
        & ~((uv[:, 0] >= 750) & (uv[:, 1] < 850))
    )


def point_depth_visibility(uv, ranges, size, bin_px=4, tolerance_m=0.04):
    """Approximate occlusion among supported landmarks, not a dense visibility proof."""
    width, height = size
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(ranges) & (ranges > 0)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    ids = np.flatnonzero(valid)
    result = np.zeros(len(uv), dtype=bool)
    if not len(ids):
        return result
    cols = (width + bin_px - 1) // bin_px
    rows = (height + bin_px - 1) // bin_px
    cells = (uv[ids] // bin_px).astype(int)
    depth = np.full((rows, cols), np.inf)
    np.minimum.at(depth, (cells[:, 1], cells[:, 0]), ranges[ids])
    depth = minimum_filter(depth, size=3, mode="constant", cval=np.inf)
    nearest = depth[cells[:, 1], cells[:, 0]]
    result[ids] = ranges[ids] <= nearest + np.maximum(tolerance_m, ranges[ids] * 0.01)
    return result


def bilinear_rgb(image, uv):
    x, y = uv.T
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, image.shape[1] - 1), np.minimum(y0 + 1, image.shape[0] - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return (
        image[y0, x0] * (1 - fx) * (1 - fy) + image[y0, x1] * fx * (1 - fy)
        + image[y1, x0] * (1 - fx) * fy + image[y1, x1] * fx * fy
    )


def fuse_colors(samples, min_views=3, max_mad=18.0):
    count = np.isfinite(samples[:, :, 0]).sum(axis=0)
    median = np.zeros(samples.shape[1:], dtype=float)
    mad = np.full(samples.shape[1], np.inf)
    eligible = count >= min_views
    if eligible.any():
        median[eligible] = np.nanmedian(samples[:, eligible], axis=0)
        mad[eligible] = np.nanmedian(
            np.max(np.abs(samples[:, eligible] - median[eligible]), axis=2), axis=0
        )
    return np.rint(median).clip(0, 255).astype(np.uint8), count, mad, eligible & (mad <= max_mad)


def supported_triangles(points, pixels, max_edge_m=0.15, min_area_px=2.0):
    faces = Delaunay(pixels).simplices
    xyz = points[faces]
    longest = np.maximum.reduce([
        np.linalg.norm(xyz[:, 0] - xyz[:, 1], axis=1),
        np.linalg.norm(xyz[:, 1] - xyz[:, 2], axis=1),
        np.linalg.norm(xyz[:, 2] - xyz[:, 0], axis=1),
    ])
    uv = pixels[faces]
    e1, e2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    area = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) / 2
    return faces[(longest <= max_edge_m) & (area >= min_area_px)], len(faces)


def read_observation_subset(path, requested_times_us, cache_dir):
    identity = {
        "path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
        "timestamps_us": sorted(map(int, requested_times_us)),
    }
    meta, cache = cache_dir / "observation_cache.json", cache_dir / "selected_observations.csv"
    if meta.exists() and cache.exists() and json.loads(meta.read_text()) == identity:
        return pd.read_csv(cache, dtype={"camera_serial": str})
    subsets = []
    for i, chunk in enumerate(pd.read_csv(path, chunksize=250_000, dtype={"camera_serial": str})):
        subset = chunk[chunk.frame_tracking_timestamp_us.isin(requested_times_us)]
        if len(subset):
            subsets.append(subset)
        if i % 20 == 0:
            print(f"Observation scan: {(i + 1) * 250_000:,} rows (bounded-memory full scan)", flush=True)
    if not subsets:
        raise RuntimeError("No observations match the selected SLAM exposure timestamps.")
    result = pd.concat(subsets, ignore_index=True)
    result.to_csv(cache, index=False)
    meta.write_text(json.dumps(identity, indent=2) + "\n")
    return result


def annotated_rgb(image, uv, keep, output, record_s):
    source = Image.fromarray(image)
    draw = ImageDraw.Draw(source, "RGBA")
    draw.rectangle((750, 0, image.shape[1], 850), fill=(25, 25, 25, 130))
    for u, v in uv[keep][::max(1, int(keep.sum()) // 2500)]:
        draw.ellipse((u - 2, v - 2, u + 2, v + 2), fill=(57, 244, 175, 220))
    source = source.transpose(Image.Transpose.ROTATE_270)
    source.thumbnail((900, 900))
    draw = ImageDraw.Draw(source)
    draw.rectangle((0, 0, source.width, 46), fill="#17231c")
    draw.text((12, 10), f"RGB at {record_s:.3f}s | green = projected supported landmarks", fill="white")
    draw.text((12, 27), "Dark region excluded: hand / phone. Display rotated only.", fill="white")
    source.save(output)


def write_frame_contact_sheet(frames, output):
    sheet = Image.new("RGB", (4 * 280, 2 * 310), "#f6f4eb")
    for i, frame in enumerate(frames):
        image = Image.fromarray(frame["image"]).transpose(Image.Transpose.ROTATE_270)
        image.thumbnail((280, 280))
        x, y = (i % 4) * 280, (i // 4) * 310
        sheet.paste(image, (x, y + 30))
        ImageDraw.Draw(sheet).text((x + 8, y + 8), f'{frame["recording_s"]:.3f}s (original VRS)', fill="#17231c")
    sheet.save(output)


def write_experiment_page(output, mesh, points, colors, count, summary):
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs
    import plotly.io as pio
    f = mesh.faces
    v = mesh.vertices
    rgba = mesh.visual.vertex_colors
    rgb_strings = [f"rgb({r},{g},{b})" for r, g, b in colors]
    traces = [
        go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                  vertexcolor=rgba[:, :3], name="Local vertex-colored patch", flatshading=True, visible=False,
                  lighting={"ambient": 1, "diffuse": 0}, hoverinfo="skip"),
        go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers", visible=True,
                     marker={"size": 2, "color": rgb_strings}, customdata=count,
                     name="RGB sampled on MPS landmarks",
                     hovertemplate="%{customdata} accepted views<br>XYZ %{x:.2f}, %{y:.2f}, %{z:.2f} m<extra></extra>"),
    ]
    figure = go.Figure(traces)
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 20, "b": 0}, paper_bgcolor="#f6f4eb", showlegend=False,
        scene={"aspectmode": "data", "xaxis": {"title": "X (m)"}, "yaxis": {"title": "Y (m)"},
               "zaxis": {"title": "Z / up (m)"}, "camera": {"eye": {"x": 1.3, "y": -1.7, "z": 0.9}}},
        uirevision="local-rgb",
    )
    validation = summary["slam_reprojection"]
    rows = "".join(
        f'<tr><td>{html.escape(x["label"])}</td><td>{x["recording_s"]:.3f}s</td>'
        f'<td>{x["n_valid"]:,}</td><td>{x["median_px"]:.2f}</td><td>{x["p95_px"]:.2f}</td></tr>'
        for x in validation["frames"]
    )
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Ntrial3 RGB vertex-color test</title>
<style>
*{box-sizing:border-box}body{margin:0;font-family:"Avenir Next","Gill Sans",sans-serif;color:#17231c;background:linear-gradient(120deg,#f4efe4,#eaf2eb)}
main{max-width:1500px;margin:auto;padding:24px;display:grid;gap:20px}.card{background:#fffffff0;border:1px solid #d8e1d7;border-radius:24px;padding:24px;min-width:0}
h1{margin:12px 0;font-size:34px;letter-spacing:-.04em}h2{margin:0 0 12px}p{line-height:1.6;color:#627067}a{color:#187859}button,a.button{display:inline-block;padding:10px 15px;border:1px solid #d8e1d7;background:white;border-radius:99px;color:#17231c;font:inherit;font-weight:800;cursor:pointer;text-decoration:none}
.toolbar{display:flex;gap:8px;flex-wrap:wrap}.toolbar button[aria-pressed="true"]{background:#187859;color:white;border-color:#187859}.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(280px,420px);gap:20px}.grid>*{min-width:0}.grid img{width:100%;border-radius:14px}.notice{padding:14px;border-left:4px solid #c79027;background:#fff7e5}#rgbPlot{height:700px;width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;text-align:left;border-bottom:1px solid #dde5dc}.table-wrap{overflow-x:auto}.metrics{font-size:18px;font-weight:800}.downloads{display:flex;gap:10px;flex-wrap:wrap}
@media(max-width:800px){main{padding:10px}.card{padding:16px}.grid{grid-template-columns:1fr}#rgbPlot{height:480px}h1{font-size:28px}}
</style><script>__PLOTLY__</script></head><body><main>
<header class="card"><a class="button" href="../index.html">Back to Environment</a><h1>Local RGB Vertex-Color Test</h1>
<p>Ntrial3 · original recording __WINDOW__ · raw VRS pixels, exact device times and online camera calibration. No UV atlas or full-room texture.</p>
<p class="metrics">__METRICS__</p><p class="notice">Colors are an appearance experiment on a partial surface. SLAM reprojection tests the coordinate/calibration chain; RGB alignment is visually checked, not independently ground-truth validated. Hidden surfaces and moving objects remain a limitation.</p></header>
<section class="card grid"><div><h2>Vertex colors, not a pasted photograph</h2><div class="toolbar"><button id="mesh" aria-pressed="false">Colored triangles</button><button id="points" aria-pressed="true">Colored landmarks</button><button id="minus" aria-label="Zoom out">−</button><button id="plus" aria-label="Zoom in">+</button><button id="reset">Reset view</button></div><div id="rgbPlot"></div><p>Drag to rotate; use + / − to zoom. No wheel zoom. Vertex XYZ comes from MPS; triangles longer than 0.15 m are rejected. Unseen geometry is not filled in.</p></div>
<aside><h2>Projection check</h2><a href="rgb_projection_overlay.jpg"><img src="rgb_projection_overlay.jpg" alt="Raw RGB frame with supported landmark projections and foreground exclusion"></a><p>Green dots: camera-projected MPS landmarks. Dark area: conservative hand/phone exclusion. Rotation is only for display.</p><a href="rgb_reference.jpg"><img src="rgb_reference.jpg" alt="Unannotated RGB reference frame"></a></aside></section>
<section class="card"><h2>What was checked?</h2><p>Each row compares projected 3D landmarks with their recorded grayscale SLAM-camera pixels at the same exposure timestamp. These are input-consistency errors, not independent 3D accuracy or RGB pixel errors.</p><a href="slam_projection_overlay.png">Inspect recorded pixels versus projections</a> · <a href="rgb_frame_contact_sheet.jpg">Inspect all seven RGB views</a><div class="table-wrap"><table><thead><tr><th>Camera</th><th>Recording time</th><th>Points</th><th>Median (px)</th><th>95th percentile (px)</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p>__DETAILS__</p><div class="downloads"><a class="button" href="local_rgb_vertex_mesh.glb">Vertex-color GLB</a><a class="button" href="local_rgb_vertex_mesh.ply">Vertex-color PLY</a><a class="button" href="rgb_supported_points.ply">Colored point PLY</a><a class="button" href="validation_summary.json">Validation / provenance JSON</a><a class="button" href="../../../../ARIA_ENVIRONMENT_RECONSTRUCTION.md">English methodology</a></div></section>
</main><script>const fig=__FIGURE__;const plot=document.getElementById('rgbPlot');Plotly.newPlot(plot,fig.data,fig.layout,{responsive:true,scrollZoom:false,displayModeBar:false});
function layer(mesh){Plotly.restyle(plot,{visible:[mesh,!mesh]});document.getElementById('mesh').setAttribute('aria-pressed',String(mesh));document.getElementById('points').setAttribute('aria-pressed',String(!mesh))}
document.getElementById('mesh').onclick=()=>layer(true);document.getElementById('points').onclick=()=>layer(false);
function zoom(f){const eye=plot.layout.scene.camera.eye;const r=Math.hypot(eye.x,eye.y,eye.z)*f;if(r<.15||r>12)return;Plotly.relayout(plot,{'scene.camera.eye':{x:eye.x*f,y:eye.y*f,z:eye.z*f}})}
document.getElementById('plus').onclick=()=>zoom(.8);document.getElementById('minus').onclick=()=>zoom(1.25);document.getElementById('reset').onclick=()=>Plotly.relayout(plot,{'scene.camera.eye':{x:1.3,y:-1.7,z:.9}});
</script></body></html>'''
    metrics = f'{len(points):,} colored landmarks · {len(v):,} mesh vertices · {len(f):,} short-edge triangles'
    details = (f'Colors require at least 3 of {summary["rgb_frame_count"]} views and median RGB deviation ≤18/255. '
               'A sparse landmark depth test is used; it cannot guarantee dense occlusion. '
               'RGB poses subtract the online camera time offset before SLERP. '
               'Coverage is sparse: the landmark view opens first; only supported small triangles are exported. '
               'No original video, manual label, classification or height mesh is modified.')
    for token, value in {
        "__PLOTLY__": get_plotlyjs(), "__FIGURE__": pio.to_json(figure).replace("</", "<\\/"),
        "__METRICS__": metrics, "__ROWS__": rows, "__DETAILS__": html.escape(details),
        "__WINDOW__": "–".join(f"{x:g}" for x in summary["recording_window_s"]) + " s",
    }.items():
        page = page.replace(token, value)
    output.write_text(page)


def main():
    from projectaria_tools.core import data_provider, mps
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-id", default="Ntrial3", choices=["Ntrial3"])
    parser.add_argument("--start-s", type=float, default=6.0)
    parser.add_argument("--end-s", type=float, default=12.0)
    args = parser.parse_args()
    if not 5 <= args.start_s < args.end_s <= 15:
        parser.error("The manually inspected foreground mask is valid only inside 5–15 s.")
    output = ROOT / "runs" / args.trial_id / "environment_reconstruction" / "rgb_vertex_color_trial"
    output.mkdir(parents=True, exist_ok=True)
    inputs = DATA_ROOT / args.trial_id
    vrs = inputs / "NTrial_3.vrs"
    slam = inputs / "mps_NTrial_3_vrs" / "slam"
    provider = data_provider.create_vrs_data_provider(str(vrs))
    if provider is None:
        raise RuntimeError("VRS provider could not open the recording.")
    rgb_sid = provider.get_stream_id_from_label("camera-rgb")
    first_image, first_record = provider.get_image_data_by_index(rgb_sid, 0)
    origin_ns = int(first_record.capture_timestamp_ns)
    calibrations = mps.read_online_calibration(str(slam / "online_calibration.jsonl"))
    calibration_ns = np.array([round(c.tracking_timestamp.total_seconds() * 1e9) for c in calibrations], dtype=np.int64)
    trajectory = pd.read_csv(slam / "closed_loop_trajectory.csv")
    if trajectory.graph_uid.nunique() != 1:
        raise ValueError("Multiple world frames cannot be mixed.")
    times = trajectory.tracking_timestamp_us.to_numpy(dtype=np.int64) * 1000
    if not (np.diff(times) > 0).all():
        raise ValueError("Trajectory timestamps must be strictly increasing.")
    xyz = trajectory[[f"t{a}_world_device" for a in "xyz"]].to_numpy()
    quats = trajectory[[f"q{a}_world_device" for a in "xyzw"]].to_numpy()

    def at_camera(capture_ns, label, shape):
        idx = int(np.argmin(np.abs(calibration_ns - capture_ns)))
        if abs(int(calibration_ns[idx]) - capture_ns) > 60_000_000:
            raise ValueError("No temporally nearby online calibration.")
        cam = calibrations[idx].get_camera_calib(label)
        if cam is None or list(cam.get_image_size()) != [shape[1], shape[0]]:
            raise ValueError("Online calibration does not match original frame dimensions.")
        corrected_ns = capture_ns - round(cam.get_time_offset_sec_device_camera() * 1e9)
        pose, gap = interpolate_pose(times, xyz, quats, corrected_ns)
        pose_idx = int(np.searchsorted(times, corrected_ns))
        if trajectory.quality_score.iloc[max(0, pose_idx - 1):pose_idx + 1].min() < 0.9:
            raise ValueError("Pose quality is too low for this experiment.")
        return cam, pose @ cam.get_transform_device_camera().to_matrix(), corrected_ns, gap

    rgb_frames, slam_frames, timestamp_set = [], [], set()
    for requested_s in np.linspace(args.start_s, args.end_s, 7):
        query = origin_ns + round(requested_s * 1e9)
        image, record = provider.get_image_data_by_time_ns(rgb_sid, query, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)
        if not image.is_valid():
            raise ValueError("Selected RGB image is invalid.")
        pixels = image.to_numpy_array()
        cam, pose, corrected, gap = at_camera(int(record.capture_timestamp_ns), "camera-rgb", pixels.shape)
        frames_for_rgb = []
        for label in ["camera-slam-left", "camera-slam-right"]:
            sid = provider.get_stream_id_from_label(label)
            gray, rec = provider.get_image_data_by_time_ns(sid, corrected, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)
            gray_pixels = gray.to_numpy_array()
            c, t, _, _ = at_camera(int(rec.capture_timestamp_ns), label, gray_pixels.shape)
            frame = {"label": label, "image": gray_pixels, "cam": c, "pose": t,
                     "timestamp_us": int(rec.capture_timestamp_ns) // 1000,
                     "recording_s": (int(rec.capture_timestamp_ns) - origin_ns) / 1e9}
            slam_frames.append(frame)
            frames_for_rgb.append(frame)
            timestamp_set.add(frame["timestamp_us"])
        rgb_frames.append({"image": pixels, "cam": cam, "pose": pose, "capture_ns": int(record.capture_timestamp_ns),
                           "corrected_ns": corrected, "gap_ns": gap, "slam": frames_for_rgb,
                           "exposure_duration_s": float(record.exposure_duration),
                           "recording_s": (int(record.capture_timestamp_ns) - origin_ns) / 1e9})
    print("VRS restored: 7 exact RGB frames and 14 matched SLAM frames loaded.", flush=True)
    observations = read_observation_subset(slam / "semidense_observations.csv.gz", timestamp_set, output)
    point_table = pd.read_csv(slam / "semidense_points.csv.gz")
    point_table = point_table[point_table.uid.isin(observations.uid)]
    point_table = point_table[(point_table.inv_dist_std <= .005) & (point_table.dist_std <= .01)]
    point_table = point_table.replace([np.inf, -np.inf], np.nan).dropna()
    if set(point_table.graph_uid) != set(trajectory.graph_uid):
        raise ValueError("Point and pose world frames differ.")
    point_table = point_table.set_index("uid")
    uids = point_table.index.to_numpy()
    positions = point_table[[f"p{a}_world" for a in "xyz"]].to_numpy()
    validations, all_errors = [], []
    for frame in slam_frames:
        obs = observations[(observations.frame_tracking_timestamp_us == frame["timestamp_us"])
                           & (observations.camera_serial == frame["cam"].get_serial_number())]
        obs = obs[obs.uid.isin(uids)].drop_duplicates("uid")
        p = point_table.loc[obs.uid, [f"p{a}_world" for a in "xyz"]].to_numpy()
        uv, _ = project_points(p, frame["pose"], frame["cam"])
        valid = np.isfinite(uv).all(axis=1)
        error = np.linalg.norm(uv[valid] - obs[["u", "v"]].to_numpy()[valid], axis=1)
        if not len(error):
            raise RuntimeError("No valid SLAM reprojection evidence.")
        all_errors.extend(error)
        frame["supported_uids"] = set(obs.uid.to_numpy()[valid][error < 2.0])
        validations.append({"label": frame["label"], "recording_s": frame["recording_s"],
                            "n_valid": int(len(error)), "n_observed": len(obs),
                            "median_px": float(np.median(error)), "p95_px": float(np.quantile(error, .95))})
        if frame is slam_frames[6]:
            overlay = Image.fromarray(frame["image"]).convert("RGB")
            draw = ImageDraw.Draw(overlay)
            for observed_uv, projected_uv in zip(obs[["u", "v"]].to_numpy()[valid][::10], uv[valid][::10]):
                u, v = observed_uv
                draw.ellipse((u-3, v-3, u+3, v+3), outline="#f946ac", width=1)
                u, v = projected_uv
                draw.line((u-2, v, u+2, v), fill="#35f2a1", width=1)
                draw.line((u, v-2, u, v+2), fill="#35f2a1", width=1)
            draw.rectangle((0, 0, overlay.width, 26), fill="#17231c")
            draw.text((8, 8), "Pink circles: recorded pixels | green crosses: 3D reprojection", fill="white")
            overlay.save(output / "slam_projection_overlay.png")
    median_error, p95_error = float(np.median(all_errors)), float(np.quantile(all_errors, .95))
    if median_error > 1.0 or p95_error > 3.0:
        raise RuntimeError(f"Projection validation failed: median={median_error:.3f}, p95={p95_error:.3f} px.")
    samples = np.full((len(rgb_frames), len(uids), 3), np.nan)
    frame_summaries = []
    for i, frame in enumerate(rgb_frames):
        uv, ranges = project_points(positions, frame["pose"], frame["cam"])
        observed = set.union(*(x["supported_uids"] for x in frame["slam"]))
        support = np.isin(uids, list(observed))
        mask = support & static_pixel_mask(uv, frame["cam"].get_image_size()) & (ranges >= .5) & (ranges <= 8.0)
        visibility = np.zeros(len(uids), dtype=bool)
        visibility[mask] = point_depth_visibility(uv[mask], ranges[mask], frame["cam"].get_image_size())
        accepted = mask & visibility
        samples[i, accepted] = bilinear_rgb(frame["image"].astype(float), uv[accepted])
        frame["uv"], frame["accepted"] = uv, accepted
        frame_summaries.append({"recording_s": frame["recording_s"], "capture_device_ns": frame["capture_ns"],
                                "corrected_pose_device_ns": frame["corrected_ns"],
                                "camera_time_offset_s": frame["cam"].get_time_offset_sec_device_camera(),
                                "readout_time_s": frame["cam"].get_readout_time_sec(),
                                "exposure_duration_s": frame["exposure_duration_s"],
                                "camera_projection_model": str(frame["cam"].get_model_name()),
                                "camera_projection_params": frame["cam"].get_projection_params().tolist(),
                                "exposure_model": "center-time pose; row-wise rolling shutter not corrected",
                                "pose_gap_ns": frame["gap_ns"], "accepted_landmarks": int(accepted.sum()),
                                "transform_world_camera": frame["pose"].tolist()})
        print(f"RGB {frame['recording_s']:.3f}s: accepted {accepted.sum():,} landmarks", flush=True)
    colors, count, mad, kept = fuse_colors(samples)
    reference = rgb_frames[len(rgb_frames) // 2]
    kept &= reference["accepted"]
    if kept.sum() < 100:
        raise RuntimeError("Insufficient multi-view supported color coverage; no mesh exported.")
    points, pixels, rgb = positions[kept], reference["uv"][kept], colors[kept]
    faces, candidate_count = supported_triangles(points, pixels)
    mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False,
                           vertex_colors=np.column_stack([rgb, np.full(len(rgb), 255, dtype=np.uint8)]))
    mesh.remove_unreferenced_vertices()
    if not len(mesh.faces):
        raise RuntimeError("No short-edge supported surface patch found.")
    mesh.export(output / "local_rgb_vertex_mesh.ply")
    mesh.export(output / "local_rgb_vertex_mesh.glb")
    trimesh.points.PointCloud(points, colors=rgb).export(output / "rgb_supported_points.ply")
    np.savez_compressed(output / "vertex_color_support.npz", uid=uids[kept], xyz=points, rgb=rgb,
                        accepted_view_count=count[kept], color_mad=mad[kept], reference_uv=pixels,
                        mesh_faces_before_vertex_compaction=faces)
    annotated_rgb(reference["image"], reference["uv"], kept, output / "rgb_projection_overlay.jpg", reference["recording_s"])
    Image.fromarray(reference["image"]).transpose(Image.Transpose.ROTATE_270).save(output / "rgb_reference.jpg")
    Image.fromarray(reference["image"]).save(output / "raw_rgb_reference.png")
    write_frame_contact_sheet(rgb_frames, output / "rgb_frame_contact_sheet.jpg")
    summary = {
        "trial_id": args.trial_id, "status": "local_vertex_color_experiment",
        "vrs_read_restored": True, "vrs_rgb_frame_count": provider.get_num_data(rgb_sid),
        "raw_rgb_size": [int(x) for x in reference["cam"].get_image_size()], "video_start_device_ns": origin_ns,
        "recording_window_s": [args.start_s, args.end_s], "rgb_frame_count": len(rgb_frames),
        "graph_uid": trajectory.graph_uid.iloc[0], "rgb_frames": frame_summaries,
        "slam_reprojection": {"frames": validations, "median_px": median_error, "p95_px": p95_error,
                              "n_valid": len(all_errors), "gate_median_px": 1.0, "gate_p95_px": 3.0,
                              "interpretation": "Input consistency on SLAM cameras, not independent RGB or geometry accuracy"},
        "coloring": {"quality_filtered_observed_points": len(point_table), "colored_points": int(kept.sum()),
                     "retained_percent_of_observed_points": float(100 * kept.sum() / len(point_table)),
                     "min_views": 3, "max_rgb_mad": 18, "foreground_exclusion": "raw RGB u >= 750 AND v < 850 excluded in inspected 5-15 s window",
                     "visibility": "SLAM observation support plus sparse 4 px, 3x3 min-range bins; tolerance max(0.04 m, 1% range)",
                     "geometry_source": "MPS observed landmark XYZ, not the coarse density-envelope mesh"},
        "mesh": {"vertices": len(mesh.vertices), "faces": len(mesh.faces), "candidate_faces": candidate_count,
                 "max_edge_m": .15, "min_projected_triangle_area_px": 2,
                 "method": "single-view Delaunay on multi-view colored landmarks; reject long 3D edges",
                 "watertight": bool(mesh.is_watertight), "uv_atlas": False},
        "motion": {"linear_speed_p90_m_s": float(np.quantile(np.linalg.norm(trajectory.loc[(times >= origin_ns + args.start_s*1e9) & (times <= origin_ns + args.end_s*1e9), [f"device_linear_velocity_{a}_device" for a in "xyz"]], axis=1), .9))},
        "inputs": {"vrs": str(vrs), "slam_folder": str(slam)},
        "runtime": {x: importlib.metadata.version(x) for x in ["projectaria-tools", "numpy", "scipy", "pandas", "trimesh"]},
        "limitations": ["RGB has no independent annotated pixel ground truth in this test.",
                        "Low-motion frames limit timing sensitivity; moving-view validation remains necessary.",
                        "Sparse depth and color consistency cannot rule out all occlusion or stationary foreground contamination.",
                        "Row-wise rolling-shutter and exposure normalization are not implemented.",
                        "Triangulation may connect nearby different surfaces; not a metrically validated room mesh.",
                        "Hand/phone mask is manually inspected and not transferable to other segments.",
                        "Unseen areas are absent, not photographically invented."],
    }
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_experiment_page(output / "index.html", mesh, points, rgb, count[kept], summary)
    print(f"Finished: {output}\nSLAM reprojection median {median_error:.3f}px; p95 {p95_error:.3f}px", flush=True)


if __name__ == "__main__":
    main()
