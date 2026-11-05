#!/usr/bin/env python3
"""Local-only camera/landmark footprint prototype; never a verified surface mesh."""
from __future__ import annotations

import json
from pathlib import Path
import argparse

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from reconstruct_dedicated_room import Reader, OUTPUT, inputs, save_json, sha256, verify_stats

ENV = OUTPUT.parent
UPRIGHT = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
THREE_TO_CV = np.diag([1., -1., -1.])


def camera_rays(camera, size=9):
    """Viewer-local rays, with an ordered boundary for the rectified crop."""
    def ray(u, v):
        x = (u * camera['width'] - .5 - camera['cx']) / camera['fx']
        y = (v * camera['height'] - .5 - camera['cy']) / camera['fy']
        p = np.array([x, -y, -1.])
        return (p / np.linalg.norm(p)).tolist()
    grid = [ray(u, v) for v in np.linspace(.05, .95, size) for u in np.linspace(.05, .95, size)]
    edge = np.linspace(0, 1, 9)[:-1]
    boundary = [ray(u, 0) for u in edge] + [ray(1, v) for v in edge]
    boundary += [ray(1-u, 1) for u in edge] + [ray(0, 1-v) for v in edge]
    return np.asarray(grid + boundary), len(grid)


def surface_patches(points, reference=None):
    """Finite discs on locally planar, well-surrounded landmark neighborhoods."""
    reference = points if reference is None else reference
    tree = cKDTree(reference)
    distances, indices = tree.query(points, k=16, workers=2)
    neighbors = reference[indices]
    centered = neighbors - neighbors.mean(axis=1)[:, None, :]
    cov = np.einsum('nki,nkj->nij', centered, centered) / 16
    values, vectors = np.linalg.eigh(cov)
    normals = vectors[:, :, 0]
    rms = np.sqrt(np.maximum(values[:, 0], 0))
    delta = neighbors - points[:, None, :]
    a = np.einsum('nki,ni->nk', delta, vectors[:, :, 1])
    b = np.einsum('nki,ni->nk', delta, vectors[:, :, 2])
    sectors = np.floor((np.arctan2(b[:, 1:], a[:, 1:]) + np.pi) / (2*np.pi) * 8).astype(int) % 8
    occupied = np.array([len(np.unique(row)) for row in sectors])
    keep = (distances[:, -1] <= .22) & (rms <= .012) & (values[:, 1] >= .00003)
    keep &= (values[:, 0] / np.maximum(values[:, 1], 1e-12) <= .08) & (occupied >= 6)
    radius = np.clip(distances[:, 4] * .45, .018, .055)
    result = np.column_stack([points[keep], normals[keep], radius[keep]])
    return result, {'input_landmarks': len(points), 'fitting_reference_landmarks': len(reference), 'patches': int(keep.sum()),
        'checks_individually': {'nearby': int((distances[:,-1] <= .22).sum()), 'flat_rms': int((rms <= .012).sum()),
            'two_dimensional': int((values[:,1]>=.00003).sum()), 'planar_ratio': int((values[:,0]/np.maximum(values[:,1],1e-12) <= .08).sum()),
            'surrounded': int((occupied>=6).sum())},
        'neighbors': 16, 'max_neighbor_distance_m': .22, 'max_local_plane_rms_m': .012,
        'radius_cap_m': .055, 'occupied_angular_sectors_min': 6,
        'plane_rms_median_m': float(np.median(rms[keep])) if keep.any() else None,
        'scope': 'Local fitted discs only. No triangles bridging gaps, no closed surface, no ground-truth surface validation.'}


def geometry(landmarks):
    import pandas as pd
    from build_scene_spatial_overview import display_sample
    chunks = []
    columns = ['graph_uid', 'px_world', 'py_world', 'pz_world', 'inv_dist_std', 'dist_std']
    for chunk in pd.read_csv(inputs()['points'], usecols=columns, chunksize=250000):
        if not chunk.graph_uid.eq(landmarks['graph_uid']).all():
            raise ValueError('Mixed landmark coordinate systems')
        xyz = chunk[['px_world','py_world','pz_world']].to_numpy()
        keep = np.isfinite(xyz).all(axis=1) & chunk.inv_dist_std.between(0,.005) & chunk.dist_std.between(0,.01)
        if keep.any():
            chunks.append(display_sample(xyz[keep], voxel=.02, limit=len(xyz))[0])
    reference = display_sample(np.concatenate(chunks), voxel=.02, limit=4000000)[0]
    print(f'Fitting local patches against {len(reference):,} quality-filtered landmarks', flush=True)
    centers = np.asarray(landmarks['points'])
    local, audit = surface_patches(centers, reference)
    # Robust plane proposals tolerate off-surface landmarks; only locally
    # surrounded inliers become small discs, never a filled global plane.
    rng = np.random.default_rng(42)
    remaining = centers.copy()
    extra, planes = [], []
    for _ in range(12):
        if len(remaining) < 1000:
            break
        sample = remaining[rng.choice(len(remaining), min(12000, len(remaining)), replace=False)]
        triples = sample[rng.integers(0, len(sample), (1200, 3))]
        normal = np.cross(triples[:,1]-triples[:,0], triples[:,2]-triples[:,0])
        length = np.linalg.norm(normal, axis=1)
        good = length > .05
        normal = normal[good]/length[good,None]
        offset = np.einsum('ij,ij->i',normal,triples[good,0])
        scores = []
        for start in range(0, len(normal), 100):
            scores.extend((np.abs(sample @ normal[start:start+100].T-offset[start:start+100]) <= .015).sum(axis=0))
        best = int(np.argmax(scores))
        if scores[best] < 180:
            break
        n, d = normal[best], offset[best]
        for _ in range(3):
            inliers = remaining[np.abs(remaining @ n-d) <= .015]
            center = inliers.mean(axis=0)
            _, _, vh = np.linalg.svd(inliers-center, full_matrices=False)
            n = vh[-1]; d = center @ n
        plane_ref = reference[np.abs(reference @ n-d) <= .015]
        candidate = centers[np.abs(centers @ n-d) <= .015]
        if len(plane_ref) < 32:
            break
        # Fit the local support in this robust plane, keeping only its footprint.
        plane_ref -= (plane_ref @ n-d)[:,None]*n
        candidate -= (candidate @ n-d)[:,None]*n
        patches, detail = surface_patches(candidate, plane_ref)
        extra.append(patches)
        planes.append({'normal': n.tolist(), 'offset_m': float(d), 'inlier_landmarks': len(plane_ref),
            'local_patches': len(patches), 'max_inlier_residual_m': .015})
        remaining = remaining[np.abs(remaining @ n-d) > .025]
        print(f'Robust plane {len(planes)}: {len(plane_ref)} inliers, {len(patches)} finite supported patches', flush=True)
    combined = np.concatenate([local, *extra]) if extra else local
    if len(combined):
        _, ids = np.unique(np.round(combined[:,:3]/.02).astype(int),axis=0,return_index=True)
        combined = combined[np.sort(ids)]
    audit.update({'strict_local_patches': len(local), 'patches': len(combined), 'robust_planes': planes,
        'robust_plane_method': 'Seed 42; up to 12 RANSAC plane proposals, 1200 hypotheses; 15 mm inliers. Plane proposals are NOT rendered. Only small locally surrounded inlier discs are rendered; no closed or filled surfaces.',
        'plane_rms_median_m': audit['plane_rms_median_m']})
    return combined, audit


def rotated_video_uv(camera, rays):
    # Undo viewer/OpenCV and upright-basis changes before applying the actual fisheye lens.
    raw_rays = rays @ (UPRIGHT @ THREE_TO_CV).T
    pixels = np.asarray([camera.project(p) for p in raw_rays], dtype=float)
    if pixels.shape != (len(rays), 2) or not np.isfinite(pixels).all():
        raise ValueError('Rectified crop lies outside calibrated RGB image')
    width, height = camera.get_image_size()
    uv = np.column_stack([(height - .5 - pixels[:, 1]) / height, (pixels[:, 0] + .5) / width])
    if np.any((uv < 0) | (uv > 1)):
        raise ValueError('Projection outside review video')
    return uv


def build(geometry_only=False):
    data_path, map_path = ENV / 'scene_playback_data.json', ENV / 'spatial_overview_data.json'
    data = json.loads(data_path.read_text())
    landmarks = json.loads(map_path.read_text())
    manifest = json.loads((OUTPUT / 'dataset_manifest.json').read_text())
    verify_stats(manifest['source_stats'])
    assert data['trial_id'] == landmarks['trial_id'] == 'RoomScan3c'
    assert data['provenance']['graph_uid'] == landmarks['graph_uid'] == manifest['world_frame']['graph_uid']
    assert sha256(data_path) == landmarks['playback_sha256']
    assert sha256(OUTPUT / 'trained_30000/splat.ply') == data['provenance']['model_sha256']
    patches, geometry_audit = geometry(landmarks)
    if len(patches) < 100:
        raise ValueError('Insufficient locally supported geometry')
    if geometry_only:
        path = ENV / 'view_footprint/footprint_data.json'
        payload = json.loads(path.read_text())
        if payload['playback_sha256'] != sha256(data_path) or payload['map_sha256'] != sha256(map_path):
            raise ValueError('Existing calibration cache no longer matches the playback/map')
        payload['patches'] = patches.round(6).tolist()
        payload['audit']['geometry'] = geometry_audit
        save_json(path.parent / 'alignment_audit.json', payload['audit'])
        path.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False)+'\n')
        print(json.dumps(geometry_audit, indent=2))
        return
    rays, hit_ray_count = camera_rays(manifest['camera'])
    reader = Reader()
    if reader.graph != landmarks['graph_uid'] or len(reader.timestamps) != len(data['poses']):
        raise ValueError('Playback/VRS identity mismatch')
    if not np.allclose((reader.timestamps-reader.origin)/1e9, data['recording_times'], atol=1e-8, rtol=0):
        raise ValueError('Playback does not map to exact VRS frame timestamps')
    # Reuse a LUT only while every ray differs by <=0.20 source pixels. Each
    # supported frame is checked, rather than assuming static camera intrinsics.
    lut, mapping, maximum_error, maximum_pose_error = [], [], 0., 0.
    last = None
    selected = {r['source_frame_index']: r for r in manifest['frames']['items']}
    pose_checks = set(selected) | set(range(0, len(reader.timestamps), 300))
    checked_poses, roundtrip = 0, []
    for i, stamp in enumerate(reader.timestamps):
        if data['poses'][i] is None:
            mapping.append(-1)
            continue
        camera, transform, _ = reader.pose(int(stamp))
        uv = rotated_video_uv(camera, rays)
        error = np.inf if last is None else np.max(np.linalg.norm((uv-last)*1408, axis=1))
        if error > .20:
            last = uv.astype(np.float32).astype(float)
            lut.append(np.round(last, 8).tolist())
        maximum_error = max(maximum_error, float(np.max(np.linalg.norm((uv-last)*1408, axis=1))))
        mapping.append(len(lut)-1)
        if i in pose_checks:
            transform[:3, :3] = transform[:3, :3] @ UPRIGHT
            p = data['poses'][i]
            actual = np.eye(4)
            actual[:3, :3] = Rotation.from_quat(p[3:]).as_matrix() @ THREE_TO_CV
            actual[:3, 3] = p[:3]
            pose_error = float(np.max(np.abs(actual-transform)))
            if i in selected:
                pose_error = max(pose_error, float(np.max(np.abs(actual-selected[i]['T_world_camera']))))
            maximum_pose_error = max(maximum_pose_error, pose_error)
            checked_poses += 1
            for ray in rays[::10]:
                raw = UPRIGHT @ THREE_TO_CV @ ray
                pixel = camera.project(raw)
                restored = camera.unproject(pixel)
                restored /= np.linalg.norm(restored)
                roundtrip.append(float(np.linalg.norm(restored-raw)))
        if i % 3000 == 0:
            print(f'Calibration/pose audit {i}/{len(reader.timestamps)}, LUTs {len(lut)}', flush=True)
    if maximum_pose_error > 1e-6 or max(roundtrip) > 1e-6:
        raise ValueError('Coordinate alignment audit failed')
    audit = {'status': 'coordinate_checks_passed_surface_unvalidated', 'trial_id': data['trial_id'],
        'world_frame': data['provenance']['graph_uid'], 'exact_vrs_frames': len(mapping),
        'supported_frame_projections_checked': sum(v >= 0 for v in mapping),
        'pose_checks': checked_poses, 'max_pose_matrix_difference': maximum_pose_error,
        'max_lens_roundtrip_ray_error': max(roundtrip), 'video_lut_max_error_pixels': maximum_error,
        'video_lut_count': len(lut), 'geometry': geometry_audit,
        'not_validated': ['physical surface accuracy', 'true occlusion or visibility', 'object identity', 'eye gaze',
            'moving objects', 'rolling-shutter correction', 'full fisheye peripheral field'],
        'interpretation': 'Internal coordinate and lens consistency only; not an independent accuracy measurement. Nearest local-disc hits are experimental candidates, not confirmed object intersections.',
        'sources': ['https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/mps_calibration',
            'https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/core_code_snippets/mps']}
    payload = {'schema_version': 1, 'trial_id': data['trial_id'], 'graph_uid': landmarks['graph_uid'],
        'units': 'meters', 'up_axis': 'Z', 'fps': data['fps'], 'frame_count': len(mapping),
        'playback_sha256': sha256(data_path), 'map_sha256': sha256(map_path),
        'model_sha256': data['provenance']['model_sha256'], 'camera': manifest['camera'],
        'rays': rays.round(10).tolist(), 'hit_ray_count': hit_ray_count,
        'patches': patches.round(6).tolist(), 'video_uv_lut': lut, 'frame_lut': mapping, 'audit': audit}
    output = ENV / 'view_footprint'
    save_json(output / 'alignment_audit.json', audit)
    # Compact browser-only geometry/LUT data; original media/models remain untouched.
    output.mkdir(exist_ok=True)
    (output / 'footprint_data.json').write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-only', action='store_true')
    build(parser.parse_args().geometry_only)
