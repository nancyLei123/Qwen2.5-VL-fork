#!/usr/bin/env python3
"""Bounded, annotation-assisted gait illustration. Never modifies source inputs."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.ndimage import uniform_filter1d

from reconstruct_dedicated_room import Reader, sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/RoomScan3c/environment_reconstruction'


def mapped_time(value, source, target):
    if not source[0] <= value <= source[-1]:
        raise ValueError('Time outside calibrated frame table')
    return float(np.interp(value, source, target))


def make_cycles(cues, sample_t, positions):
    cycles = []
    for side in ('left', 'right'):
        anchors = [c for c in cues if c['side'] == side]
        for a, b in zip(anchors, anchors[1:]):
            if not .4 <= b['recording_s'] - a['recording_s'] <= 2.2:
                continue
            # Smoothed device path is a scale cue, not the path of either foot.
            t = np.r_[a['recording_s'], sample_t[(sample_t > a['recording_s']) &
                (sample_t < b['recording_s'])], b['recording_s']]
            xy = np.column_stack([np.interp(t, sample_t, positions[:, i]) for i in range(2)])
            path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
            cycles.append({'side': side, 'start_s': a['time_s'], 'end_s': b['time_s'],
                'head_path_m': path, 'display_excursion_m': float(np.clip(path / 4, .035, .22)),
                'contact_measured': False})
    return cycles


def build(cues_path=None):
    evidence = OUT / 'gait_fusion'
    cues_path = Path(cues_path) if cues_path else ROOT / 'scripts/room_gait_cues.json'
    if not cues_path.is_file():
        raise FileNotFoundError('Private gait annotations are not in GitHub. '
                                'Supply the authorized data file with --cues PATH.')
    notes = json.loads(cues_path.read_text())
    alignment = json.loads((evidence / 'sync.json').read_text())
    if not alignment['valid']:
        raise ValueError('Audio alignment has not passed the consistency checks')
    data_path = OUT / 'scene_playback_data.json'
    data = json.loads(data_path.read_text())
    reader = Reader()
    if reader.graph != data['provenance']['graph_uid']:
        raise ValueError('MPS graph mismatch')
    rt = np.asarray(data['recording_times'])
    if not np.allclose((reader.timestamps - reader.origin) / 1e9, rt, atol=1e-8):
        raise ValueError('VRS frame table mismatch')
    vt = np.arange(len(rt)) / data['fps']
    scale, offset = alignment['recording_scale'], alignment['recording_offset_s']
    record = lambda down: down * scale + offset
    playback = lambda rec: mapped_time(rec, rt, vt)
    source_t = (reader.t - reader.origin) / 1e9
    windows = []
    for item in notes['windows']:
        start, end = record(item['cues'][0][0]), record(item['cues'][-1][0])
        times = np.arange(start - .5, end + .51, .02)
        right = np.searchsorted(source_t, times).clip(1, len(source_t) - 1)
        left = right - 1
        supported = ((times >= source_t[0]) & (times <= source_t[-1]) &
            (source_t[right] - source_t[left] <= .1) &
            (np.minimum(reader.quality[left], reader.quality[right]) >= .9))
        frame_ids = np.flatnonzero((rt >= start - .5) & (rt <= end + .5))
        if not supported.all() or any(data['poses'][i] is None for i in frame_ids):
            raise ValueError(f'Unsupported pose in {item["id"]}; do not bridge it')
        xyz = np.column_stack([np.interp(times, source_t, reader.p[:, i]) for i in range(3)])
        if np.max(np.linalg.norm(np.diff(xyz, axis=0), axis=1)) > .15:
            raise ValueError('Trajectory discontinuity in review window')
        smooth = uniform_filter1d(xyz, size=9, axis=0, mode='nearest')
        cues = [{'downward_s': t, 'recording_s': record(t), 'time_s': playback(record(t)),
                 'side': side, 'contact_measured': False} for t, side in item['cues']]
        samples = []
        for i in range(13, len(times) - 13, 2):
            dxy = smooth[i + 12, :2] - smooth[i - 12, :2]
            samples.append([playback(times[i]), float(np.arctan2(dxy[1], dxy[0])),
                            float(np.linalg.norm(dxy) / .48)])
        windows.append({'id': item['id'], 'title': item['title'], 'start_s': playback(start),
            'end_s': playback(end), 'cues': cues, 'cycles': make_cycles(cues, times, smooth),
            'direction_speed_samples': samples,
            'evidence_sheets': [f'gait_fusion/{p}' for p in item['evidence_sheets']]})
    clip = dict(notes['clip'])
    clip['url'] = 'gait_fusion/downward_review.mp4'
    clip['start_s'], clip['end_s'] = [playback(record(clip[k])) for k in ('source_start_s', 'source_end_s')]
    output = OUT / clip['url']
    if not output.exists():
        # 84.000 s is an observed source-frame PTS; preserve its subsequent VFR spacing.
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-ss', str(clip['source_start_s']),
            '-i', alignment['downward_video'], '-t', str(clip['source_end_s'] - clip['source_start_s']),
            '-vf', 'setpts=PTS-STARTPTS,scale=960:-2', '-an', '-c:v', 'libx264',
            '-preset', 'fast', '-crf', '22', '-fps_mode', 'vfr', '-movflags', '+faststart', str(output)], check=True)
    payload = {'schema_version': 1, 'trial_id': notes['trial_id'], 'graph_uid': reader.graph,
        'duration_s': data['duration_s'], 'fps': data['fps'],
        'source_playback_sha256': sha256(data_path),
        'annotation_method': notes['annotation_method'], 'cue_definition': notes['cue_definition'],
        'side_definition': notes['side_definition'],
        'timing_review_half_window_s': notes['timing_review_half_window_s'],
        'timing_note': notes['timing_note'],
        'sync': {k: alignment[k] for k in ('recording_scale', 'recording_offset_s',
            'max_fit_residual_s', 'strong_windows')},
        'method': 'Video-reviewed per-foot phase anchors + audio time alignment + smoothed MPS device travel direction and distance. Each foot interpolates only between its own anchors inside one review window. No forced alternation across missing evidence.',
        'display_prior': 'Fixed-size body; foot excursion = smoothed same-foot-anchor device-path distance / 4, clamped to 0.035-0.22 m. Cosine interpolation; nominal 0.045 m lift only in second half-cycle. Neither floor contact nor lift is measured. Body heading follows head heading for readability.',
        'limitations': 'Annotation-assisted prototype, not automatic tracking, motion capture, ground-truth gait, or clinical validation. Head-path distance is not anatomical stride length. Side and phase cues require human review. Neutral feet outside support mean unavailable evidence, NOT standing. No floor calibration or recovered joint angles.',
        'clip': clip, 'windows': windows, 'excluded': notes['excluded'],
        'summary': {'cue_count': sum(len(w['cues']) for w in windows),
                    'review_window_seconds': sum(w['end_s'] - w['start_s'] for w in windows),
                    'window_count': len(windows)}}
    (OUT / 'room_gait_fusion.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'summary': payload['summary'], 'clip': clip}, indent=2))
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cues', type=Path, help='Private, locally stored gait annotation JSON.')
    build(parser.parse_args().cues)
