#!/usr/bin/env python3
"""Screen RoomScan3c for stride-proxy review; never infer measured foot contacts."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from generate_step_stride_diagnostics import analyze_bout, rolling_mean
from reconstruct_dedicated_room import Reader, sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'runs/RoomScan3c/environment_reconstruction'


def screen_result(result) -> list[str]:
    """Conservative display gates, not a validated walking/PD classifier."""
    b, events, series = result.bout, result.events, result.series
    reasons = []
    if b['horizontal_path_length_m'] < 1 or b['net_horizontal_displacement_m'] < .35:
        reasons.append('little translation')
    speed = rolling_mean(series.linear_speed_mps.to_numpy(), 25)
    if np.median(speed) < .12 or np.percentile(speed, 25) < .06:
        reasons.append('slow or interrupted translation')
    if b['straightness_net_over_path'] < .55 or np.ptp(series.local_heading_deg) > 65:
        reasons.append('turning or unstable path direction')
    if len(events) < 5 or b['valid_step_count'] < 3 or b['lateral_range_m'] < .018:
        reasons.append('insufficient step-like sway')
    if not .6 <= b['cadence_steps_per_s'] <= 2.4 or b['alternation_rate'] < .7:
        reasons.append('irregular sway rhythm')
    if b['step_length_cv'] is not None and b['step_length_cv'] > .60:
        reasons.append('variable path spacing')
    if len(events) > 1:
        intervals = np.diff(events.session_s)
        if intervals.min() < .24 or intervals.max() > 1.5 or intervals.std() / intervals.mean() > .45:
            reasons.append('irregular event timing')
    return reasons


def resample_reader(reader, duration: float, fps=50.) -> pd.DataFrame:
    source_t = (reader.t - reader.origin) / 1e9
    times = np.arange(0, duration, 1 / fps)
    right = np.searchsorted(source_t, times).clip(1, len(source_t) - 1)
    left = right - 1
    valid = ((times >= source_t[0]) & (times <= source_t[-1])
             & (source_t[right] - source_t[left] <= .1)
             & (np.minimum(reader.quality[left], reader.quality[right]) >= .9))
    frame = pd.DataFrame({'session_s': times, 'supported': valid})
    for index, name in enumerate('xyz'):
        frame[f't{name}_world_device'] = np.interp(times, source_t, reader.p[:, index])
    for index, name in enumerate('xyzw'):
        frame[f'q{name}_world_device'] = np.interp(times, source_t, reader.q[:, index])
    xyz = frame[[f't{axis}_world_device' for axis in 'xyz']].to_numpy()
    velocity = np.gradient(xyz, 1 / fps, axis=0)
    frame['linear_speed_mps'] = np.linalg.norm(velocity[:, :2], axis=1)
    distance = np.r_[0, np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)]
    distance[~valid | ~np.r_[False, valid[:-1]]] = 0
    frame['session_horizontal_path_m'] = np.cumsum(distance)
    return frame


def build():
    source = OUTPUT / 'scene_playback_data.json'
    data = json.loads(source.read_text())
    reader = Reader()
    if reader.graph != data['provenance']['graph_uid']:
        raise ValueError('The stride trajectory and scene must share the same MPS frame.')
    recording_times = np.asarray(data['recording_times'])
    if not np.allclose((reader.timestamps - reader.origin) / 1e9, recording_times, atol=1e-8):
        raise ValueError('VRS timestamps do not match the preserved playback frame table.')
    video_times = np.arange(len(recording_times)) / data['fps']
    to_video = lambda value: float(np.interp(value, recording_times, video_times))
    frame = resample_reader(reader, recording_times[-1])
    accepted, audit = [], []
    for i, start in enumerate(np.arange(0, recording_times[-1] - 4, 6)):
        end = min(start + 6, recording_times[-1])
        section = frame[(frame.session_s >= start) & (frame.session_s <= end)]
        rgb_indices = np.flatnonzero((recording_times >= start) & (recording_times <= end))
        reasons = []
        if not section.supported.all() or any(data['poses'][j] is None for j in rgb_indices):
            reasons = ['unsupported trajectory or RGB pose']
        result = None if reasons else analyze_bout('RoomScan3c', i, section,
            {'start_s': start, 'end_s': end, 'parent_start_s': start, 'parent_end_s': end,
             'label': 'unverified_scan_motion'}, 50.)
        if result is None and not reasons:
            reasons = ['insufficient samples']
        if result is not None:
            reasons = screen_result(result)
        audit.append({'start_s': to_video(start), 'end_s': to_video(end),
                      'status': 'candidate' if not reasons else 'not_estimated', 'reasons': reasons})
        if reasons:
            continue
        b = result.bout
        events = [{'time_s': to_video(row.session_s),
                   'step_proxy_m': float(row.valid_step_path_length_m) if pd.notna(row.valid_step_path_length_m) else None,
                   'stride_proxy_m': float(row.valid_stride_path_length_m) if pd.notna(row.valid_stride_path_length_m) else None}
                  for row in result.events.itertuples()]
        sample = result.series.iloc[::5]
        accepted.append({'id': i, 'start_s': to_video(start), 'end_s': to_video(end),
            'step_proxy_m': b['median_step_path_length_m'], 'stride_proxy_m': b['median_stride_path_length_m'],
            'cadence_events_per_min': b['cadence_steps_per_s'] * 60,
            'events': events, 'direction_samples': [
                [to_video(row.session_s), float(row.local_heading_deg)] for row in sample.itertuples()],
            'foot_side': 'unknown', 'validation': 'unvalidated head-motion proxy'})
    payload = {'schema_version': 1, 'trial_id': 'RoomScan3c', 'source_playback_sha256': sha256(source),
        'graph_uid': reader.graph, 'duration_s': data['duration_s'], 'fps': data['fps'],
        'method': 'Existing lateral head-sway estimator, 50 Hz, 6-second review windows. Step = horizontal device-path distance between adjacent sway-derived events; stride = distance across two events. No cross-window bridging.',
        'limitations': 'Room-scanning movement is not confirmed walking. Slow, turning, irregular and pose-gap windows are not estimated; this does not mean no steps occurred. Foot sides, contacts, floor height and foot clearance are not measured. No diagnostic or accuracy claim.',
        'downward_validation': {'file': str(ROOT.parent / 'downward videos/3c.MP4'),
                                'status': 'available locally, not synchronized or used for validation in this preview'},
        'thresholds': {'resample_hz': 50, 'window_s': 6, 'min_path_m': 1, 'min_displacement_m': .35,
            'min_median_speed_mps': .12, 'min_speed_p25_mps': .06, 'min_straightness': .55,
            'max_path_heading_range_deg': 65, 'min_events': 5, 'min_sway_range_m': .018,
            'cadence_events_per_s': [.6, 2.4], 'min_alternation': .7, 'max_step_length_cv': .60,
            'event_interval_s': [.24, 1.5], 'max_event_interval_cv': .45},
        'summary': {'windows_screened': len(audit), 'candidate_windows': len(accepted),
                    'candidate_seconds': sum(b['end_s'] - b['start_s'] for b in accepted),
                    'rejection_reasons': dict(Counter(r for a in audit for r in a['reasons']))},
        'bouts': accepted, 'windows': audit}
    (OUTPUT / 'room_stride_preview.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps(payload['summary'], indent=2))
    return payload


if __name__ == '__main__':
    build()
