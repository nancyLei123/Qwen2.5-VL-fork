#!/usr/bin/env python3
"""Local-only evidence preparation for the RoomScan3c gait playback prototype."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from scipy.signal import butter, correlate, resample_poly, sosfiltfilt

from reconstruct_dedicated_room import inputs
from suggest_downward_audio_sync import build_envelope
from verify_trial_audio_sync import local_offset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/RoomScan3c/environment_reconstruction/gait_fusion'
DOWN = ROOT.parent / 'downward videos/3c.MP4'


def audio():
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.sensor_data import TimeDomain
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / 'sync_audio.npz'
    rate = 8000
    if not cache.exists():
        provider = data_provider.create_vrs_data_provider(str(inputs()['vrs']))
        sid = provider.get_stream_id_from_label('mic')
        rgb = provider.get_stream_id_from_label('camera-rgb')
        config = provider.get_audio_configuration(sid)
        assert config.sample_rate == 48000
        n = provider.get_num_data(sid)
        first = provider.get_audio_data_by_index(sid, 0)[1].capture_timestamps_ns[0]
        last = provider.get_audio_data_by_index(sid, n - 1)[1].capture_timestamps_ns[-1]
        origin = provider.get_first_time_ns(rgb, TimeDomain.DEVICE_TIME)
        aria = np.zeros(round((last - first) / 1e9 * rate) + 2, dtype=np.float32)
        covered = np.zeros(len(aria), dtype=bool)
        for i in range(n):
            a, r = provider.get_audio_data_by_index(sid, i)
            raw = np.asarray(a.data, dtype=np.int32).reshape(-1, config.num_channels)[:, 0]
            chunk = resample_poly(raw.astype(np.float32) / 2**31, 1, 6)
            start = round((r.capture_timestamps_ns[0] - first) / 1e9 * rate)
            count = min(len(chunk), len(aria) - start)
            aria[start:start + count] = chunk[:count]
            covered[start:start + count] = True
            if i % 5000 == 0:
                print(f'audio {i}/{n}', flush=True)
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(DOWN), '-vn',
            '-ac', '1', '-ar', str(rate), '-f', 'f32le', 'pipe:1'])
        down = np.frombuffer(raw, dtype=np.float32)
        np.savez_compressed(cache, aria=aria, down=down,
            audio_origin_recording_s=(first - origin) / 1e9, missing_audio_samples=(~covered).sum())
    with np.load(cache) as cached:
        aria, down = cached['aria'], cached['down']
        origin = float(cached['audio_origin_recording_s'])
        missing = int(cached['missing_audio_samples'])
    ae, hop = build_envelope(aria, rate, 20)
    de, _ = build_envelope(down, rate, 20)
    corr = correlate(ae, de, mode='full', method='fft')
    approximate = (np.argmax(corr) - len(de) + 1) * hop
    band = butter(4, [300, 3000], btype='bandpass', fs=rate, output='sos')
    aria = sosfiltfilt(band, aria)
    down = sosfiltfilt(band, down)
    windows = [local_offset(aria, down, rate, float(t), 12., approximate)
               for t in np.linspace(max(5., -approximate + 3), len(down) / rate - 20, 11)]
    windows = [w for w in windows if w]
    strong = [w for w in windows if w['absolute_waveform_correlation'] >= .12]
    report = dict(method='timestamped VRS audio and downward audio; independent bandpassed waveform windows',
        source_vrs=str(inputs()['vrs']), downward_video=str(DOWN),
        approximate_audio_offset_s=approximate, audio_origin_recording_s=origin,
        missing_audio_samples=missing, windows=windows, valid=False)
    if len(strong) >= 4:
        times = np.array([w['downward_start_s'] + 6 for w in strong])
        offsets = np.array([w['audio_offset_s'] for w in strong])
        slope, intercept = np.polyfit(times, offsets, 1)
        residual = offsets - (times * slope + intercept)
        valid = bool(np.max(np.abs(residual)) < .06)
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
            '-show_entries', 'stream=codec_type,start_time', '-of', 'json', str(DOWN)]))
        streams = {s['codec_type']: float(s.get('start_time', 0)) for s in probe['streams']}
        av_offset = streams['audio'] - streams['video']
        report.update(valid=valid, strong_windows=len(strong),
            recording_scale=1 + float(slope),
            recording_offset_s=float(intercept + origin - (1 + slope) * av_offset),
            max_fit_residual_s=float(np.max(np.abs(residual))),
            drift_over_recording_s=float(slope * len(down) / rate),
            mapping='Aria VRS recording seconds = scale * downward video seconds + offset; then use exact RGB frame table for dashboard time')
    (OUT / 'sync.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


def contact_sheet(start, end, step, name):
    OUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(DOWN))
    times = np.arange(start, end, step)
    cells = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (480, 270))
        cv2.rectangle(frame, (0, 0), (220, 27), (0, 0, 0), -1)
        cv2.putText(frame, f'downward {t:.2f} s', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        cells.append(frame)
    cap.release()
    columns = 4
    while len(cells) % columns:
        cells.append(np.zeros((270, 480, 3), np.uint8))
    sheet = np.vstack([np.hstack(cells[i:i + columns]) for i in range(0, len(cells), columns)])
    cv2.imwrite(str(OUT / name), sheet)
    print(OUT / name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', action='store_true')
    parser.add_argument('--sheet', nargs=3, type=float, metavar=('START', 'END', 'STEP'))
    parser.add_argument('--name', default='contact_sheet.jpg')
    args = parser.parse_args()
    if args.audio:
        audio()
    if args.sheet:
        contact_sheet(*args.sheet, args.name)
