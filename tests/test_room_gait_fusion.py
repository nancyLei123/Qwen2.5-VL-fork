import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_room_gait_fusion import make_cycles, mapped_time


class GaitFusionTests(unittest.TestCase):
    def test_outside_clock_is_rejected_not_clamped(self):
        self.assertEqual(mapped_time(1.5, [1, 2], [0, 1]), .5)
        with self.assertRaises(ValueError):
            mapped_time(0, [1, 2], [0, 1])

    def test_long_missing_foot_interval_not_bridged(self):
        cues = [dict(side='left', time_s=t, recording_s=t) for t in [0, 1.4, 8]]
        times = np.arange(0, 9, .02)
        cycles = make_cycles(cues, times, np.c_[times, times * 0, times * 0])
        self.assertEqual(len(cycles), 1)
        self.assertAlmostEqual(cycles[0]['head_path_m'], 1.4)
        self.assertAlmostEqual(cycles[0]['display_excursion_m'], .22)
        self.assertFalse(cycles[0]['contact_measured'])

    def test_source_clip_and_declared_scope(self):
        path = ROOT / 'runs/RoomScan3c/environment_reconstruction'
        if not (path / 'room_gait_fusion.json').is_file():
            self.skipTest('Private RoomScan3c artifacts are not included in a code-only checkout')
        data = json.loads((path / 'room_gait_fusion.json').read_text())
        self.assertIn('not expert-validated', data['annotation_method'])
        self.assertIn('not anatomical stride length', data['limitations'])
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
            '-show_entries', 'format=duration:stream=codec_type,width,height,start_time',
            '-of', 'json', str(path / data['clip']['url'])]))
        self.assertAlmostEqual(float(probe['format']['duration']), 17, delta=.1)
        self.assertEqual(len(probe['streams']), 1)
        self.assertEqual(probe['streams'][0]['codec_type'], 'video')
        self.assertEqual(probe['streams'][0]['width'], 960)
        self.assertEqual(float(probe['streams'][0]['start_time']), 0)


if __name__ == '__main__':
    unittest.main()
