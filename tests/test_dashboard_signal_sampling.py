import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_downward_dual_view_dashboards import sample_signal_for_interactive_display


class DashboardSignalSamplingTests(unittest.TestCase):
    @staticmethod
    def signal_frame(duration_s: float, source_hz: float = 100.0) -> pd.DataFrame:
        time_s = np.arange(0.0, duration_s, 1.0 / source_hz)
        return pd.DataFrame(
            {
                "session_s": time_s,
                "tz_world_device": 1.6 + 0.02 * np.sin(2 * np.pi * 1.8 * time_s),
                "rolling_speed_mps": np.ones_like(time_s),
            }
        )

    def test_long_session_retains_gait_scale_temporal_resolution(self) -> None:
        source = self.signal_frame(600.0)

        sampled, metadata = sample_signal_for_interactive_display(
            source,
            base_point_budget=500,
            target_hz=15.0,
            maximum_points=60_000,
        )

        self.assertGreater(metadata["display_sample_rate_hz"], 12.0)
        self.assertGreater(len(sampled), 7_000)
        self.assertGreater(
            sampled["tz_world_device"].max() - sampled["tz_world_device"].min(),
            0.037,
        )
        self.assertEqual(metadata["method"], "block-mean-time-preserving")

    def test_short_session_keeps_legacy_point_budget_when_it_is_denser(self) -> None:
        source = self.signal_frame(216.0)

        sampled, metadata = sample_signal_for_interactive_display(
            source,
            base_point_budget=5_200,
            target_hz=15.0,
            maximum_points=60_000,
        )

        self.assertGreater(metadata["display_sample_rate_hz"], 19.0)
        self.assertGreater(len(sampled), 4_000)


if __name__ == "__main__":
    unittest.main()
