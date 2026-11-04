import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from continuous_motion_predictions import (
    DEFAULT_CONFIG,
    build_continuous_segments,
    refine_boundaries,
    summarize_grid_accuracy,
)


class ContinuousMotionPredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_segments = [
            {"label": "walking", "start_s": 0.0, "end_s": 4.0},
            {"label": "standing", "start_s": 4.0, "end_s": 8.0},
        ]

    def test_strong_dense_evidence_moves_boundary(self) -> None:
        dense = pd.DataFrame(
            {
                "time_s": np.arange(1.0, 7.5, 0.5),
                "dense_v2_evidence": [
                    "walking",
                    "walking",
                    "walking",
                    "walking",
                    "standing",
                    "standing",
                    "standing",
                    "standing",
                    "standing",
                    "standing",
                    "standing",
                    "standing",
                    "standing",
                ],
            }
        )

        decisions = refine_boundaries(self.previous_segments, dense)

        self.assertTrue(decisions[0]["applied"])
        self.assertEqual(decisions[0]["refined_time_s"], 3.0)
        self.assertEqual(decisions[0]["reason"], "denser_evidence_improved_boundary")

    def test_weak_unrelated_evidence_keeps_v4_boundary(self) -> None:
        dense = pd.DataFrame(
            {
                "time_s": np.arange(1.0, 7.5, 0.5),
                "dense_v2_evidence": ["walking"] * 13,
            }
        )

        decisions = refine_boundaries(self.previous_segments, dense)

        self.assertFalse(decisions[0]["applied"])
        self.assertEqual(decisions[0]["refined_time_s"], 4.0)

    def test_continuous_segments_are_gap_free_and_keep_state_order(self) -> None:
        decisions = [
            {
                "refined_time_s": 3.5,
                "applied": True,
                "confidence": "high",
            }
        ]

        segments = build_continuous_segments(self.previous_segments, decisions, 8.0)

        self.assertEqual([segment["label"] for segment in segments], ["walking", "standing"])
        self.assertEqual(segments[0]["start_s"], 0.0)
        self.assertEqual(segments[0]["end_s"], segments[1]["start_s"])
        self.assertEqual(segments[-1]["end_s"], 8.0)

    def test_accuracy_comparison_uses_one_shared_grid(self) -> None:
        reference = np.array(["sitting", "sitting", "standing", "standing"], dtype=object)
        previous = np.array(["sitting", "sitting", "sitting", "standing"], dtype=object)
        continuous = np.array(["sitting", "sitting", "standing", "standing"], dtype=object)

        summary = summarize_grid_accuracy(
            reference,
            previous,
            continuous,
            DEFAULT_CONFIG.grid_step_s,
            "test_reference",
            [],
        )

        self.assertEqual(summary["comparable_sample_count"], 4)
        self.assertEqual(summary["comparable_duration_s"], 2.0)
        self.assertEqual(summary["previous_v4_accuracy"], 0.75)
        self.assertEqual(summary["continuous_v1_accuracy"], 1.0)
        self.assertAlmostEqual(summary["accuracy_change_percentage_points"], 25.0)


if __name__ == "__main__":
    unittest.main()
