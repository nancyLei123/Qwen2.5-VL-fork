import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from reconstruct_slam_motion_features import apply_contextual_posture_predictions


def posture_rows(base_labels: list[str], delta_5_by_index: dict[int, float]) -> pd.DataFrame:
    rows = []
    for index, label in enumerate(base_labels):
        delta_5 = delta_5_by_index.get(index, 0.0)
        rows.append(
            {
                "clip_idx": index,
                "criteria_v2_prediction": label,
                "horizontal_path_length_m": 0.2,
                "window_height_delta_m": 0.0,
                "height_slope_mps": 0.0,
                "context5_height_delta_m": delta_5,
                "context5_height_slope_mps": delta_5 / 5.0,
                "context10_height_delta_m": 0.0,
                "context10_height_slope_mps": 0.0,
            }
        )
    return pd.DataFrame(rows)


class ContextualPosturePredictionTests(unittest.TestCase):
    def test_upward_transition_corrects_later_false_sitting(self) -> None:
        frame = posture_rows(["sitting", "sitting", "sitting", "sitting"], {1: 0.4})
        result = apply_contextual_posture_predictions(
            frame,
            {"sitting_baseline_z_m": 0.0, "standing_baseline_z_m": 0.4},
        )

        self.assertEqual(result.loc[1, "criteria_v4_reason"], "posture_transition_boundary")
        self.assertEqual(result.loc[2, "criteria_v4_prediction"], "standing")
        self.assertEqual(result.loc[3, "criteria_v4_prediction"], "standing")

    def test_downward_override_is_blocked_for_reliable_global_band(self) -> None:
        frame = posture_rows(["standing", "standing", "standing"], {1: -0.4})
        result = apply_contextual_posture_predictions(
            frame,
            {"sitting_baseline_z_m": 0.0, "standing_baseline_z_m": 0.4},
        )

        self.assertEqual(result.loc[2, "criteria_v4_context_state"], "sitting")
        self.assertEqual(result.loc[2, "criteria_v4_prediction"], "standing")

    def test_downward_override_is_allowed_when_global_band_mixes_floor_levels(self) -> None:
        frame = posture_rows(["standing", "standing", "standing"], {1: -0.8})
        result = apply_contextual_posture_predictions(
            frame,
            {"sitting_baseline_z_m": 0.0, "standing_baseline_z_m": 2.0},
        )

        self.assertEqual(result.loc[2, "criteria_v4_prediction"], "sitting")
        self.assertEqual(result.loc[2, "criteria_v4_reason"], "context_overrides_unreliable_global_band")

    def test_locomotion_label_is_never_replaced(self) -> None:
        frame = posture_rows(["sitting", "walking", "sitting"], {0: 0.4, 1: -0.8})
        result = apply_contextual_posture_predictions(
            frame,
            {"sitting_baseline_z_m": 0.0, "standing_baseline_z_m": 2.0},
        )

        self.assertEqual(result.loc[1, "criteria_v4_prediction"], "walking")
        self.assertEqual(result.loc[1, "criteria_v4_reason"], "criteria_v2_locomotion_retained")


if __name__ == "__main__":
    unittest.main()
