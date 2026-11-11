#!/usr/bin/env python3
"""Create synthetic trials for the browser tests from a local dual-view trial (default Parki).

- ``E2E_Attached``: downward footage attached to the Aria clock (``downward_attachment``) with a
  recorded window of 20-120 s and upright rotation, the layout of newer paired trials.
- ``E2E_AriaOnly``: ``media_mode: aria_only`` with no downward video.
- ``E2E_Predictions``: a plain copy used for the automatic-prediction tests. A small hand-written
  detector run (clearly marked as a test fixture, not detector output) is saved for it in the
  temporary symptom store named by ``PD_INSIGHTER_E2E_STORE``.

Only manifests and pages are written, under ``runs/`` (ignored by git). The videos are referenced
from the source trial, not copied, and nothing in the source trial is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import build_symptom_annotation_page as builder  # noqa: E402
import symptom_detection_store as detection_store  # noqa: E402

ATTACHMENT = {"start_s": 20.0, "end_s": 120.0, "rotation_quarter_turns": 0, "applied_offset_s": 20.0,
              "verification": {"valid": True}, "padding": "synthetic test fixture"}


def make(source: str, name: str, update: dict) -> None:
    src, dst = builder.RUNS_DIR / source, builder.RUNS_DIR / name
    session = json.loads((src / "session_dashboard_manifest.json").read_text(encoding="utf-8"))
    for key in ("aria_session_video_uri", "downward_session_video_uri"):
        if session.get(key):
            session[key] = f"../{source}/{session[key]}"
    session.update(update, trial_id=name, recording_stem=name)
    dst.mkdir(exist_ok=True)
    (dst / "session_dashboard_manifest.json").write_text(json.dumps(session, indent=1), encoding="utf-8")
    trial = {"trial_id": name, **{k: v for k, v in update.items() if k in ("media_mode", "downward_attachment")}}
    (dst / "trial_manifest.json").write_text(json.dumps(trial, indent=1), encoding="utf-8")
    shutil.copyfile(src / "index.html", dst / "index.html")
    builder.build(name)


def evidence(key: str, source: str, points: float, maximum: float, text: str, kind: str = "support") -> dict:
    return {"id": key, "kind": kind, "source": source, "points": points, "max_points": maximum, "text": text, "value": None, "unit": ""}


def fixture_run(trial: str, duration: float) -> dict:
    """Three made-up predictions with the same shape as real detector output."""
    def item(label_id: str, name: str, start: float, end: float, score: int, strength: str, summary: str, extra: list[dict]) -> dict:
        return {"id": f"{label_id}-{int(start * 1000):08d}", "label_id": label_id, "label_name": name, "kind": "",
                "start_s": start, "end_s": end, "duration_s": end - start, "score": score,
                "score_kind": "uncalibrated detection score (0-100), not a probability", "evidence_strength": strength,
                "summary": summary, "evidence": extra, "sources": ["head_track"], "independent_sources": ["head_track"],
                "measurements": {"duration_s": end - start},
                "limitations": ["Test fixture, not detector output.", "Confirm with the video."]}
    detections = [
        item("freezing_like", "Freezing-like episode", 30.0, 36.0, 74, "high", "Progress stopped for 6.0 s while the body kept moving in place.", [
            evidence("progress_stopped", "head_track", 15, 15, "Net forward progress is 0.01 m/s for 6.0 s."),
            evidence("in_place_video", "downward_video", 34, 40, "Downward view: image motion during the stop is 2.10 px/frame."),
            evidence("turning", "head_track", -6, 18, "The head turns about 90 degrees during the stop.", "penalty")]),
        item("gait_hesitation", "Gait hesitation", 62.0, 64.5, 38, "low", "Walking is interrupted for 2.5 s and then continues.", [
            evidence("progress_dip", "head_track", 14, 25, "Movement drops to 0.04 m/s for 2.5 s."),
            evidence("in_place_video", "downward_video", 0, 20, "The downward camera has no footage for this interval.", "missing")]),
        item("slow_gait", "Slowed gait", 90.0, 110.0, 61, "medium", "Level walking at 0.18 m/s for 20 s.", [
            evidence("speed", "head_track", 36, 40, "Walking speed is 0.18 m/s.")]),
    ]
    return {
        "schema": detection_store.DETECTIONS_SCHEMA, "trial_id": trial, "run_id": "e2e-fixture-0.0.1-00000000",
        "generated_at": "2026-01-01T00:00:00.000+00:00",
        "detector": {"name": "e2e-fixture", "version": "0.0.1", "config_sha256": "0" * 64, "config": {}},
        "session_clock": {"name": "dashboard session time", "units": "s", "duration_s": duration},
        "downward_coverage_s": None, "sources": {"head_track": {"available": True}},
        "source_labels": {"head_track": "Head track", "downward_video": "Downward video"},
        "baselines": {},
        "labels": [{"id": "freezing_like", "name": "Freezing-like episode", "color": "#b5475f", "description": "Test label."},
                   {"id": "gait_hesitation", "name": "Gait hesitation", "color": "#c77f2a", "description": "Test label."},
                   {"id": "slow_gait", "name": "Slowed gait", "color": "#3f7f9c", "description": "Test label."}],
        "detections": detections,
        "summary": {"count": len(detections), "by_label": {}, "not_reported": {},
                    "notes": ["Candidates for manual review.", "Scores are uncalibrated.", "Test fixture."]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="Parki")
    args = parser.parse_args()
    if not (builder.RUNS_DIR / args.source / "session_dashboard_manifest.json").is_file():
        raise SystemExit(f"runs/{args.source} is needed for the browser tests (see docs/symptom_annotation/README.md).")
    builder.build(args.source)                  # current page assets for the source trial as well
    duration = json.loads((builder.RUNS_DIR / args.source / "session_dashboard_manifest.json").read_text())["session_duration_s"]
    make(args.source, "E2E_Attached", {"media_mode": "dual_view",
                                        "downward_attachment": {**ATTACHMENT, "session_duration_s": duration}})
    make(args.source, "E2E_AriaOnly", {"media_mode": "aria_only", "downward_session_video_uri": ""})
    make(args.source, "E2E_Predictions", {})
    store = os.environ.get("PD_INSIGHTER_E2E_STORE")
    if store:                                   # never the real annotation store
        detection_store.save_detections(Path(store), fixture_run("E2E_Predictions", float(duration)))
    print("Prepared E2E_Attached, E2E_AriaOnly and E2E_Predictions")


if __name__ == "__main__":
    main()
