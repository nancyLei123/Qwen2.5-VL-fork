#!/usr/bin/env python3
"""Create two synthetic trials for the browser tests from a local dual-view trial (default Parki).

- ``E2E_Attached``: downward footage attached to the Aria clock (``downward_attachment``) with a
  recorded window of 20-120 s and upright rotation, the layout of newer paired trials.
- ``E2E_AriaOnly``: ``media_mode: aria_only`` with no downward video.

Only manifests and pages are written, under ``runs/`` (ignored by git). The videos are referenced
from the source trial, not copied, and nothing in the source trial is modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import build_symptom_annotation_page as builder  # noqa: E402

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="Parki")
    args = parser.parse_args()
    if not (builder.RUNS_DIR / args.source / "session_dashboard_manifest.json").is_file():
        raise SystemExit(f"runs/{args.source} is needed for the browser tests (see docs/symptom_annotation/README.md).")
    duration = json.loads((builder.RUNS_DIR / args.source / "session_dashboard_manifest.json").read_text())["session_duration_s"]
    make(args.source, "E2E_Attached", {"media_mode": "dual_view",
                                        "downward_attachment": {**ATTACHMENT, "session_duration_s": duration}})
    make(args.source, "E2E_AriaOnly", {"media_mode": "aria_only", "downward_session_video_uri": ""})
    print("Prepared E2E_Attached and E2E_AriaOnly")


if __name__ == "__main__":
    main()
