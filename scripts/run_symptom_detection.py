#!/usr/bin/env python3
"""Run the automatic symptom detector on trials in runs/ and save the predictions.

Without ``--apply`` this is a dry run: the detector runs and the candidates are listed, but
nothing is written. With ``--apply`` each trial's predictions are saved to

    <store>/<trial>/symptom_annotations/detections/<run_id>.json

where ``<store>`` is ``runs/`` unless ``--symptom-store`` names another folder (the same option
as scripts/pd_insighter_server.py). Manual annotations, review decisions, analysis results and
recordings are only read. Running again with the same detector version and configuration gives
the same ``run_id``; the file is rewritten only if the result changed, and the previous file is
kept as ``*.bak``. The Symptom Annotation page shows the newest run of a trial.

The downward video analysis takes about a minute per ten minutes of video the first time; its
result is cached next to the predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import symptom_detection as detector  # noqa: E402
import symptom_detection_signals as signals_module  # noqa: E402
import symptom_detection_store as detection_store  # noqa: E402

QWEN_REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = QWEN_REPO / "runs"


def clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def discover(runs_dir: Path) -> list[str]:
    return [path.parent.name for path in sorted(runs_dir.glob("*/session_dashboard_manifest.json"))]


def run_trial(runs_dir: Path, trial_id: str, config: dict[str, Any] | None, *, use_video: bool,
              use_audio: bool, cache_dir: Path | None) -> dict[str, Any]:
    trial_signals = signals_module.load_trial(runs_dir, trial_id, use_video=use_video, use_audio=use_audio,
                                              cache_dir=cache_dir)
    return detector.detect(trial_signals, config)


def describe(document: dict[str, Any], verbose: bool) -> list[str]:
    lines = [f"{document['trial_id']}: {document['summary']['count']} candidate(s), run {document['run_id']}"]
    for name, report in document["sources"].items():
        state = "used" if report.get("available") else "not available"
        lines.append(f"  source {name:15} {state}" + (f" ({report.get('note')})" if not report.get("available") and report.get("note") else ""))
    for item in document["detections"]:
        lines.append(f"  {clock(item['start_s']):>8}-{clock(item['end_s']):<8} {item['label_name']:22} "
                     f"score {item['score']:3d}  {item['evidence_strength']:6}  {item['summary']}")
        if verbose:
            lines += [f"      {entry['points']:+6.1f}/{entry['max_points']:<4} {entry['text']}" for entry in item["evidence"]]
    skipped = {key: value for key, value in document["summary"]["not_reported"].items() if value}
    if skipped:
        lines.append(f"  not reported: {skipped}")
    return lines


def report_markdown(documents: list[dict[str, Any]]) -> str:
    labels: dict[str, str] = {}
    for document in documents:
        labels.update({label["id"]: label["name"] for label in document["labels"]})
    lines = ["# Automatic symptom detection: candidates per trial", "",
             "Candidates for manual review from the motion baseline. Scores are uncalibrated detection scores,",
             "not probabilities, and nothing here has been validated against reviewed labels.", "",
             "| Trial | Duration | " + " | ".join(labels.values()) + " | Total | High / Medium / Low |",
             "| --- | --- | " + " | ".join("---" for _ in labels) + " | --- | --- |"]
    for document in documents:
        by_label = document["summary"]["by_label"]
        strengths = [sum(bucket[level] for bucket in by_label.values()) for level in ("high", "medium", "low")]
        lines.append(f"| {document['trial_id']} | {clock(document['session_clock']['duration_s'])} | "
                     + " | ".join(str(by_label.get(key, {}).get("total", 0)) for key in labels)
                     + f" | {document['summary']['count']} | {strengths[0]} / {strengths[1]} / {strengths[2]} |")
    for document in documents:
        lines += ["", f"## {document['trial_id']}", "", f"Run `{document['run_id']}`.", ""]
        for name, report in document["sources"].items():
            lines.append(f"- {name}: " + ("used" if report.get("available") else f"not available. {report.get('note') or report.get('reason') or ''}".strip()))
        if document["detections"]:
            lines += ["", "| Start | End | Type | Score | Evidence | Summary |", "| --- | --- | --- | --- | --- | --- |"]
            lines += [f"| {clock(item['start_s'])} | {clock(item['end_s'])} | {item['label_name']} | {item['score']} | "
                      f"{item['evidence_strength']} | {item['summary']} |" for item in document["detections"]]
        skipped = {key.replace("_", " "): value for key, value in document["summary"]["not_reported"].items() if value}
        if skipped:
            lines += ["", "Not reported: " + ", ".join(f"{value} {key}" for key, value in skipped.items()) + "."]
    lines += ["", "## Notes", ""] + [f"- {note}" for note in documents[0]["summary"]["notes"]] if documents else []
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", nargs="*", help="Trial IDs. Defaults to every trial with a session manifest.")
    parser.add_argument("--apply", action="store_true", help="Save the predictions. Without it nothing is written.")
    parser.add_argument("--symptom-store", type=Path, default=None,
                        help="Folder for predictions (default: runs/). Use the same value as for the server.")
    parser.add_argument("--config", type=Path, help="JSON file with overrides for the default configuration.")
    parser.add_argument("--no-video", action="store_true", help="Skip the downward video analysis.")
    parser.add_argument("--no-audio", action="store_true", help="Skip the audio envelope.")
    parser.add_argument("--report", type=Path, help="Also write a Markdown summary of the candidates to this file.")
    parser.add_argument("--verbose", action="store_true", help="Print the evidence items of every candidate.")
    args = parser.parse_args(argv)

    trials = args.trials or discover(RUNS_DIR)
    if not trials:
        raise SystemExit(f"No trials with session_dashboard_manifest.json under {RUNS_DIR}.")
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
    store_dir = (args.symptom_store or RUNS_DIR).resolve()
    documents = []
    for trial_id in trials:
        cache_dir = detection_store.detections_dir(store_dir, trial_id) / "cache" if args.apply else None
        try:
            document = run_trial(RUNS_DIR, trial_id, config, use_video=not args.no_video,
                                 use_audio=not args.no_audio, cache_dir=cache_dir)
        except (FileNotFoundError, ValueError) as error:
            print(f"skip    {trial_id}: {error}")
            continue
        documents.append(document)
        print("\n".join(describe(document, args.verbose)))
        if args.apply:
            path, status = detection_store.save_detections(store_dir, document)
            try:
                shown = path.relative_to(QWEN_REPO).as_posix()
            except ValueError:
                shown = str(path)
            print(f"  {status}: {shown}")
    if args.report and documents:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_markdown(documents), encoding="utf-8")
        print(f"Report written to {args.report}")
    if not args.apply:
        print("Dry run: nothing was written. Add --apply to save the predictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
