#!/usr/bin/env python3
"""Compare automatic symptom predictions with reviewed manual annotations.

Nothing is computed until reviewed annotations exist; the script then says so instead of
reporting numbers. It reads the detector runs, the manual annotations and the review decisions
from the symptom store and writes nothing unless ``--output`` is given.

What counts as what
-------------------
* A reference event is a manual annotation whose label matches the predicted symptom type
  (same label ID or same name, or an explicit ``--map prediction_label=annotation_label``).
* Reviewed time is the union of all manual annotations of a trial, whatever their label
  ("No Target Symptom" marks time that was looked at and found clear). Time nobody annotated
  is unreviewed: predictions there are reported as ``unscored``, never as false positives.
  A prediction the reviewer rejected is a false positive wherever it is.
* Annotations labelled "Uncertain" or "Not Observable" are left out on both sides.

Two kinds of agreement are reported separately:

* event level: predictions and reference events are paired one-to-one (highest overlap first);
  a pair needs an intersection-over-union of at least ``--min-iou``. Gives true/false positives,
  false negatives, precision, recall, F1, mean IoU and start/end timing errors.
* time level: seconds of agreement on a 0.1 s grid inside reviewed time.

Annotations created by accepting a prediction are not independent evidence: a detector can
never miss them. They are counted, and every metric is also given for the independent
annotations alone. Use the independent numbers to judge the detector, and do not tune the
detector on the same reviewed events you report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import symptom_annotations as store  # noqa: E402
import symptom_detection_store as detection_store  # noqa: E402

QWEN_REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = QWEN_REPO / "runs"
GRID_S = 0.1
IGNORED_LABELS = ("uncertain", "not_observable")
Interval = tuple[float, float]


def overlap(a: Interval, b: Interval) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def iou(a: Interval, b: Interval) -> float:
    shared = overlap(a, b)
    union = (a[1] - a[0]) + (b[1] - b[0]) - shared
    return shared / union if union > 0 else 0.0


def merge(intervals: list[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def covered(interval: Interval, scope: list[Interval]) -> float:
    """Fraction of an interval that lies inside the scope."""
    length = interval[1] - interval[0]
    return sum(overlap(interval, part) for part in scope) / length if length > 0 else 0.0


def match_events(predictions: list[dict[str, Any]], references: list[dict[str, Any]], min_iou: float) -> list[tuple[int, int, float]]:
    """One-to-one pairs ``(prediction index, reference index, IoU)``, best overlap first."""
    candidates = []
    for i, prediction in enumerate(predictions):
        for j, reference in enumerate(references):
            value = iou((prediction["start_s"], prediction["end_s"]), (reference["start_s"], reference["end_s"]))
            if value >= min_iou and value > 0:
                candidates.append((value, i, j))
    pairs, used_p, used_r = [], set(), set()
    for value, i, j in sorted(candidates, reverse=True):
        if i not in used_p and j not in used_r:
            pairs.append((i, j, value))
            used_p.add(i)
            used_r.add(j)
    return pairs


def ratios(tp: float, fp: float, fn: float) -> dict[str, float | None]:
    precision = tp / (tp + fp) if tp + fp > 0 else None
    recall = tp / (tp + fn) if tp + fn > 0 else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if precision is not None and recall is not None else None)
    return {"precision": precision, "recall": recall, "f1": f1}


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def event_metrics(predictions: list[dict[str, Any]], references: list[dict[str, Any]], scope: list[Interval],
                  rejected: set[str], min_iou: float, min_scope_fraction: float = 0.5) -> dict[str, Any]:
    pairs = match_events(predictions, references, min_iou)
    matched_p = {i for i, _, _ in pairs}
    matched_r = {j for _, j, _ in pairs}
    false_positive, unscored = [], []
    for i, prediction in enumerate(predictions):
        if i in matched_p:
            continue
        inside = covered((prediction["start_s"], prediction["end_s"]), scope) >= min_scope_fraction
        (false_positive if inside or prediction["id"] in rejected else unscored).append(prediction["id"])
    tp, fp, fn = len(pairs), len(false_positive), len(references) - len(matched_r)
    start_errors = [predictions[i]["start_s"] - references[j]["start_s"] for i, j, _ in pairs]
    end_errors = [predictions[i]["end_s"] - references[j]["end_s"] for i, j, _ in pairs]
    return {
        "true_positives": tp, "false_positives": fp, "false_negatives": fn, "unscored_predictions": len(unscored),
        **ratios(tp, fp, fn),
        "mean_iou": mean([value for _, _, value in pairs]),
        "start_error_s": {"mean_signed": mean(start_errors), "mean_absolute": mean([abs(e) for e in start_errors])},
        "end_error_s": {"mean_signed": mean(end_errors), "mean_absolute": mean([abs(e) for e in end_errors])},
        "pairs": [{"prediction": predictions[i]["id"], "reference": references[j]["id"], "iou": round(value, 4)} for i, j, value in pairs],
        "false_positive_ids": false_positive, "unscored_ids": unscored,
        "missed_reference_ids": [reference["id"] for j, reference in enumerate(references) if j not in matched_r],
    }


def time_metrics(predictions: list[dict[str, Any]], references: list[dict[str, Any]], scope: list[Interval]) -> dict[str, Any]:
    """Seconds of agreement inside reviewed time, on a fixed grid."""
    def cells(intervals: list[Interval]) -> set[int]:
        out: set[int] = set()
        for start, end in intervals:
            out.update(range(int(round(start / GRID_S)), int(round(end / GRID_S))))
        return out

    in_scope = cells(scope)
    predicted = cells([(p["start_s"], p["end_s"]) for p in predictions]) & in_scope
    actual = cells([(r["start_s"], r["end_s"]) for r in references]) & in_scope
    tp, fp, fn = len(predicted & actual), len(predicted - actual), len(actual - predicted)
    union = tp + fp + fn
    return {"true_positive_s": round(tp * GRID_S, 1), "false_positive_s": round(fp * GRID_S, 1),
            "false_negative_s": round(fn * GRID_S, 1), "reviewed_s": round(len(in_scope) * GRID_S, 1),
            **ratios(tp, fp, fn), "iou": tp / union if union else None}


def threshold_sweep(predictions: list[dict[str, Any]], references: list[dict[str, Any]], scope: list[Interval],
                    rejected: set[str], min_iou: float, thresholds: tuple[int, ...] = tuple(range(0, 100, 10))) -> list[dict[str, Any]]:
    rows = []
    for threshold in thresholds:
        kept = [p for p in predictions if p.get("score", 0) >= threshold]
        result = event_metrics(kept, references, scope, rejected, min_iou)
        rows.append({"min_score": threshold, "predictions": len(kept),
                     **{key: result[key] for key in ("true_positives", "false_positives", "false_negatives", "precision", "recall", "f1")}})
    return rows


def reliability_table(outcomes: list[tuple[int, bool]], edges: tuple[int, ...] = (0, 30, 45, 60, 70, 80, 101)) -> list[dict[str, Any]]:
    """Share of scored predictions that were confirmed, per score band. The basis for calibrating scores later."""
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = [confirmed for score, confirmed in outcomes if low <= score < high]
        rows.append({"score_from": low, "score_below": min(high, 100), "predictions": len(inside),
                     "confirmed": sum(inside), "confirmed_fraction": (sum(inside) / len(inside)) if inside else None})
    return rows


def label_matches(annotation_labels: list[dict[str, Any]], prediction_label: dict[str, Any], mapping: dict[str, set[str]]) -> set[str]:
    """Annotation label IDs that count as the predicted symptom type."""
    wanted = set(mapping.get(prediction_label["id"], set())) | {prediction_label["id"]}
    name = str(prediction_label.get("name", "")).strip().casefold()
    wanted |= {label["id"] for label in annotation_labels if str(label.get("name", "")).strip().casefold() == name}
    return wanted


def evaluate_trial(detections: dict[str, Any], annotations: dict[str, Any], reviews: dict[str, Any] | None,
                   labels: list[dict[str, Any]], *, mapping: dict[str, set[str]] | None = None,
                   min_iou: float = 0.1) -> dict[str, Any]:
    """Metrics for one trial and one detector run. ``status`` is ``no_reference`` when nothing was reviewed."""
    mapping = mapping or {}
    items = [a for a in annotations.get("annotations", []) if a.get("label_id") not in IGNORED_LABELS]
    result: dict[str, Any] = {"trial_id": detections.get("trial_id"), "run_id": detections.get("run_id"),
                              "min_iou": min_iou, "predictions": len(detections.get("detections", []))}
    if not items:
        result.update(status="no_reference",
                      message="No reviewed annotations for this trial yet, so no accuracy can be reported.")
        return result
    scope = merge([(a["start_s"], a["end_s"]) for a in items])
    decisions = ((reviews or {}).get("reviews") or {}).get(detections.get("run_id"), {})
    rejected = {key for key, entry in decisions.items() if entry.get("decision") == "rejected"}
    from_predictions = [a for a in items if (a.get("origin") or {}).get("type") == "detection"]
    result.update(status="ok", reviewed_s=round(sum(end - start for start, end in scope), 1),
                  session_s=(detections.get("session_clock") or {}).get("duration_s"),
                  reference_annotations=len(items), references_from_accepted_predictions=len(from_predictions),
                  rejected_predictions=len(rejected), by_label={}, outcomes=[])
    for label in detections.get("labels", []):
        wanted = label_matches(labels, label, mapping)
        predictions = [d for d in detections.get("detections", []) if d.get("label_id") == label["id"]]
        references = [a for a in items if a.get("label_id") in wanted]
        independent = [a for a in references if (a.get("origin") or {}).get("type") != "detection"]
        derived_ids = {(a.get("origin") or {}).get("detection_id") for a in references if a not in independent}
        events = event_metrics(predictions, references, scope, rejected, min_iou)
        result["by_label"][label["id"]] = {
            "label_name": label.get("name"), "predictions": len(predictions), "references": len(references),
            "independent_references": len(independent),
            "event_level": events,
            "event_level_independent_only": event_metrics([p for p in predictions if p["id"] not in derived_ids],
                                                          independent, scope, rejected, min_iou),
            "time_level": time_metrics(predictions, references, scope),
            "by_min_score": threshold_sweep(predictions, references, scope, rejected, min_iou),
        }
        confirmed = {pair["prediction"] for pair in events["pairs"]}
        unscored = set(events["unscored_ids"])
        result["outcomes"] += [(int(p.get("score", 0)), p["id"] in confirmed) for p in predictions if p["id"] not in unscored]
    return result


def combine(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled event counts over trials (micro average) per symptom type and overall."""
    scored = [trial for trial in trials if trial.get("status") == "ok"]
    if not scored:
        return {"status": "no_reference", "message": "No trial has reviewed annotations yet; no accuracy is reported."}
    totals: dict[str, dict[str, int]] = {}
    for trial in scored:
        for label_id, entry in trial["by_label"].items():
            bucket = totals.setdefault(label_id, {"true_positives": 0, "false_positives": 0, "false_negatives": 0, "unscored_predictions": 0})
            for key in bucket:
                bucket[key] += entry["event_level"][key]
    overall = {key: sum(bucket[key] for bucket in totals.values()) for key in ("true_positives", "false_positives", "false_negatives", "unscored_predictions")}
    outcomes = [tuple(outcome) for trial in scored for outcome in trial["outcomes"]]
    return {
        "status": "ok", "trials_with_reference": [trial["trial_id"] for trial in scored],
        "event_level_by_label": {key: {**bucket, **ratios(bucket["true_positives"], bucket["false_positives"], bucket["false_negatives"])}
                                 for key, bucket in totals.items()},
        "event_level_overall": {**overall, **ratios(overall["true_positives"], overall["false_positives"], overall["false_negatives"])},
        "score_reliability": reliability_table(outcomes),
        "note": "Scores stay uncalibrated until each score band holds enough independently reviewed predictions "
                "to fit a calibration curve (for example isotonic regression on score versus confirmed).",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", nargs="*", help="Trial IDs. Defaults to every trial with saved predictions.")
    parser.add_argument("--symptom-store", type=Path, default=None, help="Symptom store (default: runs/).")
    parser.add_argument("--run-id", help="Detector run to evaluate (default: the newest run of each trial).")
    parser.add_argument("--min-iou", type=float, default=0.1, help="Smallest IoU for a prediction to match an annotation.")
    parser.add_argument("--map", action="append", default=[], metavar="PREDICTION=ANNOTATION",
                        help="Count an annotation label ID as a predicted symptom type. Can be repeated.")
    parser.add_argument("--output", type=Path, help="Write the full result as JSON.")
    args = parser.parse_args(argv)

    store_dir = (args.symptom_store or RUNS_DIR).resolve()
    mapping: dict[str, set[str]] = {}
    for pair in args.map:
        left, _, right = pair.partition("=")
        mapping.setdefault(left.strip(), set()).add(right.strip())
    trials = args.trials or sorted(path.name for path in store_dir.iterdir()
                                   if path.is_dir() and detection_store.detections_dir(store_dir, path.name).is_dir()) if store_dir.is_dir() else []
    labels = store.load_labels(store_dir)["labels"]
    results = []
    for trial_id in trials:
        detections = detection_store.load_detections(store_dir, trial_id, args.run_id)
        if detections is None:
            print(f"{trial_id}: no saved predictions.")
            continue
        duration = float((detections.get("session_clock") or {}).get("duration_s") or 0)
        result = evaluate_trial(detections, store.load_annotations(store_dir, trial_id, duration),
                                detection_store.load_reviews(store_dir, trial_id), labels, mapping=mapping, min_iou=args.min_iou)
        results.append(result)
        if result["status"] != "ok":
            print(f"{trial_id}: {result['message']}")
            continue
        print(f"{trial_id}: {result['reference_annotations']} reviewed annotation(s) over {result['reviewed_s']} s "
              f"({result['references_from_accepted_predictions']} created by accepting a prediction), run {result['run_id']}")
        for label_id, entry in result["by_label"].items():
            event = entry["event_level"]
            show = lambda value: "n/a" if value is None else f"{value:.2f}"  # noqa: E731
            print(f"  {entry['label_name']:22} TP {event['true_positives']} FP {event['false_positives']} FN {event['false_negatives']} "
                  f"unscored {event['unscored_predictions']} | precision {show(event['precision'])} recall {show(event['recall'])} "
                  f"F1 {show(event['f1'])} | time IoU {show(entry['time_level']['iou'])}")
    summary = combine(results)
    if summary["status"] != "ok":
        print(summary["message"])
    else:
        overall = summary["event_level_overall"]
        print(f"All trials: TP {overall['true_positives']} FP {overall['false_positives']} FN {overall['false_negatives']} "
              f"(unscored predictions in unreviewed time: {overall['unscored_predictions']})")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"trials": results, "summary": summary}, indent=1), encoding="utf-8")
        print(f"Written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
