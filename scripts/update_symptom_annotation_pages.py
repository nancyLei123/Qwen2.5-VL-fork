#!/usr/bin/env python3
"""Add or refresh the Symptom Annotation view on an installation with trial data in runs/.

Without ``--apply`` this is a dry run: it lists eligible trials, missing prerequisites and the
pages that would be written, and writes nothing. With ``--apply`` it:

1. builds ``runs/<trial>/pd_insighter_symptom_annotation.html`` (and ``symptom_annotation/``
   assets) for each eligible trial;
2. refreshes the navigation of the trial's existing views so they show the tab (navigation
   markup only, the same refresh pd_insighter_environment_navigation.py performs);
3. rebuilds the trial library so its cards link the new page.

Eligible: ``runs/<trial>/session_dashboard_manifest.json`` with a session duration. Saved symptom
annotations, labels, manual activity labels and analysis results are never read or written here.
Running it again is safe. Restart scripts/pd_insighter_server.py afterwards so it serves
/api/symptom-labels and /api/symptom-annotations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_symptom_annotation_page as builder  # noqa: E402
import pd_insighter_environment_navigation as navigation  # noqa: E402
from symptom_annotation_navigation import SYMPTOM_PAGE  # noqa: E402


def planned_pages(trials: list[str]) -> list[tuple[Path, str]]:
    """(page, "create" | "update") for every page whose content would change."""
    changes = []
    building = frozenset(trials)
    for trial in trials:
        trial_dir = builder.RUNS_DIR / trial
        page = trial_dir / SYMPTOM_PAGE
        html = builder.build_html(builder.load_payload(trial, building), trial_dir)
        current = page.read_text(encoding="utf-8") if page.is_file() else None
        if current != html:
            changes.append((page, "create" if current is None else "update"))
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", nargs="*", help="Trial IDs. Defaults to every trial with a session manifest.")
    parser.add_argument("--apply", action="store_true", help="Write the changes. Without it nothing is written.")
    parser.add_argument("--skip-library", action="store_true", help="Do not rebuild the trial library.")
    args = parser.parse_args()
    requested = args.trials or builder.discover()
    if not requested:
        raise SystemExit(f"No trials with session_dashboard_manifest.json under {builder.RUNS_DIR}.")

    trials, skipped = [], []
    for trial in requested:
        issues = builder.problems(trial)
        blocking = [issue for issue in issues if "no session video" not in issue]
        for issue in issues:
            print(f"{'skip   ' if blocking else 'warning'} {issue}")
        (skipped if blocking else trials).append(trial)
    if not trials:
        raise SystemExit("No eligible trials.")

    pages = planned_pages(trials)
    for path, reason in pages:
        print(f"{reason:7} {path.relative_to(builder.QWEN_REPO).as_posix()}")
    if not args.apply:
        print(f"Dry run: {len(trials)} eligible trial(s), {len(pages)} page(s) to write, {len(skipped)} skipped. "
              "Navigation and the trial library are refreshed with --apply.")
        return

    for trial in trials:
        builder.build(trial, frozenset(trials))
    refreshed = navigation.refresh_navigation(trials)
    for path in refreshed:
        print(f"navigation {path.relative_to(builder.QWEN_REPO).as_posix()}")
    if not args.skip_library:
        import build_pd_insighter_trial_selector as library
        library.main()
    print(f"Done: {len(pages)} page(s) written, {len(refreshed)} view(s) refreshed. "
          "Restart scripts/pd_insighter_server.py to serve the annotation API.")


if __name__ == "__main__":
    main()
