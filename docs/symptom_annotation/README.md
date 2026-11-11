# Symptom Annotation

A review view for marking time intervals in a trial with symptom labels such as
"No Target Symptom", "Simulated Symptom", "Uncertain" and "Not Observable". It sits
next to Full Session, Manual Labels, Stride, Transitions and Parkinson Review in the
trial navigation and in the trial library.

The page shows the Aria and downward session videos side by side, an overview strip of
the whole session, the activity row (automatic segments, manual activity labels or
hidden), the symptom track and a table of saved annotations. Times are dashboard
session time, the same clock as Full Session and Manual Labels.

## Add the view to an installation

The page is generated per trial from the trial's saved manifests, like the other views.
With trial data in `runs/`:

```bash
python scripts/update_symptom_annotation_pages.py
python scripts/update_symptom_annotation_pages.py --apply
```

The first command is a dry run. It lists eligible trials, missing prerequisites and the
pages that would be written, and writes nothing. `--apply`:

1. writes `runs/<trial>/pd_insighter_symptom_annotation.html` and its assets in
   `runs/<trial>/symptom_annotation/`;
2. refreshes the navigation of the trial's existing views so they show the new tab;
3. rebuilds the trial library so each card links the page (`--skip-library` skips this).

A trial is eligible when it has `session_dashboard_manifest.json` with a session
duration. A trial without any session video still gets a page, with a warning. Running
the script again is safe; it never reads or writes saved annotations, manual activity
labels or analysis results. Use `--trials NAME ...` to limit it to some trials.

Restart `scripts/pd_insighter_server.py` afterwards so it serves the annotation API.

For the Fuchs / Fuchs22 / Fuchs333 package, follow
[the reproduction quick start](../reproduction/QUICK_START.md) (import into `runs/` and
relocate), then run the two commands above.

## Trial layouts

- **Dual view (older trials):** both session videos cover the whole session. The
  downward camera keeps the 180 degree mount used by Manual Labels.
- **Downward footage attached to the Aria timeline:** when the session or trial manifest
  has `downward_attachment`, the page reads the recorded window (`start_s` to `end_s`)
  and the rotation (`rotation_quarter_turns`) from it. The video tag shows the window,
  and outside it the downward panel says there is no downward recording at that time.
  The sync details describe the attachment and whether it was audio-verified.
- **Aria-only:** with `media_mode: "aria_only"` the downward panel says the trial has no
  downward recording. Everything else works the same.

Audio uses the shared PD-Insighter page-audio control: one Audio button, off by default,
sound only from the Aria video, and the same saved preference as the other views.

## Where annotations are stored

By default under `runs/`, which git ignores:

- `runs/symptom_annotation_labels.json`: the label set shared by all trials
  (`pd-insighter/symptom-labels@1`);
- `runs/<trial>/symptom_annotations/annotations.json`: one file per trial
  (`pd-insighter/symptom-annotations@1`).

Each save is checked against the stored revision, so two open tabs cannot silently
overwrite each other (the second gets a conflict and reloads). Writes are atomic and keep
the previous file as `.bak`. Labels have stable IDs; a label in use can be archived but
not deleted.

To keep test or practice annotations away from real ones, start the server with a
separate store:

```bash
python scripts/pd_insighter_server.py --host 127.0.0.1 --port 8782 --symptom-store /path/to/scratch-store
```

## API

Served by `scripts/pd_insighter_server.py`:

| Method | Path | Body / query |
| --- | --- | --- |
| GET | `/api/symptom-labels` | |
| POST | `/api/symptom-labels` | `{"revision": n, "labels": [...]}` |
| GET | `/api/symptom-annotations?trial=NAME` | |
| POST | `/api/symptom-annotations?trial=NAME` | `{"revision": n, "annotations": [...]}` |

Errors: 400 invalid data, 404 unknown trial, 409 revision conflict.

## Files

- `scripts/symptom_annotation_navigation.py`: the tab, trial-switcher and session-time
  hooks used by `pd_insighter_environment_navigation.py`.
- `scripts/build_symptom_annotation_page.py`: builds the page for one or more trials.
- `scripts/update_symptom_annotation_pages.py`: the install/update script above.
- `scripts/symptom_annotations.py`: storage and validation.
- `scripts/templates/symptom_annotation/`: page template, CSS and JavaScript.

## Tests

```bash
python -m pytest tests/test_symptom_annotations.py
node --test tests/test_symptom_annotation_core.mjs
```

These need no trial data. The browser tests need a dual-view trial in `runs/` (default
`Parki`). They build two synthetic trials from it, one with attached downward footage
and one Aria-only, and run the server on port 8792 with a temporary annotation store:

```bash
cd tests/e2e
npm ci
npx playwright install chromium
npx playwright test
```
