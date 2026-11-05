# Working Together on PD Insighter

**Shared code branch: `pd-insighter/develop`. Private results: PD Project Shared Drive.**
Start with [QUICK_START](docs/reproduction/QUICK_START.md). Keep each developer's
own checkout/data copy separate from the shared baseline. Do not use cloud sync
to synchronize a live Git repository or annotation edits.

## Make a Change

First finish or commit your current source edits. `git status` must have no pending
tracked edits before switching branches or pulling; do not reset or discard work
to bypass a conflict. Ignored research data can remain in place.

```bash
git switch pd-insighter/develop
git pull --ff-only
git switch -c feature/short-description
# Edit scripts, templates, tests or documentation.
python -m pytest tests/test_collaboration_setup.py tests/test_environment_navigation.py tests/test_downward_attachment.py tests/test_page_audio.py tests/test_registered_aria_only.py tests/test_trial_library.py
node --test tests/*.mjs
git status --short
git add scripts/NAME.py tests/test_NAME.py
git diff --cached --stat
git diff --cached
git commit -m "Describe the change"
git push -u origin feature/short-description
```

Replace the example file paths with only the source files you actually changed.
Open a GitHub pull request with **base `pd-insighter/develop`**, not `main`.
Describe the change, tests run and any data/manifest versions it requires.
Never force-push the shared branch. Do not attach private data or identifiable
screenshots/logs to public issues or pull requests.

A collaborator needs repository Write access to push branches in this repository.
Google Drive permissions and an email address do not grant GitHub access. If not
invited, fork the public repository, push the feature branch to your own fork,
and open a PR back to `nancyLei123/Qwen2.5-VL-fork:pd-insighter/develop`.

## Get a Labmate's Accepted Changes

After the PR is reviewed and merged, on a clean source checkout:

```bash
git switch pd-insighter/develop
git pull --ff-only
```

If Git reports local changes or diverged history, stop and inspect `git status`
and `git log --oneline --graph --all -15`; do not overwrite either person's work.
Reinstall Python/npm dependencies only when their manifests changed. Restart
the local Python server after server-code changes.

## Source Templates vs Saved Pages

The browser serves generated files in ignored `runs/`. Editing a template or
pulling code does **not** automatically replace existing pages. Edit
`scripts/templates/`, not only a generated copy in `runs/`.

After changing the 3D playback template/modules, use these UI-only commands on
your downloaded working copy; they preserve playback data, labels and models:

```bash
python scripts/build_aria_scene_playback.py --trial-id Fuchs --ui-only
python scripts/build_aria_scene_playback.py --trial-id Fuchs22 --ui-only
python scripts/build_aria_scene_playback.py --trial-id Fuchs333 --ui-only
python scripts/pd_insighter_environment_navigation.py --trials Fuchs Fuchs22 Fuchs333
```

Then reload the local page. The UI-only path does not need the Aria SDK or a GPU.
For the selector use `python scripts/build_pd_insighter_trial_selector.py`.
Other page/analysis generators can change numerical results: read their help,
keep a baseline and follow [REPROCESSING](docs/fuchs_handoff/REPROCESSING.md).
Changes to classifiers require recomputation, not just a browser refresh.

## What Belongs in Git

- Include application scripts, templates, tests with synthetic inputs, dependency
  manifests, setup documentation and small source-only supporting tools.
- Keep original recordings, MPS files, trajectories, manual annotations, experiment
  outputs, checkpoints, generated pages and data packages in restricted storage.
- Do not commit `.env`, tokens, account configuration, installed environments or
  `node_modules`. `.gitignore` is a guardrail, not a security review.

Private RoomScan3c gait cues are deliberately excluded; pass the authorized file
via `python scripts/build_room_gait_fusion.py --cues /private/path/cues.json`.
Scientific artifacts must not be silently replaced by synthetic or inferred data.

The original upstream project and vendored tools retain their licenses. Passing
tests is not clinical validation, cross-platform training certification or proof
of complete 3D surface reconstruction.
