# Code Release Verification

Source snapshot: 2026-11-03, branch `pd-insighter/research-20261103`.
This is a source release, not a claim of complete clinical or cross-platform
validation. No new reconstruction training was performed for this release.

## Checks Performed

| Check | Observed result |
| --- | --- |
| Python tests in a staged-source-only export, without trial data | 255 passed, 14 skipped |
| JavaScript tests in that export, after installing locked npm dependencies | 48 passed |
| Python tests in the original workspace with saved trial artifacts | 268 passed, 1 failed; see below |
| Staged content audit | UTF-8 text only; no research media, model binaries, raw data files, or detected credentials |
| Supporting source copies | 184 files matched original bytes and SHA-256 manifest |
| OneDrive folder map | 14 folder references matched the saved snapshot inventory |

Tests used the existing local Python environment: Python 3.11.8, NumPy 2.4.6,
SciPy 1.17.1, pytest 9.2.0, and Node.js 24.20.0. The source-only export reused
that Python environment; this was not an installation test on a clean machine.
JavaScript dependencies were restored from the committed npm lockfile with
`npm ci --prefix vendor/gaussian-viewer --ignore-scripts --no-audit --no-fund`.

Two synthetic rotation fixtures now use explicit `(N, 1)` angle arrays for
single-axis batches, compatible with the installed SciPy. This does not change
the processing algorithms, recorded poses, or reconstruction outputs.

## Known Saved-Artifact Mismatch

`tests/test_long_trial_route_artifacts.py::test_deployed_route_interface_matches_source`
fails in the original workspace because the saved `60minutes_1` playback JavaScript
does not exactly match today's shared source template. The template includes newer
dedicated-room options, including hiding action tracks when labels do not apply.

This is a byte-equality deployment check, not evidence of corrupted videos or
retrained models. The test remains intact; the historical saved dashboard was not
overwritten to make it pass. The GitHub branch contains the current source, while
OneDrive preserves the separately dated data/results snapshot. Reviewing saved
results and regenerating their interface are different operations.

## Limits

- Content scanning reduces accidental disclosure risk but is not a security audit.
- Folder-map checks do not independently verify all cloud bytes or grant access.
- Tests do not establish medical accuracy, reconstruction quality on new data,
  or compatibility with every OS/GPU.
- Historical scripts still need platform-specific dependency/path configuration.
  Preserve annotations and work on a copy before regenerating trial pages.
