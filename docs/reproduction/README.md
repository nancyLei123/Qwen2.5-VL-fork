# PD Insighter: Research Reproduction Handoff

**For the current GitHub-code / OneDrive-data workflow, use
[PD_INSIGHTER.md](../../PD_INSIGHTER.md) and [Data Locations](DATA_LOCATIONS.md).**
The archive instructions below document the earlier ZIP-based handoff and do
not describe the direct `workspace/` directory now provided on OneDrive.

Prepared 2026-11-01. This package preserves the current local research work, not
just the earlier lightweight public demo. Read INVENTORY.md and inventory.json
beside the archives for the exact included files and unresolved gaps.

## Choose What You Need

1. **Explore the current results:** download code, shared results, the desired
   trial's results, and its environment package if using 3D reconstruction.
   Synchronized dashboard videos live in the trial results package. Some older
   pages reference extracted media or shared plots; those packages are also
   needed when following those links.
2. **Recompute motion/stride analysis:** additionally download that trial's raw
   VRS/MPS, extracted Aria media, and original downward recording. Preserve the
   audio synchronization metadata and manual labels, not just the videos.
3. **Retrain 3D reconstruction:** additionally download calibrated images,
   COLMAP files, poses, calibration, point clouds, saved training settings,
   evaluation splits, and reference models from the environment package.

The inventory is deliberately broader than a demo. It includes early trials,
current normal trials, the three simulated-Parkinson trials, RoomScan3c, and
shared analysis where files exist. Some archival recordings, notably Longoria,
are not on this laptop. Their absence must not be mistaken for successful backup.

## Package Layout

All ZIPs extract into the **same parent directory**. Each ZIP contains part of
`workspace/`, plus a `_package_checksums/` record. Do not put each ZIP into a
separate nested folder. Parts are independent ZIPs, not split ZIP volumes.

```text
workspace/
  Qwen2.5-VL-fork/
    scripts/                  analysis, viewers, annotation server, tests
    runs/<trial>/             current results, labels, video alignment, dashboards
    vendor/                   browser Gaussian renderer
    docs/reproduction/        this guide and relocation tool
  VRS_Pre_Processing/
    scripts/                  VRS media extraction
    data/<recording>/          original VRS/JSON and MPS products
    data/extracted_videos/     full Aria MP4/audio and extraction timestamps
  downward videos/            original/combined downward videos
  projectaria_tools/          source snapshot; not an installed environment
  tools/OpenSplat-1.2.2/       source snapshot, including local modifications
```

Group names: `00_code`, `10_results_*`, `20_environment_*`, `30_raw_mps_*`,
`40_extracted_*`, `50_downward_*`, `60_support_data`.
The approximate ZIP target is 2 GiB; a single large VRS/video can exceed this.
Original recordings are not recompressed or cut. Existing ZIPs and model
experiments may overlap in content; nothing is deleted from the source project.

## Start the Dashboard

Use Python 3.11 and a modern WebGL-capable browser. No MPS account or GPU training
is required to view saved results. A GPU/WebGL implementation is needed for 3D.

From the extracted `workspace/` directory:

```bash
python3.11 -m venv .venvs/aria-reconstruction
source .venvs/aria-reconstruction/bin/activate
python -m pip install -r Qwen2.5-VL-fork/docs/reproduction/requirements-viewer.txt
python Qwen2.5-VL-fork/docs/reproduction/relocate_copy.py --root "$PWD"
```

The last command is a **dry run**. Review its output. Then, only in your extracted
copy, update historical absolute paths:

```bash
python Qwen2.5-VL-fork/docs/reproduction/relocate_copy.py --root "$PWD" --apply
python Qwen2.5-VL-fork/scripts/pd_insighter_server.py --host 127.0.0.1 --port 8782
```

Open `http://127.0.0.1:8782/Qwen2.5-VL-fork/runs/pd_insighter_trial_selector.html`.
Use the annotation server rather than opening HTML with `file://`; otherwise
video seeking, fetch requests, or saved labels may not work. Keep the terminal
running. Do not expose this research server to the public Internet: it has no
production authentication layer. A read-only static server is an alternative
for viewing but does not provide annotation persistence.

The viewer requirements are an installation starting point. The separate
`environment_versions.json` is the observed environment used on the original
Mac, not a guaranteed cross-platform lockfile. A clean-machine end-to-end
installation has not yet been certified for this handoff.

## Preserve the Scientific State

- Ntrial10, Parki, and Parki2 are **simulated Parkinson movement**, not diagnosed
  patient recordings. Their saved method is `parkinson_high_v1`.
- Normal trials use the separately developed normal-trial pipeline. Historical
  scripts remain in the snapshot; running a generic old refresh command can
  replace a newer continuous-classification export. Inspect the saved manifest
  and source/version metadata before rerunning anything.
- Ntrial10 action labels remain in
  `runs/Ntrial10/parkinson_motion_review/manual_segments.json`.
- Other full-session labels generally use
  `runs/<trial>/manual_labels/session_manual_segments.json`.
- Freezing/gait markers are a separate overlapping layer, generally
  `parkinson_motion_review/manual_freezing_segments.json`. They must not replace
  mutually exclusive base-action labels.
- Label seconds refer to the existing synchronized **Aria session clock**.
  Do not shift labels when changing a downward-video offset. Keep the build JSON,
  synchronization results, and session manifest with each pair of videos.
- Step/stride and left/right outputs are head-motion-based estimates requiring
  video validation, not direct foot-contact measurements or a diagnosis.
- Existing accuracy is agreement with the available reference labels. The
  training/tuning trials are not an independent clinical validation cohort.

## Recompute Analysis Carefully

The principal entry points are:

- `register_trial.py`, `extract_registered_trial.py`, `run_registered_trial.py`:
  registration, VRS extraction, and the foundational clip/SLAM analysis.
- `continuous_motion_predictions.py` and
  `refresh_continuous_motion_dashboards.py`: continuous normal-trial outputs.
- `build_parkinson_motion_review.py` and `refresh_simulated_pd_dashboards.py`:
  the simulated-PD workflow; the latter republishes saved classifications.
- `generate_step_stride_diagnostics.py`: step/stride diagnostics.
- `build_downward_dual_view_dashboards.py`,
  `build_full_session_manual_labeling.py`, and
  `refresh_transition_review_pages.py`: synchronized review pages.

Start from a separate working copy, install the analysis dependencies appropriate
to your platform, and run each entry point with `--help`. Preserve the delivered
results as a baseline. Do not run MPS again merely to open or recompute existing
local results; the delivered MPS products should be reused. Access to new MPS
requests requires your own authorized Meta account, not credentials in this pack.

The older STANDARD_TRIAL_WORKFLOW.md explains the history, but its original
Python paths and older clip-level stages are not a one-command recipe for the
latest complete application. Some figures, spreadsheets and trial-specific
experiments were generated separately. The inventory and saved output manifests
are the authoritative record of what is present.

## Reconstruct the Environment

The current approach uses Project Aria Tools to retrieve RGB frames, calibration,
and MPS poses, prepares calibrated images/COLMAP-format inputs, and trains
OpenSplat scenes initialized by MPS geometry. GaussianSplats3D renders the saved
models in the browser, with synchronized video, position, orientation and action
labels. Local OpenSplat modifications are included; rebuilding stock upstream
alone may not reproduce all latest experiments.

Relevant entry points include `export_aria_gaussian_dataset.py`,
`train_aria_gaussian_scene.py`, `compare_aria_gaussian_quality.py`,
`build_aria_scene_playback.py`, and the trial-specific long-session/room scripts.
Keep each dataset manifest, camera convention, image timestamp, metric MPS frame,
train/validation split, training recipe and evaluation result together.
Do not promise identical floating-point/GPU output on a different machine.

Metashape comparison outputs may be included, but the proprietary application
and license are not. RealityCapture is not a reproduced result in this snapshot.
The source snapshots retain their original licenses; inclusion is not a new
blanket license for the research data or third-party software.

## Sharing and Verification

This package contains private scene imagery, voices, device metadata, trajectories
and annotations. It is **not anonymized**. Use a restricted research folder with
named collaborators. Do not publish raw videos, maps or account credentials to
a public GitHub repository. GitHub is suitable for separately reviewed code and
documentation, not as a substitute for this data handoff.

Every completed ZIP has a SHA-256 report and internal per-file checksums. Packing
checks that each file did not change while being read and tests the ZIP CRC.
An inventory alone does **not** mean a package was built or uploaded. Upload status
must be verified separately against the destination folder. Keep local originals
until a complete remote verification and an independent backup are available.

The relocation utility skips annotation stores and writes original text backups
inside the extracted copy. It does not fix missing source data, install GPU
dependencies, or automatically certify all HTML links.
