# Reprocessing notes

Opening all saved pages and recomputing every scientific result are different
tasks. `start.py` opens saved results and the existing local annotation API. It
does not rerun classification, request MPS, or retrain models.

## Preserved inputs

- `workspace/Qwen2.5-VL-fork/runs/<trial>/`: full session results, dual-view review
  videos, synchronization metadata, motion/stride/transition payloads, annotation
  stores when present, 3D viewer, selected reconstruction images and trained models.
- `workspace/VRS_Pre_Processing/data/Fuchs_Home_20261103/`: VRS/JSON and MPS SLAM
  trajectory, calibration, semi-dense points/observations and health checks.
- `workspace/VRS_Pre_Processing/data/extracted_videos/<trial>/`: full Aria
  MP4/audio and frame/extraction metadata.
- `workspace/downward videos/<trial>.mp4`: original downward recording.
- `workspace/Qwen2.5-VL-fork/scripts/` and `tests/`: current source snapshot,
  including uncommitted changes, not merely an older GitHub commit.
- `workspace/tools/OpenSplat-1.2.2/`: source snapshot without machine-specific
  build products; preserve the locally modified implementation when comparing.

## Repeat analyses in a separate working copy

Use the delivered results as a baseline. Install the required analysis packages
separately: the viewer requirements alone do not install Project Aria Tools,
matplotlib, ffmpeg/ffprobe or a GPU reconstruction toolchain. The observed package
versions are in `ENVIRONMENT_VERSIONS.json`, not a certified cross-platform lockfile.

Read each script's `--help` and the saved trial/session manifests before running:

1. `extract_registered_trial.py` and VRS preprocessing scripts extract Aria
   images/audio. Reuse delivered MPS results; no new cloud submission is needed.
2. `build_parkinson_motion_review.py` and `refresh_simulated_pd_dashboards.py`
   implement/publish the simulated-PD classification workflow.
3. `generate_step_stride_diagnostics.py` generates the gait-proxy diagnostics.
4. `attach_downward_to_existing_trial.py` aligns the paired downward recording
   without shifting the Aria clock. Preserve the delivered offsets, quantization,
   padding and waveform-correlation verification records.
5. `build_downward_dual_view_dashboards.py`, `build_full_session_manual_labeling.py`
   and `refresh_transition_review_pages.py` publish review interfaces.
6. `export_aria_gaussian_dataset.py`, `train_aria_gaussian_scene.py`,
   `run_registered_reconstruction.py` and `build_registered_scene_playback.py`
   prepare calibrated views, train and publish Gaussian submaps.

Do **not** run `process_registered_aria_only.py` on these paired trials: it is an
earlier Aria-only stage and intentionally rejects trials with downward sync data.
Do not blindly run a historical normal-motion pipeline over simulated-PD results.

Each submap's `trained_7000_fullres/training_summary.json` records the actual
command and parameters, held-out image and observed metrics. Those commands
contain the original GPU binary location and must be adapted to the new machine.
Training used 7,000 iterations, 1,024-square calibrated images, SH2 and Apple MPS;
other GPUs may not reproduce identical floating-point results. The reconstruction
driver skips existing trained results. To compare new training, pass a new output
directory to the trainer, rather than overwriting the delivered baseline.

The launcher relocates known workspace paths only in the downloaded copy and
preserves original text under `.relocation_originals`. It cannot fix external
Homebrew, GPU/compiler or OS-specific dependencies. Fresh-machine retraining of
all stages is not certified by this handoff. See `VALIDATION.json` for precisely
which checks were performed; passing software tests is not clinical validation.
