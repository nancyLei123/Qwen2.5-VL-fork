# PD Insighter: Code on GitHub, Data on OneDrive

This branch preserves the research source as of 2026-11-03. It is separate from
the older `PD-Insighter-Demo` repository. The matching data snapshot is
`PD_Insighter_Reproduction_20261101`; code and data have separate version dates.

For the shortest path to opening the saved dashboard, see the
[Collaborator Quick Start](docs/reproduction/QUICK_START.md). A complete OneDrive
download already contains snapshot code; cloning GitHub is an alternative, not
a required extra step. Use GitHub for separately versioned ongoing development.

## What Is Here

- `scripts/`: synchronization, automatic classification, stride and transition
  analysis, reconstruction preparation/training, and the local annotation server.
- `scripts/templates/`: synchronized video/3D playback and route-view interfaces.
- `tests/`: Python and JavaScript checks; some require separately downloaded data.
- `vendor/gaussian-viewer/`: npm manifest and lockfile, not installed packages.
- `support/workspace/`: extraction/helper scripts and the locally modified
  OpenSplat source, preserved in their workspace-relative layout.

This branch does not add research videos, images, labels, trajectories, model
weights, generated trial pages, private meeting reports, accounts, or environments.
Existing upstream Qwen example assets remain in the repository history.

## View Saved Results

1. Clone this branch into a folder named **`Qwen2.5-VL-fork`**, inside a new local
   `workspace` directory. Keep this layout because the server uses sibling paths.
2. From [Data Locations](docs/reproduction/DATA_LOCATIONS.md), download the desired
   `workspace/Qwen2.5-VL-fork/runs/<trial>/` directory into the clone's `runs/`.
   Include its `environment_reconstruction/` for 3D playback. Preserve nested paths.
3. Download the shared files directly under OneDrive's `runs/`, including the trial
   selector and shared UI assets. For links to other trials or shared plots, download
   those targets too. Copy only data/results from the snapshot; do not overwrite
   newer GitHub source with older snapshot code.

From the new local `workspace/` directory:

```bash
git clone --branch pd-insighter/research-20261103 --single-branch \
  https://github.com/nancyLei123/Qwen2.5-VL-fork.git Qwen2.5-VL-fork
# Download the selected data into the layout above before continuing.
python3.11 -m venv .venvs/aria-reconstruction
source .venvs/aria-reconstruction/bin/activate
python -m pip install -r Qwen2.5-VL-fork/docs/reproduction/requirements-viewer.txt
npm ci --prefix Qwen2.5-VL-fork/vendor/gaussian-viewer
python Qwen2.5-VL-fork/docs/reproduction/relocate_copy.py --root "$PWD"
```

The last command is a dry run. Review it, then update legacy absolute paths in
this **downloaded copy only** and start the local server:

```bash
python Qwen2.5-VL-fork/docs/reproduction/relocate_copy.py --root "$PWD" --apply
python Qwen2.5-VL-fork/scripts/pd_insighter_server.py --host 127.0.0.1 --port 8782
```

Open `http://127.0.0.1:8782/Qwen2.5-VL-fork/runs/pd_insighter_trial_selector.html`.
RoomScan3c can be opened directly at
`http://127.0.0.1:8782/Qwen2.5-VL-fork/runs/RoomScan3c/environment_reconstruction/playback.html`.
Use HTTP, not `file://`, for video seeking and data loading. Keep the server on
localhost: it is a research server without production authentication.

OneDrive folder links are **download/navigation references**, not streaming URLs
for the dashboard. Each collaborator needs permission and a local copy of the
selected data. Keep work on a separate copy to avoid syncing annotation edits
back into the shared baseline.

## Rerun Analysis or Reconstruction

Additionally obtain the trial's raw VRS/JSON, MPS products, extracted Aria media,
original downward video, and synchronization metadata. The folder map identifies
these locations. Preserve saved manual labels before rerunning any generator.

- Normal continuous classification: `continuous_motion_predictions.py` and
  `refresh_continuous_motion_dashboards.py`.
- Simulated-PD review: `build_parkinson_motion_review.py` and
  `refresh_simulated_pd_dashboards.py`.
- Step/stride diagnostics: `generate_step_stride_diagnostics.py`.
- Calibrated reconstruction inputs: `export_aria_gaussian_dataset.py`.
- Training and playback: `train_aria_gaussian_scene.py` and
  `build_aria_scene_playback.py`.
- Dedicated room workflow: `reconstruct_dedicated_room.py` and
  `build_room_scan_route_playback.py`.

Read each script's `--help` and the saved trial/experiment manifests before use.
These are research workflows, not one universally interchangeable pipeline.
The viewer requirements do not install all analysis/training dependencies.
VRS access needs Project Aria Tools; media processing needs ffmpeg/ffprobe;
OpenSplat must be built for the target GPU/platform. Metashape is optional,
proprietary, and not included. Do not rerun MPS merely to view existing results.

Some historical scripts retain Mac/Homebrew paths, locally chosen binaries, or
an external `TORCH_LIB` path. The relocation tool fixes the known workspace root,
not those external dependencies. Adapt them in the collaborator's copy before
training. Clean-machine, cross-platform reproduction has not been certified.

## Verification and Scientific Limits

OneDrive's `SOURCE_INVENTORY.json`, `SOURCE_SHA256.jsonl`, and `VERIFY_DOWNLOAD.py`
describe the **2026-11-01 snapshot**, not newly edited GitHub code. Verify downloaded
data against that snapshot before relocation; use Git commits for code versions.
`support/SOURCE_MANIFEST.json` separately records copied supporting source hashes.

```bash
# From Qwen2.5-VL-fork, after installing the needed test dependencies:
python -m pytest tests/
node --test tests/*.mjs
```

Artifact-dependent tests may skip or fail when required trials are not downloaded.
See [Release Verification](docs/reproduction/VALIDATION.md) for observed test results
and the known mismatch between the older `60minutes_1` page and today's template.
Passing code tests is not clinical validation or a fresh GPU-training comparison.
Ntrial10, Parki, and Parki2 are simulated-PD recordings. Stride/left-right measures
are head-motion-based estimates requiring video validation; a rendered camera
pose is not a measured full-body skeleton. Keep unsupported 3D/pose intervals
marked as missing rather than treating them as observed movement.

## Publishing Discipline

Keep recordings, coordinates, annotations, saved scientific outputs, and account
information in access-controlled storage. `.gitignore` helps prevent accidental
additions but is not a privacy guarantee. Review the staged diff before each push.
A branch in a public GitHub repository is also public; OneDrive permissions remain
independent. No OneDrive access is granted by publishing a folder link here.
