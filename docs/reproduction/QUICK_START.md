# PD Insighter: Run and Develop the Three-Trial Demo

Use **`pd-insighter/develop`**, not the upstream `main` branch or the older
`PD-Insighter-Demo` repository. These instructions open saved results; no new MPS
submission, motion analysis or GPU training is needed.

## 1. Download the Private Result Package

From the [Fuchs / Fuchs22 / Fuchs333 Shared Drive folder](https://drive.google.com/drive/folders/1r1Zq2lqtibt5cRs3-8nIVU-BDW_jY2Gk),
download **`01_PD_Insighter_App_and_Results.zip`** once its upload is complete
(expected size: 1,901,900,497 bytes). Unzip it completely. The extracted folder
is `PD_Insighter_Fuchs_20261105`. Keep this unchanged as your reference copy.
Access is restricted; a GitHub account or this link does not grant data access.
The three `02_Raw_*.zip` files are optional for reprocessing, not viewing.

For viewing only, the package's own README and `start.py` are sufficient.
For editing and exchanging code, continue below in a separate local workspace,
outside cloud-synced folders. Budget approximately another 2 GB for the result copy.

## 2. Clone Current Code

Install Git, Python 3.11 or newer, and Node.js 20+ with npm. In a new directory:

```bash
mkdir pd-workspace
cd pd-workspace
git clone --single-branch --branch pd-insighter/develop https://github.com/nancyLei123/Qwen2.5-VL-fork.git Qwen2.5-VL-fork
cd Qwen2.5-VL-fork
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r docs/reproduction/requirements-dev.txt
npm ci --prefix vendor/gaussian-viewer --ignore-scripts
```

Windows PowerShell: replace the two environment commands with
`py -3 -m venv .venv` and `.\.venv\Scripts\Activate.ps1`.
Full Windows/Linux processing has not been certified; the viewer is intended to
use the same layout across platforms. Training dependencies are separate.

## 3. Import Data Without Overwriting Code

Replace `/path/to/PD_Insighter_Fuchs_20261105` with your extracted package path.
Run from the cloned `Qwen2.5-VL-fork` directory:

```bash
python docs/reproduction/import_fuchs_data.py --package "/path/to/PD_Insighter_Fuchs_20261105"
python docs/reproduction/import_fuchs_data.py --package "/path/to/PD_Insighter_Fuchs_20261105" --apply
python docs/reproduction/relocate_copy.py --root .. --data-only
python docs/reproduction/relocate_copy.py --root .. --data-only --apply
```

The first command previews the import. `--apply` verifies the package's per-file
SHA-256 records and copies only `runs/`, including all three trials and shared
assets. It does not copy old scripts over GitHub code. It refuses an existing
`runs/`, so it cannot overwrite your annotations or results. On an interrupted
copy, retain it and use a fresh clone rather than merging blindly. Relocation
changes known paths only in your data copy, backs up changed text and skips
manual annotation stores; it leaves tracked source unchanged.

Do not run the package launcher to verify this new mixed code/data checkout;
its snapshot checksums describe the unchanged ZIP, not newer Git commits.

## 4. Run

```bash
python scripts/pd_insighter_server.py --host 127.0.0.1 --port 8782
```

Open [the trial library](http://127.0.0.1:8782/Qwen2.5-VL-fork/runs/pd_insighter_trial_selector.html).
Keep the terminal open. Use another port, such as 8797, if 8782 is occupied.
Use Chrome or Edge with WebGL, not `file://`. Keep the server on localhost.

Each trial has **Full Session / Manual Labels / Stride / Transitions /
Parkinson Review / Environment**. Missing video, poses or scene coverage remain
marked as missing. These are simulated movements and research estimates, not
clinically validated measurements or reconstructed full-body motion.

## 5. Work Together

See [CONTRIBUTING](../../CONTRIBUTING.md) for feature branches, pull requests,
safe `git pull --ff-only`, and regenerating saved pages from edited templates.
Git updates code, not ignored `runs/` or shared research data. Never copy snapshot
scripts over this checkout after a pull. Do not commit videos, model files,
coordinates, labels, private reports or credentials, even to a feature branch.

For other trials, see the historical [data map](DATA_LOCATIONS.md). The November 1
OneDrive package is a different baseline; do not assume it includes these newest
trials. For raw-data reprocessing, see [REPROCESSING](../fuchs_handoff/REPROCESSING.md).
