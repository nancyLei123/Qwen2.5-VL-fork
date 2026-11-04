# PD Insighter: Collaborator Quick Start

**Goal: open the saved dashboard, videos, labels, and 3D scenes on your computer.**
You do not need to rerun MPS, motion classification, or reconstruction training.

## 1. Choose One Starting Point

| What you have | What to do |
| --- | --- |
| The complete OneDrive snapshot | Use its `workspace/` folder. Code and saved results are already included; GitHub is optional. |
| Only GitHub code | Also download results/data from OneDrive. Code alone cannot display the recordings or trained scenes. |
| A working dashboard, and you want to develop new code | Keep the OneDrive snapshot unchanged; use the GitHub branch in a separate working copy. |

[OneDrive snapshot](https://adminliveunc-my.sharepoint.com/my?id=%2Fpersonal%2Fjlei_ad_unc_edu%2FDocuments%2FPD_Insighter_Reproduction_20261101)
is a **2026-11-01 snapshot**, not a live mirror. The GitHub branch
[`pd-insighter/research-20261103`](https://github.com/nancyLei123/Qwen2.5-VL-fork/tree/pd-insighter/research-20261103)
contains separately versioned code. New source templates do not automatically
update older saved pages. Neither package is the older `PD-Insighter-Demo`.

## 2. Keep the Folder Layout

Download files fully, not just cloud placeholders. Work in a local copy outside
the shared sync folder, so annotation edits do not modify the shared baseline.

```text
workspace/
  Qwen2.5-VL-fork/
    scripts/
    docs/reproduction/
    vendor/gaussian-viewer/
    runs/                       saved pages, labels, videos, and 3D models
  VRS_Pre_Processing/           raw/MPS and extracted media, if downloaded
  downward videos/             original downward media, if downloaded
```

For a **complete snapshot**, also keep `VERIFY_DOWNLOAD.py` and
`SOURCE_SHA256.jsonl` next to `workspace/`. From that parent directory, run
`python3 VERIFY_DOWNLOAD.py --size-only`, then `python3 VERIFY_DOWNLOAD.py`
for the full checksum check. Use Python 3.11; on Windows use `py -3.11` instead
of `python3`. Verify **before** changing paths or mixing in newer code. A partial
download will report missing files; the full check reads about 449 GB.

**GitHub alternative:** in a fresh `workspace/`, run:

```bash
git clone --branch pd-insighter/research-20261103 --single-branch https://github.com/nancyLei123/Qwen2.5-VL-fork.git Qwen2.5-VL-fork
```

Or select that branch on GitHub, choose **Code > Download ZIP**, extract it, and
rename the extracted folder to exactly `Qwen2.5-VL-fork` inside `workspace/`.
Add OneDrive's `runs/` and needed sibling data directories in the layout above.
Do not copy older OneDrive code over the newer GitHub code. For selected trials,
use the [folder map](DATA_LOCATIONS.md); include shared `runs/` assets and the full
trial folder, not just its HTML page. Some older pages reference extracted media.

## 3. Install and Start

Install **Python 3.11** and **Node.js/npm**. Use a modern WebGL-capable browser.
In Terminal, change to your local **`workspace/`** directory.

On macOS/Linux:

```bash
python3.11 -m venv .venvs/aria-reconstruction
source .venvs/aria-reconstruction/bin/activate
```

On Windows PowerShell, use these two lines instead:

```powershell
py -3.11 -m venv .venvs/aria-reconstruction
.\.venvs\aria-reconstruction\Scripts\Activate.ps1
```

Then, on either platform:

```bash
python -m pip install -r Qwen2.5-VL-fork/docs/reproduction/requirements-viewer.txt
npm ci --prefix Qwen2.5-VL-fork/vendor/gaussian-viewer --ignore-scripts
python Qwen2.5-VL-fork/docs/reproduction/relocate_copy.py --root .
```

The last line previews path changes. After reviewing them, apply them **only to
your downloaded working copy**, then start the server:

```bash
python Qwen2.5-VL-fork/docs/reproduction/relocate_copy.py --root . --apply
python Qwen2.5-VL-fork/scripts/pd_insighter_server.py --host 127.0.0.1 --port 8782
```

Open [PD Insighter](http://127.0.0.1:8782/Qwen2.5-VL-fork/runs/pd_insighter_trial_selector.html).
Keep the terminal running. To reopen later, activate the same environment and
run the server command again; installation and relocation are one-time steps.

## 4. Know What This Reproduces

- This opens **saved results**, not a fresh execution of every analysis algorithm.
  Retraining/recomputation needs additional dependencies and the saved experiment
  settings; see the [technical guide](../../PD_INSIGHTER.md).
- Do not double-click HTML or substitute OneDrive URLs for local media paths.
  OneDrive links are for downloading, not dashboard video streaming.
- A missing page/video usually requires checking the download and folder paths.
  For blank 3D, also check the npm installation, model files, and browser WebGL.
  Explicit "No 3D coverage" intervals may be limitations of the saved results.
- The instructions are based on the Mac research workspace. Full Windows/Linux
  reproduction has not been certified; some historical tools use Mac-specific paths.
- Keep the server on `127.0.0.1`. Share code and this guide publicly if desired,
  but grant research-data access only to approved collaborators in OneDrive.
