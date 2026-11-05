# PD Insighter: Fuchs, Fuchs22 and Fuchs333

Private research handoff, November 5, 2026. Please share only with authorized
project collaborators. This includes identifiable home imagery, voices and
movement trajectories; it is not an anonymized public dataset.

## Quick start: view every existing section

Download **01_PD_Insighter_App_and_Results.zip** and extract it completely.
This is enough for all saved results, both synchronized review videos, 3D models,
and local annotation. It includes the current code; cloning GitHub is not required.

Install Python 3.11 or newer, then open a terminal in the extracted
`PD_Insighter_Fuchs_20261105` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python start.py
```

Windows: use `py -3 -m venv .venv` and `.venv\Scripts\activate` for the first two
commands. Keep the terminal open. The browser opens the trial selector at
http://127.0.0.1:8782/Qwen2.5-VL-fork/runs/pd_insighter_trial_selector.html .
If the port is occupied, use `python start.py --port 8797` instead.
Do not open the HTML files by double-clicking. Use Chrome or Edge with WebGL.

Use the tabs: **Full Session / Manual Labels / Stride / Transitions /
Parkinson Review / Environment**. Environment includes synchronized video,
camera position/facing, nearby route navigation, point-cloud overview and the
saved Gaussian submaps. Annotation changes are saved only to your downloaded
copy. Back up that copy before experiments; do not overwrite the shared baseline.

Internet is needed to install Python dependencies, not to fetch private data while
viewing. Browser libraries are bundled. No Meta account or GPU training setup is
required to view saved results. The local server is not a public hosting service;
keep it bound to localhost. The Shared Drive link is for downloading, not hosting
or streaming the website.

## Optional: original data for reprocessing

Each `02_Raw_*.zip` contains that trial's original VRS and VRS JSON, available
MPS SLAM outputs, original downward recording, full extracted Aria video/audio
and extraction metadata. Download only the trials needed. Extract additional
ZIPs into the **same parent directory** as the app ZIP, merging the common
`PD_Insighter_Fuchs_20261105` folder. They are independent ZIPs, not split volumes.

The app package already includes the calibrated reconstruction images, COLMAP
inputs, selected-frame manifests, reference models, evaluation renders and
training logs/settings. It also includes processing scripts and the locally
modified OpenSplat source, excluding installed environments and credentials.
See `REPROCESSING.md` for scope and limitations.

Before starting the app for the first time, optionally verify all downloaded
files with `python start.py --verify`. Verification can take minutes for raw VRS
files. Run it **before** relocation or annotation edits, which legitimately change
some files. `PACKAGE_INDEX.json` records ZIP sizes and SHA-256 hashes. A file listed
there is not proof its cloud upload has completed; wait for the actual ZIP.

## What is and is not available

| Trial | Aria session | Downward footage on Aria clock | Trained submaps |
|---|---|---|---|
| Fuchs | 19:12.8 | 00:09.5 - 13:50.1 | 8 |
| Fuchs22 | 11:35.2 | 00:12.9 - 07:52.6 | 5 |
| Fuchs333 | 02:46.9 | 00:11.6 - 02:44.0 | 1 |

These are simulated-Parkinson movement recordings, not diagnosed-patient ground
truth. Automatic classification uses the saved `parkinson_high_v1` workflow.
Step/stride and left-right values are head-motion-based estimates requiring
independent video validation. No accuracy or clinical validity is claimed.

At this snapshot there are no saved manual action/foot-contact annotations for
these three trials; the labeling interfaces are available to add them. Eye-gaze
and hand-tracking MPS outputs are not present. Missing downward footage is shown
as unavailable, with black padding preserving the Aria clock. Missing poses stay
missing. The 14 separately trained, overlapping Gaussian submaps represent static
surroundings; model availability is not complete surface reconstruction or a
reconstructed moving body. Blur, holes and foreground remnants remain.

Only these three trials are included in the selector. Original source files on
Nancy's computer were not modified. This snapshot is newer than the November 3
GitHub research branch; mixing those older scripts with this data is not advised.

For ongoing code collaboration, the newer `pd-insighter/develop` branch now
contains the current source and a data-only importer. Follow
[the GitHub quick start](https://github.com/nancyLei123/Qwen2.5-VL-fork/blob/pd-insighter/develop/docs/reproduction/QUICK_START.md) rather than overwriting
new source with this package's snapshot scripts. This does not change the ZIP's
recorded checksum or certify that its cloud upload has finished.
