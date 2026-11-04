# OneDrive Data Locations

Data snapshot: **PD_Insighter_Reproduction_20261101**. These are authenticated
folder links, not public download links. Request access from the folder owner
and sign in with the approved account. GitHub access alone does not grant access.

[Open the snapshot root](https://adminliveunc-my.sharepoint.com/my?id=%2Fpersonal%2Fjlei_ad_unc_edu%2FDocuments%2FPD_Insighter_Reproduction_20261101)
contains the source inventory, SHA-256 manifest, download verifier, and reproduction
guide. Use its **workspace** directory for data; do not mix it with the older ZIPs
also present at the root. The snapshot is not a live mirror of every later edit.

## Select Only What You Need

| Purpose | Folder inside the OneDrive snapshot | Keep locally at |
| --- | --- | --- |
| Dashboard, synchronized videos, labels and saved analysis | [workspace/Qwen2.5-VL-fork/runs](https://adminliveunc-my.sharepoint.com/my?id=%2Fpersonal%2Fjlei_ad_unc_edu%2FDocuments%2FPD_Insighter_Reproduction_20261101%2Fworkspace%2FQwen2.5-VL-fork%2Fruns) | `workspace/Qwen2.5-VL-fork/runs/` |
| Original VRS, metadata and MPS outputs | [workspace/VRS_Pre_Processing/data](https://adminliveunc-my.sharepoint.com/my?id=%2Fpersonal%2Fjlei_ad_unc_edu%2FDocuments%2FPD_Insighter_Reproduction_20261101%2Fworkspace%2FVRS_Pre_Processing%2Fdata) | `workspace/VRS_Pre_Processing/data/` |
| Extracted Aria video/audio and timestamps | [workspace/VRS_Pre_Processing/data/extracted_videos](https://adminliveunc-my.sharepoint.com/my?id=%2Fpersonal%2Fjlei_ad_unc_edu%2FDocuments%2FPD_Insighter_Reproduction_20261101%2Fworkspace%2FVRS_Pre_Processing%2Fdata%2Fextracted_videos) | Same relative path |
| Original downward-camera recordings | [workspace/downward videos](https://adminliveunc-my.sharepoint.com/my?id=%2Fpersonal%2Fjlei_ad_unc_edu%2FDocuments%2FPD_Insighter_Reproduction_20261101%2Fworkspace%2Fdownward%20videos) | `workspace/downward videos/` |
| Historical Aria SDK source | [workspace/projectaria_tools](https://adminliveunc-my.sharepoint.com/my?id=%2Fpersonal%2Fjlei_ad_unc_edu%2FDocuments%2FPD_Insighter_Reproduction_20261101%2Fworkspace%2Fprojectaria_tools) | Optional; install/build the SDK separately |
| Historical VRS CLI source | [workspace/vrs_chunking_tools](https://adminliveunc-my.sharepoint.com/my?id=%2Fpersonal%2Fjlei_ad_unc_edu%2FDocuments%2FPD_Insighter_Reproduction_20261101%2Fworkspace%2Fvrs_chunking_tools) | Optional; build a platform-compatible CLI |

**Viewing:** start with the chosen trial's complete `runs/<trial>/` folder and
shared files directly under `runs/`. Its `downward_master_session/` contains the
aligned dashboard video pair where available. For 3D, keep the full
`environment_reconstruction/` directory and all relative dependencies, not just
`playback.html` or a single PLY. Some older links also need extracted media or
shared plot directories. RoomScan3c uses its own Aria review video and is a room
scan, not a fully labeled movement trial.

**Recomputing:** additionally obtain the raw/MPS and extracted folders below,
the matching downward recording, and saved alignment/annotation files. Refer to
the actual snapshot inventory and files: older trial manifests can contain stale
`exists` flags or extraction paths.

## Trial Folder Map

Raw folders below are relative to `workspace/VRS_Pre_Processing/data/`.
For ordinary movement trials, extracted media is under `extracted_videos/<trial>/`.

| Trial | Saved results and 3D scenes | Raw VRS / MPS location |
| --- | --- | --- |
| Ntrial3 | [runs/Ntrial3](https://adminliveunc-my.sharepoint.com/my?id=%2Fpersonal%2Fjlei_ad_unc_edu%2FDocuments%2FPD_Insighter_Reproduction_20261101%2Fworkspace%2FQwen2.5-VL-fork%2Fruns%2FNtrial3) | `Ntrial3/NTrial_3.vrs`; `Ntrial3/mps_NTrial_3_vrs/` |
| Ntrial10 | [runs/Ntrial10](https://adminliveunc-my.sharepoint.com/my?id=%2Fpersonal%2Fjlei_ad_unc_edu%2FDocuments%2FPD_Insighter_Reproduction_20261101%2Fworkspace%2FQwen2.5-VL-fork%2Fruns%2FNtrial10) | `Ntrial10/Five_mins111.vrs`; `Ntrial10/mps_Five_mins111_vrs/` |
| Parki | [runs/Parki](https://adminliveunc-my.sharepoint.com/my?id=%2Fpersonal%2Fjlei_ad_unc_edu%2FDocuments%2FPD_Insighter_Reproduction_20261101%2Fworkspace%2FQwen2.5-VL-fork%2Fruns%2FParki) | `3Dreconstruction/Parki_.vrs`; `3Dreconstruction/mps_Parki__vrs/` |
| Parki2 | [runs/Parki2](https://adminliveunc-my.sharepoint.com/my?id=%2Fpersonal%2Fjlei_ad_unc_edu%2FDocuments%2FPD_Insighter_Reproduction_20261101%2Fworkspace%2FQwen2.5-VL-fork%2Fruns%2FParki2) | `3Dreconstruction/Parki2.vrs`; `3Dreconstruction/mps_Parki2_vrs/` |
| RoomScan3c | [runs/RoomScan3c](https://adminliveunc-my.sharepoint.com/my?id=%2Fpersonal%2Fjlei_ad_unc_edu%2FDocuments%2FPD_Insighter_Reproduction_20261101%2Fworkspace%2FQwen2.5-VL-fork%2Fruns%2FRoomScan3c) | `3Dreconstruction/3c6ed915-3279-4edf-955b-6523e1e11bb8.vrs`; matching `mps_3c6ed915-3279-4edf-955b-6523e1e11bb8_vrs/` |
| 60minutes_1 | [runs/60minutes_1](https://adminliveunc-my.sharepoint.com/my?id=%2Fpersonal%2Fjlei_ad_unc_edu%2FDocuments%2FPD_Insighter_Reproduction_20261101%2Fworkspace%2FQwen2.5-VL-fork%2Fruns%2F60minutes_1) | `60minutes_1/60_minutes_indoor.vrs`; `60minutes_1/mps_60_minutes_indoor_vrs/` |
| Half_hour | [runs/Half_hour](https://adminliveunc-my.sharepoint.com/my?id=%2Fpersonal%2Fjlei_ad_unc_edu%2FDocuments%2FPD_Insighter_Reproduction_20261101%2Fworkspace%2FQwen2.5-VL-fork%2Fruns%2FHalf_hour) | `Half_hour/Half_hour.vrs`; `Half_hour/mps_Half_hour_vrs/` |

Names are case-sensitive: use `Ntrial3`, not `NTrial3`, for the dashboard directory.
Parki, Parki2, and RoomScan3c share a raw-data parent but are separate recordings.
Other trials are indexed in `SOURCE_INVENTORY.json`; do not infer missing data
from a trial name alone. Not every trial has a reconstructed environment.

## Keep the Correct Scientific Inputs

- Keep each original `.vrs` with its `.vrs.json` and corresponding MPS directory.
- Reconstruction needs `slam/closed_loop_trajectory.csv`,
  `slam/online_calibration.jsonl`, `slam/semidense_points.csv.gz`, and relevant
  observations/summary files. Saved experiment manifests define the selected
  frames, coordinate conventions, train/validation split, and model settings.
- Keep `session_dashboard_manifest.json`, trial/build/synchronization metadata,
  and manual-label stores with the synchronized videos. A source recording and
  a shortened review copy do not necessarily share the same zero timestamp.
- Ntrial10 action labels use `parkinson_motion_review/manual_segments.json`;
  other full-session labels generally use `manual_labels/session_manual_segments.json`.
  Freezing markers are a separate layer, not replacement base-action labels.
- Preserve the snapshot as a baseline. Work on a local copy; do not publish raw
  imagery, voices, trajectories or annotations to GitHub.

Folder references are checked against the saved snapshot inventory. They do not
independently certify every cloud byte or grant new sharing permissions.
