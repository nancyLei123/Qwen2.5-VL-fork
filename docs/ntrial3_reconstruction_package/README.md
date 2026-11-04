# NTrial3 3D Reconstruction Data Package

This is a local, portable snapshot of the available Ntrial3 inputs for rebuilding
the environment, trying another reconstruction method, or comparing against
existing Gaussian Splatting results. The original recording stem is `NTrial_3`.
The full Aria review video is 236.8 seconds long. Originals were not moved or
edited. This package is not a standalone PD Insighter website.

**Research data: keep private.** Recordings may contain people, speech, and
private indoor spaces. Share only through an approved research-data channel.
No data was uploaded when this package was created.

## Start here

- To train without extracting VRS again, use a scene dataset listed in
  `DATA_CATALOG.md`. Each has `images/`, `sparse/0/`, and a frame manifest.
- To change frame selection, resolution, or rectification, start with the raw
  VRS and MPS data. The original folder layout is preserved so the included
  exporter can find them after you move the package.
- To make a fairer RGB comparison, use the separate `quality_comparison_20261022`
  training datasets and their `split.json`, rather than evaluating on training
  images. These remain development comparisons, not clinical validation.
- To inspect previous results, use the reference Gaussian PLYs. A Gaussian PLY
  needs a Gaussian Splatting viewer; it is not a conventional triangle mesh.

## Included data types

| Data | Location within this package | Purpose |
|---|---|---|
| Original recording | `VRS_Pre_Processing/data/Ntrial3/NTrial_3.vrs` | Original sensor records and timestamps, including RGB; use for re-extraction rather than a compressed video. |
| Recording metadata | Same folder, `NTrial_3.vrs.json` | Original recording sidecar; not a replacement for camera calibration. |
| Closed-loop trajectory | `mps_NTrial_3_vrs/slam/closed_loop_trajectory.csv` within the raw folder | Metric device positions and orientations used by the reconstruction pipeline. |
| Open-loop trajectory | Same SLAM folder | Alternative trajectory for diagnostics; do not mix frames/coordinates with the closed-loop model without alignment. |
| Online calibration | Same SLAM folder, `online_calibration.jsonl` | Camera intrinsics, device-to-camera transforms, and camera time offsets. |
| Semi-dense points | Same SLAM folder, `semidense_points.csv.gz` | Measured MPS landmark geometry and quality values; initialization, not a complete surface mesh. |
| Point observations | Same SLAM folder, `semidense_observations.csv.gz` | Original MPS observation records for association/projection work. |
| Health and SLAM summaries | Raw/MPS folders | Quality and processing diagnostics. |
| Calibrated scene inputs | `Qwen2.5-VL-fork/runs/Ntrial3/environment_reconstruction/` | Seven scene datasets containing rectified RGB, COLMAP camera models/poses/points, exact timestamps, transforms, and contact sheets. |
| Aria video and audio | `VRS_Pre_Processing/data/extracted_videos/Ntrial3/` | Full-length extracted review media and extraction/timing metadata. |
| Downward original video | `downward videos/NTrial3.MP4` | Optional motion/context reference, not a calibrated second reconstruction camera. |
| Eye gaze and hand tracking | Raw MPS folder | Optional research context, not required for current static-environment reconstruction. |
| Playback poses and action context | Ntrial3 environment folder and `reference_context/` | Exact frame-time mapping, pose validity, original labeling files, and action intervals. |
| Prior model outputs | Scene training folders and comparison folders | Original seven scene PLYs, final 7k/30k comparison models, and experiment summaries. Intermediate checkpoints are excluded. |
| Source code snapshot | `Qwen2.5-VL-fork/scripts/` | Exporter, training wrapper, and comparison code used by this project. |

`PACKAGE_INDEX.json` provides package-relative paths. `FILE_INVENTORY.json`
records provenance, file sizes, and SHA-256 hashes. Some original manifests and
training logs contain the old computer's absolute paths; these are historical
provenance, not paths to use on another computer. Original dataset README text
describes its export-time status and may predate the included trained models.

## Timing and coordinates

Use the exact capture timestamps in each `dataset_manifest.json` for training.
The exporter subtracts the online camera time offset, interpolates
`T_world_device`, and applies `T_device_RGB`. It rectifies fisheye pixels into
1024 x 1024 pinhole images and rotates both the pixels and camera basis upright.
Do not rotate the images again without also changing the poses.

`T_world_camera` in the training manifest is camera-to-world. COLMAP
`images.txt`/`images.bin` uses the inverse world-to-camera transform, with
quaternions ordered **w, x, y, z**. Camera axes are right, down, forward; the MPS
world frame is gravity-aligned with Z up and distances in meters. Do not
independently normalize the scale or origin of each submap if you want to
assemble them later. Match the `graph_uid` across geometry and poses.

The playback pose arrays are for the browser viewer and use a different camera
axis convention from COLMAP. They contain 7,071 MPS-supported poses, one
explicitly marked short-gap interpolation, and 32 unavailable frames in the
7,104-frame review video. **Do not use that playback-only interpolation or fill
the missing endpoints as measured training poses.** The original MPS trajectory
and prepared training manifests remain the training sources.

The original Aria recording clock, MP4 playback clock, and trimmed dashboard
clock are not identical. `scene_playback_data.json` stores exact per-frame
recording-time mapping. Existing action labels use:

```text
dashboard_seconds = full_Aria_MP4_seconds - 15.133333
```

The dashboard covers 216 seconds; it does not label the entire untrimmed Aria
recording. Audio synchronization suggested approximately 15.12 seconds; the
saved video trim is 15.133333 seconds. Preserve the saved mapping rather than
using rounded times or aligning each stream's start to zero.

## Example use

From the extracted package root, first verify that the transfer is intact:

```bash
python3 verify_package.py
```

To export a new room dataset locally, install Project Aria Tools and the Python
dependencies in your own environment, then run:

```bash
python3 Qwen2.5-VL-fork/scripts/export_aria_gaussian_dataset.py \
  --trial-id Ntrial3 --start-s 0 --end-s 24 \
  --candidate-fps 6 --width 1024 --height 1024 --focal-px 512 \
  --min-translation-m 0.04 --min-rotation-deg 3 --max-frame-gap-s 0.5 \
  --output-dir new_exports/room
```

The installed environment at packaging time used Python 3.11.8,
`projectaria-tools 2.3.0`, `numpy 2.4.6`, `pandas 3.0.6`, `scipy 1.17.1`, and
`Pillow 11.3.0`. Version snapshots are not a guarantee of identical results on
every operating system. An environment or GPU build is not bundled.

To train an existing room dataset with your compatible OpenSplat installation:

```bash
python3 Qwen2.5-VL-fork/scripts/train_aria_gaussian_scene.py \
  --dataset-dir Qwen2.5-VL-fork/runs/Ntrial3/environment_reconstruction/gaussian_submaps/sitting_context \
  --opensplat /absolute/path/to/opensplat \
  --output-dir new_results/room --iterations 7000 --downscale 1 \
  --device auto --validation-image auto \
  --sh-degree 3 --sh-degree-interval 1000 --max-gaussians 1000000
```

The wrapper's generated HTML expects the original PD Insighter vendor assets;
it is not a portable viewer. Use the resulting PLY in a compatible viewer, or
integrate it into your own app. The comparison script is a historical snapshot
with machine-specific library paths and source-hash guards; adapt those paths
before running it elsewhere. The included split datasets themselves are portable.

Tools used in the original pipeline:

- [Project Aria Tools](https://github.com/facebookresearch/projectaria_tools)
  for VRS access and calibration/MPS reading.
- [OpenSplat](https://github.com/WebODM/OpenSplat) for Gaussian optimization.
- [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) for browser viewing.

## Limits and exclusions

The COLMAP model is a calibrated camera/point initializer, not a complete SfM
feature-match database: image keypoint tracks are not populated. A photogrammetry
tool requiring feature matches must compute them or use a supported pose import.
The MPS point cloud is not a dense depth map or ground-truth room mesh.

Prepared RGB frames are selected, rectified JPEGs, not every original frame.
The VRS is included so you can recover other frames or camera streams. Missing
views, motion blur, rolling shutter, thin surfaces, and moving hands/phones
remain limitations. Dynamic-object masks are not included because no validated
full-session masks were found in the selected pipeline inputs. No calibrated
Aria-to-downward extrinsics are provided; do not assume this is a stereo pair.

The VRS contains 7,104 RGB frames at 1408 x 1408 pixels, left/right SLAM camera
streams, eye-camera frames, IMU streams, microphone records, barometer and
magnetometer records. `PACKAGE_INDEX.json` records stream counts and the sampled
RGB decode check. These sensors are preserved inside the VRS; they are not all
exported into separate files. The image decode check samples the beginning,
middle, and end, rather than decoding every original sensor record.

The legacy scene initializers use colors from all selected frames, including
images later held out by the older training wrapper. Do not treat that older
single-view score as a leakage-free benchmark. The comparison folders provide
five held-out views with a 0.25-second temporal guard and training-only point
recoloring. MPS geometry still derives from the full recording.

Software installers, credentials, virtual environments, GPU libraries, unrelated
trials, duplicate synchronized MP4s, temporary checkpoints, and Metashape project
caches are excluded. This package preserves data needed for new reconstructions,
not every prior application project or an exact training runtime. Existing
results and dashboard files on the original computer are unchanged.
