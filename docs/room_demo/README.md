# PD Insighter: RoomScan3c Mini Demo

A self-contained **result-viewing demo**, not a complete reconstruction-training dataset.
No account, npm install, CUDA, Aria SDK, or OpenSplat installation is needed to view it.
Use Python 3.9+ and a desktop browser with WebGL enabled (Chrome or Edge suggested).
The browser and a local Python server are required; do not double-click index.html.

## Open the Demo

1. Extract the entire ZIP into a local folder, not a cloud-only placeholder folder.
2. Open a terminal in that folder.
3. Run `python3 start_demo.py` (Windows: `py -3 start_demo.py`).
4. The browser opens at http://127.0.0.1:8795/. Keep the terminal running.

If that port is occupied: `python3 start_demo.py --port 8796`.
Before first use, run `python3 start_demo.py --verify` to verify every packaged file.
No additional packages or internet connection are required for the local viewer.
External references in the methodology page are optional.

## A Quick Tour

- **Around camera:** orbit around the recorded camera position. Drag to look around;
  click to hold a viewing direction; Reset view restores it. The gold rectangle
  shows the recorded facing direction, not a reconstructed person's body.
- **Aria viewpoint:** follow the recorded camera position and orientation.
- **Nearby route:** see nearby positions rather than a distant whole-session view.
- **Play / Pause and timeline:** synchronize the video, camera pose and route.
  The demo starts near the middle of the excerpt. Drag anywhere to seek.
- **Recorded route & viewing direction:** explore the room's MPS landmarks,
  recorded route and camera frustum. Click the route to seek.
- **Display: Balanced:** use this if high-resolution rendering is slow.

## What Is Included

- The unchanged, jointly trained room Gaussian model (30,000 OpenSplat iterations).
- A 90-second, 704 x 704 H.264 review video, retaining every 30-fps source frame
  in the selected excerpt: original MP4 time **06:30 to 08:00**.
- The matching 2,700 camera poses, original RGB capture-relative timestamps and
  pose-support flags. The demo timeline starts at 00:00; original recording times
  are retained in the data/readout. Unsupported poses remain missing.
- The room's measured MPS landmarks, calibration/model metadata and a held-out
  comparison image. All viewer libraries and their licenses are bundled locally.
- The read-only PD Insighter playback source and an integrity manifest.

The **model was trained using the broader room scan**, not just these 90 seconds.
Only the video and playback interval are shortened; the model is not downsampled
or retrained. Video compression does not improve reconstruction quality.

This dedicated room scan has **no action-classification or manual-label tracks**.
No labels are invented. Room surfaces may still be blurry, distorted or missing,
especially for reflective, occluded or unobserved areas. Model availability along
the timeline is not a room-surface completeness score or clinical validation.

## If You Want to Retrain

Request the full RoomScan3c VRS, closed-loop MPS trajectory, online calibration,
semi-dense point cloud, selected calibrated RGB images, masks, train/validation
split and reconstruction scripts from the larger institutional project package.
Those large training inputs are deliberately not included in this mini demo.

The original recordings, existing models and full project have not been modified.
This contains real research imagery: share only with authorized collaborators,
not a public repository. Third-party library licenses do not grant rights to the
recordings or models. See `METHODS.html` for method, provenance and limitations.
