# RoomScan3c: camera-to-environment connection

## Original point frame (all six trials)

The original gold wireframe is retained independently of the experimental cone and its On/Off menu. It appears with the original point representation in RoomScan3c and with the point marker in Ntrial3, 60minutes_1, Fuchs, Fuchs22 and Fuchs333. It follows the recorded RGB pose; it is hidden for the first-person view, unavailable poses, or scenes without valid pinhole intrinsics.

The frame uses each displayed reconstruction's rectified image width, height, focal lengths and principal point. At an axial display depth `d`, its width is `d * image_width / fx` and its height is `d * image_height / fy`. Pixel edges at `-0.5` and `size - 0.5` preserve any off-center principal point. Non-square or asymmetric intrinsics are not forced into a square.

The fixed `d = 0.65 m` preserves the previous visual scale; it is a display choice, not a measured surface distance. Current scene intrinsics specify a centered 90 x 90 degree rectified crop, giving a 1.30 x 1.30 m guide at that depth. The 1024/512 and 1408/704 image/focal-length ratios are identical. Orbiting or zooming the observer changes only its on-screen appearance, not its world size. This rectangle does not claim full raw fisheye coverage, eye gaze, or intersection with objects.

## Use

Open Environment. The **Camera view** menu offers **Surface visibility (estimated)**, **Cone only**, **Landmark probes (diagnostic)**, and **Off**. Play or scrub the timeline. In surface mode, cyan tint follows the recorded camera on the reconstructed appearance, rather than drawing a large floating rectangle. Use **Person in room** to compare the head direction, tinted surroundings and synchronized video. **Around camera** and **Aria viewpoint** also support the tint. Changing the observer does not change the recorded projector pose.

**Surface visibility is the default for RoomScan3c only.** It is experimental model visibility, not independently verified physical visibility. The other five trials retain their original point frames without this surface experiment. Diagnostic landmark hits remain opt-in: the sparse patches do not support a reliable room-wide footprint. A visual check found an implausibly near candidate at video 12:12 even though its local plane fit passed. Local planarity and calibration consistency alone cannot validate an actual visible surface.

- Gold: the central rectified 90-degree region used by the reconstruction, not the whole raw fisheye field and not eye gaze.
- Cyan tint on the reconstruction: approximate visible region of the Gaussian model, subject to depth support. Uncolored does not mean invisible: it can be outside the crop, occluded, or unknown.
- Cyan dots in diagnostic mode: candidate intersections with small fitted MPS landmark patches, shown at the corresponding fisheye-video pixels.
- Coral: the center probe, when a supported patch is found.
- Gray video probes: no supported patch. This does not mean empty space.
- Cone reach: axial display depth in meters, not a measured distance to an object.

## Method

### Estimated surface visibility (new)

The renderer makes two 256 x 256 depth passes from the recorded calibrated RGB pose, using the exact rectified intrinsics. These passes use **all** model splats, not the observer's culled/sorted subset. They retain the nearest Gaussian fragments with opacity at least 0.20 and 0.40, within 0.3-12 m axial depth. Linear depth is RGB24-packed; empty pixels remain invalid.

For a displayed Gaussian fragment, its approximate world position is reconstructed from the observer's depth and projected into that recorded-camera depth map. Cyan tint is allowed only inside the crop when both depth estimates agree (within 0.04 m + 1.5% of depth), the candidate is at that depth (within 0.035 m + 2%), and four neighboring samples have consistent support. Thus a supported foreground surface can reject a background candidate; missing or conflicting depth is not filled. Color is blended without changing the scene's original alpha. The recorded depth passes update with the video frame; moving the observer does not move the recorded camera. Off disables the tint and preserves the original appearance and point frame.

**Important approximation:** this renderer uses projected Gaussian billboards with center depth, not physical triangles. Opacity cutoffs are engineering heuristics, not calibrated probabilities. Changing the observer can change splat footprints and the accepted pixels. Depth disagreement, low-opacity areas and depth edges can produce patchy tint. Missing occluders, floating splats and model errors can still produce false highlights. This does not label objects, prove actual visibility, measure square meters, show eye gaze, or assess attention. The gold boundary on the fisheye video indicates the calibrated crop, not a per-pixel ground-truth visibility mask. No moving people/hands are added to the static depth model.

### Sparse landmark probes (earlier diagnostic)

The preserved Gaussian model is an appearance representation, not a verified mesh. This experiment does not raycast against a supposedly solid room or invent walls across holes.

Quality-filtered MPS landmarks in the same graph/world coordinate frame provide candidate geometry. Local PCA fits use 16 neighbors, a maximum 0.22 m neighborhood, maximum 0.012 m plane RMS, a normal-to-in-plane eigenvalue ratio of 0.08, and support in at least six of eight angular sectors. Small finite discs have radii at most 0.055 m. A conservative robust-plane search was also tried; no global plane met its acceptance threshold in this export. No global plane is rendered. The current result is sparse: 466 strict local patches, reduced to 460 after coincident-center deduplication.

An upright pinhole ray grid (9 x 9) is transformed by the recorded RGB pose. Each ray retains its nearest supported disc intersection; back-of-camera, out-of-range and grazing hits are rejected. Missing geometry is not bridged. The highlight discs are drawn as readable overlays, so they can appear through Gaussian appearance; **occlusion by actual objects is not established**.

The review video is the original fisheye RGB image, displayed with a clockwise 90-degree rotation. The cone uses the rectified camera basis. To link the two, the exporter reverses the viewer/OpenCV and upright-basis transforms, then projects each ray using the exact frame's online RGB calibration. The clockwise pixel rotation is applied last. This avoids treating the fisheye video as a pinhole texture.

## Checks and limits

The audit verifies the playback hash, model hash, graph ID, units, VRS frame count and device timestamps. It compares 947 playback poses against independently re-read calibration/trajectory poses and the selected reconstruction poses. Maximum matrix-element difference: 1.64e-9. All 44,457 supported frames are lens-checked. A compact 78-entry lookup table stays within 0.20 source pixels of the SDK projections at the sampled rays. Unsupported pose/calibration frames show no hits.

These are internal consistency checks, **not independent physical accuracy validation**. The 0.20-pixel value is lookup approximation error, not total reprojection error. MPS/SLAM error, rolling shutter, sparse coverage, missing occluders, moving objects and Gaussian blur remain. The sparse candidates should not be called confirmed object visibility, contact, attention or gaze. No object identities are inferred.

The surface renderer also has a synthetic browser GPU fixture at `tests/fixtures/scene_surface_visibility.html`: a foreground patch and background patch verify foreground highlighting, a visible background region, rejection of background hidden from the recorded camera but visible to the observer, RGB24 depth precision, exact restoration with Off, and rejection of conflicting opacity depths. These tests verify the rendering logic only, not the accuracy of the reconstructed kitchen. The fixture uses synthetic planar fragments; real Gaussian billboards remain approximate. Playback, backward seeking, first-person viewing and Original point retention were also checked in RoomScan3c.

Full video-texture projection is deliberately disabled. A useful next validation would compare known physical surfaces/depths and manually reviewed pixel correspondences over several times, then test first-hit occlusion on a sufficiently complete mesh or depth reconstruction.

## Reproduce locally

From the repository root, using the existing Aria Python environment:

```sh
PYTHONPATH=scripts ../.venvs/aria-reconstruction/bin/python scripts/build_room_view_footprint.py
PYTHONPATH=scripts ../.venvs/aria-reconstruction/bin/python scripts/build_room_scan_route_playback.py --stage publish
```

Derived files are `view_footprint/footprint_data.json` and `view_footprint/alignment_audit.json` beside the playback page. Raw recordings, video exports, pose exports and trained models are not modified. The module is enabled only for RoomScan3c.

References: [Aria online calibration](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/mps_calibration), [Aria MPS tools](https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/core_code_snippets/mps). Use the audit JSON for exact counts and thresholds of the installed export.
