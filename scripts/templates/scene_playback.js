import * as THREE from 'three';
import {LineSegments2} from '/Qwen2.5-VL-fork/vendor/gaussian-viewer/node_modules/three/examples/jsm/lines/LineSegments2.js';
import {LineSegmentsGeometry} from '/Qwen2.5-VL-fork/vendor/gaussian-viewer/node_modules/three/examples/jsm/lines/LineSegmentsGeometry.js';
import {LineMaterial} from '/Qwen2.5-VL-fork/vendor/gaussian-viewer/node_modules/three/examples/jsm/lines/LineMaterial.js';
import {chooseSceneModels} from './scene_playback_quality.mjs';
import {AnchoredCameraView, ViewTap, disableViewerDebugOverlays} from './scene_camera_view.mjs';
import {SpatialOverview} from './scene_spatial_overview.mjs';
import {PoseFigure, RecordedViewFrame, supportsPoseFigure, FIGURE_NOTE, roomObserverRig, illustrativeMotionTrack, stridePreviewAt} from './scene_pose_figure.mjs';
import {CameraFootprint, validateFootprint, sha256Text} from './scene_view_footprint.mjs';
import {GaussianSurfaceVisibility} from './scene_surface_visibility.mjs';
import {compatibleGait, gaitWindowAt, gaitMotionAt, evidenceTimeAt, EvidenceClip} from './scene_gait_fusion.mjs';
import {routeSegments, nearestRouteTime, routeWindow, nearbyRouteWindow, moveOrbitAnchor, playbackSceneModel} from './scene_route_navigation.mjs';
import {clamp, frameAt, recordingAt, sceneAt, labelAt, recordingToVideo, formatTime, formatExactTime, playbackCoverage, PlaybackTransport, SceneQueue, waitForSplatTree} from './scene_playback_core.mjs';

const $ = id => document.getElementById(id);
const video = $('ariaVideo');
const seek = $('seek');
const overlay = $('sceneOverlay');
const showOverlay = (title, message) => {
  overlay.hidden = false; $('overlayTitle').textContent = title; $('overlayText').textContent = message;
};
const fetchJSON = async url => {
  const response = await fetch(url, {cache: 'no-cache'});
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
};

async function main() {
  const playbackResponse = await fetch('scene_playback_data.json', {cache: 'no-cache'});
  if (!playbackResponse.ok) throw new Error(`Playback data: HTTP ${playbackResponse.status}`);
  const playbackText = await playbackResponse.text();
  const data = JSON.parse(playbackText);
  const hasActionTracks = !data.labels_not_applicable;
  const hasPoseFigure = supportsPoseFigure(data);
  let figureDisplay = 'figure';
  let footMotion = 'illustrative', stridePreview = null, gait = null, currentGaitMotion = null;
  let gaitPlaybackEnd = null;
  const evidenceClip = new EvidenceClip($('downwardEvidence'));
  const gaitTicks = [], gaitCursors = [];
  const motionTrack = hasPoseFigure ? illustrativeMotionTrack(data) : [];
  $('figurePanel').hidden = $('roomView').hidden = !hasPoseFigure;
  if (hasPoseFigure) {
    $('figureMethod').textContent = FIGURE_NOTE;
    $('staticSceneNote').textContent = 'The room is static; the head follows the calibrated camera pose. Video-cued legs use preliminary shoe-phase observations in short review windows. Body posture, lift and foot contacts are not measured. Missing evidence is not filled with a walking cycle.';
    try {
      const candidate = await fetchJSON('room_gait_fusion.json');
      if (!compatibleGait(candidate, data)) throw new Error('Gait evidence belongs to another recording');
      gait = candidate;
      footMotion = 'video_slam';
      $('footMotion').querySelector('option[value="video_slam"]').disabled = false;
      $('footMotion').value = footMotion;
      $('gaitPanel').hidden = $('gaitEvidence').hidden = false;
      document.querySelector('.workspace').classList.add('gait-enabled');
      $('downwardEvidence').src = gait.clip.url;
      $('gaitSeek').min = gait.clip.start_s; $('gaitSeek').max = gait.clip.end_s - 1 / data.fps;
      $('gaitSummary').textContent = `${gait.summary.cue_count} visually reviewed shoe cues in ${gait.summary.window_count} short windows (${gait.summary.review_window_seconds.toFixed(1)} s). Not full-session tracking. Timing and foot side need human review.`;
      $('gaitMethod').textContent = `${gait.annotation_method}. ${gait.method} ${gait.display_prior} ${gait.limitations}`;
      $('gaitSyncNote').textContent = `Downward begins about ${gait.sync.recording_offset_s.toFixed(2)} s after Aria. ${gait.sync.strong_windows} audio windows agree (maximum fit residual ${(gait.sync.max_fit_residual_s * 1000).toFixed(1)} ms). This checks clock alignment, NOT foot-event accuracy; video cues were reviewed at 0.2 s intervals.`;
      for (const window of gait.windows) $('gaitExample').add(new Option(`${window.title} / ${formatTime(window.start_s)}-${formatTime(window.end_s)}`, window.id));
      const percent = t => 100 * (t - gait.clip.start_s) / (gait.clip.end_s - gait.clip.start_s);
      for (const [side, id, color] of [['left', 'gaitLeft', '#48b9dc'], ['right', 'gaitRight', '#e9a34d']]) {
        const lane = $(id);
        for (const window of gait.windows) {
          const cycles = window.cycles.filter(c => c.side === side);
          for (const cycle of cycles) {
            const band = document.createElement('span'); band.className = 'band';
            band.style.cssText = `left:${percent(cycle.start_s)}%;width:${percent(cycle.end_s) - percent(cycle.start_s)}%;background:${color}`;
            lane.append(band);
          }
          for (const cue of window.cues.filter(c => c.side === side)) {
            const button = document.createElement('button');
            button.style.cssText = `left:${percent(cue.time_s)}%;background:${color}`;
            button.title = button.ariaLabel = `${side} forward-shoe cue at ${formatExactTime(cue.time_s)}, approximate, not foot contact`;
            button.onclick = () => { transport.setPlaying(false); seekTo(cue.time_s); };
            lane.append(button); gaitTicks.push({button, cue});
          }
        }
        const cursor = document.createElement('span'); cursor.className = 'cursor'; lane.append(cursor); gaitCursors.push(cursor);
      }
    } catch (error) {
      $('footStatus').textContent = 'Video-cued prototype unavailable; illustrative mode is not foot tracking.';
      console.warn('Optional gait evidence:', error.message);
    }
    try {
      const preview = await fetchJSON('room_stride_preview.json');
      if (preview.trial_id !== data.trial_id || preview.graph_uid !== data.provenance.graph_uid
        || preview.duration_s !== data.duration_s || preview.fps !== data.fps) throw new Error('Stride preview belongs to another recording');
      stridePreview = preview;
      $('stridePreviewSummary').textContent = `${preview.summary.candidate_windows} candidate window(s) / ${preview.summary.windows_screened} screened. ${preview.summary.candidate_seconds.toFixed(1)} seconds pass conservative rhythm and direction checks; this is NOT walking-time coverage or validated accuracy.`;
      for (const bout of preview.bouts) $('strideCandidate').add(new Option(`${formatTime(bout.start_s)} - ${formatTime(bout.end_s)} / unvalidated`, String(bout.id)));
      $('strideCandidate').disabled = !preview.bouts.length;
    } catch {
      $('stridePreviewSummary').textContent = 'Stride preview unavailable. Illustrative feet can still show route movement; they are not stride estimates.';
    }
  }
  $('videoClockBadge').textContent = data.video_time_label || 'Original recording';
  const parameters = new URLSearchParams(location.search);
  let jointScene = null, assembly = 'submaps';
  const jointModels = {};
  if (data.trial_id === '60minutes_1') {
    try {
      const experiment = await fetchJSON('joint_first15_experiment/viewer.json');
      if (experiment.status === 'reviewed_experimental' && experiment.graph_uid === data.provenance.graph_uid) {
        jointModels.joint = experiment.scene;
      }
    } catch { /* Originals remain available until the joint experiment is reviewed. */ }
    if (data.joint_long_review_url) {
      try {
        const longer = await fetchJSON(data.joint_long_review_url);
        if (longer.status === 'reviewed_experimental' && longer.graph_uid === data.provenance.graph_uid) {
          jointModels.joint30 = longer.scene;
          $('sceneAssembly').add(new Option('Joint first 15 minutes / 30,000 steps', 'joint30'));
        }
      } catch { /* Unavailable comparisons never hide the preserved baseline. */ }
    }
    assembly = jointModels[parameters.get('assembly')] ? parameters.get('assembly') : 'submaps';
    jointScene = jointModels[assembly] || null;
    $('jointControl').hidden = !Object.keys(jointModels).length;
    $('sceneAssembly').value = assembly;
    $('jointReport').href = jointScene?.details_url || 'joint_first15_experiment/index.html';
  }
  const highQuality = parameters.get('quality') !== 'balanced';
  $('displayQuality').value = highQuality ? 'high' : 'balanced';
  const qualityBase = data.quality_base_url || (data.trial_id === 'Ntrial3' ? 'quality_comparison_20261022' : null);
  let qualityChoices = null;
  try { if (qualityBase) qualityChoices = await fetchJSON(`${qualityBase}/playback_choices.json`); }
  catch { /* No model is promoted before its comparison and visual review finish. */ }
  const requestedVersion = parameters.get('model') || qualityChoices?.default_choice || 'original';
  const qualityVersion = requestedVersion === 'compared' ? 'compared' : 'original';
  data.scenes = chooseSceneModels(data.scenes, qualityChoices, qualityVersion);
  $('modelVersion').value = qualityVersion;
  $('modelVersion').disabled = !Object.keys(qualityChoices?.scenes || {}).length;
  $('modelControl').hidden = !qualityChoices;
  $('qualityLink').hidden = !qualityBase;
  if (qualityBase) $('qualityLink').href = `${qualityBase}/index.html`;
  if (data.quality_page_url) {
    $('qualityLink').hidden = false; $('qualityLink').href = data.quality_page_url;
  }
  if (!qualityBase) $('coverageExplanation').textContent = `Green 3D regions: ${data.scenes.length} trained submaps. Amber stripes: not reconstructed. Choose a scene from the menu to inspect it. Missing model coverage does not mean video is missing. Very short pose gaps have a minimum 4 px marker; hover or open the gap list for exact times.`;
  if (data.coverage_summary) {
    const {scope_s, model_seconds} = data.coverage_summary;
    $('coverageExplanation').textContent = `${(scope_s / 60).toFixed(1)}-minute scope: ${(100 * model_seconds / scope_s).toFixed(1)}% trained-model time coverage across ${data.scenes.length} submaps. This does not mean complete surfaces or views in every direction. Amber stripes mark untrained intervals; camera-pose gaps remain separate. ${$('coverageExplanation').textContent}`;
    if (scope_s >= 1800) $('routeScope').add(new Option('First 30 minutes', 'first30'));
  }
  $('audioNote').textContent = data.video_has_audio
    ? 'Aligned camera time; image crops can differ. Use Audio On / Off to control sound.'
    : 'Aligned camera time; image crops can differ. This review copy has no audio.';
  // The shared page audio controller owns the button and actual media state.
  // Optional display-only compatibility assets never replace the source models.
  try {
    const compatibility = qualityBase ? await fetchJSON(`${qualityBase}/viewer_compat/manifest.json`) : {};
    for (const scene of data.scenes) {
      const replacement = compatibility.replacements?.[scene.ply];
      if (replacement) scene.ply = replacement;
    }
  } catch { /* Older exports remain usable without the compatibility package. */ }
  let labels = {}, labelsLoaded = false;
  if (hasActionTracks) {
    try { labels = await fetchJSON(data.labels_url); labelsLoaded = true; }
    catch { $('labelWarning').style.display = 'block'; }
  } else {
    labelsLoaded = true;
    for (const id of ['automatic', 'reference']) $(id).closest('.timeline-row').style.display = 'none';
    $('legend').style.display = 'none';
    document.querySelector('.coverage-key .gap-label')?.parentElement.remove();
    $('coverage').closest('.timeline-row').querySelector('.timeline-label').textContent = 'Model availability';
    $('coverage').setAttribute('aria-label', 'Model availability; drag to seek');
    $('routeScope').querySelector('option[value="scene"]').textContent = 'Current model';
    $('coverageExplanation').textContent = 'One static joint room model is available along this recording. This is not a claim of complete surfaces or every viewing direction. Camera-pose gaps are marked separately. No action labels were generated for this room scan.';
  }
  const automatic = labels.continuous_model_segments || labels.model_segments || [];
  const reference = labels.manual_timeline_segments || [];
  const coverage = playbackCoverage(data, automatic, reference);
  const trackIds = hasActionTracks ? ['poseCoverage', 'coverage', 'automatic', 'reference'] : ['poseCoverage', 'coverage'];
  const gapGroups = [
    {track: 'poseCoverage', gaps: coverage.poseGaps, kind: 'pose', title: 'Camera pose unavailable', short: 'No pose', detail: 'No supported calibrated position/orientation. The video is still available.'},
    {track: 'poseCoverage', gaps: coverage.poseEstimates, kind: 'estimate', title: 'Estimated camera pose', short: 'Estimated pose', detail: 'Playback-only interpolation between reliable MPS anchors no more than 50 ms apart. Not a directly supported measurement; not used for reconstruction training.'},
    {track: 'coverage', gaps: coverage.modelGaps, kind: 'model', title: 'No trained 3D model', short: 'No 3D model', detail: 'No trained scene covers this interval. The video is still available.'},
    {track: 'automatic', gaps: coverage.automaticGaps, kind: 'label', title: labelsLoaded ? 'No automatic classification' : 'Classifications not loaded', short: labelsLoaded ? 'No classification' : 'Not loaded', detail: labelsLoaded ? 'Outside available classification coverage; not a new action category.' : 'Label-file loading failed. Coverage cannot be verified.'},
    {track: 'reference', gaps: coverage.referenceGaps, kind: 'label', title: labelsLoaded ? 'No reference label' : 'Reference labels not loaded', short: labelsLoaded ? 'No reference label' : 'Not loaded', detail: labelsLoaded ? 'Outside available manual/spoken reference coverage; no label is inferred.' : 'Label-file loading failed. Coverage cannot be verified.'},
  ].filter(group => hasActionTracks || group.kind !== 'label');
  const gapButtons = [];
  const Gaussian = await import('/Qwen2.5-VL-fork/vendor/gaussian-viewer/node_modules/@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js');
  const initialIndex = data.poses.findIndex(pose => pose !== null);
  if (initialIndex < 0) throw new Error('No supported camera poses in this recording.');
  const initial = data.poses[initialIndex];
  const camera = new THREE.PerspectiveCamera(90, 1, 0.05, 300);
  camera.position.fromArray(initial); camera.up.set(0, 0, 1);
  const geometry = new THREE.Scene();
  const viewer = new Gaussian.Viewer({
    rootElement: $('sceneRoot'), camera,
    initialCameraLookAt: [initial[0], initial[1] + 3, initial[2]],
    selfDrivenMode: false, sharedMemoryForWorkers: false, gpuAcceleratedSort: false,
    integerBasedSort: false, ignoreDevicePixelRatio: !highQuality, enableOptionalEffects: true,
    // This installed browser renderer supports SH2, even when a PLY contains SH3.
    sphericalHarmonicsDegree: 2, sceneRevealMode: Gaussian.SceneRevealMode.Instant,
    renderMode: Gaussian.RenderMode.Always, logLevel: Gaussian.LogLevel.None,
  });
  viewer.controls.enableDamping = false;
  if (viewer.controls.stopListenToKeyEvents) viewer.controls.stopListenToKeyEvents();
  disableViewerDebugOverlays(viewer, window);
  const marker = new THREE.Mesh(new THREE.SphereGeometry(0.055, 12, 8), new THREE.MeshBasicMaterial({color: '#25d093', depthTest: false}));
  geometry.add(marker);
  const recordedViewFrame = new RecordedViewFrame(THREE);
  geometry.add(recordedViewFrame.object);
  const figure = hasPoseFigure ? new PoseFigure(THREE) : null;
  if (figure) geometry.add(figure.root, figure.trail);
  const viewFootprint = hasPoseFigure ? new CameraFootprint(THREE, data, geometry, {
    mode:$('viewConnectionMode'),reach:$('viewReach'),reachLabel:$('viewReachLabel'),
    status:$('viewConnectionStatus'),audit:$('viewAlignmentAudit'),overlay:$('viewVideoOverlay'),stage:$('sceneStage'),
  }) : null;
  const surfaceVisibility = hasPoseFigure ? new GaussianSurfaceVisibility(THREE, viewer, data, {
    stage:$('sceneStage'),status:$('viewConnectionStatus'),
  }) : null;
  if (viewFootprint) {
    $('viewConnection').hidden = $('viewVideoKey').hidden = false;
    (async () => {
      try {
        const payload = await fetchJSON('view_footprint/footprint_data.json');
        const validated = validateFootprint(payload, data, await sha256Text(playbackText));
        viewFootprint.install(validated);
        surfaceVisibility.install(validated);
      } catch (error) {
        surfaceVisibility.fail(error);
        $('viewConnectionStatus').textContent = `Camera footprint unavailable: ${error.message} Playback remains available; no intersections are inferred.`;
      }
    })();
  }
  const position = new THREE.Vector3(), orientation = new THREE.Quaternion();
  const forward = new THREE.Vector3(), up = new THREE.Vector3();
  const poseAt = time => data.poses[frameAt(time, data)];
  let following = false, viewMode = 'context', orbitAnchor = null;
  const worldUp = new THREE.Vector3(0, 0, 1);
  viewer.controls.enablePan = false;
  viewer.controls.minDistance = .35; viewer.controls.maxDistance = 6;
  // Only explicit distance controls zoom; scrolling the page cannot fly away.
  viewer.controls.enableZoom = false;
  const cameraView = new AnchoredCameraView(camera, viewer.controls);
  const orbitCamera = cameraView.orbitCamera;
  // The stock focal animation changes controls.target, which conflicts with
  // the wearer-centered rig. View taps below change the viewing offset only.
  viewer.onMouseClick = () => {};
  let loadedStatus = 'loading', currentScene = null, requestedScene = undefined;
  let currentPose = null, frameTime = null, renderedOnce = false;
  const transport = new PlaybackTransport(video, updatePlayText, error => {
    $('poseStatus').textContent = `Playback did not start: ${error.message}`;
  });

  function updatePlayText() {
    $('play').textContent = transport.wantsPlay ? (transport.buffering ? 'Pause (loading)' : 'Pause') : 'Play';
    $('play').setAttribute('aria-label', transport.wantsPlay ? 'Pause playback' : 'Play synchronized recording');
    $('figurePlay').textContent = $('play').textContent;
    $('figurePlay').setAttribute('aria-label', transport.wantsPlay ? 'Pause person playback' : 'Play person playback');
  }
  const queue = new SceneQueue(async scene => {
    const index = viewer.getSceneCount();
    await viewer.addSplatScene(scene.ply, {
      showLoadingUI: false, progressiveLoad: false, splatAlphaRemovalThreshold: 1,
      onProgress: percent => {
        if (currentScene?.id === scene.id) $('sceneStatus').textContent = `Loading ${scene.title} ${Math.round(percent)}%`;
      },
    });
    await waitForSplatTree(viewer.splatMesh);
    return index;
  }, index => {
    for (let i = 0; i < viewer.getSceneCount(); i++) viewer.getSplatScene(i).visible = i === index;
    viewer.forceRenderNextFrame();
  }, (status, scene, error) => {
    loadedStatus = status;
    if (status === 'loading') {
      $('sceneStatus').textContent = `Loading ${scene.title}`;
    }
    $('retry').hidden = status !== 'error';
    if (status === 'error') {
      transport.setPlaying(false);
      showOverlay('Scene could not load', error?.message || 'Please retry.');
    }
    updateVisibility();
    transport.setBuffering(status === 'loading');
  });

  function updateVisibility() {
    if (loadedStatus === 'error') return;
    if (!currentPose) {
      $('sceneStatus').textContent = 'No supported camera pose';
      const frameTime = frameAt(video.currentTime, data) / data.fps;
      const gap = coverage.poseGaps.find(item => frameTime >= item.start_s && frameTime < item.end_s);
      showOverlay('Camera pose unavailable', gap
        ? `Video ${formatExactTime(gap.start_s)} to ${formatExactTime(gap.end_s)} has no supported position/orientation. See the red Camera pose marker. Video playback remains available.`
        : 'No supported position/orientation at this frame. See the red Camera pose marker; video playback remains available.');
    } else if (!currentScene) {
      $('sceneStatus').textContent = 'No trained model at this time';
      showOverlay('No reconstructed scene in this interval', 'The amber-striped 3D coverage region marks this untrained interval. Video and supported camera poses remain available; labels are shown where available.');
    } else if (loadedStatus === 'loading') {
      showOverlay('Loading this part of the environment', 'Playback will resume when this scene is ready. You can seek elsewhere or press Pause.');
    } else if (loadedStatus === 'ready') {
      overlay.hidden = true;
      const estimated = data.pose_status?.[frameAt(video.currentTime, data)] === 'interpolated_short_gap';
      $('sceneStatus').textContent = `${viewMode === 'follow' ? 'Recorded Aria viewpoint' : viewMode === 'room' ? 'Person in room / fixed observer' : viewMode === 'context' ? 'Around the camera / close view' : 'Nearby route / moving camera'}${estimated ? ' (estimated pose)' : ''}`;
    }
  }

  function updateCamera(time) {
    currentPose = poseAt(time);
    marker.visible = false;
    const modelCamera = currentScene?.camera || {height: 1024, fy: 512};
    const fov = 2 * Math.atan(modelCamera.height / (2 * modelCamera.fy)) * 180 / Math.PI;
    if (figure) {
      currentGaitMotion = gaitMotionAt(gait, time, currentPose);
      figure.body.visible = figureDisplay === 'figure';
      figure.update(currentPose, viewMode !== 'follow' && figureDisplay !== 'point',
        figureDisplay !== 'figure' ? null : footMotion === 'video_slam' ? currentGaitMotion
          : footMotion === 'illustrative' ? motionTrack[frameAt(time, data)] : null);
      figure.updateTrail(data, time);
    }
    if (!currentPose) return;
    position.fromArray(currentPose); orientation.fromArray(currentPose, 3);
    forward.set(0, 0, -1).applyQuaternion(orientation);
    up.set(0, 1, 0).applyQuaternion(orientation);
    marker.position.copy(position);
    if (viewMode === 'follow') {
      camera.position.copy(position); camera.quaternion.copy(orientation); camera.up.copy(up);
      viewer.controls.target.copy(position).addScaledVector(forward, 3);
      camera.fov = fov;
      camera.aspect = 1; camera.updateProjectionMatrix(); camera.updateMatrixWorld(true);
      orbitAnchor = null;
    } else {
      if (!orbitAnchor) resetOrbit();
      else if (viewMode !== 'room') {
        orbitCamera.position.fromArray(moveOrbitAnchor(orbitCamera.position.toArray(), orbitAnchor.toArray(), position.toArray()));
        viewer.controls.target.copy(position);
        orbitAnchor.copy(position);
        orbitCamera.up.copy(worldUp);
        viewer.controls.update();
      }
      else viewer.controls.update();
      cameraView.apply();
      marker.visible = !figure || figureDisplay === 'point';
    }
    const distanceTarget = viewMode === 'room' ? viewer.controls.target : position;
    $('orbitDistanceLabel').textContent = `${camera.position.distanceTo(distanceTarget).toFixed(2)} m ${viewMode === 'room' ? 'from pivot' : 'from camera'}`;
    $('orbitDistance').setAttribute('aria-label', viewMode === 'room' ? 'Distance from fixed room pivot' : 'Distance from recorded camera');
    $('viewAimStatus').hidden = following;
    $('viewAimStatus').textContent = cameraView.locked ? 'Direction held / Reset view to recenter' : 'Drag to orbit / click to look';
  }

  function updateRecordedViewFrame(time) {
    const model = playbackSceneModel(currentScene, jointScene, assembly, time);
    const display = figure ? figureDisplay : 'point';
    recordedViewFrame.update(currentPose, display, viewMode, model?.camera,
      loadedStatus === 'ready' && Boolean(currentScene) && !video.seeking && video.readyState >= 2);
    $('viewFrameNote').hidden = display !== 'point' || viewMode === 'follow';
    const s = recordedViewFrame.spec;
    $('viewFrameNote').textContent = s
      ? `Gold viewing frame: ${s.horizontalFov.toFixed(1)} x ${s.verticalFov.toFixed(1)} degrees, from this model's rectified camera intrinsics. At ${s.depth.toFixed(2)} m display depth its size is ${s.width.toFixed(2)} x ${s.height.toFixed(2)} m. Display guide only, not an object distance, full fisheye view, or eye gaze. Independent of the experimental cone controls.`
      : 'Viewing frame unavailable: this scene has no verified rectified camera intrinsics. No default field of view is inferred.';
  }
  function resetOrbit() {
    if (!currentPose || viewMode === 'follow') return;
    let distance = viewMode === 'context' ? .85 : 2.4;
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(orientation);
    const offset = forward.clone().multiplyScalar(-.85).addScaledVector(right, viewMode === 'room' ? .75 : .28).addScaledVector(worldUp, viewMode === 'context' ? .3 : .75).normalize();
    const rig = viewMode === 'room' ? roomObserverRig(data, currentPose, video.currentTime) : null;
    const pivot = rig ? new THREE.Vector3(...rig.target) : position;
    // This viewer's OrbitControls also reads the original camera's up axis.
    camera.up.copy(worldUp);
    orbitCamera.up.copy(worldUp); orbitCamera.position.copy(pivot).addScaledVector(offset, distance);
    if (rig) { orbitCamera.position.fromArray(rig.position); distance = orbitCamera.position.distanceTo(pivot); }
    orbitCamera.fov = 90; orbitCamera.aspect = 1; orbitCamera.updateProjectionMatrix();
    viewer.controls.target.copy(pivot); orbitAnchor = position.clone();
    viewer.controls.update(); cameraView.reset();
    $('orbitDistance').value = distance;
  }
  function setMode(mode) {
    viewMode = mode; following = mode === 'follow'; viewer.controls.enabled = !following;
    orbitAnchor = null;
    for (const [id, value] of [['follow', 'follow'], ['explore', 'context'], ['cameraContext', 'route'], ['roomView', 'room']]) {
      $(id).classList.toggle('active', mode === value); $(id).setAttribute('aria-pressed', String(mode === value));
    }
    $('viewHint').textContent = mode === 'follow'
      ? 'Exact recorded viewpoint. Choose Around camera to see the recorded camera position as a green point.'
      : mode === 'context' ? 'The green point marks the recorded camera position. Drag to orbit around it; click to look and keep that direction. Reset view recenters it. The point follows playback; unobserved surfaces may be missing.'
        : 'The green point marks the current camera position. Click the local line to seek; drag empty space to orbit or click to look and hold that direction. Teal is before and amber is after the playhead. The point follows playback.';
    if (hasPoseFigure) $('viewHint').textContent = mode === 'room'
      ? 'Fixed observer: the person moves through the room while this view stays still. Drag to orbit; Reset view recenters on the current person. The body guide is not measured posture.'
      : mode === 'follow' ? 'Exact recorded camera viewpoint. The figure is hidden to avoid blocking the first-person view; choose Person in room to see its movement.'
        : 'Orbit follows the recorded head position. Head direction remains independent of your viewing angle. Choose Person in room for a fixed view of back-and-forth movement.';
    $('distanceControl').hidden = following;
    updateCamera(video.currentTime);
    routeLine.visible = !following && mode !== 'room';
    routeLine.material.linewidth = mode === 'context' ? 2 : 4;
    updateVisibility();
    updateRecordedViewFrame(video.currentTime);
    viewFootprint?.update(frameAt(video.currentTime, data),currentPose,viewMode,loadedStatus === 'ready' && Boolean(currentScene),
      !video.seeking && video.readyState >= 2);
  }
  $('follow').onclick = () => setMode('follow');
  $('explore').onclick = () => setMode('context');
  $('recenter').onclick = () => { resetOrbit(); updateCamera(video.currentTime); };
  $('cameraContext').onclick = () => setMode('route');
  $('roomView').onclick = () => setMode('room');
  $('figureDisplay').onchange = () => { figureDisplay = $('figureDisplay').value; sync(video.currentTime); };
  $('footMotion').onchange = () => { footMotion = $('footMotion').value; sync(video.currentTime); };
  function chooseGaitExample(play = false) {
    const window = gait?.windows.find(w => w.id === $('gaitExample').value);
    if (!window) return;
    transport.setPlaying(false); footMotion = 'video_slam'; $('footMotion').value = footMotion;
    figureDisplay = 'figure'; $('figureDisplay').value = figureDisplay;
    seekTo(window.start_s + .05); setMode('room');
    gaitPlaybackEnd = play ? window.end_s : null;
    if (play) transport.setPlaying(true);
  }
  $('gaitExample').onchange = () => chooseGaitExample();
  $('gaitReplay').onclick = () => chooseGaitExample(true);
  $('strideCandidate').onchange = () => {
    const bout = stridePreview?.bouts.find(item => String(item.id) === $('strideCandidate').value);
    if (!bout) return;
    transport.setPlaying(false); seekTo(bout.events[1]?.time_s || bout.start_s);
    setMode('room');
  };
  $('orbitDistance').oninput = () => {
    if (!currentPose || following) return;
    const pivot = viewMode === 'room' ? viewer.controls.target : position;
    const offset = orbitCamera.position.clone().sub(pivot).normalize();
    orbitCamera.position.copy(pivot).addScaledVector(offset, Number($('orbitDistance').value));
    viewer.controls.update(); updateCamera(video.currentTime);
  };
  $('displayQuality').onchange = () => {
    const url = new URL(location.href);
    url.searchParams.set('quality', $('displayQuality').value);
    url.searchParams.set('t', recordingAt(video.currentTime, data).toFixed(6));
    transport.setPlaying(false);
    location.assign(url.href);
  };
  $('modelVersion').onchange = () => {
    const url = new URL(location.href);
    url.searchParams.set('model', $('modelVersion').value);
    url.searchParams.set('t', recordingAt(video.currentTime, data).toFixed(6));
    transport.setPlaying(false);
    location.assign(url.href);
  };
  $('retry').onclick = () => queue.retry();
  $('sceneAssembly').onchange = () => {
    assembly = $('sceneAssembly').value;
    jointScene = jointModels[assembly] || null;
    $('jointReport').href = jointScene?.details_url || 'joint_first15_experiment/index.html';
    requestedScene = undefined;
    sync(video.currentTime);
    const url = new URL(location.href);
    url.searchParams.set('assembly', assembly);
    url.searchParams.set('t', recordingAt(video.currentTime, data).toFixed(6));
    history.replaceState(null, '', url);
  };

  let bounds = data.bounds, scale, center;
  const mapPoint = p => [220 + (p[0] - center[0]) * scale, 112 - (p[1] - center[1]) * scale];
  let routeEdges = [], localEdges = [], localKey = '', routeKey = '', routeRange = [0, data.duration_s], lastRouteColorTime = -1;
  const routeLine = new LineSegments2(new LineSegmentsGeometry(), new LineMaterial({vertexColors:true, linewidth:4, depthTest:false, depthWrite:false, transparent:true, opacity:.95}));
  routeLine.visible = false; routeLine.renderOrder = 10; geometry.add(routeLine);
  $('routeScope').value = data.initial_route_scope || 'all';
  const spatialOverview = new SpatialOverview(data, time => { if(!seek.disabled) routeSeekTo(time); });
  spatialOverview.init();
  function refreshLocalRoute(time) {
    const range = nearbyRouteWindow(data.duration_s, time), key = range.join(':');
    if (localKey === key) return;
    localKey = key; localEdges = routeSegments(data, ...range);
    const points = localEdges.flatMap(edge => [edge.a.position, edge.b.position]);
    routeLine.geometry.dispose(); routeLine.geometry = new LineSegmentsGeometry();
    if (points.length) {
      routeLine.geometry.setPositions(new Float32Array(points.flat()));
      routeLine.geometry.setColors(new Float32Array(points.length * 3));
    }
    lastRouteColorTime = -1;
    $('localRouteLabel').textContent = `3D route: ${formatTime(range[0])} - ${formatTime(range[1])}`;
  }
  function refreshRoute(time) {
    const range = routeWindow(data, currentScene, $('routeScope').value, time), key = range.join(':');
    if (routeKey === key) return;
    routeKey = key; routeRange = range; routeEdges = routeSegments(data, ...range);
    spatialOverview.setRange(range);
    const points = routeEdges.flatMap(edge => [edge.a.position, edge.b.position]);
    if (points.length) bounds = points.reduce((limits,p) => {
      for (let j=0;j<3;j++) { limits[0][j]=Math.min(limits[0][j],p[j]); limits[1][j]=Math.max(limits[1][j],p[j]); }
      return limits;
    }, [[Infinity,Infinity,Infinity],[-Infinity,-Infinity,-Infinity]]);
    scale = Math.min(380 / Math.max(.5, bounds[1][0] - bounds[0][0]), 190 / Math.max(.5, bounds[1][1] - bounds[0][1]));
    center = bounds[0].map((value, i) => (value + bounds[1][i]) / 2);
    $('routePath').setAttribute('d', routeEdges.map(edge => `M${mapPoint(edge.a.position).join(',')}L${mapPoint(edge.b.position).join(',')}`).join(' '));
    $('xAxis').textContent = `World X: ${bounds[0][0].toFixed(1)} to ${bounds[1][0].toFixed(1)} m`;
    $('yAxis').textContent = `World Y: ${bounds[0][1].toFixed(1)} to ${bounds[1][1].toFixed(1)} m`;
    $('routeSeek').min = range[0]; $('routeSeek').max = Math.max(range[0],range[1]-1/data.fps);
    $('routeWindow').textContent = `${formatTime(range[0])} - ${formatTime(range[1])}`;
  }
  $('routeScope').onchange = () => refreshRoute(video.currentTime);
  function routeSeekTo(time) { transport.setPlaying(false); seekTo(time); }
  $('routeSeek').oninput = () => routeSeekTo(Number($('routeSeek').value));
  $('routeBack').onclick = () => routeSeekTo(clamp(video.currentTime - 1,...routeRange));
  $('routeNext').onclick = () => routeSeekTo(clamp(video.currentTime + 1,routeRange[0],routeRange[1]-1/data.fps));
  function projectedRoute(element, isMap) {
    const box = element.getBoundingClientRect();
    const project = p => {
      if (isMap) {
        const v=mapPoint(p), matrix=element.getScreenCTM();
        return [v[0]*matrix.a+v[1]*matrix.c+matrix.e-box.left,v[0]*matrix.b+v[1]*matrix.d+matrix.f-box.top];
      }
      const v=new THREE.Vector3(...p).project(camera);
      if (v.z < -1 || v.z > 1) return [NaN,NaN];
      return [(v.x+1)*box.width/2,(1-v.y)*box.height/2];
    };
    return (isMap ? routeEdges : localEdges).map(edge=>({a:project(edge.a.position),b:project(edge.b.position),start:edge.a.time,end:edge.b.time}));
  }
  function pathHit(event, element, isMap) {
    const box=element.getBoundingClientRect();
    return nearestRouteTime(projectedRoute(element,isMap),[event.clientX-box.left,event.clientY-box.top],video.currentTime);
  }
  function bindRoutePointer(element, isMap=false) {
    let drag=null;
    element.addEventListener('pointerdown', event=>{
      if(event.button!==0 || seek.disabled) return;
      const hit = (isMap || viewMode==='route') ? pathHit(event,element,isMap) : null;
      if(hit===null) return;
      event.preventDefault();event.stopImmediatePropagation();transport.setPlaying(false);
      viewer.controls.enabled=false;
      // Keep the projection stable during a route drag even though the camera
      // anchor follows the new time. Otherwise the path moves under the pointer.
      drag={id:event.pointerId,segments:projectedRoute(element,isMap)};
      element.setPointerCapture(event.pointerId);
      if(hit!==null) routeSeekTo(hit);
    },true);
    element.addEventListener('pointermove',event=>{
      if(!drag || drag.id!==event.pointerId) return;
      event.preventDefault();event.stopImmediatePropagation();
      const box = element.getBoundingClientRect();
      const hit = nearestRouteTime(drag.segments,[event.clientX-box.left,event.clientY-box.top],video.currentTime);
      if(hit!==null) routeSeekTo(hit);
    },true);
    const finish=event=>{if(drag?.id!==event.pointerId)return;event.stopImmediatePropagation();drag=null;viewer.controls.enabled=viewMode!=='follow';if(element.hasPointerCapture(event.pointerId))element.releasePointerCapture(event.pointerId);};
    element.addEventListener('pointerup',finish,true);element.addEventListener('pointercancel',finish,true);element.addEventListener('lostpointercapture',finish,true);
  }
  bindRoutePointer($('sceneRoot'));bindRoutePointer($('routeMap'),true);
  const viewTap = new ViewTap(), sceneRoot = $('sceneRoot');
  sceneRoot.addEventListener('pointerdown', event => viewTap.down(event), true);
  sceneRoot.addEventListener('pointermove', event => viewTap.move(event), true);
  sceneRoot.addEventListener('pointercancel', () => viewTap.cancel(), true);
  sceneRoot.addEventListener('lostpointercapture', () => viewTap.cancel(), true);
  sceneRoot.addEventListener('pointerup', event => {
    if (!viewTap.up(event) || following || !currentPose || loadedStatus !== 'ready') return;
    const box = sceneRoot.getBoundingClientRect();
    const x = (event.clientX - box.left) / box.width, y = (event.clientY - box.top) / box.height;
    if (!cameraView.lookAtScreen(2 * x - 1, 1 - 2 * y)) return;
    const cue = $('viewFocusMarker');
    cue.style.left = `${100 * x}%`; cue.style.top = `${100 * y}%`;
    for (const animation of cue.getAnimations()) animation.cancel();
    cue.animate([{opacity: 1, transform: 'translate(-50%, -50%) scale(.7)'},
      {opacity: 0, transform: 'translate(-50%, -50%) scale(1.3)'}], {duration: 550});
    viewer.forceRenderNextFrame();
  }, true);
  refreshRoute(0);

  const cursors = [];
  function fillTrack(id, segments, offset = 0, coverage = false) {
    const track = $(id);
    for (const seg of segments) {
      const start = Math.max(0, seg.start_s - offset), end = Math.min(data.duration_s, seg.end_s - offset);
      if (end <= start) continue;
      const element = document.createElement('div'); element.className = 'segment';
      element.style.left = `${100 * start / data.duration_s}%`; element.style.width = `${100 * (end - start) / data.duration_s}%`;
      if (!coverage) element.style.background = seg.color || '#8a948f';
      else if (seg.iterations < 7000) element.classList.add('legacy');
      element.title = `${seg.label}: video ${formatTime(start)} - ${formatTime(end)}`;
      if (coverage && id !== 'poseCoverage' && (end - start) / data.duration_s > 0.05) {
        const text = document.createElement('span'); text.textContent = seg.label; element.append(text);
      }
      track.append(element);
    }
    const cursor = document.createElement('div'); cursor.className = 'cursor'; track.append(cursor); cursors.push(cursor);
    track.setAttribute('aria-valuemin', '0'); track.setAttribute('aria-valuemax', data.duration_s);
    return track;
  }
  fillTrack('poseCoverage', coverage.poseSegments.map(seg => ({...seg, label: 'Supported camera pose', color: '#d5e7e1'})), 0, true);
  fillTrack('coverage', coverage.modelSegments, 0, true);
  fillTrack('automatic', coverage.automaticSegments);
  fillTrack('reference', coverage.referenceSegments);
  for (const group of gapGroups) {
    const section = document.createElement('section'); section.className = 'gap-group';
    const heading = document.createElement('h3'); heading.textContent = `${group.title} (${group.gaps.length})`;
    section.append(heading);
    if (!group.gaps.length) { const text = document.createElement('p'); text.textContent = 'No gaps.'; section.append(text); }
    for (const gap of group.gaps) {
      const range = `${formatExactTime(gap.start_s)} - ${formatExactTime(gap.end_s)}`;
      const duration = `${(gap.end_s - gap.start_s).toFixed(3)} s`;
      const title = `${group.title}: ${range} (${duration}), player video time. ${group.detail}`;
      const marker = document.createElement('div'); marker.className = `segment gap gap-${group.kind}`;
      marker.style.left = `${100 * gap.start_s / data.duration_s}%`;
      marker.style.width = `${100 * (gap.end_s - gap.start_s) / data.duration_s}%`;
      // Align end-of-recording markers inward without altering their exact interval.
      if (gap.end_s >= data.duration_s) { marker.style.left = 'auto'; marker.style.right = '0'; }
      marker.title = title; marker.dataset.start = gap.start_s; marker.dataset.end = gap.end_s;
      const text = document.createElement('span'); text.textContent = group.short; marker.append(text);
      $(group.track).append(marker);
      const button = document.createElement('button'); button.className = `gap-${group.kind}`;
      button.disabled = true; button.textContent = range;
      button.setAttribute('aria-label', `Inspect ${title}`);
      const caption = document.createElement('small'); caption.textContent = `${duration} / ${group.short}`; button.append(caption);
      button.title = group.detail;
      button.onclick = () => {
        transport.setPlaying(false); seekTo((gap.start_s + gap.end_s) / 2);
      };
      gapButtons.push({button, gap, kind: group.kind}); section.append(button);
    }
    $('gapList').append(section);
  }
  const missingCount = gapButtons.filter(item => item.kind !== 'estimate').length;
  $('gapSummary').textContent = `Coverage gaps and exact times (${missingCount} missing track intervals; ${coverage.poseEstimates.length} estimated) - click to inspect`;
  const legend = new Map([...automatic, ...reference].map(seg => [seg.label, seg.color]));
  for (const [label, color] of legend) {
    const item = document.createElement('span'), swatch = document.createElement('i'); swatch.className = 'swatch'; swatch.style.background = color;
    item.append(swatch, document.createTextNode(label)); $('legend').append(item);
  }
  for (let i = 0; i <= 4; i++) { const tick = document.createElement('span'); tick.textContent = formatTime(data.duration_s * i / 4); $('ticks').append(tick); }
  for (const scene of data.scenes) {
    const option = document.createElement('option'); option.value = scene.id;
    option.textContent = `${scene.title} / ${formatTime(scene.video_start_s)}-${formatTime(scene.video_end_s)} video`;
    $('sceneSelect').append(option);
  }
  $('sceneSelect').onchange = event => {
    const scene = data.scenes.find(item => item.id === event.target.value);
    if (scene) {
      const target = recordingToVideo(scene.start_s + 1, data);
      if ($('routeScope').value !== 'nearby' && (target < routeRange[0] || target >= routeRange[1])) $('routeScope').value='scene';
      seekTo(target);
    }
  };

  function sync(time) {
    if (gaitPlaybackEnd !== null && time >= gaitPlaybackEnd) {
      transport.setPlaying(false); gaitPlaybackEnd = null;
    }
    const capture = recordingAt(time, data);
    currentScene = sceneAt(capture, data.scenes);
    refreshRoute(time); refreshLocalRoute(time); updateCamera(time);
    spatialOverview.update(time, currentPose);
    $('routeSeek').value = clamp(time,...routeRange);
    $('routeClock').textContent = formatTime(time);
    const insideRoute = time >= routeRange[0] && time < routeRange[1];
    $('routeOutside').hidden = insideRoute;
    if(Math.abs(time-lastRouteColorTime)>.1 && localEdges.length) {
      const colors=routeLine.geometry.attributes;
      localEdges.forEach((edge,i)=>{const color=edge.b.time<=time ? [0.09,.65,.47] : [.93,.61,.25]; colors.instanceColorStart.setXYZ(i,...color);colors.instanceColorEnd.setXYZ(i,...color);});
      colors.instanceColorStart.data.needsUpdate=true;lastRouteColorTime=time;
    }
    const model = playbackSceneModel(currentScene, jointScene, assembly, time);
    const desired = currentPose ? model : null;
    if (requestedScene !== (desired?.id ?? null)) {
      requestedScene = desired?.id ?? null;
      queue.request(desired);
    }
    updateVisibility();
    updateRecordedViewFrame(time);
    viewFootprint?.update(frameAt(time, data),currentPose,viewMode,loadedStatus === 'ready' && Boolean(currentScene),
      !video.seeking && video.readyState >= 2);
    $('sceneAssemblyNote').textContent = model === jointScene && jointScene
      ? `Joint experiment (${jointScene.iterations.toLocaleString()} steps): one static scene trained from all six first-15-minute groups. No model switch at their boundaries. It is not a complete room scan; blur, ghosting and unobserved surfaces remain. Compare with the originals.`
      : assembly.startsWith('joint')
        ? 'Outside the joint first-15-minute scene. Original submaps are used only where available; missing coverage is not filled in.'
        : 'Original submaps share the MPS coordinate system but were trained separately. Only the current submap is displayed; they are not fused. Time coverage is not complete surface coverage.';
    if (model !== jointScene && currentScene?.review_note) $('sceneAssemblyNote').textContent += ` Scene review: ${currentScene.review_note}`;
    if (data.scene_assembly_note) $('sceneAssemblyNote').textContent = data.scene_assembly_note;
    const action = labelAt(time, automatic, data.dashboard_offset_s, data.dashboard_duration_s);
    const manual = labelAt(time, reference, data.dashboard_offset_s, data.dashboard_duration_s);
    $('currentAction').textContent = action ? `Automatic classification: ${action.label}` : 'No automatic classification at this time';
    $('actionBadge').textContent = $('currentAction').textContent;
    $('referenceAction').textContent = manual ? `Reference label: ${manual.label}` : 'Reference label: not available';
    if (!hasActionTracks) {
      $('currentAction').textContent = 'Dedicated environment scan';
      $('actionBadge').textContent = 'Synchronized with the measured camera route';
      $('referenceAction').textContent = 'No action classification or manual labels for this scan.';
    }
    $('sceneName').textContent = currentScene?.title || 'Outside model coverage';
    $('sceneSelect').value = currentScene?.id || '';
    $('sceneDetails').hidden = !currentScene;
    if (model) $('sceneDetails').href = model.details_url;
    const dashboard = time + data.dashboard_offset_s;
    $('clock').textContent = `${formatTime(time)} / ${formatTime(data.duration_s)}`;
    if (hasPoseFigure) {
      $('figureClock').textContent = formatTime(time);
      if (!transport.scrubbing) $('figureTime').value = time;
      const projected = currentPose ? position.clone().project(camera) : null;
      $('figureStatus').textContent = !currentPose ? 'No supported pose: head and body are hidden, not extrapolated.'
        : following ? 'Figure hidden in Aria viewpoint. Video and timeline remain synchronized.'
          : figureDisplay === 'point' ? 'Original point: recorded camera position. The camera-view cone follows its measured orientation; candidate patch hits are experimental, not eye gaze.'
          : viewMode === 'room' && (Math.abs(projected.x) > 1 || Math.abs(projected.y) > 1 || Math.abs(projected.z) > 1)
            ? 'Person is outside this fixed view. Reset view to recenter, or use Around camera.'
            : 'Position and head direction follow the video clock. Body dimensions and posture are illustrative.';
      $('footStatus').textContent = footMotion === 'off' ? 'Static foot guide. No foot tracking.'
        : !figure.root.visible || figureDisplay !== 'figure' ? 'Legs are hidden in this view.'
          : footMotion === 'video_slam' ? (currentGaitMotion
            ? `Video-cued phase: ${Object.keys(currentGaitMotion.feet).join(' + ')}. Route-scaled movement; lift and joints are illustrative, not measured contacts.`
            : 'No supported foot phase here. Feet stay neutral, not a claim of standing. Choose a reviewed gait example below.')
            : figure.legsAnimated ? 'Illustrative feet follow route movement. Timing, left/right identity and foot lift are NOT measured.'
              : 'Illustrative feet held still: no sustained route translation at this moment. This does not establish whether you took a step.';
      if (gait) {
        const evidence = evidenceTimeAt(time, data, gait);
        evidenceClip.update(evidence?.clip_s ?? null, !video.paused && !video.seeking && !transport.scrubbing, video.playbackRate);
        const missingVideo = !evidence || $('downwardEvidence').readyState < 2 || $('downwardEvidence').error;
        $('downwardEvidence').hidden = !!missingVideo;
        $('gaitMissing').hidden = !missingVideo;
        $('gaitMissing').textContent = !evidence ? 'Outside the 17-second review clip. Choose a gait example.'
          : $('downwardEvidence').error ? 'Evidence clip could not load. Foot timing cannot be visually checked here.' : 'Loading synchronized downward video...';
        $('gaitEvidenceClock').textContent = evidence ? `Downward ${formatTime(evidence.downward_s)}` : 'No review clip at this time';
        const window = gaitWindowAt(gait, time);
        const cue = gait.windows.flatMap(w => w.cues).find(c => Math.abs(c.time_s - time) <= gait.timing_review_half_window_s);
        $('gaitCueReadout').textContent = cue ? `Near ${cue.side} forward-shoe cue / approximate, not foot contact`
          : window ? 'Between reviewed shoe cues / interpolated foot phase'
            : evidence ? 'Turn or visibility gap / no gait-phase estimate' : 'No foot evidence reviewed at this time.';
        if (evidenceClip.error) $('gaitEvidenceStatus').textContent = `Secondary playback issue: ${evidenceClip.error}`;
        if (!transport.scrubbing) $('gaitSeek').value = clamp(time, gait.clip.start_s, gait.clip.end_s);
        for (const cursor of gaitCursors) { cursor.hidden = !evidence; cursor.style.left = `${100 * (time - gait.clip.start_s) / (gait.clip.end_s - gait.clip.start_s)}%`; }
        for (const item of gaitTicks) item.button.setAttribute('aria-current', String(item.cue === cue));
      }
      const bout = stridePreviewAt(stridePreview, time);
      $('strideReadout').textContent = bout
        ? `Candidate window medians: step proxy ${bout.step_proxy_m.toFixed(3)} m / stride proxy ${bout.stride_proxy_m.toFixed(3)} m. Foot side unknown; video validation pending.`
        : 'No stride estimate for this time. Review a candidate above; do not infer zero steps from missing estimates.';
    }
    $('timeDetails').textContent = `VRS capture ${formatTime(capture)} / Dashboard ${dashboard >= 0 && dashboard < data.dashboard_duration_s ? formatTime(dashboard) : 'outside review interval'}`;
    if (!transport.scrubbing) seek.value = time;
    for (const cursor of cursors) cursor.style.left = `${100 * time / data.duration_s}%`;
    for (const id of trackIds) {
      $(id).setAttribute('aria-valuenow', time.toFixed(3)); $(id).setAttribute('aria-valuetext', formatTime(time));
    }
    const missing = [];
    if (!currentPose) missing.push('camera pose');
    if (!currentScene) missing.push('trained 3D model');
    if (hasActionTracks && !labelsLoaded) missing.push('label tracks could not be loaded');
    else if (hasActionTracks) {
      if (!action) missing.push('automatic classification');
      if (!manual) missing.push('reference label');
    }
    const estimated = data.pose_status?.[frameAt(time, data)] === 'interpolated_short_gap';
    let availabilityText = missing.length ? `Unavailable at this time: ${missing.join('; ')}. Video playback is still available.`
      : hasActionTracks ? 'At this time: camera pose, trained 3D model and both label tracks are available.'
      : 'At this time: calibrated camera pose and the joint room model are available. No action labels are expected for this scan.';
    if (estimated) availabilityText += ' Camera pose is estimated by short-gap interpolation, not directly MPS-supported.';
    if ($('availability').textContent !== availabilityText) {
      $('availability').textContent = availabilityText;
      $('availability').classList.toggle('has-gap', missing.length > 0 || estimated);
    }
    for (const item of gapButtons) {
      const gapTime = item.kind === 'label' ? time : frameAt(time, data) / data.fps;
      item.button.setAttribute('aria-current', String(gapTime >= item.gap.start_s && gapTime < item.gap.end_s));
    }
    $('routeMarker').style.display = $('routeDirection').style.display = currentPose && insideRoute ? '' : 'none';
    if (currentPose) {
      const [x, y] = mapPoint(currentPose);
      $('routeMarker').setAttribute('cx', x); $('routeMarker').setAttribute('cy', y);
      $('routeDirection').setAttribute('d', `M${x},${y}L${x + forward.x * 24},${y - forward.y * 24}`);
      $('worldPosition').textContent = `RGB camera XYZ: ${currentPose.slice(0, 3).map(n => n.toFixed(2)).join(', ')} m`;
      $('poseStatus').textContent = following ? '3D view follows camera position, yaw, pitch and roll.' : 'The green point in the upper 3D scene marks the recorded camera position. Your orbit follows it; a clicked viewing direction stays held until you change it or Reset view. Aria video retains the recorded direction.';
      if (hasPoseFigure && !following) $('poseStatus').textContent = 'The head is anchored to the calibrated RGB camera position and rotation. The upright body guide follows head heading for readability only, not measured body orientation or eye gaze.';
      if (estimated) $('poseStatus').textContent += ' Estimated pose: short-gap interpolation between reliable MPS anchors; not used for training.';
    } else {
      $('worldPosition').textContent = 'Camera position unavailable';
      $('poseStatus').textContent = 'No sufficiently supported pose/calibration at this exposure. No pose is extrapolated.';
    }
    // Lightweight observable state for accessibility, diagnostics and browser verification.
    $('sceneStage').dataset.frame = frameAt(time, data);
    $('sceneStage').dataset.scene = currentScene?.id || '';
    $('sceneStage').dataset.viewMode = viewMode;
    $('sceneStage').dataset.routeSegments = routeEdges.length;
    $('sceneStage').dataset.pose = currentPose ? JSON.stringify(currentPose) : '';
    $('sceneStage').dataset.poseStatus = data.pose_status?.[frameAt(time, data)] || (currentPose ? 'mps_supported' : 'unavailable');
    $('sceneStage').dataset.camera = JSON.stringify([...camera.position.toArray(), ...camera.quaternion.toArray()]);
    $('sceneStage').dataset.orbitTarget = JSON.stringify(viewer.controls.target.toArray());
    $('sceneStage').dataset.lookOffset = JSON.stringify(cameraView.lookOffset.toArray());
    $('sceneStage').dataset.directionHeld = String(!following && cameraView.locked);
    $('sceneStage').dataset.frustumVisible = String(recordedViewFrame.object.visible);
    $('sceneStage').dataset.frustumPose = recordedViewFrame.object.visible
      ? JSON.stringify([...recordedViewFrame.camera.position.toArray(), ...recordedViewFrame.camera.quaternion.toArray()]) : '';
    $('sceneStage').dataset.frustumSpec = recordedViewFrame.spec ? JSON.stringify(recordedViewFrame.spec) : '';
    $('sceneStage').dataset.footprintVisible = String(viewFootprint?.root.visible || false);
    $('sceneStage').dataset.footprintConePose = viewFootprint?.root.visible
      ? JSON.stringify([...viewFootprint.cone.position.toArray(), ...viewFootprint.cone.quaternion.toArray()]) : '';
    $('sceneStage').dataset.footprintMode = viewFootprint?.mode || 'unavailable';
    $('sceneStage').dataset.markerVisible = String(marker.visible);
    $('sceneStage').dataset.markerPosition = currentPose ? JSON.stringify(marker.position.toArray()) : '';
    $('sceneStage').dataset.meshCursorVisible = String(viewer.sceneHelper.meshCursor.visible);
    $('sceneStage').dataset.localRoute = localKey;
    $('sceneStage').dataset.model = desired?.id || '';
    $('sceneStage').dataset.assembly = assembly;
    $('sceneStage').dataset.loadedModels = viewer.getSceneCount();
    if (figure) {
      $('sceneStage').dataset.figureVisible = String(figure.root.visible);
      $('sceneStage').dataset.figurePosition = figure.root.visible ? JSON.stringify(figure.root.position.toArray()) : '';
      $('sceneStage').dataset.headQuaternion = figure.root.visible ? JSON.stringify(figure.head.quaternion.toArray()) : '';
      $('sceneStage').dataset.headForward = figure.root.visible ? JSON.stringify(figure.direction.toArray()) : '';
      $('sceneStage').dataset.bodyMeasured = 'false';
      $('sceneStage').dataset.figureDisplay = figureDisplay;
      $('sceneStage').dataset.trailSegments = String(figure.trailSegments);
      $('sceneStage').dataset.legsAnimated = String(figure.legsAnimated && figure.body.visible);
      $('sceneStage').dataset.feetMeasured = 'false';
      $('sceneStage').dataset.footMode = footMotion;
      $('sceneStage').dataset.gaitWindow = currentGaitMotion?.window_id || '';
      $('sceneStage').dataset.footPositions = figure.root.visible ? JSON.stringify(figure.feet.map(foot => foot.getWorldPosition(new THREE.Vector3()).toArray())) : '';
    }
  }
  function seekTo(time) {
    gaitPlaybackEnd = null;
    const value = clamp(time, 0, Math.max(0, data.duration_s - 1 / data.fps));
    transport.pauseForSync();
    video.currentTime = value; frameTime = null; sync(value); transport.resume();
  }
  function togglePlay() {
    if (!transport.wantsPlay && video.ended) seekTo(0);
    transport.setPlaying(!transport.wantsPlay);
  }
  let scrubPointer = null;
  function beginScrub(event) {
    if (seek.disabled || event.button !== 0 || event.isPrimary === false) return false;
    scrubPointer = event.pointerId;
    gaitPlaybackEnd = null;
    transport.beginScrub(); frameTime = null;
    return true;
  }
  function endScrub(event, commit = true) {
    if (event.pointerId !== scrubPointer) return;
    scrubPointer = null;
    transport.endScrub(commit);
  }
  seek.onpointerdown = beginScrub;
  seek.oninput = () => seekTo(Number(seek.value));
  seek.onchange = () => {
    if (!transport.scrubbing) transport.setPlaying(true);
  };
  $('figureTime').onpointerdown = beginScrub;
  $('figureTime').oninput = () => seekTo(Number($('figureTime').value));
  $('figureTime').onchange = seek.onchange;
  $('gaitSeek').onpointerdown = beginScrub;
  $('gaitSeek').oninput = () => seekTo(Number($('gaitSeek').value));
  $('gaitSeek').onchange = seek.onchange;
  window.addEventListener('pointerup', event => endScrub(event));
  window.addEventListener('pointercancel', event => endScrub(event, false));
  window.addEventListener('blur', () => { scrubPointer = null; transport.endScrub(false); });
  for (const id of trackIds) {
    const track = $(id);
    const dragTo = event => {
      const box = track.getBoundingClientRect();
      seekTo(clamp((event.clientX - box.left) / box.width, 0, 1) * data.duration_s);
    };
    track.onpointerdown = event => {
      if (!beginScrub(event)) return;
      event.preventDefault(); track.focus({preventScroll: true});
      track.setPointerCapture(event.pointerId); dragTo(event);
    };
    track.onpointermove = event => { if (track.hasPointerCapture(event.pointerId)) dragTo(event); };
    track.onpointerup = event => {
      if (track.hasPointerCapture(event.pointerId)) dragTo(event);
      endScrub(event);
      if (track.hasPointerCapture(event.pointerId)) track.releasePointerCapture(event.pointerId);
    };
    track.onlostpointercapture = event => endScrub(event, false);
    track.onkeydown = event => {
      if (event.code === 'ArrowLeft' || event.code === 'ArrowRight') {
        event.preventDefault(); seekTo(video.currentTime + (event.code === 'ArrowRight' ? 1 : -1));
        transport.setPlaying(true);
      }
    };
  }
  $('play').onclick = $('figurePlay').onclick = togglePlay; video.onclick = togglePlay;
  $('downwardEvidence').onclick = togglePlay;
  for (const [id, delta] of [['back', -1], ['forward', 1]]) $(id).onclick = () => {
    transport.setPlaying(false); seekTo((frameAt(video.currentTime, data) + delta) / data.fps);
  };
  $('speed').onchange = () => { video.playbackRate = Number($('speed').value); };
  document.addEventListener('keydown', event => {
    if (event.code !== 'Space' || event.repeat || event.altKey || event.metaKey || event.ctrlKey) return;
    if (event.target.closest('input:not([type="range"]), select, textarea, [contenteditable="true"]')) return;
    event.preventDefault(); togglePlay();
  });
  video.addEventListener('ended', () => transport.setPlaying(false));
  video.addEventListener('seeked', () => { frameTime = null; sync(video.currentTime); transport.resume(); });
  video.addEventListener('timeupdate', () => { if (video.paused) sync(video.currentTime); });
  video.addEventListener('error', () => {
    transport.setPlaying(false); $('poseStatus').textContent = `Video failed to load: ${video.error?.message || 'media error'}`;
  });
  const metadataReady = new Promise((resolve, reject) => {
    video.addEventListener('loadedmetadata', resolve, {once: true});
    video.addEventListener('error', () => reject(new Error('Aria video failed to load.')), {once: true});
  });
  video.src = data.video_url;
  await metadataReady;
  if (Math.abs(video.duration - data.duration_s) > 1 / data.fps + 0.001) throw new Error('Video duration does not match the frame timestamp table.');
  $('timingMethod').textContent = data.provenance.timing_explanation || `Each video frame is mapped to its original VRS exposure, then to the calibrated RGB pose. The last-frame clock correction is ${data.provenance.video_minus_device_at_last_frame_s.toFixed(3)} s. Dashboard labels retain their existing ${data.dashboard_offset_s.toFixed(6)} s offset from video time. All models use the same MPS world coordinate frame.`;
  for (const id of ['play', 'back', 'forward', 'seek']) $(id).disabled = false;
  if (hasPoseFigure) for (const id of ['figurePlay', 'figureTime']) $(id).disabled = false;
  for (const id of ['routeSeek','routeBack','routeNext']) $(id).disabled = false;
  for (const item of gapButtons) item.button.disabled = false;
  seek.max = Math.max(0, data.duration_s - 1 / data.fps);
  $('figureTime').max = seek.max;
  const target = Number(new URLSearchParams(location.search).get('t'));
  const requestedGait = gait?.windows.find(w => w.id === parameters.get('gait'));
  if (requestedGait) $('gaitExample').value = requestedGait.id;
  const initialTime = Number.isFinite(target) && target > 0 ? recordingToVideo(target, data)
    : requestedGait ? requestedGait.start_s + 1
    : (data.initial_video_time_s ?? Math.max(initialIndex / data.fps, -data.dashboard_offset_s + 1));
  seekTo(initialTime);
  setMode(hasPoseFigure ? 'room' : 'context');
  if (video.requestVideoFrameCallback) {
    const onFrame = (_, metadata) => { frameTime = metadata.mediaTime; video.requestVideoFrameCallback(onFrame); };
    video.requestVideoFrameCallback(onFrame);
  }
  let lastPaint = 0;
  function animate(now) {
    requestAnimationFrame(animate);
    if (now - lastPaint < 30) return;
    lastPaint = now;
    const time = !video.paused && !video.seeking && frameTime !== null ? frameTime : video.currentTime;
    sync(time);
    // OrbitControls.update() recomputes orientation even when disabled. Bypass it in follow mode.
    const controls = viewer.controls;
    if (viewMode === 'follow') viewer.controls = null;
    viewer.update();
    viewer.controls = controls;
    if (viewMode !== 'follow') cameraView.apply();
    surfaceVisibility?.update(frameAt(time, data),currentPose,viewFootprint.mode,
      loadedStatus === 'ready' && Boolean(currentScene) && overlay.hidden && viewer.splatRenderReady,
      !video.seeking && video.readyState >= 2);
    if (!overlay.hidden || !viewer.splatRenderReady) return;
    viewer.render(); renderedOnce = true;
    if (viewMode !== 'follow' && currentPose) {
      const autoClear = viewer.renderer.autoClear;
      viewer.renderer.autoClear = false;
      viewer.renderer.clearDepth();
      viewer.renderer.render(geometry, camera);
      viewer.renderer.autoClear = autoClear;
    }
    $('sceneStage').dataset.rendered = String(renderedOnce);
  }
  requestAnimationFrame(animate);
  window.addEventListener('pagehide', () => transport.setPlaying(false));
}

main().catch(error => {
  console.error(error);
  $('sceneStatus').textContent = 'Playback initialization failed';
  showOverlay('Could not prepare synchronized playback', error.message);
});
