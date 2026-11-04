import * as THREE from 'three';
import {LineSegments2} from '/Qwen2.5-VL-fork/vendor/gaussian-viewer/node_modules/three/examples/jsm/lines/LineSegments2.js';
import {LineSegmentsGeometry} from '/Qwen2.5-VL-fork/vendor/gaussian-viewer/node_modules/three/examples/jsm/lines/LineSegmentsGeometry.js';
import {LineMaterial} from '/Qwen2.5-VL-fork/vendor/gaussian-viewer/node_modules/three/examples/jsm/lines/LineMaterial.js';
import {chooseSceneModels} from './scene_playback_quality.mjs';
import {AnchoredCameraView, ViewTap} from './scene_camera_view.mjs';
import {SpatialOverview} from './scene_spatial_overview.mjs';
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
  const data = await fetchJSON('scene_playback_data.json');
  const hasActionTracks = !data.labels_not_applicable;
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
    $('coverageExplanation').textContent = `First ${scope_s / 60} minutes: ${(100 * model_seconds / scope_s).toFixed(1)}% trained-model time coverage across ${data.scenes.length} submaps. This does not mean complete surfaces or views in every direction. Amber stripes mark untrained intervals; camera-pose gaps remain separate. ${$('coverageExplanation').textContent}`;
    if (scope_s >= 1800) $('routeScope').add(new Option('First 30 minutes', 'first30'));
  }
  $('audioNote').textContent = data.video_has_audio
    ? 'Aligned camera time; image crops can differ. Audio starts muted.'
    : 'Aligned camera time; image crops can differ. This review copy has no audio.';
  $('audioToggle').hidden = !data.video_has_audio;
  $('audioToggle').onclick = () => {
    video.muted = !video.muted;
    $('audioToggle').textContent = video.muted ? 'Sound off' : 'Sound on';
    $('audioToggle').setAttribute('aria-pressed', String(!video.muted));
  };
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
  const eye = new THREE.PerspectiveCamera(90, 1, 0.06, 0.65);
  const frustum = new THREE.CameraHelper(eye);
  const gold = new THREE.Color('#f5ce72'), pale = new THREE.Color('#bddcd0');
  frustum.setColors(gold, gold, gold, pale, pale);
  frustum.material.depthTest = false;
  geometry.add(frustum);
  const marker = new THREE.Mesh(new THREE.SphereGeometry(0.055, 12, 8), new THREE.MeshBasicMaterial({color: '#25d093', depthTest: false}));
  const directionArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 0.9, '#ed9b40', 0.2, 0.1);
  geometry.add(marker, directionArrow);
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
      $('sceneStatus').textContent = `${viewMode === 'follow' ? 'Recorded Aria viewpoint' : viewMode === 'context' ? 'Around the camera / close view' : 'Nearby route / moving camera'}${estimated ? ' (estimated pose)' : ''}`;
    }
  }

  function updateCamera(time) {
    currentPose = poseAt(time);
    marker.visible = directionArrow.visible = frustum.visible = false;
    if (!currentPose) return;
    position.fromArray(currentPose); orientation.fromArray(currentPose, 3);
    forward.set(0, 0, -1).applyQuaternion(orientation);
    up.set(0, 1, 0).applyQuaternion(orientation);
    const modelCamera = currentScene?.camera || {height: 1024, fy: 512};
    const fov = 2 * Math.atan(modelCamera.height / (2 * modelCamera.fy)) * 180 / Math.PI;
    eye.fov = fov; eye.updateProjectionMatrix();
    eye.position.copy(position); eye.quaternion.copy(orientation); eye.updateMatrixWorld(true);
    frustum.update(); marker.position.copy(position);
    directionArrow.position.copy(position); directionArrow.setDirection(forward);
    if (viewMode === 'follow') {
      camera.position.copy(position); camera.quaternion.copy(orientation); camera.up.copy(up);
      viewer.controls.target.copy(position).addScaledVector(forward, 3);
      camera.fov = fov;
      camera.aspect = 1; camera.updateProjectionMatrix(); camera.updateMatrixWorld(true);
      orbitAnchor = null;
    } else {
      if (!orbitAnchor) resetOrbit();
      else {
        orbitCamera.position.fromArray(moveOrbitAnchor(orbitCamera.position.toArray(), orbitAnchor.toArray(), position.toArray()));
        viewer.controls.target.copy(position);
        orbitAnchor.copy(position);
        orbitCamera.up.copy(worldUp);
        viewer.controls.update();
      }
      cameraView.apply();
      marker.visible = directionArrow.visible = frustum.visible = true;
    }
    $('orbitDistanceLabel').textContent = `${camera.position.distanceTo(position).toFixed(2)} m from camera`;
    $('viewAimStatus').hidden = following;
    $('viewAimStatus').textContent = cameraView.locked ? 'Direction held / Reset view to recenter' : 'Drag to orbit / click to look';
  }
  function resetOrbit() {
    if (!currentPose || viewMode === 'follow') return;
    const distance = viewMode === 'context' ? .85 : 2.4;
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(orientation);
    const offset = forward.clone().multiplyScalar(-.85).addScaledVector(right, .28).addScaledVector(worldUp, viewMode === 'context' ? .3 : .75).normalize();
    // This viewer's OrbitControls also reads the original camera's up axis.
    camera.up.copy(worldUp);
    orbitCamera.up.copy(worldUp); orbitCamera.position.copy(position).addScaledVector(offset, distance);
    orbitCamera.fov = 90; orbitCamera.aspect = 1; orbitCamera.updateProjectionMatrix();
    viewer.controls.target.copy(position); orbitAnchor = position.clone();
    viewer.controls.update(); cameraView.reset();
    $('orbitDistance').value = distance;
  }
  function setMode(mode) {
    viewMode = mode; following = mode === 'follow'; viewer.controls.enabled = !following;
    orbitAnchor = null;
    for (const [id, value] of [['follow', 'follow'], ['explore', 'context'], ['cameraContext', 'route']]) {
      $(id).classList.toggle('active', mode === value); $(id).setAttribute('aria-pressed', String(mode === value));
    }
    $('viewHint').textContent = mode === 'follow'
      ? 'Exact recorded viewpoint. Choose Around camera to see the camera point and its rectangular viewing frustum.'
      : mode === 'context' ? 'Click to look in a direction and keep it. Drag to orbit around the green camera point; Reset view recenters it. Playback moves your position without resetting your chosen direction. The gold rectangle is the recorded view; unobserved surfaces may be missing.'
        : 'Click the local line to seek; drag empty space to orbit or click it to look and hold that direction. Reset view recenters the camera point. Teal is before and amber is after the playhead. Playback moves the camera and viewing rectangle together.';
    $('distanceControl').hidden = following;
    updateCamera(video.currentTime);
    routeLine.visible = !following;
    routeLine.material.linewidth = mode === 'context' ? 2 : 4;
    updateVisibility();
  }
  $('follow').onclick = () => setMode('follow');
  $('explore').onclick = () => setMode('context');
  $('recenter').onclick = () => { resetOrbit(); updateCamera(video.currentTime); };
  $('cameraContext').onclick = () => setMode('route');
  $('orbitDistance').oninput = () => {
    if (!currentPose || following) return;
    const offset = orbitCamera.position.clone().sub(position).normalize();
    orbitCamera.position.copy(position).addScaledVector(offset, Number($('orbitDistance').value));
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
      $('poseStatus').textContent = following ? '3D view follows camera position, yaw, pitch and roll.' : 'Green point and gold viewing rectangle = recorded camera. Your orbit follows its position; a clicked viewing direction stays held until you change it or Reset view. Aria video retains the recorded direction.';
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
    $('sceneStage').dataset.frustumVisible = String(frustum.visible);
    $('sceneStage').dataset.localRoute = localKey;
    $('sceneStage').dataset.model = desired?.id || '';
    $('sceneStage').dataset.assembly = assembly;
    $('sceneStage').dataset.loadedModels = viewer.getSceneCount();
  }
  function seekTo(time) {
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
  $('play').onclick = togglePlay; video.onclick = togglePlay;
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
  for (const id of ['routeSeek','routeBack','routeNext']) $(id).disabled = false;
  for (const item of gapButtons) item.button.disabled = false;
  seek.max = Math.max(0, data.duration_s - 1 / data.fps);
  const target = Number(new URLSearchParams(location.search).get('t'));
  const initialTime = Number.isFinite(target) && target > 0 ? recordingToVideo(target, data)
    : (data.initial_video_time_s ?? Math.max(initialIndex / data.fps, -data.dashboard_offset_s + 1));
  seekTo(initialTime);
  setMode('context');
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
