import {routeSegments} from './scene_route_navigation.mjs';

export const supportsPoseFigure = data => data.trial_id === 'RoomScan3c';
export const FIGURE_NOTE = 'Head + arrow: calibrated RGB camera pose, used as a headset/head proxy (not eye gaze). Body: fixed-size illustration, not measured joints, posture, or foot contacts. Video + SLAM uses preliminary, visually reviewed shoe-phase anchors in two short windows; route distance sets display excursion, not anatomical stride length. Foot lift, floor contact and between-anchor motion remain illustrative. The separate Illustrative movement mode uses a chosen 0.32 m visual step cycle, NOT measured cadence or stride length. Do not use either animation to assess gait or PD symptoms. The figure is an overlay and may show through surfaces.';

// Z is world-up. These dimensions are a display template, not anthropometry.
export const BODY_JOINTS = Object.freeze({
  neck: [0, 0, -.17], chest: [0, 0, -.35], pelvis: [0, 0, -.78],
  leftShoulder: [-.21, 0, -.29], rightShoulder: [.21, 0, -.29],
  leftElbow: [-.28, 0, -.59], rightElbow: [.28, 0, -.59],
  leftHand: [-.24, .03, -.87], rightHand: [.24, .03, -.87],
  leftHip: [-.12, 0, -.78], rightHip: [.12, 0, -.78],
  leftKnee: [-.15, .03, -1.12], rightKnee: [.15, .03, -1.12],
  leftAnkle: [-.17, 0, -1.47], rightAnkle: [.17, 0, -1.47],
});
export const BODY_BONES = [
  ['neck', 'chest'], ['chest', 'pelvis'], ['leftShoulder', 'rightShoulder'],
  ['leftShoulder', 'leftElbow'], ['leftElbow', 'leftHand'],
  ['rightShoulder', 'rightElbow'], ['rightElbow', 'rightHand'],
  ['leftHip', 'rightHip'], ['leftHip', 'leftKnee'], ['leftKnee', 'leftAnkle'],
  ['rightHip', 'rightKnee'], ['rightKnee', 'rightAnkle'],
];

export function validFigurePose(pose) {
  return Array.isArray(pose) && pose.length === 7 && pose.every(Number.isFinite)
    && Math.hypot(...pose.slice(3)) > 1e-8;
}

export const VIEW_FRAME_DEPTH_M = .65;

export function recordedViewFrameSpec(intrinsics, depth = VIEW_FRAME_DEPTH_M) {
  if (!intrinsics || (intrinsics.model && intrinsics.model !== 'PINHOLE')) return null;
  const {width, height, fx, fy, cx, cy} = intrinsics;
  if (![width, height, fx, fy, cx, cy, depth].every(Number.isFinite)
      || Math.min(width, height, fx, fy) <= 0 || depth <= .06
      || cx < -.5 || cx > width-.5 || cy < -.5 || cy > height-.5) return null;
  // Pixel centers are integral; image edges are at -0.5 and size-0.5.
  const left = (-.5-cx)/fx, right = (width-.5-cx)/fx;
  const top = (cy+.5)/fy, bottom = (cy+.5-height)/fy;
  return {left, right, top, bottom, depth, width: depth*(right-left), height: depth*(top-bottom),
    horizontalFov: (Math.atan(right)-Math.atan(left))*180/Math.PI,
    verticalFov: (Math.atan(top)-Math.atan(bottom))*180/Math.PI};
}

// This is a camera-field-of-view guide, not a surface or measured object distance.
export class RecordedViewFrame {
  constructor(THREE) {
    this.camera = new THREE.PerspectiveCamera(90, 1, .06, VIEW_FRAME_DEPTH_M);
    this.object = new THREE.CameraHelper(this.camera);
    const gold = new THREE.Color('#f5ce72'), pale = new THREE.Color('#bddcd0');
    this.object.setColors(gold, gold, gold, pale, pale);
    this.object.material.depthTest = this.object.material.depthWrite = false;
    this.object.renderOrder = 20;
    this.object.visible = false;
  }

  update(pose, display, viewMode, intrinsics, ready = true) {
    this.spec = recordedViewFrameSpec(intrinsics);
    this.object.visible = ready && display === 'point' && viewMode !== 'follow'
      && validFigurePose(pose) && Boolean(this.spec);
    if (!this.object.visible) return;
    const s = this.spec, n = this.camera.near;
    const key = JSON.stringify(s);
    if (key !== this.projectionKey) {
      this.camera.fov = s.verticalFov;
      this.camera.aspect = s.width/s.height;
      this.camera.projectionMatrix.makePerspective(n*s.left, n*s.right, n*s.top, n*s.bottom, n, s.depth);
      this.camera.projectionMatrixInverse.copy(this.camera.projectionMatrix).invert();
      this.object.update();
      this.projectionKey = key;
    }
    this.camera.position.fromArray(pose);
    this.camera.quaternion.fromArray(pose, 3).normalize();
    this.camera.updateMatrixWorld(true);
  }
}

export function illustrativeMotionTrack(data) {
  const fps = data.fps, poses = data.poses, track = Array(poses.length).fill(null);
  let start = 0;
  while (start < poses.length) {
    if (!validFigurePose(poses[start])) { start++; continue; }
    let end = start + 1;
    while (end < poses.length && validFigurePose(poses[end])
      && Math.hypot(...poses[end].slice(0, 3).map((v, j) => v - poses[end - 1][j])) <= .25) end++;
    const prefixX = [0], prefixY = [0], smooth = [];
    for (let i = start; i < end; i++) {
      prefixX.push(prefixX.at(-1) + poses[i][0]); prefixY.push(prefixY.at(-1) + poses[i][1]);
    }
    const radius = Math.max(1, Math.round(.2 * fps)), look = Math.max(1, Math.round(.5 * fps));
    for (let i = 0; i < end - start; i++) {
      const a = Math.max(0, i - radius), b = Math.min(end - start, i + radius + 1);
      smooth.push([(prefixX[b] - prefixX[a]) / (b - a), (prefixY[b] - prefixY[a]) / (b - a)]);
    }
    let distance = 0, runStart = 0;
    for (let i = 0; i < smooth.length; i++) {
      const a = Math.max(0, i - look), b = Math.min(smooth.length - 1, i + look);
      const dx = smooth[b][0] - smooth[a][0], dy = smooth[b][1] - smooth[a][1];
      const moving = Math.hypot(dx, dy) >= .12 && Math.hypot(dx, dy) / Math.max((b - a) / fps, .001) >= .12;
      if (!moving) { distance = 0; runStart = i + 1; continue; }
      if (i > runStart) distance += Math.hypot(smooth[i][0] - smooth[i - 1][0], smooth[i][1] - smooth[i - 1][1]);
      track[start + i] = {phase: distance / .32 * Math.PI, heading: Math.atan2(dy, dx),
        weight: Math.min(1, (i - runStart) / (.3 * fps)), measured: false};
    }
    let runEnd = end - 1;
    for (let i = end - 1; i >= start; i--) {
      if (!track[i]) runEnd = i - 1;
      else track[i].weight = Math.min(track[i].weight, (runEnd - i) / (.3 * fps));
    }
    start = end;
  }
  return track;
}

export function stridePreviewAt(preview, time) {
  return preview?.bouts?.find(bout => time >= bout.start_s && time < bout.end_s) || null;
}

export function recentHeadPath(data, time, seconds = 6) {
  // Recompute from timestamps when seeking backwards; never append wall-clock history.
  return routeSegments(data, Math.max(0, time - seconds), Math.min(data.duration_s, time + 1 / data.fps), .1);
}

export function roomObserverTarget(pose) {
  return [pose[0], pose[1], pose[2] - .65];
}

export function roomObserverRig(data, pose, time) {
  // Start from an actually visited position, rather than orbiting through a ceiling/wall.
  // This chooses only the observer's location; the recorded person pose is untouched.
  let best = null;
  for (let i = 0; i < data.poses.length; i += Math.max(1, Math.round(data.fps))) {
    const candidate = data.poses[i];
    if (!validFigurePose(candidate)) continue;
    const distance = Math.hypot(candidate[0] - pose[0], candidate[1] - pose[1]);
    if (distance < 1.25 || distance > 3 || Math.abs(candidate[2] - pose[2]) > .35) continue;
    const score = 2 * Math.abs(distance - 1.8) + .003 * Math.abs(i / data.fps - time);
    if (!best || score < best.score) best = {position: candidate.slice(0, 3), score};
  }
  return {position: best?.position || [pose[0] + 1.6, pose[1] - 1.2, pose[2] + .1], target: roomObserverTarget(pose)};
}

export class PoseFigure {
  constructor(THREE) {
    this.THREE = THREE;
    this.root = new THREE.Group(); this.root.name = 'Schematic wearer / camera pose proxy';
    this.head = new THREE.Group(); this.body = new THREE.Group();
    this.root.add(this.head, this.body); this.root.visible = false;
    const material = color => new THREE.MeshBasicMaterial({color, depthTest: false, depthWrite: false});
    const headMaterial = material('#ff795b'), bodyMaterial = material('#e8dfc9');
    this.head.add(new THREE.Mesh(new THREE.SphereGeometry(.095, 20, 14), headMaterial));
    const nose = new THREE.Mesh(new THREE.SphereGeometry(.028, 10, 8), material('#fff3dd'));
    nose.position.set(0, 0, -.095); this.head.add(nose);
    // Camera local -Z is forward; the head group retains full yaw, pitch and roll.
    this.arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, -1), new THREE.Vector3(0, 0, -.12), .55, '#ffd48a', .11, .075);
    this.arrow.line.material.depthTest = this.arrow.cone.material.depthTest = false;
    this.arrow.line.material.depthWrite = this.arrow.cone.material.depthWrite = false;
    this.head.add(this.arrow);
    // An asymmetric brow makes roll visible even when the direction arrow is unchanged.
    const brow = new THREE.Mesh(new THREE.BoxGeometry(.14, .025, .025), material('#402d26'));
    brow.position.set(0, .055, -.065); this.head.add(brow);
    const sphere = new THREE.SphereGeometry(.032, 10, 8);
    this.joints = {}; this.bones = [];
    for (const [name, point] of Object.entries(BODY_JOINTS)) {
      const joint = new THREE.Mesh(sphere, bodyMaterial); joint.position.fromArray(point); this.body.add(joint);
      this.joints[name] = joint;
    }
    for (const [a, b] of BODY_BONES) {
      const start = new THREE.Vector3(...BODY_JOINTS[a]), end = new THREE.Vector3(...BODY_JOINTS[b]);
      const bone = new THREE.Mesh(new THREE.CylinderGeometry(.018, .018, 1, 8), bodyMaterial);
      bone.scale.y = start.distanceTo(end);
      bone.position.copy(start).add(end).multiplyScalar(.5);
      bone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), end.sub(start).normalize());
      this.body.add(bone);
      this.bones.push({mesh: bone, a, b});
    }
    this.feet = ['left', 'right'].map(side => {
      const foot = new THREE.Mesh(new THREE.BoxGeometry(.085, .19, .045), material('#ffd48a'));
      foot.position.copy(this.joints[`${side}Ankle`].position); foot.position.y += .045;
      this.body.add(foot); return foot;
    });
    this.legsAnimated = false;
    this.trail = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({
      color: '#ffb49a', transparent: true, opacity: .85, depthTest: false, depthWrite: false,
    }));
    this.trail.visible = false; this.trailKey = ''; this.trailSegments = 0;
    this.root.traverse(object => { object.renderOrder = 30; });
    this.trail.renderOrder = 25;
    this.direction = new THREE.Vector3();
  }

  update(pose, visible = true, motion = null) {
    this.root.visible = visible && validFigurePose(pose);
    this.legsAnimated = false;
    if (!this.root.visible) return;
    this.root.position.fromArray(pose);
    this.head.quaternion.fromArray(pose, 3).normalize();
    this.direction.set(0, 0, -1).applyQuaternion(this.head.quaternion);
    const right = new this.THREE.Vector3(1, 0, 0).applyQuaternion(this.head.quaternion);
    const yaw = Math.hypot(this.direction.x, this.direction.y) > 1e-5
      ? Math.atan2(this.direction.y, this.direction.x) - Math.PI / 2
      : Math.atan2(right.y, right.x);
    // The upright body follows head heading only for readability, not body-pose inference.
    this.body.rotation.set(0, 0, yaw);
    this.updateLegs(motion);
    this.root.updateMatrixWorld(true);
  }

  updateLegs(motion) {
    const THREE = this.THREE;
    for (const [name, point] of Object.entries(BODY_JOINTS)) this.joints[name].position.fromArray(point);
    const fused = motion?.kind === 'video_slam';
    const active = motion && Number.isFinite(motion.heading) && (fused
      ? Object.values(motion.feet).some(f => f.weight > 0)
      : Number.isFinite(motion.phase) && motion.weight > 0);
    this.legsAnimated = !!active;
    if (active) {
      const travel = new THREE.Vector3(Math.cos(motion.heading), Math.sin(motion.heading), 0)
        .applyQuaternion(this.body.quaternion.clone().invert());
      for (const [index, side] of ['left', 'right'].entries()) {
        const cue = fused ? motion.feet[side] : null;
        if (fused && !cue) continue;
        const phase = motion.phase + index * Math.PI, weight = Math.min(1, fused ? cue.weight : motion.weight);
        const hip = this.joints[`${side}Hip`].position, knee = this.joints[`${side}Knee`].position;
        const ankle = this.joints[`${side}Ankle`].position;
        const upper = hip.distanceTo(knee), lower = knee.distanceTo(ankle);
        ankle.addScaledVector(travel, (fused ? cue.forward : .13 * Math.sin(phase)) * weight);
        ankle.z += (fused ? cue.lift : .055 * Math.max(0, Math.sin(phase))) * weight;
        // Two-link illustration with bounded reach, not a fitted body or floor model.
        const axis = ankle.clone().sub(hip);
        const reach = Math.min(axis.length(), upper + lower - .001);
        axis.normalize(); ankle.copy(hip).addScaledVector(axis, reach);
        const along = (upper * upper - lower * lower + reach * reach) / (2 * reach);
        const bend = new THREE.Vector3(0, 1, 0).addScaledVector(axis, -axis.y).normalize();
        const solved = hip.clone().addScaledVector(axis, along)
          .addScaledVector(bend, Math.sqrt(Math.max(0, upper * upper - along * along)));
        knee.lerp(solved, weight);
      }
    }
    for (const {mesh, a, b} of this.bones) {
      const start = this.joints[a].position, end = this.joints[b].position;
      mesh.position.copy(start).add(end).multiplyScalar(.5);
      mesh.scale.y = start.distanceTo(end);
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), end.clone().sub(start).normalize());
    }
    ['left', 'right'].forEach((side, index) => {
      this.feet[index].position.copy(this.joints[`${side}Ankle`].position);
      this.feet[index].position.y += .045;
      this.feet[index].material.color.set(fused && motion.feet[side]?.weight > 0
        ? (side === 'left' ? '#48b9dc' : '#e9a34d') : '#c5bca9');
    });
  }

  updateTrail(data, time, visible = true) {
    this.trail.visible = visible && this.root.visible;
    if (!this.trail.visible) return;
    const key = Math.floor(time * data.fps);
    if (key === this.trailKey) return;
    this.trailKey = key;
    const edges = recentHeadPath(data, time);
    this.trailSegments = edges.length;
    this.trail.geometry.dispose();
    this.trail.geometry = new this.THREE.BufferGeometry().setFromPoints(
      edges.flatMap(edge => [new this.THREE.Vector3(...edge.a.position), new this.THREE.Vector3(...edge.b.position)]));
  }
}
