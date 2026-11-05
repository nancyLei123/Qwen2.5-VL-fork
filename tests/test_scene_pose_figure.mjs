import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as THREE from '../vendor/gaussian-viewer/node_modules/three/build/three.module.js';
import {PoseFigure, RecordedViewFrame, recordedViewFrameSpec, supportsPoseFigure, BODY_JOINTS, recentHeadPath, roomObserverTarget, roomObserverRig, FIGURE_NOTE, illustrativeMotionTrack, stridePreviewAt} from '../scripts/templates/scene_pose_figure.mjs';

const near = (a, b) => assert.ok(Math.hypot(...a.map((v, i) => v - b[i])) < 1e-8);
const pose = (x = 0) => [x, 2, 1.6, ...new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0)).toArray()];
const intrinsics = (fov=90) => ({model:'PINHOLE',width:1024,height:1024,
  fx:512/Math.tan(fov*Math.PI/360),fy:512/Math.tan(fov*Math.PI/360),cx:511.5,cy:511.5});

test('Original point retains a wireframe viewing square driven by recorded pose', () => {
  const frame = new RecordedViewFrame(THREE);
  assert.equal(frame.object.isLineSegments, true);
  assert.equal(frame.object.children.length, 0);
  for (const mode of ['room', 'context', 'route']) {
    for (const x of [0, 1, -1, 0]) {
      const p = pose(x), before = [...p];
      frame.update(p, 'point', mode, intrinsics(78));
      assert.equal(frame.object.visible, true);
      near(frame.camera.position.toArray(), p.slice(0, 3));
      near(frame.camera.quaternion.toArray(), p.slice(3));
      assert.ok(Math.abs(frame.camera.fov-78) < 1e-8);
      assert.equal(frame.camera.aspect, 1);
      assert.deepEqual(p, before);
      // The far-plane corners still form a square in the camera's local frame.
      const a = new THREE.Vector3().fromBufferAttribute(frame.object.geometry.attributes.position, frame.object.pointMap.f1[0]);
      const b = new THREE.Vector3().fromBufferAttribute(frame.object.geometry.attributes.position, frame.object.pointMap.f2[0]);
      const c = new THREE.Vector3().fromBufferAttribute(frame.object.geometry.attributes.position, frame.object.pointMap.f3[0]);
      assert.ok(Math.abs(a.distanceTo(b) - a.distanceTo(c)) < 1e-8);
    }
  }
});

test('viewing square hides only in other representations, first person, or pose gaps', () => {
  const frame = new RecordedViewFrame(THREE);
  for (const [p, display, mode] of [[pose(), 'figure', 'room'], [pose(), 'head', 'context'],
    [pose(), 'point', 'follow'], [null, 'point', 'room']]) {
    frame.update(pose(), 'point', 'room', intrinsics());
    frame.update(p, display, mode, intrinsics());
    assert.equal(frame.object.visible, false);
  }
  frame.update(pose(), 'point', 'room', intrinsics());
  assert.equal(frame.object.visible, true);
});

test('frame dimensions use both focal lengths and principal point, not a forced square', () => {
  const k={model:'PINHOLE',width:1600,height:900,fx:1000,fy:800,cx:650,cy:410};
  const frame=new RecordedViewFrame(THREE);
  frame.update([0,0,0,0,0,0,1], 'point', 'context', k);
  assert.ok(Math.abs(frame.spec.width-.65*1600/1000)<1e-10);
  assert.ok(Math.abs(frame.spec.height-.65*900/800)<1e-10);
  const corners={f1:[-.5,899.5],f2:[1599.5,899.5],f3:[-.5,-.5],f4:[1599.5,-.5]};
  for(const [key,[u,v]] of Object.entries(corners)) {
    const p=new THREE.Vector3().fromBufferAttribute(frame.object.geometry.attributes.position,frame.object.pointMap[key][0]);
    assert.ok(Math.abs(p.z+.65)<1e-6);
    assert.ok(Math.abs(k.fx*p.x/-p.z+k.cx-u)<1e-3);
    assert.ok(Math.abs(k.cy-k.fy*p.y/-p.z-v)<1e-3);
  }
  const s=recordedViewFrameSpec(k,1.3);
  assert.equal(s.width,frame.spec.width*2);
  assert.equal(s.horizontalFov,frame.spec.horizontalFov);
});

test('invalid or missing intrinsics and unloaded scenes do not invent a viewing frame', () => {
  const frame=new RecordedViewFrame(THREE);
  for(const k of [null, {}, {...intrinsics(),fx:0}, {...intrinsics(),cy:NaN}, {...intrinsics(),model:'FISHEYE'}]) {
    frame.update(pose(),'point','context',k);
    assert.equal(frame.object.visible,false);assert.equal(frame.spec,null);
  }
  frame.update(pose(),'point','context',intrinsics(),false);
  assert.equal(frame.object.visible,false);
});

test('available deployed trials retain calibrated point frames at every submap', (t) => {
  let checked = 0;
  for(const trial of ['Ntrial3','60minutes_1','RoomScan3c','Fuchs','Fuchs22','Fuchs333']) {
    const path=new URL(`../runs/${trial}/environment_reconstruction/scene_playback_data.json`,import.meta.url);
    if(!fs.existsSync(path)) continue;
    checked += 1;
    const data=JSON.parse(fs.readFileSync(path,'utf8')), frame=new RecordedViewFrame(THREE);
    for(const scene of data.scenes) {
      frame.update(data.poses.find(Boolean),'point','context',scene.camera);
      assert.equal(frame.object.visible,true,`${trial}/${scene.id}`);
      assert.ok(Math.abs(frame.spec.width-1.3)<1e-8);
      assert.ok(Math.abs(frame.spec.height-1.3)<1e-8);
      assert.ok(Math.abs(frame.spec.horizontalFov-90)<1e-8);
    }
  }
  if (!checked) t.skip('Private deployed trial artifacts absent');
});

test('prototype is enabled only for the requested room scan', () => {
  assert.equal(supportsPoseFigure({trial_id: 'RoomScan3c'}), true);
  for (const trial_id of ['Ntrial3', '60minutes_1', 'Fuchs', 'Fuchs22', 'Fuchs333'])
    assert.equal(supportsPoseFigure({trial_id}), false);
  assert.match(FIGURE_NOTE, /not measured joints, posture, or foot contacts/);
  assert.match(FIGURE_NOTE, /not eye gaze/);
});

test('head follows forward, backward and repeated positions without a fake walking cycle', () => {
  const figure = new PoseFigure(THREE), original = JSON.stringify(BODY_JOINTS);
  const localBones = figure.body.children.map(o => o.position.toArray());
  for (const x of [0, 1, 2, 1, 0, -1, 0]) {
    const p = pose(x); figure.update(p);
    near(figure.head.getWorldPosition(new THREE.Vector3()).toArray(), p.slice(0, 3));
    assert.equal(figure.root.visible, true);
    assert.deepEqual(figure.body.children.map(o => o.position.toArray()), localBones);
  }
  assert.equal(JSON.stringify(BODY_JOINTS), original);
});

test('head arrow keeps measured yaw, pitch and roll, independently of travel direction', () => {
  const figure = new PoseFigure(THREE);
  for (const angles of [[1.5, .4, -.8], [.8, -.6, 2], [0, 0, .7]]) {
    const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(...angles));
    const p = [1, 3, 2, ...q.toArray()]; figure.update(p);
    assert.ok(1 - Math.abs(figure.head.quaternion.dot(q)) < 1e-10);
    near(figure.direction.toArray(), new THREE.Vector3(0, 0, -1).applyQuaternion(q).toArray());
    const up = new THREE.Vector3(0, 0, 1).applyQuaternion(figure.body.quaternion);
    near(up.toArray(), [0, 0, 1]);
    figure.update([-2, 3, 2, ...q.toArray()]);
    near(figure.direction.toArray(), new THREE.Vector3(0, 0, -1).applyQuaternion(q).toArray());
  }
});

test('missing/invalid poses and first-person mode hide the figure and its trail', () => {
  const figure = new PoseFigure(THREE), data = {fps: 10, duration_s: 2, poses: Array.from({length:20}, (_,i)=>pose(i / 10))};
  for (const p of [null, [0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [NaN, 0, 0, 0, 0, 0, 1]]) {
    figure.update(p); figure.updateTrail(data, 1);
    assert.equal(figure.root.visible, false); assert.equal(figure.trail.visible, false);
  }
  figure.update(pose(), false); figure.updateTrail(data, 1);
  assert.equal(figure.root.visible, false); assert.equal(figure.trail.visible, false);
  figure.update(pose()); figure.updateTrail(data, 1);
  assert.equal(figure.root.visible, true); assert.equal(figure.trail.visible, true);
});

test('recent trail follows recording time, preserves reversals and never bridges missing poses', () => {
  const data = {fps: 10, duration_s: 10, poses: Array.from({length:100}, (_,i)=>pose(i <= 50 ? i / 10 : (100 - i) / 10))};
  data.poses[70] = null;
  const edges = recentHeadPath(data, 8);
  assert.ok(edges.some(e=>e.b.position[0] < e.a.position[0]));
  assert.ok(edges.some(e=>e.b.position[0] > e.a.position[0]));
  assert.ok(edges.every(e=>!(e.a.time < 7 && e.b.time > 7)));
  assert.ok(edges.every(e=>e.a.time >= 2 && e.b.time <= 8));
  const rewound = recentHeadPath(data, 1);
  assert.ok(rewound.every(e=>e.b.time <= 1));
  near(roomObserverTarget(pose()), [0, 2, .95]);
});

test('room observer starts at a recorded nearby height, without changing any poses', () => {
  const data = {fps: 1, poses: [pose(), pose(1.8), [1.8, 2, 4, ...pose().slice(3)], null, pose(8)]};
  const before = JSON.stringify(data);
  const rig = roomObserverRig(data, pose(), 0);
  near(rig.position, [1.8, 2, 1.6]); near(rig.target, [0, 2, .95]);
  assert.equal(JSON.stringify(data), before);
  const fallback = roomObserverRig({fps:30,poses:[pose()]},pose(),0);
  assert.ok(fallback.position.every(Number.isFinite));
});

test('route animation stays off for stationary rotation, small head sway, and missing poses', () => {
  const stationary = {fps:30, poses:Array.from({length:300}, (_,i)=>[0,0,1.6,...new THREE.Quaternion().setFromEuler(new THREE.Euler(1,i/30,0)).toArray()])};
  assert.ok(illustrativeMotionTrack(stationary).every(x=>x===null));
  const sway = {fps:30, poses:Array.from({length:300},(_,i)=>pose(.03*Math.sin(i/5)))};
  assert.ok(illustrativeMotionTrack(sway).every(x=>x===null));
  const gap = {fps:30, poses:Array.from({length:300},(_,i)=>i>=140 && i<=160 ? null : pose(i/60))};
  const motion = illustrativeMotionTrack(gap);
  assert.ok(motion.slice(140,161).every(x=>x===null));
  assert.equal(motion[161]?.weight || 0, 0);
});

test('illustrative feet alternate, preserve the exact head pose and return when seeking backwards', () => {
  const data = {fps:30,poses:Array.from({length:300},(_,i)=>pose(i/60))};
  const track = illustrativeMotionTrack(data), before = JSON.stringify(data), figure = new PoseFigure(THREE);
  figure.update(data.poses[100],true,track[100]);
  assert.equal(figure.legsAnimated,true);
  const first = figure.feet.map(foot=>foot.getWorldPosition(new THREE.Vector3()).toArray());
  figure.update(data.poses[120],true,track[120]);
  assert.notDeepEqual(figure.feet.map(foot=>foot.getWorldPosition(new THREE.Vector3()).toArray()),first);
  near(figure.head.getWorldPosition(new THREE.Vector3()).toArray(), data.poses[120].slice(0,3));
  figure.update(data.poses[100],true,track[100]);
  assert.deepEqual(figure.feet.map(foot=>foot.getWorldPosition(new THREE.Vector3()).toArray()),first);
  figure.update(data.poses[100],true,null);
  assert.equal(figure.legsAnimated,false);
  near(figure.joints.leftAnkle.position.toArray(),BODY_JOINTS.leftAnkle);
  assert.equal(JSON.stringify(data),before);
});

test('travel reversal changes the leg travel direction, not the recorded head direction', () => {
  const data = {fps:30,poses:Array.from({length:300},(_,i)=>pose(i<150?i/60:(300-i)/60))};
  const track = illustrativeMotionTrack(data), figure=new PoseFigure(THREE);
  assert.ok(Math.abs(track[90].heading)<1e-6);
  assert.ok(Math.abs(Math.abs(track[210].heading)-Math.PI)<1e-6);
  figure.update(data.poses[90],true,track[90]); const forward=figure.direction.toArray();
  figure.update(data.poses[210],true,track[210]); near(figure.direction.toArray(),forward);
  figure.update(null,true,track[210]); assert.equal(figure.legsAnimated,false);
});

test('stride readout uses only candidate windows, independent of foot animation', () => {
  const bout={start_s:10,end_s:16,step_proxy_m:.3,stride_proxy_m:.6};
  assert.equal(stridePreviewAt({bouts:[bout]},12),bout);
  for(const time of [0,9.99,16,20]) assert.equal(stridePreviewAt({bouts:[bout]},time),null);
  assert.equal(stridePreviewAt(null,12),null);
  assert.match(FIGURE_NOTE,/NOT measured cadence or stride length/);
});
