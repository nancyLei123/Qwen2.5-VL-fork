import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../vendor/gaussian-viewer/node_modules/three/build/three.module.js';
import {OrbitControls} from '../vendor/gaussian-viewer/node_modules/@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js';
import {AnchoredCameraView, ViewTap, disableViewerDebugOverlays} from '../scripts/templates/scene_camera_view.mjs';

test('debug cones stay disabled without removing playback keyboard controls', () => {
  const target = new EventTarget();
  const visible = {cursor: true, plane: true};
  const viewer = {
    showMeshCursor: true, showControlPlane: true,
    sceneHelper: {
      setMeshCursorVisibility(value) { visible.cursor = value; },
      setControlPlaneVisibility(value) { visible.plane = value; },
    },
  };
  viewer.keyDownListener = () => { viewer.showMeshCursor = viewer.showControlPlane = true; };
  let playbackEvents = 0;
  target.addEventListener('keydown', viewer.keyDownListener);
  target.addEventListener('keydown', () => playbackEvents++);
  disableViewerDebugOverlays(viewer, target);
  for (const code of ['KeyC', 'KeyU', 'Space']) {
    const event = new Event('keydown');
    Object.defineProperty(event, 'code', {value: code});
    target.dispatchEvent(event);
  }
  assert.deepEqual(visible, {cursor: false, plane: false});
  assert.equal(viewer.showMeshCursor, false);
  assert.equal(viewer.showControlPlane, false);
  assert.equal(playbackEvents, 3);
});

function fixture() {
  const camera = new THREE.PerspectiveCamera(90, 1, .05, 300);
  camera.up.set(0, 0, 1);
  camera.position.set(1, -2, 3);
  const controls = {object: camera, target: new THREE.Vector3(1, -1, 3), update() { this.object.lookAt(this.target); }};
  const view = new AnchoredCameraView(camera, controls);
  controls.update(); view.apply();
  return {camera, controls, view};
}
const equalVector = (a, b) => assert.ok(a.distanceTo(b) < 1e-10);
const equalRotation = (a, b) => assert.ok(1 - Math.abs(a.dot(b)) < 1e-10);

test('installed viewer OrbitControls cannot overwrite the held look direction', () => {
  const camera = new THREE.PerspectiveCamera(90, 1, .05, 300);
  camera.up.set(0, 0, 1); camera.position.set(2, -3, 4);
  const element = {style: {}, addEventListener() {}, removeEventListener() {}};
  const controls = new OrbitControls(camera, element);
  controls.enableDamping = false; controls.target.set(2, -2, 4); controls.update();
  const view = new AnchoredCameraView(camera, controls);
  view.lookAtScreen(.5, .2);
  const held = camera.quaternion.clone();
  for (let i = 0; i < 180; i++) {
    controls.target.x += .01; view.orbitCamera.position.x += .01;
    controls.update(); view.apply();
    equalRotation(camera.quaternion, held);
  }
  controls.dispose();
});

test('click centers the chosen ray and survives repeated playback/control updates', () => {
  const {camera, controls, view} = fixture();
  const before = camera.quaternion.clone(), pivot = controls.target.clone(), position = camera.position.clone();
  const clicked = new THREE.Vector3(.6, .3, .5).unproject(camera).sub(camera.position).normalize();
  assert.equal(view.lookAtScreen(.6, .3), true);
  equalVector(camera.getWorldDirection(new THREE.Vector3()), clicked);
  assert.ok(camera.quaternion.angleTo(before) > .3);
  const after = camera.quaternion.clone();
  for (let i = 0; i < 180; i++) { controls.update(); view.apply(); }
  equalRotation(camera.quaternion, after);
  equalVector(camera.position, position); equalVector(controls.target, pivot);
  assert.equal(view.locked, true);
});

test('moving the wearer across time/submaps preserves chosen direction and distance', () => {
  const {camera, controls, view} = fixture();
  view.lookAtScreen(-.4, .2);
  const rotation = camera.quaternion.clone(), distance = camera.position.distanceTo(controls.target);
  const delta = new THREE.Vector3(23, -8, 2);
  controls.target.add(delta); view.orbitCamera.position.add(delta); controls.update(); view.apply();
  equalRotation(camera.quaternion, rotation);
  assert.ok(Math.abs(camera.position.distanceTo(controls.target) - distance) < 1e-10);
});

test('dragging still orbits the wearer and distance changes retain the look offset', () => {
  const {camera, controls, view} = fixture();
  view.lookAtScreen(.4, .1);
  const offset = view.lookOffset.clone(), before = camera.quaternion.clone();
  view.orbitCamera.position.sub(controls.target).applyAxisAngle(new THREE.Vector3(0, 0, 1), .3).add(controls.target);
  controls.update(); view.apply();
  equalRotation(view.lookOffset, offset);
  assert.ok(camera.quaternion.angleTo(before) > .2);
  const turned = camera.quaternion.clone();
  view.orbitCamera.position.sub(controls.target).multiplyScalar(2).add(controls.target);
  controls.update(); view.apply(); equalRotation(camera.quaternion, turned);
});

test('a second click replaces the held direction; reset restores the centered view', () => {
  const {camera, controls, view} = fixture();
  view.lookAtScreen(.5, .1);
  const clicked = new THREE.Vector3(-.7, -.2, .5).unproject(camera).sub(camera.position).normalize();
  view.lookAtScreen(-.7, -.2); equalVector(camera.getWorldDirection(new THREE.Vector3()), clicked);
  view.reset();
  equalRotation(camera.quaternion, view.orbitCamera.quaternion);
  equalVector(camera.getWorldDirection(new THREE.Vector3()), controls.target.clone().sub(camera.position).normalize());
  assert.equal(view.locked, false);
});

test('invalid/off-canvas clicks leave the view untouched', () => {
  const {view} = fixture();
  for (const point of [[NaN, 0], [0, Infinity], [1.1, 0], [0, -1.1]]) assert.equal(view.lookAtScreen(...point), false);
  assert.equal(view.locked, false);
});

const event = overrides => ({pointerId: 1, button: 0, isPrimary: true, clientX: 20, clientY: 30, timeStamp: 100, ...overrides});
test('only short primary left taps aim the view', () => {
  const tap = new ViewTap();
  tap.down(event()); assert.equal(tap.up(event({timeStamp: 200})), true);
  for (const bad of [{button: 2}, {button: 1}, {isPrimary: false}]) {
    tap.down(event(bad)); assert.equal(tap.up(event({...bad, timeStamp: 200})), false);
  }
  tap.down(event()); assert.equal(tap.up(event({timeStamp: 700})), false);
  tap.down(event()); assert.equal(tap.up(event({pointerId: 2})), false);
});

test('dragging away and returning, cancelled gestures and multi-touch are not clicks', () => {
  const tap = new ViewTap();
  tap.down(event()); tap.move(event({clientX: 40})); tap.move(event());
  assert.equal(tap.up(event({timeStamp: 200})), false);
  tap.down(event()); tap.cancel(); assert.equal(tap.up(event({timeStamp: 200})), false);
  tap.down(event()); tap.down(event({pointerId: 2, isPrimary: false}));
  assert.equal(tap.up(event({timeStamp: 200})), false);
  tap.down(event()); assert.equal(tap.up(event({clientX: 80, timeStamp: 200})), false);
});
