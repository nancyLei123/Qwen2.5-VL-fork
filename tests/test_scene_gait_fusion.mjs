import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import * as THREE from '../vendor/gaussian-viewer/node_modules/three/build/three.module.js';
import {PoseFigure, BODY_JOINTS} from '../scripts/templates/scene_pose_figure.mjs';
import {compatibleGait, gaitMotionAt, evidenceTimeAt, EvidenceClip} from '../scripts/templates/scene_gait_fusion.mjs';

const root = new URL('../runs/RoomScan3c/environment_reconstruction/', import.meta.url);
const haveArtifacts = ['scene_playback_data.json', 'room_gait_fusion.json']
  .every(name => fs.existsSync(new URL(name, root)));
const source = haveArtifacts ? fs.readFileSync(new URL('scene_playback_data.json', root)) : null;
const data = haveArtifacts ? JSON.parse(source) : null;
const gait = haveArtifacts ? JSON.parse(fs.readFileSync(new URL('room_gait_fusion.json', root))) : null;
const artifactTest = (name, run) => test(name, {skip: !haveArtifacts && 'Private RoomScan3c artifacts absent'}, run);
const poseAt = time => data.poses[Math.floor(time * data.fps)];
const near = (a, b, tolerance = 1e-7) => assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);

artifactTest('prototype belongs only to the exact preserved RoomScan recording', () => {
  assert.equal(compatibleGait(gait, data), true);
  assert.equal(compatibleGait(gait, {...data, trial_id:'Ntrial3'}), false);
  assert.equal(compatibleGait({...gait, graph_uid:'wrong'}, data), false);
  assert.equal(createHash('sha256').update(source).digest('hex'), gait.source_playback_sha256);
  assert.equal(gait.source_playback_sha256, '996e2f25a0a8d0971c2a0a4a1b3771d9e884f50f142e027798c4dd9576b33694');
  assert.equal(gait.summary.cue_count, 15);
  assert.equal(gait.summary.window_count, 2);
});

artifactTest('exact frame-table clock and audio alignment round-trip every reviewed cue', () => {
  for (const cue of gait.windows.flatMap(w => w.cues)) {
    const result = evidenceTimeAt(cue.time_s, data, gait);
    near(result.downward_s, cue.downward_s);
    near(result.clip_s, cue.downward_s - gait.clip.source_start_s);
    assert.equal(cue.contact_measured, false);
  }
  for (const time of [0, gait.clip.start_s - .01, gait.clip.end_s, 500])
    assert.equal(evidenceTimeAt(time, data, gait), null);
});

artifactTest('each foot uses its own reviewed interval, not a fixed alternating metronome', () => {
  const window = gait.windows[0], time = window.cues[3].time_s + .3;
  const motion = gaitMotionAt(gait, time, poseAt(time));
  assert.equal(motion.kind, 'video_slam'); assert.equal(motion.measured, false);
  for (const side of ['left', 'right']) {
    const cycle = window.cycles.find(c=>c.side === side && c.start_s <= time && time < c.end_s);
    near(motion.feet[side].phase, (time - cycle.start_s) / (cycle.end_s - cycle.start_s));
    near(motion.feet[side].period_s, cycle.end_s - cycle.start_s);
  }
  assert.ok(Math.abs(motion.feet.left.period_s - motion.feet.right.period_s) > .1);
});

artifactTest('no phase is invented outside review windows, across the turn or without poses', () => {
  const gap = (gait.windows[0].end_s + gait.windows[1].start_s) / 2;
  for (const t of [1, gap, 600, gait.windows[1].end_s + .01])
    assert.equal(gaitMotionAt(gait, t, poseAt(t)), null);
  const t = gait.windows[0].start_s + 1;
  assert.equal(gaitMotionAt(gait, t, null), null);
  const early = gaitMotionAt(gait, gait.windows[0].start_s + .1, poseAt(t));
  assert.ok(early.feet.left); assert.equal(early.feet.right, undefined);
});

artifactTest('new animation preserves measured head, handles rewind, resets unsupported feet', () => {
  const fig = new PoseFigure(THREE), start = gait.windows[0].start_s;
  const t = start + 1, p = poseAt(t);
  const snapshot = () => fig.feet.map(f=>f.getWorldPosition(new THREE.Vector3()).toArray());
  fig.update(p, true, gaitMotionAt(gait,t,p)); const original = snapshot();
  assert.equal(fig.legsAnimated, true);
  fig.head.getWorldPosition(new THREE.Vector3()).toArray().forEach((v,i)=>near(v,p[i]));
  assert.ok(Math.abs(fig.head.quaternion.dot(new THREE.Quaternion(...p.slice(3)))) > .999999);
  fig.update(poseAt(t+.4),true,gaitMotionAt(gait,t+.4,poseAt(t+.4)));
  assert.notDeepEqual(snapshot(), original);
  fig.update(p,true,gaitMotionAt(gait,t,p)); assert.deepEqual(snapshot(),original);
  fig.update(p,true,null);
  assert.equal(fig.legsAnimated,false);
  assert.deepEqual(fig.joints.leftAnkle.position.toArray(),BODY_JOINTS.leftAnkle);
  assert.deepEqual(fig.joints.rightAnkle.position.toArray(),BODY_JOINTS.rightAnkle);
});

test('secondary video follows pause, play, rewind and leaving clip without touching master', async () => {
  const video = {readyState:2,paused:true,seeking:false,currentTime:0,playbackRate:1,
    pause(){this.paused=true;}, play(){this.paused=false;return Promise.resolve();}};
  const follower = new EvidenceClip(video);
  follower.update(4,false); near(video.currentTime,4); assert.equal(video.paused,true);
  follower.update(4.5,true,.5); near(video.currentTime,4.5); assert.equal(video.paused,false);
  near(video.playbackRate,.5);
  await Promise.resolve(); await Promise.resolve();
  follower.update(2,false); near(video.currentTime,2); assert.equal(video.paused,true);
  follower.update(null,true); assert.equal(video.paused,true);
  video.readyState=0; follower.update(7,true); assert.equal(video.paused,true);
});
