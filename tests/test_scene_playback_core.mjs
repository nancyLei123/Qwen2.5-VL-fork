import test from 'node:test';
import assert from 'node:assert/strict';
import {frameAt, recordingAt, sceneAt, labelAt, formatExactTime, missingIntervals, playbackCoverage, videoLabelSegments, SceneQueue} from '../scripts/templates/scene_playback_core.mjs';

test('video time selects an exposure, not a proportion of total VRS duration', () => {
  const data = {fps: 30, poses: [1, 1, 1], recording_times: [0, 0.032, 0.067]};
  assert.equal(frameAt(1 / 30, data), 1);
  assert.equal(frameAt(0.033333, data), 1);
  assert.equal(frameAt(0.03330, data), 0);
  assert.equal(recordingAt(1 / 30, data), 0.032);
  assert.equal(frameAt(10, data), 2);
});

test('label offset is applied once and uncovered labels stay unknown', () => {
  const segments = [{start_s: 0, end_s: 4, label: 'standing'}, {start_s: 4, end_s: 8, label: 'walking'}];
  assert.equal(labelAt(10, segments, -15.133333, 216), null);
  assert.equal(labelAt(16, segments, -15.133333, 216).label, 'standing');
  assert.equal(labelAt(19.133333, segments, -15.133333, 216).label, 'walking');
});

test('model boundary chooses the next model and does not fill coverage gaps', () => {
  const scenes = [{id: 'a', start_s: 0, end_s: 24}, {id: 'b', start_s: 24, end_s: 60}, {id: 'c', start_s: 78, end_s: 120}];
  assert.equal(sceneAt(24, scenes).id, 'b');
  assert.equal(sceneAt(61, scenes), null);
  assert.equal(sceneAt(216, scenes), null);
});

test('exact gap timestamps retain subsecond durations and carry into minutes', () => {
  assert.equal(formatExactTime(29 / 30), '0:00.967');
  assert.equal(formatExactTime(2477 / 30), '1:22.567');
  assert.equal(formatExactTime(59.9999), '1:00.000');
});

test('missing intervals merge adjacent/overlapping coverage without inventing gaps', () => {
  assert.deepEqual(missingIntervals([
    {start_s: 5, end_s: 8}, {start_s: -2, end_s: 1}, {start_s: 3, end_s: 6},
    {start_s: 8, end_s: 9}, {start_s: 12, end_s: 15}, {start_s: NaN, end_s: 4},
  ], 10), [{start_s: 1, end_s: 3}, {start_s: 9, end_s: 10}]);
  assert.deepEqual(missingIntervals([], 3), [{start_s: 0, end_s: 3}]);
  assert.deepEqual(missingIntervals([{start_s: -1, end_s: 4}], 3), []);
});

test('label coverage uses the existing offset and clips to the reviewed dashboard interval', () => {
  const data = {dashboard_offset_s: -2, dashboard_duration_s: 5, duration_s: 10};
  const input = [{start_s: -1, end_s: 1, label: 'standing'}, {start_s: 4, end_s: 9, label: 'walking'}];
  const original = JSON.stringify(input);
  const converted = videoLabelSegments(input, data);
  assert.deepEqual(converted.map(s => [s.start_s, s.end_s]), [[2, 3], [6, 7]]);
  assert.deepEqual(missingIntervals(converted, 10), [
    {start_s: 0, end_s: 2}, {start_s: 3, end_s: 6}, {start_s: 7, end_s: 10},
  ]);
  assert.equal(JSON.stringify(input), original);
});

test('pose gaps include the initial, single-frame and final gaps independently of model coverage', () => {
  const data = {fps: 2, duration_s: 3, poses: [null, null, [1], null, [1], null],
    recording_times: [0, 0.45, 1.6, 2, 2.4, 3.5],
    scenes: [{id: 'a', title: 'A', start_s: 0, end_s: 1, iterations: 7000},
      {id: 'b', title: 'B', start_s: 2, end_s: 3, iterations: 1500}],
    dashboard_offset_s: -0.5, dashboard_duration_s: 2};
  const coverage = playbackCoverage(data, [{start_s: 0, end_s: 2, label: 'walking'}], []);
  assert.deepEqual(coverage.poseGaps.map(s => [s.start_s, s.end_s]), [[0, 1], [1.5, 2], [2.5, 3]]);
  assert.deepEqual(coverage.modelGaps.map(s => [s.start_s, s.end_s]), [[1, 1.5], [2.5, 3]]);
  assert.deepEqual(coverage.modelSegments.map(s => [s.start_s, s.end_s, s.iterations]), [[0, 1, 7000], [1.5, 2.5, 1500]]);
  assert.deepEqual(coverage.referenceGaps, [{start_s: 0, end_s: 3}]);
  for (let i = 0; i < data.poses.length; i++) {
    const t = (i + 0.5) / data.fps;
    const inGap = gaps => gaps.some(s => s.start_s <= t && t < s.end_s);
    assert.equal(inGap(coverage.poseGaps), !data.poses[i]);
    assert.equal(inGap(coverage.modelGaps), !sceneAt(recordingAt(t, data), data.scenes));
  }
});

test('estimated poses are identified separately from both supported and unavailable poses', () => {
  const data = {fps: 2, duration_s: 2, poses: [[1], [2], null, [3]],
    pose_status: ['mps_supported', 'interpolated_short_gap', 'unavailable', 'mps_supported'],
    recording_times: [0, 0.5, 1, 1.5], scenes: [], dashboard_offset_s: 0, dashboard_duration_s: 2};
  const coverage = playbackCoverage(data, [], []);
  assert.deepEqual(coverage.poseEstimates.map(s => [s.start_s, s.end_s]), [[0.5, 1]]);
  assert.deepEqual(coverage.poseGaps.map(s => [s.start_s, s.end_s]), [[1, 1.5]]);
  delete data.pose_status;
  assert.deepEqual(playbackCoverage(data, [], []).poseEstimates, []);
});

const tick = () => new Promise(resolve => setTimeout(resolve, 0));
test('a rapid seek does not display a stale scene when its asynchronous load finishes', async () => {
  const resolutions = new Map(), shows = [], starts = [];
  const queue = new SceneQueue(scene => {
    starts.push(scene.id);
    return new Promise(resolve => resolutions.set(scene.id, resolve));
  }, id => shows.push(id), () => {});
  queue.request({id: 'a'});
  queue.request({id: 'b'});
  resolutions.get('a')('model-a'); await tick();
  assert.deepEqual(starts, ['a', 'b']);
  assert.ok(!shows.includes('model-a'));
  resolutions.get('b')('model-b'); await tick();
  assert.equal(shows.at(-1), 'model-b');
  queue.request({id: 'a'});
  assert.equal(shows.at(-1), 'model-a');
});

test('seek into uncovered time clears the model despite an in-flight load', async () => {
  let finish; const shows = [];
  const queue = new SceneQueue(() => new Promise(resolve => { finish = resolve; }), id => shows.push(id), () => {});
  queue.request({id: 'a'}); queue.request(null);
  finish('model-a'); await tick();
  assert.ok(shows.every(value => value === null));
});

test('load failures are reported and retry is explicit', async () => {
  let attempts = 0; const states = [];
  const queue = new SceneQueue(async () => {
    if (++attempts === 1) throw new Error('network');
    return 'model';
  }, () => {}, status => states.push(status));
  queue.request({id: 'a'}); await tick();
  assert.equal(states.at(-1), 'error'); assert.equal(attempts, 1);
  queue.retry(); await tick();
  assert.equal(states.at(-1), 'ready'); assert.equal(attempts, 2);
});
