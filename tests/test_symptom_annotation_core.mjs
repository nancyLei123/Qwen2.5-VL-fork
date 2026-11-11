import test from 'node:test';
import assert from 'node:assert/strict';
import {
  formatTime, parseTime, videoToSession, sessionToVideo, videoCovers, inCoverage, windowAround, shiftWindow, assignLanes,
  unreviewedIntervals, nextUnreviewed, reviewedFraction, validateDraft, labelNameProblems, randomId, upsert,
  annotationsAt, tickStep,
} from '../scripts/templates/symptom_annotation/symptom_annotation_core.mjs';

const labels = [
  {id: 'simulated_symptom', name: 'Simulated Symptom', archived: false},
  {id: 'old_label', name: 'Old Label', archived: true},
];
const a = (id, start_s, end_s, label_id = 'simulated_symptom') => ({id, start_s, end_s, label_id});

test('times format and parse without losing sub-second precision', () => {
  assert.equal(formatTime(83.4), '1:23.40');
  assert.equal(formatTime(3725.125, 3), '1:02:05.125');
  assert.equal(formatTime(0), '0:00.00');
  assert.equal(parseTime('83.25'), 83.25);
  assert.equal(parseTime('1:23.25'), 83.25);
  assert.equal(parseTime('0:01:23.250'), 83.25);
  for (const bad of ['', 'abc', '1:75', '-3', '1.5:20', '1:2:3:4']) assert.equal(parseTime(bad), null, bad);
});

test('a video offset maps to the session clock and back exactly once', () => {
  // A camera that started 14.4667 s into the session.
  const offset = 14.466667;
  assert.equal(videoToSession(10, offset), 24.466667);
  assert.ok(Math.abs(sessionToVideo(videoToSession(123.456, offset), offset) - 123.456) < 1e-9);
  assert.equal(videoCovers(10, offset, 400), false);   // before this camera started
  assert.equal(videoCovers(20, offset, 400), true);
  assert.equal(videoCovers(500, offset, 400), false);  // after it ended
  assert.equal(videoCovers(0, 0, 490), true);
});

test('an interval marked while watching either camera keeps its session time', () => {
  const ariaOffset = 0, downwardOffset = 2.5;
  const fromAria = videoToSession(100, ariaOffset);
  const fromDownward = videoToSession(97.5, downwardOffset);
  assert.equal(fromAria, fromDownward);
});

test('padded downward video only counts as footage inside its recorded window', () => {
  // Aria-clock attachment: the downward file spans the session, real footage only 9.5-830.1 s.
  const window = [9.5, 830.1];
  assert.equal(inCoverage(5, window), false);
  assert.equal(inCoverage(9.5, window), true);
  assert.equal(inCoverage(830.1, window), false);
  assert.equal(inCoverage(5, null), true);
});

test('the visible window stays inside the session', () => {
  assert.deepEqual(windowAround(10, 60, 490), {start: 0, end: 60});
  assert.deepEqual(windowAround(480, 60, 490), {start: 430, end: 490});
  assert.deepEqual(windowAround(200, 60, 490), {start: 170, end: 230});
  assert.deepEqual(windowAround(200, Infinity, 490), {start: 0, end: 490});
  assert.deepEqual(shiftWindow({start: 0, end: 60}, 1, 490), {start: 54, end: 114});
  assert.deepEqual(shiftWindow({start: 400, end: 460}, 1, 490), {start: 430, end: 490});
});

test('overlapping annotations get separate lanes; touching ones share a lane', () => {
  const {lanes, count} = assignLanes([a('x', 0, 10), a('y', 5, 15), a('z', 10, 20), a('w', 6, 8)]);
  assert.equal(count, 3);
  assert.equal(lanes.get('x'), 0);
  assert.equal(lanes.get('y'), 1);
  assert.equal(lanes.get('z'), 0);
  assert.equal(lanes.get('w'), 2);
});

test('unreviewed time is everything without an annotation, never assumed normal', () => {
  assert.deepEqual(unreviewedIntervals([], 100), [{start_s: 0, end_s: 100}]);
  const gaps = unreviewedIntervals([a('x', 10, 20), a('y', 15, 30), a('z', 50, 100)], 100);
  assert.deepEqual(gaps, [{start_s: 0, end_s: 10}, {start_s: 30, end_s: 50}]);
  assert.equal(reviewedFraction([a('x', 10, 20), a('y', 15, 30), a('z', 50, 100)], 100), 0.7);
  assert.deepEqual(nextUnreviewed([a('x', 0, 20), a('z', 50, 100)], 100, 25), {start_s: 20, end_s: 50});
  // Inside an unreviewed stretch, N moves on (here: wraps to the first stretch).
  assert.deepEqual(nextUnreviewed([a('x', 10, 20)], 100, 50), {start_s: 0, end_s: 10});
  assert.deepEqual(nextUnreviewed([a('x', 10, 20)], 100, 5), {start_s: 20, end_s: 100});
  assert.deepEqual(nextUnreviewed([a('x', 10, 20)], 100, 99.999), {start_s: 0, end_s: 10});
  assert.equal(nextUnreviewed([a('x', 0, 100)], 100, 5), null);
});

test('drafts are checked before saving', () => {
  const ok = {start_s: 1.25, end_s: 3.5, label_id: 'simulated_symptom', confidence: 'medium', isNew: true};
  assert.deepEqual(validateDraft(ok, labels, 100), []);
  assert.match(validateDraft({...ok, end_s: 1}, labels, 100).join(), /after the start/);
  assert.match(validateDraft({...ok, start_s: -1}, labels, 100).join(), /negative/);
  assert.match(validateDraft({...ok, end_s: 101}, labels, 100).join(), /end of the session/);
  assert.match(validateDraft({...ok, start_s: NaN}, labels, 100).join(), /start and end/);
  assert.match(validateDraft({...ok, label_id: 'missing'}, labels, 100).join(), /Choose a label/);
  assert.match(validateDraft({...ok, label_id: 'old_label'}, labels, 100).join(), /archived/);
  // Editing an older annotation that uses an archived label is still allowed.
  assert.deepEqual(validateDraft({...ok, label_id: 'old_label', isNew: false}, labels, 100), []);
});

test('label names must be unique ignoring case and spacing', () => {
  assert.deepEqual(labelNameProblems('Freezing', labels), []);
  assert.match(labelNameProblems('  simulated   SYMPTOM ', labels)[0], /already exists/);
  assert.match(labelNameProblems('old label', labels)[0], /archived/);
  assert.deepEqual(labelNameProblems('Simulated Symptom', labels, 'simulated_symptom'), []);
  assert.match(labelNameProblems('   ', labels)[0], /Enter a label name/);
});

test('ids are stable-format and lists stay sorted by time', () => {
  const id = randomId('ann');
  assert.match(id, /^ann_[a-z0-9]{10}$/);
  let list = upsert([], a('b', 5, 6));
  list = upsert(list, a('a', 1, 2));
  list = upsert(list, {...a('b', 0, 1)});
  assert.deepEqual(list.map(item => [item.id, item.start_s]), [['b', 0], ['a', 1]]);
  assert.deepEqual(annotationsAt([a('x', 0, 10), a('y', 5, 15)], 7).map(item => item.id), ['x', 'y']);
  assert.equal(tickStep(30), 5);
  assert.equal(tickStep(490), 60);
});
