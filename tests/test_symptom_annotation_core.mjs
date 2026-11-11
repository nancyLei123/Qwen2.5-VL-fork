import test from 'node:test';
import assert from 'node:assert/strict';
import {
  formatTime, parseTime, videoToSession, sessionToVideo, videoCovers, inCoverage, windowAround, shiftWindow, assignLanes,
  unreviewedIntervals, nextUnreviewed, reviewedFraction, validateDraft, labelNameProblems, randomId, upsert,
  annotationsAt, tickStep,
} from '../scripts/templates/symptom_annotation/symptom_annotation_core.mjs';
import * as core from '../scripts/templates/symptom_annotation/symptom_annotation_core.mjs';

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

// ---- automatic predictions --------------------------------------------------------------------

const RUN = {run_id: 'motion-baseline-0.1.0-aaaaaaaa', detections: [
  {id: 'slow_gait-00010000', label_id: 'slow_gait', start_s: 10, end_s: 20, score: 80, evidence_strength: 'high'},
  {id: 'gait_hesitation-00012000', label_id: 'gait_hesitation', start_s: 12, end_s: 13, score: 35, evidence_strength: 'low'},
  {id: 'freezing_like-00040000', label_id: 'freezing_like', start_s: 40, end_s: 44, score: 55, evidence_strength: 'medium'},
]};
const REVIEWS = {revision: 2, reviews: {[RUN.run_id]: {
  'slow_gait-00010000': {decision: 'accepted', annotation_id: 'ann_one'},
  'freezing_like-00040000': {decision: 'unreviewed', history: [{decision: 'rejected'}]},
}}};

test('review decisions are looked up per run and a cleared decision counts as undecided', () => {
  assert.equal(core.reviewOf(REVIEWS, RUN.run_id, 'slow_gait-00010000').decision, 'accepted');
  assert.equal(core.reviewOf(REVIEWS, RUN.run_id, 'freezing_like-00040000'), null);
  assert.equal(core.reviewOf(REVIEWS, 'another-run', 'slow_gait-00010000'), null);
  assert.equal(core.reviewOf(null, RUN.run_id, 'slow_gait-00010000'), null);
  assert.deepEqual(core.predictionProgress(RUN, REVIEWS), {total: 3, accepted: 1, rejected: 0, needs_review: 0, unreviewed: 2});
});

test('low-evidence predictions can be hidden without changing the run', () => {
  assert.deepEqual(core.shownPredictions(RUN).map(item => item.start_s), [10, 12, 40]);
  assert.deepEqual(core.shownPredictions(RUN, {showLow: false}).map(item => item.id), ['slow_gait-00010000', 'freezing_like-00040000']);
  assert.equal(RUN.detections.length, 3);
  assert.deepEqual(core.shownPredictions(null), []);
});

test('stepping through predictions follows time order and wraps', () => {
  const list = core.shownPredictions(RUN);
  assert.equal(core.stepPrediction(list, null, 11, 1).id, 'gait_hesitation-00012000');
  assert.equal(core.stepPrediction(list, null, 100, 1).id, 'slow_gait-00010000');
  assert.equal(core.stepPrediction(list, null, 30, -1).id, 'gait_hesitation-00012000');
  assert.equal(core.stepPrediction(list, 'freezing_like-00040000', 0, 1).id, 'slow_gait-00010000');
  assert.equal(core.stepPrediction(list, 'slow_gait-00010000', 0, -1).id, 'freezing_like-00040000');
  assert.equal(core.stepPrediction([], null, 0, 1), null);
});

test('accepting a prediction makes a manual annotation that names its origin and leaves the prediction alone', () => {
  const prediction = RUN.detections[2];
  const before = JSON.stringify(prediction);
  const annotation = core.annotationFromPrediction(prediction, RUN.run_id, {id: 'ann_new', labelId: 'freezing_like', annotator: 'AB'});
  assert.deepEqual(annotation, {id: 'ann_new', label_id: 'freezing_like', start_s: 40, end_s: 44, confidence: 'medium',
    review_status: 'confirmed', notes: '', annotator: 'AB',
    origin: {type: 'detection', run_id: RUN.run_id, detection_id: 'freezing_like-00040000', edited: false}});
  assert.equal(JSON.stringify(prediction), before);
  const edited = core.annotationFromPrediction(prediction, RUN.run_id, {id: 'ann_e', labelId: 'uncertain', start_s: 39.5, end_s: 45, edited: true});
  assert.equal(edited.start_s, 39.5); assert.equal(edited.origin.edited, true);
  assert.equal(core.annotationOfPrediction([annotation], RUN.run_id, prediction.id), annotation);
  assert.equal(core.annotationOfPrediction([annotation], 'another-run', prediction.id), null);
  assert.equal(core.annotationOfPrediction([{id: 'ann_manual', label_id: 'uncertain', start_s: 40, end_s: 44}], RUN.run_id, prediction.id), null);
});

test('a prediction label reuses the catalogue label with the same id or name', () => {
  const labels = [{id: 'lbl_custom', name: 'Gait Hesitation', archived: false}, {id: 'slow_gait', name: 'Renamed', archived: false}];
  assert.equal(core.matchingLabel(labels, {id: 'slow_gait', name: 'Slowed gait'}).id, 'slow_gait');
  assert.equal(core.matchingLabel(labels, {id: 'gait_hesitation', name: 'Gait hesitation'}).id, 'lbl_custom');
  assert.equal(core.matchingLabel(labels, {id: 'freezing_like', name: 'Freezing-like episode'}), null);
});
