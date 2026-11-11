// End-to-end checks of automatic predictions on the Symptom Annotation page.
// Uses the E2E_Predictions trial and its hand-written detector run (see prepare_fixtures.py);
// decisions and annotations go to the temporary store, never into runs/.
import {test, expect} from '@playwright/test';
import {createHash} from 'node:crypto';
import {existsSync, readFileSync} from 'node:fs';
import path from 'node:path';

const TRIAL = 'E2E_Predictions';
const PAGE = `${TRIAL}/pd_insighter_symptom_annotation.html`;
const RUN = 'e2e-fixture-0.0.1-00000000';
const KNOWN = [/\/api\/manual-labels\?/, /\/api\/frame-context\?/];
const store = () => process.env.PD_INSIGHTER_E2E_STORE;
const folder = () => path.join(store(), TRIAL, 'symptom_annotations');
const runFile = () => path.join(folder(), 'detections', `${RUN}.json`);
const sha = file => createHash('sha256').update(readFileSync(file)).digest('hex');
const json = file => JSON.parse(readFileSync(file, 'utf8'));

test.describe.configure({mode: 'serial'});

function watch(page) {
  const problems = [];
  page.on('console', m => { if (m.type() === 'error' && !m.text().startsWith('Failed to load resource')) problems.push(m.text()); });
  page.on('pageerror', e => problems.push(e.message));
  page.on('response', r => { if (r.status() >= 400 && !KNOWN.some(p => p.test(r.url()))) problems.push(`${r.status()} ${r.url()}`); });
  return problems;
}

async function open(page, query = '?t=20') {
  await page.goto(`${PAGE}${query}`);
  await page.waitForFunction(() => document.documentElement.dataset.saveState === 'saved' && document.documentElement.dataset.predictions !== undefined);
  await page.waitForFunction(() => document.getElementById('ariaVideo').readyState >= 1);
  await page.selectOption('#windowSelect', '120');
}

const chip = (page, id) => page.locator(`.prediction[data-prediction-id="${id}"]`);
const sessionTime = page => page.evaluate(() => Number(document.documentElement.dataset.sessionTime));
let detectorOutput;

test('predictions are drawn apart from manual annotations and the Activity selector is unchanged', async ({page}) => {
  const problems = watch(page);
  detectorOutput = sha(runFile());
  await open(page);
  await expect(page.locator('#symptomLayers')).toBeVisible();
  await expect(page.locator('#symptomLayers option')).toHaveText(['Both', 'Manual', 'Automatic']);
  await expect(page.locator('#activitySource option')).toHaveText(['Automatic', 'Manual', 'Hide']);
  await expect(page.locator('.prediction')).toHaveCount(3);
  await expect(page.locator('#predictionFact')).toHaveText('3 predictions, 0 decided');
  await expect(page.locator('#legend')).toContainText('Predicted:');
  // Distinct treatment: a dashed outline, never the solid fill of a manual annotation.
  expect(await chip(page, 'freezing_like-00030000').evaluate(node => getComputedStyle(node).borderTopStyle)).toBe('dashed');
  await expect(page.locator('.annotation')).toHaveCount(0);
  // Position on the session clock: 30 s to 36 s of a 0-120 s window.
  const box = await chip(page, 'freezing_like-00030000').boundingBox();
  const track = await page.locator('#symptomTrack').boundingBox();
  expect((box.x - track.x) / track.width).toBeCloseTo(30 / 120, 2);
  expect(box.width / track.width).toBeCloseTo(6 / 120, 2);
  expect(problems).toEqual([]);
});

test('a trial without a detector run looks as before', async ({page}) => {
  await page.goto('Parki/pd_insighter_symptom_annotation.html?t=20');
  await page.waitForFunction(() => document.documentElement.dataset.predictions === 'none');
  await expect(page.locator('#symptomLayers')).toBeHidden();
  await expect(page.locator('.prediction')).toHaveCount(0);
  await expect(page.locator('#predictionFact')).toBeHidden();
  await expect(page.locator('#editorEmpty')).toBeVisible();
});

test('clicking a prediction jumps the videos there and explains it', async ({page}) => {
  const problems = watch(page);
  await open(page);
  await chip(page, 'freezing_like-00030000').click();
  await page.waitForFunction(() => Math.abs(Number(document.documentElement.dataset.sessionTime) - 30) < 0.05);
  await page.waitForFunction(() => Math.abs(downwardVideo.currentTime - ariaVideo.currentTime) < 0.05);
  const panel = page.locator('#predictionPanel');
  await expect(panel).toBeVisible();
  await expect(page.locator('#predLabel')).toHaveText('Freezing-like episode');
  await expect(page.locator('#predTime')).toHaveText('0:30.00 - 0:36.00 (6.0 s)');
  await expect(page.locator('#predScore')).toHaveText('Score 74 · High evidence');
  await expect(page.locator('#predScore')).toHaveAttribute('title', /not a probability/);
  await expect(page.locator('#predEvidence li')).toHaveCount(3);
  await expect(page.locator('#predEvidence li').nth(1)).toContainText('+34/40');
  await expect(page.locator('#predEvidence li').nth(1)).toContainText('Downward video');
  await expect(page.locator('#predEvidence li[data-kind="penalty"]')).toContainText('turns about 90 degrees');
  await expect(page.locator('#predLimits')).toContainText('Confirm with the video');
  await expect(chip(page, 'freezing_like-00030000')).toHaveClass(/active/);
  // Replay plays the interval with a second either side and stops by itself.
  await page.locator('#speedSelect').selectOption('2');
  await page.locator('#predReplay').click();
  await page.waitForFunction(() => Number(document.documentElement.dataset.sessionTime) > 31, null, {timeout: 20_000});
  await page.waitForFunction(() => ariaVideo.paused && Number(document.documentElement.dataset.sessionTime) >= 36.9, null, {timeout: 30_000});
  expect(await sessionTime(page)).toBeLessThan(38.5);
  // Stepping through predictions.
  await page.locator('#predNext').click();
  await expect(page.locator('#predLabel')).toHaveText('Gait hesitation');
  await page.keyboard.press(']');
  await expect(page.locator('#predLabel')).toHaveText('Slowed gait');
  expect(await sessionTime(page)).toBeCloseTo(90, 1);
  await page.keyboard.press('Escape');
  await expect(panel).toBeHidden();
  expect(problems).toEqual([]);
});

test('layers: both, manual only (predictions hidden) and automatic only', async ({page}) => {
  await open(page);
  await page.selectOption('#symptomLayers', 'manual');
  await expect(page.locator('.prediction')).toHaveCount(0);
  await expect(page.locator('#legend')).not.toContainText('Predicted:');
  await page.reload();
  await page.waitForFunction(() => document.documentElement.dataset.predictions !== undefined && document.documentElement.dataset.saveState === 'saved');
  await expect(page.locator('#symptomLayers')).toHaveValue('manual');          // the choice is remembered
  await expect(page.locator('.prediction')).toHaveCount(0);
  await page.selectOption('#symptomLayers', 'automatic');
  await page.selectOption('#windowSelect', '120');
  await expect(page.locator('.prediction')).toHaveCount(3);
  await expect(page.locator('.unreviewed')).toHaveCount(0);
  await page.selectOption('#symptomLayers', 'both');
  await expect(page.locator('.prediction')).toHaveCount(3);
  // Low-evidence predictions can be hidden from the settings menu.
  await page.locator('#settingsBtn').click();
  await expect(page.locator('#predictionRunNote')).toContainText('uncalibrated');
  await page.locator('#showLowPredictions').uncheck();
  await expect(page.locator('.prediction')).toHaveCount(2);
  await page.locator('#showLowPredictions').check();
  await page.keyboard.press('Escape');
  await expect(page.locator('.prediction')).toHaveCount(3);
});

test('reject keeps the prediction and stores the decision apart from it', async ({page}) => {
  const problems = watch(page);
  await open(page);
  await chip(page, 'gait_hesitation-00062000').click();
  await page.locator('#predReject').click();
  await expect(page.locator('#predState')).toHaveText('Rejected');
  await expect(chip(page, 'gait_hesitation-00062000')).toHaveAttribute('data-decision', 'rejected');
  await expect(page.locator('#predictionFact')).toHaveText('3 predictions, 1 decided');
  const reviews = json(path.join(folder(), 'detection_reviews.json'));
  expect(reviews.reviews[RUN]['gait_hesitation-00062000'].decision).toBe('rejected');
  expect(reviews.reviews[RUN]['gait_hesitation-00062000'].detection.score).toBe(38);
  expect(sha(runFile())).toBe(detectorOutput);                                   // detector output untouched
  expect(existsSync(path.join(folder(), 'annotations.json'))).toBe(false);       // no manual annotation was made
  await page.reload();
  await page.waitForFunction(() => document.documentElement.dataset.predictions !== undefined && document.documentElement.dataset.saveState === 'saved');
  await page.selectOption('#windowSelect', '120');
  await expect(chip(page, 'gait_hesitation-00062000')).toHaveAttribute('data-decision', 'rejected');
  // A decision can be cleared; the earlier one stays in the history.
  await chip(page, 'gait_hesitation-00062000').click();
  await page.locator('#predLater').click();
  await expect(page.locator('#predState')).toHaveText('Needs review');
  await page.locator('#predUndo').click();
  await expect(page.locator('#predState')).toBeHidden();
  const after = json(path.join(folder(), 'detection_reviews.json')).reviews[RUN]['gait_hesitation-00062000'];
  expect(after.history.map(entry => entry.decision)).toEqual(['rejected', 'needs_review']);
  await page.locator('#predReject').click();
  await expect(page.locator('#predState')).toHaveText('Rejected');
  expect(problems).toEqual([]);
});

test('accept creates a manual annotation with the suggested label and range', async ({page}) => {
  const problems = watch(page);
  await open(page);
  await chip(page, 'freezing_like-00030000').click();
  await page.locator('#predAccept').click();
  await expect(page.locator('#predState')).toHaveText('Accepted');
  await expect(page.locator('#annotationRows tr')).toHaveCount(1);
  await expect(page.locator('#annotationRows tr').first()).toContainText('Freezing-like episode');
  await expect(page.locator('#annotationRows tr').first()).toContainText('0:30.00');
  await expect(page.locator('.annotation')).toHaveCount(1);
  await expect(chip(page, 'freezing_like-00030000')).toHaveAttribute('data-decision', 'accepted');
  const saved = json(path.join(folder(), 'annotations.json')).annotations;
  expect(saved).toHaveLength(1);
  expect(saved[0]).toMatchObject({label_id: 'freezing_like', start_s: 30, end_s: 36, source: 'manual', confidence: 'high',
    origin: {type: 'detection', run_id: RUN, detection_id: 'freezing_like-00030000', edited: false}});
  const review = json(path.join(folder(), 'detection_reviews.json')).reviews[RUN]['freezing_like-00030000'];
  expect(review).toMatchObject({decision: 'accepted', annotation_id: saved[0].id, edited: false});
  expect(sha(runFile())).toBe(detectorOutput);
  // The label was added to the shared catalogue so it can also be used by hand.
  expect(json(path.join(store(), 'symptom_annotation_labels.json')).labels.map(label => label.id)).toContain('freezing_like');
  // Accepting twice does not create a second annotation.
  await expect(page.locator('#predAccept')).toBeDisabled();
  expect(problems).toEqual([]);
});

test('edit adjusts the range before accepting and records that it was edited', async ({page}) => {
  const problems = watch(page);
  await open(page);
  await chip(page, 'slow_gait-00090000').click();
  await page.locator('#predEdit').click();
  await expect(page.locator('#editorTitle')).toHaveText('Adjust prediction, then save');
  await expect(page.locator('#startInput')).toHaveValue('1:30.000');
  await page.locator('#endInput').fill('1:45');
  await page.locator('#endInput').press('Tab');
  await page.locator('.label-choice', {hasText: 'Uncertain'}).click();
  await page.locator('#saveBtn').click();
  await expect(page.locator('#saveStatus')).toHaveText('Prediction accepted with changes');
  await expect(page.locator('#annotationRows tr')).toHaveCount(2);
  const saved = json(path.join(folder(), 'annotations.json')).annotations.find(item => item.origin?.detection_id === 'slow_gait-00090000');
  expect(saved).toMatchObject({label_id: 'uncertain', start_s: 90, end_s: 105, origin: {edited: true}});
  const review = json(path.join(folder(), 'detection_reviews.json')).reviews[RUN]['slow_gait-00090000'];
  expect(review).toMatchObject({decision: 'accepted', edited: true, accepted: {label_id: 'uncertain', start_s: 90, end_s: 105}});
  expect(review.detection).toMatchObject({label_id: 'slow_gait', start_s: 90, end_s: 110});   // what the detector said is kept
  expect(sha(runFile())).toBe(detectorOutput);
  await expect(page.locator('#predictionFact')).toHaveText('3 predictions, 3 decided');
  // Cancelling an edit accepts nothing.
  await chip(page, 'gait_hesitation-00062000').click();
  await page.locator('#predEdit').click();
  await page.locator('#cancelBtn').click();
  await expect(page.locator('#annotationRows tr')).toHaveCount(2);
  expect(problems).toEqual([]);
});

test('missed events can still be annotated by hand and manual edits never change predictions', async ({page}) => {
  const problems = watch(page);
  await open(page);
  const track = await page.locator('#symptomTrack').boundingBox();
  const y = track.y + track.height - 14;                                          // below the prediction lane
  await page.mouse.move(track.x + track.width * 0.42, y);
  await page.mouse.down();
  await page.mouse.move(track.x + track.width * 0.47, y, {steps: 6});
  await page.mouse.up();
  await expect(page.locator('#editorTitle')).toHaveText('New annotation');
  await page.locator('.label-choice', {hasText: 'Simulated Symptom'}).click();
  await page.locator('#saveBtn').click();
  await expect(page.locator('#saveStatus')).toHaveText('Annotation saved');
  await expect(page.locator('#annotationRows tr')).toHaveCount(3);
  const manual = json(path.join(folder(), 'annotations.json')).annotations.find(item => item.label_id === 'simulated_symptom');
  expect(manual.origin).toBeUndefined();
  // Deleting the annotation made from a prediction leaves the prediction and its decision in place.
  page.once('dialog', dialog => dialog.accept());
  await page.locator('#annotationRows tr', {hasText: 'Freezing-like episode'}).locator('.row-menu-btn').click();
  await page.locator('#rowMenu [data-action="delete"]').click();
  await expect(page.locator('#annotationRows tr')).toHaveCount(2);
  await expect(page.locator('.prediction')).toHaveCount(3);
  expect(sha(runFile())).toBe(detectorOutput);
  expect(json(runFile()).detections).toHaveLength(3);
  expect(json(path.join(folder(), 'detection_reviews.json')).reviews[RUN]['freezing_like-00030000'].decision).toBe('accepted');
  // Nothing was written into the trial data.
  expect(existsSync(path.join(process.env.PD_INSIGHTER_E2E_RUNS, TRIAL, 'symptom_annotations'))).toBe(false);
  expect(problems).toEqual([]);
});
