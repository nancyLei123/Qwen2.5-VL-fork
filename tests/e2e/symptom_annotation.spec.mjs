// End-to-end checks of the Symptom Annotation page on the Parki trial.
// Needs the generated runs/Parki/pd_insighter_symptom_annotation.html in this checkout.
import {test, expect} from '@playwright/test';
import {existsSync, readdirSync} from 'node:fs';
import path from 'node:path';

const PAGE = 'Parki/pd_insighter_symptom_annotation.html';
// Existing dashboard behaviour, not part of this feature: the saved metadata points at a Mac
// audio path (manual-labels), and the frame strip asks for downward frames past its end.
const KNOWN = [/\/api\/manual-labels\?/, /\/api\/frame-context\?/];

test.describe.configure({mode: 'serial'});

function watch(page) {
  const problems = [];
  page.on('console', m => { if (m.type() === 'error' && !m.text().startsWith('Failed to load resource')) problems.push(m.text()); });
  page.on('pageerror', e => problems.push(e.message));
  page.on('response', r => { if (r.status() >= 400 && !KNOWN.some(p => p.test(r.url()))) problems.push(`${r.status()} ${r.url()}`); });
  return problems;
}

async function open(page, query = '') {
  await page.goto(`${PAGE}${query}`);
  await page.waitForFunction(() => ['saved', 'error'].includes(document.documentElement.dataset.saveState));
  await page.waitForFunction(() => document.getElementById('ariaVideo').readyState >= 1);
}

async function dragSymptoms(page, fromFraction, toFraction) {
  const box = await page.locator('#symptomTrack').boundingBox();
  const y = box.y + 20;
  await page.mouse.move(box.x + box.width * fromFraction, y);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * toFraction, y, {steps: 8});
  await page.mouse.up();
}

async function goTo(page, text) {
  await page.locator('#clockBtn').click();
  await page.locator('#gotoInput').fill(text);
  await page.locator('#gotoInput').press('Enter');
}

const view = page => page.evaluate(() => {
  const label = document.getElementById('windowLabel').textContent;
  return {label, time: Number(document.documentElement.dataset.sessionTime)};
});

test('the tab is part of the trial navigation and keeps the playback position', async ({page}) => {
  const problems = watch(page);
  await page.goto('Parki/index.html');
  const tab = page.locator('a.nav-tab[data-view="symptoms"]');
  await expect(tab).toHaveText('Symptom Annotation');
  await page.waitForFunction(() => document.getElementById('ariaVideo').readyState >= 1);
  await page.evaluate(() => seekSession(42.5, {preservePlayback: false}));
  await tab.click();
  await expect(page).toHaveURL(/pd_insighter_symptom_annotation\.html\?t=42\.5/);
  await page.waitForFunction(() => Math.abs(Number(document.documentElement.dataset.sessionTime) - 42.5) < 0.05);
  await expect(page.locator('span.nav-tab.active')).toHaveText('Symptom Annotation');
  await page.locator('#forward5').click();
  await page.locator('a.nav-tab', {hasText: 'Full Session'}).click();
  await expect(page).toHaveURL(/index\.html\?t=47\.5/);
  await page.waitForFunction(() => document.documentElement.dataset.restoredSessionTime !== undefined);
  expect(await page.evaluate(() => currentTime)).toBeCloseTo(47.5, 1);
  for (const other of ['pd_insighter_manual_labeling.html', 'pd_insighter_stride_review.html', 'pd_insighter_parkinson_review.html']) {
    await page.goto(`Parki/${other}`);
    await expect(page.locator('a.nav-tab[data-view="symptoms"]')).toHaveCount(1);
  }
  expect(problems).toEqual([]);
});

test('videos play together on the session clock', async ({page}) => {
  const problems = watch(page);
  await open(page, '?t=100');
  await page.waitForFunction(() => document.getElementById('downwardVideo').readyState >= 1);
  await page.locator('#playBtn').click();
  await page.waitForTimeout(1500);
  await page.locator('#playBtn').click();
  await page.waitForTimeout(300);
  const t = await page.evaluate(() => ({aria: ariaVideo.currentTime, down: downwardVideo.currentTime, session: Number(document.documentElement.dataset.sessionTime)}));
  expect(t.aria).toBeGreaterThan(100.5);
  expect(Math.abs(t.aria - t.down)).toBeLessThan(0.1);
  expect(Math.abs(t.session - t.aria)).toBeLessThan(0.05);
  await goTo(page, '2:30.5');
  await page.waitForFunction(() => Math.abs(Number(document.documentElement.dataset.sessionTime) - 150.5) < 0.01);
  await page.locator('#forwardFrame').click();
  await page.waitForFunction(() => Math.abs(Number(document.documentElement.dataset.sessionTime) - (Math.round(150.5 * 30) + 1) / 30) < 0.005);
  expect(problems).toEqual([]);
});

test('timeline windows move and follow the video', async ({page}) => {
  await open(page, '?t=200');
  await page.locator('#windowSelect').selectOption('30');
  let v = await view(page);
  expect(v.label).toBe('3:05.0 - 3:35.0');
  await page.locator('#nextWindow').click();
  v = await view(page);
  expect(v.label).toBe('3:32.0 - 4:02.0');
  await page.locator('#centerWindow').click();
  expect((await view(page)).label).toBe('3:05.0 - 3:35.0');
  await page.locator('#windowSelect').selectOption('full');
  expect((await view(page)).label).toBe('0:00.0 - 8:10.0');
});

test('create, reload, edit and delete an annotation', async ({page}) => {
  const problems = watch(page);
  await open(page, '?t=60');
  await page.locator('#windowSelect').selectOption('60');
  await dragSymptoms(page, 0.2, 0.45);
  await expect(page.locator('#editorForm')).toBeVisible();
  await page.locator('#startInput').fill('0:40.250');
  await page.locator('#startInput').press('Tab');
  await page.locator('#endInput').fill('0:52.5');
  await page.locator('#endInput').press('Tab');
  await page.locator('.label-choice', {hasText: 'Simulated Symptom'}).click();
  await page.locator('#confidenceChoices button', {hasText: 'High'}).click();
  await page.locator('#notesInput').fill('Short steps near the door');
  await expect(page.locator('#saveStatus')).toHaveText('Unsaved selection');
  await page.locator('#saveBtn').click();
  await expect(page.locator('#saveStatus')).toHaveText('Annotation saved');
  await expect(page.locator('#annotationRows tr')).toHaveCount(1);
  await expect(page.locator('#symptomTrack .annotation')).toHaveCount(1);

  await page.reload();
  await page.waitForFunction(() => document.documentElement.dataset.saveState === 'saved');
  const row = page.locator('#annotationRows tr').first();
  await expect(row).toContainText('Simulated Symptom');
  await expect(row).toContainText('0:40.25');
  await expect(row).toContainText('0:52.50');

  await row.click();
  await expect(page.locator('#editorTitle')).toHaveText('Edit annotation');
  await expect(page.locator('#notesInput')).toHaveValue('Short steps near the door');
  await page.locator('#endInput').fill('0:55');
  await page.locator('#endInput').press('Tab');
  await page.locator('#statusSelect').selectOption('needs_review');
  await page.locator('#saveBtn').click();
  await expect(page.locator('#saveStatus')).toHaveText('Changes saved');
  await page.reload();
  await page.waitForFunction(() => document.documentElement.dataset.saveState === 'saved');
  await expect(page.locator('#annotationRows tr').first()).toContainText('0:55.00');
  await expect(page.locator('#annotationRows tr').first()).toContainText('Needs review');

  page.once('dialog', dialog => dialog.accept());
  await page.locator('#annotationRows .row-menu-btn').first().click();
  await expect(page.locator('#rowMenu')).toBeVisible();
  await page.locator('#rowMenu [data-action="delete"]').click();
  await expect(page.locator('#saveStatus')).toHaveText('Annotation deleted');
  await page.reload();
  await page.waitForFunction(() => document.documentElement.dataset.saveState === 'saved');
  await expect(page.locator('#listEmpty')).toBeVisible();
  expect(problems).toEqual([]);
});

test('marking with the playhead, overlapping intervals and the next unreviewed time', async ({page}) => {
  await open(page, '?t=10');
  await page.locator('#windowSelect').selectOption('60');
  await page.keyboard.press('i');
  await goTo(page, '25');
  await page.keyboard.press('o');
  await expect(page.locator('#startInput')).toHaveValue('0:10.000');
  await expect(page.locator('#endInput')).toHaveValue('0:25.000');
  await page.locator('.label-choice', {hasText: 'Uncertain'}).click();
  await page.locator('#saveBtn').click();
  await expect(page.locator('#saveStatus')).toHaveText('Annotation saved');

  // A second, overlapping interval with a different label is kept as entered.
  await dragSymptoms(page, 0.05, 0.3);
  await page.locator('#startInput').fill('20'); await page.locator('#startInput').press('Tab');
  await page.locator('#endInput').fill('30'); await page.locator('#endInput').press('Tab');
  await page.locator('.label-choice', {hasText: 'Not Observable'}).click();
  await page.keyboard.press('Enter');
  await expect(page.locator('#annotationRows tr')).toHaveCount(2);
  const tops = await page.locator('#symptomTrack .annotation').evaluateAll(nodes => nodes.map(n => n.style.top));
  expect(new Set(tops).size).toBe(2);

  await goTo(page, '0');
  await page.locator('#nextUnreviewed').click();
  await page.waitForFunction(() => Math.abs(Number(document.documentElement.dataset.sessionTime) - 30) < 0.05);
  await expect(page.locator('#reviewedFact')).toHaveText(/^\d+% reviewed$/);
});

test('custom labels: add, use, rename and archive without breaking annotations', async ({page}) => {
  await open(page, '?t=300');
  await page.locator('#manageLabelsBtn').click();
  await page.locator('#addLabelBtn').click();
  const row = page.locator('.label-row').last();
  await row.locator('input[type="text"]').first().fill('Shuffling (provisional)');
  await row.locator('input[type="color"]').evaluate(input => { input.value = '#8e44ad'; input.dispatchEvent(new Event('input')); });
  await row.locator('input[type="text"]').nth(1).fill('Short, dragging steps');
  await page.locator('#labelSaveBtn').click();
  await expect(page.locator('#labelDialog')).not.toBeVisible();
  await expect(page.locator('#legend')).toContainText('Shuffling (provisional)');

  await dragSymptoms(page, 0.4, 0.6);
  await page.locator('.label-choice', {hasText: 'Shuffling (provisional)'}).click();
  await page.locator('#saveBtn').click();
  await expect(page.locator('#saveStatus')).toHaveText('Annotation saved');

  // Duplicate names are refused.
  await page.locator('#manageLabelsBtn').click();
  await page.locator('#addLabelBtn').click();
  await page.locator('.label-row').last().locator('input[type="text"]').first().fill('uncertain');
  await page.locator('#labelSaveBtn').click();
  await expect(page.locator('#labelError')).toContainText('already exists');
  await page.locator('#labelCancelBtn').click();

  // Rename and archive; the saved annotation keeps its label.
  await page.locator('#manageLabelsBtn').click();
  const shuffling = page.locator('.label-row').filter({has: page.locator('input[value="Shuffling (provisional)"]')});
  await shuffling.locator('input[type="text"]').first().fill('Shuffling gait');
  await shuffling.locator('button', {hasText: 'Archive'}).click();
  await page.locator('#labelSaveBtn').click();
  await page.reload();
  await page.waitForFunction(() => document.documentElement.dataset.saveState === 'saved');
  await expect(page.locator('#annotationRows')).toContainText('Shuffling gait');
  await expect(page.locator('#legend')).toContainText('Shuffling gait (archived)');
  await dragSymptoms(page, 0.7, 0.8);
  await expect(page.locator('.label-choice', {hasText: 'Shuffling gait'})).toHaveCount(0);
  await page.keyboard.press('Escape');
});

test('failed saves keep the work and say so', async ({page}) => {
  await open(page, '?t=400');
  await page.route('**/api/symptom-annotations', route => route.request().method() === 'POST'
    ? route.fulfill({status: 500, contentType: 'application/json', body: JSON.stringify({error: 'Disk full'})})
    : route.continue());
  await dragSymptoms(page, 0.3, 0.5);
  await page.locator('.label-choice', {hasText: 'Uncertain'}).click();
  await page.locator('#notesInput').fill('keep me');
  await page.locator('#saveBtn').click();
  await expect(page.locator('#saveStatus')).toHaveText('Not saved');
  await expect(page.locator('#formError')).toContainText('Disk full');
  await expect(page.locator('#notesInput')).toHaveValue('keep me');
});

test('a missing video shows a clear message and the other camera still works', async ({page}) => {
  await page.route('**/aria_session_video.mp4*', route => route.abort());
  await page.goto(`${PAGE}?t=50`);
  await expect(page.locator('[data-video="aria"] .video-empty')).toContainText('could not be loaded');
  await page.waitForFunction(() => document.getElementById('downwardVideo').readyState >= 1);
  await goTo(page, '61');
  await page.waitForFunction(() => Math.abs(downwardVideo.currentTime - 61) < 0.1);
});

test('test annotations go to the separate store, never into the trial data', async () => {
  const runs = process.env.PD_INSIGHTER_E2E_RUNS;
  expect(existsSync(path.join(runs, 'symptom_annotation_labels.json'))).toBe(false);
  for (const trial of readdirSync(runs)) expect(existsSync(path.join(runs, trial, 'symptom_annotations'))).toBe(false);
  expect(existsSync(path.join(process.env.PD_INSIGHTER_E2E_STORE, 'Parki', 'symptom_annotations', 'annotations.json'))).toBe(true);
});

test('activity row: automatic, manual and hidden', async ({page}) => {
  const problems = watch(page);
  await page.addInitScript(() => {
    if (!sessionStorage.getItem('activity-reset')) {
      localStorage.removeItem('pd-insighter-symptom-activity-source');
      sessionStorage.setItem('activity-reset', '1');
    }
  });
  await open(page, '?t=60');
  await page.locator('#windowSelect').selectOption('full');
  const saved = await page.evaluate(async () => (await fetch('session_dashboard_manifest.json')).json());
  const automatic = saved.continuous_model_segments || saved.model_segments;
  await expect(page.locator('#activitySource')).toHaveValue('automatic');
  await expect(page.locator('#activityTrack .seg')).toHaveCount(automatic.length);
  await expect(page.locator('#activityEmpty')).toBeHidden();

  // Parki has no saved manual activity labels: say so instead of showing automatic ones.
  await page.locator('#activitySource').selectOption('manual');
  await expect(page.locator('#activityEmpty')).toHaveText('No manual activity labels for this trial.');
  await expect(page.locator('#activityTrack .seg')).toHaveCount(0);

  await page.locator('#activitySource').selectOption('hide');
  await expect(page.locator('#activityTrack')).toBeHidden();
  await expect(page.locator('#overviewTrack .seg.activity')).toHaveCount(0);
  await page.reload();
  await page.waitForFunction(() => document.documentElement.dataset.saveState === 'saved');
  await expect(page.locator('#activitySource')).toHaveValue('hide');
  await page.locator('#activitySource').selectOption('automatic');
  expect(problems).toEqual([]);
});

test('manual activity labels are read from the existing store, unchanged', async ({page}) => {
  const requests = [];
  await page.route('**/api/session-manual-segments?*', route => {
    requests.push(route.request().method());
    return route.fulfill({status: 200, contentType: 'application/json', body: JSON.stringify({trial_id: 'Parki', version: 1, segments: [
      {id: 's1', start_s: 10.25, end_s: 30.5, motion_label: 'walking', confidence: 'high'},
      {id: 's2', start_s: 30.5, end_s: 42, motion_label: 'standing', confidence: 'medium'},
    ]})});
  });
  await page.addInitScript(() => localStorage.setItem('pd-insighter-symptom-activity-source', 'manual'));
  await open(page, '?t=20');
  await page.locator('#windowSelect').selectOption('60');
  await expect(page.locator('#activityTrack .seg.manual')).toHaveCount(2);
  const first = page.locator('#activityTrack .seg.manual').first();
  await expect(first).toHaveAttribute('title', /Manual: walking 0:10\.25 - 0:30\.50 \(high\)/);
  await expect(page.locator('#legend')).toContainText('walking');
  expect(requests).toEqual(['GET']);  // read only; nothing is written back
  await page.evaluate(() => localStorage.setItem('pd-insighter-symptom-activity-source', 'automatic'));
});

test('settings menu holds the annotator, options and shortcuts', async ({page}) => {
  await open(page);
  await expect(page.locator('#settingsPanel')).toBeHidden();
  await page.locator('#settingsBtn').click();
  await expect(page.locator('#settingsPanel')).toBeVisible();
  await expect(page.locator('#settingsPanel')).toContainText('Mark start / end');
  await page.locator('#annotatorInput').fill('AB');
  await page.locator('#annotatorInput').press('Tab');
  await page.keyboard.press('Escape');
  await expect(page.locator('#settingsPanel')).toBeHidden();
  await page.reload();
  await page.waitForFunction(() => document.documentElement.dataset.saveState === 'saved');
  await expect(page.locator('#annotatorInput')).toHaveValue('AB');
});

test('attached downward footage: recorded window, padding note and upright rotation', async ({page}) => {
  const problems = watch(page);
  await page.goto('E2E_Attached/pd_insighter_symptom_annotation.html?t=10');
  await page.waitForFunction(() => document.getElementById('downwardVideo').readyState >= 1 && document.documentElement.dataset.saveState === 'saved');
  const panel = page.locator('[data-video="downward"]');
  await expect(panel.locator('.video-tag')).toHaveText('Downward \u00b7 0:20.0-2:00.0');
  await expect(panel.locator('.video-note')).toHaveText('No downward recording at this time.');
  await expect(panel).toHaveAttribute('data-coverage', 'none');
  expect(await page.locator('#downwardVideo').evaluate(v => v.style.transform)).toBe('rotate(0deg)');
  await goTo(page, '1:00');
  await expect(panel).toHaveAttribute('data-coverage', 'recorded');
  await expect(panel.locator('.video-note')).toBeHidden();
  // Times stay on the session clock: the downward video sits at the same time as Aria.
  await page.waitForFunction(() => Math.abs(downwardVideo.currentTime - ariaVideo.currentTime) < 0.05);
  await goTo(page, '2:30');
  await expect(panel).toHaveAttribute('data-coverage', 'none');
  await page.locator('#settingsBtn').click();
  await expect(page.locator('#syncDetails')).toContainText('padding');
  expect(problems).toEqual([]);
});

test('older dual-view trials keep the 180-degree downward mount', async ({page}) => {
  await open(page, '?t=30');
  await page.waitForFunction(() => document.getElementById('downwardVideo').readyState >= 1);
  expect(await page.locator('#downwardVideo').evaluate(v => v.style.transform)).toBe('rotate(180deg)');
  await expect(page.locator('[data-video="downward"] .video-tag')).toHaveText('Downward');
});

test('Aria-only trials stay fully usable', async ({page}) => {
  const problems = watch(page);
  await page.goto('E2E_AriaOnly/pd_insighter_symptom_annotation.html?t=30');
  await page.waitForFunction(() => document.getElementById('ariaVideo').readyState >= 1 && document.documentElement.dataset.saveState === 'saved');
  await expect(page.locator('[data-video="downward"] .video-empty')).toHaveText('No downward camera recording for this trial.');
  await expect(page.locator('#syncStatus')).toBeHidden();
  await dragSymptoms(page, 0.3, 0.5);
  await page.locator('.label-choice', {hasText: 'Not Observable'}).click();
  await page.locator('#saveBtn').click();
  await expect(page.locator('#saveStatus')).toHaveText('Annotation saved');
  await page.reload();
  await page.waitForFunction(() => document.documentElement.dataset.saveState === 'saved');
  await expect(page.locator('#annotationRows tr')).toHaveCount(1);
  expect(problems).toEqual([]);
});

test('audio uses the shared PD-Insighter control: one source, off by default', async ({page}) => {
  await page.addInitScript(() => localStorage.removeItem('pd-insighter-audio-enabled-v1'));
  await open(page, '?t=40');
  const toggle = page.locator('#audioToggle');
  await expect(toggle).toHaveClass(/pd-audio-toggle/);
  await expect(page.locator('.pd-audio-toggle')).toHaveCount(1);
  await expect(toggle).toHaveText(/Audio Off|No audio track/);
  expect(await page.evaluate(() => ariaVideo.muted && downwardVideo.muted)).toBe(true);
  if ((await toggle.textContent()) === 'Audio Off') {
    await toggle.click();
    await expect(toggle).toHaveText('Audio On');
    expect(await page.evaluate(() => [ariaVideo.muted, downwardVideo.muted])).toEqual([false, true]);
    // Space on the focused button toggles audio, not playback.
    await toggle.press('Space');
    expect(await page.evaluate(() => ariaVideo.paused)).toBe(true);
  }
});

test('the trial library links the Symptom Annotation page', async ({page}) => {
  await page.goto('pd_insighter_trial_selector.html');
  const link = page.locator('.trial-card[data-trial-id="Parki"] a', {hasText: 'Symptom Annotation'});
  await expect(link).toHaveAttribute('href', /^Parki\/pd_insighter_symptom_annotation\.html\?v=\d+$/);
  await link.click();
  await expect(page.locator('span.nav-tab.active')).toHaveText('Symptom Annotation');
});
