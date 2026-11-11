// Symptom Annotation page: synchronized video, timeline, editor, saved list and label manager.
// Data: /api/symptom-labels (shared catalogue), /api/symptom-annotations?trial_id=... and, when a detector
// run is saved for the trial, /api/symptom-detections?trial_id=... (read-only predictions + review decisions).
import {
  WINDOW_SIZES, CONFIDENCE, REVIEW_STATUS, clamp, formatTime, parseTime, videoToSession, sessionToVideo,
  videoCovers, inCoverage, windowAround, shiftWindow, assignLanes, unreviewedIntervals, nextUnreviewed, reviewedFraction,
  validateDraft, labelNameProblems, randomId, upsert, tickStep,
  SYMPTOM_LAYERS, DECISION_LABELS, STRENGTH_LABELS, reviewOf, shownPredictions, predictionProgress, stepPrediction,
  matchingLabel, annotationFromPrediction, annotationOfPrediction,
} from './symptom_annotation_core.mjs';

const $ = id => document.getElementById(id);
const data = JSON.parse($('symptomPageData').textContent);
const trialId = data.trial_id;
const duration = Number(data.session_duration_s);
const fps = Number(data.session_fps) || 30;
const LANE_HEIGHT = 34;
const PREDICTION_LANE_HEIGHT = 24;
const SOFT_SYNC_S = 0.03;
const HARD_SYNC_S = 0.15;
const ACTIVITY_COLORS = {sitting: '#e6ab02', standing: '#7570b3', walking: '#d95f02', upstairs: '#1b9e77', downstairs: '#666666'};

const state = {
  labels: [], labelsRevision: 0,
  annotations: [], revision: 0,
  view: {start: 0, end: duration},
  selection: null,          // {start_s, end_s} while choosing a new interval
  editingId: null,          // id of the saved annotation being edited
  draft: null,              // editor contents
  dirty: false,
  saving: false,
  time: 0,
  playing: false,
  activitySource: 'automatic',  // automatic | manual | hide
  manualActivity: null,         // null = not loaded; {segments} or {error}
  layers: 'both',               // both | manual | automatic (which symptom layers are drawn)
  predictions: {runs: [], run: null, reviews: null},   // detector output is read-only here
  predictionId: null,           // prediction shown in the review panel
  acceptingId: null,            // prediction being adjusted in the editor before it is accepted
  showLowPredictions: true,
  playUntil: null,              // session time at which a replay stops
};

/** Activity segments for the chosen source, or null when that source has nothing to show. */
function activitySegments() {
  if (state.activitySource === 'automatic') return data.activity_segments || [];
  if (state.activitySource === 'manual') return state.manualActivity?.segments || [];
  return [];
}

async function loadManualActivity() {
  if (state.manualActivity || !data.manual_activity_api) return;
  try {
    const body = await api(`${data.manual_activity_api}?trial_id=${encodeURIComponent(trialId)}`);
    const segments = (Array.isArray(body?.segments) ? body.segments : []).map(seg => ({
      start_s: Number(seg.start_s), end_s: Number(seg.end_s),
      label: String(seg.motion_label || seg.label || '').toLowerCase(), confidence: seg.confidence || '',
    })).filter(seg => Number.isFinite(seg.start_s) && Number.isFinite(seg.end_s) && seg.end_s > seg.start_s);
    state.manualActivity = {segments};
  } catch (error) {
    state.manualActivity = {segments: [], error: error.message};
  }
}

async function setActivitySource(source) {
  state.activitySource = source;
  $('activitySource').value = source;
  try { localStorage.setItem('pd-insighter-symptom-activity-source', source); } catch { /* optional */ }
  if (source === 'manual' && !state.manualActivity) {
    $('activityEmpty').hidden = false; $('activityEmpty').textContent = 'Loading manual activity labels';
    await loadManualActivity();
  }
  document.documentElement.dataset.activitySource = source;
  renderTimeline(); renderLegend();
}

// ---- automatic predictions: state helpers ------------------------------------------------------------

const predictionRun = () => state.predictions.run;
const predictionsOn = () => !!predictionRun() && state.layers !== 'manual';
const manualOn = () => state.layers !== 'automatic';
const predictionList = () => (predictionsOn() ? shownPredictions(predictionRun(), {showLow: state.showLowPredictions}) : []);
const predictionById = id => (predictionRun()?.detections || []).find(item => item.id === id) || null;
const predictionLabel = id => (predictionRun()?.labels || []).find(label => label.id === id) || {id, name: id, color: '#555555', description: ''};
const decisionOf = id => reviewOf(state.predictions.reviews, predictionRun()?.run_id, id)?.decision || '';

// ---- small helpers -----------------------------------------------------------------------------

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === 'class') node.className = value;
    else if (key === 'style') Object.assign(node.style, value);
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (value !== undefined && value !== null && value !== false) node.setAttribute(key, value === true ? '' : value);
  }
  node.append(...children.filter(child => child !== null && child !== undefined));
  return node;
}
const labelById = id => state.labels.find(label => label.id === id);
const activeLabels = () => state.labels.filter(label => !label.archived);
const isTyping = target => target?.isContentEditable || ['TEXTAREA', 'SELECT'].includes(target?.tagName)
  || (target?.tagName === 'INPUT' && !['checkbox', 'button', 'range', 'color'].includes(target.type));

function setStatus(kind, text) {
  const status = $('saveStatus');
  status.className = `save-status ${kind}`;
  status.textContent = text;
  document.documentElement.dataset.saveState = kind;
}

async function api(path, options = {}) {
  const response = await fetch(path, {cache: 'no-store', headers: {'Content-Type': 'application/json'}, ...options});
  let body = null;
  try { body = await response.json(); } catch { /* handled below */ }
  if (!response.ok) {
    const error = new Error(body?.error || `Request failed (HTTP ${response.status}).`);
    error.status = response.status; error.body = body;
    throw error;
  }
  return body;
}

// ---- video -------------------------------------------------------------------------------------

const videos = (data.videos || []).map(spec => ({
  ...spec, el: $(`${spec.key}Video`), panel: document.querySelector(`[data-video="${spec.key}"]`),
  turns: spec.turns || 0, ready: false, failed: !spec.available,
}));
let master = null;

function applyRotation(video) {
  const n = ((video.turns % 4) + 4) % 4;
  const node = video.el;
  node.style.transform = n % 2 ? `translate(-50%, -50%) rotate(${n * 90}deg)` : `rotate(${n * 90}deg)`;
  node.classList.toggle('rotated-quarter', n % 2 === 1);
  if (n % 2) {
    const rect = node.parentElement.getBoundingClientRect();
    const w = node.videoWidth || rect.height, h = node.videoHeight || rect.width;
    const scale = Math.min(rect.width / h, rect.height / w);
    node.style.setProperty('--rotated-video-width', `${w * scale}px`);
    node.style.setProperty('--rotated-video-height', `${h * scale}px`);
  }
}

function showVideoProblem(video, text) {
  video.failed = true;
  video.el.hidden = true;
  const empty = video.panel.querySelector('.video-empty');
  empty.hidden = false; empty.textContent = text;
  video.panel.querySelector('[data-rotate]').disabled = true;
}

function chooseMaster() {
  master = videos.find(video => !video.failed) || null;
  const usable = videos.filter(video => !video.failed);
  $('syncStatus').hidden = usable.length < 2;
  for (const id of ['playBtn', 'back5', 'forward5', 'backFrame', 'forwardFrame']) $(id).disabled = !master;
  if (!master) setStatus('error', 'No video available');
}

function setupVideos() {
  for (const video of videos) {
    if (video.failed) {
      showVideoProblem(video, video.missing === 'not_recorded'
        ? `No ${video.title.toLowerCase()} recording for this trial.`
        : `The ${video.title.toLowerCase()} video is not available for this trial.`);
      continue;
    }
    if (video.coverage_s) {
      const tag = video.panel.querySelector('.video-tag');
      tag.textContent += ` \u00b7 ${formatTime(video.coverage_s[0], 1)}-${formatTime(video.coverage_s[1], 1)}`;
      tag.title = 'Recorded footage on the session clock; outside this the video is padding.';
    }
    video.el.src = video.uri;
    video.el.addEventListener('loadedmetadata', () => { video.ready = true; applyRotation(video); syncFollowers(true); });
    video.el.addEventListener('error', () => { showVideoProblem(video, `The ${video.title.toLowerCase()} video could not be loaded.`); chooseMaster(); });
    video.panel.querySelector('[data-rotate]').addEventListener('click', event => { event.stopPropagation(); video.turns += 1; applyRotation(video); });
  }
  chooseMaster();
  window.addEventListener('resize', () => videos.forEach(video => !video.failed && applyRotation(video)));
  for (const video of videos) {
    video.el.addEventListener('play', () => { if (video === master) { state.playing = true; updatePlayButton(); startLoop(); } });
    video.el.addEventListener('pause', () => { if (video === master) { state.playing = false; updatePlayButton(); pauseFollowers(); tick(); } });
    video.el.addEventListener('seeked', () => { if (video === master) tick(); });
    video.el.addEventListener('ended', () => { if (video === master) pause(); });
  }
}

function sessionTime() {
  if (!master) return state.time;
  return clamp(videoToSession(master.el.currentTime || 0, master.offset_s), 0, duration);
}

/** Keeps every other video on the master's session time; never fakes footage it does not have. */
function syncFollowers(force = false) {
  const t = sessionTime();
  let worst = 0;
  for (const video of videos) {
    if (video === master || video.failed || !video.ready) continue;
    const covers = videoCovers(t, video.offset_s, video.duration_s) && inCoverage(t, video.coverage_s);
    const note = video.panel.querySelector('.video-note');
    note.hidden = covers;
    video.panel.dataset.coverage = covers ? 'recorded' : 'none';
    if (!covers) {
      note.textContent = video.coverage_s ? 'No downward recording at this time.' : 'No footage from this camera at this time.';
      // A padded video keeps playing in sync (its padding is black); a shorter one waits.
      if (!video.coverage_s) { video.el.pause(); continue; }
    }
    const target = sessionToVideo(t, video.offset_s);
    const delta = (video.el.currentTime || 0) - target;
    const rate = master.el.playbackRate || 1;
    if (force || Math.abs(delta) > HARD_SYNC_S || master.el.paused) {
      if (Math.abs(delta) > 0.5 / fps || force) video.el.currentTime = clamp(target, 0, video.duration_s);
      video.el.playbackRate = rate;
      worst = Math.max(worst, Math.abs((video.el.currentTime || 0) - target));
      if (!master.el.paused && video.el.paused) video.el.play().catch(() => {});
      continue;
    }
    worst = Math.max(worst, Math.abs(delta));
    if (Math.abs(delta) > SOFT_SYNC_S) {
      video.el.playbackRate = rate * (1 + clamp(-delta * 0.5, -0.06, 0.06));
    } else video.el.playbackRate = rate;
    if (!master.el.paused && video.el.paused) video.el.play().catch(() => {});
  }
  const sync = $('syncStatus');
  if (!sync.hidden) {
    const ms = Math.round(worst * 1000);
    const good = worst <= SOFT_SYNC_S + 0.01;
    sync.className = `sync-dot ${good ? 'good' : 'adjusting'}`;
    sync.textContent = good ? 'In sync' : `Syncing ${ms} ms`;
    sync.title = 'Downward video follows the Aria video on the session clock';
  }
}

function pauseFollowers() {
  for (const video of videos) if (video !== master && !video.failed) video.el.pause();
  syncFollowers(true);
}

async function play() {
  if (!master) return;
  if (sessionTime() >= duration - 0.05) seek(0);
  try { await master.el.play(); syncFollowers(true); }
  catch { setStatus('pending', 'Click Play again to start the video'); }
}
function pause() { if (master) master.el.pause(); }
function togglePlay() { state.playUntil = null; if (state.playing) pause(); else play(); }
function updatePlayButton() { $('playBtn').textContent = state.playing ? 'Pause' : 'Play'; }

function seek(t) {
  const target = clamp(t, 0, duration);
  state.time = target;
  if (master) master.el.currentTime = clamp(sessionToVideo(target, master.offset_s), 0, master.duration_s);
  syncFollowers(true);
  if ($('followVideo').checked) keepInView(target);
  renderTime();
}

let loopId = null;
function startLoop() {
  if (loopId !== null) return;
  const step = () => {
    loopId = null;
    tick();
    if (state.playing) loopId = requestAnimationFrame(step);
  };
  loopId = requestAnimationFrame(step);
}
function tick() {
  state.time = sessionTime();
  if (state.playUntil !== null && state.time >= state.playUntil) { state.playUntil = null; pause(); }
  if (state.playing) syncFollowers(false);
  if ($('followVideo').checked) keepInView(state.time);
  renderTime();
}

// ---- timeline geometry ---------------------------------------------------------------------------

const span = () => state.view.end - state.view.start;
const pct = t => `${(100 * (t - state.view.start)) / span()}%`;
const width = (a, b) => `${(100 * (b - a)) / span()}%`;
const fullPct = t => `${(100 * t) / duration}%`;

function windowSeconds() {
  const option = WINDOW_SIZES.find(item => item.value === $('windowSelect').value);
  return option ? option.seconds : Infinity;
}
function setView(view) {
  state.view = view;
  renderTimeline();
}
function keepInView(t) {
  if (t < state.view.start || t > state.view.end) setView(windowAround(t + span() * 0.4, span(), duration));
}
function timeAt(event, track, full = false) {
  const rect = track.getBoundingClientRect();
  const ratio = clamp((event.clientX - rect.left) / Math.max(1, rect.width), 0, 1);
  return full ? ratio * duration : state.view.start + ratio * span();
}

// ---- rendering ---------------------------------------------------------------------------------

function renderTime() {
  $('clockNow').textContent = formatTime(state.time);
  for (const head of document.querySelectorAll('.playhead')) {
    const full = head.dataset.full === '1';
    const visible = full || (state.time >= state.view.start && state.time <= state.view.end);
    head.hidden = !visible;
    if (visible) head.style.left = full ? fullPct(state.time) : pct(state.time);
  }
  document.documentElement.dataset.sessionTime = state.time.toFixed(3);
}

function playhead(full = false) { return el('div', {class: 'playhead', dataset: {full: full ? '1' : '0'}}); }

function visibleAnnotations() {
  return state.annotations.filter(item => item.end_s > state.view.start && item.start_s < state.view.end);
}

function renderTimeline() {
  const {start, end} = state.view;
  $('windowLabel').textContent = `${formatTime(start, 1)} - ${formatTime(end, 1)}`;

  const overview = $('overviewTrack');
  const activity_ = activitySegments();
  overview.replaceChildren(
    ...activity_.map(seg => el('div', {class: 'seg activity', style: {
      left: fullPct(seg.start_s), width: `${(100 * (seg.end_s - seg.start_s)) / duration}%`, background: ACTIVITY_COLORS[seg.label] || '#9aa39d'}})),
    ...(manualOn() ? state.annotations : []).map(item => el('div', {class: 'seg symptom', style: {
      left: fullPct(item.start_s), width: `${(100 * (item.end_s - item.start_s)) / duration}%`, background: labelById(item.label_id)?.color || '#888'}})),
    ...predictionList().map(item => el('div', {class: 'seg prediction-mark', style: {
      left: fullPct(item.start_s), width: `${(100 * (item.end_s - item.start_s)) / duration}%`, background: predictionLabel(item.label_id).color}})),
    el('div', {class: 'overview-window', style: {left: fullPct(start), width: `${(100 * (end - start)) / duration}%`}}),
    playhead(true),
  );

  const activity = $('activityTrack');
  const empty = $('activityEmpty');
  activity.hidden = state.activitySource === 'hide';
  empty.hidden = activity_.length > 0;
  empty.textContent = state.activitySource === 'manual'
    ? (state.manualActivity?.error ? 'Manual activity labels could not be loaded.' : 'No manual activity labels for this trial.')
    : 'No automatic classification for this trial.';
  const manual = state.activitySource === 'manual';
  activity.replaceChildren(
    empty,
    ...activity_.filter(seg => seg.end_s > start && seg.start_s < end).map(seg => el('div', {
      class: manual ? 'seg manual' : 'seg', dataset: {source: state.activitySource},
      title: `${manual ? 'Manual' : 'Automatic'}: ${seg.label} ${formatTime(seg.start_s)} - ${formatTime(seg.end_s)}${seg.confidence ? ` (${seg.confidence})` : ''}`,
      style: {left: pct(Math.max(seg.start_s, start)), width: width(Math.max(seg.start_s, start), Math.min(seg.end_s, end)), background: ACTIVITY_COLORS[seg.label] || '#9aa39d'}})),
    playhead(),
  );

  const symptoms = $('symptomTrack');
  const showManual = manualOn();
  const {lanes, count} = assignLanes(showManual ? state.annotations : []);
  const predictions = predictionList();
  const predictionLanes = assignLanes(predictions);
  // The prediction lane keeps its height while predictions are shown, so the track does not jump while scrolling.
  const lanesTop = predictionsOn() ? predictionLanes.count * PREDICTION_LANE_HEIGHT + 6 : 0;
  symptoms.style.height = `${Math.max(46, lanesTop + (showManual ? count * LANE_HEIGHT : 0) + 12)}px`;
  symptoms.dataset.layers = state.layers;
  const items = [];
  const trackWidth = symptoms.clientWidth || 1000;
  if (lanesTop) items.push(el('div', {class: 'prediction-lane', style: {height: `${lanesTop}px`}}));
  const open = predictionsOn() ? predictionById(state.predictionId) : null;
  if (open && open.end_s > start && open.start_s < end) {
    const a = Math.max(open.start_s, start), b = Math.min(open.end_s, end);
    items.push(el('div', {class: 'prediction-range', style: {left: pct(a), width: width(a, b)}}));
  }
  for (const item of predictions) {
    if (item.end_s <= start || item.start_s >= end) continue;
    const label = predictionLabel(item.label_id);
    const decision = decisionOf(item.id);
    const a = Math.max(item.start_s, start), b = Math.min(item.end_s, end);
    const chip = el('div', {
      class: `prediction${item.id === state.predictionId ? ' active' : ''}`,
      dataset: {predictionId: item.id, strength: item.evidence_strength, decision},
      title: `Automatic prediction: ${label.name}, ${formatTime(item.start_s)} - ${formatTime(item.end_s)}\n`
        + `Score ${item.score} (uncalibrated), ${STRENGTH_LABELS[item.evidence_strength] || item.evidence_strength} evidence`
        + (decision ? `\n${DECISION_LABELS[decision]}` : ''),
      style: {left: pct(a), width: width(a, b), top: `${3 + predictionLanes.lanes.get(item.id) * PREDICTION_LANE_HEIGHT}px`},
    });
    const room = ((b - a) / span()) * trackWidth;
    if (room >= 96) chip.append(label.name, el('span', {class: 'score'}, String(item.score)));
    else if (room >= 28) chip.append(el('span', {class: 'score solo'}, String(item.score)));
    chip.style.setProperty('--pred', label.color);
    items.push(chip);
  }
  for (const gap of showManual ? unreviewedIntervals(state.annotations, duration) : []) {
    if (gap.end_s <= start || gap.start_s >= end) continue;
    const a = Math.max(gap.start_s, start), b = Math.min(gap.end_s, end);
    items.push(el('div', {class: 'unreviewed', title: `Unreviewed: ${formatTime(gap.start_s)} - ${formatTime(gap.end_s)}`,
      style: {left: pct(a), width: width(a, b), top: `${lanesTop}px`}}));
  }
  for (const item of showManual ? visibleAnnotations() : []) {
    const label = labelById(item.label_id);
    const a = Math.max(item.start_s, start), b = Math.min(item.end_s, end);
    items.push(el('div', {
      class: `annotation${item.id === state.editingId ? ' active' : ''}${item.review_status === 'needs_review' ? ' needs-review' : ''}`,
      dataset: {id: item.id}, title: `${label?.name || 'Unknown label'}: ${formatTime(item.start_s)} - ${formatTime(item.end_s)}${item.notes ? `\n${item.notes}` : ''}`,
      style: {left: pct(a), width: width(a, b), top: `${lanesTop + 6 + lanes.get(item.id) * LANE_HEIGHT}px`, background: label?.color || '#888'},
    }, label?.name || 'Unknown label'));
  }
  const range = state.selection || (state.editingId && state.draft ? state.draft : null);
  if (range && Number.isFinite(range.start_s) && Number.isFinite(range.end_s) && range.end_s > start && range.start_s < end) {
    const a = Math.max(range.start_s, start), b = Math.min(range.end_s, end);
    items.push(el('div', {class: 'selection', style: {left: pct(a), width: width(a, b)}}));
  }
  items.push(playhead());
  symptoms.replaceChildren(...items);

  const axis = $('axis');
  const step = tickStep(span());
  const ticks = [];
  for (let t = Math.ceil(start / step) * step; t <= end + 1e-9; t += step) {
    ticks.push(el('span', {style: {left: pct(t)}}, formatTime(t, step < 1 ? 1 : 0).replace(/\.0+$/, '')));
  }
  axis.replaceChildren(...ticks);
  renderTime();
}

function renderLegend() {
  const legend = $('legend');
  const used = new Set(state.annotations.map(item => item.label_id));
  const shown = state.labels.filter(label => !label.archived || used.has(label.id));
  const activityShown = new Set(activitySegments().map(seg => seg.label));
  legend.replaceChildren(
    ...shown.map(label => el('span', {class: 'item', title: label.description || ''},
      el('i', {class: 'swatch', style: {background: label.color}}), label.name + (label.archived ? ' (archived)' : ''))),
    el('span', {class: 'item'}, el('i', {class: 'swatch hatch'}), 'Unreviewed'),
    ...predictionLegend(),
    ...(activityShown.size ? [el('span', {class: 'sep'}),
      ...Object.entries(ACTIVITY_COLORS).filter(([name]) => activityShown.has(name))
        .map(([name, color]) => el('span', {class: 'item'}, el('i', {class: 'swatch', style: {background: color}}), name))] : []),
  );
}

function predictionLegend() {
  const used = new Set(predictionList().map(item => item.label_id));
  if (!used.size) return [];
  return [el('span', {class: 'sep'}), el('span', {class: 'item'}, 'Predicted:'),
    ...(predictionRun().labels || []).filter(label => used.has(label.id)).map(label => {
      const swatch = el('i', {class: 'swatch predicted'});
      swatch.style.setProperty('--pred', label.color);
      return el('span', {class: 'item', title: label.description || ''}, swatch, label.name);
    })];
}

function renderSummary() {
  const reviewed = reviewedFraction(state.annotations, duration);
  $('reviewedFact').textContent = `${Math.round(reviewed * 100)}% reviewed`;
  $('annotationCount').textContent = state.annotations.length ? `(${state.annotations.length})` : '';
  const run = predictionRun();
  $('predictionFact').hidden = !run;
  if (run) {
    const progress = predictionProgress(run, state.predictions.reviews);
    $('predictionFact').textContent = `${progress.total} prediction${progress.total === 1 ? '' : 's'}, ${progress.total - progress.unreviewed} decided`;
    $('predictionFact').title = `Automatic predictions (${run.run_id}): ${progress.accepted} accepted, ${progress.rejected} rejected, `
      + `${progress.needs_review} marked for review, ${progress.unreviewed} not decided. Time without a prediction is not marked symptom-free.`;
  }
}

function renderList() {
  const filter = $('listFilter');
  const previous = filter.value || 'all';
  const used = new Set(state.annotations.map(item => item.label_id));
  filter.replaceChildren(
    el('option', {value: 'all'}, 'All labels'),
    el('option', {value: 'needs_review'}, 'Needs review'),
    ...state.labels.filter(label => used.has(label.id)).map(label => el('option', {value: label.id}, label.name)),
  );
  filter.value = [...filter.options].some(option => option.value === previous) ? previous : 'all';
  const rows = state.annotations.filter(item => filter.value === 'all'
    || (filter.value === 'needs_review' ? item.review_status === 'needs_review' : item.label_id === filter.value));
  $('listEmpty').hidden = state.annotations.length > 0;
  $('tableWrap').hidden = state.annotations.length === 0;
  filter.hidden = state.annotations.length === 0;
  $('annotationRows').replaceChildren(...rows.map(item => {
    const label = labelById(item.label_id);
    return el('tr', {class: item.id === state.editingId ? 'active' : '', dataset: {id: item.id}, onclick: () => openAnnotation(item.id, true)},
      el('td', {}, el('span', {class: 'label-cell'}, el('i', {class: 'swatch', style: {background: label?.color || '#888'}}), label?.name || 'Unknown label',
        item.notes ? el('span', {class: 'note-tag', title: item.notes}, 'Note') : null)),
      el('td', {class: 'mono'}, formatTime(item.start_s)),
      el('td', {class: 'mono'}, formatTime(item.end_s)),
      el('td', {class: 'mono'}, `${(item.end_s - item.start_s).toFixed(2)} s`),
      el('td', {}, el('span', {class: `status-tag ${item.review_status}`}, REVIEW_STATUS[item.review_status] || item.review_status)),
      el('td', {class: 'actions'},
        el('button', {type: 'button', class: 'row-menu-btn', title: 'Actions', 'aria-label': 'Actions', 'aria-haspopup': 'menu',
          onclick: event => { event.stopPropagation(); openRowMenu(event.currentTarget, item.id); }}, '\u22EF')),
    );
  }));
}

let rowMenuId = null;
function openRowMenu(button, id) {
  const menu = $('rowMenu');
  if (!menu.hidden && rowMenuId === id) { closeMenus(); return; }
  closeMenus();
  rowMenuId = id;
  const rect = button.getBoundingClientRect();
  menu.hidden = false;
  menu.style.top = `${Math.min(rect.bottom + 4, window.innerHeight - menu.offsetHeight - 8)}px`;
  menu.style.left = `${Math.max(8, rect.right - menu.offsetWidth)}px`;
  menu.querySelector('button').focus();
}
function closeMenus() {
  $('rowMenu').hidden = true; rowMenuId = null;
  $('settingsPanel').hidden = true; $('settingsBtn').setAttribute('aria-expanded', 'false');
}

function renderAll() {
  renderTimeline();
  renderLegend();
  renderSummary();
  renderList();
  renderEditor();
  renderPrediction();
}

// ---- editor ------------------------------------------------------------------------------------

function lastLabelId() {
  try { const id = localStorage.getItem('pd-insighter-symptom-last-label'); if (id && activeLabels().some(l => l.id === id)) return id; } catch { /* optional */ }
  return activeLabels()[0]?.id || '';
}

function beginSelection(startS, endS) {
  const a = clamp(Math.min(startS, endS), 0, duration), b = clamp(Math.max(startS, endS), 0, duration);
  state.editingId = null; state.acceptingId = null; state.predictionId = null;
  state.selection = {start_s: a, end_s: b};
  state.draft = {isNew: true, start_s: a, end_s: b, label_id: state.draft?.isNew ? state.draft.label_id : lastLabelId(),
    confidence: 'medium', review_status: 'confirmed', notes: ''};
  state.dirty = true;
  renderTimeline(); renderList(); renderEditor(); revealEditor();
}

function openAnnotation(id, seekTo) {
  const item = state.annotations.find(entry => entry.id === id);
  if (!item) return;
  state.selection = null; state.acceptingId = null; state.predictionId = null;
  state.editingId = id;
  state.draft = {...item, isNew: false};
  state.dirty = false;
  if (seekTo) { pause(); seek(item.start_s); if (!(item.start_s >= state.view.start && item.end_s <= state.view.end)) setView(windowAround((item.start_s + item.end_s) / 2, span(), duration)); }
  renderTimeline(); renderList(); renderEditor(); revealEditor();
}

function closeEditor() {
  state.selection = null; state.editingId = null; state.draft = null; state.dirty = false;
  state.acceptingId = null;
  $('formError').textContent = '';
  renderTimeline(); renderList(); renderEditor(); renderPrediction();
  updateSaveState();
}

function renderEditor() {
  const draft = state.draft;
  $('editorEmpty').hidden = !!draft || !!state.loadError || !!state.predictionId;
  $('editorForm').hidden = !draft;
  $('editorTitle').textContent = state.acceptingId ? 'Adjust prediction, then save' : draft && !draft.isNew ? 'Edit annotation' : 'New annotation';
  $('editorTitle').title = draft && !draft.isNew ? `Added ${new Date(draft.created_at).toLocaleString()}${draft.annotator ? ` by ${draft.annotator}` : ''}` : '';
  $('deleteBtn').hidden = !draft || draft.isNew;
  if (!draft) return;
  if (document.activeElement !== $('startInput')) $('startInput').value = Number.isFinite(draft.start_s) ? formatTime(draft.start_s, 3) : '';
  if (document.activeElement !== $('endInput')) $('endInput').value = Number.isFinite(draft.end_s) ? formatTime(draft.end_s, 3) : '';
  $('durationNote').textContent = Number.isFinite(draft.start_s) && Number.isFinite(draft.end_s) && draft.end_s > draft.start_s
    ? `Length ${(draft.end_s - draft.start_s).toFixed(3)} s` : '';
  const choices = state.labels.filter(label => !label.archived || label.id === draft.label_id);
  $('labelChoices').replaceChildren(...choices.map(label => el('button', {
    type: 'button', class: 'label-choice', 'aria-pressed': String(label.id === draft.label_id), title: label.description || '',
    dataset: {labelId: label.id}, onclick: () => { draft.label_id = label.id; state.dirty = true; renderEditor(); },
  }, el('i', {class: 'swatch', style: {background: label.color}}), label.name + (label.archived ? ' (archived)' : ''))),
  ...(choices.length ? [] : [el('span', {class: 'muted'}, 'No labels yet. Use Manage Labels to add one.')]));
  $('confidenceChoices').replaceChildren(...CONFIDENCE.map(value => el('button', {
    type: 'button', 'aria-pressed': String(draft.confidence === value), dataset: {confidence: value},
    onclick: () => { draft.confidence = value; state.dirty = true; renderEditor(); },
  }, value[0].toUpperCase() + value.slice(1))));
  $('statusSelect').value = draft.review_status;
  if (document.activeElement !== $('notesInput')) $('notesInput').value = draft.notes || '';
  updateSaveState();
}

function revealEditor() {
  const form = $('editorForm');
  const rect = form.getBoundingClientRect();
  if (rect.bottom > window.innerHeight) form.scrollIntoView({block: 'nearest', behavior: 'smooth'});
}

function readTimeInput(input, key) {
  const value = parseTime(input.value);
  if (value === null) { $('formError').textContent = 'Use a time like 83.25 or 1:23.25.'; return; }
  $('formError').textContent = '';
  state.draft[key] = value; state.dirty = true;
  if (state.draft.isNew) state.selection = {start_s: state.draft.start_s, end_s: state.draft.end_s};
  renderTimeline(); renderEditor();
}

function markBoundary(key) {
  const t = state.time;
  if (!state.draft) { beginSelection(t, key === 'start_s' ? Math.min(duration, t + 2) : Math.max(0, t - 2)); }
  state.draft[key] = t;
  if (key === 'start_s' && !(state.draft.end_s > t)) state.draft.end_s = Math.min(duration, t + 1);
  if (key === 'end_s' && !(state.draft.start_s < t)) state.draft.start_s = Math.max(0, t - 1);
  if (state.draft.isNew) state.selection = {start_s: state.draft.start_s, end_s: state.draft.end_s};
  state.dirty = true;
  renderTimeline(); renderEditor();
}

// ---- saving ------------------------------------------------------------------------------------

function updateSaveState() {
  if (state.saving) return;
  if (state.loadError) setStatus('error', state.loadError);
  else if (state.dirty && state.draft) setStatus('pending', state.draft.isNew ? 'Unsaved selection' : 'Unsaved changes');
  else if (state.revision === 0 && !state.annotations.length) setStatus('saved', 'No annotations yet');
  else setStatus('saved', 'All changes saved');
}

async function persist(nextAnnotations, successText) {
  state.saving = true;
  setStatus('pending', 'Saving');
  try {
    const document_ = await api('/api/symptom-annotations', {method: 'POST', body: JSON.stringify({
      trial_id: trialId, base_revision: state.revision, annotator: annotator(),
      annotations: nextAnnotations.map(({isNew, ...item}) => item),
    })});
    state.annotations = document_.annotations; state.revision = document_.revision;
    state.saving = false;
    setStatus('saved', successText);
    return true;
  } catch (error) {
    state.saving = false;
    if (error.status === 409) {
      setStatus('error', 'Changed elsewhere');
      $('formError').textContent = 'These annotations were changed in another window. Reload the page to see the latest version; your edit is still in the form.';
    } else {
      setStatus('error', 'Not saved');
      $('formError').textContent = `Not saved: ${error.message}`;
    }
    return false;
  }
}

function annotator() {
  return $('annotatorInput').value.trim();
}

async function saveDraft() {
  const draft = state.draft;
  if (!draft || state.saving) return;
  if (document.activeElement === $('startInput')) readTimeInput($('startInput'), 'start_s');
  if (document.activeElement === $('endInput')) readTimeInput($('endInput'), 'end_s');
  draft.notes = $('notesInput').value;
  draft.review_status = $('statusSelect').value;
  const problems = validateDraft(draft, state.labels, duration);
  if (problems.length) { $('formError').textContent = problems.join(' '); return; }
  $('formError').textContent = '';
  const id = draft.isNew ? randomId('ann') : draft.id;
  const item = {id, label_id: draft.label_id, start_s: draft.start_s, end_s: draft.end_s, confidence: draft.confidence,
    review_status: draft.review_status, notes: draft.notes.trim(), annotator: draft.annotator || annotator()};
  if (draft.origin) item.origin = draft.origin;
  const accepting = state.acceptingId;
  const ok = await persist(upsert(state.annotations, item), accepting ? 'Prediction accepted with changes' : draft.isNew ? 'Annotation saved' : 'Changes saved');
  if (!ok) return;
  if (!accepting) { try { localStorage.setItem('pd-insighter-symptom-last-label', item.label_id); } catch { /* optional */ } }
  state.selection = null; state.editingId = null; state.draft = null; state.dirty = false; state.acceptingId = null;
  if (accepting) {
    state.predictionId = accepting;
    await decide(accepting, 'accepted', {annotation_id: item.id, edited: true,
      accepted: {label_id: item.label_id, start_s: item.start_s, end_s: item.end_s}});
  }
  renderAll();
}

async function deleteAnnotation(id) {
  const item = state.annotations.find(entry => entry.id === id);
  if (!item || state.saving) return;
  const name = labelById(item.label_id)?.name || 'this annotation';
  if (!window.confirm(`Delete "${name}" from ${formatTime(item.start_s)} to ${formatTime(item.end_s)}?`)) return;
  const ok = await persist(state.annotations.filter(entry => entry.id !== id), 'Annotation deleted');
  if (!ok) { renderEditor(); return; }
  if (state.editingId === id) { state.editingId = null; state.draft = null; state.dirty = false; }
  renderAll();
}

// ---- automatic predictions: review panel ----------------------------------------------------------

function renderPrediction() {
  const item = predictionsOn() && !state.draft ? predictionById(state.predictionId) : null;
  $('predictionPanel').hidden = !item;
  if (!item) return;
  const label = predictionLabel(item.label_id);
  const swatch = el('i', {class: 'swatch predicted'});
  swatch.style.setProperty('--pred', label.color);
  $('predLabel').replaceChildren(swatch, label.name);
  $('predLabel').title = label.description || '';
  $('predTime').textContent = `${formatTime(item.start_s)} - ${formatTime(item.end_s)} (${(item.end_s - item.start_s).toFixed(1)} s)`;
  $('predScore').textContent = `Score ${item.score} \u00b7 ${STRENGTH_LABELS[item.evidence_strength] || item.evidence_strength} evidence`;
  $('predScore').dataset.strength = item.evidence_strength;
  const decision = decisionOf(item.id);
  $('predState').hidden = !decision;
  $('predState').className = `status-tag ${decision}`;
  $('predState').textContent = DECISION_LABELS[decision] || '';
  $('predSummary').textContent = item.summary || '';
  const sources = predictionRun().source_labels || {};
  $('predEvidence').replaceChildren(...(item.evidence || []).map(entry => el('li', {
    dataset: {kind: entry.kind, zero: entry.kind === 'support' && !(entry.points > 0) ? '1' : '0'},
  },
    el('span', {class: 'points', title: entry.kind === 'penalty' ? 'Points taken off' : 'Points earned of the maximum for this item'},
      entry.kind === 'missing' ? `n/a /${entry.max_points}` : entry.kind === 'penalty' ? `${entry.points}` : `+${entry.points}/${entry.max_points}`),
    el('span', {class: 'source'}, sources[entry.source] || entry.source),
    el('span', {}, entry.text))));
  $('predLimits').textContent = (item.limitations || []).join(' ');
  $('predUndo').hidden = !decision;
  $('predAccept').disabled = decision === 'accepted';
  $('predAccept').textContent = decision === 'accepted' ? 'Accepted' : 'Accept';
  const accepted = annotationOfPrediction(state.annotations, predictionRun().run_id, item.id);
  $('predEdit').textContent = accepted ? 'Edit annotation' : 'Edit';
  document.documentElement.dataset.prediction = item.id;
}

function openPrediction(id, seekTo = true) {
  const item = predictionById(id);
  if (!item) return;
  if (state.dirty && state.draft && !window.confirm('Discard the unsaved selection?')) return;
  state.selection = null; state.editingId = null; state.draft = null; state.dirty = false; state.acceptingId = null;
  state.predictionId = id;
  state.playUntil = null;
  $('predError').textContent = '';
  if (seekTo) {
    pause(); seek(item.start_s);
    if (!(item.start_s >= state.view.start && item.end_s <= state.view.end)) setView(windowAround((item.start_s + item.end_s) / 2, span(), duration));
  }
  renderAll(); updateSaveState();
  const panel = $('predictionPanel');
  if (panel.getBoundingClientRect().bottom > window.innerHeight) panel.scrollIntoView({block: 'nearest', behavior: 'smooth'});
}

function closePrediction() {
  state.predictionId = null; state.playUntil = null;
  delete document.documentElement.dataset.prediction;
  renderAll();
}

function gotoPrediction(direction) {
  const next = stepPrediction(predictionList(), state.predictionId, state.time, direction);
  if (next) openPrediction(next.id, true);
}

function replayPrediction() {
  const item = predictionById(state.predictionId);
  if (!item) return;
  seek(Math.max(0, item.start_s - 1));
  state.playUntil = Math.min(duration, item.end_s + 1);
  play();
}

/** Stores one review decision. The detector output is never changed by this. */
async function decide(id, decision, extra = {}) {
  const run = predictionRun();
  if (!run) return false;
  try {
    state.predictions.reviews = await api('/api/symptom-detection-reviews', {method: 'POST', body: JSON.stringify({
      trial_id: trialId, base_revision: state.predictions.reviews?.revision ?? 0, run_id: run.run_id,
      detection_id: id, decision, reviewer: annotator(), ...extra,
    })});
    $('predError').textContent = '';
    return true;
  } catch (error) {
    if (error.status === 409 && error.body?.current) {
      state.predictions.reviews = error.body.current;
      $('predError').textContent = 'Review decisions were changed in another window. They are reloaded now; choose again.';
    } else $('predError').textContent = `Decision not saved: ${error.message}`;
    return false;
  }
}

async function setDecision(decision) {
  const id = state.predictionId;
  if (!id || state.saving) return;
  await decide(id, decision);
  renderAll();
}

/** Adds the prediction's label to the shared label catalogue when it is not there yet. */
async function ensureLabel(wanted) {
  const existing = matchingLabel(state.labels, wanted);
  if (existing) return existing;
  try {
    const document_ = await api('/api/symptom-labels', {method: 'POST', body: JSON.stringify({
      base_revision: state.labelsRevision,
      labels: [...state.labels.map(({created_at, updated_at, ...label}) => label),
        {id: wanted.id, name: wanted.name, color: wanted.color, description: wanted.description || '', archived: false}],
    })});
    state.labels = document_.labels; state.labelsRevision = document_.revision;
    return matchingLabel(state.labels, wanted);
  } catch (error) {
    $('predError').textContent = error.status === 409 ? 'Labels were changed elsewhere. Reload the page and try again.' : `Label not added: ${error.message}`;
    return null;
  }
}

async function acceptPrediction() {
  const item = predictionById(state.predictionId);
  const run = predictionRun();
  if (!item || !run || state.saving) return;
  let annotation = annotationOfPrediction(state.annotations, run.run_id, item.id);
  if (!annotation) {
    const label = await ensureLabel(predictionLabel(item.label_id));
    if (!label) return;
    annotation = annotationFromPrediction(item, run.run_id, {id: randomId('ann'), labelId: label.id, annotator: annotator()});
    if (!await persist(upsert(state.annotations, annotation), 'Prediction accepted')) {
      $('predError').textContent = $('formError').textContent; $('formError').textContent = '';
      return;
    }
  }
  await decide(item.id, 'accepted', {annotation_id: annotation.id, edited: !!annotation.origin?.edited,
    accepted: {label_id: annotation.label_id, start_s: annotation.start_s, end_s: annotation.end_s}});
  renderAll();
}

/** Opens the normal editor with the prediction's label and times; saving accepts it with the changes. */
async function editPrediction() {
  const item = predictionById(state.predictionId);
  const run = predictionRun();
  if (!item || !run) return;
  const existing = annotationOfPrediction(state.annotations, run.run_id, item.id);
  if (existing) { openAnnotation(existing.id, false); return; }
  const label = await ensureLabel(predictionLabel(item.label_id));
  if (!label) return;
  const draft = annotationFromPrediction(item, run.run_id, {id: '', labelId: label.id, annotator: annotator(), edited: true});
  state.editingId = null;
  state.selection = {start_s: draft.start_s, end_s: draft.end_s};
  state.draft = {...draft, isNew: true};
  state.acceptingId = item.id;
  state.dirty = true;
  renderAll(); revealEditor();
}

async function loadPredictions(runId = null) {
  try {
    const query = `trial_id=${encodeURIComponent(trialId)}${runId ? `&run_id=${encodeURIComponent(runId)}` : ''}`;
    const body = await api(`/api/symptom-detections?${query}`);
    state.predictions = {runs: body.runs || [], run: body.run || null, reviews: body.reviews || null};
  } catch {
    state.predictions = {runs: [], run: null, reviews: null};   // older server or nothing saved: page works without predictions
  }
  const run = predictionRun();
  $('symptomLayers').hidden = !run;
  $('predictionSettings').hidden = !run;
  document.documentElement.dataset.predictions = run ? String(run.detections.length) : 'none';
  if (!run) return;
  const select = $('predictionRun');
  select.replaceChildren(...state.predictions.runs.map(item => el('option', {value: item.run_id, selected: item.run_id === run.run_id},
    `${item.run_id} (${item.count})`)));
  $('predictionRunField').hidden = state.predictions.runs.length < 2;
  const generated = run.generated_at ? new Date(run.generated_at).toLocaleString() : 'unknown time';
  $('predictionRunNote').textContent = `${run.detector?.name || 'Detector'} ${run.detector?.version || ''}, run ${generated}. `
    + 'Predictions are candidates for review. Scores are uncalibrated and are not probabilities. '
    + (run.summary?.notes || []).slice(2).join(' ');
}

function setLayers(value) {
  state.layers = SYMPTOM_LAYERS.includes(value) ? value : 'both';
  $('symptomLayers').value = state.layers;
  try { localStorage.setItem('pd-insighter-symptom-layers', state.layers); } catch { /* optional */ }
  document.documentElement.dataset.symptomLayers = state.layers;
  if (state.layers === 'manual') state.predictionId = null;
  renderAll();
}

// ---- labels ------------------------------------------------------------------------------------

let labelWorking = [];

function openLabelDialog() {
  labelWorking = state.labels.map(label => ({...label}));
  $('labelError').textContent = '';
  renderLabelRows();
  $('labelDialog').showModal();
}

function renderLabelRows() {
  const counts = new Map();
  for (const item of state.annotations) counts.set(item.label_id, (counts.get(item.label_id) || 0) + 1);
  $('labelRows').replaceChildren(...labelWorking.map(label => {
    const row = el('div', {class: `label-row${label.archived ? ' archived' : ''}`, dataset: {labelId: label.id}},
      el('input', {type: 'color', value: label.color, 'aria-label': `Color for ${label.name || 'new label'}`, oninput: event => { label.color = event.target.value; }}),
      el('input', {type: 'text', value: label.name, maxlength: '60', placeholder: 'Label name', 'aria-label': 'Label name', oninput: event => { label.name = event.target.value; }}),
      el('input', {type: 'text', value: label.description || '', maxlength: '300', placeholder: 'Short description (optional)', 'aria-label': 'Description', oninput: event => { label.description = event.target.value; }}),
      el('div', {style: {display: 'flex', gap: '6px', alignItems: 'center'}},
        counts.get(label.id) ? el('span', {class: 'usage', title: 'Annotations in this trial'}, `${counts.get(label.id)} used`) : null,
        label.isNew
          ? el('button', {type: 'button', onclick: () => { labelWorking = labelWorking.filter(item => item !== label); renderLabelRows(); }}, 'Remove')
          : el('button', {type: 'button', onclick: () => { label.archived = !label.archived; renderLabelRows(); }}, label.archived ? 'Restore' : 'Archive')),
    );
    return row;
  }));
}

function addLabel() {
  const palette = ['#2f6db5', '#8e44ad', '#16a085', '#c0392b', '#d35400', '#7f8c3d', '#b0458a', '#34495e'];
  const used = new Set(labelWorking.map(label => label.color));
  labelWorking.push({id: randomId('lbl'), name: '', color: palette.find(color => !used.has(color)) || '#2f6db5', description: '', archived: false, isNew: true});
  renderLabelRows();
  const inputs = $('labelRows').querySelectorAll('.label-row:last-child input[type="text"]');
  inputs[0]?.focus();
}

async function saveLabels(event) {
  event.preventDefault();
  for (const label of labelWorking) {
    const problems = labelNameProblems(label.name, labelWorking, label.id);
    if (problems.length) { $('labelError').textContent = problems[0]; return; }
  }
  $('labelSaveBtn').disabled = true;
  try {
    const document_ = await api('/api/symptom-labels', {method: 'POST', body: JSON.stringify({
      base_revision: state.labelsRevision,
      labels: labelWorking.map(({isNew, created_at, updated_at, ...label}) => ({...label, name: label.name.trim(), description: (label.description || '').trim()})),
    })});
    state.labels = document_.labels; state.labelsRevision = document_.revision;
    $('labelDialog').close();
    setStatus('saved', 'Labels saved');
    renderAll();
  } catch (error) {
    $('labelError').textContent = error.status === 409 ? 'Labels were changed elsewhere. Reload the page and try again.' : error.message;
  } finally {
    $('labelSaveBtn').disabled = false;
  }
}

// ---- timeline interaction ----------------------------------------------------------------------

function bindTrack(track) {
  let drag = null;
  track.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    const hit = event.target.closest('.annotation');
    const prediction = event.target.closest('.prediction');
    const t = timeAt(event, track);
    drag = {id: event.pointerId, x: event.clientX, start: $('dragFromPlayhead').checked ? state.time : t, hitId: hit?.dataset.id || null,
      predictionId: prediction?.dataset.predictionId || null, moved: false};
    track.setPointerCapture(event.pointerId);
    event.preventDefault();
  });
  track.addEventListener('pointermove', event => {
    if (!drag || event.pointerId !== drag.id) return;
    if (!drag.moved && Math.abs(event.clientX - drag.x) < 4) return;
    drag.moved = true;
    const t = timeAt(event, track);
    state.selection = {start_s: Math.min(drag.start, t), end_s: Math.max(drag.start, t)};
    renderTimeline();
  });
  const finish = event => {
    if (!drag || event.pointerId !== drag.id) return;
    const current = drag; drag = null;
    if (track.hasPointerCapture(event.pointerId)) track.releasePointerCapture(event.pointerId);
    if (event.type === 'pointercancel') { renderTimeline(); return; }
    const t = timeAt(event, track);
    if (current.moved && Math.abs(t - current.start) >= 0.05) { beginSelection(current.start, t); return; }
    if (current.predictionId) { openPrediction(current.predictionId, true); return; }
    if (current.hitId) { openAnnotation(current.hitId, false); seek(t); return; }
    state.selection = state.draft?.isNew ? state.selection : null;
    seek(t);
    renderTimeline();
  };
  track.addEventListener('pointerup', finish);
  track.addEventListener('pointercancel', finish);
}

function bindOverview() {
  const track = $('overviewTrack');
  track.addEventListener('pointerdown', event => {
    const t = timeAt(event, track, true);
    setView(windowAround(t, span(), duration));
    seek(t);
  });
}

// ---- keyboard ----------------------------------------------------------------------------------

function bindKeys() {
  document.addEventListener('keydown', event => {
    if ($('labelDialog').open) return;
    if (event.key === 'Escape' && (!$('rowMenu').hidden || !$('settingsPanel').hidden)) { event.preventDefault(); closeMenus(); return; }
    if (event.key === 'Escape' && !$('gotoInput').hidden) { event.preventDefault(); closeGoto(); return; }
    if (event.key === 'Escape' && state.draft) { event.preventDefault(); closeEditor(); return; }
    if (event.key === 'Escape' && state.predictionId) { event.preventDefault(); closePrediction(); return; }
    if (event.key === 'Enter' && state.draft && !event.target?.closest?.('#gotoForm, #settingsPanel') && event.target?.tagName !== 'BUTTON' && event.target?.tagName !== 'SELECT') {
      event.preventDefault(); saveDraft(); return;
    }
    if (isTyping(event.target) || event.ctrlKey || event.metaKey || event.altKey) return;
    const key = event.key.toLowerCase();
    if (event.code === 'Space') { event.preventDefault(); if (!event.repeat) togglePlay(); }
    else if (key === 'arrowleft' || key === 'arrowright') { event.preventDefault(); seek(state.time + (key === 'arrowright' ? 1 : -1) * (event.shiftKey ? 5 : 1)); }
    else if (key === ',' || key === '.') { event.preventDefault(); stepFrame(key === '.' ? 1 : -1); }
    else if (key === 'i') { event.preventDefault(); markBoundary('start_s'); }
    else if (key === 'o') { event.preventDefault(); markBoundary('end_s'); }
    else if (key === 'n') { event.preventDefault(); jumpToUnreviewed(); }
    else if ((key === '[' || key === ']') && predictionsOn()) { event.preventDefault(); gotoPrediction(key === ']' ? 1 : -1); }
  }, true);
}

function openGoto() {
  $('clockBtn').hidden = true;
  $('gotoInput').hidden = false;
  $('gotoInput').value = formatTime(state.time, 2);
  $('gotoInput').focus(); $('gotoInput').select();
}
function closeGoto() {
  $('gotoInput').hidden = true; $('gotoInput').setCustomValidity('');
  $('clockBtn').hidden = false;
}

function stepFrame(direction) {
  pause();
  const frame = Math.round(state.time * fps) + direction;
  seek(frame / fps);
}

function jumpToUnreviewed() {
  const gap = nextUnreviewed(state.annotations, duration, state.time + 0.01);
  if (!gap) { setStatus('saved', 'Everything is reviewed'); return; }
  pause();
  seek(gap.start_s);
  setView(windowAround(gap.start_s + span() * 0.4, span(), duration));
}

// ---- navigation keeps the moment -----------------------------------------------------------------

function bindNavigation() {
  const select = $('trialJumpSelect');
  for (const item of data.trial_jump_options || []) {
    select.append(el('option', {value: item.href, selected: item.trial_id === trialId}, item.label));
  }
  select.addEventListener('change', () => { if (select.value) window.location.href = select.value; });
  // Links to the trial's other views carry the current time; Full Session restores it.
  document.querySelectorAll('.nav-tabs a.nav-tab').forEach(link => {
    const update = () => {
      const url = new URL(link.getAttribute('href'), location.href);
      url.searchParams.set('t', state.time.toFixed(3));
      link.href = url.href;
    };
    for (const name of ['pointerdown', 'focus']) link.addEventListener(name, update);
  });
  window.addEventListener('beforeunload', event => {
    if (state.dirty && state.draft) { event.preventDefault(); event.returnValue = ''; }
  });
}

// ---- start -------------------------------------------------------------------------------------

async function load() {
  setStatus('pending', 'Loading');
  try {
    const [labels, annotations] = await Promise.all([
      api('/api/symptom-labels'),
      api(`/api/symptom-annotations?trial_id=${encodeURIComponent(trialId)}`),
      loadPredictions(),
    ]);
    state.labels = labels.labels; state.labelsRevision = labels.revision;
    state.annotations = annotations.annotations; state.revision = annotations.revision;
    state.loadError = null;
    if (labels.warning || annotations.warning) setStatus('pending', 'Loaded from backup');
    else updateSaveState();
  } catch (error) {
    state.loadError = 'Could not load annotations';
    setStatus('error', state.loadError);
    $('editorEmpty').hidden = false;
    $('editorEmpty').textContent = `Annotations could not be loaded: ${error.message}`;
    for (const id of ['markStart', 'markEnd', 'manageLabelsBtn']) $(id).disabled = true;
  }
  renderAll();
}

function init() {
  $('durationFact').textContent = `${formatTime(duration, 1)} session`;
  $('clockTotal').textContent = formatTime(duration);
  try { $('annotatorInput').value = localStorage.getItem('pd-insighter-annotator') || ''; } catch { /* optional */ }
  $('annotatorInput').addEventListener('change', () => { try { localStorage.setItem('pd-insighter-annotator', annotator()); } catch { /* optional */ } });
  $('syncDetails').textContent = data.sync_note || '';
  let savedSource = 'automatic';
  try { savedSource = localStorage.getItem('pd-insighter-symptom-activity-source') || 'automatic'; } catch { /* optional */ }
  state.activitySource = ['automatic', 'manual', 'hide'].includes(savedSource) ? savedSource : 'automatic';
  $('activitySource').value = state.activitySource;
  document.documentElement.dataset.activitySource = state.activitySource;
  $('activitySource').addEventListener('change', () => setActivitySource($('activitySource').value));
  if (state.activitySource === 'manual') setActivitySource('manual');

  $('settingsBtn').addEventListener('click', event => {
    event.stopPropagation();
    const open = $('settingsPanel').hidden;
    closeMenus();
    $('settingsPanel').hidden = !open;
    $('settingsBtn').setAttribute('aria-expanded', String(open));
  });
  $('settingsPanel').addEventListener('click', event => event.stopPropagation());
  $('rowMenu').addEventListener('click', event => {
    const action = event.target.closest('button')?.dataset.action;
    const id = rowMenuId;
    closeMenus();
    if (action === 'edit') openAnnotation(id, true);
    if (action === 'delete') deleteAnnotation(id);
  });
  document.addEventListener('click', event => { if (!event.target.closest('.row-menu-btn')) closeMenus(); });
  window.addEventListener('scroll', () => { if (!$('rowMenu').hidden) closeMenus(); }, {passive: true});
  $('clockBtn').addEventListener('click', openGoto);
  $('gotoInput').addEventListener('blur', () => setTimeout(closeGoto, 120));

  const windowSelect = $('windowSelect');
  windowSelect.replaceChildren(...WINDOW_SIZES.map(option => el('option', {value: option.value}, option.label)));
  windowSelect.value = duration <= 120 ? 'full' : '60';
  windowSelect.addEventListener('change', () => setView(windowAround(state.time, windowSeconds(), duration)));
  $('prevWindow').addEventListener('click', () => { $('followVideo').checked = false; setView(shiftWindow(state.view, -1, duration)); });
  $('nextWindow').addEventListener('click', () => { $('followVideo').checked = false; setView(shiftWindow(state.view, 1, duration)); });
  $('centerWindow').addEventListener('click', () => setView(windowAround(state.time, span(), duration)));
  $('markStart').addEventListener('click', () => markBoundary('start_s'));
  $('markEnd').addEventListener('click', () => markBoundary('end_s'));
  $('nextUnreviewed').addEventListener('click', jumpToUnreviewed);

  $('playBtn').addEventListener('click', togglePlay);
  $('back5').addEventListener('click', () => seek(state.time - 5));
  $('forward5').addEventListener('click', () => seek(state.time + 5));
  $('backFrame').addEventListener('click', () => stepFrame(-1));
  $('forwardFrame').addEventListener('click', () => stepFrame(1));
  $('speedSelect').addEventListener('change', () => { for (const video of videos) video.el.playbackRate = Number($('speedSelect').value); });
  $('gotoForm').addEventListener('submit', event => {
    event.preventDefault();
    const t = parseTime($('gotoInput').value);
    if (t === null || t > duration) { $('gotoInput').setCustomValidity('Enter a time within the session'); $('gotoInput').reportValidity(); return; }
    $('gotoInput').setCustomValidity('');
    closeGoto();
    pause(); seek(t); setView(windowAround(t, span(), duration));
  });

  $('editorForm').addEventListener('submit', event => { event.preventDefault(); saveDraft(); });
  $('cancelBtn').addEventListener('click', closeEditor);
  $('deleteBtn').addEventListener('click', () => state.draft && deleteAnnotation(state.draft.id));
  $('startInput').addEventListener('change', () => readTimeInput($('startInput'), 'start_s'));
  $('endInput').addEventListener('change', () => readTimeInput($('endInput'), 'end_s'));
  $('startFromPlayhead').addEventListener('click', () => markBoundary('start_s'));
  $('endFromPlayhead').addEventListener('click', () => markBoundary('end_s'));
  $('statusSelect').addEventListener('change', () => { state.draft.review_status = $('statusSelect').value; state.dirty = true; updateSaveState(); });
  $('notesInput').addEventListener('input', () => { state.draft.notes = $('notesInput').value; state.dirty = true; updateSaveState(); });
  $('listFilter').addEventListener('change', renderList);

  let savedLayers = 'both';
  try { savedLayers = localStorage.getItem('pd-insighter-symptom-layers') || 'both'; } catch { /* optional */ }
  state.layers = SYMPTOM_LAYERS.includes(savedLayers) ? savedLayers : 'both';
  $('symptomLayers').value = state.layers;
  document.documentElement.dataset.symptomLayers = state.layers;
  $('symptomLayers').addEventListener('change', () => setLayers($('symptomLayers').value));
  try { state.showLowPredictions = localStorage.getItem('pd-insighter-symptom-show-low') !== '0'; } catch { /* optional */ }
  $('showLowPredictions').checked = state.showLowPredictions;
  $('showLowPredictions').addEventListener('change', () => {
    state.showLowPredictions = $('showLowPredictions').checked;
    try { localStorage.setItem('pd-insighter-symptom-show-low', state.showLowPredictions ? '1' : '0'); } catch { /* optional */ }
    if (state.predictionId && !predictionList().some(item => item.id === state.predictionId)) state.predictionId = null;
    renderAll();
  });
  $('predictionRun').addEventListener('change', async () => { state.predictionId = null; await loadPredictions($('predictionRun').value); renderAll(); });
  $('predClose').addEventListener('click', closePrediction);
  $('predPrev').addEventListener('click', () => gotoPrediction(-1));
  $('predNext').addEventListener('click', () => gotoPrediction(1));
  $('predReplay').addEventListener('click', replayPrediction);
  $('predAccept').addEventListener('click', acceptPrediction);
  $('predEdit').addEventListener('click', editPrediction);
  $('predReject').addEventListener('click', () => setDecision('rejected'));
  $('predLater').addEventListener('click', () => setDecision('needs_review'));
  $('predUndo').addEventListener('click', () => setDecision('unreviewed'));

  $('manageLabelsBtn').addEventListener('click', openLabelDialog);
  $('addLabelBtn').addEventListener('click', addLabel);
  $('labelCancelBtn').addEventListener('click', () => $('labelDialog').close());
  $('labelForm').addEventListener('submit', saveLabels);

  bindTrack($('symptomTrack'));
  bindTrack($('activityTrack'));
  bindOverview();
  bindKeys();
  bindNavigation();
  setupVideos();

  const requested = Number(new URLSearchParams(location.search).get('t'));
  const start = Number.isFinite(requested) && requested >= 0 && requested <= duration ? requested : 0;
  state.view = windowAround(start, windowSeconds(), duration);
  if (master) {
    const apply = () => seek(start);
    if (master.el.readyState >= 1) apply(); else master.el.addEventListener('loadedmetadata', apply, {once: true});
  }
  state.time = start;
  load();
}

init();
