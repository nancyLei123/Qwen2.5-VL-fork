// Pure helpers for the Symptom Annotation page (no DOM). Tested in tests/test_symptom_annotation_core.mjs.
//
// Clock: every time here is session time in seconds, the same clock as Full Session and Manual
// Labels. A video's own time maps to it as  session = video + offset_s  (offset 0 for the
// trial's session videos, which are already trimmed to the session).

export const WINDOW_SIZES = [
  {value: '30', label: '30 s', seconds: 30},
  {value: '60', label: '1 min', seconds: 60},
  {value: '120', label: '2 min', seconds: 120},
  {value: '300', label: '5 min', seconds: 300},
  {value: 'full', label: 'Full session', seconds: Infinity},
];
export const CONFIDENCE = ['high', 'medium', 'low'];
export const REVIEW_STATUS = {confirmed: 'Confirmed', needs_review: 'Needs review'};
export const MIN_INTERVAL_S = 0.05;

export const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

/** 83.4 -> "1:23.40"; hours when needed. */
export function formatTime(seconds, digits = 2) {
  if (!Number.isFinite(seconds)) return '--';
  const sign = seconds < 0 ? '-' : '';
  const scale = 10 ** digits;
  const total = Math.round(Math.abs(seconds) * scale) / scale;
  const h = Math.floor(total / 3600), m = Math.floor((total % 3600) / 60), s = total - 3600 * h - 60 * m;
  const sec = s.toFixed(digits).padStart(digits ? digits + 3 : 2, '0');
  return h ? `${sign}${h}:${String(m).padStart(2, '0')}:${sec}` : `${sign}${m}:${sec}`;
}

/** Accepts "83.4", "1:23.4", "0:01:23.4". Returns seconds or null. */
export function parseTime(text) {
  if (typeof text !== 'string') return null;
  const parts = text.trim().split(':');
  if (!parts.length || parts.length > 3 || parts.some(part => !/^\d+(\.\d+)?$/.test(part.trim()))) return null;
  const numbers = parts.map(Number);
  if (numbers.slice(0, -1).some(n => !Number.isInteger(n))) return null;
  if (numbers.length > 1 && numbers.slice(1).some(n => n >= 60)) return null;
  return numbers.reduce((total, n) => total * 60 + n, 0);
}

export function videoToSession(videoTime, offset = 0) {
  return videoTime + offset;
}

export function sessionToVideo(sessionTime, offset = 0) {
  return sessionTime - offset;
}

/** Whether a video covers session time t, given its offset and own duration. */
export function videoCovers(sessionTime, offset, videoDuration) {
  const local = sessionToVideo(sessionTime, offset);
  return local >= 0 && local <= videoDuration;
}

/** Whether t lies in a recorded-footage window [start, end); no window means recorded throughout. */
export function inCoverage(sessionTime, coverage) {
  return !coverage || (sessionTime >= coverage[0] && sessionTime < coverage[1]);
}

/** Visible window [start, end) of the given size, kept inside the session. */
export function windowAround(center, size, duration) {
  if (!Number.isFinite(size) || size >= duration) return {start: 0, end: duration};
  const start = clamp(center - size / 2, 0, duration - size);
  return {start, end: start + size};
}

export function shiftWindow(view, direction, duration) {
  const size = view.end - view.start;
  if (size >= duration) return {...view};
  const start = clamp(view.start + direction * size * 0.9, 0, duration - size);
  return {start, end: start + size};
}

/** Lane index per annotation so overlapping intervals are drawn on separate rows. */
export function assignLanes(annotations) {
  const order = [...annotations].sort((a, b) => a.start_s - b.start_s || a.end_s - b.end_s || a.id.localeCompare(b.id));
  const laneEnds = [];
  const lanes = new Map();
  for (const item of order) {
    let lane = laneEnds.findIndex(end => end <= item.start_s);
    if (lane < 0) { lane = laneEnds.length; laneEnds.push(item.end_s); } else laneEnds[lane] = item.end_s;
    lanes.set(item.id, lane);
  }
  return {lanes, count: Math.max(1, laneEnds.length)};
}

/** Session intervals with no annotation of any label (not reviewed yet). */
export function unreviewedIntervals(annotations, duration, minLength = 0) {
  const sorted = [...annotations].sort((a, b) => a.start_s - b.start_s);
  const gaps = [];
  let cursor = 0;
  for (const item of sorted) {
    if (item.start_s > cursor + 1e-9) gaps.push({start_s: cursor, end_s: Math.min(item.start_s, duration)});
    cursor = Math.max(cursor, item.end_s);
    if (cursor >= duration) break;
  }
  if (cursor < duration - 1e-9) gaps.push({start_s: cursor, end_s: duration});
  return gaps.filter(gap => gap.end_s - gap.start_s > minLength);
}

/** First unreviewed interval starting at or after t; wraps to the first one. */
export function nextUnreviewed(annotations, duration, t, minLength = 0.5) {
  const gaps = unreviewedIntervals(annotations, duration, minLength);
  return gaps.find(gap => gap.start_s >= t - 1e-6) || gaps[0] || null;
}

export function reviewedFraction(annotations, duration) {
  if (!(duration > 0)) return 0;
  const missing = unreviewedIntervals(annotations, duration).reduce((sum, gap) => sum + gap.end_s - gap.start_s, 0);
  return clamp(1 - missing / duration, 0, 1);
}

/** Problems with a draft annotation, as short messages (empty when valid). */
export function validateDraft(draft, labels, duration) {
  const problems = [];
  const {start_s: start, end_s: end} = draft;
  if (!Number.isFinite(start) || !Number.isFinite(end)) problems.push('Enter a start and end time.');
  else {
    if (start < 0) problems.push('Start time cannot be negative.');
    if (end > duration + 1e-6) problems.push(`End time is after the end of the session (${formatTime(duration)}).`);
    if (end - start < MIN_INTERVAL_S) problems.push('End time must be after the start time.');
  }
  const label = labels.find(item => item.id === draft.label_id);
  if (!label) problems.push('Choose a label.');
  else if (label.archived && draft.isNew) problems.push('This label is archived. Choose another label.');
  if (!CONFIDENCE.includes(draft.confidence)) problems.push('Choose a confidence.');
  return problems;
}

/** Problems with a label name (empty when fine); names are compared without case or spacing. */
export function labelNameProblems(name, labels, ownId = null) {
  const clean = (name || '').split(/\s+/).filter(Boolean).join(' ');
  if (!clean) return ['Enter a label name.'];
  if (clean.length > 60) return ['Keep the name under 60 characters.'];
  const taken = labels.find(item => item.id !== ownId && item.name.split(/\s+/).filter(Boolean).join(' ').toLowerCase() === clean.toLowerCase());
  return taken ? [`A label named "${taken.name}" already exists${taken.archived ? ' (archived)' : ''}.`] : [];
}

export function randomId(prefix, random = () => Math.random()) {
  let id = '';
  for (let i = 0; i < 10; i++) id += Math.floor(random() * 36).toString(36);
  return `${prefix}_${id}`;
}

/** Inserts or replaces one annotation by id, keeping the list sorted by time. */
export function upsert(annotations, item) {
  const next = annotations.filter(existing => existing.id !== item.id);
  next.push(item);
  return next.sort((a, b) => a.start_s - b.start_s || a.end_s - b.end_s || a.id.localeCompare(b.id));
}

export function annotationsAt(annotations, t) {
  return annotations.filter(item => item.start_s <= t && t < item.end_s);
}

/** Readable tick spacing for a visible span. */
export function tickStep(span) {
  return [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800].find(step => span / step <= 10) || 3600;
}
