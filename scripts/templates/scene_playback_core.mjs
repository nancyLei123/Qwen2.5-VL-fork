export const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

export function frameAt(videoTime, data) {
  // Browsers quantize media time to microseconds, including just below a frame boundary.
  return clamp(Math.floor((Math.max(0, videoTime) + 0.000001) * data.fps), 0, data.poses.length - 1);
}

export function recordingAt(videoTime, data) {
  return data.recording_times[frameAt(videoTime, data)];
}

export function sceneAt(recordingTime, scenes) {
  return scenes.find(scene => recordingTime >= scene.start_s && recordingTime < scene.end_s) || null;
}

export function labelAt(videoTime, segments, offset, duration) {
  const dashboardTime = videoTime + offset;
  if (dashboardTime < 0 || dashboardTime >= duration) return null;
  return segments.find(segment => dashboardTime >= segment.start_s && dashboardTime < segment.end_s) || null;
}

export function recordingToVideo(time, data) {
  let low = 0, high = data.recording_times.length - 1;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (data.recording_times[mid] < time) low = mid + 1;
    else high = mid;
  }
  return low / data.fps;
}

export function formatTime(time) {
  const safe = Math.round(Math.max(0, time) * 10) / 10;
  return `${Math.floor(safe / 60)}:${(safe % 60).toFixed(1).padStart(4, '0')}`;
}

export function formatExactTime(time) {
  const ms = Math.round(Math.max(0, time) * 1000);
  return `${Math.floor(ms / 60000)}:${((ms % 60000) / 1000).toFixed(3).padStart(6, '0')}`;
}

// Each displayed frame occupies [index / fps, (index + 1) / fps) on the video clock.
export function frameIntervals(data, keyForFrame) {
  const intervals = [];
  if (!data.poses.length) return intervals;
  let start = 0, key = keyForFrame(0);
  for (let index = 1; index <= data.poses.length; index++) {
    const next = index < data.poses.length ? keyForFrame(index) : undefined;
    if (index === data.poses.length || next !== key) {
      intervals.push({start_s: start / data.fps, end_s: Math.min(index / data.fps, data.duration_s), key});
      start = index; key = next;
    }
  }
  return intervals;
}

export function videoLabelSegments(segments, data) {
  return segments.map(segment => ({
    ...segment,
    start_s: Math.max(0, Math.max(0, segment.start_s) - data.dashboard_offset_s),
    end_s: Math.min(data.duration_s, Math.min(data.dashboard_duration_s, segment.end_s) - data.dashboard_offset_s),
  })).filter(segment => Number.isFinite(segment.start_s) && Number.isFinite(segment.end_s) && segment.end_s > segment.start_s);
}

export function missingIntervals(segments, duration) {
  const covered = segments.map(segment => ({
    start_s: clamp(segment.start_s, 0, duration), end_s: clamp(segment.end_s, 0, duration),
  })).filter(segment => Number.isFinite(segment.start_s) && Number.isFinite(segment.end_s) && segment.end_s > segment.start_s)
    .sort((a, b) => a.start_s - b.start_s);
  const gaps = [];
  let end = 0;
  for (const segment of covered) {
    if (segment.start_s > end + 1e-9) gaps.push({start_s: end, end_s: segment.start_s});
    end = Math.max(end, segment.end_s);
  }
  if (end < duration - 1e-9) gaps.push({start_s: end, end_s: duration});
  return gaps;
}

export function playbackCoverage(data, automatic, reference) {
  const poses = frameIntervals(data, index => data.poses[index] != null);
  const estimates = frameIntervals(data, index => data.poses[index] != null && data.pose_status?.[index] === 'interpolated_short_gap');
  const models = frameIntervals(data, index => sceneAt(data.recording_times[index], data.scenes)?.id || null);
  const automaticSegments = videoLabelSegments(automatic, data);
  const referenceSegments = videoLabelSegments(reference, data);
  return {
    poseSegments: poses.filter(interval => interval.key),
    poseGaps: poses.filter(interval => !interval.key),
    poseEstimates: estimates.filter(interval => interval.key),
    modelSegments: models.filter(interval => interval.key).map(interval => {
      const scene = data.scenes.find(item => item.id === interval.key);
      return {...interval, label: scene.title, iterations: scene.iterations};
    }),
    modelGaps: models.filter(interval => !interval.key),
    automaticSegments, referenceSegments,
    automaticGaps: missingIntervals(automaticSegments, data.duration_s),
    referenceGaps: missingIntervals(referenceSegments, data.duration_s),
  };
}

// Keep user intent separate from pauses needed for seeking or loading a 3D scene.
export class PlaybackTransport {
  constructor(video, changed = () => {}, failed = () => {}) {
    this.video = video; this.changed = changed; this.failed = failed;
    this.wantsPlay = false; this.buffering = false; this.scrubbing = false;
    this.pending = false; this.expectedPauses = 0; this.revision = 0;
    video.addEventListener('pause', () => {
      if (this.expectedPauses) this.expectedPauses--;
      else { this.wantsPlay = false; this.revision++; }
      this.changed();
    });
  }
  pauseForSync() {
    this.revision++;
    if (!this.video.paused) this.expectedPauses++;
    this.video.pause();
  }
  setPlaying(playing) {
    this.wantsPlay = playing; this.revision++;
    if (!playing) this.pauseForSync();
    this.resume();
  }
  setBuffering(buffering) {
    if (this.buffering !== buffering) this.revision++;
    this.buffering = buffering;
    if (buffering) this.pauseForSync();
    this.resume();
  }
  beginScrub() {
    if (this.scrubbing) return;
    this.scrubbing = true; this.pauseForSync(); this.changed();
  }
  endScrub(commit = true) {
    if (!this.scrubbing) return;
    this.scrubbing = false; this.revision++;
    if (commit) this.wantsPlay = true;
    this.resume();
  }
  resume() {
    this.changed();
    if (!this.wantsPlay || this.buffering || this.scrubbing || this.video.seeking || !this.video.paused || this.pending) return;
    const revision = this.revision;
    this.pending = true;
    Promise.resolve().then(() => {
      if (revision !== this.revision) return;
      return this.video.play();
    }).catch(error => {
      if (revision !== this.revision || error.name === 'AbortError') return;
      this.wantsPlay = false;
      this.failed(error);
    }).finally(() => {
      this.pending = false;
      // A newer seek/play can arrive while an interrupted play() is still settling.
      if (revision !== this.revision) this.resume();
      else this.changed();
    });
  }
}

// This renderer resolves addSplatScene before its asynchronous spatial-tree build.
// Starting the next mesh build too early can dispose the tree still being built.
export async function waitForSplatTree(mesh, timeoutMs = 30000, pollMs = 25) {
  const deadline = Date.now() + timeoutMs;
  while (mesh.baseSplatTree || !mesh.getSplatTree()) {
    if (mesh.disposed) throw new Error('3D scene was disposed while preparing its spatial index.');
    if (Date.now() >= deadline) throw new Error('3D spatial index did not finish. Please retry loading the scene.');
    await new Promise(resolve => setTimeout(resolve, pollMs));
  }
}

// Scene loads are serialized. A newer seek always wins, even if an older load finishes later.
export class SceneQueue {
  constructor(load, show, status) {
    this.load = load; this.show = show; this.status = status;
    this.loaded = new Map(); this.failed = new Map(); this.desired = null; this.running = false;
  }
  request(scene) {
    this.desired = scene;
    if (scene && this.loaded.has(scene.id)) {
      this.show(this.loaded.get(scene.id)); this.status('ready', scene); return;
    }
    this.show(null);
    if (!scene) { this.status('gap', null); return; }
    if (this.failed.has(scene.id)) { this.status('error', scene, this.failed.get(scene.id)); return; }
    this.status('loading', scene);
    if (!this.running) this.drain();
  }
  retry() {
    if (this.desired) this.failed.delete(this.desired.id);
    this.request(this.desired);
  }
  async drain() {
    this.running = true;
    try {
      while (this.desired && !this.loaded.has(this.desired.id) && !this.failed.has(this.desired.id)) {
        const scene = this.desired;
        try { this.loaded.set(scene.id, await this.load(scene)); }
        catch (error) { this.failed.set(scene.id, error); }
      }
    } finally {
      this.running = false;
      this.request(this.desired);
    }
  }
}
