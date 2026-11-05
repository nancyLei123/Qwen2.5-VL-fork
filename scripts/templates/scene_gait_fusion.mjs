import {validFigurePose} from './scene_pose_figure.mjs';

export function compatibleGait(gait, data) {
  return data.trial_id === 'RoomScan3c' && gait?.schema_version === 1
    && gait.trial_id === data.trial_id && gait.graph_uid === data.provenance?.graph_uid
    && gait.duration_s === data.duration_s && gait.fps === data.fps
    && Number.isFinite(gait.sync?.recording_offset_s) && gait.sync.recording_scale > 0
    && Array.isArray(gait.windows);
}

export function gaitWindowAt(gait, time) {
  return gait?.windows.find(w => time >= w.start_s && time <= w.end_s) || null;
}

export function gaitMotionAt(gait, time, pose) {
  const window = gaitWindowAt(gait, time);
  if (!window || !validFigurePose(pose)) return null;
  const samples = window.direction_speed_samples;
  let i = samples.findIndex(s => s[0] >= time);
  if (i < 1) i = i === -1 ? samples.length - 1 : 1;
  const a = samples[i - 1], b = samples[i];
  if (!a || !b || b[0] - a[0] > .15 || time < a[0] || time > b[0]) return null;
  const f = (time - a[0]) / (b[0] - a[0]);
  const heading = Math.atan2((1 - f) * Math.sin(a[1]) + f * Math.sin(b[1]),
    (1 - f) * Math.cos(a[1]) + f * Math.cos(b[1]));
  const feet = {};
  for (const side of ['left', 'right']) {
    const cycles = window.cycles.filter(c => c.side === side);
    const cycle = cycles.find(c => time >= c.start_s && time < c.end_s);
    if (!cycle || cycle.end_s - cycle.start_s > 2.25) continue;
    const phase = (time - cycle.start_s) / (cycle.end_s - cycle.start_s);
    // Foot lift and between-anchor motion are display priors, not observed contacts.
    const weight = Math.max(0, Math.min(1, (time - cycles[0].start_s) / .2,
      (cycles.at(-1).end_s - time) / .2));
    feet[side] = {forward: cycle.display_excursion_m * Math.cos(phase * 2 * Math.PI),
      lift: phase > .5 ? .045 * Math.sin((phase - .5) * 2 * Math.PI) : 0,
      phase, weight, period_s: cycle.end_s - cycle.start_s, head_path_m: cycle.head_path_m};
  }
  return Object.values(feet).some(foot => foot.weight > 0)
    ? {kind: 'video_slam', feet, heading, measured: false, window_id: window.id} : null;
}

export function evidenceTimeAt(time, data, gait) {
  if (!gait || time < gait.clip.start_s || time >= gait.clip.end_s) return null;
  const frame = time * data.fps, lo = Math.floor(frame), hi = Math.ceil(frame);
  const r = data.recording_times;
  if (lo < 0 || hi >= r.length || r[hi] - r[lo] > .1) return null;
  const recording = r[lo] + (r[hi] - r[lo]) * (frame - lo);
  const downward = (recording - gait.sync.recording_offset_s) / gait.sync.recording_scale;
  const clip = downward - gait.clip.source_start_s;
  return clip >= -1e-6 && clip < gait.clip.source_end_s - gait.clip.source_start_s
    ? {clip_s: Math.max(0, clip), downward_s: downward} : null;
}

// One master clock. This secondary clip never seeks or starts the Aria video.
export class EvidenceClip {
  constructor(video) { this.video = video; this.pending = false; this.lastTarget = null; this.error = ''; }
  update(target, playing, rate = 1) {
    const v = this.video;
    if (target === null || v.readyState < 1) { v.pause(); this.lastTarget = null; return; }
    const jumped = this.lastTarget === null || target < this.lastTarget - .01 || Math.abs(target - this.lastTarget) > .3;
    this.lastTarget = target;
    v.playbackRate = rate;
    if (!playing) v.pause();
    if (!v.seeking && (jumped || Math.abs(v.currentTime - target) > (playing ? .10 : .018))) v.currentTime = target;
    if (playing && v.paused && !this.pending) {
      this.pending = true;
      Promise.resolve(v.play()).catch(error => { if (error.name !== 'AbortError') this.error = error.message; })
        .finally(() => { this.pending = false; });
    }
  }
}
