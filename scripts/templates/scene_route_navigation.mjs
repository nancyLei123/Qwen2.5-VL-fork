// Display-only sampling; the player still uses the original per-frame poses.
export function routeSegments(data, start, end, stepSeconds = 0.2) {
  const segments = [];
  const step = Math.max(1, Math.round(stepSeconds * data.fps));
  let previous = null;
  for (let i = Math.max(0, Math.ceil(start * data.fps)); i < Math.min(data.poses.length, Math.ceil(end * data.fps)); i++) {
    const pose = data.poses[i];
    if (!pose) { previous = null; continue; }
    const sample = {time: i / data.fps, position: pose.slice(0, 3)};
    if (!previous) { previous = sample; continue; }
    if (i % step !== 0 && i + 1 < Math.min(data.poses.length, Math.ceil(end * data.fps))) continue;
    const distance = Math.hypot(...sample.position.map((v, j) => v - previous.position[j]));
    if (distance <= 3 && sample.time - previous.time <= stepSeconds + 2 / data.fps) {
      segments.push({a: previous, b: sample});
    }
    previous = sample;
  }
  return segments;
}

// Projected segments can cross or represent repeated visits. Prefer the time
// nearest the current playhead only when screen distances are effectively tied.
export function nearestRouteTime(segments, point, currentTime, radius = 14) {
  let best = null;
  for (const segment of segments) {
    const {a, b, start, end} = segment;
    if (![...a, ...b, start, end].every(Number.isFinite)) continue;
    const dx = b[0] - a[0], dy = b[1] - a[1], length2 = dx * dx + dy * dy;
    const fraction = length2 < 1e-6 ? Math.max(0, Math.min(1, (currentTime - start) / Math.max(1e-9, end - start)))
      : Math.max(0, Math.min(1, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2));
    const distance = Math.hypot(point[0] - a[0] - fraction * dx, point[1] - a[1] - fraction * dy);
    if (distance > radius) continue;
    const time = start + fraction * (end - start);
    if (!best || distance < best.distance - 1 || (Math.abs(distance - best.distance) <= 1 && Math.abs(time - currentTime) < Math.abs(best.time - currentTime))) {
      best = {time, distance};
    }
  }
  return best?.time ?? null;
}

export function routeWindow(data, scene, scope, time) {
  if (scope === 'nearby') return nearbyRouteWindow(data.duration_s, time);
  if (scope === 'first15') return [0, Math.min(900, data.duration_s)];
  if (scope === 'first30') return [0, Math.min(1800, data.duration_s)];
  if (scope === 'all') return [0, data.duration_s];
  if (scene) return [scene.video_start_s, scene.video_end_s];
  const start = Math.max(0, Math.floor(time / 60) * 60);
  return [start, Math.min(start + 60, data.duration_s)];
}

// A short, stable window around the playhead, never a bounding box of the
// entire session. This affects the route overlay only, not the model or poses.
export function nearbyRouteWindow(duration, time, seconds = 30) {
  const width = Math.min(seconds, duration);
  const start = Math.max(0, Math.min(duration - width, Math.round(time / 5) * 5 - width / 2));
  return [start, start + width];
}

export function moveOrbitAnchor(cameraPosition, previousAnchor, anchor) {
  return cameraPosition.map((value, i) => value + (anchor[i] - previousAnchor[i]));
}

export function playbackSceneModel(scene, joint, choice, time) {
  if (!scene || !['joint', 'joint30'].includes(choice) || !joint || time < joint.video_start_s || time >= joint.video_end_s) return scene;
  return joint;
}
