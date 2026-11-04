const byId = id => document.getElementById(id);
let report, selected = 'room';
function show(label) {
  selected = label;
  const scene = report.scenes[label], m = scene.metrics;
  document.querySelectorAll('[data-scene]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.scene === label)));
  byId('sceneTitle').textContent = scene.title;
  const frame = scene.validation_frame;
  const timestamp = frame.recording_s;
  byId('sceneInfo').textContent = `Recording ${scene.segment.start_s}-${scene.segment.end_s} s. ${scene.training_frames} training images + 1 withheld reference at ${timestamp.toFixed(2)} s (${frame.name}). Same calibrated camera for all three images.`;
  byId('reference').src = `${label}/reference.jpg`;
  byId('gaussian').src = `${label}/gaussian.png`;
  byId('metashape').src = `${label}/metashape_checkered.jpg`;
  byId('coverage').textContent = `${m.opaque_mesh_coverage_pct.toFixed(1)}%`;
  byId('missing').textContent = `${m.missing_or_edge_pct.toFixed(1)}%`;
  byId('faces').textContent = `${scene.metashape.faces.toLocaleString()} faces`;
  byId('visualNote').textContent = scene.visual_observation ? `Visual review: ${scene.visual_observation}` : '';
  for (const [id, key] of Object.entries({gsFull:'gaussian_full_psnr_db', msFull:'metashape_full_black_psnr_db', gsMask:'gaussian_common_mask_psnr_db', msMask:'metashape_common_mask_psnr_db'})) byId(id).textContent = m[key] == null ? 'No supported pixels' : m[key].toFixed(2);
  byId('gaussianLink').href = `../gaussian_submaps/${scene.submap}/trained_7000_fullres/viewer.html`;
  byId('comparisonLink').href = `${label}/comparison.jpg`;
  byId('meshPanel').hidden = true;
  byId('meshFrame').removeAttribute('src');
  byId('loadMesh').textContent = 'Explore Metashape in 3D';
  byId('integrity').textContent = report.source_integrity;
  history.replaceState(null, '', `?scene=${label}`);
}
document.querySelectorAll('[data-scene]').forEach(button => button.addEventListener('click', () => report && show(button.dataset.scene)));
byId('zoom').addEventListener('change', event => document.documentElement.style.setProperty('--image-zoom', event.target.value));
document.querySelectorAll('.image-window').forEach(window => window.addEventListener('pointermove', event => {
  const rect = window.getBoundingClientRect();
  document.documentElement.style.setProperty('--image-origin', `${100*(event.clientX-rect.left)/rect.width}% ${100*(event.clientY-rect.top)/rect.height}%`);
}));
byId('loadMesh').addEventListener('click', () => {
  if (!report) return;
  const open = byId('meshPanel').hidden;
  byId('meshPanel').hidden = !open;
  if (open) byId('meshFrame').src = `mesh.html?scene=${selected}`;
  else byId('meshFrame').removeAttribute('src');
  byId('loadMesh').textContent = open ? 'Close 3D viewer' : 'Explore Metashape in 3D';
});
fetch('comparison_results.json').then(response => {
  if (!response.ok) throw new Error(`Results unavailable (${response.status})`);
  return response.json();
}).then(data => { report = data; const scene = new URLSearchParams(location.search).get('scene'); show(Object.hasOwn(report.scenes, scene) ? scene : 'room'); })
.catch(error => {byId('sceneTitle').textContent = 'Results could not be loaded'; byId('sceneInfo').textContent = error.message;});
