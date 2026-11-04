export function chooseSceneModels(scenes, choices, selection) {
  return scenes.map(scene => {
    const candidate = selection === 'compared' ? choices?.scenes?.[scene.id] : null;
    const sameCamera = candidate?.camera && Object.keys(scene.camera).every(
      key => candidate.camera[key] === scene.camera[key]);
    if (!candidate?.visually_reviewed || !candidate.ply || !Number.isFinite(candidate.iterations)
        || candidate.original_ply !== scene.ply || !sameCamera) {
      return {...scene, qualityVersion: 'original'};
    }
    // Keep original time coverage, calibration, labels and coordinate mapping.
    return {...scene, ply: candidate.ply, iterations: candidate.iterations,
      details_url: candidate.details_url, qualityVersion: 'compared',
      qualityEvidence: {views: candidate.validation_views, psnr: candidate.mean_psnr_db}};
  });
}
