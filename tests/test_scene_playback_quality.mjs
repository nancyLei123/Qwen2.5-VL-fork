import test from 'node:test';
import assert from 'node:assert/strict';
import {chooseSceneModels} from '../scripts/templates/scene_playback_quality.mjs';

const scenes = [{id: 'stairs', ply: 'old.ply', iterations: 7000, start_s: 78, end_s: 120,
  camera: {fx:512}, video_start_s:78.1, video_end_s:120.1}];
const choices = {scenes: {stairs: {ply:'new.ply', iterations:15000, visually_reviewed:true,
  validation_views:5, mean_psnr_db:29, start_s:0, camera:{fx:512}, original_ply:'old.ply', details_url:'comparison.html'}}};

test('reviewed candidate replaces only model fields, never the clock or calibration', () => {
  const result = chooseSceneModels(scenes, choices, 'compared');
  assert.equal(result[0].ply, 'new.ply');
  assert.equal(result[0].start_s, 78);
  assert.equal(result[0].video_start_s, 78.1);
  assert.deepEqual(result[0].camera, {fx:512});
  assert.equal(scenes[0].ply, 'old.ply');
  assert.equal(scenes[0].qualityVersion, undefined);
});
test('stale candidates cannot replace a different original model or camera', () => {
  const changedModel = [{...scenes[0], ply:'replacement_original.ply'}];
  assert.equal(chooseSceneModels(changedModel, choices, 'compared')[0].ply, 'replacement_original.ply');
  const changedCamera = [{...scenes[0], camera:{fx:600}}];
  assert.equal(chooseSceneModels(changedCamera, choices, 'compared')[0].ply, 'old.ply');
});
test('original selection and unavailable/unreviewed candidates remain original', () => {
  assert.equal(chooseSceneModels(scenes, choices, 'original')[0].ply, 'old.ply');
  assert.equal(chooseSceneModels(scenes, null, 'compared')[0].ply, 'old.ply');
  const unreviewed = {scenes: {stairs: {...choices.scenes.stairs, visually_reviewed:false}}};
  assert.equal(chooseSceneModels(scenes, unreviewed, 'compared')[0].ply, 'old.ply');
});
