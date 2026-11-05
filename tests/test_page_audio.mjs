import test from 'node:test';
import assert from 'node:assert/strict';
import { createAudioController, AUDIO_PREFERENCE_KEY } from '../scripts/templates/page_audio.mjs';

class Media extends EventTarget {
  dataset = {};
  muted = false;
  volume = 1;
  currentTime = 127;
  paused = false;
  error = null;
}
class Button extends EventTarget {
  dataset = {};
  attributes = {};
  setAttribute(key, value) { this.attributes[key] = value; }
}
function fixture(saved = null) {
  const video = new Media(), follower = new Media(), button = new Button();
  const values = new Map(saved === null ? [] : [[AUDIO_PREFERENCE_KEY, saved]]);
  const storage = { getItem: key => values.get(key), setItem: (key, value) => values.set(key, value) };
  const controller = createAudioController(video, button, [follower], storage);
  return { video, follower, button, values, controller };
}
test('starts muted and toggles exactly once without changing time or pause state', () => {
  const { video, follower, button } = fixture();
  assert.equal(button.textContent, 'Audio Off');
  button.dispatchEvent(new Event('click'));
  assert.equal(video.muted, false);
  assert.equal(button.textContent, 'Audio On');
  assert.equal(button.attributes['aria-pressed'], 'true');
  assert.equal(follower.muted, true);
  assert.equal(video.currentTime, 127);
  assert.equal(video.paused, false);
  button.dispatchEvent(new Event('click'));
  assert.equal(video.muted, true);
  assert.equal(button.attributes['aria-pressed'], 'false');
});
test('restores preference after metadata/source changes; seeking does not alter it', () => {
  const { video, button, values } = fixture('true');
  assert.equal(video.muted, false);
  video.muted = true;
  video.dispatchEvent(new Event('loadedmetadata'));
  assert.equal(video.muted, false);
  button.dispatchEvent(new Event('click'));
  assert.equal(values.get(AUDIO_PREFERENCE_KEY), 'false');
  video.dispatchEvent(new Event('seeking'));
  video.dispatchEvent(new Event('loadstart'));
  assert.equal(video.muted, true);
});
test('zero volume can be turned on and native volume changes update the label', () => {
  const { video, button } = fixture('true');
  video.volume = 0;
  video.dispatchEvent(new Event('volumechange'));
  assert.equal(button.textContent, 'Audio Off');
  button.dispatchEvent(new Event('click'));
  assert.equal(video.volume, 1);
  assert.equal(button.textContent, 'Audio On');
});
test('autoplay fallback remains muted and button reflects the actual state', () => {
  const { video, button } = fixture('true');
  video.muted = true;
  video.dispatchEvent(new Event('volumechange'));
  assert.equal(button.textContent, 'Audio Off');
  assert.equal(video.muted, true);
});
test('secondary cameras cannot produce duplicate audio', () => {
  const { follower } = fixture('true');
  follower.muted = false;
  follower.dispatchEvent(new Event('volumechange'));
  assert.equal(follower.muted, true);
});
test('silent video is explicitly unavailable, not a misleading Audio On button', () => {
  const { video, button, controller } = fixture('true');
  video.dataset.pdAudioAvailable = 'false';
  controller.applyPreference();
  assert.equal(button.disabled, true);
  assert.equal(button.hidden, false);
  assert.equal(button.textContent, 'No audio track');
  button.dispatchEvent(new Event('click'));
  assert.equal(video.muted, true);
});
test('failed source is distinguished from a muted source and recovers on reload', () => {
  const { video, button } = fixture();
  video.error = { code: 4 };
  video.dispatchEvent(new Event('error'));
  assert.equal(button.textContent, 'Audio unavailable');
  assert.equal(button.disabled, true);
  video.error = null;
  video.dispatchEvent(new Event('loadedmetadata'));
  assert.equal(button.textContent, 'Audio Off');
  assert.equal(button.disabled, false);
});
test('blocked localStorage does not prevent controls working', () => {
  const video = new Media(), button = new Button();
  createAudioController(video, button, [], { getItem() { throw Error(); }, setItem() { throw Error(); } });
  button.dispatchEvent(new Event('click'));
  assert.equal(video.muted, false);
});
