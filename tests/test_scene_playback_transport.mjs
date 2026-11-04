import test from 'node:test';
import assert from 'node:assert/strict';
import {PlaybackTransport} from '../scripts/templates/scene_playback_core.mjs';

class Video extends EventTarget {
  paused = true;
  seeking = false;
  currentTime = 0;
  starts = [];
  pauseEvents = [];
  nextPlay = null;
  play() {
    this.paused = false;
    this.starts.push(this.currentTime);
    return this.nextPlay || Promise.resolve();
  }
  pause() {
    if (!this.paused) this.pauseEvents.push(() => this.dispatchEvent(new Event('pause')));
    this.paused = true;
  }
  flushPauseEvents() {
    this.pauseEvents.splice(0).forEach(dispatch => dispatch());
  }
}

const settled = () => new Promise(resolve => setImmediate(resolve));

test('selecting a middle time while paused starts there, only after seeking finishes', async () => {
  const video = new Video(), transport = new PlaybackTransport(video);
  transport.beginScrub();
  video.currentTime = 118.3; video.seeking = true;
  transport.endScrub();
  await settled();
  assert.equal(transport.wantsPlay, true);
  assert.deepEqual(video.starts, []);
  video.seeking = false; transport.resume();
  await settled();
  assert.deepEqual(video.starts, [118.3]);
  assert.equal(video.paused, false);
});

test('scene loading during a drag does not play an intermediate position', async () => {
  const video = new Video(), transport = new PlaybackTransport(video);
  transport.beginScrub();
  video.currentTime = 80;
  transport.setBuffering(true);
  transport.setBuffering(false);
  await settled();
  assert.deepEqual(video.starts, []);
  video.currentTime = 155;
  transport.endScrub();
  await settled();
  assert.deepEqual(video.starts, [155]);
});

test('release while a new scene loads remembers the selected time and play intent', async () => {
  const video = new Video(), transport = new PlaybackTransport(video);
  transport.beginScrub(); video.currentTime = 221;
  transport.setBuffering(true); transport.endScrub();
  await settled();
  assert.equal(video.paused, true);
  assert.equal(transport.wantsPlay, true);
  transport.setBuffering(false);
  await settled();
  assert.deepEqual(video.starts, [221]);
});

test('a queued internal pause cannot cancel playback after dragging while playing', async () => {
  const video = new Video(), transport = new PlaybackTransport(video);
  transport.setPlaying(true); await settled();
  transport.beginScrub(); video.currentTime = 65; transport.endScrub();
  await settled();
  video.flushPauseEvents();
  assert.equal(video.paused, false);
  assert.equal(transport.wantsPlay, true);
  assert.deepEqual(video.starts, [0, 65]);
});

test('rapid seek retries the latest play after the earlier play promise aborts', async () => {
  const video = new Video(), transport = new PlaybackTransport(video);
  let reject;
  video.nextPlay = new Promise((resolve, fail) => { reject = fail; });
  transport.setPlaying(true); await settled();
  transport.beginScrub(); video.currentTime = 170; transport.endScrub();
  video.nextPlay = null;
  reject(Object.assign(new Error('Seek interrupted play'), {name: 'AbortError'}));
  await settled();
  video.flushPauseEvents();
  assert.deepEqual(video.starts, [0, 170]);
  assert.equal(video.paused, false);
  assert.equal(transport.wantsPlay, true);
});

test('Pause during loading wins over a later ready notification', async () => {
  const video = new Video(), transport = new PlaybackTransport(video);
  transport.beginScrub(); video.currentTime = 180;
  transport.setBuffering(true); transport.endScrub();
  transport.setPlaying(false); transport.setBuffering(false);
  await settled();
  assert.equal(video.paused, true);
  assert.deepEqual(video.starts, []);
});

test('canceled pointer interactions preserve the previous play intent', async () => {
  const video = new Video(), transport = new PlaybackTransport(video);
  transport.beginScrub(); transport.endScrub(false);
  await settled();
  assert.deepEqual(video.starts, []);
  transport.setPlaying(true); await settled();
  transport.beginScrub(); transport.endScrub(false);
  await settled();
  assert.equal(video.paused, false);
});

test('frame and gap inspection remain paused even when a scene finishes loading', async () => {
  const video = new Video(), transport = new PlaybackTransport(video);
  transport.setPlaying(true); await settled();
  transport.setPlaying(false); transport.pauseForSync();
  video.currentTime = 82.583;
  transport.setBuffering(true); transport.setBuffering(false);
  video.flushPauseEvents(); transport.resume();
  await settled();
  assert.deepEqual(video.starts, [0]);
  assert.equal(video.paused, true);
});

test('genuine playback failure is surfaced without an endless retry loop', async () => {
  const video = new Video(), errors = [];
  const transport = new PlaybackTransport(video, () => {}, error => errors.push(error.message));
  video.play = () => Promise.reject(Object.assign(new Error('Blocked'), {name: 'NotAllowedError'}));
  transport.setPlaying(true); await settled();
  assert.equal(transport.wantsPlay, false);
  assert.deepEqual(errors, ['Blocked']);
});
