// One audible source per synchronized player. Never change the playback clock.
export const AUDIO_PREFERENCE_KEY = 'pd-insighter-audio-enabled-v1';

export function createAudioController(video, button, followers = [], storage = null) {
  let preferred = false;
  try { preferred = storage?.getItem(AUDIO_PREFERENCE_KEY) === 'true'; } catch {}

  function silenceFollowers() {
    for (const follower of followers) {
      follower.defaultMuted = true;
      if (!follower.muted) follower.muted = true;
    }
  }
  function render() {
    const unavailable = video.dataset.pdAudioAvailable === 'false';
    const failed = !!video.error;
    const enabled = !unavailable && !failed && !video.muted && video.volume > 0;
    button.hidden = false;
    button.disabled = unavailable || failed;
    button.textContent = unavailable ? 'No audio track' : failed ? 'Audio unavailable' : enabled ? 'Audio On' : 'Audio Off';
    button.setAttribute('aria-pressed', String(enabled));
    button.dataset.muted = String(!enabled);
    button.setAttribute('aria-label', unavailable ? 'This video has no audio track' : failed ? 'Video could not load; audio unavailable' : enabled ? 'Turn audio off' : 'Turn audio on');
    button.title = unavailable ? 'This review video was exported without audio.' : failed ? 'Check the video source and reload the page.' : 'Audio from Aria only; the downward view stays muted.';
  }
  function applyPreference() {
    silenceFollowers();
    if (preferred && video.volume <= 0) video.volume = 1;
    video.muted = video.dataset.pdAudioAvailable === 'false' || !preferred;
    render();
  }
  function toggle() {
    if (button.disabled) return;
    preferred = video.muted || video.volume <= 0;
    try { storage?.setItem(AUDIO_PREFERENCE_KEY, String(preferred)); } catch {}
    applyPreference();
  }
  button.addEventListener('click', toggle);
  video.addEventListener('volumechange', render);
  video.addEventListener('error', render);
  video.addEventListener('loadedmetadata', applyPreference);
  video.addEventListener('loadstart', applyPreference);
  for (const follower of followers) {
    follower.addEventListener('volumechange', silenceFollowers);
    follower.addEventListener('play', silenceFollowers);
  }
  applyPreference();
  return { render, toggle, applyPreference };
}

export function installPageAudio(doc, availability = null) {
  const video = doc.getElementById('ariaVideo');
  if (!video || video.dataset.pdAudioInstalled) return;
  video.dataset.pdAudioInstalled = 'true';
  if (typeof availability === 'boolean') video.dataset.pdAudioAvailable = String(availability);
  const oldButton = doc.querySelector('#audioToggleButton, #audioToggleBtn, #audioToggle');
  // Legacy pages bind their own click handlers. Replace only the control, keeping
  // their other transport listeners and synchronized video elements untouched.
  const button = oldButton ? oldButton.cloneNode(true) : doc.createElement('button');
  button.removeAttribute('onclick');
  button.type = 'button';
  button.id = oldButton?.id || 'pageAudioToggle';
  button.classList.add('pd-audio-toggle');
  if (oldButton) oldButton.replaceWith(button);
  else {
    const transport = doc.querySelector('#playBtn, #playPauseButton, #playPause');
    if (transport) transport.after(button);
    else video.parentElement.after(button);
  }
  let storage = null;
  try { storage = doc.defaultView.localStorage; } catch {}
  // Manual-label and scene pages have global Space shortcuts. A focused audio
  // button must keep its normal keyboard activation, without toggling playback.
  for (const type of ['keydown', 'keyup']) doc.defaultView.addEventListener(type, event => {
    if ((event.code === 'Space' || event.code === 'Enter') && event.target === button) event.stopPropagation();
  }, true);
  const followers = [...doc.querySelectorAll('video')].filter(item => item !== video);
  return createAudioController(video, button, followers, storage);
}
