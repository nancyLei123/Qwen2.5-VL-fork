// Presentation-only export of the live viewer; never changes frames, poses or labels.
export function createDemoCapture(video) {
  const canvas = document.createElement('canvas');
  canvas.width = 1280; canvas.height = 960;
  const ctx = canvas.getContext('2d');
  const panel = document.createElement('div');
  panel.className = 'card toolbar';
  panel.style.cssText = 'position:sticky;top:0;z-index:20;border:2px solid #187859';
  panel.innerHTML = '<strong>Local presentation capture</strong><button id="captureStart">Start recording</button><button id="captureStop" disabled>Stop recording</button><a id="captureDownload" hidden download="NTrial3_3D_interactive_demo.webm">Download video</a><button id="captureStill">Save screenshot</button><span id="captureState" role="status">Live viewer + Aria video, no audio</span>';
  document.querySelector('main').prepend(panel);
  const $ = id => document.getElementById(id);
  let recorder, chunks = [], started = 0, objectURL, lastClick = null, pointer = null, stopTimer;
  const mime = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find(x => MediaRecorder.isTypeSupported(x));
  const root = $('sceneRoot');
  root.addEventListener('pointermove', event => {
    const rect = root.getBoundingClientRect();
    pointer = {x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height, at: performance.now(), down: event.buttons > 0};
  });
  document.addEventListener('pointerdown', event => {
    const button = event.target.closest('button');
    if (button && !panel.contains(button)) lastClick = {text: button.textContent.trim(), at: performance.now()};
  });
  function download(blob, name) {
    const url = URL.createObjectURL(blob), link = document.createElement('a');
    link.href = url; link.download = name; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }
  $('captureStill').onclick = () => canvas.toBlob(blob => download(blob, 'NTrial3_3D_' + $('sceneStage').dataset.viewMode + '.png'));
  $('captureStart').onclick = () => {
    chunks = []; started = performance.now();
    if (objectURL) URL.revokeObjectURL(objectURL);
    $('captureDownload').hidden = true;
    recorder = new MediaRecorder(canvas.captureStream(30), {mimeType: mime, videoBitsPerSecond: 5500000});
    recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
    recorder.onstop = () => {
      clearTimeout(stopTimer);
      recorder.stream.getTracks().forEach(track => track.stop());
      objectURL = URL.createObjectURL(new Blob(chunks, {type: mime}));
      $('captureDownload').href = objectURL; $('captureDownload').hidden = false;
      $('captureStart').disabled = false; $('captureStop').disabled = true;
      $('captureState').textContent = 'Ready to download. Original data unchanged.';
    };
    recorder.start(1000); $('captureStart').disabled = true; $('captureStop').disabled = false;
    stopTimer = setTimeout(() => { if (recorder.state === 'recording') recorder.stop(); }, 60000);
  };
  $('captureStop').onclick = () => { if (recorder?.state === 'recording') recorder.stop(); };
  function text(value, x, y, size = 16, color = '#182921', weight = 400) {
    ctx.font = `${weight} ${size}px "Avenir Next", sans-serif`; ctx.fillStyle = color;
    ctx.fillText(value, x, y);
  }
  function box(x, y, width, height, fill, radius = 16) {
    ctx.fillStyle = fill; ctx.beginPath(); ctx.roundRect(x, y, width, height, radius); ctx.fill();
  }
  function wrapped(value, x, y, maxWidth, size = 16) {
    ctx.font = `${size}px "Avenir Next", sans-serif`;
    let line = '';
    for (const word of value.split(' ')) {
      if (line && ctx.measureText(line + ' ' + word).width > maxWidth) { text(line, x, y, size); y += size * 1.4; line = word; }
      else line += (line ? ' ' : '') + word;
    }
    text(line, x, y, size);
  }
  function track(id, label, y) {
    const x = 250, width = 982;
    text(label, 48, y + 16, 14, '#63756b', 700);
    box(x, y, width, 22, '#e3e9e2', 5);
    ctx.save(); ctx.beginPath(); ctx.rect(x, y, width, 22); ctx.clip();
    for (const segment of $(id).querySelectorAll('.segment')) {
      ctx.fillStyle = segment.classList.contains('gap') ? '#bbbfc4' : getComputedStyle(segment).backgroundColor;
      ctx.fillRect(x + parseFloat(segment.style.left) * width / 100, y, parseFloat(segment.style.width) * width / 100, 22);
    }
    const now = x + video.currentTime / video.duration * width;
    ctx.fillStyle = '#fff'; ctx.fillRect(now - 3, y, 6, 22);
    ctx.fillStyle = '#182921'; ctx.fillRect(now - 1, y, 2, 22); ctx.restore();
  }
  return {
    paint(sceneCanvas) {
      const following = $('sceneStage').dataset.viewMode === 'follow';
      ctx.fillStyle = '#f0f3ec'; ctx.fillRect(0, 0, 1280, 960);
      text('PD INSIGHTER / NTRIAL3', 32, 34, 15, '#187859', 800);
      text('Movement in its spatial context', 32, 76, 32, '#182921', 800);
      text(following ? 'Follow the wearer\'s recorded camera position and orientation' : 'Explore around the recorded camera: green marker + orange facing arrow', 32, 106, 17, '#63756b');
      box(24, 128, 736, 626, '#fff'); box(780, 128, 476, 626, '#fff');
      text('3D environment', 44, 160, 19, '#182921', 800);
      text(following ? 'FOLLOW ARIA VIEW' : 'EXPLORE IN 3D', 538, 160, 14, '#187859', 800);
      box(32, 180, 720, 530, '#111c16', 8);
      // Square projection is preserved, matching the calibrated interactive camera.
      ctx.drawImage(sceneCanvas, 127, 180, 530, 530);
      if (pointer && performance.now() - pointer.at < 1600) {
        const x = 127 + pointer.x * 530, y = 180 + pointer.y * 530;
        ctx.beginPath(); ctx.arc(x, y, pointer.down ? 12 : 7, 0, Math.PI * 2);
        ctx.fillStyle = pointer.down ? '#ffcb59aa' : '#ffffffbb'; ctx.fill();
        ctx.strokeStyle = '#18392b'; ctx.lineWidth = 2; ctx.stroke();
      }
      text($('sceneName').textContent, 44, 738, 16, '#63756b');
      text('Original Aria video', 800, 160, 19, '#182921', 800);
      box(808, 180, 420, 330, '#111c16', 8);
      ctx.save(); ctx.translate(1018, 345); ctx.rotate(Math.PI / 2);
      ctx.drawImage(video, -165, -165, 330, 330); ctx.restore();
      text($('clock').textContent, 808, 550, 23, '#187859', 800);
      text($('actionBadge').textContent, 808, 589, 18, '#182921', 800);
      wrapped($('referenceAction').textContent, 808, 620, 420, 16);
      wrapped(following ? 'Synchronized original video, calibrated camera pose and action labels.' : 'The marker represents the RGB camera, not a reconstructed person. Moving objects remain in the video.', 808, 658, 410, 16);
      if (lastClick && performance.now() - lastClick.at < 2200) {
        box(150, 660, 484, 36, '#f5f9f1ed', 10); text('Clicked: ' + lastClick.text, 168, 684, 16, '#187859', 800);
      }
      box(24, 770, 1232, 126, '#fff');
      track('coverage', '3D coverage', 787); track('automatic', 'Automatic classification', 823); track('reference', 'Manual / spoken reference', 859);
      text('Recorded dashboard interaction | Static Gaussian scene; residual blur and incomplete surfaces remain.', 32, 925, 15, '#63756b');
      if (recorder?.state === 'recording') $('captureState').textContent = `Recording ${((performance.now() - started) / 1000).toFixed(1)} s (silent)`;
    },
  };
}
