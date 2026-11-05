"""Add consistent audio controls without rebuilding analysis or changing media."""

from functools import lru_cache
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote, urlsplit

TEMPLATE = Path(__file__).parent / 'templates/page_audio.mjs'


@lru_cache(maxsize=128)
def _probe_audio(path: str, size: int, modified_ns: int) -> bool | None:
    probe = shutil.which('ffprobe')
    if not probe:
        return None
    try:
        result = subprocess.run([probe, '-v', 'error', '-select_streams', 'a',
                                 '-show_entries', 'stream=index', '-of', 'json', path],
                                capture_output=True, text=True, timeout=15, check=True)
        return bool(json.loads(result.stdout)['streams'])
    except (subprocess.SubprocessError, ValueError, KeyError):
        return None


def audio_availability(document: str, page_dir: Path) -> bool | None:
    """Probe the exact local review asset, not the existence of its source VRS."""
    match = re.search(r'<video\b[^>]*id="ariaVideo"[^>]*src="([^"]+)"', document)
    uri = match[1] if match else ''
    if not uri:
        match = re.search(r'"aria_session_video_uri"\s*:\s*"([^"\n]+)"', document)
        uri = match[1] if match else ''
    payload = page_dir / 'scene_playback_data.json'
    if not uri and payload.is_file():
        uri = json.loads(payload.read_text()).get('video_url', '')
    if not uri or urlsplit(uri).scheme or urlsplit(uri).netloc:
        return None
    path = (page_dir / unquote(urlsplit(uri).path)).resolve()
    if not path.is_file():
        return None
    stat = path.stat()
    return _probe_audio(str(path), stat.st_size, stat.st_mtime_ns)


def inject_page_audio(document: str, page_dir: Path) -> str:
    if not re.search(r'<video\b[^>]*id="ariaVideo"', document):
        return document
    document = re.sub(r'<script type="module" id="pd-page-audio">.*?</script>', '', document, flags=re.S)
    document = re.sub(r'<style id="pd-page-audio-style">.*?</style>', '', document, flags=re.S)
    # These legacy initializers ran before the shared controller or after an
    # asynchronous data load. New visitors should never start with sound on.
    document = document.replace('let enabled = true;', 'let enabled = false;')
    document = document.replace('ariaVideo.muted = false;', 'ariaVideo.muted = true;')
    available = audio_availability(document, page_dir)
    script = ('<script type="module" id="pd-page-audio">\n' + TEMPLATE.read_text()
              + '\ninstallPageAudio(document, ' + json.dumps(available) + ');\n</script>')
    style = '''<style id="pd-page-audio-style">
    .pd-audio-toggle { display:inline-flex; align-items:center; justify-content:center;
      gap:8px; min-height:40px; padding:9px 15px; border:1px solid #cbdcd3;
      border-radius:999px; background:#f5f8f5; color:#28493a; cursor:pointer;
      font:inherit; font-size:13px; font-weight:800; white-space:nowrap; }
    .pd-audio-toggle::before { content:''; width:7px; height:7px; border-radius:50%; background:#89958d; }
    .pd-audio-toggle[aria-pressed="true"] { background:#e4f2eb; color:#126345; border-color:#8fbea7; }
    .pd-audio-toggle[aria-pressed="true"]::before { background:#15815a; }
    .pd-audio-toggle:disabled { cursor:default; opacity:.7; }
    .pd-audio-toggle:focus-visible { outline:3px solid #d89e25; outline-offset:3px; }
    </style>'''
    return document.replace('</head>', style + '</head>').replace('</body>', script + '</body>')
