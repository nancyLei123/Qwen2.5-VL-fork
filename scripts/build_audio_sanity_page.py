#!/usr/bin/env python3
"""Build a very small local HTML page for testing extracted audio playback."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-id", required=True, help="Trial id, e.g. trial2")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        help="Optional explicit path to a trial_manifest.json file.",
    )
    return parser.parse_args()


def relative_href(page_path: Path, target_path: Path) -> str:
    return Path(os.path.relpath(target_path.resolve(), start=page_path.resolve().parent)).as_posix()


def main() -> None:
    args = parse_args()
    default_manifest = (
        Path(__file__).resolve().parents[1] / "runs" / args.trial_id / "trial_manifest.json"
    )
    manifest_path = (args.manifest_path or default_manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    run_dir = Path(manifest["analysis_outputs"]["run_dir"]["path"]).resolve()
    extracted_media = manifest.get("extracted_media", {})
    extracted_audio = Path(extracted_media["audio_wav"]["path"]).resolve()
    clip_audio_dir = run_dir / "clip_audio"
    output_html = run_dir / "audio_sanity_check.html"

    clip_files = sorted(clip_audio_dir.glob("clip_*.wav"))
    clip_records = [
        {
            "name": clip_path.name,
            "uri": relative_href(output_html, clip_path),
        }
        for clip_path in clip_files
    ]

    recording_stem = str(manifest.get("recording_stem", args.trial_id))
    title = f"{args.trial_id} audio sanity check"

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f6f6;
      color: #1a1a1a;
      margin: 0;
      padding: 24px;
    }}
    .page {{
      max-width: 900px;
      margin: 0 auto;
      display: grid;
      gap: 18px;
    }}
    .card {{
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    }}
    audio {{
      width: 100%;
      margin-top: 10px;
    }}
    select, button {{
      font: inherit;
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid #ccc;
      background: #fafafa;
    }}
    .row {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
      color: #555;
      word-break: break-all;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="card">
      <h1>Audio Sanity Check</h1>
      <p><b>Trial:</b> {html.escape(args.trial_id)}</p>
      <p><b>Recording:</b> {html.escape(recording_stem)}</p>
      <p>This page is only for checking whether the extracted audio itself plays correctly.</p>
    </section>

    <section class="card">
      <h2>Full Recording Audio</h2>
      <div class="mono">{html.escape(str(extracted_audio))}</div>
      <div class="row">
        <button id="playFullButton" type="button">Play Full Audio</button>
        <button id="pauseFullButton" type="button">Pause</button>
        <span id="fullStatus" class="mono"></span>
      </div>
      <audio id="fullAudio" controls preload="metadata" src="{relative_href(output_html, extracted_audio)}"></audio>
    </section>

    <section class="card">
      <h2>Clip Audio</h2>
      <div class="row">
        <label for="clipPicker"><b>Choose clip:</b></label>
        <select id="clipPicker"></select>
        <button id="loadClipButton" type="button">Load Clip Audio</button>
        <button id="playClipButton" type="button">Play Clip Audio</button>
        <button id="pauseClipButton" type="button">Pause</button>
      </div>
      <div class="mono" id="clipPath"></div>
      <div class="mono" id="clipStatus"></div>
      <audio id="clipAudio" controls preload="metadata"></audio>
    </section>
  </div>

  <script>
    const clips = {json.dumps(clip_records)};
    const fullAudio = document.getElementById("fullAudio");
    const fullStatus = document.getElementById("fullStatus");
    const playFullButton = document.getElementById("playFullButton");
    const pauseFullButton = document.getElementById("pauseFullButton");
    const clipPicker = document.getElementById("clipPicker");
    const clipPath = document.getElementById("clipPath");
    const clipStatus = document.getElementById("clipStatus");
    const clipAudio = document.getElementById("clipAudio");
    const loadClipButton = document.getElementById("loadClipButton");
    const playClipButton = document.getElementById("playClipButton");
    const pauseClipButton = document.getElementById("pauseClipButton");

    function attachStatus(audioEl, statusEl, label) {{
      const update = (msg) => {{
        statusEl.textContent = `${{label}}: ${{msg}}`;
      }};
      audioEl.addEventListener("loadstart", () => update("loading"));
      audioEl.addEventListener("loadedmetadata", () => update(`ready (${{audioEl.duration.toFixed(2)}}s)`));
      audioEl.addEventListener("canplay", () => update("can play"));
      audioEl.addEventListener("play", () => update("playing"));
      audioEl.addEventListener("pause", () => update("paused"));
      audioEl.addEventListener("ended", () => update("ended"));
      audioEl.addEventListener("error", () => update(`error code ${{audioEl.error ? audioEl.error.code : "unknown"}}`));
      update("idle");
    }}

    function loadClip() {{
      const idx = Number(clipPicker.value || 0);
      const clip = clips[idx];
      if (!clip) return;
      clipAudio.pause();
      clipAudio.src = clip.uri;
      clipAudio.load();
      clipPath.textContent = clip.uri;
    }}

    async function tryPlay(audioEl, statusEl, label) {{
      try {{
        await audioEl.play();
        statusEl.textContent = `${{label}}: playing`;
      }} catch (err) {{
        statusEl.textContent = `${{label}}: play failed - ${{err && err.message ? err.message : err}}`;
      }}
    }}

    clips.forEach((clip, idx) => {{
      const option = document.createElement("option");
      option.value = idx;
      option.textContent = clip.name;
      clipPicker.appendChild(option);
    }});

    clipPicker.addEventListener("change", loadClip);
    loadClipButton.addEventListener("click", loadClip);
    playFullButton.addEventListener("click", () => tryPlay(fullAudio, fullStatus, "Full audio"));
    pauseFullButton.addEventListener("click", () => fullAudio.pause());
    playClipButton.addEventListener("click", () => tryPlay(clipAudio, clipStatus, "Clip audio"));
    pauseClipButton.addEventListener("click", () => clipAudio.pause());
    attachStatus(fullAudio, fullStatus, "Full audio");
    attachStatus(clipAudio, clipStatus, "Clip audio");
    loadClip();
  </script>
</body>
</html>
"""

    output_html.write_text(html_text)
    print(f"Wrote audio sanity page to: {output_html}")


if __name__ == "__main__":
    main()
