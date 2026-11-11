"""Connect trial views and shared navigation without rebuilding trial analysis."""

from __future__ import annotations

import argparse
from functools import wraps
import html
import json
import os
from pathlib import Path
import re
from typing import Any, Callable
from pd_insighter_audio import inject_page_audio
import symptom_annotation_navigation as symptoms


RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"
VIEW_FILES = (
    "index.html",
    "pd_insighter_session_review.html",
    "pd_insighter_manual_labeling.html",
    "pd_insighter_stride_review.html",
    "pd_insighter_transition_review.html",
    "pd_insighter_parkinson_review.html",
    symptoms.SYMPTOM_PAGE,
)
VIEW_TARGETS = {
    "full": ("Full Session", "index.html"),
    "manual": ("Manual Labels", "pd_insighter_manual_labeling.html"),
    "stride": ("Stride", "pd_insighter_stride_review.html"),
    "transitions": ("Transitions", "pd_insighter_transition_review.html"),
    "parkinson": ("Parkinson Review", "pd_insighter_parkinson_review.html"),
}
SIMULATED_TRIALS = {"Ntrial10", "Parki", "Parki2", "Fuchs", "Fuchs22", "Fuchs333"}


def adapt_aria_only(document: str, trial_dir: Path, active_view: str) -> str:
    """Keep legacy controls present, but never load or wait for an absent camera."""
    manifest_path = trial_dir / "trial_manifest.json"
    if not manifest_path.is_file() or json.loads(manifest_path.read_text()).get("media_mode") != "aria_only":
        return document
    document = re.sub(r'(downwardVideo\.src\s*=\s*[^;]+;)', r'/* No downward recording supplied. */', document)
    document = document.replace('id="downwardVideo" src=""', 'id="downwardVideo"')
    document = document.replace("downwardVideo.play()", "(downwardVideo.getAttribute('src') ? downwardVideo.play() : Promise.resolve())")
    document = document.replace("function syncFollower(force = false) {", "function syncFollower(force = false) {\n      if (!downwardVideo.getAttribute('src')) return;")
    document = document.replace("async function seekVideoPrecisely(videoEl, targetSeconds) {", "async function seekVideoPrecisely(videoEl, targetSeconds) {\n      if (!videoEl.getAttribute('src')) return;")
    document = document.replace('renderFrameStrip(downwardFrameStrip, frames, "downward");', 'downwardFrameStrip.replaceChildren();')
    document = document.replace("One shared transport controls both views.", "The transport controls the Aria video.")
    document = document.replace("Review the synchronized videos", "Review the Aria video")
    document = document.replace("seek both videos", "seek the Aria video")
    document = document.replace("with the synchronized videos", "with the Aria video")
    document = document.replace("start the synchronized videos anywhere", "start the Aria video anywhere")
    document = document.replace("Use the downward video to check whether each detected side-to-side peak matches a real step.", "Foot-contact validation is pending because a downward-facing video is not available.")
    document = document.replace("Please validate with the downward view.", "Foot-contact validation requires a matching downward video.")
    if active_view == "parkinson":
        document = document.replace("downwardVideo.currentTime || 0", "ariaVideo.currentTime || 0")
        document = document.replace("downwardVideo.paused ?", "ariaVideo.paused ?")
        document = document.replace("function syncVideos(source) {", "function syncVideos(source) {\n  if (!downwardVideo.getAttribute('src')) return;")
        document = document.replace("downwardVideo.addEventListener('play',", "ariaVideo.addEventListener('play',")
        document = document.replace("ariaVideo.addEventListener('timeupdate', () => syncVideos(ariaVideo));", "ariaVideo.addEventListener('timeupdate', () => { updatePlayhead(); });")
    style = '''<style id="aria-only-layout">
      .view-card:has(#downwardVideo), .video-card:has(#downwardVideo),
      #rotateDownwardButton, #rotateDownwardBtn, #syncStatus, #timeInformationSection,
      .frame-strip-card:has(#downwardFrameStrip) { display:none!important; }
      .media-grid, .video-grid, .frame-review-grid { grid-template-columns:minmax(0,1fr)!important; }
      .media-grid .view-frame, .video-grid .video-frame, .video-grid .video-wrap { max-height:560px; }
      .aria-only-notice { margin:12px 0; padding:12px 18px; border:1px solid #d8e4da; border-radius:14px; background:#f3f7f1; color:#385246; font-size:14px; }
    </style>'''
    notice = '''<div class="aria-only-notice" role="note"><strong>Aria-only recording.</strong> A matching downward-facing video has not been provided. Automatic classification, transitions and head-motion-derived stride estimates need manual validation; no accuracy score or measured foot contacts are available.</div>'''
    document = document.replace('</head>', style + '</head>')
    return re.sub(r'(<nav\b.*?</nav>)', lambda m: m[0] + notice, document, count=1, flags=re.S)


def adapt_partial_downward(document: str, trial_dir: Path) -> str:
    """Explain black padding in aligned media without changing any review clock."""
    path = trial_dir / "trial_manifest.json"
    if not path.is_file():
        return document
    attachment = json.loads(path.read_text()).get("downward_attachment")
    if not attachment or 'id="downwardVideo"' not in document:
        return document
    start, end = float(attachment["start_s"]), float(attachment["end_s"])
    duration = float(attachment["session_duration_s"])
    rotation = int(attachment.get("rotation_quarter_turns", 0)) % 4
    # Older players hard-code a 180-degree mount. These supplied files are upright.
    for variable in ("downwardRotationQuarterTurns", "downwardTurns"):
        document = document.replace(f"{variable} = 2;", f"{variable} = {rotation};")
    document = document.replace("#downwardVideo { transform: rotate(180deg); }",
                                f"#downwardVideo {{ transform: rotate({rotation * 90}deg); }}")
    document = re.sub(r'^[^\n]*localStorage.getItem\(`pd-insighter-downward-rotation-v2:[^\n]*',
                      lambda m: m[0].replace(" || 2", f" ?? {rotation}").replace("v2:", "v3:"),
                      document, flags=re.M)
    document = document.replace("setItem(`pd-insighter-downward-rotation-v2:", "setItem(`pd-insighter-downward-rotation-v3:")
    def clock(seconds):
        return f"{int(seconds // 60)}:{seconds % 60:04.1f}"
    notice = f'''<section class="downward-coverage" aria-label="Downward recording coverage">
      <div><strong>Downward footage: {clock(start)} - {clock(end)}</strong>
      <span> on the unchanged Aria timeline. Striped sections have no downward recording.</span></div>
      <div class="downward-coverage-track" role="img" aria-label="Downward footage available from {clock(start)} to {clock(end)}; other times unavailable">
        <span class="downward-coverage-available" style="left:{start / duration * 100:.6f}%;width:{(end-start) / duration * 100:.6f}%"></span>
        <span class="downward-coverage-cursor"></span></div>
      <div class="downward-coverage-labels"><span>0:00</span><span>{clock(duration)}</span></div>
    </section>'''
    style = '''<style id="partial-downward-style">
      .downward-coverage { margin:14px 0; padding:14px 18px; border:1px solid #d8e4da; border-radius:16px; background:#f6f8f2; color:#385246; font-size:14px; }
      .downward-coverage-track { position:relative; height:12px; margin-top:10px; border-radius:5px; overflow:hidden; background:repeating-linear-gradient(135deg,#dfe4df 0,#dfe4df 5px,#f7f8f6 5px,#f7f8f6 10px); }
      .downward-coverage-available { position:absolute; top:0; bottom:0; background:#167a55; }
      .downward-coverage-cursor { position:absolute; width:2px; top:0; bottom:0; background:#111; }
      .downward-coverage-labels { display:flex; justify-content:space-between; font-size:11px; margin-top:3px; }
      .downward-gap-overlay { position:absolute; inset:0; align-items:center; justify-content:center; text-align:center; padding:24px; background:#14241d; color:#f0f5ee; z-index:2; pointer-events:none; font:600 16px/1.5 sans-serif; }
      .downward-gap-overlay[hidden] { display:none!important; }
      .media-grid .view-card { grid-template-rows:auto auto minmax(0,1fr); }
    </style>'''
    script = '''<script id="partial-downward-coverage-script">
    (() => {
      const coverage = __COVERAGE__;
      const master = document.getElementById('ariaVideo');
      const follower = document.getElementById('downwardVideo');
      if (!master || !follower) return;
      const frame = follower.parentElement;
      frame.style.position = 'relative';
      const overlay = document.createElement('div');
      overlay.className = 'downward-gap-overlay';
      overlay.setAttribute('role', 'status');
      frame.appendChild(overlay);
      const cursor = document.querySelector('.downward-coverage-cursor');
      let previous = '';
      function updateCoverage() {
        const t = Number(master.currentTime || 0);
        const available = t >= coverage.start && t < coverage.end;
        const state = available ? 'available' : t < coverage.start ? 'before' : 'after';
        if (cursor) cursor.style.left = `${Math.min(100, Math.max(0, t / coverage.duration * 100))}%`;
        if (state === previous) return;
        previous = state;
        overlay.hidden = available;
        overlay.style.display = available ? 'none' : 'flex';
        overlay.textContent = available ? '' : 'No downward recording at this time. Aria video and analyses remain available.';
        follower.dataset.coverage = state;
      }
      ['timeupdate', 'seeking', 'seeked', 'loadedmetadata'].forEach(event => master.addEventListener(event, updateCoverage));
      updateCoverage();
    })();
    </script>'''.replace("__COVERAGE__", json.dumps({"start": start, "end": end, "duration": duration}))
    document = document.replace("</head>", style + "</head>")
    document = re.sub(r'(<nav\b.*?</nav>)', lambda m: m[0] + notice, document, count=1, flags=re.S)
    return document.replace("</body>", script + "</body>")


def action_label_path(trial_dir: Path) -> Path:
    # Keep the original Ntrial10 annotation store and timestamps, without migration.
    if trial_dir.name == "Ntrial10":
        return trial_dir / "parkinson_motion_review/manual_segments.json"
    return trial_dir / "manual_labels/session_manual_segments.json"


def parkinson_navigation(trial_id: str) -> str:
    tabs = "".join(f'<a class="nav-tab" href="{filename}">{label}</a>'
                   for key, (label, filename) in VIEW_TARGETS.items() if key != "parkinson")
    return f'''<nav class="card utility-bar" aria-label="PD Insighter navigation">
      <div class="nav-context"><span class="nav-mark" aria-hidden="true">PD</span>
        <div class="nav-location"><strong>{html.escape(trial_id)}</strong><span>Parkinson Review</span></div></div>
      <div class="nav-tabs" aria-label="Trial views">{tabs}</div>
      <label class="trial-switcher" for="trialJumpSelect"><span>Trial</span>
        <select id="trialJumpSelect" aria-label="Switch trial or return to all trials" onchange="if(this.value) window.location.href=this.value"></select></label>
    </nav>'''


def environment_entry(trial_dir: Path) -> Path | None:
    directory = trial_dir / "environment_reconstruction"
    if (directory / "playback.html").is_file() and (directory / "scene_playback_data.json").is_file():
        return directory / "playback.html"
    if (directory / "index.html").is_file() and (directory / "reconstruction_summary.json").is_file():
        return directory / "index.html"
    return None


def _is_redirect(document: str) -> bool:
    return bool(re.search(r'<meta\b[^>]*http-equiv=[\"\']refresh[\"\']', document, re.I))


def _is_redirect_file(path: Path) -> bool:
    # Generated redirect stubs put this meta tag first; avoid reading embedded plots.
    with path.open() as stream:
        return _is_redirect(stream.read(4096))


def _shared_review_shell(document: str, trial_dir: Path, active_view: str, page_dir: Path | None = None) -> str:
    page_dir = page_dir or trial_dir
    document = document.replace('class="page-nav"', 'class="page-nav utility-bar"')
    document = document.replace('class="nav-trial-switcher"', 'class="nav-trial-switcher trial-switcher"')
    for element_id, filename in (("fullReviewLink", "index.html"),
                                 ("strideReviewLink", "pd_insighter_stride_review.html"),
                                 ("transitionReviewLink", "pd_insighter_transition_review.html")):
        document = document.replace(f'id="{element_id}" class="nav-tab" href="#"',
                                    f'id="{element_id}" class="nav-tab" href="{html.escape(os.path.relpath(trial_dir / filename, page_dir), quote=True)}"')
    style = '''<style id="shared-simulation-navigation">
    .page:has(> .utility-bar), .shell:has(> .utility-bar), main:has(> .utility-bar) { box-sizing:border-box; width:100%; max-width:1800px; margin:0 auto; padding:22px; }
    .utility-bar { box-sizing:border-box; display:flex; flex-wrap:wrap; align-items:center; gap:12px; position:sticky; top:10px; z-index:30; padding:12px 14px; margin-bottom:20px; background:rgba(255,255,255,.97); border:1px solid #dce5dc; border-radius:22px; box-shadow:0 8px 28px #17251c0b; font-family:"Avenir Next","Gill Sans",sans-serif; line-height:1.25; }
    .utility-bar * { box-sizing:border-box; }
    .utility-bar .nav-context { display:flex; gap:10px; align-items:center; min-width:150px; flex:0 0 auto; }
    .utility-bar .nav-mark { display:grid; place-items:center; width:36px; height:36px; border-radius:12px; background:#167852; color:white; font-size:12px; font-weight:900; }
    .utility-bar .nav-location { display:grid; gap:2px; line-height:1.25; }
    .utility-bar .nav-location strong { font-size:14px; font-weight:800; }
    .utility-bar .nav-location span { font-size:12px; font-weight:500; color:#5f6b61; }
    .utility-bar .nav-tabs { display:flex; flex:1 1 540px; min-width:0; align-items:center; justify-content:flex-start; gap:3px; padding:4px; border:1px solid #dce5dc; border-radius:15px; background:#f1f5f1; overflow-x:auto; }
    .utility-bar .nav-tab { display:inline-flex; align-items:center; justify-content:center; min-height:38px; padding:0 11px; border:0; border-radius:11px; white-space:nowrap; text-decoration:none; font-size:13px; font-weight:800; color:#52645a; background:transparent; }
    .utility-bar .nav-tab.active { background:#167852; color:white; box-shadow:none; }
    .utility-bar .nav-tab[aria-disabled="true"] { opacity:.45; cursor:not-allowed; }
    .utility-bar a.nav-tab:hover { background:#e4ede6; color:#234f3b; }
    .utility-bar a:focus-visible, .utility-bar select:focus-visible { outline:3px solid #cc9629; outline-offset:2px; }
    .utility-bar .trial-switcher { display:flex; align-items:center; gap:8px; flex:0 1 270px; min-width:200px; }
    .utility-bar .trial-switcher > span { font-size:11px; font-weight:800; text-transform:uppercase; color:#5f6b61; }
    .utility-bar select { min-width:0; width:100%; max-width:100%; min-height:40px; padding:8px 10px; border:1px solid #dce5dc; border-radius:12px; background:white; font:inherit; font-size:13px; color:#1e2520; }
    .environment-tools { display:flex; flex-wrap:wrap; align-items:center; gap:8px 18px; font-size:13px; margin:0 0 12px; color:#63756b; }
    .environment-tools a { color:#187859; font-weight:700; text-underline-offset:3px; }
    @media(max-width:760px) { .page:has(> .utility-bar), .shell:has(> .utility-bar), main:has(> .utility-bar) { padding:10px; } .utility-bar { position:relative; top:0; } .utility-bar .nav-tabs { order:3; flex-basis:100%; flex-wrap:wrap; overflow:visible; } .utility-bar .nav-tab { flex:1 0 auto; min-height:34px; } .utility-bar .trial-switcher { margin-left:auto; flex:1 1 140px; min-width:140px; } }
    </style>'''
    items = [{"label": "All Trials", "value": os.path.relpath(trial_dir.parent / "pd_insighter_trial_selector.html", page_dir)}]
    directories = {p.parent for p in trial_dir.parent.glob("*/session_dashboard_manifest.json")}
    directories.update(p.parent.parent for p in trial_dir.parent.glob("*/environment_reconstruction/scene_playback_data.json"))
    directories.add(trial_dir)
    for source in sorted(directories):
        target = environment_entry(source) if active_view == "environment" else source / symptoms.view_filename(active_view, VIEW_TARGETS)
        suffix = ""
        if target is None or not target.is_file() or (active_view != "environment" and _is_redirect_file(target)):
            target = source / "index.html"
            suffix = " (Full Session)"
            if not target.is_file() or _is_redirect_file(target):
                target = environment_entry(source)
                suffix = " (Environment)"
        if target is None or not target.is_file():
            continue
        items.append({"label": source.name + suffix,
                      "value": os.path.relpath(target, page_dir), "selected": source == trial_dir})
    script = '''<script id="shared-simulation-trial-switcher">
    document.addEventListener('DOMContentLoaded', () => {
      const select = document.getElementById('trialJumpSelect');
      if (!select) return;
      select.replaceChildren();
      __ITEMS__.forEach(item => {
        const option = document.createElement('option');
        option.value = item.value; option.textContent = item.label; option.selected = !!item.selected;
        select.appendChild(option);
      });
    });
    </script>'''.replace("__ITEMS__", json.dumps(items).replace("<", "\\u003c"))
    document = re.sub(r'<style id="shared-simulation-navigation">.*?</style>', '', document, flags=re.S)
    document = re.sub(r'<script id="shared-simulation-trial-switcher">.*?</script>', '', document, flags=re.S)
    return document.replace('</head>', style + '</head>').replace('</body>', script + '</body>')


def inject_environment_tab(document: str, trial_dir: Path, page_dir: Path | None = None, active_view: str = "full") -> str:
    target = environment_entry(trial_dir)
    if active_view == "environment":
        tab = '<span class="nav-tab active" data-view="environment" aria-current="page">Environment</span>'
    elif target is not None:
        href = html.escape(os.path.relpath(target, page_dir or trial_dir), quote=True)
        tab = f'<a class="nav-tab" data-view="environment" href="{href}">Environment</a>'
    else:
        tab = (
            '<span class="nav-tab" data-view="environment" aria-disabled="true" '
            'title="Environment reconstruction has not been generated for this trial" '
            'style="opacity:.45;cursor:not-allowed">Environment</span>'
        )

    def replace(match: re.Match[str]) -> str:
        contents = re.sub(
            r'<(?:a|span)\b[^>]*data-view="environment"[^>]*>.*?</(?:a|span)>',
            "", match.group(2), flags=re.DOTALL,
        )
        contents = re.sub(r'<(?:a|span)\b[^>]*data-view="parkinson"[^>]*>.*?</(?:a|span)>', "", contents, flags=re.DOTALL)
        contents = symptoms.strip_tab(contents).rstrip()
        clinical_page = trial_dir / "pd_insighter_parkinson_review.html"
        if clinical_page.exists() or trial_dir.name in SIMULATED_TRIALS:
            href = html.escape(os.path.relpath(clinical_page, page_dir or trial_dir), quote=True)
            contents += ('\n        <span class="nav-tab active" data-view="parkinson" aria-current="page">Parkinson Review</span>'
                         if active_view == "parkinson" else f'\n        <a class="nav-tab" data-view="parkinson" href="{href}">Parkinson Review</a>')
        contents += symptoms.symptom_tab(trial_dir, page_dir, active_view)
        return f"{match.group(1)}{contents.rstrip()}\n        {tab}\n      </div>"

    document = re.sub(
        r'(<div\b[^>]*class="nav-tabs"[^>]*>)(.*?)</div>',
        replace, document, count=1, flags=re.DOTALL,
    )
    document = re.sub(r'<span class="nav-tab active"(?![^>]*aria-current=)',
                      '<span class="nav-tab active" aria-current="page"', document)
    if trial_dir.name == "Ntrial10":
        document = document.replace("/api/session-manual-segments", "/api/parkinson-manual-segments")
    document = symptoms.add_session_time_link(document, trial_dir, active_view)
    return inject_page_audio(_shared_review_shell(document, trial_dir, active_view, page_dir), page_dir or trial_dir)


def inject_environment_navigation(document: str, trial_dir: Path, page_path: Path) -> str:
    """Keep reconstruction views in the same trial shell; never rebuild their data."""
    if _is_redirect(document):
        return document
    tabs = []
    for key, (label, filename) in VIEW_TARGETS.items():
        if key == "parkinson":
            continue  # Added by inject_environment_tab when the review is available.
        target = trial_dir / filename
        if target.is_file() and not _is_redirect_file(target):
            href = html.escape(os.path.relpath(target, page_path.parent), quote=True)
            tabs.append(f'<a class="nav-tab" href="{href}">{label}</a>')
        else:
            tabs.append(f'<span class="nav-tab" aria-disabled="true" title="Not available for this trial">{label}</span>')
    nav = f'''<nav class="card utility-bar" aria-label="PD Insighter navigation">
      <div class="nav-context"><span class="nav-mark" aria-hidden="true">PD</span>
        <div class="nav-location"><strong>{html.escape(trial_dir.name)}</strong><span>Environment</span></div></div>
      <div class="nav-tabs" aria-label="Trial views">{"".join(tabs)}</div>
      <label class="trial-switcher" for="trialJumpSelect"><span>Trial</span>
        <select id="trialJumpSelect" aria-label="Switch trial or return to all trials" onchange="if(this.value) window.location.href=this.value"></select></label>
    </nav>'''
    if '<!-- PD_INSIGHTER_NAVIGATION -->' in document:
        document = document.replace('<!-- PD_INSIGHTER_NAVIGATION -->', nav, 1)
    else:
        document = re.sub(r'<nav\b.*?</nav>', lambda _: nav, document, count=1, flags=re.S)
    # Reconstruction-specific links belong below the main tabs, not in a second nav.
    document = re.sub(r'<div class="environment-tools".*?</div>', '', document, flags=re.S)
    links = []
    directory = trial_dir / "environment_reconstruction"
    for label, target in (("Synchronized playback", directory / "playback.html"),
                          ("Methods & results", directory / "index.html"),
                          ("Submaps", directory / "gaussian_submaps/index.html"),
                          ("Quality comparison", directory / "quality_20261029/index.html")):
        if target != page_path and target.is_file() and not _is_redirect_file(target):
            href = html.escape(os.path.relpath(target, page_path.parent), quote=True)
            links.append(f'<a href="{href}">{html.escape(label)}</a>')
    if links:
        local_links = '<div class="environment-tools" aria-label="Reconstruction resources">' + ''.join(links) + '</div>'
        document = re.sub(r'(</header>)', lambda m: m[0] + local_links, document, count=1)
    return inject_environment_tab(document, trial_dir, page_path.parent, "environment")


def environment_navigation(builder: Callable[..., str]) -> Callable[..., str]:
    @wraps(builder)
    def wrapped(payload: dict[str, Any], *args: Any, **kwargs: Any) -> str:
        filename = Path(builder.__code__.co_filename).name
        active = ("parkinson" if filename == "build_parkinson_motion_review.py" else
                  "manual" if filename == "build_full_session_manual_labeling.py" else
                  "stride" if builder.__name__ == "build_stride_review_html" else
                  "transitions" if builder.__name__ == "build_transition_review_html" else "full")
        trial_dir = RUNS_ROOT / str(payload["trial_id"])
        document = inject_environment_tab(builder(payload, *args, **kwargs), trial_dir, active_view=active)
        return adapt_partial_downward(adapt_aria_only(document, trial_dir, active), trial_dir)
    return wrapped


def refresh_navigation(trial_ids: list[str] | None = None) -> list[Path]:
    directories = [RUNS_ROOT / name for name in trial_ids] if trial_ids else sorted(RUNS_ROOT.iterdir())
    changed = []
    for directory in directories:
        for name in VIEW_FILES:
            path = directory / name
            if not path.is_file():
                continue
            original = path.read_text()
            if _is_redirect(original):
                continue
            active = symptoms.SYMPTOM_VIEW if name == symptoms.SYMPTOM_PAGE else next(
                (key for key, (_, filename) in VIEW_TARGETS.items() if filename == name), "full")
            updated = inject_environment_tab(original, directory, active_view=active)
            if updated != original:
                path.write_text(updated)
                changed.append(path)
        for name in ("index.html", "playback.html"):
            path = directory / "environment_reconstruction" / name
            if not path.is_file():
                continue
            original = path.read_text()
            updated = inject_environment_navigation(original, directory, path)
            if updated != original:
                path.write_text(updated)
                changed.append(path)
    return changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", nargs="*")
    for path in refresh_navigation(parser.parse_args().trials):
        print(f"Updated navigation only: {path}")
