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


RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"
VIEW_FILES = (
    "index.html",
    "pd_insighter_session_review.html",
    "pd_insighter_manual_labeling.html",
    "pd_insighter_stride_review.html",
    "pd_insighter_transition_review.html",
    "pd_insighter_parkinson_review.html",
)
VIEW_TARGETS = {
    "full": ("Full Session", "index.html"),
    "manual": ("Manual Labels", "pd_insighter_manual_labeling.html"),
    "stride": ("Stride", "pd_insighter_stride_review.html"),
    "transitions": ("Transitions", "pd_insighter_transition_review.html"),
    "parkinson": ("Parkinson Review", "pd_insighter_parkinson_review.html"),
}
SIMULATED_TRIALS = {"Ntrial10", "Parki", "Parki2"}


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


def _shared_review_shell(document: str, trial_dir: Path, active_view: str) -> str:
    if trial_dir.name not in SIMULATED_TRIALS:
        return document
    document = document.replace('class="page-nav"', 'class="page-nav utility-bar"')
    document = document.replace('class="nav-trial-switcher"', 'class="nav-trial-switcher trial-switcher"')
    for element_id, filename in (("fullReviewLink", "index.html"),
                                 ("strideReviewLink", "pd_insighter_stride_review.html"),
                                 ("transitionReviewLink", "pd_insighter_transition_review.html")):
        document = document.replace(f'id="{element_id}" class="nav-tab" href="#"',
                                    f'id="{element_id}" class="nav-tab" href="{filename}"')
    style = '''<style id="shared-simulation-navigation">
    .utility-bar { display:flex; flex-wrap:wrap; align-items:center; gap:12px; position:sticky; top:10px; z-index:30; padding:12px 14px; margin-bottom:20px; background:rgba(255,255,255,.97); border:1px solid #dce5dc; border-radius:22px; box-shadow:0 8px 28px #17251c0b; }
    .utility-bar .nav-context { display:flex; gap:10px; align-items:center; min-width:150px; flex:0 0 auto; }
    .utility-bar .nav-mark { display:grid; place-items:center; width:36px; height:36px; border-radius:12px; background:#167852; color:white; font-size:12px; font-weight:900; }
    .utility-bar .nav-location { display:grid; gap:2px; }
    .utility-bar .nav-location strong { font-size:14px; }
    .utility-bar .nav-location span { font-size:12px; color:#5f6b61; }
    .utility-bar .nav-tabs { display:flex; flex:1 1 540px; min-width:0; justify-content:flex-start; gap:3px; padding:4px; border:1px solid #dce5dc; border-radius:15px; background:#f1f5f1; overflow-x:auto; }
    .utility-bar .nav-tab { display:inline-flex; align-items:center; justify-content:center; min-height:38px; padding:0 11px; border-radius:11px; white-space:nowrap; text-decoration:none; font-size:13px; font-weight:800; color:#52645a; }
    .utility-bar .nav-tab.active { background:#167852; color:white; box-shadow:none; }
    .utility-bar .trial-switcher { display:flex; align-items:center; gap:8px; flex:0 1 270px; min-width:200px; }
    .utility-bar .trial-switcher > span { font-size:11px; font-weight:800; text-transform:uppercase; color:#5f6b61; }
    .utility-bar select { min-width:0; width:100%; max-width:100%; min-height:40px; padding:8px 10px; border:1px solid #dce5dc; border-radius:12px; background:white; font:inherit; font-size:13px; color:#1e2520; }
    @media(max-width:760px) { .utility-bar { position:relative; top:0; } .utility-bar .nav-tabs { order:3; flex-basis:100%; flex-wrap:wrap; overflow:visible; } .utility-bar .nav-tab { flex:1 0 auto; min-height:34px; } .utility-bar .trial-switcher { margin-left:auto; flex:1 1 140px; min-width:140px; } }
    </style>'''
    items = [{"label": "All Trials", "value": "../pd_insighter_trial_selector.html"}]
    for source in sorted(trial_dir.parent.glob("*/session_dashboard_manifest.json")):
        name = source.parent.name
        filename = VIEW_TARGETS[active_view][1]
        has_view = (source.parent / filename).is_file()
        if not has_view:
            filename = "index.html"
        items.append({"label": name + (" (Full Session)" if not has_view else ""),
                      "value": f"../{name}/{filename}", "selected": name == trial_dir.name})
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
    target = trial_dir / "environment_reconstruction" / "index.html"
    ready = target.is_file() and (target.parent / "reconstruction_summary.json").is_file()
    if ready:
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
        contents = re.sub(r'<(?:a|span)\b[^>]*data-view="parkinson"[^>]*>.*?</(?:a|span)>', "", contents, flags=re.DOTALL).rstrip()
        clinical_page = trial_dir / "pd_insighter_parkinson_review.html"
        if clinical_page.exists() or trial_dir.name in SIMULATED_TRIALS:
            href = html.escape(os.path.relpath(clinical_page, page_dir or trial_dir), quote=True)
            contents += ('\n        <span class="nav-tab active" data-view="parkinson" aria-current="page">Parkinson Review</span>'
                         if active_view == "parkinson" else f'\n        <a class="nav-tab" data-view="parkinson" href="{href}">Parkinson Review</a>')
        return f"{match.group(1)}{contents.rstrip()}\n        {tab}\n      </div>"

    document = re.sub(
        r'(<div\b[^>]*class="nav-tabs"[^>]*>)(.*?)</div>',
        replace, document, count=1, flags=re.DOTALL,
    )
    if trial_dir.name in SIMULATED_TRIALS:
        document = re.sub(r'<span class="nav-tab active"(?![^>]*aria-current=)',
                          '<span class="nav-tab active" aria-current="page"', document)
    if trial_dir.name == "Ntrial10":
        document = document.replace("/api/session-manual-segments", "/api/parkinson-manual-segments")
    return _shared_review_shell(document, trial_dir, active_view)


def environment_navigation(builder: Callable[..., str]) -> Callable[..., str]:
    @wraps(builder)
    def wrapped(payload: dict[str, Any], *args: Any, **kwargs: Any) -> str:
        filename = Path(builder.__code__.co_filename).name
        active = ("parkinson" if filename == "build_parkinson_motion_review.py" else
                  "manual" if filename == "build_full_session_manual_labeling.py" else
                  "stride" if builder.__name__ == "build_stride_review_html" else
                  "transitions" if builder.__name__ == "build_transition_review_html" else "full")
        return inject_environment_tab(
            builder(payload, *args, **kwargs), RUNS_ROOT / str(payload["trial_id"]), active_view=active
        )
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
            active = next((key for key, (_, filename) in VIEW_TARGETS.items() if filename == name), "full")
            updated = inject_environment_tab(original, directory, active_view=active)
            if updated != original:
                path.write_text(updated)
                changed.append(path)
    return changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", nargs="*")
    for path in refresh_navigation(parser.parse_args().trials):
        print(f"Updated navigation only: {path}")
