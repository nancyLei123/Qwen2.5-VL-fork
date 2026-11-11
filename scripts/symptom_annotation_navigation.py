"""Navigation pieces for the Symptom Annotation view.

pd_insighter_environment_navigation.py calls these few functions, so the Symptom Annotation tab
follows the same refresh path as the other trial views:

- ``symptom_tab`` adds the tab where ``runs/<trial>/pd_insighter_symptom_annotation.html`` exists
  (or marks it active on that page). Trials without the page get no tab.
- ``view_filename`` maps the active view to its page, including ``"symptoms"``.
- ``add_session_time_link`` keeps the current time when moving from Full Session to Symptom
  Annotation (``?t=<session seconds>``) and seeks there when Full Session is opened with ``?t=``.

Links always use forward slashes, also when pages are generated on Windows.
"""

from __future__ import annotations

import html
import os
import re
from pathlib import Path

SYMPTOM_PAGE = "pd_insighter_symptom_annotation.html"
SYMPTOM_VIEW = "symptoms"
TAB_PATTERN = r'<(?:a|span)\b[^>]*data-view="symptoms"[^>]*>.*?</(?:a|span)>'


def href(target: Path, start: Path) -> str:
    return html.escape(os.path.relpath(target, start).replace(os.sep, "/"), quote=True)


def view_filename(view: str, view_targets: dict[str, tuple[str, str]]) -> str:
    return SYMPTOM_PAGE if view == SYMPTOM_VIEW else view_targets[view][1]


def strip_tab(contents: str) -> str:
    return re.sub(TAB_PATTERN, "", contents, flags=re.DOTALL)


def symptom_tab(trial_dir: Path, page_dir: Path | None, active_view: str) -> str:
    """The tab markup to append to a trial's nav tabs, or "" when the trial has no page."""
    if active_view == SYMPTOM_VIEW:
        return '\n        <span class="nav-tab active" data-view="symptoms" aria-current="page">Symptom Annotation</span>'
    page = trial_dir / SYMPTOM_PAGE
    if not page.is_file():
        return ""
    return f'\n        <a class="nav-tab" data-view="symptoms" href="{href(page, page_dir or trial_dir)}">Symptom Annotation</a>'


SESSION_TIME_SCRIPT = """<script id="pd-insighter-session-time">
    (() => {
      // Keeps the moment when moving between Full Session and Symptom Annotation:
      // the tab link carries ?t=<session seconds>, and ?t= on this page seeks there.
      if (typeof seekSession !== "function" || typeof currentTime === "undefined") return;
      for (const link of document.querySelectorAll('a[data-view="symptoms"]')) {
        const update = () => {
          const url = new URL(link.getAttribute("href"), location.href);
          url.searchParams.set("t", (Number(currentTime) || 0).toFixed(3));
          link.href = url.href;
        };
        for (const name of ["pointerdown", "focus"]) link.addEventListener(name, update);
      }
      const params = new URLSearchParams(location.search);
      const t = Number(params.get("t"));
      if (!params.has("t") || !Number.isFinite(t)) return;
      const start = () => setTimeout(() => {
        Promise.resolve(seekSession(t, { preservePlayback: false }))
          .then(() => { document.documentElement.dataset.restoredSessionTime = String(currentTime); })
          .catch(error => console.error(error));
      }, 400);
      if (ariaVideo.readyState >= 1 && document.readyState === "complete") start();
      else window.addEventListener("load", () => {
        if (ariaVideo.readyState >= 1) start();
        else ariaVideo.addEventListener("loadedmetadata", start, { once: true });
      }, { once: true });
    })();
    </script>"""


def add_session_time_link(document: str, trial_dir: Path, active_view: str) -> str:
    """Full Session only: idempotent, and only where the trial has a Symptom Annotation page."""
    document = re.sub(r'<script id="pd-insighter-session-time">.*?</script>', "", document, flags=re.S)
    if active_view == "full" and (trial_dir / SYMPTOM_PAGE).is_file() and 'id="ariaVideo"' in document:
        document = document.replace("</body>", SESSION_TIME_SCRIPT + "</body>", 1)
    return document
