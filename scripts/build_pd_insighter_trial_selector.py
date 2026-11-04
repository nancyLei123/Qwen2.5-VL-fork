#!/usr/bin/env python3
"""Build a multi-trial PD-Insighter selector page.

The selector is kept separate from individual trial dashboards so that:
- each trial keeps its own self-contained HTML output
- adding a new trial does not break old links
- future analyses can simply rebuild this index automatically
"""

from __future__ import annotations

import json
from pathlib import Path


QWEN_REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = QWEN_REPO / "runs"
FULL_SELECTOR_HTML = RUNS_DIR / "pd_insighter_trial_selector.html"
CLIP_SELECTOR_HTML = RUNS_DIR / "pd_insighter_trial_selector_clip.html"
LEGACY_SELECTOR_HTML = RUNS_DIR / "pd_insighter_trial_selector_legacy.html"
FULL_SESSION_DASHBOARD_NAME = "pd_insighter_session_review.html"
CANONICAL_TRIAL_ENTRY_NAME = "index.html"
LEGACY_DASHBOARD_NAME = "pd_insighter_rule_interactive.html"
ADDITIONAL_NORMAL_TRIALS = {"Half_hour", "Half_hour2", "60minutes_1", "60minutes_2"}


def trial_sort_key(trial_id: str) -> tuple[int, str]:
    if trial_id.startswith("trial"):
        suffix = trial_id.removeprefix("trial")
    elif trial_id.startswith("Ntrial"):
        suffix = trial_id.removeprefix("Ntrial")
    else:
        suffix = trial_id
    try:
        return int(suffix), trial_id
    except ValueError:
        return 10**9, trial_id


def versioned_href(path: Path, rel_href: str) -> str:
    try:
        version = int(path.stat().st_mtime)
    except FileNotFoundError:
        return rel_href
    joiner = "&" if "?" in rel_href else "?"
    return f"{rel_href}{joiner}v={version}"


def collect_trials() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for manifest_path in sorted(RUNS_DIR.glob("*/trial_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        run_dir = manifest_path.parent
        full_dashboard_path = run_dir / CANONICAL_TRIAL_ENTRY_NAME
        if not full_dashboard_path.exists():
            full_dashboard_path = run_dir / FULL_SESSION_DASHBOARD_NAME
        if not full_dashboard_path.exists():
            continue
        trial_id = str(manifest.get("trial_id", run_dir.name))
        if not trial_id.startswith("Ntrial") and trial_id not in ADDITIONAL_NORMAL_TRIALS and manifest.get("trial_type") != "parkinson_simulated":
            continue
        recording_stem = str(manifest.get("recording_stem", run_dir.name))
        plots_dir = run_dir / "plots"
        parkinson_review_path = run_dir / "pd_insighter_parkinson_review.html"
        manual_labeling_path = run_dir / "pd_insighter_manual_labeling.html"
        environment_reconstruction_path = run_dir / "environment_reconstruction" / "index.html"
        item = {
            "trial_id": trial_id,
            "recording_stem": recording_stem,
            "label": f"{trial_id} - {recording_stem}",
            "full_dashboard_href": versioned_href(
                full_dashboard_path,
                f"{trial_id}/{full_dashboard_path.name}",
            ),
            "metrics_href": f"{trial_id}/clip_metrics.csv",
            "timeline_href": f"{trial_id}/timeline_predictions.json",
            "point_cloud_href": versioned_href(
                plots_dir / "point_cloud_3d.png",
                f"{trial_id}/plots/point_cloud_3d.png",
            ),
            "horizontal_href": versioned_href(
                plots_dir / "horizontal_displacement_by_clip.png",
                f"{trial_id}/plots/horizontal_displacement_by_clip.png",
            ),
            "vertical_href": versioned_href(
                plots_dir / "vertical_displacement_by_clip.png",
                f"{trial_id}/plots/vertical_displacement_by_clip.png",
            ),
            "parkinson_review_href": versioned_href(
                parkinson_review_path,
                f"{trial_id}/pd_insighter_parkinson_review.html",
            )
            if parkinson_review_path.exists()
            else "",
            "manual_labeling_href": versioned_href(
                manual_labeling_path,
                f"{trial_id}/pd_insighter_manual_labeling.html",
            )
            if manual_labeling_path.exists()
            else "",
            "environment_reconstruction_href": versioned_href(
                environment_reconstruction_path,
                f"{trial_id}/environment_reconstruction/index.html",
            )
            if environment_reconstruction_path.exists()
            else "",
            "plots_exist": str(plots_dir.exists()).lower(),
        }
        items.append(item)

    return sorted(items, key=lambda item: trial_sort_key(item["trial_id"]))


def collect_legacy_trials() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for manifest_path in sorted(RUNS_DIR.glob("*/trial_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        run_dir = manifest_path.parent
        dashboard_path = run_dir / LEGACY_DASHBOARD_NAME
        if not dashboard_path.exists():
            continue

        trial_id = str(manifest.get("trial_id", run_dir.name))
        if not trial_id.startswith("trial"):
            continue
        recording_stem = str(manifest.get("recording_stem", run_dir.name))
        items.append(
            {
                "trial_id": trial_id,
                "recording_stem": recording_stem,
                "label": f"{trial_id} - {recording_stem}",
                "legacy_dashboard_href": versioned_href(
                    dashboard_path,
                    f"{trial_id}/{dashboard_path.name}",
                ),
            }
        )
    return sorted(items, key=lambda item: trial_sort_key(item["trial_id"]))


def build_html(
    trials: list[dict[str, str]],
    *,
    page_title: str,
    hero_title: str,
    hero_copy: str,
    selector_heading: str,
    selector_copy: str,
    target_key: str,
) -> str:
    trials_json = json.dumps(trials)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{page_title}</title>
  <style>
    :root {{
      --bg: #eef1f3;
      --card: #ffffff;
      --text: #1b1b1b;
      --muted: #65707a;
      --border: #d8dde3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .page {{
      max-width: 1680px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      gap: 18px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.06);
    }}
    .hero {{
      padding: 22px;
      display: grid;
      gap: 10px;
    }}
    .hero h1 {{
      margin: 0;
      font-size: 30px;
      line-height: 1.1;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
    }}
    .selector-card {{
      padding: 18px;
      display: grid;
      gap: 14px;
    }}
    .selector-row {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 12px;
    }}
    .selector-copy h2 {{
      margin: 0;
      font-size: 18px;
    }}
    .selector-copy p {{
      margin: 4px 0 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .selector-controls {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: flex-start;
      gap: 10px;
    }}
    .trial-select {{
      width: min(420px, 100%);
      min-width: 260px;
      max-width: 100%;
      padding: 11px 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      font-size: 14px;
      background: white;
    }}
    .helper {{
      color: var(--muted);
      font-size: 13px;
    }}
    .special-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .special-links a {{
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 10px 13px;
      color: var(--text);
      background: #f8faf9;
      text-decoration: none;
      font-weight: 700;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="card hero">
      <h1>{hero_title}</h1>
      <p>{hero_copy}</p>
    </section>
    <section class="card selector-card">
      <div class="selector-row">
        <div class="selector-copy">
          <h2>{selector_heading}</h2>
          <p>{selector_copy}</p>
        </div>
        <div class="selector-controls">
          <select id="trialSelect" class="trial-select"></select>
        </div>
      </div>
      <div class="helper" id="helperText"></div>
      <div class="special-links" id="specialLinks"></div>
    </section>
  </div>
  <script>
    const trials = {trials_json};
    const trialSelect = document.getElementById("trialSelect");
    const helperText = document.getElementById("helperText");
    const specialLinks = document.getElementById("specialLinks");

    function buildSelector() {{
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = "Choose a trial…";
      placeholder.selected = true;
      trialSelect.appendChild(placeholder);

      for (const item of trials) {{
        const option = document.createElement("option");
        option.value = item.trial_id;
        option.textContent = item.label;
        trialSelect.appendChild(option);
      }}

      helperText.textContent = trials.length
        ? `Currently available: ${{trials.map(t => t.trial_id).join(", ")}}`
        : "No analyzed trials are available yet.";
      const specialReviews = trials.filter(t => t.parkinson_review_href);
      const manualReviews = trials.filter(t => t.manual_labeling_href);
      const environmentReviews = trials.filter(t => t.environment_reconstruction_href);
      const specialReviewLinks = [
        ...environmentReviews.map(t => `<a href="${{t.environment_reconstruction_href}}">${{t.trial_id}} Environment 3D</a>`),
        ...manualReviews.map(t => `<a href="${{t.manual_labeling_href}}">${{t.trial_id}} Manual Labeling</a>`),
        ...specialReviews.map(t => `<a href="${{t.parkinson_review_href}}">${{t.trial_id}} Parkinson Review</a>`)
      ];
      specialLinks.innerHTML = specialReviewLinks.join("");
      trialSelect.addEventListener("change", (event) => {{
        const item = trials.find(t => t.trial_id === event.target.value);
        if (item && item["{target_key}"]) {{
          window.location.href = item["{target_key}"];
        }}
      }});
    }}

    buildSelector();
  </script>
</body>
</html>
"""


def build_redirect_html(target_href: str, message: str) -> str:
    escaped_target = target_href.replace('"', "&quot;")
    escaped_message = message.replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta http-equiv="refresh" content="0; url={escaped_target}" />
  <title>Redirecting</title>
  <script>window.location.replace("{escaped_target}");</script>
</head>
<body>
  <p>{escaped_message}</p>
</body>
</html>
"""


def main() -> None:
    trials = collect_trials()
    legacy_trials = collect_legacy_trials()
    FULL_SELECTOR_HTML.write_text(
        build_html(
            trials,
            page_title="PD-Insighter Trial Selector",
            hero_title="PD-Insighter Trial Selector",
            hero_copy="Select a trial and it will open the canonical full-session clinician review. Newly analyzed trials will appear here automatically when the selector is rebuilt.",
            selector_heading="Full-Session Review",
            selector_copy="Choose a processed trial to open its full-session review directly.",
            target_key="full_dashboard_href",
        )
    )
    CLIP_SELECTOR_HTML.write_text(
        build_redirect_html(
            "pd_insighter_trial_selector.html",
            "Clip-focused review has been retired. Redirecting to the full-session trial selector.",
        )
    )
    LEGACY_SELECTOR_HTML.write_text(
        build_html(
            legacy_trials,
            page_title="PD-Insighter Legacy Trial Selector",
            hero_title="PD-Insighter Legacy Trial Selector",
            hero_copy="Select a legacy trial dashboard from the original trial1-trial8 workflow. This is the older interface that was built before the Ntrial full-session redesign.",
            selector_heading="Legacy Trial Review",
            selector_copy="Choose a legacy processed trial to open its original dashboard directly.",
            target_key="legacy_dashboard_href",
        )
    )
    print(f"Wrote selector HTML to: {FULL_SELECTOR_HTML}")
    print(f"Wrote retired clip selector redirect to: {CLIP_SELECTOR_HTML}")
    print(f"Wrote legacy selector HTML to: {LEGACY_SELECTOR_HTML}")
    print(f"Included {len(trials)} analyzed trial(s).")


if __name__ == "__main__":
    main()
