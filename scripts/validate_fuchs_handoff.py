#!/usr/bin/env python3
"""Smoke-test an extracted handoff; annotation writes require an isolated copy."""
import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse
from urllib.request import Request, urlopen


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ("href", "src"):
            value = attrs.get(key, "")
            if value and not value.startswith(("#", "data:", "javascript:")):
                self.urls.append(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8797)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not str(root).startswith("/private/tmp/pd-fuchs-handoff-check-"):
        raise SystemExit("Only the isolated temporary test copy may be modified.")
    workspace = root / "workspace"
    base = f"http://127.0.0.1:{args.port}"
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Extracted local copy, not cloud upload verification",
              "pages": [], "resources": [], "video_ranges": [],
              "models": [], "annotations": [], "frame_context": [],
              "inactive_optional_links": [], "errors": []}
    # The shared viewer template contains other trials' optional experiments.
    # Exempt only links independently observed as hidden in this trial's live DOM.
    browser_report = args.report.with_name("BROWSER_VALIDATION.json")
    hidden_optional = set()
    if browser_report.exists():
        for check in json.loads(browser_report.read_text()):
            if urlparse(check["url"]).netloc != urlparse(base).netloc:
                continue
            hidden_optional.update(urljoin(check["url"], entry["path"])
                for entry in check["optional"] if entry["hidden"] and not entry["visible"])
    pages = ["index.html", "pd_insighter_manual_labeling.html",
             "pd_insighter_stride_review.html", "pd_insighter_transition_review.html",
             "pd_insighter_parkinson_review.html", "environment_reconstruction/playback.html"]
    references = set()
    for trial in ("Fuchs", "Fuchs22", "Fuchs333"):
        trial_base = f"/Qwen2.5-VL-fork/runs/{trial}/"
        for page in pages:
            url = base + trial_base + page
            with urlopen(url, timeout=30) as response:
                html = response.read().decode()
                report["pages"].append({"url": url, "status": response.status})
            refs = References()
            refs.feed(html)
            for ref in refs.urls:
                resolved = urljoin(url, ref)
                if urlparse(resolved).netloc == urlparse(base).netloc:
                    references.add(resolved)
        for view in ("aria", "downward"):
            url = base + trial_base + f"downward_master_session/{view}_session_video.mp4"
            req = Request(url, headers={"Range": "bytes=1000000-1001023"})
            with urlopen(req, timeout=30) as response:
                data = response.read()
                assert response.status == 206 and len(data) == 1024
                report["video_ranges"].append({"trial": trial, "view": view,
                    "status": response.status, "content_range": response.headers["Content-Range"]})
        for model in (workspace / trial_base.lstrip("/")).glob("environment_reconstruction/gaussian_submaps/*/trained_7000_fullres/splat_7000.ply"):
            url = base + "/" + quote(str(model.relative_to(workspace)))
            with urlopen(Request(url, headers={"Range": "bytes=0-4095"}), timeout=30) as response:
                data = response.read()
                assert data.startswith(b"ply\n") and b"element vertex" in data
                report["models"].append({"trial": trial, "file": model.name,
                    "submap": model.parent.parent.name, "bytes": model.stat().st_size})
        url = base + f"/api/session-manual-segments?trial_id={trial}"
        with urlopen(url, timeout=30) as response:
            original = json.load(response)
        assert original["segments"] == [], "Unexpected existing test labels; refusing overwrite"
        payload = {"trial_id": trial, "segments": [{"id": "handoff_smoke_test",
            "start_s": 1, "end_s": 2, "motion_label": "standing", "confidence": "uncertain",
            "note": "Synthetic test in disposable copy only, not a research annotation."}]}
        def post(value):
            request = Request(base + "/api/session-manual-segments", data=json.dumps(value).encode(),
                              headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        try:
            assert post(payload)["saved"]
            with urlopen(url, timeout=30) as response:
                assert json.load(response)["segments"][0]["id"] == "handoff_smoke_test"
            report["annotations"].append({"trial": trial, "save_and_readback": True})
        finally:
            post({"trial_id": trial, "segments": []})
        with urlopen(base + f"/api/frame-context?trial_id={trial}&center_time_s=100&offsets=0", timeout=30) as response:
            frames = json.load(response)["frames"]
        for frame in frames:
            for key in ("aria_url", "downward_url"):
                with urlopen(base + frame[key], timeout=30) as response:
                    assert response.read(2) == b"\xff\xd8"
        report["frame_context"].append({"trial": trial, "center_time_s": 100, "paired_frames": len(frames)})
    for url in sorted(references):
        try:
            with urlopen(Request(url, method="HEAD"), timeout=30) as response:
                report["resources"].append({"url": url, "status": response.status})
        except Exception as exc:
            key = "inactive_optional_links" if url in hidden_optional else "errors"
            report[key].append({"url": url, "error": str(exc)})
    report["passed"] = not report["errors"] and len(report["models"]) == 14
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: len(report[key]) for key in
        ("pages", "resources", "video_ranges", "models", "annotations", "errors")}))
    print("PASS" if report["passed"] else "FAIL")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
