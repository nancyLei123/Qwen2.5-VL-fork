#!/usr/bin/env python3
"""Snapshot only three Fuchs trials; preserve originals and build independent ZIPs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

from prepare_pd_reproduction_bundle import candidate_secret, excluded_reason, environment_snapshot

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "Qwen2.5-VL-fork"
TRIALS = ("Fuchs", "Fuchs22", "Fuchs333")
PACKAGE = "PD_Insighter_Fuchs_20261105"
DOCS = REPO / "docs/fuchs_handoff"
STORED = {".vrs", ".mp4", ".gz", ".jpg", ".jpeg", ".png", ".zip"}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def clone(source, target):
    before = source.stat()
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.uname().sysname != "Darwin" or subprocess.run(
            ["/bin/cp", "-c", "-p", str(source), str(target)], capture_output=True).returncode:
        shutil.copy2(source, target)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed during snapshot: {source}")


def limit_switcher(text):
    # Limit only the copied UI options; analytical payloads and numerical results stay unchanged.
    patch = """<script id="handoff-trials-only">
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('select').forEach(select => {
    if (!/trial/i.test(select.id + ' ' + (select.getAttribute('aria-label') || ''))) return;
    for (const option of [...select.options]) {
      if (!/^(All Trials|Fuchs|Fuchs22|Fuchs333)$/i.test(option.textContent.trim())) option.remove();
    }
  });
});
</script>"""
    return text.replace("</body>", patch + "</body>")


def selector():
    cards = []
    pages = [("Full Session", "index.html"), ("Manual Labels", "pd_insighter_manual_labeling.html"),
             ("Stride", "pd_insighter_stride_review.html"), ("Transitions", "pd_insighter_transition_review.html"),
             ("Parkinson Review", "pd_insighter_parkinson_review.html"),
             ("Environment", "environment_reconstruction/playback.html")]
    for trial in TRIALS:
        summary = json.loads((REPO / "runs" / trial / "summary.json").read_text())
        links = " ".join(f'<a href="{trial}/{page}">{title}</a>' for title, page in pages)
        cards.append(f'<article><h2>{trial}</h2><p>{summary["session_duration_s"]/60:.2f} minutes'
                     f' | simulated movement | both camera review views</p><nav>{links}</nav></article>')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PD Insighter | Fuchs trial handoff</title><style>
body{margin:0;background:linear-gradient(130deg,#e4eddf,#f8f5ee);color:#182b23;font-family:"Avenir Next",Verdana,sans-serif}
main{max-width:1080px;margin:50px auto;padding:0 24px}h1{font-size:38px}p{color:#52685d;line-height:1.6}
article{background:#fffdf9;border:1px solid #d7e2d7;border-radius:22px;padding:24px;margin:20px 0}
nav{display:flex;gap:10px;flex-wrap:wrap}a{display:inline-block;padding:12px 16px;background:#176b52;color:white;border-radius:24px;text-decoration:none}
</style><main><p>PD INSIGHTER / RESEARCH HANDOFF / NOVEMBER 5, 2026</p><h1>Three recordings. Every current view.</h1>
<p>Saved results and synchronized video, with local annotation support. No retraining is needed to explore.
These are simulated movements; classification, stride and reconstructed geometry are not clinically validated.</p>
''' + "".join(cards) + '<p>The original VRS/MPS archives are optional for viewing. Read README.md and REPROCESSING.md in the package root.</p></main></html>'


def build(output):
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing handoffs are never overwritten.")
    stage = output / PACKAGE
    workspace = stage / "workspace"
    workspace.mkdir(parents=True)
    groups = {"01_PD_Insighter_App_and_Results": []}
    excluded = []
    source_audit = []

    def add(source, group, vendor=False):
        rel = source.relative_to(ROOT)
        reason = excluded_reason(rel)
        if vendor and "node_modules" in rel.parts:
            reason = None
        if source.is_symlink():
            reason = "symlink excluded; no external targets followed"
        if reason:
            excluded.append({"path": str(rel), "reason": reason})
            return
        if not source.is_file():
            return
        if candidate_secret(source):
            raise RuntimeError(f"Possible credential requires review: {source}")
        target = workspace / rel
        clone(source, target)
        source_audit.append({"path": str(rel), "bytes": source.stat().st_size,
                             "mtime_ns": source.stat().st_mtime_ns})
        if target.suffix == ".html" and "runs" in rel.parts:
            target.write_text(limit_switcher(target.read_text()))
        groups.setdefault(group, []).append(target)

    def tree(path, group, vendor=False):
        for directory, dirs, files in os.walk(path, followlinks=False):
            here = Path(directory)
            dirs[:] = sorted(d for d in dirs if not (here/d).is_symlink()
                             and (not excluded_reason((here/d).relative_to(ROOT))
                                  or vendor and d == "node_modules"))
            for name in sorted(files):
                add(here / name, group, vendor)

    app = "01_PD_Insighter_App_and_Results"
    for relative in ("Qwen2.5-VL-fork/scripts", "Qwen2.5-VL-fork/tests",
                     "Qwen2.5-VL-fork/docs/reproduction", "Qwen2.5-VL-fork/vendor/gaussian-viewer",
                     "VRS_Pre_Processing/scripts", "scripts", "tools/OpenSplat-1.2.2"):
        tree(ROOT / relative, app)
    # Preserve the installed browser runtime without unrelated npm caches or executables.
    for relative in ("three/build", "three/examples/jsm", "@mkkellogg/gaussian-splats-3d/build"):
        base = REPO / "vendor/gaussian-viewer/node_modules" / relative
        for source in sorted(base.rglob("*")):
            if source.is_file():
                add(source, app, vendor=True)
    for name in ("three", "@mkkellogg/gaussian-splats-3d"):
        add(REPO / "vendor/gaussian-viewer/node_modules" / name / "LICENSE", app, vendor=True)
    for name in ("STANDARD_TRIAL_WORKFLOW.md", "SLAM_COORDINATES_STEP_STRIDE_EXPLANATION.md",
                 "PARKINSON_MOTION_CRITERIA_NTRIAL10.md", "ARIA_ENVIRONMENT_RECONSTRUCTION.md",
                 "MOTION_PARAMETER_CATALOG_NTRIALS.md", "RAW_SLAM_PARAMETER_INVESTIGATION.md"):
        add(REPO / name, app)
    for name in ("README.md", "REPROCESSING.md", "requirements.txt", "start.py"):
        target = stage / name
        clone(DOCS / name, target)
        groups[app].append(target)
    for trial in TRIALS:
        tree(REPO / "runs" / trial, app)
    home = workspace / "Qwen2.5-VL-fork/runs/pd_insighter_trial_selector.html"
    home.write_text(selector())
    groups[app].append(home)
    save(stage / "ENVIRONMENT_VERSIONS.json", environment_snapshot(ROOT))
    groups[app].append(stage / "ENVIRONMENT_VERSIONS.json")
    raw = ROOT / "VRS_Pre_Processing/data/Fuchs_Home_20261103"
    for trial in TRIALS:
        group = "02_Raw_" + trial
        for suffix in (".vrs", ".vrs.json"):
            add(raw / (trial + suffix), group)
        tree(raw / f"mps_{trial}_vrs", group)
        tree(ROOT / "VRS_Pre_Processing/data/extracted_videos" / trial, group)
        add(ROOT / "downward videos" / (trial + ".mp4"), group)
        add(raw / "local_checks_20261104" / (trial + "_local_validation.json"), group)
    save(output / "SOURCE_SNAPSHOT.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
         "files": source_audit, "excluded": excluded})
    index = {"created_utc": datetime.now(timezone.utc).isoformat(), "trials": TRIALS,
             "originals_modified": False, "cloud_upload_verified": False, "archives": []}
    for group, files in groups.items():
        print(f"Packing {group}: {len(files)} files, {sum(p.stat().st_size for p in files)/1024**3:.2f} GiB", flush=True)
        records = []
        archive_path = output / (group + ".zip")
        with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            for number, path in enumerate(sorted(files)):
                records.append({"path": path.relative_to(stage).as_posix(),
                                "bytes": path.stat().st_size, "sha256": digest(path)})
                archive.write(path, path.relative_to(output), compress_type=(
                    zipfile.ZIP_STORED if path.suffix.lower() in STORED else zipfile.ZIP_DEFLATED))
                if number and number % 500 == 0:
                    print(f"  {number}/{len(files)}", flush=True)
            checksum = stage / "checksums" / (group + ".json")
            save(checksum, {"files": records, "group": group})
            archive.write(checksum, checksum.relative_to(output))
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("ZIP CRC verification failed")
        entry = {"name": archive_path.name, "bytes": archive_path.stat().st_size,
                 "sha256": digest(archive_path), "files": len(records), "zip_crc_verified": True}
        index["archives"].append(entry)
        save(output / "PACKAGE_INDEX.json", index)
        print("Completed: " + json.dumps(entry), flush=True)
    for name in ("README.md", "REPROCESSING.md"):
        clone(DOCS / name, output / name)
    print(f"READY: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output.resolve())
