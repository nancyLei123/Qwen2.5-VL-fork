#!/usr/bin/env python3
"""Inventory an allowlisted research snapshot and optionally make independent ZIPs.

This is local-only: it never uploads, deletes sources, changes sharing, follows
external symlinks, or includes installed environments and account credentials.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile


WORKSPACE = Path(__file__).resolve().parents[2]
REPO = WORKSPACE / "Qwen2.5-VL-fork"
BLOCKED_PARTS = {".git", ".venv", ".venvs", "__pycache__", ".pytest_cache", ".cache", "build", "_deps",
                 "node_modules", ".aws", ".ssh", ".codex", ".DS_Store"}
TEXT_SUFFIXES = {".py", ".js", ".mjs", ".ts", ".json", ".jsonl", ".md", ".txt",
                 ".html", ".yaml", ".yml", ".toml", ".sh", ".log", ".ipynb"}
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{25,}"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9_.-]{24,}"),
    re.compile(r'''(?i)(?:password|access_token|client_secret|api_key)\s*[:=]\s*["'][A-Za-z0-9_./+=-]{12,}["']'''),
    re.compile(r"https?://[^\s\"<>]+[?&](?:access_token|X-Amz-Signature)=[^\s\"<>]+"),
]
DOC_NAMES = ["STANDARD_TRIAL_WORKFLOW.md", "SLAM_COORDINATES_STEP_STRIDE_EXPLANATION.md",
             "PARKINSON_MOTION_CRITERIA_NTRIAL10.md", "ARIA_ENVIRONMENT_RECONSTRUCTION.md",
             "MOTION_PARAMETER_CATALOG_NTRIALS.md", "RAW_SLAM_PARAMETER_INVESTIGATION.md",
             "SLAM_FEATURE_RECONSTRUCTION_RESULTS_JUNE_2026.md", "THESIS_REPLICATION_GUIDE.md"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def excluded_reason(relative: Path) -> str | None:
    parts = relative.parts
    if any(part in BLOCKED_PARTS or part.startswith("build-") or part.endswith(".app") for part in parts):
        return "cache, installed environment, VCS metadata, or build directory"
    name = relative.name.lower()
    if name.startswith(".env") or name in {"credentials.json", "token.json", "tokens.json", "auth.json"}:
        return "account or environment secret file"
    if name.endswith((".enc", ".vrs.enc", ".pyc", ".pyo", ".partial", ".tmp", ".pem", ".key", ".lic")):
        return "credential, disposable encrypted upload, cache, or incomplete file"
    return None


def candidate_secret(path: Path) -> bool:
    # This is a preflight screen, not a guarantee that research content is anonymous.
    if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 16 * 1024 * 1024:
        return False
    content = path.read_text(errors="replace")
    return any(pattern.search(content) for pattern in SECRET_PATTERNS)


def inventory(workspace: Path):
    entries, excluded = {}, []

    def add(path: Path, group: str):
        relative = path.relative_to(workspace)
        if relative.as_posix() in entries:
            return
        reason = excluded_reason(relative)
        if path.is_symlink():
            reason = "symlink: must be reviewed rather than silently dereferenced"
        if reason:
            excluded.append({"path": relative.as_posix(), "reason": reason})
            return
        if not path.is_file():
            return
        if candidate_secret(path):
            excluded.append({"path": relative.as_posix(), "reason": "possible embedded credential; manual review required"})
            return
        stat = path.stat()
        annotation = ("manual" in path.name.lower() and path.suffix == ".json") or "manual_labels" in relative.parts
        entries[relative.as_posix()] = {
            "path": relative.as_posix(), "group": group, "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "annotation": annotation,
            "sha256": sha256(path) if annotation else None,
        }

    def tree(root: Path, group):
        if not root.exists():
            return
        for directory, subdirs, files in os.walk(root, followlinks=False):
            here = Path(directory)
            kept = []
            for name in sorted(subdirs):
                child = here / name
                reason = excluded_reason(child.relative_to(workspace))
                if child.is_symlink():
                    reason = "symlink directory: not traversed"
                if reason:
                    excluded.append({"path": child.relative_to(workspace).as_posix(), "reason": reason})
                else:
                    kept.append(name)
            subdirs[:] = kept
            for name in sorted(files):
                path = here / name
                add(path, group(path) if callable(group) else group)

    for repo_name in ("Qwen2.5-VL-fork", "projectaria_tools", "VRS_Pre_Processing"):
        repo = workspace / repo_name
        tracked = subprocess.run(["git", "ls-files", "-z"], cwd=repo, check=True, capture_output=True).stdout
        for item in tracked.decode().split("\0"):
            if not item:
                continue
            # Do not import tracked example datasets, personal documents, or demos.
            first = Path(item).parts[0]
            if first not in {"data", "runs", "deliverables", "pd_insighter_demo"} and not any(part in {"demo", "demos", "test_data"} for part in Path(item).parts):
                add(repo / item, "00_code")
    for relative in ("Qwen2.5-VL-fork/scripts", "Qwen2.5-VL-fork/tests", "Qwen2.5-VL-fork/vendor",
                     "Qwen2.5-VL-fork/docs/reproduction", "Qwen2.5-VL-fork/docs/ntrial3_reconstruction_package",
                     "VRS_Pre_Processing/scripts", "projectaria_tools/scripts", "scripts", "vrs_chunking_tools"):
        tree(workspace / relative, "00_code")
    for name in DOC_NAMES:
        add(workspace / "Qwen2.5-VL-fork" / name, "00_code")
    tree(workspace / "tools/OpenSplat-1.2.2", "00_code")

    runs = workspace / "Qwen2.5-VL-fork/runs"
    def result_group(path):
        relative = path.relative_to(runs)
        if len(relative.parts) == 1:
            return "10_results_shared"
        trial = relative.parts[0]
        if "environment_reconstruction" in relative.parts:
            return "20_environment_" + trial
        return "10_results_" + trial
    tree(runs, result_group)

    raw = workspace / "VRS_Pre_Processing/data"
    def raw_group(path):
        parts = path.relative_to(raw).parts
        if parts[0].startswith("extracted_videos"):
            return "40_extracted_" + (parts[1] if len(parts) > 2 else "shared")
        return "30_raw_mps_" + (parts[0] if len(parts) > 1 else "shared")
    tree(raw, raw_group)
    tree(workspace / "downward videos", lambda p: "50_downward_" + (p.relative_to(workspace / "downward videos").parts[0].rsplit(".", 1)[0]))
    tree(workspace / "data", "60_support_data")
    return list(entries.values()), excluded


def trial_audit(workspace):
    output = []
    runs = workspace / "Qwen2.5-VL-fork/runs"
    for trial in sorted(p for p in runs.iterdir() if p.is_dir()):
        manifest_path = trial / "trial_manifest.json"
        session_path = trial / "session_dashboard_manifest.json"
        if not (manifest_path.exists() or session_path.exists() or (trial / "environment_reconstruction").exists()):
            continue
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        session = json.loads(session_path.read_text()) if session_path.exists() else {}
        checked = {}
        for section in ("raw_data", "mps_outputs", "extracted_media"):
            for key, value in manifest.get(section, {}).items():
                if isinstance(value, dict) and "path" in value and not key.endswith("dir"):
                    source = Path(value["path"])
                    try:
                        relative = source.relative_to(workspace).as_posix()
                    except ValueError:
                        relative = "EXTERNAL/" + source.name
                    checked[section + "." + key] = {"path": relative, "exists": source.is_file()}
        for view in ("aria", "downward"):
            path = trial / "downward_master_session" / f"{view}_session_video.mp4"
            if session:
                checked[view + "_session_video"] = {"path": path.relative_to(workspace).as_posix(), "exists": path.is_file()}
        pages = [name for name in ("index.html", "pd_insighter_manual_labeling.html", "pd_insighter_stride_review.html",
                                  "pd_insighter_transition_review.html", "pd_insighter_parkinson_review.html",
                                  "environment_reconstruction/index.html", "environment_reconstruction/playback.html")
                 if (trial / name).is_file()]
        annotations = []
        for folder in ("manual_labels", "parkinson_motion_review", "spoken_action_labels"):
            for path in sorted((trial / folder).glob("*manual*.json")):
                payload = json.loads(path.read_text())
                annotations.append({"path": path.relative_to(workspace).as_posix(), "sha256": sha256(path),
                                    "segment_count": len(payload.get("segments", [])) if isinstance(payload, dict) else None})
        output.append({"trial_id": trial.name, "trial_type": manifest.get("trial_type", "not explicitly recorded"),
                       "has_full_session_manifest": bool(session), "session_duration_s": session.get("session_duration_s"),
                       "pages": pages, "source_checks": checked, "annotations": annotations,
                       "missing_sources": [key for key, value in checked.items() if not value["exists"]]})
    return output


def environment_snapshot(workspace):
    result = {}
    for name in ("aria-reconstruction", "aria-mps"):
        python = workspace / ".venvs" / name / "bin/python"
        if python.exists():
            script = "import importlib.metadata as m,json,platform; print(json.dumps({'python':platform.python_version(),'platform':platform.platform(),'packages':sorted([{'name':d.metadata['Name'],'version':d.version} for d in m.distributions()],key=lambda p:p['name'].lower())}))"
            process = subprocess.run([str(python), "-c", script], capture_output=True, text=True)
            result[name] = json.loads(process.stdout) if process.returncode == 0 else {"unavailable": True}
    for name in ("Qwen2.5-VL-fork", "projectaria_tools", "VRS_Pre_Processing"):
        process = subprocess.run(["git", "rev-parse", "HEAD"], cwd=workspace / name, capture_output=True, text=True)
        result[name] = {"git_head": process.stdout.strip(), "note": "Working files, including untracked research scripts, are snapshotted; commit alone is insufficient."}
    return result


def archive_group(workspace, output, group, entries, max_bytes=2 * 1024**3):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", group) or ".." in group:
        raise ValueError("Invalid package group")
    files = sorted((entry for entry in entries if entry["group"] == group), key=lambda e: e["path"])
    if not files:
        raise ValueError("Unknown or empty group: " + group)
    parts, part, size = [], [], 0
    for entry in files:
        if part and size + entry["bytes"] > max_bytes:
            parts.append(part)
            part, size = [], 0
        part.append(entry)
        size += entry["bytes"]
    if part:
        parts.append(part)
    reports = []
    for number, members in enumerate(parts, 1):
        name = f"{group}.part{number:03d}.zip"
        target = output / "archives" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        report_path = target.with_suffix(".verified.json")
        if target.exists() or report_path.exists():
            raise FileExistsError(f"Existing artifact is never overwritten: {target}")
        expected = sum(e["bytes"] for e in members)
        if shutil.disk_usage(target.parent).free < expected + 3 * 1024**3:
            raise RuntimeError("Not enough free space to archive without risking the research workspace")
        temporary = target.with_suffix(".zip.partial")
        checks = []
        with zipfile.ZipFile(temporary, "x", allowZip64=True) as archive:
            for entry in members:
                source = workspace / entry["path"]
                if Path(entry["path"]).is_absolute() or ".." in Path(entry["path"]).parts or source.is_symlink() or not source.resolve().is_relative_to(workspace.resolve()):
                    raise ValueError("Unsafe source path: " + entry["path"])
                stat = source.stat()
                if (stat.st_size, stat.st_mtime_ns) != (entry["bytes"], entry["mtime_ns"]):
                    raise RuntimeError("Source changed since inventory; make a new snapshot: " + entry["path"])
                info = zipfile.ZipInfo("workspace/" + entry["path"])
                info.compress_type = zipfile.ZIP_DEFLATED if source.suffix.lower() in TEXT_SUFFIXES else zipfile.ZIP_STORED
                info.external_attr = (stat.st_mode & 0xFFFF) << 16
                digest = hashlib.sha256()
                with source.open("rb") as reader, archive.open(info, "w", force_zip64=True) as writer:
                    for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                        writer.write(chunk)
                        digest.update(chunk)
                after = source.stat()
                if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                    raise RuntimeError("Source changed while packing: " + entry["path"])
                checks.append({"path": info.filename, "bytes": stat.st_size, "sha256": digest.hexdigest()})
            archive.writestr("_package_checksums/" + name + ".json", json.dumps(checks, indent=2))
        with zipfile.ZipFile(temporary) as archive:
            bad = archive.testzip()
            if bad:
                raise RuntimeError("ZIP integrity failure: " + bad)
        temporary.rename(target)
        report = {"archive": target.name, "bytes": target.stat().st_size, "sha256": sha256(target),
                  "entries": len(checks), "zip_crc_test": "passed", "source_hashes_recorded": True,
                  "upload_status": "NOT_UPLOADED"}
        write_json(report_path, report)
        reports.append(report)
        print(f"Built {target.name}: {target.stat().st_size / 1024**2:.1f} MiB; verified locally, not uploaded", flush=True)
    return reports


def build_inventory(workspace, output):
    output.mkdir(parents=True, exist_ok=False)
    entries, excluded = inventory(workspace)
    trials = trial_audit(workspace)
    groups = defaultdict(lambda: {"files": 0, "bytes": 0, "upload_status": "NOT_UPLOADED"})
    for entry in entries:
        groups[entry["group"]]["files"] += 1
        groups[entry["group"]]["bytes"] += entry["bytes"]
    payload = {"snapshot_date": "2026-11-01", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "files": entries, "excluded": excluded, "groups": dict(sorted(groups.items())), "trials": trials,
               "scope": "Allowlisted local research workspace; external drives and unrelated user folders were not copied.",
               "not_included": ["account credentials", "virtual environments", "installed proprietary applications/licenses",
                                "Downloads duplicates", "previous delivery ZIP duplicates", "email drafts and meeting correspondence"],
               "upload_status": "NOT_UPLOADED; destination sharing needs confirmation"}
    write_json(output / "inventory.json", payload)
    write_json(output / "environment_versions.json", environment_snapshot(workspace))
    total = sum(item["bytes"] for item in entries)
    lines = ["# PD Insighter Reproduction Inventory", "", "Snapshot: 2026-11-01 (America/New_York).",
             "", "**LOCAL PREPARATION ONLY. No research data has been uploaded.**", "",
             f"Selected: {len(entries):,} files, {total / 1024**3:.2f} GiB. These are uncompressed sizes, not upload estimates.",
             "", "This is a research snapshot, not an anonymized public dataset. Sharing approval is required.", "",
             "## Package Groups", "", "| Group | Files | GiB |", "|---|---:|---:|"]
    for group, item in sorted(groups.items()):
        lines.append(f"| {group} | {item['files']} | {item['bytes'] / 1024**3:.3f} |")
    lines += ["", "## Trial Readiness", "", "Pages and source availability below are file-existence checks, not an independent re-run of every analysis.",
              "", "| Trial | Full session | Available pages | Missing manifest sources |", "|---|---|---:|---|"]
    for trial in trials:
        lines.append(f"| {trial['trial_id']} | {'yes' if trial['has_full_session_manifest'] else 'no'} | {len(trial['pages'])} | {', '.join(trial['missing_sources']) or 'none listed'} |")
    lines += ["", "## Important Gaps", "", "Longoria archival data was previously moved to NDrive. That drive is not mounted in this inventory; old manifests do not prove the files are present.",
              "", "Original source code and manifests retain historical local paths. Use the documented relocation step on a COPY before regenerating outputs.",
              "", "Do not reuse normal-trial criteria for Ntrial10, Parki, or Parki2. Preserve the saved simulated-PD classifications and annotation stores.",
              "", "No claim is made that a rerun on a different OS/GPU is numerically identical, or that simulated labels are clinical ground truth.",
              "", "## Exclusions Requiring Review", ""]
    for reason, count in Counter(item["reason"] for item in excluded).items():
        lines.append(f"- {reason}: {count} paths (see inventory.json).")
    (output / "INVENTORY.md").write_text("\n".join(lines) + "\n")
    print(f"Inventoried {len(entries):,} files in {len(groups)} groups: {total / 1024**3:.2f} GiB", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pack", nargs="*", default=[], help="Groups to pack after inventory, e.g. 00_code. No uploads.")
    parser.add_argument("--existing-inventory", action="store_true")
    parser.add_argument("--part-gib", type=float, default=2)
    args = parser.parse_args()
    if args.part_gib <= 0:
        parser.error("--part-gib must be positive")
    output = args.output.resolve()
    payload = json.loads((output / "inventory.json").read_text()) if args.existing_inventory else build_inventory(WORKSPACE, output)
    for group in args.pack:
        archive_group(WORKSPACE, output, group, payload["files"], int(args.part_gib * 1024**3))


if __name__ == "__main__":
    main()
