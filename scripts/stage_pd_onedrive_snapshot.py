#!/usr/bin/env python3
"""Stage the reviewed allowlist with independent APFS clones; never sync sources."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def save(path: Path, data) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def clone(source: Path, target: Path) -> None:
    if sys.platform != "darwin":
        raise RuntimeError("This staging mode requires macOS APFS clonefile support")
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    function = lib.clonefile
    function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    function.restype = ctypes.c_int
    target.parent.mkdir(parents=True, exist_ok=True)
    if function(os.fsencode(source), os.fsencode(target), 0) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))


def stage(root: Path, inventory: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    if output == root or root.is_relative_to(output):
        raise ValueError("Snapshot must be a separate directory, never the source root or its parent")
    if output.exists():
        raise FileExistsError("Use a fresh output directory; existing snapshots are never overwritten")
    payload = json.loads(inventory.read_text())
    files = payload["files"]
    # Validate everything before creating a partial snapshot.
    for entry in files:
        relative = Path(entry["path"])
        source = root / relative
        if relative.is_absolute() or ".." in relative.parts or source.is_symlink():
            raise ValueError("Unsafe inventory path: " + str(relative))
        if not source.resolve().is_relative_to(root) or source.resolve().is_relative_to(output):
            raise ValueError("Source outside the approved tree: " + str(relative))
        stat = source.stat()
        if not source.is_file() or (stat.st_size, stat.st_mtime_ns) != (entry["bytes"], entry["mtime_ns"]):
            raise RuntimeError("Source changed since inventory: " + str(relative))
    free_before = shutil.disk_usage(output.parent).free
    if free_before < 50 * 1024**3:
        raise RuntimeError("Keep at least 50 GiB free before preparing the upload snapshot")
    output.mkdir()
    status = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "STAGING",
              "source_inventory_sha256": digest(inventory), "files_total": len(files),
              "source_bytes": sum(e["bytes"] for e in files), "files_staged": 0,
              "storage_method": "APFS independent copy-on-write clones, not hardlinks",
              "cloud_status": "NOT_UPLOADED", "source_changed": False}
    save(output / "STAGING_STATUS.json", status)
    for number, entry in enumerate(files, 1):
        source, target = root / entry["path"], output / "workspace" / entry["path"]
        before = source.stat()
        if (before.st_size, before.st_mtime_ns) != (entry["bytes"], entry["mtime_ns"]):
            raise RuntimeError("Source changed before clone: " + entry["path"])
        clone(source, target)
        after, copied = source.stat(), target.stat()
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            raise RuntimeError("Source changed during clone: " + entry["path"])
        if copied.st_ino == after.st_ino or copied.st_size != entry["bytes"]:
            raise RuntimeError("Clone must have independent inode and matching size: " + entry["path"])
        if entry.get("annotation") and entry.get("sha256") and digest(target) != entry["sha256"]:
            raise RuntimeError("Annotation checksum mismatch: " + entry["path"])
        status["files_staged"] = number
        if number % 1000 == 0:
            save(output / "STAGING_STATUS.json", status)
            print(f"Staged {number}/{len(files)} files", flush=True)
    clone(inventory, output / "SOURCE_INVENTORY.json")
    status.update(status="STAGED_LOCALLY_NOT_UPLOADED", finished_utc=datetime.now(timezone.utc).isoformat(),
                  free_disk_before_bytes=free_before, free_disk_after_bytes=shutil.disk_usage(output).free,
                  annotation_hashes="All available source annotation hashes matched",
                  note="Independent snapshot only. No cloud upload has been performed by this script.")
    save(output / "STAGING_STATUS.json", status)
    return status


def verify(snapshot: Path) -> dict:
    payload = json.loads((snapshot / "SOURCE_INVENTORY.json").read_text())
    report = snapshot / "SOURCE_SHA256.jsonl"
    if report.exists():
        raise FileExistsError("Existing checksum report is not overwritten")
    temporary = report.with_suffix(".jsonl.partial")
    total = 0
    with temporary.open("x") as writer:
        for i, entry in enumerate(payload["files"], 1):
            path = snapshot / "workspace" / entry["path"]
            before = path.stat()
            if before.st_size != entry["bytes"]:
                raise RuntimeError("Snapshot size mismatch: " + entry["path"])
            checksum = digest(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError("Snapshot changed while hashing: " + entry["path"])
            if entry.get("sha256") and checksum != entry["sha256"]:
                raise RuntimeError("Reference checksum mismatch: " + entry["path"])
            writer.write(json.dumps({"path": "workspace/" + entry["path"], "bytes": entry["bytes"], "sha256": checksum}) + "\n")
            total += entry["bytes"]
            if i % 1000 == 0:
                writer.flush()
                print(f"Hashed {i}/{len(payload['files'])} files, {total / 1024**3:.2f} GiB", flush=True)
    temporary.rename(report)
    result = {"status": "LOCAL_SHA256_COMPLETE", "files": len(payload["files"]), "bytes": total,
              "manifest": report.name, "manifest_sha256": digest(report),
              "finished_utc": datetime.now(timezone.utc).isoformat(),
              "note": "Local verification only; cloud transfer completion requires separate verification."}
    save(snapshot / "LOCAL_VERIFICATION.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify.resolve()), indent=2))
    elif args.root and args.inventory and args.output:
        print(json.dumps(stage(args.root, args.inventory, args.output), indent=2))
    else:
        parser.error("Use --root --inventory --output, or --verify")


if __name__ == "__main__":
    main()
