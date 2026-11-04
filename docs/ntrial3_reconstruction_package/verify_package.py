#!/usr/bin/env python3
"""Verify the extracted package against its recorded SHA-256 inventory."""
import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parent
    inventory = json.loads((root / "FILE_INVENTORY.json").read_text())
    failures = []
    for number, item in enumerate(inventory["files"], 1):
        path = root / item["path"]
        if not path.resolve().is_relative_to(root):
            failures.append(item["path"] + ": unsafe path")
            continue
        if not path.is_file() or path.is_symlink():
            failures.append(item["path"] + ": missing or symlink")
        elif path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            failures.append(item["path"] + ": size/checksum mismatch")
        if number % 250 == 0:
            print(f"Checked {number}/{len(inventory['files'])} files", flush=True)
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Verified {len(inventory['files'])} payload files; all SHA-256 hashes match.")
    print("The inventory itself is protected by the ZIP's external SHA-256 checksum.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
