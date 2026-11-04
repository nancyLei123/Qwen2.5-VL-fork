#!/usr/bin/env python3
"""Rebase legacy paths in an extracted research COPY; dry-run unless --apply."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import quote

ORIGINAL = "/Users/leijianan/gen1_glasses"
SUFFIXES = {".json", ".html", ".py", ".js", ".mjs", ".md", ".sh", ".txt", ".yaml", ".yml", ".toml"}


def protected(relative: Path):
    return ("manual_labels" in relative.parts or "spoken_action_labels" in relative.parts
            or ("manual" in relative.name.lower() and relative.suffix == ".json"))


def rewrite(text, destination):
    # Match URI encoding as well as filesystem paths. Only one known source root
    # is replaced; unrelated external paths remain visible for human review.
    return text.replace(quote(ORIGINAL, safe=""), quote(destination, safe="")).replace(ORIGINAL, destination)


def relocate(root: Path, apply=False):
    root = root.resolve()
    if root == Path(ORIGINAL).resolve():
        raise ValueError("Refusing to modify the live original workspace. Extract a copy elsewhere.")
    if not (root / "Qwen2.5-VL-fork/scripts/pd_insighter_server.py").is_file():
        raise ValueError("--root must point to the extracted workspace directory")
    rows = []
    backup = root / ".relocation_originals"
    for folder in ("Qwen2.5-VL-fork", "VRS_Pre_Processing/scripts", "projectaria_tools/scripts", "scripts"):
        for path in sorted((root / folder).rglob("*")):
            rel = path.relative_to(root)
            if rel.as_posix() == "Qwen2.5-VL-fork/docs/reproduction/relocate_copy.py":
                continue
            if path.is_symlink() or any(part.startswith(".") for part in rel.parts):
                continue
            if not path.is_file() or path.suffix not in SUFFIXES or protected(rel):
                continue
            original = path.read_text(errors="strict")
            updated = rewrite(original, root.as_posix())
            if original == updated:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({"path": rel.as_posix(), "original_sha256": digest})
            if apply:
                saved = backup / rel
                saved.parent.mkdir(parents=True, exist_ok=True)
                if not saved.exists():
                    shutil.copy2(path, saved)
                path.write_text(updated)
    if apply:
        (root / "relocation_report.json").write_text(json.dumps({"changed": rows, "annotation_files_skipped": True}, indent=2))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    changed = relocate(args.root, args.apply)
    print(f"{'Updated' if args.apply else 'Would update'} {len(changed)} text files; annotation files excluded.")
    for row in changed[:30]:
        print(row["path"])
    if len(changed) > 30:
        print("... additional files omitted from this console summary")
