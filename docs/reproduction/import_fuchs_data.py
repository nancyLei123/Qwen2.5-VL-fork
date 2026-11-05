#!/usr/bin/env python3
"""Import verified Fuchs result data into a fresh clone; never copy snapshot code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil

REPO = Path(__file__).resolve().parents[2]
PREFIX = PurePosixPath('workspace/Qwen2.5-VL-fork/runs')
TRIALS = ('Fuchs', 'Fuchs22', 'Fuchs333')
MANIFEST = 'checksums/01_PD_Insighter_App_and_Results.json'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def plan(package, repo):
    package, repo = package.resolve(), repo.resolve()
    if (repo / 'runs').exists() or (repo / 'runs').is_symlink():
        raise FileExistsError('runs/ already exists. Nothing overwritten; use a fresh clone.')
    if not (repo / 'scripts/pd_insighter_server.py').is_file():
        raise ValueError('Destination is not a PD Insighter source checkout.')
    manifest = package / MANIFEST
    if not manifest.is_file():
        raise ValueError('Select the fully extracted PD_Insighter_Fuchs_20261105 folder.')
    entries, seen = [], set()
    for item in json.loads(manifest.read_text())['files']:
        path = PurePosixPath(item['path'])
        if path.is_absolute() or '..' in path.parts or '\\' in str(path):
            raise ValueError('Unsafe manifest path')
        if not path.is_relative_to(PREFIX):
            continue
        relative = Path(*path.relative_to(PREFIX).parts)
        if not relative.parts or relative in seen:
            raise ValueError('Empty or duplicate data path')
        seen.add(relative)
        source = package.joinpath(*path.parts)
        if any(p.is_symlink() for p in (source, *source.parents) if p != package.parent):
            raise ValueError(f'Symlink not allowed: {relative}')
        if not source.resolve().is_relative_to(package) or not source.is_file():
            raise ValueError(f'Missing data: {relative}')
        if source.stat().st_size != item['bytes']:
            raise ValueError(f'Incomplete/changed data: {relative}')
        entries.append((source, relative, item))
    for trial in TRIALS:
        if Path(trial, 'index.html') not in seen:
            raise ValueError(f'Missing required trial: {trial}')
    if Path('pd_insighter_trial_selector.html') not in seen:
        raise ValueError('Missing trial selector')
    return entries


def import_data(package, repo=REPO, apply=False):
    repo = repo.resolve()
    entries = plan(package, repo)
    print(f'{len(entries)} result files, {sum(item[2]["bytes"] for item in entries):,} bytes.')
    if not apply:
        print('Dry run only. Add --apply to verify hashes and copy into a new runs/.')
        return len(entries)
    # Preflight the entire selected payload before creating any destination data.
    for source, relative, item in entries:
        if digest(source) != item['sha256']:
            raise ValueError(f'Checksum mismatch: {relative}. Use the unchanged extracted ZIP.')
    destination = repo / 'runs'
    destination.mkdir()  # Exclusive: a concurrent import cannot overwrite this tree.
    for source, relative, item in entries:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as incoming, target.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=8 * 1024**2)
        if digest(target) != item['sha256']:
            raise ValueError(f'Copied file failed verification: {relative}')
    print('Imported and SHA-256 checked result data. Source package and Git code unchanged.')
    print('Next: relocate_copy.py --root .. --data-only, then --data-only --apply.')
    return len(entries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    import_data(args.package, apply=args.apply)
