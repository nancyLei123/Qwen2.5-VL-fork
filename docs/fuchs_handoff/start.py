#!/usr/bin/env python3
"""Start only the downloaded Fuchs workspace, never the original research tree."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import webbrowser


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspace"


def verify():
    errors = []
    manifests = sorted((ROOT / "checksums").glob("*.json"))
    if not manifests:
        raise SystemExit("No package checksums found; extract the entire ZIP first.")
    count = 0
    for manifest in manifests:
        for item in json.loads(manifest.read_text())["files"]:
            path = ROOT / item["path"]
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                errors.append(item["path"])
                continue
            value = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                    value.update(chunk)
            if value.hexdigest() != item["sha256"]:
                errors.append(item["path"])
            count += 1
    print(f"Checked {count} files; {len(errors)} missing/changed.")
    for path in errors[:30]:
        print(path)
    if errors:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8782)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
        return
    if sys.version_info < (3, 11):
        raise SystemExit("Please use Python 3.11 or newer.")
    if WORKSPACE.resolve() == Path("/Users/leijianan/gen1_glasses"):
        raise SystemExit("Refusing to run against the original workspace.")
    missing = [name for name in ("numpy", "pandas", "scipy", "openpyxl", "cv2")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit("Missing dependencies: " + ", ".join(missing) +
                         "\nRun: python -m pip install -r requirements.txt")
    # Rebase paths only inside the downloaded copy; the existing utility backs up text.
    scripts = WORKSPACE / "Qwen2.5-VL-fork/scripts"
    subprocess.run([sys.executable, str(WORKSPACE / "Qwen2.5-VL-fork/docs/reproduction/relocate_copy.py"),
                    "--root", str(WORKSPACE), "--apply"], check=True)
    url = f"http://127.0.0.1:{args.port}/Qwen2.5-VL-fork/runs/pd_insighter_trial_selector.html"
    print("Open " + url, flush=True)
    print("Keep this terminal open. Labels are saved only in this local copy.", flush=True)
    if not args.no_browser:
        timer = threading.Timer(1.5, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    try:
        subprocess.run([sys.executable, str(scripts / "pd_insighter_server.py"),
                        "--host", "127.0.0.1", "--port", str(args.port)], cwd=WORKSPACE, check=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
