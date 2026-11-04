#!/usr/bin/env python3
"""Batch helper for registered-trial workflows.

This script reduces the per-trial manual work by:
- discovering the real .vrs filename inside each trial folder
- registering manifests automatically
- running extraction, MPS, and/or analysis in sequence
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


GEN1_ROOT = Path(__file__).resolve().parents[2]
QWEN_REPO = GEN1_ROOT / "Qwen2.5-VL-fork"
VRS_REPO = GEN1_ROOT / "VRS_Pre_Processing"
PROJECTARIA_BIN = GEN1_ROOT / "projectaria_tools" / ".venv" / "bin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        nargs="+",
        required=True,
        help="Trial ids, e.g. trial6 trial7 trial8",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["register", "extract"],
        choices=["register", "extract", "mps", "analyze"],
        help="Ordered steps to run for each trial.",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        choices=["EYE_GAZE", "SLAM", "HAND_TRACKING"],
        help="Optional MPS feature subset.",
    )
    parser.add_argument(
        "--force-mps",
        action="store_true",
        help="Pass --force to aria_mps.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Pass --retry-failed to aria_mps.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining trials even if one step fails.",
    )
    parser.add_argument(
        "--username",
        help="Optional MPS username/email. Useful when keychain token lookup fails.",
    )
    parser.add_argument(
        "--password",
        help="Optional MPS password. Useful when keychain token lookup fails.",
    )
    return parser.parse_args()


def discover_recording_stem(trial_id: str) -> str:
    trial_dir = VRS_REPO / "data" / trial_id
    vrs_files = sorted(path for path in trial_dir.glob("*.vrs") if path.is_file())
    if not vrs_files:
        raise FileNotFoundError(f"No .vrs file found in {trial_dir}")
    if len(vrs_files) > 1:
        names = ", ".join(path.name for path in vrs_files)
        raise RuntimeError(f"Multiple .vrs files found in {trial_dir}: {names}")
    return vrs_files[0].name.removesuffix(".vrs")


def manifest_path(trial_id: str) -> Path:
    return QWEN_REPO / "runs" / trial_id / "trial_manifest.json"


def run_cmd(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def register_trial(trial_id: str) -> None:
    stem = discover_recording_stem(trial_id)
    cmd = [
        sys.executable,
        str(QWEN_REPO / "scripts" / "register_trial.py"),
        "--trial-id",
        trial_id,
        "--recording-stem",
        stem,
    ]
    run_cmd(cmd, cwd=QWEN_REPO)


def extract_trial(trial_id: str) -> None:
    cmd = [
        sys.executable,
        str(QWEN_REPO / "scripts" / "extract_registered_trial.py"),
        "--trial-id",
        trial_id,
    ]
    run_cmd(cmd, cwd=QWEN_REPO)


def load_manifest(trial_id: str) -> dict[str, object]:
    path = manifest_path(trial_id)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return json.loads(path.read_text())


def run_mps(
    trial_id: str,
    *,
    features: list[str] | None,
    force: bool,
    retry_failed: bool,
    username: str | None,
    password: str | None,
) -> None:
    manifest = load_manifest(trial_id)
    vrs_path = Path(manifest["raw_data"]["vrs"]["path"])

    cmd = [str(PROJECTARIA_BIN / "aria_mps"), "single", "-i", str(vrs_path), "--no-ui"]
    if username:
        cmd.extend(["-u", username])
    if password:
        cmd.extend(["-p", password])
    if features:
        cmd.extend(["--features", *features])
    if force:
        cmd.append("--force")
    if retry_failed:
        cmd.append("--retry-failed")

    env = os.environ.copy()
    env["PATH"] = f"{PROJECTARIA_BIN}:{env.get('PATH', '')}"
    env["SSL_CERT_FILE"] = "/etc/ssl/cert.pem"
    run_cmd(cmd, cwd=QWEN_REPO, env=env)


def analyze_trial(trial_id: str) -> None:
    cmd = [
        sys.executable,
        str(QWEN_REPO / "scripts" / "run_registered_trial.py"),
        "--trial-id",
        trial_id,
    ]
    run_cmd(cmd, cwd=QWEN_REPO)


def main() -> None:
    args = parse_args()
    for trial_id in args.trials:
        print(f"\n=== {trial_id} ===")
        try:
            for step in args.steps:
                if step == "register":
                    register_trial(trial_id)
                elif step == "extract":
                    extract_trial(trial_id)
                elif step == "mps":
                    run_mps(
                        trial_id,
                        features=args.features,
                        force=args.force_mps,
                        retry_failed=args.retry_failed,
                        username=args.username,
                        password=args.password,
                    )
                elif step == "analyze":
                    analyze_trial(trial_id)
        except Exception as exc:
            print(f"[ERROR] {trial_id}: {exc}")
            if not args.continue_on_error:
                raise


if __name__ == "__main__":
    main()
