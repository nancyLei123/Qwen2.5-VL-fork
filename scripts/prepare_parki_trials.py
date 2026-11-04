#!/usr/bin/env python3
"""Pair the three October recordings without moving or deleting originals."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "VRS_Pre_Processing/data/3Dreconstruction"
PAIRS = [
    ("Parki", "Parki_", "Parki_2026-10-27 21:21:00.MP4"),
    ("Parki2", "Parki2", "Parki2_2026-10-27 21:32:35.MP4"),
    ("3c", "3c6ed915-3279-4edf-955b-6523e1e11bb8", "3c_2026-10-27 19:43:03.MP4"),
]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    records = []
    for trial, stem, filename in PAIRS:
        source = Path.home() / "Downloads" / filename
        target = ROOT / "downward videos" / f"{trial}.MP4"
        vrs = RAW / f"{stem}.vrs"
        sidecar = RAW / f"{stem}.vrs.json"
        mps = RAW / f"mps_{stem}_vrs"
        for required in (source, vrs, sidecar, mps / "slam/closed_loop_trajectory.csv"):
            if not required.is_file():
                raise FileNotFoundError(required)
        source_hash = digest(source)
        if not target.exists():
            # Exclusive creation prevents overwriting any prior research media.
            with source.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
        if digest(target) != source_hash:
            raise ValueError(f"Copy mismatch; original retained: {target}")
        records.append(dict(trial_id=trial, recording_stem=stem,
                            raw_vrs=str(vrs), raw_json=str(sidecar), mps_dir=str(mps),
                            downward_original=str(source), downward_video=str(target),
                            downward_sha256=source_hash, pairing_basis="User-provided filenames",
                            processing_requested=trial in {"Parki", "Parki2"}))
        print(f"Verified downward copy: {trial} -> {target}", flush=True)
        if trial in {"Parki", "Parki2"}:
            manifest = ROOT / "Qwen2.5-VL-fork/runs" / trial / "trial_manifest.json"
            if not manifest.exists():
                subprocess.run([sys.executable, str(Path(__file__).with_name("register_trial.py")),
                                "--trial-id", trial, "--recording-stem", stem,
                                "--data-dir", str(RAW), "--trial-type", "parkinson_simulated"], check=True)
            metadata = json.loads(manifest.read_text())
            metadata["downward_media"] = {
                "original_path": str(source), "path": str(target), "sha256": source_hash,
                "creation_time_timezone": "UTC",
                "timestamp_basis": "Embedded UTC creation time agrees with the user-named local-time filename (America/New_York).",
            }
            for group in ("raw_data", "mps_outputs", "extracted_media", "analysis_outputs"):
                for entry in metadata.get(group, {}).values():
                    if isinstance(entry, dict) and "path" in entry:
                        entry["exists"] = Path(entry["path"]).exists()
            metadata["recommended_commands"] = {
                "extract_video": f"{sys.executable} scripts/extract_registered_trial.py --trial-id {trial}",
                "build_dashboard": f"{sys.executable} scripts/build_downward_dual_view_dashboards.py --trials {trial}",
                "stride_analysis": f"{sys.executable} scripts/generate_step_stride_diagnostics.py --trials {trial} --out-dir runs/{trial}/stride_step_diagnostics",
                "note": "Run from Qwen2.5-VL-fork. MPS is already complete; no resubmission is needed.",
            }
            manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    report = RAW / "recording_pairs_20261029.json"
    report.write_text(json.dumps({"originals_preserved": True, "recordings": records}, indent=2) + "\n")
    print(f"Pairing inventory: {report}")


if __name__ == "__main__":
    main()
