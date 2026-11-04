#!/usr/bin/env python3
"""Export motion-centered Aria submaps without overwriting existing scenes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

SUBMAPS = (
    {
        "name": "sitting_context",
        "start_s": 0.0,
        "end_s": 24.0,
        "focus": "approach, stand-to-sit, seated context, and return to standing",
        "states": ["standing", "walking", "sitting"],
        "motion_sequence": [
            [0.0, 4.0, "standing"],
            [4.0, 13.0, "walking"],
            [13.0, 16.0, "standing"],
            [16.0, 22.0, "sitting"],
            [22.0, 24.0, "standing"],
        ],
    },
    {
        "name": "upstairs_context",
        "start_s": 78.0,
        "end_s": 120.0,
        "focus": "pre-stair standing, two upstairs bouts, landing, and connecting walk",
        "states": ["standing", "upstairs", "walking"],
        "motion_sequence": [
            [78.0, 82.0, "standing"],
            [82.0, 96.0, "upstairs"],
            [96.0, 105.0, "walking"],
            [105.0, 118.0, "upstairs"],
            [118.0, 120.0, "standing"],
        ],
    },
    {
        "name": "corridor_walking",
        "start_s": 120.0,
        "end_s": 190.0,
        "focus": "long continuous walking context through the corridor",
        "states": ["walking"],
        "motion_sequence": [[120.0, 190.0, "walking"]],
    },
    {
        "name": "downstairs_context",
        "start_s": 190.0,
        "end_s": 216.0,
        "focus": "pre-stair standing, downstairs bouts, connecting walk, and final standing",
        "states": ["standing", "downstairs", "walking"],
        "motion_sequence": [
            [190.0, 195.0, "standing"],
            [195.0, 201.0, "downstairs"],
            [201.0, 203.0, "walking"],
            [203.0, 210.0, "downstairs"],
            [210.0, 214.0, "walking"],
            [214.0, 216.0, "standing"],
        ],
    },
    {
        "name": "connecting_walk_gap",
        "selection_source": "untrained_coverage",
        "start_s": 58.0,
        "end_s": 80.0,
        "focus": "corridor coverage for the previously untrained 60-78 s interval, with boundary overlap",
        "states": [],
        "motion_sequence": [],
    },
    {
        "name": "final_walk_gap",
        "selection_source": "untrained_coverage",
        "start_s": 212.0,
        "end_s": 236.52,
        "focus": "final stairwell context after 216 s, through the last supported exported camera view",
        "states": [],
        "motion_sequence": [],
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-id", default="Ntrial3")
    parser.add_argument(
        "--only",
        action="append",
        choices=[item["name"] for item in SUBMAPS],
        help="Export only the named submap; repeat for more than one.",
    )
    parser.add_argument("--candidate-fps", type=float, default=6.0)
    return parser.parse_args()


def training_command(dataset: Path, opensplat: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/train_aria_gaussian_scene.py"),
        "--dataset-dir",
        str(dataset),
        "--opensplat",
        str(opensplat),
        "--output-dir",
        str(dataset / "trained_7000_fullres"),
        "--iterations",
        "7000",
        "--downscale",
        "1",
        "--validation-image",
        "auto",
        "--device",
        "auto",
        "--max-gaussians",
        "1000000",
        "--sh-degree",
        "3",
        "--sh-degree-interval",
        "1000",
        "--refine-every",
        "500",
        "--densify-from",
        "500",
        "--densify-until",
        "-1",
    ]


def write_gpu_script(root: Path, commands: list[list[str]]) -> Path:
    destination = root / "run_all_7000_fullres_gpu.sh"
    lines = [
        "#!/bin/zsh",
        "set -euo pipefail",
        "",
        "# Override this path when using an NVIDIA or remote GPU build.",
        f'OPEN_SPLAT="${{OPEN_SPLAT:-{ROOT.parent / "tools/OpenSplat-1.2.2/build-mps-torch213/opensplat"}}}"',
        'TORCH_LIB="${TORCH_LIB:-/Users/leijianan/Downloads/创业项目/无人机/outputs/uav_material_analysis/.venv_uav_bridge/lib/python3.11/site-packages/torch/lib}"',
        'export DYLD_LIBRARY_PATH="$TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"',
        'if [[ ! -x "$OPEN_SPLAT" ]]; then',
        '  echo "GPU-enabled OpenSplat not found: $OPEN_SPLAT" >&2',
        '  echo "Install Xcode + Metal Toolchain and build OpenSplat with GPU_RUNTIME=MPS first." >&2',
        "  exit 2",
        "fi",
        "",
    ]
    for command in commands:
        command = command.copy()
        command[command.index("--opensplat") + 1] = '"$OPEN_SPLAT"'
        rendered = []
        for part in command:
            rendered.append(part if part == '"$OPEN_SPLAT"' else shlex.quote(part))
        lines.append(" \\\n  ".join(rendered))
        lines.append("")
    destination.write_text("\n".join(lines).rstrip() + "\n")
    destination.chmod(0o755)
    return destination


def _training_result(dataset: Path) -> tuple[str, dict | None, str | None]:
    full = dataset / "trained_7000_fullres" / "training_summary.json"
    smoke = dataset / "trained_smoke_fullres" / "training_summary.json"
    if full.is_file():
        return "7,000-iteration result available", json.loads(full.read_text()), "trained_7000_fullres/index.html"
    if smoke.is_file():
        return "full-resolution smoke test available", json.loads(smoke.read_text()), "trained_smoke_fullres/index.html"
    return "dataset ready; training pending", None, None


def write_gallery(root: Path, trial_id: str, records: list[dict], gpu_script: Path) -> Path:
    cards = []
    playback_available = (root.parent / "playback.html").is_file()
    playback_link = '<a class="button" href="../playback.html">Synchronized video + 3D playback</a>' if playback_available else ""
    completed = sum(
        (Path(record["dataset_dir"]) / "trained_7000_fullres/training_summary.json").is_file()
        for record in records
    )
    for record in records:
        dataset = Path(record["dataset_dir"])
        status, result, result_href = _training_result(dataset)
        sequence = "Times are VRS recording seconds. Submap names describe training regions, not verified action labels. Use synchronized playback for time-aligned classifications."
        metrics = ""
        action = ""
        image_href = record["name"] + "/contact_sheet.jpg"
        if result:
            metrics = (
                f"<p><strong>{result['backend']}; {result['iterations']:,} iterations; "
                f"held-out PSNR {result['validation_psnr_db']:.2f} dB.</strong></p>"
            )
            image_href = record["name"] + "/" + result_href
            action = f'<a class="button" href="{html.escape(image_href, quote=True)}">Open interactive 3D scene</a>'
            if playback_available:
                action += f'<a class="button" href="../playback.html?t={record["start_s"] + 1:g}">Play video + scene</a>'
        cards.append(
            f'''<article class="submap-card">
              <a href="{html.escape(image_href, quote=True)}"><img src="{html.escape(record['name'] + '/contact_sheet.jpg', quote=True)}" alt="Open the {html.escape(record['name'])} interactive 3D reconstruction"></a>
              <div class="copy"><span class="status">{html.escape(status)}</span><h2>{html.escape(record['name'].replace('_', ' ').title())}</h2>
              <p>{record['start_s']:g}-{record['end_s']:g} s. {html.escape(record['focus'])}.</p>
              <p class="sequence">{html.escape(sequence)}</p>
              <div class="stats"><span><strong>{record['frames_selected']:,}</strong> calibrated views</span><span><strong>{record['initializer_points']:,}</strong> initializer points</span><span><strong>{record['median_adjacent_translation_m']:.3f} m</strong> median camera baseline</span></div>
              {metrics}<div class="actions">{action}<a class="button secondary" href="{html.escape(record['name'] + '/contact_sheet.jpg', quote=True)}">Source frames</a><a class="button secondary" href="{html.escape(record['name'] + '/dataset_manifest.json', quote=True)}">Dataset manifest</a></div></div>
            </article>'''
        )
    destination = root / "index.html"
    destination.write_text(
        f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
        <title>{html.escape(trial_id)} Motion-Centered Gaussian Submaps | PD Insighter</title>
        <style>
        :root{{--ink:#17231c;--muted:#66736b;--green:#187859;--line:#d9e3da;--paper:#f5f4ec;--amber:#9a6712}}
        *{{box-sizing:border-box}} body{{margin:0;color:var(--ink);background:radial-gradient(circle at 10% 0,#e4e9dc,transparent 48%),var(--paper);font-family:"Avenir Next","Gill Sans",sans-serif}}
        main{{max-width:1500px;margin:auto;padding:24px}} header{{background:#ffffffeb;border:1px solid var(--line);border-radius:26px;padding:28px;margin-bottom:18px}}
        h1{{font-size:clamp(30px,4vw,52px);letter-spacing:-.045em;margin:0 0 8px}} h2{{margin:6px 0;font-size:24px}} p{{color:var(--muted);line-height:1.55}}
        .top{{display:flex;justify-content:space-between;gap:20px;align-items:start;flex-wrap:wrap}} .badge,.status{{display:inline-block;border-radius:999px;padding:8px 12px;background:#fff2ce;color:#744f0d;font-weight:800;font-size:13px}}
        .grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}} .submap-card{{overflow:hidden;background:#ffffffed;border:1px solid var(--line);border-radius:24px;box-shadow:0 16px 42px #263f2910}}
        .submap-card img{{width:100%;display:block;aspect-ratio:2/1;object-fit:cover;background:#19201b}} .copy{{padding:22px}} .sequence{{font-size:13px;background:#eef4ef;padding:10px 12px;border-radius:12px}}
        .stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:14px 0}} .stats span{{padding:10px;background:#f6f7f1;border-radius:12px;color:var(--muted);font-size:12px}} .stats strong{{display:block;color:var(--ink);font-size:17px}}
        .actions{{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}} .button{{text-decoration:none;background:var(--green);color:white;padding:10px 14px;border-radius:999px;font-weight:800}} .secondary{{background:white;color:var(--ink);border:1px solid var(--line)}}
        .notice{{border-left:4px solid #c88b22;background:#fff8e8;border-radius:10px;padding:10px 14px}} code{{overflow-wrap:anywhere}} @media(max-width:800px){{.grid{{grid-template-columns:1fr}}.stats{{grid-template-columns:1fr}}}}
        </style></head><body><main>
        <header><div class="top"><div><a class="button secondary" href="../index.html">Back to Environment</a><h1>Motion-Centered Gaussian Submaps</h1><p>{html.escape(trial_id)} is divided by clinically relevant motion context rather than reconstructed as one long session.</p></div><span class="badge">{len(records)} auditable submaps</span></div>
        <p class="notice"><strong>Current status:</strong> {completed} of {len(records)} calibrated submaps have completed the formal 7,000-iteration full-resolution run. Each completed card reports its held-out-view PSNR and links to the MPS-trained result.</p>
        <div class="actions">{playback_link}</div><p>Prepared command: <code>{html.escape(str(gpu_script))}</code></p></header>
        <section class="grid">{''.join(cards)}</section>
        </main></body></html>'''
    )
    return destination


def main() -> None:
    args = parse_args()
    selected = [item for item in SUBMAPS if not args.only or item["name"] in args.only]
    root = (
        ROOT
        / "runs"
        / args.trial_id
        / "environment_reconstruction"
        / "gaussian_submaps"
    )
    root.mkdir(parents=True, exist_ok=True)
    exporter = ROOT / "scripts/export_aria_gaussian_dataset.py"
    gpu_binary = ROOT.parent / "tools/OpenSplat-1.2.2/build-mps-torch213/opensplat"
    records = []
    commands = []

    for profile in selected:
        output = root / profile["name"]
        manifest_path = output / "dataset_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            segment = manifest.get("segment", {})
            if (
                segment.get("start_s") != profile["start_s"]
                or segment.get("end_s") != profile["end_s"]
            ):
                raise RuntimeError(f"Existing submap has different bounds: {output}")
            print(f"Keeping existing submap: {output}", flush=True)
        else:
            command = [
                sys.executable,
                str(exporter),
                "--trial-id",
                args.trial_id,
                "--start-s",
                str(profile["start_s"]),
                "--end-s",
                str(profile["end_s"]),
                "--candidate-fps",
                str(args.candidate_fps),
                "--min-translation-m",
                "0.04",
                "--min-rotation-deg",
                "3",
                "--max-frame-gap-s",
                "0.5",
                "--max-point-path-distance-m",
                "5",
                "--max-points",
                "350000",
                "--output-dir",
                str(output),
            ]
            print(f"Exporting {profile['name']} ({profile['start_s']:g}-{profile['end_s']:g}s).", flush=True)
            subprocess.run(command, check=True)
            manifest = json.loads(manifest_path.read_text())

        train = training_command(output, gpu_binary)
        commands.append(train)
        records.append(
            {
                **profile,
                "dataset_dir": str(output),
                "frames_selected": manifest["frames"]["selected"],
                "initializer_points": manifest["points"]["written"],
                "median_adjacent_translation_m": manifest["frames"][
                    "adjacent_translation_m"
                ]["median"],
                "median_sharpness": manifest["frames"]["sharpness"]["median"],
                "gpu_training_command": train,
            }
        )

    gpu_script = write_gpu_script(root, commands)
    summary = {
        "trial_id": args.trial_id,
        "status": "datasets_exported_gpu_training_complete"
        if all(
            (Path(record["dataset_dir"]) / "trained_7000_fullres/training_summary.json").is_file()
            for record in records
        )
        else "datasets_exported_gpu_training_pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_motion_labels": str(
            ROOT
            / "runs"
            / args.trial_id
            / "raw_parameter_diagnostics"
            / "continuous_motion_segments.json"
        ),
        "selection_policy": {
            "candidate_fps": args.candidate_fps,
            "minimum_translation_m": 0.04,
            "minimum_rotation_deg": 3.0,
            "maximum_frame_gap_s": 0.5,
            "context_rule": "Include stable motion before and after each clinical action.",
        },
        "training_target": {
            "iterations": 7000,
            "downscale_factor": 1,
            "spherical_harmonics_degree": 3,
            "validation_frame": "automatic middle selected frame",
            "backend": "Metal or CUDA OpenSplat; never label the CPU fallback as GPU",
        },
        "submaps": records,
        "gpu_script": str(gpu_script),
    }
    (root / "submaps_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    gallery = write_gallery(root, args.trial_id, records, gpu_script)
    print(f"Submap manifest: {root / 'submaps_manifest.json'}")
    print(f"GPU training script: {gpu_script}")
    print(f"Submap gallery: {gallery}")


if __name__ == "__main__":
    main()
