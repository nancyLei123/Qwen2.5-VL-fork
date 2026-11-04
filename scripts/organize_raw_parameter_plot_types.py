#!/usr/bin/env python3
"""Group raw-parameter diagnostic plots by plot type for review/presentation."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


QWEN_REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = QWEN_REPO / "runs"


PLOT_TYPES = [
    {
        "folder": "01_height_over_time",
        "goal": "Basic Parameter Understanding",
        "description": "Raw SLAM height over session time with manual-label background.",
        "per_trial": "height_over_time.png",
    },
    {
        "folder": "02_instantaneous_speed_over_time",
        "goal": "Basic Parameter Understanding",
        "description": "Instantaneous speed with 0.5s/2s rolling means and old 2s window means.",
        "per_trial": "instantaneous_linear_speed_over_time.png",
    },
    {
        "folder": "03_height_and_speed_over_time",
        "goal": "Basic Parameter Understanding",
        "description": "Raw SLAM height and 0.5s rolling speed on the same time axis.",
        "per_trial": "height_and_speed_over_time.png",
    },
    {
        "folder": "04_xy_path_colored_by_height",
        "goal": "Basic Parameter Understanding",
        "description": "XY trajectory colored by raw SLAM height.",
        "per_trial": "xy_path_colored_by_height.png",
    },
    {
        "folder": "05_vertical_delta_by_window_size",
        "goal": "Basic Parameter Understanding",
        "description": "Vertical delta over time for multiple window sizes.",
        "per_trial": "vertical_delta_by_window_size.png",
    },
    {
        "folder": "06_mean_height_vs_vertical_delta_2s",
        "goal": "Basic Parameter Understanding",
        "description": "2s mean height line with 2s vertical-delta bars.",
        "per_trial": "mean_height_vs_vertical_delta_2s.png",
    },
    {
        "folder": "07_boxplot_mean_z_by_label",
        "goal": "Categorizing Motions",
        "description": "2s mean raw SLAM z grouped by effective manual label.",
        "per_trial": "mean_z_by_motion_label_boxplot.png",
        "combined": "mean_z_by_motion_label_boxplot_Ntrial1_Ntrial2_Ntrial3.png",
    },
    {
        "folder": "08_boxplot_speed_by_label",
        "goal": "Categorizing Motions",
        "description": "2s mean linear speed grouped by effective manual label.",
        "per_trial": "mean_linear_speed_by_motion_label_boxplot.png",
        "combined": "mean_linear_speed_by_motion_label_boxplot_Ntrial1_Ntrial2_Ntrial3.png",
    },
    {
        "folder": "09_boxplot_horizontal_displacement_by_label",
        "goal": "Categorizing Motions",
        "description": "2s endpoint horizontal displacement grouped by effective manual label.",
        "per_trial": "horizontal_displacement_by_motion_label_boxplot.png",
        "combined": "horizontal_displacement_by_motion_label_boxplot_Ntrial1_Ntrial2_Ntrial3.png",
    },
    {
        "folder": "10_boxplot_horizontal_path_length_by_label",
        "goal": "Categorizing Motions",
        "description": "2s frame-to-frame horizontal path length grouped by effective manual label.",
        "per_trial": "horizontal_path_length_by_motion_label_boxplot.png",
        "combined": "horizontal_path_length_by_motion_label_boxplot_Ntrial1_Ntrial2_Ntrial3.png",
    },
    {
        "folder": "11_boxplot_vertical_displacement_by_label",
        "goal": "Categorizing Motions",
        "description": "2s vertical displacement grouped by effective manual label.",
        "per_trial": "vertical_displacement_by_motion_label_boxplot.png",
        "combined": "vertical_displacement_by_motion_label_boxplot_Ntrial1_Ntrial2_Ntrial3.png",
    },
    {
        "folder": "12_boxplot_height_slope_by_label",
        "goal": "Categorizing Motions",
        "description": "2s height slope grouped by effective manual label.",
        "per_trial": "height_slope_by_motion_label_boxplot.png",
        "combined": "height_slope_by_motion_label_boxplot_Ntrial1_Ntrial2_Ntrial3.png",
    },
    {
        "folder": "13_scatter_mean_z_vs_speed",
        "goal": "Categorizing Motions",
        "description": "2s mean raw SLAM z vs 2s mean speed, colored by label.",
        "per_trial": "mean_z_vs_mean_linear_speed_scatter.png",
        "combined": "mean_z_vs_mean_linear_speed_scatter_Ntrial1_Ntrial2_Ntrial3.png",
    },
    {
        "folder": "14_scatter_horizontal_displacement_vs_speed",
        "goal": "Categorizing Motions",
        "description": "2s endpoint horizontal displacement vs 2s mean speed, colored by label.",
        "per_trial": "horizontal_displacement_vs_mean_linear_speed_scatter.png",
        "combined": "horizontal_displacement_vs_mean_linear_speed_scatter_Ntrial1_Ntrial2_Ntrial3.png",
    },
    {
        "folder": "19_scatter_horizontal_path_length_vs_speed",
        "goal": "Categorizing Motions",
        "description": "2s horizontal path length vs 2s mean speed, colored by label.",
        "per_trial": "horizontal_path_length_vs_mean_linear_speed_scatter.png",
        "combined": "horizontal_path_length_vs_mean_linear_speed_scatter_Ntrial1_Ntrial2_Ntrial3.png",
    },
    {
        "folder": "15_scatter_vertical_displacement_vs_speed",
        "goal": "Categorizing Motions",
        "description": "2s vertical displacement vs 2s mean speed, colored by label.",
        "per_trial": "vertical_displacement_vs_mean_linear_speed_scatter.png",
        "combined": "vertical_displacement_vs_mean_linear_speed_scatter_Ntrial1_Ntrial2_Ntrial3.png",
    },
    {
        "folder": "16_scatter_vertical_delta_vs_horizontal_path",
        "goal": "Categorizing Motions",
        "description": "2s vertical displacement vs 2s horizontal path length, colored by label.",
        "per_trial": "vertical_delta_vs_horizontal_path_length_scatter.png",
        "combined": "vertical_delta_vs_horizontal_path_length_scatter_Ntrial1_Ntrial2_Ntrial3.png",
    },
    {
        "folder": "17_scatter_height_slope_vs_horizontal_path",
        "goal": "Categorizing Motions",
        "description": "2s height slope vs 2s horizontal path length, colored by label.",
        "per_trial": "height_slope_vs_horizontal_path_length_scatter.png",
        "combined": "height_slope_vs_horizontal_path_length_scatter_Ntrial1_Ntrial2_Ntrial3.png",
    },
    {
        "folder": "18_scatter_raw_z_vs_instantaneous_speed",
        "goal": "Basic Parameter Understanding",
        "description": "Sampled raw SLAM z vs instantaneous speed, colored by label.",
        "per_trial": "instantaneous_z_vs_speed_scatter.png",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", nargs="+", default=["Ntrial1", "Ntrial2", "Ntrial3"])
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "raw_parameter_diagnostics" / "by_plot_type")
    parser.add_argument(
        "--goal-output-dir",
        type=Path,
        default=RUNS_DIR / "raw_parameter_diagnostics" / "by_analysis_goal",
        help="Also write a clearer goal-based organization with combined plots separated at top level.",
    )
    parser.add_argument(
        "--curated-output-dir",
        type=Path,
        default=RUNS_DIR / "raw_parameter_diagnostics" / "most_meaningful_graphs",
        help="Also write a small presentation-ready folder with only the highest-signal plots.",
    )
    return parser.parse_args()


def copy_if_exists(source: Path, destination: Path, rows: list[dict[str, str]], plot_type: str, description: str) -> None:
    if not source.exists():
        rows.append(
            {
                "plot_type": plot_type,
                "description": description,
                "source": str(source),
                "organized_path": "",
                "status": "missing",
            }
        )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    rows.append(
        {
            "plot_type": plot_type,
            "description": description,
            "source": str(source),
            "organized_path": str(destination),
            "status": "copied",
        }
    )


def reset_generated_output_dir(output_dir: Path) -> None:
    """Clear a generated diagnostics folder while guarding against accidental deletes."""
    diagnostics_root = (RUNS_DIR / "raw_parameter_diagnostics").resolve()
    target = output_dir.resolve()
    if diagnostics_root not in target.parents:
        raise ValueError(f"Refusing to clear non-diagnostics directory: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def organize_by_analysis_goal(trials: list[str], output_dir: Path) -> tuple[Path, int, int]:
    rows: list[dict[str, str]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_root = output_dir / "00_Combined Summary Plots"

    for spec in PLOT_TYPES:
        goal = spec["goal"]
        goal_folder = output_dir / goal / spec["folder"]
        description = spec["description"]
        plot_type = spec["folder"]

        for trial_id in trials:
            source = RUNS_DIR / trial_id / "raw_parameter_diagnostics" / spec["per_trial"]
            destination = goal_folder / f"{trial_id}_{spec['per_trial']}"
            copy_if_exists(source, destination, rows, plot_type, description)

        combined_name = spec.get("combined")
        if combined_name:
            source = RUNS_DIR / "raw_parameter_diagnostics" / combined_name
            destination = combined_root / goal / f"{spec['folder']}__combined_{combined_name}"
            copy_if_exists(source, destination, rows, plot_type, description)

    manifest_path = output_dir / "analysis_goal_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["plot_type", "description", "source", "organized_path", "status"])
        writer.writeheader()
        writer.writerows(rows)

    readme_lines = [
        "# Raw Parameter Diagnostic Plots by Analysis Goal",
        "",
        "This folder is the clearer replacement view for the older `by_plot_type` organization.",
        "",
        "## Why Some Plots Have Combined Versions",
        "",
        "Combined Ntrial1-3 versions are only created for boxplots and scatter plots where the x/y axes are comparable across trials.",
        "Time-series plots stay per-trial because each trial has a different duration, sequence, and time axis; combining them would be visually confusing.",
        "",
        "## Folder Labels",
        "",
        "- `Basic Parameter Understanding`: plots for understanding what the raw SLAM parameters mean over time.",
        "- `Categorizing Motions`: plots that directly help separate sitting, standing, walking, upstairs, and downstairs.",
        "- `00_Combined Summary Plots`: combined Ntrial1-3 boxplots/scatters, kept outside the per-trial plot folders so they are easy to find.",
        "",
        "## Basic Parameter Understanding",
        "",
    ]
    for spec in PLOT_TYPES:
        if spec["goal"] == "Basic Parameter Understanding":
            readme_lines.append(f"- `{spec['folder']}`: {spec['description']}")
    readme_lines.extend(["", "## Categorizing Motions", ""])
    for spec in PLOT_TYPES:
        if spec["goal"] == "Categorizing Motions":
            readme_lines.append(f"- `{spec['folder']}`: {spec['description']}")
    readme_lines.extend(["", f"Manifest: `{manifest_path.name}`"])
    (output_dir / "README.md").write_text("\n".join(readme_lines))

    copied = sum(1 for row in rows if row["status"] == "copied")
    missing = sum(1 for row in rows if row["status"] == "missing")
    return manifest_path, copied, missing


def organize_curated_plots(trials: list[str], output_dir: Path) -> tuple[Path, int, int]:
    """Create a compact folder of only the most useful plots for discussion."""
    rows: list[dict[str, str]] = []
    reset_generated_output_dir(output_dir)

    per_trial_choices = [
        {
            "category": "Basic Parameter",
            "plot_type": "03_height_and_speed_over_time",
            "filename": "height_and_speed_over_time.png",
            "title": "Height and speed over time",
            "why": "Best first-look plot for checking how raw SLAM height and movement speed change across the session.",
        },
        {
            "category": "Basic Parameter",
            "plot_type": "06_mean_height_vs_vertical_delta_2s",
            "filename": "mean_height_vs_vertical_delta_2s.png",
            "title": "Mean height versus 2s vertical delta",
            "why": "Shows how the old 2-second derived feature relates to the underlying height signal.",
        },
    ]
    combined_choices = [
        {
            "category": "Categorizing Motions",
            "filename": "mean_z_by_motion_label_boxplot_Ntrial1_Ntrial2_Ntrial3.png",
            "title": "Mean height by label",
            "why": "Most direct evidence for sitting versus standing posture height.",
        },
        {
            "category": "Categorizing Motions",
            "filename": "mean_linear_speed_by_motion_label_boxplot_Ntrial1_Ntrial2_Ntrial3.png",
            "title": "Mean linear speed by label",
            "why": "Clear summary of which labels are high-motion versus low-motion.",
        },
        {
            "category": "Categorizing Motions",
            "filename": "horizontal_path_length_by_motion_label_boxplot_Ntrial1_Ntrial2_Ntrial3.png",
            "title": "Horizontal path length by label",
            "why": "Better movement-amount cue than endpoint displacement because it accumulates frame-to-frame path.",
        },
        {
            "category": "Categorizing Motions",
            "filename": "horizontal_path_length_vs_mean_linear_speed_scatter_Ntrial1_Ntrial2_Ntrial3.png",
            "title": "Horizontal path length versus mean speed",
            "why": "Directly shows the stronger total-movement feature against movement intensity.",
        },
        {
            "category": "Categorizing Motions",
            "filename": "height_slope_by_motion_label_boxplot_Ntrial1_Ntrial2_Ntrial3.png",
            "title": "Height slope by label",
            "why": "Key cue for upward versus downward motion trends.",
        },
        {
            "category": "Categorizing Motions",
            "filename": "height_slope_vs_horizontal_path_length_scatter_Ntrial1_Ntrial2_Ntrial3.png",
            "title": "Height slope versus horizontal path",
            "why": "Most compact scatter for separating walking, upstairs, downstairs, and low-motion posture clips.",
        },
    ]

    for choice_index, choice in enumerate(per_trial_choices, start=1):
        for trial_id in trials:
            source = RUNS_DIR / trial_id / "raw_parameter_diagnostics" / choice["filename"]
            safe_category = choice["category"].replace(" ", "_")
            destination = output_dir / f"{choice_index:02d}_{safe_category}__{trial_id}__{choice['filename']}"
            copy_if_exists(source, destination, rows, choice["plot_type"], f"{choice['title']}: {choice['why']}")

    start_index = len(per_trial_choices) + 1
    for offset, choice in enumerate(combined_choices, start=start_index):
        source = RUNS_DIR / "raw_parameter_diagnostics" / choice["filename"]
        safe_category = choice["category"].replace(" ", "_")
        destination = output_dir / f"{offset:02d}_{safe_category}__combined__{choice['filename']}"
        copy_if_exists(source, destination, rows, "combined_summary", f"{choice['title']}: {choice['why']}")

    manifest_path = output_dir / "most_meaningful_graphs_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["plot_type", "description", "source", "organized_path", "status"])
        writer.writeheader()
        writer.writerows(rows)

    readme_lines = [
        "# Most Meaningful Raw-Parameter Graphs",
        "",
        "This is the small, presentation-ready subset of the raw-parameter diagnostics.",
        "The full diagnostic collections remain in `by_analysis_goal` and `by_plot_type`.",
        "",
        "## Basic Parameter",
        "",
        "- `Height and speed over time`: best first-look plot for understanding raw SLAM height and movement speed during each trial.",
        "- `Mean height versus 2s vertical delta`: shows how the older 2-second feature relates to the underlying height signal.",
        "",
        "## Categorizing Motions",
        "",
        "- `Mean height by label`: strongest simple posture cue for sitting versus standing.",
        "- `Mean linear speed by label`: separates high-motion and low-motion states.",
        "- `Horizontal path length by label`: summarizes how much horizontal movement occurred in each window.",
        "- `Horizontal path length versus mean speed`: directly compares total horizontal movement with movement intensity.",
        "- `Height slope by label`: summarizes upward/downward height trend.",
        "- `Height slope versus horizontal path`: compact scatter for walking, stairs, and posture separation.",
        "",
        f"Manifest: `{manifest_path.name}`",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines))

    copied = sum(1 for row in rows if row["status"] == "copied")
    missing = sum(1 for row in rows if row["status"] == "missing")
    return manifest_path, copied, missing


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for spec in PLOT_TYPES:
        folder = args.output_dir / spec["folder"]
        description = spec["description"]
        plot_type = spec["folder"]
        for trial_id in args.trials:
            source = RUNS_DIR / trial_id / "raw_parameter_diagnostics" / spec["per_trial"]
            destination = folder / f"{trial_id}_{spec['per_trial']}"
            copy_if_exists(source, destination, rows, plot_type, description)

        combined_name = spec.get("combined")
        if combined_name:
            source = RUNS_DIR / "raw_parameter_diagnostics" / combined_name
            destination = folder / f"combined_{combined_name}"
            copy_if_exists(source, destination, rows, plot_type, description)

    manifest_path = args.output_dir / "plot_type_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["plot_type", "description", "source", "organized_path", "status"])
        writer.writeheader()
        writer.writerows(rows)

    readme_lines = [
        "# Raw Parameter Diagnostic Plots by Type",
        "",
        "This folder groups the Ntrial1-3 raw-parameter diagnostic plots by plot type.",
        "Each plot uses effective manual labels from the workbook when available.",
        "",
        "## Plot Types",
        "",
    ]
    for spec in PLOT_TYPES:
        readme_lines.append(f"- `{spec['folder']}`: {spec['description']}")
    readme_lines.append("")
    readme_lines.append(f"Manifest: `{manifest_path.name}`")
    (args.output_dir / "README.md").write_text("\n".join(readme_lines))

    copied = sum(1 for row in rows if row["status"] == "copied")
    missing = sum(1 for row in rows if row["status"] == "missing")
    print(f"Organized {copied} plots into {args.output_dir}")
    if missing:
        print(f"Missing {missing} expected plots; see {manifest_path}")
    else:
        print(f"All expected plots were found. Manifest: {manifest_path}")

    goal_manifest_path, goal_copied, goal_missing = organize_by_analysis_goal(args.trials, args.goal_output_dir)
    print(f"Organized {goal_copied} goal-labeled plots into {args.goal_output_dir}")
    if goal_missing:
        print(f"Missing {goal_missing} expected goal-labeled plots; see {goal_manifest_path}")
    else:
        print(f"All expected goal-labeled plots were found. Manifest: {goal_manifest_path}")

    curated_manifest_path, curated_copied, curated_missing = organize_curated_plots(
        args.trials,
        args.curated_output_dir,
    )
    print(f"Organized {curated_copied} curated plots into {args.curated_output_dir}")
    if curated_missing:
        print(f"Missing {curated_missing} expected curated plots; see {curated_manifest_path}")
    else:
        print(f"All expected curated plots were found. Manifest: {curated_manifest_path}")


if __name__ == "__main__":
    main()
