#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import List

import cv2
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_PROJECTARIA_REPO = ROOT_DIR / "projectaria_tools"
from projectaria_tools.core import calibration, data_provider  # noqa: E402


DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "VRS"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "extracted_videos"
DEFAULT_MASK_DIR = LOCAL_PROJECTARIA_REPO / "data" / "gen1" / "vignetting_mask"
RGB_LABEL = "camera-rgb"


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract RGB MP4 videos and thumbnails from Project Aria VRS files."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory to search recursively for VRS files.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where extracted media will be written.",
    )
    parser.add_argument(
        "--vrs_file",
        type=str,
        default="",
        help="Specific VRS filename or absolute path to process.",
    )
    parser.add_argument(
        "--undistort",
        type=str2bool,
        default=True,
        help="Whether to undistort RGB frames to a linear camera model.",
    )
    parser.add_argument(
        "--color_correct",
        type=str2bool,
        default=True,
        help="Whether to enable Project Aria color correction.",
    )
    parser.add_argument(
        "--devignette",
        type=str2bool,
        default=True,
        help="Whether to enable Project Aria devignetting.",
    )
    parser.add_argument(
        "--output_width",
        type=int,
        default=1048,
        help="Output video width.",
    )
    parser.add_argument(
        "--output_height",
        type=int,
        default=1048,
        help="Output video height.",
    )
    parser.add_argument(
        "--focal_length",
        type=float,
        default=300.0,
        help="Focal length for the linear undistorted camera model.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Optional cap on frames written, useful for quick tests.",
    )
    parser.add_argument(
        "--devignetting_mask_dir",
        type=Path,
        default=None,
        help="Optional folder containing old_isp/new_isp devignetting mask .bin files.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Process only the smallest discovered VRS file.",
    )
    return parser.parse_args()


def discover_vrs_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        return []
    return sorted(path for path in input_dir.rglob("*.vrs") if path.is_file())


def resolve_vrs_selection(input_dir: Path, vrs_file: str, debug: bool) -> List[Path]:
    discovered = discover_vrs_files(input_dir)
    if vrs_file:
        explicit = Path(vrs_file).expanduser()
        if explicit.exists():
            return [explicit.resolve()]
        matches = [path for path in discovered if path.name == vrs_file]
        if not matches:
            raise FileNotFoundError(
                f"Could not find '{vrs_file}' in {input_dir} or as an absolute path."
            )
        if len(matches) > 1:
            match_text = "\n".join(str(path) for path in matches)
            raise RuntimeError(
                f"Multiple files named '{vrs_file}' were found. Please use an absolute path.\n{match_text}"
            )
        return matches

    if debug and discovered:
        return [min(discovered, key=lambda path: path.stat().st_size)]
    return discovered


def ensure_bgr(frame):
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def format_output_dir(vrs_path: Path, input_dir: Path, output_root: Path, flags: str) -> Path:
    relative_parent = Path(".")
    try:
        relative_parent = vrs_path.parent.relative_to(input_dir)
    except ValueError:
        relative_parent = Path(vrs_path.parent.name)
    return output_root / relative_parent / flags / vrs_path.name


def write_metadata(
    metadata_path: Path,
    *,
    source_vrs: Path,
    output_video: Path,
    thumbnail: Path,
    fps: float,
    written_frames: int,
    source_resolution,
    output_resolution,
    undistort: bool,
    color_correct: bool,
    devignette: bool,
) -> None:
    payload = {
        "source_vrs": str(source_vrs),
        "output_video": str(output_video),
        "thumbnail": str(thumbnail),
        "fps": float(fps),
        "written_frames": int(written_frames),
        "source_resolution": [int(value) for value in source_resolution],
        "output_resolution": [int(value) for value in output_resolution],
        "undistort": undistort,
        "color_correct": color_correct,
        "devignette": devignette,
    }
    metadata_path.write_text(json.dumps(payload, indent=2))


def resolve_devignetting_mask_dir(mask_dir: Path | None) -> Path | None:
    candidates = []
    if mask_dir is not None:
        candidates.append(mask_dir.expanduser().resolve())
    if DEFAULT_MASK_DIR.exists():
        candidates.append(DEFAULT_MASK_DIR.resolve())

    required_files = [
        Path("old_isp/rgb_half_devignetting_mask.bin"),
        Path("old_isp/rgb_full_devignetting_mask.bin"),
        Path("old_isp/slam_devignetting_mask.bin"),
    ]
    for candidate in candidates:
        if all((candidate / relative).exists() for relative in required_files):
            return candidate
    return None


def process_vrs_file(vrs_path: Path, args: argparse.Namespace) -> Path:
    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    rgb_stream_id = provider.get_stream_id_from_label(RGB_LABEL)
    if rgb_stream_id is None:
        raise RuntimeError(f"{vrs_path} does not contain a '{RGB_LABEL}' stream.")

    if args.color_correct:
        provider.set_color_correction(True)
    devignette_applied = False
    if args.devignette:
        resolved_mask_dir = resolve_devignetting_mask_dir(args.devignetting_mask_dir)
        if resolved_mask_dir is None:
            print(
                "Warning: devignetting was requested, but compatible mask binaries were not found. "
                "Continuing without devignetting."
            )
        else:
            provider.set_devignetting_mask_folder_path(str(resolved_mask_dir))
            provider.set_devignetting(True)
            devignette_applied = True

    device_calib = provider.get_device_calibration()
    src_calib = device_calib.get_camera_calib(RGB_LABEL)
    src_width, src_height = src_calib.get_image_size()
    dst_calib = None
    if args.undistort:
        dst_calib = calibration.get_linear_camera_calibration(
            int(args.output_width),
            int(args.output_height),
            float(args.focal_length),
            "camera-rgb-linear",
            src_calib.get_transform_device_camera(),
        )

    image_config = provider.get_image_configuration(rgb_stream_id)
    fps = float(image_config.nominal_rate_hz or provider.get_nominal_rate_hz(rgb_stream_id) or 30.0)
    total_frames = provider.get_num_data(rgb_stream_id)
    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    preview_image_data, _ = provider.get_image_data_by_index(rgb_stream_id, 0)
    if not preview_image_data.is_valid():
        raise RuntimeError(f"Could not read the first RGB frame from {vrs_path}")

    preview_frame = preview_image_data.to_numpy_array()
    if args.undistort:
        preview_frame = calibration.distort_by_calibration(
            preview_frame, dst_calib, src_calib
        )
    elif (
        preview_frame.shape[1] != args.output_width
        or preview_frame.shape[0] != args.output_height
    ):
        preview_frame = cv2.resize(
            preview_frame,
            (args.output_width, args.output_height),
            interpolation=cv2.INTER_LINEAR,
        )
    preview_bgr = ensure_bgr(preview_frame)

    flags = (
        f"undistort_{args.undistort}_"
        f"colorcorrect_{args.color_correct}_"
        f"devignette_{devignette_applied}"
    )
    output_dir = format_output_dir(vrs_path, args.input_dir, args.output_root, flags)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / f"{vrs_path.name}.mp4"
    thumbnail_path = output_dir / f"{vrs_path.name}.png"
    metadata_path = output_dir / "metadata.json"

    writer = None
    first_frame_bgr = None
    written_frames = 0

    progress = tqdm(
        range(total_frames),
        desc=f"Extracting {vrs_path.name}",
        unit="frame",
    )
    try:
        for index in progress:
            image_data, _ = provider.get_image_data_by_index(rgb_stream_id, index)
            if not image_data.is_valid():
                continue

            frame = image_data.to_numpy_array()
            if args.undistort:
                frame = calibration.distort_by_calibration(frame, dst_calib, src_calib)
            elif frame.shape[1] != args.output_width or frame.shape[0] != args.output_height:
                frame = cv2.resize(
                    frame,
                    (args.output_width, args.output_height),
                    interpolation=cv2.INTER_LINEAR,
                )

            frame_bgr = ensure_bgr(frame)
            if first_frame_bgr is None:
                first_frame_bgr = frame_bgr

            if writer is None:
                height, width = frame_bgr.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer for {video_path}")

            writer.write(frame_bgr)
            written_frames += 1
    finally:
        if writer is not None:
            writer.release()

    if written_frames == 0 or first_frame_bgr is None:
        raise RuntimeError(f"No valid RGB frames were extracted from {vrs_path}")

    cv2.imwrite(str(thumbnail_path), first_frame_bgr)
    write_metadata(
        metadata_path,
        source_vrs=vrs_path,
        output_video=video_path,
        thumbnail=thumbnail_path,
        fps=fps,
        written_frames=written_frames,
        source_resolution=(src_width, src_height),
        output_resolution=(first_frame_bgr.shape[1], first_frame_bgr.shape[0]),
        undistort=args.undistort,
        color_correct=args.color_correct,
        devignette=devignette_applied,
    )
    return output_dir


def main() -> int:
    args = parse_args()
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()

    vrs_files = resolve_vrs_selection(args.input_dir, args.vrs_file, args.debug)
    if not vrs_files:
        print(
            f"No VRS files found. Put recordings under {args.input_dir} or pass --vrs_file explicitly."
        )
        return 1

    print("Processing the following VRS files:")
    for path in vrs_files:
        print(f"  - {path}")

    output_dirs: List[Path] = []
    for vrs_path in vrs_files:
        output_dir = process_vrs_file(vrs_path, args)
        output_dirs.append(output_dir)
        print(f"Finished {vrs_path.name} -> {output_dir}")

    print("\nAll done. Output folders:")
    for output_dir in output_dirs:
        print(f"  - {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
