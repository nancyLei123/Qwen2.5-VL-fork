#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rotate videos and images under a directory by 90, 180, or 270 degrees."
    )
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "extracted_videos",
        help="Directory to scan recursively.",
    )
    parser.add_argument(
        "--degrees",
        type=int,
        default=90,
        choices=[90, 180, 270],
        help="Rotation amount clockwise.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the input file instead of creating a suffixed copy.",
    )
    return parser.parse_args()


def rotation_code(degrees: int) -> int:
    if degrees == 90:
        return cv2.ROTATE_90_CLOCKWISE
    if degrees == 180:
        return cv2.ROTATE_180
    return cv2.ROTATE_90_COUNTERCLOCKWISE


def output_path_for(path: Path, degrees: int, overwrite: bool) -> Path:
    if overwrite:
        return path
    return path.with_name(f"{path.stem}_rot{degrees}{path.suffix}")


def rotate_image(path: Path, degrees: int, overwrite: bool) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    rotated = cv2.rotate(image, rotation_code(degrees))
    out_path = output_path_for(path, degrees, overwrite)
    cv2.imwrite(str(out_path), rotated)


def rotate_video(path: Path, degrees: int, overwrite: bool) -> None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_size = (height, width) if degrees in {90, 270} else (width, height)
    out_path = output_path_for(path, degrees, overwrite)

    temp_path = out_path if not overwrite else path.with_name(f"{path.stem}.tmp{path.suffix}")
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        out_size,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video writer: {temp_path}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rotated = cv2.rotate(frame, rotation_code(degrees))
            writer.write(rotated)
    finally:
        capture.release()
        writer.release()

    if overwrite:
        temp_path.replace(path)


def main() -> int:
    args = parse_args()
    root_dir = args.root_dir.expanduser().resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    files = sorted(path for path in root_dir.rglob("*") if path.is_file())
    targets = [
        path
        for path in files
        if path.suffix.lower() in VIDEO_EXTENSIONS.union(IMAGE_EXTENSIONS)
        and f"_rot{args.degrees}" not in path.stem
    ]
    if not targets:
        print(f"No media files found under {root_dir}")
        return 0

    for path in tqdm(targets, desc="Rotating media", unit="file"):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            rotate_image(path, args.degrees, args.overwrite)
        else:
            rotate_video(path, args.degrees, args.overwrite)

    print(f"Finished rotating {len(targets)} files under {root_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
