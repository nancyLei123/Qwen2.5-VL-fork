#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay semicolon-separated ELAN-style annotations onto a video."
    )
    parser.add_argument("--video_path", type=Path, required=True, help="Input MP4 file.")
    parser.add_argument("--csv_path", type=Path, required=True, help="Annotation CSV file.")
    parser.add_argument(
        "--output_path", type=Path, required=True, help="Output MP4 file with labels."
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="all_tiers",
        help="CSV column containing the label text.",
    )
    parser.add_argument(
        "--font_scale", type=float, default=1.0, help="OpenCV font scale for labels."
    )
    parser.add_argument(
        "--thickness", type=int, default=2, help="OpenCV text thickness."
    )
    return parser.parse_args()


def load_annotations(csv_path: Path, label_column: str) -> list[dict[str, object]]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        rows = []
        for row in reader:
            if not row:
                continue
            rows.append(
                {
                    "start": float(row["#starttime"]),
                    "end": float(row["#endtime"]),
                    "label": row[label_column].strip(),
                }
            )
    rows.sort(key=lambda row: row["start"])
    return rows


def draw_label(frame, text: str, font_scale: float, thickness: int):
    if not text:
        return frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = 24
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    top_left = (margin, margin)
    bottom_right = (margin + text_width + 20, margin + text_height + baseline + 20)
    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        text,
        (margin + 10, margin + text_height + 10),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return frame


def main() -> int:
    args = parse_args()
    annotations = load_annotations(args.csv_path, args.label_column)
    if not annotations:
        raise RuntimeError(f"No annotations found in {args.csv_path}")

    capture = cv2.VideoCapture(str(args.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {args.output_path}")

    annotation_index = 0
    progress = tqdm(total=frame_count, desc="Overlaying labels", unit="frame")
    try:
        frame_idx = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            current_time = frame_idx / fps
            while (
                annotation_index < len(annotations)
                and annotations[annotation_index]["end"] <= current_time
            ):
                annotation_index += 1

            active_label = ""
            if annotation_index < len(annotations):
                current = annotations[annotation_index]
                if current["start"] <= current_time < current["end"]:
                    active_label = str(current["label"])

            draw_label(frame, active_label, args.font_scale, args.thickness)
            writer.write(frame)
            frame_idx += 1
            progress.update(1)
    finally:
        progress.close()
        capture.release()
        writer.release()

    print(f"Labeled video written to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
