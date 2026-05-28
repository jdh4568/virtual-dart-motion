import argparse
import csv
from pathlib import Path

import cv2


def find_bright_object(frame, threshold, min_area, max_area):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        candidates.append((area, cx, cy))

    if not candidates:
        return None

    _, cx, cy = max(candidates, key=lambda item: item[0])
    return cx, cy


def track_bright_object(
    video_path,
    output_csv,
    threshold=210,
    min_area=8,
    max_area=3000,
    start_frame=0,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tracked_count = 0

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "frame_index",
                "time_sec",
                "object_detected",
                "x",
                "y",
                "pixel_x",
                "pixel_y",
            ],
        )
        writer.writeheader()

        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            point = None
            if frame_index >= start_frame:
                point = find_bright_object(frame, threshold, min_area, max_area)

            row = {
                "frame_index": frame_index,
                "time_sec": frame_index / fps,
                "object_detected": point is not None,
                "x": "",
                "y": "",
                "pixel_x": "",
                "pixel_y": "",
            }
            if point is not None:
                px, py = point
                row["x"] = px / width
                row["y"] = py / height
                row["pixel_x"] = px
                row["pixel_y"] = py
                tracked_count += 1

            writer.writerow(row)
            frame_index += 1

    cap.release()

    print(f"Video: {video_path}")
    print(f"Object CSV: {output_csv}")
    print(f"Threshold: {threshold}")
    print(f"Area range: {min_area}-{max_area}")
    print(f"Tracked frames: {tracked_count}/{frame_index}")


def main():
    parser = argparse.ArgumentParser(description="Track a bright thrown object in a side-view video.")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/object_trajectory.csv"),
        help="Output object trajectory CSV path",
    )
    parser.add_argument("--threshold", type=int, default=210, help="Brightness threshold")
    parser.add_argument("--min-area", type=float, default=8, help="Minimum blob area")
    parser.add_argument("--max-area", type=float, default=3000, help="Maximum blob area")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame to start tracking")
    args = parser.parse_args()

    track_bright_object(
        video_path=args.video,
        output_csv=args.out,
        threshold=args.threshold,
        min_area=args.min_area,
        max_area=args.max_area,
        start_frame=args.start_frame,
    )


if __name__ == "__main__":
    main()
