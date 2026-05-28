import argparse
import csv
from pathlib import Path

import cv2


def centroid_from_mask(mask, min_area, max_area):
    mask = cv2.medianBlur(mask, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

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


def find_bright_object(frame, threshold, min_area, max_area):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return centroid_from_mask(mask, min_area, max_area)


def find_hsv_object(frame, hsv_lower, hsv_upper, min_area, max_area):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    return centroid_from_mask(mask, min_area, max_area)


def find_object(frame, mode, threshold, hsv_lower, hsv_upper, min_area, max_area):
    if mode == "bright":
        return find_bright_object(frame, threshold, min_area, max_area)
    return find_hsv_object(frame, hsv_lower, hsv_upper, min_area, max_area)


def track_object(
    video_path,
    output_csv,
    mode="lime",
    threshold=210,
    hsv_lower=(35, 60, 60),
    hsv_upper=(90, 255, 255),
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
                point = find_object(
                    frame,
                    mode,
                    threshold,
                    hsv_lower,
                    hsv_upper,
                    min_area,
                    max_area,
                )

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
    print(f"Mode: {mode}")
    if mode == "bright":
        print(f"Threshold: {threshold}")
    else:
        print(f"HSV lower: {hsv_lower}")
        print(f"HSV upper: {hsv_upper}")
    print(f"Area range: {min_area}-{max_area}")
    print(f"Tracked frames: {tracked_count}/{frame_index}")


def track_bright_object(
    video_path,
    output_csv,
    threshold=210,
    min_area=8,
    max_area=3000,
    start_frame=0,
):
    track_object(
        video_path=video_path,
        output_csv=output_csv,
        mode="bright",
        threshold=threshold,
        min_area=min_area,
        max_area=max_area,
        start_frame=start_frame,
    )


def main():
    parser = argparse.ArgumentParser(description="Track a colored thrown object in a side-view video.")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/object_trajectory.csv"),
        help="Output object trajectory CSV path",
    )
    parser.add_argument(
        "--mode",
        choices=["lime", "bright"],
        default="lime",
        help="Object tracking mode. lime uses HSV color tracking; bright uses grayscale thresholding.",
    )
    parser.add_argument("--threshold", type=int, default=210, help="Brightness threshold for bright mode")
    parser.add_argument("--h-min", type=int, default=35, help="Minimum HSV hue for lime mode")
    parser.add_argument("--h-max", type=int, default=90, help="Maximum HSV hue for lime mode")
    parser.add_argument("--s-min", type=int, default=60, help="Minimum HSV saturation for lime mode")
    parser.add_argument("--s-max", type=int, default=255, help="Maximum HSV saturation for lime mode")
    parser.add_argument("--v-min", type=int, default=60, help="Minimum HSV value for lime mode")
    parser.add_argument("--v-max", type=int, default=255, help="Maximum HSV value for lime mode")
    parser.add_argument("--min-area", type=float, default=8, help="Minimum blob area")
    parser.add_argument("--max-area", type=float, default=3000, help="Maximum blob area")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame to start tracking")
    args = parser.parse_args()

    track_object(
        video_path=args.video,
        output_csv=args.out,
        mode=args.mode,
        threshold=args.threshold,
        hsv_lower=(args.h_min, args.s_min, args.v_min),
        hsv_upper=(args.h_max, args.s_max, args.v_max),
        min_area=args.min_area,
        max_area=args.max_area,
        start_frame=args.start_frame,
    )


if __name__ == "__main__":
    main()
