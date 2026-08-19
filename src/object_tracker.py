import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


COLOR_RANGES = {
    "green": [((35, 60, 60), (90, 255, 255))],
    "blue": [((90, 60, 60), (130, 255, 255))],
    "red": [((0, 70, 60), (10, 255, 255)), ((170, 70, 60), (180, 255, 255))],
    "yellow": [((20, 70, 70), (35, 255, 255))],
}


def read_release_frame(analysis_csv):
    df = pd.read_csv(analysis_csv)
    release_rows = df[df["is_release_candidate"] == True]
    if release_rows.empty:
        raise ValueError("No release candidate frame found in analysis CSV.")
    return int(release_rows.iloc[0]["frame_index"])


def read_release_direction(analysis_csv):
    df = pd.read_csv(analysis_csv)
    release_rows = df[df["is_release_candidate"] == True]
    if release_rows.empty:
        return 1.0, 0.0

    row = release_rows.iloc[0]
    dx = row.get("throw_direction_x", 1.0)
    dy = row.get("throw_direction_y", 0.0)
    if pd.isna(dx) or pd.isna(dy):
        return 1.0, 0.0

    length = math.hypot(dx, dy)
    if length == 0:
        return 1.0, 0.0
    return dx / length, dy / length


def read_release_info(analysis_csv):
    df = pd.read_csv(analysis_csv)
    release_rows = df[df["is_release_candidate"] == True]
    if release_rows.empty:
        raise ValueError("No release candidate frame found in analysis CSV.")

    row = release_rows.iloc[0]
    release_frame = int(row["frame_index"])

    x = row.get("throw_release_x", row.get("right_wrist_x", 0.5))
    y = row.get("throw_release_y", row.get("right_wrist_y", 0.5))
    if pd.isna(x) or pd.isna(y):
        x, y = 0.5, 0.5

    dx = row.get("throw_direction_x", 1.0)
    dy = row.get("throw_direction_y", 0.0)
    if pd.isna(dx) or pd.isna(dy):
        dx, dy = 1.0, 0.0

    length = math.hypot(dx, dy)
    if length == 0:
        dx, dy = 1.0, 0.0
    else:
        dx, dy = dx / length, dy / length

    return release_frame, (float(x), float(y)), (dx, dy)


def build_mask(hsv, color):
    ranges = COLOR_RANGES[color]
    mask = None
    for lower, upper in ranges:
        partial = cv2.inRange(hsv, lower, upper)
        mask = partial if mask is None else cv2.bitwise_or(mask, partial)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def largest_contour_center(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < min_area:
        return None

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    return cx, cy, area


def track_object_color(
    video_path,
    output_csv,
    release_frame,
    color="green",
    min_area=20,
    max_frames_after_release=40,
    flip_horizontal=False,
):
    if color not in COLOR_RANGES:
        raise ValueError(f"--object-color must be one of: {sorted(COLOR_RANGES)}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if flip_horizontal:
            frame = cv2.flip(frame, 1)

        if release_frame <= frame_index <= release_frame + max_frames_after_release:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = build_mask(hsv, color)
            center = largest_contour_center(mask, min_area)

            if center:
                cx, cy, area = center
                rows.append(
                    {
                        "frame_index": frame_index,
                        "time_sec": frame_index / fps,
                        "x": cx / width,
                        "y": cy / height,
                        "area": area,
                        "object_detected": True,
                    }
                )

        frame_index += 1

    cap.release()

    track_df = pd.DataFrame(
        rows,
        columns=["frame_index", "time_sec", "x", "y", "area", "object_detected"],
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    track_df.to_csv(output_csv, index=False)

    return output_csv


def create_forward_roi_mask(width, height, release_point, direction):
    release_x = int(release_point[0] * width)
    release_y = int(release_point[1] * height)

    dir_x, dir_y = direction
    length = math.hypot(dir_x, dir_y)
    if length == 0:
        dir_x, dir_y = 1.0, 0.0
    else:
        dir_x, dir_y = dir_x / length, dir_y / length

    perp_x, perp_y = -dir_y, dir_x
    far = max(width, height) * 1.2
    near_width = max(24, int(min(width, height) * 0.04))
    far_width = max(120, int(min(width, height) * 0.28))

    origin = np.array([release_x, release_y], dtype=np.float32)
    direction_vec = np.array([dir_x, dir_y], dtype=np.float32)
    perp_vec = np.array([perp_x, perp_y], dtype=np.float32)

    p1 = origin + perp_vec * near_width
    p2 = origin - perp_vec * near_width
    p3 = origin + direction_vec * far - perp_vec * far_width
    p4 = origin + direction_vec * far + perp_vec * far_width

    polygon = np.array([p1, p2, p3, p4], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    return mask


def pick_flow_point(prev_points, next_points, status, direction, min_motion_px, release_point_px):
    best = None
    dir_x, dir_y = direction
    release_x, release_y = release_point_px

    for prev, nxt, ok in zip(prev_points.reshape(-1, 2), next_points.reshape(-1, 2), status.reshape(-1)):
        if not ok:
            continue

        ahead_x = nxt[0] - release_x
        ahead_y = nxt[1] - release_y
        if ahead_x * dir_x + ahead_y * dir_y < 0:
            continue

        move_x = nxt[0] - prev[0]
        move_y = nxt[1] - prev[1]
        distance = math.hypot(move_x, move_y)
        if distance < min_motion_px:
            continue

        motion_alignment = (move_x * dir_x + move_y * dir_y) / distance
        if motion_alignment < 0.15:
            continue

        score = distance * (0.5 + motion_alignment)
        if best is None or score > best[0]:
            best = (score, nxt[0], nxt[1], distance)

    return best


def track_object_flow(
    video_path,
    output_csv,
    release_frame,
    release_point,
    direction,
    max_frames_after_release=40,
    flip_horizontal=False,
    min_motion_px=4.0,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = {}
    frame_index = 0
    wanted_start = release_frame
    wanted_end = release_frame + max_frames_after_release

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if flip_horizontal:
            frame = cv2.flip(frame, 1)
        if wanted_start <= frame_index <= wanted_end:
            frames[frame_index] = frame
        if frame_index > wanted_end:
            break
        frame_index += 1

    cap.release()

    rows = []
    if wanted_start not in frames:
        return save_track_rows(rows, output_csv)

    prev_gray = cv2.cvtColor(frames[wanted_start], cv2.COLOR_BGR2GRAY)

    mask = create_forward_roi_mask(width, height, release_point, direction)
    prev_points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=250,
        qualityLevel=0.01,
        minDistance=6,
        blockSize=5,
        mask=mask,
    )
    if prev_points is None:
        return save_track_rows(rows, output_csv)

    lk_params = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    }

    release_point_px = (release_point[0] * width, release_point[1] * height)
    locked_point = None

    for idx in range(release_frame + 1, wanted_end + 1):
        if idx not in frames:
            continue

        gray = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)

        if next_points is None or status is None:
            break

        best = pick_flow_point(
            prev_points,
            next_points,
            status,
            direction,
            min_motion_px,
            release_point_px,
        )
        if best:
            _, cx, cy, distance = best
            locked_point = (cx, cy)
            rows.append(
                {
                    "frame_index": idx,
                    "time_sec": idx / fps,
                    "x": cx / width,
                    "y": cy / height,
                    "area": distance,
                    "object_detected": True,
                }
            )

        if locked_point is not None:
            prev_points = np.array([[locked_point]], dtype=np.float32)
        else:
            good_next = next_points[status.reshape(-1) == 1]
            if len(good_next) < 5:
                break
            prev_points = good_next.reshape(-1, 1, 2)

        if len(prev_points) < 1:
            break

        prev_gray = gray

    return save_track_rows(rows, output_csv)


def save_track_rows(rows, output_csv):
    track_df = pd.DataFrame(
        rows,
        columns=["frame_index", "time_sec", "x", "y", "area", "object_detected"],
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    track_df.to_csv(output_csv, index=False)
    return track_df


def correct_release_from_object_track(analysis_csv, object_track_csv, lead_frames=3):
    track_df = pd.read_csv(object_track_csv)
    if track_df.empty:
        return None

    analysis_df = pd.read_csv(analysis_csv)
    detected = track_df[track_df["object_detected"] == True]
    if detected.empty:
        return None

    first_object_frame = int(detected.iloc[0]["frame_index"])
    target_frame = max(0, first_object_frame - lead_frames)
    candidates = analysis_df[analysis_df["frame_index"] <= target_frame]
    candidates = candidates[candidates["tracking_ok"] == True]
    if candidates.empty:
        candidates = analysis_df[analysis_df["frame_index"] <= target_frame]
    if candidates.empty:
        return None

    release_idx = candidates.index[-1]
    release_row = analysis_df.loc[release_idx]
    motion_point = release_row.get("throw_motion_point", "wrist")
    point_prefix = f"right_{motion_point}"
    x_column = f"{point_prefix}_x"
    y_column = f"{point_prefix}_y"
    if x_column not in analysis_df.columns or y_column not in analysis_df.columns:
        x_column = "right_wrist_x"
        y_column = "right_wrist_y"

    analysis_df["is_release_candidate"] = False
    analysis_df.loc[release_idx, "is_release_candidate"] = True
    analysis_df["throw_release_x"] = release_row[x_column]
    analysis_df["throw_release_y"] = release_row[y_column]
    if "filtered_speed" in analysis_df.columns:
        analysis_df["throw_release_speed"] = release_row["filtered_speed"]
    analysis_df.to_csv(analysis_csv, index=False)

    return {
        "first_object_frame": first_object_frame,
        "corrected_release_frame": int(release_row["frame_index"]),
    }


def track_object(
    video_path,
    analysis_csv,
    output_csv,
    method="flow",
    color="green",
    min_area=20,
    max_frames_after_release=40,
    flip_horizontal=False,
    min_motion_px=4.0,
):
    release_frame, release_point, direction = read_release_info(analysis_csv)

    if method == "color":
        output_path = track_object_color(
            video_path=video_path,
            output_csv=output_csv,
            release_frame=release_frame,
            color=color,
            min_area=min_area,
            max_frames_after_release=max_frames_after_release,
            flip_horizontal=flip_horizontal,
        )
        track_df = pd.read_csv(output_path)
    elif method == "flow":
        track_df = track_object_flow(
            video_path=video_path,
            output_csv=output_csv,
            release_frame=release_frame,
            release_point=release_point,
            direction=direction,
            max_frames_after_release=max_frames_after_release,
            flip_horizontal=flip_horizontal,
            min_motion_px=min_motion_px,
        )
    else:
        raise ValueError("--object-method must be either color or flow")

    print("Object tracking result")
    print(f"Video: {video_path}")
    print(f"Analysis CSV: {analysis_csv}")
    print(f"Release frame: {release_frame}")
    print(f"Object method: {method}")
    if method == "color":
        print(f"Object color: {color}")
    print(f"Detected object frames: {len(track_df)}")
    print(f"Object track CSV saved: {output_csv}")
    return output_csv


def main():
    parser = argparse.ArgumentParser(description="Track a projectile after release.")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("analysis_csv", type=Path, help="CSV created by analyze_throw.py")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/object_track.csv"),
        help="Output object track CSV path",
    )
    parser.add_argument(
        "--object-method",
        choices=["color", "flow"],
        default="flow",
        help="Projectile tracking method",
    )
    parser.add_argument(
        "--object-color",
        choices=sorted(COLOR_RANGES),
        default="green",
        help="Projectile color to track",
    )
    parser.add_argument("--min-area", type=float, default=20, help="Minimum contour area")
    parser.add_argument(
        "--max-frames-after-release",
        type=int,
        default=40,
        help="Number of frames to scan after release",
    )
    parser.add_argument(
        "--flip-horizontal",
        action="store_true",
        help="Flip mirrored/selfie videos before tracking",
    )
    parser.add_argument(
        "--min-motion-px",
        type=float,
        default=4.0,
        help="Minimum optical-flow motion in pixels",
    )
    args = parser.parse_args()

    track_object(
        video_path=args.video,
        analysis_csv=args.analysis_csv,
        output_csv=args.out,
        method=args.object_method,
        color=args.object_color,
        min_area=args.min_area,
        max_frames_after_release=args.max_frames_after_release,
        flip_horizontal=args.flip_horizontal,
        min_motion_px=args.min_motion_px,
    )


if __name__ == "__main__":
    main()
