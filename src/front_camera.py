import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyze_throw import MOTION_POINTS, clamp, point_prefix, resolve_motion_point


def _nearest_frame_row(df, frame_index):
    if df.empty:
        return None

    distances = (df["frame_index"] - frame_index).abs()
    return df.loc[distances.idxmin()]


def _front_window(df, release_frame, window):
    start_frame = max(0, release_frame - window)
    recent = df[(df["frame_index"] >= start_frame) & (df["frame_index"] <= release_frame)]
    recent = recent[recent["front_tracking_ok"] == True]
    return recent


def analyze_front_direction(
    front_landmarks_csv,
    release_frame,
    hand,
    motion_point="auto",
    direction_window=20,
    horizontal_gain=1.0,
):
    front_df = pd.read_csv(front_landmarks_csv)
    hand = hand.lower()

    if motion_point == "auto":
        motion_point = resolve_motion_point(front_df, hand, "auto")
    if motion_point not in MOTION_POINTS:
        raise ValueError(f"--motion-point must be one of: auto, {sorted(MOTION_POINTS)}")

    point = point_prefix(hand, motion_point)
    required = ["frame_index", f"{point}_x", f"{point}_y"]
    missing = [column for column in required if column not in front_df.columns]
    if missing:
        raise ValueError(f"Missing required front camera columns: {missing}")

    for column in required:
        front_df[column] = pd.to_numeric(front_df[column], errors="coerce")

    front_df["front_tracking_ok"] = (
        front_df[f"{point}_x"].notna() & front_df[f"{point}_y"].notna()
    )
    recent = _front_window(front_df, release_frame, direction_window)
    if len(recent) < 2:
        raise ValueError("Not enough front camera tracking points near release frame.")

    start_row = recent.iloc[0]
    front_release_row = _nearest_frame_row(recent, release_frame)
    if front_release_row is None:
        raise ValueError("Cannot find matching front camera release frame.")

    dx = front_release_row[f"{point}_x"] - start_row[f"{point}_x"]
    dy = front_release_row[f"{point}_y"] - start_row[f"{point}_y"]
    distance = (dx**2 + dy**2) ** 0.5
    if pd.isna(distance) or distance == 0:
        front_direction_x = 0.0
    else:
        front_direction_x = clamp((dx / distance) * horizontal_gain, -1.0, 1.0)

    return {
        "release_frame": release_frame,
        "front_release_frame": int(front_release_row["frame_index"]),
        "front_start_frame": int(start_row["frame_index"]),
        "front_motion_point": motion_point,
        "front_direction_x": front_direction_x,
        "front_raw_dx": dx,
        "front_raw_dy": dy,
        "front_start_x": start_row[f"{point}_x"],
        "front_start_y": start_row[f"{point}_y"],
        "front_release_x": front_release_row[f"{point}_x"],
        "front_release_y": front_release_row[f"{point}_y"],
    }


def render_front_direction_image(
    front_video,
    result,
    output_image,
    flip_horizontal=False,
):
    cap = cv2.VideoCapture(str(front_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open front video: {front_video}")

    release_frame = int(result["front_release_frame"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, release_frame)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError(f"Cannot read front release frame: {release_frame}")
    cap.release()

    if flip_horizontal:
        frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]
    start = (
        int(float(result["front_start_x"]) * width),
        int(float(result["front_start_y"]) * height),
    )
    release = (
        int(float(result["front_release_x"]) * width),
        int(float(result["front_release_y"]) * height),
    )

    cv2.arrowedLine(frame, start, release, (0, 255, 255), 6, cv2.LINE_AA, tipLength=0.25)
    cv2.circle(frame, start, 12, (0, 128, 255), -1)
    cv2.circle(frame, release, 12, (0, 0, 255), -1)
    cv2.putText(
        frame,
        "front start",
        (max(0, start[0] - 80), max(25, start[1] - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 128, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"front direction x: {result['front_direction_x']:.3f}",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    output_path = Path(output_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    return output_path


def apply_front_camera_direction(
    side_analysis_csv,
    front_landmarks_csv,
    hand,
    motion_point,
    output_csv=None,
    direction_window=20,
    horizontal_gain=1.0,
    frame_offset=5,
    front_video=None,
    front_direction_image=None,
    front_flip_horizontal=False,
):
    side_df = pd.read_csv(side_analysis_csv)
    hand = hand.lower()

    release_rows = side_df[side_df["is_release_candidate"] == True]
    if release_rows.empty:
        raise ValueError("No release candidate frame found in side analysis CSV.")

    release_row = release_rows.iloc[0]
    side_release_frame = int(release_row["frame_index"])
    front_release_frame = max(0, side_release_frame + int(frame_offset))

    if motion_point == "auto":
        motion_point = release_row.get("throw_motion_point", "auto")
    result = analyze_front_direction(
        front_landmarks_csv=front_landmarks_csv,
        release_frame=front_release_frame,
        hand=hand,
        motion_point=motion_point,
        direction_window=direction_window,
        horizontal_gain=horizontal_gain,
    )

    result["side_release_frame"] = side_release_frame
    result["front_frame_offset"] = int(frame_offset)
    side_df["throw_side_direction_x"] = side_df["throw_direction_x"]
    side_df["throw_direction_x"] = result["front_direction_x"]
    side_df["front_camera_enabled"] = True
    side_df["front_side_release_frame"] = side_release_frame
    side_df["front_frame_offset"] = int(frame_offset)
    side_df["front_motion_point"] = result["front_motion_point"]
    side_df["front_release_frame"] = result["front_release_frame"]
    side_df["front_start_frame"] = result["front_start_frame"]
    side_df["front_direction_x"] = result["front_direction_x"]
    side_df["front_raw_dx"] = result["front_raw_dx"]
    side_df["front_raw_dy"] = result["front_raw_dy"]
    side_df["front_direction_window"] = direction_window
    side_df["front_horizontal_gain"] = horizontal_gain

    output_path = Path(output_csv) if output_csv else Path(side_analysis_csv)
    side_df.to_csv(output_path, index=False)

    print("Front camera direction correction")
    print(f"Front landmarks CSV: {front_landmarks_csv}")
    print(f"Side release frame: {side_release_frame}")
    print(f"Front frame offset: {frame_offset}")
    print(f"Front frame used: {result['front_release_frame']}")
    print(f"Front motion point: {result['front_motion_point']}")
    print(f"Front raw movement: dx={result['front_raw_dx']:.4f}, dy={result['front_raw_dy']:.4f}")
    print(f"Front direction x: {result['front_direction_x']:.4f}")
    print(f"Updated analysis CSV: {output_path}")

    if front_video and front_direction_image:
        image_path = render_front_direction_image(
            front_video=front_video,
            result=result,
            output_image=front_direction_image,
            flip_horizontal=front_flip_horizontal,
        )
        print(f"Front direction image: {image_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Test front camera left-right direction from landmarks.")
    parser.add_argument("front_landmarks_csv", type=Path, help="CSV created by extract_landmarks.py")
    parser.add_argument(
        "--release-frame",
        type=int,
        required=True,
        help="Frame number selected by visual inspection as the release moment.",
    )
    parser.add_argument("--hand", choices=["right", "left"], default="right")
    parser.add_argument(
        "--motion-point",
        choices=["auto", "wrist", "thumb_tip", "index_tip", "middle_tip"],
        default="auto",
    )
    parser.add_argument("--direction-window", type=int, default=20)
    parser.add_argument("--horizontal-gain", type=float, default=1.0)
    args = parser.parse_args()

    result = analyze_front_direction(
        front_landmarks_csv=args.front_landmarks_csv,
        release_frame=args.release_frame,
        hand=args.hand,
        motion_point=args.motion_point,
        direction_window=args.direction_window,
        horizontal_gain=args.horizontal_gain,
    )

    print("Front camera standalone test")
    print(f"Front landmarks CSV: {args.front_landmarks_csv}")
    print(f"Release frame input: {args.release_frame}")
    print(f"Front frame used: {result['front_release_frame']}")
    print(f"Front start frame: {result['front_start_frame']}")
    print(f"Front motion point: {result['front_motion_point']}")
    print(f"Front raw movement: dx={result['front_raw_dx']:.4f}, dy={result['front_raw_dy']:.4f}")
    print(f"Front direction x: {result['front_direction_x']:.4f}")


if __name__ == "__main__":
    main()
