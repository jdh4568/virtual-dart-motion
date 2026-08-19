import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def video_writer_fps(fps):
    if not fps or fps <= 0:
        return 30
    return max(1, int(round(fps)))


def read_markers(analysis_csv):
    df = pd.read_csv(analysis_csv)
    start_rows = df[df["is_start_frame"] == True]
    release_rows = df[df["is_release_candidate"] == True]

    if start_rows.empty:
        raise ValueError("No start frame found in analysis CSV.")
    if release_rows.empty:
        raise ValueError("No release candidate frame found in analysis CSV.")

    return df, start_rows.iloc[0], release_rows.iloc[0]


def load_trajectory_points(trajectory_csv, width, height, y_offset_px=0):
    traj_df = pd.read_csv(trajectory_csv)

    points = []

    for _, row in traj_df.iterrows():
        x = row["x"]
        y = row["y"]

        if pd.isna(x) or pd.isna(y):
            continue

        px = int(x * width)
        py = int(y * height) - y_offset_px

        points.append((px, py))

    return points


def draw_label(frame, text, origin, color):
    x, y = origin
    cv2.putText(
        frame,
        text,
        (x + 2, y + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def point_from_row(row, hand, point_kind):
    if point_kind == "start" and hasattr(row, "throw_start_x") and hasattr(row, "throw_start_y"):
        x = getattr(row, "throw_start_x")
        y = getattr(row, "throw_start_y")
        if not pd.isna(x) and not pd.isna(y):
            return x, y

    if point_kind == "release" and hasattr(row, "throw_release_x") and hasattr(row, "throw_release_y"):
        x = getattr(row, "throw_release_x")
        y = getattr(row, "throw_release_y")
        if not pd.isna(x) and not pd.isna(y):
            return x, y

    return getattr(row, f"{hand}_wrist_x"), getattr(row, f"{hand}_wrist_y")


def current_motion_point_from_row(row, hand):
    motion_point = getattr(row, "throw_motion_point", "wrist")
    x_name = f"{hand}_{motion_point}_x"
    y_name = f"{hand}_{motion_point}_y"

    if hasattr(row, x_name) and hasattr(row, y_name):
        x = getattr(row, x_name)
        y = getattr(row, y_name)
        if not pd.isna(x) and not pd.isna(y):
            return x, y

    return getattr(row, f"{hand}_wrist_x"), getattr(row, f"{hand}_wrist_y")


def draw_point_marker(frame, row, hand, point_kind, label, color):
    height, width = frame.shape[:2]
    x, y = point_from_row(row, hand, point_kind)

    if pd.isna(x) or pd.isna(y):
        return

    px = int(x * width)
    py = int(y * height)

    cv2.circle(frame, (px, py), 12, color, -1)
    cv2.circle(frame, (px, py), 16, (255, 255, 255), 2)
    draw_label(frame, label, (px + 18, py - 12), color)


def draw_animated_trajectory(frame, trajectory_points, release_frame, frame_index):
    if frame_index < release_frame:
        return

    if not trajectory_points:
        return

    frames_after_release = frame_index - release_frame + 1

    points_to_draw = min(
        frames_after_release,
        len(trajectory_points)
    )

    if points_to_draw <= 0:
        return

    for i in range(1, points_to_draw):
        cv2.line(
            frame,
            trajectory_points[i - 1],
            trajectory_points[i],
            (0, 255, 255),
            4,
            cv2.LINE_AA,
        )

    current_point = trajectory_points[points_to_draw - 1]
    cv2.circle(frame, current_point, 8, (255, 0, 0), -1)
    cv2.circle(frame, current_point, 11, (255, 255, 255), 2)

    if points_to_draw == len(trajectory_points):
        draw_label(frame, "ENDPOINT", (current_point[0] + 12, current_point[1] - 12), (255, 0, 0))


def draw_grid(frame, cell_size_px):
    height, width = frame.shape[:2]

    for x in range(0, width + 1, cell_size_px):
        color = (210, 210, 210) if x % (cell_size_px * 5) else (170, 170, 170)
        cv2.line(frame, (x, 0), (x, height), color, 1, cv2.LINE_AA)

    for y in range(0, height + 1, cell_size_px):
        color = (210, 210, 210) if y % (cell_size_px * 5) else (170, 170, 170)
        cv2.line(frame, (0, y), (width, y), color, 1, cv2.LINE_AA)


def draw_full_trajectory(frame, trajectory_points):
    if not trajectory_points:
        return

    for i in range(1, len(trajectory_points)):
        cv2.line(
            frame,
            trajectory_points[i - 1],
            trajectory_points[i],
            (0, 220, 255),
            4,
            cv2.LINE_AA,
        )

    release_point = trajectory_points[0]
    endpoint = trajectory_points[-1]
    cv2.circle(frame, release_point, 8, (0, 0, 255), -1)
    cv2.circle(frame, release_point, 11, (255, 255, 255), 2)
    draw_label(frame, "RELEASE", (release_point[0] + 12, release_point[1] - 12), (0, 0, 255))
    cv2.circle(frame, endpoint, 8, (255, 0, 0), -1)
    cv2.circle(frame, endpoint, 11, (255, 255, 255), 2)
    draw_label(frame, "ENDPOINT", (endpoint[0] + 12, endpoint[1] - 12), (255, 0, 0))


def render_grid_trajectory(
    trajectory_csv,
    output_image,
    width=1920,
    height=1080,
    grid_size_px=30,
    trajectory_y_offset_px=0,
):
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    draw_grid(frame, grid_size_px)

    trajectory_points = load_trajectory_points(
        trajectory_csv,
        width,
        height,
        y_offset_px=trajectory_y_offset_px,
    )
    draw_full_trajectory(frame, trajectory_points)

    output_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image), frame)

    print(f"Grid trajectory image: {output_image}")
    print(f"Canvas: {width}x{height}")
    print(f"Grid size: {grid_size_px}px")
    print(f"Trajectory points: {len(trajectory_points)}")


def render_preview(
    video_path,
    analysis_csv,
    trajectory_csv,
    output_video,
    hand,
    flip_horizontal=False,
    trajectory_y_offset_px=0,
):
    df, start_row, release_row = read_markers(analysis_csv)
    frame_lookup = {int(row.frame_index): row for row in df.itertuples(index=False)}

    start_frame = int(start_row["frame_index"])
    release_frame = int(release_row["frame_index"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    trajectory_points = load_trajectory_points(
        trajectory_csv,
        width,
        height,
        y_offset_px=trajectory_y_offset_px,
    )

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        video_writer_fps(fps),
        (width, height),
    )

    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if flip_horizontal:
            frame = cv2.flip(frame, 1)

        draw_label(frame, f"Frame: {frame_index}", (24, 36), (255, 255, 255))

        if frame_index in frame_lookup:
            row = frame_lookup[frame_index]
            point_x, point_y = current_motion_point_from_row(row, hand)

            if not pd.isna(point_x) and not pd.isna(point_y):
                px = int(point_x * width)
                py = int(point_y * height)
                cv2.circle(frame, (px, py), 7, (0, 255, 255), -1)

        if frame_index == start_frame:
            draw_point_marker(frame, start_row, hand, "start", "START", (128, 128, 128))
            draw_label(frame, "START FRAME", (24, 72), (200, 200, 200))

        if frame_index == release_frame:
            draw_point_marker(frame, release_row, hand, "release", "RELEASE", (0, 0, 255))
            draw_label(frame, "RELEASE CANDIDATE", (24, 72), (0, 0, 255))

        if start_frame <= frame_index <= release_frame:
            draw_label(frame, "THROW WINDOW", (24, height - 30), (0, 255, 255))

        draw_animated_trajectory(
            frame=frame,
            trajectory_points=trajectory_points,
            release_frame=release_frame,
            frame_index=frame_index,
        )

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    print(f"Video: {video_path}")
    print(f"Flip horizontal: {flip_horizontal}")
    print(f"Analysis CSV: {analysis_csv}")
    print(f"Trajectory CSV: {trajectory_csv}")
    print(f"Start frame: {start_frame}")
    print(f"Release candidate frame: {release_frame}")
    print(f"Trajectory points: {len(trajectory_points)}")
    print(f"Trajectory Y offset: {trajectory_y_offset_px}px")
    print(f"Preview saved: {output_video}")


def render_release_frame_image(
    video_path,
    analysis_csv,
    output_image,
    hand="right",
    flip_horizontal=False,
):
    _, start_row, release_row = read_markers(analysis_csv)
    release_frame = int(release_row["frame_index"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, release_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot read release frame: {release_frame}")

    if flip_horizontal:
        frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]
    start_x, start_y = point_from_row(start_row, hand, "start")
    release_x, release_y = point_from_row(release_row, hand, "release")

    if not any(pd.isna(value) for value in [start_x, start_y, release_x, release_y]):
        start_px = (int(start_x * width), int(start_y * height))
        release_px = (int(release_x * width), int(release_y * height))
        cv2.arrowedLine(
            frame,
            start_px,
            release_px,
            (0, 255, 255),
            5,
            cv2.LINE_AA,
            tipLength=0.2,
        )

    draw_point_marker(frame, start_row, hand, "start", "START", (0, 255, 0))
    draw_point_marker(frame, release_row, hand, "release", "RELEASE", (0, 0, 255))
    draw_label(
        frame,
        f"release frame: {release_frame}",
        (24, 42),
        (0, 255, 255),
    )

    output_path = Path(output_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)

    print(f"Release frame image: {output_path}")
    print(f"Release candidate frame: {release_frame}")


def main():
    parser = argparse.ArgumentParser(
        description="Render a video preview with start, release candidate, and animated trajectory."
    )
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("analysis_csv", type=Path, help="CSV created by analyze_throw.py")
    parser.add_argument("trajectory_csv", type=Path, help="CSV created by trajectory.py")
    parser.add_argument(
        "--hand",
        choices=["right", "left"],
        default="right",
        help="Throwing hand to visualize",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/analysis_preview.mp4"),
        help="Output annotated video path",
    )
    parser.add_argument(
        "--flip-horizontal",
        action="store_true",
        help="Flip mirrored/selfie videos before drawing markers",
    )
    parser.add_argument(
        "--trajectory-y-offset-px",
        type=int,
        default=0,
        help="Move the rendered trajectory upward by this many pixels.",
    )

    args = parser.parse_args()

    render_preview(
        video_path=args.video,
        analysis_csv=args.analysis_csv,
        trajectory_csv=args.trajectory_csv,
        output_video=args.out,
        hand=args.hand,
        flip_horizontal=args.flip_horizontal,
        trajectory_y_offset_px=args.trajectory_y_offset_px,
    )


if __name__ == "__main__":
    main()
