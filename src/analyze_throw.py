import argparse
import math
from pathlib import Path

import pandas as pd


MOTION_POINTS = {"wrist", "thumb_tip", "index_tip", "middle_tip"}


def moving_average(values, window):
    return values.rolling(window=window, center=True, min_periods=1).mean()


def point_prefix(hand, motion_point):
    return f"{hand}_{motion_point}"


def resolve_motion_point(df, hand, motion_point):
    if motion_point != "auto":
        return motion_point

    for candidate in ["index_tip", "middle_tip", "thumb_tip"]:
        x_column = f"{hand}_{candidate}_x"
        y_column = f"{hand}_{candidate}_y"
        if x_column in df.columns and y_column in df.columns:
            detected_ratio = df[[x_column, y_column]].notna().all(axis=1).mean()
            if detected_ratio >= 0.2:
                return candidate

    return "wrist"


def calculate_speed(df, hand, motion_point):
    point = point_prefix(hand, motion_point)

    df["raw_dx"] = df[f"{point}_x"].diff()
    df["raw_dy"] = df[f"{point}_y"].diff()
    df["dx"] = df[f"{point}_x"].diff()
    df["dy"] = df[f"{point}_y"].diff()
    df["dt"] = df["time_sec"].diff()
    df["speed"] = ((df["dx"] ** 2 + df["dy"] ** 2) ** 0.5) / df["dt"]
    return df


def calculate_elbow_angle(row, hand):
    shoulder = (row[f"{hand}_shoulder_x"], row[f"{hand}_shoulder_y"])
    elbow = (row[f"{hand}_elbow_x"], row[f"{hand}_elbow_y"])
    wrist = (row[f"{hand}_wrist_x"], row[f"{hand}_wrist_y"])

    if any(pd.isna(v) for point in [shoulder, elbow, wrist] for v in point):
        return math.nan

    upper = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
    lower = (wrist[0] - elbow[0], wrist[1] - elbow[1])

    upper_len = math.hypot(*upper)
    lower_len = math.hypot(*lower)
    if upper_len == 0 or lower_len == 0:
        return math.nan

    dot = upper[0] * lower[0] + upper[1] * lower[1]
    cos_theta = max(-1.0, min(1.0, dot / (upper_len * lower_len)))
    return math.degrees(math.acos(cos_theta))


def filter_unreliable_motion(df, hand, motion_point, min_visibility):
    visibility_columns = [
        f"{hand}_shoulder_visibility",
        f"{hand}_elbow_visibility",
        f"{hand}_wrist_visibility",
    ]
    point_visibility = f"{point_prefix(hand, motion_point)}_visibility"
    if point_visibility not in visibility_columns:
        visibility_columns.append(point_visibility)
    available_visibility = [column for column in visibility_columns if column in df.columns]

    if available_visibility:
        df["tracking_ok"] = df["pose_detected"] == True
        for column in available_visibility:
            df["tracking_ok"] = df["tracking_ok"] & (df[column] >= min_visibility)
    else:
        df["tracking_ok"] = df["pose_detected"] == True

    point = point_prefix(hand, motion_point)
    df["tracking_ok"] = (
        df["tracking_ok"]
        & df[f"{point}_x"].notna()
        & df[f"{point}_y"].notna()
    )

    speed_cap = df.loc[df["tracking_ok"], "speed"].quantile(0.98)
    if pd.isna(speed_cap) or speed_cap <= 0:
        speed_cap = df["speed"].max()

    # Very large single-frame jumps usually mean landmark tracking jitter.
    df["filtered_speed"] = df["speed"].where(df["tracking_ok"], 0).clip(upper=speed_cap)
    df["smooth_speed"] = moving_average(df["filtered_speed"].fillna(0), 5)
    return df


def find_release_candidate(df, speed_threshold_ratio=0.45):
    max_speed = df["smooth_speed"].max()
    if pd.isna(max_speed) or max_speed <= 0:
        raise ValueError("Cannot find release candidate because speed values are empty.")

    threshold = max_speed * speed_threshold_ratio
    candidates = df[df["smooth_speed"] >= threshold].copy()
    candidates = candidates[candidates["tracking_ok"] == True]

    if candidates.empty:
        release_idx = df["smooth_speed"].idxmax()
        return release_idx, threshold

    # Pick the strongest actual wrist-speed frame inside the high-motion region.
    # This avoids selecting an earlier frame only because of centered smoothing.
    release_idx = candidates["filtered_speed"].idxmax()
    return release_idx, threshold


def direction_from_recent_motion(df, release_idx, hand, motion_point, direction_window):
    release_pos = df.index.get_loc(release_idx)
    start_pos = max(0, release_pos - direction_window)
    recent = df.iloc[start_pos : release_pos + 1]
    recent = recent[recent["tracking_ok"] == True]

    if len(recent) < 2:
        return None

    start_row = recent.iloc[0]
    release_row = recent.iloc[-1]
    point = point_prefix(hand, motion_point)
    dx = release_row[f"{point}_x"] - start_row[f"{point}_x"]
    dy = release_row[f"{point}_y"] - start_row[f"{point}_y"]
    distance = math.hypot(dx, dy)

    if distance == 0 or pd.isna(distance):
        return None

    return dx / distance, dy / distance


def clamp(value, low, high):
    return max(low, min(high, value))


def calculate_hit_position(direction_x, direction_y, speed, board_w, board_h, sensitivity):
    center_x = (board_w - 1) / 2
    center_y = (board_h - 1) / 2

    if any(pd.isna(value) for value in [direction_x, direction_y, speed]):
        direction_x, direction_y, speed = 0.0, 0.0, 0.0

    hit_x = center_x + direction_x * speed * sensitivity
    hit_y = center_y + direction_y * speed * sensitivity

    return (
        int(round(clamp(hit_x, 0, board_w - 1))),
        int(round(clamp(hit_y, 0, board_h - 1))),
    )


def analyze(
    csv_path,
    hand,
    motion_point,
    output_csv,
    window,
    lookback,
    direction_window,
    min_visibility,
    board_w,
    board_h,
    sensitivity,
):
    df = pd.read_csv(csv_path)
    hand = hand.lower()
    if hand not in {"right", "left"}:
        raise ValueError("--hand must be either right or left")
    if motion_point not in MOTION_POINTS | {"auto"}:
        raise ValueError(f"--motion-point must be one of: auto, {sorted(MOTION_POINTS)}")

    required = [
        "frame_index",
        "time_sec",
        "pose_detected",
        f"{hand}_shoulder_x",
        f"{hand}_shoulder_y",
        f"{hand}_elbow_x",
        f"{hand}_elbow_y",
        f"{hand}_wrist_x",
        f"{hand}_wrist_y",
    ]
    motion_point = resolve_motion_point(df, hand, motion_point)
    point = point_prefix(hand, motion_point)
    required.extend([f"{point}_x", f"{point}_y"])

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_columns = [column for column in required if column not in {"pose_detected"}]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = calculate_speed(df, hand, motion_point)
    df = filter_unreliable_motion(df, hand, motion_point, min_visibility)
    df["smooth_speed"] = moving_average(df["filtered_speed"].fillna(0), window)
    df["elbow_angle"] = df.apply(lambda row: calculate_elbow_angle(row, hand), axis=1)

    release_idx, threshold = find_release_candidate(df)
    release_row = df.loc[release_idx]

    release_pos = df.index.get_loc(release_idx)
    start_pos = max(0, release_pos - lookback)
    start_candidates = df.iloc[start_pos : release_pos + 1]
    start_candidates = start_candidates[start_candidates["tracking_ok"] == True]
    if len(start_candidates) >= 2:
        start_idx = start_candidates.index[0]
    else:
        start_idx = df.index[start_pos]
    start_row = df.loc[start_idx]

    dx = release_row[f"{point}_x"] - start_row[f"{point}_x"]
    dy = release_row[f"{point}_y"] - start_row[f"{point}_y"]
    dt = release_row["time_sec"] - start_row["time_sec"]

    if dt <= 0 or pd.isna(dt) or any(pd.isna(value) for value in [dx, dy]):
        avg_speed = 0.0
    else:
        avg_speed = math.hypot(dx, dy) / dt

    distance = math.hypot(dx, dy)
    if distance == 0 or pd.isna(distance):
        direction_x, direction_y = 0.0, 0.0
    else:
        direction_x, direction_y = dx / distance, dy / distance

    recent_direction = direction_from_recent_motion(
        df,
        release_idx,
        hand,
        motion_point,
        direction_window,
    )
    if recent_direction:
        direction_x, direction_y = recent_direction

    angle_deg = math.degrees(math.atan2(direction_y, direction_x))
    hit_x, hit_y = calculate_hit_position(
        direction_x,
        direction_y,
        avg_speed,
        board_w,
        board_h,
        sensitivity,
    )

    df["is_release_candidate"] = False
    df.loc[release_idx, "is_release_candidate"] = True
    df["is_start_frame"] = False
    df.loc[start_idx, "is_start_frame"] = True
    df["throw_avg_speed"] = avg_speed
    df["throw_direction_x"] = direction_x
    df["throw_direction_y"] = direction_y
    df["throw_release_speed"] = release_row["filtered_speed"]
    df["throw_motion_point"] = motion_point
    df["throw_release_x"] = release_row[f"{point}_x"]
    df["throw_release_y"] = release_row[f"{point}_y"]
    df["throw_start_x"] = start_row[f"{point}_x"]
    df["throw_start_y"] = start_row[f"{point}_y"]

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

    print("Throw analysis result")
    print(f"Input CSV: {csv_path}")
    print(f"Hand: {hand}")
    print(f"Motion point: {motion_point}")
    print(f"Start frame: {int(start_row['frame_index'])} ({start_row['time_sec']:.4f}s)")
    print(f"Release candidate frame: {int(release_row['frame_index'])} ({release_row['time_sec']:.4f}s)")
    print(f"Release speed threshold: {threshold:.4f}")
    print(f"Release raw speed: {release_row['speed']:.4f}")
    print(f"Release smooth speed: {release_row['smooth_speed']:.4f}")
    print(f"Tracking OK at release: {release_row['tracking_ok']}")
    print(
        f"Release point: ({release_row[f'{point}_x']:.4f}, {release_row[f'{point}_y']:.4f})"
    )
    print(f"Average throw speed: {avg_speed:.4f}")
    print(f"Direction vector: ({direction_x:.4f}, {direction_y:.4f})")
    print(f"Direction angle: {angle_deg:.2f} degrees")
    print(f"Elbow angle at release: {release_row['elbow_angle']:.2f} degrees")
    print(f"Board hit position: ({hit_x}, {hit_y}) on {board_w}x{board_h}")
    if output_csv:
        print(f"Analysis CSV saved: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Analyze throw motion from MediaPipe landmark CSV.")
    parser.add_argument(
        "csv",
        type=Path,
        help="Input CSV created by extract_landmarks.py",
    )
    parser.add_argument(
        "--hand",
        choices=["right", "left"],
        default="right",
        help="Throwing hand to analyze",
    )
    parser.add_argument(
        "--motion-point",
        choices=["auto", "wrist", "thumb_tip", "index_tip", "middle_tip"],
        default="auto",
        help="Landmark used for release timing and throw direction",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/throw_analysis.csv"),
        help="Output CSV with speed and release markers",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Moving average window for speed smoothing",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=10,
        help="Frames before release candidate used as start frame",
    )
    parser.add_argument(
        "--direction-window",
        type=int,
        default=4,
        help="Recent frames before release used to estimate final direction",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.5,
        help="Minimum shoulder/elbow/wrist visibility used for reliable tracking",
    )
    parser.add_argument("--board-w", type=int, default=16, help="Virtual board width")
    parser.add_argument("--board-h", type=int, default=16, help="Virtual board height")
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.35,
        help="Scale factor from throw speed to board position",
    )
    args = parser.parse_args()

    analyze(
        args.csv,
        args.hand,
        args.motion_point,
        args.out,
        args.window,
        args.lookback,
        args.direction_window,
        args.min_visibility,
        args.board_w,
        args.board_h,
        args.sensitivity,
    )


if __name__ == "__main__":
    main()
