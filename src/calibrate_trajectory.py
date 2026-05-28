import argparse
import json
import math
from pathlib import Path

import pandas as pd


DEFAULT_SEARCH = {
    "velocity_scale": [0.5, 0.8, 1.0, 1.2, 1.5],
    "gravity": [0.0, 0.1, 0.25, 0.4, 0.6],
    "board_scale": [12.0, 16.0, 20.0, 24.0, 30.0],
    "speed_to_mps": [1.5, 2.0, 2.5, 3.0, 3.5],
    "release_offset_frames": [0, 2, 4, 6, 8],
}


def marked_row(df, column):
    rows = df[df[column] == True]
    if rows.empty:
        raise ValueError(f"No {column} row found")
    return rows.iloc[0]


def point_prefix(hand, motion_point):
    return f"{hand}_{motion_point}"


def row_motion_point(row, hand, motion_point):
    point = point_prefix(hand, motion_point)
    x_name = f"{point}_x"
    y_name = f"{point}_y"
    if x_name in row.index and y_name in row.index:
        x = row[x_name]
        y = row[y_name]
        if not pd.isna(x) and not pd.isna(y):
            return x, y
    return row[f"{hand}_wrist_x"], row[f"{hand}_wrist_y"]


def row_has_motion_point(row, hand, motion_point):
    x, y = row_motion_point(row, hand, motion_point)
    if pd.isna(x) or pd.isna(y):
        return False
    if "filtered_speed" in row.index and pd.isna(row["filtered_speed"]):
        return False
    return True


def release_relative_object_points(object_csv, release_x, release_y, release_frame, limit):
    df = pd.read_csv(object_csv)
    df = df[df["object_detected"] == True].copy()
    df = df[df["frame_index"] >= release_frame]
    df = df.dropna(subset=["x", "y"])
    if df.empty:
        raise ValueError("No object points found after release frame")

    points = []
    for _, row in df.head(limit).iterrows():
        points.append((row["x"] - release_x, -(row["y"] - release_y)))
    return points


def build_release_model_row(analysis_df, base_release_row, hand, params):
    offset = int(params["release_offset_frames"])
    base_pos = analysis_df.index.get_loc(base_release_row.name)
    motion_point = base_release_row.get("throw_motion_point", "wrist")

    target_pos = max(0, base_pos - offset)
    release_row = None
    for pos in range(target_pos, base_pos + 1):
        candidate = analysis_df.iloc[pos]
        if row_has_motion_point(candidate, hand, motion_point):
            release_row = candidate
            break
    if release_row is None:
        for pos in range(target_pos, -1, -1):
            candidate = analysis_df.iloc[pos]
            if row_has_motion_point(candidate, hand, motion_point):
                release_row = candidate
                break
    if release_row is None:
        release_row = base_release_row

    release_x, release_y = row_motion_point(release_row, hand, motion_point)
    start_x = base_release_row.get("throw_start_x", math.nan)
    start_y = base_release_row.get("throw_start_y", math.nan)

    if pd.isna(start_x) or pd.isna(start_y):
        start_row = marked_row(analysis_df, "is_start_frame")
        start_x, start_y = row_motion_point(start_row, hand, motion_point)

    dx = release_x - start_x
    dy = release_y - start_y
    distance = math.hypot(dx, dy)
    if distance == 0 or pd.isna(distance):
        direction_x, direction_y = 0.0, 0.0
    else:
        direction_x, direction_y = dx / distance, dy / distance

    release_speed = release_row.get("filtered_speed", base_release_row.get("throw_release_speed", 0.0))

    return {
        "frame_index": int(release_row["frame_index"]),
        "release_x": release_x,
        "release_y": release_y,
        "direction_x": direction_x,
        "direction_y": direction_y,
        "release_speed": release_speed,
    }


def predicted_points(model, params, count, board_distance, duration, steps):
    release_x = model["release_x"]
    release_y = model["release_y"]
    direction_x = model["direction_x"]
    direction_y = model["direction_y"]
    release_speed = model["release_speed"]

    vx = direction_x * release_speed * params["velocity_scale"]
    vy = direction_y * release_speed * params["velocity_scale"]

    forward_speed = release_speed * params["speed_to_mps"]
    if board_distance and forward_speed > 0:
        flight_duration = max(0.25, min(1.2, board_distance / forward_speed))
    else:
        flight_duration = duration

    sample_count = max(count, steps)
    points = []
    for i in range(sample_count):
        t = flight_duration * i / max(1, sample_count - 1)
        x = release_x + vx * t
        y = release_y + vy * t + 0.5 * params["gravity"] * (t**2)
        points.append((x - release_x, -(y - release_y)))
    return points[:count]


def mean_error(predicted, actual):
    count = min(len(predicted), len(actual))
    if count == 0:
        return math.inf
    total = 0.0
    for i in range(count):
        px, py = predicted[i]
        ax, ay = actual[i]
        total += math.hypot(px - ax, py - ay)
    return total / count


def iter_params(search):
    for velocity_scale in search["velocity_scale"]:
        for gravity in search["gravity"]:
            for board_scale in search["board_scale"]:
                for speed_to_mps in search["speed_to_mps"]:
                    for release_offset_frames in search["release_offset_frames"]:
                        yield {
                            "velocity_scale": velocity_scale,
                            "gravity": gravity,
                            "board_scale": board_scale,
                            "speed_to_mps": speed_to_mps,
                            "release_offset_frames": release_offset_frames,
                        }


def calibrate(
    analysis_csv,
    object_csv,
    output_json,
    hand="right",
    board_distance=2.0,
    duration=0.8,
    steps=30,
    search=None,
):
    search = search or DEFAULT_SEARCH
    analysis_df = pd.read_csv(analysis_csv)
    release_row = marked_row(analysis_df, "is_release_candidate")

    best = None
    for params in iter_params(search):
        model = build_release_model_row(analysis_df, release_row, hand, params)
        try:
            actual = release_relative_object_points(
                object_csv,
                model["release_x"],
                model["release_y"],
                model["frame_index"],
                steps,
            )
        except ValueError:
            continue
        predicted = predicted_points(model, params, len(actual), board_distance, duration, steps)
        error = mean_error(predicted, actual)
        if best is None or error < best["mean_error"]:
            best = {"mean_error": error, **params}

    if best is None:
        raise ValueError("No calibration candidate could be evaluated.")

    output = {
        "version": 1,
        "source_analysis_csv": str(analysis_csv),
        "source_object_csv": str(object_csv),
        "hand": hand,
        "board_distance": board_distance,
        "mean_error": best["mean_error"],
        "velocity_scale": best["velocity_scale"],
        "gravity": best["gravity"],
        "board_scale": best["board_scale"],
        "speed_to_mps": best["speed_to_mps"],
        "release_offset_frames": best["release_offset_frames"],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Calibration saved: {output_json}")
    print(f"Mean error: {best['mean_error']:.6f}")
    print(
        "Recommended: "
        f"velocity_scale={best['velocity_scale']}, "
        f"gravity={best['gravity']}, "
        f"board_scale={best['board_scale']}, "
        f"speed_to_mps={best['speed_to_mps']}, "
        f"release_offset_frames={best['release_offset_frames']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Calibrate virtual trajectory parameters from object tracking.")
    parser.add_argument("analysis_csv", type=Path, help="Throw analysis CSV")
    parser.add_argument("object_csv", type=Path, help="Object trajectory CSV")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/calibration.json"),
        help="Output calibration JSON path",
    )
    parser.add_argument("--board-distance", type=float, default=2.0)
    parser.add_argument("--hand", choices=["right", "left"], default="right")
    parser.add_argument("--duration", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    calibrate(
        analysis_csv=args.analysis_csv,
        object_csv=args.object_csv,
        output_json=args.out,
        hand=args.hand,
        board_distance=args.board_distance,
        duration=args.duration,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()
