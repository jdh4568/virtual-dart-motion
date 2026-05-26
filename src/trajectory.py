import argparse
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import pandas as pd


def clamp(value, low, high):
    return max(low, min(high, value))


def get_marked_row(df, column_name):
    if column_name not in df.columns:
        raise ValueError(f"Missing marker column: {column_name}")

    marked = df[df[column_name] == True]
    if marked.empty:
        raise ValueError(f"No frame is marked as {column_name}")

    return marked.iloc[0]


def normalize_vector(x, y):
    length = math.hypot(x, y)
    if length == 0 or math.isnan(length):
        return 0.0, 0.0
    return x / length, y / length


def row_point(row, hand, point_kind):
    if point_kind == "start" and {"throw_start_x", "throw_start_y"}.issubset(row.index):
        if not any(pd.isna(row[column]) for column in ["throw_start_x", "throw_start_y"]):
            return row["throw_start_x"], row["throw_start_y"]

    if point_kind == "release" and {"throw_release_x", "throw_release_y"}.issubset(row.index):
        if not any(pd.isna(row[column]) for column in ["throw_release_x", "throw_release_y"]):
            return row["throw_release_x"], row["throw_release_y"]

    return row[f"{hand}_wrist_x"], row[f"{hand}_wrist_y"]


def estimate_initial_velocity(start_row, release_row, hand, velocity_scale):
    if {
        "throw_direction_x",
        "throw_direction_y",
        "throw_release_speed",
    }.issubset(release_row.index):
        direction_x = release_row["throw_direction_x"]
        direction_y = release_row["throw_direction_y"]
        release_speed = release_row["throw_release_speed"]
        if not any(pd.isna(v) for v in [direction_x, direction_y, release_speed]):
            vx = direction_x * release_speed * velocity_scale
            vy = direction_y * release_speed * velocity_scale
            speed = math.hypot(vx, vy)
            return vx, vy, speed

    start_x, start_y = row_point(start_row, hand, "start")
    release_x, release_y = row_point(release_row, hand, "release")
    dt = release_row["time_sec"] - start_row["time_sec"]

    if dt <= 0 or pd.isna(dt):
        return 0.0, 0.0, 0.0

    vx = ((release_x - start_x) / dt) * velocity_scale
    vy = ((release_y - start_y) / dt) * velocity_scale
    speed = math.hypot(vx, vy)
    return vx, vy, speed


def estimate_flight_duration(
    release_speed,
    board_distance,
    speed_to_mps,
    min_duration,
    max_duration,
    fallback_duration,
):
    if board_distance is None or board_distance <= 0:
        return fallback_duration

    if pd.isna(release_speed) or release_speed <= 0:
        return fallback_duration

    forward_speed_mps = release_speed * speed_to_mps
    if forward_speed_mps <= 0:
        return fallback_duration

    duration = board_distance / forward_speed_mps
    return clamp(duration, min_duration, max_duration)


def build_trajectory(release_x, release_y, vx, vy, gravity, duration, steps):
    points = []
    if steps < 2:
        raise ValueError("--steps must be at least 2")

    for i in range(steps):
        t = duration * i / (steps - 1)
        x = release_x + vx * t
        y = release_y + vy * t + 0.5 * gravity * (t**2)
        points.append(
            {
                "point_index": i,
                "t": t,
                "x": x,
                "y": y,
            }
        )

    return points


def endpoint_to_board(endpoint, release_x, release_y, board_w, board_h, board_scale):
    center_x = (board_w - 1) / 2
    center_y = (board_h - 1) / 2

    delta_x = endpoint["x"] - release_x
    delta_y = endpoint["y"] - release_y

    hit_x = center_x + delta_x * board_scale
    hit_y = center_y + delta_y * board_scale

    return (
        int(round(clamp(hit_x, 0, board_w - 1))),
        int(round(clamp(hit_y, 0, board_h - 1))),
    )


def save_plot(points, start_row, release_row, hand, hit, plot_path, board_w, board_h, board_distance):
    import matplotlib.pyplot as plt

    endpoint = points[-1]
    start_x, start_y = row_point(start_row, hand, "start")
    release_x, release_y = row_point(release_row, hand, "release")
    motion_point = release_row.get("throw_motion_point", "wrist")

    xs = [point["x"] - release_x for point in points]
    ys = [-(point["y"] - release_y) for point in points]
    start_plot_x = start_x - release_x
    start_plot_y = -(start_y - release_y)
    endpoint_plot_x = endpoint["x"] - release_x
    endpoint_plot_y = -(endpoint["y"] - release_y)

    plt.figure(figsize=(7, 7))
    plt.plot(xs, ys, marker="o", linewidth=2, markersize=3, label="Predicted trajectory")
    plt.scatter(
        [start_plot_x],
        [start_plot_y],
        color="gray",
        s=80,
        label="Start frame",
    )
    plt.scatter(
        [0],
        [0],
        color="red",
        s=80,
        label=f"Release candidate ({motion_point})",
    )
    plt.scatter(
        [endpoint_plot_x],
        [endpoint_plot_y],
        color="dodgerblue",
        edgecolor="black",
        s=90,
        label="Trajectory endpoint",
    )
    plt.annotate(
        "release",
        (0, 0),
        textcoords="offset points",
        xytext=(8, -14),
    )
    plt.annotate(
        "endpoint",
        (endpoint_plot_x, endpoint_plot_y),
        textcoords="offset points",
        xytext=(8, 8),
    )
    margin = 0.08
    min_x = min(min(xs), start_plot_x, 0) - margin
    max_x = max(max(xs), start_plot_x, 0) + margin
    min_y = min(min(ys), start_plot_y, 0) - margin
    max_y = max(max(ys), start_plot_y, 0) + margin
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.axhline(0, color="black", linewidth=1, alpha=0.25)
    plt.axvline(0, color="black", linewidth=1, alpha=0.25)
    plt.grid(True, alpha=0.25)
    distance_label = f"{board_distance}m" if board_distance else "virtual"
    plt.title(
        f"Release-relative Trajectory / Board Hit: {hit} on {board_w}x{board_h} / Distance: {distance_label}"
    )
    plt.xlabel("Horizontal movement from release")
    plt.ylabel("Vertical movement from release")
    plt.legend()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def predict(
    analysis_csv,
    hand,
    output_csv,
    plot_path,
    gravity,
    duration,
    steps,
    velocity_scale,
    board_w,
    board_h,
    board_scale,
    board_distance,
    speed_to_mps,
    min_duration,
    max_duration,
):
    df = pd.read_csv(analysis_csv)
    hand = hand.lower()

    start_row = get_marked_row(df, "is_start_frame")
    release_row = get_marked_row(df, "is_release_candidate")

    vx, vy, speed = estimate_initial_velocity(start_row, release_row, hand, velocity_scale)
    direction_x, direction_y = normalize_vector(vx, vy)

    release_x, release_y = row_point(release_row, hand, "release")
    release_speed = release_row.get("throw_release_speed", speed)
    flight_duration = estimate_flight_duration(
        release_speed=release_speed,
        board_distance=board_distance,
        speed_to_mps=speed_to_mps,
        min_duration=min_duration,
        max_duration=max_duration,
        fallback_duration=duration,
    )

    points = build_trajectory(release_x, release_y, vx, vy, gravity, flight_duration, steps)
    endpoint = points[-1]
    hit = endpoint_to_board(endpoint, release_x, release_y, board_w, board_h, board_scale)

    trajectory_df = pd.DataFrame(points)
    trajectory_df["relative_x_from_release"] = trajectory_df["x"] - release_x
    trajectory_df["relative_y_from_release"] = -(trajectory_df["y"] - release_y)
    trajectory_df["vx"] = vx
    trajectory_df["vy"] = vy
    trajectory_df["speed"] = speed
    trajectory_df["direction_x"] = direction_x
    trajectory_df["direction_y"] = direction_y
    trajectory_df["board_distance_m"] = board_distance
    trajectory_df["speed_to_mps"] = speed_to_mps
    trajectory_df["flight_duration"] = flight_duration
    trajectory_df["board_hit_x"] = hit[0]
    trajectory_df["board_hit_y"] = hit[1]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    trajectory_df.to_csv(output_csv, index=False)

    if plot_path:
        save_plot(
            points,
            start_row,
            release_row,
            hand,
            hit,
            plot_path,
            board_w,
            board_h,
            board_distance,
        )

    print("Trajectory prediction result")
    print(f"Input analysis CSV: {analysis_csv}")
    print(f"Hand: {hand}")
    print(f"Motion point: {release_row.get('throw_motion_point', 'wrist')}")
    print(f"Initial velocity: ({vx:.4f}, {vy:.4f})")
    print(f"Initial speed: {speed:.4f}")
    print(f"Direction: ({direction_x:.4f}, {direction_y:.4f})")
    print(f"Gravity: {gravity}")
    print(f"Board distance: {board_distance}m")
    print(f"Flight duration: {flight_duration:.4f}s")
    print(f"Endpoint: ({endpoint['x']:.4f}, {endpoint['y']:.4f})")
    print(f"Board hit position: {hit} on {board_w}x{board_h}")
    print(f"Trajectory CSV saved: {output_csv}")
    if plot_path:
        print(f"Trajectory plot saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict a virtual dart trajectory from throw analysis CSV."
    )
    parser.add_argument(
        "analysis_csv",
        type=Path,
        help="CSV created by analyze_throw.py",
    )
    parser.add_argument(
        "--hand",
        choices=["right", "left"],
        default="right",
        help="Throwing hand to analyze",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/trajectory.csv"),
        help="Output CSV for trajectory points",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("output/trajectory.png"),
        help="Optional trajectory plot path",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=0.25,
        help="Virtual downward acceleration in normalized camera units",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.8,
        help="Fallback virtual flight duration in seconds",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of trajectory sample points",
    )
    parser.add_argument(
        "--velocity-scale",
        type=float,
        default=0.8,
        help="Scale factor applied to wrist velocity before trajectory prediction",
    )
    parser.add_argument("--board-w", type=int, default=16, help="Virtual board width")
    parser.add_argument("--board-h", type=int, default=16, help="Virtual board height")
    parser.add_argument(
        "--board-scale",
        type=float,
        default=20.0,
        help="Scale factor from normalized trajectory displacement to board cells",
    )
    parser.add_argument(
        "--board-distance",
        type=float,
        default=2.0,
        help="Fixed distance from thrower to virtual board in meters",
    )
    parser.add_argument(
        "--speed-to-mps",
        type=float,
        default=2.5,
        help="Scale factor from normalized release speed to estimated forward m/s",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.25,
        help="Minimum estimated flight duration in seconds",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=1.2,
        help="Maximum estimated flight duration in seconds",
    )
    args = parser.parse_args()

    predict(
        args.analysis_csv,
        args.hand,
        args.out,
        args.plot,
        args.gravity,
        args.duration,
        args.steps,
        args.velocity_scale,
        args.board_w,
        args.board_h,
        args.board_scale,
        args.board_distance,
        args.speed_to_mps,
        args.min_duration,
        args.max_duration,
    )


if __name__ == "__main__":
    main()
