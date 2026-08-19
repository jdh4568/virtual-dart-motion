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


def build_dart_trajectory(
    release_x,
    release_y,
    direction_x,
    direction_y,
    dart_speed_mps,
    board_distance,
    board_width_m,
    board_height_m,
    gravity_mps2,
    max_horizontal_angle_deg,
    max_vertical_angle_deg,
    steps,
    screen_endpoint_x=None,
):
    if steps < 2:
        raise ValueError("--steps must be at least 2")
    if dart_speed_mps <= 0:
        raise ValueError("--dart-speed-mps must be greater than 0")
    if board_distance <= 0:
        raise ValueError("--board-distance must be greater than 0 in dart mode")

    horizontal_angle = math.radians(direction_x * max_horizontal_angle_deg)
    vertical_angle = math.radians(-direction_y * max_vertical_angle_deg)

    forward_speed = dart_speed_mps * math.cos(horizontal_angle) * math.cos(vertical_angle)
    if forward_speed <= 0:
        raise ValueError("Estimated forward speed must be greater than 0")

    vx_mps = dart_speed_mps * math.sin(horizontal_angle)
    vy_mps = dart_speed_mps * math.sin(vertical_angle)
    flight_duration = board_distance / forward_speed

    points = []
    for i in range(steps):
        progress = i / (steps - 1)
        t = flight_duration * i / (steps - 1)
        x_m = vx_mps * t
        y_m = vy_mps * t - 0.5 * gravity_mps2 * (t**2)

        # Approximate a screen-space path for the video overlay while preserving
        # meter-based columns for physics and board hit calculation.
        if screen_endpoint_x is None:
            x = release_x + (x_m / board_width_m)
        else:
            x = release_x + (screen_endpoint_x - release_x) * progress
        y = release_y - (y_m / board_height_m)
        points.append(
            {
                "point_index": i,
                "t": t,
                "progress": progress,
                "x": x,
                "y": y,
                "x_m": x_m,
                "y_m": y_m,
                "z_m": board_distance * progress,
                "is_2m_endpoint": i == steps - 1,
            }
        )

    return points, flight_duration, vx_mps, vy_mps, forward_speed


def marker_extension_direction(start_row, release_row, hand, fallback_x, fallback_y):
    start_x, start_y = row_point(start_row, hand, "start")
    release_x, release_y = row_point(release_row, hand, "release")

    if not any(pd.isna(value) for value in [start_x, start_y, release_x, release_y]):
        marker_direction_x = release_x - start_x
        marker_direction_y = release_y - start_y
        direction_x, direction_y = normalize_vector(marker_direction_x, marker_direction_y)
        if direction_x != 0.0 or direction_y != 0.0:
            return direction_x, direction_y

    return side_extension_direction(release_row, fallback_x, fallback_y)


def side_extension_direction(release_row, direction_x, direction_y):
    side_direction_x = release_row.get("throw_side_direction_x", direction_x)
    if pd.isna(side_direction_x):
        side_direction_x = direction_x

    if pd.isna(direction_y):
        direction_y = 0.0

    return normalize_vector(float(side_direction_x), float(direction_y))


def build_extended_trajectory(
    release_x,
    release_y,
    direction_x,
    direction_y,
    screen_endpoint_x,
    duration,
    gravity,
    steps,
):
    if steps < 2:
        raise ValueError("--steps must be at least 2")
    if screen_endpoint_x is None:
        screen_endpoint_x = 1.0

    direction_x, direction_y = normalize_vector(direction_x, direction_y)
    if abs(direction_x) < 1e-6:
        direction_x = 1.0
    if direction_x < 0:
        direction_x *= -1.0
        direction_y *= -1.0

    endpoint_x = screen_endpoint_x
    delta_x = endpoint_x - release_x
    if delta_x <= 0:
        endpoint_x = min(1.0, max(release_x + 0.01, screen_endpoint_x))
        delta_x = endpoint_x - release_x

    slope = direction_y / direction_x
    arc_height = min(0.035, max(0.012, gravity * 0.08))
    points = []
    for i in range(steps):
        progress = i / (steps - 1)
        t = duration * progress
        x = release_x + delta_x * progress
        visual_lift = 4.0 * arc_height * progress * (1.0 - progress)
        y = release_y + slope * delta_x * progress - visual_lift
        points.append(
            {
                "point_index": i,
                "t": t,
                "progress": progress,
                "x": x,
                "y": y,
                "is_2m_endpoint": i == steps - 1,
            }
        )

    return points, duration, direction_x, direction_y


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


def endpoint_meters_to_board(endpoint, board_w, board_h, board_width_m, board_height_m):
    center_x = (board_w - 1) / 2
    center_y = (board_h - 1) / 2

    hit_x = center_x + (endpoint["x_m"] / board_width_m) * board_w
    hit_y = center_y - (endpoint["y_m"] / board_height_m) * board_h

    return (
        int(round(clamp(hit_x, 0, board_w - 1))),
        int(round(clamp(hit_y, 0, board_h - 1))),
    )


def build_object_corrected_trajectory(object_track_csv, release_x, release_y, min_object_points):
    if not object_track_csv:
        return None

    track_df = pd.read_csv(object_track_csv)
    if track_df.empty or len(track_df) < min_object_points:
        return None

    points = []
    first_time = track_df.iloc[0]["time_sec"]
    for i, row in enumerate(track_df.itertuples(index=False)):
        if pd.isna(row.x) or pd.isna(row.y):
            continue
        points.append(
            {
                "point_index": i,
                "t": row.time_sec - first_time,
                "x": row.x,
                "y": row.y,
                "object_area": row.area,
            }
        )

    if len(points) < min_object_points:
        return None

    return points


def save_plot(
    points,
    start_row,
    release_row,
    hand,
    hit,
    plot_path,
    board_w,
    board_h,
    board_distance,
    board_height_m,
    physics_mode,
):
    import matplotlib.pyplot as plt

    endpoint = points[-1]
    start_x, start_y = row_point(start_row, hand, "start")
    release_x, release_y = row_point(release_row, hand, "release")
    motion_point = release_row.get("throw_motion_point", "wrist")

    if physics_mode == "dart" and {"z_m", "y_m"}.issubset(points[-1]):
        xs = [point["z_m"] for point in points]
        ys = [point["y_m"] for point in points]
        start_plot_x = 0
        start_plot_y = -(start_y - release_y) * board_height_m
        x_label = "Forward distance from release (m)"
        y_label = "Vertical displacement from release (m)"
    else:
        xs = [point["x"] - release_x for point in points]
        ys = [-(point["y"] - release_y) for point in points]
        start_plot_x = start_x - release_x
        start_plot_y = -(start_y - release_y)
        x_label = "Horizontal movement from release"
        y_label = "Vertical movement from release"

    endpoint_plot_x = xs[-1]
    endpoint_plot_y = ys[-1]

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
    if physics_mode == "dart" and {"z_m", "y_m"}.issubset(points[-1]):
        min_x = 0
        max_x = board_distance if board_distance else max(xs)
    else:
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
    plt.xlabel(x_label)
    plt.ylabel(y_label)
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
    physics_mode,
    dart_speed_mps,
    board_width_m,
    board_height_m,
    gravity_mps2,
    max_horizontal_angle_deg,
    max_vertical_angle_deg,
    object_track_csv,
    min_object_points,
    screen_endpoint_x,
):
    df = pd.read_csv(analysis_csv)
    hand = hand.lower()

    start_row = get_marked_row(df, "is_start_frame")
    release_row = get_marked_row(df, "is_release_candidate")

    vx, vy, speed = estimate_initial_velocity(start_row, release_row, hand, velocity_scale)
    direction_x, direction_y = normalize_vector(vx, vy)

    release_x, release_y = row_point(release_row, hand, "release")
    release_speed = release_row.get("throw_release_speed", speed)

    points = build_object_corrected_trajectory(
        object_track_csv=object_track_csv,
        release_x=release_x,
        release_y=release_y,
        min_object_points=min_object_points,
    )
    using_object_correction = points is not None

    if physics_mode == "extend" and not using_object_correction:
        direction_x, direction_y = marker_extension_direction(
            start_row,
            release_row,
            hand,
            direction_x,
            direction_y,
        )
        flight_duration = estimate_flight_duration(
            release_speed=release_speed,
            board_distance=board_distance,
            speed_to_mps=speed_to_mps,
            min_duration=min_duration,
            max_duration=max_duration,
            fallback_duration=duration,
        )
        points, flight_duration, direction_x, direction_y = build_extended_trajectory(
            release_x=release_x,
            release_y=release_y,
            direction_x=direction_x,
            direction_y=direction_y,
            screen_endpoint_x=screen_endpoint_x,
            duration=flight_duration,
            gravity=gravity,
            steps=steps,
        )
        dart_vx_mps = math.nan
        dart_vy_mps = math.nan
        dart_forward_mps = math.nan
    elif physics_mode == "dart" and not using_object_correction:
        points, flight_duration, dart_vx_mps, dart_vy_mps, dart_forward_mps = build_dart_trajectory(
            release_x=release_x,
            release_y=release_y,
            direction_x=direction_x,
            direction_y=direction_y,
            dart_speed_mps=dart_speed_mps,
            board_distance=board_distance,
            board_width_m=board_width_m,
            board_height_m=board_height_m,
            gravity_mps2=gravity_mps2,
            max_horizontal_angle_deg=max_horizontal_angle_deg,
            max_vertical_angle_deg=max_vertical_angle_deg,
            steps=steps,
            screen_endpoint_x=screen_endpoint_x,
        )
    else:
        dart_vx_mps = math.nan
        dart_vy_mps = math.nan
        dart_forward_mps = math.nan
        if using_object_correction:
            flight_duration = points[-1]["t"]
        else:
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
    if using_object_correction:
        hit = endpoint_to_board(endpoint, release_x, release_y, board_w, board_h, board_scale)
    elif physics_mode == "dart":
        hit = endpoint_meters_to_board(endpoint, board_w, board_h, board_width_m, board_height_m)
    else:
        hit = endpoint_to_board(endpoint, release_x, release_y, board_w, board_h, board_scale)

    trajectory_df = pd.DataFrame(points)
    trajectory_df["relative_x_from_release"] = trajectory_df["x"] - release_x
    trajectory_df["relative_y_from_release"] = -(trajectory_df["y"] - release_y)
    trajectory_df["vx"] = vx
    trajectory_df["vy"] = vy
    trajectory_df["speed"] = speed
    trajectory_df["direction_x"] = direction_x
    trajectory_df["direction_y"] = direction_y
    trajectory_df["physics_mode"] = physics_mode
    trajectory_df["trajectory_source"] = "object_track" if using_object_correction else physics_mode
    trajectory_df["board_distance_m"] = board_distance
    trajectory_df["board_width_m"] = board_width_m
    trajectory_df["board_height_m"] = board_height_m
    trajectory_df["speed_to_mps"] = speed_to_mps
    trajectory_df["dart_speed_mps"] = dart_speed_mps
    trajectory_df["dart_vx_mps"] = dart_vx_mps
    trajectory_df["dart_vy_mps"] = dart_vy_mps
    trajectory_df["dart_forward_mps"] = dart_forward_mps
    trajectory_df["gravity_mps2"] = gravity_mps2
    trajectory_df["flight_duration"] = flight_duration
    trajectory_df["screen_endpoint_x"] = screen_endpoint_x
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
            board_height_m,
            physics_mode,
        )

    print("Trajectory prediction result")
    print(f"Input analysis CSV: {analysis_csv}")
    print(f"Hand: {hand}")
    print(f"Motion point: {release_row.get('throw_motion_point', 'wrist')}")
    print(f"Initial velocity: ({vx:.4f}, {vy:.4f})")
    print(f"Initial speed: {speed:.4f}")
    print(f"Direction: ({direction_x:.4f}, {direction_y:.4f})")
    print(f"Physics mode: {physics_mode}")
    print(f"Trajectory source: {'object_track' if using_object_correction else physics_mode}")
    print(f"Gravity: {gravity_mps2 if physics_mode == 'dart' else gravity}")
    print(f"Board distance: {board_distance}m")
    if physics_mode == "dart":
        print(f"Dart speed: {dart_speed_mps}m/s")
        print(f"Dart velocity: ({dart_vx_mps:.4f}, {dart_vy_mps:.4f}, {dart_forward_mps:.4f}) m/s")
    print(f"Flight duration: {flight_duration:.4f}s")
    print(f"Endpoint: ({endpoint['x']:.4f}, {endpoint['y']:.4f})")
    if screen_endpoint_x is not None:
        print(f"Screen endpoint x: {screen_endpoint_x:.4f}")
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
        "--physics-mode",
        choices=["simple", "dart", "extend"],
        default="extend",
        help="Trajectory model to use",
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
    parser.add_argument(
        "--dart-speed-mps",
        type=float,
        default=8.0,
        help="Initial dart speed in meters per second for dart physics mode",
    )
    parser.add_argument(
        "--board-width-m",
        type=float,
        default=0.6,
        help="Physical board width in meters for dart physics mode",
    )
    parser.add_argument(
        "--board-height-m",
        type=float,
        default=0.6,
        help="Physical board height in meters for dart physics mode",
    )
    parser.add_argument(
        "--gravity-mps2",
        type=float,
        default=9.81,
        help="Gravity in meters per second squared for dart physics mode",
    )
    parser.add_argument(
        "--max-horizontal-angle-deg",
        type=float,
        default=15.0,
        help="Maximum horizontal launch angle mapped from camera direction",
    )
    parser.add_argument(
        "--max-vertical-angle-deg",
        type=float,
        default=15.0,
        help="Maximum vertical launch angle mapped from camera direction",
    )
    parser.add_argument(
        "--object-track-csv",
        type=Path,
        help="Optional object tracking CSV used to correct trajectory",
    )
    parser.add_argument(
        "--min-object-points",
        type=int,
        default=3,
        help="Minimum detected object points required for correction",
    )
    parser.add_argument(
        "--screen-endpoint-x",
        type=float,
        help="Optional normalized x position for the rendered 2m endpoint.",
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
        args.physics_mode,
        args.dart_speed_mps,
        args.board_width_m,
        args.board_height_m,
        args.gravity_mps2,
        args.max_horizontal_angle_deg,
        args.max_vertical_angle_deg,
        args.object_track_csv,
        args.min_object_points,
        args.screen_endpoint_x,
    )


if __name__ == "__main__":
    main()
