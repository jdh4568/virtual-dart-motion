import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt


DEFAULT_HIT_RADIUS_PX = 140
DEFAULT_FRONT_DIRECTION_THRESHOLD = 0.15
DEFAULT_FRONT_DIRECTION_FULL_SCALE = 0.35


def default_targets(width, height):
    return {
        "top": [int(width * 0.5), int(height * 0.28)],
        "left": [int(width * 0.32), int(height * 0.5)],
        "center": [int(width * 0.5), int(height * 0.5)],
        "right": [int(width * 0.68), int(height * 0.5)],
        "bottom": [int(width * 0.5), int(height * 0.72)],
    }


def load_target_config(config_path, width, height):
    if not config_path:
        return {
            "targets": default_targets(width, height),
            "hit_radius_px": DEFAULT_HIT_RADIUS_PX,
            "front_direction_threshold": DEFAULT_FRONT_DIRECTION_THRESHOLD,
            "front_direction_full_scale": DEFAULT_FRONT_DIRECTION_FULL_SCALE,
        }

    with Path(config_path).open("r", encoding="utf-8") as file:
        config = json.load(file)

    targets = config.get("targets") or default_targets(width, height)
    hit_radius_px = config.get("hit_radius_px", DEFAULT_HIT_RADIUS_PX)
    front_direction_threshold = config.get(
        "front_direction_threshold",
        DEFAULT_FRONT_DIRECTION_THRESHOLD,
    )
    front_direction_full_scale = config.get(
        "front_direction_full_scale",
        DEFAULT_FRONT_DIRECTION_FULL_SCALE,
    )
    return {
        "targets": targets,
        "hit_radius_px": hit_radius_px,
        "front_direction_threshold": front_direction_threshold,
        "front_direction_full_scale": front_direction_full_scale,
    }


def endpoint_pixel_from_trajectory(trajectory_csv, width, height):
    df = pd.read_csv(trajectory_csv)
    if df.empty:
        raise ValueError("Trajectory CSV is empty.")

    endpoint = df.iloc[-1]
    return float(endpoint["x"]) * width, float(endpoint["y"]) * height


def front_direction_from_analysis(analysis_csv):
    if not analysis_csv:
        return None

    df = pd.read_csv(analysis_csv)
    release_rows = df[df["is_release_candidate"] == True]
    if release_rows.empty:
        return None

    row = release_rows.iloc[0]
    if "front_camera_enabled" not in row.index or not bool(row["front_camera_enabled"]):
        return None
    if "front_direction_x" not in row.index or pd.isna(row["front_direction_x"]):
        return None

    return float(row["front_direction_x"])


def nearest_target(endpoint_px, targets, hit_radius_px):
    endpoint_x, endpoint_y = endpoint_px
    best_name = None
    best_distance = math.inf

    for name, (target_x, target_y) in targets.items():
        distance = math.hypot(endpoint_x - target_x, endpoint_y - target_y)
        if distance < best_distance:
            best_name = name
            best_distance = distance

    return {
        "target": best_name,
        "distance_px": best_distance,
        "hit": best_distance <= hit_radius_px,
    }


def clamp(value, low, high):
    return max(low, min(high, value))


def endpoint_with_front_direction(endpoint_y, targets, front_direction_x, full_scale):
    center_x = float(targets["center"][0])
    if front_direction_x is None:
        return center_x, endpoint_y, "side_vertical_only"

    if full_scale <= 0:
        full_scale = DEFAULT_FRONT_DIRECTION_FULL_SCALE

    left_x = float(targets["left"][0])
    right_x = float(targets["right"][0])
    if front_direction_x < 0:
        max_offset = center_x - left_x
    else:
        max_offset = right_x - center_x

    ratio = clamp(front_direction_x / full_scale, -1.0, 1.0)
    endpoint_x = center_x + ratio * max_offset
    endpoint_x = clamp(endpoint_x, left_x, right_x)
    return endpoint_x, endpoint_y, "front_continuous_x"


def render_pixel_targets(endpoint_px, targets, hit_result, hit_radius_px, output_png, width, height):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")

    for name, (target_x, target_y) in targets.items():
        is_hit = bool(hit_result["hit"])
        is_target_hit = name == hit_result["target"] and is_hit
        if is_target_hit:
            color = "#ffcc33"
            edge = "#111111"
        else:
            color = "#dddddd"
            edge = "#777777"
        circle = plt.Circle(
            (target_x, target_y),
            hit_radius_px,
            facecolor=color,
            edgecolor=edge,
            linewidth=2,
            alpha=0.5,
        )
        ax.add_patch(circle)
        ax.scatter([target_x], [target_y], color=edge, s=50)
        ax.text(target_x, target_y - hit_radius_px - 18, name, ha="center", fontsize=12)

    endpoint_x, endpoint_y = endpoint_px
    ax.scatter([endpoint_x], [endpoint_y], color="#ff3333", s=120, zorder=5)
    ax.text(endpoint_x + 16, endpoint_y - 16, "endpoint", color="#ff3333", fontsize=12)
    if hit_result.get("front_direction_x") is not None:
        ax.text(
            width * 0.02,
            height * 0.08,
            f"front_direction_x={hit_result['front_direction_x']:.3f}",
            fontsize=12,
            color="#333333",
        )
    status = "HIT" if hit_result["hit"] else "MISS"
    ax.set_title(
        f"Pixel Target Result: {status} / nearest={hit_result['target']} / "
        f"hit={hit_result['hit']} / mode={hit_result.get('mode', 'nearest')} / "
        f"distance={hit_result['distance_px']:.1f}px"
    )
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.grid(True, alpha=0.2)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def save_result_csv(endpoint_px, hit_result, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "endpoint_x_px": endpoint_px[0],
                "endpoint_y_px": endpoint_px[1],
                "target": hit_result["target"],
                "distance_px": hit_result["distance_px"],
                "hit": hit_result["hit"],
                "mode": hit_result.get("mode"),
                "front_direction_x": hit_result.get("front_direction_x"),
                "front_direction_full_scale": hit_result.get("front_direction_full_scale"),
                "target_mirror_x": hit_result.get("target_mirror_x"),
            }
        ]
    ).to_csv(output_csv, index=False)


def evaluate_pixel_targets(
    trajectory_csv,
    output_png,
    output_csv,
    width=1920,
    height=1080,
    config_path=None,
    analysis_csv=None,
    mirror_x=True,
):
    config = load_target_config(config_path, width, height)
    targets = config["targets"]
    hit_radius_px = config["hit_radius_px"]
    front_direction_threshold = config["front_direction_threshold"]
    front_direction_full_scale = config["front_direction_full_scale"]

    trajectory_endpoint_x, endpoint_y = endpoint_pixel_from_trajectory(
        trajectory_csv,
        width,
        height,
    )
    front_direction_x = front_direction_from_analysis(analysis_csv)

    endpoint_x, endpoint_y, mode = endpoint_with_front_direction(
        endpoint_y=endpoint_y,
        targets=targets,
        front_direction_x=front_direction_x,
        full_scale=front_direction_full_scale,
    )
    endpoint_px = (endpoint_x, endpoint_y)
    if mirror_x:
        endpoint_x = width - endpoint_x
        endpoint_px = (endpoint_x, endpoint_y)

    hit_result = nearest_target(endpoint_px, targets, hit_radius_px)
    hit_result["mode"] = mode
    hit_result["front_direction_x"] = front_direction_x
    hit_result["front_direction_full_scale"] = front_direction_full_scale
    hit_result["target_mirror_x"] = mirror_x

    render_pixel_targets(endpoint_px, targets, hit_result, hit_radius_px, output_png, width, height)
    save_result_csv(endpoint_px, hit_result, output_csv)

    print("Pixel target result")
    print(f"Trajectory endpoint px: ({trajectory_endpoint_x:.1f}, {endpoint_y:.1f})")
    print(f"Final endpoint px: ({endpoint_x:.1f}, {endpoint_y:.1f})")
    print(f"Target mirror x: {mirror_x}")
    if front_direction_x is not None:
        print(f"Front direction x: {front_direction_x:.4f}")
        print(f"Front threshold: {front_direction_threshold:.4f}")
        print(f"Front full scale: {front_direction_full_scale:.4f}")
    print(f"Target: {hit_result['target']}")
    print(f"Mode: {hit_result.get('mode')}")
    print(f"Distance: {hit_result['distance_px']:.1f}px")
    print(f"Hit: {hit_result['hit']}")
    print(f"Result CSV: {output_csv}")
    print(f"Result image: {output_png}")

    return hit_result


def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted endpoint against 5 pixel targets.")
    parser.add_argument("trajectory_csv", type=Path)
    parser.add_argument("--analysis-csv", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--mirror-x", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", type=Path, default=Path("output/pixel_targets.png"))
    parser.add_argument("--csv-out", type=Path, default=Path("output/pixel_targets.csv"))
    args = parser.parse_args()

    evaluate_pixel_targets(
        trajectory_csv=args.trajectory_csv,
        output_png=args.out,
        output_csv=args.csv_out,
        width=args.width,
        height=args.height,
        config_path=args.config,
        analysis_csv=args.analysis_csv,
        mirror_x=args.mirror_x,
    )


if __name__ == "__main__":
    main()
