import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

from src.adb_capture import AdbCaptureError, capture_video
from src.extract_landmarks import extract_pose
from src.analyze_throw import analyze
from src.front_camera import apply_front_camera_direction
from src.object_tracker import track_object, correct_release_from_object_track, COLOR_RANGES
from src.pixel_targets import evaluate_pixel_targets
from src.trajectory import predict
from src.trajectory_quality import evaluate_trajectory_quality
from src.simulate_board import read_hit_position, render_board
from src.render_analysis_preview import (
    render_grid_trajectory,
    render_preview,
    render_release_frame_image,
)
from src.mqtt_led_publisher import publish_led_result


def find_latest_video(videos_dir):
    videos = []

    for pattern in ["*.mp4", "*.mov", "*.MOV", "*.MP4"]:
        videos.extend(videos_dir.glob(pattern))

    if not videos:
        raise FileNotFoundError(f"No video file found in {videos_dir}")

    return max(videos, key=lambda path: path.stat().st_mtime)


def build_run_name(video_path, flip_horizontal):
    run_name = video_path.stem

    if flip_horizontal:
        run_name = f"{run_name}_flipped"

    return run_name


def normalized_screen_endpoint_x(video_path, endpoint_margin_px):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    cap.release()

    if width <= 0:
        return None

    margin = max(0, min(endpoint_margin_px, width - 1))

    return (width - margin) / width


def safe_float(value, default=None):
    if value is None:
        return default

    if pd.isna(value):
        return default

    return float(value)


def safe_string(value, default=None):
    if value is None:
        return default

    if pd.isna(value):
        return default

    return str(value)


def safe_bool(value):
    if value is None:
        return False

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    if isinstance(value, str):
        return value.strip().lower() in ["true", "1", "yes", "y"]

    return bool(value)


def write_latest_result(
    run_name,
    video_path,
    hit_x,
    hit_y,
    trajectory_png,
    board_png,
    pixel_target_png,
    pixel_target_csv,
    analysis_preview,
    grid_trajectory_png,
    trajectory_quality,
):

    if not pixel_target_csv.exists():
        raise FileNotFoundError(
            f"Pixel target CSV not found: {pixel_target_csv}"
        )

    pixel_df = pd.read_csv(pixel_target_csv)

    if pixel_df.empty:
        raise ValueError(
            f"Pixel target CSV is empty: {pixel_target_csv}"
        )

    pixel_row = pixel_df.iloc[0]

    target = safe_string(
        pixel_row.get("target"),
        default="center"
    )

    endpoint_x_px = safe_float(
        pixel_row.get("endpoint_x_px"),
        default=960.0
    )

    endpoint_y_px = safe_float(
        pixel_row.get("endpoint_y_px"),
        default=540.0
    )

    distance_px = safe_float(
        pixel_row.get("distance_px"),
        default=None
    )

    pixel_hit = safe_bool(
        pixel_row.get("hit")
    )

    mode = safe_string(
        pixel_row.get("mode"),
        default=None
    )

    front_direction_x = safe_float(
        pixel_row.get("front_direction_x"),
        default=None
    )

    latest_result = {
        "success": True,

        "run_name": run_name,

        "hit_x": int(hit_x),
        "hit_y": int(hit_y),

        "target": target,
        "endpoint_x_px": endpoint_x_px,
        "endpoint_y_px": endpoint_y_px,
        "distance_px": distance_px,
        "hit": pixel_hit,
        "mode": mode,
        "front_direction_x": front_direction_x,

        "video_name": video_path.name,

        "trajectory_image":
            f"/output/{run_name}/{trajectory_png.name}",

        "board_image":
            f"/output/{run_name}/{board_png.name}",

        "pixel_target_image":
            f"/output/{run_name}/{pixel_target_png.name}",

        "preview_video":
            f"/output/{run_name}/{analysis_preview.name}",

        "grid_trajectory_image":
            f"/output/{run_name}/{grid_trajectory_png.name}",

        "pixel_target_csv":
            f"/output/{run_name}/{pixel_target_csv.name}",

        **trajectory_quality,
    }

    latest_result_path = Path("output") / "latest_result.json"

    latest_result_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with latest_result_path.open(
        "w",
        encoding="utf-8"
    ) as result_file:
        json.dump(
            latest_result,
            result_file,
            ensure_ascii=False,
            indent=2
        )

    print(f"Latest result saved: {latest_result_path}")
    print(f"Target: {target}")
    print(f"Endpoint px: ({endpoint_x_px}, {endpoint_y_px})")
    print(f"Trajectory valid: {trajectory_quality['trajectory_valid']}")
    print(f"Trajectory quality: {trajectory_quality['trajectory_quality_reason']}")

    return latest_result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run virtual dart motion analysis."
    )

    parser.add_argument(
        "input_video",
        nargs="?",
        type=Path,
        help="Input video path. If omitted, the latest video in videos/ is used.",
    )

    parser.add_argument(
        "--video",
        type=Path,
        help="Input video path. Kept for compatibility with the ADB workflow.",
    )

    parser.add_argument(
        "--adb-capture",
        action="store_true",
        help="Record a new video over ADB, pull it into videos/, then analyze it.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("adb_config.json"),
        help="ADB capture config path.",
    )

    parser.add_argument(
        "--hand",
        choices=["right", "left"],
        default="right",
        help="Throwing hand to analyze.",
    )

    parser.add_argument(
        "--motion-point",
        choices=["auto", "wrist", "thumb_tip", "index_tip", "middle_tip"],
        default="auto",
        help="Landmark used for release timing and trajectory start.",
    )

    parser.add_argument(
        "--start-mode",
        choices=["recent", "video-start"],
        default="video-start",
        help="How to choose the throw start frame.",
    )

    parser.add_argument(
        "--flip-horizontal",
        action="store_true",
        help="Flip mirrored/selfie videos before analysis.",
    )

    parser.add_argument(
        "--front-video",
        type=Path,
        help="Optional front camera video used to correct left-right direction.",
    )

    parser.add_argument(
        "--front-flip-horizontal",
        action="store_true",
        help="Flip the front camera video before extracting landmarks.",
    )

    parser.add_argument(
        "--front-direction-window",
        type=int,
        default=20,
        help="Frames before the side release frame used for front left-right direction.",
    )

    parser.add_argument(
        "--front-frame-offset",
        type=int,
        default=5,
        help="Frame offset added to the side release frame for front camera direction.",
    )

    parser.add_argument(
        "--front-horizontal-gain",
        type=float,
        default=1.0,
        help="Scale applied to front camera left-right direction.",
    )

    parser.add_argument(
        "--board-distance",
        type=float,
        default=2.0,
        help="Fixed distance from thrower to virtual board in meters.",
    )

    parser.add_argument(
        "--physics-mode",
        choices=["simple", "dart", "extend"],
        default="extend",
        help="Trajectory model to use.",
    )

    parser.add_argument(
        "--dart-speed-mps",
        type=float,
        default=8.0,
        help="Initial dart speed in meters per second for dart physics mode.",
    )

    parser.add_argument(
        "--direction-window",
        type=int,
        default=20,
        help="Recent frames before release used to estimate throw direction.",
    )

    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.5,
        help="Minimum MediaPipe landmark visibility used for side camera tracking.",
    )

    parser.add_argument(
        "--release-offset-frames",
        type=int,
        default=0,
        help="Move the detected release frame this many frames earlier.",
    )

    parser.add_argument(
        "--trajectory-y-offset-px",
        type=int,
        default=0,
        help="Move the rendered trajectory upward by this many pixels in the preview video.",
    )

    parser.add_argument(
        "--endpoint-margin-px",
        type=int,
        default=10,
        help="Fix the rendered 2m endpoint this many pixels before the right edge.",
    )

    parser.add_argument(
        "--target-config",
        type=Path,
        help="Optional JSON config with 5 pixel target centers and hit radius.",
    )

    parser.add_argument(
        "--target-mirror-x",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror the final pixel target endpoint across the vertical center line.",
    )

    parser.add_argument(
        "--track-object",
        action="store_true",
        help="Track a colored projectile after release and use it to correct trajectory.",
    )

    parser.add_argument(
        "--object-method",
        choices=["color", "flow"],
        default="flow",
        help="Projectile tracking method when --track-object is enabled.",
    )

    parser.add_argument(
        "--object-color",
        choices=sorted(COLOR_RANGES),
        default="green",
        help="Projectile color to track when --track-object is enabled.",
    )

    parser.add_argument(
        "--object-min-area",
        type=float,
        default=20,
        help="Minimum contour area for projectile tracking.",
    )

    parser.add_argument(
        "--object-max-frames",
        type=int,
        default=40,
        help="Maximum frames to scan after release for projectile tracking.",
    )

    parser.add_argument(
        "--object-min-motion-px",
        type=float,
        default=4.0,
        help="Minimum optical-flow motion in pixels.",
    )

    parser.add_argument(
        "--object-release-lead-frames",
        type=int,
        default=3,
        help="Move release marker this many frames before first tracked object frame.",
    )

    return parser.parse_args()


def run_analysis(
    video_path,
    hand,
    motion_point,
    start_mode,
    flip_horizontal,
    front_video,
    front_flip_horizontal,
    front_direction_window,
    front_frame_offset,
    front_horizontal_gain,
    board_distance,
    physics_mode,
    dart_speed_mps,
    direction_window,
    min_visibility,
    release_offset_frames,
    trajectory_y_offset_px,
    endpoint_margin_px,
    target_config,
    target_mirror_x,
    track_object_enabled,
    object_method,
    object_color,
    object_min_area,
    object_max_frames,
    object_min_motion_px,
    object_release_lead_frames,
):
    run_name = build_run_name(video_path, flip_horizontal)

    front_run_name = None

    if front_video:
        front_run_name = build_run_name(front_video, front_flip_horizontal)

    output_dir = Path("output") / run_name

    landmarks_csv = output_dir / f"{run_name}_landmarks.csv"

    front_landmarks_csv = (
        output_dir / f"{front_run_name}_landmarks.csv"
        if front_run_name
        else None
    )

    front_pose_preview = (
        output_dir / f"{front_run_name}_pose_preview.mp4"
        if front_run_name
        else None
    )

    front_direction_png = (
        output_dir / f"{front_run_name}_direction.png"
        if front_run_name
        else None
    )

    pose_preview = output_dir / f"{run_name}_pose_preview.mp4"
    analysis_csv = output_dir / f"{run_name}_analysis.csv"
    trajectory_csv = output_dir / f"{run_name}_trajectory.csv"
    object_track_csv = output_dir / f"{run_name}_object_track.csv"
    trajectory_png = output_dir / f"{run_name}_trajectory.png"
    board_png = output_dir / f"{run_name}_board.png"
    pixel_target_png = output_dir / f"{run_name}_pixel_targets.png"
    pixel_target_csv = output_dir / f"{run_name}_pixel_targets.csv"
    analysis_preview = output_dir / f"{run_name}_analysis_preview.mp4"
    release_frame_png = output_dir / f"{run_name}_release_frame.png"
    grid_trajectory_png = output_dir / f"{run_name}_grid_trajectory.png"

    board_w = 16
    board_h = 16

    print("실행 설정")
    print(f"Video: {video_path}")
    print(f"Hand: {hand}")
    print(f"Motion point: {motion_point}")
    print(f"Start mode: {start_mode}")
    print(f"Flip horizontal: {flip_horizontal}")
    print(f"Front video: {front_video}")
    print(f"Front flip horizontal: {front_flip_horizontal}")
    print(f"Board distance: {board_distance}m")
    print(f"Physics mode: {physics_mode}")
    print(f"Dart speed: {dart_speed_mps}m/s")
    print(f"Direction window: {direction_window}")
    print(f"Min visibility: {min_visibility}")
    print(f"Release offset frames: {release_offset_frames}")
    print(f"Trajectory Y offset: {trajectory_y_offset_px}px")
    print(f"Endpoint margin: {endpoint_margin_px}px")
    print(f"Target config: {target_config}")
    print(f"Target mirror x: {target_mirror_x}")
    print(f"Front direction window: {front_direction_window}")
    print(f"Front frame offset: {front_frame_offset}")
    print(f"Front horizontal gain: {front_horizontal_gain}")
    print(f"Track object: {track_object_enabled}")
    print(f"Object method: {object_method}")
    print(f"Output folder: {output_dir}")

    print("\n1. 관절 및 손가락 좌표 추출 중...")
    extract_pose(
        video_path=video_path,
        output_csv=landmarks_csv,
        preview_path=pose_preview,
        flip_horizontal=flip_horizontal,
    )

    print("\n2. 투척 동작 분석 중...")
    analyze(
        csv_path=landmarks_csv,
        hand=hand,
        motion_point=motion_point,
        start_mode=start_mode,
        output_csv=analysis_csv,
        window=5,
        lookback=10,
        direction_window=direction_window,
        min_visibility=min_visibility,
        board_w=board_w,
        board_h=board_h,
        sensitivity=0.35,
        release_offset_frames=release_offset_frames,
    )

    if front_video:
        print("\n2-1. 정면 카메라 좌우 방향 보정 중...")
        extract_pose(
            video_path=front_video,
            output_csv=front_landmarks_csv,
            preview_path=front_pose_preview,
            flip_horizontal=front_flip_horizontal,
        )

        apply_front_camera_direction(
            side_analysis_csv=analysis_csv,
            front_landmarks_csv=front_landmarks_csv,
            hand=hand,
            motion_point=motion_point,
            output_csv=analysis_csv,
            direction_window=front_direction_window,
            horizontal_gain=front_horizontal_gain,
            frame_offset=front_frame_offset,
            front_video=front_video,
            front_direction_image=front_direction_png,
            front_flip_horizontal=front_flip_horizontal,
        )

    if track_object_enabled:
        print("\n2-2. 릴리즈 이후 물체 추적 중...")
        track_object(
            video_path=video_path,
            analysis_csv=analysis_csv,
            output_csv=object_track_csv,
            method=object_method,
            color=object_color,
            min_area=object_min_area,
            max_frames_after_release=object_max_frames,
            flip_horizontal=flip_horizontal,
            min_motion_px=object_min_motion_px,
        )

        correction = correct_release_from_object_track(
            analysis_csv=analysis_csv,
            object_track_csv=object_track_csv,
            lead_frames=object_release_lead_frames,
        )

        if correction:
            print(
                "Release corrected from object track: "
                f"{correction['corrected_release_frame']} "
                f"(first object frame: {correction['first_object_frame']})"
            )
    else:
        object_track_csv = None

    print("\n2-3. 릴리즈 시점 이미지 저장 중...")
    render_release_frame_image(
        video_path=video_path,
        analysis_csv=analysis_csv,
        output_image=release_frame_png,
        hand=hand,
        flip_horizontal=flip_horizontal,
    )

    print("\n3. 가상 다트 궤적 예측 중...")
    screen_endpoint_x = normalized_screen_endpoint_x(
        video_path,
        endpoint_margin_px
    )

    predict(
        analysis_csv=analysis_csv,
        hand=hand,
        output_csv=trajectory_csv,
        plot_path=trajectory_png,
        gravity=0.25,
        duration=0.8,
        steps=30,
        velocity_scale=0.8,
        board_w=board_w,
        board_h=board_h,
        board_scale=20.0,
        board_distance=board_distance,
        speed_to_mps=2.5,
        min_duration=0.25,
        max_duration=1.2,
        physics_mode=physics_mode,
        dart_speed_mps=dart_speed_mps,
        board_width_m=0.6,
        board_height_m=0.6,
        gravity_mps2=9.81,
        max_horizontal_angle_deg=15.0,
        max_vertical_angle_deg=15.0,
        object_track_csv=object_track_csv,
        min_object_points=3,
        screen_endpoint_x=screen_endpoint_x,
    )

    print("\n4. 궤적 유효성 검사 중...")
    trajectory_quality = evaluate_trajectory_quality(
        trajectory_csv=trajectory_csv,
        expected_endpoint_x=screen_endpoint_x,
    )

    print(trajectory_quality["trajectory_quality_message"])

    print("\n5. 가상 보드 결과 이미지 생성 중...")
    hit_x, hit_y = read_hit_position(
        trajectory_csv
    )

    render_board(
        hit_x=hit_x,
        hit_y=hit_y,
        board_w=board_w,
        board_h=board_h,
        output_png=board_png,
    )

    print("\n6. 픽셀 과녁 결과 생성 중...")
    evaluate_pixel_targets(
        trajectory_csv=trajectory_csv,
        analysis_csv=analysis_csv,
        output_png=pixel_target_png,
        output_csv=pixel_target_csv,
        width=1920,
        height=1080,
        config_path=target_config,
        mirror_x=target_mirror_x,
    )

    print("\n7. 분석 미리보기 영상 생성 중...")
    render_preview(
        video_path=video_path,
        analysis_csv=analysis_csv,
        trajectory_csv=trajectory_csv,
        output_video=analysis_preview,
        hand=hand,
        flip_horizontal=flip_horizontal,
        trajectory_y_offset_px=trajectory_y_offset_px,
    )

    print("\n8. 1920x1080 격자 궤적 이미지 생성 중...")
    render_grid_trajectory(
        trajectory_csv=trajectory_csv,
        output_image=grid_trajectory_png,
        width=1920,
        height=1080,
        grid_size_px=30,
        trajectory_y_offset_px=trajectory_y_offset_px,
    )

    print("\n전체 실행 완료")
    print(f"명중 위치: ({hit_x}, {hit_y})")
    print(f"결과 폴더: {output_dir}")
    print(f"좌표 CSV: {landmarks_csv}")
    print(f"분석 CSV: {analysis_csv}")
    print(f"궤적 이미지: {trajectory_png}")
    print(f"픽셀 과녁 이미지: {pixel_target_png}")
    print(f"픽셀 과녁 CSV: {pixel_target_csv}")
    print(f"분석 영상: {analysis_preview}")
    print(f"릴리즈 이미지: {release_frame_png}")
    print(f"격자 궤적 이미지: {grid_trajectory_png}")

    write_latest_result(
        run_name=run_name,
        video_path=video_path,
        hit_x=hit_x,
        hit_y=hit_y,
        trajectory_png=trajectory_png,
        board_png=board_png,
        pixel_target_png=pixel_target_png,
        pixel_target_csv=pixel_target_csv,
        analysis_preview=analysis_preview,
        grid_trajectory_png=grid_trajectory_png,
        trajectory_quality=trajectory_quality,
    )

    if trajectory_quality["trajectory_valid"]:
        print("\n9. MQTT LED 결과 전송 중...")

        try:
            publish_led_result()

        except Exception as exc:
            print(
                f"[MQTT LED WARNING] LED MQTT publish failed: {exc}",
                file=sys.stderr
            )

    else:
        print("\n9. 비정상 궤적이므로 MQTT LED 결과 전송을 건너뜁니다.")


def main():
    args = parse_args()

    try:
        if args.adb_capture:
            video_path = capture_video(
                args.config
            )
        else:
            video_path = (
                args.video
                or args.input_video
                or find_latest_video(Path("videos"))
            )

        run_analysis(
            video_path=video_path,
            hand=args.hand,
            motion_point=args.motion_point,
            start_mode=args.start_mode,
            flip_horizontal=args.flip_horizontal,
            front_video=args.front_video,
            front_flip_horizontal=args.front_flip_horizontal,
            front_direction_window=args.front_direction_window,
            front_frame_offset=args.front_frame_offset,
            front_horizontal_gain=args.front_horizontal_gain,
            board_distance=args.board_distance,
            physics_mode=args.physics_mode,
            dart_speed_mps=args.dart_speed_mps,
            direction_window=args.direction_window,
            min_visibility=args.min_visibility,
            release_offset_frames=args.release_offset_frames,
            trajectory_y_offset_px=args.trajectory_y_offset_px,
            endpoint_margin_px=args.endpoint_margin_px,
            target_config=args.target_config,
            target_mirror_x=args.target_mirror_x,
            track_object_enabled=args.track_object,
            object_method=args.object_method,
            object_color=args.object_color,
            object_min_area=args.object_min_area,
            object_max_frames=args.object_max_frames,
            object_min_motion_px=args.object_min_motion_px,
            object_release_lead_frames=args.object_release_lead_frames,
        )

    except AdbCaptureError as exc:
        print(f"[ADB 오류] {exc}", file=sys.stderr)
        return 1

    return 0
#안녕

if __name__ == "__main__":
    raise SystemExit(main())