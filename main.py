import argparse
import sys
from pathlib import Path

from src.adb_capture import AdbCaptureError, capture_video
from src.extract_landmarks import extract_pose
from src.analyze_throw import analyze
from src.trajectory import predict
from src.simulate_board import read_hit_position, render_board
from src.render_analysis_preview import render_preview


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


def parse_args():
    parser = argparse.ArgumentParser(description="Run virtual dart motion analysis.")
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
        "--flip-horizontal",
        action="store_true",
        help="Flip mirrored/selfie videos before analysis.",
    )
    parser.add_argument(
        "--board-distance",
        type=float,
        default=2.0,
        help="Fixed distance from thrower to virtual board in meters.",
    )
    return parser.parse_args()


def run_analysis(video_path, hand, motion_point, flip_horizontal, board_distance):
    run_name = build_run_name(video_path, flip_horizontal)

    output_dir = Path("output") / run_name
    landmarks_csv = output_dir / f"{run_name}_landmarks.csv"
    pose_preview = output_dir / f"{run_name}_pose_preview.mp4"
    analysis_csv = output_dir / f"{run_name}_analysis.csv"
    trajectory_csv = output_dir / f"{run_name}_trajectory.csv"
    trajectory_png = output_dir / f"{run_name}_trajectory.png"
    board_png = output_dir / f"{run_name}_board.png"
    analysis_preview = output_dir / f"{run_name}_analysis_preview.mp4"

    board_w = 16
    board_h = 16

    print("실행 설정")
    print(f"Video: {video_path}")
    print(f"Hand: {hand}")
    print(f"Motion point: {motion_point}")
    print(f"Flip horizontal: {flip_horizontal}")
    print(f"Board distance: {board_distance}m")
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
        output_csv=analysis_csv,
        window=5,
        lookback=10,
        direction_window=4,
        min_visibility=0.5,
        board_w=board_w,
        board_h=board_h,
        sensitivity=0.35,
    )

    print("\n3. 가상 다트 궤적 예측 중...")
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
    )

    print("\n4. 가상 보드 결과 이미지 생성 중...")
    hit_x, hit_y = read_hit_position(trajectory_csv)
    render_board(
        hit_x=hit_x,
        hit_y=hit_y,
        board_w=board_w,
        board_h=board_h,
        output_png=board_png,
    )

    print("\n5. 분석 미리보기 영상 생성 중...")
    render_preview(
        video_path=video_path,
        analysis_csv=analysis_csv,
        trajectory_csv=trajectory_csv,
        output_video=analysis_preview,
        hand=hand,
        flip_horizontal=flip_horizontal,
    )

    print("\n전체 실행 완료")
    print(f"명중 위치: ({hit_x}, {hit_y})")
    print(f"결과 폴더: {output_dir}")
    print(f"좌표 CSV: {landmarks_csv}")
    print(f"분석 CSV: {analysis_csv}")
    print(f"궤적 이미지: {trajectory_png}")
    print(f"분석 영상: {analysis_preview}")


def main():
    args = parse_args()

    try:
        if args.adb_capture:
            video_path = capture_video(args.config)
        else:
            video_path = args.video or args.input_video or find_latest_video(Path("videos"))

        run_analysis(
            video_path=video_path,
            hand=args.hand,
            motion_point=args.motion_point,
            flip_horizontal=args.flip_horizontal,
            board_distance=args.board_distance,
        )
    except AdbCaptureError as exc:
        print(f"[ADB 오류] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
