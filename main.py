import argparse
import sys
from pathlib import Path

from src.adb_capture import AdbCaptureError, capture_video
from src.extract_landmarks import extract_pose
from src.analyze_throw import analyze
from src.trajectory import predict
from src.simulate_board import read_hit_position, render_board
from src.render_analysis_preview import render_preview


def run_analysis(video_path):
    hand = "right"
    flip_horizontal = False

    output_dir = Path("output")
    landmarks_csv = output_dir / "landmarks.csv"   
    pose_preview = output_dir / "pose_preview.mp4"
    analysis_csv = output_dir / "throw_analysis.csv"
    trajectory_csv = output_dir / "trajectory.csv"
    trajectory_png = output_dir / "trajectory.png"
    board_png = output_dir / "board_result.png"
    analysis_preview = output_dir / "analysis_preview.mp4"

    board_w = 16
    board_h = 16

    print("1. 관절 좌표 추출 중...")
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


def main():
    parser = argparse.ArgumentParser(description="Run virtual dart motion analysis.")
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("videos/dart9.mp4"),
        help="Input video path. Used unless --adb-capture is set.",
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
    args = parser.parse_args()

    try:
        video_path = capture_video(args.config) if args.adb_capture else args.video
    except AdbCaptureError as exc:
        print(f"[ADB 오류] {exc}", file=sys.stderr)
        return 1

    run_analysis(video_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
