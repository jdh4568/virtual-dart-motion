import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from src.adb_capture import AdbCaptureError, capture_video
from src.analyze_throw import analyze
from src.calibrate_trajectory import calibrate
from src.extract_landmarks import extract_pose
from src.object_tracking import track_object
from src.render_analysis_preview import render_preview
from src.simulate_board import read_hit_position, render_board
from src.trajectory import predict


DEFAULTS = {
    "hand": "right",
    "motion_point": "auto",
    "start_mode": "initial",
    "start_window": 60,
    "release_offset_frames": 0,
    "board_distance": 2.0,
    "gravity": 0.25,
    "velocity_scale": 0.8,
    "board_scale": 20.0,
    "speed_to_mps": 2.5,
    "min_duration": 0.25,
    "max_duration": 1.2,
}


def find_latest_video(videos_dir):
    videos = []
    for pattern in ["*.mp4", "*.MP4", "*.mov", "*.MOV"]:
        videos.extend(videos_dir.glob(pattern))

    if not videos:
        raise FileNotFoundError(f"No video file found in {videos_dir}")

    return max(videos, key=lambda path: path.stat().st_mtime)


def build_run_name(video_path, flip_horizontal):
    run_name = video_path.stem
    if flip_horizontal:
        run_name = f"{run_name}_flipped"
    return run_name


def load_calibration(path):
    if not path:
        return {}

    calibration_path = Path(path)
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    with calibration_path.open("r", encoding="utf-8") as calibration_file:
        calibration = json.load(calibration_file)

    if not isinstance(calibration, dict):
        raise ValueError(f"Calibration file must contain a JSON object: {calibration_path}")

    return calibration


def choose_value(name, cli_value, calibration):
    if cli_value is not None:
        return cli_value
    if name in calibration:
        return calibration[name]
    return DEFAULTS[name]


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
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Calibration JSON created by src/calibrate_trajectory.py.",
    )
    parser.add_argument("--hand", choices=["right", "left"], default=DEFAULTS["hand"])
    parser.add_argument(
        "--motion-point",
        choices=["auto", "wrist", "thumb_tip", "index_tip", "middle_tip"],
        default=DEFAULTS["motion_point"],
    )
    parser.add_argument("--flip-horizontal", action="store_true")
    parser.add_argument("--start-mode", choices=["initial", "lookback"], default=DEFAULTS["start_mode"])
    parser.add_argument("--start-window", type=int, default=None)
    parser.add_argument("--release-offset-frames", type=int, default=None)
    parser.add_argument("--board-distance", type=float, default=None)
    parser.add_argument("--gravity", type=float, default=None)
    parser.add_argument("--velocity-scale", type=float, default=None)
    parser.add_argument("--board-scale", type=float, default=None)
    parser.add_argument("--speed-to-mps", type=float, default=None)
    return parser.parse_args()


def default_runtime_args(**overrides):
    values = {
        "hand": DEFAULTS["hand"],
        "motion_point": DEFAULTS["motion_point"],
        "flip_horizontal": False,
        "start_mode": DEFAULTS["start_mode"],
        "start_window": None,
        "release_offset_frames": None,
        "board_distance": None,
        "gravity": None,
        "velocity_scale": None,
        "board_scale": None,
        "speed_to_mps": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def run_analysis(video_path, args, calibration):
    hand = args.hand
    motion_point = args.motion_point
    flip_horizontal = args.flip_horizontal
    start_mode = args.start_mode
    start_window = int(choose_value("start_window", args.start_window, calibration))
    release_offset_frames = int(
        choose_value("release_offset_frames", args.release_offset_frames, calibration)
    )
    board_distance = float(choose_value("board_distance", args.board_distance, calibration))
    gravity = float(choose_value("gravity", args.gravity, calibration))
    velocity_scale = float(choose_value("velocity_scale", args.velocity_scale, calibration))
    board_scale = float(choose_value("board_scale", args.board_scale, calibration))
    speed_to_mps = float(choose_value("speed_to_mps", args.speed_to_mps, calibration))


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
    print(f"Start mode: {start_mode}")
    print(f"Start window: {start_window} frames")
    print(f"Release offset frames: {release_offset_frames}")
    print(f"Flip horizontal: {flip_horizontal}")
    print(f"Board distance: {board_distance}m")
    print(f"Velocity scale: {velocity_scale}")
    print(f"Gravity: {gravity}")
    print(f"Board scale: {board_scale}")
    print(f"Speed to m/s: {speed_to_mps}")
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
        start_mode=start_mode,
        start_window=start_window,
        release_offset_frames=release_offset_frames,
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
        gravity=gravity,
        duration=0.8,
        steps=30,
        velocity_scale=velocity_scale,
        board_w=board_w,
        board_h=board_h,
        board_scale=board_scale,
        board_distance=board_distance,
        speed_to_mps=speed_to_mps,
        min_duration=DEFAULTS["min_duration"],
        max_duration=DEFAULTS["max_duration"],
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

    return {
        "run_name": run_name,
        "output_dir": output_dir,
        "landmarks_csv": landmarks_csv,
        "analysis_csv": analysis_csv,
        "trajectory_csv": trajectory_csv,
        "trajectory_png": trajectory_png,
        "board_png": board_png,
        "analysis_preview": analysis_preview,
    }


def run_calibration_flow(config_path=Path("adb_config.json"), calibration_path=Path("calibration.json")):
    print("\n[캘리브레이션] 실제 연두색 물체를 던지는 영상으로 보정값을 생성합니다.")
    video_path = capture_video(config_path)
    args = default_runtime_args()
    outputs = run_analysis(video_path, args, calibration={})

    object_csv = outputs["output_dir"] / f"{outputs['run_name']}_object_trajectory.csv"
    print("\n[캘리브레이션] 연두색 물체 궤적 추적 중...")
    track_object(
        video_path=video_path,
        output_csv=object_csv,
        mode="lime",
    )

    print("\n[캘리브레이션] 예측 궤적과 물체 궤적 비교 중...")
    calibrate(
        analysis_csv=outputs["analysis_csv"],
        object_csv=object_csv,
        output_json=calibration_path,
        hand=args.hand,
        board_distance=DEFAULTS["board_distance"],
    )
    print(f"\n[캘리브레이션 완료] 보정 파일: {calibration_path}")


def run_game_flow(config_path=Path("adb_config.json"), calibration_path=Path("calibration.json")):
    print("\n[게임] 무물체 가상 다트 실행")
    calibration = load_calibration(calibration_path) if calibration_path.exists() else {}
    if calibration:
        print(f"[게임] 보정 파일 적용: {calibration_path}")
    else:
        print(f"[게임] 보정 파일 없음. 기본 파라미터로 실행합니다: {calibration_path}")

    video_path = capture_video(config_path)
    args = default_runtime_args()
    run_analysis(video_path, args, calibration)


def run_interactive_menu():
    print("Virtual Dart Motion")
    print("1. 캘리브레이션")
    print("2. 시뮬레이션")
    print("0. 종료")
    choice = input("번호를 입력하세요: ").strip()

    if choice == "1":
        run_calibration_flow()
        return 0
    if choice == "2":
        run_game_flow()
        return 0
    if choice == "0":
        print("종료합니다.")
        return 0

    print("잘못된 입력입니다. 1, 2, 0 중 하나를 입력하세요.", file=sys.stderr)
    return 1


def main():
    if len(sys.argv) == 1:
        try:
            return run_interactive_menu()
        except AdbCaptureError as exc:
            print(f"[ADB 오류] {exc}", file=sys.stderr)
            return 1

    args = parse_args()

    try:
        calibration = load_calibration(args.calibration)
        if args.adb_capture:
            video_path = capture_video(args.config)
        else:
            video_path = args.video or args.input_video or find_latest_video(Path("videos"))

        run_analysis(video_path, args, calibration)
    except AdbCaptureError as exc:
        print(f"[ADB 오류] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
