import argparse
import concurrent.futures
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from main import run_analysis
from src.adb_capture import (
    AdbCaptureError,
    find_latest_mp4,
    load_config,
    play_beep,
    run_adb,
    tap_shutter,
)
from src.object_tracker import COLOR_RANGES


def authorized_devices():
    result = run_adb(["devices"])
    devices = []
    unauthorized = []

    for line in result.stdout.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        serial, state = parts[0], parts[1]
        if state == "device":
            devices.append(serial)
        elif state == "unauthorized":
            unauthorized.append(serial)

    if unauthorized:
        raise AdbCaptureError(
            "ADB device is unauthorized. Approve USB debugging on every camera phone."
        )

    return devices


def resolve_camera_serials(side_serial, front_serial):
    devices = authorized_devices()

    if side_serial and side_serial not in devices:
        raise AdbCaptureError(f"Side camera serial is not connected: {side_serial}")
    if front_serial and front_serial not in devices:
        raise AdbCaptureError(f"Front camera serial is not connected: {front_serial}")
    if side_serial and front_serial and side_serial == front_serial:
        raise AdbCaptureError("Side and front camera serials must be different.")

    if side_serial and front_serial:
        return side_serial, front_serial

    if len(devices) != 2:
        raise AdbCaptureError(
            "Connect exactly two authorized ADB devices, or pass --side-serial and --front-serial."
        )

    missing = [serial for serial in devices if serial not in {side_serial, front_serial}]
    if side_serial:
        return side_serial, missing[0]
    if front_serial:
        return missing[0], front_serial

    side_serial, front_serial = sorted(devices)
    print("[ADB] --side-serial/--front-serial이 없어 정렬된 장치 순서로 배정합니다.")
    print(f"[ADB] Side camera: {side_serial}")
    print(f"[ADB] Front camera: {front_serial}")
    return side_serial, front_serial


def safe_serial(serial):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", serial)


def load_double_config(config_path):
    path = Path(config_path)
    if not path.exists():
        return {}, {}

    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise AdbCaptureError(f"Double camera config must be a JSON object: {path}")

    side_config = config.get("side_camera", config.get("side", {}))
    front_config = config.get("front_camera", config.get("front", {}))
    if not isinstance(side_config, dict) or not isinstance(front_config, dict):
        raise AdbCaptureError(
            "Double camera config must contain side_camera and front_camera objects."
        )

    return side_config, front_config


def merge_camera_config(base_config_path, camera_config):
    config_path = camera_config.get("config")
    config = load_config(config_path or base_config_path)

    for key, value in camera_config.items():
        if key not in {"serial", "config", "model", "role", "note"}:
            config[key] = value

    return config


def tap_both(side_serial, front_serial, side_config, front_config):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                tap_shutter,
                side_serial,
                int(side_config["shutter_x"]),
                int(side_config["shutter_y"]),
            ),
            executor.submit(
                tap_shutter,
                front_serial,
                int(front_config["shutter_x"]),
                int(front_config["shutter_y"]),
            ),
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def pull_labeled_video(serial, remote_file, local_video_dir, label, started_at):
    local_dir = Path(local_video_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    local_path = local_dir / f"{label}_capture_{timestamp}_{safe_serial(serial)}.mp4"
    run_adb(["-s", serial, "pull", remote_file, str(local_path)])

    if not local_path.exists():
        raise AdbCaptureError(f"adb pull finished, but local file was not created: {local_path}")

    return local_path


def capture_double_video(
    side_serial,
    front_serial,
    side_config,
    front_config,
):
    record_seconds = max(
        float(side_config["record_seconds"]),
        float(front_config["record_seconds"]),
    )
    save_wait_seconds = max(
        float(side_config["save_wait_seconds"]),
        float(front_config["save_wait_seconds"]),
    )
    beep_enabled = bool(side_config["beep_enabled"]) or bool(front_config["beep_enabled"])

    print(f"[ADB] Side camera device: {side_serial}")
    print(f"[ADB] Front camera device: {front_serial}")
    print("[ADB] 2초 뒤 두 카메라 녹화를 동시에 시작합니다.")
    time.sleep(2)

    if beep_enabled:
        play_beep()

    started_at = datetime.now()
    print("[ADB] 두 카메라 녹화 시작")
    tap_both(side_serial, front_serial, side_config, front_config)

    print(f"[ADB] {record_seconds:.1f}초 동안 녹화 중")
    time.sleep(record_seconds)

    print("[ADB] 두 카메라 녹화 종료")
    tap_both(side_serial, front_serial, side_config, front_config)

    if beep_enabled:
        play_beep()

    print(f"[ADB] 파일 저장 대기: {save_wait_seconds:.1f}초")
    time.sleep(save_wait_seconds)

    print("[ADB] 최신 mp4 검색")
    side_remote_file = find_latest_mp4(side_serial, str(side_config["remote_camera_dir"]))
    front_remote_file = find_latest_mp4(front_serial, str(front_config["remote_camera_dir"]))
    print(f"[ADB] Side video: {side_remote_file}")
    print(f"[ADB] Front video: {front_remote_file}")

    print("[ADB] 영상 파일 전송")
    side_video = pull_labeled_video(
        side_serial,
        side_remote_file,
        str(side_config["local_video_dir"]),
        "side",
        started_at,
    )
    front_video = pull_labeled_video(
        front_serial,
        front_remote_file,
        str(front_config["local_video_dir"]),
        "front",
        started_at,
    )
    print(f"[ADB] Side saved: {side_video}")
    print(f"[ADB] Front saved: {front_video}")

    return side_video, front_video


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture front and side cameras together, then run trajectory analysis."
    )
    parser.add_argument("--side-video", type=Path, help="Existing side camera video.")
    parser.add_argument("--front-video", type=Path, help="Existing front camera video.")
    parser.add_argument("--side-serial", help="ADB serial for the side camera phone.")
    parser.add_argument("--front-serial", help="ADB serial for the front camera phone.")
    parser.add_argument(
        "--double-config",
        type=Path,
        default=Path("double_adb_config.json"),
        help="Double camera ADB config with side/front serials and phone-specific settings.",
    )
    parser.add_argument(
        "--side-config",
        type=Path,
        default=Path("adb_config.json"),
        help="ADB capture config for the side camera.",
    )
    parser.add_argument(
        "--front-config",
        type=Path,
        help="ADB capture config for the front camera. Defaults to --side-config.",
    )
    parser.add_argument("--hand", choices=["right", "left"], default="right")
    parser.add_argument(
        "--motion-point",
        choices=["auto", "wrist", "thumb_tip", "index_tip", "middle_tip"],
        default="auto",
    )
    parser.add_argument(
        "--start-mode",
        choices=["recent", "video-start"],
        default="video-start",
    )
    parser.add_argument("--flip-horizontal", action="store_true")
    parser.add_argument("--front-flip-horizontal", action="store_true")
    parser.add_argument("--front-direction-window", type=int, default=20)
    parser.add_argument("--front-frame-offset", type=int, default=5)
    parser.add_argument("--front-horizontal-gain", type=float, default=1.0)
    parser.add_argument("--board-distance", type=float, default=2.0)
    parser.add_argument(
        "--physics-mode",
        choices=["simple", "dart", "extend"],
        default="extend",
    )
    parser.add_argument("--dart-speed-mps", type=float, default=8.0)
    parser.add_argument("--direction-window", type=int, default=20)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--release-offset-frames", type=int, default=0)
    parser.add_argument("--trajectory-y-offset-px", type=int, default=0)
    parser.add_argument("--endpoint-margin-px", type=int, default=10)
    parser.add_argument(
        "--target-config",
        type=Path,
        help="Optional JSON config with 5 pixel target centers and hit radius.",
    )
    parser.add_argument("--target-mirror-x", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--track-object", action="store_true")
    parser.add_argument(
        "--object-method",
        choices=["color", "flow"],
        default="flow",
    )
    parser.add_argument(
        "--object-color",
        choices=sorted(COLOR_RANGES),
        default="green",
    )
    parser.add_argument("--object-min-area", type=float, default=20)
    parser.add_argument("--object-max-frames", type=int, default=40)
    parser.add_argument("--object-min-motion-px", type=float, default=4.0)
    parser.add_argument("--object-release-lead-frames", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        side_camera_config, front_camera_config = load_double_config(args.double_config)

        if args.side_video or args.front_video:
            if not args.side_video or not args.front_video:
                raise AdbCaptureError("--side-video and --front-video must be used together.")
            side_video = args.side_video
            front_video = args.front_video
        else:
            side_serial_arg = args.side_serial or side_camera_config.get("serial")
            front_serial_arg = args.front_serial or front_camera_config.get("serial")
            side_config = merge_camera_config(args.side_config, side_camera_config)
            front_config = merge_camera_config(
                args.front_config or args.side_config,
                front_camera_config,
            )
            side_serial, front_serial = resolve_camera_serials(
                side_serial_arg,
                front_serial_arg,
            )
            side_video, front_video = capture_double_video(
                side_serial=side_serial,
                front_serial=front_serial,
                side_config=side_config,
                front_config=front_config,
            )

        run_analysis(
            video_path=side_video,
            hand=args.hand,
            motion_point=args.motion_point,
            start_mode=args.start_mode,
            flip_horizontal=args.flip_horizontal,
            front_video=front_video,
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


if __name__ == "__main__":
    raise SystemExit(main())
