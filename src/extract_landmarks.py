import argparse
import csv
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import cv2
import mediapipe as mp


POSE_POINTS = {
    "right_shoulder": mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    "right_elbow": mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    "right_wrist": mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    "left_shoulder": mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    "left_elbow": mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    "left_wrist": mp.solutions.pose.PoseLandmark.LEFT_WRIST,
}

HAND_POINTS = {
    "thumb_tip": mp.solutions.hands.HandLandmark.THUMB_TIP,
    "index_tip": mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
    "middle_tip": mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
}


def video_writer_fps(fps):
    if not fps or fps <= 0:
        return 30
    return max(1, int(round(fps)))


def empty_point(row, name):
    row[f"{name}_x"] = ""
    row[f"{name}_y"] = ""
    row[f"{name}_z"] = ""
    row[f"{name}_visibility"] = ""


def write_point(row, name, point, visibility=1.0):
    row[f"{name}_x"] = point.x
    row[f"{name}_y"] = point.y
    row[f"{name}_z"] = point.z
    row[f"{name}_visibility"] = visibility


def assign_hands_to_pose_sides(pose_landmarks, hand_landmarks):
    assigned = {"right": None, "left": None}
    if pose_landmarks is None or not hand_landmarks:
        return assigned

    pose_wrists = {
        "right": pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value],
        "left": pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value],
    }

    candidates = []
    for hand in hand_landmarks:
        wrist = hand.landmark[mp.solutions.hands.HandLandmark.WRIST.value]
        for side, pose_wrist in pose_wrists.items():
            distance = ((wrist.x - pose_wrist.x) ** 2 + (wrist.y - pose_wrist.y) ** 2) ** 0.5
            candidates.append((distance, side, hand))

    used_hands = set()
    for _, side, hand in sorted(candidates, key=lambda item: item[0]):
        hand_id = id(hand)
        if assigned[side] is None and hand_id not in used_hands:
            assigned[side] = hand
            used_hands.add(hand_id)

    return assigned


def landmark_row(frame_index, time_sec, pose_landmarks, hand_sides):
    row = {
        "frame_index": frame_index,
        "time_sec": time_sec,
        "pose_detected": pose_landmarks is not None,
        "right_hand_detected": hand_sides["right"] is not None,
        "left_hand_detected": hand_sides["left"] is not None,
    }

    for name, landmark_id in POSE_POINTS.items():
        if pose_landmarks is None:
            empty_point(row, name)
            continue

        point = pose_landmarks[landmark_id.value]
        write_point(row, name, point, point.visibility)

    for side in ["right", "left"]:
        hand = hand_sides[side]
        for point_name, landmark_id in HAND_POINTS.items():
            column_name = f"{side}_{point_name}"
            if hand is None:
                empty_point(row, column_name)
                continue
            write_point(row, column_name, hand.landmark[landmark_id.value])

    return row


def extract_pose(video_path, output_csv, preview_path=None, flip_horizontal=False):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fieldnames = [
        "frame_index",
        "time_sec",
        "pose_detected",
        "right_hand_detected",
        "left_hand_detected",
    ]
    for name in POSE_POINTS:
        fieldnames.extend(
            [
                f"{name}_x",
                f"{name}_y",
                f"{name}_z",
                f"{name}_visibility",
            ]
        )
    for side in ["right", "left"]:
        for point_name in HAND_POINTS:
            name = f"{side}_{point_name}"
            fieldnames.extend(
                [
                    f"{name}_x",
                    f"{name}_y",
                    f"{name}_z",
                    f"{name}_visibility",
                ]
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    preview_writer = None

    if preview_path:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        preview_writer = cv2.VideoWriter(
            str(preview_path),
            fourcc,
            video_writer_fps(fps),
            (width, height),
        )

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    drawing = mp.solutions.drawing_utils

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands, output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        frame_index = 0
        pose_detected_count = 0
        right_hand_detected_count = 0
        left_hand_detected_count = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if flip_horizontal:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = pose.process(rgb)
            hand_result = hands.process(rgb)

            pose_landmarks = None
            if pose_result.pose_landmarks:
                pose_landmarks = pose_result.pose_landmarks.landmark
                pose_detected_count += 1

            hand_sides = assign_hands_to_pose_sides(
                pose_landmarks,
                hand_result.multi_hand_landmarks if hand_result.multi_hand_landmarks else [],
            )
            if hand_sides["right"] is not None:
                right_hand_detected_count += 1
            if hand_sides["left"] is not None:
                left_hand_detected_count += 1

            time_sec = frame_index / fps
            writer.writerow(landmark_row(frame_index, time_sec, pose_landmarks, hand_sides))

            if preview_writer:
                annotated = frame.copy()
                if pose_result.pose_landmarks:
                    drawing.draw_landmarks(
                        annotated,
                        pose_result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )
                if hand_result.multi_hand_landmarks:
                    for hand_landmarks in hand_result.multi_hand_landmarks:
                        drawing.draw_landmarks(
                            annotated,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                        )
                preview_writer.write(annotated)

            frame_index += 1

    cap.release()
    if preview_writer:
        preview_writer.release()

    print(f"Video: {video_path}")
    print(f"Flip horizontal: {flip_horizontal}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {frame_index}/{total_frames}")
    print(f"Pose detected: {pose_detected_count} frames")
    print(f"Right hand detected: {right_hand_detected_count} frames")
    print(f"Left hand detected: {left_hand_detected_count} frames")
    print(f"CSV saved: {output_csv}")
    if preview_path:
        print(f"Preview saved: {preview_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe pose landmarks from a video.")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/landmarks.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help="Optional annotated preview video path",
    )
    parser.add_argument(
        "--flip-horizontal",
        action="store_true",
        help="Flip mirrored/selfie videos before running MediaPipe",
    )
    args = parser.parse_args()

    extract_pose(args.video, args.out, args.preview, args.flip_horizontal)


if __name__ == "__main__":
    main()
