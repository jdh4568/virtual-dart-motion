from pathlib import Path
import json
import subprocess
import sys
import time

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request


app = FastAPI()

templates = Jinja2Templates(
    directory="app/templates"
)

app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static"
)

app.mount(
    "/output",
    StaticFiles(directory="output"),
    name="output"
)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


def preview_frames(video_path):

    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = 1.0 / max(1.0, fps)

    try:

        while capture.isOpened():

            ok, frame = capture.read()

            if not ok:
                break

            encoded, jpeg = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 82]
            )

            if not encoded:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )

            time.sleep(frame_delay)

    finally:

        capture.release()


@app.get("/preview/{run_name}")
def preview_video(run_name: str):

    safe_run_name = Path(run_name).name
    video_path = (
        Path("output")
        / safe_run_name
        / f"{safe_run_name}_analysis_preview.mp4"
    )

    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Preview video not found: {video_path}"
        )

    return StreamingResponse(
        preview_frames(video_path),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


def read_latest_result():

    result_path = Path("output") / "latest_result.json"

    if not result_path.exists():

        raise FileNotFoundError(
            "No latest_result.json found. double_camera.py must create output/latest_result.json."
        )

    with result_path.open("r", encoding="utf-8") as result_file:

        result = json.load(result_file)

    required_keys = {
        "success",
        "hit_x",
        "hit_y",
        "video_name",
        "trajectory_image",
        "board_image",
        "preview_video",
        "target",
        "endpoint_x_px",
        "endpoint_y_px",
    }

    missing_keys = required_keys - set(result.keys())

    if missing_keys:

        raise ValueError(
            f"latest_result.json is missing keys: {sorted(missing_keys)}"
        )

    return result


@app.get("/latest-result")
def latest_result():

    try:

        return read_latest_result()

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc)
        ) from exc


async def read_request_json(request: Request):

    try:

        body = await request.json()

        if body is None:

            return {}

        if not isinstance(body, dict):

            return {}

        return body

    except Exception:

        return {}


def get_float_value(data, key, default_value):

    value = data.get(key, default_value)

    try:

        return float(value)

    except (TypeError, ValueError):

        return float(default_value)


def get_int_value(data, key, default_value):

    value = data.get(key, default_value)

    try:

        return int(value)

    except (TypeError, ValueError):

        return int(default_value)


@app.post("/analyze")
async def analyze_video(request: Request):

    try:

        request_data = await read_request_json(request)

        dart_speed_mps = get_float_value(
            request_data,
            "dart_speed_mps",
            10.0
        )

        trajectory_y_offset_px = get_int_value(
            request_data,
            "trajectory_y_offset_px",
            0
        )

        front_horizontal_gain = get_float_value(
            request_data,
            "front_horizontal_gain",
            0.4
        )

        print("\n====================")
        print("RUN DOUBLE CAMERA")
        print("====================")
        print(f"Dart speed mps: {dart_speed_mps}")
        print(f"Trajectory Y offset px: {trajectory_y_offset_px}")
        print(f"Front horizontal gain: {front_horizontal_gain}")

        command = [
            sys.executable,
            "double_camera.py",

            "--dart-speed-mps",
            str(dart_speed_mps),

            "--trajectory-y-offset-px",
            str(trajectory_y_offset_px),

            "--front-horizontal-gain",
            str(front_horizontal_gain),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        if result.stdout:

            print(result.stdout)

        if result.stderr:

            print(result.stderr)

        if result.returncode != 0:

            return {
                "success": False,
                "error": result.stderr or "double_camera.py failed."
            }

        latest_result = read_latest_result()

        latest_result["dart_speed_mps"] = dart_speed_mps
        latest_result["trajectory_y_offset_px"] = trajectory_y_offset_px
        latest_result["front_horizontal_gain"] = front_horizontal_gain

        print("\n====================")
        print("WEB RESPONSE")
        print("====================")
        print(f"Video: {latest_result['video_name']}")
        print(f"Target: {latest_result.get('target')}")
        print(f"Hit: ({latest_result['hit_x']}, {latest_result['hit_y']})")
        print(
            "Endpoint: "
            f"({latest_result.get('endpoint_x_px')}, "
            f"{latest_result.get('endpoint_y_px')})"
        )

        return latest_result

    except Exception as exc:

        print(f"[WEB ERROR] {exc}")

        return {
            "success": False,
            "error": str(exc)
        }
