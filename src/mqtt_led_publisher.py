import json
from pathlib import Path

import paho.mqtt.client as mqtt


LATEST_RESULT_FILE = Path("output/latest_result.json")

MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "virtual-dart/led/result"

DEFAULT_DURATION_SECONDS = 5


def load_latest_result(result_path=LATEST_RESULT_FILE):
    result_path = Path(result_path)

    if not result_path.exists():
        raise FileNotFoundError(
            f"Latest result file not found: {result_path}"
        )

    with result_path.open("r", encoding="utf-8") as result_file:
        return json.load(result_file)


def safe_bool(value):
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
            "hit",
        }

    return bool(value)


def normalize_target(target):
    if target is None:
        return "miss"

    normalized = str(target).strip().lower()

    if normalized in {
        "top",
        "left",
        "center",
        "right",
        "bottom",
    }:
        return normalized

    return "miss"


def build_payload(
    result,
    duration=DEFAULT_DURATION_SECONDS,
):
    hit = safe_bool(
        result.get("hit")
    )

    target = normalize_target(
        result.get("target")
    )

    payload = {
        "hit": hit,
        "target": target,
        "duration": duration,

        "video_name":
            result.get("video_name"),

        "endpoint_x_px":
            result.get("endpoint_x_px"),

        "endpoint_y_px":
            result.get("endpoint_y_px"),

        "distance_px":
            result.get("distance_px"),

        "mode":
            result.get("mode"),

        "front_direction_x":
            result.get("front_direction_x"),
    }

    return payload


def publish_led_result(
    result_path=LATEST_RESULT_FILE,
    broker_host=MQTT_BROKER_HOST,
    broker_port=MQTT_BROKER_PORT,
    topic=MQTT_TOPIC,
    duration=DEFAULT_DURATION_SECONDS,
):
    result = load_latest_result(
        result_path
    )

    payload = build_payload(
        result=result,
        duration=duration,
    )

    payload_text = json.dumps(
        payload,
        ensure_ascii=False,
    )

    client = mqtt.Client()

    print("\n====================")
    print("MQTT LED PUBLISH")
    print("====================")
    print(f"Broker : {broker_host}:{broker_port}")
    print(f"Topic  : {topic}")
    print(f"Payload: {payload_text}")

    client.connect(
        broker_host,
        broker_port,
        keepalive=30,
    )

    publish_result = client.publish(
        topic,
        payload_text,
        qos=0,
        retain=False,
    )

    publish_result.wait_for_publish()

    client.disconnect()

    print("MQTT publish complete")

    return payload


def main():
    publish_led_result()


if __name__ == "__main__":
    main()