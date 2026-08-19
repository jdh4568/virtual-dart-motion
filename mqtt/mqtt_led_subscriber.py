import json
import time

import paho.mqtt.client as mqtt

try:
    import RPi.GPIO as GPIO
except (ImportError, RuntimeError) as exc:
    GPIO = None
    GPIO_ERROR = exc
else:
    GPIO_ERROR = None


MQTT_BROKER_HOST = "192.168.0.10"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "virtual-dart/led/result"

DEFAULT_DURATION_SECONDS = 5


LED_PINS = {
    "top": 17,
    "left": 23,
    "center": 27,
    "right": 24,
    "bottom": 22,
}


def setup_gpio():
    if GPIO is None:
        print(f"[GPIO ERROR] RPi.GPIO is not available: {GPIO_ERROR}")
        return False

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    for pin in LED_PINS.values():
        GPIO.setup(pin, GPIO.OUT)

    clear_leds()

    return True


def clear_leds():
    if GPIO is None:
        return

    for pin in LED_PINS.values():
        GPIO.output(pin, GPIO.LOW)


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

    if normalized in LED_PINS:
        return normalized

    return "miss"


def light_target(
    target,
    hit,
    duration=DEFAULT_DURATION_SECONDS,
):
    clear_leds()

    if not hit:
        print("[LED] MISS - all LEDs off")
        return "miss"

    normalized_target = normalize_target(
        target
    )

    pin = LED_PINS.get(
        normalized_target
    )

    if pin is None:
        print(
            f"[LED] Unknown target: {target}. "
            "All LEDs off."
        )
        return "miss"

    GPIO.output(
        pin,
        GPIO.HIGH
    )

    print(
        f"[LED] ON - target={normalized_target}, "
        f"pin=GPIO{pin}, duration={duration}s"
    )

    time.sleep(duration)

    clear_leds()

    print("[LED] OFF")

    return normalized_target


def handle_payload(payload):
    hit = safe_bool(
        payload.get("hit")
    )

    target = normalize_target(
        payload.get("target")
    )

    duration = payload.get(
        "duration",
        DEFAULT_DURATION_SECONDS,
    )

    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = DEFAULT_DURATION_SECONDS

    video_name = payload.get("video_name")
    endpoint_x_px = payload.get("endpoint_x_px")
    endpoint_y_px = payload.get("endpoint_y_px")
    distance_px = payload.get("distance_px")

    print("\n====================")
    print("MQTT LED MESSAGE")
    print("====================")
    print(f"Video       : {video_name}")
    print(f"Hit         : {hit}")
    print(f"Target      : {target}")
    print(f"Duration    : {duration}")
    print(f"Endpoint X  : {endpoint_x_px}")
    print(f"Endpoint Y  : {endpoint_y_px}")
    print(f"Distance px : {distance_px}")

    if GPIO is None:
        print(f"[LED skipped] RPi.GPIO is not available: {GPIO_ERROR}")
        return

    light_target(
        target=target,
        hit=hit,
        duration=duration,
    )


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("\n====================")
        print("MQTT CONNECTED")
        print("====================")
        print(f"Broker: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
        print(f"Topic : {MQTT_TOPIC}")

        client.subscribe(
            MQTT_TOPIC
        )

    else:
        print(f"[MQTT ERROR] Connection failed. rc={rc}")


def on_message(client, userdata, msg):
    try:
        payload_text = msg.payload.decode(
            "utf-8"
        )

        payload = json.loads(
            payload_text
        )

    except Exception as exc:
        print(f"[MQTT ERROR] Invalid payload: {exc}")
        print(f"Raw payload: {msg.payload}")
        return

    handle_payload(
        payload
    )


def main():
    setup_gpio()

    client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message

    print("\n====================")
    print("MQTT LED SUBSCRIBER START")
    print("====================")

    try:
        client.connect(
            MQTT_BROKER_HOST,
            MQTT_BROKER_PORT,
            keepalive=30,
        )

        client.loop_forever()

    except KeyboardInterrupt:
        print("\n[MQTT] Subscriber stopped by user.")

    finally:
        clear_leds()

        if GPIO is not None:
            GPIO.cleanup()


if __name__ == "__main__":
    main()