import json
import time
from pathlib import Path

try:
    import RPi.GPIO as GPIO
except (ImportError, RuntimeError) as exc:
    GPIO = None
    GPIO_ERROR = exc
else:
    GPIO_ERROR = None


LATEST_RESULT_FILE = Path("output/latest_result.json")

LIGHT_DURATION_SECONDS = 5


LED_PINS = {
    "top": 17,
    "left": 23,
    "center": 27,
    "right": 24,
    "bottom": 22,
}


def setup_gpio():
    if GPIO is None:
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


def load_latest_result(result_path=LATEST_RESULT_FILE):
    result_path = Path(result_path)

    if not result_path.exists():
        raise FileNotFoundError(
            f"Latest result file not found: {result_path}"
        )

    with result_path.open("r", encoding="utf-8") as result_file:
        return json.load(result_file)


def normalize_target(target):
    if target is None:
        return "miss"

    normalized = str(target).strip().lower()

    if normalized in LED_PINS:
        return normalized

    return "miss"


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


def target_to_pin(target):
    normalized_target = normalize_target(target)

    return LED_PINS.get(normalized_target)


def light_target(target, hit):
    if GPIO is None:
        print(f"[LED skipped] RPi.GPIO is not available: {GPIO_ERROR}")
        return None

    clear_leds()

    if not hit:
        return None

    normalized_target = normalize_target(target)

    pin = target_to_pin(normalized_target)

    if pin is None:
        return None

    GPIO.output(pin, GPIO.HIGH)

    return pin


def process_hit(result_path=LATEST_RESULT_FILE):
    result = load_latest_result(result_path)

    target = normalize_target(
        result.get("target")
    )

    hit = safe_bool(
        result.get("hit")
    )

    endpoint_x_px = result.get("endpoint_x_px")
    endpoint_y_px = result.get("endpoint_y_px")
    distance_px = result.get("distance_px")
    video_name = result.get("video_name")

    active_pin = light_target(
        target=target,
        hit=hit
    )

    led_target = target if hit and active_pin is not None else "miss"

    print("\n====================")
    print("LED RESULT")
    print("====================")
    print(f"Video       : {video_name}")
    print(f"Hit         : {hit}")
    print(f"Target      : {target}")
    print(f"LED Target  : {led_target}")
    print(f"GPIO Pin    : {active_pin}")
    print(f"Endpoint X  : {endpoint_x_px}")
    print(f"Endpoint Y  : {endpoint_y_px}")
    print(f"Distance px : {distance_px}")

    return led_target


def cleanup():
    if GPIO is None:
        return

    clear_leds()
    GPIO.cleanup()


def run_once(
    result_path=LATEST_RESULT_FILE,
    duration=LIGHT_DURATION_SECONDS,
    cleanup_after=True,
):
    try:
        gpio_ready = setup_gpio()

        led_target = process_hit(result_path)

        if gpio_ready:
            time.sleep(duration)

        return led_target

    finally:
        if cleanup_after:
            cleanup()
        else:
            clear_leds()


def main():
    run_once()


if __name__ == "__main__":
    main()