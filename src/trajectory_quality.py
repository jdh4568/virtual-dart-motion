from pathlib import Path

import pandas as pd


REASON_MESSAGES = {
    "valid": "궤적이 정상적으로 형성되었습니다.",
    "empty": "궤적 데이터가 생성되지 않았습니다. 다시 촬영하세요.",
    "invalid_values": "궤적 좌표를 계산할 수 없습니다. 다시 촬영하세요.",
    "wrong_direction": "투척 방향이 오른쪽을 향하지 않습니다. 다시 촬영하세요.",
    "sky_exit": "궤적이 화면 위쪽으로 벗어났습니다. 다시 촬영하세요.",
    "ground_exit": "궤적이 화면 아래쪽으로 벗어났습니다. 다시 촬영하세요.",
    "right_edge_not_reached": "궤적이 화면 오른쪽 끝에 도달하지 못했습니다. 다시 촬영하세요.",
}


def evaluate_trajectory_quality(
    trajectory_csv,
    expected_endpoint_x=None,
    right_edge_tolerance=0.02,
    vertical_tolerance=0.0,
):
    """Check whether a left-to-right trajectory reaches the right edge on screen."""
    trajectory_csv = Path(trajectory_csv)
    df = pd.read_csv(trajectory_csv)

    if df.empty:
        return _result(False, "empty")

    if not {"x", "y"}.issubset(df.columns):
        return _result(False, "invalid_values")

    points = df[["x", "y"]].apply(pd.to_numeric, errors="coerce")
    if points.isna().any().any():
        return _result(False, "invalid_values")

    start_x = float(points.iloc[0]["x"])
    endpoint_x = float(points.iloc[-1]["x"])
    endpoint_y = float(points.iloc[-1]["y"])

    if expected_endpoint_x is None:
        expected_endpoint_x = 1.0

    minimum_endpoint_x = float(expected_endpoint_x) - right_edge_tolerance

    # A left-to-right throw should make meaningful progress toward the right edge.
    if endpoint_x <= start_x or endpoint_x - start_x < 0.05:
        return _result(
            False,
            "wrong_direction",
            endpoint_x=endpoint_x,
            endpoint_y=endpoint_y,
        )

    upper_limit = -vertical_tolerance
    lower_limit = 1.0 + vertical_tolerance

    start_y = float(points.iloc[0]["y"])
    if start_y < upper_limit:
        return _exit_result(False, "sky_exit", endpoint_x, endpoint_y, 0, start_x, start_y)
    if start_y > lower_limit:
        return _exit_result(False, "ground_exit", endpoint_x, endpoint_y, 0, start_x, start_y)

    # Compare segment intersection times so a path that reaches the right edge
    # before leaving vertically is accepted even if later samples are off-screen.
    for position in range(1, len(points)):
        previous = points.iloc[position - 1]
        current = points.iloc[position]
        x0, y0 = float(previous["x"]), float(previous["y"])
        x1, y1 = float(current["x"]), float(current["y"])
        events = []

        if x0 < minimum_endpoint_x <= x1 and x1 != x0:
            events.append(((minimum_endpoint_x - x0) / (x1 - x0), "valid"))
        if y0 >= upper_limit > y1 and y1 != y0:
            events.append(((upper_limit - y0) / (y1 - y0), "sky_exit"))
        if y0 <= lower_limit < y1 and y1 != y0:
            events.append(((lower_limit - y0) / (y1 - y0), "ground_exit"))

        if not events:
            continue

        _, first_event = min(events, key=lambda event: event[0])
        if first_event == "valid":
            return _result(
                True,
                "valid",
                endpoint_x=endpoint_x,
                endpoint_y=endpoint_y,
            )

        return _exit_result(
            False,
            first_event,
            endpoint_x,
            endpoint_y,
            position,
            x1,
            y1,
        )

    if endpoint_x < minimum_endpoint_x:
        return _result(False, "right_edge_not_reached", endpoint_x=endpoint_x, endpoint_y=endpoint_y)

    return _result(
        True,
        "valid",
        endpoint_x=endpoint_x,
        endpoint_y=endpoint_y,
    )


def _result(valid, reason, **details):
    return {
        "trajectory_valid": valid,
        "retry_required": not valid,
        "trajectory_quality_reason": reason,
        "trajectory_quality_message": REASON_MESSAGES[reason],
        **details,
    }


def _exit_result(valid, reason, endpoint_x, endpoint_y, index, exit_x, exit_y):
    return _result(
        valid,
        reason,
        endpoint_x=endpoint_x,
        endpoint_y=endpoint_y,
        exit_point_index=index,
        exit_x=exit_x,
        exit_y=exit_y,
    )
