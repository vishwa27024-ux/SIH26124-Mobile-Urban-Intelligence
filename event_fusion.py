from __future__ import annotations

from datetime import datetime
from typing import Any

from event_engine import (
    is_same_incident,
    merge_repeated_event,
    normalize_event,
)


# Same-bus observations are grouped into fixed 5-second
# incident windows instead of a rolling window.
SAME_BUS_WINDOW_SECONDS = 5.0

# Different buses can confirm the same incident over
# a wider time interval.
MULTI_BUS_WINDOW_SECONDS = 60.0


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _event_type(
    event: dict[str, Any],
) -> str:
    return str(
        event.get(
            "event_type",
            "",
        )
    ).strip().lower()


def _get_buses(
    event: dict[str, Any],
) -> list[str]:

    buses: list[str] = []

    reporting = event.get(
        "reporting_buses",
        [],
    )

    if isinstance(
        reporting,
        list,
    ):
        buses.extend(
            str(bus)
            for bus in reporting
            if bus
        )

    bus_id = event.get(
        "bus_id"
    )

    if bus_id:
        buses.append(
            str(bus_id)
        )

    return list(
        dict.fromkeys(
            buses
        )
    )


def _video_time(
    event: dict[str, Any],
) -> float | None:

    value = event.get(
        "video_time_seconds"
    )

    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass

    frame_number = event.get(
        "frame_number"
    )

    fps = event.get(
        "fps"
    )

    if (
        frame_number is not None
        and fps is not None
    ):
        try:
            fps_value = float(fps)

            if fps_value > 0:
                return (
                    float(frame_number)
                    / fps_value
                )
        except (TypeError, ValueError):
            pass

    return None


def _parse_timestamp(
    value: Any,
) -> datetime | None:

    if not value:
        return None

    try:
        return datetime.fromisoformat(
            str(value)
        )
    except (TypeError, ValueError):
        return None


def _time_difference(
    event_a: dict[str, Any],
    event_b: dict[str, Any],
) -> float | None:

    time_a = _video_time(
        event_a
    )

    time_b = _video_time(
        event_b
    )

    if (
        time_a is not None
        and time_b is not None
    ):
        return abs(
            time_a - time_b
        )

    timestamp_a = _parse_timestamp(
        event_a.get(
            "timestamp"
        )
    )

    timestamp_b = _parse_timestamp(
        event_b.get(
            "timestamp"
        )
    )

    if (
        timestamp_a is not None
        and timestamp_b is not None
    ):
        try:
            return abs(
                (
                    timestamp_a
                    - timestamp_b
                ).total_seconds()
            )
        except Exception:
            return None

    return None


def _window_id(
    event: dict[str, Any],
) -> str | None:

    existing = event.get(
        "fusion_window_id"
    )

    if existing is not None:
        return str(existing)

    time_value = _video_time(
        event
    )

    if time_value is None:
        return None

    bucket = int(
        time_value
        // SAME_BUS_WINDOW_SECONDS
    )

    bus_ids = _get_buses(
        event
    )

    bus_id = (
        bus_ids[0]
        if bus_ids
        else "UNKNOWN"
    )

    return (
        f"{bus_id}:"
        f"{bucket}"
    )


def _same_location(
    event_a: dict[str, Any],
    event_b: dict[str, Any],
) -> bool:

    try:
        return bool(
            is_same_incident(
                event_a,
                event_b,
            )
        )
    except Exception:
        return False


def can_fuse_events(
    event_a: dict[str, Any],
    event_b: dict[str, Any],
) -> bool:
    """
    Same type + same location + temporal compatibility.

    Same bus:
        fixed 5-second window

    Different buses:
        up to 60-second confirmation window
    """

    if (
        _event_type(event_a)
        != _event_type(event_b)
    ):
        return False

    if not _same_location(
        event_a,
        event_b,
    ):
        return False

    buses_a = set(
        _get_buses(event_a)
    )

    buses_b = set(
        _get_buses(event_b)
    )

    shared_bus = bool(
        buses_a & buses_b
    )

    time_a = _video_time(
        event_a
    )

    time_b = _video_time(
        event_b
    )

    # For video observations from the same bus,
    # use a FIXED window rather than comparing
    # to the latest merged observation.
    if shared_bus:

        if (
            time_a is not None
            and time_b is not None
        ):
            return (
                int(
                    time_a
                    // SAME_BUS_WINDOW_SECONDS
                )
                ==
                int(
                    time_b
                    // SAME_BUS_WINDOW_SECONDS
                )
            )

        difference = _time_difference(
            event_a,
            event_b,
        )

        if difference is None:
            return True

        return (
            difference
            <= SAME_BUS_WINDOW_SECONDS
        )

    # Different buses can confirm the same location
    # over a wider time interval.
    difference = _time_difference(
        event_a,
        event_b,
    )

    if difference is None:
        return True

    return (
        difference
        <= MULTI_BUS_WINDOW_SECONDS
    )


def _restore_custom_fields(
    event: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:

    updated = dict(
        event
    )

    custom_fields = (
        "video_time_seconds",
        "video_timestamp",
        "fps",
        "frame_number",
        "fusion_window_id",
    )

    for field in custom_fields:

        if field in source:
            updated[field] = source[field]

    return updated


def _restore_fleet_fields(
    event: dict[str, Any],
    buses: list[str],
) -> dict[str, Any]:

    updated = dict(
        event
    )

    unique_buses = list(
        dict.fromkeys(
            str(bus)
            for bus in buses
            if bus
        )
    )

    updated[
        "reporting_buses"
    ] = unique_buses

    updated[
        "reporting_bus_count"
    ] = len(
        unique_buses
    )

    if unique_buses:
        updated[
            "bus_id"
        ] = unique_buses[0]

    return updated


def prepare_event(
    event: dict[str, Any],
) -> dict[str, Any]:

    normalized = normalize_event(
        event
    )

    buses = _get_buses(
        event
    )

    prepared = _restore_fleet_fields(
        normalized,
        buses,
    )

    window_id = _window_id(
        event
    )

    if window_id is not None:
        prepared[
            "fusion_window_id"
        ] = window_id

    for field in (
        "video_time_seconds",
        "video_timestamp",
        "fps",
        "frame_number",
    ):
        if field in event:
            prepared[field] = event[field]

    return prepared


def fuse_two_events(
    event_a: dict[str, Any],
    event_b: dict[str, Any],
) -> dict[str, Any]:

    if not can_fuse_events(
        event_a,
        event_b,
    ):
        raise ValueError(
            "Events cannot be fused."
        )

    normalized_a = prepare_event(
        event_a
    )

    normalized_b = prepare_event(
        event_b
    )

    buses_a = _get_buses(
        normalized_a
    )

    buses_b = _get_buses(
        normalized_b
    )

    all_buses = list(
        dict.fromkeys(
            buses_a
            + buses_b
        )
    )

    count_a = _safe_int(
        normalized_a.get(
            "detection_count",
            1,
        ),
        1,
    )

    count_b = _safe_int(
        normalized_b.get(
            "detection_count",
            1,
        ),
        1,
    )

    confidence_a = _safe_float(
        normalized_a.get(
            "highest_confidence",
            normalized_a.get(
                "confidence",
                0.0,
            ),
        )
    )

    confidence_b = _safe_float(
        normalized_b.get(
            "highest_confidence",
            normalized_b.get(
                "confidence",
                0.0,
            ),
        )
    )

    strongest_confidence = max(
        confidence_a,
        confidence_b,
    )

    merged = merge_repeated_event(
        normalized_a,
        normalized_b,
    )

    merged = normalize_event(
        merged
    )

    merged = _restore_fleet_fields(
        merged,
        all_buses,
    )

    merged[
        "detection_count"
    ] = (
        count_a
        + count_b
    )

    merged[
        "highest_confidence"
    ] = round(
        strongest_confidence,
        4,
    )

    merged[
        "confidence"
    ] = round(
        strongest_confidence,
        4,
    )

    # Preserve the fixed incident window.
    window_id = _window_id(
        normalized_a
    )

    if window_id is None:
        window_id = _window_id(
            normalized_b
        )

    if window_id is not None:
        merged[
            "fusion_window_id"
        ] = window_id

    # Keep the latest raw observation as the event
    # representative, without changing the fixed window.
    time_a = _video_time(
        normalized_a
    )

    time_b = _video_time(
        normalized_b
    )

    latest = normalized_a

    if (
        time_b is not None
        and (
            time_a is None
            or time_b >= time_a
        )
    ):
        latest = normalized_b

    merged = _restore_custom_fields(
        merged,
        latest,
    )

    if (
        len(all_buses)
        >= 2
    ):
        merged[
            "status"
        ] = "VERIFIED"

    return merged


def _find_candidate(
    fused_events: list[
        dict[str, Any]
    ],
    incoming: dict[str, Any],
) -> int | None:

    best_index = None
    best_difference = float(
        "inf"
    )

    incoming_buses = set(
        _get_buses(
            incoming
        )
    )

    for index, existing in enumerate(
        fused_events
    ):

        existing_buses = set(
            _get_buses(
                existing
            )
        )

        shared_bus = bool(
            incoming_buses
            &
            existing_buses
        )

        if not can_fuse_events(
            existing,
            incoming,
        ):
            continue

        difference = _time_difference(
            existing,
            incoming,
        )

        if difference is None:
            return index

        # For same-bus events in a fixed window,
        # prefer the matching window explicitly.
        if shared_bus:

            existing_window = _window_id(
                existing
            )

            incoming_window = _window_id(
                incoming
            )

            if (
                existing_window is not None
                and incoming_window is not None
                and existing_window
                != incoming_window
            ):
                continue

        if (
            difference
            < best_difference
        ):
            best_difference = difference
            best_index = index

    return best_index


def fuse_events(
    events: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    """
    Fuse raw observations into actual incidents.
    """

    prepared = [
        prepare_event(event)
        for event in events
    ]

    prepared.sort(
        key=lambda event: (
            _video_time(event)
            if _video_time(event)
            is not None
            else float("inf")
        )
    )

    fused: list[
        dict[str, Any]
    ] = []

    for incoming in prepared:

        index = _find_candidate(
            fused,
            incoming,
        )

        if index is None:

            fused.append(
                incoming
            )

        else:

            fused[index] = (
                fuse_two_events(
                    fused[index],
                    incoming,
                )
            )

    return fused


def add_event_to_fleet(
    fleet_events: list[
        dict[str, Any]
    ],
    new_event: dict[str, Any],
) -> list[
    dict[str, Any]
]:

    current = list(
        fleet_events
    )

    incoming = prepare_event(
        new_event
    )

    index = _find_candidate(
        current,
        incoming,
    )

    if index is None:

        current.append(
            incoming
        )

    else:

        current[index] = (
            fuse_two_events(
                current[index],
                incoming,
            )
        )

    return current


def get_fusion_summary(
    events: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:

    fused = fuse_events(
        events
    )

    raw = len(
        events
    )

    unique = len(
        fused
    )

    reporting_buses = set()

    for event in fused:

        reporting_buses.update(
            _get_buses(
                event
            )
        )

    multi_bus = sum(
        1
        for event in fused
        if len(
            _get_buses(
                event
            )
        ) >= 2
    )

    verified = sum(
        1
        for event in fused
        if str(
            event.get(
                "status",
                "",
            )
        ).upper()
        == "VERIFIED"
    )

    high_priority = sum(
        1
        for event in fused
        if str(
            event.get(
                "priority",
                "",
            )
        ).upper()
        in {
            "HIGH",
            "CRITICAL",
        }
    )

    return {
        "raw_detections": raw,
        "unique_incidents": unique,
        "merged_detections": max(
            raw - unique,
            0,
        ),
        "multi_bus_incidents": multi_bus,
        "verified_incidents": verified,
        "reporting_buses": len(
            reporting_buses
        ),
        "high_priority_incidents": high_priority,
    }


def self_test() -> None:

    print(
        "EVENT FUSION SELF-TEST"
    )

    print(
        "----------------------------------------"
    )

    event_1 = {
        "event_id": "T-001",
        "event_type": "traffic",
        "status": "DETECTED",
        "severity": "HIGH",
        "priority": "HIGH",
        "traffic_level": "HIGH",
        "vehicle_count": 15,
        "detection_count": 1,
        "confidence": 0.72,
        "highest_confidence": 0.72,
        "latitude": 13.3161,
        "longitude": 75.7720,
        "bus_id": "BUS-01",
        "reporting_buses": ["BUS-01"],
        "video_time_seconds": 10.0,
    }

    event_2 = {
        **event_1,
        "event_id": "T-002",
        "vehicle_count": 16,
        "confidence": 0.84,
        "highest_confidence": 0.84,
        "video_time_seconds": 13.0,
    }

    assert can_fuse_events(
        event_1,
        event_2,
    )

    print(
        "Same-window fusion: PASSED"
    )

    event_3 = {
        **event_1,
        "event_id": "T-003",
        "video_time_seconds": 15.0,
    }

    assert not can_fuse_events(
        event_1,
        event_3,
    )

    print(
        "Fixed-window separation: PASSED"
    )

    event_4 = {
        **event_1,
        "event_id": "T-004",
        "bus_id": "BUS-02",
        "reporting_buses": ["BUS-02"],
        "video_time_seconds": 40.0,
    }

    assert can_fuse_events(
        event_1,
        event_4,
    )

    print(
        "Multi-bus confirmation window: PASSED"
    )

    repeated = fuse_events(
        [
            event_1,
            event_2,
            event_3,
        ]
    )

    assert len(
        repeated
    ) == 2

    print(
        "Repeated observation separation: PASSED"
    )

    pothole_1 = {
        "event_id": "P-001",
        "event_type": "pothole",
        "status": "DETECTED",
        "severity": "MEDIUM",
        "priority": "MEDIUM",
        "detection_count": 1,
        "confidence": 0.72,
        "highest_confidence": 0.72,
        "latitude": 13.3200,
        "longitude": 75.7800,
        "bus_id": "BUS-01",
        "reporting_buses": ["BUS-01"],
        "video_time_seconds": 10.0,
    }

    pothole_2 = {
        **pothole_1,
        "event_id": "P-002",
        "confidence": 0.86,
        "highest_confidence": 0.86,
        "bus_id": "BUS-02",
        "reporting_buses": ["BUS-02"],
        "video_time_seconds": 20.0,
    }

    fused_pothole = fuse_events(
        [
            pothole_1,
            pothole_2,
        ]
    )

    assert len(
        fused_pothole
    ) == 1

    assert (
        fused_pothole[0][
            "reporting_bus_count"
        ] == 2
    )

    print(
        "Two-bus fusion: PASSED"
    )

    summary = get_fusion_summary(
        [
            event_1,
            event_2,
            event_3,
        ]
    )

    assert (
        summary[
            "raw_detections"
        ] == 3
    )

    assert (
        summary[
            "unique_incidents"
        ] == 2
    )

    print(
        "Fusion summary: PASSED"
    )

    print(
        "----------------------------------------"
    )

    print(
        "SELF-TEST PASSED"
    )


if __name__ == "__main__":
    self_test()