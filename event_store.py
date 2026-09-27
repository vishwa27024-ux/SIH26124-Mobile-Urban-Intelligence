from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from event_engine import normalize_event


BASE_DIR = Path(__file__).resolve().parent

EVENT_STORE_FILE = BASE_DIR / "events.json"
TRAFFIC_EVENTS_FILE = BASE_DIR / "traffic_events.json"


INITIAL_EVENTS = [
    {
        "event_id": "P100",
        "event_type": "pothole",
        "severity": "NONE",
        "priority": "NONE",
        "status": "DETECTED",
        "vehicle_count": 0,
        "traffic_level": "LOW",
        "congestion": 0.0,
        "confidence": 0.0,
        "highest_confidence": 0.0,
        "detection_count": 0,
        "reporting_buses": [],
        "bus_id": "BUS-01",
        "route_id": "ROUTE-01",
        "latitude": 13.3185,
        "longitude": 75.7788,
        "location_name": "Chikkamagaluru",
        "timestamp": "2026-09-18T10:45:00+05:30",
        "source": "pothole-ai",
        "vehicle_breakdown": {},
        "evidence": None,
        "intervention_latitude": None,
        "intervention_longitude": None,
    }
]


def _normalize_preserve(
    event: dict[str, Any],
) -> dict[str, Any]:
    """
    Normalize the core event schema while preserving
    additional fields such as:

    video_time_seconds
    video_timestamp
    fps
    frame_number
    evidence_image
    etc.
    """

    original = dict(event)

    try:
        normalized = normalize_event(
            original
        )
    except Exception:
        normalized = dict(original)

    for key, value in original.items():

        if key not in normalized:
            normalized[key] = value

    return normalized


def _normalize_events(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    return [
        _normalize_preserve(event)
        for event in events
    ]


def _read_json_file(
    path: Path,
) -> Any:

    if not path.exists():
        return None

    try:

        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):

        return None


def _extract_events(
    data: Any,
) -> list[dict[str, Any]]:

    if isinstance(data, list):

        return [
            item
            for item in data
            if isinstance(item, dict)
        ]

    if isinstance(data, dict):

        events = data.get(
            "events"
        )

        if isinstance(
            events,
            list,
        ):

            return [
                item
                for item in events
                if isinstance(
                    item,
                    dict,
                )
            ]

    return []


def _event_type(
    event: dict[str, Any],
) -> str:

    return str(
        event.get(
            "event_type",
            "",
        )
    ).strip().lower()


def _load_generated_traffic() -> list[
    dict[str, Any]
]:

    data = _read_json_file(
        TRAFFIC_EVENTS_FILE
    )

    return _extract_events(
        data
    )


def load_events(
    prefer_generated: bool = True,
) -> list[dict[str, Any]]:
    """
    Load raw events for the MUI intelligence pipeline.

    IMPORTANT:
    Generated traffic observations are intentionally
    NOT spatially fused here.

    event_fusion.py is responsible for:
        time-aware fusion
        spatial fusion
        multi-bus confirmation

    Priority:

        generated traffic events
        +
        non-traffic demo events

    This prevents the 78 YOLO observations from being
    incorrectly reduced to one event before fusion.
    """

    generated_traffic = (
        _load_generated_traffic()
        if prefer_generated
        else []
    )

    if generated_traffic:

        # Keep only non-traffic demo events from the
        # built-in data. Real generated traffic replaces
        # the old static traffic records.
        non_traffic_demo = [
            dict(event)
            for event in INITIAL_EVENTS
            if _event_type(event)
            != "traffic"
        ]

        events = (
            generated_traffic
            + non_traffic_demo
        )

        return _normalize_events(
            events
        )

    # --------------------------------------------------------
    # Fallback to persistent event store
    # --------------------------------------------------------

    persistent_data = _read_json_file(
        EVENT_STORE_FILE
    )

    persistent_events = _extract_events(
        persistent_data
    )

    if persistent_events:

        return _normalize_events(
            persistent_events
        )

    # --------------------------------------------------------
    # Final fallback
    # --------------------------------------------------------

    return _normalize_events(
        [
            dict(event)
            for event in INITIAL_EVENTS
        ]
    )


def find_event(
    event_id: str,
) -> dict[str, Any] | None:

    for event in load_events():

        if str(
            event.get(
                "event_id"
            )
        ) == str(event_id):

            return event

    return None


def add_event(
    event: dict[str, Any],
    save: bool = True,
) -> dict[str, Any]:
    """
    Add one event.

    Event fusion is NOT performed here.
    """

    events = load_events(
        prefer_generated=False
    )

    normalized = _normalize_preserve(
        event
    )

    existing_id = str(
        normalized.get(
            "event_id",
            "",
        )
    )

    for index, existing in enumerate(
        events
    ):

        if str(
            existing.get(
                "event_id",
                "",
            )
        ) == existing_id:

            events[index] = normalized

            if save:
                save_events(
                    events
                )

            return normalized

    events.append(
        normalized
    )

    if save:
        save_events(
            events
        )

    return normalized


def upsert_event(
    events: list[dict[str, Any]],
    new_event: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Insert or replace by exact event ID.

    Spatial/time fusion is deliberately handled
    by event_fusion.py.
    """

    normalized = _normalize_preserve(
        new_event
    )

    event_id = str(
        normalized.get(
            "event_id",
            "",
        )
    )

    for index, existing in enumerate(
        events
    ):

        if str(
            existing.get(
                "event_id",
                "",
            )
        ) == event_id:

            events[index] = normalized

            return events

    events.append(
        normalized
    )

    return events


def update_event(
    event_id: str,
    updates: dict[str, Any],
    save: bool = True,
) -> dict[str, Any] | None:

    events = load_events(
        prefer_generated=False
    )

    for index, event in enumerate(
        events
    ):

        if str(
            event.get(
                "event_id"
            )
        ) != str(event_id):

            continue

        updated = dict(
            event
        )

        updated.update(
            updates
        )

        updated = _normalize_preserve(
            updated
        )

        events[index] = updated

        if save:
            save_events(
                events
            )

        return updated

    return None


def replace_event(
    event: dict[str, Any],
    save: bool = False,
) -> bool:

    event_id = event.get(
        "event_id"
    )

    if event_id is None:
        return False

    events = load_events(
        prefer_generated=False
    )

    for index, existing in enumerate(
        events
    ):

        if str(
            existing.get(
                "event_id"
            )
        ) == str(event_id):

            events[index] = (
                _normalize_preserve(
                    event
                )
            )

            if save:
                save_events(
                    events
                )

            return True

    return False


def delete_event(
    event_id: str,
    save: bool = True,
) -> bool:

    events = load_events(
        prefer_generated=False
    )

    original_count = len(
        events
    )

    events = [
        event
        for event in events
        if str(
            event.get(
                "event_id"
            )
        ) != str(event_id)
    ]

    if len(events) == original_count:
        return False

    if save:
        save_events(
            events
        )

    return True


def save_events(
    events: list[
        dict[str, Any]
    ],
) -> None:

    normalized_events = (
        _normalize_events(
            events
        )
    )

    payload = {
        "project": (
            "MUI - Mobile Urban Intelligence"
        ),
        "event_count": len(
            normalized_events
        ),
        "events": normalized_events,
    }

    EVENT_STORE_FILE.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def get_event_summary(
    events: list[
        dict[str, Any]
    ] | None = None,
) -> dict[str, Any]:

    if events is None:

        events = load_events()

    total = len(
        events
    )

    traffic = sum(
        1
        for event in events
        if _event_type(event)
        == "traffic"
    )

    potholes = sum(
        1
        for event in events
        if _event_type(event)
        == "pothole"
    )

    critical = sum(
        1
        for event in events
        if str(
            event.get(
                "priority",
                "",
            )
        ).upper()
        == "CRITICAL"
    )

    high = sum(
        1
        for event in events
        if str(
            event.get(
                "priority",
                "",
            )
        ).upper()
        == "HIGH"
    )

    medium = sum(
        1
        for event in events
        if str(
            event.get(
                "priority",
                "",
            )
        ).upper()
        == "MEDIUM"
    )

    low = sum(
        1
        for event in events
        if str(
            event.get(
                "priority",
                "",
            )
        ).upper()
        == "LOW"
    )

    resolved = sum(
        1
        for event in events
        if str(
            event.get(
                "status",
                "",
            )
        ).upper()
        == "RESOLVED"
    )

    detected = sum(
        1
        for event in events
        if str(
            event.get(
                "status",
                "",
            )
        ).upper()
        == "DETECTED"
    )

    buses = set()

    for event in events:

        reporting = event.get(
            "reporting_buses",
            [],
        )

        if isinstance(
            reporting,
            list,
        ):

            buses.update(
                str(bus)
                for bus in reporting
                if bus
            )

        bus_id = event.get(
            "bus_id"
        )

        if bus_id:

            buses.add(
                str(bus_id)
            )

    repeated = sum(
        1
        for event in events
        if int(
            event.get(
                "detection_count",
                1,
            )
            or 1
        ) > 1
    )

    return {
        "total_events": total,
        "traffic_events": traffic,
        "pothole_events": potholes,
        "critical_events": critical,
        "high_priority_events": high,
        "medium_priority_events": medium,
        "low_priority_events": low,
        "resolved_events": resolved,
        "detected_events": detected,
        "active_buses": len(
            buses
        ),
        "repeated_events": repeated,
    }


def self_test() -> None:

    print(
        "EVENT STORE SELF-TEST"
    )

    print(
        "----------------------------------------"
    )

    events = load_events(
        prefer_generated=True
    )

    traffic_events = [
        event
        for event in events
        if _event_type(event)
        == "traffic"
    ]

    pothole_events = [
        event
        for event in events
        if _event_type(event)
        == "pothole"
    ]

    print(
        f"Total raw events: "
        f"{len(events)}"
    )

    print(
        f"Traffic observations: "
        f"{len(traffic_events)}"
    )

    print(
        f"Pothole events: "
        f"{len(pothole_events)}"
    )

    # Real generated traffic should currently
    # contain 78 observations.
    if TRAFFIC_EVENTS_FILE.exists():

        assert len(
            traffic_events
        ) > 1

        print(
            "Generated traffic preserved: PASSED"
        )

        # Ensure video timing survived normalization.
        first_traffic = (
            traffic_events[0]
        )

        assert (
            "video_time_seconds"
            in first_traffic
        )

        print(
            "Video timing preserved: PASSED"
        )

    pothole = find_event(
        "P100"
    )

    assert pothole is not None

    print(
        "Pothole event: FOUND"
    )

    test_event = {
        "event_id": "STORE-TEST-001",
        "event_type": "traffic",
        "status": "DETECTED",
        "severity": "MEDIUM",
        "priority": "MEDIUM",
        "vehicle_count": 8,
        "traffic_level": "MEDIUM",
        "latitude": 13.5000,
        "longitude": 75.9000,
        "bus_id": "BUS-TEST",
        "route_id": "ROUTE-TEST",
        "reporting_buses": [
            "BUS-TEST"
        ],
        "video_time_seconds": 12.5,
    }

    temporary = list(
        events
    )

    temporary = upsert_event(
        temporary,
        test_event,
    )

    assert any(
        event.get(
            "event_id"
        )
        == "STORE-TEST-001"
        for event in temporary
    )

    print(
        "Event insertion: PASSED"
    )

    updated = update_event(
        "P100",
        {
            "status": "VERIFIED"
        },
        save=False,
    )

    assert updated is not None

    assert (
        updated["status"]
        == "VERIFIED"
    )

    print(
        "Event update: PASSED"
    )

    summary = get_event_summary(
        events
    )

    assert (
        summary["total_events"]
        == len(events)
    )

    print(
        "Event summary: PASSED"
    )

    print(
        "----------------------------------------"
    )

    print(
        "SELF-TEST PASSED"
    )


if __name__ == "__main__":
    self_test()