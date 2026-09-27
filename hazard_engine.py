from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from event_engine import normalize_event


BASE_DIR = Path(__file__).resolve().parent


def calculate_hazard_severity(
    detection_count: int,
    confidence: float,
) -> str:
    """
    Determine pothole severity from detection count
    and model confidence.
    """

    detection_count = max(
        0,
        int(detection_count),
    )

    confidence = max(
        0.0,
        min(1.0, float(confidence)),
    )

    if detection_count >= 2 and confidence >= 0.70:
        return "HIGH"

    if detection_count >= 1 and confidence >= 0.60:
        return "MEDIUM"

    if detection_count >= 1:
        return "LOW"

    return "NONE"


def calculate_hazard_priority(
    detection_count: int,
    confidence: float,
    reporting_bus_count: int = 1,
) -> str:
    """
    Determine operational priority.

    Multiple buses confirming the same hazard increases
    the priority of the incident.
    """

    detection_count = max(
        0,
        int(detection_count),
    )

    confidence = max(
        0.0,
        min(1.0, float(confidence)),
    )

    reporting_bus_count = max(
        0,
        int(reporting_bus_count),
    )

    # Multiple buses + strong detection = critical
    if (
        reporting_bus_count >= 3
        and detection_count >= 2
        and confidence >= 0.70
    ):
        return "CRITICAL"

    if (
        reporting_bus_count >= 2
        and detection_count >= 1
        and confidence >= 0.60
    ):
        return "HIGH"

    if (
        detection_count >= 2
        and confidence >= 0.70
    ):
        return "HIGH"

    if detection_count >= 1 and confidence >= 0.60:
        return "MEDIUM"

    if detection_count >= 1:
        return "LOW"

    return "NONE"


def _safe_confidence(
    value: Any,
) -> float:
    """Safely convert confidence to a number."""

    try:
        confidence = float(
            value or 0.0
        )
    except (
        TypeError,
        ValueError,
    ):
        confidence = 0.0

    return round(
        max(
            0.0,
            min(1.0, confidence),
        ),
        4,
    )


def _safe_boxes(
    boxes: Any,
) -> list[Any]:
    """Return a safe list of detection boxes."""

    if boxes is None:
        return []

    if isinstance(boxes, list):
        return boxes

    try:
        return list(boxes)
    except TypeError:
        return []


def create_pothole_event(
    detection_count: int,
    confidence: float,
    latitude: float | None = None,
    longitude: float | None = None,
    bus_id: str = "BUS-01",
    route_id: str = "ROUTE-01",
    timestamp: str | None = None,
    evidence_image: str | None = None,
    boxes: list[Any] | None = None,
    event_id: str | None = None,
    source: str = "pothole-ai",
) -> dict[str, Any]:
    """
    Create a unified MUI pothole event.

    This converts pothole AI output into the same event
    structure used by traffic intelligence.
    """

    detection_count = max(
        0,
        int(detection_count),
    )

    confidence = _safe_confidence(
        confidence
    )

    if timestamp is None:
        timestamp = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

    reporting_buses = []

    if bus_id:
        reporting_buses.append(
            str(bus_id)
        )

    severity = calculate_hazard_severity(
        detection_count=detection_count,
        confidence=confidence,
    )

    priority = calculate_hazard_priority(
        detection_count=detection_count,
        confidence=confidence,
        reporting_bus_count=len(
            reporting_buses
        ),
    )

    if event_id is None:
        event_id = (
            f"P-{bus_id}-"
            f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        )

    safe_boxes = _safe_boxes(
        boxes
    )

    event = {
        "event_id": event_id,
        "event_type": "pothole",

        "status": "DETECTED",

        "latitude": latitude,
        "longitude": longitude,

        "timestamp": timestamp,

        "bus_id": bus_id,
        "route_id": route_id,

        "source": source,

        "detection_count": detection_count,

        "confidence": confidence,

        "highest_confidence": confidence,

        "severity": severity,

        "priority": priority,

        "reporting_buses": reporting_buses,

        "evidence": evidence_image,

        "evidence_image": evidence_image,

        "boxes": safe_boxes,

        "vehicle_count": 0,

        "traffic_level": "LOW",

        "congestion": 0.0,

        "location_verified": (
            latitude is not None
            and longitude is not None
        ),
    }

    return normalize_event(
        event
    )


def create_event_from_detection_result(
    detection_result: dict[str, Any],
    bus_id: str = "BUS-01",
    route_id: str = "ROUTE-01",
    latitude: float | None = None,
    longitude: float | None = None,
    timestamp: str | None = None,
    source: str = "pothole-ai",
    event_id: str | None = None,
) -> dict[str, Any]:
    """
    Convert the result returned by the existing pothole
    detection pipeline into a unified event.
    """

    detection_count = int(
        detection_result.get(
            "count",
            detection_result.get(
                "detection_count",
                0,
            ),
        )
        or 0
    )

    confidence = _safe_confidence(
        detection_result.get(
            "highest_confidence",
            detection_result.get(
                "confidence",
                0.0,
            ),
        )
    )

    evidence_image = detection_result.get(
        "image_path",
        detection_result.get(
            "annotated_image",
        ),
    )

    boxes = detection_result.get(
        "boxes",
        [],
    )

    return create_pothole_event(
        detection_count=detection_count,
        confidence=confidence,
        latitude=latitude,
        longitude=longitude,
        bus_id=bus_id,
        route_id=route_id,
        timestamp=timestamp,
        evidence_image=evidence_image,
        boxes=boxes,
        event_id=event_id,
        source=source,
    )


def attach_visual_evidence(
    event: dict[str, Any],
    evidence_image: str | None,
    boxes: list[Any] | None = None,
) -> dict[str, Any]:
    """
    Attach visual evidence to an existing hazard event.
    """

    updated_event = dict(event)

    updated_event["evidence"] = evidence_image
    updated_event["evidence_image"] = evidence_image

    if boxes is not None:
        updated_event["boxes"] = _safe_boxes(
            boxes
        )

    return normalize_event(
        updated_event
    )


def update_hazard_priority(
    event: dict[str, Any],
) -> dict[str, Any]:
    """
    Recalculate severity and priority after an event
    receives additional detections or bus confirmations.
    """

    updated_event = dict(event)

    detection_count = int(
        updated_event.get(
            "detection_count",
            0,
        )
        or 0
    )

    confidence = _safe_confidence(
        updated_event.get(
            "highest_confidence",
            updated_event.get(
                "confidence",
                0.0,
            ),
        )
    )

    reporting_buses = updated_event.get(
        "reporting_buses",
        [],
    )

    if not isinstance(
        reporting_buses,
        list,
    ):
        reporting_buses = []

    reporting_bus_count = len(
        set(
            str(bus)
            for bus in reporting_buses
            if bus
        )
    )

    updated_event["severity"] = (
        calculate_hazard_severity(
            detection_count,
            confidence,
        )
    )

    updated_event["priority"] = (
        calculate_hazard_priority(
            detection_count,
            confidence,
            reporting_bus_count,
        )
    )

    return normalize_event(
        updated_event
    )


def merge_hazard_detection(
    existing_event: dict[str, Any],
    new_detection: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge a new pothole detection into an existing event.

    This is useful when multiple buses or repeated frames
    report the same pothole.
    """

    updated_event = dict(
        existing_event
    )

    old_count = int(
        existing_event.get(
            "detection_count",
            0,
        )
        or 0
    )

    new_count = int(
        new_detection.get(
            "detection_count",
            new_detection.get(
                "count",
                0,
            ),
        )
        or 0
    )

    updated_event["detection_count"] = (
        old_count + new_count
    )

    old_confidence = _safe_confidence(
        existing_event.get(
            "highest_confidence",
            existing_event.get(
                "confidence",
                0.0,
            ),
        )
    )

    new_confidence = _safe_confidence(
        new_detection.get(
            "highest_confidence",
            new_detection.get(
                "confidence",
                0.0,
            ),
        )
    )

    updated_event["confidence"] = round(
        max(
            old_confidence,
            new_confidence,
        ),
        4,
    )

    updated_event["highest_confidence"] = (
        updated_event["confidence"]
    )

    old_buses = existing_event.get(
        "reporting_buses",
        [],
    )

    new_buses = new_detection.get(
        "reporting_buses",
        [],
    )

    if not isinstance(
        old_buses,
        list,
    ):
        old_buses = []

    if not isinstance(
        new_buses,
        list,
    ):
        new_buses = []

    merged_buses = list(
        dict.fromkeys(
            [
                str(bus)
                for bus in (
                    old_buses + new_buses
                )
                if bus
            ]
        )
    )

    updated_event["reporting_buses"] = (
        merged_buses
    )

    if merged_buses:
        updated_event["bus_id"] = (
            merged_buses[0]
        )

    latest_timestamp = new_detection.get(
        "timestamp"
    )

    if latest_timestamp:
        updated_event["timestamp"] = (
            latest_timestamp
        )

    new_evidence = new_detection.get(
        "evidence_image",
        new_detection.get(
            "evidence"
        ),
    )

    if new_evidence:
        updated_event["evidence"] = (
            new_evidence
        )

        updated_event["evidence_image"] = (
            new_evidence
        )

    new_boxes = new_detection.get(
        "boxes"
    )

    if new_boxes:
        updated_event["boxes"] = (
            _safe_boxes(new_boxes)
        )

    return update_hazard_priority(
        updated_event
    )


def self_test() -> None:
    """Run hazard-engine tests without requiring a YOLO model."""

    print(
        "HAZARD ENGINE SELF-TEST"
    )

    print(
        "----------------------------------------"
    )

    event = create_pothole_event(
        detection_count=2,
        confidence=0.76,
        latitude=13.3161,
        longitude=75.7720,
        bus_id="BUS-01",
        route_id="ROUTE-01",
        evidence_image="ai_pothole_annotated.jpg",
        boxes=[
            {
                "x1": 100,
                "y1": 120,
                "x2": 260,
                "y2": 240,
            }
        ],
    )

    assert event["event_type"] == "pothole"

    print(
        "Pothole event creation: PASSED"
    )

    assert event["severity"] == "HIGH"

    print(
        "Severity calculation: PASSED"
    )

    assert event["priority"] == "HIGH"

    print(
        "Priority calculation: PASSED"
    )

    assert (
        event["evidence_image"]
        == "ai_pothole_annotated.jpg"
    )

    assert len(
        event["boxes"]
    ) == 1

    print(
        "Visual evidence attachment: PASSED"
    )

    merged = merge_hazard_detection(
        event,
        {
            "detection_count": 1,
            "confidence": 0.88,
            "timestamp": (
                "2026-09-18T12:30:00+05:30"
            ),
            "reporting_buses": [
                "BUS-02"
            ],
            "evidence_image": (
                "ai_pothole_annotated_2.jpg"
            ),
        },
    )

    assert (
        merged["detection_count"] == 3
    )

    assert (
        "BUS-02"
        in merged["reporting_buses"]
    )

    assert (
        merged["highest_confidence"]
        == 0.88
    )

    print(
        "Repeated detection merge: PASSED"
    )

    assert merged["priority"] == "HIGH"

    normalized = normalize_event(
        merged
    )

    assert (
        normalized["event_type"]
        == "pothole"
    )

    print(
        "Unified event schema: PASSED"
    )

    print(
        "----------------------------------------"
    )

    print(
        "SELF-TEST PASSED"
    )


if __name__ == "__main__":
    self_test()