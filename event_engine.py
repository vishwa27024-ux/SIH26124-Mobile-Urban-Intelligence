"""
Unified urban event engine for SIH26124.

This module converts detections from traffic, pothole, and future
hazard pipelines into one consistent event structure.

No external packages are required.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from math import atan2, cos, radians, sin, sqrt
from typing import Any


DEFAULT_STATUS = "DETECTED"
DEFAULT_SEVERITY = "UNKNOWN"
DEFAULT_PRIORITY = "UNASSIGNED"


def _now_timestamp() -> str:
    """Return the current local timestamp in a consistent format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _validate_coordinate(value: Any, name: str) -> float | None:
    """Validate and normalize a latitude/longitude value."""
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number or None.") from exc

    if name == "latitude" and not -90 <= number <= 90:
        raise ValueError("latitude must be between -90 and 90.")

    if name == "longitude" and not -180 <= number <= 180:
        raise ValueError("longitude must be between -180 and 180.")

    return number


def create_event(
    *,
    event_id: int | str,
    event_type: str,
    source: str,
    confidence: float | None = None,
    timestamp: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    location: str | None = None,
    bus_id: str | None = None,
    route_id: str | None = None,
    severity: str = DEFAULT_SEVERITY,
    priority: str = DEFAULT_PRIORITY,
    evidence_image: str | None = None,
    recommendation: str | None = None,
    status: str = DEFAULT_STATUS,
    **extra: Any,
) -> dict[str, Any]:
    """
    Create one normalized SIH26124 urban event.

    All detection pipelines should eventually use this function.
    """

    if not event_type or not isinstance(event_type, str):
        raise ValueError("event_type must be a non-empty string.")

    if not source or not isinstance(source, str):
        raise ValueError("source must be a non-empty string.")

    if confidence is not None:
        confidence = float(confidence)

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")

    event = {
        "event_id": event_id,
        "event_type": event_type,
        "source": source,
        "confidence": confidence,
        "timestamp": timestamp or _now_timestamp(),
        "latitude": _validate_coordinate(latitude, "latitude"),
        "longitude": _validate_coordinate(longitude, "longitude"),
        "location": location,
        "bus_id": bus_id,
        "route_id": route_id,
        "severity": severity,
        "priority": priority,
        "evidence_image": evidence_image,
        "recommendation": recommendation,
        "status": status,
    }

    event.update(extra)

    return event


def normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize an existing event dictionary.

    Useful when migrating the current hard-coded EVENTS list.
    """

    if not isinstance(event, dict):
        raise TypeError("event must be a dictionary.")

    normalized = deepcopy(event)

    normalized.setdefault("event_id", None)
    normalized.setdefault("event_type", "unknown")
    normalized.setdefault("source", "unknown")
    normalized.setdefault("confidence", None)
    normalized.setdefault("timestamp", _now_timestamp())
    normalized.setdefault("latitude", None)
    normalized.setdefault("longitude", None)
    normalized.setdefault("location", None)
    normalized.setdefault("bus_id", None)
    normalized.setdefault("route_id", None)
    normalized.setdefault("severity", DEFAULT_SEVERITY)
    normalized.setdefault("priority", DEFAULT_PRIORITY)
    normalized.setdefault("evidence_image", None)
    normalized.setdefault("recommendation", None)
    normalized.setdefault("status", DEFAULT_STATUS)

    normalized["latitude"] = _validate_coordinate(
        normalized["latitude"],
        "latitude",
    )

    normalized["longitude"] = _validate_coordinate(
        normalized["longitude"],
        "longitude",
    )

    if normalized["confidence"] is not None:
        normalized["confidence"] = float(normalized["confidence"])

    return normalized


def update_event(
    event: dict[str, Any],
    **updates: Any,
) -> dict[str, Any]:
    """Return a normalized copy of an event with selected fields updated."""

    normalized = normalize_event(event)
    normalized.update(updates)

    return normalize_event(normalized)


def _haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate distance between two GPS coordinates in kilometres."""

    radius_km = 6371.0

    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1_rad)
        * cos(lat2_rad)
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return radius_km * c


def is_same_incident(
    existing_event: dict[str, Any],
    new_event: dict[str, Any],
    *,
    distance_km: float = 0.10,
) -> bool:
    """
    Decide whether two events probably represent the same incident.

    Current rule:
    - same event type
    - both have GPS
    - within the configured distance
    """

    existing = normalize_event(existing_event)
    new = normalize_event(new_event)

    if existing["event_type"] != new["event_type"]:
        return False

    if (
        existing["latitude"] is None
        or existing["longitude"] is None
        or new["latitude"] is None
        or new["longitude"] is None
    ):
        return False

    distance = _haversine_km(
        existing["latitude"],
        existing["longitude"],
        new["latitude"],
        new["longitude"],
    )

    return distance <= distance_km


def merge_repeated_event(
    existing_event: dict[str, Any],
    new_event: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge a repeated detection into one event.

    This is intentionally simple for the first version.
    Fleet-level confirmation will be expanded later.
    """

    existing = normalize_event(existing_event)
    new = normalize_event(new_event)

    merged = deepcopy(existing)

    # Keep the strongest confidence.
    existing_conf = existing.get("confidence")
    new_conf = new.get("confidence")

    if existing_conf is None:
        merged["confidence"] = new_conf
    elif new_conf is not None:
        merged["confidence"] = max(
            existing_conf,
            new_conf,
        )

    # Prefer the newest timestamp.
    if new.get("timestamp"):
        merged["timestamp"] = new["timestamp"]

    # Preserve useful evidence.
    if new.get("evidence_image"):
        merged["evidence_image"] = new["evidence_image"]

    # Preserve location if the old event did not have one.
    if merged.get("latitude") is None:
        merged["latitude"] = new.get("latitude")

    if merged.get("longitude") is None:
        merged["longitude"] = new.get("longitude")

    if merged.get("location") is None:
        merged["location"] = new.get("location")

    # Add fleet-confirmation information.
    reporting_buses = set()

    if existing.get("bus_id"):
        reporting_buses.add(existing["bus_id"])

    if new.get("bus_id"):
        reporting_buses.add(new["bus_id"])

    if reporting_buses:
        merged["reporting_buses"] = sorted(
            reporting_buses
        )
        merged["reporting_bus_count"] = len(
            reporting_buses
        )

    previous_count = existing.get(
        "detection_count",
        1,
    )

    merged["detection_count"] = (
        int(previous_count) + 1
    )

    return normalize_event(merged)


if __name__ == "__main__":
    # Basic self-test.
    traffic_event = create_event(
        event_id="T001",
        event_type="traffic",
        source="traffic-ai",
        confidence=0.91,
        latitude=13.3161,
        longitude=75.7720,
        location="Chikkamagaluru",
        bus_id="BUS-01",
        route_id="R01",
        severity="HIGH",
        priority="HIGH",
        recommendation="Review signal timing and monitor queue growth.",
    )

    repeated_traffic = create_event(
        event_id="T002",
        event_type="traffic",
        source="traffic-ai",
        confidence=0.94,
        latitude=13.3164,
        longitude=75.7722,
        location="Chikkamagaluru",
        bus_id="BUS-02",
        route_id="R01",
        severity="HIGH",
        priority="HIGH",
    )

    print("EVENT ENGINE SELF-TEST")
    print("-" * 40)
    print("Event created:", traffic_event["event_id"])
    print("Event type:", traffic_event["event_type"])
    print("GPS:", traffic_event["latitude"], traffic_event["longitude"])
    print("Status:", traffic_event["status"])

    same = is_same_incident(
        traffic_event,
        repeated_traffic,
    )

    print("Same incident:", same)

    if same:
        merged = merge_repeated_event(
            traffic_event,
            repeated_traffic,
        )

        print(
            "Reporting buses:",
            merged.get("reporting_buses"),
        )

        print(
            "Detection count:",
            merged.get("detection_count"),
        )

    print("-" * 40)
    print("SELF-TEST PASSED")
    