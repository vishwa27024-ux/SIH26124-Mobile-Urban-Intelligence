from __future__ import annotations

from datetime import datetime
from typing import Any

from event_engine import normalize_event


VALID_STATUSES = (
    "DETECTED",
    "VERIFIED",
    "PRIORITIZED",
    "ASSIGNED",
    "IN_PROGRESS",
    "RESOLVED",
)


STATUS_ORDER = {
    "DETECTED": 0,
    "VERIFIED": 1,
    "PRIORITIZED": 2,
    "ASSIGNED": 3,
    "IN_PROGRESS": 4,
    "RESOLVED": 5,
}


def _timestamp() -> str:
    """Return the current local timestamp."""

    return datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


def normalize_status(status: Any) -> str:
    """Convert a status value into a valid lifecycle status."""

    if status is None:
        return "DETECTED"

    value = str(status).strip().upper()

    aliases = {
        "NEW": "DETECTED",
        "CONFIRMED": "VERIFIED",
        "VERIFY": "VERIFIED",
        "PRIORITY": "PRIORITIZED",
        "INPROGRESS": "IN_PROGRESS",
        "IN PROGRESS": "IN_PROGRESS",
        "DONE": "RESOLVED",
        "CLOSED": "RESOLVED",
    }

    value = aliases.get(
        value,
        value,
    )

    if value not in VALID_STATUSES:
        return "DETECTED"

    return value


def get_status_order(
    status: Any,
) -> int:
    """Return lifecycle order for a status."""

    normalized = normalize_status(
        status
    )

    return STATUS_ORDER.get(
        normalized,
        0,
    )


def can_transition(
    current_status: Any,
    new_status: Any,
) -> bool:
    """
    Determine whether an event can move to a new
    lifecycle state.

    Lifecycle:

    DETECTED
       ↓
    VERIFIED
       ↓
    PRIORITIZED
       ↓
    ASSIGNED
       ↓
    IN_PROGRESS
       ↓
    RESOLVED
    """

    current = normalize_status(
        current_status
    )

    new = normalize_status(
        new_status
    )

    if current == new:
        return True

    # Allow moving backwards only for operational
    # correction from VERIFIED back to DETECTED.
    if (
        current == "VERIFIED"
        and new == "DETECTED"
    ):
        return True

    return (
        get_status_order(new)
        == get_status_order(current) + 1
    )


def transition_event(
    event: dict[str, Any],
    new_status: str,
    changed_by: str = "MUI-SYSTEM",
    note: str | None = None,
) -> dict[str, Any]:
    """
    Move an event to the next lifecycle state.

    The original event dictionary is not modified.
    """

    updated_event = dict(
        event
    )

    current_status = normalize_status(
        updated_event.get(
            "status",
            "DETECTED",
        )
    )

    target_status = normalize_status(
        new_status
    )

    if not can_transition(
        current_status,
        target_status,
    ):
        raise ValueError(
            f"Invalid lifecycle transition: "
            f"{current_status} -> {target_status}"
        )

    now = _timestamp()

    updated_event["status"] = target_status

    updated_event["status_updated_at"] = now

    updated_event["status_updated_by"] = (
        changed_by
    )

    if note:
        updated_event["status_note"] = note

    # Keep a complete lifecycle history.
    history = updated_event.get(
        "status_history",
        [],
    )

    if not isinstance(
        history,
        list,
    ):
        history = []

    history = list(
        history
    )

    history.append(
        {
            "from": current_status,
            "to": target_status,
            "timestamp": now,
            "changed_by": changed_by,
            "note": note,
        }
    )

    updated_event["status_history"] = history

    return normalize_event(
        updated_event
    )


def verify_event(
    event: dict[str, Any],
    changed_by: str = "MUI-SYSTEM",
    note: str | None = None,
) -> dict[str, Any]:
    """Move DETECTED event to VERIFIED."""

    return transition_event(
        event,
        "VERIFIED",
        changed_by=changed_by,
        note=note
        or "Incident confirmed by additional evidence.",
    )


def prioritize_event(
    event: dict[str, Any],
    changed_by: str = "MUI-SYSTEM",
    note: str | None = None,
) -> dict[str, Any]:
    """Move VERIFIED event to PRIORITIZED."""

    return transition_event(
        event,
        "PRIORITIZED",
        changed_by=changed_by,
        note=note
        or "Event priority determined by the decision engine.",
    )


def assign_event(
    event: dict[str, Any],
    assigned_to: str,
    changed_by: str = "MUI-AUTHORITY",
    note: str | None = None,
) -> dict[str, Any]:
    """
    Assign an event to a maintenance team,
    authority, or operational unit.
    """

    updated_event = transition_event(
        event,
        "ASSIGNED",
        changed_by=changed_by,
        note=note
        or f"Assigned to {assigned_to}.",
    )

    updated_event["assigned_to"] = (
        assigned_to
    )

    updated_event["assigned_at"] = (
        updated_event.get(
            "status_updated_at"
        )
    )

    return updated_event


def start_work(
    event: dict[str, Any],
    changed_by: str = "MUI-AUTHORITY",
    note: str | None = None,
) -> dict[str, Any]:
    """Move an assigned event into active work."""

    return transition_event(
        event,
        "IN_PROGRESS",
        changed_by=changed_by,
        note=note
        or "Corrective work has started.",
    )


def resolve_event(
    event: dict[str, Any],
    changed_by: str = "MUI-AUTHORITY",
    note: str | None = None,
    resolution_evidence: str | None = None,
) -> dict[str, Any]:
    """Mark an event as resolved."""

    updated_event = transition_event(
        event,
        "RESOLVED",
        changed_by=changed_by,
        note=note
        or "Incident resolved.",
    )

    updated_event["resolved_at"] = (
        updated_event.get(
            "status_updated_at"
        )
    )

    if resolution_evidence:
        updated_event[
            "resolution_evidence"
        ] = resolution_evidence

    return updated_event


def get_lifecycle_summary(
    events: list[dict[str, Any]],
) -> dict[str, int]:
    """Return counts for each lifecycle state."""

    summary = {
        status: 0
        for status in VALID_STATUSES
    }

    for event in events:
        status = normalize_status(
            event.get(
                "status",
                "DETECTED",
            )
        )

        summary[status] += 1

    return summary


def initialize_event_lifecycle(
    event: dict[str, Any],
) -> dict[str, Any]:
    """
    Prepare an event for lifecycle management.

    Existing status is preserved.
    """

    updated_event = dict(
        event
    )

    status = normalize_status(
        updated_event.get(
            "status",
            "DETECTED",
        )
    )

    updated_event["status"] = status

    if "status_history" not in updated_event:
        updated_event["status_history"] = [
            {
                "from": None,
                "to": status,
                "timestamp": (
                    updated_event.get(
                        "timestamp"
                    )
                    or _timestamp()
                ),
                "changed_by": "MUI-SYSTEM",
                "note": "Event lifecycle initialized.",
            }
        ]

    return normalize_event(
        updated_event
    )


def self_test() -> None:
    """Run lifecycle tests."""

    print(
        "EVENT LIFECYCLE SELF-TEST"
    )

    print(
        "----------------------------------------"
    )

    event = {
        "event_id": "P-LIFE-001",
        "event_type": "pothole",
        "status": "DETECTED",
        "severity": "HIGH",
        "priority": "HIGH",
        "detection_count": 2,
        "confidence": 0.82,
        "highest_confidence": 0.82,
        "latitude": 13.3161,
        "longitude": 75.7720,
        "bus_id": "BUS-01",
        "route_id": "ROUTE-01",
        "reporting_buses": [
            "BUS-01",
            "BUS-02",
        ],
        "timestamp": _timestamp(),
    }

    event = initialize_event_lifecycle(
        event
    )

    assert event["status"] == "DETECTED"

    print(
        "Lifecycle initialization: PASSED"
    )

    event = verify_event(
        event,
        changed_by="BUS-FUSION",
    )

    assert event["status"] == "VERIFIED"

    print(
        "Detected → Verified: PASSED"
    )

    event = prioritize_event(
        event,
        changed_by="MUI-PRIORITY",
    )

    assert event["status"] == "PRIORITIZED"

    print(
        "Verified → Prioritized: PASSED"
    )

    event = assign_event(
        event,
        assigned_to="Road Maintenance Team",
    )

    assert event["status"] == "ASSIGNED"

    assert (
        event["assigned_to"]
        == "Road Maintenance Team"
    )

    print(
        "Prioritized → Assigned: PASSED"
    )

    event = start_work(
        event
    )

    assert event["status"] == "IN_PROGRESS"

    print(
        "Assigned → In Progress: PASSED"
    )

    event = resolve_event(
        event,
        resolution_evidence="repair_photo.jpg",
    )

    assert event["status"] == "RESOLVED"

    assert (
        event["resolution_evidence"]
        == "repair_photo.jpg"
    )

    print(
        "In Progress → Resolved: PASSED"
    )

    assert len(
        event["status_history"]
    ) >= 6

    print(
        "Lifecycle history: PASSED"
    )

    summary = get_lifecycle_summary(
        [
            event
        ]
    )

    assert (
        summary["RESOLVED"] == 1
    )

    print(
        "Lifecycle summary: PASSED"
    )

    assert not can_transition(
        "DETECTED",
        "RESOLVED",
    )

    print(
        "Invalid transition protection: PASSED"
    )

    print(
        "----------------------------------------"
    )

    print(
        "SELF-TEST PASSED"
    )


if __name__ == "__main__":
    self_test()