from __future__ import annotations

from typing import Any

from event_engine import normalize_event
from event_fusion import fuse_events, get_fusion_summary
from event_lifecycle import (
    get_lifecycle_summary,
    initialize_event_lifecycle,
    prioritize_event,
    verify_event,
)
from recommendation_engine import attach_recommendation


def _reporting_buses(
    event: dict[str, Any],
) -> list[str]:
    buses = event.get(
        "reporting_buses",
        [],
    )

    if not isinstance(
        buses,
        list,
    ):
        buses = []

    unique_buses = list(
        dict.fromkeys(
            str(bus)
            for bus in buses
            if bus
        )
    )

    bus_id = event.get(
        "bus_id"
    )

    if bus_id:
        bus_id = str(bus_id)

        if bus_id not in unique_buses:
            unique_buses.append(
                bus_id
            )

    return unique_buses


def _restore_fleet_fields(
    event: dict[str, Any],
    source_event: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(event)

    buses = _reporting_buses(
        source_event
    )

    updated[
        "reporting_buses"
    ] = buses

    updated[
        "reporting_bus_count"
    ] = len(buses)

    if buses:
        updated[
            "bus_id"
        ] = buses[0]

    return updated


def process_event(
    event: dict[str, Any],
) -> dict[str, Any]:
    normalized = normalize_event(
        event
    )

    initialized = initialize_event_lifecycle(
        normalized
    )

    return attach_recommendation(
        initialized
    )


def process_fleet_events(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Complete fleet-intelligence pipeline:

    Multiple detections
            ↓
       Normalization
            ↓
        Event Fusion
            ↓
     Multi-bus Verify
            ↓
         Priority
            ↓
      Recommendation
    """

    if not events:
        return []

    normalized_events = [
        normalize_event(event)
        for event in events
    ]

    fused_events = fuse_events(
        normalized_events
    )

    processed_events: list[
        dict[str, Any]
    ] = []

    for fused_event in fused_events:

        fleet_buses = _reporting_buses(
            fused_event
        )

        initialized = initialize_event_lifecycle(
            fused_event
        )

        initialized = _restore_fleet_fields(
            initialized,
            fused_event,
        )

        current_status = str(
            initialized.get(
                "status",
                "DETECTED",
            )
        ).upper()

        # ----------------------------------------------------
        # MULTI-BUS VERIFICATION
        # ----------------------------------------------------

        if (
            len(fleet_buses) >= 2
            and current_status == "DETECTED"
        ):
            initialized = verify_event(
                initialized,
                changed_by="MUI-FLEET-FUSION",
                note=(
                    f"Incident confirmed by "
                    f"{len(fleet_buses)} "
                    f"reporting buses."
                ),
            )

            initialized = _restore_fleet_fields(
                initialized,
                fused_event,
            )

            current_status = str(
                initialized.get(
                    "status",
                    "DETECTED",
                )
            ).upper()

        # ----------------------------------------------------
        # PRIORITIZATION
        # ----------------------------------------------------

        priority = str(
            initialized.get(
                "priority",
                "LOW",
            )
        ).upper()

        if (
            current_status == "VERIFIED"
            and priority
            in {
                "CRITICAL",
                "HIGH",
                "MEDIUM",
            }
        ):
            initialized = prioritize_event(
                initialized,
                changed_by="MUI-PRIORITY-ENGINE",
                note=(
                    f"{priority} priority "
                    f"assigned from event context."
                ),
            )

            initialized = _restore_fleet_fields(
                initialized,
                fused_event,
            )

        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------

        recommended = attach_recommendation(
            initialized
        )

        recommended = _restore_fleet_fields(
            recommended,
            fused_event,
        )

        processed_events.append(
            recommended
        )

    return processed_events


def get_system_summary(
    raw_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Generate the complete MUI dashboard summary.

    Important:
    A multi-bus incident counts as VERIFIED even if its
    current lifecycle status has already advanced to
    PRIORITIZED.
    """

    fleet_events = process_fleet_events(
        raw_events
    )

    fusion_summary = get_fusion_summary(
        raw_events
    )

    lifecycle_summary = get_lifecycle_summary(
        fleet_events
    )

    event_types: dict[str, int] = {}

    for event in fleet_events:

        event_type = str(
            event.get(
                "event_type",
                "unknown",
            )
        ).lower()

        event_types[event_type] = (
            event_types.get(
                event_type,
                0,
            )
            + 1
        )

    priority_counts = {
        "CRITICAL": 0,
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0,
        "NONE": 0,
    }

    for event in fleet_events:

        priority = str(
            event.get(
                "priority",
                "LOW",
            )
        ).upper()

        if priority not in priority_counts:
            priority = "NONE"

        priority_counts[
            priority
        ] += 1

    # --------------------------------------------------------
    # IMPORTANT FIX
    # --------------------------------------------------------
    # Multi-bus incidents are verified incidents even when
    # they have progressed to PRIORITIZED.
    verified_incidents = (
        fusion_summary[
            "multi_bus_incidents"
        ]
    )

    prioritized_incidents = sum(
        1
        for event in fleet_events
        if str(
            event.get(
                "status",
                "",
            )
        ).upper()
        == "PRIORITIZED"
    )

    return {
        "raw_detections": fusion_summary[
            "raw_detections"
        ],

        "unique_incidents": fusion_summary[
            "unique_incidents"
        ],

        "merged_detections": fusion_summary[
            "merged_detections"
        ],

        "multi_bus_incidents": fusion_summary[
            "multi_bus_incidents"
        ],

        "verified_incidents": verified_incidents,

        "prioritized_incidents": prioritized_incidents,

        "reporting_buses": fusion_summary[
            "reporting_buses"
        ],

        "event_types": event_types,

        "priority_counts": priority_counts,

        "lifecycle": lifecycle_summary,
    }


def run_demo_pipeline() -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    demo_events = [
        {
            "event_id": "P-BUS01-001",
            "event_type": "pothole",
            "status": "DETECTED",
            "severity": "MEDIUM",
            "priority": "MEDIUM",
            "detection_count": 1,
            "confidence": 0.72,
            "highest_confidence": 0.72,
            "latitude": 13.3161,
            "longitude": 75.7720,
            "bus_id": "BUS-01",
            "route_id": "ROUTE-01",
            "reporting_buses": [
                "BUS-01"
            ],
            "evidence_image": (
                "pothole_bus01.jpg"
            ),
        },
        {
            "event_id": "P-BUS02-001",
            "event_type": "pothole",
            "status": "DETECTED",
            "severity": "MEDIUM",
            "priority": "MEDIUM",
            "detection_count": 1,
            "confidence": 0.81,
            "highest_confidence": 0.81,
            "latitude": 13.3164,
            "longitude": 75.7721,
            "bus_id": "BUS-02",
            "route_id": "ROUTE-02",
            "reporting_buses": [
                "BUS-02"
            ],
            "evidence_image": (
                "pothole_bus02.jpg"
            ),
        },
        {
            "event_id": "T-BUS03-001",
            "event_type": "traffic",
            "status": "DETECTED",
            "severity": "HIGH",
            "priority": "HIGH",
            "vehicle_count": 18,
            "traffic_level": "HIGH",
            "congestion": 0.90,
            "confidence": 0.86,
            "highest_confidence": 0.92,
            "detection_count": 1,
            "latitude": 13.3215,
            "longitude": 75.7845,
            "bus_id": "BUS-03",
            "route_id": "ROUTE-03",
            "reporting_buses": [
                "BUS-03"
            ],
        },
    ]

    processed = process_fleet_events(
        demo_events
    )

    summary = get_system_summary(
        demo_events
    )

    return processed, summary


def self_test() -> None:

    print(
        "MUI PIPELINE SELF-TEST"
    )

    print(
        "----------------------------------------"
    )

    events, summary = run_demo_pipeline()

    assert len(events) == 2

    print(
        "Multi-event processing: PASSED"
    )

    pothole_events = [
        event
        for event in events
        if event.get(
            "event_type"
        ) == "pothole"
    ]

    traffic_events = [
        event
        for event in events
        if event.get(
            "event_type"
        ) == "traffic"
    ]

    assert len(
        pothole_events
    ) == 1

    assert len(
        traffic_events
    ) == 1

    print(
        "Event-type processing: PASSED"
    )

    fused_pothole = pothole_events[0]

    assert (
        fused_pothole[
            "reporting_bus_count"
        ] == 2
    )

    assert (
        len(
            fused_pothole[
                "reporting_buses"
            ]
        ) == 2
    )

    print(
        "Multi-bus fusion through pipeline: PASSED"
    )

    assert (
        fused_pothole[
            "status"
        ] == "PRIORITIZED"
    )

    print(
        "Multi-bus verification + prioritization: PASSED"
    )

    assert (
        "recommendation"
        in fused_pothole
    )

    assert (
        "recommended_action"
        in fused_pothole
    )

    print(
        "Recommendation integration: PASSED"
    )

    assert (
        summary[
            "unique_incidents"
        ] == 2
    )

    assert (
        summary[
            "multi_bus_incidents"
        ] == 1
    )

    assert (
        summary[
            "verified_incidents"
        ] == 1
    )

    assert (
        summary[
            "prioritized_incidents"
        ] == 1
    )

    assert (
        summary[
            "reporting_buses"
        ] == 3
    )

    print(
        "System summary: PASSED"
    )

    print(
        "----------------------------------------"
    )

    print(
        "SELF-TEST PASSED"
    )


if __name__ == "__main__":
    self_test()