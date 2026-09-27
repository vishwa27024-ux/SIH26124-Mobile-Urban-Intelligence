from __future__ import annotations

from typing import Any


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_level(
    value: Any,
    default: str = "LOW",
) -> str:
    value = str(
        value if value is not None else default
    ).strip().upper()

    return value or default


def _get_reporting_bus_count(
    event: dict[str, Any],
) -> int:
    buses = event.get(
        "reporting_buses",
        [],
    )

    if isinstance(
        buses,
        list,
    ):
        unique_buses = {
            str(bus)
            for bus in buses
            if bus
        }

        if unique_buses:
            return len(unique_buses)

    return max(
        1,
        _safe_int(
            event.get(
                "reporting_bus_count",
                1,
            ),
            1,
        ),
    )


def traffic_recommendation(
    event: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate a rule-based operational recommendation
    for a traffic event.

    This is explainable decision support, not generative AI.
    """

    vehicle_count = _safe_int(
        event.get(
            "vehicle_count",
            0,
        )
    )

    traffic_level = _normalize_level(
        event.get(
            "traffic_level",
            "LOW",
        )
    )

    severity = _normalize_level(
        event.get(
            "severity",
            traffic_level,
        )
    )

    priority = _normalize_level(
        event.get(
            "priority",
            severity,
        )
    )

    congestion = _safe_float(
        event.get(
            "congestion",
            0.0,
        )
    )

    reporting_buses = _get_reporting_bus_count(
        event
    )

    detection_count = max(
        1,
        _safe_int(
            event.get(
                "detection_count",
                1,
            ),
            1,
        ),
    )

    immediate_actions: list[str] = []
    long_term_actions: list[str] = []

    if (
        traffic_level == "HIGH"
        or severity == "HIGH"
        or vehicle_count >= 15
    ):
        primary_action = (
            "Deploy traffic-control measures "
            "and conduct junction-level intervention"
        )

        immediate_actions = [
            "Deploy traffic personnel or temporary control measures",
            "Optimize signal timing for the affected approach",
            "Monitor congestion from additional bus observations",
        ]

        long_term_actions = [
            "Evaluate junction redesign and channelization",
            "Assess corridor capacity and recurring congestion patterns",
            "Consider grade-separated or alternate-corridor intervention where justified",
        ]

        reason = (
            f"High traffic intensity detected with "
            f"{vehicle_count} vehicles and "
            f"{reporting_buses} reporting bus(es)."
        )

    elif (
        traffic_level == "MEDIUM"
        or severity == "MEDIUM"
        or vehicle_count >= 6
    ):
        primary_action = (
            "Optimize traffic flow and monitor the corridor"
        )

        immediate_actions = [
            "Review signal timing and lane utilization",
            "Monitor the location through additional fleet observations",
            "Check for temporary obstruction or bottleneck conditions",
        ]

        long_term_actions = [
            "Evaluate intersection channelization",
            "Study recurring traffic patterns by time and route",
            "Consider targeted corridor-capacity improvements",
        ]

        reason = (
            f"Moderate traffic intensity detected with "
            f"{vehicle_count} vehicles."
        )

    else:
        primary_action = (
            "Continue monitoring"
        )

        immediate_actions = [
            "Continue fleet-based monitoring",
            "Track changes in vehicle density",
        ]

        long_term_actions = [
            "Use repeated observations to identify recurring congestion",
        ]

        reason = (
            f"Low traffic intensity detected with "
            f"{vehicle_count} vehicles."
        )

    if reporting_buses >= 2:
        immediate_actions.append(
            "Cross-check the incident with multiple bus observations"
        )

    if detection_count >= 3:
        immediate_actions.append(
            "Increase confidence through repeated fleet detections"
        )

    return {
        "recommendation_type": "traffic",
        "primary_action": primary_action,
        "reason": reason,
        "immediate_actions": immediate_actions,
        "long_term_actions": long_term_actions,
        "priority": priority,
        "vehicle_count": vehicle_count,
        "congestion_score": round(
            min(
                max(
                    congestion,
                    0.0,
                ),
                1.0,
            ),
            2,
        ),
        "reporting_bus_count": reporting_buses,
        "detection_count": detection_count,
        "decision_basis": [
            "vehicle_count",
            "traffic_level",
            "severity",
            "priority",
            "fleet_confirmation",
        ],
    }


def pothole_recommendation(
    event: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate an operational recommendation for a pothole
    or road-hazard event.
    """

    detection_count = _safe_int(
        event.get(
            "detection_count",
            0,
        )
    )

    confidence = _safe_float(
        event.get(
            "highest_confidence",
            event.get(
                "confidence",
                0.0,
            ),
        )
    )

    severity = _normalize_level(
        event.get(
            "severity",
            "NONE",
        ),
        "NONE",
    )

    priority = _normalize_level(
        event.get(
            "priority",
            severity,
        ),
        "NONE",
    )

    reporting_buses = _get_reporting_bus_count(
        event
    )

    evidence = event.get(
        "evidence_image",
        event.get(
            "evidence"
        ),
    )

    immediate_actions: list[str] = []
    long_term_actions: list[str] = []

    if priority == "CRITICAL":
        primary_action = (
            "Initiate urgent road inspection and repair response"
        )

        immediate_actions = [
            "Inspect the affected road segment immediately",
            "Place temporary warning or traffic-safety measures where required",
            "Notify the responsible road-maintenance authority",
        ]

        long_term_actions = [
            "Complete permanent road-surface repair",
            "Inspect the surrounding road segment for related defects",
            "Record the repair outcome for future fleet verification",
        ]

        reason = (
            f"Critical road hazard supported by "
            f"{detection_count} detection(s) and "
            f"{reporting_buses} reporting bus(es)."
        )

    elif (
        priority == "HIGH"
        or severity == "HIGH"
        or (
            detection_count >= 2
            and confidence >= 0.70
        )
    ):
        primary_action = (
            "Schedule priority road inspection and repair"
        )

        immediate_actions = [
            "Verify the road defect at the reported location",
            "Prioritize the location for maintenance scheduling",
            "Monitor the location through additional bus observations",
        ]

        long_term_actions = [
            "Complete permanent pothole repair",
            "Inspect nearby pavement for additional damage",
        ]

        reason = (
            f"High-severity road hazard detected with "
            f"confidence {confidence:.2f}."
        )

    elif (
        priority == "MEDIUM"
        or severity == "MEDIUM"
        or (
            detection_count >= 1
            and confidence >= 0.60
        )
    ):
        primary_action = (
            "Verify the road hazard and schedule maintenance"
        )

        immediate_actions = [
            "Verify the reported location",
            "Add the location to the maintenance queue",
            "Continue fleet observation for confirmation",
        ]

        long_term_actions = [
            "Repair the pavement defect",
            "Monitor recurrence after maintenance",
        ]

        reason = (
            f"Medium-confidence road hazard detected "
            f"with confidence {confidence:.2f}."
        )

    elif detection_count > 0:
        primary_action = (
            "Verify the detected road hazard"
        )

        immediate_actions = [
            "Collect additional visual evidence",
            "Verify the road location before assigning maintenance",
        ]

        long_term_actions = [
            "Repair the defect if confirmed",
            "Continue monitoring the road segment",
        ]

        reason = (
            f"Potential road hazard detected with "
            f"confidence {confidence:.2f}."
        )

    else:
        primary_action = (
            "Continue monitoring"
        )

        immediate_actions = [
            "Continue road-condition monitoring",
        ]

        long_term_actions = [
            "Reassess the location when new fleet evidence becomes available",
        ]

        reason = (
            "No confirmed pothole detection is currently available."
        )

    if reporting_buses >= 2:
        immediate_actions.append(
            "Use multi-bus confirmation to strengthen incident verification"
        )

    if evidence:
        immediate_actions.append(
            "Retain the visual evidence with the incident record"
        )

    return {
        "recommendation_type": "pothole",
        "primary_action": primary_action,
        "reason": reason,
        "immediate_actions": immediate_actions,
        "long_term_actions": long_term_actions,
        "priority": priority,
        "confidence": round(
            max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            ),
            4,
        ),
        "detection_count": detection_count,
        "reporting_bus_count": reporting_buses,
        "visual_evidence": evidence,
        "decision_basis": [
            "detection_count",
            "confidence",
            "severity",
            "priority",
            "fleet_confirmation",
            "visual_evidence",
        ],
    }


def generate_recommendation(
    event: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate the appropriate recommendation based on
    the unified event type.
    """

    event_type = str(
        event.get(
            "event_type",
            "",
        )
    ).strip().lower()

    if event_type in {
        "pothole",
        "road_hazard",
        "hazard",
    }:
        return pothole_recommendation(
            event
        )

    if event_type == "traffic":
        return traffic_recommendation(
            event
        )

    return {
        "recommendation_type": "general",
        "primary_action": (
            "Verify the event and assess the affected location"
        ),
        "reason": (
            "The event type is not covered by a specialized recommendation rule."
        ),
        "immediate_actions": [
            "Verify the event",
            "Review available evidence",
            "Assess the affected location",
        ],
        "long_term_actions": [
            "Record the appropriate corrective action",
            "Monitor the event location",
        ],
        "priority": _normalize_level(
            event.get(
                "priority",
                "LOW",
            ),
            "LOW",
        ),
        "decision_basis": [
            "event_type",
            "priority",
        ],
    }


def attach_recommendation(
    event: dict[str, Any],
) -> dict[str, Any]:
    """
    Attach recommendation information directly to an
    event without changing the original event type.
    """

    updated_event = dict(
        event
    )

    recommendation = generate_recommendation(
        updated_event
    )

    updated_event["recommendation"] = (
        recommendation
    )

    updated_event["recommendation_reason"] = (
        recommendation.get(
            "reason",
            "",
        )
    )

    updated_event["immediate_actions"] = (
        recommendation.get(
            "immediate_actions",
            [],
        )
    )

    updated_event["long_term_actions"] = (
        recommendation.get(
            "long_term_actions",
            [],
        )
    )

    updated_event["recommended_action"] = (
        recommendation.get(
            "primary_action",
            "",
        )
    )

    return updated_event


def self_test() -> None:
    print(
        "RECOMMENDATION ENGINE SELF-TEST"
    )

    print(
        "----------------------------------------"
    )

    traffic_event = {
        "event_id": "T-001",
        "event_type": "traffic",
        "traffic_level": "HIGH",
        "severity": "HIGH",
        "priority": "HIGH",
        "vehicle_count": 18,
        "congestion": 0.90,
        "detection_count": 3,
        "reporting_buses": [
            "BUS-01",
            "BUS-02",
            "BUS-03",
        ],
    }

    traffic_result = traffic_recommendation(
        traffic_event
    )

    assert (
        traffic_result[
            "recommendation_type"
        ] == "traffic"
    )

    assert (
        traffic_result[
            "priority"
        ] == "HIGH"
    )

    assert (
        traffic_result[
            "reporting_bus_count"
        ] == 3
    )

    assert (
        len(
            traffic_result[
                "immediate_actions"
            ]
        ) > 0
    )

    print(
        "Traffic recommendation: PASSED"
    )

    pothole_event = {
        "event_id": "P-001",
        "event_type": "pothole",
        "severity": "HIGH",
        "priority": "CRITICAL",
        "detection_count": 3,
        "confidence": 0.86,
        "highest_confidence": 0.86,
        "reporting_buses": [
            "BUS-01",
            "BUS-03",
            "BUS-05",
        ],
        "evidence_image": "pothole.jpg",
    }

    pothole_result = pothole_recommendation(
        pothole_event
    )

    assert (
        pothole_result[
            "recommendation_type"
        ] == "pothole"
    )

    assert (
        pothole_result[
            "priority"
        ] == "CRITICAL"
    )

    assert (
        pothole_result[
            "reporting_bus_count"
        ] == 3
    )

    assert (
        pothole_result[
            "visual_evidence"
        ] == "pothole.jpg"
    )

    print(
        "Pothole recommendation: PASSED"
    )

    attached = attach_recommendation(
        pothole_event
    )

    assert (
        "recommendation"
        in attached
    )

    assert (
        attached[
            "recommended_action"
        ]
    )

    assert (
        isinstance(
            attached[
                "immediate_actions"
            ],
            list,
        )
    )

    assert (
        isinstance(
            attached[
                "long_term_actions"
            ],
            list,
        )
    )

    print(
        "Recommendation attachment: PASSED"
    )

    general_event = {
        "event_id": "G-001",
        "event_type": "unknown",
        "priority": "LOW",
    }

    general_result = generate_recommendation(
        general_event
    )

    assert (
        general_result[
            "recommendation_type"
        ] == "general"
    )

    print(
        "Fallback recommendation: PASSED"
    )

    print(
        "----------------------------------------"
    )

    print(
        "SELF-TEST PASSED"
    )


if __name__ == "__main__":
    self_test()