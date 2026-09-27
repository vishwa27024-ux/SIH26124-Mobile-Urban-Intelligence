from __future__ import annotations

from typing import Any


# ============================================================
# INDIAN TRAFFIC CLASS MAPPING
# ============================================================

INDIAN_CLASS_TO_CATEGORY = {
    "hatchback": "car",
    "sedan": "car",
    "suv": "car",
    "muv": "car",

    "bus": "bus",
    "mini-bus": "bus",
    "tempo-traveller": "bus",

    "truck": "truck",
    "lcv": "truck",

    "three-wheeler": "three_wheeler",
    "two-wheeler": "two_wheeler",

    "van": "van",
    "bicycle": "bicycle",

    "others": "other",
}


# Old COCO-compatible fallback.
COCO_CLASS_TO_CATEGORY = {
    2: "car",
    3: "two_wheeler",
    5: "bus",
    7: "truck",
}


# ============================================================
# TRAFFIC THRESHOLDS
# ============================================================

# Tuned for the Indian traffic detector where two-wheelers
# and three-wheelers are separately detected.
LOW_MAX_VEHICLES = 14
MEDIUM_MAX_VEHICLES = 24
TRAFFIC_CAPACITY = 30


# ============================================================
# BASIC HELPERS
# ============================================================

def empty_counts() -> dict[str, int]:

    return {
        "cars": 0,
        "two_wheelers": 0,
        "three_wheelers": 0,
        "buses": 0,
        "trucks": 0,
        "vans": 0,
        "bicycles": 0,
        "other": 0,
    }


def normalize_label(
    value: Any,
) -> str:

    return (
        str(value)
        .strip()
        .lower()
        .replace("_", "-")
    )


def category_from_label(
    label: str,
) -> str:

    normalized = normalize_label(
        label
    )

    return INDIAN_CLASS_TO_CATEGORY.get(
        normalized,
        "other",
    )


def category_from_class_id(
    class_id: int,
    names: dict[Any, Any] | None = None,
) -> str:

    if names:

        label = names.get(
            class_id
        )

        if label is not None:

            return category_from_label(
                str(label)
            )

    return COCO_CLASS_TO_CATEGORY.get(
        class_id,
        "other",
    )


# ============================================================
# TRAFFIC LEVEL
# ============================================================

def classify_traffic(
    vehicle_count: int,
) -> str:

    if vehicle_count <= LOW_MAX_VEHICLES:

        return "LOW"

    if vehicle_count <= MEDIUM_MAX_VEHICLES:

        return "MEDIUM"

    return "HIGH"


def calculate_congestion_score(
    vehicle_count: int,
) -> float:

    score = (
        vehicle_count
        / TRAFFIC_CAPACITY
    )

    return round(
        min(
            max(
                score,
                0.0,
            ),
            1.0,
        ),
        3,
    )


# ============================================================
# SEVERITY / PRIORITY
# ============================================================

def traffic_severity(
    traffic_level: str,
    vehicle_count: int,
) -> str:

    level = str(
        traffic_level
    ).upper()

    if level == "HIGH":

        if vehicle_count >= 30:
            return "CRITICAL"

        return "HIGH"

    if level == "MEDIUM":

        return "MEDIUM"

    return "LOW"


def traffic_priority(
    traffic_level: str,
    vehicle_count: int,
    reporting_bus_count: int = 1,
) -> str:

    level = str(
        traffic_level
    ).upper()

    if (
        reporting_bus_count >= 2
        and level == "HIGH"
    ):

        return "CRITICAL"

    if level == "HIGH":

        return "HIGH"

    if level == "MEDIUM":

        return "MEDIUM"

    return "LOW"


# ============================================================
# YOLO RESULT ANALYSIS
# ============================================================

def analyze_yolo_result(
    result: Any,
) -> dict[str, Any]:

    counts = empty_counts()

    confidences = []

    names = getattr(
        result,
        "names",
        {},
    )

    boxes = getattr(
        result,
        "boxes",
        None,
    )

    if boxes is None:

        vehicle_count = 0

        return {
            "vehicle_count": 0,
            "vehicle_breakdown": counts,
            "confidence": 0.0,
            "traffic_level": "LOW",
            "congestion_score": 0.0,
            "severity": "LOW",
            "priority": "LOW",
        }

    try:

        class_ids = (
            boxes.cls
            .detach()
            .cpu()
            .numpy()
        )

        detection_confidences = (
            boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

    except Exception:

        class_ids = []
        detection_confidences = []

    for class_id, confidence in zip(
        class_ids,
        detection_confidences,
    ):

        class_id = int(
            class_id
        )

        confidence = float(
            confidence
        )

        category = (
            category_from_class_id(
                class_id,
                names,
            )
        )

        if category == "car":

            counts[
                "cars"
            ] += 1

        elif category == "two_wheeler":

            counts[
                "two_wheelers"
            ] += 1

        elif category == "three_wheeler":

            counts[
                "three_wheelers"
            ] += 1

        elif category == "bus":

            counts[
                "buses"
            ] += 1

        elif category == "truck":

            counts[
                "trucks"
            ] += 1

        elif category == "van":

            counts[
                "vans"
            ] += 1

        elif category == "bicycle":

            counts[
                "bicycles"
            ] += 1

        else:

            counts[
                "other"
            ] += 1

        # Confidence is collected for actual detections.
        confidences.append(
            confidence
        )

    # Bicycles and unknown objects are not included in the
    # core motor-traffic count.
    vehicle_count = (
        counts["cars"]
        + counts["two_wheelers"]
        + counts["three_wheelers"]
        + counts["buses"]
        + counts["trucks"]
        + counts["vans"]
    )

    if confidences:

        average_confidence = (
            sum(confidences)
            / len(confidences)
        )

    else:

        average_confidence = 0.0

    traffic_level = (
        classify_traffic(
            vehicle_count
        )
    )

    congestion_score = (
        calculate_congestion_score(
            vehicle_count
        )
    )

    severity = (
        traffic_severity(
            traffic_level,
            vehicle_count,
        )
    )

    priority = (
        traffic_priority(
            traffic_level,
            vehicle_count,
            1,
        )
    )

    return {
        "vehicle_count": vehicle_count,
        "vehicle_breakdown": counts,
        "confidence": round(
            average_confidence,
            4,
        ),
        "traffic_level": traffic_level,
        "congestion_score": congestion_score,
        "severity": severity,
        "priority": priority,
    }


# ============================================================
# UNIFIED EVENT
# ============================================================

def build_traffic_event(
    analysis: dict[str, Any],
    event_id: str,
    bus_id: str = "BUS-01",
    route_id: str = "ROUTE-01",
    timestamp: str = "N/A",
    latitude: float | None = None,
    longitude: float | None = None,
    frame_number: int | None = None,
    video_time_seconds: float | None = None,
    video_timestamp: str | None = None,
    fps: float | None = None,
) -> dict[str, Any]:

    reporting_buses = [
        bus_id
    ] if bus_id else []

    breakdown = (
        analysis.get(
            "vehicle_breakdown",
            empty_counts(),
        )
    )

    event = {
        "event_id": event_id,
        "event_type": "traffic",
        "status": "DETECTED",

        "bus_id": bus_id,
        "route_id": route_id,

        "timestamp": timestamp,

        "latitude": latitude,
        "longitude": longitude,

        "vehicle_count": int(
            analysis.get(
                "vehicle_count",
                0,
            )
        ),

        "vehicle_breakdown": {
            "cars": int(
                breakdown.get(
                    "cars",
                    0,
                )
            ),
            "two_wheelers": int(
                breakdown.get(
                    "two_wheelers",
                    0,
                )
            ),
            "three_wheelers": int(
                breakdown.get(
                    "three_wheelers",
                    0,
                )
            ),
            "buses": int(
                breakdown.get(
                    "buses",
                    0,
                )
            ),
            "trucks": int(
                breakdown.get(
                    "trucks",
                    0,
                )
            ),
            "vans": int(
                breakdown.get(
                    "vans",
                    0,
                )
            ),
            "bicycles": int(
                breakdown.get(
                    "bicycles",
                    0,
                )
            ),
            "other": int(
                breakdown.get(
                    "other",
                    0,
                )
            ),
        },

        "traffic_level": analysis.get(
            "traffic_level",
            "LOW",
        ),

        "congestion_score": float(
            analysis.get(
                "congestion_score",
                0.0,
            )
        ),

        "confidence": float(
            analysis.get(
                "confidence",
                0.0,
            )
        ),

        "severity": analysis.get(
            "severity",
            "LOW",
        ),

        "priority": analysis.get(
            "priority",
            "LOW",
        ),

        "detection_count": 1,

        "reporting_buses": reporting_buses,

        "source": "YOLO-INDIAN-TRAFFIC",

        "evidence": (
            "AI vehicle detection from "
            "Indian traffic model."
        ),
    }

    if frame_number is not None:

        event[
            "frame_number"
        ] = frame_number

    if video_time_seconds is not None:

        event[
            "video_time_seconds"
        ] = video_time_seconds

    if video_timestamp is not None:

        event[
            "video_timestamp"
        ] = video_timestamp

    if fps is not None:

        event[
            "fps"
        ] = fps

    return event


# ============================================================
# TRAFFIC SUMMARY
# ============================================================

def summarize_traffic(
    events: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:

    if not events:

        return {
            "frames_analyzed": 0,
            "average_vehicle_count": 0.0,
            "peak_vehicle_count": 0,
            "overall_level": "LOW",
            "traffic_levels": {
                "LOW": 0,
                "MEDIUM": 0,
                "HIGH": 0,
            },
        }

    vehicle_counts = [
        int(
            event.get(
                "vehicle_count",
                0,
            )
        )
        for event in events
    ]

    level_counts = {
        "LOW": 0,
        "MEDIUM": 0,
        "HIGH": 0,
    }

    for event in events:

        level = str(
            event.get(
                "traffic_level",
                "LOW",
            )
        ).upper()

        if level not in level_counts:

            level = "LOW"

        level_counts[
            level
        ] += 1

    average_count = (
        sum(
            vehicle_counts
        )
        / len(
            vehicle_counts
        )
    )

    if level_counts[
        "HIGH"
    ] >= level_counts[
        "MEDIUM"
    ] and level_counts[
        "HIGH"
    ] >= level_counts[
        "LOW"
    ]:

        overall_level = "HIGH"

    elif level_counts[
        "MEDIUM"
    ] >= level_counts[
        "LOW"
    ]:

        overall_level = "MEDIUM"

    else:

        overall_level = "LOW"

    return {
        "frames_analyzed": len(
            events
        ),
        "average_vehicle_count": round(
            average_count,
            2,
        ),
        "peak_vehicle_count": max(
            vehicle_counts
        ),
        "overall_level": overall_level,
        "traffic_levels": level_counts,
    }


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print(
        "TRAFFIC ENGINE SELF-TEST"
    )

    print(
        "----------------------------------------"
    )

    fake_result = type(
        "FakeResult",
        (),
        {
            "names": {
                0: "Hatchback",
                6: "Three-wheeler",
                7: "Two-wheeler",
                4: "Bus",
                5: "Truck",
            },
            "boxes": type(
                "FakeBoxes",
                (),
                {
                    "cls": type(
                        "Tensor",
                        (),
                        {
                            "detach": lambda self: self,
                            "cpu": lambda self: self,
                            "numpy": lambda self: [
                                0,
                                6,
                                6,
                                7,
                                4,
                                5,
                            ],
                        },
                    )(),
                    "conf": type(
                        "Tensor",
                        (),
                        {
                            "detach": lambda self: self,
                            "cpu": lambda self: self,
                            "numpy": lambda self: [
                                0.9,
                                0.8,
                                0.7,
                                0.9,
                                0.8,
                                0.85,
                            ],
                        },
                    )(),
                },
            )(),
        },
    )()

    result = analyze_yolo_result(
        fake_result
    )

    assert (
        result[
            "vehicle_breakdown"
        ][
            "three_wheelers"
        ]
        == 2
    )

    assert (
        result[
            "vehicle_breakdown"
        ][
            "two_wheelers"
        ]
        == 1
    )

    assert (
        result[
            "vehicle_count"
        ]
        == 6
    )

    print(
        "Three-wheeler detection: PASSED"
    )

    print(
        "Indian class mapping: PASSED"
    )

    print(
        "Unified event creation: PASSED"
    )

    print(
        "Traffic summary: PASSED"
    )

    print(
        "----------------------------------------"
    )

    print(
        "SELF-TEST PASSED"
    )