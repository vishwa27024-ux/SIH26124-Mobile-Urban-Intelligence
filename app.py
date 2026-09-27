from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any

import cv2
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from ultralytics import YOLO

from event_store import load_events
from event_lifecycle import initialize_event_lifecycle
from hazard_engine import (
    calculate_hazard_priority,
    calculate_hazard_severity,
)
from mui_pipeline import process_fleet_events
from recommendation_engine import generate_recommendation


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="MUI — Mobile Urban Intelligence",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# FILE PATHS
# ============================================================

TRAFFIC_EVENTS_FILE = (
    BASE_DIR / "traffic_events.json"
)

TRAFFIC_VIDEO = (
    BASE_DIR / "13020022_3840_2160_30fps.mp4"
)

AI_TRAFFIC_VIDEO = (
    BASE_DIR / "traffic_ai_h264.mp4"
)

AI_TRAFFIC_VIDEO_FALLBACK = (
    BASE_DIR / "traffic_ai_detected.mp4"
)

YOLO_TRAFFIC_MODEL = (
    BASE_DIR / "yolo11n.pt"
)

# Dedicated pothole-only model
POTHOLE_MODEL = (
    BASE_DIR / "weights" / "pothole_only.pt"
)

# Three separate Indian road-damage images
POTHOLE_HIGH_IMAGE = (
    BASE_DIR / "pothole_high.jpg"
)

POTHOLE_MEDIUM_IMAGE = (
    BASE_DIR / "pothole_medium.jpg"
)

POTHOLE_LOW_IMAGE = (
    BASE_DIR / "pothole_low.jpg"
)

POTHOLE_LATITUDE = 13.3185
POTHOLE_LONGITUDE = 75.7788


# ============================================================
# SESSION STATE
# ============================================================

st.session_state.setdefault(
    "page",
    "Monitor",
)

st.session_state.setdefault(
    "selected_map_id",
    None,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_int(
    value: Any,
    default: int = 0,
) -> int:

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def event_type(
    event: dict[str, Any],
) -> str:

    return str(
        event.get(
            "event_type",
            "unknown",
        )
    ).lower()


def event_priority(
    event: dict[str, Any],
) -> str:

    return str(
        event.get(
            "priority",
            "LOW",
        )
    ).upper()


def event_status(
    event: dict[str, Any],
) -> str:

    return str(
        event.get(
            "status",
            "DETECTED",
        )
    ).upper()


def reporting_buses(
    event: dict[str, Any],
) -> list[str]:

    buses = event.get(
        "reporting_buses",
        [],
    )

    if isinstance(
        buses,
        list,
    ):

        unique = list(
            dict.fromkeys(
                str(bus)
                for bus in buses
                if bus
            )
        )

        if unique:
            return unique

    bus_id = event.get(
        "bus_id"
    )

    if bus_id:
        return [str(bus_id)]

    return []


def haversine(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:

    radius = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dlat = math.radians(
        lat2 - lat1
    )

    dlon = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(
            dlat / 2
        ) ** 2
        +
        math.cos(p1)
        * math.cos(p2)
        * math.sin(
            dlon / 2
        ) ** 2
    )

    return (
        2
        * radius
        * math.asin(
            math.sqrt(a)
        )
    )


def image_is_valid(
    image_path: Path,
) -> bool:

    if not image_path.exists():
        return False

    try:

        image = cv2.imread(
            str(image_path)
        )

        return (
            image is not None
            and image.size > 0
        )

    except Exception:

        return False


def image_to_data_uri(
    image_path: Path,
) -> str | None:

    if not image_is_valid(
        image_path
    ):
        return None

    try:

        encoded = base64.b64encode(
            image_path.read_bytes()
        ).decode(
            "ascii"
        )

        return (
            "data:image/jpeg;base64,"
            f"{encoded}"
        )

    except OSError:

        return None


def get_vehicle_breakdown(
    event: dict[str, Any],
) -> dict[str, int]:
    """Return a normalized Indian-traffic vehicle breakdown.

    The traffic generator uses categories such as cars, three_wheelers,
    two_wheelers, vans, bicycles and other. Older event files may use
    singular names or COCO-style names, so all supported aliases are
    normalized here for the dashboard.
    """

    raw = (
        event.get(
            "vehicle_breakdown"
        )
        or event.get(
            "vehicle_counts"
        )
        or {}
    )

    if not isinstance(
        raw,
        dict,
    ):

        raw = {}

    def value(
        *keys: str,
    ) -> int:

        for key in keys:

            if key in raw:

                return safe_int(
                    raw.get(
                        key
                    ),
                    0,
                )

        return 0

    return {
        "Cars": value(
            "cars",
            "car",
        ),
        "Auto-rickshaws": value(
            "three_wheelers",
            "three_wheeler",
            "three-wheelers",
            "three-wheeler",
            "auto_rickshaws",
            "auto_rickshaw",
        ),
        "Two-wheelers": value(
            "two_wheelers",
            "two_wheeler",
            "motorcycles",
            "motorcycle",
            "scooters",
            "scooter",
        ),
        "Buses": value(
            "buses",
            "bus",
        ),
        "Trucks": value(
            "trucks",
            "truck",
        ),
        "Vans": value(
            "vans",
            "van",
        ),
        "Bicycles": value(
            "bicycles",
            "bicycle",
        ),
        "Other": value(
            "other",
            "others",
        ),
    }


def get_pedestrian_count(
    event: dict[str, Any],
) -> int:
    """Return the pedestrian/person count recorded by the AI generator."""

    return max(
        0,
        safe_int(
            event.get(
                "person_count",
                event.get(
                    "pedestrian_count",
                    0,
                ),
            ),
            0,
        ),
    )


def get_event_confidence(
    event: dict[str, Any],
) -> float:
    """Return the strongest available event confidence."""

    return safe_float(
        event.get(
            "highest_confidence",
            event.get(
                "confidence",
                event.get(
                    "average_confidence",
                    0.0,
                ),
            ),
        ),
        0.0,
    )


def inspect_video(
    video_path: Path,
) -> dict[str, Any]:
    """Read basic runtime properties of a video without decoding every frame."""

    result = {
        "exists": video_path.exists(),
        "opened": False,
        "frames": 0,
        "fps": 0.0,
        "duration": 0.0,
        "width": 0,
        "height": 0,
    }

    if not video_path.exists():
        return result

    capture = cv2.VideoCapture(
        str(video_path)
    )

    try:

        result["opened"] = bool(
            capture.isOpened()
        )

        if not result["opened"]:
            return result

        result["frames"] = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        result["fps"] = float(
            capture.get(
                cv2.CAP_PROP_FPS
            )
            or 0.0
        )

        result["width"] = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        result["height"] = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        if result["fps"] > 0:

            result["duration"] = (
                result["frames"]
                / result["fps"]
            )

        return result

    finally:

        capture.release()


# ============================================================
# TRAFFIC SUMMARY
# ============================================================

def load_traffic_summary():

    default = {
        "frames": 0,
        "frames_processed": 0,
        "frames_sampled": 0,
        "average_vehicle_count": 0.0,
        "peak_vehicle_count": 0,
        "overall_level": "LOW",
        "low_frames": 0,
        "medium_frames": 0,
        "high_frames": 0,
        "average_confidence": 0.0,
        "video_frames": 0,
        "video_fps": 0.0,
    }

    if not TRAFFIC_EVENTS_FILE.exists():

        return default

    try:

        data = json.loads(
            TRAFFIC_EVENTS_FILE.read_text(
                encoding="utf-8"
            )
        )

        summary = data.get(
            "summary",
            {},
        )

        levels = summary.get(
            "traffic_levels",
            {},
        )

        frames_processed = safe_int(
            summary.get(
                "frames_processed",
                summary.get(
                    "frames_analyzed",
                    0,
                ),
            )
        )

        frames_sampled = safe_int(
            summary.get(
                "frames_sampled",
                summary.get(
                    "frames_analyzed",
                    0,
                ),
            )
        )

        return {
            "frames": frames_processed,
            "frames_processed": frames_processed,
            "frames_sampled": frames_sampled,
            "average_vehicle_count": safe_float(
                summary.get(
                    "average_vehicle_count",
                    0.0,
                )
            ),
            "peak_vehicle_count": safe_int(
                summary.get(
                    "peak_vehicle_count",
                    0,
                )
            ),
            "overall_level": str(
                summary.get(
                    "overall_level",
                    "LOW",
                )
            ).upper(),
            "low_frames": safe_int(
                levels.get(
                    "LOW",
                    summary.get(
                        "low_frames",
                        0,
                    ),
                )
            ),
            "medium_frames": safe_int(
                levels.get(
                    "MEDIUM",
                    summary.get(
                        "medium_frames",
                        0,
                    ),
                )
            ),
            "high_frames": safe_int(
                levels.get(
                    "HIGH",
                    summary.get(
                        "high_frames",
                        0,
                    ),
                )
            ),
            "average_confidence": safe_float(
                summary.get(
                    "average_confidence",
                    0.0,
                )
            ),
            "video_frames": safe_int(
                summary.get(
                    "video_frames",
                    0,
                )
            ),
            "video_fps": safe_float(
                summary.get(
                    "video_fps",
                    0.0,
                )
            ),
        }

    except (
        OSError,
        json.JSONDecodeError,
    ):

        return default


TRAFFIC_SUMMARY = (
    load_traffic_summary()
)


# ============================================================
# LOAD REAL EVENTS
# ============================================================

BASE_EVENTS = load_events(
    prefer_generated=True
)

REAL_TRAFFIC_EVENTS = [
    event
    for event in BASE_EVENTS
    if event_type(event)
    == "traffic"
]

REAL_RAW_COUNT = len(
    REAL_TRAFFIC_EVENTS
)

REAL_VEHICLE_TOTALS = {
    "Cars": 0,
    "Auto-rickshaws": 0,
    "Two-wheelers": 0,
    "Buses": 0,
    "Trucks": 0,
    "Vans": 0,
    "Bicycles": 0,
    "Other": 0,
}

REAL_PEDESTRIAN_OBSERVATIONS = 0
REAL_PEDESTRIAN_FRAMES = 0

for real_event in REAL_TRAFFIC_EVENTS:

    breakdown = get_vehicle_breakdown(
        real_event
    )

    for category in REAL_VEHICLE_TOTALS:

        REAL_VEHICLE_TOTALS[
            category
        ] += breakdown.get(
            category,
            0,
        )

    people = get_pedestrian_count(
        real_event
    )

    REAL_PEDESTRIAN_OBSERVATIONS += people

    if people > 0:

        REAL_PEDESTRIAN_FRAMES += 1


AI_VIDEO_INFO = inspect_video(
    AI_TRAFFIC_VIDEO
)

AI_VIDEO_FALLBACK_INFO = inspect_video(
    AI_TRAFFIC_VIDEO_FALLBACK
)

ORIGINAL_VIDEO_INFO = inspect_video(
    TRAFFIC_VIDEO
)

AI_VIDEO_IS_HEALTHY = bool(
    AI_VIDEO_INFO.get("opened")
    and AI_VIDEO_INFO.get("frames", 0) > 1
    and AI_VIDEO_INFO.get("fps", 0.0) > 0
    and AI_VIDEO_INFO.get("duration", 0.0) > 0.1
)



# ============================================================
# SIMULATED GPS ROUTE
# ============================================================

def apply_prototype_route(
    events: list[
        dict[str, Any]
    ],
):

    updated_events = []

    route_lat_start = 13.3095
    route_lon_start = 75.7585

    route_lat_end = 13.3315
    route_lon_end = 75.7905

    max_time = 39.0

    for event in events:

        updated = dict(
            event
        )

        if event_type(event) != "traffic":

            updated_events.append(
                updated
            )

            continue

        video_time = safe_float(
            event.get(
                "video_time_seconds",
                0.0,
            )
        )

        ratio = min(
            max(
                video_time
                / max_time,
                0.0,
            ),
            1.0,
        )

        updated[
            "latitude"
        ] = (
            route_lat_start
            +
            (
                route_lat_end
                -
                route_lat_start
            )
            * ratio
        )

        updated[
            "longitude"
        ] = (
            route_lon_start
            +
            (
                route_lon_end
                -
                route_lon_start
            )
            * ratio
        )

        updated[
            "gps_mode"
        ] = "SIMULATED_ROUTE"

        updated_events.append(
            updated
        )

    return updated_events


BASE_EVENTS = (
    apply_prototype_route(
        BASE_EVENTS
    )
)


# ============================================================
# SIMULATED FLEET CONFIRMATIONS
# ============================================================

def build_demo_fleet_events(
    events: list[
        dict[str, Any]
    ],
):

    output = [
        dict(event)
        for event in events
    ]

    high_candidates = [
        event
        for event in output
        if (
            event_type(event)
            == "traffic"
            and str(
                event.get(
                    "traffic_level",
                    "",
                )
            ).upper()
            == "HIGH"
        )
    ]

    selected = high_candidates[:3]

    for original in selected:

        original_time = safe_float(
            original.get(
                "video_time_seconds",
                0.0,
            )
        )

        original_lat = safe_float(
            original.get(
                "latitude",
                0.0,
            )
        )

        original_lon = safe_float(
            original.get(
                "longitude",
                0.0,
            )
        )

        for (
            bus_id,
            route_id,
            offset,
        ) in (
            (
                "BUS-02",
                "ROUTE-02",
                0.00012,
            ),
            (
                "BUS-03",
                "ROUTE-03",
                -0.00010,
            ),
        ):

            simulated = dict(
                original
            )

            simulated[
                "event_id"
            ] = (
                f"{original.get(
                    'event_id'
                )}-{bus_id}"
            )

            simulated[
                "bus_id"
            ] = bus_id

            simulated[
                "route_id"
            ] = route_id

            simulated[
                "reporting_buses"
            ] = [
                bus_id
            ]

            simulated[
                "latitude"
            ] = (
                original_lat
                + offset
            )

            simulated[
                "longitude"
            ] = (
                original_lon
                + offset
            )

            simulated[
                "video_time_seconds"
            ] = (
                original_time
                + 1.0
            )

            simulated[
                "source"
            ] = (
                "SIMULATED-FLEET-CONFIRMATION"
            )

            simulated[
                "is_simulated"
            ] = True

            simulated[
                "simulation_note"
            ] = (
                "Prototype-only fleet confirmation."
            )

            output.append(
                simulated
            )

    return output


PIPELINE_INPUT_EVENTS = (
    build_demo_fleet_events(
        BASE_EVENTS
    )
)


# ============================================================
# UNIFIED MUI PIPELINE
# ============================================================

EVENTS = process_fleet_events(
    PIPELINE_INPUT_EVENTS
)

for event in EVENTS:

    buses = reporting_buses(
        event
    )

    if len(
        buses
    ) >= 2:

        event[
            "fleet_confirmation_mode"
        ] = "SIMULATED"

        event[
            "fleet_confirmation_note"
        ] = (
            "Only BUS-01 has real video evidence. "
            "Other bus confirmations are simulated."
        )


# ============================================================
# POTHOLE MODEL
# ============================================================

@st.cache_resource
def get_pothole_model():

    if not POTHOLE_MODEL.exists():

        return None

    try:

        return YOLO(
            str(
                POTHOLE_MODEL
            )
        )

    except Exception:

        return None


# ============================================================
# POTHOLE IMAGE INFERENCE
# ============================================================

def infer_pothole_image(
    model,
    image_path: Path,
):

    if not image_is_valid(
        image_path
    ):

        return {
            "ok": False,
            "path": image_path,
            "count": 0,
            "confidence": 0.0,
            "boxes": [],
            "annotated": None,
        }

    image = cv2.imread(
        str(
            image_path
        )
    )

    try:

        results = model.predict(
            source=image,
            conf=0.20,
            iou=0.45,
            verbose=False,
        )

    except Exception as error:

        return {
            "ok": False,
            "path": image_path,
            "count": 0,
            "confidence": 0.0,
            "boxes": [],
            "annotated": None,
            "error": str(
                error
            ),
        }

    annotated = image.copy()

    boxes = []

    for result in results:

        if result.boxes is None:

            continue

        xyxy = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

        for (
            box,
            confidence,
        ) in zip(
            xyxy,
            confidences,
        ):

            x1, y1, x2, y2 = [
                int(
                    max(
                        0,
                        value,
                    )
                )
                for value in box
            ]

            confidence = float(
                confidence
            )

            boxes.append(
                {
                    "box": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
                    "confidence": (
                        confidence
                    ),
                }
            )

            cv2.rectangle(
                annotated,
                (
                    x1,
                    y1,
                ),
                (
                    x2,
                    y2,
                ),
                (
                    0,
                    0,
                    255,
                ),
                4,
            )

            cv2.putText(
                annotated,
                (
                    f"Pothole "
                    f"{confidence * 100:.1f}%"
                ),
                (
                    x1,
                    max(
                        35,
                        y1 - 10,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (
                    0,
                    0,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

    annotated_path = (
        BASE_DIR
        / (
            "ai_pothole_annotated_"
            f"{image_path.stem}.jpg"
        )
    )

    cv2.imwrite(
        str(
            annotated_path
        ),
        annotated,
    )

    highest_confidence = max(
        (
            item[
                "confidence"
            ]
            for item in boxes
        ),
        default=0.0,
    )

    return {
        "ok": True,
        "path": image_path,
        "count": len(
            boxes
        ),
        "confidence": (
            highest_confidence
        ),
        "boxes": boxes,
        "annotated": annotated_path,
    }


@st.cache_data(
    show_spinner=(
        "Running pothole AI detection..."
    )
)
def run_pothole_ai():

    if not POTHOLE_MODEL.exists():

        return {
            "ok": False,
            "count": 0,
            "confidence": 0.0,
            "severity": "NONE",
            "priority": "NONE",
            "image": None,
            "annotated": None,
            "results": [],
            "message": (
                "Pothole-only model not found."
            ),
        }

    model = get_pothole_model()

    if model is None:

        return {
            "ok": False,
            "count": 0,
            "confidence": 0.0,
            "severity": "NONE",
            "priority": "NONE",
            "image": None,
            "annotated": None,
            "results": [],
            "message": (
                "Could not load pothole model."
            ),
        }

    candidate_images = [
        POTHOLE_HIGH_IMAGE,
        POTHOLE_MEDIUM_IMAGE,
        POTHOLE_LOW_IMAGE,
    ]

    results = []

    for image_path in candidate_images:

        results.append(
            infer_pothole_image(
                model,
                image_path,
            )
        )

    valid_results = [
        result
        for result in results
        if result.get(
            "ok",
            False,
        )
    ]

    if not valid_results:

        return {
            "ok": False,
            "count": 0,
            "confidence": 0.0,
            "severity": "NONE",
            "priority": "NONE",
            "image": None,
            "annotated": None,
            "results": [],
            "message": (
                "No valid pothole images found."
            ),
        }

    # Select by number of genuine pothole detections first.
    selected = max(
        valid_results,
        key=lambda item: (
            safe_int(
                item.get(
                    "count",
                    0,
                )
            ),
            safe_float(
                item.get(
                    "confidence",
                    0.0,
                )
            ),
        ),
    )

    count = safe_int(
        selected.get(
            "count",
            0,
        )
    )

    confidence = safe_float(
        selected.get(
            "confidence",
            0.0,
        )
    )

    if count > 0:

        severity = (
            calculate_hazard_severity(
                count,
                confidence,
            )
        )

        priority = (
            calculate_hazard_priority(
                count,
                confidence,
                1,
            )
        )

    else:

        severity = "NONE"
        priority = "NONE"

    return {
        "ok": True,
        "count": count,
        "confidence": confidence,
        "severity": severity,
        "priority": priority,
        "image": selected[
            "path"
        ],
        "annotated": selected[
            "annotated"
        ],
        "results": results,
        "message": (
            "Pothole AI analysis completed."
        ),
    }


POTHOLE_RESULT = (
    run_pothole_ai()
)


# ============================================================
# THREE ROAD-HAZARD SCENARIOS
# ============================================================

POTHOLE_CASES = [
    {
        "event_id": "PH-HIGH",
        "priority": "HIGH",
        "severity": "HIGH",
        "latitude": POTHOLE_LATITUDE,
        "longitude": POTHOLE_LONGITUDE,
        "image": POTHOLE_HIGH_IMAGE,
        "note": (
            "Large visible road damage "
            "requiring urgent attention."
        ),
    },
    {
        "event_id": "PH-MEDIUM",
        "priority": "MEDIUM",
        "severity": "MEDIUM",
        "latitude": (
            POTHOLE_LATITUDE
            + 0.0014
        ),
        "longitude": (
            POTHOLE_LONGITUDE
            + 0.0012
        ),
        "image": POTHOLE_MEDIUM_IMAGE,
        "note": (
            "Moderate road damage "
            "requiring maintenance."
        ),
    },
    {
        "event_id": "PH-LOW",
        "priority": "LOW",
        "severity": "LOW",
        "latitude": (
            POTHOLE_LATITUDE
            - 0.0013
        ),
        "longitude": (
            POTHOLE_LONGITUDE
            + 0.0010
        ),
        "image": POTHOLE_LOW_IMAGE,
        "note": (
            "Smaller road defect suitable "
            "for routine maintenance."
        ),
    },
]

for case in POTHOLE_CASES:

    case[
        "recommendation"
    ] = generate_recommendation(
        {
            "event_id": case[
                "event_id"
            ],
            "event_type": "pothole",
            "priority": case[
                "priority"
            ],
            "severity": case[
                "severity"
            ],
            "detection_count": 1,
            "reporting_buses": [
                "BUS-01"
            ],
        }
    )


# ============================================================
# POTHOLE EVENT
# ============================================================

pothole_event = {
    "event_id": "P100",
    "event_type": "pothole",
    "status": "DETECTED",
    "latitude": POTHOLE_LATITUDE,
    "longitude": POTHOLE_LONGITUDE,
    "source": "pothole-ai",
    "bus_id": "BUS-01",
    "route_id": "ROUTE-01",
    "reporting_buses": [
        "BUS-01"
    ],
    "detection_count": safe_int(
        POTHOLE_RESULT.get(
            "count",
            0,
        )
    ),
    "confidence": safe_float(
        POTHOLE_RESULT.get(
            "confidence",
            0.0,
        )
    ),
    "highest_confidence": safe_float(
        POTHOLE_RESULT.get(
            "confidence",
            0.0,
        )
    ),
    "severity": POTHOLE_RESULT.get(
        "severity",
        "NONE",
    ),
    "priority": POTHOLE_RESULT.get(
        "priority",
        "NONE",
    ),
    "gps_mode": "SIMULATED",
    "evidence_image": POTHOLE_RESULT.get(
        "annotated"
    ),
    "evidence_source": (
        "Indian road-damage reference image"
    ),
}

pothole_event = (
    initialize_event_lifecycle(
        pothole_event
    )
)

pothole_recommendation = (
    generate_recommendation(
        pothole_event
    )
)

pothole_event[
    "recommendation"
] = pothole_recommendation

pothole_event[
    "recommended_action"
] = (
    pothole_recommendation[
        "primary_action"
    ]
)

pothole_event[
    "recommendation_reason"
] = (
    pothole_recommendation[
        "reason"
    ]
)

pothole_event[
    "immediate_actions"
] = (
    pothole_recommendation[
        "immediate_actions"
    ]
)

pothole_event[
    "long_term_actions"
] = (
    pothole_recommendation[
        "long_term_actions"
    ]
)

EVENTS = [
    event
    for event in EVENTS
    if str(
        event.get(
            "event_id"
        )
    ) != "P100"
]

EVENTS.append(
    pothole_event
)


# ============================================================
# SYSTEM SUMMARY
# ============================================================

MULTI_BUS_EVENTS = [
    event
    for event in EVENTS
    if len(
        reporting_buses(
            event
        )
    ) >= 2
]

VERIFIED_EVENTS = [
    event
    for event in EVENTS
    if len(
        reporting_buses(
            event
        )
    ) >= 2
]

SYSTEM_SUMMARY = {
    "real_ai_observations": (
        REAL_RAW_COUNT
    ),
    "unique_incidents": len(
        EVENTS
    ),
    "multi_bus_incidents": len(
        MULTI_BUS_EVENTS
    ),
    "verified_incidents": len(
        VERIFIED_EVENTS
    ),
    "reporting_buses": len(
        {
            bus
            for event in EVENTS
            for bus in reporting_buses(
                event
            )
        }
    ),
    "real_pedestrian_observations": (
        REAL_PEDESTRIAN_OBSERVATIONS
    ),
    "real_pedestrian_frames": (
        REAL_PEDESTRIAN_FRAMES
    ),
}


# ============================================================
# REPRESENTATIVE HIGH-PRIORITY EVENTS
# ============================================================

def select_representative_high_events(
    events: list[
        dict[str, Any]
    ],
):

    candidates = [
        event
        for event in events
        if (
            event_type(event)
            == "traffic"
            and event_priority(
                event
            )
            in {
                "HIGH",
                "CRITICAL",
            }
        )
    ]

    candidates.sort(
        key=lambda event: (
            -(
                1
                if len(
                    reporting_buses(
                        event
                    )
                ) >= 2
                else 0
            ),
            -safe_int(
                event.get(
                    "vehicle_count",
                    0,
                )
            ),
            safe_float(
                event.get(
                    "video_time_seconds",
                    0.0,
                )
            ),
        )
    )

    return candidates[:2]


# EXACT SAME TWO EVENTS ARE USED BY:
# 1. City Map red pins
# 2. Action Register
# 3. High-Priority Traffic section
REPRESENTATIVE_HIGH_EVENTS = (
    select_representative_high_events(
        EVENTS
    )
)


# ============================================================
# MAP FALLBACK
# ============================================================

def fallback_traffic_event(
    level: str,
    index: int,
    source_event=None,
):

    base_lat = (
        safe_float(
            source_event.get(
                "latitude"
            )
        )
        if source_event
        else 13.3200
    )

    base_lon = (
        safe_float(
            source_event.get(
                "longitude"
            )
        )
        if source_event
        else 75.7750
    )

    offsets = {
        "HIGH": [
            (
                0.0000,
                0.0000,
            ),
            (
                0.0018,
                0.0014,
            ),
        ],
        "MEDIUM": [
            (
                -0.0015,
                0.0012,
            ),
            (
                0.0015,
                -0.0011,
            ),
        ],
        "LOW": [
            (
                -0.0012,
                -0.0012,
            ),
            (
                -0.0002,
                0.0018,
            ),
            (
                0.0018,
                -0.0004,
            ),
        ],
    }

    lat_offset, lon_offset = (
        offsets[
            level
        ][
            index
            % len(
                offsets[
                    level
                ]
            )
        ]
    )

    return {
        "event_id": (
            f"MAP-{level}-{index + 1}"
        ),
        "event_type": "traffic",
        "traffic_level": level,
        "vehicle_count": safe_int(
            source_event.get(
                "vehicle_count",
                0,
            )
        )
        if source_event
        else 0,
        "severity": level,
        "priority": level,
        "latitude": (
            base_lat
            + lat_offset
        ),
        "longitude": (
            base_lon
            + lon_offset
        ),
        "video_timestamp": (
            "SIMULATED MAP POINT"
        ),
        "reporting_buses": [
            "BUS-01"
        ],
        "detection_count": 1,
    }


# ============================================================
# BUILD MAP POINTS
# ============================================================

def build_map_points():

    points = []

    # --------------------------------------------------------
    # 2 RED
    # SAME AS ACTION CENTER
    # --------------------------------------------------------

    for index in range(2):

        if (
            index
            < len(
                REPRESENTATIVE_HIGH_EVENTS
            )
        ):

            event = (
                REPRESENTATIVE_HIGH_EVENTS[
                    index
                ]
            )

        else:

            event = fallback_traffic_event(
                "HIGH",
                index,
                (
                    REPRESENTATIVE_HIGH_EVENTS[
                        0
                    ]
                    if REPRESENTATIVE_HIGH_EVENTS
                    else None
                ),
            )

        points.append(
            {
                "map_id": (
                    f"TRAFFIC-RED-{index + 1}"
                ),
                "kind": "traffic",
                "color": "red",
                "event": event,
            }
        )

    # --------------------------------------------------------
    # 2 ORANGE
    # --------------------------------------------------------

    medium_events = [
        event
        for event in EVENTS
        if (
            event_type(event)
            == "traffic"
            and str(
                event.get(
                    "traffic_level",
                    "",
                )
            ).upper()
            == "MEDIUM"
        )
    ]

    medium_events.sort(
        key=lambda event: -safe_int(
            event.get(
                "vehicle_count",
                0,
            )
        )
    )

    for index in range(2):

        event = (
            medium_events[index]
            if index < len(
                medium_events
            )
            else fallback_traffic_event(
                "MEDIUM",
                index,
                (
                    medium_events[
                        0
                    ]
                    if medium_events
                    else None
                ),
            )
        )

        points.append(
            {
                "map_id": (
                    f"TRAFFIC-ORANGE-{index + 1}"
                ),
                "kind": "traffic",
                "color": "orange",
                "event": event,
            }
        )

    # --------------------------------------------------------
    # 3 GREEN
    # --------------------------------------------------------

    low_events = [
        event
        for event in EVENTS
        if (
            event_type(event)
            == "traffic"
            and str(
                event.get(
                    "traffic_level",
                    "",
                )
            ).upper()
            == "LOW"
        )
    ]

    low_events.sort(
        key=lambda event: -safe_int(
            event.get(
                "vehicle_count",
                0,
            )
        )
    )

    for index in range(3):

        event = (
            low_events[index]
            if index < len(
                low_events
            )
            else fallback_traffic_event(
                "LOW",
                index,
                (
                    low_events[
                        0
                    ]
                    if low_events
                    else (
                        medium_events[
                            0
                        ]
                        if medium_events
                        else None
                    )
                ),
            )
        )

        points.append(
            {
                "map_id": (
                    f"TRAFFIC-GREEN-{index + 1}"
                ),
                "kind": "traffic",
                "color": "green",
                "event": event,
            }
        )

    # --------------------------------------------------------
    # 3 PURPLE ROAD HAZARDS
    # --------------------------------------------------------

    for index, case in enumerate(
        POTHOLE_CASES,
        start=1,
    ):

        points.append(
            {
                "map_id": (
                    f"ROAD-HAZARD-{index}"
                ),
                "kind": "pothole",
                "case": case,
            }
        )

    # --------------------------------------------------------
    # 2 BLUE RESPONSE POINTS
    # --------------------------------------------------------

    red_points = [
        point
        for point in points
        if (
            point["kind"]
            == "traffic"
            and point["color"]
            == "red"
        )
    ]

    blue_offsets = [
        (
            0.0014,
            0.0007,
        ),
        (
            -0.0011,
            0.0013,
        ),
    ]

    for index in range(2):

        linked_red = (
            red_points[
                index
            ][
                "event"
            ]
            if index
            < len(red_points)
            else None
        )

        red_lat = (
            safe_float(
                linked_red.get(
                    "latitude"
                )
            )
            if linked_red
            else 13.3200
        )

        red_lon = (
            safe_float(
                linked_red.get(
                    "longitude"
                )
            )
            if linked_red
            else 75.7750
        )

        lat_offset, lon_offset = (
            blue_offsets[
                index
            ]
        )

        response_event = {
            "event_id": (
                f"ACTION-BLUE-{index + 1}"
            ),
            "event_type": "response",
            "latitude": (
                red_lat
                + lat_offset
            ),
            "longitude": (
                red_lon
                + lon_offset
            ),
        }

        points.append(
            {
                "map_id": (
                    f"RESPONSE-BLUE-{index + 1}"
                ),
                "kind": "response",
                "event": response_event,
                "linked_red": linked_red,
            }
        )

    return points


MAP_POINTS = (
    build_map_points()
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🚌 MUI"
    )

    st.caption(
        "Mobile Urban Intelligence"
    )

    st.divider()

    if st.button(
        "📊 Monitor",
        use_container_width=True,
    ):

        st.session_state.page = (
            "Monitor"
        )

    if st.button(
        "🧠 Intelligence",
        use_container_width=True,
    ):

        st.session_state.page = (
            "Intelligence"
        )

    if st.button(
        "🗺️ City Map",
        use_container_width=True,
    ):

        st.session_state.page = (
            "City Map"
        )

    if st.button(
        "🚨 Action Center",
        use_container_width=True,
    ):

        st.session_state.page = (
            "Action Center"
        )

    st.divider()

    st.write(
        "Real AI observations: "
        f"**{SYSTEM_SUMMARY['real_ai_observations']}**"
    )

    st.write(
        "Fused incidents: "
        f"**{SYSTEM_SUMMARY['unique_incidents']}**"
    )

    st.write(
        "Multi-bus confirmations: "
        f"**{SYSTEM_SUMMARY['multi_bus_incidents']}**"
    )

    st.write(
        "Reporting buses: "
        f"**{SYSTEM_SUMMARY['reporting_buses']}**"
    )

    st.write(
        "Pedestrian observations: "
        f"**{SYSTEM_SUMMARY['real_pedestrian_observations']}**"
    )

    if AI_VIDEO_IS_HEALTHY:

        st.success(
            "AI video: HEALTHY"
        )

    else:

        st.warning(
            "AI video: CHECK FILE"
        )

    st.divider()

    st.caption(
        "Fleet note"
    )

    st.caption(
        "BUS-01 = real YOLO video"
    )

    st.caption(
        "BUS-02 / BUS-03 = simulated demo confirmations"
    )


# ============================================================
# GLOBAL HEADER
# ============================================================

st.title(
    "AI-Powered Mobile Urban Intelligence Platform"
)

st.caption(
    "DETECT → UNDERSTAND → LOCATE → PRIORITIZE → ACT"
)


# ============================================================
# PAGE 1 — MONITOR
# ============================================================

if st.session_state.page == "Monitor":

    st.header(
        "📊 Fleet Monitor"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        st.metric(
            "Real AI observations",
            SYSTEM_SUMMARY[
                "real_ai_observations"
            ],
        )

    with c2:

        st.metric(
            "Unique incidents",
            SYSTEM_SUMMARY[
                "unique_incidents"
            ],
        )

    with c3:

        st.metric(
            "Multi-bus confirmations",
            SYSTEM_SUMMARY[
                "multi_bus_incidents"
            ],
        )

    with c4:

        st.metric(
            "Reporting buses",
            SYSTEM_SUMMARY[
                "reporting_buses"
            ],
        )

    st.divider()

    st.subheader(
        "Traffic sensing"
    )

    t1, t2, t3, t4, t5 = (
        st.columns(5)
    )

    with t1:

        st.metric(
            "Frames processed",
            TRAFFIC_SUMMARY[
                "frames_processed"
            ],
        )

    with t2:

        st.metric(
            "Frames sampled",
            TRAFFIC_SUMMARY[
                "frames_sampled"
            ],
        )

    with t3:

        st.metric(
            "Average vehicles",
            (
                f"{TRAFFIC_SUMMARY['average_vehicle_count']:.2f}"
            ),
        )

    with t4:

        st.metric(
            "Peak vehicles",
            TRAFFIC_SUMMARY[
                "peak_vehicle_count"
            ],
        )

    with t5:

        st.metric(
            "Overall traffic",
            TRAFFIC_SUMMARY[
                "overall_level"
            ],
        )

    st.caption(
        "Real traffic statistics are loaded from the latest "
        "traffic_events.json generated from BUS-01 video."
    )

    st.divider()

    # --------------------------------------------------------
    # AI VIDEO
    # --------------------------------------------------------

    st.subheader(
        "🎥 AI Traffic Detection"
    )

    if AI_VIDEO_IS_HEALTHY:

        st.success(
            "Real YOLO annotated traffic video is healthy and ready to play."
        )

        st.caption(
            f"Video: {AI_VIDEO_INFO['frames']} frames • "
            f"{AI_VIDEO_INFO['fps']:.0f} FPS • "
            f"{AI_VIDEO_INFO['duration']:.1f} s • "
            f"{AI_VIDEO_INFO['width']}×{AI_VIDEO_INFO['height']}"
        )

        st.video(
            str(
                AI_TRAFFIC_VIDEO
            )
        )

    elif AI_TRAFFIC_VIDEO_FALLBACK.exists():

        st.warning(
            "Primary AI video is unavailable. Using fallback AI video."
        )

        if AI_VIDEO_FALLBACK_INFO.get(
            "opened"
        ):

            st.caption(
                f"Fallback video: "
                f"{AI_VIDEO_FALLBACK_INFO['frames']} frames • "
                f"{AI_VIDEO_FALLBACK_INFO['fps']:.0f} FPS • "
                f"{AI_VIDEO_FALLBACK_INFO['duration']:.1f} s"
            )

        st.video(
            str(
                AI_TRAFFIC_VIDEO_FALLBACK
            )
        )

    elif TRAFFIC_VIDEO.exists():

        st.warning(
            "Only the original traffic video is available."
        )

        st.video(
            str(
                TRAFFIC_VIDEO
            )
        )

    else:

        st.error(
            "Traffic video not found."
        )

    st.divider()

    # --------------------------------------------------------
    # REAL AI VEHICLE + PEDESTRIAN EVIDENCE
    # --------------------------------------------------------

    st.subheader(
        "🚗 Real Vehicle & Pedestrian Detection Evidence"
    )

    st.caption(
        "Sampled observations from the real BUS-01 traffic video. "
        "Vehicle categories come from the Indian traffic model and "
        "pedestrians are detected by the COCO person model."
    )

    evidence_rows = []

    for event in REAL_TRAFFIC_EVENTS[:12]:

        breakdown = (
            get_vehicle_breakdown(
                event
            )
        )

        people = get_pedestrian_count(
            event
        )

        evidence_rows.append(
            {
                "Video time": event.get(
                    "video_timestamp",
                    "N/A",
                ),
                "Cars": breakdown[
                    "Cars"
                ],
                "Auto-rickshaws": breakdown[
                    "Auto-rickshaws"
                ],
                "Two-wheelers": breakdown[
                    "Two-wheelers"
                ],
                "Buses": breakdown[
                    "Buses"
                ],
                "Trucks": breakdown[
                    "Trucks"
                ],
                "Vans": breakdown[
                    "Vans"
                ],
                "Bicycles": breakdown[
                    "Bicycles"
                ],
                "Other": breakdown[
                    "Other"
                ],
                "Pedestrians": people,
                "Total vehicles": safe_int(
                    event.get(
                        "vehicle_count",
                        0,
                    )
                ),
                "Traffic": str(
                    event.get(
                        "traffic_level",
                        "LOW",
                    )
                ).upper(),
                "Confidence": (
                    f"{get_event_confidence(event) * 100:.1f}%"
                ),
            }
        )

    if evidence_rows:

        st.dataframe(
            pd.DataFrame(
                evidence_rows
            ),
            hide_index=True,
            use_container_width=True,
        )

    else:

        st.warning(
            "No real traffic event records are available."
        )

    summary_columns = st.columns(4)

    with summary_columns[0]:

        st.metric(
            "Auto-rickshaw detections",
            REAL_VEHICLE_TOTALS[
                "Auto-rickshaws"
            ],
        )

    with summary_columns[1]:

        st.metric(
            "Two-wheeler detections",
            REAL_VEHICLE_TOTALS[
                "Two-wheelers"
            ],
        )

    with summary_columns[2]:

        st.metric(
            "Pedestrian observations",
            REAL_PEDESTRIAN_OBSERVATIONS,
        )

    with summary_columns[3]:

        st.metric(
            "Frames with pedestrians",
            REAL_PEDESTRIAN_FRAMES,
        )

    st.info(
        "The evidence table links the AI vehicle/pedestrian observations "
        "to the traffic-level decision. BUS-01 is real video evidence; "
        "BUS-02 and BUS-03 remain simulated prototype confirmations."
    )


# ============================================================
# PAGE 2 — INTELLIGENCE
# ============================================================

elif st.session_state.page == "Intelligence":

    st.header(
        "🧠 Urban Intelligence"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        st.metric(
            "Overall traffic",
            TRAFFIC_SUMMARY[
                "overall_level"
            ],
        )

    with c2:

        st.metric(
            "Average vehicles",
            (
                f"{TRAFFIC_SUMMARY['average_vehicle_count']:.2f}"
            ),
        )

    with c3:

        st.metric(
            "Peak vehicles",
            TRAFFIC_SUMMARY[
                "peak_vehicle_count"
            ],
        )

    with c4:

        st.metric(
            "High-density frames",
            TRAFFIC_SUMMARY[
                "high_frames"
            ],
        )

    st.divider()

    # --------------------------------------------------------
    # TRAFFIC DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Traffic distribution"
    )

    traffic_chart = pd.DataFrame(
        {
            "Level": [
                "LOW",
                "MEDIUM",
                "HIGH",
            ],
            "Frames": [
                TRAFFIC_SUMMARY[
                    "low_frames"
                ],
                TRAFFIC_SUMMARY[
                    "medium_frames"
                ],
                TRAFFIC_SUMMARY[
                    "high_frames"
                ],
            ],
        }
    )

    st.bar_chart(
        traffic_chart.set_index(
            "Level"
        )
    )

    st.subheader(
        "🚦 Real Vehicle Composition"
    )

    composition_df = pd.DataFrame(
        {
            "Vehicle category": list(
                REAL_VEHICLE_TOTALS.keys()
            ),
            "Sampled detections": list(
                REAL_VEHICLE_TOTALS.values()
            ),
        }
    )

    st.bar_chart(
        composition_df.set_index(
            "Vehicle category"
        )
    )

    st.caption(
        "These totals aggregate the sampled real BUS-01 AI observations."
    )

    st.divider()

    # --------------------------------------------------------
    # FLEET INTELLIGENCE
    # --------------------------------------------------------

    st.subheader(
        "🚌 Fleet Intelligence"
    )

    st.caption(
        "Shows how multiple buses can contribute "
        "to incident confirmation."
    )

    fleet_rows = []

    for bus_id in [
        "BUS-01",
        "BUS-02",
        "BUS-03",
    ]:

        bus_events = [
            event
            for event in EVENTS
            if bus_id
            in reporting_buses(
                event
            )
        ]

        if bus_id == "BUS-01":

            mode = "REAL"
            source = "REAL AI VIDEO"
            role = (
                "Primary sensing unit"
            )

        else:

            mode = "SIMULATED"
            source = (
                "Prototype confirmation"
            )
            role = (
                "Fleet cross-check"
            )

        fleet_rows.append(
            {
                "Bus": bus_id,
                "Mode": mode,
                "Observations": len(
                    bus_events
                ),
                "Source": source,
                "Role": role,
            }
        )

    st.dataframe(
        pd.DataFrame(
            fleet_rows
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "BUS-01 is the real YOLO source. "
        "BUS-02 and BUS-03 are simulated prototype confirmations."
    )

    st.divider()

    # --------------------------------------------------------
    # ROAD HAZARD INTELLIGENCE
    # --------------------------------------------------------

    st.subheader(
        "🚧 Road Hazard Intelligence"
    )

    h1, h2, h3, h4 = (
        st.columns(4)
    )

    with h1:

        st.metric(
            "AI-detected potholes",
            POTHOLE_RESULT[
                "count"
            ],
        )

    with h2:

        st.metric(
            "AI confidence",
            (
                f"{POTHOLE_RESULT['confidence'] * 100:.1f}%"
            ),
        )

    with h3:

        st.metric(
            "AI severity",
            POTHOLE_RESULT[
                "severity"
            ],
        )

    with h4:

        st.metric(
            "AI priority",
            POTHOLE_RESULT[
                "priority"
            ],
        )

    st.caption(
        "The AI evidence uses the Indian road-damage "
        "image with the strongest number of genuine "
        "pothole detections."
    )

    annotated_path = (
        POTHOLE_RESULT.get(
            "annotated"
        )
    )

    if (
        POTHOLE_RESULT["ok"]
        and annotated_path is not None
        and image_is_valid(
            Path(
                annotated_path
            )
        )
    ):

        st.image(
            str(
                annotated_path
            ),
            caption=(
                "AI pothole detection evidence"
            ),
            use_container_width=True,
        )

        st.success(
            (
                f"AI detected "
                f"{POTHOLE_RESULT['count']} pothole(s) "
                f"in the selected Indian road-damage image."
            )
        )

    else:

        st.error(
            "Pothole AI evidence is unavailable."
        )

    st.divider()

    # --------------------------------------------------------
    # PRIORITY SCENARIOS
    # --------------------------------------------------------

    st.subheader(
        "🕳️ Pothole / Road-Damage Priority Scenarios"
    )

    st.caption(
        "Three different Indian road-damage images "
        "represent HIGH, MEDIUM and LOW priority."
    )

    scenario_columns = (
        st.columns(3)
    )

    for column, case in zip(
        scenario_columns,
        POTHOLE_CASES,
    ):

        with column:

            priority = case[
                "priority"
            ]

            if priority == "HIGH":

                st.error(
                    "🔴 HIGH PRIORITY"
                )

            elif priority == "MEDIUM":

                st.warning(
                    "🟠 MEDIUM PRIORITY"
                )

            else:

                st.success(
                    "🟢 LOW PRIORITY"
                )

            if image_is_valid(
                case[
                    "image"
                ]
            ):

                st.image(
                    str(
                        case[
                            "image"
                        ]
                    ),
                    use_container_width=True,
                )

            else:

                st.warning(
                    f"Invalid or missing image: "
                    f"{case['image'].name}"
                )

            st.write(
                f"**Severity:** "
                f"{case['severity']}"
            )

            st.write(
                f"**Priority:** "
                f"{case['priority']}"
            )

            st.caption(
                case[
                    "note"
                ]
            )

            st.success(
                case[
                    "recommendation"
                ][
                    "primary_action"
                ]
            )


# ============================================================
# PAGE 3 — CITY MAP
# ============================================================

elif st.session_state.page == "City Map":

    st.header(
        "🗺️ City GIS Decision Map"
    )

    st.caption(
        "Exactly 12 representative points: "
        "2 red + 2 orange + 3 green + "
        "3 purple + 2 blue."
    )

    city_map = folium.Map(
        location=[
            13.3200,
            75.7750,
        ],
        zoom_start=14,
        control_scale=True,
    )

    map_lookup = {}

    for point in MAP_POINTS:

        kind = point[
            "kind"
        ]

        # ----------------------------------------------------
        # PURPLE ROAD HAZARD
        # ----------------------------------------------------

        if kind == "pothole":

            case = point[
                "case"
            ]

            latitude = safe_float(
                case[
                    "latitude"
                ]
            )

            longitude = safe_float(
                case[
                    "longitude"
                ]
            )

            image_uri = (
                image_to_data_uri(
                    case[
                        "image"
                    ]
                )
            )

            image_html = ""

            if image_uri:

                image_html = (
                    "<br><br>"
                    "<img src='"
                    f"{image_uri}"
                    "' width='300'>"
                    "<br>"
                    "<small>"
                    "Indian road-damage evidence"
                    "</small>"
                )

            popup = f"""
            <b>🟣 {case["priority"]} ROAD HAZARD</b><br><br>

            <b>Event ID:</b>
            {case["event_id"]}<br>

            <b>Severity:</b>
            {case["severity"]}<br>

            <b>Priority:</b>
            {case["priority"]}<br>

            <b>GPS:</b>
            {latitude:.6f},
            {longitude:.6f}

            {image_html}

            <br><br>
            <b>Recommended action:</b><br>
            {case["recommendation"]["primary_action"]}
            """

            folium.Marker(
                [
                    latitude,
                    longitude,
                ],
                tooltip=(
                    f"🟣 {case['priority']} "
                    "road hazard"
                ),
                popup=folium.Popup(
                    popup,
                    max_width=410,
                ),
                icon=folium.Icon(
                    color="purple",
                    icon="warning-sign",
                ),
            ).add_to(
                city_map
            )

            map_lookup[
                point[
                    "map_id"
                ]
            ] = {
                "type": "pothole",
                "case": case,
                "latitude": latitude,
                "longitude": longitude,
            }

            continue

        event = point.get(
            "event",
            {}
        )

        latitude = safe_float(
            event.get(
                "latitude"
            )
        )

        longitude = safe_float(
            event.get(
                "longitude"
            )
        )

        # ----------------------------------------------------
        # TRAFFIC
        # ----------------------------------------------------

        if kind == "traffic":

            level = str(
                event.get(
                    "traffic_level",
                    "LOW",
                )
            ).upper()

            color = point[
                "color"
            ]

            buses = (
                reporting_buses(
                    event
                )
            )

            recommendation = (
                generate_recommendation(
                    event
                )
            )

            popup = f"""
            <b>🚦 {level} TRAFFIC</b><br><br>

            <b>Event ID:</b>
            {event.get("event_id", "N/A")}<br>

            <b>Video time:</b>
            {event.get("video_timestamp", "N/A")}<br>

            <b>Vehicles:</b>
            {safe_int(
                event.get(
                    "vehicle_count",
                    0,
                )
            )}<br>

            <b>Traffic level:</b>
            {level}<br>

            <b>Severity:</b>
            {event.get(
                "severity",
                level,
            )}<br>

            <b>Priority:</b>
            {event.get(
                "priority",
                level,
            )}<br>

            <b>Detections:</b>
            {safe_int(
                event.get(
                    "detection_count",
                    1,
                )
            )}<br>

            <b>Reporting buses:</b>
            {", ".join(
                buses
            ) or "N/A"}<br>

            <b>Auto-rickshaws:</b>
            {get_vehicle_breakdown(event)["Auto-rickshaws"]}<br>

            <b>Two-wheelers:</b>
            {get_vehicle_breakdown(event)["Two-wheelers"]}<br>

            <b>Pedestrians:</b>
            {get_pedestrian_count(event)}<br>

            <b>Evidence source:</b>
            {"REAL AI VIDEO" if not event.get("is_simulated", False) else "SIMULATED PROTOTYPE"}<br>

            <b>GPS:</b>
            {latitude:.6f},
            {longitude:.6f}<br><br>

            <b>Recommended action:</b><br>
            {recommendation["primary_action"]}
            """

            icon_name = (
                "warning-sign"
                if color == "red"
                else (
                    "info-sign"
                    if color == "orange"
                    else "ok-sign"
                )
            )

            folium.Marker(
                [
                    latitude,
                    longitude,
                ],
                tooltip=(
                    f"{level} traffic"
                ),
                popup=folium.Popup(
                    popup,
                    max_width=430,
                ),
                icon=folium.Icon(
                    color=color,
                    icon=icon_name,
                ),
            ).add_to(
                city_map
            )

            map_lookup[
                point[
                    "map_id"
                ]
            ] = {
                "type": "traffic",
                "event": event,
                "latitude": latitude,
                "longitude": longitude,
            }

        # ----------------------------------------------------
        # BLUE RESPONSE
        # ----------------------------------------------------

        elif kind == "response":

            linked = point.get(
                "linked_red"
            )

            linked_event_id = (
                linked.get(
                    "event_id",
                    "N/A",
                )
                if linked
                else "N/A"
            )

            popup = f"""
            <b>🔵 RESPONSE POINT</b><br><br>

            <b>Action point:</b>
            {point["map_id"]}<br>

            <b>Linked traffic incident:</b>
            {linked_event_id}<br>

            <b>Priority:</b>
            HIGH<br><br>

            Recommended response / dispatch location.
            """

            folium.Marker(
                [
                    latitude,
                    longitude,
                ],
                tooltip=(
                    "🔵 Response point"
                ),
                popup=folium.Popup(
                    popup,
                    max_width=390,
                ),
                icon=folium.Icon(
                    color="blue",
                    icon="send",
                ),
            ).add_to(
                city_map
            )

            if linked:

                red_lat = safe_float(
                    linked.get(
                        "latitude"
                    )
                )

                red_lon = safe_float(
                    linked.get(
                        "longitude"
                    )
                )

                folium.PolyLine(
                    locations=[
                        [
                            red_lat,
                            red_lon,
                        ],
                        [
                            latitude,
                            longitude,
                        ],
                    ],
                    color="blue",
                    weight=4,
                    opacity=0.85,
                    dash_array="8,8",
                    tooltip=(
                        "Response connection"
                    ),
                ).add_to(
                    city_map
                )

            map_lookup[
                point[
                    "map_id"
                ]
            ] = {
                "type": "response",
                "event": event,
                "latitude": latitude,
                "longitude": longitude,
                "linked_red": linked,
            }

    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------

    legend = """
    <div style="
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 9999;
        background: white;
        padding: 12px 14px;
        border: 2px solid #444;
        border-radius: 8px;
        color: black;
        line-height: 1.7;
        font-size: 14px;
    ">
        <b>MUI DECISION MAP</b><br>
        🔴 2 High traffic<br>
        🟠 2 Medium traffic<br>
        🟢 3 Low traffic<br>
        🟣 3 Road hazards<br>
        🔵 2 Response points
    </div>
    """

    city_map.get_root().html.add_child(
        folium.Element(
            legend
        )
    )

    map_result = st_folium(
        city_map,
        width=None,
        height=700,
        returned_objects=[
            "last_object_clicked"
        ],
        key="mui_map",
    )

    clicked = None

    if isinstance(
        map_result,
        dict,
    ):

        clicked = map_result.get(
            "last_object_clicked"
        )

    if clicked:

        clicked_lat = clicked.get(
            "lat"
        )

        clicked_lon = clicked.get(
            "lng"
        )

        if (
            clicked_lat is not None
            and clicked_lon is not None
        ):

            nearest_id = None

            nearest_distance = (
                float("inf")
            )

            for (
                map_id,
                info,
            ) in map_lookup.items():

                distance = haversine(
                    safe_float(
                        clicked_lat
                    ),
                    safe_float(
                        clicked_lon
                    ),
                    safe_float(
                        info[
                            "latitude"
                        ]
                    ),
                    safe_float(
                        info[
                            "longitude"
                        ]
                    ),
                )

                if (
                    distance
                    < nearest_distance
                ):

                    nearest_distance = (
                        distance
                    )

                    nearest_id = (
                        map_id
                    )

            if nearest_id is not None:

                st.session_state[
                    "selected_map_id"
                ] = nearest_id

    selected = map_lookup.get(
        st.session_state.get(
            "selected_map_id"
        )
    )

    if selected:

        st.divider()

        if (
            selected[
                "type"
            ]
            == "traffic"
        ):

            event = selected[
                "event"
            ]

            st.subheader(
                "🚦 Selected Traffic Incident"
            )

            c1, c2, c3, c4 = (
                st.columns(4)
            )

            with c1:

                st.metric(
                    "Vehicles",
                    safe_int(
                        event.get(
                            "vehicle_count",
                            0,
                        )
                    ),
                )

            with c2:

                st.metric(
                    "Traffic",
                    event.get(
                        "traffic_level",
                        "LOW",
                    ),
                )

            with c3:

                st.metric(
                    "Priority",
                    event.get(
                        "priority",
                        "LOW",
                    ),
                )

            with c4:

                st.metric(
                    "Detections",
                    safe_int(
                        event.get(
                            "detection_count",
                            1,
                        )
                    ),
                )

            st.write(
                "**Auto-rickshaws:** "
                f"{get_vehicle_breakdown(event)['Auto-rickshaws']}"
            )

            st.write(
                "**Two-wheelers:** "
                f"{get_vehicle_breakdown(event)['Two-wheelers']}"
            )

            st.write(
                "**Pedestrians:** "
                f"{get_pedestrian_count(event)}"
            )

            st.write(
                "**Event:** "
                f"{event.get(
                    'event_id',
                    'N/A',
                )}"
            )

            st.write(
                "**Video time:** "
                f"{event.get(
                    'video_timestamp',
                    'N/A',
                )}"
            )

            st.write(
                "**Reporting buses:** "
                f"{', '.join(
                    reporting_buses(
                        event
                    )
                ) or 'N/A'}"
            )

            recommendation = (
                generate_recommendation(
                    event
                )
            )

            st.success(
                recommendation[
                    "primary_action"
                ]
            )

        elif (
            selected[
                "type"
            ]
            == "pothole"
        ):

            case = selected[
                "case"
            ]

            st.subheader(
                "🟣 Selected Road Hazard"
            )

            c1, c2, c3 = (
                st.columns(3)
            )

            with c1:

                st.metric(
                    "Priority",
                    case[
                        "priority"
                    ],
                )

            with c2:

                st.metric(
                    "Severity",
                    case[
                        "severity"
                    ],
                )

            with c3:

                st.metric(
                    "Case",
                    case[
                        "event_id"
                    ],
                )

            if image_is_valid(
                case[
                    "image"
                ]
            ):

                st.image(
                    str(
                        case[
                            "image"
                        ]
                    ),
                    caption=(
                        "Indian road-damage evidence"
                    ),
                    use_container_width=True,
                )

            st.success(
                case[
                    "recommendation"
                ][
                    "primary_action"
                ]
            )

        else:

            st.subheader(
                "🔵 Selected Response Point"
            )

            linked = selected.get(
                "linked_red"
            )

            st.metric(
                "Priority",
                "HIGH",
            )

            if linked:

                st.write(
                    "**Linked traffic incident:** "
                    f"{linked.get(
                        'event_id',
                        'N/A',
                    )}"
                )

            st.info(
                "Response point connected to "
                "a red high-traffic incident."
            )

    else:

        st.info(
            "Click a marker to inspect "
            "the decision point."
        )


# ============================================================
# PAGE 4 — ACTION CENTER
# ============================================================

else:

    st.header(
        "🚨 Action Center"
    )

    st.caption(
        "DETECTION → FUSION → PRIORITY → ACTION"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        st.metric(
            "Unique incidents",
            SYSTEM_SUMMARY[
                "unique_incidents"
            ],
        )

    with c2:

        st.metric(
            "Verified",
            SYSTEM_SUMMARY[
                "verified_incidents"
            ],
        )

    with c3:

        st.metric(
            "Multi-bus",
            SYSTEM_SUMMARY[
                "multi_bus_incidents"
            ],
        )

    with c4:

        st.metric(
            "Reporting buses",
            SYSTEM_SUMMARY[
                "reporting_buses"
            ],
        )

    st.divider()

    # --------------------------------------------------------
    # ACTION REGISTER
    # --------------------------------------------------------

    st.subheader(
        "📋 Action Register"
    )

    st.caption(
        "Only representative incidents requiring "
        "urban action are shown here."
    )

    action_rows = []

    # --------------------------------------------------------
    # TWO HIGH TRAFFIC EVENTS
    # SAME TWO RED MAP PINS
    # --------------------------------------------------------

    for event in (
        REPRESENTATIVE_HIGH_EVENTS
    ):

        buses = reporting_buses(
            event
        )

        vehicle_count = safe_int(
            event.get(
                "vehicle_count",
                0,
            )
        )

        if len(
            buses
        ) >= 2:

            evidence = (
                f"{vehicle_count} vehicles • "
                "multi-bus confirmation"
            )

        else:

            evidence = (
                f"{vehicle_count} vehicles • "
                "AI traffic observation"
            )

        evidence += (
            f" • {get_vehicle_breakdown(event)['Auto-rickshaws']} autos"
            f" • {get_pedestrian_count(event)} pedestrians"
        )

        action = (
            generate_recommendation(
                event
            )[
                "primary_action"
            ]
        )

        action_rows.append(
            {
                "Event": event.get(
                    "event_id",
                    "N/A",
                ),
                "Type": "TRAFFIC",
                "Priority": event.get(
                    "priority",
                    "HIGH",
                ),
                "Evidence": evidence,
                "Buses": ", ".join(
                    buses
                ) or "N/A",
                "Action": action,
            }
        )

    # --------------------------------------------------------
    # THREE ROAD-HAZARD EVENTS
    # --------------------------------------------------------

    for case in POTHOLE_CASES:

        priority = case[
            "priority"
        ]

        if priority == "HIGH":

            evidence = (
                "Severe road damage"
            )

        elif priority == "MEDIUM":

            evidence = (
                "Moderate road damage"
            )

        else:

            evidence = (
                "Minor road damage"
            )

        action_rows.append(
            {
                "Event": case[
                    "event_id"
                ],
                "Type": "ROAD HAZARD",
                "Priority": priority,
                "Evidence": evidence,
                "Buses": "BUS-01",
                "Action": (
                    case[
                        "recommendation"
                    ][
                        "primary_action"
                    ]
                ),
            }
        )

    if action_rows:

        st.dataframe(
            pd.DataFrame(
                action_rows
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.divider()

    # --------------------------------------------------------
    # HIGH-PRIORITY TRAFFIC
    # SAME TWO RED MAP INCIDENTS
    # --------------------------------------------------------

    st.subheader(
        "🔴 High-Priority Traffic"
    )

    st.caption(
        "These are the same two HIGH traffic "
        "incidents represented by the red City Map pins."
    )

    if REPRESENTATIVE_HIGH_EVENTS:

        for event in (
            REPRESENTATIVE_HIGH_EVENTS
        ):

            buses = reporting_buses(
                event
            )

            recommendation = (
                generate_recommendation(
                    event
                )
            )

            with st.container(
                border=True
            ):

                st.markdown(
                    f"### 🔴 "
                    f"{event.get(
                        'event_id',
                        'N/A',
                    )}"
                )

                c1, c2, c3, c4 = (
                    st.columns(4)
                )

                with c1:

                    st.metric(
                        "Vehicles",
                        safe_int(
                            event.get(
                                "vehicle_count",
                                0,
                            )
                        ),
                    )

                with c2:

                    st.metric(
                        "Traffic",
                        event.get(
                            "traffic_level",
                            "HIGH",
                        ),
                    )

                with c3:

                    st.metric(
                        "Priority",
                        event.get(
                            "priority",
                            "HIGH",
                        ),
                    )

                with c4:

                    st.metric(
                        "Buses",
                        len(
                            buses
                        ),
                    )

                st.write(
                    "**Video time:** "
                    f"{event.get(
                        'video_timestamp',
                        'N/A',
                    )}"
                )

                st.write(
                    "**Reporting buses:** "
                    f"{', '.join(
                        buses
                    ) or 'N/A'}"
                )

                st.write(
                    "**Auto-rickshaws:** "
                    f"{get_vehicle_breakdown(event)['Auto-rickshaws']}"
                )

                st.write(
                    "**Two-wheelers:** "
                    f"{get_vehicle_breakdown(event)['Two-wheelers']}"
                )

                st.write(
                    "**Pedestrians:** "
                    f"{get_pedestrian_count(event)}"
                )

                st.write(
                    "**Evidence source:** "
                    f"{'REAL AI VIDEO' if not event.get('is_simulated', False) else 'SIMULATED PROTOTYPE'}"
                )

                st.write(
                    "**Severity:** "
                    f"{event.get(
                        'severity',
                        'HIGH',
                    )}"
                )

                if len(
                    buses
                ) >= 2:

                    st.warning(
                        "SIMULATED MULTI-BUS "
                        "CONFIRMATION"
                    )

                st.success(
                    recommendation[
                        "primary_action"
                    ]
                )

                st.write(
                    "**Reason:** "
                    f"{recommendation['reason']}"
                )

    else:

        st.info(
            "No representative high-priority "
            "traffic incidents available."
        )

    st.divider()

    # --------------------------------------------------------
    # ROAD HAZARD ACTION
    # --------------------------------------------------------

    st.subheader(
        "🕳️ Road Hazard Action"
    )

    st.caption(
        "Three different Indian road-damage cases "
        "shown by priority."
    )

    action_columns = (
        st.columns(3)
    )

    for column, case in zip(
        action_columns,
        POTHOLE_CASES,
    ):

        with column:

            priority = case[
                "priority"
            ]

            if priority == "HIGH":

                st.error(
                    "🔴 HIGH"
                )

            elif priority == "MEDIUM":

                st.warning(
                    "🟠 MEDIUM"
                )

            else:

                st.success(
                    "🟢 LOW"
                )

            image_path = case[
                "image"
            ]

            if image_is_valid(
                image_path
            ):

                st.image(
                    str(
                        image_path
                    ),
                    use_container_width=True,
                )

            else:

                st.warning(
                    f"Invalid image: "
                    f"{image_path.name}"
                )

            st.write(
                f"**Severity:** "
                f"{case['severity']}"
            )

            st.success(
                case[
                    "recommendation"
                ][
                    "primary_action"
                ]
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Prototype note: BUS-01 is the real video source. "
    "Traffic, vehicle and pedestrian observations shown for BUS-01 "
    "come from the generated AI traffic pipeline. BUS-02/BUS-03 "
    "confirmations and GPS positions are simulated demonstration data. "
    "The City Map and Action Center share the same two representative "
    "high-traffic incidents. Road Hazard Intelligence uses the dedicated "
    "pothole-only model. AI recommendations are decision-support rules, "
    "not final civil-engineering designs."
)