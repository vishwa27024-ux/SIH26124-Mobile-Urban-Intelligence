import json
import math
from pathlib import Path

import cv2
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from ultralytics import YOLO


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="AI-Powered Mobile Urban Intelligence Platform",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VIDEO_FILE = (
    BASE_DIR
    / "13020022_3840_2160_30fps.mp4"
)

ANNOTATED_VIDEO = (
    BASE_DIR
    / "runs"
    / "detect"
    / "predict-2"
    / "ai_detected_h264.mp4"
)

HAZARD_IMAGE = (
    BASE_DIR
    / "pexels-sliceisop-24827286.jpg"
)

POTHOLE_MODEL = (
    BASE_DIR
    / "weights"
    / "best.pt"
)

TRAFFIC_JSON = (
    BASE_DIR
    / "traffic_results.json"
)

AI_POTHOLE_IMAGE = (
    BASE_DIR
    / "ai_pothole_annotated.jpg"
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Monitor"

if "selected_event" not in st.session_state:
    st.session_state.selected_event = None

if "selected_type" not in st.session_state:
    st.session_state.selected_type = None

if "map_last_click" not in st.session_state:
    st.session_state.map_last_click = None


# ============================================================
# EVENT DATA
# ============================================================
#
# IMPORTANT:
# These GPS values are SIMULATED prototype coordinates.
# They are not GPS metadata extracted from the pothole image.
# ============================================================

EVENTS = [

    {
        "event_id": 1,
        "event_type": "traffic",
        "traffic_level": "HIGH",
        "vehicle_count": 14,
        "latitude": 13.316100,
        "longitude": 75.772000,
        "location": "Chikkamagaluru",
        "timestamp": "2026-09-11 10:15:00",
        "source": "recorded-road-video",
        "congestion_km": 0.80,
        "reporting_buses": 3,
        "junction_type": "single_corridor",
        "intervention_latitude": 13.319000,
        "intervention_longitude": 75.779500,
    },

    {
        "event_id": 2,
        "event_type": "traffic",
        "traffic_level": "MEDIUM",
        "vehicle_count": 9,
        "latitude": 13.321500,
        "longitude": 75.784500,
        "location": "Chikkamagaluru",
        "timestamp": "2026-09-11 10:18:00",
        "source": "recorded-road-video",
        "congestion_km": 0.45,
        "reporting_buses": 2,
        "junction_type": "multi_approach",
        "intervention_latitude": 13.323500,
        "intervention_longitude": 75.790000,
    },

    {
        "event_id": 3,
        "event_type": "traffic",
        "traffic_level": "LOW",
        "vehicle_count": 4,
        "latitude": 13.309800,
        "longitude": 75.764200,
        "location": "Chikkamagaluru",
        "timestamp": "2026-09-11 10:20:00",
        "source": "recorded-road-video",
        "congestion_km": 0.20,
        "reporting_buses": 1,
        "junction_type": "single_corridor",
        "intervention_latitude": 13.307500,
        "intervention_longitude": 75.760500,
    },

    {
        "event_id": 100,
        "event_type": "pothole",
        "latitude": 13.318000,
        "longitude": 75.778000,
        "location": "Chikkamagaluru",
        "timestamp": "2026-09-11 10:22:00",
        "source": "pothole-ai-detection",
        "pothole_count": 0,
        "confidence": 0.0,
        "severity": "NONE",
        "priority": "NONE",
    },
]


# ============================================================
# FIND EVENT
# ============================================================

def get_event(event_id):

    for event in EVENTS:

        if event["event_id"] == event_id:
            return event

    return None


# ============================================================
# HAVERSINE
# ============================================================

def haversine(
    lat1,
    lon1,
    lat2,
    lon2,
):

    R = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dlat = math.radians(
        lat2 - lat1
    )

    dlon = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(p1)
        * math.cos(p2)
        * math.sin(dlon / 2) ** 2
    )

    return (
        2
        * R
        * math.asin(
            math.sqrt(a)
        )
    )


# ============================================================
# POTHOLE MODEL
# ============================================================

@st.cache_resource
def get_pothole_model():

    if not POTHOLE_MODEL.exists():
        return None

    return YOLO(
        str(POTHOLE_MODEL)
    )


# ============================================================
# ROAD ROI
# ============================================================

def create_road_mask(
    height,
    width,
):

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    # Main paved-road trapezoid.
    #
    # This deliberately excludes the
    # far-left grass/shoulder and
    # far-right landscape.

    points = np.array(
        [
            [
                int(width * 0.34),
                int(height * 0.53),
            ],
            [
                int(width * 0.70),
                int(height * 0.53),
            ],
            [
                int(width * 0.91),
                int(height * 0.99),
            ],
            [
                int(width * 0.10),
                int(height * 0.99),
            ],
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(
        mask,
        [points],
        255,
    )

    return mask, points


# ============================================================
# IOU
# ============================================================

def iou(
    box1,
    box2,
):

    x1 = max(
        box1[0],
        box2[0],
    )

    y1 = max(
        box1[1],
        box2[1],
    )

    x2 = min(
        box1[2],
        box2[2],
    )

    y2 = min(
        box1[3],
        box2[3],
    )

    intersection = (
        max(0, x2 - x1)
        * max(0, y2 - y1)
    )

    area1 = (
        (box1[2] - box1[0])
        *
        (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0])
        *
        (box2[3] - box2[1])
    )

    union = (
        area1
        + area2
        - intersection
    )

    if union <= 0:
        return 0

    return (
        intersection / union
    )


# ============================================================
# POTHOLE DETECTION
# ============================================================

@st.cache_data(
    show_spinner="AI is analysing the road..."
)
def run_pothole_detection():

    if not HAZARD_IMAGE.exists():

        return {
            "ok": False,
            "count": 0,
            "confidence": 0,
            "boxes": [],
            "image": None,
            "message": (
                "Pothole image not found."
            ),
        }

    model = get_pothole_model()

    if model is None:

        return {
            "ok": False,
            "count": 0,
            "confidence": 0,
            "boxes": [],
            "image": None,
            "message": (
                "Pothole model not found: "
                "weights/best.pt"
            ),
        }

    image = cv2.imread(
        str(HAZARD_IMAGE)
    )

    if image is None:

        return {
            "ok": False,
            "count": 0,
            "confidence": 0,
            "boxes": [],
            "image": None,
            "message": (
                "Could not read pothole image."
            ),
        }

    height, width = (
        image.shape[:2]
    )

    road_mask, road_polygon = (
        create_road_mask(
            height,
            width,
        )
    )

    # ----------------------------------------
    # CREATE ROAD-ONLY IMAGE
    # ----------------------------------------

    road_only = np.zeros_like(
        image
    )

    road_only[
        road_mask > 0
    ] = image[
        road_mask > 0
    ]

    # ----------------------------------------
    # MODEL
    # ----------------------------------------

    try:

        results = model.predict(
            source=road_only,
            conf=0.55,
            iou=0.45,
            verbose=False,
        )

    except Exception as error:

        return {
            "ok": False,
            "count": 0,
            "confidence": 0,
            "boxes": [],
            "image": None,
            "message": str(error),
        }

    candidates = []

    # ----------------------------------------
    # FILTER DETECTIONS
    # ----------------------------------------

    for result in results:

        if result.boxes is None:
            continue

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

        for box, confidence in zip(
            boxes,
            confidences,
        ):

            x1, y1, x2, y2 = map(
                float,
                box,
            )

            confidence = float(
                confidence
            )

            # Center point.
            cx = int(
                (x1 + x2) / 2
            )

            cy = int(
                (y1 + y2) / 2
            )

            # Must be inside image.
            if (
                cx < 0
                or cx >= width
                or cy < 0
                or cy >= height
            ):
                continue

            # --------------------------------
            # CENTER MUST BE ON ROAD
            # --------------------------------

            if (
                road_mask[cy, cx]
                == 0
            ):
                continue

            # --------------------------------
            # BOX OVERLAP WITH ROAD
            # --------------------------------

            bx1 = max(
                0,
                int(x1),
            )

            by1 = max(
                0,
                int(y1),
            )

            bx2 = min(
                width,
                int(x2),
            )

            by2 = min(
                height,
                int(y2),
            )

            if bx2 <= bx1 or by2 <= by1:
                continue

            box_mask = road_mask[
                by1:by2,
                bx1:bx2
            ]

            overlap = (
                np.count_nonzero(
                    box_mask
                )
                /
                box_mask.size
            )

            # At least 70% of the
            # detection must be on road.
            if overlap < 0.70:
                continue

            candidates.append(
                {
                    "box": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
                    "confidence": confidence,
                }
            )

    # ----------------------------------------
    # SORT
    # ----------------------------------------

    candidates.sort(
        key=lambda item: item[
            "confidence"
        ],
        reverse=True,
    )

    # ----------------------------------------
    # REMOVE DUPLICATES
    # ----------------------------------------

    final_boxes = []

    for candidate in candidates:

        duplicate = False

        for accepted in final_boxes:

            if (
                iou(
                    candidate["box"],
                    accepted["box"],
                )
                >= 0.45
            ):

                duplicate = True
                break

        if not duplicate:

            final_boxes.append(
                candidate
            )

    # ----------------------------------------
    # DRAW AI RESULTS ONLY
    # ----------------------------------------

    annotated = image.copy()

    # No original image will be displayed
    # anywhere in the dashboard.

    for index, detection in enumerate(
        final_boxes,
        start=1,
    ):

        x1, y1, x2, y2 = map(
            int,
            detection["box"],
        )

        confidence = (
            detection["confidence"]
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            4,
        )

        label = (
            f"Pothole "
            f"{confidence * 100:.1f}%"
        )

        cv2.putText(
            annotated,
            label,
            (
                x1,
                max(
                    30,
                    y1 - 10,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # ----------------------------------------
    # SAVE AI IMAGE
    # ----------------------------------------

    cv2.imwrite(
        str(AI_POTHOLE_IMAGE),
        annotated,
    )

    highest_confidence = 0.0

    if final_boxes:

        highest_confidence = max(
            item["confidence"]
            for item in final_boxes
        )

    return {
        "ok": True,
        "count": len(
            final_boxes
        ),
        "confidence": (
            highest_confidence
        ),
        "boxes": final_boxes,
        "image": str(
            AI_POTHOLE_IMAGE
        ),
        "message": (
            "AI pothole detection completed."
        ),
    }


# ============================================================
# RUN AI
# ============================================================

pothole = (
    run_pothole_detection()
)

pothole_event = get_event(
    100
)

if pothole_event:

    pothole_event[
        "pothole_count"
    ] = pothole["count"]

    pothole_event[
        "confidence"
    ] = pothole["confidence"]

    if (
        pothole["count"] >= 2
        and pothole["confidence"] >= 0.70
    ):

        pothole_event[
            "severity"
        ] = "HIGH"

        pothole_event[
            "priority"
        ] = "HIGH"

    elif (
        pothole["count"] >= 1
        and pothole["confidence"] >= 0.60
    ):

        pothole_event[
            "severity"
        ] = "MEDIUM"

        pothole_event[
            "priority"
        ] = "MEDIUM"

    elif pothole["count"] >= 1:

        pothole_event[
            "severity"
        ] = "LOW"

        pothole_event[
            "priority"
        ] = "LOW"

    else:

        pothole_event[
            "severity"
        ] = "NONE"

        pothole_event[
            "priority"
        ] = "NONE"


# ============================================================
# TRAFFIC DATA
# ============================================================

@st.cache_data
def load_traffic():

    default = {
        "frames": 624,
        "average_vehicle_count": 9.81,
        "peak_vehicle_count": 19,
        "low_frames": 30,
        "medium_frames": 343,
        "high_frames": 251,
        "overall_level": "MEDIUM",
    }

    if not TRAFFIC_JSON.exists():

        return default

    try:

        with open(
            TRAFFIC_JSON,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(
                file
            )

    except Exception:

        return default


traffic = load_traffic()

frames = traffic.get(
    "frames",
    624,
)

average_vehicles = traffic.get(
    "average_vehicle_count",
    9.81,
)

peak_vehicles = traffic.get(
    "peak_vehicle_count",
    19,
)

low_frames = traffic.get(
    "low_frames",
    30,
)

medium_frames = traffic.get(
    "medium_frames",
    343,
)

high_frames = traffic.get(
    "high_frames",
    251,
)

overall_level = traffic.get(
    "overall_level",
    "MEDIUM",
)


# ============================================================
# TRAFFIC SOLUTION
# ============================================================

def traffic_solution(event):

    level = event[
        "traffic_level"
    ]

    count = event[
        "vehicle_count"
    ]

    congestion = event[
        "congestion_km"
    ]

    buses = event[
        "reporting_buses"
    ]

    if level == "HIGH":

        if (
            event[
                "junction_type"
            ]
            == "multi_approach"
        ):

            concept = (
                "Flyover / "
                "grade-separated junction"
            )

            reason = (
                "Multiple approaches can create "
                "turning conflicts and queue spillback."
            )

        else:

            concept = (
                "Bypass / ring-road / "
                "corridor improvement"
            )

            reason = (
                "The observed queue is concentrated "
                "along a single corridor."
            )

        immediate = [
            "Deploy traffic-control personnel.",
            "Review signal timing.",
            "Monitor queue growth.",
            "Check the next bus observation.",
        ]

        long_term = [
            "Conduct traffic-volume study.",
            "Evaluate corridor-capacity options.",
            "Study grade separation or bypass feasibility.",
        ]

    elif level == "MEDIUM":

        concept = (
            "Signal optimization / "
            "junction channelization"
        )

        reason = (
            "Traffic is elevated but does not yet "
            "justify a major infrastructure concept."
        )

        immediate = [
            "Monitor the corridor.",
            "Review signal timing.",
            "Track repeated observations.",
        ]

        long_term = [
            "Optimize signal phases.",
            "Review turning lanes.",
        ]

    else:

        concept = (
            "Monitoring only"
        )

        reason = (
            "Traffic is currently low."
        )

        immediate = [
            "Continue monitoring."
        ]

        long_term = [
            "Reassess if traffic increases."
        ]

    return {
        "concept": concept,
        "reason": reason,
        "immediate": immediate,
        "long_term": long_term,
    }


# ============================================================
# MAP
# ============================================================

def build_map():

    city_map = folium.Map(
        location=[
            13.3165,
            75.7770,
        ],
        zoom_start=14,
        control_scale=True,
    )

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
        padding: 12px;
        border: 2px solid #444;
        border-radius: 8px;
        font-size: 14px;
        line-height: 1.7;
        box-shadow: 0 2px 8px rgba(0,0,0,.25);
    ">

    <b>MAP LEGEND</b><br>

    <span style="color:#800080">●</span>
    Purple — Pothole / road hazard<br>

    <span style="color:red">●</span>
    Red — HIGH traffic<br>

    <span style="color:orange">●</span>
    Orange — MEDIUM traffic<br>

    <span style="color:green">●</span>
    Green — LOW traffic<br>

    <span style="color:#1769aa">●</span>
    Blue — AI-assisted intervention

    </div>
    """

    city_map.get_root().html.add_child(
        folium.Element(
            legend
        )
    )

    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    for event in EVENTS:

        # ====================================================
        # POTHOLE
        # ====================================================

        if event[
            "event_type"
        ] == "pothole":

            popup = f"""
            <b>🟣 ROAD HAZARD</b><br><br>

            Potholes:
            {event["pothole_count"]}<br>

            Confidence:
            {event["confidence"] * 100:.1f}%<br>

            Severity:
            {event["severity"]}<br>

            Time:
            {event["timestamp"]}<br>

            GPS:
            {event["latitude"]:.6f},
            {event["longitude"]:.6f}
            """

            folium.Marker(
                [
                    event[
                        "latitude"
                    ],
                    event[
                        "longitude"
                    ],
                ],
                tooltip=(
                    "🟣 Pothole • click for details"
                ),
                popup=folium.Popup(
                    popup,
                    max_width=330,
                ),
                icon=folium.Icon(
                    color="purple",
                    icon="exclamation-sign",
                ),
            ).add_to(
                city_map
            )

        # ====================================================
        # TRAFFIC
        # ====================================================

        else:

            if (
                event[
                    "traffic_level"
                ]
                == "HIGH"
            ):

                color = "red"
                icon = "warning-sign"

            elif (
                event[
                    "traffic_level"
                ]
                == "MEDIUM"
            ):

                color = "orange"
                icon = "info-sign"

            else:

                color = "green"
                icon = "ok-sign"

            solution = traffic_solution(
                event
            )

            popup = f"""
            <b>{event["traffic_level"]} TRAFFIC</b><br><br>

            Vehicles:
            {event["vehicle_count"]}<br>

            Congestion:
            {event["congestion_km"]:.2f} km<br>

            Reporting buses:
            {event["reporting_buses"]}<br>

            Time:
            {event["timestamp"]}<br>

            GPS:
            {event["latitude"]:.6f},
            {event["longitude"]:.6f}<br><br>

            AI concept:
            {solution["concept"]}
            """

            folium.Marker(
                [
                    event[
                        "latitude"
                    ],
                    event[
                        "longitude"
                    ],
                ],
                tooltip=(
                    f"{event['traffic_level']} traffic"
                ),
                popup=folium.Popup(
                    popup,
                    max_width=350,
                ),
                icon=folium.Icon(
                    color=color,
                    icon=icon,
                ),
            ).add_to(
                city_map
            )

            # ------------------------------------------------
            # BLUE INTERVENTION
            # ------------------------------------------------

            if (
                event[
                    "traffic_level"
                ]
                == "HIGH"
            ):

                distance = haversine(
                    event["latitude"],
                    event["longitude"],
                    event[
                        "intervention_latitude"
                    ],
                    event[
                        "intervention_longitude"
                    ],
                )

                blue_popup = f"""
                <b>🔵 AI-ASSISTED INTERVENTION</b><br><br>

                Linked event:
                #{event["event_id"]}<br>

                Concept:
                {solution["concept"]}<br>

                Distance:
                {distance:.2f} km<br>

                Hotspot GPS:
                {event["latitude"]:.6f},
                {event["longitude"]:.6f}<br>

                Intervention GPS:
                {event["intervention_latitude"]:.6f},
                {event["intervention_longitude"]:.6f}<br><br>

                {solution["reason"]}
                """

                folium.Marker(
                    [
                        event[
                            "intervention_latitude"
                        ],
                        event[
                            "intervention_longitude"
                        ],
                    ],
                    tooltip=(
                        "🔵 Intervention concept"
                    ),
                    popup=folium.Popup(
                        blue_popup,
                        max_width=380,
                    ),
                    icon=folium.Icon(
                        color="blue",
                        icon="road",
                    ),
                ).add_to(
                    city_map
                )

                folium.PolyLine(
                    [
                        [
                            event[
                                "latitude"
                            ],
                            event[
                                "longitude"
                            ],
                        ],
                        [
                            event[
                                "intervention_latitude"
                            ],
                            event[
                                "intervention_longitude"
                            ],
                        ],
                    ],
                    color="blue",
                    weight=4,
                    dash_array="8,8",
                ).add_to(
                    city_map
                )

    return city_map


# ============================================================
# MAP CLICK PROCESSING
# ============================================================

def select_map_event(
    clicked_lat,
    clicked_lng,
):

    # --------------------------------------------------------
    # CHECK BLUE INTERVENTION FIRST
    # --------------------------------------------------------

    best = None
    best_distance = 999

    for event in EVENTS:

        if (
            event["event_type"]
            != "traffic"
        ):
            continue

        if (
            event["traffic_level"]
            != "HIGH"
        ):
            continue

        distance = haversine(
            clicked_lat,
            clicked_lng,
            event[
                "intervention_latitude"
            ],
            event[
                "intervention_longitude"
            ],
        )

        if distance < best_distance:

            best = event
            best_distance = distance

    if (
        best is not None
        and best_distance < 0.12
    ):

        st.session_state.selected_event = (
            best["event_id"]
        )

        st.session_state.selected_type = (
            "intervention"
        )

        return

    # --------------------------------------------------------
    # CHECK NORMAL EVENTS
    # --------------------------------------------------------

    best = None
    best_distance = 999

    for event in EVENTS:

        distance = haversine(
            clicked_lat,
            clicked_lng,
            event[
                "latitude"
            ],
            event[
                "longitude"
            ],
        )

        if distance < best_distance:

            best = event
            best_distance = distance

    if (
        best is not None
        and best_distance < 0.12
    ):

        st.session_state.selected_event = (
            best["event_id"]
        )

        st.session_state.selected_type = (
            "event"
        )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.title(
        "🚌 MUI Dashboard"
    )

    st.caption(
        "Mobile Urban Intelligence"
    )

    st.divider()

    st.subheader(
        "Navigation"
    )

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

    st.subheader(
        "Map Legend"
    )

    st.write(
        "🟣 Pothole / road hazard"
    )

    st.write(
        "🔴 HIGH traffic"
    )

    st.write(
        "🟠 MEDIUM traffic"
    )

    st.write(
        "🟢 LOW traffic"
    )

    st.write(
        "🔵 AI-assisted intervention"
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "AI-Powered Mobile Urban Intelligence Platform"
)

st.caption(
    "DETECT → UNDERSTAND → LOCATE → PRIORITIZE → ACT"
)


# ============================================================
# MONITOR
# ============================================================

if (
    st.session_state.page
    == "Monitor"
):

    st.header(
        "📊 Fleet Monitor"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Frames analysed",
            frames,
        )

    with c2:

        st.metric(
            "Average vehicles",
            f"{average_vehicles:.2f}",
        )

    with c3:

        st.metric(
            "Peak vehicles",
            peak_vehicles,
        )

    st.divider()

    st.subheader(
        "AI Traffic Detection"
    )

    if ANNOTATED_VIDEO.exists():

        st.video(
            str(
                ANNOTATED_VIDEO
            )
        )

    elif VIDEO_FILE.exists():

        st.video(
            str(
                VIDEO_FILE
            )
        )

        st.warning(
            "AI annotated video not found. "
            "Showing original video."
        )

    else:

        st.error(
            "Road video not found."
        )


# ============================================================
# INTELLIGENCE
# ============================================================

elif (
    st.session_state.page
    == "Intelligence"
):

    st.header(
        "🧠 Urban Intelligence"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Overall traffic",
            overall_level,
        )

    with c2:

        st.metric(
            "Average vehicles",
            f"{average_vehicles:.2f}",
        )

    with c3:

        st.metric(
            "Peak vehicles",
            peak_vehicles,
        )

    with c4:

        st.metric(
            "High-density frames",
            high_frames,
        )

    st.divider()

    st.subheader(
        "Traffic distribution"
    )

    chart = pd.DataFrame(
        {
            "Level": [
                "LOW",
                "MEDIUM",
                "HIGH",
            ],
            "Frames": [
                low_frames,
                medium_frames,
                high_frames,
            ],
        }
    )

    st.bar_chart(
        chart.set_index(
            "Level"
        )
    )

    st.divider()

    st.subheader(
        "Road hazard intelligence"
    )

    h1, h2, h3, h4 = st.columns(4)

    with h1:

        st.metric(
            "Potholes",
            pothole[
                "count"
            ],
        )

    with h2:

        st.metric(
            "Confidence",
            f"{pothole['confidence'] * 100:.1f}%",
        )

    with h3:

        st.metric(
            "Severity",
            pothole_event[
                "severity"
            ],
        )

    with h4:

        st.metric(
            "Priority",
            pothole_event[
                "priority"
            ],
        )


# ============================================================
# CITY MAP
# ============================================================

elif (
    st.session_state.page
    == "City Map"
):

    st.header(
        "🗺️ City GIS Decision Map"
    )

    city_map = build_map()

    map_result = st_folium(
        city_map,
        width=None,
        height=650,
        returned_objects=[
            "last_object_clicked"
        ],
        key="main_city_map",
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

        lat = clicked.get(
            "lat"
        )

        lng = clicked.get(
            "lng"
        )

        if (
            lat is not None
            and lng is not None
        ):

            click_key = (
                round(lat, 5),
                round(lng, 5),
            )

            if (
                click_key
                != st.session_state.map_last_click
            ):

                st.session_state.map_last_click = (
                    click_key
                )

                select_map_event(
                    lat,
                    lng,
                )

    # ========================================================
    # SELECTED EVENT
    # ========================================================

    selected = get_event(
        st.session_state.selected_event
    )

    if selected is None:

        st.info(
            "Click a marker on the map to inspect the event."
        )

    # ========================================================
    # POTHOLE DETAILS
    # ========================================================

    elif (
        selected["event_type"]
        == "pothole"
    ):

        st.divider()

        st.subheader(
            "🟣 Selected Road Hazard"
        )

        # ----------------------------------------------------
        # ONLY ONE IMAGE
        # ----------------------------------------------------

        st.markdown(
            "### AI-detected pothole image"
        )

        if (
            pothole["ok"]
            and pothole["image"]
            and Path(
                pothole["image"]
            ).exists()
        ):

            st.image(
                pothole["image"],
                use_container_width=True,
            )

        else:

            st.error(
                pothole["message"]
            )

        st.divider()

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Potholes",
                selected[
                    "pothole_count"
                ],
            )

        with c2:

            st.metric(
                "Confidence",
                (
                    f"{selected['confidence'] * 100:.1f}%"
                ),
            )

        with c3:

            st.metric(
                "Severity",
                selected[
                    "severity"
                ],
            )

        with c4:

            st.metric(
                "Priority",
                selected[
                    "priority"
                ],
            )

        # ----------------------------------------------------
        # EXACT GPS
        # ----------------------------------------------------

        st.subheader(
            "📍 Exact GPS location"
        )

        st.code(
            (
                f"Latitude : "
                f"{selected['latitude']:.6f}\n"
                f"Longitude: "
                f"{selected['longitude']:.6f}"
            )
        )

        google_maps_link = (
            "https://www.google.com/maps/search/"
            f"?api=1&query="
            f"{selected['latitude']},"
            f"{selected['longitude']}"
        )

        st.link_button(
            "📍 Open exact location in Google Maps",
            google_maps_link,
        )

        st.caption(
            "Prototype note: this GPS coordinate is "
            "simulated demo data. The current photograph "
            "does not provide verified live GPS."
        )

        st.write(
            f"**Location:** "
            f"{selected['location']}"
        )

        st.write(
            f"**Detection time:** "
            f"{selected['timestamp']}"
        )

        st.write(
            f"**Source:** "
            f"{selected['source']}"
        )

        st.subheader(
            "Recommended action"
        )

        st.success(
            "Immediate: verify the detected pothole "
            "and dispatch road maintenance if confirmed."
        )

        st.info(
            "Long-term: compare repeated bus observations "
            "to determine whether the road deficiency persists."
        )

    # ========================================================
    # TRAFFIC DETAILS
    # ========================================================

    else:

        st.divider()

        solution = (
            traffic_solution(
                selected
            )
        )

        intervention_distance = (
            haversine(
                selected["latitude"],
                selected["longitude"],
                selected[
                    "intervention_latitude"
                ],
                selected[
                    "intervention_longitude"
                ],
            )
        )

        st.subheader(
            "🚦 Selected Traffic Intelligence"
        )

        c1, c2, c3, c4, c5 = (
            st.columns(5)
        )

        with c1:

            st.metric(
                "Traffic",
                selected[
                    "traffic_level"
                ],
            )

        with c2:

            st.metric(
                "Vehicles",
                selected[
                    "vehicle_count"
                ],
            )

        with c3:

            st.metric(
                "Congestion",
                f"{selected['congestion_km']:.2f} km",
            )

        with c4:

            st.metric(
                "Reporting buses",
                selected[
                    "reporting_buses"
                ],
            )

        with c5:

            st.metric(
                "Hotspot → intervention",
                f"{intervention_distance:.2f} km",
            )

        st.subheader(
            "📍 GPS"
        )

        st.code(
            (
                f"Hotspot Latitude : "
                f"{selected['latitude']:.6f}\n"
                f"Hotspot Longitude: "
                f"{selected['longitude']:.6f}\n"
                f"Intervention Latitude : "
                f"{selected['intervention_latitude']:.6f}\n"
                f"Intervention Longitude: "
                f"{selected['intervention_longitude']:.6f}"
            )
        )

        st.write(
            f"**Detection time:** "
            f"{selected['timestamp']}"
        )

        st.subheader(
            "AI-assisted infrastructure concept"
        )

        st.info(
            solution[
                "concept"
            ]
        )

        st.write(
            f"**Reason:** "
            f"{solution['reason']}"
        )

        st.subheader(
            "Immediate actions"
        )

        for item in solution[
            "immediate"
        ]:

            st.write(
                f"• {item}"
            )

        st.subheader(
            "Long-term actions"
        )

        for item in solution[
            "long_term"
        ]:

            st.write(
                f"• {item}"
            )

        st.warning(
            "The infrastructure concept is AI-assisted "
            "decision support, not a final civil-engineering design."
        )


# ============================================================
# ACTION CENTER
# ============================================================

elif (
    st.session_state.page
    == "Action Center"
):

    st.header(
        "🚨 Action Center"
    )

    st.subheader(
        "High-priority traffic"
    )

    for event in EVENTS:

        if (
            event["event_type"]
            != "traffic"
        ):
            continue

        if (
            event["traffic_level"]
            != "HIGH"
        ):
            continue

        solution = (
            traffic_solution(
                event
            )
        )

        distance = (
            haversine(
                event["latitude"],
                event["longitude"],
                event[
                    "intervention_latitude"
                ],
                event[
                    "intervention_longitude"
                ],
            )
        )

        with st.container(
            border=True
        ):

            st.markdown(
                f"### 🔴 Event #{event['event_id']}"
            )

            st.write(
                f"Time: "
                f"{event['timestamp']}"
            )

            st.write(
                f"Vehicles: "
                f"{event['vehicle_count']}"
            )

            st.write(
                f"Congestion: "
                f"{event['congestion_km']:.2f} km"
            )

            st.write(
                f"Reporting buses: "
                f"{event['reporting_buses']}"
            )

            st.write(
                f"Intervention distance: "
                f"{distance:.2f} km"
            )

            st.write(
                f"AI concept: "
                f"{solution['concept']}"
            )

            st.write(
                f"Reason: "
                f"{solution['reason']}"
            )

    st.divider()

    st.subheader(
        "Road hazard"
    )

    st.write(
        f"Potholes: "
        f"{pothole['count']}"
    )

    st.write(
        f"Confidence: "
        f"{pothole['confidence'] * 100:.1f}%"
    )

    st.write(
        f"Severity: "
        f"{pothole_event['severity']}"
    )

    st.write(
        f"Priority: "
        f"{pothole_event['priority']}"
    )

    st.write(
        f"GPS: "
        f"{pothole_event['latitude']:.6f}, "
        f"{pothole_event['longitude']:.6f}"
    )

    st.write(
        f"Time: "
        f"{pothole_event['timestamp']}"
    )