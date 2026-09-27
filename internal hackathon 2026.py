import json
import streamlit as st
from ultralytics import YOLO
from datetime import datetime
from map_data import events


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Mobile Urban Intelligence",
    page_icon="🚍",
    layout="wide"
)


# =========================================================
# LOAD POTHOLE MODEL ONCE
# =========================================================

@st.cache_resource
def load_pothole_model():
    return YOLO("weights/best.pt")


pothole_model = load_pothole_model()


# =========================================================
# FILES
# =========================================================

# Browser-friendly AI-annotated video
camera_video = "runs/detect/predict-2/ai_detected_h264.mp4"

# Pothole test image
hazard_image = "pexels-sliceisop-24827286.jpg"

# Cached traffic analysis
traffic_cache = "traffic_results.json"


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "analyzed": False,
    "traffic_data": None,
    "hazard_data": None,
    "priority": None,
    "recommendation": None
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# HEADER
# =========================================================

st.title(
    "🚍 Mobile Urban Intelligence Platform"
)

st.caption(
    "Turning transport-camera observations into urban intelligence"
)

st.divider()


# =========================================================
# SYSTEM STATUS
# =========================================================

s1, s2, s3, s4 = st.columns(4)

s1.metric(
    "AI Engine",
    "READY"
)

s2.metric(
    "Camera Feed",
    "CONNECTED"
)

s3.metric(
    "GIS",
    "ACTIVE"
)

s4.metric(
    "Platform",
    "ONLINE"
)

st.divider()


# =========================================================
# CAMERA FEED
# =========================================================

left, right = st.columns([2, 1])


with left:

    st.subheader(
        "🎥 Demo Fleet Camera Feed"
    )

    st.video(
        camera_video
    )

    st.caption(
        "Recorded road footage simulating a public-transport camera stream."
    )

    analyze_button = st.button(
        "🤖 Analyze Urban Scene",
        type="primary"
    )


with right:

    st.subheader(
        "🧠 System Overview"
    )

    st.write(
        "The platform analyzes transport-camera "
        "observations and converts them into "
        "traffic, road-hazard and priority information."
    )

    st.info(
        "Prototype mode: GPS locations are simulated."
    )


# =========================================================
# ANALYSIS
# =========================================================

if analyze_button:

    st.info(
        "Loading cached traffic intelligence..."
    )

    # -----------------------------------------------------
    # LOAD TRAFFIC CACHE
    # -----------------------------------------------------

    try:

        with open(
            traffic_cache,
            "r"
        ) as file:

            traffic = json.load(file)

    except FileNotFoundError:

        st.error(
            "traffic_results.json not found. "
            "Run analysis_cache.py first."
        )

        st.stop()


    # -----------------------------------------------------
    # POTHOLE ANALYSIS
    # -----------------------------------------------------

    hazard_results = pothole_model(
        hazard_image,
        verbose=False
    )

    hazard_result = hazard_results[0]


    pothole_count = len(
        hazard_result.boxes
    )


    confidences = []


    for box in hazard_result.boxes:

        confidence = float(
            box.conf[0]
        )

        confidences.append(
            confidence
        )


    if pothole_count > 0:

        highest_confidence = max(
            confidences
        )

    else:

        highest_confidence = 0


    # -----------------------------------------------------
    # HAZARD SEVERITY
    # -----------------------------------------------------

    if (
        pothole_count >= 2
        and highest_confidence >= 0.70
    ):

        hazard_severity = "HIGH"

    elif (
        pothole_count >= 1
        and highest_confidence >= 0.60
    ):

        hazard_severity = "MEDIUM"

    else:

        hazard_severity = "LOW"


    # -----------------------------------------------------
    # URBAN PRIORITY
    # -----------------------------------------------------

    if (
        traffic["overall"] == "HIGH"
        and hazard_severity == "HIGH"
    ):

        priority = "CRITICAL"

        recommendation = (
            "Immediate road inspection recommended."
        )

    elif (
        traffic["overall"] == "HIGH"
        or hazard_severity == "HIGH"
    ):

        priority = "HIGH"

        recommendation = (
            "Priority monitoring and inspection recommended."
        )

    elif (
        traffic["overall"] == "MEDIUM"
        or hazard_severity == "MEDIUM"
    ):

        priority = "MEDIUM"

        recommendation = (
            "Continue monitoring this location."
        )

    else:

        priority = "LOW"

        recommendation = (
            "No immediate intervention indicated."
        )


    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    st.session_state.traffic_data = traffic


    st.session_state.hazard_data = {

        "count": pothole_count,

        "confidence": highest_confidence,

        "severity": hazard_severity

    }


    st.session_state.priority = priority


    st.session_state.recommendation = (
        recommendation
    )


    st.session_state.analyzed = True


    st.success(
        "✅ AI analysis complete!"
    )


# =========================================================
# RESULTS
# =========================================================

if st.session_state.analyzed:

    traffic = (
        st.session_state.traffic_data
    )

    hazard = (
        st.session_state.hazard_data
    )

    priority = (
        st.session_state.priority
    )

    recommendation = (
        st.session_state.recommendation
    )


    # =====================================================
    # CURRENT URBAN SITUATION
    # =====================================================

    st.divider()

    st.subheader(
        "🏙️ Current Urban Situation"
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Average Vehicles",
        traffic["average"]
    )


    c2.metric(
        "Peak Vehicles",
        traffic["peak"]
    )


    c3.metric(
        "Road Hazards",
        hazard["count"]
    )


    c4.metric(
        "Priority",
        priority
    )


    # =====================================================
    # TRAFFIC INTELLIGENCE
    # =====================================================

    st.divider()

    st.subheader(
        "📊 Traffic Intelligence"
    )


    t1, t2, t3, t4 = st.columns(4)


    t1.metric(
        "Frames Analyzed",
        traffic["frames"]
    )


    t2.metric(
        "Average Vehicles",
        traffic["average"]
    )


    t3.metric(
        "Peak Vehicles",
        traffic["peak"]
    )


    t4.metric(
        "Overall Traffic",
        traffic["overall"]
    )


    st.write(
        f"🟢 Low traffic frames: "
        f"{traffic['low']}"
    )


    st.write(
        f"🟠 Medium traffic frames: "
        f"{traffic['medium']}"
    )


    st.write(
        f"🔴 High traffic frames: "
        f"{traffic['high']}"
    )


    # =====================================================
    # ROAD HAZARD
    # =====================================================

    st.divider()

    st.subheader(
        "🕳️ Road Hazard Intelligence"
    )


    h1, h2, h3 = st.columns(3)


    h1.metric(
        "Potholes Detected",
        hazard["count"]
    )


    h2.metric(
        "Highest Confidence",
        f"{hazard['confidence']:.1%}"
    )


    h3.metric(
        "Severity",
        hazard["severity"]
    )


    st.caption(
        "Prototype severity rule based on detection "
        "count and model confidence."
    )


    # =====================================================
    # CITY ACTION CENTER
    # =====================================================

    st.divider()

    st.subheader(
        "🚨 City Action Center"
    )


    a1, a2 = st.columns(2)


    a1.metric(
        "Priority Level",
        priority
    )


    with a2:

        st.info(
            f"**Recommended Action**\n\n"
            f"{recommendation}"
        )


    # =====================================================
    # HAZARD EVENT
    # =====================================================

    hazard_event = {

        "event_id": 100,

        "event_type": "road_hazard",

        "hazard_type": "pothole",

        "count": hazard["count"],

        "confidence": round(
            hazard["confidence"],
            4
        ),

        "severity": hazard["severity"],

        "priority": priority,

        "location": "Chikkamagaluru",

        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    }


    with st.expander(
        "🚨 Latest Hazard Event Details"
    ):

        st.json(
            hazard_event
        )


# =========================================================
# CITY INTELLIGENCE MAP
# =========================================================

st.divider()

st.subheader(
    "🗺️ City Intelligence Map"
)


st.caption(
    "Demo map showing simulated urban events."
)


map_points = []


for event in events:

    map_points.append({

        "lat": event["latitude"],

        "lon": event["longitude"]

    })


st.map(
    map_points,
    use_container_width=True
)


# =========================================================
# MAP LEGEND
# =========================================================

st.write(
    "### Map Legend"
)


legend1, legend2, legend3 = st.columns(3)


legend1.error(
    "🔴 HIGH PRIORITY\n\n"
    "Traffic hotspot / critical event"
)


legend2.warning(
    "🟠 MEDIUM PRIORITY\n\n"
    "Moderate traffic condition"
)


legend3.success(
    "🟢 LOW PRIORITY\n\n"
    "Low traffic condition"
)


# =========================================================
# PRIORITY LOCATIONS
# =========================================================

st.write(
    "### 📍 Priority Locations"
)


for event in events:

    if event["traffic_level"] == "HIGH":

        st.error(

            f"🔴 **HIGH PRIORITY** — "
            f"Event {event['event_id']} | "
            f"{event['vehicle_count']} vehicles | "
            f"Location: Chikkamagaluru"

        )

    elif event["traffic_level"] == "MEDIUM":

        st.warning(

            f"🟠 **MEDIUM PRIORITY** — "
            f"Event {event['event_id']} | "
            f"{event['vehicle_count']} vehicles | "
            f"Location: Chikkamagaluru"

        )

    else:

        st.success(

            f"🟢 **LOW PRIORITY** — "
            f"Event {event['event_id']} | "
            f"{event['vehicle_count']} vehicles | "
            f"Location: Chikkamagaluru"

        )


# =========================================================
# URBAN EVENTS
# =========================================================

st.divider()

st.subheader(
    "🚨 Urban Events"
)


for event in events:

    if event["traffic_level"] == "HIGH":

        priority_label = "HIGH"

    elif event["traffic_level"] == "MEDIUM":

        priority_label = "MEDIUM"

    else:

        priority_label = "LOW"


    st.write(

        f"**Event {event['event_id']}** | "
        f"Traffic: {event['traffic_level']} | "
        f"Vehicles: {event['vehicle_count']} | "
        f"Priority: {priority_label}"

    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Mobile Urban Intelligence Platform • "
    "SIH26124 Prototype"
)