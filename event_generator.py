from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
from ultralytics import YOLO


# ============================================================
# DEFAULT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT = (
    BASE_DIR
    / "13020022_3840_2160_30fps.mp4"
)

DEFAULT_TRAFFIC_MODEL = (
    BASE_DIR
    / "weights"
    / "indian_traffic_yolo11s.pt"
)

DEFAULT_PERSON_MODEL = (
    BASE_DIR
    / "yolo11n.pt"
)

DEFAULT_OUTPUT = (
    BASE_DIR
    / "traffic_ai_h264.mp4"
)

DEFAULT_FALLBACK_OUTPUT = (
    BASE_DIR
    / "traffic_ai_detected.mp4"
)

DEFAULT_JSON = (
    BASE_DIR
    / "traffic_events.json"
)


# ============================================================
# TRAFFIC RULES
# ============================================================

LOW_MAX = 14
MEDIUM_MAX = 24

TRAFFIC_LEVELS = (
    "LOW",
    "MEDIUM",
    "HIGH",
)


# ============================================================
# INDIAN TRAFFIC MODEL CLASSES
# ============================================================

INDIAN_CLASS_NAMES = {
    0: "Hatchback",
    1: "Sedan",
    2: "SUV",
    3: "MUV",
    4: "Bus",
    5: "Truck",
    6: "Three-wheeler",
    7: "Two-wheeler",
    8: "LCV",
    9: "Mini-bus",
    10: "tempo-traveller",
    11: "bicycle",
    12: "Van",
    13: "Others",
}


def normalize_indian_class(
    class_name: str,
) -> str:

    value = str(
        class_name
    ).strip().lower()

    if value in {
        "hatchback",
        "sedan",
        "suv",
        "muv",
    }:
        return "cars"

    if value in {
        "bus",
        "mini-bus",
        "mini bus",
        "tempo-traveller",
        "tempo traveller",
    }:
        return "buses"

    if value in {
        "truck",
        "lcv",
    }:
        return "trucks"

    if value in {
        "three-wheeler",
        "three wheeler",
        "auto",
        "auto-rickshaw",
        "autorickshaw",
    }:
        return "three_wheelers"

    if value in {
        "two-wheeler",
        "two wheeler",
        "motorcycle",
        "motorbike",
        "scooter",
    }:
        return "two_wheelers"

    if value == "van":
        return "vans"

    if value == "bicycle":
        return "bicycles"

    return "other"


def empty_breakdown() -> dict[str, int]:

    return {
        "cars": 0,
        "buses": 0,
        "trucks": 0,
        "three_wheelers": 0,
        "two_wheelers": 0,
        "vans": 0,
        "bicycles": 0,
        "other": 0,
    }


# ============================================================
# TRAFFIC CLASSIFICATION
# ============================================================

def classify_traffic(
    vehicle_count: int,
) -> str:

    if vehicle_count <= LOW_MAX:
        return "LOW"

    if vehicle_count <= MEDIUM_MAX:
        return "MEDIUM"

    return "HIGH"


def calculate_priority(
    traffic_level: str,
) -> str:

    return str(
        traffic_level
    ).upper()


def calculate_severity(
    traffic_level: str,
) -> str:

    return str(
        traffic_level
    ).upper()


def calculate_congestion_score(
    vehicle_count: int,
) -> float:

    if vehicle_count <= 0:
        return 0.0

    score = (
        vehicle_count / 30.0
    )

    return round(
        min(
            score,
            1.0,
        ),
        3,
    )


# ============================================================
# YOLO EXTRACTION
# ============================================================

def extract_detection_data(
    result: Any,
    class_names: dict[int, str] | list[str] | None = None,
):

    detections = []

    boxes = getattr(
        result,
        "boxes",
        None,
    )

    if boxes is None:
        return detections

    classes = getattr(
        boxes,
        "cls",
        None,
    )

    confidences = getattr(
        boxes,
        "conf",
        None,
    )

    xyxy = getattr(
        boxes,
        "xyxy",
        None,
    )

    if classes is None:
        return detections

    try:
        class_values = (
            classes.cpu()
            .numpy()
            .tolist()
        )
    except Exception:
        class_values = []

    try:
        confidence_values = (
            confidences.cpu()
            .numpy()
            .tolist()
        )
    except Exception:
        confidence_values = []

    try:
        box_values = (
            xyxy.cpu()
            .numpy()
            .tolist()
        )
    except Exception:
        box_values = []

    for index, class_id in enumerate(
        class_values
    ):

        class_id = int(
            class_id
        )

        confidence = (
            float(
                confidence_values[index]
            )
            if index
            < len(
                confidence_values
            )
            else 0.0
        )

        box = (
            box_values[index]
            if index
            < len(
                box_values
            )
            else [
                0,
                0,
                0,
                0,
            ]
        )

        if class_names is None:
            name = str(
                class_id
            )

        elif isinstance(
            class_names,
            dict,
        ):
            name = str(
                class_names.get(
                    class_id,
                    class_id,
                )
            )

        else:
            name = (
                str(
                    class_names[class_id]
                )
                if 0
                <= class_id
                < len(
                    class_names
                )
                else str(
                    class_id
                )
            )

        detections.append(
            {
                "class_id": class_id,
                "class_name": name,
                "confidence": confidence,
                "box": box,
            }
        )

    return detections


# ============================================================
# SIMULATED GPS ROUTE
# ============================================================

def calculate_route_position(
    video_time_seconds: float,
    duration_seconds: float,
):

    route_lat_start = 13.3095
    route_lon_start = 75.7585

    route_lat_end = 13.3315
    route_lon_end = 75.7905

    if duration_seconds <= 0:
        ratio = 0.0

    else:
        ratio = (
            video_time_seconds
            / duration_seconds
        )

    ratio = max(
        0.0,
        min(
            ratio,
            1.0,
        ),
    )

    latitude = (
        route_lat_start
        + (
            route_lat_end
            - route_lat_start
        )
        * ratio
    )

    longitude = (
        route_lon_start
        + (
            route_lon_end
            - route_lon_start
        )
        * ratio
    )

    return (
        round(latitude, 7),
        round(longitude, 7),
    )


# ============================================================
# TRAFFIC EVENT
# ============================================================

def build_event(
    frame_number: int,
    video_time_seconds: float,
    duration_seconds: float,
    breakdown: dict[str, int],
    detections: list[dict[str, Any]],
    person_count: int,
    bus_id: str,
    route_id: str,
):

    vehicle_count = sum(
        breakdown.values()
    )

    traffic_level = classify_traffic(
        vehicle_count
    )

    confidences = [
        float(
            item.get(
                "confidence",
                0.0,
            )
        )
        for item in detections
    ]

    average_confidence = (
        sum(confidences)
        / len(confidences)
        if confidences
        else 0.0
    )

    highest_confidence = (
        max(confidences)
        if confidences
        else 0.0
    )

    latitude, longitude = (
        calculate_route_position(
            video_time_seconds,
            duration_seconds,
        )
    )

    congestion_score = (
        calculate_congestion_score(
            vehicle_count
        )
    )

    event = {
        "event_id": (
            f"BUS01-{frame_number}"
        ),
        "event_type": "traffic",

        "status": "DETECTED",

        "bus_id": bus_id,
        "route_id": route_id,

        "reporting_buses": [
            bus_id
        ],

        "frame_number": frame_number,

        "video_time_seconds": round(
            video_time_seconds,
            3,
        ),

        "video_timestamp": (
            f"{video_time_seconds:.2f}s"
        ),

        "vehicle_count": vehicle_count,

        "vehicle_breakdown": breakdown,

        "traffic_level": traffic_level,

        "severity": calculate_severity(
            traffic_level
        ),

        "priority": calculate_priority(
            traffic_level
        ),

        "congestion_score": congestion_score,

        "congestion_km": round(
            congestion_score,
            3,
        ),

        "average_confidence": round(
            average_confidence,
            4,
        ),

        "confidence": round(
            highest_confidence,
            4,
        ),

        "highest_confidence": round(
            highest_confidence,
            4,
        ),

        "detection_count": len(
            detections
        ),

        "person_count": int(
            person_count
        ),

        "pedestrian_detected": (
            person_count > 0
        ),

        "latitude": latitude,
        "longitude": longitude,

        "location": (
            "Chikkamagaluru"
        ),

        "source": (
            "Indian traffic YOLO + "
            "COCO person YOLO"
        ),

        "gps_mode": (
            "SIMULATED_ROUTE"
        ),

        "is_simulated": False,
    }

    return event


# ============================================================
# DRAW TRAFFIC BOXES
# ============================================================

def draw_traffic_detections(
    frame,
    detections,
):

    for detection in detections:

        box = detection[
            "box"
        ]

        confidence = float(
            detection[
                "confidence"
            ]
        )

        class_name = str(
            detection[
                "class_name"
            ]
        )

        category = normalize_indian_class(
            class_name
        )

        x1, y1, x2, y2 = [
            int(
                max(
                    0,
                    value,
                )
            )
            for value in box
        ]

        label = (
            f"{class_name} "
            f"{confidence * 100:.1f}%"
        )

        # Green for ordinary traffic
        # Red for buses/trucks
        # Orange for auto-rickshaws
        if category in {
            "buses",
            "trucks",
        }:
            box_color = (
                0,
                0,
                255,
            )

        elif category == (
            "three_wheelers"
        ):
            box_color = (
                0,
                165,
                255,
            )

        else:
            box_color = (
                0,
                255,
                0,
            )

        cv2.rectangle(
            frame,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            box_color,
            2,
        )

        text_y = max(
            25,
            y1 - 8,
        )

        cv2.putText(
            frame,
            label,
            (
                x1,
                text_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2,
            cv2.LINE_AA,
        )

    return frame


# ============================================================
# DRAW PERSON BOXES
# ============================================================

def draw_person_detections(
    frame,
    person_detections,
):

    for detection in (
        person_detections
    ):

        box = detection[
            "box"
        ]

        confidence = float(
            detection[
                "confidence"
            ]
        )

        x1, y1, x2, y2 = [
            int(
                max(
                    0,
                    value,
                )
            )
            for value in box
        ]

        box_color = (
            255,
            255,
            0,
        )

        label = (
            f"Person "
            f"{confidence * 100:.1f}%"
        )

        cv2.rectangle(
            frame,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            box_color,
            2,
        )

        text_y = max(
            25,
            y1 - 8,
        )

        cv2.putText(
            frame,
            label,
            (
                x1,
                text_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2,
            cv2.LINE_AA,
        )

    return frame


# ============================================================
# FRAME STATUS OVERLAY
# ============================================================

def draw_status(
    frame,
    frame_number: int,
    video_time_seconds: float,
    vehicle_count: int,
    traffic_level: str,
    person_count: int,
):

    cv2.rectangle(
        frame,
        (
            0,
            0,
        ),
        (
            frame.shape[1],
            95,
        ),
        (
            0,
            0,
            0,
        ),
        -1,
    )

    text = (
        f"MUI AI TRAFFIC | "
        f"Frame: {frame_number} | "
        f"Time: {video_time_seconds:.2f}s | "
        f"Vehicles: {vehicle_count} | "
        f"Traffic: {traffic_level} | "
        f"Persons: {person_count}"
    )

    cv2.putText(
        frame,
        text,
        (
            20,
            38,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "BUS-01 | AI MOBILE URBAN INTELLIGENCE",
        (
            20,
            75,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    return frame


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate(
    input_path: Path,
    traffic_model_path: Path,
    person_model_path: Path,
    output_path: Path,
    fallback_output_path: Path,
    json_path: Path,
    sample_every: int,
    confidence: float,
    bus_id: str,
    route_id: str,
):

    print()
    print("=" * 70)
    print("MUI TRAFFIC AI VIDEO GENERATOR")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # CHECK INPUTS
    # --------------------------------------------------------

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input video not found: {input_path}"
        )

    if not traffic_model_path.exists():
        raise FileNotFoundError(
            "Indian traffic model not found: "
            f"{traffic_model_path}"
        )

    if not person_model_path.exists():
        raise FileNotFoundError(
            "Person model not found: "
            f"{person_model_path}"
        )

    # --------------------------------------------------------
    # VIDEO INPUT
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(input_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open input video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if fps <= 0:
        fps = 30.0

    duration_seconds = (
        total_frames / fps
        if fps > 0
        else 0.0
    )

    print(
        f"Input video : {input_path.name}"
    )
    print(
        f"Frames      : {total_frames}"
    )
    print(
        f"FPS         : {fps:.2f}"
    )
    print(
        f"Duration    : {duration_seconds:.2f}s"
    )
    print(
        f"Resolution  : {width}x{height}"
    )
    print()

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    print(
        "Loading Indian traffic model..."
    )

    traffic_model = YOLO(
        str(
            traffic_model_path
        )
    )

    print(
        "Loading COCO person model..."
    )

    person_model = YOLO(
        str(
            person_model_path
        )
    )

    print(
        "Models loaded."
    )

    # --------------------------------------------------------
    # FFMPEG
    # --------------------------------------------------------

    ffmpeg_exe = (
        imageio_ffmpeg
        .get_ffmpeg_exe()
    )

    temporary_output = (
        output_path.with_suffix(
            ".part.mp4"
        )
    )

    if temporary_output.exists():
        temporary_output.unlink()

    # --------------------------------------------------------
    # START FFMPEG ENCODER
    # --------------------------------------------------------

    ffmpeg_command = [
        ffmpeg_exe,

        "-y",

        "-f",
        "rawvideo",

        "-vcodec",
        "rawvideo",

        "-pix_fmt",
        "bgr24",

        "-s",
        f"{width}x{height}",

        "-r",
        str(fps),

        "-i",
        "-",

        "-an",

        "-c:v",
        "libx264",

        "-preset",
        "medium",

        "-crf",
        "23",

        "-pix_fmt",
        "yuv420p",

        "-movflags",
        "+faststart",

        str(
            temporary_output
        ),
    ]

    print(
        "Starting H.264 encoder..."
    )

    ffmpeg_process = subprocess.Popen(
        ffmpeg_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # --------------------------------------------------------
    # PROCESS VIDEO
    # --------------------------------------------------------

    events = []

    frame_number = 0

    sampled_frames = 0

    vehicle_counts = []

    confidence_values = []

    traffic_level_counts = {
        "LOW": 0,
        "MEDIUM": 0,
        "HIGH": 0,
    }

    try:

        while True:

            success, frame = (
                cap.read()
            )

            if not success:
                break

            frame_number += 1

            # ------------------------------------------------
            # TRAFFIC INFERENCE
            # ------------------------------------------------

            traffic_results = (
                traffic_model.predict(
                    source=frame,
                    conf=confidence,
                    verbose=False,
                    device="cpu",
                )
            )

            traffic_result = (
                traffic_results[0]
            )

            traffic_detections = (
                extract_detection_data(
                    traffic_result,
                    INDIAN_CLASS_NAMES,
                )
            )

            breakdown = (
                empty_breakdown()
            )

            for detection in (
                traffic_detections
            ):

                category = (
                    normalize_indian_class(
                        detection[
                            "class_name"
                        ]
                    )
                )

                breakdown[
                    category
                ] += 1

            vehicle_count = sum(
                breakdown.values()
            )

            # ------------------------------------------------
            # PERSON INFERENCE
            # ------------------------------------------------

            person_results = (
                person_model.predict(
                    source=frame,
                    conf=0.25,
                    classes=[0],
                    verbose=False,
                    device="cpu",
                )
            )

            person_result = (
                person_results[0]
            )

            person_detections = (
                extract_detection_data(
                    person_result
                )
            )

            person_count = len(
                person_detections
            )

            # ------------------------------------------------
            # ANNOTATE EVERY FRAME
            # ------------------------------------------------

            annotated = frame.copy()

            annotated = (
                draw_traffic_detections(
                    annotated,
                    traffic_detections,
                )
            )

            annotated = (
                draw_person_detections(
                    annotated,
                    person_detections,
                )
            )

            current_time = (
                frame_number / fps
            )

            current_level = (
                classify_traffic(
                    vehicle_count
                )
            )

            annotated = draw_status(
                annotated,
                frame_number,
                current_time,
                vehicle_count,
                current_level,
                person_count,
            )

            # ------------------------------------------------
            # SEND FRAME TO FFMPEG
            # ------------------------------------------------

            if (
                ffmpeg_process.stdin
                is None
            ):
                raise RuntimeError(
                    "FFmpeg stdin is unavailable."
                )

            ffmpeg_process.stdin.write(
                annotated.tobytes()
            )

            # ------------------------------------------------
            # SAMPLE EVENT DATA
            # ------------------------------------------------

            if (
                frame_number == 1
                or frame_number
                % sample_every
                == 0
            ):

                event = build_event(
                    frame_number=frame_number,
                    video_time_seconds=(
                        current_time
                    ),
                    duration_seconds=(
                        duration_seconds
                    ),
                    breakdown=breakdown,
                    detections=traffic_detections,
                    person_count=person_count,
                    bus_id=bus_id,
                    route_id=route_id,
                )

                events.append(
                    event
                )

                sampled_frames += 1

                vehicle_counts.append(
                    vehicle_count
                )

                confidence_values.append(
                    event[
                        "confidence"
                    ]
                )

                traffic_level_counts[
                    current_level
                ] += 1

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if (
                frame_number % 100
                == 0
                or frame_number
                == total_frames
            ):

                percentage = (
                    frame_number
                    / max(
                        total_frames,
                        1,
                    )
                    * 100
                )

                print(
                    f"Processed "
                    f"{frame_number}/"
                    f"{total_frames} "
                    f"({percentage:.1f}%)"
                )

    except BrokenPipeError:

        raise RuntimeError(
            "FFmpeg stopped accepting frames. "
            "The video encoder failed."
        )

    finally:

        cap.release()

        if (
            ffmpeg_process.stdin
            is not None
        ):

            try:
                ffmpeg_process.stdin.close()
            except Exception:
                pass

    # --------------------------------------------------------
    # FINISH FFMPEG
    # --------------------------------------------------------

    stderr_bytes = (
        ffmpeg_process.stderr.read()
        if ffmpeg_process.stderr
        else b""
    )

    return_code = (
        ffmpeg_process.wait()
    )

    if return_code != 0:

        stderr_text = (
            stderr_bytes.decode(
                "utf-8",
                errors="replace",
            )
        )

        raise RuntimeError(
            "FFmpeg encoding failed.\n\n"
            f"{stderr_text[-5000:]}"
        )

    if not temporary_output.exists():

        raise RuntimeError(
            "FFmpeg reported success, "
            "but the output video was not created."
        )

    # --------------------------------------------------------
    # VERIFY GENERATED VIDEO
    # --------------------------------------------------------

    verification = cv2.VideoCapture(
        str(
            temporary_output
        )
    )

    if not verification.isOpened():

        raise RuntimeError(
            "Generated video cannot be opened "
            "after encoding."
        )

    generated_frames = int(
        verification.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    generated_fps = (
        verification.get(
            cv2.CAP_PROP_FPS
        )
    )

    generated_width = int(
        verification.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    generated_height = int(
        verification.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    verification.release()

    print()
    print("VIDEO VERIFICATION")
    print("-" * 40)
    print(
        f"Frames    : {generated_frames}"
    )
    print(
        f"FPS       : {generated_fps:.2f}"
    )
    print(
        f"Resolution: "
        f"{generated_width}x"
        f"{generated_height}"
    )
    print()

    if (
        generated_frames
        < total_frames - 2
    ):

        raise RuntimeError(
            "Generated video contains too few frames. "
            f"Expected about {total_frames}, "
            f"got {generated_frames}."
        )

    # --------------------------------------------------------
    # INSTALL FINAL OUTPUT
    # --------------------------------------------------------

    if output_path.exists():
        output_path.unlink()

    temporary_output.replace(
        output_path
    )

    # Also create the fallback filename
    if fallback_output_path.exists():
        fallback_output_path.unlink()

    shutil.copy2(
        output_path,
        fallback_output_path,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    if vehicle_counts:

        average_vehicle_count = (
            sum(vehicle_counts)
            / len(vehicle_counts)
        )

        peak_vehicle_count = max(
            vehicle_counts
        )

    else:

        average_vehicle_count = 0.0
        peak_vehicle_count = 0

    if traffic_level_counts[
        "HIGH"
    ] > 0:

        overall_level = "HIGH"

    elif traffic_level_counts[
        "MEDIUM"
    ] > 0:

        overall_level = "MEDIUM"

    else:

        overall_level = "LOW"

    average_confidence = (
        sum(confidence_values)
        / len(confidence_values)
        if confidence_values
        else 0.0
    )

    payload = {
        "summary": {
            "frames_processed": (
                total_frames
            ),

            "frames_sampled": (
                sampled_frames
            ),

            "frames_analyzed": (
                sampled_frames
            ),

            "average_vehicle_count": (
                round(
                    average_vehicle_count,
                    2,
                )
            ),

            "peak_vehicle_count": (
                peak_vehicle_count
            ),

            "traffic_levels": {
                "LOW": traffic_level_counts[
                    "LOW"
                ],
                "MEDIUM": traffic_level_counts[
                    "MEDIUM"
                ],
                "HIGH": traffic_level_counts[
                    "HIGH"
                ],
            },

            "low_frames": (
                traffic_level_counts[
                    "LOW"
                ]
            ),

            "medium_frames": (
                traffic_level_counts[
                    "MEDIUM"
                ]
            ),

            "high_frames": (
                traffic_level_counts[
                    "HIGH"
                ]
            ),

            "overall_level": (
                overall_level
            ),

            "average_confidence": round(
                average_confidence,
                4,
            ),

            "video_file": (
                output_path.name
            ),

            "video_frames": (
                generated_frames
            ),

            "video_fps": (
                generated_fps
            ),
        },

        "events": events,
    }

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    output_size_mb = (
        output_path.stat().st_size
        / (
            1024 * 1024
        )
    )

    print()
    print("=" * 70)
    print("TRAFFIC GENERATION COMPLETE")
    print("=" * 70)
    print(
        f"Frames processed : "
        f"{total_frames}"
    )
    print(
        f"Frames sampled   : "
        f"{sampled_frames}"
    )
    print(
        f"Average vehicles : "
        f"{average_vehicle_count:.2f}"
    )
    print(
        f"Peak vehicles    : "
        f"{peak_vehicle_count}"
    )
    print(
        f"Overall traffic  : "
        f"{overall_level}"
    )
    print(
        f"LOW frames       : "
        f"{traffic_level_counts['LOW']}"
    )
    print(
        f"MEDIUM frames    : "
        f"{traffic_level_counts['MEDIUM']}"
    )
    print(
        f"HIGH frames      : "
        f"{traffic_level_counts['HIGH']}"
    )
    print(
        f"Events written   : "
        f"{json_path.name}"
    )
    print(
        f"AI video written : "
        f"{output_path.name}"
    )
    print(
        f"Fallback video   : "
        f"{fallback_output_path.name}"
    )
    print(
        f"Video size       : "
        f"{output_size_mb:.2f} MB"
    )
    print(
        f"Verified frames  : "
        f"{generated_frames}"
    )
    print("=" * 70)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Generate browser-compatible "
            "AI traffic video and traffic events."
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT
        ),
    )

    parser.add_argument(
        "--traffic-model",
        default=str(
            DEFAULT_TRAFFIC_MODEL
        ),
    )

    parser.add_argument(
        "--person-model",
        default=str(
            DEFAULT_PERSON_MODEL
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT
        ),
    )

    parser.add_argument(
        "--fallback-output",
        default=str(
            DEFAULT_FALLBACK_OUTPUT
        ),
    )

    parser.add_argument(
        "--json",
        default=str(
            DEFAULT_JSON
        ),
    )

    parser.add_argument(
        "--sample-every",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--bus-id",
        default="BUS-01",
    )

    parser.add_argument(
        "--route-id",
        default="ROUTE-01",
    )

    return parser.parse_args()


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    args = parse_arguments()

    generate(
        input_path=Path(
            args.input
        ),

        traffic_model_path=Path(
            args.traffic_model
        ),

        person_model_path=Path(
            args.person_model
        ),

        output_path=Path(
            args.output
        ),

        fallback_output_path=Path(
            args.fallback_output
        ),

        json_path=Path(
            args.json
        ),

        sample_every=max(
            1,
            args.sample_every,
        ),

        confidence=max(
            0.01,
            min(
                args.confidence,
                0.99,
            ),
        ),

        bus_id=args.bus_id,

        route_id=args.route_id,
    )


if __name__ == "__main__":
    main()