from ultralytics import YOLO
from datetime import datetime

model = YOLO(
    "https://huggingface.co/peterhdd/pothole-detection-yolov8/resolve/main/best.pt"
)

results = model("pexels-sliceisop-24827286.jpg", verbose=False)

result = results[0]

pothole_count = len(result.boxes)

confidences = []

for box in result.boxes:
    confidence = float(box.conf[0])
    confidences.append(confidence)

if pothole_count == 0:
    severity = "LOW"
    highest_confidence = 0
else:
    highest_confidence = max(confidences)

    if pothole_count >= 2 and highest_confidence >= 0.70:
        severity = "HIGH"
    elif highest_confidence >= 0.60:
        severity = "MEDIUM"
    else:
        severity = "LOW"

event = {
    "event_id": 1,
    "event_type": "road_hazard",
    "hazard_type": "pothole",
    "pothole_count": pothole_count,
    "highest_confidence": round(highest_confidence, 4),
    "severity": severity,
    "location": "Chikkamagaluru",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

print("\n===== ROAD HAZARD EVENT =====")
print(event)