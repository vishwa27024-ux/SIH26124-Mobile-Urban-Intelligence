from ultralytics import YOLO
from datetime import datetime

model = YOLO("yolo11n.pt")

vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

results = model(
    source="13020022_3840_2160_30fps.mp4",
    stream=True
)

event_id = 1

for result in results:

    counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0
    }

    for class_id in result.boxes.cls:
        class_id = int(class_id)

        if class_id in vehicle_classes:
            vehicle_name = vehicle_classes[class_id]
            counts[vehicle_name] += 1

    vehicle_count = sum(counts.values())

    if vehicle_count <= 5:
        traffic_level = "LOW"
    elif vehicle_count <= 10:
        traffic_level = "MEDIUM"
    else:
        traffic_level = "HIGH"

    event = {
        "event_id": event_id,
        "event_type": "traffic_density",
        "vehicle_count": vehicle_count,
        "traffic_level": traffic_level,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": "Chikkamagaluru"
    }

    print(event)

    event_id += 1