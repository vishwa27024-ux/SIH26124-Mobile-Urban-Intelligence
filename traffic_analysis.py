from ultralytics import YOLO

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

    total_vehicles = sum(counts.values())

    if total_vehicles <= 5:
        traffic_level = "LOW"
    elif total_vehicles <= 10:
        traffic_level = "MEDIUM"
    else:
        traffic_level = "HIGH"

    print(
        f"Cars: {counts['car']} | "
        f"Motorcycles: {counts['motorcycle']} | "
        f"Buses: {counts['bus']} | "
        f"Trucks: {counts['truck']} | "
        f"Total: {total_vehicles} | "
        f"Traffic: {traffic_level}"
    )