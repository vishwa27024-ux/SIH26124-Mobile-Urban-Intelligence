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
    stream=True,
    verbose=False
)

vehicle_counts = []

for result in results:

    total_vehicles = 0

    for class_id in result.boxes.cls:
        class_id = int(class_id)

        if class_id in vehicle_classes:
            total_vehicles += 1

    vehicle_counts.append(total_vehicles)

average_vehicles = sum(vehicle_counts) / len(vehicle_counts)
peak_vehicles = max(vehicle_counts)

high_count = 0
medium_count = 0
low_count = 0

for count in vehicle_counts:

    if count <= 5:
        low_count += 1
    elif count <= 10:
        medium_count += 1
    else:
        high_count += 1

if average_vehicles <= 5:
    overall_traffic = "LOW"
elif average_vehicles <= 10:
    overall_traffic = "MEDIUM"
else:
    overall_traffic = "HIGH"

print("\n===== URBAN TRAFFIC INTELLIGENCE =====")
print(f"Average vehicles: {average_vehicles:.2f}")
print(f"Peak vehicles: {peak_vehicles}")
print(f"Low traffic frames: {low_count}")
print(f"Medium traffic frames: {medium_count}")
print(f"High traffic frames: {high_count}")
print(f"Overall traffic: {overall_traffic}")