import json
from ultralytics import YOLO


VIDEO_FILE = "13020022_3840_2160_30fps.mp4"
CACHE_FILE = "traffic_results.json"


model = YOLO("yolo11n.pt")


print("Starting AI analysis...")

results = model(
    source=VIDEO_FILE,
    stream=True,
    verbose=False
)

vehicle_counts = []

for result in results:

    vehicle_count = 0

    for class_id in result.boxes.cls:

        class_id = int(class_id)

        if class_id in [2, 3, 5, 7]:
            vehicle_count += 1

    vehicle_counts.append(vehicle_count)


# Calculate summary

average_vehicles = (
    sum(vehicle_counts) / len(vehicle_counts)
)

peak_vehicles = max(vehicle_counts)

low_frames = 0
medium_frames = 0
high_frames = 0


for count in vehicle_counts:

    if count <= 5:
        low_frames += 1

    elif count <= 10:
        medium_frames += 1

    else:
        high_frames += 1


if average_vehicles <= 5:
    overall_traffic = "LOW"

elif average_vehicles <= 10:
    overall_traffic = "MEDIUM"

else:
    overall_traffic = "HIGH"


# Create data to save

traffic_data = {

    "frames": len(vehicle_counts),

    "average": round(
        average_vehicles,
        2
    ),

    "peak": peak_vehicles,

    "low": low_frames,

    "medium": medium_frames,

    "high": high_frames,

    "overall": overall_traffic

}


# Save to JSON

with open(
    CACHE_FILE,
    "w"
) as file:

    json.dump(
        traffic_data,
        file,
        indent=4
    )


print("\n===== TRAFFIC CACHE CREATED =====")

print(
    f"Frames: {traffic_data['frames']}"
)

print(
    f"Average vehicles: {traffic_data['average']}"
)

print(
    f"Peak vehicles: {traffic_data['peak']}"
)

print(
    f"Overall traffic: {traffic_data['overall']}"
)

print(
    f"Saved to: {CACHE_FILE}"
)