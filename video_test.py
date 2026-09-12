from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.predict(
    source="13020022_3840_2160_30fps.mp4",
    save=True
)

print("VIDEO PROCESSING COMPLETE")