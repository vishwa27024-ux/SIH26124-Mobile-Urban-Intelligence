from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.predict(
    source="13020022_3840_2160_30fps.mp4",
    save=True,
    conf=0.40
)

print("AI annotated video created!")