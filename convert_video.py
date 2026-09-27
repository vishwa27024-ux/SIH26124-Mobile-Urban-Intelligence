import cv2

input_video = "runs/detect/predict-2/13020022_3840_2160_30fps.avi"
output_video = "runs/detect/predict-2/ai_detected.mp4"

cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    out.write(frame)

cap.release()
out.release()

print("MP4 conversion complete!")
print(f"Saved to: {output_video}")