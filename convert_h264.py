import subprocess
import imageio_ffmpeg


input_video = r"runs\detect\predict-2\13020022_3840_2160_30fps.avi"

output_video = r"runs\detect\predict-2\ai_detected_h264.mp4"


ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()


command = [
    ffmpeg,
    "-y",
    "-i", input_video,
    "-c:v", "libx264",
    "-preset", "fast",
    "-crf", "23",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    output_video
]


subprocess.run(command, check=True)


print("H.264 conversion complete!")
print(f"Saved to: {output_video}")