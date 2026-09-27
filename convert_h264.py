from pathlib import Path
import subprocess
import imageio_ffmpeg


INPUT_VIDEO = Path("traffic_ai_detected.mp4")
OUTPUT_VIDEO = Path("traffic_ai_h264.mp4")


def main():
    print("=" * 50)
    print("MUI H.264 VIDEO CONVERSION")
    print("=" * 50)

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Input video not found:\n{INPUT_VIDEO.resolve()}"
        )

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(INPUT_VIDEO),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(OUTPUT_VIDEO),
    ]

    print(f"Input : {INPUT_VIDEO}")
    print(f"Output: {OUTPUT_VIDEO}")
    print()
    print("Converting...")
    print()

    subprocess.run(command, check=True)

    if not OUTPUT_VIDEO.exists():
        raise RuntimeError(
            "FFmpeg finished, but the output video was not created."
        )

    size_mb = OUTPUT_VIDEO.stat().st_size / (1024 * 1024)

    print()
    print("=" * 50)
    print("H.264 CONVERSION COMPLETE")
    print("=" * 50)
    print(f"Input video : {INPUT_VIDEO}")
    print(f"Output video: {OUTPUT_VIDEO}")
    print(f"Output size : {size_mb:.2f} MB")
    print("=" * 50)


if __name__ == "__main__":
    main()