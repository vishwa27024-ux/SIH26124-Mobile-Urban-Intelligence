from pathlib import Path
import subprocess
import sys

try:
    import imageio_ffmpeg
except ImportError:
    print("ERROR: imageio-ffmpeg is not installed.")
    print("Run:")
    print(r".\.venv\Scripts\python.exe -m pip install imageio-ffmpeg")
    sys.exit(1)


BASE_DIR = Path(__file__).resolve().parent

INPUT_VIDEO = BASE_DIR / "traffic_ai_detected.mp4"
OUTPUT_VIDEO = BASE_DIR / "traffic_ai_h264.mp4"

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def main():
    print("=" * 60)
    print("MUI BROWSER-COMPATIBLE VIDEO CONVERTER")
    print("=" * 60)

    if not INPUT_VIDEO.exists():
        print()
        print("ERROR: Input video not found:")
        print(INPUT_VIDEO)
        print()
        print("Make sure traffic_ai_detected.mp4 is in the project folder.")
        sys.exit(1)

    print(f"Input : {INPUT_VIDEO}")
    print(f"Output: {OUTPUT_VIDEO}")
    print()
    print("Converting...")
    print()

    command = [
        FFMPEG,
        "-y",
        "-i",
        str(INPUT_VIDEO),

        # Browser-compatible video codec
        "-c:v",
        "libx264",

        # Maximum compatibility with Chrome/Edge/Firefox
        "-pix_fmt",
        "yuv420p",

        # Good quality / reasonable file size
        "-preset",
        "medium",
        "-crf",
        "23",

        # Keep original frame rate
        "-r",
        "30",

        # MP4 metadata at beginning of file
        "-movflags",
        "+faststart",

        # Video-only output
        "-an",

        str(OUTPUT_VIDEO),
    ]

    try:
        result = subprocess.run(
            command,
            check=False,
        )

    except Exception as error:
        print("ERROR while running FFmpeg:")
        print(error)
        sys.exit(1)

    if result.returncode != 0:
        print()
        print("ERROR: FFmpeg conversion failed.")
        sys.exit(result.returncode)

    if not OUTPUT_VIDEO.exists():
        print()
        print("ERROR: Output video was not created.")
        sys.exit(1)

    size_mb = OUTPUT_VIDEO.stat().st_size / (1024 * 1024)

    print()
    print("=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Output file : {OUTPUT_VIDEO.name}")
    print(f"File size   : {size_mb:.2f} MB")
    print()
    print("Browser compatibility settings:")
    print("Codec       : H.264")
    print("Pixel format: yuv420p")
    print("Fast start  : enabled")
    print("=" * 60)


if __name__ == "__main__":
    main()