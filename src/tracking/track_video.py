
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "models" / "yolo11n.pt"
VIDEO_PATH = ROOT / "data" / "samples" / "traffic.mp4"
OUTPUT_DIR = ROOT / "outputs" / "tracking"
OUTPUT_PATH = OUTPUT_DIR / "tracked_traffic.mp4"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))
    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {OUTPUT_PATH}")

    frame_count = 0
    tracked_objects = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            result = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.25,
                device="cpu",
                verbose=False,
            )[0]

            annotated = result.plot()
            writer.write(annotated)

            frame_count += 1

            if result.boxes is not None and result.boxes.id is not None:
                tracked_objects += len(result.boxes.id)

            print(
                f"\rProcessed frames: {frame_count} | "
                f"Tracked detections: {tracked_objects}",
                end="",
                flush=True,
            )
    finally:
        capture.release()
        writer.release()

    print(f"\nSaved tracked video: {OUTPUT_PATH}")
    print(f"Total frames processed: {frame_count}")
    print(f"Total tracked detections across frames: {tracked_objects}")


if __name__ == "__main__":
    main()
