from ultralytics import YOLO
import cv2
from pathlib import Path


def test_bytetrack_generates_persistent_ids():
    root = Path(__file__).resolve().parents[1]

    model_path = root / "artifacts/models/yolo11n.pt"
    video_path = root / "data/samples/traffic.mp4"

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video_path))

    assert cap.isOpened(), "Could not open test video"

    frames_processed = 0
    frames_with_ids = 0

    try:
        while frames_processed < 10:
            success, frame = cap.read()

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

            frames_processed += 1

            if result.boxes is not None and result.boxes.id is not None:
                frames_with_ids += 1

        assert frames_processed == 10
        assert frames_with_ids > 0

    finally:
        cap.release()