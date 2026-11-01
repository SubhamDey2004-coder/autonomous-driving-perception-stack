
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "models" / "yolo11n.pt"
IMAGE_PATH = ROOT / "data" / "samples" / "test_road.png"
OUTPUT_DIR = ROOT / "outputs" / "detection"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Test image not found: {IMAGE_PATH}\n"
            "Place a road image at data/samples/test_road.jpg"
        )

    model = YOLO(str(MODEL_PATH)) if MODEL_PATH.exists() else YOLO("yolo11n.pt")

    results = model.predict(
        source=str(IMAGE_PATH),
        conf=0.35,
        device="cpu",
        verbose=False,
    )

    result = results[0]
    annotated = result.plot()

    output_path = OUTPUT_DIR / "detection_result.jpg"
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Could not save image: {output_path}")

    if MODEL_PATH.exists() is False:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(MODEL_PATH))

    print(f"Detected objects: {len(result.boxes)}")
    print(f"Annotated image saved to: {output_path}")


if __name__ == "__main__":
    main()
