
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "models" / "yolo11n.pt"


class ObjectDetector:
    def __init__(self, confidence=0.35, device="cpu"):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model weights not found: {MODEL_PATH}"
            )

        self.model = YOLO(str(MODEL_PATH))
        self.confidence = confidence
        self.device = device

    def detect(self, image):
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )
        return results[0]


if __name__ == "__main__":
    import cv2

    image_path = ROOT / "data" / "samples" / "test_road.png"
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    detector = ObjectDetector()
    result = detector.detect(image)

    print(f"Detected objects: {len(result.boxes)}")
    print("Classes:", [
        result.names[int(box.cls[0])]
        for box in result.boxes
    ])

    output_path = ROOT / "outputs" / "detection" / "detector_test.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), result.plot()):
        raise RuntimeError(f"Could not save output: {output_path}")

    print(f"Saved result: {output_path}")
