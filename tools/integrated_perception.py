
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "external_models" / "Ultra-Fast-Lane-Detection-v2"

sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(REPO))

from model.model_culane import parsingNet
from tools.evaluate_ufld_video import (
    decode_predictions,
    smooth_lane,
    draw_lanes,
    SAMPLE_STRIDE,
    SMOOTHING_ALPHA,
    MAX_MISSED_INFERENCES,
)

VIDEO_PATH = REPO / "example.mp4"
YOLO_PATH = ROOT / "artifacts" / "models" / "yolo11n.pt"
LANE_PATH = ROOT / "artifacts" / "models" / "culane_res18.pth"
OUTPUT_PATH = ROOT / "outputs" / "perception" / "integrated_traffic.mp4"


def build_lane_model():
    model = parsingNet(
        pretrained=False,
        backbone="18",
        num_grid_row=200,
        num_cls_row=72,
        num_grid_col=100,
        num_cls_col=81,
        num_lane_on_row=4,
        num_lane_on_col=4,
        use_aux=False,
        input_height=320,
        input_width=1600,
        fc_norm=True,
    )

    checkpoint = torch.load(
        LANE_PATH, map_location="cpu", weights_only=False
    )
    if "model" not in checkpoint:
        raise KeyError(f"Missing model weights in {LANE_PATH}")

    state = {
        key.removeprefix("module."): value
        for key, value in checkpoint["model"].items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main():
    for path in (VIDEO_PATH, YOLO_PATH, LANE_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

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
        raise RuntimeError(f"Cannot create output: {OUTPUT_PATH}")

    frame_index = 0
    inference_count = 0
    stable_lanes = [None] * 4
    missed_counts = [0] * 4

    lane_time = 0.0
    tracking_time = 0.0
    writing_time = 0.0
    preprocessing_time = 0.0
    annotation_time = 0.0

    transform = transforms.Compose([
        transforms.Resize((int(320 / 0.6), 1600)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    row_anchor = np.linspace(0.42, 1.0, 72)
    col_anchor = np.linspace(0.0, 1.0, 81)

    start_time = time.perf_counter()

    try:
        print("Loading lane model on CPU...")
        lane_model = build_lane_model()

        print("Loading YOLO vehicle detector...")
        vehicle_model = YOLO(str(YOLO_PATH))

        print(f"Processing video: {VIDEO_PATH}")
        print(f"Video dimensions: {width}x{height} at {fps:.2f} FPS")
        print(f"Lane inference every {SAMPLE_STRIDE} frames.")

        with torch.inference_mode():
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                # Lane inference and stabilization.
                if frame_index % SAMPLE_STRIDE == 0:
                    prep_start = time.perf_counter()

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb)
                    tensor = transform(image)[:, -320:, :].unsqueeze(0)

                    preprocessing_time += time.perf_counter() - prep_start

                    lane_start = time.perf_counter()
                    predictions = lane_model(tensor)
                    lane_time += time.perf_counter() - lane_start

                    current_lanes = decode_predictions(
                        predictions, width, height, row_anchor, col_anchor
                    )

                    for lane_idx, current in enumerate(current_lanes):
                        candidate = smooth_lane(
                            stable_lanes[lane_idx],
                            current,
                            lane_idx,
                            SMOOTHING_ALPHA,
                            width,
                            height,
                        )

                        if candidate is not None:
                            stable_lanes[lane_idx] = candidate
                            missed_counts[lane_idx] = 0
                        else:
                            missed_counts[lane_idx] += 1
                            if missed_counts[lane_idx] > MAX_MISSED_INFERENCES:
                                stable_lanes[lane_idx] = None

                    inference_count += 1

                # YOLO detection and ByteTrack tracking.
                tracking_start = time.perf_counter()
                result = vehicle_model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=0.25,
                    device="cpu",
                    verbose=False,
                )[0]
                tracking_time += time.perf_counter() - tracking_start

                # Draw vehicle and lane annotations.
                annotation_start = time.perf_counter()
                annotated = result.plot()
                draw_lanes(annotated, stable_lanes)
                annotation_time += time.perf_counter() - annotation_start

                # Write the combined frame.
                writing_start = time.perf_counter()
                writer.write(annotated)
                writing_time += time.perf_counter() - writing_start

                frame_index += 1

                if frame_index % 30 == 0:
                    print(f"Processed {frame_index} frames", flush=True)

    finally:
        capture.release()
        writer.release()

    elapsed = time.perf_counter() - start_time
    effective_fps = frame_index / elapsed if elapsed > 0 else 0.0

    print("\nIntegration and profiling complete.")
    print(f"Frames processed: {frame_index}")
    print(f"Lane inference frames: {inference_count}")
    print(f"Total processing time: {elapsed:.2f} seconds")
    print(f"Effective processing FPS: {effective_fps:.2f}")
    print("\nComponent timings:")
    print(f"  Lane model forward pass: {lane_time:.2f} seconds")
    print(f"  YOLO + tracking:         {tracking_time:.2f} seconds")
    print(f"  Lane preprocessing:      {preprocessing_time:.2f} seconds")
    print(f"  Annotation/drawing:      {annotation_time:.2f} seconds")
    print(f"  Video writing:           {writing_time:.2f} seconds")
    print(f"\nOutput video: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
