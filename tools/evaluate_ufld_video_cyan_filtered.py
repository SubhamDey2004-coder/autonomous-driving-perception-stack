import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "external_models" / "Ultra-Fast-Lane-Detection-v2"
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(REPO))

from model.model_culane import parsingNet


# Paths
VIDEO_PATH = REPO / "example.mp4"
CHECKPOINT_PATH = ROOT / "artifacts" / "models" / "culane_res18.pth"
OUTPUT_PATH = ROOT / "outputs" / "lanes" / "traffic_ufld_cyan_filtered.mp4"

# Inference and smoothing settings
SAMPLE_STRIDE = 3
SMOOTHING_ALPHA = 0.35
MAX_MISSED_INFERENCES = 15
MIN_POINTS = 6
MAX_MEDIAN_SHIFT_RATIO = 0.08
MAX_POINT_SHIFT_RATIO = 0.04
ALIGNED_POINT_COUNT = 48

LANE_COLORS = [
    (0, 255, 0),      # Lane 0: green
    (0, 255, 255),    # Lane 1: yellow
    (255, 0, 255),    # Lane 2: magenta
    (255, 255, 0),    # Lane 3: cyan
]


def build_model():
    """Load the pretrained CULane model on CPU."""
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
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint has no 'model' state dictionary: {CHECKPOINT_PATH}")

    state = {
        key.removeprefix("module."): value
        for key, value in checkpoint["model"].items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def decode_predictions(pred, width, height, row_anchor, col_anchor):
    """Decode predictions into four lane point lists, ordered by lane ID 0-3."""
    row_lanes = {}
    col_lanes = {}

    row_exist = pred["exist_row"].argmax(1)[0]
    col_exist = pred["exist_col"].argmax(1)[0]

    # Row-based lanes: lane IDs 1 and 2.
    for lane_idx in (1, 2):
        points = []
        valid = row_exist[:, lane_idx]

        if valid.sum().item() > 72 / 2:
            for k in range(72):
                if valid[k].item() != 1:
                    continue

                logits = pred["loc_row"][0, :, k, lane_idx]
                center = int(logits.argmax().item())
                indices = torch.arange(
                    max(0, center - 1),
                    min(199, center + 1) + 1,
                    device=logits.device,
                )
                x_ref = (
                    logits[indices].softmax(0) * indices.float()
                ).sum().item() + 0.5

                x = int(x_ref / 199 * width)
                y = int(float(row_anchor[k]) * height)
                if 0 <= x < width and 0 <= y < height:
                    points.append((x, y))

        row_lanes[lane_idx] = points

    # Column-based lanes: lane IDs 0 and 3.
    for lane_idx in (0, 3):
        points = []
        valid = col_exist[:, lane_idx]

        if valid.sum().item() > 81 / 4:
            for k in range(81):
                if valid[k].item() != 1:
                    continue

                logits = pred["loc_col"][0, :, k, lane_idx]
                center = int(logits.argmax().item())
                indices = torch.arange(
                    max(0, center - 1),
                    min(99, center + 1) + 1,
                    device=logits.device,
                )
                y_ref = (
                    logits[indices].softmax(0) * indices.float()
                ).sum().item() + 0.5

                y = int(y_ref / 99 * height)
                x = int(float(col_anchor[k]) * width)
                if 0 <= x < width and 0 <= y < height:
                    points.append((x, y))

        col_lanes[lane_idx] = points

    # Return lanes in actual model lane-ID order: 0, 1, 2, 3.
    return [
        col_lanes.get(0, []),
        row_lanes.get(1, []),
        row_lanes.get(2, []),
        col_lanes.get(3, []),
    ]


def remove_spikes(points, variable_axis, window_size=5):
    """Filter isolated coordinate spikes while preserving fixed scanline anchors."""
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 5:
        return points

    # The window must be odd so every point has a central median.
    window_size = max(3, int(window_size))
    if window_size % 2 == 0:
        window_size += 1
    if len(points) < window_size:
        window_size = len(points) if len(points) % 2 == 1 else len(points) - 1
    if window_size < 3:
        return points

    result = points.copy()
    radius = window_size // 2
    values = points[:, variable_axis]
    padded = np.pad(values, (radius, radius), mode="edge")
    result[:, variable_axis] = np.array(
        [
            np.median(padded[i:i + window_size])
            for i in range(len(values))
        ],
        dtype=np.float32,
    )
    return result


def align_lane(points, lane_idx, count=ALIGNED_POINT_COUNT, axis_range=None):
    """Represent each lane on a fixed spatial axis before comparing frames.

    Row-predicted lanes (IDs 1 and 2) have fixed y anchors, so interpolate x(y).
    Column-predicted lanes (IDs 0 and 3) have fixed x anchors, so interpolate y(x).
    """
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 2:
        return np.empty((0, 2), dtype=np.float32)

    axis = 1 if lane_idx in (1, 2) else 0
    variable = 0 if axis == 1 else 1
    points = points[np.argsort(points[:, axis])]
    axis_values = points[:, axis]
    variable_values = points[:, variable]

    # Remove duplicate anchor coordinates before interpolation.
    unique_axis, unique_indices = np.unique(axis_values, return_index=True)
    variable_values = variable_values[unique_indices]
    if len(unique_axis) < 2:
        return np.empty((0, 2), dtype=np.float32)

    if axis_range is None:
        low, high = float(unique_axis[0]), float(unique_axis[-1])
    else:
        low = max(float(unique_axis[0]), float(axis_range[0]))
        high = min(float(unique_axis[-1]), float(axis_range[1]))
    if high - low < 2.0:
        return np.empty((0, 2), dtype=np.float32)

    target_axis = np.linspace(low, high, count, dtype=np.float32)
    target_variable = np.interp(target_axis, unique_axis, variable_values)
    if axis == 1:
        return np.column_stack((target_variable, target_axis)).astype(np.float32)
    return np.column_stack((target_axis, target_variable)).astype(np.float32)


def smooth_lane(previous, current, lane_idx, alpha, width, height):
    """Align points spatially, then smooth; reject weak or implausible updates."""
    if current is None or len(current) < MIN_POINTS:
        return None

    axis = 1 if lane_idx in (1, 2) else 0
    variable_axis = 0 if axis == 1 else 1
    # Lane 3 (cyan) is the column-predicted lane and shows the most local
    # jaggedness in the current video. Use a wider median window for this lane
    # only; keep the already-good green, yellow, and magenta lanes unchanged.
    spike_window = 9 if lane_idx == 3 else 5
    cleaned = remove_spikes(current, variable_axis, window_size=spike_window)
    new = align_lane(cleaned, lane_idx)
    if len(new) < MIN_POINTS:
        return None

    if previous is None or len(previous) < MIN_POINTS:
        return [(int(round(x)), int(round(y))) for x, y in new]

    old_raw = np.asarray(previous, dtype=np.float32)
    old = align_lane(old_raw, lane_idx)
    if len(old) < MIN_POINTS:
        return [(int(round(x)), int(round(y))) for x, y in new]

    # Compare only the spatial range supported by both predictions. This prevents
    # points at different road positions from being paired by list index.
    old_axis = old[:, axis]
    new_axis = new[:, axis]
    low = max(float(old_axis.min()), float(new_axis.min()))
    high = min(float(old_axis.max()), float(new_axis.max()))
    if high - low < 2.0:
        return None
    old = align_lane(old, lane_idx, axis_range=(low, high))
    new = align_lane(new, lane_idx, axis_range=(low, high))
    count = min(len(old), len(new))
    if count < MIN_POINTS:
        return None
    old = old[:count]
    new = new[:count]

    max_dimension = float(max(width, height))
    displacement = np.linalg.norm(new - old, axis=1)
    if float(np.median(displacement)) > MAX_MEDIAN_SHIFT_RATIO * max_dimension:
        return None

    delta = new - old
    distances = np.linalg.norm(delta, axis=1)
    max_point_shift = MAX_POINT_SHIFT_RATIO * max_dimension
    scale = np.ones_like(distances)
    mask = distances > max_point_shift
    scale[mask] = max_point_shift / distances[mask]
    limited_new = old + delta * scale[:, None]

    smoothed = alpha * limited_new + (1.0 - alpha) * old
    # Keep the anchor axis strictly ordered for clean polyline drawing.
    smoothed = smoothed[np.argsort(smoothed[:, axis])]
    return [(int(round(x)), int(round(y))) for x, y in smoothed]

def draw_lanes(frame, lanes):
    """Draw the current stable lane predictions."""
    height, width = frame.shape[:2]
    for points, color in zip(lanes, LANE_COLORS):
        if points is None or len(points) < 2:
            continue

        pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
        pts[:, 0, 0] = np.clip(pts[:, 0, 0], 0, width - 1)
        pts[:, 0, 1] = np.clip(pts[:, 0, 1], 0, height - 1)
        cv2.polylines(frame, [pts], False, color, 3, cv2.LINE_AA)


def main():
    if SAMPLE_STRIDE < 1:
        raise ValueError("SAMPLE_STRIDE must be at least 1.")
    if not 0.0 < SMOOTHING_ALPHA <= 1.0:
        raise ValueError("SMOOTHING_ALPHA must be in the range (0, 1].")
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("The input video reported invalid frame dimensions.")
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {OUTPUT_PATH}")

    frame_index = 0
    inference_count = 0
    stable_lanes = [None, None, None, None]
    missed_counts = [0, 0, 0, 0]

    try:
        model = build_model()
        # Match the official CULane demo preprocessing.
        transform = transforms.Compose(
            [
                transforms.Resize((int(320 / 0.6), 1600)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        row_anchor = np.linspace(0.42, 1.0, 72)
        col_anchor = np.linspace(0.0, 1.0, 81)

        print(f"Input: {width}x{height} at {fps:.2f} FPS")
        print(f"Total frames: {total_frames}")
        print("Model device: CPU")
        print(f"Inference every {SAMPLE_STRIDE} frame(s)")
        print("Running stabilized lane inference...")

        with torch.inference_mode():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_index % SAMPLE_STRIDE == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb)
                    tensor = transform(image)[:, -320:, :].unsqueeze(0)
                    predictions = model(tensor)

                    current_lanes = decode_predictions(
                        predictions,
                        width,
                        height,
                        row_anchor,
                        col_anchor,
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

                # Draw the latest stable prediction on every frame.
                draw_lanes(frame, stable_lanes)
                writer.write(frame)
                frame_index += 1

                if frame_index % 30 == 0:
                    print(f"Processed {frame_index}/{total_frames} frames")

        print("\nEvaluation finished.")
        print(f"Frames written: {frame_index}")
        print(f"Inference frames: {inference_count}")
        print(f"Output: {OUTPUT_PATH}")

    finally:
        cap.release()
        writer.release()


if __name__ == "__main__":
    main()
