
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "external_models" / "Ultra-Fast-Lane-Detection-v2"
sys.path.insert(0, str(REPO))

from model.model_culane import parsingNet  # noqa: E402


def decode_lanes(pred, row_anchor, col_anchor, image_width, image_height):
    """Decode standard, single-batch CULane predictions."""
    loc_row = pred["loc_row"].detach().cpu()
    loc_col = pred["loc_col"].detach().cpu()
    exist_row = pred["exist_row"].argmax(dim=1).detach().cpu()
    exist_col = pred["exist_col"].argmax(dim=1).detach().cpu()

    ref_width, ref_height = 1640.0, 590.0
    scale_x = image_width / ref_width
    scale_y = image_height / ref_height
    lanes = []
    
    for lane_idx in (1, 2):
        points = []
        valid = exist_row[0, :, lane_idx]
        if valid.sum().item() > loc_row.shape[2] / 2:
            for k in range(loc_row.shape[2]):
                if valid[k].item() != 1:
                    continue
                logits = loc_row[0, :, k, lane_idx]
                center = int(logits.argmax().item())
                indices = torch.arange(
                    max(0, center - 1),
                    min(loc_row.shape[1] - 1, center + 1) + 1,
                )
                x_ref = (
                    logits[indices].softmax(0) * indices.float()
                ).sum().item() + 0.5
                x = int(x_ref / (loc_row.shape[1] - 1) * image_width)
                y = int(float(row_anchor[k]) * image_height)
                if 0 <= x < image_width and 0 <= y < image_height:
                    points.append((x, y))
        lanes.append(points)

    for lane_idx in (0, 3):
        points = []
        valid = exist_col[0, :, lane_idx]
        if valid.sum().item() > loc_col.shape[2] / 4:
            for k in range(loc_col.shape[2]):
                if valid[k].item() != 1:
                    continue
                logits = loc_col[0, :, k, lane_idx]
                center = int(logits.argmax().item())
                indices = torch.arange(
                    max(0, center - 1),
                    min(loc_col.shape[1] - 1, center + 1) + 1,
                )
                y_ref = (
                    logits[indices].softmax(0) * indices.float()
                ).sum().item() + 0.5
                y = int(y_ref / (loc_col.shape[1] - 1) * image_height)
                x = int(float(col_anchor[k]) * image_width)
                if 0 <= x < image_width and 0 <= y < image_height:
                    points.append((x, y))
        lanes.append(points)

    return lanes


def decode_lanes_tta(pred, row_anchor, col_anchor, image_width, image_height):
    """Decode five-view TTA predictions using the official CULane logic."""
    row_preds = torch.chunk(pred["loc_row"].detach().cpu(), 5, dim=0)
    row_exists = torch.chunk(
        pred["exist_row"].argmax(dim=1).detach().cpu(), 5, dim=0
    )

    col_preds = torch.chunk(pred["loc_col"].detach().cpu(), 5, dim=0)
    col_exists = torch.chunk(
        pred["exist_col"].argmax(dim=1).detach().cpu(), 5, dim=0
    )

    ref_width, ref_height = 1640.0, 590.0
    scale_x = image_width / ref_width
    scale_y = image_height / ref_height

    def decode_position(logits, grid_size):
        center = int(logits.argmax().item())
        indices = torch.arange(
            max(0, center - 1),
            min(grid_size - 1, center + 1) + 1,
        )
        probabilities = logits[indices].softmax(dim=0)
        return (
            (probabilities * indices.float()).sum().item() + 0.5
        ) / (grid_size - 1)

    lanes = []

    # Row predictions: normal, left-shifted, right-shifted.
    grid_row = row_preds[0].shape[1]
    num_row = row_preds[0].shape[2]

    for lane_idx in (1, 2):
        points = []
        valid = row_exists[0][0, :, lane_idx]

        if valid.sum().item() >= num_row / 2:
            for k in range(num_row):
                if valid[k].item() != 1:
                    continue

                values = []

                for pred_idx, offset in ((0, 0.0), (1, 1640.0 / 25),
                                         (2, -1640.0 / 25)):
                    if row_exists[pred_idx][0, k, lane_idx].item() == 1:
                        x_ref = decode_position(
                            row_preds[pred_idx][0, :, k, lane_idx],
                            grid_row,
                        ) * ref_width
                        values.append(x_ref + offset)

                if len(values) >= 2:
                    x = int(round(np.mean(values) * scale_x))
                    y = int(round(float(row_anchor[k]) * ref_height * scale_y))

                    if 0 <= x < image_width and 0 <= y < image_height:
                        points.append((x, y))

        if len(points) >= num_row / 2:
            lanes.append(points)
        else:
            lanes.append([])

    # Column predictions: normal, up-shifted, down-shifted.
    grid_col = col_preds[0].shape[1]
    num_col = col_preds[0].shape[2]

    up_offset = 32.0 / 534.0 * ref_height

    for lane_idx in (0, 3):
        points = []
        valid = col_exists[0][0, :, lane_idx]

        if valid.sum().item() >= num_col / 4:
            for k in range(num_col):
                if valid[k].item() != 1:
                    continue

                values = []

                for pred_idx, offset in ((0, 0.0), (3, up_offset),
                                         (4, -up_offset)):
                    if col_exists[pred_idx][0, k, lane_idx].item() == 1:
                        y_ref = decode_position(
                            col_preds[pred_idx][0, :, k, lane_idx],
                            grid_col,
                        ) * ref_height
                        values.append(y_ref + offset)

                if len(values) >= 2:
                    x = int(round(float(col_anchor[k]) * ref_width * scale_x))
                    y = int(round(np.mean(values) * scale_y))

                    if 0 <= x < image_width and 0 <= y < image_height:
                        points.append((x, y))

        if len(points) >= num_col / 4:
            lanes.append(points)
        else:
            lanes.append([])

    return lanes



def main():
    parser = argparse.ArgumentParser(
        description="Ultra-Fast Lane Detection with CPU inference."
    )
    parser.add_argument(
        "--image",
        default="data/samples/test_road.png",
        help="Input image path relative to the project root.",
    )
    parser.add_argument(
        "--output",
        default="outputs/lanes/ultra_fast_lane_detection.png",
        help="Output image path relative to the project root.",
    )
    parser.add_argument(
        "--save-preprocessed",
        action="store_true",
        help="Save the resized-and-cropped model input for diagnostics.",
    )
    args = parser.parse_args()

    image_path = ROOT / args.image
    output_path = ROOT / args.output
    checkpoint_path = ROOT / "artifacts" / "models" / "culane_res18.pth"

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    original = cv2.imread(str(image_path))
    if original is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    image_height, image_width = original.shape[:2]
    device = torch.device("cpu")

    # CULane ResNet18 configuration.
    cfg_width = 1600
    cfg_height = 320
    crop_ratio = 0.6

    num_grid_row = 200
    num_cls_row = 72
    num_grid_col = 100
    num_cls_col = 81

    print("Building model...")

    net = parsingNet(
        pretrained=False,
        backbone="18",
        num_grid_row=num_grid_row,
        num_cls_row=num_cls_row,
        num_grid_col=num_grid_col,
        num_cls_col=num_cls_col,
        num_lane_on_row=4,
        num_lane_on_col=4,
        use_aux=False,
        input_height=cfg_height,
        input_width=cfg_width,
        fc_norm=True,
    ).to(device)

    print("Loading pretrained checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state = {
        key.removeprefix("module."): value
        for key, value in checkpoint["model"].items()
    }

    net.load_state_dict(state, strict=True)
    net.eval()

    # Match the official demo preprocessing.
    image = Image.open(image_path).convert("RGB")

    image_transform = transforms.Compose([
        transforms.Resize(
            (int(cfg_height / crop_ratio), cfg_width)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    tensor = image_transform(image)
    tensor = tensor[:, -cfg_height:, :]

    if args.save_preprocessed:
        # Undo normalization so the exact model input can be visually inspected.
        mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
        preview = (tensor * std + mean).clamp(0, 1)
        preview = (preview.permute(1, 2, 0).numpy()[:, :, ::-1] * 255).astype(np.uint8)
        preview_path = output_path.with_name(output_path.stem + "_model_input.png")
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(preview_path), preview):
            raise RuntimeError(f"Could not save preprocessed input: {preview_path}")
        print(f"Model input preview: {preview_path}")

    tensor = tensor.unsqueeze(0).to(device)

    print("Running CPU inference...")

    with torch.inference_mode():
        pred = net(tensor)

    required_keys = {"loc_row", "loc_col", "exist_row", "exist_col"}

    if not isinstance(pred, dict):
        raise TypeError(f"Expected model output dict, got {type(pred).__name__}")

    if not required_keys.issubset(pred.keys()):
        raise KeyError(f"Missing model outputs: {required_keys - set(pred.keys())}")

    row_anchor = np.linspace(0.42, 1.0, num_cls_row)
    col_anchor = np.linspace(0.0, 1.0, num_cls_col)

    print(
        "Row existence counts:",
        pred["exist_row"].argmax(1)[0].sum(dim=0).tolist(),
    )
    print(
        "Column existence counts:",
        pred["exist_col"].argmax(1)[0].sum(dim=0).tolist(),
    )

    lanes = decode_lanes(
        pred,
        row_anchor,
        col_anchor,
        image_width,
        image_height,
    )

    # BGR colors for OpenCV.
    colors = [
        (0, 255, 0),
        (0, 200, 255),
        (255, 100, 0),
        (255, 0, 255),
    ]

    for lane, color in zip(lanes, colors):
        if len(lane) < 2:
            continue

        points = sorted(lane, key=lambda point: point[1])

        for first, second in zip(points, points[1:]):
            cv2.line(original, first, second, color, 3, cv2.LINE_AA)

        for point in points:
            cv2.circle(original, point, 3, color, -1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), original):
        raise RuntimeError(f"Could not save output: {output_path}")

    print("\n--- Lane Detection Results ---")
    print(f"Device:       {device}")
    print(f"Input image:  {image_path}")
    print(f"Output image: {output_path}")
    print(f"Lane points:  {[len(lane) for lane in lanes]}")
    print("Inference completed.")


if __name__ == "__main__":
    main()
