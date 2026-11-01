
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
from src.lanes.ultra_fast_lane_detection import decode_lanes
from demo import pred2coords


def main():
    image_path = ROOT / "data/samples/ufldv2_test_frame.png"
    checkpoint_path = ROOT / "artifacts/models/culane_res18.pth"

    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device("cpu")
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    transform = transforms.Compose([
        transforms.Resize((int(320 / 0.6), 1600)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    tensor = transform(image)[:, -320:, :].unsqueeze(0)

    net = parsingNet(
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
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    state = {
        key.removeprefix("module."): value
        for key, value in checkpoint["model"].items()
    }
    net.load_state_dict(state, strict=True)
    net.eval()

    with torch.inference_mode():
        pred = net(tensor)

    row_anchor = np.linspace(0.42, 1.0, 72)
    col_anchor = np.linspace(0.0, 1.0, 81)

    # The official decoder moves loc_row/loc_col to CPU in-place.
    official_pred = {
        key: value.clone() for key, value in pred.items()
    }

    official = pred2coords(
        official_pred,
        row_anchor,
        col_anchor,
        original_image_width=image_width,
        original_image_height=image_height,
    )

    custom = decode_lanes(
        pred,
        row_anchor,
        col_anchor,
        image_width,
        image_height,
    )

    print("\nOfficial lane point counts:", [len(lane) for lane in official])
    print("Custom lane point counts:  ", [len(lane) for lane in custom])

    
    print("\nCoordinate comparison by lane:")
    for lane_index in range(4):
        official_lane = (
            official[lane_index] if lane_index < len(official) else []
        )
        custom_lane = custom[lane_index]

        print(f"\nLane {lane_index}:")
        print(f"  Official points: {len(official_lane)}")
        print(f"  Custom points:   {len(custom_lane)}")

        official_set = set(map(tuple, official_lane))
        custom_set = set(map(tuple, custom_lane))

        missing = sorted(official_set - custom_set)
        extra = sorted(custom_set - official_set)

        print(f"  Missing from custom: {missing}")
        print(f"  Extra in custom:     {extra}")

        if not missing and not extra:
            print("  Result: EXACT MATCH")


    print("\nDecoder comparison complete.")


if __name__ == "__main__":
    main()
