
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = ROOT / "data" / "samples" / "test_road.png"
OUTPUT_DIR = ROOT / "outputs" / "lanes"
OUTPUT_PATH = OUTPUT_DIR / "lane_detection.jpg"



def detect_lanes(image):
    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Restrict processing to the road area.
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.38), int(height * 0.50)),
        (int(width * 0.62), int(height * 0.50)),
        (int(width * 0.95), height),
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    road_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        road_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=30,
    )

    # Collect candidate lines on the left and right.
    left_candidates = []
    right_candidates = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(
                int, np.asarray(line).reshape(-1)
            )

            dx = x2 - x1
            dy = y2 - y1

            if abs(dx) < 5:
                continue

            slope = dy / dx
            length = np.hypot(dx, dy)

            # Ignore horizontal edges and weakly sloped segments.
            if abs(slope) < 0.4 or abs(slope) > 3.0:
                continue

            # Represent the line as x = m*y + b.
            m = dx / dy if abs(dy) > 1 else None

            if m is None:
                continue

            b = x1 - m * y1
            candidate = (m, b, length)

            midpoint_x = (x1 + x2) / 2

            if slope < 0 and midpoint_x < width * 0.55:
                left_candidates.append(candidate)
            elif slope > 0 and midpoint_x > width * 0.45:
                right_candidates.append(candidate)

    overlay = image.copy()
    y_bottom = height - 1
    y_top = int(height * 0.55)

    # Draw one length-weighted representative line per side.
    for candidates in (left_candidates, right_candidates):
        if not candidates:
            continue

        weights = np.array(
            [item[2] for item in candidates], dtype=float
        )
        parameters = np.array(
            [[item[0], item[1]] for item in candidates],
            dtype=float,
        )

        m, b = np.average(parameters, axis=0, weights=weights)

        x_bottom = int(m * y_bottom + b)
        x_top = int(m * y_top + b)

        if -width <= x_bottom <= 2 * width and \
           -width <= x_top <= 2 * width:
            cv2.line(
                overlay,
                (x_bottom, y_bottom),
                (x_top, y_top),
                (0, 255, 0),
                5,
            )

    return overlay, road_edges


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise RuntimeError(f"Could not read image: {IMAGE_PATH}")

    annotated, edges = detect_lanes(image)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(OUTPUT_PATH), annotated):
        raise RuntimeError(f"Could not save output: {OUTPUT_PATH}")

    cv2.imwrite(str(OUTPUT_DIR / "road_edges.jpg"), edges)

    print("Lane detection baseline completed.")
    print(f"Annotated image: {OUTPUT_PATH}")
    print(f"Road edges: {OUTPUT_DIR / 'road_edges.jpg'}")


if __name__ == "__main__":
    main()
