
import numpy as np
import cv2
from tools.evaluate_ufld_video import draw_lanes
from tools.evaluate_ufld_video import remove_spikes
from tools.evaluate_ufld_video import align_lane, ALIGNED_POINT_COUNT

def test_remove_spikes_filters_isolated_outlier():
    points = np.array(
        [
            [10, 10],
            [11, 20],
            [12, 30],
            [100, 40],  # Isolated outlier in x
            [14, 50],
            [15, 60],
            [16, 70],
        ],
        dtype=np.float32,
    )

    result = remove_spikes(points, variable_axis=0)

    assert result.shape == points.shape
    assert result.dtype == np.float32

    # The isolated x-coordinate should be reduced.
    assert result[3, 0] < points[3, 0]

    # The y-coordinates must remain unchanged.
    np.testing.assert_array_equal(result[:, 1], points[:, 1])


def test_remove_spikes_preserves_short_input():
    points = np.array(
        [
            [10, 10],
            [20, 20],
            [30, 30],
        ],
        dtype=np.float32,
    )

    result = remove_spikes(points, variable_axis=0)

    np.testing.assert_array_equal(result, points)

def test_align_lane_returns_fixed_number_of_points():
    points = np.array(
        [
            [100, 100],
            [120, 200],
            [140, 300],
            [160, 400],
            [180, 500],
        ],
        dtype=np.float32,
    )

    result = align_lane(points, lane_idx=1)

    assert result.shape == (ALIGNED_POINT_COUNT, 2)
    assert result.dtype == np.float32

    # Row-predicted lanes must have increasing y-coordinates.
    assert np.all(np.diff(result[:, 1]) >= 0)


def test_align_lane_rejects_insufficient_points():
    points = np.array(
        [
            [100, 100],
        ],
        dtype=np.float32,
    )

    result = align_lane(points, lane_idx=1)

    assert result.shape == (0, 2)

def test_align_lane_handles_duplicate_axis_values():
    points = np.array(
        [
            [100, 100],
            [120, 200],
            [130, 200],  # Duplicate y-coordinate
            [140, 300],
            [160, 400],
        ],
        dtype=np.float32,
    )

    result = align_lane(points, lane_idx=1)

    assert result.shape == (ALIGNED_POINT_COUNT, 2)
    assert np.all(np.isfinite(result))


def test_align_lane_respects_axis_range():
    points = np.array(
        [
            [100, 100],
            [120, 200],
            [140, 300],
            [160, 400],
            [180, 500],
        ],
        dtype=np.float32,
    )

    result = align_lane(
        points,
        lane_idx=1,
        axis_range=(200, 400),
    )

    assert result.shape == (ALIGNED_POINT_COUNT, 2)
    assert result[:, 1].min() >= 200
    assert result[:, 1].max() <= 400
    
def test_draw_lanes_modifies_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    lanes = [
        [(100, 100), (110, 200), (120, 300)],
        [(300, 100), (310, 200), (320, 300)],
    ]

    original = frame.copy()

    draw_lanes(frame, lanes)

    assert frame.shape == original.shape
    assert frame.dtype == np.uint8
    assert np.any(frame != original)