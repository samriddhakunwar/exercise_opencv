"""Known-value tests for angle_utils."""

import pytest

from angle_utils import (
    AngleSmoother,
    body_line_deviation,
    LandmarkSmoother,
    calculate_3d_angle,
    calculate_angle,
    compute_torso_angle,
    knee_cave_detected,
    map_range,
)


@pytest.mark.parametrize("a, b, c, expected", [
    ((1, 0), (0, 0), (0, 1), 90.0),
    ((1, 0), (0, 0), (-1, 0), 180.0),
    ((1, 0), (0, 0), (1, 0), 0.0),
    ((1, 0), (0, 0), (1, 1), 45.0),
    ((0, 0), (0, 0), (1, 1), 0.0),          # degenerate vector → 0
])
def test_calculate_angle(a, b, c, expected):
    assert calculate_angle(a, b, c) == pytest.approx(expected)


@pytest.mark.parametrize("a, b, c, expected", [
    ((1, 0, 0), (0, 0, 0), (0, 0, 1), 90.0),
    ((1, 1, 0), (0, 0, 0), (1, 1, 0), 0.0),
    ((1, 0, 0), (0, 0, 0), (-1, 0, 0), 180.0),
    ((1, 0, 0), (0, 0, 0), (1, 0, 1), 45.0),
])
def test_calculate_3d_angle(a, b, c, expected):
    assert calculate_3d_angle(a, b, c) == pytest.approx(expected, abs=1e-4)


def test_angle_smoother_moving_average_and_window():
    s = AngleSmoother(window=3)
    assert s.update(90) == 90
    assert s.update(120) == pytest.approx(105)
    assert s.update(150) == pytest.approx(120)
    assert s.update(180) == pytest.approx(150)    # 90 dropped out of the window
    s.reset()
    assert s.update(10) == 10


def test_landmark_smoother_ema_per_index():
    s = LandmarkSmoother(alpha=0.5)
    assert s.update(0, 100, 100) == (100, 100)    # first sample passes through
    assert s.update(0, 200, 0) == (150, 50)
    assert s.update(1, 7, 7) == (7, 7)            # independent per landmark
    s.reset()
    assert s.update(0, 1, 1) == (1, 1)


@pytest.mark.parametrize("shoulder, hip, expected", [
    ((0, 0), (0, 100), 0.0),       # upright
    ((100, 0), (0, 100), 45.0),    # leaning forward
    ((-100, 0), (0, 100), 45.0),   # leaning the other way — unsigned
])
def test_compute_torso_angle(shoulder, hip, expected):
    assert compute_torso_angle(shoulder, hip) == pytest.approx(expected)


def test_knee_cave_detected():
    hip, ankle = (100, 0), (100, 200)
    assert not knee_cave_detected(hip, (100, 100), ankle)
    # Hip/ankle 100 px apart; knee 20 px off the midline → 0.2 > 0.08
    assert knee_cave_detected((0, 0), (70, 100), (100, 200), threshold=0.08)
    assert not knee_cave_detected((0, 0), (52, 100), (100, 200), threshold=0.08)


@pytest.mark.parametrize("value, expected", [
    (155, 0.0), (100, 1.0), (127.5, 0.5),
    (200, 0.0), (50, 1.0),                 # clamped
])
def test_map_range_inverted_and_clamped(value, expected):
    assert map_range(value, 155, 100, 0.0, 1.0) == pytest.approx(expected)


def test_map_range_degenerate_input_range():
    assert map_range(5, 3, 3, 10, 20) == 10


@pytest.mark.parametrize("shoulder, ankle", [
    ((400, 400), (800, 400)),     # head on the left
    ((800, 400), (400, 400)),     # head on the right
])
def test_body_line_deviation_sign_is_facing_independent(shoulder, ankle):
    mid_x = (shoulder[0] + ankle[0]) / 2
    assert body_line_deviation(shoulder, (mid_x, 400), ankle) == pytest.approx(0)
    assert body_line_deviation(shoulder, (mid_x, 440), ankle) == pytest.approx(0.1)    # sag
    assert body_line_deviation(shoulder, (mid_x, 360), ankle) == pytest.approx(-0.1)   # pike
