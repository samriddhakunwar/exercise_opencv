"""
angle_utils.py
--------------
Geometry utilities for pose analysis.
All functions are pure, stateless, and reusable across exercise modules.
"""

from __future__ import annotations

import math
import collections
from typing import Tuple, Sequence


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Core angle maths
# ---------------------------------------------------------------------------

def calculate_angle(a: Point2D, b: Point2D, c: Point2D) -> float:
    """Return the angle (degrees) at vertex *b* formed by points a-b-c.

    Uses the law of cosines / dot-product formulation.  The result is always
    in the range [0, 180].

    Args:
        a: First point  (x, y).
        b: Vertex point (x, y).
        c: Last point   (x, y).

    Returns:
        Angle in degrees between vectors BA and BC.
    """
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]

    dot   = ax * cx + ay * cy
    mag_a = math.sqrt(ax ** 2 + ay ** 2)
    mag_c = math.sqrt(cx ** 2 + cy ** 2)

    if mag_a < 1e-6 or mag_c < 1e-6:
        return 0.0

    cosine = dot / (mag_a * mag_c)
    cosine = max(-1.0, min(1.0, cosine))   # Clamp for numerical safety
    return math.degrees(math.acos(cosine))


def calculate_3d_angle(a: Point3D, b: Point3D, c: Point3D) -> float:
    """Return angle at *b* using full 3-D coordinates (MediaPipe world coords).

    Args:
        a, b, c: (x, y, z) tuples.

    Returns:
        Angle in degrees.
    """
    ax, ay, az = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    cx, cy, cz = c[0] - b[0], c[1] - b[1], c[2] - b[2]

    dot   = ax * cx + ay * cy + az * cz
    mag_a = math.sqrt(ax**2 + ay**2 + az**2)
    mag_c = math.sqrt(cx**2 + cy**2 + cz**2)

    if mag_a < 1e-6 or mag_c < 1e-6:
        return 0.0

    cosine = max(-1.0, min(1.0, dot / (mag_a * mag_c)))
    return math.degrees(math.acos(cosine))


def vector_angle_2d(p1: Point2D, p2: Point2D) -> float:
    """Return the angle (degrees) that the vector p1→p2 makes with the x-axis.

    Args:
        p1: Start point.
        p2: End point.

    Returns:
        Angle in degrees in [-180, 180].
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------

class AngleSmoother:
    """Running-window moving-average smoother for a scalar angle value.

    Attributes:
        window: Number of frames to average over.
    """

    def __init__(self, window: int = 5) -> None:
        """Initialise with the given moving-average window size."""
        self._buffer: collections.deque[float] = collections.deque(maxlen=window)
        self.window = window

    def update(self, value: float) -> float:
        """Push a new value and return the smoothed result.

        Args:
            value: Latest raw angle measurement.

        Returns:
            Smoothed angle.
        """
        self._buffer.append(value)
        return sum(self._buffer) / len(self._buffer)

    def reset(self) -> None:
        """Clear the internal buffer."""
        self._buffer.clear()


class LandmarkSmoother:
    """Exponential moving average smoother for 2-D landmark positions.

    Reduces per-frame jitter without introducing significant lag.

    Attributes:
        alpha: EMA weight for the new sample (0 = ignore new, 1 = raw).
    """

    def __init__(self, alpha: float = 0.4) -> None:
        """Initialise the EMA smoother.

        Args:
            alpha: Smoothing factor in (0, 1].  Higher = less smoothing.
        """
        self.alpha = alpha
        self._state: dict[int, Tuple[float, float]] = {}

    def update(self, idx: int, x: float, y: float) -> Tuple[float, float]:
        """Update and return the smoothed (x, y) for landmark *idx*.

        Args:
            idx: Landmark index (unique key per joint).
            x:   Raw x coordinate.
            y:   Raw y coordinate.

        Returns:
            Smoothed (x, y) tuple.
        """
        if idx not in self._state:
            self._state[idx] = (x, y)
        else:
            px, py = self._state[idx]
            sx = self.alpha * x + (1 - self.alpha) * px
            sy = self.alpha * y + (1 - self.alpha) * py
            self._state[idx] = (sx, sy)
        return self._state[idx]

    def reset(self) -> None:
        """Clear all smoothed state."""
        self._state.clear()


# ---------------------------------------------------------------------------
# Body-alignment helpers
# ---------------------------------------------------------------------------

def euclidean_distance(p1: Point2D, p2: Point2D) -> float:
    """Return Euclidean distance between two 2-D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def normalise_distance(p1: Point2D, p2: Point2D, reference: float) -> float:
    """Return distance between *p1* and *p2* normalised by *reference* length.

    Args:
        p1, p2:    Points to measure.
        reference: A reference body-segment length (e.g., torso length).

    Returns:
        Normalised distance.  Returns 0 if *reference* is near zero.
    """
    if reference < 1e-6:
        return 0.0
    return euclidean_distance(p1, p2) / reference


def compute_torso_angle(shoulder: Point2D, hip: Point2D) -> float:
    """Return the forward-lean angle of the torso relative to vertical.

    Args:
        shoulder: Shoulder landmark (x, y) in image coordinates.
        hip:      Hip landmark (x, y).

    Returns:
        Lateral lean angle in degrees (0 = perfectly upright).
    """
    dx = shoulder[0] - hip[0]
    dy = hip[1] - shoulder[1]      # y inverted in image space
    vertical_angle = math.degrees(math.atan2(abs(dx), max(dy, 1e-6)))
    return vertical_angle


def is_hip_below_knee(hip: Point2D, knee: Point2D, threshold: float = 0.0) -> bool:
    """Return True if the hip y-coordinate is at or below the knee.

    In OpenCV image coordinates y increases downward, so 'below' means
    a *higher* y value.

    Args:
        hip:       Hip landmark (x, y).
        knee:      Knee landmark (x, y).
        threshold: Extra tolerance in normalised units.

    Returns:
        True when the hip is at or below the knee level.
    """
    return hip[1] >= knee[1] - threshold


def knee_cave_detected(
    hip: Point2D,
    knee: Point2D,
    ankle: Point2D,
    threshold: float = 0.08,
) -> bool:
    """Detect whether the knee has collapsed inward (valgus).

    Compares the horizontal offset of the knee relative to the hip-ankle
    midline.  Works best from a front-facing view.

    Args:
        hip:       Hip landmark (x, y).
        knee:      Knee landmark (x, y).
        ankle:     Ankle landmark (x, y).
        threshold: Normalised threshold for collapse (0-1).

    Returns:
        True when significant knee cave is detected.
    """
    midline_x = (hip[0] + ankle[0]) / 2.0
    knee_offset = abs(knee[0] - midline_x)
    # Normalise by approximate leg width (hip-to-ankle horizontal span)
    span = abs(hip[0] - ankle[0]) + 1e-6
    return (knee_offset / span) > threshold


def map_range(
    value: float,
    in_min: float,
    in_max: float,
    out_min: float,
    out_max: float,
) -> float:
    """Linearly map *value* from [in_min, in_max] to [out_min, out_max].

    The result is clamped to the output range.

    Args:
        value:   Value to map.
        in_min:  Lower bound of input range.
        in_max:  Upper bound of input range.
        out_min: Lower bound of output range.
        out_max: Upper bound of output range.

    Returns:
        Mapped and clamped value.
    """
    if abs(in_max - in_min) < 1e-9:
        return out_min
    ratio = (value - in_min) / (in_max - in_min)
    ratio = max(0.0, min(1.0, ratio))
    return out_min + ratio * (out_max - out_min)
