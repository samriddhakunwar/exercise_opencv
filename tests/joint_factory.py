"""
joint_factory.py
----------------
Builds synthetic JointData from joint angles so counter tests read like
"go from 170° to 90° and back" instead of lists of pixel coordinates.

All coordinates are image pixels in a 720×1280 frame (y grows downward).
"""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple

from exercise_base import JointData

FRAME: Tuple[int, int] = (720, 1280)   # (height, width) passed to process_frame

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def point_at(origin: Point, direction_deg: float, length: float) -> Point:
    """Point *length* px from *origin* in *direction_deg* (0 = +x, 90 = down)."""
    rad = math.radians(direction_deg)
    return origin[0] + length * math.cos(rad), origin[1] + length * math.sin(rad)


def bend(a: Point, b: Point, angle: float, length: float, turn: int = 1) -> Point:
    """Return c such that the interior angle a-b-c equals *angle* degrees."""
    base = math.degrees(math.atan2(a[1] - b[1], a[0] - b[0]))
    return point_at(b, base + turn * angle, length)


def make_joints(**points) -> JointData:
    """JointData with full visibility; ``left_x`` defaults to ``right_x``."""
    for name in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
        left, right = f"left_{name}", f"right_{name}"
        if left in points and right not in points:
            points[right] = points[left]
        if right in points and left not in points:
            points[left] = points[right]
    vis = dict(left_visibility=0.9, right_visibility=0.9,
               left_arm_visibility=0.9, right_arm_visibility=0.9)
    vis.update({k: points.pop(k) for k in list(points) if k.endswith("visibility")})
    return JointData(**points, **vis)


# ---------------------------------------------------------------------------
# Exercise poses
# ---------------------------------------------------------------------------

def squat_pose(knee_angle: float, torso_lean: float = 0.0) -> JointData:
    """Side-view squat: vertical shin, thigh bent back by *knee_angle*."""
    ankle = (600.0, 650.0)
    knee  = (600.0, 500.0)
    hip   = bend(ankle, knee, knee_angle, 150)
    shoulder = point_at(hip, -90 + torso_lean, 200)
    return make_joints(left_ankle=ankle, left_knee=knee, left_hip=hip,
                       left_shoulder=shoulder)


def pushup_pose(elbow_angle: float, hip_offset: float = 0.0,
                upper_arm_dir: float = 110.0) -> JointData:
    """Side-view push-up: horizontal body, arm bent to *elbow_angle*.

    Args:
        hip_offset:    Hip drop below the shoulder-ankle line, as a fraction
                       of body length (positive = sag).
        upper_arm_dir: Upper-arm direction; 90 = straight down (front view).
    """
    shoulder = (400.0, 400.0)
    ankle    = (800.0, 400.0)
    hip      = (600.0, 400.0 + hip_offset * 400.0)
    elbow    = point_at(shoulder, upper_arm_dir, 120)
    wrist    = bend(shoulder, elbow, elbow_angle, 120)
    return make_joints(left_shoulder=shoulder, left_elbow=elbow, left_wrist=wrist,
                       left_hip=hip, left_knee=(700.0, 400.0), left_ankle=ankle)


def high_knee_pose(left_lift: float, right_lift: float) -> JointData:
    """Front-view marching pose.

    *lift* is the counter's own metric ``(hip_y - knee_y) / hip_y`` in
    normalised units; -0.5 ≈ standing, ≥ 0.72 = high knee.
    """
    hip_y = 360.0

    def leg(x: float, lift: float):
        knee = (x, hip_y * (1 - lift))
        return (x, hip_y), knee, (x, knee[1] + 150)

    lh, lk, la = leg(560.0, left_lift)
    rh, rk, ra = leg(720.0, right_lift)
    return make_joints(left_hip=lh, left_knee=lk, left_ankle=la,
                       right_hip=rh, right_knee=rk, right_ankle=ra,
                       left_shoulder=(560.0, 160.0), right_shoulder=(720.0, 160.0))


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------

def ramp(start: float, end: float, steps: int = 10) -> List[float]:
    """Linear sequence from *start* to *end* inclusive."""
    return [start + (end - start) * i / (steps - 1) for i in range(steps)]


def hold(value: float, frames: int = 10) -> List[float]:
    """*value* repeated for *frames* frames."""
    return [value] * frames


def feed(counter, poses: Iterable[JointData]):
    """Run *poses* through *counter*; return the list of non-None RepResults."""
    results = []
    for joints in poses:
        result, _, _ = counter.process_frame(joints, FRAME)
        if result is not None:
            results.append(result)
    return results
