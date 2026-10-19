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
    for name in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "foot"):
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


def jumping_jack_pose(arm_abduction: float, feet_ratio: float) -> JointData:
    """Front-view jumping jack.

    Args:
        arm_abduction: 0 = arms at sides, 90 = horizontal, 180 = overhead.
        feet_ratio:    Ankle spread / shoulder width.
    """
    ls, rs = (560.0, 200.0), (720.0, 200.0)               # shoulder width 160
    spread = feet_ratio * 160.0 / 2
    return make_joints(
        nose=(640.0, 140.0),
        left_shoulder=ls, right_shoulder=rs,
        left_elbow=point_at(ls, 90 + arm_abduction, 75),
        right_elbow=point_at(rs, 90 - arm_abduction, 75),
        left_wrist=point_at(ls, 90 + arm_abduction, 150),
        right_wrist=point_at(rs, 90 - arm_abduction, 150),
        left_hip=(590.0, 400.0), right_hip=(690.0, 400.0),
        left_knee=(640.0 - spread, 525.0), right_knee=(640.0 + spread, 525.0),
        left_ankle=(640.0 - spread, 650.0), right_ankle=(640.0 + spread, 650.0),
    )


def curl_pose(left_angle: float, right_angle: float, upper_arm_tilt: float = 0.0,
              elbow_rise: float = 0.0, right_arm_visibility: float = 0.9) -> JointData:
    """Front-view curl with both arms.

    Args:
        upper_arm_tilt: Upper-arm angle from vertical (elbow swinging out).
        elbow_rise:     Elbow raised by this many px (elbow coming forward/up).
    """
    def arm(shoulder: Point, angle: float, outward: int):
        elbow = point_at(shoulder, 90 + outward * upper_arm_tilt, 120)
        elbow = (elbow[0], elbow[1] - elbow_rise)
        return elbow, bend(shoulder, elbow, angle, 110, turn=-outward)

    ls, rs = (560.0, 200.0), (720.0, 200.0)
    le, lw = arm(ls, left_angle, 1)
    re, rw = arm(rs, right_angle, -1)
    return make_joints(nose=(640.0, 140.0), left_shoulder=ls, right_shoulder=rs,
                       left_elbow=le, right_elbow=re, left_wrist=lw, right_wrist=rw,
                       left_hip=(590.0, 450.0), right_hip=(690.0, 450.0),
                       right_arm_visibility=right_arm_visibility)


def lunge_pose(front_angle: float, front: str = "left",
               knee_forward: float = 0.0) -> JointData:
    """Side-view lunge facing image-right; *front* leg bent to *front_angle*.

    Args:
        knee_forward: Push the front knee this many px forward (past the toes).
    """
    f_ankle = (700.0, 650.0)
    f_knee  = (700.0 + knee_forward, 500.0)
    hip     = bend(f_ankle, f_knee, front_angle, 150)
    b_ankle = (hip[0] - 150.0, 650.0)
    b_knee  = (hip[0] - 60.0, 560.0)
    shoulder = (hip[0], hip[1] - 200.0)
    back = "right" if front == "left" else "left"
    return make_joints(
        nose=(hip[0] + 40.0, hip[1] - 240.0),
        left_shoulder=shoulder, right_shoulder=shoulder,
        left_hip=hip, right_hip=hip,
        **{f"{front}_knee": f_knee, f"{front}_ankle": f_ankle,
           f"{front}_foot": (740.0, 650.0),
           f"{back}_knee": b_knee, f"{back}_ankle": b_ankle,
           f"{back}_foot": (b_ankle[0] + 40.0, 650.0)},
    )


def mirror(joints: JointData, width: float = FRAME[1]) -> JointData:
    """Flip a pose horizontally (user facing the other way)."""
    flipped = {}
    for name, value in vars(joints).items():
        if isinstance(value, tuple):
            value = (width - value[0], value[1])
        flipped[name] = value
    return JointData(**flipped)


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


class FakeClock:
    """Stand-in for ``time.time`` that advances only when told to."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def feed_timed(counter, poses: Iterable[JointData], clock: FakeClock, dt: float = 0.1):
    """Like ``feed`` but advances *clock* by *dt* before each frame."""
    results = []
    for joints in poses:
        clock.now += dt
        result, _, _ = counter.process_frame(joints, FRAME)
        if result is not None:
            results.append(result)
    return results
