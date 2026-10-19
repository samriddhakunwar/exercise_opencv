"""
auto_detect.py
--------------
Rule-based automatic exercise detection.

Every frame is labelled from the last ``AUTO_WINDOW_SECONDS`` of pose
features, in priority order:

    body horizontal ─┬─ elbows moving ──────────────► pushup
                     └─ elbows still ───────────────► plank
    body upright ────┬─ wrists above head + feet wide ► jumpingjack
                     ├─ one knee up at a time, L and R ► highknee
                     └─ knees bending ─┬─ split stance ► lunge
                                       └─ otherwise ───► squat

A label is only *detected* once it has held for ``AUTO_STABLE_SECONDS``
with at least ``AUTO_MIN_AGREEMENT`` of frame labels agreeing — so the
app doesn't flicker between counters.  Curls aren't detected.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import config
from angle_utils import calculate_angle, euclidean_distance


@dataclass
class _Features:
    """Per-frame measurements used by the rules."""
    t:            float
    incline:      float            # Shoulder→ankle angle from horizontal (deg)
    elbow:        float            # Elbow angle of the better-visible arm
    min_knee:     float            # Smaller of the two knee angles
    jack_open:    bool             # Wrists above nose and feet wide
    left_up:      bool             # Only the left knee is raised
    right_up:     bool             # Only the right knee is raised
    split_stance: bool             # Lunge-like stance


def _mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def extract_features(joints, t: float) -> _Features:
    """Measure everything the rules need from one JointData."""
    sh, an = _mid(joints.left_shoulder, joints.right_shoulder), _mid(joints.left_ankle, joints.right_ankle)
    incline = math.degrees(math.atan2(abs(an[1] - sh[1]), abs(an[0] - sh[0]) + 1e-6))

    side = "left" if joints.left_arm_visibility >= joints.right_arm_visibility else "right"
    elbow = calculate_angle(getattr(joints, f"{side}_shoulder"),
                            getattr(joints, f"{side}_elbow"), getattr(joints, f"{side}_wrist"))

    knees = [calculate_angle(getattr(joints, f"{s}_hip"), getattr(joints, f"{s}_knee"),
                             getattr(joints, f"{s}_ankle")) for s in ("left", "right")]

    # Jumping jacks are judged front-on: in a side view the shoulders overlap
    # and "feet wider than shoulders" is meaningless
    shoulder_w = abs(joints.left_shoulder[0] - joints.right_shoulder[0])
    torso_len  = euclidean_distance(sh, _mid(joints.left_hip, joints.right_hip))
    front_view = shoulder_w > config.AUTO_FRONT_VIEW_MIN * torso_len
    feet_ratio = abs(joints.left_ankle[0] - joints.right_ankle[0]) / max(shoulder_w, 1e-6)
    wrists_up  = max(joints.left_wrist[1], joints.right_wrist[1]) < joints.nose[1]

    def knee_rise(s: str) -> float:
        hip, knee, ankle = (getattr(joints, f"{s}_{p}") for p in ("hip", "knee", "ankle"))
        return (hip[1] - knee[1]) / max(euclidean_distance(hip, ankle), 1e-6)

    l_up = knee_rise("left") > config.AUTO_KNEE_LIFT
    r_up = knee_rise("right") > config.AUTO_KNEE_LIFT

    leg = max(euclidean_distance(_mid(joints.left_hip, joints.right_hip), an), 1e-6)
    split = (abs(joints.left_ankle[0] - joints.right_ankle[0]) / leg > config.AUTO_LUNGE_STANCE
             and abs(joints.left_knee[1] - joints.right_knee[1]) / leg > config.AUTO_LUNGE_KNEE_DIFF)

    return _Features(t, incline, elbow, min(knees),
                     front_view and wrists_up and feet_ratio >= config.JJ_FEET_OPEN_RATIO,
                     l_up and not r_up, r_up and not l_up, split)


class ExerciseClassifier:
    """Labels the current exercise from a sliding window of pose features.

    Attributes:
        detected:   Stable exercise key (None until something holds).
        candidate:  Most common recent frame label.
        confidence: Share of recent frame labels agreeing with ``candidate``.
    """

    def __init__(self) -> None:
        self._features: Deque[_Features] = collections.deque()
        self._labels: Deque[Tuple[float, Optional[str]]] = collections.deque()
        self.detected:   Optional[str] = None
        self.candidate:  Optional[str] = None
        self.confidence: float = 0.0

    def update(self, joints, t: float) -> Optional[str]:
        """Add a frame (joints may be None) and return the stable detection."""
        if joints is not None:
            self._features.append(extract_features(joints, t))
        while self._features and t - self._features[0].t > config.AUTO_WINDOW_SECONDS:
            self._features.popleft()

        self._labels.append((t, self._label() if joints is not None else None))
        while t - self._labels[0][0] > config.AUTO_STABLE_SECONDS:
            self._labels.popleft()

        counts = collections.Counter(label for _, label in self._labels)
        self.candidate, n = counts.most_common(1)[0]
        self.confidence = n / len(self._labels)
        span = t - self._labels[0][0]
        if (self.candidate is not None
                and span >= config.AUTO_STABLE_SECONDS * 0.9
                and self.confidence >= config.AUTO_MIN_AGREEMENT):
            self.detected = self.candidate
        return self.detected

    def _label(self) -> Optional[str]:
        """Classify the current feature window."""
        f = self._features
        incline = f[-1].incline
        if incline <= config.AUTO_HORIZONTAL_MAX:
            elbows = [x.elbow for x in f]
            return "pushup" if max(elbows) - min(elbows) > config.AUTO_ELBOW_MOTION else "plank"
        if incline < config.AUTO_VERTICAL_MIN:
            return None                                    # in between — no opinion
        if any(x.jack_open for x in f):
            return "jumpingjack"
        if any(x.left_up for x in f) and any(x.right_up for x in f):
            return "highknee"
        bending = [x for x in f if x.min_knee < config.AUTO_KNEE_BEND]
        if bending:
            lunge_frames = sum(x.split_stance for x in bending)
            return "lunge" if lunge_frames * 2 > len(bending) else "squat"
        return None
