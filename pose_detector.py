"""
pose_detector.py
----------------
Wraps MediaPipe Pose to provide clean, smoothed landmark data and
helpers for extracting named joint positions in image-space coordinates.
"""

from __future__ import annotations

# Patches the protobuf/TensorFlow conflict before mediapipe loads.
import _mp_compat  # noqa: F401

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

import config
from angle_utils import LandmarkSmoother

# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------
mp_pose     = mp.solutions.pose
mp_drawing  = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# MediaPipe landmark indices we care about
_LM = mp_pose.PoseLandmark


@dataclass
class JointData:
    """Container for key joint positions (normalised 0-1 coords).

    Attributes:
        left_shoulder:   (x, y) left shoulder.
        right_shoulder:  (x, y) right shoulder.
        left_elbow:      (x, y) left elbow.
        right_elbow:     (x, y) right elbow.
        left_wrist:      (x, y) left wrist.
        right_wrist:     (x, y) right wrist.
        left_hip:        (x, y) left hip.
        right_hip:       (x, y) right hip.
        left_knee:       (x, y) left knee.
        right_knee:      (x, y) right knee.
        left_ankle:      (x, y) left ankle.
        right_ankle:     (x, y) right ankle.
        left_visibility:      Visibility score for the left leg chain.
        right_visibility:     Visibility score for the right leg chain.
        left_arm_visibility:  Visibility score for the left arm chain.
        right_arm_visibility: Visibility score for the right arm chain.
    """
    left_shoulder:        Tuple[float, float] = (0.0, 0.0)
    right_shoulder:       Tuple[float, float] = (0.0, 0.0)
    left_elbow:           Tuple[float, float] = (0.0, 0.0)
    right_elbow:          Tuple[float, float] = (0.0, 0.0)
    left_wrist:           Tuple[float, float] = (0.0, 0.0)
    right_wrist:          Tuple[float, float] = (0.0, 0.0)
    left_hip:             Tuple[float, float] = (0.0, 0.0)
    right_hip:            Tuple[float, float] = (0.0, 0.0)
    left_knee:            Tuple[float, float] = (0.0, 0.0)
    right_knee:           Tuple[float, float] = (0.0, 0.0)
    left_ankle:           Tuple[float, float] = (0.0, 0.0)
    right_ankle:          Tuple[float, float] = (0.0, 0.0)
    left_visibility:      float = 0.0
    right_visibility:     float = 0.0
    left_arm_visibility:  float = 0.0
    right_arm_visibility: float = 0.0
    raw_landmarks:        Optional[object] = field(default=None, repr=False)


class PoseDetector:
    """MediaPipe Pose wrapper with smoothing and landmark extraction.

    Automatically selects the better-visible leg side (left or right) and
    applies EMA smoothing to reduce landmark jitter.

    Usage::

        detector = PoseDetector()
        joints, annotated_frame = detector.process(frame)
        if joints:
            hip, knee, ankle = detector.get_active_leg(joints)
    """

    def __init__(self) -> None:
        """Initialise MediaPipe Pose and the landmark smoother."""
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=config.MODEL_COMPLEXITY,
            smooth_landmarks=config.SMOOTH_LANDMARKS,
            enable_segmentation=False,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._smoother = LandmarkSmoother(alpha=config.LANDMARK_SMOOTH_ALPHA)

        # Custom drawing specs for a cleaner skeleton overlay
        self._landmark_spec = mp_drawing.DrawingSpec(
            color=(80, 200, 120), thickness=2, circle_radius=4
        )
        self._connection_spec = mp_drawing.DrawingSpec(
            color=(200, 200, 255), thickness=2
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        frame: np.ndarray,
    ) -> Tuple[Optional[JointData], np.ndarray]:
        """Run MediaPipe Pose on *frame* and return joint data + annotated frame.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            A (JointData | None, annotated_frame) tuple.
            JointData is None when no person is detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        rgb.flags.writeable = True

        annotated = frame.copy()

        if results.pose_landmarks is None:
            return None, annotated

        # Optionally draw the MediaPipe skeleton
        if config.DRAW_LANDMARKS and config.SHOW_SKELETON:
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self._landmark_spec,
                connection_drawing_spec=self._connection_spec,
            )

        joints = self._extract_joints(results.pose_landmarks.landmark, frame.shape)
        return joints, annotated

    def get_active_leg(
        self,
        joints: JointData,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Return (hip, knee, ankle) for the better-visible leg side.

        Args:
            joints: Extracted joint data.

        Returns:
            Tuple of three (x, y) pixel-coordinate points.
        """
        if joints.left_visibility >= joints.right_visibility:
            return joints.left_hip, joints.left_knee, joints.left_ankle
        return joints.right_hip, joints.right_knee, joints.right_ankle

    def get_active_shoulder(
        self,
        joints: JointData,
    ) -> Tuple[float, float]:
        """Return the shoulder of the active (more visible) leg side.

        Args:
            joints: Extracted joint data.

        Returns:
            (x, y) pixel coordinate of the shoulder.
        """
        if joints.left_visibility >= joints.right_visibility:
            return joints.left_shoulder
        return joints.right_shoulder

    def release(self) -> None:
        """Release MediaPipe resources."""
        self._pose.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_joints(
        self,
        landmarks,
        shape: Tuple[int, int, int],
    ) -> JointData:
        """Extract and smooth key joint positions from raw landmarks.

        Args:
            landmarks: MediaPipe landmark list.
            shape:     Frame shape (H, W, C).

        Returns:
            Populated JointData instance.
        """
        h, w = shape[:2]

        def lm(idx: int) -> Tuple[float, float]:
            lmk = landmarks[idx]
            sx, sy = self._smoother.update(idx, lmk.x * w, lmk.y * h)
            return sx, sy

        def vis(idx: int) -> float:
            return landmarks[idx].visibility

        left_vis  = min(vis(_LM.LEFT_HIP),  vis(_LM.LEFT_KNEE),  vis(_LM.LEFT_ANKLE))
        right_vis = min(vis(_LM.RIGHT_HIP), vis(_LM.RIGHT_KNEE), vis(_LM.RIGHT_ANKLE))

        # Arm visibility — used by PushUpCounter
        left_arm_vis  = min(
            vis(_LM.LEFT_SHOULDER), vis(_LM.LEFT_ELBOW), vis(_LM.LEFT_WRIST)
        )
        right_arm_vis = min(
            vis(_LM.RIGHT_SHOULDER), vis(_LM.RIGHT_ELBOW), vis(_LM.RIGHT_WRIST)
        )

        return JointData(
            left_shoulder        = lm(_LM.LEFT_SHOULDER),
            right_shoulder       = lm(_LM.RIGHT_SHOULDER),
            left_elbow           = lm(_LM.LEFT_ELBOW),
            right_elbow          = lm(_LM.RIGHT_ELBOW),
            left_wrist           = lm(_LM.LEFT_WRIST),
            right_wrist          = lm(_LM.RIGHT_WRIST),
            left_hip             = lm(_LM.LEFT_HIP),
            right_hip            = lm(_LM.RIGHT_HIP),
            left_knee            = lm(_LM.LEFT_KNEE),
            right_knee           = lm(_LM.RIGHT_KNEE),
            left_ankle           = lm(_LM.LEFT_ANKLE),
            right_ankle          = lm(_LM.RIGHT_ANKLE),
            left_visibility      = left_vis,
            right_visibility     = right_vis,
            left_arm_visibility  = left_arm_vis,
            right_arm_visibility = right_arm_vis,
            raw_landmarks        = landmarks,
        )
