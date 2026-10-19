"""
curl_counter.py
---------------
Bicep-curl counter — independent elbow-angle FSM per arm.

State machine (per arm)::

    DOWN     ─(angle < EXTENDED)──────────────► CURLING
    CURLING  ─(angle ≤ TOP)───────────────────► UP
             ─(back to EXTENDED, never at top)► DOWN      (rejected: curl higher)
    UP       ─(angle > TOP_EXIT)──────────────► LOWERING
    LOWERING ─(angle ≥ EXTENDED)──────────────► DOWN      ← rep evaluated here
             ─(curls again before extending)──► CURLING   (rejected: extend arm)

Form: the upper arm must stay still — its angle from vertical must stay
under CURL_ELBOW_DRIFT_ANGLE and the elbow must not rise more than
CURL_ELBOW_RISE_RATIO × upper-arm length from where the rep started.
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import config
from angle_utils import AngleSmoother, calculate_angle, euclidean_distance, map_range
from exercise_base import FormIssue, RepResult, SidedExerciseBase


class CurlState(str, enum.Enum):
    """States in the per-arm curl finite state machine."""
    DOWN     = "down"
    CURLING  = "curling"
    UP       = "up"
    LOWERING = "lowering"


@dataclass
class _Arm:
    """Per-arm tracking state."""
    state:        CurlState = CurlState.DOWN
    smoother:     AngleSmoother = field(
        default_factory=lambda: AngleSmoother(config.ANGLE_SMOOTHING_WINDOW))
    angle:        float = 180.0
    start_time:   float = 0.0
    min_angle:    float = 180.0
    max_angle:    float = 0.0          # Highest angle while lowering
    start_elbow_y: float = 0.0
    upper_len:    float = 0.0
    drifted:      bool  = False


class CurlCounter(SidedExerciseBase):
    """Counts bicep curls for each arm separately."""

    TARGET_REPS      = config.CURL_TARGET_REPS
    TARGET_SETS      = config.CURL_TARGET_SETS
    CALORIES_PER_REP = config.CALORIES_PER_CURL

    def __init__(self) -> None:
        """Initialise both arms."""
        super().__init__()
        self._arms = {"left": _Arm(), "right": _Arm()}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_frame(
        self,
        joints,
        frame_shape: Tuple[int, int],
    ) -> Tuple[Optional[RepResult], FormIssue, float]:
        """Update both arm FSMs.

        Returns:
            (RepResult or None, current FormIssue, most-bent elbow angle).
        """
        if joints is None:
            self.coaching_message = FormIssue.BODY_NOT_FOUND.value
            return None, FormIssue.BODY_NOT_FOUND, 0.0

        result: Optional[RepResult] = None
        issue = FormIssue.NONE
        angles = []
        for side in ("left", "right"):
            if getattr(joints, f"{side}_arm_visibility") < config.CURL_MIN_VISIBILITY:
                continue
            r, i = self._update_arm(
                side,
                getattr(joints, f"{side}_shoulder"),
                getattr(joints, f"{side}_elbow"),
                getattr(joints, f"{side}_wrist"),
            )
            angles.append(self._arms[side].angle)
            if r is not None and (result is None or r.counted):
                result = r
            if i != FormIssue.NONE:
                issue = i

        if not angles:
            self.coaching_message = FormIssue.BODY_NOT_FOUND.value
            return None, FormIssue.BODY_NOT_FOUND, 0.0

        angle = min(angles)
        self.depth_progress = map_range(angle, config.CURL_EXTENDED_ANGLE,
                                        config.CURL_TOP_ANGLE, 0.0, 1.0)
        if issue != FormIssue.NONE and result is None:
            self.coaching_message = issue.value
        return result, issue, angle

    def reset(self) -> None:
        """Reset all counters, stats, and FSM state."""
        self.__init__()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_arm(self, side, shoulder, elbow, wrist) -> Tuple[Optional[RepResult], FormIssue]:
        """Advance one arm's FSM. Returns (RepResult | None, live FormIssue)."""
        arm   = self._arms[side]
        angle = arm.smoother.update(calculate_angle(shoulder, elbow, wrist))
        arm.angle = angle
        issue = FormIssue.NONE

        if arm.state != CurlState.DOWN and self._elbow_drifted(arm, shoulder, elbow):
            if not arm.drifted:
                self._cues.append("Keep your elbow still")
            arm.drifted = True
        if arm.drifted and arm.state != CurlState.DOWN:
            issue = FormIssue.ELBOW_DRIFT

        if arm.state == CurlState.DOWN:
            if angle < config.CURL_EXTENDED_ANGLE:
                self._start_rep(arm, angle, shoulder, elbow)
                arm.state = CurlState.CURLING

        elif arm.state == CurlState.CURLING:
            arm.min_angle = min(arm.min_angle, angle)
            if angle <= config.CURL_TOP_ANGLE:
                arm.state = CurlState.UP
                self.coaching_message = "SQUEEZE! NOW LOWER"
            elif angle >= config.CURL_EXTENDED_ANGLE:
                arm.state = CurlState.DOWN
                if arm.min_angle < config.CURL_EXTENDED_ANGLE - config.CURL_REVERSAL_DELTA:
                    return self._reject(FormIssue.CURL_HIGHER, arm), FormIssue.CURL_HIGHER

        elif arm.state == CurlState.UP:
            arm.min_angle = min(arm.min_angle, angle)
            if angle > config.CURL_TOP_EXIT_ANGLE:
                arm.state     = CurlState.LOWERING
                arm.max_angle = angle

        elif arm.state == CurlState.LOWERING:
            arm.max_angle = max(arm.max_angle, angle)
            if angle >= config.CURL_EXTENDED_ANGLE:
                arm.state = CurlState.DOWN
                if arm.drifted:
                    return self._reject(FormIssue.ELBOW_DRIFT, arm), FormIssue.ELBOW_DRIFT
                duration = time.time() - arm.start_time
                self._accept_side_rep(side, duration, arm.min_angle)
                self.coaching_message = ("SET COMPLETE!" if self.stats.reps_in_current_set == 0
                                         else f"GOOD {side.upper()} REP! ✓")
                return RepResult(True, FormIssue.NONE, arm.min_angle, duration), FormIssue.NONE
            if angle < arm.max_angle - config.CURL_REVERSAL_DELTA:
                # Started curling again without straightening the arm
                result = self._reject(FormIssue.EXTEND_ARM, arm)
                self._start_rep(arm, angle, shoulder, elbow)
                arm.state = CurlState.CURLING
                return result, FormIssue.EXTEND_ARM

        return None, issue

    @staticmethod
    def _start_rep(arm: _Arm, angle: float, shoulder, elbow) -> None:
        """Record the baseline for a new rep."""
        arm.start_time    = time.time()
        arm.min_angle     = angle
        arm.start_elbow_y = elbow[1]
        arm.upper_len     = euclidean_distance(shoulder, elbow)
        arm.drifted       = False

    @staticmethod
    def _elbow_drifted(arm: _Arm, shoulder, elbow) -> bool:
        """True when the upper arm swings away from vertical or the elbow rises."""
        dx, dy = elbow[0] - shoulder[0], elbow[1] - shoulder[1]
        from_vertical = math.degrees(math.atan2(abs(dx), max(dy, 1e-6)))
        rise = (arm.start_elbow_y - elbow[1]) / max(arm.upper_len, 1e-6)
        return (from_vertical > config.CURL_ELBOW_DRIFT_ANGLE
                or rise > config.CURL_ELBOW_RISE_RATIO)

    def _reject(self, issue: FormIssue, arm: _Arm) -> RepResult:
        """Build a rejected RepResult and update the coaching message."""
        self.coaching_message = f"BAD REP — {issue.value}"
        return RepResult(False, issue, arm.min_angle, time.time() - arm.start_time)
