"""
squat_counter.py
----------------
Squat counter — knee-angle finite state machine.

Inherits from ExerciseBase (exercise_base.py) for shared session
statistics, rep acceptance, and coaching infrastructure.
"""

from __future__ import annotations

import time
import enum
from typing import Optional, Tuple

import config
from angle_utils import (
    AngleSmoother,
    calculate_angle,
    compute_torso_angle,
    is_hip_below_knee,
    knee_cave_detected,
    map_range,
)
from exercise_base import ExerciseBase, FormIssue, RepResult, SessionStats


# ---------------------------------------------------------------------------
# FSM states
# ---------------------------------------------------------------------------

class SquatState(str, enum.Enum):
    """States in the squat finite state machine."""
    IDLE         = "idle"
    COUNTDOWN    = "countdown"
    STANDING     = "standing"
    GOING_DOWN   = "going_down"
    BOTTOM       = "bottom"
    GOING_UP     = "going_up"


# ---------------------------------------------------------------------------
# Squat counter — concrete implementation
# ---------------------------------------------------------------------------

class SquatCounter(ExerciseBase):
    """Counts bodyweight squats using a knee-angle finite state machine.

    State machine:
        COUNTDOWN → STANDING → GOING_DOWN → BOTTOM → GOING_UP → STANDING

    Form is validated at the BOTTOM and STANDING transitions; bad reps
    are not counted.
    """

    TARGET_REPS       = config.TARGET_REPS
    TARGET_SETS       = config.TARGET_SETS
    CALORIES_PER_REP  = config.CALORIES_PER_SQUAT

    def __init__(self) -> None:
        """Initialise the counter, smoother, and all tracking fields."""
        super().__init__()

        self._angle_smoother = AngleSmoother(window=config.ANGLE_SMOOTHING_WINDOW)
        self._state          = SquatState.STANDING

        # Rep tracking
        self._rep_start_time:  Optional[float] = None
        self._rep_min_angle:   float           = 180.0
        self._rep_hip_ok:      bool            = False    # hip ≥ knee at bottom
        self._rep_stand_ok:    bool            = False    # full extension achieved
        self._rep_back_ok:     bool            = True     # back straight throughout
        self._last_form_issue: FormIssue       = FormIssue.NONE

        # Guard flag: prevents double-counting while at bottom
        self._counted_this_rep: bool = False

        # Coaching message for the current rep
        self.coaching_message: str = "GET READY"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_frame(
        self,
        joints,
        frame_shape: Tuple[int, int],
    ) -> Tuple[Optional[RepResult], FormIssue, float]:
        """Update the FSM with this frame's pose data.

        Args:
            joints:      JointData from PoseDetector (may be None).
            frame_shape: (height, width).

        Returns:
            Tuple of (RepResult or None, current FormIssue, smoothed knee angle).
        """
        if joints is None:
            self._last_form_issue = FormIssue.BODY_NOT_FOUND
            self.coaching_message = FormIssue.BODY_NOT_FOUND.value
            return None, FormIssue.BODY_NOT_FOUND, 0.0

        h, w = frame_shape
        hip, knee, ankle = self._get_active_leg_coords(joints)

        # ---- Calculate & smooth the knee angle -----------------------
        raw_angle     = calculate_angle(hip, knee, ankle)
        smooth_angle  = self._angle_smoother.update(raw_angle)

        # ---- Update depth progress bar (0 = standing, 1 = full squat)
        self.depth_progress = map_range(
            smooth_angle,
            config.STANDING_ANGLE_MIN,
            config.SQUAT_DEPTH_ANGLE,
            0.0,
            1.0,
        )
        self.depth_progress = max(0.0, min(1.0, self.depth_progress))

        # ---- Form checks --------------------------------------------
        shoulder = self._get_active_shoulder(joints)
        torso_angle  = compute_torso_angle(shoulder, hip)
        back_straight = torso_angle < config.TORSO_LEAN_THRESHOLD
        hip_ok        = is_hip_below_knee(hip, knee)
        cave_detected = knee_cave_detected(hip, knee, ankle, config.KNEE_COLLAPSE_THRESHOLD)

        # Accumulate form issues within a rep
        if not back_straight:
            self._rep_back_ok = False
        if hip_ok and smooth_angle <= config.SQUAT_DEPTH_ANGLE:
            self._rep_hip_ok = True

        # ---- FSM transitions ----------------------------------------
        rep_result = None
        form_issue = FormIssue.NONE

        if self._state == SquatState.STANDING:
            self.coaching_message = "SQUAT DOWN"
            if smooth_angle < config.STANDING_ANGLE_MIN:
                self._state          = SquatState.GOING_DOWN
                self._rep_start_time = time.time()
                self._rep_min_angle  = smooth_angle
                self._rep_hip_ok     = False
                self._rep_stand_ok   = False
                self._rep_back_ok    = back_straight
                self._counted_this_rep = False

        elif self._state == SquatState.GOING_DOWN:
            self._rep_min_angle = min(self._rep_min_angle, smooth_angle)
            if not back_straight:
                self._rep_back_ok = False
                form_issue = FormIssue.BAD_BACK
                self.coaching_message = FormIssue.BAD_BACK.value
            elif cave_detected:
                form_issue = FormIssue.KNEE_CAVE
                self.coaching_message = FormIssue.KNEE_CAVE.value
            else:
                self.coaching_message = "KEEP GOING..."

            if smooth_angle <= config.SQUAT_DEPTH_ANGLE:
                self._state = SquatState.BOTTOM

        elif self._state == SquatState.BOTTOM:
            self._rep_min_angle = min(self._rep_min_angle, smooth_angle)
            if not self._counted_this_rep:
                self._rep_hip_ok = self._rep_hip_ok or hip_ok

            if smooth_angle <= config.SQUAT_DEPTH_ANGLE:
                self.coaching_message = "GREAT DEPTH! NOW UP!"
            elif smooth_angle <= config.BAD_DEPTH_ANGLE:
                form_issue = FormIssue.GO_LOWER
                self.coaching_message = FormIssue.GO_LOWER.value

            if smooth_angle > config.BAD_DEPTH_ANGLE:
                self._state = SquatState.GOING_UP

        elif self._state == SquatState.GOING_UP:
            if smooth_angle < config.FULL_EXT_ANGLE:
                self.coaching_message = "ALMOST THERE!"
            else:
                self.coaching_message = "STAND TALL!"
                self._rep_stand_ok = True

            if smooth_angle >= config.STANDING_ANGLE_MIN and not self._counted_this_rep:
                # ---- Rep completed: validate and count ---------------
                rep_result = self._finalise_rep()
                self._counted_this_rep = True
                self._state = SquatState.STANDING

        # ---- Propagate dominant form issue to caller ----------------
        self._last_form_issue = form_issue
        return rep_result, form_issue, smooth_angle

    def reset(self) -> None:
        """Reset all counters, stats, and FSM state."""
        self.stats             = SessionStats()
        self.stats.configure(config.TARGET_REPS, config.TARGET_SETS)
        self._state            = SquatState.STANDING
        self._rep_start_time   = None
        self._rep_min_angle    = 180.0
        self._rep_hip_ok       = False
        self._rep_stand_ok     = False
        self._rep_back_ok      = True
        self._counted_this_rep = False
        self._last_form_issue  = FormIssue.NONE
        self.depth_progress    = 0.0
        self.coaching_message  = "GET READY"
        self._angle_smoother.reset()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _finalise_rep(self) -> RepResult:
        """Evaluate the completed rep and update session statistics.

        Returns:
            RepResult with counted status and form metadata.
        """
        duration   = time.time() - (self._rep_start_time or time.time())
        min_angle  = self._rep_min_angle

        # Determine whether to count this rep
        issue = FormIssue.NONE
        if not self._rep_hip_ok and min_angle > config.SQUAT_DEPTH_ANGLE:
            issue = FormIssue.GO_LOWER
        elif not self._rep_stand_ok:
            issue = FormIssue.STAND_FULLY
        elif not self._rep_back_ok:
            issue = FormIssue.BAD_BACK

        counted = (issue == FormIssue.NONE)

        if counted:
            self._accept_rep(
                duration, min_angle, config.CALORIES_PER_SQUAT,
                config.TARGET_REPS, config.TARGET_SETS,
            )
            self.coaching_message = "GOOD REP! ✓"
            if self.stats.reps_in_current_set == 0 and self.stats.sets_completed > 0:
                self.coaching_message = "SET COMPLETE!"
        else:
            self.coaching_message = f"BAD REP — {issue.value}"

        return RepResult(
            counted    = counted,
            form_issue = issue,
            min_angle  = min_angle,
            duration   = duration,
        )

    @staticmethod
    def _get_active_leg_coords(joints) -> Tuple[
        Tuple[float, float], Tuple[float, float], Tuple[float, float]
    ]:
        """Select hip/knee/ankle for the better-visible leg."""
        if joints.left_visibility >= joints.right_visibility:
            return joints.left_hip, joints.left_knee, joints.left_ankle
        return joints.right_hip, joints.right_knee, joints.right_ankle

    @staticmethod
    def _get_active_shoulder(joints) -> Tuple[float, float]:
        """Return the shoulder on the active leg side."""
        if joints.left_visibility >= joints.right_visibility:
            return joints.left_shoulder
        return joints.right_shoulder
