"""
pushup_counter.py
-----------------
Push-up counter — elbow-angle finite state machine.

Architecture
~~~~~~~~~~~~
Inherits from ExerciseBase (in squat_counter.py) for shared session
statistics, rep acceptance, and coaching infrastructure.

State machine::

    TOP → LOWERING → BOTTOM → PUSHING_UP → TOP  ← rep evaluated here

Landmarks used (side with highest arm visibility):
    Shoulder → Elbow → Wrist  (elbow angle)
    Shoulder → Hip → Ankle    (body-alignment check)

Key fix vs earlier implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The original body-alignment check divided the perpendicular deviation by
``abs(ankle_y - shoulder_y)``.  In a side-view push-up the body is nearly
horizontal, so that denominator approaches zero and the check always fires
"STRAIGHTEN YOUR BODY".

This version normalises by the *body length* (shoulder-to-ankle distance),
which is stable regardless of body orientation.  The deviation is also
signed via the 2-D cross-product so that sag and pike are correctly
distinguished without depending on which axis is vertical.
"""

from __future__ import annotations

import math
import time
import enum
from typing import Optional, Tuple

import config
from angle_utils import AngleSmoother, calculate_angle, map_range
from squat_counter import ExerciseBase, FormIssue, RepResult, SessionStats


# ---------------------------------------------------------------------------
# FSM states
# ---------------------------------------------------------------------------

class PushUpState(str, enum.Enum):
    """States in the push-up finite state machine."""
    IDLE       = "idle"
    TOP        = "top"          # Arms locked out at the top
    LOWERING   = "lowering"     # Descending toward the ground
    BOTTOM     = "bottom"       # Elbows fully bent at depth
    PUSHING_UP = "pushing_up"   # Rising back toward the top


# ---------------------------------------------------------------------------
# Push-up counter
# ---------------------------------------------------------------------------

class PushUpCounter(ExerciseBase):
    """Counts push-ups using an elbow-angle finite state machine.

    State machine::

        TOP → LOWERING → BOTTOM → PUSHING_UP → TOP  ← rep evaluated here

    Form validation:
        * Full arm extension at the top  (elbow angle ≥ PUSHUP_FULL_EXT_ANGLE).
        * Sufficient depth at the bottom (elbow angle ≤ PUSHUP_BOTTOM_ANGLE).
        * Body stays roughly straight    (perpendicular hip deviation ≤ thresholds).
        * User viewed from the side      (elbow-shoulder horizontal offset check).
    """

    TARGET_REPS      = config.TARGET_PUSHUPS
    TARGET_SETS      = config.TARGET_PUSHUP_SETS
    CALORIES_PER_REP = config.CALORIES_PER_PUSHUP

    def __init__(self) -> None:
        """Initialise the push-up counter."""
        super().__init__()
        self._angle_smoother   = AngleSmoother(window=config.ANGLE_SMOOTHING_WINDOW)
        self._state            = PushUpState.TOP   # type: ignore[assignment]

        # Per-rep tracking
        self._rep_start_time:   Optional[float] = None
        self._rep_min_angle:    float           = 180.0
        self._rep_ext_ok:       bool            = False   # Full extension seen at top
        self._rep_depth_ok:     bool            = False   # Proper depth seen at bottom
        self._rep_body_ok:      bool            = True    # Body line OK throughout
        self._counted_this_rep: bool            = False
        self._last_form_issue:  FormIssue       = FormIssue.NONE

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_frame(
        self,
        joints,
        frame_shape: Tuple[int, int],
    ) -> Tuple[Optional[RepResult], FormIssue, float]:
        """Update the push-up FSM with this frame's pose data.

        Args:
            joints:      JointData from PoseDetector (may be None).
            frame_shape: (height, width).

        Returns:
            Tuple of (RepResult or None, current FormIssue, smoothed elbow angle).
        """
        if joints is None:
            self._last_form_issue = FormIssue.BODY_NOT_FOUND
            self.coaching_message = FormIssue.BODY_NOT_FOUND.value
            return None, FormIssue.BODY_NOT_FOUND, 0.0

        shoulder, elbow, wrist, hip, ankle = self._get_active_arm_coords(joints)

        # ---- Elbow angle (Shoulder → Elbow → Wrist) -----------------
        raw_angle    = calculate_angle(shoulder, elbow, wrist)
        smooth_angle = self._angle_smoother.update(raw_angle)

        # ---- Depth progress bar (0 = top/extended, 1 = full depth) --
        self.depth_progress = map_range(
            smooth_angle,
            config.PUSHUP_TOP_ANGLE,
            config.PUSHUP_BOTTOM_ANGLE,
            0.0,
            1.0,
        )
        self.depth_progress = max(0.0, min(1.0, self.depth_progress))

        # ---- Form checks --------------------------------------------
        body_straight, body_issue = self._check_body_alignment(shoulder, hip, ankle)
        side_view_ok = self._check_side_view(joints)

        if not body_straight:
            self._rep_body_ok = False

        # ---- FSM transitions ----------------------------------------
        rep_result = None
        form_issue = FormIssue.NONE

        # Side-view gate: halt tracking (but don't penalise reps) if
        # the user isn't in profile view.
        if not side_view_ok:
            form_issue = FormIssue.MOVE_SIDEWAYS
            self.coaching_message = FormIssue.MOVE_SIDEWAYS.value
            self._last_form_issue = form_issue
            return None, form_issue, smooth_angle

        if self._state == PushUpState.TOP:
            if smooth_angle >= config.PUSHUP_FULL_EXT_ANGLE:
                self._rep_ext_ok = True
            self.coaching_message = "LOWER DOWN"
            if smooth_angle < config.PUSHUP_TOP_ANGLE:
                # Start descent
                self._state            = PushUpState.LOWERING
                self._rep_start_time   = time.time()
                self._rep_min_angle    = smooth_angle
                self._rep_depth_ok     = False
                self._rep_ext_ok       = smooth_angle >= config.PUSHUP_FULL_EXT_ANGLE
                self._rep_body_ok      = body_straight
                self._counted_this_rep = False

        elif self._state == PushUpState.LOWERING:
            self._rep_min_angle = min(self._rep_min_angle, smooth_angle)
            if not body_straight:
                form_issue = body_issue
                self.coaching_message = body_issue.value
            else:
                self.coaching_message = "KEEP GOING..."

            if smooth_angle <= config.PUSHUP_BOTTOM_ANGLE:
                self._state        = PushUpState.BOTTOM
                self._rep_depth_ok = True

        elif self._state == PushUpState.BOTTOM:
            self._rep_min_angle = min(self._rep_min_angle, smooth_angle)
            if smooth_angle <= config.PUSHUP_BOTTOM_ANGLE:
                self._rep_depth_ok = True
                self.coaching_message = "GREAT DEPTH! PUSH UP!"
            elif smooth_angle <= config.PUSHUP_LOWER_LIMIT:
                form_issue = FormIssue.GO_LOWER_PU
                self.coaching_message = FormIssue.GO_LOWER_PU.value

            if smooth_angle > config.PUSHUP_LOWER_LIMIT:
                self._state = PushUpState.PUSHING_UP

        elif self._state == PushUpState.PUSHING_UP:
            if smooth_angle < config.PUSHUP_FULL_EXT_ANGLE:
                self.coaching_message = "ALMOST THERE!"
            else:
                self.coaching_message = "LOCK OUT!"
                self._rep_ext_ok = True

            if smooth_angle >= config.PUSHUP_TOP_ANGLE and not self._counted_this_rep:
                rep_result             = self._finalise_rep(smooth_angle)
                self._counted_this_rep = True
                self._state            = PushUpState.TOP

        self._last_form_issue = form_issue
        return rep_result, form_issue, smooth_angle

    def reset(self) -> None:
        """Reset all counters, stats, and FSM state."""
        self.stats             = SessionStats()
        self.stats.configure(config.TARGET_PUSHUPS, config.TARGET_PUSHUP_SETS)
        self._state            = PushUpState.TOP   # type: ignore[assignment]
        self._rep_start_time   = None
        self._rep_min_angle    = 180.0
        self._rep_ext_ok       = False
        self._rep_depth_ok     = False
        self._rep_body_ok      = True
        self._counted_this_rep = False
        self._last_form_issue  = FormIssue.NONE
        self.depth_progress    = 0.0
        self.coaching_message  = "GET READY"
        self._angle_smoother.reset()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _finalise_rep(self, current_angle: float) -> RepResult:
        """Evaluate the completed push-up rep and update session stats.

        Args:
            current_angle: Elbow angle when returning to the top.

        Returns:
            RepResult with counted status and form metadata.
        """
        duration  = time.time() - (self._rep_start_time or time.time())
        min_angle = self._rep_min_angle

        issue = FormIssue.NONE
        if not self._rep_depth_ok:
            issue = FormIssue.GO_LOWER_PU
        elif not self._rep_ext_ok:
            issue = FormIssue.LOCK_ARMS
        elif not self._rep_body_ok:
            issue = FormIssue.STRAIGHTEN

        counted = (issue == FormIssue.NONE)

        if counted:
            self._accept_rep(
                duration, min_angle, config.CALORIES_PER_PUSHUP,
                config.TARGET_PUSHUPS, config.TARGET_PUSHUP_SETS,
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
    def _get_active_arm_coords(joints) -> Tuple[
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
    ]:
        """Return (shoulder, elbow, wrist, hip, ankle) for the more-visible arm."""
        if joints.left_arm_visibility >= joints.right_arm_visibility:
            return (
                joints.left_shoulder,
                joints.left_elbow,
                joints.left_wrist,
                joints.left_hip,
                joints.left_ankle,
            )
        return (
            joints.right_shoulder,
            joints.right_elbow,
            joints.right_wrist,
            joints.right_hip,
            joints.right_ankle,
        )

    @staticmethod
    def _check_body_alignment(
        shoulder: Tuple[float, float],
        hip: Tuple[float, float],
        ankle: Tuple[float, float],
    ) -> Tuple[bool, FormIssue]:
        """Check whether the body forms a straight line from shoulder to ankle.

        Uses true perpendicular (point-to-line) distance normalised by the
        full body length.  This is robust for any body orientation including
        horizontal push-up position where the old y-span denominator breaks.

        2-D cross product gives the signed deviation so sag and pike can be
        distinguished without relying on which axis is "vertical" in frame.

        Args:
            shoulder: Shoulder (x, y) in image coordinates.
            hip:      Hip (x, y).
            ankle:    Ankle (x, y).

        Returns:
            (is_ok, FormIssue) — FormIssue.NONE when the body is straight.
        """
        # Direction vector of the body line
        dx = ankle[0] - shoulder[0]
        dy = ankle[1] - shoulder[1]
        body_len = math.sqrt(dx * dx + dy * dy)

        if body_len < 1e-3:
            return True, FormIssue.NONE           # Can't compute — give benefit of doubt

        # Perpendicular (signed) distance from hip to the shoulder→ankle line,
        # normalised by body length so scale doesn't matter.
        # cross = (ankle - shoulder) × (hip - shoulder)
        cross = dx * (hip[1] - shoulder[1]) - dy * (hip[0] - shoulder[0])
        deviation = cross / (body_len * body_len)

        # Positive cross → hip is "above" the line in right-hand convention
        # (which direction that is in image space depends on camera orientation;
        # we rely on the magnitude and use a generous threshold).
        if deviation > config.PUSHUP_HIP_SAG_THRESHOLD:
            return False, FormIssue.HIPS_UP
        if deviation < -config.PUSHUP_PIKE_THRESHOLD:
            return False, FormIssue.HIPS_DOWN
        return True, FormIssue.NONE

    @staticmethod
    def _check_side_view(joints) -> bool:
        """Heuristic check that the user is in a side-on (profile) view.

        From a profile view, the visible elbow is displaced horizontally
        relative to the shoulder.  Head-on the x-offset is near zero.

        Returns:
            True when a side view is detected (or when confidence is too
            low to judge — give the user the benefit of the doubt).
        """
        if joints.left_arm_visibility >= joints.right_arm_visibility:
            sh, el = joints.left_shoulder, joints.left_elbow
        else:
            sh, el = joints.right_shoulder, joints.right_elbow

        dx = abs(el[0] - sh[0])
        dy = abs(el[1] - sh[1])
        if dy < 1e-3:
            return True   # Cannot determine — allow
        ratio = dx / dy
        # Relaxed threshold: 0.10 instead of original 0.15
        return ratio >= 0.10
