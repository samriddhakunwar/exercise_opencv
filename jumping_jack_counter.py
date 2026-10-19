"""
jumping_jack_counter.py
-----------------------
Jumping-jack counter (front view) with hysteresis on arms and feet.

State machine::

    CLOSED  ─(arms leave sides or feet leave together)─► OPENING
    OPENING ─(wrists above head AND feet apart)────────► OPEN
    OPENING ─(back to closed without opening fully)────► CLOSED  (rejected)
    OPEN    ─(arms down AND feet together)─────────────► CLOSED  ← rep counted

"Open" and "closed" use separate thresholds (JJ_FEET_OPEN_RATIO vs
JJ_FEET_CLOSED_RATIO, wrists above the nose vs below the shoulders) so
jitter around a single boundary can't double count.
"""

from __future__ import annotations

import enum
from typing import Optional, Tuple

import clock
import config
from angle_utils import AngleSmoother, calculate_angle, map_range
from exercise_base import ExerciseBase, FormIssue, RepResult


class JackState(str, enum.Enum):
    """States in the jumping-jack finite state machine."""
    CLOSED  = "closed"    # Arms down, feet together
    OPENING = "opening"   # Moving out, not yet fully open
    OPEN    = "open"      # Arms overhead, feet apart


class JumpingJackCounter(ExerciseBase):
    """Counts jumping jacks from arm height and foot spread."""

    TARGET_REPS      = config.JJ_TARGET_REPS
    TARGET_SETS      = config.JJ_TARGET_SETS
    CALORIES_PER_REP = config.CALORIES_PER_JUMPING_JACK
    BEST_LOWER_IS_BETTER = False   # depths hold peak arm angle — higher is better

    def __init__(self) -> None:
        """Initialise the jumping-jack counter."""
        super().__init__()
        self._state: JackState = JackState.CLOSED
        self._feet_smoother = AngleSmoother(window=config.ANGLE_SMOOTHING_WINDOW)
        self._arm_smoother  = AngleSmoother(window=config.ANGLE_SMOOTHING_WINDOW)
        self._rep_start_time: Optional[float] = None
        self._peak_arm_angle: float = 0.0
        self._seen_arms_up:   bool  = False
        self._seen_feet_open: bool  = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_frame(
        self,
        joints,
        frame_shape: Tuple[int, int],
    ) -> Tuple[Optional[RepResult], FormIssue, float]:
        """Update the FSM with this frame's pose data.

        Returns:
            (RepResult or None, current FormIssue, average arm angle).
        """
        if joints is None:
            self.coaching_message = FormIssue.BODY_NOT_FOUND.value
            return None, FormIssue.BODY_NOT_FOUND, 0.0

        shoulder_w = abs(joints.left_shoulder[0] - joints.right_shoulder[0])
        if shoulder_w < 1e-3:
            self.coaching_message = FormIssue.BODY_NOT_FOUND.value
            return None, FormIssue.BODY_NOT_FOUND, 0.0

        feet_ratio = self._feet_smoother.update(
            abs(joints.left_ankle[0] - joints.right_ankle[0]) / shoulder_w
        )
        arm_angle = self._arm_smoother.update((
            calculate_angle(joints.left_hip, joints.left_shoulder, joints.left_wrist)
            + calculate_angle(joints.right_hip, joints.right_shoulder, joints.right_wrist)
        ) / 2.0)
        self.depth_progress = map_range(arm_angle, 20.0, 160.0, 0.0, 1.0)

        head_y     = joints.nose[1]
        shoulder_y = (joints.left_shoulder[1] + joints.right_shoulder[1]) / 2.0
        wrists_y   = (joints.left_wrist[1], joints.right_wrist[1])
        arms_up    = max(wrists_y) < head_y                     # both above head
        arms_down  = min(wrists_y) > shoulder_y + config.JJ_ARMS_DOWN_MARGIN * shoulder_w
        feet_open  = feet_ratio >= config.JJ_FEET_OPEN_RATIO
        feet_shut  = feet_ratio <= config.JJ_FEET_CLOSED_RATIO
        closed     = arms_down and feet_shut

        result: Optional[RepResult] = None
        issue = FormIssue.NONE

        if self._state == JackState.CLOSED:
            self.coaching_message = "JUMP!"
            if not closed:
                self._state          = JackState.OPENING
                self._rep_start_time = clock.now()
                self._peak_arm_angle = arm_angle
                self._seen_arms_up   = False
                self._seen_feet_open = False

        if self._state == JackState.OPENING:
            self._track(arm_angle, arms_up, feet_open)
            if arms_up and feet_open:
                self._state = JackState.OPEN
                self.coaching_message = "AND BACK!"
            elif closed:
                # Half a jack (one of arms / feet made it) → reject with a cue;
                # neither → just fidgeting, return quietly.
                if self._seen_arms_up or self._seen_feet_open:
                    issue = (FormIssue.ARMS_OVERHEAD if not self._seen_arms_up
                             else FormIssue.FEET_WIDER)
                    result = RepResult(False, issue, self._peak_arm_angle, self._duration())
                    self.coaching_message = f"BAD REP — {issue.value}"
                self._state = JackState.CLOSED

        elif self._state == JackState.OPEN:
            self._track(arm_angle, arms_up, feet_open)
            if closed:
                duration = self._duration()
                self._accept_rep(duration, self._peak_arm_angle, self.CALORIES_PER_REP,
                                 self.TARGET_REPS, self.TARGET_SETS)
                result = RepResult(True, FormIssue.NONE, self._peak_arm_angle, duration)
                self.coaching_message = ("SET COMPLETE!" if self.stats.reps_in_current_set == 0
                                         else "GOOD REP! ✓")
                self._state = JackState.CLOSED

        return result, issue, arm_angle

    def reset(self) -> None:
        """Reset all counters, stats, and FSM state."""
        self.__init__()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _track(self, arm_angle: float, arms_up: bool, feet_open: bool) -> None:
        """Remember the best arm height / foot spread seen this rep."""
        self._peak_arm_angle = max(self._peak_arm_angle, arm_angle)
        self._seen_arms_up   = self._seen_arms_up or arms_up
        self._seen_feet_open = self._seen_feet_open or feet_open

    def _duration(self) -> float:
        """Seconds since this rep started."""
        return clock.now() - (self._rep_start_time or clock.now())
