"""
lunge_counter.py
----------------
Lunge counter (side view) — front-knee-angle FSM, counted per leg.

State machine (front knee)::

    STANDING   ─(angle < STANDING)──────► GOING_DOWN
    GOING_DOWN ─(angle ≤ DEPTH)─────────► BOTTOM      (front leg locked here)
               ─(back to STANDING)──────► STANDING    (rejected: go lower)
    BOTTOM     ─(angle > BOTTOM_EXIT)───► GOING_UP
    GOING_UP   ─(angle ≥ STANDING)──────► STANDING    ← rep evaluated here

Front leg: the side the user faces is taken from nose vs. shoulder midpoint;
the front leg is the one whose ankle is further in that direction.  In a
front view (no clear facing) the leg with the higher knee is used.

Form: the front knee must not travel past the toes by more than
LUNGE_KNEE_TOE_TOLERANCE × shin length.
"""

from __future__ import annotations

import enum
from typing import Optional, Tuple

import clock
import config
from angle_utils import AngleSmoother, calculate_angle, euclidean_distance, map_range
from exercise_base import FormIssue, RepResult, SidedExerciseBase


class LungeState(str, enum.Enum):
    """States in the lunge finite state machine."""
    STANDING   = "standing"
    GOING_DOWN = "going_down"
    BOTTOM     = "bottom"
    GOING_UP   = "going_up"


class LungeCounter(SidedExerciseBase):
    """Counts lunges per front leg."""

    TARGET_REPS      = config.LUNGE_TARGET_REPS
    TARGET_SETS      = config.LUNGE_TARGET_SETS
    CALORIES_PER_REP = config.CALORIES_PER_LUNGE

    def __init__(self) -> None:
        """Initialise the lunge counter."""
        super().__init__()
        self._state: LungeState = LungeState.STANDING
        self._smoothers = {s: AngleSmoother(config.ANGLE_SMOOTHING_WINDOW)
                           for s in ("left", "right")}
        self.front_leg:       str   = "left"
        self._rep_start_time: float = 0.0
        self._rep_min_angle:  float = 180.0
        self._rep_knee_ok:    bool  = True

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
            (RepResult or None, current FormIssue, smoothed front-knee angle).
        """
        if joints is None:
            self.coaching_message = FormIssue.BODY_NOT_FOUND.value
            return None, FormIssue.BODY_NOT_FOUND, 0.0

        angles = {
            s: self._smoothers[s].update(calculate_angle(
                getattr(joints, f"{s}_hip"), getattr(joints, f"{s}_knee"),
                getattr(joints, f"{s}_ankle")))
            for s in ("left", "right")
        }
        facing = self._facing(joints)
        if self._state in (LungeState.STANDING, LungeState.GOING_DOWN):
            self.front_leg = self._detect_front_leg(joints, facing)
        angle = angles[self.front_leg]
        self.depth_progress = map_range(angle, config.LUNGE_STANDING_ANGLE,
                                        config.LUNGE_DEPTH_ANGLE, 0.0, 1.0)

        issue = FormIssue.NONE
        if facing and self._state in (LungeState.GOING_DOWN, LungeState.BOTTOM):
            if self._knee_past_toes(joints, self.front_leg, facing):
                if self._rep_knee_ok:
                    self._cues.append("Knee behind toes")
                self._rep_knee_ok = False
                issue = FormIssue.KNEE_OVER_TOES
                self.coaching_message = issue.value

        result: Optional[RepResult] = None

        if self._state == LungeState.STANDING:
            self.coaching_message = "STEP AND LUNGE"
            if angle < config.LUNGE_STANDING_ANGLE:
                self._state          = LungeState.GOING_DOWN
                self._rep_start_time = clock.now()
                self._rep_min_angle  = angle
                self._rep_knee_ok    = True

        elif self._state == LungeState.GOING_DOWN:
            self._rep_min_angle = min(self._rep_min_angle, angle)
            if angle <= config.LUNGE_DEPTH_ANGLE:
                self._state = LungeState.BOTTOM
            elif angle >= config.LUNGE_STANDING_ANGLE:
                self._state = LungeState.STANDING
                if self._rep_min_angle < config.LUNGE_STANDING_ANGLE - 10:
                    result = self._reject(FormIssue.GO_LOWER)
                    issue  = FormIssue.GO_LOWER

        elif self._state == LungeState.BOTTOM:
            self._rep_min_angle = min(self._rep_min_angle, angle)
            if issue == FormIssue.NONE:
                self.coaching_message = "GREAT DEPTH! PUSH UP!"
            if angle > config.LUNGE_BOTTOM_EXIT_ANGLE:
                self._state = LungeState.GOING_UP

        elif self._state == LungeState.GOING_UP:
            self.coaching_message = "STAND TALL!"
            if angle >= config.LUNGE_STANDING_ANGLE:
                self._state = LungeState.STANDING
                if not self._rep_knee_ok:
                    result = self._reject(FormIssue.KNEE_OVER_TOES)
                else:
                    duration = clock.now() - self._rep_start_time
                    self._accept_side_rep(self.front_leg, duration, self._rep_min_angle)
                    result = RepResult(True, FormIssue.NONE, self._rep_min_angle, duration)
                    self.coaching_message = (
                        "SET COMPLETE!" if self.stats.reps_in_current_set == 0
                        else f"GOOD {self.front_leg.upper()} REP! ✓")

        return result, issue, angle

    def reset(self) -> None:
        """Reset all counters, stats, and FSM state."""
        self.__init__()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _facing(joints) -> int:
        """+1 facing image-right, -1 facing image-left, 0 = front view / unsure."""
        sh_mid_x  = (joints.left_shoulder[0] + joints.right_shoulder[0]) / 2.0
        hip_mid   = ((joints.left_hip[0] + joints.right_hip[0]) / 2.0,
                     (joints.left_hip[1] + joints.right_hip[1]) / 2.0)
        sh_mid    = (sh_mid_x, (joints.left_shoulder[1] + joints.right_shoulder[1]) / 2.0)
        torso_len = euclidean_distance(sh_mid, hip_mid)
        offset    = joints.nose[0] - sh_mid_x
        if torso_len < 1e-3 or abs(offset) / torso_len < config.LUNGE_FACING_MIN:
            return 0
        return 1 if offset > 0 else -1

    @staticmethod
    def _detect_front_leg(joints, facing: int) -> str:
        """Return "left" or "right" for the leg that is in front."""
        if facing:
            return ("left" if facing * joints.left_ankle[0] >= facing * joints.right_ankle[0]
                    else "right")
        # Front view: the front knee is the higher one (smaller y)
        return "left" if joints.left_knee[1] <= joints.right_knee[1] else "right"

    @staticmethod
    def _knee_past_toes(joints, side: str, facing: int) -> bool:
        """True when the front knee is ahead of the toes beyond the tolerance."""
        knee  = getattr(joints, f"{side}_knee")
        ankle = getattr(joints, f"{side}_ankle")
        toe   = getattr(joints, f"{side}_foot")
        shin  = euclidean_distance(knee, ankle)
        return facing * (knee[0] - toe[0]) > config.LUNGE_KNEE_TOE_TOLERANCE * shin

    def _reject(self, issue: FormIssue) -> RepResult:
        """Build a rejected RepResult and update the coaching message."""
        self.coaching_message = f"BAD REP — {issue.value}"
        return RepResult(False, issue, self._rep_min_angle,
                         clock.now() - self._rep_start_time)
