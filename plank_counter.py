"""
plank_counter.py
----------------
Plank timer — times holds instead of counting reps.

State machine::

    WAITING ─(body horizontal)──────────► HOLDING   (timer runs)
    HOLDING ─(hips sag / pike)──────────► PAUSED    (timer paused, coach)
    PAUSED  ─(body straight again)──────► HOLDING
    HOLDING/PAUSED ─(out of position for PLANK_BREAK_GRACE s)─► WAITING

A set completes when one continuous hold reaches PLANK_TARGET_SECONDS.  The
timer keeps running past the target (so "longest plank" is meaningful) until
the user leaves the position.  A hold that ends before the target is logged
as a failed attempt and does not complete the set.
"""

from __future__ import annotations

import enum
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import config
from angle_utils import body_line_deviation, calculate_angle
from exercise_base import ExerciseBase, FormIssue, RepResult


class PlankState(str, enum.Enum):
    """States in the plank finite state machine."""
    WAITING = "waiting"   # Not in plank position
    HOLDING = "holding"   # Good form — timer running
    PAUSED  = "paused"    # In position but hips out of line — timer paused


class PlankCounter(ExerciseBase):
    """Times plank holds with sag / pike detection."""

    TARGET_REPS      = 1          # One hold per set
    TARGET_SETS      = config.PLANK_TARGET_SETS
    CALORIES_PER_REP = 0.0        # Calories are per second, see below

    def __init__(self) -> None:
        """Initialise the plank timer."""
        super().__init__()
        self._state: PlankState = PlankState.WAITING
        self.hold_seconds:  float = 0.0          # Current hold (good-form time)
        self.total_hold:    float = 0.0          # All good-form time this session
        self.attempts:      List[float] = []     # Holds that ended before the target
        self._target_hit:   bool  = False
        self._halfway_said: bool  = False
        self._body_angles:  List[float] = []     # Shoulder-hip-ankle angle samples
        self._out_time:     float = 0.0          # Time spent out of position
        self._last_time:    Optional[float] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def longest_hold(self) -> float:
        """Longest hold this session, completed or not (seconds)."""
        return max(self.stats.rep_durations + self.attempts + [self.hold_seconds])

    def process_frame(
        self,
        joints,
        frame_shape: Tuple[int, int],
    ) -> Tuple[Optional[RepResult], FormIssue, float]:
        """Advance the hold timer with this frame.

        Returns:
            (RepResult when a set completes, active FormIssue, body angle).
        """
        now = time.time()
        dt  = 0.0 if self._last_time is None else min(now - self._last_time,
                                                       config.PLANK_MAX_FRAME_DT)
        self._last_time = now

        if joints is None:
            self._out_of_position(dt)
            self.coaching_message = FormIssue.BODY_NOT_FOUND.value
            return None, FormIssue.BODY_NOT_FOUND, 0.0

        shoulder, hip, ankle = self._active_side(joints)
        body_angle = calculate_angle(shoulder, hip, ankle)
        incline = math.degrees(math.atan2(abs(ankle[1] - shoulder[1]),
                                          abs(ankle[0] - shoulder[0])))

        if incline > config.PLANK_MAX_INCLINE:
            self._out_of_position(dt)
            if self._state == PlankState.WAITING:
                self.coaching_message = FormIssue.GET_IN_PLANK.value
                return None, FormIssue.GET_IN_PLANK, body_angle
            return None, FormIssue.NONE, body_angle
        self._out_time = 0.0

        deviation = body_line_deviation(shoulder, hip, ankle)
        issue = FormIssue.NONE
        if deviation > config.PLANK_SAG_THRESHOLD:
            issue = FormIssue.HIPS_UP
        elif deviation < -config.PLANK_PIKE_THRESHOLD:
            issue = FormIssue.HIPS_DOWN

        if self._state == PlankState.WAITING:
            self._start_hold()

        result = None
        if issue != FormIssue.NONE:
            if self._state != PlankState.PAUSED:
                self._cues.append("Hips up" if issue == FormIssue.HIPS_UP else "Hips down")
            self._state = PlankState.PAUSED
            self.coaching_message = issue.value
        else:
            self._state = PlankState.HOLDING
            self.hold_seconds += dt
            self.total_hold   += dt
            self._body_angles.append(body_angle)
            result = self._check_milestones()

        self.stats.calories = self.total_hold * config.CALORIES_PER_PLANK_SECOND
        self.depth_progress = min(1.0, self.hold_seconds / config.PLANK_TARGET_SECONDS)
        return result, issue, body_angle

    def reset(self) -> None:
        """Reset all counters, stats, and FSM state."""
        self.__init__()

    def rep_announcement(self) -> Optional[str]:
        """Set completion is announced by the app; nothing per 'rep'."""
        return None

    def hud_fields(self) -> Dict[str, Any]:
        """Hold-time fields for the HUD."""
        return {
            "is_timed":       True,
            "hold_seconds":   self.hold_seconds,
            "target_seconds": float(config.PLANK_TARGET_SECONDS),
            "best_hold":      self.longest_hold,
        }

    def csv_rows(self, angle_label: str) -> List[List[Any]]:
        """Hold summary rows for the session CSV."""
        return [
            ["Longest Hold (s)",     f"{self.longest_hold:.1f}"],
            ["Total Hold Time (s)",  f"{self.total_hold:.1f}"],
            [f"Avg {angle_label} Angle (°)", f"{self.stats.average_depth:.1f}"],
            ["Failed Attempts (s)",  " ".join(f"{a:.1f}" for a in self.attempts)],
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _start_hold(self) -> None:
        """Begin a new hold."""
        self.hold_seconds  = 0.0
        self._target_hit   = False
        self._halfway_said = False
        self._body_angles  = []
        self._cues.append("Hold it")

    def _check_milestones(self) -> Optional[RepResult]:
        """Halfway cue, set completion at the target, and coaching text."""
        target = config.PLANK_TARGET_SECONDS
        if not self._halfway_said and self.hold_seconds >= target / 2:
            self._halfway_said = True
            self._cues.append("Halfway there")

        if not self._target_hit and self.hold_seconds >= target:
            self._target_hit = True
            self.stats.total_reps     += 1
            self.stats.sets_completed += 1
            self.stats.current_set    += 1
            result = RepResult(True, FormIssue.NONE, self._avg_angle(), self.hold_seconds)
            if self.stats.workout_complete:
                self._end_hold()      # counting stops now — record it
            self.coaching_message = "SET COMPLETE! ✓"
            return result

        if self._target_hit:
            self.coaching_message = f"BONUS +{int(self.hold_seconds - target)}s"
        else:
            self.coaching_message = f"HOLD IT! {int(target - self.hold_seconds)}s"
        return None

    def _out_of_position(self, dt: float) -> None:
        """Accumulate time out of position and end the hold after the grace period."""
        if self._state == PlankState.WAITING:
            return
        self._out_time += dt
        if self._out_time >= config.PLANK_BREAK_GRACE:
            self._end_hold()

    def _end_hold(self) -> None:
        """Finish the current hold and record it."""
        if self._target_hit:
            self.stats.rep_durations.append(self.hold_seconds)
            self.stats.depths.append(self._avg_angle())
            self.coaching_message = "NICE HOLD! REST UP"
        elif self.hold_seconds > 0:
            self.attempts.append(self.hold_seconds)
            self.coaching_message = "HOLD ENDED EARLY"
        self._state      = PlankState.WAITING
        self._target_hit = False
        self._out_time   = 0.0

    def _avg_angle(self) -> float:
        """Average shoulder-hip-ankle angle over the current hold."""
        return sum(self._body_angles) / len(self._body_angles) if self._body_angles else 0.0

    @staticmethod
    def _active_side(joints) -> Tuple[Tuple[float, float], ...]:
        """(shoulder, hip, ankle) for the better-visible side."""
        if joints.left_visibility >= joints.right_visibility:
            return joints.left_shoulder, joints.left_hip, joints.left_ankle
        return joints.right_shoulder, joints.right_hip, joints.right_ankle
