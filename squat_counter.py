"""
squat_counter.py
----------------
Finite-state-machine squat counter with form validation.

Architecture note
~~~~~~~~~~~~~~~~~
ExerciseBase  ← abstract base class for all exercises
    └── SquatCounter  ← concrete implementation

New exercises (PushUpCounter, LungeCounter, etc.) should inherit from
ExerciseBase and implement the abstract methods.
"""

from __future__ import annotations

import time
import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import config
from angle_utils import (
    AngleSmoother,
    calculate_angle,
    compute_torso_angle,
    is_hip_below_knee,
    knee_cave_detected,
    map_range,
)

# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

class FormIssue(str, enum.Enum):
    """Enumeration of detectable form problems."""
    NONE           = ""
    GO_LOWER       = "GO LOWER"
    STAND_FULLY    = "STAND FULLY"
    BAD_BACK       = "STRAIGHTEN YOUR BACK"
    KNEE_CAVE      = "KNEES OUT!"
    FACE_SIDEWAYS  = "TURN SIDEWAYS"
    BODY_NOT_FOUND = "STEP INTO FRAME"


class SquatState(str, enum.Enum):
    """States in the squat finite state machine."""
    IDLE         = "idle"
    COUNTDOWN    = "countdown"
    STANDING     = "standing"
    GOING_DOWN   = "going_down"
    BOTTOM       = "bottom"
    GOING_UP     = "going_up"


@dataclass
class RepResult:
    """Outcome of a completed repetition.

    Attributes:
        counted:      Whether the rep was counted (True) or rejected (False).
        form_issue:   FormIssue describing why the rep was rejected (if any).
        min_angle:    Minimum knee angle reached during this rep.
        duration:     Duration of the rep in seconds.
    """
    counted:    bool
    form_issue: FormIssue
    min_angle:  float
    duration:   float


@dataclass
class SessionStats:
    """Running statistics for the current workout session.

    Attributes:
        total_reps:       Accepted rep count across all sets.
        current_set:      1-based current set number.
        sets_completed:   Number of fully completed sets.
        rep_durations:    Duration (seconds) of each accepted rep.
        depths:           Min knee angle achieved for each accepted rep.
        start_time:       Session wall-clock start timestamp.
        calories:         Estimated calories burned.
    """
    total_reps:      int   = 0
    current_set:     int   = 1
    sets_completed:  int   = 0
    rep_durations:   List[float] = field(default_factory=list)
    depths:          List[float] = field(default_factory=list)
    start_time:      float = field(default_factory=time.time)
    calories:        float = 0.0

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def elapsed_seconds(self) -> float:
        """Seconds elapsed since the session started."""
        return time.time() - self.start_time

    @property
    def reps_in_current_set(self) -> int:
        """Reps completed in the current (active) set."""
        return self.total_reps - self.sets_completed * config.TARGET_REPS

    @property
    def average_depth(self) -> float:
        """Average minimum knee angle across all accepted reps."""
        return sum(self.depths) / len(self.depths) if self.depths else 0.0

    @property
    def fastest_rep(self) -> float:
        """Duration of the fastest completed rep (seconds)."""
        return min(self.rep_durations) if self.rep_durations else 0.0

    @property
    def slowest_rep(self) -> float:
        """Duration of the slowest completed rep (seconds)."""
        return max(self.rep_durations) if self.rep_durations else 0.0

    @property
    def workout_complete(self) -> bool:
        """True when all target sets have been completed."""
        return self.sets_completed >= config.TARGET_SETS


# ---------------------------------------------------------------------------
# Abstract base exercise class
# ---------------------------------------------------------------------------

class ExerciseBase(ABC):
    """Abstract base class for exercise counter modules.

    Every new exercise should inherit this class and implement
    ``process_frame``.  Audio feedback and session statistics are provided
    here so subclasses can share the infrastructure.
    """

    def __init__(self) -> None:
        """Initialise base state."""
        self.stats  = SessionStats()
        self._state = SquatState.IDLE

    @abstractmethod
    def process_frame(
        self,
        joints,
        frame_shape: Tuple[int, int],
    ) -> Tuple[Optional[RepResult], FormIssue, float]:
        """Analyse one frame and update internal state.

        Args:
            joints:      JointData from PoseDetector.
            frame_shape: (height, width) of the video frame.

        Returns:
            (RepResult | None, active_form_issue, current_angle)
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset all counters and state to initial values."""
        ...

    @property
    def state(self) -> SquatState:
        """Current FSM state."""
        return self._state


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

        # Depth-progress value for the progress bar (0-1)
        self.depth_progress: float = 0.0

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
            self.stats.total_reps    += 1
            self.stats.rep_durations.append(duration)
            self.stats.depths.append(min_angle)
            self.stats.calories       = self.stats.total_reps * config.CALORIES_PER_SQUAT
            self.coaching_message     = "GOOD REP! ✓"

            # Check set completion
            if self.stats.reps_in_current_set >= config.TARGET_REPS:
                self.stats.sets_completed += 1
                self.stats.current_set    += 1
                self.coaching_message      = "SET COMPLETE!"
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


# ---------------------------------------------------------------------------
# Stub classes — extend these for future exercises
# ---------------------------------------------------------------------------

class PushUpCounter(ExerciseBase):
    """Placeholder for push-up counting (not yet implemented)."""

    def process_frame(self, joints, frame_shape):  # type: ignore[override]
        raise NotImplementedError

    def reset(self) -> None:
        self.stats = SessionStats()


class LungeCounter(ExerciseBase):
    """Placeholder for lunge counting (not yet implemented)."""

    def process_frame(self, joints, frame_shape):  # type: ignore[override]
        raise NotImplementedError

    def reset(self) -> None:
        self.stats = SessionStats()


class CurlCounter(ExerciseBase):
    """Placeholder for bicep-curl counting (not yet implemented)."""

    def process_frame(self, joints, frame_shape):  # type: ignore[override]
        raise NotImplementedError

    def reset(self) -> None:
        self.stats = SessionStats()


class PlankCounter(ExerciseBase):
    """Placeholder for plank timing (not yet implemented)."""

    def process_frame(self, joints, frame_shape):  # type: ignore[override]
        raise NotImplementedError

    def reset(self) -> None:
        self.stats = SessionStats()


class JumpingJackCounter(ExerciseBase):
    """Placeholder for jumping jack counting (not yet implemented)."""

    def process_frame(self, joints, frame_shape):  # type: ignore[override]
        raise NotImplementedError

    def reset(self) -> None:
        self.stats = SessionStats()
