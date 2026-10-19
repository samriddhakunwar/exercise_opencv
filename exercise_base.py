"""
exercise_base.py
----------------
Shared base infrastructure for every exercise counter.

Architecture note
~~~~~~~~~~~~~~~~~
ExerciseBase  ← abstract base class for all exercises
    ├── SquatCounter      ← knee-angle FSM          (squat_counter.py)
    ├── PushUpCounter     ← elbow-angle FSM          (pushup_counter.py)
    └── HighKneeCounter   ← dual-leg marching FSM    (high_knee_counter.py)

New exercises should inherit from ExerciseBase and implement
``process_frame`` and ``reset``, then add one entry to ``exercises.EXERCISES``.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import clock
import config


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

@dataclass
class JointData:
    """Container for key joint positions (image pixel coords).

    Attributes:
        nose:            (x, y) nose — used as the "head" reference.
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
        left_foot:       (x, y) left foot index (toe tip).
        right_foot:      (x, y) right foot index (toe tip).
        left_visibility:      Visibility score for the left leg chain.
        right_visibility:     Visibility score for the right leg chain.
        left_arm_visibility:  Visibility score for the left arm chain.
        right_arm_visibility: Visibility score for the right arm chain.
    """
    nose:                 Tuple[float, float] = (0.0, 0.0)
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
    left_foot:            Tuple[float, float] = (0.0, 0.0)
    right_foot:           Tuple[float, float] = (0.0, 0.0)
    left_visibility:      float = 0.0
    right_visibility:     float = 0.0
    left_arm_visibility:  float = 0.0
    right_arm_visibility: float = 0.0
    raw_landmarks:        Optional[object] = field(default=None, repr=False)


class FormIssue(str, enum.Enum):
    """Enumeration of detectable form problems (shared across all exercises)."""
    NONE           = ""
    # Squat-specific
    GO_LOWER       = "GO LOWER"
    STAND_FULLY    = "STAND FULLY"
    BAD_BACK       = "STRAIGHTEN YOUR BACK"
    KNEE_CAVE      = "KNEES OUT!"
    FACE_SIDEWAYS  = "TURN SIDEWAYS"
    BODY_NOT_FOUND = "STEP INTO FRAME"
    # Push-up-specific
    LOCK_ARMS      = "LOCK OUT YOUR ARMS"
    GO_LOWER_PU    = "GO LOWER"
    STRAIGHTEN     = "STRAIGHTEN YOUR BODY"
    HIPS_UP        = "KEEP YOUR HIPS UP"
    HIPS_DOWN      = "LOWER YOUR HIPS"
    MOVE_SIDEWAYS  = "MOVE SIDEWAYS"
    # High Knee-specific
    RAISE_KNEES_HIGHER = "LIFT ABOVE HIP!"
    ALTERNATE_LEGS     = "ALTERNATE LEGS"
    MOVE_FASTER        = "MOVE FASTER!"
    SLOW_DOWN          = "SLOW DOWN"
    # Plank-specific (HIPS_UP / HIPS_DOWN shared with push-ups)
    GET_IN_PLANK       = "GET INTO PLANK POSITION"
    # Jumping-jack-specific
    ARMS_OVERHEAD      = "ARMS ALL THE WAY UP"
    FEET_WIDER         = "JUMP FEET WIDER"
    # Curl-specific
    ELBOW_DRIFT        = "KEEP ELBOWS STILL"
    EXTEND_ARM         = "EXTEND YOUR ARM FULLY"
    CURL_HIGHER        = "CURL ALL THE WAY UP"
    # Lunge-specific
    KNEE_OVER_TOES     = "KNEE BEHIND TOES"


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
    start_time:      float = field(default_factory=clock.now)
    calories:        float = 0.0
    # Runtime targets — set via configure() by each exercise
    _target_reps:    int   = field(default=config.TARGET_REPS, repr=False)
    _target_sets:    int   = field(default=config.TARGET_SETS, repr=False)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(self, target_reps: int, target_sets: int) -> None:
        """Set exercise-specific workout targets.

        Must be called once during exercise __init__ so that the
        derived properties (reps_in_current_set, workout_complete)
        use the correct values for the active exercise.

        Args:
            target_reps: Number of reps per set.
            target_sets: Total sets to complete.
        """
        self._target_reps = target_reps
        self._target_sets = target_sets

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def target_reps(self) -> int:
        """Reps per set (as configured)."""
        return self._target_reps

    @property
    def elapsed_seconds(self) -> float:
        """Seconds elapsed since the session started."""
        return clock.now() - self.start_time

    @property
    def reps_in_current_set(self) -> int:
        """Reps completed in the current (active) set."""
        return self.total_reps - self.sets_completed * self._target_reps

    @property
    def average_depth(self) -> float:
        """Average minimum angle across all accepted reps."""
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
        return self.sets_completed >= self._target_sets


# ---------------------------------------------------------------------------
# Abstract base exercise class
# ---------------------------------------------------------------------------

class ExerciseBase(ABC):
    """Abstract base class for exercise counter modules.

    Every new exercise should inherit this class and implement
    ``process_frame``.  Audio feedback and session statistics are provided
    here so subclasses can share the infrastructure.
    """

    # Subclasses should override these class-level defaults
    TARGET_REPS: int = config.TARGET_REPS
    TARGET_SETS: int = config.TARGET_SETS
    CALORIES_PER_REP: float = config.CALORIES_PER_SQUAT
    # Direction of best_metric(): True for angles (deeper = smaller angle)
    BEST_LOWER_IS_BETTER: bool = True

    def __init__(self) -> None:
        """Initialise base state."""
        self.stats  = SessionStats()
        self.stats.configure(self.TARGET_REPS, self.TARGET_SETS)
        self._state: Optional[enum.Enum] = None
        # Depth progress (0-1) used by the progress bar
        self.depth_progress: float = 0.0
        # Live coaching text
        self.coaching_message: str = "GET READY"
        # Phrases for the app to speak (mid-rep cues like "Hips up")
        self._cues: List[str] = []

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
    def state(self):
        """Current FSM state (SquatState, PushUpState, …)."""
        return self._state

    # ------------------------------------------------------------------
    # Hooks used by the app (override where an exercise differs)
    # ------------------------------------------------------------------

    @property
    def session_complete(self) -> bool:
        """True when the workout for this exercise is finished."""
        return self.stats.workout_complete

    def set_targets(
        self,
        reps: Optional[int] = None,
        sets: Optional[int] = None,
        seconds: Optional[float] = None,
    ) -> None:
        """Override this instance's targets (workout routines) and reset.

        Args:
            reps:    Reps per set (per side for sided exercises).
            sets:    Total sets.
            seconds: Hold target for timed exercises (plank).
        """
        if reps is not None:
            self.TARGET_REPS = reps
        if sets is not None:
            self.TARGET_SETS = sets
        if seconds is not None:
            self.TARGET_SECONDS = seconds
        self.reset()

    def on_session_start(self) -> None:
        """Called once when the countdown ends and counting begins."""

    def on_session_end(self) -> None:
        """Called before the session is saved (finish any open hold, …)."""

    def avg_metric(self) -> Optional[float]:
        """Average of the primary metric for the history table."""
        return self.stats.average_depth if self.stats.depths else None

    def best_metric(self) -> Optional[float]:
        """Best single-rep value of the primary metric (deepest angle, …)."""
        if not self.stats.depths:
            return None
        return (min if self.BEST_LOWER_IS_BETTER else max)(self.stats.depths)

    def record_value(self) -> float:
        """Value compared against history for "New record!" (default: reps)."""
        return float(self.stats.total_reps)

    def pop_cues(self) -> List[str]:
        """Return and clear phrases queued for the audio dispatcher."""
        cues, self._cues = self._cues, []
        return cues

    def rep_announcement(self) -> Optional[str]:
        """Phrase spoken when a rep is counted (None = stay quiet)."""
        return str(self.stats.total_reps)

    def hud_fields(self) -> Dict[str, Any]:
        """Extra RenderData keyword arguments specific to this exercise."""
        return {}

    def csv_rows(self, angle_label: str) -> List[List[Any]]:
        """Exercise-specific summary rows for the session CSV.

        Args:
            angle_label: Human-readable label of the primary angle.
        """
        return [
            [f"Avg {angle_label} Angle (°)", f"{self.stats.average_depth:.1f}"],
            ["Fastest Rep (s)", f"{self.stats.fastest_rep:.2f}"],
            ["Slowest Rep (s)", f"{self.stats.slowest_rep:.2f}"],
        ]

    def _accept_rep(
        self,
        duration: float,
        angle: float,
        calories_per_rep: float,
        target_reps: int,
        target_sets: int,
    ) -> None:
        """Increment stats for an accepted rep and handle set transitions.

        Args:
            duration:         Rep duration in seconds.
            angle:            Primary angle metric for this rep (min elbow angle, etc.).
            calories_per_rep: Calories burned per rep.
            target_reps:      Reps per set.
            target_sets:      Total sets (unused here — kept for symmetry).
        """
        self.stats.total_reps     += 1
        self.stats.rep_durations.append(duration)
        self.stats.depths.append(angle)
        self.stats.calories = self.stats.total_reps * calories_per_rep
        if self.stats.reps_in_current_set >= target_reps:
            self.stats.sets_completed += 1
            self.stats.current_set    += 1


# ---------------------------------------------------------------------------
# Per-side exercises (curls, lunges)
# ---------------------------------------------------------------------------

class SidedExerciseBase(ExerciseBase):
    """Base for exercises counted per side with a per-side target.

    A set completes when *both* sides reach ``TARGET_REPS``.  SessionStats is
    configured with ``2 × TARGET_REPS`` so ``reps_in_current_set`` is the
    combined left + right count for the current set.
    """

    def __init__(self) -> None:
        """Initialise per-side counters."""
        super().__init__()
        self.stats.configure(2 * self.TARGET_REPS, self.TARGET_SETS)
        self.side_reps:  Dict[str, int] = {"left": 0, "right": 0}   # current set
        self.side_total: Dict[str, int] = {"left": 0, "right": 0}   # whole session
        self._last_side: str = "left"

    def _accept_side_rep(self, side: str, duration: float, angle: float) -> None:
        """Record one accepted rep on *side* and roll the set over if done.

        Reps beyond the target on one side are kept in ``side_total`` (and
        calories) but not in ``stats.total_reps`` / the current set, so
        ``reps_in_current_set`` never exceeds the combined target.
        """
        self._last_side = side
        self.side_total[side] += 1
        self.stats.rep_durations.append(duration)
        self.stats.depths.append(angle)
        self.stats.calories = sum(self.side_total.values()) * self.CALORIES_PER_REP
        if self.side_reps[side] >= self.TARGET_REPS:
            return
        self.side_reps[side] += 1
        self.stats.total_reps += 1
        if all(n >= self.TARGET_REPS for n in self.side_reps.values()):
            self.stats.sets_completed += 1
            self.stats.current_set    += 1
            self.side_reps = {"left": 0, "right": 0}

    def rep_announcement(self) -> Optional[str]:
        """Speak the side and its count, e.g. "Left 4"."""
        side = self._last_side
        return f"{side.capitalize()} {self.side_total[side]}"

    def hud_fields(self) -> Dict[str, Any]:
        """Per-side counts for the HUD."""
        return {"left_reps": self.side_reps["left"], "right_reps": self.side_reps["right"]}

    def csv_rows(self, angle_label: str) -> List[List[Any]]:
        """Base rows plus per-side totals."""
        return super().csv_rows(angle_label) + [
            ["Left Reps",  self.side_total["left"]],
            ["Right Reps", self.side_total["right"]],
        ]
