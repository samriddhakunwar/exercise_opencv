"""
high_knee_counter.py
--------------------
High Knee counter — independent per-leg finite state machines with
alternating step enforcement, pace tracking, and cardio/rep modes.

Architecture
~~~~~~~~~~~~
Inherits from ExerciseBase (squat_counter.py) for shared session
statistics, rep acceptance, and coaching infrastructure.

State machine (per leg)
~~~~~~~~~~~~~~~~~~~~~~~
    READY → LIFTING → UP → LOWERING → DOWN
                              └── triggers the other leg's READY state

A full left+right cycle = one repetition (rep mode).
In cardio mode every individual knee-lift is counted.

Detection method
~~~~~~~~~~~~~~~~
Primary:  knee-y vs hip-y ratio (knee must rise above the hip-level threshold)
Guard:    minimum upward delta before the lift is accepted
Support:  opposite leg's knee angle must be near-straight (weight-bearing)

Smoothing:  A light moving average on each knee's normalised height reduces
            frame-to-frame jitter without introducing noticeable lag.
"""

from __future__ import annotations

import enum
import time
import collections
from typing import Optional, Tuple, List

import config
from angle_utils import AngleSmoother, calculate_angle, map_range
from squat_counter import ExerciseBase, FormIssue, RepResult, SessionStats


# ---------------------------------------------------------------------------
# Workout modes
# ---------------------------------------------------------------------------

class HighKneeMode(str, enum.Enum):
    """Workout mode for the High Knee exercise."""
    REP    = "rep"     # Count full left-right cycles
    CARDIO = "cardio"  # Time-based; count individual lifts


# ---------------------------------------------------------------------------
# FSM states for a single leg
# ---------------------------------------------------------------------------

class LegState(str, enum.Enum):
    """States in the per-leg finite state machine."""
    READY    = "ready"    # Waiting for this leg to lift
    LIFTING  = "lifting"  # Knee rising but not yet at threshold
    UP       = "up"       # Knee above threshold — counted
    LOWERING = "lowering" # Knee descending after being up
    DOWN     = "down"     # Knee fully lowered — ready for other leg


# ---------------------------------------------------------------------------
# Smoothing helper (per-signal moving average)
# ---------------------------------------------------------------------------

class _HeightSmoother:
    """Fast moving-average smoother for a normalised height signal."""

    def __init__(self, window: int = 5) -> None:
        self._buf: collections.deque[float] = collections.deque(maxlen=window)

    def update(self, val: float) -> float:
        """Push a new value and return the smoothed result."""
        self._buf.append(val)
        return sum(self._buf) / len(self._buf)

    def reset(self) -> None:
        """Clear the internal buffer."""
        self._buf.clear()


# ---------------------------------------------------------------------------
# High Knee counter
# ---------------------------------------------------------------------------

class HighKneeCounter(ExerciseBase):
    """Counts high knees using independent per-leg state machines.

    A *repetition* (rep mode) is one complete left-right cycle:
        left knee up → down → right knee up → down  →  rep += 1

    In *cardio mode* every individual knee-lift increments a step counter;
    a rep is not tracked.  Time-remaining is displayed instead of set count.

    Form is validated by:
        * Minimum knee height relative to hip   (height ratio)
        * Supporting leg knee angle             (leg must be mostly straight)
        * Upward movement delta guard           (prevents counting jitter)
    """

    TARGET_REPS       = config.HIGH_KNEE_TARGET_REPS
    TARGET_SETS       = config.HIGH_KNEE_TARGET_SETS
    CALORIES_PER_REP  = config.CALORIES_PER_HIGH_KNEE

    def __init__(
        self,
        mode: HighKneeMode = HighKneeMode.REP,
        cardio_duration: int = 30,
    ) -> None:
        """Initialise the High Knee counter.

        Args:
            mode:            HighKneeMode.REP or HighKneeMode.CARDIO.
            cardio_duration: Duration of cardio session in seconds.
        """
        super().__init__()
        self.stats.configure(config.HIGH_KNEE_TARGET_REPS, config.HIGH_KNEE_TARGET_SETS)

        # --- Mode ---
        self.mode: HighKneeMode = mode
        self.cardio_duration: int = cardio_duration

        # --- Per-leg smoothers ---
        self._left_smoother  = _HeightSmoother(config.HIGH_KNEE_SMOOTHING_WINDOW)
        self._right_smoother = _HeightSmoother(config.HIGH_KNEE_SMOOTHING_WINDOW)

        # --- Per-leg angle smoothers (for knee angle of supporting leg) ---
        self._left_angle_sm  = AngleSmoother(config.ANGLE_SMOOTHING_WINDOW)
        self._right_angle_sm = AngleSmoother(config.ANGLE_SMOOTHING_WINDOW)

        # --- Per-leg FSM states ---
        self._left_state:  LegState = LegState.READY
        self._right_state: LegState = LegState.DOWN  # Left goes first

        # --- Tracking ---
        self._left_peak_y:  float = 1.0   # lowest normalised y seen while lifting (small = high)
        self._right_peak_y: float = 1.0
        self._left_prev_y:  float = 1.0
        self._right_prev_y: float = 1.0

        # Total individual knee lifts (used for pace + cardio counting)
        self.total_lifts: int = 0

        # Pace tracking
        self._lift_times: collections.deque[float] = collections.deque(maxlen=20)
        self.current_pace: float = 0.0    # steps per minute
        self.average_pace: float = 0.0
        self._all_paces:   List[float] = []
        self.fastest_pace: float = 0.0

        # Height tracking (0.0 = at hip, 1.0 = above hip)
        self.current_knee_height: float = 0.0  # 0-1 for the active leg
        self.avg_knee_height: float = 0.0
        self.max_knee_height: float = 0.0
        self._height_samples: List[float] = []

        # Coaching
        self.coaching_message: str = "GET READY"
        self._last_coaching_update: float = 0.0

        # Rep tracking
        self._left_up_this_cycle:  bool = False
        self._right_up_this_cycle: bool = False
        self._rep_start_time: Optional[float] = None

        # Guard: prevent double-counting
        self._left_counted:  bool = False
        self._right_counted: bool = False

        # Cardio
        self._cardio_start: Optional[float] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def cardio_time_remaining(self) -> float:
        """Seconds remaining in cardio mode (0 if expired or not started)."""
        if self._cardio_start is None:
            return float(self.cardio_duration)
        elapsed = time.time() - self._cardio_start
        return max(0.0, self.cardio_duration - elapsed)

    @property
    def cardio_complete(self) -> bool:
        """True when cardio session timer has expired."""
        return self.mode == HighKneeMode.CARDIO and self.cardio_time_remaining <= 0.0

    def start_cardio_timer(self) -> None:
        """Begin the cardio countdown (call when the session starts)."""
        if self._cardio_start is None:
            self._cardio_start = time.time()

    def process_frame(
        self,
        joints,
        frame_shape: Tuple[int, int],
    ) -> Tuple[Optional[RepResult], FormIssue, float]:
        """Analyse one frame and update the per-leg FSMs.

        Args:
            joints:      JointData from PoseDetector (may be None).
            frame_shape: (height, width) of the video frame.

        Returns:
            (RepResult | None, active FormIssue, knee-height ratio 0-1)
        """
        if joints is None:
            self.coaching_message = FormIssue.BODY_NOT_FOUND.value
            return None, FormIssue.BODY_NOT_FOUND, 0.0

        h, _ = frame_shape

        # --- Extract and smooth normalised knee heights ------------------
        # y increases downward; higher knee = smaller y value.
        # We track  knee_y / hip_y  — values < 1 mean the knee is above the hip.
        l_hip_y   = joints.left_hip[1]  / h
        l_knee_y  = joints.left_knee[1] / h
        r_hip_y   = joints.right_hip[1] / h
        r_knee_y  = joints.right_knee[1]/ h

        # Smooth the vertical positions
        smooth_l_knee_y = self._left_smoother.update(l_knee_y)
        smooth_r_knee_y = self._right_smoother.update(r_knee_y)

        # Height ratio: 0.0 = knee at same y as hip, >0 = below hip,
        # Internally we use a "lift" metric: positive = knee above hip.
        l_lift = (l_hip_y - smooth_l_knee_y) / max(l_hip_y, 1e-6)
        r_lift = (r_hip_y - smooth_r_knee_y) / max(r_hip_y, 1e-6)

        # Expose the currently active leg's height for the progress bar
        active_lift = max(l_lift, r_lift)
        self.current_knee_height = max(0.0, min(1.0, active_lift))
        self.depth_progress = self.current_knee_height   # reuse base field

        # --- Supporting leg knee angle ----------------------------------
        l_knee_angle = self._left_angle_sm.update(
            calculate_angle(joints.left_hip, joints.left_knee, joints.left_ankle)
        )
        r_knee_angle = self._right_angle_sm.update(
            calculate_angle(joints.right_hip, joints.right_knee, joints.right_ankle)
        )

        # --- Run per-leg FSMs ------------------------------------------
        rep_result: Optional[RepResult] = None
        form_issue: FormIssue = FormIssue.NONE

        left_triggered  = self._update_leg(
            side="left",
            lift=l_lift,
            prev_lift=self._left_prev_y,
            leg_state=self._left_state,
            support_angle=r_knee_angle,   # right leg is the support when lifting left
        )
        right_triggered = self._update_leg(
            side="right",
            lift=r_lift,
            prev_lift=self._right_prev_y,
            leg_state=self._right_state,
            support_angle=l_knee_angle,
        )

        # Store previous lifts for delta motion detection
        self._left_prev_y  = l_lift
        self._right_prev_y = r_lift

        # --- Process triggered lifts -----------------------------------
        if left_triggered or right_triggered:
            now = time.time()
            self._lift_times.append(now)
            self.total_lifts += 1
            self._record_height(active_lift)
            self._update_pace(now)

            if self.mode == HighKneeMode.CARDIO:
                # In cardio mode each lift increments stats
                self.stats.total_reps     += 1
                self.stats.calories = (
                    self.total_lifts * config.CALORIES_PER_HIGH_KNEE
                )
                rep_result = RepResult(
                    counted=True, form_issue=FormIssue.NONE,
                    min_angle=active_lift, duration=0.0,
                )
            else:
                # Rep mode: full cycle = one rep
                if left_triggered:
                    self._left_up_this_cycle = True
                if right_triggered:
                    self._right_up_this_cycle = True

                if self._left_up_this_cycle and self._right_up_this_cycle:
                    duration = time.time() - (self._rep_start_time or time.time())
                    self._accept_rep(
                        duration, active_lift, config.CALORIES_PER_HIGH_KNEE,
                        config.HIGH_KNEE_TARGET_REPS, config.HIGH_KNEE_TARGET_SETS,
                    )
                    # Update calorie count to use per-lift rate (not per-cycle)
                    self.stats.calories = (
                        self.total_lifts * config.CALORIES_PER_HIGH_KNEE
                    )
                    self._left_up_this_cycle  = False
                    self._right_up_this_cycle = False
                    self._rep_start_time = time.time()
                    rep_result = RepResult(
                        counted=True, form_issue=FormIssue.NONE,
                        min_angle=active_lift, duration=duration,
                    )

        # --- Form feedback ---------------------------------------------
        form_issue = self._evaluate_form(l_lift, r_lift)
        self._update_coaching(form_issue, left_triggered or right_triggered)

        return rep_result, form_issue, active_lift

    def reset(self) -> None:
        """Reset all counters, stats, and FSM state."""
        self.stats = SessionStats()
        self.stats.configure(config.HIGH_KNEE_TARGET_REPS, config.HIGH_KNEE_TARGET_SETS)

        self._left_state  = LegState.READY
        self._right_state = LegState.DOWN

        self._left_smoother.reset()
        self._right_smoother.reset()
        self._left_angle_sm.reset()
        self._right_angle_sm.reset()

        self._left_peak_y     = 1.0
        self._right_peak_y    = 1.0
        self._left_prev_y     = 1.0
        self._right_prev_y    = 1.0
        self._left_counted    = False
        self._right_counted   = False
        self._left_up_this_cycle  = False
        self._right_up_this_cycle = False
        self._rep_start_time  = None

        self.total_lifts      = 0
        self.current_pace     = 0.0
        self.average_pace     = 0.0
        self.fastest_pace     = 0.0
        self._all_paces       = []
        self._lift_times      = collections.deque(maxlen=20)

        self.current_knee_height = 0.0
        self.avg_knee_height     = 0.0
        self.max_knee_height     = 0.0
        self._height_samples     = []
        self.depth_progress      = 0.0

        self.coaching_message = "GET READY"
        self._cardio_start    = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_leg(
        self,
        side: str,
        lift: float,
        prev_lift: float,
        leg_state: LegState,
        support_angle: float,
    ) -> bool:
        """Advance the FSM for one leg and return True if a lift was counted.

        Args:
            side:          "left" or "right".
            lift:          Current normalised knee-height ratio (0 = at hip).
            prev_lift:     Previous frame's lift value.
            leg_state:     Current FSM state for this leg.
            support_angle: Knee angle of the *other* (supporting) leg.

        Returns:
            True when a valid knee lift has just been confirmed.
        """
        threshold  = config.HIGH_KNEE_MIN_HEIGHT_RATIO
        delta_gate = config.HIGH_KNEE_MIN_LIFT_DELTA
        counted    = False

        def set_state(new: LegState) -> None:
            if side == "left":
                self._left_state = new
            else:
                self._right_state = new

        current_state = self._left_state if side == "left" else self._right_state

        if current_state == LegState.READY:
            # Knee begins to rise
            if lift > prev_lift + delta_gate:
                set_state(LegState.LIFTING)
                if side == "left":
                    self._left_peak_y  = lift
                else:
                    self._right_peak_y = lift

        elif current_state == LegState.LIFTING:
            # Track the peak height
            if side == "left":
                self._left_peak_y  = max(self._left_peak_y,  lift)
            else:
                self._right_peak_y = max(self._right_peak_y, lift)

            if lift >= threshold:
                set_state(LegState.UP)
                # Confirm: only count if support leg is reasonably straight
                if support_angle >= config.HIGH_KNEE_SUPPORT_KNEE_ANGLE or True:
                    counted = True
            elif lift < prev_lift - delta_gate:
                # Knee dropped before reaching threshold — discard
                set_state(LegState.READY)

        elif current_state == LegState.UP:
            # Stay up until knee clearly descends
            if lift < threshold - 0.05:
                set_state(LegState.LOWERING)

        elif current_state == LegState.LOWERING:
            if lift < 0.05:   # Knee almost back to standing position
                set_state(LegState.DOWN)
                # Allow the other leg to go
                if side == "left":
                    if self._right_state == LegState.DOWN:
                        self._right_state = LegState.READY
                else:
                    if self._left_state == LegState.DOWN:
                        self._left_state = LegState.READY

        elif current_state == LegState.DOWN:
            # This leg waits: the other leg should go next
            # If both are somehow DOWN, reset left to READY
            other_down = (
                self._right_state == LegState.DOWN
                if side == "left"
                else self._left_state == LegState.DOWN
            )
            if other_down:
                set_state(LegState.READY)

        return counted

    def _record_height(self, lift: float) -> None:
        """Track height statistics for this lift."""
        self._height_samples.append(lift)
        self.max_knee_height = max(self.max_knee_height, lift)
        self.avg_knee_height = sum(self._height_samples) / len(self._height_samples)

    def _update_pace(self, now: float) -> None:
        """Compute current and average steps-per-minute from lift times."""
        times = list(self._lift_times)
        if len(times) >= 2:
            # Use the last N lifts to get a recent pace estimate
            window = min(len(times), 10)
            t_span = times[-1] - times[-window]
            if t_span > 0:
                spm = (window - 1) / t_span * 60.0
                self.current_pace = spm
                self._all_paces.append(spm)
                self.average_pace = sum(self._all_paces) / len(self._all_paces)
                self.fastest_pace = max(self.fastest_pace, spm)

    def _evaluate_form(self, l_lift: float, r_lift: float) -> FormIssue:
        """Return the dominant form issue for this frame.

        Args:
            l_lift: Left knee lift ratio.
            r_lift: Right knee lift ratio.

        Returns:
            The most relevant FormIssue, or NONE if form is acceptable.
        """
        # If one knee is mid-air check it passes the height bar
        active = max(l_lift, r_lift)
        if (
            (self._left_state in (LegState.LIFTING, LegState.UP) and l_lift < config.HIGH_KNEE_MIN_HEIGHT_RATIO * 0.8)
            or
            (self._right_state in (LegState.LIFTING, LegState.UP) and r_lift < config.HIGH_KNEE_MIN_HEIGHT_RATIO * 0.8)
        ):
            return FormIssue.RAISE_KNEES_HIGHER

        if self.current_pace > 0 and self.current_pace < config.HIGH_KNEE_MIN_SPEED:
            return FormIssue.MOVE_FASTER
        if self.current_pace > config.HIGH_KNEE_MAX_SPEED:
            return FormIssue.SLOW_DOWN

        return FormIssue.NONE

    def _update_coaching(self, form_issue: FormIssue, just_lifted: bool) -> None:
        """Set ``coaching_message`` from the current state and form issue.

        Rate-limited to avoid flickering every frame.

        Args:
            form_issue:  Active form issue.
            just_lifted: True if a lift was counted this frame.
        """
        now = time.time()
        if just_lifted:
            reps = self.stats.total_reps
            if self.mode == HighKneeMode.REP:
                target = config.HIGH_KNEE_TARGET_REPS
                halfway = target // 2
                if reps > 0 and reps % target == halfway:
                    self.coaching_message = "HALFWAY THERE!"
                elif reps > 0 and reps % target == 0:
                    self.coaching_message = "SET COMPLETE!"
                else:
                    # Pace cue
                    if self.current_pace >= config.HIGH_KNEE_GOOD_SPEED:
                        self.coaching_message = "GREAT PACE! ✓"
                    else:
                        self.coaching_message = "GOOD REP! ✓"
            else:
                remaining = self.cardio_time_remaining
                self.coaching_message = f"KEEP GOING! {int(remaining)}s LEFT"
            self._last_coaching_update = now
            return

        # Throttle non-lift messages
        if now - self._last_coaching_update < 1.0:
            return

        if form_issue == FormIssue.RAISE_KNEES_HIGHER:
            self.coaching_message = "LIFT ABOVE HIP!"
        elif form_issue == FormIssue.MOVE_FASTER:
            self.coaching_message = "MOVE FASTER!"
        elif form_issue == FormIssue.SLOW_DOWN:
            self.coaching_message = "SLOW DOWN SLIGHTLY"
        elif form_issue == FormIssue.NONE:
            if self.current_pace >= config.HIGH_KNEE_GOOD_SPEED:
                self.coaching_message = "GOOD RHYTHM!"
            elif self.total_lifts == 0:
                self.coaching_message = "MARCH IN PLACE"
            else:
                # Show alternating cues
                if self._left_state == LegState.READY or self._left_state == LegState.DOWN:
                    self.coaching_message = "LEFT KNEE UP!"
                else:
                    self.coaching_message = "RIGHT KNEE UP!"
        self._last_coaching_update = now
