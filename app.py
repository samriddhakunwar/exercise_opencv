"""
app.py
------
Entry point for the Squat Tracker application.

Responsibilities:
    * Camera capture loop
    * Gluing pose detection → exercise counter → UI renderer
    * Keyboard input (Space, R, Q, S)
    * Audio feedback dispatch
    * Session CSV export
    * Screenshot / recording management
"""

from __future__ import annotations

# Must be the very first local import — patches the protobuf/TF conflict
# before MediaPipe can attempt to import tensorflow.tools.docs.
import _mp_compat  # noqa: F401

import sys

import csv
import os
import time
import threading
import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config
from pose_detector import PoseDetector
from squat_counter import SquatCounter, FormIssue, RepResult
from ui import UIRenderer, RenderData

# ---------------------------------------------------------------------------
# Optional audio — graceful fallback when pyttsx3 is absent
# ---------------------------------------------------------------------------
try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False


class AudioFeedback:
    """Thread-safe text-to-speech dispatcher with a per-phrase cooldown.

    Runs TTS in a background thread to avoid blocking the render loop.
    """

    def __init__(self) -> None:
        """Initialise the TTS engine if available."""
        self._enabled   = config.AUDIO_ENABLED and _TTS_AVAILABLE
        self._cooldown  = config.AUDIO_COOLDOWN
        self._last_time: dict[str, float] = {}
        self._lock      = threading.Lock()
        self._engine: Optional[object] = None

        if self._enabled:
            try:
                engine = pyttsx3.init()   # type: ignore[attr-defined]
                engine.setProperty("rate", 165)
                engine.setProperty("volume", 0.9)
                self._engine = engine
            except Exception:
                self._enabled = False

    def say(self, phrase: str) -> None:
        """Speak *phrase* if the cooldown has elapsed.

        Args:
            phrase: Text to synthesise.
        """
        if not self._enabled or not self._engine:
            return
        now = time.time()
        with self._lock:
            last = self._last_time.get(phrase, 0.0)
            if now - last < self._cooldown:
                return
            self._last_time[phrase] = now

        threading.Thread(
            target=self._speak,
            args=(phrase,),
            daemon=True,
        ).start()

    def _speak(self, phrase: str) -> None:
        """Internal: run TTS (called in background thread)."""
        try:
            self._engine.say(phrase)       # type: ignore[union-attr]
            self._engine.runAndWait()      # type: ignore[union-attr]
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

class StatsLogger:
    """Saves session statistics to a CSV file in the workouts directory."""

    def __init__(self) -> None:
        """Ensure the output directory exists."""
        self._dir = Path(config.STATS_DIR)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, stats, elapsed: float) -> Path:
        """Write session stats to a timestamped CSV file.

        Args:
            stats:   SessionStats object from the counter.
            elapsed: Total session duration in seconds.

        Returns:
            Path of the written file.
        """
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._dir / f"session_{ts}.csv"

        rows = [
            ["Metric", "Value"],
            ["Date",          datetime.datetime.now().strftime("%Y-%m-%d %H:%M")],
            ["Total Reps",    stats.total_reps],
            ["Sets Completed",stats.sets_completed],
            ["Duration (s)",  f"{elapsed:.1f}"],
            ["Avg Depth (°)", f"{stats.average_depth:.1f}"],
            ["Fastest Rep (s)", f"{stats.fastest_rep:.2f}"],
            ["Slowest Rep (s)", f"{stats.slowest_rep:.2f}"],
            ["Calories",       f"{stats.calories:.1f}"],
        ]
        # Rep-by-rep breakdown
        rows.append([])
        rows.append(["Rep #", "Duration (s)", "Min Angle (°)"])
        for i, (dur, depth) in enumerate(
            zip(stats.rep_durations, stats.depths), start=1
        ):
            rows.append([i, f"{dur:.2f}", f"{depth:.1f}"])

        with open(path, "w", newline="") as f:
            csv.writer(f).writerows(rows)

        return path


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------

class SquatTrackerApp:
    """Orchestrates the full squat-tracking session.

    Handles:
        * Video capture initialisation
        * Frame processing pipeline
        * Keyboard shortcuts
        * Audio feedback timing
        * Screenshot / optional recording
        * CSV export at session end
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialise all subsystems."""
        self._detector = PoseDetector()
        self._counter  = SquatCounter()
        self._renderer = UIRenderer()
        self._audio    = AudioFeedback()
        self._logger   = StatsLogger()

        # Capture
        self._cap: Optional[cv2.VideoCapture] = None

        # Application state
        self._paused    = False
        self._running   = True
        self._recording = False
        self._writer: Optional[cv2.VideoWriter] = None

        # Countdown state
        self._countdown         = config.COUNTDOWN_SECONDS
        self._countdown_start   = 0.0
        self._in_countdown      = True

        # FPS tracking
        self._fps_buffer: list[float] = []
        self._last_frame_time  = time.time()

        # Rep audio tracking
        self._last_spoken_rep  = 0
        self._set_complete_spoken = False

        # Screenshot dir
        Path("screenshots").mkdir(exist_ok=True)

        # Recording dir
        if config.ENABLE_RECORDING:
            Path(config.RECORDING_DIR).mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Open the webcam and start the main processing loop."""
        self._cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self._cap.isOpened():
            print(f"[ERROR] Cannot open camera index {config.CAMERA_INDEX}.")
            sys.exit(1)

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)

        cv2.namedWindow("Squat Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Squat Tracker", config.FRAME_WIDTH, config.FRAME_HEIGHT)

        self._countdown_start = time.time()

        # Announce startup
        self._audio.say(f"Get ready. Workout starts in {config.COUNTDOWN_SECONDS}.")

        try:
            while self._running:
                ok, frame = self._cap.read()
                if not ok:
                    print("[WARN] Empty frame, retrying...")
                    continue

                frame = cv2.flip(frame, 1)     # Mirror for natural feedback
                self._process_frame(frame)

                key = cv2.waitKey(1) & 0xFF
                self._handle_key(key, frame)

                if not cv2.getWindowProperty(
                    "Squat Tracker", cv2.WND_PROP_VISIBLE
                ):
                    break
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Core per-frame pipeline
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> None:
        """Run detection, counting, and rendering for a single frame.

        Args:
            frame: Raw BGR frame from the webcam.
        """
        now  = time.time()
        fps  = self._compute_fps(now)
        stats = self._counter.stats

        # ---- Countdown phase ----------------------------------------
        countdown_val = 0
        if self._in_countdown:
            elapsed_cd = now - self._countdown_start
            remaining  = config.COUNTDOWN_SECONDS - int(elapsed_cd)
            countdown_val = max(0, remaining)
            if countdown_val == 0:
                self._in_countdown = False
                self._audio.say("Go!")
                self._counter.stats.start_time = time.time()  # Reset timer

        # ---- Pose detection -----------------------------------------
        joints, annotated = self._detector.process(frame)

        # ---- Exercise counting (skip during countdown/pause) ---------
        rep_result: Optional[RepResult] = None
        form_issue = FormIssue.NONE
        knee_angle = 0.0

        if not self._paused and not self._in_countdown and not stats.workout_complete:
            rep_result, form_issue, knee_angle = self._counter.process_frame(
                joints, frame.shape[:2]
            )
            self._dispatch_audio(rep_result, form_issue, stats)

        # ---- Build render data --------------------------------------
        data = RenderData(
            reps_in_set      = stats.reps_in_current_set,
            total_reps       = stats.total_reps,
            current_set      = stats.current_set,
            sets_completed   = stats.sets_completed,
            target_reps      = config.TARGET_REPS,
            target_sets      = config.TARGET_SETS,
            coaching_msg     = self._counter.coaching_message,
            form_issue       = form_issue.value,
            knee_angle       = knee_angle,
            depth_progress   = self._counter.depth_progress,
            elapsed_seconds  = stats.elapsed_seconds,
            fps              = fps,
            is_paused        = self._paused,
            countdown        = countdown_val,
            session_complete = stats.workout_complete,
            calories         = stats.calories,
            fastest_rep      = stats.fastest_rep,
            slowest_rep      = stats.slowest_rep,
            avg_depth        = stats.average_depth,
        )

        # ---- Render -------------------------------------------------
        output = self._renderer.render(annotated, data)

        # ---- Optional recording ------------------------------------
        if self._writer is not None:
            self._writer.write(output)

        cv2.imshow("Squat Tracker", output)

    # ------------------------------------------------------------------
    # Audio dispatch
    # ------------------------------------------------------------------

    def _dispatch_audio(
        self,
        rep_result: Optional[RepResult],
        form_issue: FormIssue,
        stats,
    ) -> None:
        """Trigger spoken feedback based on the latest rep result.

        Args:
            rep_result:  RepResult from this frame (may be None).
            form_issue:  Active form issue.
            stats:       Current session statistics.
        """
        if rep_result is not None:
            if rep_result.counted:
                rep_num = stats.total_reps
                if rep_num != self._last_spoken_rep:
                    self._last_spoken_rep = rep_num
                    self._audio.say(str(rep_num))
                    halfway = config.TARGET_REPS // 2
                    if rep_num % config.TARGET_REPS == halfway:
                        self._audio.say("Halfway there!")
            else:
                # Bad rep feedback
                phrase = {
                    FormIssue.GO_LOWER:    "Go lower",
                    FormIssue.STAND_FULLY: "Stand fully",
                    FormIssue.BAD_BACK:    "Straighten your back",
                    FormIssue.KNEE_CAVE:   "Knees out",
                }.get(rep_result.form_issue, "Bad form")
                self._audio.say(phrase)

        # Set completion announcement
        if (stats.sets_completed > 0
                and stats.reps_in_current_set == 0
                and not self._set_complete_spoken):
            if stats.workout_complete:
                self._audio.say("Workout complete. Great job!")
                if config.SAVE_STATS_CSV:
                    path = self._logger.save(stats, stats.elapsed_seconds)
                    print(f"[INFO] Stats saved to {path}")
            else:
                self._audio.say(f"Set {stats.sets_completed} complete. Rest up!")
            self._set_complete_spoken = True
        elif stats.reps_in_current_set > 0:
            self._set_complete_spoken = False

    # ------------------------------------------------------------------
    # Keyboard input
    # ------------------------------------------------------------------

    def _handle_key(self, key: int, frame: np.ndarray) -> None:
        """Process keyboard shortcuts.

        Args:
            key:   Key code from cv2.waitKey.
            frame: Current (unrendered) frame, used for screenshot.
        """
        if key == ord("q") or key == 27:       # Q or Escape → quit
            self._running = False
        elif key == ord(" "):                   # Space → pause/resume
            self._paused = not self._paused
            phrase = "Paused." if self._paused else "Resuming."
            self._audio.say(phrase)
        elif key == ord("r"):                   # R → reset session
            self._counter.reset()
            self._in_countdown    = True
            self._countdown_start = time.time()
            self._last_spoken_rep = 0
            self._paused          = False
            self._audio.say("Session reset. Get ready.")
        elif key == ord("s"):                   # S → screenshot
            self._take_screenshot()
        elif key == ord("v"):                   # V → toggle recording
            self._toggle_recording()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _compute_fps(self, now: float) -> float:
        """Compute current FPS using a rolling average.

        Args:
            now: Current timestamp from time.time().

        Returns:
            Smoothed frames-per-second value.
        """
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            self._fps_buffer.append(1.0 / dt)
        if len(self._fps_buffer) > 30:
            self._fps_buffer.pop(0)
        return sum(self._fps_buffer) / len(self._fps_buffer) if self._fps_buffer else 0.0

    def _take_screenshot(self) -> None:
        """Save the current frame as a PNG file in the screenshots folder."""
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"screenshots/squat_{ts}.png"
        if self._cap and self._cap.isOpened():
            ok, frame = self._cap.read()
            if ok:
                cv2.imwrite(path, cv2.flip(frame, 1))
                print(f"[INFO] Screenshot saved: {path}")

    def _toggle_recording(self) -> None:
        """Start or stop video recording of the session."""
        if self._writer is None:
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{config.RECORDING_DIR}/session_{ts}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                path, fourcc, config.TARGET_FPS,
                (config.FRAME_WIDTH, config.FRAME_HEIGHT)
            )
            print(f"[INFO] Recording started: {path}")
        else:
            self._writer.release()
            self._writer = None
            print("[INFO] Recording stopped.")

    def _shutdown(self) -> None:
        """Release all resources gracefully."""
        print("[INFO] Shutting down…")
        if self._cap:
            self._cap.release()
        if self._writer:
            self._writer.release()
        self._detector.release()

        # Save stats if session has any reps
        stats = self._counter.stats
        if config.SAVE_STATS_CSV and stats.total_reps > 0:
            path = self._logger.save(stats, stats.elapsed_seconds)
            print(f"[INFO] Final stats saved to {path}")

        cv2.destroyAllWindows()
        print("[INFO] Bye!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = SquatTrackerApp()
    app.run()
