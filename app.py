"""
app.py
------
Entry point for the AI Fitness Tracker application.

Responsibilities:
    * Camera capture loop
    * Gluing pose detection → exercise counter → UI renderer
    * Keyboard input (Space, R, Q, S, E)
    * Audio feedback dispatch
    * Session CSV export
    * Screenshot / recording management

Command line
~~~~~~~~~~~~
    python app.py                         webcam, starts with squats
    python app.py --exercise plank        pick the starting exercise
    python app.py --video clip.mp4        process a video file instead
    python app.py --video clip.mp4 --headless --exercise pushup
                                          no window; print final stats
    --camera N   override CAMERA_INDEX     --no-audio   disable TTS
    --calibrate  run calibration for the starting exercise first

Keyboard shortcuts
~~~~~~~~~~~~~~~~~~
    Space  → Pause / Resume
    R      → Reset current session
    C      → Calibrate thresholds for the current exercise
    E      → Switch exercise (cycles through exercises.EXERCISES)
    S      → Save screenshot
    V      → Toggle video recording
    Q / Esc→ Quit
"""

from __future__ import annotations

# Must be the very first local import — patches the protobuf/TF conflict
# before MediaPipe can attempt to import tensorflow.tools.docs.
import _mp_compat  # noqa: F401

import sys
import csv
import sqlite3
import argparse
import os
import time
import threading
import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import calibration
import clock
import config
import history
from pose_detector import PoseDetector
from exercise_base import ExerciseBase, FormIssue, RepResult
from exercises import EXERCISES
from ui import UIRenderer, RenderData

# ---------------------------------------------------------------------------
# Optional audio — graceful fallback when pyttsx3 is absent
# ---------------------------------------------------------------------------
try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Audio feedback
# ---------------------------------------------------------------------------

class AudioFeedback:
    """Thread-safe text-to-speech dispatcher with a per-phrase cooldown.

    Runs TTS in a background thread to avoid blocking the render loop.
    """

    def __init__(self, enabled: bool = True) -> None:
        """Initialise the TTS engine if available (and *enabled*)."""
        self._enabled   = enabled and config.AUDIO_ENABLED and _TTS_AVAILABLE
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

    def save(
        self,
        counter: ExerciseBase,
        elapsed: float,
        exercise: str = "squat",
    ) -> Path:
        """Write session stats to a timestamped CSV file.

        Args:
            counter:  The exercise counter whose session is being saved.
            elapsed:  Total session duration in seconds.
            exercise: Exercise key (a key of ``EXERCISES``).

        Returns:
            Path of the written file.
        """
        ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path  = self._dir / f"{exercise}_session_{ts}.csv"
        stats = counter.stats

        spec = EXERCISES[exercise]
        rows = [
            ["Metric", "Value"],
            ["Exercise",        spec.name],
            ["Date",            datetime.datetime.now().strftime("%Y-%m-%d %H:%M")],
            ["Total Reps",      stats.total_reps],
            ["Sets Completed",  stats.sets_completed],
            ["Duration (s)",    f"{elapsed:.1f}"],
            ["Calories",        f"{stats.calories:.1f}"],
        ]
        rows += counter.csv_rows(spec.angle_label)
        # Rep-by-rep breakdown
        rows.append([])
        rows.append(["Rep #", "Duration (s)", f"Min {spec.angle_label} Angle (°)"])
        for i, (dur, depth) in enumerate(
            zip(stats.rep_durations, stats.depths), start=1
        ):
            rows.append([i, f"{dur:.2f}", f"{depth:.3f}"])

        # utf-8-sig writes a BOM so Excel shows "°" correctly
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(rows)

        return path


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------

WINDOW_TITLE = "AI Fitness Tracker"


class FitnessTrackerApp:
    """Orchestrates the full fitness-tracking session.

    Supports multiple exercises (Squat, Push-Up) with live switching via
    the E key.  All subsystems (detection, counting, UI, audio, logging)
    are shared; only the active counter swaps out on exercise change.

    Keyboard shortcuts
    ~~~~~~~~~~~~~~~~~~
        Space  → Pause / Resume
        R      → Reset session
        E      → Cycle exercise
        S      → Screenshot
        V      → Toggle recording
        Q/Esc  → Quit
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        exercise: Optional[str] = None,
        video: Optional[str] = None,
        camera: Optional[int] = None,
        audio: bool = True,
        headless: bool = False,
        calibrate: bool = False,
    ) -> None:
        """Initialise all subsystems.

        Args:
            exercise: Starting exercise key (default: first registry entry).
            video:    Video file to process instead of the webcam.
            camera:   Webcam index (default: config.CAMERA_INDEX).
            audio:    Enable spoken feedback.
            headless: Run without a window and print the final stats.
            calibrate: Start with calibration for the starting exercise.
        """
        self._video    = video
        self._camera   = config.CAMERA_INDEX if camera is None else camera
        self._headless = headless
        self._detector = PoseDetector()
        self._renderer = UIRenderer()
        self._audio    = AudioFeedback(enabled=audio and not headless)
        self._logger   = StatsLogger()

        # Capture
        self._cap: Optional[cv2.VideoCapture] = None
        self._video_time = 0.0                         # Seconds into the video

        # Calibration: saved per-exercise thresholds override config defaults
        applied = calibration.apply(calibration.load())
        if applied:
            print(f"[INFO] Loaded calibration for: {', '.join(applied)}")
        self._calibrate_on_start = calibrate
        self._calibrator: Optional[calibration.Calibrator] = None
        self._cal_counter: Optional[ExerciseBase] = None

        # Exercise state
        self._exercise_key  = exercise or next(iter(EXERCISES))
        self._counter       = self._build_counter()
        self._session_saved = False                    # Guards duplicate saves
        self._load_history()

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
        self._announced_sets = 0

        # Dirs
        Path("screenshots").mkdir(exist_ok=True)
        if config.ENABLE_RECORDING:
            Path(config.RECORDING_DIR).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Exercise management
    # ------------------------------------------------------------------

    def _build_counter(self) -> ExerciseBase:
        """Create and return the counter for the current exercise."""
        return EXERCISES[self._exercise_key].factory()

    def _save_session(self) -> None:
        """Save the current session once: CSV + one history row."""
        if self._session_saved:
            return
        self._counter.on_session_end()
        stats = self._counter.stats
        if stats.total_reps == 0 and self._counter.record_value() == 0:
            return
        self._session_saved = True
        if config.SAVE_STATS_CSV:
            path = self._logger.save(self._counter, stats.elapsed_seconds, self._exercise_key)
            print(f"[INFO] Stats for {self._exercise_key} saved to {path}")
        if config.SAVE_HISTORY:
            try:
                history.add_session(self._exercise_key, self._counter, stats.elapsed_seconds)
            except sqlite3.Error as exc:
                print(f"[WARN] Could not write history: {exc}")

    def _load_history(self) -> None:
        """Fetch "last time" text and the record to beat for this exercise."""
        self._new_record = False
        try:
            self._last_text      = history.last_time_text(self._exercise_key)
            self._record_to_beat = history.personal_record(self._exercise_key)
        except sqlite3.Error as exc:
            print(f"[WARN] Could not read history: {exc}")
            self._last_text, self._record_to_beat = "", None

    def _check_record(self) -> None:
        """Announce once per session when the user beats their best."""
        if (
            not self._new_record
            and self._record_to_beat is not None
            and self._counter.record_value() > self._record_to_beat
        ):
            self._new_record = True
            self._audio.say("New record!")

    def _switch_exercise(self) -> None:
        """Cycle to the next exercise and reset everything."""
        # Save stats for the exercise being left if it has reps
        self._save_session()

        keys = list(EXERCISES)
        self._exercise_key  = keys[(keys.index(self._exercise_key) + 1) % len(keys)]
        self._counter       = self._build_counter()
        self._session_saved       = False
        self._load_history()
        self._last_spoken_rep     = 0
        self._announced_sets = 0
        self._paused              = False
        self._begin_countdown()

        spec = EXERCISES[self._exercise_key]
        self._audio.say(f"Switching to {spec.name}. Get ready.")
        print(f"[INFO] Exercise switched to: {spec.name}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Open the webcam or video file and run the main processing loop."""
        if self._video:
            self._cap = cv2.VideoCapture(self._video)
            if not self._cap.isOpened():
                print(f"[ERROR] Cannot open video file {self._video}.")
                sys.exit(1)
            # Counters time reps / holds on the video's own timeline
            fps   = self._cap.get(cv2.CAP_PROP_FPS) or config.TARGET_FPS
            epoch = time.time()
            clock.use(lambda: epoch + self._video_time)
        else:
            self._cap = cv2.VideoCapture(self._camera)
            if not self._cap.isOpened():
                print(f"[ERROR] Cannot open camera index {self._camera}.")
                sys.exit(1)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            self._cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)

        if not self._headless:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_TITLE, config.FRAME_WIDTH, config.FRAME_HEIGHT)

        self._begin_countdown()
        spec = EXERCISES[self._exercise_key]
        if self._calibrate_on_start:
            self._start_calibration()
        elif not self._video:
            self._audio.say(
                f"Get ready. {spec.name} workout starts in {config.COUNTDOWN_SECONDS}."
            )

        frame_idx = 0
        try:
            while self._running:
                ok, frame = self._cap.read()
                if not ok:
                    if self._video:
                        break                  # End of file = end of session
                    print("[WARN] Empty frame, retrying...")
                    continue

                if self._video:
                    self._video_time = frame_idx / fps
                    frame_idx += 1
                else:
                    frame = cv2.flip(frame, 1)     # Mirror for natural feedback
                self._process_frame(frame)

                if self._headless:
                    continue
                key = cv2.waitKey(1) & 0xFF
                self._handle_key(key, frame)

                if not cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _begin_countdown(self) -> None:
        """Start the pre-set countdown (skipped for video input)."""
        if self._video:
            self._in_countdown = False
            self._start_counting()
        else:
            self._in_countdown    = True
            self._countdown_start = time.time()

    def _start_calibration(self) -> None:
        """Begin guided calibration for the current exercise."""
        key = self._exercise_key
        if key not in calibration.SPECS:
            print(f"[INFO] Calibration isn't available for {EXERCISES[key].name}.")
            self._audio.say("Calibration isn't available for this exercise.")
            return
        self._calibrator   = calibration.Calibrator(key)
        self._cal_counter  = self._build_counter()     # used only to read the metric
        self._in_countdown = False
        self._paused       = False
        print(f"[INFO] Calibrating {EXERCISES[key].name}…")

    def _update_calibration(self, joints, frame_shape) -> float:
        """Feed one frame to the calibrator; finish up when it's done.

        Returns:
            The metric measured this frame (0.0 when no body is found).
        """
        cal = self._calibrator
        metric = None
        if joints is not None:
            _, _, metric = self._cal_counter.process_frame(joints, frame_shape)
        cal.update(metric, clock.now())
        for cue in cal.cues:
            self._audio.say(cue)
        cal.cues.clear()
        if not cal.finished:
            return metric or 0.0

        if cal.result is not None:
            calibration.save(cal.exercise, cal.result)
            calibration.apply({cal.exercise: cal.result})
            print(f"[INFO] Calibration saved for {cal.exercise}: {cal.result}")
        else:
            print("[INFO] Calibration failed (range too small) — keeping current thresholds.")
        self._calibrator  = None
        self._cal_counter = None
        self._counter.reset()
        self._begin_countdown()
        return metric or 0.0

    def _start_counting(self) -> None:
        """Countdown finished: reset the session timer and start counting."""
        self._counter.stats.start_time = clock.now()
        self._counter.on_session_start()

    # ------------------------------------------------------------------
    # Core per-frame pipeline
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> None:
        """Run detection, counting, and rendering for a single frame.

        Args:
            frame: Raw BGR frame from the webcam.
        """
        now   = time.time()
        fps   = self._compute_fps(now)
        stats = self._counter.stats
        spec  = EXERCISES[self._exercise_key]

        # ---- Countdown phase ----------------------------------------
        countdown_val = 0
        if self._in_countdown:
            elapsed_cd = now - self._countdown_start
            remaining  = config.COUNTDOWN_SECONDS - int(elapsed_cd)
            countdown_val = max(0, remaining)
            if countdown_val == 0:
                self._in_countdown = False
                self._audio.say("Go!")
                self._start_counting()

        # --- Pose detection -----------------------------------------
        joints, annotated = self._detector.process(frame)

        calibrating = self._calibrator is not None
        if calibrating:
            cal_message = self._calibrator.message
            cal_metric  = self._update_calibration(joints, frame.shape[:2])

        # ---- Exercise counting (skip during countdown/pause) ---------
        rep_result: Optional[RepResult] = None
        form_issue = FormIssue.NONE
        current_angle = 0.0

        if (
            not calibrating
            and not self._paused
            and not self._in_countdown
            and not self._counter.session_complete
        ):
            rep_result, form_issue, current_angle = self._counter.process_frame(
                joints, frame.shape[:2]
            )
            self._dispatch_audio(rep_result, form_issue, stats)
            self._check_record()

            if self._counter.session_complete:
                self._audio.say(f"{spec.name} workout complete. Great job!")
                self._save_session()

        # ---- Build render data --------------------------------------
        fields = dict(
            reps_in_set      = stats.reps_in_current_set,
            total_reps       = stats.total_reps,
            current_set      = stats.current_set,
            sets_completed   = stats.sets_completed,
            target_reps      = spec.target_reps,
            target_sets      = spec.target_sets,
            coaching_msg     = self._counter.coaching_message,
            form_issue       = form_issue.value,
            knee_angle       = current_angle,
            depth_progress   = self._counter.depth_progress,
            elapsed_seconds  = stats.elapsed_seconds,
            fps              = fps,
            is_paused        = self._paused,
            countdown        = countdown_val,
            session_complete = self._counter.session_complete,
            calories         = stats.calories,
            fastest_rep      = stats.fastest_rep,
            slowest_rep      = stats.slowest_rep,
            avg_depth        = stats.average_depth,
            exercise_name    = spec.name,
            angle_label      = spec.angle_label,
            bar_label        = spec.bar_label,
            last_session     = self._last_text,
            banner           = "NEW RECORD!" if self._new_record else "",
        )
        fields.update(self._counter.hud_fields())   # exercise-specific extras
        if calibrating:
            fields.update(coaching_msg=cal_message, banner="CALIBRATING", countdown=0,
                          knee_angle=cal_metric)
        data = RenderData(**fields)

        if self._headless:
            return

        # ---- Render -------------------------------------------------
        output = self._renderer.render(annotated, data)

        # ---- Optional recording ------------------------------------
        if self._writer is not None:
            self._writer.write(output)

        cv2.imshow(WINDOW_TITLE, output)

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
                    phrase = self._counter.rep_announcement()
                    if phrase:
                        self._audio.say(phrase)
                    halfway = stats.target_reps // 2
                    if halfway and stats.reps_in_current_set == halfway:
                        self._audio.say("Halfway there!")
            else:
                # Bad rep feedback — covers all exercises
                phrase = {
                    # Squat issues
                    FormIssue.GO_LOWER:          "Go lower",
                    FormIssue.STAND_FULLY:        "Stand fully",
                    FormIssue.BAD_BACK:           "Straighten your back",
                    FormIssue.KNEE_CAVE:          "Knees out",
                    # Push-up issues
                    FormIssue.GO_LOWER_PU:        "Go lower",
                    FormIssue.LOCK_ARMS:          "Lock out your arms",
                    FormIssue.STRAIGHTEN:         "Straighten your body",
                    FormIssue.HIPS_UP:            "Keep your hips up",
                    FormIssue.HIPS_DOWN:          "Lower your hips",
                    FormIssue.MOVE_SIDEWAYS:      "Move sideways",
                    # High Knee issues
                    FormIssue.RAISE_KNEES_HIGHER: "Lift above hip",
                    FormIssue.ALTERNATE_LEGS:     "Alternate legs",
                    FormIssue.MOVE_FASTER:        "Move faster",
                    FormIssue.SLOW_DOWN:          "Slow down slightly",
                    # Jumping jack issues
                    FormIssue.ARMS_OVERHEAD:      "Arms all the way up",
                    FormIssue.FEET_WIDER:         "Jump your feet wider",
                    # Curl issues
                    FormIssue.ELBOW_DRIFT:        "Keep your elbows still",
                    FormIssue.EXTEND_ARM:         "Extend your arm fully",
                    FormIssue.CURL_HIGHER:        "Curl all the way up",
                    # Lunge issues
                    FormIssue.KNEE_OVER_TOES:     "Keep your knee behind your toes",
                }.get(rep_result.form_issue, "Bad form")
                self._audio.say(phrase)

        # Mid-rep cues queued by the counter ("Hips up", "Halfway there" …)
        for cue in self._counter.pop_cues():
            self._audio.say(cue)

        # Set completion announcement (workout completion is announced in
        # _process_frame via counter.session_complete)
        if stats.sets_completed > self._announced_sets:
            self._announced_sets = stats.sets_completed
            if not stats.workout_complete:
                self._audio.say(f"Set {stats.sets_completed} complete. Rest up!")

    # ------------------------------------------------------------------
    # Keyboard input
    # ------------------------------------------------------------------

    def _handle_key(self, key: int, frame: np.ndarray) -> None:
        """Process keyboard shortcuts.

        Args:
            key:   Key code from cv2.waitKey.
            frame: Current (unrendered) frame, used for screenshot.

        Shortcuts
        ---------
        Space  Pause / Resume
        R      Reset session
        E      Switch exercise
        S      Screenshot
        V      Toggle recording
        Q/Esc  Quit
        """
        if key == ord("q") or key == 27:       # Q or Escape → quit
            self._running = False
        elif key == ord(" "):                   # Space → pause/resume
            self._paused = not self._paused
            phrase = "Paused." if self._paused else "Resuming."
            self._audio.say(phrase)
        elif key == ord("r"):                   # R → reset session
            self._save_session()                # keep what was done before the reset
            self._counter.reset()
            self._session_saved   = False
            self._load_history()
            self._announced_sets = 0
            self._begin_countdown()
            self._last_spoken_rep = 0
            self._paused          = False
            self._audio.say("Session reset. Get ready.")
        elif key == ord("e"):                   # E → switch exercise
            self._calibrator = None
            self._switch_exercise()
        elif key == ord("c"):                   # C → calibrate / cancel
            if self._calibrator is None:
                self._start_calibration()
            else:
                self._calibrator = None
                self._begin_countdown()
                self._audio.say("Calibration cancelled.")
        elif key == ord("s"):                   # S → screenshot
            self._take_screenshot(frame)
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

    def _take_screenshot(self, frame: np.ndarray) -> None:
        """Save *frame* (the current one) as a PNG in the screenshots folder."""
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"screenshots/{self._exercise_key}_{ts}.png"
        cv2.imwrite(path, frame)
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
        self._save_session()
        if self._video or self._headless:
            print(self.summary())
        clock.use_wall_clock()

        if not self._headless:
            cv2.destroyAllWindows()
        print("[INFO] Bye!")

    def summary(self) -> str:
        """Plain-text final stats for the current exercise."""
        spec, c = EXERCISES[self._exercise_key], self._counter
        best = c.best_metric()
        avg  = c.avg_metric()
        lines = [
            "",
            "=== SESSION SUMMARY ===",
            f"Exercise : {spec.name}",
            f"Reps     : {c.stats.total_reps}",
            f"Sets     : {c.stats.sets_completed} / {spec.target_sets}",
            f"Duration : {c.stats.elapsed_seconds:.1f} s",
            f"Calories : {c.stats.calories:.1f} kcal",
            f"Avg {spec.angle_label:<5}: {'-' if avg is None else f'{avg:.1f}'}",
            f"{spec.best_label} : {'-' if best is None else spec.best_fmt.format(best)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description="AI Fitness Tracker")
    parser.add_argument("--video", metavar="PATH",
                        help="process a video file instead of the webcam")
    parser.add_argument("--exercise", choices=list(EXERCISES),
                        help="starting exercise (default: %(default)s)",
                        default=next(iter(EXERCISES)))
    parser.add_argument("--camera", type=int, metavar="N",
                        help=f"webcam index (default: {config.CAMERA_INDEX})")
    parser.add_argument("--no-audio", action="store_true", help="disable spoken feedback")
    parser.add_argument("--headless", action="store_true",
                        help="no window; print final stats (use with --video)")
    parser.add_argument("--calibrate", action="store_true",
                        help="calibrate thresholds for the starting exercise first")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    app = FitnessTrackerApp(
        exercise=args.exercise,
        video=args.video,
        camera=args.camera,
        audio=not args.no_audio,
        headless=args.headless,
        calibrate=args.calibrate,
    )
    app.run()
