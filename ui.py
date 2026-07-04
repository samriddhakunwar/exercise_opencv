"""
ui.py
-----
All OpenCV drawing routines for the squat-tracker overlay.

The renderer is intentionally decoupled from the business logic:
it receives pure data and writes to a numpy frame.  No MediaPipe or
counter state is imported here — only config and numpy.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

import config

# ---------------------------------------------------------------------------
# Convenience alias
# ---------------------------------------------------------------------------
Color = Tuple[int, int, int]

FONT = cv2.FONT_HERSHEY_DUPLEX


# ---------------------------------------------------------------------------
# Helper drawing functions (private)
# ---------------------------------------------------------------------------

def _alpha_rect(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Color,
    alpha: float = 0.55,
    radius: int = 12,
) -> None:
    """Draw a semi-transparent filled rounded rectangle.

    Args:
        frame:  Frame to draw on (mutated in place).
        x1, y1: Top-left corner.
        x2, y2: Bottom-right corner.
        color:  BGR fill colour.
        alpha:  Opacity of the overlay (0 = fully transparent).
        radius: Corner radius in pixels.
    """
    overlay = frame.copy()
    # Draw rounded rectangle on overlay
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _text_size(text: str, scale: float, thickness: int) -> Tuple[int, int]:
    """Return (width, height) of a rendered text string."""
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return w, h


def _draw_text_shadow(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    scale: float,
    color: Color,
    thickness: int = 2,
    shadow_offset: int = 2,
) -> None:
    """Draw text with a dark drop-shadow for legibility."""
    sx, sy = pos[0] + shadow_offset, pos[1] + shadow_offset
    cv2.putText(frame, text, (sx, sy), FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, FONT, scale, color, thickness, cv2.LINE_AA)


def _pulse_alpha(period: float = 1.5) -> float:
    """Return a sinusoidal alpha value for smooth pulsing animations."""
    t = time.time()
    return 0.5 + 0.5 * math.sin(2 * math.pi * t / period)


# ---------------------------------------------------------------------------
# Data passed to the renderer
# ---------------------------------------------------------------------------

@dataclass
class RenderData:
    """All data needed for one frame of the UI overlay.

    Attributes:
        reps_in_set:      Reps done in the current set.
        total_reps:       Total reps across all sets.
        current_set:      1-based set index.
        sets_completed:   Number of completed sets.
        target_reps:      Reps per set target.
        target_sets:      Total sets target.
        coaching_msg:     Main coaching message to display.
        form_issue:       Current form issue string (empty = none).
        knee_angle:       Smoothed knee angle in degrees.
        depth_progress:   0-1 squat depth fraction.
        elapsed_seconds:  Seconds since session start.
        fps:              Current frames-per-second.
        is_paused:        Whether playback is paused.
        countdown:        Countdown integer (0 = not counting down).
        session_complete: Whether the workout is done.
        calories:         Estimated calories so far.
        fastest_rep:      Fastest rep duration in seconds.
        slowest_rep:      Slowest rep duration in seconds.
        avg_depth:        Average minimum knee angle.
    """
    reps_in_set:      int   = 0
    total_reps:       int   = 0
    current_set:      int   = 1
    sets_completed:   int   = 0
    target_reps:      int   = config.TARGET_REPS
    target_sets:      int   = config.TARGET_SETS
    coaching_msg:     str   = "GET READY"
    form_issue:       str   = ""
    knee_angle:       float = 0.0
    depth_progress:   float = 0.0
    elapsed_seconds:  float = 0.0
    fps:              float = 0.0
    is_paused:        bool  = False
    countdown:        int   = 0
    session_complete: bool  = False
    calories:         float = 0.0
    fastest_rep:      float = 0.0
    slowest_rep:      float = 0.0
    avg_depth:        float = 0.0
    # Exercise identity
    exercise_name:    str   = "SQUAT"   # "SQUAT" or "PUSH-UP"
    angle_label:      str   = "Knee"    # label shown next to the live angle


# ---------------------------------------------------------------------------
# Main renderer class
# ---------------------------------------------------------------------------

class UIRenderer:
    """Draws all HUD elements onto the video frame.

    Designed for a 1280×720 frame.  All positions scale with the frame size.
    """

    def __init__(self) -> None:
        """Pre-compute layout constants."""
        # These will be updated on the first render call
        self._w = 1280
        self._h = 720
        # Animated coaching glow state
        self._last_msg      = ""
        self._msg_change_t  = 0.0
        self._good_rep_flash = 0.0

    # ------------------------------------------------------------------
    # Public render call
    # ------------------------------------------------------------------

    def render(self, frame: np.ndarray, data: RenderData) -> np.ndarray:
        """Draw all HUD elements onto *frame* and return it.

        Args:
            frame: BGR input frame (will be modified in place).
            data:  All display data for this frame.

        Returns:
            The annotated frame.
        """
        self._h, self._w = frame.shape[:2]

        if data.session_complete:
            return self._render_summary(frame, data)

        if data.countdown > 0:
            return self._render_countdown(frame, data)

        if data.is_paused:
            self._render_pause_banner(frame)

        # ---- Standard HUD -------------------------------------------
        self._render_top_left_panel(frame, data)
        self._render_top_right_panel(frame, data)
        self._render_bottom_bar(frame, data)
        self._render_coaching_message(frame, data)
        self._render_hotkey_hint(frame)

        return frame

    # ------------------------------------------------------------------
    # Panel renderers
    # ------------------------------------------------------------------

    def _render_top_left_panel(self, frame: np.ndarray, data: RenderData) -> None:
        """Stats panel — top left corner."""
        pad = 14
        w_box, h_box = 260, 170
        _alpha_rect(frame, pad, pad, pad + w_box, pad + h_box,
                    config.COLOR_SEMI_BG, alpha=0.72)

        # Title — exercise specific
        title = f"{data.exercise_name} TRACKER"
        _draw_text_shadow(frame, title,
                          (pad + 12, pad + 32), 0.55, config.COLOR_ACCENT, 1)

        # Separator line
        cv2.line(frame,
                 (pad + 10, pad + 40),
                 (pad + w_box - 10, pad + 40),
                 config.COLOR_ACCENT, 1)

        # Rep count — big number
        rep_str = str(data.reps_in_set)
        rw, rh = _text_size(rep_str, 2.8, 4)
        rep_color = config.COLOR_GREEN if data.reps_in_set > 0 else config.COLOR_WHITE
        _draw_text_shadow(frame, rep_str,
                          (pad + 20, pad + 110), 2.8, rep_color, 4)

        # Label next to big number
        _draw_text_shadow(frame, "REPS",
                          (pad + 20 + rw + 8, pad + 88), 0.55, config.COLOR_WHITE, 1)
        _draw_text_shadow(frame, f"/ {data.target_reps}",
                          (pad + 20 + rw + 8, pad + 115), 0.65, config.COLOR_ORANGE, 1)

        # Set info
        set_str = f"Set  {data.current_set} / {data.target_sets}"
        _draw_text_shadow(frame, set_str,
                          (pad + 12, pad + 155), 0.58, config.COLOR_WHITE, 1)

    def _render_top_right_panel(self, frame: np.ndarray, data: RenderData) -> None:
        """Progress bar panel — top right corner."""
        if not config.SHOW_PROGRESS_BAR:
            return

        bar_w  = 36
        bar_h  = 200
        margin = 14
        bx     = self._w - margin - bar_w
        by     = margin

        # Background track
        _alpha_rect(frame, bx - 6, by - 6,
                    bx + bar_w + 6, by + bar_h + 6,
                    config.COLOR_SEMI_BG, alpha=0.72)

        # Progress colour based on depth
        p = data.depth_progress
        if p < 0.4:
            bar_color = config.COLOR_RED
        elif p < 0.75:
            bar_color = config.COLOR_YELLOW
        else:
            bar_color = config.COLOR_GREEN

        # Draw filled bar (bottom-up)
        filled_h = int(bar_h * p)
        if filled_h > 0:
            bottom_y = by + bar_h
            _alpha_rect(frame,
                        bx, bottom_y - filled_h,
                        bx + bar_w, bottom_y,
                        bar_color, alpha=0.90, radius=6)

        # Tick marks at 40 % and 75 %
        for pct in (0.4, 0.75, 1.0):
            ty = by + bar_h - int(bar_h * pct)
            cv2.line(frame, (bx - 4, ty), (bx + bar_w + 4, ty),
                     config.COLOR_WHITE, 1)

        # Label
        pct_str = f"{int(p * 100)}%"
        tw, _ = _text_size(pct_str, 0.5, 1)
        _draw_text_shadow(frame, pct_str,
                          (bx + bar_w // 2 - tw // 2, by + bar_h + 22),
                          0.5, config.COLOR_WHITE, 1)
        _draw_text_shadow(frame, "DEPTH",
                          (bx + bar_w // 2 - 20, by - 12),
                          0.45, config.COLOR_WHITE, 1)

    def _render_bottom_bar(self, frame: np.ndarray, data: RenderData) -> None:
        """Metrics strip along the bottom of the frame."""
        bar_h = 56
        by    = self._h - bar_h
        _alpha_rect(frame, 0, by, self._w, self._h,
                    config.COLOR_SEMI_BG, alpha=0.75, radius=0)

        # Elapsed time
        elapsed  = int(data.elapsed_seconds)
        mins, secs = divmod(elapsed, 60)
        time_str = f"⏱  {mins:02d}:{secs:02d}"
        _draw_text_shadow(frame, time_str,
                          (18, self._h - 18), 0.65, config.COLOR_WHITE, 1)

        # FPS
        if config.SHOW_FPS:
            fps_str  = f"FPS: {data.fps:.1f}"
            fw, _    = _text_size(fps_str, 0.55, 1)
            fps_col  = config.COLOR_GREEN if data.fps >= 25 else config.COLOR_RED
            _draw_text_shadow(frame, fps_str,
                              (self._w // 2 - fw // 2, self._h - 18),
                              0.55, fps_col, 1)

        # Angle readout — label depends on exercise
        if config.SHOW_ANGLE:
            ang_str = f"{data.angle_label}: {data.knee_angle:.1f}°"
            aw, _   = _text_size(ang_str, 0.60, 1)
            _draw_text_shadow(frame, ang_str,
                              (self._w - aw - 18, self._h - 18),
                              0.60, config.COLOR_ORANGE, 1)

        # Calories
        cal_str = f"🔥 {data.calories:.1f} kcal"
        _draw_text_shadow(frame, cal_str,
                          (self._w // 2 - 60, self._h - 40),
                          0.52, config.COLOR_YELLOW, 1)

    def _render_coaching_message(self, frame: np.ndarray, data: RenderData) -> None:
        """Centred large coaching message with animated glow."""
        msg = data.coaching_msg or ""
        if not msg:
            return

        # Detect message change for flash animation
        if msg != self._last_msg:
            self._last_msg     = msg
            self._msg_change_t = time.time()
            if "GOOD" in msg or "✓" in msg or "COMPLETE" in msg:
                self._good_rep_flash = 1.0

        # Decay good-rep flash
        dt = time.time() - self._msg_change_t
        self._good_rep_flash = max(0.0, 1.0 - dt / 0.8)

        # Choose colour
        if "GOOD" in msg or "✓" in msg or "COMPLETE" in msg or "GREAT" in msg:
            msg_color = config.COLOR_GREEN
        elif "BAD" in msg or "LOWER" in msg or "CAVE" in msg or "BACK" in msg:
            msg_color = config.COLOR_RED
        elif "ALMOST" in msg or "KEEP" in msg or "TALL" in msg:
            msg_color = config.COLOR_YELLOW
        else:
            msg_color = config.COLOR_WHITE

        scale = 1.4
        mw, mh = _text_size(msg, scale, 2)
        cx = self._w // 2
        cy = self._h // 2

        # Background pill
        pad_x, pad_y = 28, 14
        bx1 = cx - mw // 2 - pad_x
        bx2 = cx + mw // 2 + pad_x
        by1 = cy - mh - pad_y
        by2 = cy + pad_y
        _alpha_rect(frame, bx1, by1, bx2, by2, config.COLOR_SEMI_BG, alpha=0.65)

        # Glow border on new message
        if self._good_rep_flash > 0.1:
            alpha_border = self._good_rep_flash
            overlay = frame.copy()
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), msg_color, 3)
            cv2.addWeighted(overlay, alpha_border, frame, 1 - alpha_border, 0, frame)

        _draw_text_shadow(frame, msg,
                          (cx - mw // 2, cy),
                          scale, msg_color, 2, shadow_offset=3)

    def _render_pause_banner(self, frame: np.ndarray) -> None:
        """Full-frame pause overlay."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self._w, self._h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        α = _pulse_alpha(1.2)
        col = tuple(int(c * α) for c in config.COLOR_ORANGE)
        pw, ph = _text_size("PAUSED", 3.0, 4)
        cx, cy = self._w // 2 - pw // 2, self._h // 2 + ph // 2
        _draw_text_shadow(frame, "PAUSED", (cx, cy), 3.0, col, 4)  # type: ignore[arg-type]

        hint = "SPACE to resume  |  R to reset  |  Q to quit"
        hw, hh = _text_size(hint, 0.6, 1)
        _draw_text_shadow(frame, hint,
                          (self._w // 2 - hw // 2, cy + 50), 0.6, config.COLOR_WHITE, 1)

    def _render_countdown(self, frame: np.ndarray, data: RenderData) -> np.ndarray:
        """Countdown screen drawn over the live feed."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self._w, self._h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)

        num_str = str(data.countdown)
        α = _pulse_alpha(0.8)
        scale = 10.0
        nw, nh = _text_size(num_str, scale, 8)
        col = tuple(int(c * (0.5 + 0.5 * α)) for c in config.COLOR_GREEN)
        _draw_text_shadow(frame, num_str,
                          (self._w // 2 - nw // 2, self._h // 2 + nh // 2),
                          scale, col, 8, shadow_offset=6)  # type: ignore[arg-type]

        ready = f"GET READY TO {data.exercise_name.upper()}"
        rw, _ = _text_size(ready, 0.9, 2)
        _draw_text_shadow(frame, ready,
                          (self._w // 2 - rw // 2, self._h // 4),
                          0.9, config.COLOR_WHITE, 2)
        return frame

    def _render_summary(self, frame: np.ndarray, data: RenderData) -> np.ndarray:
        """Full-screen workout summary overlay."""
        # Dark bg
        overlay = np.zeros_like(frame)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cx = self._w // 2

        # Title — exercise specific
        title = f"{data.exercise_name} COMPLETE!"
        tw, _ = _text_size(title, 2.0, 3)
        _draw_text_shadow(frame, title, (cx - tw // 2, 100), 2.0, config.COLOR_GREEN, 3)

        # Stats box
        box_w, box_h = 550, 380
        bx1 = cx - box_w // 2
        by1 = 130
        _alpha_rect(frame, bx1, by1, bx1 + box_w, by1 + box_h,
                    config.COLOR_SEMI_BG, alpha=0.85)

        lines = [
            ("Total Reps",      str(data.total_reps),          config.COLOR_GREEN),
            ("Sets Completed",  f"{data.sets_completed} / {data.target_sets}", config.COLOR_WHITE),
            ("Duration",        _fmt_time(data.elapsed_seconds), config.COLOR_WHITE),
            (f"Avg {data.angle_label} Angle", f"{data.avg_depth:.1f}°", config.COLOR_ORANGE),
            ("Fastest Rep",     f"{data.fastest_rep:.2f}s",     config.COLOR_GREEN),
            ("Slowest Rep",     f"{data.slowest_rep:.2f}s",     config.COLOR_RED),
            ("Calories Burned", f"{data.calories:.1f} kcal",    config.COLOR_YELLOW),
        ]
        y = by1 + 50
        for label, value, color in lines:
            _draw_text_shadow(frame, f"{label}:", (bx1 + 30, y), 0.70, config.COLOR_WHITE, 1)
            vw, _ = _text_size(value, 0.80, 2)
            _draw_text_shadow(frame, value, (bx1 + box_w - vw - 30, y), 0.80, color, 2)
            cv2.line(frame,
                     (bx1 + 20, y + 12),
                     (bx1 + box_w - 20, y + 12),
                     (60, 60, 60), 1)
            y += 48

        # Footer
        footer = "Press  R  to restart  |  Q  to quit"
        fw, _ = _text_size(footer, 0.7, 1)
        _draw_text_shadow(frame, footer,
                          (cx - fw // 2, self._h - 30), 0.7, config.COLOR_WHITE, 1)
        return frame

    def _render_hotkey_hint(self, frame: np.ndarray) -> None:
        """Small hotkey reference in the bottom-left area."""
        hints = "SPACE=Pause  R=Reset  E=Exercise  Q=Quit  S=Screenshot"
        hw, _ = _text_size(hints, 0.42, 1)
        _draw_text_shadow(frame, hints,
                          (self._w // 2 - hw // 2, self._h - 68),
                          0.42, (150, 150, 150), 1)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS string."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
